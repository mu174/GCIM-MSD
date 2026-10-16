#!/usr/bin/env python3
"""Create the cached text, visual, and acoustic features used by GCIM-MSD."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "test")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mustard", "memotion"))
    parser.add_argument("--modality", required=True, choices=("text", "image", "audio", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "features.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    raw = {}
    for name, value in config["raw_data"].items():
        value = Path(value)
        raw[name] = value if value.is_absolute() else ROOT / value
    config["data_root"] = data_root
    config["raw_data"] = raw
    return config


def preprocess_text(text):
    text = re.sub(r"(@.*?)[\s]", " ", str(text))
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?P<url>https?://[^\s]+)", "", text)
    text = re.sub(r"\@(\w+)", "", text)
    return text.replace("#", "")


def mustard_dialogues(csv_path):
    dialogues = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("KEY", "")
            if key:
                dialogues.setdefault(key, []).append(row["SENTENCE"])
    return dialogues


def dependency_tokens(nlp, text, memotion=False):
    tokens = []
    for token in nlp("" if pd.isna(text) else str(text)):
        item = {
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.text,
            "pos": token.pos_,
        }
        if memotion:
            item["head_pos"] = token.head.pos_
        else:
            item["tag"] = token.tag_
        tokens.append(item)
    return tokens


def encode_text(model, tokenizer, text, max_length, device):
    encoded = tokenizer.encode_plus(
        text=preprocess_text(text),
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_count = int(attention_mask.sum().item())
    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    token_features = hidden[:, 1:token_count, :].squeeze(0).cpu().numpy()
    full_feature = hidden[:, 0, :].cpu().numpy()
    return token_features.astype(np.float32), full_feature.astype(np.float32)


def extract_text(config, dataset, device):
    import spacy
    from transformers import BertModel, BertTokenizer

    data_root = config["data_root"]
    output = data_root / dataset / "features" / "text"
    output.mkdir(parents=True, exist_ok=True)
    model_name = config["models"]["text"]
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = BertModel.from_pretrained(model_name, return_dict=True).to(device).eval()
    nlp = spacy.load(config["models"]["dependency_parser"])
    max_length = int(config["text"][f"{dataset}_max_length"])

    for split in SPLITS:
        split_csv = data_root / dataset / "splits" / f"{split}.csv"
        if dataset == "mustard":
            dialogues = mustard_dialogues(split_csv)
            dependency_graph = {}
            for key, utterances in tqdm(dialogues.items(), desc=f"mustard text {split}"):
                dependency_graph[key] = []
                for index, utterance in enumerate(utterances):
                    token, full = encode_text(model, tokenizer, utterance, max_length, device)
                    np.save(output / f"{key}_{index}.npy", token)
                    np.save(output / f"{key}_{index}_full_text.npy", full)
                    dependency_graph[key].append(dependency_tokens(nlp, utterance))
            target = output / f"dependency_graph_{split}.json"
            target.write_text(json.dumps(dependency_graph, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            frame = pd.read_csv(split_csv, encoding="latin")
            for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"memotion text {split}"):
                image_name = str(getattr(row, "image_name"))
                key = Path(image_name).stem
                text = getattr(row, "text_corrected")
                token, full = encode_text(model, tokenizer, text, max_length, device)
                np.save(output / f"{key}.npy", token)
                np.save(output / f"{key}_full_text.npy", full)
                dependency = dependency_tokens(nlp, text, memotion=True)
                (output / f"{key}_dep_tree.json").write_text(
                    json.dumps(dependency, ensure_ascii=False, indent=2), encoding="utf-8"
                )


def vision_models(config, device):
    import torchvision.transforms as transforms
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from transformers import ViTImageProcessor, ViTModel

    if config["models"]["object_detector"] != "fasterrcnn_resnet50_fpn":
        raise ValueError("Only fasterrcnn_resnet50_fpn is supported")
    detector = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    processor = ViTImageProcessor.from_pretrained(config["models"]["vision"])
    encoder = ViTModel.from_pretrained(config["models"]["vision"]).to(device).eval()
    size = int(config["vision"]["image_size"])
    detector_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return detector, processor, encoder, detector_transform


def encode_image(image, detector, processor, encoder, detector_transform, device):
    detector_input = detector_transform(image).to(device)
    with torch.no_grad():
        boxes = detector([detector_input])[0]["boxes"].cpu().numpy().astype(int)
        whole = encoder(**processor(images=image, return_tensors="pt").to(device))
        whole = whole.last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
    objects = []
    width, height = image.size
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        with torch.no_grad():
            feature = encoder(**processor(images=crop, return_tensors="pt").to(device))
        objects.append(feature.last_hidden_state[:, 0].cpu().numpy())
    local = np.concatenate(objects, axis=0).astype(np.float32) if objects else None
    return whole, local


def extract_image(config, dataset, device):
    data_root = config["data_root"]
    output = data_root / dataset / "features" / "image"
    output.mkdir(parents=True, exist_ok=True)
    detector, processor, encoder, detector_transform = vision_models(config, device)

    if dataset == "memotion":
        image_root = config["raw_data"]["memotion_images"]
        names = []
        for split in SPLITS:
            frame = pd.read_csv(data_root / dataset / "splits" / f"{split}.csv", encoding="latin")
            names.extend(frame["image_name"].astype(str).tolist())
        for name in tqdm(dict.fromkeys(names), desc="memotion images"):
            path = image_root / name
            with Image.open(path) as source:
                image = source.convert("RGB")
            whole, local = encode_image(image, detector, processor, encoder, detector_transform, device)
            key = Path(name).stem
            np.save(output / f"{key}_full_image.npy", whole)
            if local is not None:
                np.save(output / f"{key}.npy", local)
    else:
        frame_root = config["raw_data"]["mustard_frames"]
        for split in SPLITS:
            dialogues = mustard_dialogues(data_root / dataset / "splits" / f"{split}.csv")
            for key, utterances in tqdm(dialogues.items(), desc=f"mustard images {split}"):
                sample_dir = frame_root / key
                frame_count = len([path for path in sample_dir.iterdir() if path.is_file()])
                indices = np.linspace(1, frame_count, num=len(utterances), endpoint=True, dtype=int)
                for index, frame_index in enumerate(indices):
                    path = sample_dir / f"{frame_index:05d}.jpg"
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                    whole, local = encode_image(
                        image, detector, processor, encoder, detector_transform, device
                    )
                    np.save(output / f"{key}_{index}_full_image.npy", whole)
                    if local is not None:
                        np.save(output / f"{key}_{index}.npy", local)


def interpolate_rows(values, output_dim):
    old_axis = np.linspace(0.0, 1.0, values.shape[1])
    new_axis = np.linspace(0.0, 1.0, output_dim)
    return np.stack([np.interp(new_axis, old_axis, row) for row in values])


def extract_audio(config):
    data_root = config["data_root"]
    source_root = config["raw_data"]["mustard_audio_features"]
    segments = int(config["audio"]["segments"])
    output_dim = int(config["audio"]["output_dim"])
    for split in SPLITS:
        output = data_root / "mustard" / "features" / "audio" / split
        output.mkdir(parents=True, exist_ok=True)
        dialogues = mustard_dialogues(data_root / "mustard" / "splits" / f"{split}.csv")
        for key in tqdm(dialogues, desc=f"mustard audio {split}"):
            values = np.load(source_root / f"{key}.npy", allow_pickle=False)
            if values.ndim != 2 or values.shape[0] < segments:
                raise ValueError(f"Expected a 2-D array with at least {segments} rows: {key}")
            usable = values.shape[0] - values.shape[0] % segments
            chunks = np.split(values[:usable], segments)
            pooled = np.stack([chunk.mean(axis=0) for chunk in chunks])
            resized = interpolate_rows(pooled, output_dim).astype(np.float32)
            np.savetxt(output / key, resized, fmt="%f", delimiter=",")


def main():
    args = arguments()
    if args.dataset == "memotion" and args.modality == "audio":
        raise SystemExit("Memotion has no acoustic modality")
    config = load_config(args.config)
    if args.modality in ("text", "all"):
        extract_text(config, args.dataset, args.device)
    if args.modality in ("image", "all"):
        extract_image(config, args.dataset, args.device)
    if args.dataset == "mustard" and args.modality in ("audio", "all"):
        extract_audio(config)


if __name__ == "__main__":
    main()
