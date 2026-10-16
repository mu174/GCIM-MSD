"""Evaluate one released GCIM-MSD checkpoint on its fixed test split."""

import argparse
import json
from pathlib import Path

import torch
from dgl.dataloading import GraphDataLoader

import engine_memotion
import engine_mustard
import model_memotion
import model_mustard
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("mustard", "memotion"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mcm-margin", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "mustard":
        train_set, test_set, _ = utils.set_up_mustard()
        model = model_mustard.MModel().to(device)
        engine = engine_mustard
        batch_size = args.batch_size or 32
        counts = torch.zeros(3, dtype=torch.float64)
        for index in range(len(train_set)):
            counts[int(torch.as_tensor(train_set[index][-1]).argmax().item())] += 1
        sentiment_weights = (counts.sum() / (3 * counts)).float().to(device)
    else:
        _, test_set, _ = utils.set_up_memotion()
        model = model_memotion.MModel().to(device)
        engine = engine_memotion
        batch_size = args.batch_size or 48
        sentiment_weights = None

    loader = GraphDataLoader(
        test_set, batch_size=batch_size, drop_last=False, shuffle=False
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    test_loss, sarcasm_report, sentiment_report = engine.eval_func(
        model,
        loader,
        device,
        gamma_1=0.5,
        gamma_2=0.5,
        gamma_3=1.0,
        mcm_margin=args.mcm_margin,
        sentiment_class_weights=sentiment_weights,
    )
    record = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "test_loss": test_loss,
        "test_sarcasm_weighted": sarcasm_report["weighted avg"],
        "test_sentiment_weighted": sentiment_report["weighted avg"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
