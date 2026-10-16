#!/usr/bin/env python3
"""Train one GCIM-MSD ablation and save structured metrics."""

from __future__ import annotations

import os
import argparse
import json
import math
from pathlib import Path

import torch
from dgl.dataloading import GraphDataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset_ablation
import engine_memotion
import engine_mustard
import model_ablation
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mustard", "memotion"), required=True)
    parser.add_argument(
        "--variant", choices=model_ablation.SUPPORTED_VARIANTS, required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lambda-sarcasm", type=float, default=0.5)
    parser.add_argument("--lambda-sentiment", type=float, default=0.5)
    parser.add_argument("--lambda-mcm", type=float, default=1.0)
    parser.add_argument("--mcm-margin", type=float, default=0.5)
    parser.add_argument("--selection-sarcasm-weight", type=float, default=0.7)
    parser.add_argument("--selection-sentiment-weight", type=float, default=0.3)
    parser.add_argument(
        "--sentiment-weighting", choices=("balanced", "none"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0 or args.gradient_accumulation_steps < 1:
        raise ValueError("learning rate and accumulation steps must be positive")
    if args.selection_sarcasm_weight < 0 or args.selection_sentiment_weight < 0:
        raise ValueError("selection weights must be non-negative")
    if not math.isclose(
        args.selection_sarcasm_weight + args.selection_sentiment_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("selection weights must sum to 1")


def weighted(report):
    return report["weighted avg"]


def base_dataset(dataset):
    return getattr(dataset, "base_dataset", dataset)


def sentiment_weights(dataset, dataset_name, device):
    source = base_dataset(dataset)
    if dataset_name == "memotion":
        labels = torch.as_tensor(
            source.df["overall_sentiment"].to_numpy(), dtype=torch.long
        )
        counts = torch.bincount(labels, minlength=3).double()
    else:
        labels = [int(item["sentiment-label"]) + 1 for item in source.uttList]
        counts = torch.bincount(torch.tensor(labels), minlength=3).double()
    if torch.any(counts == 0):
        raise ValueError(f"Missing sentiment class: {counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    return weights.float().to(device), counts.tolist()


def effective_loss_weights(args):
    if args.variant == "wo_mcm":
        return args.lambda_sarcasm, args.lambda_sentiment, 0.0
    if args.variant == "stl_sarcasm":
        return 1.0, 0.0, args.lambda_mcm
    if args.variant == "stl_sentiment":
        return 0.0, 1.0, 0.0
    return args.lambda_sarcasm, args.lambda_sentiment, args.lambda_mcm


def selection(args, sarcasm_f1, sentiment_f1):
    if args.variant == "stl_sarcasm":
        return sarcasm_f1, "sarcasm_weighted_f1", None
    if args.variant == "stl_sentiment":
        return sentiment_f1, "sentiment_weighted_f1", None
    score = (
        args.selection_sarcasm_weight * sarcasm_f1
        + args.selection_sentiment_weight * sentiment_f1
    )
    return (
        score,
        "weighted_joint_f1",
        {
            "sarcasm": args.selection_sarcasm_weight,
            "sentiment": args.selection_sentiment_weight,
        },
    )


def set_up(args):
    if args.dataset == "mustard":
        train, test, validation = utils.set_up_mustard()
        engine = engine_mustard
    else:
        train, test, validation = utils.set_up_memotion()
        engine = engine_memotion
    return (
        dataset_ablation.apply_modality_ablation(train, args.dataset, args.variant),
        dataset_ablation.apply_modality_ablation(test, args.dataset, args.variant),
        dataset_ablation.apply_modality_ablation(
            validation, args.dataset, args.variant
        ),
        engine,
    )


def main():
    args = parse_args()
    validate_args(args)
    utils.set_seed(args.seed)
    train_dataset, test_dataset, validation_dataset, engine = set_up(args)
    train_loader = GraphDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    validation_loader = GraphDataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = GraphDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    device = torch.device(os.environ.get("GCIM_MSD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model_ablation.AblationModel(args.dataset, args.variant).to(device)
    if args.sentiment_weighting == "balanced":
        class_weights, sentiment_counts = sentiment_weights(
            train_dataset, args.dataset, device
        )
    else:
        class_weights, sentiment_counts = None, None

    gamma_1, gamma_2, gamma_3 = effective_loss_weights(args)
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=updates_per_epoch * args.epochs,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=False)

    print("Dataset:", args.dataset)
    print("Variant:", args.variant)
    print("Variant definition:", model_ablation.VARIANT_DEFINITIONS[args.variant])
    print("Active modalities:", model.active_modalities)
    print("Device:", device)
    print("Effective loss weights:", gamma_1, gamma_2, gamma_3)
    print("Training sentiment counts:", sentiment_counts)
    print(
        "Sentiment class weights:",
        None if class_weights is None else class_weights.detach().cpu().tolist(),
    )
    print("Optimizer updates per epoch:", updates_per_epoch)

    best_candidate = (-math.inf, -math.inf)
    best_epoch = None
    best_loss = None
    best_sarcasm = None
    best_sentiment = None
    best_selection = None
    selection_metric = None
    selection_weights = None

    for epoch in range(1, args.epochs + 1):
        results = engine.train_func_epoch(
            epoch,
            model,
            train_loader,
            device,
            optimizer,
            scheduler,
            val_loader=validation_loader,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            gamma_3=gamma_3,
            mcm_margin=args.mcm_margin,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            sentiment_class_weights=class_weights,
        )
        (
            train_loss,
            train_sarcasm,
            train_sentiment,
            validation_loss,
            validation_sarcasm,
            validation_sentiment,
        ) = results
        sarcasm_f1 = weighted(validation_sarcasm)["f1-score"]
        sentiment_f1 = weighted(validation_sentiment)["f1-score"]
        score, metric, weights = selection(args, sarcasm_f1, sentiment_f1)
        tie_breaker = sarcasm_f1 if metric == "weighted_joint_f1" else score
        candidate = (score, tie_breaker)

        print(
            f"Epoch {epoch} | train loss {train_loss:.6f} "
            f"| validation loss {validation_loss:.6f}"
        )
        print("Train Sarcasm:", weighted(train_sarcasm))
        print("Train Sentiment:", weighted(train_sentiment))
        print("Validation Sarcasm:", weighted(validation_sarcasm))
        print("Validation Sentiment:", weighted(validation_sentiment))
        print(f"Validation selection score ({metric}): {score:.6f}")

        if candidate > best_candidate:
            best_candidate = candidate
            best_epoch = epoch
            best_loss = validation_loss
            best_sarcasm = validation_sarcasm
            best_sentiment = validation_sentiment
            best_selection = score
            selection_metric = metric
            selection_weights = weights
            temporary = args.checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(args.checkpoint)
            print(f"Best model saved at epoch {epoch}: {score:.6f}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_loss, test_sarcasm, test_sentiment = engine.eval_func(
        model,
        test_loader,
        device,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        mcm_margin=args.mcm_margin,
        sentiment_class_weights=class_weights,
    )
    print("Best epoch:", best_epoch)
    print("Test Sarcasm:", weighted(test_sarcasm))
    print("Test Sentiment:", weighted(test_sentiment))

    record = {
        "dataset": "MUStARD" if args.dataset == "mustard" else "Memotion",
        "implementation": "isolated_gcim_msd_ablation_v1",
        "variant": args.variant,
        "variant_definition": model_ablation.VARIANT_DEFINITIONS[args.variant],
        "active_modalities": list(model.active_modalities),
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint": str(args.checkpoint),
        "selection": {
            "split": "validation",
            "metric": selection_metric,
            "weights": selection_weights,
            "mode": "max",
            "tie_breaker": (
                "sarcasm_weighted_f1"
                if selection_metric == "weighted_joint_f1"
                else None
            ),
        },
        "loss": {
            "lambda_sarcasm": gamma_1,
            "lambda_sentiment": gamma_2,
            "lambda_mcm": gamma_3,
            "mcm_margin": args.mcm_margin,
            "sentiment_weighting": args.sentiment_weighting,
            "sentiment_training_counts": sentiment_counts,
            "sentiment_class_weights": (
                None
                if class_weights is None
                else class_weights.detach().cpu().tolist()
            ),
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "updates_per_epoch": updates_per_epoch,
        },
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_selection,
        "best_validation_loss": best_loss,
        "best_validation_sarcasm_weighted": weighted(best_sarcasm),
        "best_validation_sentiment_weighted": weighted(best_sentiment),
        "test_loss": test_loss,
        "test_sarcasm_weighted": weighted(test_sarcasm),
        "test_sentiment_weighted": weighted(test_sentiment),
        "reported_tasks": (
            ["sarcasm"]
            if args.variant == "stl_sarcasm"
            else ["sentiment"]
            if args.variant == "stl_sentiment"
            else ["sarcasm", "sentiment"]
        ),
    }
    metadata = args.checkpoint.with_suffix(".json")
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("Run metadata saved to", metadata)


if __name__ == "__main__":
    main()
