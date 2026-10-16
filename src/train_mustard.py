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
import engine_mustard as engine
import model_mustard as model_mu
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=config.lr)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=config.gradient_accumulation_steps,
    )
    parser.add_argument("--lambda-sarcasm", type=float, default=0.5)
    parser.add_argument("--lambda-sentiment", type=float, default=0.5)
    parser.add_argument("--lambda-mcm", type=float, default=1.0)
    parser.add_argument("--selection-sarcasm-weight", type=float, default=0.7)
    parser.add_argument("--selection-sentiment-weight", type=float, default=0.3)
    parser.add_argument(
        "--mcm-margin",
        type=float,
        required=True,
        help="MCM discrepancy margin delta; no value was recorded in the old run",
    )
    parser.add_argument(
        "--sentiment-weighting",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="New checkpoint path; its parent directory must not already exist",
    )
    return parser.parse_args()


def weighted_metrics(report):
    return report["weighted avg"]


def sentiment_weights(dataset, device):
    counts = torch.zeros(3, dtype=torch.float64)
    for index in range(len(dataset)):
        label = dataset[index][-1]
        target = int(torch.as_tensor(label).argmax().item())
        counts[target] += 1
    if torch.any(counts == 0):
        raise ValueError(f"Missing sentiment class in training data: {counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    return weights.float().to(device), counts.tolist()


if __name__ == "__main__":
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    if args.selection_sarcasm_weight < 0 or args.selection_sentiment_weight < 0:
        raise ValueError("selection weights must be non-negative")
    if not math.isclose(
        args.selection_sarcasm_weight + args.selection_sentiment_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("selection weights must sum to 1")

    utils.set_seed(args.seed)
    dataset_train, dataset_test, dataset_dev = utils.set_up_mustard()

    dataloader_train = GraphDataLoader(
        dataset_train,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
    )
    dataloader_test = GraphDataLoader(
        dataset_test,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )
    dataloader_dev = GraphDataLoader(
        dataset_dev,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    device = torch.device(os.environ.get("GCIM_MSD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model_mu.MModel().to(device)
    if args.sentiment_weighting == "balanced":
        sentiment_class_weights, sentiment_counts = sentiment_weights(
            dataset_train, device
        )
    else:
        sentiment_class_weights = None
        sentiment_counts = None

    updates_per_epoch = math.ceil(
        len(dataloader_train) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=total_updates,
    )

    print("Device:", device)
    print("Training sentiment counts:", sentiment_counts)
    print(
        "Sentiment class weights:",
        None
        if sentiment_class_weights is None
        else sentiment_class_weights.detach().cpu().tolist(),
    )
    print("Optimizer updates per epoch:", updates_per_epoch)
    print(
        "Checkpoint selection: validation joint weighted F1",
        args.selection_sarcasm_weight,
        args.selection_sentiment_weight,
    )

    best_score = -math.inf
    best_val_sarcasm_f1 = -math.inf
    best_epoch = None
    best_val_loss = None
    best_val_report_sar = None
    best_val_report_sent = None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=False)

    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_report_sar,
            train_report_sent,
            val_loss,
            val_report_sar,
            val_report_sent,
        ) = engine.train_func_epoch(
            epoch,
            model,
            dataloader_train,
            device,
            optimizer,
            scheduler,
            val_loader=dataloader_dev,
            gamma_1=args.lambda_sarcasm,
            gamma_2=args.lambda_sentiment,
            gamma_3=args.lambda_mcm,
            mcm_margin=args.mcm_margin,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            sentiment_class_weights=sentiment_class_weights,
        )

        val_sarcasm_f1 = weighted_metrics(val_report_sar)["f1-score"]
        val_sentiment_f1 = weighted_metrics(val_report_sent)["f1-score"]
        selection_score = (
            args.selection_sarcasm_weight * val_sarcasm_f1
            + args.selection_sentiment_weight * val_sentiment_f1
        )
        print(
            f"Epoch {epoch} | train loss {train_loss:.6f} "
            f"| validation loss {val_loss:.6f}"
        )
        print("Train Sarcasm:", weighted_metrics(train_report_sar))
        print("Train Sentiment:", weighted_metrics(train_report_sent))
        print("Validation Sarcasm:", weighted_metrics(val_report_sar))
        print("Validation Sentiment:", weighted_metrics(val_report_sent))
        print(f"Validation joint selection score: {selection_score:.6f}")

        # Use validation data only. Sarcasm F1 breaks an exact joint-score tie.
        candidate = (selection_score, val_sarcasm_f1)
        incumbent = (best_score, best_val_sarcasm_f1)
        if candidate > incumbent:
            best_score = selection_score
            best_val_sarcasm_f1 = val_sarcasm_f1
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_report_sar = val_report_sar
            best_val_report_sent = val_report_sent
            temporary_path = args.checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), temporary_path)
            temporary_path.replace(args.checkpoint)
            print(
                f"Best model saved at epoch {best_epoch} "
                f"(validation joint score: {best_score:.6f}; "
                f"sarcasm F1: {val_sarcasm_f1:.6f}; "
                f"sentiment F1: {val_sentiment_f1:.6f})"
            )

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_loss, test_report_sar, test_report_sent = engine.eval_func(
        model,
        dataloader_test,
        device,
        gamma_1=args.lambda_sarcasm,
        gamma_2=args.lambda_sentiment,
        gamma_3=args.lambda_mcm,
        mcm_margin=args.mcm_margin,
        sentiment_class_weights=sentiment_class_weights,
    )
    print("Best epoch:", best_epoch)
    print("Test Sarcasm:", weighted_metrics(test_report_sar))
    print("Test Sentiment:", weighted_metrics(test_report_sent))

    record = {
        "dataset": "MUStARD",
        "implementation": "fixed_mcm_accumulation_joint_selection",
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint": str(args.checkpoint),
        "selection": {
            "split": "validation",
            "metric": f"{args.selection_sarcasm_weight} * sarcasm_weighted_f1 + {args.selection_sentiment_weight} * sentiment_weighted_f1",
            "mode": "max",
            "sarcasm_weight": args.selection_sarcasm_weight,
            "sentiment_weight": args.selection_sentiment_weight,
            "tie_breaker": "sarcasm_weighted_f1",
        },
        "loss": {
            "lambda_sarcasm": args.lambda_sarcasm,
            "lambda_sentiment": args.lambda_sentiment,
            "lambda_mcm": args.lambda_mcm,
            "mcm_margin": args.mcm_margin,
            "sentiment_weighting": args.sentiment_weighting,
            "sentiment_training_counts": sentiment_counts,
            "sentiment_class_weights": (
                None
                if sentiment_class_weights is None
                else sentiment_class_weights.detach().cpu().tolist()
            ),
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "updates_per_epoch": updates_per_epoch,
        },
        "best_epoch": best_epoch,
        "best_validation_joint_score": best_score,
        "best_validation_loss": best_val_loss,
        "best_validation_sarcasm_weighted": weighted_metrics(
            best_val_report_sar
        ),
        "best_validation_sentiment_weighted": weighted_metrics(
            best_val_report_sent
        ),
        "test_loss": test_loss,
        "test_sarcasm_weighted": weighted_metrics(test_report_sar),
        "test_sentiment_weighted": weighted_metrics(test_report_sent),
    }
    metadata_path = args.checkpoint.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    print("Run metadata saved to", metadata_path)
