import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm


def mcm_margin_loss(js_divergence, sarcasm_targets, margin):
    sarcasm_targets = sarcasm_targets.float()
    return (
        sarcasm_targets * F.relu(margin - js_divergence)
        + (1.0 - sarcasm_targets) * js_divergence
    ).mean()


def _reports(sarcasm_targets, sarcasm_predictions, sentiment_targets,
             sentiment_predictions):
    sarcasm_report = classification_report(
        sarcasm_targets,
        sarcasm_predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    sentiment_report = classification_report(
        sentiment_targets,
        sentiment_predictions,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0,
    )
    return sarcasm_report, sentiment_report


def train_func_epoch(
    epoch,
    model,
    data_loader,
    device,
    optimizer,
    scheduler,
    val_loader=None,
    gamma_1=0.5,
    gamma_2=0.5,
    gamma_3=1.0,
    mcm_margin=0.5,
    gradient_accumulation_steps=1,
    sentiment_class_weights=None,
):
    model.train()
    total_loss = 0.0
    sarcasm_predictions = []
    sentiment_predictions = []
    sarcasm_targets = []
    sentiment_targets = []
    sarcasm_criterion = torch.nn.CrossEntropyLoss()
    sentiment_criterion = torch.nn.CrossEntropyLoss(
        weight=sentiment_class_weights
    )
    optimizer.zero_grad(set_to_none=True)

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as progress:
        for step, batch in enumerate(progress):
            progress.set_description(f"Training- Epoch {epoch}")
            batched_graph, text_graph, image_graph, sarcasm, sentiment = batch
            batched_graph = batched_graph.to(device)
            text_graph = text_graph.to(device)
            image_graph = image_graph.to(device)
            sarcasm = sarcasm.long().to(device)
            sentiment = sentiment.long().to(device)

            sar, sent, js_divergence = model(
                batched_graph, text_graph, image_graph
            )
            loss_sarcasm = sarcasm_criterion(sar, sarcasm)
            loss_sentiment = sentiment_criterion(sent, sentiment)
            loss_mcm = mcm_margin_loss(
                js_divergence, sarcasm, mcm_margin
            )
            loss = (
                gamma_1 * loss_sarcasm
                + gamma_2 * loss_sentiment
                + gamma_3 * loss_mcm
            )
            total_loss += loss.item()

            (loss / gradient_accumulation_steps).backward()
            should_update = (
                (step + 1) % gradient_accumulation_steps == 0
                or step + 1 == len(data_loader)
            )
            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            sarcasm_predictions.append(sar.argmax(dim=1).detach().cpu().numpy())
            sentiment_predictions.append(sent.argmax(dim=1).detach().cpu().numpy())
            sarcasm_targets.append(sarcasm.detach().cpu().numpy())
            sentiment_targets.append(sentiment.detach().cpu().numpy())
            progress.set_postfix(train_loss=total_loss / (step + 1))

    reports = _reports(
        np.concatenate(sarcasm_targets),
        np.concatenate(sarcasm_predictions),
        np.concatenate(sentiment_targets),
        np.concatenate(sentiment_predictions),
    )
    train_loss = total_loss / len(data_loader)
    if val_loader is None:
        return train_loss, *reports

    val_loss, val_report_sar, val_report_sent = eval_func(
        model,
        val_loader,
        device,
        epoch,
        gamma_1,
        gamma_2,
        gamma_3,
        mcm_margin,
        sentiment_class_weights,
    )
    return (
        train_loss,
        reports[0],
        reports[1],
        val_loss,
        val_report_sar,
        val_report_sent,
    )


def eval_func(
    model,
    data_loader,
    device,
    epoch=1,
    gamma_1=0.5,
    gamma_2=0.5,
    gamma_3=1.0,
    mcm_margin=0.5,
    sentiment_class_weights=None,
):
    model.eval()
    total_loss = 0.0
    sarcasm_predictions = []
    sentiment_predictions = []
    sarcasm_targets = []
    sentiment_targets = []
    sarcasm_criterion = torch.nn.CrossEntropyLoss()
    sentiment_criterion = torch.nn.CrossEntropyLoss(
        weight=sentiment_class_weights
    )

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as progress:
        for step, batch in enumerate(progress):
            progress.set_description(f"Evaluating- Epoch {epoch}")
            batched_graph, text_graph, image_graph, sarcasm, sentiment = batch
            batched_graph = batched_graph.to(device)
            text_graph = text_graph.to(device)
            image_graph = image_graph.to(device)
            sarcasm = sarcasm.long().to(device)
            sentiment = sentiment.long().to(device)

            with torch.no_grad():
                sar, sent, js_divergence = model(
                    batched_graph, text_graph, image_graph
                )
                loss_sarcasm = sarcasm_criterion(sar, sarcasm)
                loss_sentiment = sentiment_criterion(sent, sentiment)
                loss_mcm = mcm_margin_loss(
                    js_divergence, sarcasm, mcm_margin
                )
                loss = (
                    gamma_1 * loss_sarcasm
                    + gamma_2 * loss_sentiment
                    + gamma_3 * loss_mcm
                )
            total_loss += loss.item()
            sarcasm_predictions.append(sar.argmax(dim=1).cpu().numpy())
            sentiment_predictions.append(sent.argmax(dim=1).cpu().numpy())
            sarcasm_targets.append(sarcasm.cpu().numpy())
            sentiment_targets.append(sentiment.cpu().numpy())
            progress.set_postfix(loss=loss.item())

    reports = _reports(
        np.concatenate(sarcasm_targets),
        np.concatenate(sarcasm_predictions),
        np.concatenate(sentiment_targets),
        np.concatenate(sentiment_predictions),
    )
    return total_loss / len(data_loader), reports[0], reports[1]
