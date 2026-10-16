# Importing libraries
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn.functional as F


def mcm_margin_loss(js_divergence, sarcasm_targets, margin):
    """Label-conditioned MCM objective from the paper, averaged over samples."""
    sarcasm_targets = sarcasm_targets.float()
    return (
        sarcasm_targets * F.relu(margin - js_divergence)
        + (1.0 - sarcasm_targets) * js_divergence
    ).mean()


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
    """Function for a single training epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (GraphDataLoader)
        device (str)
        optimizer (torch.optim)
        scheduler (torch.optim.lr_scheduler)
        val_loader (GraphDataLoader, optional): Validation data loader. Defaults to None.
    """

    model.train()
    total_loss = 0
    losses = []
    targets = []
    predictions = []
    sarcasm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    sentiment_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1, weight=sentiment_class_weights
    )
    optimizer.zero_grad(set_to_none=True)

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:
        outputsar, outputsent, outputemo = [], [], []
        tarsar, tarsent, taremo = [], [], []
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Training- Epoch {epoch}")

            batched_graph, text_graph, image_graph, audio_graph, sarcasm, sentiment = batch
            batched_graph, text_graph, image_graph, audio_graph, sarcasm, sentiment = (
                batched_graph.to(device),
                text_graph.to(device),
                image_graph.to(device),
                audio_graph.to(device),
                sarcasm.to(device),
                sentiment.to(device)
            )

            sar, sent, js_divergence = model(
                batched_graph, text_graph, image_graph, audio_graph
            )

            sarArgmax = torch.argmax(sarcasm, dim=-1)
            sentArgmax = torch.argmax(sentiment, dim=-1)

            label_sar = np.argmax(
                sarcasm.cpu().numpy(), axis=-1)
            label_sent = np.argmax(
                sentiment.cpu().numpy(), axis=-1)
            pred_sar = np.argmax(
                sar.cpu().detach().numpy(), axis=1)
            pred_sent = np.argmax(
                sent.cpu().detach().numpy(), axis=1)

            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            tarsar.append(label_sar)
            tarsent.append(label_sent)

            loss1 = sarcasm_criterion(sar, sarArgmax)
            loss2 = sentiment_criterion(sent, sentArgmax)
            loss3 = mcm_margin_loss(js_divergence, sarArgmax, mcm_margin)

            losses.append((loss1.item(), loss2.item()))

            loss = gamma_1 * loss1 + gamma_2 * loss2 + gamma_3 * loss3
            total_loss += loss.item()

            # Accumulate the mean gradient over the configured number of batches.
            (loss / gradient_accumulation_steps).backward()
            should_update = (
                (step + 1) % gradient_accumulation_steps == 0
                or step + 1 == len(data_loader)
            )
            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            single_epoch.set_postfix(train_loss=total_loss / (step + 1))

    outputsar = np.concatenate(
        np.array(outputsar, dtype=object), axis=0)
    outputsent = np.concatenate(
        np.array(outputsent, dtype=object), axis=0)
    # outputemo = np.concatenate(
    #     np.array(outputemo, dtype=object), axis=0)
    tarsar = np.concatenate(
        np.array(tarsar, dtype=object), axis=0)
    tarsent = np.concatenate(
        np.array(tarsent, dtype=object), axis=0)

    report_sar = classification_report(tarsar, outputsar, output_dict=True, labels=np.unique(tarsar), zero_division=0)
    report_sent = classification_report(tarsent, outputsent, output_dict=True, labels=np.unique(tarsent), zero_division=0)

    epoch_train_loss = total_loss / len(data_loader)

    # Run validation if val_loader is provided
    if val_loader:
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
        return epoch_train_loss, report_sar, report_sent, val_loss, val_report_sar, val_report_sent

    return epoch_train_loss, report_sar


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
    """Function for a single validation epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (GraphDataLoader)
        device (str)
    """

    model.eval()
    total_loss = 0
    targets = []
    predictions = []
    sarcasm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    sentiment_criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1, weight=sentiment_class_weights
    ).to(device)

    with (tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch):
        outputsar, outputsent, outputemo = [], [], []
        tarsar, tarsent, taremo = [], [], []
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            batched_graph, text_graph, image_graph, audio_graph, sarcasm, sentiment = batch
            batched_graph, text_graph, image_graph, audio_graph, sarcasm, sentiment = (
                batched_graph.to(device),
                text_graph.to(device),
                image_graph.to(device),
                audio_graph.to(device),
                sarcasm.to(device),
                sentiment.to(device)
            )

            with torch.no_grad():
                sar, sent, js_divergence = model(
                    batched_graph, text_graph, image_graph, audio_graph
                )

            sarArgmax = torch.argmax(sarcasm, dim=-1)
            sentArgmax = torch.argmax(sentiment, dim=-1)

            loss1_val = sarcasm_criterion(sar, sarArgmax)
            loss2_val = sentiment_criterion(sent, sentArgmax)
            loss3_val = mcm_margin_loss(js_divergence, sarArgmax, mcm_margin)

            eval_loss = (
                gamma_1 * loss1_val
                + gamma_2 * loss2_val
                + gamma_3 * loss3_val
            )

            total_loss += eval_loss.item()

            single_epoch.set_postfix(loss=eval_loss.item())

            label_sar = np.argmax(
                sarcasm.cpu().detach().numpy(), axis=-1)
            label_sent = np.argmax(
                sentiment.cpu().detach().numpy(), axis=-1)

            pred_sar = np.argmax(
                sar.cpu().detach().numpy(), axis=1)
            pred_sent = np.argmax(
                sent.cpu().detach().numpy(), axis=1)

            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            tarsar.append(label_sar)
            tarsent.append(label_sent)

        outputsar = np.concatenate(
            np.array(outputsar, dtype=object), axis=0)
        outputsent = np.concatenate(
            np.array(outputsent, dtype=object), axis=0)

        tarsar = np.concatenate(
            np.array(tarsar, dtype=object), axis=0)
        tarsent = np.concatenate(
            np.array(tarsent, dtype=object), axis=0)

    epoch_validation_loss = total_loss / len(data_loader)

    report_sar = classification_report(tarsar, outputsar, output_dict=True, labels=np.unique(tarsar), zero_division=0)
    report_sent = classification_report(tarsent, outputsent, output_dict=True, labels=np.unique(tarsent), zero_division=0)

    return epoch_validation_loss, report_sar, report_sent
