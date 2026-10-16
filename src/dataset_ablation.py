"""Dataset wrappers used only by the isolated ablation entry point."""

from __future__ import annotations

import dgl
import torch
from torch.utils.data import Dataset


def combine_modality_graphs(graphs: list[dgl.DGLGraph]) -> dgl.DGLGraph:
    """Rebuild the directed multimodal graph from active modality graphs."""

    if not graphs:
        raise ValueError("At least one modality graph must remain")
    source_parts = []
    destination_parts = []
    offsets = []
    offset = 0
    for graph in graphs:
        offsets.append(offset)
        source, destination = graph.edges()
        source_parts.append(source + offset)
        destination_parts.append(destination + offset)
        offset += graph.num_nodes()

    # Match dataset.py: every earlier modality points to every later modality.
    for left_index in range(len(graphs)):
        for right_index in range(left_index + 1, len(graphs)):
            left_nodes = graphs[left_index].num_nodes()
            right_nodes = graphs[right_index].num_nodes()
            source_parts.append(
                torch.arange(left_nodes).repeat_interleave(right_nodes)
                + offsets[left_index]
            )
            destination_parts.append(
                torch.arange(right_nodes).repeat(left_nodes)
                + offsets[right_index]
            )

    combined = dgl.graph(
        (torch.cat(source_parts), torch.cat(destination_parts)),
        num_nodes=offset,
    )
    combined.ndata["features"] = torch.cat(
        [graph.ndata["features"] for graph in graphs], dim=0
    )
    return combined


class ModalityAblationDataset(Dataset):
    """Remove a modality from every model input while retaining labels."""

    def __init__(self, base_dataset, dataset: str, variant: str):
        self.base_dataset = base_dataset
        self.dataset = dataset
        self.variant = variant
        if dataset not in {"mustard", "memotion"}:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if variant not in {"wo_t", "wo_v"}:
            raise ValueError("Wrapper is only valid for wo_t and wo_v")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        if self.dataset == "mustard":
            _, text_graph, image_graph, audio_graph, sarcasm, sentiment = item
            active = {
                "text": text_graph,
                "image": image_graph,
                "audio": audio_graph,
            }
            active.pop("text" if self.variant == "wo_t" else "image")
            global_graph = combine_modality_graphs(list(active.values()))
            return (
                global_graph,
                text_graph,
                image_graph,
                audio_graph,
                sarcasm,
                sentiment,
            )

        _, text_graph, image_graph, sarcasm, sentiment = item
        active = {"text": text_graph, "image": image_graph}
        active.pop("text" if self.variant == "wo_t" else "image")
        global_graph = combine_modality_graphs(list(active.values()))
        return global_graph, text_graph, image_graph, sarcasm, sentiment


def apply_modality_ablation(base_dataset, dataset: str, variant: str):
    if variant in {"wo_t", "wo_v"}:
        return ModalityAblationDataset(base_dataset, dataset, variant)
    return base_dataset
