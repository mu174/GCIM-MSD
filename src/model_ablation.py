"""Isolated GCIM-MSD models for ablation and modality experiments.

This module intentionally does not modify or monkey-patch ``model_mu_fixed``
or ``model_me_fixed``.  Every experiment is trained from scratch with an
explicit variant recorded by the training entry point.
"""

from __future__ import annotations

import math

import dgl
import torch
from dgl.nn.pytorch import conv
from torch import nn
import torch.nn.functional as F


SUPPORTED_VARIANTS = (
    "wo_crmd",
    "wo_mcm",
    "wo_gate",
    "wo_sp",
    "wo_t",
    "wo_v",
    "stl_sarcasm",
    "stl_sentiment",
)

VARIANT_DEFINITIONS = {
    "wo_crmd": (
        "Remove the CRMD output from the sarcasm decoder; its 256-dimensional "
        "slot is replaced by zeros so the decoder input size stays unchanged."
    ),
    "wo_mcm": (
        "Keep the forward representation unchanged and disable the MCM margin "
        "loss in the training entry point."
    ),
    "wo_gate": (
        "Replace every learned sigmoid gate in the sentiment and sarcasm "
        "decoders with an identity multiplier of one."
    ),
    "wo_sp": (
        "Replace the shared additional-modality projection with one independent "
        "projection per active modality."
    ),
    "wo_t": (
        "Remove text from the rebuilt global graph and from CRMD/MCM inputs."
    ),
    "wo_v": (
        "Remove vision from the rebuilt global graph and from CRMD/MCM inputs."
    ),
    "stl_sarcasm": (
        "Train only with sarcasm supervision; sentiment cross-entropy is zero."
    ),
    "stl_sentiment": (
        "Train only with sentiment supervision; sarcasm and MCM losses are zero."
    ),
}


def generalized_js_divergence(distributions: list[torch.Tensor]) -> torch.Tensor:
    """Return one normalized JS-divergence value per sample.

    Each input is ``[batch, heads, feature_dim]``.  With one remaining
    modality (Memotion w/o-T or w/o-V), conflict is defined as zero.
    """

    if not distributions:
        raise ValueError("At least one active modality is required")
    if len(distributions) == 1:
        return distributions[0].sum(dim=(1, 2)) * 0.0

    stacked = torch.stack(distributions, dim=1).float()
    probabilities = F.softmax(stacked, dim=2)
    mean_probability = probabilities.mean(dim=1, keepdim=True)
    eps = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp_min(eps)
    mean_probability = mean_probability.clamp_min(eps)
    kl = probabilities * (probabilities.log() - mean_probability.log())
    kl = kl.sum(dim=2)
    return kl.mean(dim=(1, 2)) / math.log(2.0)


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        if num_modalities < 1:
            raise ValueError("CRMD requires at least one active modality")
        self.feature_dim = feature_dim
        self.transform_layer = nn.Linear(feature_dim, feature_dim)
        self.projection_layer = nn.Linear(
            num_modalities * feature_dim, feature_dim
        )
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        self.layer_norm = nn.LayerNorm(feature_dim)

    @staticmethod
    def calculate_attention(query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(query.size(-1))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

    def forward(self, shared_features, modality_features):
        if shared_features.size(-1) != self.feature_dim:
            raise ValueError("Unexpected shared feature dimension")
        if any(item.size(-1) != self.feature_dim for item in modality_features):
            raise ValueError("Unexpected modality feature dimension")

        query = self.transform_layer(shared_features)
        transformed_modalities = [
            self.transform_layer(item) for item in modality_features
        ]
        attended_reference = self.calculate_attention(query, query, query)
        attended_modalities = [
            self.calculate_attention(query, item, item)
            for item in transformed_modalities
        ]
        differences = [
            self.layer_norm(torch.abs(attended_reference - item))
            for item in attended_modalities
        ]
        modality_weights = F.softmax(self.modality_weights, dim=0)
        combined_difference = sum(
            weight * difference
            for weight, difference in zip(modality_weights, differences)
        )
        mask = torch.sigmoid(combined_difference)
        adjusted = [item * mask for item in attended_modalities]
        return self.projection_layer(torch.cat(adjusted, dim=-1))


class AblationModel(nn.Module):
    """GCIM-MSD backbone with one explicit ablation variant."""

    def __init__(self, dataset: str, variant: str):
        super().__init__()
        if dataset not in {"mustard", "memotion"}:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}")

        self.dataset = dataset
        self.variant = variant
        modality_names = (
            ["text", "image", "audio"]
            if dataset == "mustard"
            else ["text", "image"]
        )
        if variant == "wo_t":
            modality_names.remove("text")
        elif variant == "wo_v":
            modality_names.remove("image")
        self.active_modalities = tuple(modality_names)

        self.mm_embedding_space = nn.Sequential(
            nn.Linear(768, 768),
            nn.ELU(),
            nn.Dropout(0.4),
        )
        self.gat_layer = conv.GATConv(
            in_feats=768,
            out_feats=256,
            num_heads=4,
            feat_drop=0.5,
            attn_drop=0.0,
            residual=True,
            activation=nn.ELU(),
        )

        def projection():
            return nn.Sequential(
                nn.Linear(768, 768),
                nn.ELU(),
                nn.Dropout(0.4),
            )

        if variant == "wo_sp":
            self.modality_projections = nn.ModuleDict(
                {name: projection() for name in self.active_modalities}
            )
            self.additional_modalities_projection = None
        else:
            self.additional_modalities_projection = projection()
            self.modality_projections = None

        # Instantiate the CRMD layer for every variant so common downstream
        # layers keep a stable declaration order. wo_crmd skips it in forward.
        self.cross_modal_attention = CrossModalAttention(
            feature_dim=256,
            num_modalities=len(self.active_modalities),
        )
        self.sentimentGRU = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )
        self.sarcasmGRU = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )
        self.directSent = nn.Linear(256, 3)
        self.directSar = nn.Linear(512, 2)
        self.sent_gate = nn.Linear(256, 1)
        self.sar_gate_x = nn.Linear(512, 1)
        self.sar_gate_sent = nn.Linear(256, 1)
        self.sar_gate_sar = nn.Linear(256, 1)

    def _project_modality(self, name: str, graph: dgl.DGLGraph):
        features = graph.ndata["features"]
        if self.modality_projections is None:
            features = self.additional_modalities_projection(features)
        else:
            features = self.modality_projections[name](features)
        features = self.gat_layer(graph, features)
        graph.ndata["ablation_mcm_features"] = features
        return dgl.mean_nodes(graph, "ablation_mcm_features")

    def forward(self, g, text_graph, image_graph, audio_graph=None):
        graphs = {"text": text_graph, "image": image_graph}
        if self.dataset == "mustard":
            if audio_graph is None:
                raise ValueError("MUStARD requires an audio graph")
            graphs["audio"] = audio_graph

        x_shared = self.mm_embedding_space(g.ndata["features"])
        x_fused = self.gat_layer(g, x_shared).mean(1)
        g.ndata["ablation_global_features"] = x_fused
        global_reference = dgl.mean_nodes(g, "ablation_global_features")

        modality_summaries = [
            self._project_modality(name, graphs[name])
            for name in self.active_modalities
        ]
        js_divergence = generalized_js_divergence(modality_summaries)

        if self.variant == "wo_crmd":
            cross_reference = torch.zeros_like(global_reference)
        else:
            cross_nodes = []
            node_groups = torch.split(
                x_fused,
                g.batch_num_nodes().detach().cpu().tolist(),
                dim=0,
            )
            for sample_index, shared_nodes in enumerate(node_groups):
                sample_modalities = [
                    item[sample_index].unsqueeze(0)
                    for item in modality_summaries
                ]
                cross_nodes.append(
                    self.cross_modal_attention(
                        shared_nodes.unsqueeze(0), sample_modalities
                    ).squeeze(0)
                )
            g.ndata["ablation_cross_features"] = torch.cat(cross_nodes, dim=0)
            cross_reference = dgl.mean_nodes(g, "ablation_cross_features")

        sequence = global_reference.unsqueeze(1)
        _, sentiment_hidden = self.sentimentGRU(sequence)
        if self.variant == "wo_gate":
            sentiment_features = sentiment_hidden[-1]
        else:
            sentiment_gate = torch.sigmoid(self.sent_gate(sentiment_hidden[-1]))
            sentiment_features = sentiment_gate * sentiment_hidden[-1]
        sentiment_output = self.directSent(sentiment_features)

        sarcasm_input = torch.cat(
            [sequence, cross_reference.unsqueeze(1)], dim=2
        )
        if self.variant == "wo_gate":
            gated_sarcasm_input = sarcasm_input
        else:
            input_gate = torch.sigmoid(self.sar_gate_x(sarcasm_input))
            gated_sarcasm_input = input_gate * sarcasm_input
        _, sarcasm_hidden = self.sarcasmGRU(gated_sarcasm_input)

        if self.variant == "wo_gate":
            sarcasm_features = torch.cat(
                [sarcasm_hidden[-1], sentiment_hidden[-1]], dim=1
            )
        else:
            sentiment_gate = torch.sigmoid(
                self.sar_gate_sent(sentiment_hidden[-1])
            )
            sarcasm_gate = torch.sigmoid(
                self.sar_gate_sar(sarcasm_hidden[-1])
            )
            sarcasm_features = torch.cat(
                [
                    sarcasm_gate * sarcasm_hidden[-1],
                    sentiment_gate * sentiment_hidden[-1],
                ],
                dim=1,
            )
        sarcasm_output = self.directSar(sarcasm_features)
        return sarcasm_output, sentiment_output, js_divergence
