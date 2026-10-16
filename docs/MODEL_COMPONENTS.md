# Implementation map

| Paper component | Release implementation |
|---|---|
| shared multimodal projection | `MModel.mm_embedding_space` |
| shared single-modality projection | `MModel.additional_modalities_projection` |
| graph encoding / global reference | shared `MModel.gat_layer`, followed by mean over attention heads |
| CRMD interaction | `CrossModalAttention.forward` and `calculate_attention` |
| modality discrepancy / MCM signal | `jensen_shannon_divergence` and `mcm_margin_loss` |
| task gates | `sent_gate`, `sar_gate_x`, `sar_gate_sent`, `sar_gate_sar` |

Text and visual encoder outputs are 768-dimensional. MUStARD acoustic features
are initially 128-dimensional and are aligned to 768 dimensions during offline
feature preparation; the cached node arrays consumed by this training code are
therefore 768-dimensional. Both projection modules map `768 -> 768`; they are
separate parameter sets. A four-head GAT maps each node to four
256-dimensional head outputs. Averaging over the head axis gives the
256-dimensional global reference used by CRMD.

`CrossModalAttention.transform_layer` is one shared `Linear(256, 256)` used to
form Q, K, and V for the reference and modality inputs. Attention is
`softmax(Q K^T / sqrt(256), dim=-1) V`. Its learnable modality coefficients
are scalars normalized with softmax; the resulting deviation mask is a
256-dimensional sigmoid vector per reference node. The downstream task gates
produce one scalar per sample and broadcast it across the corresponding
feature vector.
