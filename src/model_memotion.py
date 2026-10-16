# Importing libraries
import numpy as np
import dgl
from dgl.nn.pytorch import conv
import torch
from torch import nn
import torch.nn.functional as F


# 模态间不一致学习模块
def calculate_js_divergence(text_distribution, image_distribution):
    """Return normalized T/V JS divergence for every graph in the batch.

    Inputs have shape ``[batch, heads, feature_dim]``.  Normalization is over
    the attention-head axis and gradients are preserved.
    """
    distributions = torch.stack(
        [text_distribution, image_distribution], dim=1
    ).float()  # [batch, modalities, heads, feature_dim]
    probabilities = F.softmax(distributions, dim=2)
    mean_probability = probabilities.mean(dim=1, keepdim=True)

    eps = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp_min(eps)
    mean_probability = mean_probability.clamp_min(eps)
    kl = (
        probabilities
        * (probabilities.log() - mean_probability.log())
    ).sum(dim=2)
    return kl.mean(dim=(1, 2)) / np.log(2.0)


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim=256, num_modalities=2):
        super().__init__()
        # 保存输入参数
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        # 定义查询、键、值的线性变换层
        self.transform_layer = nn.Linear(feature_dim, feature_dim)
        # 定义注意力分数的 softmax 层
        self.softmax = nn.Softmax(dim=-1)
        # 定义特征投影层
        self.projection_layer = nn.Linear(num_modalities * feature_dim, feature_dim)
        # 定义可学习的模态权重
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        # 定义 LayerNorm 用于归一化
        self.layer_norm = nn.LayerNorm(feature_dim)

    def calculate_attention(self, query, key, value):
        """计算注意力特征"""
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        # 应用 softmax 函数得到注意力权重
        attention_weights = self.softmax(attention_scores)
        # 计算注意力特征
        return torch.matmul(attention_weights, value)

    def generate_mask(self, attended, attended_modalities):
        """根据各模态的注意力特征与共享特征的差异生成掩码"""
        # 计算各模态注意力特征与共享特征的差异
        differences = [torch.abs(attended - attended_modality) for attended_modality in attended_modalities]
        # 对差异进行 LayerNorm 归一化处理
        norm_differences = [self.layer_norm(diff) for diff in differences]
        # 对模态权重进行 softmax 处理，确保权重之和为 1
        weights = nn.functional.softmax(self.modality_weights, dim=0)
        # 结合各模态的权重计算综合差异
        combined_diff = sum(w * diff for w, diff in zip(weights, norm_differences))
        # 应用 sigmoid 函数生成掩码
        return torch.sigmoid(combined_diff)

    def forward(self, shared_features, text_features, image_features):
        # 检查输入特征的维度
        for features in [shared_features, text_features, image_features]:
            assert features.size(-1) == self.feature_dim, f"Input feature dimension must be {self.feature_dim}"

        features = [shared_features, text_features, image_features]
        # 对所有特征进行变换
        transformed_features = [self.transform_layer(f) for f in features]

        query = transformed_features[0]
        keys = transformed_features[1:]
        values = transformed_features[1:]

        # 计算共享特征的注意力
        attended = self.calculate_attention(query, query, query)
        # 计算各模态的注意力
        attended_modalities = [self.calculate_attention(query, key, value) for key, value in zip(keys, values)]

        # 根据综合差异生成掩码
        mask = self.generate_mask(attended, attended_modalities)

        # 动态调整特征的贡献
        adjusted_modalities = [attended_modality * mask for attended_modality in attended_modalities]

        # 合并各模态的注意力特征
        combined_features = torch.cat(adjusted_modalities, dim=-1)
        # 投影到最终维度
        return self.projection_layer(combined_features)


class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()

        # Projection layer for shared space
        self.mm_embedding_space = nn.Sequential(
            nn.Linear(768, 768),
            nn.ELU(),
            nn.Dropout(0.4)
        )

        # Multi-head GAT Layer for shared space
        self.gat_layer = conv.GATConv(
            in_feats=768,
            out_feats=256,
            num_heads=4,
            feat_drop=0.5,
            attn_drop=0.0,
            residual=True,
            activation=nn.ELU()
        )

        # Projection layers for additional modalities
        self.additional_modalities_projection =nn.Sequential(
                nn.Linear(768, 768),
                nn.ELU(),
                nn.Dropout(0.4)
            )

        # Cross-modal attention layer
        self.cross_modal_attention = CrossModalAttention()  # Adjust input dimension accordingly

        # Multi-task decoder module
        self.sentimentGRU = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.sarcasmGRU = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.directSent = nn.Linear(
            in_features=256, out_features=3)

        self.directSar = nn.Linear(
            in_features=512, out_features=2)

        # 情感分析的门控机制
        self.sent_gate = nn.Linear(256, 1)
        # 讽刺检测的门控机制
        self.sar_gate_x = nn.Linear(512, 1)
        self.sar_gate_sent = nn.Linear(256, 1)
        self.sar_gate_sar = nn.Linear(256, 1)

    def forward(self, g, text_graph, image_graph):
        x = g.ndata['features']
        text_features = text_graph.ndata['features']
        image_features = image_graph.ndata['features']

        # Project to common multimodal space
        x_shared = self.mm_embedding_space(x)

        # Apply GCN layer to shared space (multi-modal features)
        x_fused = self.gat_layer(g, x_shared)
        x_fused = x_fused.mean(1)

        # Process additional modalities separately
        text_feature_nodes = self.additional_modalities_projection(text_features)
        text_feature_nodes = self.gat_layer(text_graph, text_feature_nodes)
        text_graph.ndata['mcm_features'] = text_feature_nodes
        text_mcm_feature = dgl.mean_nodes(text_graph, 'mcm_features')

        image_feature_nodes = self.additional_modalities_projection(image_features)
        image_feature_nodes = self.gat_layer(image_graph, image_feature_nodes)
        image_graph.ndata['mcm_features'] = image_feature_nodes
        image_mcm_feature = dgl.mean_nodes(image_graph, 'mcm_features')

        g.ndata['features'] = x_fused
        x = dgl.mean_nodes(g, 'features')

        # Apply CRMD independently to each graph and use both T and V.
        x_cross_per_graph = []
        x_fused_per_graph = torch.split(
            x_fused, g.batch_num_nodes().detach().cpu().tolist(), dim=0
        )
        for batch_index, shared_nodes in enumerate(x_fused_per_graph):
            x_cross_per_graph.append(
                self.cross_modal_attention(
                    shared_nodes.unsqueeze(0),
                    text_mcm_feature[batch_index].unsqueeze(0),
                    image_mcm_feature[batch_index].unsqueeze(0),
                ).squeeze(0)
            )
        x_cross_nodes = torch.cat(x_cross_per_graph, dim=0)

        g.ndata['features'] = x_cross_nodes

        # Take mean representation of the multimodal graph
        x_cross = dgl.mean_nodes(g, 'features')

        js_divergence = calculate_js_divergence(
            text_mcm_feature, image_mcm_feature
        )
        # return sar, sent, emo
        x = x.unsqueeze(1)  # Add sequence dimension for GRU

        # Multi-task processing

        # 1. Sentiment Analysis
        sentGRU, sentHidden = self.sentimentGRU(x)
        # #
        # sentOutput = self.directSent(sentHidden[-1])

        # 情感分析的门控
        sent_gate_value = torch.sigmoid(self.sent_gate(sentHidden[-1]))
        sentOutput = self.directSent(sent_gate_value * sentHidden[-1])

        # 4. Sarcasm Detection (combine all features)
        sarInput = torch.cat([x, x_cross.unsqueeze(1)], dim=2)

        # 输入特征的门控
        sar_gate_x_value = torch.sigmoid(self.sar_gate_x(sarInput))
        sarGRU, sarHidden = self.sarcasmGRU(sar_gate_x_value * sarInput)

        # 情感隐藏状态的门控
        sar_gate_sent_value = torch.sigmoid(self.sar_gate_sent(sentHidden[-1]))
        # 讽刺检测隐藏状态的门控
        sar_gate_sar_value = torch.sigmoid(self.sar_gate_sar(sarHidden[-1]))
        sarOutput = self.directSar(
            torch.cat([sar_gate_sar_value * sarHidden[-1], sar_gate_sent_value * sentHidden[-1]], dim=1))

        # return sentOutput
        return sarOutput, sentOutput, js_divergence
