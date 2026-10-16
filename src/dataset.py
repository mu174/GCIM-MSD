# Importing libraries
import pandas as pd
import dgl
from dgl.data import DGLDataset
from torch import nn
import torch
import numpy as np
import csv
import json
from pathlib import Path


def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='latin') as f:
        reader = csv.reader(f)
        uttNameList = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                uttNameList.append(i[0])
        uttNameList = list(set(uttNameList))
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = ''
            uttDict[name]['sentiment-label'] = ''
            uttDict[name]['emotion-label'] = ''
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='latin') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'] = item[4]
            uttDict[item[0]]['sentiment-label'] = item[5]
            uttDict[item[0]]['emotion-label'] = item[7]
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList


class GraphDataset_Memotion(DGLDataset):
    def __init__(self, df, root_dir, image_vec_dir, text_vec_dir, dataset_name="GraphDataset"):
        """Create a multimodal graph dataset for Memotion.

        Args:
            df (pd.DataFrame)
            root_dir (str)
            image_id (str)
            text_id (str)
            image_vec_dir (str)
            text_vec_dir (str)
            dataset_name (str, optional). Defaults to "GraphDataset".
        """
        super(GraphDataset_Memotion, self).__init__(name=dataset_name,
                                           verbose=True)

        # Main CSV file
        self.df = df

        # Base data folder
        self.root_dir = root_dir

        # directory that contains node embeddings for image graph
        self.image_vec_dir = image_vec_dir

        # directory that contains node embeddings for image graph
        self.text_vec_dir = text_vec_dir

        # to resize the imagefeature vector from pre-trained model
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(768)

    def __len__(self):
        return len(self.df)

    def load_dependency_tree(self, filename):
        """
        从 JSON 文件中加载给定 utterance 名称的依存树。
        """
        dep_tree_filename = f"{self.root_dir}{self.text_vec_dir}{filename}_dep_tree.json"
        with open(dep_tree_filename, 'r', encoding='utf-8') as f:
            dep_tree = json.load(f)
        return dep_tree

    def __getitem__(self, idx):
        """

        Args:
            idx : index of sample to be created

        Returns:
            dgl.graph: multimodal graph for a news post
            torch.tensor(): classification label corresponding to the news post
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        # filenames for the index
        file_name = row.iloc[1].split(".")[0]
        dep_tree = self.load_dependency_tree(file_name)
        # print(file_name)
        # file_name = self.df['image_name'][idx].split(".")[0]
        # print(file_name)

        # Load full image node embedding
        image_vec_full = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name}_full_image.npy')

        # Load node embeddings for objects present in the image
        try:
            image_vec = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name}.npy')
            all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
            # image_vec = all_image_vec
        except:
            image_vec = image_vec_full

        # Resize the image vectors to match the text embedding dimension
        # image_vec = self.adaptive_pooling(torch.tensor(image_vec).float().unsqueeze(0)).squeeze(0)
        # print(image_vec.shape)

        # 创建图像图
        num_local_image_nodes = image_vec.shape[0]
        num_global_image_nodes = image_vec_full.shape[0]

        # 节点ID
        local_image_node_ids = torch.arange(num_local_image_nodes)
        global_image_node_ids = torch.arange(num_local_image_nodes, num_local_image_nodes + num_global_image_nodes)

        # 边：局部图像特征节点与局部图像特征节点之间的边
        local_image_edges = (torch.arange(num_local_image_nodes).repeat(num_local_image_nodes),
                             torch.arange(num_local_image_nodes).repeat_interleave(num_local_image_nodes))

        # 边：全局图像特征节点与全局图像特征节点之间的边
        global_image_edges = (torch.arange(num_global_image_nodes).repeat(num_global_image_nodes),
                              torch.arange(num_global_image_nodes).repeat_interleave(num_global_image_nodes))

        # 边：局部图像特征节点与全局图像特征节点之间的边
        local_to_global_image_edges = (local_image_node_ids.repeat(num_global_image_nodes),
                                       global_image_node_ids.repeat(num_local_image_nodes))

        # 创建图
        edges = torch.cat([local_image_edges[0], global_image_edges[0], local_to_global_image_edges[0]]), \
            torch.cat([local_image_edges[1], global_image_edges[1], local_to_global_image_edges[1]])

        image_graph = dgl.graph(edges)

        all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)

        # 添加节点特征
        image_all_vec = torch.cat([
            torch.tensor(image_vec).float(),
            torch.tensor(image_vec_full).float()
        ])

        image_graph.ndata['features'] = image_all_vec

        # Adding text modality in Graph Dict
        # Load node embeddings for tokens present in the text
        text_vec = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name}.npy')

        # Load full image node embedding
        text_vec_full = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name}_full_text.npy')

        sentence_nodes_to_keep = []
        sentence_edges_src = []
        sentence_edges_dst = []

        # 过滤标点符号节点并保留特征
        non_punct_indices = []
        for parent_idx, token in enumerate(dep_tree):
            if token['dep'] != "punct":  # 保留非标点符号节点
                sentence_nodes_to_keep.append(parent_idx)  # 仅保留非标点符号节点
                non_punct_indices.append(parent_idx)

                # 添加非标点符号节点的依存关系
                for child in token.get('children', []):
                    if child['dep'] != "punct":  # 排除标点符号的子节点
                        # 使用 parent_idx 和 child_idx 作为边的源和目标
                        child_idx = next((idx for idx, t in enumerate(dep_tree) if t == child), None)
                        if child_idx is not None:
                            sentence_edges_src.append(len(non_punct_indices) - 1)  # 父节点
                            sentence_edges_dst.append(next(
                                (i for i, idx in enumerate(non_punct_indices) if idx == child_idx),
                                None))  # 子节点
            else:
                # 对于标点符号节点，仅保留其特征，不添加边
                pass

        # 检查 non_punct_indices 中的索引是否有效
        valid_non_punct_indices = [idx for idx in non_punct_indices if idx < text_vec.shape[0]]
        # 删除 text_vec 中对应的标点符号节点的特征
        text_vec = text_vec[valid_non_punct_indices]

        # 句子的整体特征（即句子的“超级节点”）
        sentence_vec = np.concatenate([text_vec_full, text_vec], axis=0)
        # sentence_vec = self.adaptive_pooling(torch.tensor(sentence_vec).float().unsqueeze(0)).squeeze(0)

        # 确保每个 token 与句子的超级节点连接
        for token_idx in range(len(sentence_nodes_to_keep)):
            sentence_edges_src.append(token_idx)
            sentence_edges_dst.append(len(sentence_nodes_to_keep))  # 句子的“超级节点”

        # 如果没有依存关系的边（所有依存关系都被过滤掉了），确保每个 token 与自己连接
        if not sentence_edges_src and not sentence_edges_dst:
            for idx in range(len(sentence_nodes_to_keep)):
                sentence_edges_src.append(idx)
                sentence_edges_dst.append(idx)  # 自连接

        # 创建文本图
        text_edges_src = torch.tensor(sentence_edges_src, dtype=torch.int64)
        text_edges_dst = torch.tensor(sentence_edges_dst, dtype=torch.int64)
        text_all_edges = (text_edges_src, text_edges_dst)
        text_all_vec = torch.tensor(sentence_vec, dtype=torch.float)

        text_graph = dgl.graph(text_all_edges)

        # 如果特征数量少了
        if text_graph.num_nodes() > len(text_all_vec):
            num_missing = text_graph.num_nodes() - len(text_all_vec)
            # 假设特征维度是 feature_dim
            feature_dim = text_all_vec.shape[1]
            missing_features = torch.zeros(num_missing, feature_dim, dtype=torch.float)
            text_all_features = torch.cat([text_all_vec, missing_features])
        else:
            text_all_features = text_all_vec

        # 如果特征数量多了
        if text_graph.num_nodes() < len(text_all_features):
            text_all_features = text_all_features[:text_graph.num_nodes()]

        # 为图像图中入度为 0 的节点添加自环
        in_degrees = text_graph.in_degrees()
        zero_in_degree_nodes = torch.nonzero(in_degrees == 0).squeeze()
        if zero_in_degree_nodes.numel() > 0:
            text_graph.add_edges(zero_in_degree_nodes, zero_in_degree_nodes)

        text_graph.ndata['features'] = text_all_features

        # 合并所有特征
        all_vec = torch.cat([text_all_features, image_all_vec], dim=0)
        # print(all_vec.shape)

        # 节点数量
        num_text_nodes = text_all_features.shape[0]
        num_image_nodes = image_all_vec.shape[0]

        # 计算图像和音频节点的索引偏移
        image_offset = num_text_nodes

        # 1. 创建自模态边（不需要改变）
        text_edges = text_graph.edges()
        image_edges = (image_graph.edges()[0] + image_offset, image_graph.edges()[1] + image_offset)

        # 2. 创建跨模态边，并为非文本模态增加偏移
        text_to_image_edges = (torch.arange(num_text_nodes).unsqueeze(1).repeat(1, num_image_nodes).view(-1),
                               torch.arange(num_image_nodes).unsqueeze(0).repeat(num_text_nodes, 1).view(
                                   -1) + image_offset)

        # 3. 合并所有边
        u = torch.cat([text_edges[0], image_edges[0], text_to_image_edges[0]])

        v = torch.cat([text_edges[1], image_edges[1], text_to_image_edges[1]])

        # 4. 创建图，包含所有节点和边
        g = dgl.graph((u, v), num_nodes=num_text_nodes + num_image_nodes)

        # 添加节点特征
        g.ndata['features'] = all_vec

        # find the label
        sarcasm = row['sarcasm']
        sentiment = row['overall_sentiment']

        return g, text_graph, image_graph, sarcasm, sentiment


class GraphDataset_Mustard(DGLDataset):
    def __init__(self, datatye, root_dir, image_vec_dir, text_vec_dir, audio_vec_dir, dataset_name="GraphDataset"):
        super(GraphDataset_Mustard, self).__init__(name=dataset_name, verbose=True)
        self.datatye = datatye
        if self.datatye not in {'train', 'dev', 'test'}:
            raise ValueError(f"Unknown MUStARD split: {self.datatye}")
        datafile = str(
            Path(root_dir) / 'mustard' / 'splits' / f'{self.datatye}.csv'
        )
        self.uttDict, self.uttNameList = readcsv(datafile)
        self.uttList = list(self.uttDict.values())

        self.root_dir = root_dir
        self.image_vec_dir = image_vec_dir
        self.text_vec_dir = text_vec_dir
        self.audio_vec_dir = audio_vec_dir
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(768)

        self.graphs = []
        self.text_graph = []
        self.image_graph = []
        self.audio_graph = []

        self.Dep_tree = self.load_dependency_tree(datatye)

        for idx in range(len(self.uttNameList)):
            uttName = self.uttNameList[idx]
            text = list(self.uttDict[uttName]['utterance'])
            # print(len(text))
            filename_image = f'{self.root_dir}{self.image_vec_dir}' + uttName
            filename_text = f'{self.root_dir}{self.text_vec_dir}' + uttName
            filename_audio = f'{self.root_dir}{self.audio_vec_dir}' + datatye + '/' + uttName
            # 初始化全局节点特征和边的容器
            all_image_vecs = []
            all_image_edges_src = []
            all_image_edges_dst = []

            all_audio_edge_src = []
            all_audio_edge_dst = []
            all_audio_vecs = []

            all_text_vecs = []
            all_text_edges_src = []
            all_text_edges_dst = []

            image_node_offset = 0  # 用来记录当前图的节点ID偏移
            text_node_offset = 0
            audio_node_offset = 0

            # 通过加载的依存树文件获取依存树
            dep_tree = self.Dep_tree[uttName]

            # 处理每个句子
            for i in range(len(text)):
                # 加载图像、音频和文本特征
                image_vec_full = np.load(f"{filename_image}_{i}_full_image.npy")
                image_vec_full = image_vec_full.astype(np.float32)  # Ensure correct type

                try:
                    image_vec = np.load(f"{filename_image}_{i}.npy")
                    image_vec = image_vec.astype(np.float32)  # Ensure correct type
                except FileNotFoundError:
                    image_vec = image_vec_full

                # image_vec = self.adaptive_pooling(torch.tensor(image_vec).float().unsqueeze(0)).squeeze(0)

                text_vec_full = np.load(f"{filename_text}_{i}_full_text.npy")
                text_vec = np.load(f"{filename_text}_{i}.npy")

                # 获取当前句子的依存树
                sentence_dep_tree = dep_tree[i]  # 当前句子的依存树

                sentence_nodes_to_keep = []
                sentence_edges_src = []
                sentence_edges_dst = []

                # 过滤标点符号节点并保留特征
                non_punct_indices = []
                for parent_idx, token in enumerate(sentence_dep_tree):
                    if token['dep'] != "punct":  # 保留非标点符号节点
                        sentence_nodes_to_keep.append(parent_idx)  # 仅保留非标点符号节点
                        non_punct_indices.append(parent_idx)

                        # 添加非标点符号节点的依存关系
                        for child in token.get('children', []):
                            if child['dep'] != "punct":  # 排除标点符号的子节点
                                # 使用 parent_idx 和 child_idx 作为边的源和目标
                                child_idx = next((idx for idx, t in enumerate(sentence_dep_tree) if t == child), None)
                                if child_idx is not None:
                                    sentence_edges_src.append(text_node_offset + len(non_punct_indices) - 1)  # 父节点
                                    sentence_edges_dst.append(text_node_offset + next(
                                        (i for i, idx in enumerate(non_punct_indices) if idx == child_idx),
                                        None))  # 子节点
                    else:
                        # 对于标点符号节点，仅保留其特征，不添加边
                        pass

                # 检查 non_punct_indices 中的索引是否有效
                valid_non_punct_indices = [idx for idx in non_punct_indices if idx < text_vec.shape[0]]
                # 删除 text_vec 中对应的标点符号节点的特征
                text_vec = text_vec[valid_non_punct_indices]

                # 句子的整体特征（即句子的“超级节点”）
                sentence_vec = np.concatenate([text_vec_full, text_vec], axis=0)
                # sentence_vec = self.adaptive_pooling(torch.tensor(sentence_vec).float().unsqueeze(0)).squeeze(0)

                # 确保每个 token 与句子的超级节点连接
                for token_idx in range(len(sentence_nodes_to_keep)):
                    sentence_edges_src.append(text_node_offset + token_idx)
                    sentence_edges_dst.append(text_node_offset + len(sentence_nodes_to_keep))  # 句子的“超级节点”

                # 如果没有依存关系的边（所有依存关系都被过滤掉了），确保每个 token 与自己连接
                if not sentence_edges_src and not sentence_edges_dst:
                    for idx in range(len(sentence_nodes_to_keep)):
                        sentence_edges_src.append(text_node_offset + idx)
                        sentence_edges_dst.append(text_node_offset + idx)  # 自连接

                # 将句子内部的依存树边添加到全局边容器
                all_text_edges_src.append(torch.tensor(sentence_edges_src))
                all_text_edges_dst.append(torch.tensor(sentence_edges_dst))

                # 为句子的超级节点添加特征
                all_text_vecs.append(torch.tensor(sentence_vec, dtype=torch.float))

                # 更新偏移量，确保下一个句子的节点ID不重叠
                num_nodes_in_sentence = len(sentence_nodes_to_keep) + 1  # +1 for the super node of the sentence
                text_node_offset += num_nodes_in_sentence

                # 处理句子之间的连接（超级节点之间全连接）
                if i > 0:  # 从第二个句子开始，才能连接前一个句子
                    prev_sentence_offset = text_node_offset - num_nodes_in_sentence  # 前一个句子的节点偏移
                    sentence_edges_src = [prev_sentence_offset - 1]
                    sentence_edges_dst = [text_node_offset - 1]
                    all_text_edges_src.append(torch.tensor(sentence_edges_src))
                    all_text_edges_dst.append(torch.tensor(sentence_edges_dst))

                # ---------------------------- 图像图构建 ---------------------------
                num_local_image_nodes = image_vec.shape[0]
                num_global_image_nodes = image_vec_full.shape[0]

                # 节点ID (带有偏移)
                local_image_node_ids = torch.arange(num_local_image_nodes) + image_node_offset
                global_image_node_ids = torch.arange(num_local_image_nodes,
                                                     num_local_image_nodes + num_global_image_nodes) + image_node_offset

                # 边：局部图像特征节点与局部图像特征节点之间的边
                local_image_edges = (local_image_node_ids.repeat(num_local_image_nodes),
                                     local_image_node_ids.repeat_interleave(num_local_image_nodes))

                # 边：全局图像特征节点与全局图像特征节点之间的边
                global_image_edges = (global_image_node_ids.repeat(num_global_image_nodes),
                                      global_image_node_ids.repeat_interleave(num_global_image_nodes))

                # 边：局部图像特征节点与全局图像特征节点之间的边
                local_to_global_image_edges = (local_image_node_ids.repeat(num_global_image_nodes),
                                               global_image_node_ids.repeat(num_local_image_nodes))

                # 累积边 (将每次循环生成的边存入全局边容器)
                all_image_edges_src.append(
                    torch.cat([local_image_edges[0], global_image_edges[0], local_to_global_image_edges[0]]))
                all_image_edges_dst.append(
                    torch.cat([local_image_edges[1], global_image_edges[1], local_to_global_image_edges[1]]))

                all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
                image_vec = all_image_vec

                # 累积节点特征 (将每次循环生成的节点特征存入全局特征容器)
                all_image_vecs.append(torch.tensor(image_vec).float())

                # 更新偏移量 (确保下一个图的节点ID不重叠)
                image_node_offset += num_local_image_nodes + num_global_image_nodes

            # Create text graph
            text_edges_src = torch.cat(all_text_edges_src)
            text_edges_dst = torch.cat(all_text_edges_dst)
            text_all_edges = (text_edges_src, text_edges_dst)
            text_all_features = torch.cat(all_text_vecs)

            text_graph = dgl.graph(text_all_edges)
            # print(text_graph.num_nodes())
            # print(text_node_offset)
            # 如果特征数量少了
            if text_graph.num_nodes() > len(text_all_features):
                num_missing = text_graph.num_nodes() - len(text_all_features)
                # 假设特征维度是 feature_dim
                feature_dim = text_all_features.shape[1]
                missing_features = torch.zeros(num_missing, feature_dim, dtype=torch.float)
                text_all_features = torch.cat([text_all_features, missing_features])

            # 如果特征数量多了
            if text_graph.num_nodes() < len(text_all_features):
                text_all_features = text_all_features[:text_graph.num_nodes()]

            text_graph.ndata['features'] = text_all_features

            # 为图像图中入度为 0 的节点添加自环
            in_degrees = text_graph.in_degrees()
            zero_in_degree_nodes = torch.nonzero(in_degrees == 0).squeeze()
            if zero_in_degree_nodes.numel() > 0:
                text_graph.add_edges(zero_in_degree_nodes, zero_in_degree_nodes)

            self.text_graph.append(text_graph)

            # Create image graph
            image_edges_src = torch.cat(all_image_edges_src)
            image_edges_dst = torch.cat(all_image_edges_dst)
            image_all_edges = (image_edges_src, image_edges_dst)
            image_all_features = torch.cat(all_image_vecs)

            image_graph = dgl.graph(image_all_edges)
            image_graph.ndata['features'] = image_all_features
            #
            # # 为图像图中入度为 0 的节点添加自环
            # in_degrees = image_graph.in_degrees()
            # zero_in_degree_nodes = torch.nonzero(in_degrees == 0).squeeze()
            # if zero_in_degree_nodes.numel() > 0:
            #     image_graph.add_edges(zero_in_degree_nodes, zero_in_degree_nodes)

            self.image_graph.append(image_graph)

            # Create audio graph
            audio_vec_full = np.loadtxt(
                filename_audio,
                dtype=float,
                delimiter=",")
            audio_vec_full = audio_vec_full.astype(np.float32)
            # 创建音频图
            num_audio_nodes = audio_vec_full.shape[0]
            audio_node_ids = torch.arange(num_audio_nodes) + audio_node_offset

            # 边：音频特征节点与自身之间的边
            audio_edges = (audio_node_ids.repeat(num_audio_nodes),
                           audio_node_ids.repeat_interleave(num_audio_nodes))

            # 累积边
            all_audio_edge_src.append(audio_edges[0])
            all_audio_edge_dst.append(audio_edges[1])

            # 累积音频特征
            all_audio_vecs.append(torch.tensor(audio_vec_full).float())

            # 更新偏移量
            audio_node_offset += num_audio_nodes

            # 创建全局音频图
            audio_edges_src = torch.cat(all_audio_edge_src)
            audio_edges_dst = torch.cat(all_audio_edge_dst)
            audio_all_edges = (audio_edges_src, audio_edges_dst)
            audio_all_features = torch.cat(all_audio_vecs)

            audio_graph = dgl.graph(audio_all_edges)

            # in_degrees = audio_graph.in_degrees()
            # zero_in_degree_nodes = torch.nonzero(in_degrees == 0).squeeze()
            # if zero_in_degree_nodes.numel() > 0:
            #     audio_graph.add_edges(zero_in_degree_nodes, zero_in_degree_nodes)

            audio_graph.ndata['features'] = audio_all_features

            self.audio_graph.append(audio_graph)

            # 合并所有特征
            all_vec = torch.cat([text_all_features, image_all_features, audio_all_features], dim=0)
            # print(all_vec.shape)

            # 节点数量
            num_text_nodes = text_all_features.shape[0]
            num_image_nodes = image_all_features.shape[0]
            num_audio_nodes = audio_all_features.shape[0]

            # 计算图像和音频节点的索引偏移
            image_offset = num_text_nodes
            audio_offset = num_text_nodes + num_image_nodes

            # 1. 创建自模态边（不需要改变）
            text_edges = text_graph.edges()
            image_edges = (image_graph.edges()[0] + image_offset, image_graph.edges()[1] + image_offset)
            audio_edges = (audio_graph.edges()[0] + audio_offset, audio_graph.edges()[1] + audio_offset)

            # 2. 创建跨模态边，并为非文本模态增加偏移
            text_to_image_edges = (torch.arange(num_text_nodes).unsqueeze(1).repeat(1, num_image_nodes).view(-1),
                                   torch.arange(num_image_nodes).unsqueeze(0).repeat(num_text_nodes, 1).view(
                                       -1) + image_offset)

            text_to_audio_edges = (torch.arange(num_text_nodes).unsqueeze(1).repeat(1, num_audio_nodes).view(-1),
                                   torch.arange(num_audio_nodes).unsqueeze(0).repeat(num_text_nodes, 1).view(
                                       -1) + audio_offset)

            image_to_audio_edges = (
                torch.arange(num_image_nodes).unsqueeze(1).repeat(1, num_audio_nodes).view(-1) + image_offset,
                torch.arange(num_audio_nodes).unsqueeze(0).repeat(num_image_nodes, 1).view(-1) + audio_offset)

            # 3. 合并所有边
            u = torch.cat([text_edges[0], image_edges[0], audio_edges[0],
                           text_to_image_edges[0], text_to_audio_edges[0], image_to_audio_edges[0]])

            v = torch.cat([text_edges[1], image_edges[1], audio_edges[1],
                           text_to_image_edges[1], text_to_audio_edges[1], image_to_audio_edges[1]])

            # 4. 创建图，包含所有节点和边
            g = dgl.graph((u, v), num_nodes=num_text_nodes + num_image_nodes + num_audio_nodes)

            # 添加节点特征
            g.ndata['features'] = all_vec

            self.graphs.append(g)

    def load_dependency_tree(self, datatye):
        """
        从 JSON 文件中加载给定 utterance 名称的依存树。
        """
        dep_tree_filename = f"{self.root_dir}{self.text_vec_dir}dependency_graph_{datatye}.json"
        with open(dep_tree_filename, 'r', encoding='utf-8') as f:
            dep_tree = json.load(f)
        return dep_tree

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        sarcasmStr = self.uttList[idx]['sarcasm-label']
        sentimentStr = self.uttList[idx]['sentiment-label']

        if 'True' == sarcasmStr:
            sarcasmLabel = np.array([0, 1], dtype=np.int8)
        else:
            sarcasmLabel = np.array([1, 0], dtype=np.int8)

        sentimentLabel = np.zeros(3, dtype=np.int8)
        if -1 == int(sentimentStr):
            sentimentLabel[0] = 1
        elif 0 == int(sentimentStr):
            sentimentLabel[1] = 1
        else:
            sentimentLabel[2] = 1

        return self.graphs[idx], self.text_graph[idx], self.image_graph[idx], self.audio_graph[idx], sarcasmLabel, sentimentLabel
