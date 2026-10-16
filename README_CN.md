# GCIM-MSD

**Global Context-Guided Incongruity Modeling for Multimodal Sarcasm Detection** 的实现，面向 MUStARD 和 Memotion 的讽刺检测与辅助情感任务。

![GCIM-MSD 模型结构](assets/figure2.png)

## 环境

记录环境为 Python 3.10.14、PyTorch 1.12.1+cu113、DGL 0.9.1.post1 和 Transformers 4.30.2。先安装与平台匹配的 PyTorch 和 DGL，再运行：

```bash
pip install -r requirements.txt
```

## 数据

本仓库不提供原始数据、数据划分文件、缓存特征和训练权重。训练前需在本地准备数据和特征，并按照 [docs/DATASETS.md](docs/DATASETS.md) 进行检查：

```bash
python scripts/preflight.py /数据根目录
```

## 训练

```bash
bash scripts/train_mustard.sh /数据根目录
bash scripts/train_memotion.sh /数据根目录
```

启动器读取 `configs/<dataset>.yaml`，默认使用 CPU，并在新的输出目录保存源码快照和运行设置。checkpoint 按验证集联合指标选择：

```text
0.7 × sarcasm weighted F1 + 0.3 × sentiment weighted F1
```

消融和模态移除实验：

```bash
bash scripts/train_ablation_suite.sh mustard /数据根目录
bash scripts/train_ablation_suite.sh memotion /数据根目录
```

## 评估

评估需要使用本地训练得到的兼容 checkpoint。

## 主结果

以下为固定测试集上的 support-weighted 指标。

| Dataset | Sarcasm P | Sarcasm R | Sarcasm F1 | Sentiment R | Sentiment F1 |
|---|---:|---:|---:|---:|---:|
| MUStARD | 86.04 | 86.00 | 85.98 | 47.00 | 48.39 |
| Memotion | 69.08 | 76.05 | 71.56 | 55.15 | 49.30 |

完整配置与实验设置见 `configs/` 和 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。仓库维护者：Jie Tang（`teganjay0715@gmail.com`）。代码采用 Apache License 2.0。
