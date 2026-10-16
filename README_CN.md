# GCIM-MSD

**Global Context-Guided Incongruity Modeling for Multimodal Sarcasm Detection** 的实现，面向 MUStARD 和 Memotion 的讽刺检测与辅助情感任务。

![GCIM-MSD 模型结构](assets/figure2.png)

MUStARD 使用文本、视觉和音频模态；Memotion 使用文本和视觉模态。

## 环境

核心环境为 Python 3.10.14、PyTorch 1.12.1+cu113、DGL 0.9.1.post1 和 Transformers 4.30.2。先安装与平台匹配的 PyTorch 和 DGL，再运行：

```bash
pip install -r requirements.txt
```

## 数据

按照 [docs/DATASETS.md](docs/DATASETS.md) 准备处理后的数据划分和模态特征，然后运行检查：

```bash
python scripts/preflight.py /处理后数据目录
```

## 训练

```bash
bash scripts/train_mustard.sh /处理后数据目录
bash scripts/train_memotion.sh /处理后数据目录
```

启动器读取 `configs/<dataset>.yaml`，默认使用 CPU，并为每次运行创建新的输出目录。checkpoint 按验证集联合指标选择：

```text
0.7 × sarcasm weighted F1 + 0.3 × sentiment weighted F1
```

## GCIM-MSD 结果

以下为测试集上的 support-weighted 指标。

| Dataset | Sarcasm P | Sarcasm R | Sarcasm F1 | Sentiment R | Sentiment F1 |
|---|---:|---:|---:|---:|---:|
| MUStARD | 86.04 | 86.00 | 85.98 | 47.00 | 48.39 |
| Memotion | 69.08 | 76.05 | 71.56 | 55.15 | 49.30 |

完整配置与实验设置见 `configs/` 和 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。仓库维护邮箱：`teganjay0715@gmail.com`。代码采用 Apache License 2.0。
