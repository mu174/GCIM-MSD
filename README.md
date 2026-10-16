<div align="center">

<h1>GCIM-MSD</h1>

<p><strong>Global Context-Guided Incongruity Modeling for Multimodal Sarcasm Detection</strong></p>

<p>This repository provides the implementation of the paper.</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 1.12.1">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0"></a>
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#key-features">Key Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#repository-structure">Repository</a> ·
  <a href="#data">Data</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#main-results">Results</a>
</p>

</div>

---

<a id="overview"></a>

## 🔎 Overview

- **Problem:** Existing methods have difficulty fully modeling cross-modal incongruity in multimodal sarcasm.
- **Task:** Multimodal sarcasm detection with auxiliary sentiment analysis on MUStARD and Memotion.
- **Core Mechanism:** GCR, CRMD, and MCM.

<a id="key-features"></a>

## ✨ Key Features

- Global Contextual Reference
- Context-Referenced Modality Deviation
- Modality-level Conflict Modeling

<a id="architecture"></a>

## 🧩 Architecture

<p align="center">
  <img src="assets/figure2.png" width="95%" alt="GCIM-MSD architecture">
</p>

<p align="center"><em>Overview of the GCIM-MSD framework.</em></p>

<a id="setup"></a>

## ⚙️ Setup

Core environment: Python 3.10.14, PyTorch 1.12.1+cu113, DGL 0.9.1.post1, and Transformers 4.30.2. Install platform-compatible PyTorch and DGL builds first, then run:

```bash
pip install -r requirements.txt
```

Package versions are listed in `requirements-recorded.txt`.

<a id="repository-structure"></a>

## 🗂️ Repository Structure

```text
GCIM-MSD_release/
├── assets/                 # Architecture figure
├── configs/                # Training and feature configurations
├── data/                   # Fixed MUStARD and Memotion split CSVs
├── docs/                   # Dataset, model, and reproducibility notes
├── scripts/                # Training, evaluation, and feature utilities
└── src/                    # Dataset loaders, models, and training code
```

The training launchers read the experiment settings from `configs/` and use
the corresponding implementations in `src/`. Each run is written to a new
archive directory together with its configuration, logs, and source snapshot.

<a id="data"></a>

## 📁 Data

The fixed split CSVs are included under `data/`. Configure the raw media and feature models in `configs/features.yaml`, then create the modality caches as described in [docs/DATASETS.md](docs/DATASETS.md). Check the prepared layout with:

```bash
python scripts/preflight.py data
```

### Dataset Summary

| Dataset | Modalities | Train | Validation | Test |
|---|---|---:|---:|---:|
| MUStARD | Text, visual, acoustic | 500 | 90 | 100 |
| Memotion | Text, visual | 4,896 | 1,048 | 1,048 |

MUStARD counts refer to target utterances; its split CSVs also contain the
associated dialogue context. Memotion counts refer to individual memes.

### Feature Preparation

- **Text:** BERT token and `[CLS]` representations, accompanied by dependency information produced with spaCy.
- **Visual:** ViT `[CLS]` representations for complete images and Faster R-CNN representations for detected regions.
- **Acoustic:** MUStARD acoustic inputs are divided into four temporal segments, mean-pooled, and interpolated to 768 dimensions.

Generated files follow the cache layout described in
[`docs/DATASETS.md`](docs/DATASETS.md) and are consumed directly by the graph
dataset loaders.

<a id="quick-start"></a>

## 🚀 Quick Start

```bash
bash scripts/train_mustard.sh data
bash scripts/train_memotion.sh data
```

MUStARD and Memotion are trained independently with their corresponding
configuration files. Launchers default to CPU and save each run in a new
output directory. Checkpoints are selected on the validation split, and the
selected checkpoint is evaluated once on the test split.

<a id="main-results"></a>

## 📊 Main Results

All values are support-weighted percentages on the test sets.

| Dataset | Sarcasm P | Sarcasm R | Sarcasm F1 | Sentiment R | Sentiment F1 |
|---|---:|---:|---:|---:|---:|
| MUStARD | 86.04 | 86.00 | 85.98 | 47.00 | 48.39 |
| Memotion | 69.08 | 76.05 | 71.56 | 55.15 | 49.30 |

Configurations and experimental settings are available in `configs/` and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## 📬 Contact and License

Repository maintenance: `teganjay0715@gmail.com`. This code is released under the Apache License 2.0. Dataset and third-party licenses apply separately.
