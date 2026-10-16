<div align="center">

<h1>GCIM-MSD</h1>

<p><strong>Global Context-Guided Incongruity Modeling for Multimodal Sarcasm Detection</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 1.12.1">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0"></a>
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#data">Data</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#main-results">Results</a>
</p>

</div>

---

## Overview

GCIM-MSD performs multimodal sarcasm detection with auxiliary sentiment supervision on MUStARD and Memotion.

| | Description |
|---|---|
| **Problem** | Model cross-modal incongruity using a shared global contextual reference. |
| **Task** | Sarcasm detection with auxiliary sentiment analysis. |
| **Modalities** | MUStARD: text, visual, and acoustic; Memotion: text and visual. |

The method contains three main components:

- **Global Contextual Reference (GCR):** aggregates multimodal information into a shared contextual reference.
- **Context-Referenced Modality Deviation (CRMD):** models the interaction between each modality and the GCR.
- **Modality-level Conflict Modeling (MCM):** captures conflict among modality representations.

## Architecture

<p align="center">
  <img src="assets/figure2.png" width="95%" alt="GCIM-MSD architecture">
</p>

<p align="center"><em>Overview of the GCIM-MSD framework.</em></p>

## Setup

Core environment: Python 3.10.14, PyTorch 1.12.1+cu113, DGL 0.9.1.post1, and Transformers 4.30.2. Install platform-compatible PyTorch and DGL builds first, then run:

```bash
pip install -r requirements.txt
```

Package versions are listed in `requirements-recorded.txt`.

## Data

The fixed split CSVs are included under `data/`. Configure the raw media and feature models in `configs/features.yaml`, then create the modality caches as described in [docs/DATASETS.md](docs/DATASETS.md). Check the prepared layout with:

```bash
python scripts/preflight.py data
```

## Quick Start

```bash
bash scripts/train_mustard.sh data
bash scripts/train_memotion.sh data
```

Launchers read `configs/<dataset>.yaml`, default to CPU, and save each run in a new output directory. Checkpoints are selected on the validation split using:

```text
0.7 * sarcasm weighted F1 + 0.3 * sentiment weighted F1
```

The selected checkpoint is evaluated once on the test split.

## Main Results

All values are support-weighted percentages on the test sets.

| Dataset | Sarcasm P | Sarcasm R | Sarcasm F1 | Sentiment R | Sentiment F1 |
|---|---:|---:|---:|---:|---:|
| MUStARD | 86.04 | 86.00 | 85.98 | 47.00 | 48.39 |
| Memotion | 69.08 | 76.05 | 71.56 | 55.15 | 49.30 |

Configurations and experimental settings are available in `configs/` and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Contact and license

Repository maintenance: `teganjay0715@gmail.com`. This code is released under the Apache License 2.0. Dataset and third-party licenses apply separately.
