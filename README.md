# GCIM-MSD

Code for **Global Context-Guided Incongruity Modeling for Multimodal Sarcasm Detection**. GCIM-MSD performs sarcasm detection with auxiliary sentiment supervision on MUStARD and Memotion.

![GCIM-MSD architecture](assets/figure2.png)

MUStARD uses text, visual, and acoustic modalities; Memotion uses text and visual modalities.

## Requirements

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

## Training

```bash
bash scripts/train_mustard.sh data
bash scripts/train_memotion.sh data
```

Launchers read `configs/<dataset>.yaml`, default to CPU, and save each run in a new output directory. Checkpoints are selected on the validation split using:

```text
0.7 * sarcasm weighted F1 + 0.3 * sentiment weighted F1
```

The selected checkpoint is evaluated once on the test split.

## GCIM-MSD Results

All values are support-weighted percentages on the test sets.

| Dataset | Sarcasm P | Sarcasm R | Sarcasm F1 | Sentiment R | Sentiment F1 |
|---|---:|---:|---:|---:|---:|
| MUStARD | 86.04 | 86.00 | 85.98 | 47.00 | 48.39 |
| Memotion | 69.08 | 76.05 | 71.56 | 55.15 | 49.30 |

Configurations and experimental settings are available in `configs/` and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Contact and license

Repository maintenance: `teganjay0715@gmail.com`. This code is released under the Apache License 2.0. Dataset and third-party licenses apply separately.
