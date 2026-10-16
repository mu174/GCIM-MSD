# Dataset and feature layout

The fixed split files are included in this repository. Generated graph features use the following layout:

```text
DATA_ROOT/
├── mustard/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── features/
│       ├── text/
│       │   ├── dependency_graph_train.json
│       │   ├── dependency_graph_dev.json
│       │   └── dependency_graph_test.json
│       ├── image/
│       └── audio/
│           ├── train/
│           ├── dev/
│           └── test/
└── memotion/
    ├── splits/
    │   ├── train.csv
    │   ├── dev.csv
    │   └── test.csv
    └── features/
        ├── text/
        └── image/
```

## Fixed splits used by the recorded main run

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| MUStARD | 500 target utterances | 90 | 100 |
| Memotion | 4,896 memes | 1,048 | 1,048 |

MUStARD split CSVs contain multiple context/target utterance rows per sample;
the counts above are target sample counts loaded by the dataset class.

## Label mapping

### MUStARD

- sarcasm: `False -> 0`, `True -> 1`;
- sentiment: `-1 -> 0` (Negative), `0 -> 1` (Neutral), and positive
  integer values `-> 2` (Positive).

### Memotion

- sarcasm: `not_sarcastic -> 0`; `general`, `twisted_meaning`, and
  `very_twisted -> 1`;
- sentiment: `very_negative` and `negative -> 0`, `neutral -> 1`,
  `positive` and `very_positive -> 2`.

The training code expects these Memotion labels to have already been converted
to integer columns named `sarcasm` and `overall_sentiment`.

## Cached feature contract

Both datasets load NumPy node-feature arrays and dependency-graph JSON files.
Place the prepared feature files in the layout above before training.

## Feature extraction

Set the raw media locations and model identifiers in `configs/features.yaml`.
The configured raw paths use these file names:

```text
raw/
├── memotion/images/<image_name>
└── mustard/
    ├── frames/<KEY>/<five-digit-frame-index>.jpg
    └── audio_features/<KEY>.npy
```

Install the dependency parser once:

```bash
python -m spacy download en_core_web_sm
```

Generate the caches with:

```bash
python scripts/extract_features.py --dataset mustard --modality all --device cuda
python scripts/extract_features.py --dataset memotion --modality all --device cuda
```

Text features are BERT token and `[CLS]` representations. Visual features are
ViT `[CLS]` representations of the complete image and Faster R-CNN object
regions. MUStARD visual frames are sampled uniformly to match the number of
utterances. Its acoustic input is a per-sample NumPy array; the script divides
the temporal axis into four equal segments, mean-pools each segment, and
linearly interpolates each vector to 768 dimensions.
