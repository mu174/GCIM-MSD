# Dataset and feature layout

The release does not redistribute MUStARD, Memotion, or their media. Obtain
each dataset under its own terms, then arrange the processed split files and
cached graph features as follows:

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

## Split-file checksums

These hashes identify the exact local split files used by the archived run.

| File | SHA-256 |
|---|---|
| MUStARD train | `c01edad8c270f1c56a5aa6126dfe5f98ada2ff41bd8b083c85a44f34312e932a` |
| MUStARD validation | `017f3d97273600a42cf0a095d36b06c41b311885aa33e177f10e3b8e238b4c97` |
| MUStARD test | `b364d37a55f4e53b4c4859c5bf3897e63c275e87fd5f034a2bd081fa3f30db18` |
| Memotion train | `3e5714a8af835351dfa0d73be303ba4df0b894d09f5971f2f5b49583cb612d4d` |
| Memotion validation | `63925a1eb06a3fb9cb9e5262e9768da8dbafeeaa3c1c50e0747239e2cc1c4baa` |
| Memotion test | `caa2fd07ad5cf1d7e6df2889e911825eb014265fc71f3e1b77424934ac91cc5b` |

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
The training code does not download encoders or extract features on demand.
Feature-extraction scripts are not included. Supply the prepared feature files
using the layout above.
