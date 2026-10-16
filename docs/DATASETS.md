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
