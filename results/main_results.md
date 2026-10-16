# Recorded main results

Source run: `20261015_074731_489865_joint70_30_both_mu_balanced_me_none`

All P/R/F1 values are support-weighted percentages on the fixed test split.
The checkpoint for each dataset was selected on validation joint weighted F1.

| Dataset | Best epoch | Validation joint F1 | Sarcasm P | Sarcasm R | Sarcasm F1 | Sentiment P | Sentiment R | Sentiment F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MUStARD | 9 | 68.29 | 86.04 | 86.00 | 85.98 | 51.13 | 47.00 | 48.39 |
| Memotion | 18 | 60.80 | 69.08 | 76.05 | 71.56 | 45.71 | 55.15 | 49.30 |

The corresponding full JSON records are `mustard_main.json` and
`memotion_main.json`.
