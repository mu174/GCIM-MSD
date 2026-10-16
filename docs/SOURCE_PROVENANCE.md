# Source provenance

The executable main chain was copied from the immutable source snapshot of
archived run `20261015_074731_489865_joint70_30_both_mu_balanced_me_none`.

| Release file | Archived source file |
|---|---|
| `src/train_mustard.py` | `main_mu_joint.py` |
| `src/train_memotion.py` | `main_me_joint.py` |
| `src/model_mustard.py` | `model_mu_fixed.py` |
| `src/model_memotion.py` | `model_me_fixed.py` |
| `src/engine_mustard.py` | `engine_mu_fixed.py` |
| `src/engine_memotion.py` | `engine_me_fixed.py` |
| `src/dataset.py` | `dataset.py` |
| `src/utils.py` | `utils.py` |

Release-only changes are documented in `docs/REPRODUCIBILITY.md`. Original
source hashes are stored in `results/provenance.json` so the release can be
compared against the archived run without publishing machine-specific paths.
