#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {mustard|memotion} DATA_ROOT [RUNS_ROOT] [VARIANT ...]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$1"
DATA_ROOT="$2"
RUNS_ROOT="${3:-${REPO_ROOT}/runs_ablation}"
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(wo_crmd wo_mcm wo_gate wo_sp wo_t wo_v stl_sarcasm stl_sentiment)
fi
exec python "${REPO_ROOT}/scripts/launch.py" --dataset "$DATASET" \
  --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --variants "${VARIANTS[@]}"
