#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {mustard|memotion} DATA_ROOT CHECKPOINT OUTPUT_JSON" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GCIM_MSD_DATA_ROOT="$(cd "$2" && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python "${REPO_ROOT}/src/evaluate.py" "$1" --checkpoint "$3" --output "$4"
