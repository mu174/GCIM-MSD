#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATA_ROOT [RUNS_ROOT]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${REPO_ROOT}/scripts/launch.py" --dataset memotion \
  --data-root "$1" --runs-root "${2:-${REPO_ROOT}/runs}"
