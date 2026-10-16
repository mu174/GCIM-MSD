#!/usr/bin/env python3
"""Check runtime packages and the local GCIM-MSD data layout."""

import argparse
import importlib.util
import json
from pathlib import Path


PACKAGES = ("torch", "dgl", "transformers", "numpy", "pandas", "sklearn", "tqdm")
REQUIRED_FILES = (
    "mustard/splits/train.csv",
    "mustard/splits/dev.csv",
    "mustard/splits/test.csv",
    "mustard/features/text/dependency_graph_train.json",
    "mustard/features/text/dependency_graph_dev.json",
    "mustard/features/text/dependency_graph_test.json",
    "memotion/splits/train.csv",
    "memotion/splits/dev.csv",
    "memotion/splits/test.csv",
)
REQUIRED_FEATURE_DIRS = (
    "mustard/features/text",
    "mustard/features/image",
    "mustard/features/audio/train",
    "mustard/features/audio/dev",
    "mustard/features/audio/test",
    "memotion/features/text",
    "memotion/features/image",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    packages = {name: importlib.util.find_spec(name) is not None for name in PACKAGES}
    files = {name: (root / name).is_file() for name in REQUIRED_FILES}
    feature_dirs = {
        name: (root / name).is_dir() and any((root / name).iterdir())
        for name in REQUIRED_FEATURE_DIRS
    }
    report = {
        "data_root": str(root),
        "packages": packages,
        "required_files": files,
        "nonempty_feature_directories": feature_dirs,
    }
    print(json.dumps(report, indent=2))
    if not all((*packages.values(), *files.values(), *feature_dirs.values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
