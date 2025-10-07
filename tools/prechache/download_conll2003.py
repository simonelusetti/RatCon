#!/usr/bin/env python3
"""Download and cache the conll2003 dataset for offline training."""

import argparse
import os
from datasets import load_dataset


def download_conll(cache_dir: str, revision: str) -> None:
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir

    # Ensure we are allowed to reach the Hub while downloading.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(var, None)

    dataset = load_dataset("conll2003", revision=revision, cache_dir=os.environ.get("HF_HOME"))
    sizes = {split: len(ds) for split, ds in dataset.items()}
    print("Cached conll2003 splits:")
    for split, size in sizes.items():
        print(f"  {split}: {size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/hf-cache")),
        help="Directory to use for the HF cache (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default="refs/convert/parquet",
        help="Dataset revision to download (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_conll(args.cache_dir, args.revision)
