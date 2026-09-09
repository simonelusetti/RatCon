"""Warm the probe cache from the shell.

    python -m tagger wikiann --family bert --seeds 0,1,2 --device cuda
    python -m tagger conll2003 conll2000 --family bert --seeds 0,1,2

The tagger is not a forge experiment, so there are no signatures to pass around:
a dataset and an encoder family name the cache entry completely. Datasets
already cached are skipped unless --retrain is given.
"""
import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.utils import configure_runtime

from .tagging import headline_f1, performances

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--family", default="bert", help="token encoder the probe reads")
    parser.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--class-weighted", action="store_true",
                        help="inverse-frequency loss weights; for skewed corpora "
                             "such as movie_rationales")
    parser.add_argument("--retrain", action="store_true", help="ignore any cached model")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra dotted config override, repeatable")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = OmegaConf.load(ROOT / "conf/config.yaml")
    cfg.data.encoder.family = args.family
    cfg.tagger.class_weighted = args.class_weighted
    if args.device:
        cfg.runtime.device = args.device
    if args.epochs:
        cfg.train.epochs = args.epochs
    cfg.merge_with_dotlist(args.set)

    cfg.runtime, fell_back = configure_runtime(cfg.runtime)
    if fell_back:
        logging.warning("CUDA requested but unavailable, using CPU.")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for dataset in args.datasets:
        reports = performances(cfg, dataset=dataset, seeds=seeds, retrain=args.retrain)
        f1 = [headline_f1(r) for r in reports]
        print(f"{dataset}/{args.family}: macro F1 "
              + ", ".join(f"seed{s}={v:.4f}" for s, v in zip(seeds, f1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
