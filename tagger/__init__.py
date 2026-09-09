"""The tagger: this study's evaluation object, not one of its experiments.

The selector is what the project measures. The probe only answers a fixed
question about a corpus -- "how well can a small MLP recover each tag from
this frozen encoder's token embeddings?" -- so it is a cache keyed by that
question, not a forge experiment with runs and checkpoints.

    from tagger import performances
    reports = performances(cfg, seeds=(0, 1, 2))   # trains only what is missing
    reports = load("wikiann", "bert")              # read-only, for plots
"""
from .tagging import load, performances, store_dir

__all__ = ["load", "performances", "store_dir"]
