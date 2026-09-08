"""Shared entry point for the rationale selector and oracle mask search.

    forge run data.dataset=wikiann data.encoder.family=sbert
    forge run task=oracle data.dataset=wikiann data.encoder.family=bert

Forge's default entry point is `train.py` beside `conf/` at the project
root, so this is a shim: the implementation lives in src/train.py, which
stays importable as a package module (its `from .metrics import ...`
relative imports need that).
"""
from src.train import main  # noqa: F401  -- forge calls main(cfg)
