"""Entry point for the rationale selector (the bias test).

    forge run data.dataset=wikiann data.encoder.family=sbert

Forge's default entry point is `train.py` beside `conf/` at the project
root, so this is a shim: the implementation lives in src/train.py, which
stays importable as a package module (its `from .metrics import ...`
relative imports need that).
"""
from src.train import main  # noqa: F401  -- forge calls main(cfg)
