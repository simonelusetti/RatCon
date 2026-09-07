"""Entry point for the NER MLP probe (downstream validation).

    forge -M ner_probe run data.dataset=conll2003 data.encoder.family=sbert

Shim for the same reason as train.py -- see its docstring.
"""
from src.ner_probe import main  # noqa: F401  -- forge calls main(cfg)
