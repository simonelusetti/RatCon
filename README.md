# RatCon Package

This repository now has packaging metadata (`pyproject.toml`, `setup.cfg`). Install it alongside MoE so the branching scripts can import the rationale selector models:

```bash
cd ../RatCon
pip install -e .
```

The MoE project expects RatCon to be importable from the sibling directory (`../RatCon`).
