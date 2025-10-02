# Dataset Precache Utilities

Helpers for building the cached datasets required by RatCon grid sweeps live in
this package. You can run them directly via Python or submit them to Slurm using
the provided batch script.

## Command-line usage

```bash
module load python/3.11.7
source ~/RatCon/.venv/bin/activate
export HF_HOME=$HOME/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

python -m tools.prechache --subset 1.0
```

Key options:
- `--subset`: fraction (<=1) or absolute number of CNN **train** examples to
  keep when building the cached files (default: full train set).
- `--cnn-splits` / `--wiki-splits`: restrict which splits are materialised.
- `--skip-cnn` / `--skip-wiki`: skip an entire dataset.
- `--rerun-grid`: automatically invoke `dora grid <name> --clear` after caching
  (use `--grid-name` to pick which grid).

## Slurm batch helper

Submit the batch script from the repo root to cache datasets on a compute node
and automatically restart the grid sweep:

```bash
sbatch tools/prechache/precache_and_rerun.sbatch
```

Outputs are written to `logs/precache_and_rerun-<jobid>.out|err`. Adjust the
`#SBATCH` directives in the script if you need different resources or a
non-default grid name.
