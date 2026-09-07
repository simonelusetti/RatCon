#!/usr/bin/env bash
# Full CoNLL-2003 selector/preference grid: 12 strategies x 3 seeds.
#
# Intended detached launch:
#   tmux new-session -d -s iolex-conll-preferences \
#     "bash scripts/run_preferences_conll2003.sh 2>&1 | tee outputs/grid_preferences_conll2003.log"
set -u
cd "$(dirname "$0")/.."

# The execution sandbox can read the shared Hugging Face model cache but its
# datasets cache must live in the writable project tree. Offline mode makes
# this deterministic and avoids metadata checks: every encoder and the
# converted CoNLL-2003 parquet files are already cached on this machine.
export HF_DATASETS_CACHE="$PWD/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="$PWD/outputs/matplotlib-cache"

mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

.venv/bin/forge grid --file utils/grid_preferences_conll2003.yaml runtime.device=cuda
echo "[$(date -Is)] GRID_EXIT=$?"
