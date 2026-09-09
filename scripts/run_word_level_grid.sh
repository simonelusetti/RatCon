#!/usr/bin/env bash
# Full re-run under word-level selection (src/words.py).
#
#   tmux new-session -d -s iolex-word \
#     "bash scripts/run_word_level_grid.sh 2>&1 | tee outputs/word_level_grid.log"
#
# Every subword-era selector and oracle was purged first: the selector MLP is
# D->scalar under either unit, so an old checkpoint would load without error
# and silently rank for a different unit and budget. train.py now refuses one
# via the selection_unit field in the checkpoint meta, but purging is what
# makes the store honest.
#
# taggers are deliberately NOT re-run. The probe reads word-level
# embeddings already (tagger/model.py's gather_word_level), so nothing about it
# changes; outputs/probe/ stays valid and its 9 forge experiments were kept.
#
# llm gets its own calls throughout: data.encoder.pooling is part of the
# experiment signature and pythia uses `last`, so it cannot share a --sweep
# axis with the mean-pooled families. Two --sweep axes cross-produce
# (5 families x 3 seeds = 15); --run and --sweep do NOT cross.
#
# batch_size stays at the 16 the original sweep used. 32 allocated 31.3 GB
# of a 32 GB card, and `forge grid` swallows per-entry exceptions -- an OOM
# ten runs in would mark runs `failed` and carry on silently, which is a far
# worse trade than the extra hours. runtime.* is excluded from the signature,
# so this is free to tune without forking experiments.
#
# Order is by value: wikiann selectors first (the primary corpus, 6 tags),
# then the wikiann oracles that pair with them, then conll2003. Word-level
# search is ~2.7x cheaper than subword and reaches ~98% exhaustive rather
# than ~87%, so the oracle phase is much shorter than it used to be.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FAM=bert,electra,roberta,sbert,e5

echo "[$(date -Is)] === phase 1: wikiann selectors, 5 families x 3 seeds ==="
.venv/bin/forge grid data.dataset=wikiann data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
    train.continue=true \
    --sweep data.encoder.family=$FAM --sweep runtime.seed=0,1,2
echo "[$(date -Is)] P1_EXIT=$?"

echo "[$(date -Is)] === phase 2: wikiann selector, pythia (last pooling) ==="
.venv/bin/forge grid data.dataset=wikiann data.encoder.family=llm \
    data.encoder.pooling=last runtime.device=cuda runtime.grid=true \
    runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] P2_EXIT=$?"

echo "[$(date -Is)] === phase 3: wikiann oracles (deterministic, no seeds) ==="
.venv/bin/forge grid task=oracle data.dataset=wikiann data.encoder.pooling=mean \
    runtime.device=cuda runtime.data.batch_size=128 \
    --sweep data.encoder.family=$FAM
echo "[$(date -Is)] P3_EXIT=$?"
.venv/bin/forge run task=oracle data.dataset=wikiann data.encoder.family=llm \
    data.encoder.pooling=last runtime.device=cuda runtime.data.batch_size=128 \
    runtime.oracle.chunk_tokens=65536
echo "[$(date -Is)] P3B_EXIT=$?"

echo "[$(date -Is)] === phase 4: conll2003 selectors ==="
.venv/bin/forge grid data.dataset=conll2003 data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 \
    train.continue=true \
    --sweep data.encoder.family=$FAM --sweep runtime.seed=0,1,2
echo "[$(date -Is)] P4_EXIT=$?"
.venv/bin/forge grid data.dataset=conll2003 data.encoder.family=llm \
    data.encoder.pooling=last runtime.device=cuda runtime.grid=true \
    runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] P4B_EXIT=$?"
echo "[$(date -Is)] === done ==="
