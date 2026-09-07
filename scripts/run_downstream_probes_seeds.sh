#!/usr/bin/env bash
# Re-run the NER probes on the non-wikiann corpora with 3 seeds each.
#
#   tmux new-session -d -s iolex-probes \
#     "bash scripts/run_downstream_probes_seeds.sh 2>&1 | tee outputs/downstream_probes.log"
#
# Why two calls instead of one grid: forge's `direct` (--run) and `product`
# (--sweep) entries are INDEPENDENT -- they are not crossed. The previous
# attempt combined `--run data.dataset=...` with `--sweep runtime.seed=0,1,2`
# and got 3 datasets at the default seed PLUS 3 default-dataset runs, rather
# than the 9 intended. Two --sweep axes DO cross with each other, so the
# datasets that share settings go in one call, and movie_rationales gets its
# own because it needs class_weighted.
#
# movie_rationales keeps ner.class_weighted=true: its rationale split is
# ~5.4:1 and unweighted cross-entropy collapses the probe onto the majority
# class (see conf/config.yaml), which would make its per-tag F1 meaningless.
# Seeds are excluded from the signature, so these land as extra runs of the
# probe experiments that already exist rather than new ones.
set -u
cd "$(dirname "$0")/.."

echo "[$(date -Is)] === conll2003 + conll2000, seeds 0/1/2 ==="
.venv/bin/forge -M ner_probe grid task=ner data.encoder.family=bert \
    runtime.device=cuda runtime.data.batch_size=16 train.continue=true \
    --sweep data.dataset=conll2003,conll2000 \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_A=$?"

echo "[$(date -Is)] === movie_rationales (class-weighted), seeds 0/1/2 ==="
.venv/bin/forge -M ner_probe grid task=ner data.encoder.family=bert \
    data.dataset=movie_rationales ner.class_weighted=true \
    runtime.device=cuda runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=0,1,2
echo "[$(date -Is)] EXIT_B=$?"
