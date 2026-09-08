#!/usr/bin/env bash
# Brute-force oracle mask search on the full wikiann test set (bert x mean).
#
#   tmux new-session -d -s iolex-oracle "bash scripts/run_oracle.sh 2>&1 | tee outputs/oracle_bert_mean.log"
#
# batch_size is large on purpose: (sentence, rho) pairs are bucketed by
# (n, k), so more sentences per batch means bigger buckets and fuller GPU
# batches. Internal chunking keeps memory bounded regardless.
set -u
cd "$(dirname "$0")/.."
.venv/bin/forge run task=oracle \
    data.dataset=wikiann data.encoder.family=bert data.encoder.pooling=mean \
    runtime.device=cuda runtime.data.batch_size=512
echo "ORACLE_EXIT=$?"
