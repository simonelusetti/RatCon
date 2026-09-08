#!/usr/bin/env bash
# bert/mean oracle at rho=0.8 on the corpora other than wikiann, plus the NER
# probes those oracles have to be correlated against.
#
#   tmux new-session -d -s iolex-downstream \
#     "bash scripts/run_oracle_downstream.sh 2>&1 | tee outputs/oracle_downstream.log"
#
# Probes first: they are minutes each, and without a probe on the same corpus
# an oracle run has no per-tag F1 to correlate with, so it would produce bias
# curves and nothing to compare them to. 3 seeds each, matching the wikiann
# probes so the F1 side is averaged the same way.
#
# movie_rationales gets ner.class_weighted=true: its rationale/not-rationale
# split is ~5.4:1 and unweighted cross-entropy collapses the probe onto the
# majority class (see the note on ner.class_weighted in conf/config.yaml),
# which would make its per-tag F1 -- and any correlation built on it -- junk.
#
# model.loss.sweep_range=[0.8,0.8,1] restricts the search to the single
# rho asked for. It is part of the experiment signature, so these are
# distinct experiments from any full-sweep oracle.
#
# Ordered conll2003 -> conll2000 -> movie_rationales deliberately: the first
# two are genuine searches (69% / 37% of pairs enumerated exhaustively), while
# movie_rationales is 0% exhaustive -- its documents are ~510 tokens, so
# C(510,408) is far past any cap and every document falls back to 10k uniform
# samples. That run is best-of-random rather than an oracle; it is last so it
# can be killed without losing the useful results.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] === NER probes ==="
bash scripts/run_downstream_probes_seeds.sh
echo "[$(date -Is)] PROBES_EXIT=$?"

echo "[$(date -Is)] === oracles (rho=0.8) ==="
.venv/bin/forge grid task=oracle \
    data.encoder.family=bert data.encoder.pooling=mean \
    model.loss.sweep_range=[0.8,0.8,1] \
    runtime.device=cuda runtime.data.batch_size=256 \
    --run data.dataset=conll2003 \
    --run data.dataset=conll2000 \
    --run data.dataset=movie_rationales runtime.oracle.chunk_tokens=65536
echo "[$(date -Is)] ORACLES_EXIT=$?"
