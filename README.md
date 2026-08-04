# Iolex

Code for **"No Checked Baggage: Forcing Sentence Encoders to Keep Only
Essential Tokens"** — a study of whether sentence encoders (SBERT, E5,
Pythia) have structural token-selection biases. A differentiable top-ρ
token selector is trained on top of a frozen sentence encoder; its
selections are then checked for label-dependent bias, tested for
downstream-task sufficiency, and correlated against an independent NER
probe.

## Setup

```bash
pip install -r requirements.txt
```

Python >= 3.10. GPU is optional (`runtime.device=cuda`); every experiment
in the paper was also runnable on CPU, just slower. This is a standalone
project, not an installable library — everything is run directly from the
repo root (`dora run ...`, `python3 utils/plot_*.py ...`); there's no
package to `pip install -e .`.

## How an experiment is identified

Every training run is a [dora](https://github.com/facebookresearch/dora)
experiment, hashed from its config overrides into a short **signature**
(e.g. `97d170e1`). All of that run's outputs live under
`outputs/xps/<sig>/`:

```
outputs/xps/<sig>/
  data/       # JSON artifacts (curves, histories, summaries) -- what the plot scripts read
  plots/      # PNGs saved directly by the training run
  state/models/  # checkpoints
  train.log
```

Launch a run with `dora run <overrides...>`; it prints `Exp signature: <sig>`
on completion. The `utils/plot_*.py` scripts below all take one or more
`--*-sig` arguments that point at these directories — **there is no
hidden lookup table**; every sig has to come from a run you (or the
checkpoint bundle you were given) actually produced.

Two entry points share the same config (`src/conf/default.yaml`):

| Task | Entry point | Invocation |
|---|---|---|
| Rationale selector (the bias test) | `src/train.py` | `dora run <overrides>` |
| NER MLP probe (downstream validation) | `src/ner_probe.py` | `dora --main_module ner_probe run <overrides>` |

Key config overrides:

- `data.dataset` — one of `wikiann`, `conll2003`, `conll2000`,
  `movie_rationales`, `stsb`
- `data.encoder.family` — one of `sbert`, `e5`, `llm` (Pythia)
- `train.seed` — seed (excluded from the dora signature, so re-running the
  same overrides with a different seed reuses the same `<sig>` directory
  layout only if you also bump `run=N`; see `utils/grid_*.yaml` for the
  pattern used to launch multi-seed sweeps)

A plain `dora run data.dataset=wikiann data.encoder.family=sbert` trains
the selector, then automatically runs the bias-test evaluation
(`runtime.eval.skip=false` is the default) and writes every `data/*.json`
artifact the plot scripts need — no separate eval step required.

## Reproducing the paper's figures

Each row below trains whatever the figure needs (skip if you already have
matching sigs) and then calls the corresponding plot script.

**Figure 4 — per-encoder bias heatmap (WikiAnn / CoNLL-2003)**
```bash
dora run data.dataset=wikiann data.encoder.family=sbert
dora run data.dataset=wikiann data.encoder.family=e5
dora run data.dataset=wikiann data.encoder.family=llm

python3 utils/plot_signed_heatmap.py --dataset wikiann \
    --sbert-sig <sig> --e5-sig <sig> --llm-sig <sig>
```

**Figure 5 — grounding test (bias rate vs. NER F1, + B/I convergence)**
```bash
dora run data.dataset=conll2003 data.encoder.family=sbert
dora --main_module ner_probe run data.dataset=conll2003 data.encoder.family=sbert
# ...repeat both for e5 and llm

python3 utils/plot_paper_figure_conll2003.py \
    --sbert-bias-sig <sig> --sbert-ner-sig <sig> \
    --e5-bias-sig <sig>    --e5-ner-sig <sig> \
    --llm-bias-sig <sig>   --llm-ner-sig <sig> \
    --rho 0.8
```

**Figure 6 — SBERT bias across 4 corpora**
```bash
dora run data.dataset=wikiann data.encoder.family=sbert
dora run data.dataset=conll2003 data.encoder.family=sbert
dora run data.dataset=conll2000 data.encoder.family=sbert
dora run data.dataset=movie_rationales data.encoder.family=sbert

python3 utils/plot_signed_heatmap_sbert_corpora.py \
    --wikiann-sig <sig> --conll2003-sig <sig> \
    --conll2000-sig <sig> --movie-rationales-sig <sig>
```

**Figure 3 — STS-B sufficiency test (3 panels)**
```bash
# panel 1 + 2: trained+evaluated on STS-B, 3 seeds per encoder
dora run data.dataset=stsb data.encoder.family=sbert train.seed=0
dora run data.dataset=stsb data.encoder.family=sbert train.seed=1
dora run data.dataset=stsb data.encoder.family=sbert train.seed=2
# ...repeat for e5, llm

# panel 3: SBERT trained on each of the other 4 corpora, evaluated on STS-B
# (already have these sigs if you ran Figure 6 above -- STS-B evaluation
# runs automatically on every training run regardless of data.dataset)

python3 utils/plot_stsb_sufficiency.py \
    --panel1-sigs <sig> <sig> <sig> \
    --panel2-sbert-sigs <sig> <sig> <sig> \
    --panel2-e5-sigs <sig> <sig> <sig> \
    --panel2-llm-sigs <sig> <sig> <sig> \
    --panel3-stsb-sigs <sig> <sig> <sig> \
    --panel3-wikiann-sigs <sig> <sig> <sig> \
    --panel3-conll2003-sigs <sig> <sig> <sig> \
    --panel3-conll2000-sigs <sig> <sig> <sig> \
    --panel3-movie-rationales-sigs <sig> <sig> <sig>
```
This script has no default sigs at all (unlike the others) — the STS-B
training sweep hasn't been run in this repo yet, so there is nothing
correct to default to.

## Multi-seed / multi-encoder sweeps

`utils/grid.py` drives `utils/grid_<name>.yaml` configs that launch a
whole sweep sequentially via `dora run` and print a signature summary
table at the end:

```bash
python3 utils/grid.py --config utils/grid_conll2003.yaml
```

`grid_ner.yaml` sets `main_module: ner_probe` so its sweep launches
`ner_probe.py` runs instead of `train.py` runs. Every plot script's
`--help` documents exactly which flags it needs and which sig each one
should come from.

## Every plot/analysis script's own docstring

Each `utils/plot_*.py` file has a module docstring describing what figure
it produces and its exact CLI usage — run `python3 utils/<script>.py --help`
for the authoritative, current argument list.
