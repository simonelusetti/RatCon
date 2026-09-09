"""Correlation of every selection metric against probe F1, as a plain table.

One number per (series, metric, variant, seed, rho):

    x  the series' per-tag selection bias  (see _common.load_bias)
         selection = observed rate - rho          division = signed z
         sgn       = as measured                  abs      = |value|, per run
    y  that tag's token-level F1 from the tagger cache, averaged over the
       probe's seeds
    r  Pearson correlation of x against y ACROSS TAGS

So each cell is a correlation over `n_tags` points -- 6 for wikiann. That is
the binding constraint on everything here: at n=6, |r| must exceed 0.811 to
reach p<0.05, so read these as descriptive, not significant.

Per seed by default, deliberately. The alternative -- averaging each tag's
bias over seeds and correlating once (--pooled) -- gives a *different and
systematically stronger* number, because averaging removes noise from x and
less noise in x means less attenuation. On this data the pooled figure has
come out more extreme than any individual run it summarises. Use --pooled for
a plot's central line; quote per-seed numbers.

Usage:
  python3 utils/correlation_tables.py [--dataset wikiann] [--pooled]
      [--series bert/mean,sbert/mean] [--output PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.core import ExperimentStore  # noqa: E402
from utils._common import entity_tags, load_bias, tagger_f1, probe_reports  # noqa: E402
from utils.plot_grounding import discover_series  # noqa: E402

RHOS = [None] + [round(0.1 * i, 1) for i in range(1, 10)]
METRICS = [("selection", False), ("selection", True), ("division", False), ("division", True)]


def critical_r(n: int, alpha: float = 0.05) -> float:
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(np.sqrt(t**2 / (t**2 + n - 2)))


def seed_of(store: ExperimentStore, path: Path):
    return OmegaConf.select(store.load_run(path.parent.name, path.name).config, "seed")


def correlations(paths, dataset, tags, f1, metric, absolute):
    """One row of correlations across rho for a set of runs (pooled if >1)."""
    row = []
    for rho in RHOS:
        biases = [load_bias(p, dataset, metric, rho) for p in paths]
        x = np.array([np.mean([abs(b[g]) if absolute else b[g] for b in biases]) for g in tags])
        row.append(stats.pearsonr(x, f1)[0] if x.std() > 0 else float("nan"))
    return row


def main(dataset: str, pooled: bool, wanted: list[str] | None, out) -> None:
    store = ExperimentStore(root=ROOT / "outputs")
    tags = entity_tags(dataset)
    series = {s["label"]: s for s in discover_series(dataset)}
    if wanted:
        missing = [w for w in wanted if w not in series]
        if missing:
            raise SystemExit(f"unknown series {missing}; have {sorted(series)}")
        series = {k: v for k, v in series.items() if k in wanted}
    if not series:
        raise SystemExit(f"no selection experiments for {dataset!r}")

    head = f"{'metric/variant':16s}{'seed':>6s}" + "".join(
        f"{('avg' if r is None else f'{r:.1f}'):>8s}" for r in RHOS)
    print(f"dataset={dataset}  tags={len(tags)} ({', '.join(tags)})", file=out)
    print(f"|r| for p<0.05 at n={len(tags)}: {critical_r(len(tags)):.3f}"
          f"   mode: {'POOLED over seeds' if pooled else 'per seed'}\n", file=out)

    for label in sorted(series):
        e = series[label]
        reports = probe_reports(dataset, e["family"])
        f1 = np.array([np.mean([tagger_f1(r, tags)[g] for r in reports]) for g in tags])
        runs = sorted(e["bias_runs"], key=lambda p: (seed_of(store, p) is None, seed_of(store, p)))
        print("=" * len(head), file=out)
        print(f"{label}   {e['signature']}   {len(runs)} run(s), probe seeds={len(reports)}", file=out)
        print(head, file=out)
        for metric, absolute in METRICS:
            name = f"{metric}/{'abs' if absolute else 'sgn'}"
            groups = [("pooled", runs)] if pooled else [(seed_of(store, p), [p]) for p in runs]
            for i, (seed, paths) in enumerate(groups):
                row = correlations(paths, dataset, tags, f1, metric, absolute)
                print(f"{name if i == 0 else '':16s}{str(seed):>6s}"
                      + "".join(f"{v:+8.3f}" for v in row), file=out)
            print(file=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="wikiann")
    p.add_argument("--pooled", action="store_true",
                   help="average bias over seeds before correlating (see docstring)")
    p.add_argument("--series", default=None, help="comma-separated subset")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    sel = [s.strip() for s in a.series.split(",")] if a.series else None
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("w") as fh:
            main(a.dataset, a.pooled, sel, fh)
        print(f"wrote {a.output}")
    else:
        main(a.dataset, a.pooled, sel, sys.stdout)
