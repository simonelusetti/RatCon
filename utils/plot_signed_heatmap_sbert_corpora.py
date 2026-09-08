"""Signed effect heatmap for SBERT across three corpora, in the paper's
Figure 3 layout: WikiAnn / Movie Review stacked in a left column,
CoNLL2000 (many more chunk tags) as one tall panel on the right, one
shared colorbar. CoNLL2003 (NER) is intentionally omitted -- WikiAnn
already covers the NER task, and plot_bias_heatmaps.py provides the
WikiAnn/CoNLL2003 per-encoder comparison separately.

Same --metric choice as plot_bias_heatmaps.py (division doesn't saturate;
pvalue matches the original figure's exact "sign x -log10(p)" convention
but is prone to hitting the exact test's precision floor on strong effects).

Usage: python3 utils/plot_signed_heatmap_sbert_corpora.py [--metric division|pvalue] [--output PATH]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LABEL_DISPLAY_NAMES  # noqa: E402
from utils.forge_paths import run_dir  # noqa: E402
from utils._common import make_flat_grey_cmap, make_flat_grey_norm  # noqa: E402

# SBERT bias-test (rationale task) signatures, one per corpus -- used only as
# CLI argument DEFAULTS (see __main__), not a hardcoded lookup; every sig is
# a real, overridable --*-sig argument.
# default_sig is a CLI argument DEFAULT only, never a hardcoded lookup.
# All three are None since the move from dora to forge: forge hashes the
# config differently, so no dora-era signature resolves any more, and a
# default that always fails is worse than none. Fill one back in once its
# sweep has been re-run under forge -- `forge info -S` lists signatures.
_CORPORA = [
    {"name": "WikiAnn (NER)", "dataset": "wikiann", "flag": "wikiann", "default_sig": None},
    {"name": "CoNLL2000 (Chunking)", "dataset": "conll2000", "flag": "conll2000", "default_sig": None},
    {"name": "Movie Review (Rationale)", "dataset": "movie_rationales", "flag": "movie-rationales", "default_sig": None},
]
# Background / non-of-interest classes to drop from every panel's rows.
EXCLUDE_LABELS = {"O", "special", "not rationale"}

# Two-sided threshold at p=0.05 (uncorrected) -- a visual "is this even
# nominally significant" control, marked on the colorbar, distinct from the
# per-dataset Bonferroni threshold used for the actual significance decision.
Z_CRIT_P05 = 1.9599639845400545
NEGLOGP_CRIT_P05 = -math.log10(0.05)


def resolve_name(raw_key: str, dataset: str) -> str:
    return LABEL_DISPLAY_NAMES.get(dataset, {}).get(raw_key, raw_key)


def load_matrix(entry: dict, metric: str) -> tuple[list[float], dict[str, list[float]]]:
    sig = entry["sig"]
    effect = json.loads((run_dir(sig) / "data/effect_size_curves.json").read_text())
    rho = [float(r) for r in effect["rho"]]
    if metric == "pvalue":
        pvalue = json.loads((run_dir(sig) / "data/pvalue_curves.json").read_text())

    matrix = {}
    for raw_key, z_curve in effect["curves"].items():
        name = resolve_name(raw_key, entry["dataset"])
        if name in EXCLUDE_LABELS:
            continue
        z = np.array(z_curve)
        if metric == "division":
            matrix[name] = z.tolist()
        else:
            matrix[name] = (np.sign(z) * np.array(pvalue["curves"][raw_key])).tolist()
    return rho, matrix


def main(sigs: dict[str, str], metric: str, height: float, hspace: float, output: Path) -> None:
    p05 = Z_CRIT_P05 if metric == "division" else NEGLOGP_CRIT_P05
    panels = {}
    for corpus in _CORPORA:
        entry = {"name": corpus["name"], "dataset": corpus["dataset"], "sig": sigs[corpus["dataset"]]}
        rho, matrix = load_matrix(entry, metric)
        panels[entry["name"]] = (rho, matrix)
    cbar_label = "Preference score" if metric == "division" else "sign × -log10(p)"

    all_abs = [abs(v) for _, matrix in panels.values() for curve in matrix.values() for v in curve if np.isfinite(v)]
    vmax = float(np.percentile(all_abs, 95)) if all_abs else 1.0
    norm = make_flat_grey_norm(vmax, thresh=p05)
    cmap = make_flat_grey_cmap()

    row_counts = {name: len(matrix) for name, (_, matrix) in panels.items()}
    names = [corpus["name"] for corpus in _CORPORA]

    fig = plt.figure(figsize=(6.5, height))
    gs = fig.add_gridspec(
        nrows=len(names), ncols=1,
        height_ratios=[row_counts[n] for n in names],
        hspace=hspace,
    )

    im = None
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i, 0])
        rho, matrix = panels[name]
        rows = sorted(matrix.keys())
        data = np.array([matrix[r] for r in rows])
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, origin="upper")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(rho)))
        ax.set_xticklabels([f"{r:.2f}" for r in rho], fontsize=7, rotation=45, ha="right")
        ax.text(1.02, 0, "ρ", transform=ax.transAxes, fontsize=8, ha="left", va="center")

    cbar = fig.colorbar(im, ax=fig.get_axes(), shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    for sign in (1, -1):
        cbar.ax.axhline(sign * p05, color="#333333", linestyle="--", linewidth=1)
    default_ticks = list(cbar.get_ticks())
    all_ticks = sorted(default_ticks + [p05])
    labels = [("p = 0.05" if t == p05 else f"{t:g}") for t in all_ticks]
    cbar.set_ticks(all_ticks)
    cbar.set_ticklabels(labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for corpus in _CORPORA:
        parser.add_argument(f"--{corpus['flag']}-sig", default=corpus["default_sig"],
                             help=f"bias-test forge sig for {corpus['name']}")
    parser.add_argument("--metric", choices=["division", "pvalue"], default="division")
    parser.add_argument("--height", type=float, default=13.0, help="figure height in inches (default 13)")
    parser.add_argument("--hspace", type=float, default=0.4, help="vertical gap between stacked panels, as a fraction of panel height (default 0.4; increase if titles collide with the panel above)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sigs = {corpus["dataset"]: getattr(args, f"{corpus['flag'].replace('-', '_')}_sig") for corpus in _CORPORA}
    missing = [f"--{c['flag']}-sig" for c in _CORPORA if not sigs[c["dataset"]]]
    if missing:
        raise ValueError(f"Missing required signature argument(s): {', '.join(missing)}")
    output = args.output or ROOT / f"outputs/analysis/signed_heatmap_sbert_corpora_{args.metric}.pdf"
    main(sigs, args.metric, args.height, args.hspace, output)
