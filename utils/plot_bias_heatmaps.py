"""Signed selection-bias heatmap (entity tag x rho) for every strategy.

Discovers whatever is in the forge store and gives each strategy a panel, on
one shared colour scale so panels are comparable by eye rather than only
within themselves.

This also produces the paper's Figure 4, which used to be a separate script
taking three hardcoded --sbert/--e5/--llm signature flags: `--ncols 1` gives
that figure's stacked single-column layout, and `--strategies` picks the
encoders it showed.

    python3 utils/plot_bias_heatmaps.py --dataset wikiann --ncols 1 \
        --strategies sbert,e5,llm

Reading a panel: colour saturation is the magnitude of the effect and hue is
its direction -- blue for over-selected, red for under-selected, with a flat
grey band for anything inside the significance threshold. A tag kept 20
points above chance and one kept 20 points below are equally saturated in
opposite hues.

Metrics
-------
--metric division (default): the signed z-score, the raw effect divided by
    the exact null's standard deviation. Standardising this way lets a rare
    tag's small absolute excess count for as much as a common tag's large
    one. It keeps resolving where p-values have bottomed out.
--metric pvalue: sign(z) * -log10(p). Matches the original chi-square-era
    figure's convention, but at this scale the p-values hit the exact test's
    floating-point floor (~1e-14), so most cells saturate identically no
    matter how much stronger one effect is than another.

Multi-seed runs are averaged cell-by-cell before plotting.

Usage:
  python3 utils/plot_bias_heatmaps.py [--dataset wikiann] [--metric division|pvalue]
      [--strategies bert/mean,electra/mean] [--ncols 3] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LABEL_DISPLAY_NAMES, PAD_TAG  # noqa: E402
from utils._common import make_flat_grey_cmap, make_flat_grey_norm  # noqa: E402
from utils.forge_paths import discover_strategies  # noqa: E402

# Not entity types, and O's magnitude would dwarf every real tag and flatten
# the colour scale for everything else. PAD_TAG is padding rather than a
# class at all; it only appears in some runs (e5's, which prefixes "query: ")
# and plotting it as a tag row would be meaningless.
EXCLUDE_LABELS = {"O", "special", "not rationale", PAD_TAG}

# Two-sided z at p=0.05, uncorrected: a visual "is this even nominally
# significant" marker on the colourbar, distinct from the per-dataset
# Bonferroni threshold used for the actual significance decision.
Z_CRIT_P05 = 1.9599639845400545
NEGLOGP_CRIT_P05 = -math.log10(0.05)


def load_matrix(runs: list[Path], dataset: str, metric: str) -> tuple[list[float], dict[str, list[float]]]:
    """Mean effect per (tag, rho) across a strategy's runs."""
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    rho: list[float] = []
    stacks: dict[str, list[np.ndarray]] = {}

    for run in runs:
        effect = json.loads((run / "data/effect_size_curves.json").read_text())
        rho = [float(r) for r in effect["rho"]]
        pvalue = (json.loads((run / "data/pvalue_curves.json").read_text())
                  if metric == "pvalue" else None)
        for idx, curve in effect["curves"].items():
            name = label_map.get(idx, idx)
            if name in EXCLUDE_LABELS:
                continue
            z = np.array(curve, dtype=float)
            value = z if metric == "division" else np.sign(z) * np.array(pvalue["curves"][idx], dtype=float)
            stacks.setdefault(name, []).append(value)

    return rho, {name: np.mean(np.stack(v), axis=0).tolist() for name, v in stacks.items()}


def main(dataset: str, metric: str, strategies: list[str] | None, ncols: int, output: Path) -> None:
    found = discover_strategies(dataset)
    if strategies:
        available = {e["label"] for e in found}
        unknown = [s for s in strategies if s not in available]
        if unknown:
            raise SystemExit(f"Unknown strategy label(s): {', '.join(unknown)}.\n"
                             f"Available: {', '.join(sorted(available))}")
        found = [e for e in found if e["label"] in strategies]
    if not found:
        raise SystemExit(f"No selection experiments found for dataset {dataset!r}.")

    # Group by encoder, then pooling, so neighbouring panels are the ones
    # worth comparing.
    found.sort(key=lambda e: (e["family"], e["task"] != "rationale", str(e["pooling"])))
    panels = {e["label"]: load_matrix(e["runs"], dataset, metric) for e in found}

    p05 = Z_CRIT_P05 if metric == "division" else NEGLOGP_CRIT_P05
    all_abs = [abs(v) for _, m in panels.values() for c in m.values() for v in c if np.isfinite(v)]
    vmax = float(np.percentile(all_abs, 95)) if all_abs else 1.0
    norm = make_flat_grey_norm(vmax, thresh=p05)
    cmap = make_flat_grey_cmap()

    rows_labels = sorted({r for _, m in panels.values() for r in m})
    ncols = max(1, min(ncols, len(panels)))
    nrows = -(-len(panels) // ncols)
    # A single column is the paper's Figure 4 layout, where each panel spans
    # the full width and needs room for ~10 rho ticks; tiled panels are
    # narrower because several sit side by side.
    width = 6.5 if ncols == 1 else 4.4 * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.5 * nrows),
                             squeeze=False, sharex=True)

    im = None
    for ax, (label, (rho, matrix)) in zip(axes.ravel(), panels.items()):
        present = [r for r in rows_labels if r in matrix]
        data = np.array([matrix[r] for r in present])
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, origin="upper")
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present, fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(len(rho)))
        ax.set_xticklabels([f"{r:.1f}" for r in rho], fontsize=6, rotation=45, ha="right")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("ρ", fontsize=8)

    cbar = fig.colorbar(im, ax=list(axes.ravel()), shrink=0.6, pad=0.02)
    cbar.set_label("Preference score" if metric == "division" else "sign × -log10(p)", fontsize=9)
    for sign in (1, -1):
        cbar.ax.axhline(sign * p05, color="#333333", linestyle="--", linewidth=1)
    ticks = sorted(list(cbar.get_ticks()) + [p05])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([("p = 0.05" if t == p05 else f"{t:g}") for t in ticks])

    fig.suptitle(f"{dataset}: selection bias by tag and ρ, per strategy "
                 f"(mean over seeds; blue = over-selected, red = under-selected)", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved plot to {output}")
    for e in found:
        print(f"    {e['label']:24s} {e['signature']}  ({len(e['runs'])} run(s))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="wikiann")
    parser.add_argument("--metric", default="division", choices=["division", "pvalue"])
    parser.add_argument("--strategies", default=None,
                        help="comma-separated subset; default is everything in the store")
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metric_tag = {"division": "div", "pvalue": "pval"}[args.metric]
    selected = [s.strip() for s in args.strategies.split(",")] if args.strategies else None
    # A filtered figure gets the encoders it kept in its name: without this,
    # the paper's Figure 4 (--strategies sbert,e5,llm) and the full
    # all-strategy heatmap overwrite each other's file.
    slug = ""
    if selected:
        families = sorted({s.split("/")[0].split(" ")[0] for s in selected})
        slug = "_" + ("all" if len(families) > 3 else "-".join(families))
    out = args.output or ROOT / "outputs/analysis" / f"heat_{args.dataset}_{metric_tag}{slug}.pdf"
    main(args.dataset, args.metric, selected, args.ncols, out)
