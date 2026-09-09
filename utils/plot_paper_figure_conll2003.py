"""Paper figure: preference-vs-F1 scatter (left) next to tagger training-progress
curves split by B/I (right), for conll2003. Per the supervisor's simplification
request: no titles, no correlation numbers in the plot, just axis labels and
legends. Per-point tag labels are kept on the scatter panel.

Usage: python3 utils/plot_paper_figure_conll2003.py [--rho 0.8] [--output PATH]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from utils._common import (  # noqa: E402
    ENCODER_COLOR,
    ENCODER_DISPLAY,
    ENCODER_MARKER,
    entity_tags,
    load_bias,
    tagger_f1,
    probe_reports,
)
from utils.forge_paths import run_dir  # noqa: E402
from utils.plot_convergence import BIO_LINESTYLE, load_history  # noqa: E402

DATASET = "conll2003"
ENCODERS = ("sbert", "e5", "llm")


def draw_scatter(ax, dataset: str, bias_sigs: dict[str, str], rho: float) -> None:
    tags = entity_tags(DATASET)

    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

    for encoder in bias_sigs:
        rate_by_tag = load_bias(run_dir(bias_sigs[encoder]), dataset, "selection", rho)
        # Averaged over the probe cache's seeds; this panel shows one point
        # per tag, so the spread is not drawn (see plot_ner_grounding.py for
        # the version that does).
        per_seed = [tagger_f1(r, tags) for r in probe_reports(dataset, encoder)]
        f1_by_tag = {t: float(np.mean([f[t] for f in per_seed])) for t in tags}
        xs = [rate_by_tag[t] for t in tags]
        ys = [f1_by_tag[t] for t in tags]
        ax.scatter(
            xs, ys,
            color=ENCODER_COLOR[encoder],
            marker=ENCODER_MARKER[encoder],
            s=70,
            edgecolors="white",
            linewidths=0.8,
            label=ENCODER_DISPLAY[encoder],
            zorder=3,
        )
        for tag, x, y in zip(tags, xs, ys):
            ax.annotate(
                tag, (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color="#555555",
            )

    ax.set_xlabel("Over/under-selection rate")
    ax.set_ylabel("tagger token-level F1")
    ax.legend(title="encoder", frameon=False, loc="lower right")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.12 * (yhi - ylo))


def draw_convergence(ax, dataset: str, encoders: list[str]) -> None:
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

    for encoder in encoders:
        epochs, b_f1s, i_f1s = load_history(dataset, encoder)
        for prefix, f1s in (("B", b_f1s), ("I", i_f1s)):
            ax.plot(
                epochs, f1s,
                color=ENCODER_COLOR[encoder],
                linestyle=BIO_LINESTYLE[prefix],
                marker=ENCODER_MARKER[encoder],
                markersize=5,
                linewidth=1.6,
                zorder=3,
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("token-level macro F1")

    encoder_handles = [
        Line2D([0], [0], color=ENCODER_COLOR[e], marker=ENCODER_MARKER[e], linewidth=1.6, label=ENCODER_DISPLAY[e])
        for e in encoders
    ]
    bio_handles = [
        Line2D([0], [0], color="#555555", linestyle=BIO_LINESTYLE[p], linewidth=1.6, label=f"{p}-*")
        for p in ("B", "I")
    ]
    legend1 = ax.legend(handles=encoder_handles, title="encoder", frameon=False, loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=bio_handles, title="tag type", frameon=False, loc="lower center")


def main(bias_sigs: dict[str, str], rho: float, output: Path) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.2))
    draw_scatter(ax_left, "conll2003", bias_sigs, rho)
    draw_convergence(ax_right, "conll2003", list(bias_sigs))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Only the bias side needs a signature. The tagger is a cache keyed by
    # (dataset, encoder family), so naming the encoder already names it.
    for enc in ENCODERS:
        parser.add_argument(f"--{enc}-bias-sig", required=True,
                            help=f"bias-test (rationale) forge sig for {ENCODER_DISPLAY[enc]}")
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bias_sigs = {enc: getattr(args, f"{enc}_bias_sig") for enc in ENCODERS}
    output = args.output or ROOT / "outputs/analysis" / "paper_figure_conll2003.pdf"
    main(bias_sigs, args.rho, output)
