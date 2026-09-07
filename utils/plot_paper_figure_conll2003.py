"""Paper figure: preference-vs-F1 scatter (left) next to NER training-progress
curves split by B/I (right), for conll2003. Per the supervisor's simplification
request: no titles, no correlation numbers in the plot, just axis labels and
legends. Per-point tag labels are kept on the scatter panel.

Usage: python3 utils/plot_paper_figure_conll2003.py [--rho 0.8] [--output PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.plot_ner_correlation import (  # noqa: E402
    ENCODER_COLOR,
    ENCODER_DISPLAY,
    ENCODER_MARKER,
    add_sig_args,
    entity_tags,
    load_ner_f1,
    sigs_from_args,
)
from utils.plot_ner_correlation_rate import load_bias_rate  # noqa: E402
from utils.plot_ner_convergence import BIO_LINESTYLE, load_history  # noqa: E402

DATASET = "conll2003"


def draw_scatter(ax, bias_sigs: dict[str, str], ner_sigs: dict[str, str], rho: float) -> None:
    tags = entity_tags(DATASET)

    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

    for encoder in bias_sigs:
        rate_by_tag = load_bias_rate(DATASET, bias_sigs[encoder], rho)
        f1_by_tag = load_ner_f1(ner_sigs[encoder], tags)
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
    ax.set_ylabel("NER token-level F1")
    ax.legend(title="encoder", frameon=False, loc="lower right")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.12 * (yhi - ylo))


def draw_convergence(ax, ner_sigs: dict[str, str]) -> None:
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

    for encoder, sig in ner_sigs.items():
        epochs, b_f1s, i_f1s = load_history(sig)
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
        for e in ner_sigs
    ]
    bio_handles = [
        Line2D([0], [0], color="#555555", linestyle=BIO_LINESTYLE[p], linewidth=1.6, label=f"{p}-*")
        for p in ("B", "I")
    ]
    legend1 = ax.legend(handles=encoder_handles, title="encoder", frameon=False, loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=bio_handles, title="tag type", frameon=False, loc="lower center")


def main(bias_sigs: dict[str, str], ner_sigs: dict[str, str], rho: float, output: Path) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.2))
    draw_scatter(ax_left, bias_sigs, ner_sigs, rho)
    draw_convergence(ax_right, ner_sigs)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_sig_args(parser, dataset_for_defaults=DATASET)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bias_sigs, ner_sigs = sigs_from_args(args)
    output = args.output or ROOT / "outputs/analysis" / "paper_figure_conll2003.pdf"
    main(bias_sigs, ner_sigs, args.rho, output)
