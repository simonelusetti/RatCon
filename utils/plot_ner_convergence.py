"""NER MLP probe training-progress curves, one line per (encoder, B/I) pair,
for a single dataset -- token-level F1 macro-averaged separately across all
B-* tags and all I-* tags per epoch (span-level has no B/I distinction, since
it collapses B/I into whole entity spans). Independent of the bias-test rho
-- rho is a retention-rate parameter for the exact preference test on the
rationale-selection task and plays no role in training the NER probe, which
has its own separate per-epoch training loop.

Usage: python3 utils/plot_ner_convergence.py [--dataset conll2003] [--output PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ner  # noqa: E402
from utils._common import ENCODER_COLOR, ENCODER_DISPLAY, ENCODER_MARKER  # noqa: E402

BIO_LINESTYLE = {"B": "-", "I": "--"}


def load_history(dataset: str, family: str) -> tuple[list[int], list[float], list[float]]:
    """(epochs, macro-avg F1 over B-* tags, macro-avg F1 over I-* tags).

    Averaged over every cached probe seed. The per-epoch history comes from
    the probe cache (see `ner/`); probes are not forge experiments, so there
    is no signature to pass.
    """
    reports = [r for r in ner.load(dataset, family) if r.get("history")]
    if not reports:
        raise SystemExit(
            f"No cached NER probe history for {dataset}/{family}. "
            f"Build one with: python -m ner {dataset} --family {family}")

    def macro(token: dict, prefix: str) -> float:
        scores = [v["f1-score"] for k, v in token.items() if k.startswith(prefix)]
        return sum(scores) / len(scores)

    epochs = [entry["epoch"] for entry in reports[0]["history"]]
    curves = [[[macro(e["token_level"], p) for e in r["history"]] for r in reports]
              for p in ("B-", "I-")]
    means = [[sum(seed[i] for seed in c) / len(c) for i in range(len(epochs))] for c in curves]
    return epochs, means[0], means[1]


def main(dataset: str, encoders: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
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
    ax.set_title(f"{dataset}: NER probe training progress, by encoder and B/I")

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

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default="conll2003", choices=["wikiann", "conll2003"])
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--encoders", default="sbert,e5,llm",
                        help="comma-separated token encoders whose cached probes to plot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/analysis" / f"conv_{args.dataset}.pdf"
    main(args.dataset, [e.strip() for e in args.encoders.split(",") if e.strip()], output)
