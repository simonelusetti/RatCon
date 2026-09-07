"""Grounding test across every pooling strategy, aggregated over seeds.

Supersedes plot_ner_correlation.py / plot_ner_correlation_rate.py, which
plotted three hardcoded encoders from a single run each. This one:

  * treats a *pooling strategy* as the unit of comparison -- both the
    explicit reductions (bert/mean, bert/max, bert/min) and the trained
    sentence encoders (sbert, e5, pythia), which are themselves pooling
    strategies, just learned ones rather than specified ones;
  * discovers those series from the forge store instead of taking
    signatures, so there is nothing to keep in sync by hand;
  * uses EVERY run of each experiment rather than the most recent, plotting
    the per-tag mean as a point and the seed spread as a shadow.

Each panel is one strategy: x is its selection bias for an entity tag, y is
how well an MLP probe recovers that tag from the same frozen token encoder.
A positive slope means "the tokens it keeps are the ones it understands".

Usage:
  python3 utils/plot_ner_grounding.py [--dataset wikiann]
      [--metric selection|division] [--rho 0.3 | --rho-average]
      [--spread sd|sem] [--output PATH]

  # Canonical, non-overlapping view: every experiment in each of sentence
  # selectors / token+pooling selectors / sentence oracles / token+pooling
  # oracles, aggregating only its newest three completed runs.
  python3 utils/plot_ner_grounding.py --grouped-latest --variant both


On the error shadows
--------------------
The shadow is an AXIS-ALIGNED ellipse, deliberately: x and y come from two
*different and unpaired* sets of runs. x is measured on the selector's
bias-test runs, y on the NER probe's runs -- separate trainings that share
only a token encoder. Seed 0 of one has no correspondence to seed 0 of the
other, so pairing them to estimate a covariance would manufacture a
correlation out of an arbitrary ordering. With no meaningful joint
distribution, the honest region is the product of the two marginals, which
is an ellipse with no tilt.

The radii default to one standard deviation across seeds (`--spread sd`),
matching the +/-std bands this repo already uses for its multi-seed figures
(see plot_stsb_sufficiency.py). SD answers "how much does an individual run
move"; `--spread sem` divides by sqrt(n) to answer the narrower "how
precisely is the mean pinned down" -- with n=3 that band is optimistic, so
it is not the default.

Note that the three bert/* strategies share ONE NER probe: the probe reads
word-level token embeddings and never pools, so pooling cannot change it.
Their y values and y spreads are therefore identical by construction, and
only x differs -- which is exactly the controlled comparison this figure is
for.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from omegaconf import OmegaConf
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.core import ExperimentStore  # noqa: E402

from src.data import LABEL_DISPLAY_NAMES  # noqa: E402

# Background / non-entity classes: not entity types, and "O" in particular
# would dwarf every real tag's effect and flatten the axis.
EXCLUDE_LABELS = {"O", "special"}

# Three channels, because a strategy is a (token encoder, pooling) pair and
# each strategy contains several entity tags: colour is the TOKEN ENCODER,
# line style is the POOLING, and point marker is the ENTITY TAG. This keeps
# every tag identifiable even in the 3 encoders x 3 poolings figure without
# overloading the plot with repeated text annotations.
#
# Fixed order, never cycled. Okabe-Ito, checked with the palette validator:
# worst adjacent CVD separation dE 9.6 (deutan), worst normal-vision pair
# 16.4, both clear of their floors. sbert/e5/llm keep the assignments the
# earlier figures used so a strategy does not change colour between plots.
ENCODER_COLOR = {
    "sbert":   "#0072B2",
    "e5":      "#E69F00",
    "llm":     "#009E73",
    "bert":    "#D55E00",
    "electra": "#CC79A7",
    "roberta": "#56B4E9",
}

# The marker half of each tuple is retained for compatibility with the older
# single-legend layout; render() now uses only the line style because markers
# are reserved for entity tags.
POOLING_STYLE = {
    "mean": ("o", "-"),
    "max":  ("^", "--"),
    "min":  ("v", ":"),
    "last": ("P", "-."),
    None:   ("s", "-"),
}

# Supports WikiAnn's six entity tags and CoNLL-2003's eight without cycling.
# Pooling no longer consumes this visual channel, so the mapping is stable in
# every selector/oracle group.
TAG_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*")
_FALLBACK_HUE = "#666666"

DISPLAY_NAME = {"llm": "pythia"}

SENTENCE_ENCODERS = {"sbert", "e5", "llm"}
TOKEN_ENCODERS = {"bert", "electra", "roberta"}

# Ordered deliberately: this is also the order in which grouped plots are
# written.
GROUNDING_GROUPS = {
    "sentence": {
        "title": "sentence encoders — selectors",
        "task": "rationale",
        "families": SENTENCE_ENCODERS,
    },
    "token_pooling": {
        "title": "token encoder + pooling — selectors",
        "task": "rationale",
        "families": TOKEN_ENCODERS,
    },
    "sentence_oracle": {
        "title": "sentence encoders — oracles",
        "task": "oracle",
        "families": SENTENCE_ENCODERS,
    },
    "token_pooling_oracle": {
        "title": "token encoder + pooling — oracles",
        "task": "oracle",
        "families": TOKEN_ENCODERS,
    },
}


def entity_tags(dataset: str) -> list[str]:
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    return [name for name in label_map.values() if name not in EXCLUDE_LABELS]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def series_label(family: str, pooling: str | None, task: str = "rationale") -> str:
    """`bert/max` for an explicit strategy, bare `sbert` for a trained one.

    Runs made before data.encoder.pooling existed have pooling=None; they are
    labelled by family alone rather than guessing a strategy for them.

    The task suffix matters: the brute-force oracle (task=oracle) shares a
    family and pooling with the selector it is the ceiling for, so without it
    the two would collide on one key and the dict would keep whichever was
    discovered last -- silently plotting one and dropping the other.
    """
    base = f"{family}/{pooling}" if pooling else family
    return base if task == "rationale" else f"{base} [{task}]"


def discover_series(dataset: str) -> list[dict]:
    """Pair every selector experiment with the NER probe of its token encoder.

    The probe is keyed by family alone: it never pools, so one probe serves
    every pooling strategy built on the same token encoder.
    """
    store = ExperimentStore(root=ROOT / "outputs")
    selectors: dict[str, dict] = {}
    probes: dict[str, dict] = {}

    for selection in store.all_selections():
        cfg = selection.experiment.config
        if str(OmegaConf.select(cfg, "data.dataset")) != dataset:
            continue
        done_runs = sorted(
            (r for r in (selection.runs or []) if r.status == "done"),
            key=lambda r: r.launched_on,
        )
        if not done_runs:
            continue
        done = [r.path for r in done_runs]
        newest_launch = done_runs[-1].launched_on

        family = str(OmegaConf.select(cfg, "data.encoder.family"))
        if str(OmegaConf.select(cfg, "task")) == "ner":
            # Several probe experiments for one family are possible after a
            # config migration. Pooling never affects a probe, so use only
            # the newest experiment rather than silently merging incompatible
            # probe configurations or relying on store traversal order.
            previous = probes.get(family)
            if previous is None or newest_launch > previous["launched_on"]:
                probes[family] = {"runs": done, "launched_on": newest_launch}
        else:
            pooling = OmegaConf.select(cfg, "data.encoder.pooling")
            label = series_label(family, None if pooling is None else str(pooling),
                                 str(OmegaConf.select(cfg, "task")))
            candidate = {
                "label": label,
                "family": family,
                "task": str(OmegaConf.select(cfg, "task")),
                "pooling": None if pooling is None else str(pooling),
                "signature": selection.experiment.signature,
                "bias_runs": done,
                "launched_on": newest_launch,
            }
            # Same label can occur on both sides of a config migration. A
            # grounding series is one experiment, never a mixture; retain the
            # newest completed one deterministically.
            previous = selectors.get(label)
            if previous is None or newest_launch > previous["launched_on"]:
                selectors[label] = candidate

    series = []
    order = list(ENCODER_COLOR)
    for label in sorted(selectors, key=lambda k: (order.index(selectors[k]["family"])
                                                  if selectors[k]["family"] in order else 99, k)):
        entry = selectors[label]
        probe = probes.get(entry["family"])
        if not probe:
            print(f"skipping {label}: no completed NER probe for token encoder "
                  f"{entry['family']!r} (run `forge -M ner_probe grid ...`)", file=sys.stderr)
            continue
        entry["ner_runs"] = probe["runs"]
        series.append(entry)
    return series


def grouped_series_with_latest_runs(series: list[dict], count: int = 3) -> dict[str, list[dict]]:
    """Every experiment in each group, capped to its newest *count* runs.

    Bias and NER runs are capped independently because they are unpaired
    experiments. The plot combines their marginal mean/spread, never a
    run-by-run covariance (see the module docstring).
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    grouped: dict[str, list[dict]] = {}
    for slug, spec in GROUNDING_GROUPS.items():
        candidates = [
            entry for entry in series
            if entry["task"] == spec["task"] and entry["family"] in spec["families"]
        ]
        candidates.sort(key=lambda entry: (
            list(ENCODER_COLOR).index(entry["family"])
            if entry["family"] in ENCODER_COLOR else 99,
            entry["pooling"] or "",
            entry["label"],
        ))
        grouped[slug] = [
            {
                **entry,
                "bias_runs": entry["bias_runs"][-count:],
                "ner_runs": entry["ner_runs"][-count:],
            }
            for entry in candidates
        ]
    return grouped


# ---------------------------------------------------------------------------
# Per-run loaders (one run directory in, one value per tag out)
# ---------------------------------------------------------------------------

def load_bias(run_path: Path, dataset: str, metric: str, rho: float | None) -> dict[str, float]:
    """Selection bias per entity tag for a single bias-test run.

    metric="selection": the observed selection rate minus rho. Under the
    null every token is kept with probability rho regardless of label, so
    this is the raw over/under-selection in probability units -- an effect
    size, unscaled by how variable that tag's count could be by chance.
    metric="division": the same effect divided by the exact null's standard
    deviation, i.e. the signed z-score from the hypergeometric-convolution
    test. Standardising this way lets a rare tag's small raw excess count
    for as much as a common tag's large one. It is an effect size, not a
    significance value -- the p-value is a separate artifact
    (pvalue_curves.json).

    rho=None averages over every rho except 1.0, where keeping everything
    forces the effect to 0 by construction rather than by measurement.
    """
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    filename = "selection_rate_curves.json" if metric == "selection" else "effect_size_curves.json"
    payload = json.loads((run_path / "data" / filename).read_text())
    rhos = np.array(payload["rho"], dtype=float)
    offset = rhos if metric == "selection" else np.zeros_like(rhos)

    # label_map.get(idx, idx), not label_map[idx]: datasets with an upstream
    # ClassLabel (wikiann, conll2003) key their curves by stringified index,
    # but conll2000 has none -- NLTK yields the tag string itself, so its
    # artifacts are keyed "B-NP" directly. Requiring a map hit silently
    # dropped every conll2000 tag and produced an empty result rather than
    # an error.
    if rho is None:
        keep = ~np.isclose(rhos, 1.0)
        return {
            label_map.get(idx, idx): float(np.mean(np.array(curve, dtype=float)[keep] - offset[keep]))
            for idx, curve in payload["curves"].items()
        }
    i = int(np.argmin(np.abs(rhos - rho)))
    return {
        label_map.get(idx, idx): float(curve[i] - offset[i])
        for idx, curve in payload["curves"].items()
    }


def load_ner_f1(run_path: Path, tags: list[str]) -> dict[str, float]:
    report = json.loads((run_path / "data" / "ner_classification_report.json").read_text())
    return {tag: float(report["token_level"][tag]["f1-score"]) for tag in tags}


def load_binary_entity_f1(run_path: Path) -> float:
    """Token-level F1 of "is this token part of any entity" (B-*/I-* collapsed
    against O) -- a real detection metric rather than an average of per-tag F1s."""
    report = json.loads((run_path / "data" / "ner_classification_report.json").read_text())
    return float(report["binary_entity_level"]["entity"]["f1-score"])


def mean_spread(values: list[float], spread: str) -> tuple[float, float]:
    """Mean and its shadow radius over seeds. ddof=1: these are a sample of
    seeds, not the population of them."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()), 0.0
    sd = float(arr.std(ddof=1))
    return float(arr.mean()), sd / np.sqrt(arr.size) if spread == "sem" else sd


def safe_pearson(x, y) -> float:
    """Pearson r, or NaN when either side is constant/too short.

    Constant y is expected when several pooling strategies share one token
    encoder: they necessarily share the same NER probe, so there is no
    across-strategy NER variance to correlate against.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 3 or y_arr.size != x_arr.size:
        return float("nan")
    if np.isclose(x_arr.std(), 0.0) or np.isclose(y_arr.std(), 0.0):
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def style_axis(ax) -> None:
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")


def build_points(series, tags, dataset, metric, rho, spread, absolute):
    """points[label][tag] = (x_mean, x_radius, y_mean, y_radius).

    With absolute=True the magnitude is taken PER RUN before averaging, not
    on the averaged value: the question is "how far from chance was this run",
    so a tag whose bias flips sign across seeds must not cancel itself out.
    """
    points, binary_f1 = {}, {}
    for entry in series:
        bias_runs = [load_bias(p, dataset, metric, rho) for p in entry["bias_runs"]]
        f1_runs = [load_ner_f1(p, tags) for p in entry["ner_runs"]]
        points[entry["label"]] = {
            tag: (*mean_spread([abs(b[tag]) if absolute else b[tag] for b in bias_runs], spread),
                  *mean_spread([f[tag] for f in f1_runs], spread))
            for tag in tags
        }
        binary_f1[entry["label"]] = mean_spread(
            [load_binary_entity_f1(p) for p in entry["ner_runs"]], spread
        )
    return points, binary_f1


def render(series, tags, points, binary_f1, dataset, metric, rho, spread, absolute, output,
           group_title: str | None = None):
    labels = [e["label"] for e in series]
    within = {}
    for label in labels:
        xs = np.array([points[label][t][0] for t in tags])
        ys = np.array([points[label][t][2] for t in tags])
        within[label] = safe_pearson(xs, ys)
    agg = np.array([np.mean([abs(points[l][t][0]) for t in tags]) for l in labels])
    across = safe_pearson(agg, np.array([binary_f1[l][0] for l in labels]))

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    style_axis(ax)
    if not absolute:
        ax.axvline(0.0, color="#BBBBBB", linewidth=1.0, linestyle=":", zorder=1)

    strategy_handles = []
    for entry in series:
        label = entry["label"]
        color = ENCODER_COLOR.get(entry["family"], _FALLBACK_HUE)
        _, linestyle = POOLING_STYLE.get(entry["pooling"], ("o", "-"))
        # The oracle shares an encoder AND a pooling with the selector it is
        # the ceiling for. Mixed legacy plots still need a distinct line;
        # grouped oracle plots already separate the task in their own figure.
        if entry.get("task") == "oracle" and group_title is None:
            linestyle = "-."
        for tag_i, tag in enumerate(tags):
            xm, xr, ym, yr = points[label][tag]
            if xr > 0 or yr > 0:
                ax.add_patch(Ellipse((xm, ym), width=2 * xr, height=2 * yr,
                                     facecolor=color, alpha=0.18, edgecolor="none", zorder=2))
            ax.scatter([xm], [ym], color=color, marker=TAG_MARKERS[tag_i],
                       s=58, edgecolors="white", linewidths=0.8, zorder=4)
        # Least-squares line per strategy: makes the sign of each correlation
        # readable directly, which is the whole point of overlaying them.
        xs = np.array([points[label][t][0] for t in tags])
        ys = np.array([points[label][t][2] for t in tags])
        slope, intercept = np.polyfit(xs, ys, 1)
        span = np.linspace(xs.min(), xs.max(), 2)
        ax.plot(span, slope * span + intercept, color=color, linewidth=1.5,
                linestyle=linestyle, alpha=0.65, zorder=3)
        r_label = f"{within[label]:+.2f}" if np.isfinite(within[label]) else "n/a"
        strategy_handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                                       linewidth=1.8,
                                       label=f"{DISPLAY_NAME.get(label, label)}  (r = {r_label})"))

    first = ax.legend(handles=strategy_handles, title="encoder x pooling", frameon=False,
                      fontsize=8, title_fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      ncol=1)
    ax.add_artist(first)
    tag_handles = [
        Line2D([0], [0], color="#666666", marker=TAG_MARKERS[i], linestyle="none",
               markerfacecolor="#888888", markeredgecolor="white", markersize=7, label=tag)
        for i, tag in enumerate(tags)
    ]
    second = ax.legend(handles=tag_handles, title="entity tag", frameon=False,
                       fontsize=8, title_fontsize=9, loc="lower left",
                       bbox_to_anchor=(1.01, 0.0))

    base = ("Over/under-selection rate (observed - rho)" if metric == "selection"
            else "Preference score (signed z, effect / null SD)")
    if absolute:
        base = ("|over/under-selection rate|" if metric == "selection"
                else "|preference score| (|z|, distance from chance in null SDs)")
    rho_desc = "averaged over rho" if rho is None else f"at rho={rho:g}"
    ax.set_xlabel(f"{base}, {rho_desc}", fontsize=10)
    ax.set_ylabel("NER token-level F1", fontsize=10)

    band = "±1 SD" if spread == "sd" else "±1 SEM"
    question = ("does it treat the tags it understands differently, either way?"
                if absolute else "does it keep the tags it understands?")
    title_scope = f"{dataset} — {group_title}" if group_title else dataset
    ax.set_title(f"{title_scope}: {question}\nmean over seeds, shadow = {band}", fontsize=11)

    # Reserve the right margin explicitly rather than calling tight_layout():
    # tight_layout re-fits the axes to the *figure* and does not know about
    # legends anchored outside it, so it grows the axes over them and the
    # bbox_inches="tight" pass then crops mid-label.
    fig.subplots_adjust(left=0.08, right=0.74, top=0.90, bottom=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    # bbox_extra_artists is required: a "tight" bbox only measures artists
    # parented to the axes, so legends anchored outside it get cropped
    # mid-label unless they are named here explicitly.
    fig.savefig(output, bbox_inches="tight", bbox_extra_artists=(first, second))
    print(f"Saved plot to {output}")
    for label in labels:
        r_text = f"{within[label]:+.4f}" if np.isfinite(within[label]) else "n/a"
        print(f"    {DISPLAY_NAME.get(label, label):11s} r={r_text}")
    across_text = f"{across:+.4f}" if np.isfinite(across) else "n/a"
    print(f"    across-strategy r={across_text} (n={len(labels)}, low power)")


def main(dataset: str, metric: str, rho: float | None, spread: str,
         strategies: list[str] | None, variant: str, output: Path | None) -> None:
    tags = entity_tags(dataset)
    series = discover_series(dataset)
    if strategies:
        available = {e["label"] for e in series}
        unknown = [s for s in strategies if s not in available]
        if unknown:
            raise SystemExit(
                f"Unknown strategy label(s): {', '.join(unknown)}.\n"
                f"Available: {', '.join(sorted(available))}"
            )
        series = [e for e in series if e["label"] in strategies]
    if not series:
        raise SystemExit(
            f"No (selector, NER probe) pairs found for dataset {dataset!r}. "
            f"Run a selector grid and `forge -M ner_probe grid --file utils/grid_ner.yaml` first."
        )

    print(f"seeds per strategy (bias runs / probe runs):")
    for entry in series:
        print(f"  {entry['label']:11s} {len(entry['bias_runs'])} / {len(entry['ner_runs'])}")

    # Short but still readable: metric (sel|div), rho (avg|0.4), sign
    # (sgn|abs). A filtered run adds the encoders it kept, with the pooling
    # dropped when every series shares one -- it is the varying part that
    # needs to be in the name.
    suffix = "avg" if rho is None else f"{rho:g}"
    metric_tag = {"selection": "sel", "division": "div"}[metric]
    wanted = {"signed": (False,), "absolute": (True,), "both": (False, True)}[variant]
    for absolute in wanted:
        points, binary_f1 = build_points(series, tags, dataset, metric, rho, spread, absolute)
        sign_tag = "abs" if absolute else "sgn"
        if output is not None:
            out = output.with_name(f"{output.stem}_{sign_tag}{output.suffix}")
        else:
            slug = ""
            if strategies:
                # Name by what varies, compactly: the distinct encoders, or
                # "all" once there are enough that listing them is longer
                # than it is useful, plus a marker for which kinds of run are
                # present. Enumerating every (family, pooling, task) triple
                # produced names over 100 characters long.
                fams = sorted({e["family"] for e in series})
                kinds = {e["task"] for e in series}
                kind = {("oracle",): "orc", ("rationale",): "sel"}.get(
                    tuple(sorted(kinds)), "selorc")
                slug = "_" + ("all" if len(fams) > 3 else "-".join(fams)) + "-" + kind
            out = (ROOT / "outputs/analysis" /
                   f"ground_{dataset}_{metric_tag}_{suffix}{slug}_{sign_tag}.pdf")
        print()
        render(series, tags, points, binary_f1, dataset, metric, rho, spread, absolute, out)


def main_grouped_latest(dataset: str, metric: str, rho: float | None, spread: str,
                        variant: str, count: int, output_dir: Path) -> None:
    """Write four canonical groups using each experiment's newest runs."""
    tags = entity_tags(dataset)
    groups = grouped_series_with_latest_runs(discover_series(dataset), count=count)
    wanted = {"signed": (False,), "absolute": (True,), "both": (False, True)}[variant]
    output_dir.mkdir(parents=True, exist_ok=True)

    for slug, series in groups.items():
        if not series:
            raise SystemExit(
                f"Group {slug!r} has no completed experiments paired with an NER probe."
            )

        print(f"\n{slug}: {len(series)} experiments, newest {count} run(s) per experiment")
        for entry in series:
            bias_run_ids = [path.name for path in entry["bias_runs"]]
            probe_run_ids = [path.name for path in entry["ner_runs"]]
            print(
                f"  {entry['signature']}  {entry['label']}: "
                f"bias={','.join(bias_run_ids)} probe={','.join(probe_run_ids)}"
            )

        for absolute in wanted:
            points, binary_f1 = build_points(
                series, tags, dataset, metric, rho, spread, absolute
            )
            sign_tag = "absolute" if absolute else "signed"
            output = output_dir / f"grounding_{slug}_{sign_tag}.pdf"
            render(
                series, tags, points, binary_f1, dataset, metric, rho, spread,
                absolute, output, group_title=GROUNDING_GROUPS[slug]["title"],
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="wikiann", choices=["wikiann", "conll2003"])
    parser.add_argument("--metric", default="selection", choices=["selection", "division"],
                        help="selection: observed-minus-rho, the raw effect size (default); "
                             "division: that effect divided by the exact null's SD (signed z)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rho", type=float, default=None)
    group.add_argument("--rho-average", dest="rho_average", action="store_true",
                       help="average over every rho except 1.0 (the default)")
    parser.add_argument("--spread", default="sd", choices=["sd", "sem"],
                        help="shadow radius: sd = seed-to-seed spread (default); "
                             "sem = uncertainty of the plotted mean")
    parser.add_argument("--variant", default="both", choices=["both", "signed", "absolute"],
                        help="which sign convention to emit (default both)")
    parser.add_argument("--strategies", default=None,
                        help="comma-separated subset to plot, e.g. "
                             "'bert/mean,electra/mean,roberta/mean'; default is every "
                             "strategy found in the store")
    parser.add_argument("--output", type=Path, default=None,
                        help="base path; _sgn / _abs is inserted before the suffix")
    parser.add_argument("--grouped-latest", action="store_true",
                        help="write four non-overlapping plots using every experiment in each "
                             "encoding/task group and only its newest runs")
    parser.add_argument("--latest-count", type=int, default=3,
                        help="newest completed runs retained per experiment (default 3)")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/analysis/grounding_latest",
                        help="directory used by --grouped-latest")
    args = parser.parse_args()

    selected = [s.strip() for s in args.strategies.split(",")] if args.strategies else None
    if args.grouped_latest:
        if selected or args.output is not None:
            parser.error("--grouped-latest cannot be combined with --strategies or --output; use --output-dir")
        main_grouped_latest(
            args.dataset, args.metric, args.rho, args.spread, args.variant,
            args.latest_count, args.output_dir,
        )
    else:
        main(args.dataset, args.metric, args.rho, args.spread, selected, args.variant, args.output)
