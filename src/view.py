import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, FuncNorm
import matplotlib.pyplot as plt
import numpy as np

from .data import LABEL_DISPLAY_NAMES


_DEFAULT_LOSS_HISTORY_PATH = Path("data") / "loss_history.json"
_DEFAULT_LOSS_PLOT_PATH = Path("plots") / "loss.pdf"
_DEFAULT_SELECTION_RATE_CURVES_PATH = Path("data") / "selection_rate_curves.json"
_DEFAULT_PVALUE_CURVES_PATH = Path("data") / "pvalue_curves.json"
_DEFAULT_EFFECT_SIZE_CURVES_PATH = Path("data") / "effect_size_curves.json"
_DEFAULT_SELECTION_RATE_PLOT_PATH = Path("plots") / "selection_rate_vs_rho.pdf"
_DEFAULT_PVALUE_PLOT_PATH = Path("plots") / "pvalue_vs_rho.pdf"
_DEFAULT_EFFECT_SIZE_PLOT_PATH = Path("plots") / "effect_size_vs_rho.pdf"
_DEFAULT_SIGNED_EFFECT_HEATMAP_PLOT_PATH = Path("plots") / "signed_effect_heatmap.pdf"
_DEFAULT_SPEARMAN_CURVES_PATH = Path("data") / "spearman_curves.json"
_DEFAULT_SPEARMAN_PLOT_PATH = Path("plots") / "spearman_vs_rho.pdf"


_METRIC_TO_DATA_PATH = {
    "selection_rate": _DEFAULT_SELECTION_RATE_CURVES_PATH,
    "pvalue": _DEFAULT_PVALUE_CURVES_PATH,
    "effect_size": _DEFAULT_EFFECT_SIZE_CURVES_PATH,
    "signed_effect": _DEFAULT_EFFECT_SIZE_CURVES_PATH,
    "spearman": _DEFAULT_SPEARMAN_CURVES_PATH,
}

_METRIC_TO_PLOT_PATH = {
    "selection_rate": _DEFAULT_SELECTION_RATE_PLOT_PATH,
    "pvalue": _DEFAULT_PVALUE_PLOT_PATH,
    "effect_size": _DEFAULT_EFFECT_SIZE_PLOT_PATH,
    "signed_effect": _DEFAULT_SIGNED_EFFECT_HEATMAP_PLOT_PATH,
    "spearman": _DEFAULT_SPEARMAN_PLOT_PATH,
}


def _plot_metric_from_artifact(metric_name: str, ylabel: str) -> Path:
    data_path = _METRIC_TO_DATA_PATH[metric_name]
    out_path = _METRIC_TO_PLOT_PATH[metric_name]

    metric_payload = _load_json(data_path)
    parsed = maybe_extract_metric_payload(metric_payload)
    if parsed is None:
        raise ValueError(f"Metric missing or invalid in artifact: {data_path}")

    x, curves, baseline = parsed
    if not curves:
        raise ValueError(f"Metric curves are empty in artifact: {data_path}")

    fig, ax = plt.subplots(figsize=(7, 5))

    if isinstance(baseline, Mapping):
        baseline_kind = baseline.get("kind")
        if baseline_kind == "constant":
            try:
                baseline_value = float(baseline.get("value"))
            except (TypeError, ValueError):
                baseline_value = None
            if baseline_value is not None:
                baseline_label = str(baseline.get("label", "baseline"))
                ax.plot(x, [baseline_value] * len(x), "--", label=baseline_label)
        elif baseline_kind == "identity":
            baseline_label = str(baseline.get("label", "baseline"))
            ax.plot(x, x, "--", label=baseline_label)

    display_labels = {
        "selector": "Trained selector",
        "random": "Random selector",
    }
    for label, values in sorted(curves.items(), key=lambda kv: kv[0]):
        ax.plot(x, values, marker="o", linewidth=2.0, label=display_labels.get(label, label))

    ax.set_xlabel("Selection rate (rho)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_selection_rate_curves() -> Path:
    return _plot_metric_from_artifact("selection_rate", "Selection rate")


def plot_pvalue_curves() -> Path:
    return _plot_metric_from_artifact("pvalue", "-log10(p-value)")


def plot_effect_size_curves() -> Path:
    return _plot_metric_from_artifact("effect_size", "Effect size (z-score)")


def plot_spearman_curves() -> Path:
    return _plot_metric_from_artifact("spearman", "Spearman correlation (STS-B)")


# ---------------------------------------------------------------------------
# Signed effect-size colormap tuning
# _Z_THRESH     : two-sided z-critical value at alpha=0.05 (~1.96).
# _GREY_FRACTION: fraction of the [0,1] colormap range devoted to the grey
#                 neutral zone (values within ±threshold).  Increase to push
#                 colour further from centre; decrease for a more gradual fade.
# ---------------------------------------------------------------------------
_Z_THRESH        = 1.9599639845400545  # two-sided z-critical value, alpha=0.05
_COLOR_GAMMA     = 0.3                 # power curve: <1 → saturates fast, >1 → gradual
_GREY_HALF_WIDTH = 0.15               # fraction of [0,1] colourmap space for the grey zone
                                       # (each side of 0.5), so grey occupies [0.35, 0.65]
_COLOR_VMAX      = 10.0  # max |z| for colour scaling; actual vmax may be lower if data is less extreme

_SIGNED_EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "RdGreyBu",
    ["#b2182b", "#d9d9d9", "#2166ac"],
)


def _make_signed_effect_norm(vmax: float, thresh: float = _Z_THRESH) -> FuncNorm:
    """
    Custom norm for the signed effect-size heatmap.

    Visual layout of [0, 1]:
      [0,          0.5 - GH]  → negative significant  (red)
      [0.5 - GH,  0.5 + GH]  → non-significant zone   (grey), linear within zone
      [0.5 + GH,  1         ] → positive significant  (blue)

    where GH = _GREY_HALF_WIDTH.  This gives the grey zone visible height in
    the colourbar so the ±thresh (Bonferroni-corrected significance) boundary
    is readable. thresh is the dataset-specific z-critical value (varies with
    m = num_labels x num_rhos being tested; defaults to the uncorrected
    two-sided alpha=0.05 value only when the payload doesn't carry one).
    """
    gamma  = _COLOR_GAMMA
    GH     = _GREY_HALF_WIDTH
    safe_v = max(vmax - thresh, 1e-8)

    # colourmap breakpoints
    g_lo = 0.5 - GH   # bottom of grey zone
    g_hi = 0.5 + GH   # top of grey zone

    def forward(x):
        x   = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        pm = x >  thresh
        nm = x < -thresh
        dm = ~pm & ~nm   # |x| <= thresh → grey

        # grey zone: linear interpolation within [−thresh, +thresh] → [g_lo, g_hi]
        out[dm] = 0.5 + (x[dm] / thresh) * GH

        # positive significant: [thresh, vmax] → [g_hi, 1.0]
        u = np.clip((x[pm] - thresh) / safe_v, 0.0, 1.0)
        out[pm] = g_hi + (1.0 - g_hi) * (u ** gamma)

        # negative significant: [-vmax, -thresh] → [0.0, g_lo]
        u = np.clip((-x[nm] - thresh) / safe_v, 0.0, 1.0)
        out[nm] = g_lo - g_lo * (u ** gamma)

        return np.clip(out, 0.0, 1.0)

    def inverse(y):
        y   = np.asarray(y, dtype=float)
        out = np.empty_like(y)

        pm = y > g_hi
        nm = y < g_lo
        dm = ~pm & ~nm

        # grey zone → data
        out[dm] = (y[dm] - 0.5) / GH * thresh

        # positive significant
        u = np.clip((y[pm] - g_hi) / (1.0 - g_hi), 0.0, 1.0)
        out[pm] = thresh + safe_v * (u ** (1.0 / gamma))

        # negative significant
        u = np.clip((g_lo - y[nm]) / g_lo, 0.0, 1.0)
        out[nm] = -(thresh + safe_v * (u ** (1.0 / gamma)))

        return out

    return FuncNorm((forward, inverse), vmin=-vmax, vmax=vmax)

def _signed_effect_heatmap_from_payload(
    payload: Mapping[str, Any],
    ax: plt.Axes,
    vmax: float | None = None,
    norm: FuncNorm | None = None,
    dataset_name: str | None = None,
) -> plt.cm.ScalarMappable:
    """Render signed effect-size (z-score) heatmap onto ax. Returns the image for colorbar reuse."""
    rho_values = [float(r) for r in payload.get("rho", [])]
    curves_raw = payload.get("curves", {})

    labels = sorted(curves_raw.keys())
    if not labels or not rho_values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return plt.cm.ScalarMappable(cmap=_SIGNED_EFFECT_CMAP)

    matrix = np.array([[float(v) for v in curves_raw[lbl]] for lbl in labels])  # [L, R]

    vmax = vmax if vmax is not None else _COLOR_VMAX
    if norm is None:
        baseline = payload.get("baseline", {})
        thresh = baseline.get("value") if isinstance(baseline, Mapping) else None
        thresh = float(thresh) if isinstance(thresh, (int, float)) and thresh > 0 else _Z_THRESH
        norm = _make_signed_effect_norm(vmax, thresh=thresh)

    label_map = LABEL_DISPLAY_NAMES.get(dataset_name or "", {})
    im = ax.imshow(matrix, aspect="auto", cmap=_SIGNED_EFFECT_CMAP, norm=norm, origin="upper")
    ax.set_xticks(range(len(rho_values)))
    ax.set_xticklabels([f"{r:.2f}" for r in rho_values], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([label_map.get(lbl, lbl) for lbl in labels], fontsize=7)

    # Bonferroni significance markers: dot = p < alpha/m (per-dataset corrected).
    significant = payload.get("significant", {})
    if isinstance(significant, Mapping):
        for li, lbl in enumerate(labels):
            sig_row = significant.get(lbl)
            if not isinstance(sig_row, list) or len(sig_row) != len(rho_values):
                continue
            for ri, sig in enumerate(sig_row):
                if sig:
                    ax.plot(ri, li, marker=".", markersize=3, color="black", alpha=0.7)

    return im


def plot_signed_effect_heatmap(dataset_name: str | None = None) -> Path:
    out_path = _DEFAULT_SIGNED_EFFECT_HEATMAP_PLOT_PATH
    payload = _load_json(_DEFAULT_EFFECT_SIZE_CURVES_PATH)

    rho_values = payload.get("rho", [])
    curves_raw = payload.get("curves", {})
    if not rho_values or not curves_raw:
        raise ValueError(f"Metric missing or invalid in artifact: {_DEFAULT_EFFECT_SIZE_CURVES_PATH}")

    n_labels = len(curves_raw)
    fig_h = max(3.0, n_labels * 0.4 + 1.5)
    fig_w = max(6.0, len(rho_values) * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = _signed_effect_heatmap_from_payload(payload, ax, dataset_name=dataset_name)
    ax.set_xlabel("Selection rate (ρ)", fontsize=9)
    ax.set_title("Signed effect size  (blue = over-selected, red = under-selected)", fontsize=9)
    fig.colorbar(im, ax=ax, label="z-score (signed)", shrink=0.8, extend="both")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_eval_plots(metric_names: Sequence[str], dataset_name: str | None = None) -> dict[str, Path]:
    _plotters = {
        "selection_rate": plot_selection_rate_curves,
        "pvalue": plot_pvalue_curves,
        "effect_size": plot_effect_size_curves,
        "signed_effect": lambda: plot_signed_effect_heatmap(dataset_name=dataset_name),
        "spearman": plot_spearman_curves,
    }
    plot_paths: dict[str, Path] = {}
    for metric_name in metric_names:
        plotter = _plotters.get(metric_name)
        if plotter is None:
            continue
        try:
            plot_paths[metric_name] = plotter()
        except ValueError:
            pass  # artifact exists but has no data (e.g. unlabelled dataset)
    return plot_paths


def save_train_eval_loss_plot(
    ema_alpha: float = 0.2,
) -> None:
    payload = _load_json(_DEFAULT_LOSS_HISTORY_PATH)
    train_loss_history = payload.get("train", []) if isinstance(payload, Mapping) else []
    eval_loss_history = payload.get("eval", []) if isinstance(payload, Mapping) else []

    if not train_loss_history and not eval_loss_history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    train_ax, eval_ax = axes

    alpha = float(ema_alpha)
    if not (0.0 < alpha <= 1.0):
        alpha = 0.2

    def _plot_history(ax, history: Sequence[Mapping[str, float]], title: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if not history:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        epochs = range(1, len(history) + 1)
        loss_keys = list(history[0].keys())
        for key in loss_keys:
            ys = [float(entry.get(key, np.nan)) for entry in history]
            ax.plot(epochs, ys, alpha=0.45, linewidth=1.5, label=f"{key} (raw)")
            ys_ema = _ema(ys, alpha)
            ax.plot(epochs, ys_ema, linewidth=2.2, label=f"{key} (EMA {alpha:.2f})")

        values = [float(v) for entry in history for v in entry.values()]
        if values:
            vmin = min(values)
            vmax = max(values)
            if vmax <= vmin:
                vmax = vmin + 1.0
            ax.set_ylim(vmin, vmax * 1.05)
        ax.legend(fontsize="small")

    _plot_history(train_ax, train_loss_history, "Train Losses")
    _plot_history(eval_ax, eval_loss_history, "Eval Losses")

    fig.tight_layout()
    fig.savefig(_DEFAULT_LOSS_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def _ema(values: Sequence[float], alpha: float) -> list[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric_payload(metric_payload: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any] | None]:
    rho_values = metric_payload.get("rho")
    curves_raw = metric_payload.get("curves")
    baseline = metric_payload.get("baseline")

    if not isinstance(rho_values, list) or not rho_values:
        raise ValueError("Metric payload has invalid rho grid")
    if not isinstance(curves_raw, Mapping) or not curves_raw:
        raise ValueError("Metric payload has invalid curves mapping")

    x = np.asarray([float(v) for v in rho_values], dtype=float)
    curves = {str(label): np.asarray([float(v) for v in values], dtype=float) for label, values in curves_raw.items()}
    return x, curves, baseline if isinstance(baseline, Mapping) else None


def maybe_extract_metric_payload(
    metric_payload: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any] | None] | None:
    try:
        return extract_metric_payload(metric_payload)
    except ValueError:
        return None
