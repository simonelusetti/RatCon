import math
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, FuncNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np

from .data import LABEL_DISPLAY_NAMES


_DEFAULT_LOSS_HISTORY_PATH = Path("data") / "loss_history.json"
_DEFAULT_LOSS_PLOT_PATH = Path("plots") / "loss.png"
_DEFAULT_SELECTION_RATE_CURVES_PATH = Path("data") / "selection_rate_curves.json"
_DEFAULT_PVALUE_CURVES_PATH = Path("data") / "pvalue_curves.json"
_DEFAULT_EFFECT_SIZE_CURVES_PATH = Path("data") / "effect_size_curves.json"
_DEFAULT_SELECTION_RATE_PLOT_PATH = Path("plots") / "selection_rate_vs_rho.png"
_DEFAULT_PVALUE_PLOT_PATH = Path("plots") / "pvalue_vs_rho.png"
_DEFAULT_EFFECT_SIZE_PLOT_PATH = Path("plots") / "effect_size_vs_rho.png"
_DEFAULT_SIGNED_EFFECT_HEATMAP_PLOT_PATH = Path("plots") / "signed_effect_heatmap.png"
_DEFAULT_SPEARMAN_CURVES_PATH = Path("data") / "spearman_curves.json"
_DEFAULT_SPEARMAN_PLOT_PATH = Path("plots") / "spearman_vs_rho.png"

_METRIC_TO_FILENAME = {
    "selection_rate": "selection_rate_curves.json",
    "pvalue": "pvalue_curves.json",
    "effect_size": "effect_size_curves.json",
    # z is already signed, so the heatmap reads the same artifact as the line plot.
    "signed_effect": "effect_size_curves.json",
    "spearman": "spearman_curves.json",
}

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


_DEFAULT_CONF_PATH = Path(__file__).resolve().parent / "conf" / "default.yaml"
_DEFAULT_DATASET_NAME: str | None = None


def _get_default_dataset_name() -> str | None:
    global _DEFAULT_DATASET_NAME
    if _DEFAULT_DATASET_NAME is not None:
        return _DEFAULT_DATASET_NAME
    try:
        import yaml
        with _DEFAULT_CONF_PATH.open() as f:
            cfg = yaml.safe_load(f)
        _DEFAULT_DATASET_NAME = str(cfg["data"]["dataset"])
    except Exception:
        pass
    return _DEFAULT_DATASET_NAME


def _dataset_name_from_overrides(overrides: list[str]) -> str | None:
    """Return the canonical dataset name from a list of Hydra override strings."""
    for o in overrides:
        if o.startswith("data.dataset="):
            return o.split("=", 1)[1]
    return None


def _legend_in_right_panel(
    ax,
    fontsize: float = 7,
    ncol: int = 1,
    width_ratio: float = 0.72,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * width_ratio, pos.height])
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=fontsize,
        ncol=ncol,
    )

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


def mean_std_curves(curves: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, : len(curve)] = np.asarray(curve, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def _ema(values: Sequence[float], alpha: float) -> list[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def plot_with_band(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.18,
    color: str | None = None,
) -> str:
    """Plot a mean curve with a ±std band. Returns the color used."""
    valid = np.isfinite(mean)
    if not np.any(valid):
        return color or "C0"
    xv = x[valid]
    yv = mean[valid]
    sv = std[valid]
    kwargs = {"marker": "o", "linewidth": 2.0, "linestyle": linestyle, "label": label}
    if color is not None:
        kwargs["color"] = color
    line, = ax.plot(xv, yv, **kwargs)
    used_color = line.get_color()
    band_valid = np.isfinite(sv)
    if np.any(band_valid):
        xb = xv[band_valid]
        yb = yv[band_valid]
        sb = sv[band_valid]
        ax.fill_between(xb, yb - sb, yb + sb, alpha=alpha, color=used_color)
    return used_color


def _build_overview_figure(n_groups: int, ncols: int, width: float = 5.8, height: float = 4.6):
    nrows = max(1, math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))
    fig.subplots_adjust(wspace=35, hspace=5)
    return fig, np.asarray(axes).reshape(-1)


def _setup_overview_axis(ax, label: str, n_runs: int, xlabel: str, ylabel: str, ylim: tuple[float, float] | None = None, custom_title: str | None = None) -> None:
    if custom_title is not None:
        ax.set_title(custom_title, fontsize=11, loc="center")
    else:
        ax.set_title(f"{label}\\nn={n_runs}", fontsize=8, loc="left", fontfamily="monospace")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _finalize_overview_figure(fig, axes: np.ndarray, n_groups: int, out_path: Path, dpi: int = 180) -> None:
    for ax in axes[n_groups:]:
        ax.set_visible(False)
    fig.tight_layout(pad=1.1, w_pad=2.2, h_pad=2.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_group_label_curves(ax, x_ref: np.ndarray, per_label_runs: dict[str, list[np.ndarray]]) -> None:
    for label, curves in sorted(per_label_runs.items(), key=lambda kv: kv[0]):
        mean, std = mean_std_curves([c.tolist() for c in curves])
        plot_with_band(ax, x_ref, mean, std, f"{label} (n={len(curves)})")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_loss_histories_for_run(run_dir: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    loss_path = run_dir / "data" / "loss_history.json"
    if not loss_path.exists():
        return [], []
    payload = _load_json(loss_path)
    train_history = payload.get("train", []) if isinstance(payload, Mapping) else []
    eval_history = payload.get("eval", []) if isinstance(payload, Mapping) else []
    if not isinstance(train_history, list) or not isinstance(eval_history, list):
        return [], []
    return train_history, eval_history


def _load_metric_payload_for_run(run_dir: Path, metric: str) -> Mapping[str, Any] | None:
    filename = _METRIC_TO_FILENAME.get(metric)
    if filename is None:
        return None
    payload_path = run_dir / "data" / filename
    if not payload_path.exists():
        return None
    payload = _load_json(payload_path)
    return payload if isinstance(payload, Mapping) else None


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


def plot_loss_overview(groups: Sequence[Any], out_path: Path, ncols: int, titles: list[str] | None = None) -> None:
    def _plot_loss_ax(
        ax,
        histories: Sequence[Sequence[Mapping[str, float]]],
        metric_key: str,
        title: str | None = None,
        xlabel: bool = False,
        custom_title: str | None = None,
    ) -> None:
        curves = [[entry[metric_key] for entry in h if metric_key in entry] for h in histories]
        curves = [c for c in curves if c]

        ax.grid(True, alpha=0.2)
        ax.set_ylabel(metric_key.replace("_", " "), fontsize=7)
        if custom_title is not None:
            ax.set_title(custom_title, fontsize=11, loc="center")
        elif title:
            ax.set_title(f"{title}\\nn={len(curves)}", fontsize=8, loc="left", fontfamily="monospace")
        if xlabel:
            ax.set_xlabel("epoch", fontsize=7)

        if not curves:
            ax.text(0.5, 0.5, f"no {metric_key}", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            return

        mean, std = mean_std_curves(curves)
        x = np.arange(1, len(mean) + 1, dtype=float)
        plot_with_band(ax, x, mean, std, "mean±std")
        ema_mean = _ema(mean.tolist(), 0.2)
        ax.plot(x[: len(ema_mean)], ema_mean, linewidth=2.2, label="EMA 0.20")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6)

    n = len(groups)
    nrows = max(1, math.ceil(n / ncols))
    fig = plt.figure(figsize=(ncols * 6.0, nrows * 8.8))
    outer_gs = fig.add_gridspec(nrows, ncols, hspace=0.7, wspace=0.5)

    for i, group in enumerate(groups):
        row, col = divmod(i, ncols)
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], hspace=0.45)
        train_ax = fig.add_subplot(inner_gs[0])
        eval_ax = fig.add_subplot(inner_gs[1])
        loaded_histories = [_load_loss_histories_for_run(run.sig_dir) for run in group.runs]
        custom = titles[i] if titles and i < len(titles) else None
        _plot_loss_ax(train_ax, [train_h for train_h, _ in loaded_histories], "train_loss", title=group.label, custom_title=custom)
        _plot_loss_ax(eval_ax, [eval_h for _, eval_h in loaded_histories], "eval_loss", xlabel=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_signed_effect_heatmap_overview(groups: Sequence[Any], out_path: Path, ncols: int, titles: list[str] | None = None) -> None:
    all_abs_vals: list[float] = []
    group_matrices: list[tuple[Any, list[float], dict[str, list[float]]]] = []
    group_thresh: float | None = None

    for group in groups:
        label_accum: dict[str, list[list[float]]] = {}
        rho_ref: list[float] | None = None

        for run in group.runs:
            payload = _load_metric_payload_for_run(run.sig_dir, metric="effect_size")
            if payload is None:
                continue
            parsed = maybe_extract_metric_payload(payload)
            if parsed is None:
                continue
            x, curves, baseline = parsed
            if group_thresh is None and isinstance(baseline, Mapping):
                value = baseline.get("value")
                if isinstance(value, (int, float)) and value > 0:
                    group_thresh = float(value)
            if rho_ref is None:
                rho_ref = x.tolist()
            for label, curve in curves.items():
                label_accum.setdefault(label, []).append(curve.tolist())
                all_abs_vals.extend(float(abs(v)) for v in curve.tolist() if np.isfinite(v))

        if rho_ref is None:
            group_matrices.append((group, [], {}))
            continue

        keep_special = any(
            "model.keep_special=true" in " ".join(run.overrides)
            for run in group.runs
        )
        mean_curves: dict[str, list[float]] = {
            lbl: list(np.mean(np.array(runs_data), axis=0))
            for lbl, runs_data in label_accum.items()
            if keep_special or lbl != "special"
        }
        group_matrices.append((group, rho_ref, mean_curves))

    global_vmax = float(np.percentile(all_abs_vals, 95)) if all_abs_vals else 1.0
    group_thresh = group_thresh if group_thresh is not None else _Z_THRESH
    shared_norm = _make_signed_effect_norm(global_vmax, thresh=group_thresh)

    # Cap columns to the actual number of groups to avoid empty gridspec columns
    # that push the colorbar far off to the right.
    effective_ncols = min(ncols, len(group_matrices))
    fig, axes = _build_overview_figure(len(groups), effective_ncols, width=5.5, height=4.2)

    im_ref = None
    for i, (ax, (group, rho_ref, mean_curves)) in enumerate(zip(axes, group_matrices)):
        custom = titles[i] if titles and i < len(titles) else None
        if custom is not None:
            ax.set_title(custom, fontsize=11, loc="center")
        else:
            ax.set_title(
                f"{group.label}\\nn={len(group.runs)}",
                fontsize=8, loc="left", fontfamily="monospace",
            )
        if not rho_ref or not mean_curves:
            ax.text(0.5, 0.5, "no signed_effect data", ha="center", va="center", transform=ax.transAxes)
            continue

        pseudo_payload = {"rho": rho_ref, "curves": mean_curves}
        ds_name = next(
            (_dataset_name_from_overrides(run.overrides) for run in group.runs
             if _dataset_name_from_overrides(run.overrides) is not None),
            _get_default_dataset_name(),
        )
        im = _signed_effect_heatmap_from_payload(
            pseudo_payload, ax, vmax=global_vmax, norm=shared_norm, dataset_name=ds_name
        )
        ax.set_xlabel("ρ", fontsize=7)
        if im_ref is None:
            im_ref = im

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    fig.tight_layout(pad=1.1, w_pad=2.2, h_pad=2.2)

    # Colorbar anchored to the used axes — matplotlib auto-sizes and positions it.
    if im_ref is None:
        sm = plt.cm.ScalarMappable(norm=shared_norm, cmap="RdBu")
        sm.set_array([])
        im_ref = sm

    cbar = fig.colorbar(
        im_ref,
        ax=list(axes[:len(groups)]),
        shrink=0.8,
        pad=0.04,
    )
    cbar.set_label("z-score (signed)", fontsize=8)
    # Ticks at the four meaningful breakpoints: extremes + both threshold boundaries
    ticks = [-global_vmax, -group_thresh, 0.0, group_thresh, global_vmax]
    labels = [
        f"−{global_vmax:.0f}",
        f"−{group_thresh:.1f}",
        "0",
        f"+{group_thresh:.1f}",
        f"+{global_vmax:.0f}",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels, fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_overview(groups: Sequence[Any], out_path: Path, ncols: int, metric: str, titles: list[str] | None = None) -> None:
    ylabel = "-log10(p)" if metric == "pvalue" else "Effect size (z-score)"
    fig, axes = _build_overview_figure(len(groups), ncols, width=7.2)

    for i, (ax, group) in enumerate(zip(axes, groups)):
        custom = titles[i] if titles and i < len(titles) else None
        _setup_overview_axis(ax, group.label, len(group.runs), "selection rate", ylabel, custom_title=custom)
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}
        baselines: list[float] = []
        baseline_label = "p=0.05"

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric=metric)
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, label_curves, baseline = parsed
            if isinstance(baseline, Mapping) and baseline.get("kind") == "constant":
                try:
                    baselines.append(float(baseline.get("value")))
                    baseline_label = str(baseline.get("label", baseline_label))
                except (TypeError, ValueError):
                    pass
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"{metric} rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, f"no {metric} data", transform=ax.transAxes, ha="center", va="center")
            continue

        _plot_group_label_curves(ax, x_ref, per_label_runs)

        if baselines:
            ax.axhline(float(np.mean(baselines)), linestyle="--", linewidth=1.5, color="0.35", label=baseline_label)
        _legend_in_right_panel(ax, fontsize=6)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


def plot_selection_rates_overview(groups: Sequence[Any], out_path: Path, ncols: int, titles: list[str] | None = None) -> None:
    fig, axes = _build_overview_figure(len(groups), ncols, width=7.2)

    for i, (ax, group) in enumerate(zip(axes, groups)):
        custom = titles[i] if titles and i < len(titles) else None
        _setup_overview_axis(ax, group.label, len(group.runs), "effective selection rate (rho)", "selection rate", ylim=(0.0, 1.05), custom_title=custom)
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}
        show_identity_baseline = False

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric="selection_rate")
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, label_curves, baseline = parsed
            if isinstance(baseline, Mapping) and baseline.get("kind") == "identity":
                show_identity_baseline = True
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Selection-rate rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no selections data", transform=ax.transAxes, ha="center", va="center")
            continue

        if show_identity_baseline:
            ax.plot(x_ref, x_ref, linestyle="--", linewidth=1.5, color="0.35", label="baseline (y=x)")
        _plot_group_label_curves(ax, x_ref, per_label_runs)
        _legend_in_right_panel(ax, fontsize=6)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


