import math
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np


_DEFAULT_LOSS_HISTORY_PATH = Path("data") / "loss_history.json"
_DEFAULT_LOSS_PLOT_PATH = Path("plots") / "loss.png"
_DEFAULT_SELECTION_RATE_CURVES_PATH = Path("data") / "selection_rate_curves.json"
_DEFAULT_CHI_SQUARE_CURVES_PATH = Path("data") / "chi_square_curves.json"
_DEFAULT_CRAMERS_V_CURVES_PATH = Path("data") / "cramers_v_curves.json"
_DEFAULT_SPEARMAN_CURVES_PATH = Path("data") / "spearman_curves.json"
_DEFAULT_SPEARMAN_PLOT_PATH = Path("plots") / "spearman_vs_rho.png"
_DEFAULT_SELECTION_RATE_PLOT_PATH = Path("plots") / "selection_rate_vs_rho.png"
_DEFAULT_CHI_SQUARE_PLOT_PATH = Path("plots") / "chi_square_vs_rho.png"
_DEFAULT_CRAMERS_V_PLOT_PATH = Path("plots") / "cramers_v_vs_rho.png"

_METRIC_TO_FILENAME = {
    "selection_rate": "selection_rate_curves.json",
    "chi_square": "chi_square_curves.json",
    "cramers_v": "cramers_v_curves.json",
    "spearman": "spearman_curves.json",
}

_METRIC_TO_DATA_PATH = {
    "selection_rate": _DEFAULT_SELECTION_RATE_CURVES_PATH,
    "chi_square": _DEFAULT_CHI_SQUARE_CURVES_PATH,
    "cramers_v": _DEFAULT_CRAMERS_V_CURVES_PATH,
    "spearman": _DEFAULT_SPEARMAN_CURVES_PATH,
}

_METRIC_TO_PLOT_PATH = {
    "selection_rate": _DEFAULT_SELECTION_RATE_PLOT_PATH,
    "chi_square": _DEFAULT_CHI_SQUARE_PLOT_PATH,
    "cramers_v": _DEFAULT_CRAMERS_V_PLOT_PATH,
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
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_selection_rate_curves() -> Path:
    return _plot_metric_from_artifact("selection_rate", "Selection rate")


def plot_chi_square_curves() -> Path:
    return _plot_metric_from_artifact("chi_square", "-log10(p-value)")


def plot_cramers_v_curves() -> Path:
    return _plot_metric_from_artifact("cramers_v", "Cramer's V")


def plot_spearman_curves() -> Path:
    return _plot_metric_from_artifact("spearman", "Spearman correlation (STS-B)")


def save_eval_plots(metric_names: Sequence[str]) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}
    for metric_name in metric_names:
        if metric_name == "selection_rate":
            plot_paths[metric_name] = plot_selection_rate_curves()
        elif metric_name == "chi_square":
            plot_paths[metric_name] = plot_chi_square_curves()
        elif metric_name == "cramers_v":
            plot_paths[metric_name] = plot_cramers_v_curves()
        elif metric_name == "spearman":
            plot_paths[metric_name] = plot_spearman_curves()
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
) -> None:
    valid = np.isfinite(mean)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = mean[valid]
    sv = std[valid]
    line, = ax.plot(xv, yv, marker="o", linewidth=2.0, linestyle=linestyle, label=label)
    band_valid = np.isfinite(sv)
    if np.any(band_valid):
        xb = xv[band_valid]
        yb = yv[band_valid]
        sb = sv[band_valid]
        ax.fill_between(xb, yb - sb, yb + sb, alpha=alpha, color=line.get_color())


def _build_overview_figure(n_groups: int, ncols: int, width: float = 5.2, height: float = 4.2):
    nrows = max(1, math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))
    return fig, np.asarray(axes).reshape(-1)


def _setup_overview_axis(ax, label: str, n_runs: int, xlabel: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_title(f"{label}\\nn={n_runs}", fontsize=8, loc="left", fontfamily="monospace")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _finalize_overview_figure(fig, axes: np.ndarray, n_groups: int, out_path: Path, dpi: int = 180) -> None:
    for ax in axes[n_groups:]:
        ax.set_visible(False)
    fig.tight_layout()
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


def plot_loss_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    def _plot_loss_ax(
        ax,
        histories: Sequence[Sequence[Mapping[str, float]]],
        metric_key: str,
        title: str | None = None,
        xlabel: bool = False,
    ) -> None:
        curves = [[entry[metric_key] for entry in h if metric_key in entry] for h in histories]
        curves = [c for c in curves if c]

        ax.grid(True, alpha=0.2)
        ax.set_ylabel(metric_key.replace("_", " "), fontsize=7)
        if title:
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
    fig = plt.figure(figsize=(ncols * 5.2, nrows * 8.0))
    outer_gs = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.35)

    for i, group in enumerate(groups):
        row, col = divmod(i, ncols)
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], hspace=0.35)
        train_ax = fig.add_subplot(inner_gs[0])
        eval_ax = fig.add_subplot(inner_gs[1])
        loaded_histories = [_load_loss_histories_for_run(run.sig_dir) for run in group.runs]
        _plot_loss_ax(train_ax, [train_h for train_h, _ in loaded_histories], "train_loss", title=group.label)
        _plot_loss_ax(eval_ax, [eval_h for _, eval_h in loaded_histories], "eval_loss", xlabel=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spearman_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = _build_overview_figure(len(groups), ncols)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "selection rate (rho)", "spearman")

        selector_curves: list[np.ndarray] = []
        random_curves: list[np.ndarray] = []
        x_ref: np.ndarray | None = None
        skipped_mismatch = 0

        for run in group.runs:
            metric_payload = _load_metric_payload_for_run(run.sig_dir, metric="spearman")
            if metric_payload is None:
                continue
            parsed = maybe_extract_metric_payload(metric_payload)
            if parsed is None:
                continue
            x, curves, _ = parsed
            y_selector = curves.get("selector")
            y_random = curves.get("random")
            if y_selector is None or y_random is None:
                continue
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                skipped_mismatch += 1
                continue
            selector_curves.append(y_selector)
            random_curves.append(y_random)

        if skipped_mismatch:
            print(f"Skipped {skipped_mismatch} spearman runs in group '{group.label}' due to rho-grid mismatch")

        if not selector_curves or x_ref is None:
            ax.text(0.5, 0.5, "no spearman data", transform=ax.transAxes, ha="center", va="center")
            continue

        selector_mean, selector_std = mean_std_curves([c.tolist() for c in selector_curves])
        random_mean, random_std = mean_std_curves([c.tolist() for c in random_curves])
        plot_with_band(ax, x_ref, selector_mean, selector_std, "selector mean+-std")
        plot_with_band(ax, x_ref, random_mean, random_std, "random mean+-std", linestyle="--", alpha=0.14)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


def plot_chi_square_overview(groups: Sequence[Any], out_path: Path, ncols: int, metric: str) -> None:
    ylabel = "-log10(p)" if metric == "chi_square" else "Cramer's V"
    fig, axes = _build_overview_figure(len(groups), ncols)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "selection rate", ylabel)
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}
        baselines: list[float] = []

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
                except (TypeError, ValueError):
                    pass
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Chi-square rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no chi-square data", transform=ax.transAxes, ha="center", va="center")
            continue

        _plot_group_label_curves(ax, x_ref, per_label_runs)

        if baselines:
            ax.axhline(float(np.mean(baselines)), linestyle="--", linewidth=1.5, color="0.35", label="p=0.05")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


def plot_selection_rates_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = _build_overview_figure(len(groups), ncols)

    for ax, group in zip(axes, groups):
        _setup_overview_axis(ax, group.label, len(group.runs), "effective selection rate (rho)", "selection rate", ylim=(0.0, 1.05))
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
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    _finalize_overview_figure(fig, axes, len(groups), out_path)


