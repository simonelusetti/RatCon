import json
import logging
import os
import sys
import time
import math
import resource
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import torch
from dora import XP
from prettytable import PrettyTable
from scipy.stats import chi2_contingency
from tqdm import tqdm


if TYPE_CHECKING:
    from src.metrics import Counts


tqdm._instances.clear()


_CHI_SQUARE_PVALUE_THRESHOLD = 0.05
_CHI2_CRITICAL_DF1_P05 = 3.841458820694124


@dataclass(frozen=True)
class RunStartCapture:
    started_perf: float
    ru_start: resource.struct_rusage


def start_run_metrics_capture() -> RunStartCapture:
    return RunStartCapture(
        started_perf=time.perf_counter(),
        ru_start=resource.getrusage(resource.RUSAGE_SELF),
    )


def _rss_bytes(ru: resource.struct_rusage) -> int:
    # macOS returns bytes, Linux returns kilobytes
    return int(ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024)


def _build_run_statistics(start_capture: RunStartCapture) -> dict:
    ru_end = resource.getrusage(resource.RUSAGE_SELF)
    finished_perf = time.perf_counter()
    return {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": finished_perf - start_capture.started_perf,
        "resources": {
            "cpu_user_seconds": ru_end.ru_utime - start_capture.ru_start.ru_utime,
            "cpu_system_seconds": ru_end.ru_stime - start_capture.ru_start.ru_stime,
            "max_rss_bytes": _rss_bytes(ru_end),
        },
    }


def _write_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metrics_artifacts(
    cfg: dict,
    xp: XP,
    train_loss_history: Sequence[Mapping[str, float]],
    eval_loss_history: Sequence[Mapping[str, float]],
    start_capture: RunStartCapture,
    epochs_completed: int,
    epochs_target: int,
    training_completed: bool,
) -> None:
    run_statistics = _build_run_statistics(start_capture)
    final_losses = dict(eval_loss_history[-1]) if eval_loss_history else {}

    compact_metrics = {
        "run_statistics": run_statistics,
        "final_losses": final_losses,
        "training_progress": {
            "epochs_completed": int(epochs_completed),
            "epochs_target": int(epochs_target),
            "completed": bool(training_completed),
        },
    }

    training_cfg = cfg.get("train", {}) if isinstance(cfg, Mapping) else {}
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, Mapping) else {}
    runtime_data_cfg = runtime_cfg.get("data", {}) if isinstance(runtime_cfg, Mapping) else {}

    details_metrics = {
        "experiment": {
            "signature": xp.sig,
            "folder": str(xp.folder),
        },
        "training": {
            "epochs_target": int(training_cfg.get("epochs", 0) or 0),
            "no_train": bool(training_cfg.get("no_train", False)),
            "continue": bool(training_cfg.get("continue", False)),
            "epochs_completed": int(epochs_completed),
            "completed": bool(training_completed),
        },
        "runtime": {
            "device": str(runtime_cfg.get("device", "")),
            "bf16": bool(runtime_cfg.get("bf16", False)),
            "compile": bool(runtime_cfg.get("compile", False)),
            "batch_size": int(runtime_data_cfg.get("batch_size", 0) or 0),
            "num_workers": int(runtime_data_cfg.get("num_workers", 0) or 0),
        },
        "run_statistics": run_statistics,
    }

    metrics_file = str(cfg.get("metrics_file", "metrics.json"))
    metrics_details_file = str(cfg.get("metrics_details_file", "metrics_details.json"))
    _write_json(metrics_details_file, details_metrics)
    if training_completed:
        _write_json(metrics_file, compact_metrics)


def configure_runtime(runtime_cfg: dict) -> tuple[dict, bool]:
    changed_device = False

    if "threads" in runtime_cfg and runtime_cfg["threads"] is not None:
        torch.set_num_threads(int(runtime_cfg["threads"]))
    if "interop_threads" in runtime_cfg and runtime_cfg["interop_threads"] is not None:
        torch.set_num_interop_threads(int(runtime_cfg["interop_threads"]))
    if runtime_cfg.get("device") == "cuda" and not torch.cuda.is_available():
        changed_device = True

    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")
    runtime_cfg["device"] = device.type
    return runtime_cfg, changed_device


def to_device(device: torch.device, batch: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def should_disable_tqdm(short_log: bool = False, grid_mode: bool = False) -> bool:
    return short_log or grid_mode or bool(os.environ.get("DISABLE_TQDM")) or not sys.stderr.isatty()


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def get_logger(logfile: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    use_tqdm = not os.environ.get("DISABLE_TQDM")
    ch = TqdmLoggingHandler() if use_tqdm else logging.StreamHandler(sys.stderr)

    ch.setLevel(logging.INFO)
    ch_format = "%(asctime)s - %(levelname)s - %(message)s"
    ch.setFormatter(logging.Formatter(ch_format))

    fh = logging.FileHandler(Path(logfile))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def make_table(fields: Sequence[str], rows: Iterable[Sequence[object]]) -> PrettyTable:
    table = PrettyTable()
    table.field_names = fields
    for row in rows:
        table.add_row(row)
    return table


def dict_to_table(losses: Mapping[str, float]) -> PrettyTable:
    fields = losses.keys()
    rows = [[f"{losses.get(k, 0.0):.6f}" for k in fields]]
    return make_table(fields, rows)


def format_dict(d: Mapping[str, int | float | str], new_liners: set[str] | None = None) -> str:
    extra_newline_after = new_liners or set()
    lines: list[str] = []
    for k, v in d.items():
        if isinstance(v, (int, float)):
            v = f"{v:.5f}"
        lines.append(f"{k}: {v}")
        if k in extra_newline_after:
            lines.append("")
    return "\n".join(lines)


def _label_sort_key(label: Any) -> tuple[int, Any]:
    if isinstance(label, (int, np.integer)):
        return 0, int(label)
    if isinstance(label, float):
        return 1, float(label)
    if isinstance(label, str):
        try:
            return 0, int(label)
        except ValueError:
            try:
                return 1, float(label)
            except ValueError:
                return 2, label
    return 3, str(label)


# ---------------------------------------------------------------------
# Helpers: extracting raw counts from your `Counts`
# ---------------------------------------------------------------------
def _infer_non_entity_label(labels: Sequence[Any]) -> Any:
    for cand in ("O", "o", 0, "0", "NON_ENTITY", "NONE", "non_entity"):
        if cand in labels:
            return cand
    for lab in labels:
        if isinstance(lab, str) and lab.strip().upper() == "O":
            return lab
    raise ValueError(
        "Could not infer non-entity label. Please ensure the non-entity label is present "
        "in Counts.data keys (commonly 'O'), or update _infer_non_entity_label()."
    )


def _get_count(count_obj: Any, label: Any) -> int:
    v = count_obj.data[label]
    if isinstance(v, float):
        raise TypeError(f"Counts.data[{label!r}] is a float (rate); chi-square needs raw integer counts.")
    return int(v)


def selection_rate_matrix_to_table(
    counts_pred: Sequence[Any],
    counts_gold: Sequence[Any],
    selection_rates: Sequence[float],
) -> PrettyTable:
    labels = sorted(
        {label for counts in counts_gold for label in counts.data.keys()},
        key=_label_sort_key,
    )

    headers = ["label"] + [f"{rate:.3f}" for rate in selection_rates]
    rows: list[list[str]] = []

    for label in labels:
        row = [str(label)]
        for pred, gold in zip(counts_pred, counts_gold):
            total = _get_count(gold, label) if label in gold.data else 0
            kept = _get_count(pred, label) if label in pred.data else 0
            value = (kept / total) if total > 0 else 0.0
            row.append(f"{value:.2f}")
        rows.append(row)

    return make_table(headers, rows)


def _contingency_2x2_one_vs_rest(
    pred: Any,
    gold: Any,
    label: Any,
) -> np.ndarray:
    tp = _get_count(pred, label)
    tot_pos = _get_count(gold, label)

    tot_selected = sum(_get_count(pred, lab) for lab in pred.data.keys())
    tot_all = sum(_get_count(gold, lab) for lab in gold.data.keys())

    fp = tot_selected - tp
    tot_neg = tot_all - tot_pos

    fn = tot_pos - tp
    tn = tot_neg - fp

    if fn < 0 or tn < 0:
        raise ValueError(
            f"Invalid counts in one-vs-rest table for label={label!r}: "
            f"tp={tp}, tot_pos={tot_pos}, fp={fp}, tot_neg={tot_neg}"
        )

    return np.array([[tp, fn], [fp, tn]], dtype=np.int64)


def _contingency_2x2_vs_negative_label(
    pred: Any,
    gold: Any,
    positive_label: Any,
    negative_label: Any,
) -> np.ndarray:
    tp = _get_count(pred, positive_label)
    fn = _get_count(gold, positive_label) - tp
    fp = _get_count(pred, negative_label)
    tn = _get_count(gold, negative_label) - fp

    if fn < 0 or tn < 0:
        raise ValueError(
            "Invalid counts in pos-vs-negative table for "
            f"label={positive_label!r}, negative={negative_label!r}: "
            f"tp={tp}, fn={fn}, fp={fp}, tn={tn}"
        )

    return np.array([[tp, fn], [fp, tn]], dtype=np.int64)


def _chi_square_stats(table_2x2: np.ndarray) -> tuple[float, float, float]:
    """
    Safe chi-square computation.
    Returns (chi2, p_value, cramers_v).
    Degenerate tables return (0.0, 1.0, 0.0).
    """
    if np.any(table_2x2.sum(axis=0) == 0) or np.any(table_2x2.sum(axis=1) == 0):
        return 0.0, 1.0, 0.0

    try:
        chi2, p, _, _ = chi2_contingency(table_2x2, correction=False)
    except ValueError:
        return 0.0, 1.0, 0.0

    n = float(table_2x2.sum())
    v = float(np.sqrt(chi2 / n)) if n > 0 else 0.0
    return float(chi2), float(p), v


def build_chi_square_payload(
    counts_pred: Sequence[Any],
    counts_gold: Sequence[Any],
    selection_rates: Sequence[float],
) -> dict[str, Any]:
    """Build chi-square and Cramer's V data for each rho and label."""
    if not counts_pred or not counts_gold:
        return {
            "mode": "one_vs_rest",
            "labels": [],
            "rows": [],
        }

    labels = sorted(
        {label for counts in counts_gold for label in counts.data.keys()},
        key=_label_sort_key,
    )

    negative_label: Any | None = None
    try:
        negative_label = _infer_non_entity_label(labels)
    except ValueError:
        negative_label = None

    if negative_label is not None:
        effective_labels = [label for label in labels if label != negative_label]
        mode = "vs_negative_label"
    else:
        effective_labels = labels
        mode = "one_vs_rest"

    rows: list[dict[str, Any]] = []
    for rate, pred, gold in zip(selection_rates, counts_pred, counts_gold):
        label_rows: list[dict[str, Any]] = []
        for label in effective_labels:
            if negative_label is not None:
                table = _contingency_2x2_vs_negative_label(pred, gold, label, negative_label)
            else:
                table = _contingency_2x2_one_vs_rest(pred, gold, label)
            chi2, p_value, cramers_v = _chi_square_stats(table)
            label_rows.append(
                {
                    "label": str(label),
                    "contingency": [[int(x) for x in row] for row in table.tolist()],
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "cramers_v": float(cramers_v),
                }
            )

        rows.append(
            {
                "rho": float(rate),
                "labels": label_rows,
            }
        )

    return {
        "mode": mode,
        "negative_label": None if negative_label is None else str(negative_label),
        "labels": [str(label) for label in effective_labels],
        "rows": rows,
    }


# ---------------------------------------------------------------------
# Loss history persistence
# ---------------------------------------------------------------------
def save_combined_loss_history(
    train_history: Sequence[Mapping[str, float]],
    eval_history: Sequence[Mapping[str, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "train": [dict(item) for item in train_history],
        "eval": [dict(item) for item in eval_history],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)


def load_combined_loss_history(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict in {path}, found {type(payload).__name__}.")

    histories: dict[str, list[dict[str, float]]] = {}
    for key in ("train", "eval"):
        raw = payload.get(key, [])
        if not isinstance(raw, list):
            raise ValueError(f"Expected a list for '{key}' in {path}, found {type(raw).__name__}.")
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(f"Expected mapping entries in '{key}' in {path}, found {type(item).__name__}.")
        histories[key] = [{str(k): float(v) for k, v in item.items()} for item in raw]

    return histories["train"], histories["eval"]


# ---------------------------------------------------------------------
# Loss plots
# ---------------------------------------------------------------------
def save_train_eval_loss_plot(
    train_loss_history: Sequence[Mapping[str, float]],
    eval_loss_history: Sequence[Mapping[str, float]],
    out_path: str,
    ema_alpha: float = 0.2,
) -> None:
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
    fig.savefig(out_path, bbox_inches="tight")
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


def make_axes(n_items: int, ncols: int) -> tuple[Any, list[Any]]:
    nrows = max(1, math.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 4.2))
    axes_arr = np.asarray(axes).reshape(-1)
    return fig, list(axes_arr)


def style_group_axis(ax, label: str, n_runs: int) -> None:
    ax.set_title(f"{label}\nn={n_runs}", fontsize=8, loc="left", fontfamily="monospace")
    ax.grid(True, alpha=0.2)


def extract_spearman_curve(stsb: dict[str, Any], keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = None
    for key in keys:
        if key in stsb:
            values = stsb[key]
            break
    if values is None:
        raise ValueError(f"Missing one of {keys} in stsb.json")

    if isinstance(values, dict):
        items = sorted(((float(k), float(v)) for k, v in values.items()), key=lambda kv: kv[0])
        x = np.asarray([k for k, _ in items], dtype=float)
        y = np.asarray([v for _, v in items], dtype=float)
        return x, y

    if isinstance(values, list):
        rhos = stsb.get("rhos")
        if not isinstance(rhos, list) or len(rhos) != len(values):
            raise ValueError("Invalid list spearman format in stsb.json")
        x = np.asarray([float(v) for v in rhos], dtype=float)
        y = np.asarray([float(v) for v in values], dtype=float)
        return x, y

    raise ValueError(f"Unsupported format for {keys} in stsb.json")


def extract_chi_square_curves(chi_square: dict[str, Any], metric: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = chi_square.get("rows") if isinstance(chi_square, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Invalid chi_square.json format: expected non-empty rows list")

    rho_values: list[float] = []
    label_points: dict[str, list[float]] = {}
    for row in rows:
        rho = float(row.get("rho"))
        rho_values.append(rho)
        labels = row.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            label = str(item.get("label"))
            if metric == "chi_square":
                p_value = float(item.get("p_value", 1.0))
                p_value = max(p_value, 1e-300)
                value = -math.log10(p_value)
            else:
                value = float(item.get("cramers_v", 0.0))
            label_points.setdefault(label, []).append(value)

    x = np.asarray(rho_values, dtype=float)
    curves = {label: np.asarray(values, dtype=float) for label, values in label_points.items()}
    return x, curves


def extract_chi_square_baseline(chi_square: dict[str, Any], metric: str) -> float | None:
    if metric == "chi_square":
        return -math.log10(_CHI_SQUARE_PVALUE_THRESHOLD)

    rows = chi_square.get("rows") if isinstance(chi_square, dict) else None
    if not isinstance(rows, list) or not rows:
        return None

    for row in rows:
        labels = row.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            contingency = item.get("contingency")
            if not isinstance(contingency, list):
                continue
            table = np.asarray(contingency, dtype=float)
            total = float(table.sum())
            if total > 0.0:
                return float(np.sqrt(_CHI2_CRITICAL_DF1_P05 / total))

    return None


def extract_selection_rate_curves(selections: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = selections.get("selections_by_rho") if isinstance(selections, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Invalid selections.json format: expected non-empty selections_by_rho list")

    labels: set[str] = set()
    for row in rows:
        pred_counts = row.get("pred_counts", {})
        gold_counts = row.get("gold_counts", {})
        if isinstance(pred_counts, dict):
            labels.update(str(label) for label in pred_counts.keys())
        if isinstance(gold_counts, dict):
            labels.update(str(label) for label in gold_counts.keys())

    sorted_labels = sorted(labels)
    rho_values: list[float] = []
    label_points: dict[str, list[float]] = {label: [] for label in sorted_labels}

    for row in rows:
        rho_values.append(float(row.get("rho")))
        pred_counts_raw = row.get("pred_counts", {})
        gold_counts_raw = row.get("gold_counts", {})

        pred_counts = {str(k): float(v) for k, v in pred_counts_raw.items()} if isinstance(pred_counts_raw, dict) else {}
        gold_counts = {str(k): float(v) for k, v in gold_counts_raw.items()} if isinstance(gold_counts_raw, dict) else {}

        for label in sorted_labels:
            kept = pred_counts.get(label, 0.0)
            total = gold_counts.get(label, 0.0)
            value = (kept / total) if total > 0 else 0.0
            label_points[label].append(value)

    x = np.asarray(rho_values, dtype=float)
    curves = {label: np.asarray(values, dtype=float) for label, values in label_points.items()}
    return x, curves


def is_negative_label(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in {"o", "0", "false", "non_entity", "none", "negative", "neg"}


def filter_negative_label_curves(chi_square: dict[str, Any], curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    negative_label_raw = chi_square.get("negative_label") if isinstance(chi_square, dict) else None
    negative_label = str(negative_label_raw).strip() if negative_label_raw is not None else None

    filtered: dict[str, np.ndarray] = {}
    for label, curve in curves.items():
        if negative_label is not None and label.strip() == negative_label:
            continue
        if is_negative_label(label):
            continue
        filtered[label] = curve
    return filtered


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
            ax.set_title(f"{title}\nn={len(curves)}", fontsize=8, loc="left", fontfamily="monospace")
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
        _plot_loss_ax(train_ax, [r.train_history for r in group.runs], "train_loss", title=group.label)
        _plot_loss_ax(eval_ax, [r.eval_history for r in group.runs], "eval_loss", xlabel=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spearman_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        selector_curves: list[np.ndarray] = []
        random_curves: list[np.ndarray] = []
        x_ref: np.ndarray | None = None
        accepted_sigs: list[tuple[str, np.ndarray]] = []

        for run in group.runs:
            if run.stsb is None:
                print(f"Skipping STS-B for {run.sig}: missing {run.sig_dir / 'data' / 'stsb.json'}")
                continue
            x, y_selector = extract_spearman_curve(run.stsb, ("ours_by_rho", "ours"))
            _, y_random = extract_spearman_curve(run.stsb, ("random_by_rho", "random"))
            if x_ref is None:
                x_ref = x
                accepted_sigs.append((run.sig, x))
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                def _fmt_grid(arr: np.ndarray) -> str:
                    return f"[{arr[0]:.3g}..{arr[-1]:.3g}] n={len(arr)}"
                print(
                    f"Skipping {run.sig}: rho grid mismatch in group '{group.label}'\n"
                    f"  this run : {_fmt_grid(x)}\n"
                    f"  accepted :"
                )
                for sig, ax_arr in accepted_sigs:
                    print(f"    {sig}  {_fmt_grid(ax_arr)}")
                continue
            else:
                accepted_sigs.append((run.sig, x))
            selector_curves.append(y_selector)
            random_curves.append(y_random)

        if not selector_curves or x_ref is None:
            style_group_axis(ax, group.label, len(group.runs))
            ax.set_xlabel("selection rate (rho)")
            ax.set_ylabel("spearman")
            ax.text(0.5, 0.5, "no stsb data", transform=ax.transAxes, ha="center", va="center")
            continue

        selector_mean, selector_std = mean_std_curves([c.tolist() for c in selector_curves])
        random_mean, random_std = mean_std_curves([c.tolist() for c in random_curves])
        plot_with_band(ax, x_ref, selector_mean, selector_std, "selector mean+-std", linestyle="-")
        plot_with_band(ax, x_ref, random_mean, random_std, "random mean+-std", linestyle="--", alpha=0.14)
        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("selection rate (rho)")
        ax.set_ylabel("spearman")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_chi_square_overview(groups: Sequence[Any], out_path: Path, ncols: int, metric: str) -> None:
    ylabel = "-log10(p)" if metric == "chi_square" else "Cramer's V"
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}
        baselines: list[float] = []

        for run in group.runs:
            if run.chi_square is None:
                continue
            x, label_curves = extract_chi_square_curves(run.chi_square, metric=metric)
            label_curves = filter_negative_label_curves(run.chi_square, label_curves)
            baseline = extract_chi_square_baseline(run.chi_square, metric=metric)
            if baseline is not None:
                baselines.append(baseline)
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Chi-square rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("selection rate")
        ax.set_ylabel(ylabel)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no chi-square data", transform=ax.transAxes, ha="center", va="center")
            continue

        for label, curves in sorted(per_label_runs.items(), key=lambda kv: kv[0]):
            mean, std = mean_std_curves([c.tolist() for c in curves])
            plot_with_band(ax, x_ref, mean, std, f"{label} (n={len(curves)})")

        if baselines:
            baseline = float(np.mean(baselines))
            ax.axhline(baseline, linestyle="--", linewidth=1.5, color="0.35", label="p=0.05")

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selection_rates_overview(groups: Sequence[Any], out_path: Path, ncols: int) -> None:
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}

        for run in group.runs:
            if run.selections is None:
                continue
            x, label_curves = extract_selection_rate_curves(run.selections)
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Selection-rate rho grid mismatch inside group: {group.label}")

            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("effective selection rate (rho)")
        ax.set_ylabel("selection rate")
        ax.set_ylim(0.0, 1.05)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no selections data", transform=ax.transAxes, ha="center", va="center")
            continue

        ax.plot(x_ref, x_ref, linestyle="--", linewidth=1.5, color="0.35", label="baseline (y=x)")

        for label, curves in sorted(per_label_runs.items(), key=lambda kv: kv[0]):
            mean, std = mean_std_curves([c.tolist() for c in curves])
            plot_with_band(ax, x_ref, mean, std, f"{label} (n={len(curves)})")

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stsb_sweep(base: float, ours: Mapping[float, float], rand: Mapping[float, float], out_path: str = "spearman_vs_rho.png") -> None:
    rhos = list(ours.keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rhos, [base] * len(rhos), "--", label="Baseline")
    ax.plot(rhos, [ours[r] for r in rhos], "o-", label="Trained selector")
    ax.plot(rhos, [rand[r] for r in rhos], "x-", label="Random selector")
    ax.set_xlabel("Selection rate (ρ)")
    ax.set_ylabel("Spearman correlation (STS-B)")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
