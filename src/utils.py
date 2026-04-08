import json
import logging
import os
import sys
import time
import resource
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from dora import XP
from prettytable import PrettyTable
from scipy.stats import chi2_contingency
from tqdm import tqdm


if TYPE_CHECKING:
    from src.metrics import Counts


tqdm._instances.clear()


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
    if short_log:
        return True
    if grid_mode:
        return True
    if os.environ.get("DISABLE_TQDM"):
        return True
    if not sys.stderr.isatty():
        return True
    return False


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
    """
    Extract a *raw integer count* for a given label from a Counts-like object.

    Expected usage in this file:
      - counts_gold: totals per label (gold support)
      - counts_pred: selected-per-label (predicted kept)

    Supported conventions:
      1) count_obj.counts[label]
      2) count_obj.data[label] is an int
      3) count_obj.data[label] is dict with {"count"} / {"n"} / {"total"} / {"selected"}
      4) count_obj.data[label] is tuple/list -> first element is count
    """
    if hasattr(count_obj, "counts"):
        return int(count_obj.counts[label])

    if hasattr(count_obj, "data"):
        v = count_obj.data[label]
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float):
            raise TypeError(
                f"Counts.data[{label!r}] is a float (rate). Chi-square needs raw counts. "
                "Make counts_pred/counts_gold store integer counts per label."
            )
        if isinstance(v, dict):
            for k in ("count", "n", "total", "selected", "keep"):
                if k in v:
                    return int(v[k])
        if isinstance(v, (tuple, list)) and len(v) >= 1:
            return int(v[0])

    raise TypeError(
        f"Cannot extract integer count for label={label!r} from Counts. "
        "Please store integer counts per label, e.g. Counts.data[label] = int."
    )


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

    rows: list[dict[str, Any]] = []
    for rate, pred, gold in zip(selection_rates, counts_pred, counts_gold):
        label_rows: list[dict[str, Any]] = []
        for label in labels:
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
        "mode": "one_vs_rest",
        "labels": [str(label) for label in labels],
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

    train_history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []

    for key in ["train", "eval"]:
        history_list = payload.get(key, [])
        if not isinstance(history_list, list):
            raise ValueError(f"Expected a list for '{key}' in {path}, found {type(history_list).__name__}.")

        for item in history_list:
            if not isinstance(item, Mapping):
                raise ValueError(f"Expected mapping entries in '{key}' in {path}, found {type(item).__name__}.")

        if key == "train":
            train_history = [{str(k): float(v) for k, v in item.items()} for item in history_list]
        else:
            eval_history = [{str(k): float(v) for k, v in item.items()} for item in history_list]

    return train_history, eval_history


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

    def _ema(values: Sequence[float], alpha_value: float) -> list[float]:
        if not values:
            return []
        smoothed = [float(values[0])]
        for value in values[1:]:
            smoothed.append(alpha_value * float(value) + (1.0 - alpha_value) * smoothed[-1])
        return smoothed

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
