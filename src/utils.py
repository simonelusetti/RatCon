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
from typing import Any

import torch
from dora import XP
from prettytable import PrettyTable
from tqdm import tqdm


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
