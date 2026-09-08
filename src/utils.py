import json
import logging
import os
import re
import sys
import time
import resource
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from forge.core import ExperimentRun, ExperimentStore
from omegaconf import OmegaConf
from tqdm import tqdm


tqdm._instances.clear()


# Captured at import time, which forge guarantees happens before the run
# starts: commands.run() imports the entry-point module first and only then
# calls main() -> start_run(), and start_run() is what chdirs into the run
# directory. Anything addressed relative to where the user launched `forge`
# (the dataset cache, most importantly) has to be resolved against this
# rather than against the live cwd. Replaces dora.to_absolute_path, which
# got the same value from hydra's original-cwd bookkeeping.
LAUNCH_CWD = Path.cwd()


def to_absolute_path(path: str | Path) -> Path:
    """Resolve *path* against the directory `forge` was launched from."""
    path = Path(path)
    return path if path.is_absolute() else (LAUNCH_CWD / path).resolve()


def remap_checkpoint_state_dict(state_dict: dict, model_state_dict: dict) -> dict:
    """Reconcile a saved state_dict with the live model's key naming.

    Handles two sources of drift between the environment that trained a
    checkpoint and the one loading it: a torch.compile `_orig_mod.` prefix,
    and sentence-transformers renaming its wrapped HF module from `model` to
    `auto_model` across versions.
    """
    ckpt_compiled = any(k.startswith("_orig_mod.") for k in state_dict)
    model_compiled = any(k.startswith("_orig_mod.") for k in model_state_dict)
    if ckpt_compiled and not model_compiled:
        remapped = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    elif model_compiled and not ckpt_compiled:
        remapped = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    else:
        remapped = dict(state_dict)

    if any(".auto_model." in k for k in model_state_dict) and not any(".auto_model." in k for k in remapped):
        remapped = {k.replace(".model.0.model.", ".model.0.auto_model."): v for k, v in remapped.items()}
    return remapped


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


def write_metrics_details(
    cfg: dict,
    run: ExperimentRun,
    start_capture: RunStartCapture,
    epochs_completed: int,
    epochs_target: int,
    training_completed: bool,
) -> None:
    """Write the nested `metrics_details.json` side-artifact.

    forge owns `metrics.json` (written by run.finish() from the flat dict
    build_final_metrics returns) and `logs.jsonl` (run.push_log); this keeps
    the richer nested payload -- resource usage, the resolved runtime -- that
    doesn't fit a flat metrics table but is worth having per run.
    """
    training_cfg = cfg.get("train", {}) if isinstance(cfg, Mapping) else {}
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, Mapping) else {}
    runtime_data_cfg = runtime_cfg.get("data", {}) if isinstance(runtime_cfg, Mapping) else {}

    _write_json("metrics_details.json", {
        "experiment": {
            "signature": run.signature,
            "folder": str(run.path),
        },
        "training": {
            "epochs_target": int(epochs_target),
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
        "run_statistics": _build_run_statistics(start_capture),
    })


def build_final_metrics(
    eval_loss_history: Sequence[Mapping[str, float]],
    start_capture: RunStartCapture,
    epochs_completed: int,
) -> dict[str, float]:
    """Flat metrics dict for ExperimentRun.finish().

    Flat and scalar-valued on purpose: this is what `forge metrics` renders
    as table columns, so nested payloads go to metrics_details.json instead.
    """
    stats = _build_run_statistics(start_capture)
    return {
        **{str(k): float(v) for k, v in (eval_loss_history[-1] if eval_loss_history else {}).items()},
        "epochs_completed": int(epochs_completed),
        "duration_seconds": float(stats["duration_seconds"]),
        "max_rss_gb": stats["resources"]["max_rss_bytes"] / 1024 ** 3,
    }


def checkpoint_epoch(checkpoint_path: Path) -> int | None:
    match = re.fullmatch(r"model_(\d+)\.pth", checkpoint_path.name)
    return int(match.group(1)) if match else None


def find_resume_checkpoint(
    run: ExperimentRun, checkpoint_name: str | None = None,
) -> tuple[int, Path] | None:
    """Find a same-seed checkpoint using Forge's per-run runtime snapshots.

    Resume chooses the highest epoch; named evaluation requires that file.
    Ties use file modification time, as before. Failed runs remain eligible
    because their last saved checkpoint can be resumed.
    """
    seed = OmegaConf.select(run.config, "seed")
    store = ExperimentStore(root=run.experiment.path.parents[1])
    candidates = [
        (checkpoint_epoch(path) or 0, path)
        for source in store.list_runs(run.experiment.signature)
        if OmegaConf.select(source.config, "seed") == seed
        for path in (source.path / "state/models").glob(checkpoint_name or "model_*.pth")
        if path.is_file() and (checkpoint_name is not None or checkpoint_epoch(path) is not None)
    ]
    if not candidates and checkpoint_name is not None:
        raise FileNotFoundError(
            f"No {checkpoint_name} for seed {seed} in experiment {run.experiment.signature}."
        )
    return max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime)) if candidates else None


# torch.set_num_interop_threads() is a once-per-process call -- it raises if
# called again after any parallel work has started. `forge grid` executes
# every entry in a single process (dora used to fork a subprocess per run),
# so without this guard the second and later runs of a grid would die here.
_interop_threads_configured = False


def configure_runtime(runtime_cfg: dict) -> tuple[dict, bool]:
    global _interop_threads_configured
    changed_device = False

    # Safe to call repeatedly, unlike its interop counterpart below.
    if "threads" in runtime_cfg and runtime_cfg["threads"] is not None:
        torch.set_num_threads(int(runtime_cfg["threads"]))
    if (
        not _interop_threads_configured
        and runtime_cfg.get("interop_threads") is not None
    ):
        torch.set_num_interop_threads(int(runtime_cfg["interop_threads"]))
        _interop_threads_configured = True
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
