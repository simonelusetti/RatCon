import json
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TextIO,
    TYPE_CHECKING,
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dora import XP
from prettytable import PrettyTable
from scipy.stats import chi2_contingency
from tqdm import tqdm


if TYPE_CHECKING:
    from src.metrics import Counts


tqdm._instances.clear()


def configure_runtime(runtime_cfg: Dict) -> Tuple[Dict, bool]:
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


def to_device(device: torch.device, batch: Dict) -> Dict:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def open_selection_writer(xp: XP, epoch: int) -> TextIO:
    out_dir = xp.folder / "selections"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"eval_epoch_{epoch:03d}.json"
    return open(path, "w")


def full_conf_matrix(
    logger: logging.Logger,
    counts_gold_history: Sequence["Counts"],
    counts_pred_history: Sequence["Counts"],
    labels_present: bool,
) -> None:
    if not labels_present:
        return

    txt = ""
    new_txt = ""

    for epoch, (gold, pred) in enumerate(zip(counts_gold_history, counts_pred_history)):
        new_txt = gold.conf_matrix(pred, epoch)
        if new_txt is None:
            return
        txt += new_txt

    out = "confusion_matrix.txt"
    with open(out, "w") as f:
        f.write(txt)

    logger.info("Last epoch confusion matrix:\n%s", new_txt)
    logger.info("Saved confusion matrix to %s", out)


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
    lines: List[str] = []
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


def tkns_to_words(
    gates: torch.Tensor,
    attn_mask: torch.Tensor,
    word_ids: torch.Tensor,
    labels: list[list[str]],
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    device = gates.device
    B, T = gates.shape
    sel = (gates > threshold) & attn_mask.bool()
    W = int(word_ids.max().item()) + 1
    word_onehot = torch.zeros(B, T, W, device=device, dtype=torch.bool)
    valid = word_ids >= 0
    word_onehot[valid] = F.one_hot(word_ids[valid], num_classes=W).bool()
    word_pred = (sel.unsqueeze(-1) & word_onehot).any(dim=1)
    word_attn = (attn_mask.bool().unsqueeze(-1) & word_onehot).any(dim=1)
    word_labels: List[List[str]] = []

    for b in range(B):
        label_by_wid: Dict[int, str] = {}
        for t, wid in enumerate(word_ids[b].tolist()):
            if wid == -100:
                continue
            if wid not in label_by_wid:
                label_by_wid[wid] = labels[b][t]

        wids_sorted = sorted(label_by_wid.keys())
        word_labels.append([label_by_wid[wid] for wid in wids_sorted])

    return word_pred, word_attn, word_labels


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


def _contingency_2x2_from_pred_gold(
    pred: Any,
    gold: Any,
    label: Any,
    non_entity_label: Any,
) -> np.ndarray:
    tp = _get_count(pred, label)
    tot_pos = _get_count(gold, label)

    fp = _get_count(pred, non_entity_label)
    tot_neg = _get_count(gold, non_entity_label)

    fn = tot_pos - tp
    tn = tot_neg - fp

    if fn < 0 or tn < 0:
        raise ValueError(
            f"Invalid counts: selected cannot exceed total. "
            f"label={label!r}: tp={tp}, tot_pos={tot_pos}; "
            f"non_entity={non_entity_label!r}: fp={fp}, tot_neg={tot_neg}"
        )

    return np.array([[tp, fn], [fp, tn]], dtype=np.int64)


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


def _chi_square_stats(table_2x2: np.ndarray) -> Tuple[float, float, float]:
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


# ---------------------------------------------------------------------
# Individual Plotters (square figures, saved individually)
# ---------------------------------------------------------------------
def save_loss_plot(loss_history: Sequence[Mapping[str, float]], out_path: str) -> None:
    if not loss_history:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Losses Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    epochs = range(1, len(loss_history) + 1)
    loss_keys = list(loss_history[0].keys())

    for key in loss_keys:
        ys = [loss[key] for loss in loss_history]
        ax.plot(epochs, ys, label=key)

    ax.legend(fontsize="small")

    all_values = [v for loss in loss_history for v in loss.values()]
    if all_values:
        ax.set_ylim(min(all_values), max(all_values) * 1.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_selection_rates_plot_from_pred_gold(
    counts_pred: Sequence[Any],
    counts_gold: Sequence[Any],
    selection_rates: Sequence[float],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title("Selection Rates vs Mean Effective Rate")
    ax.set_xlabel("Mean effective selection rate")
    ax.set_ylabel("Selection rate (kept / total)")

    labels = sorted(counts_gold[0].data.keys(), key=_label_sort_key)
    rates_arr = np.array(selection_rates, dtype=float)

    # Random baseline: under uniform selection, label-wise keep rate matches overall keep rate.
    ax.plot(rates_arr, rates_arr, linestyle="--", linewidth=2, label="Random baseline")

    for label in labels:
        ys: List[float] = []
        for pred, gold in zip(counts_pred, counts_gold):
            kept = _get_count(pred, label)
            tot = _get_count(gold, label)
            ys.append((kept / tot) if tot > 0 else 0.0)

        ax.plot(rates_arr, ys, marker="o", label=str(label))

    ax.legend(fontsize="small")
    if len(rates_arr):
        ax.set_xlim(float(rates_arr.min()), float(rates_arr.max()))
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_chi_square_plot_from_pred_gold(
    counts_pred: Sequence[Any],
    counts_gold: Sequence[Any],
    selection_rates: Sequence[float],
    out_path: str,
    non_entity_label: Optional[Any] = None,
    alpha: float = 0.05,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = sorted(counts_gold[0].data.keys(), key=_label_sort_key)
    use_one_vs_rest = False
    if non_entity_label is None:
        try:
            non_entity_label = _infer_non_entity_label(labels)
        except ValueError:
            use_one_vs_rest = True

    test_labels = labels if use_one_vs_rest else [lab for lab in labels if lab != non_entity_label]
    rates_arr = np.array(selection_rates, dtype=float)

    ax.set_title("Chi-square Heatmap")
    ax.set_xlabel("Mean effective selection rate")
    ax.set_ylabel("Label")

    # If a label is never selected at any sweep point, treat it as deterministically
    # suppressed and render it as non-significant in this p-value heatmap.
    always_zero_selected: set[Any] = set()
    for lab in test_labels:
        if all(_get_count(pred, lab) == 0 for pred in counts_pred):
            always_zero_selected.add(lab)

    values = np.zeros((len(test_labels), len(rates_arr)), dtype=float)
    for row, lab in enumerate(test_labels):
        if lab in always_zero_selected:
            values[row, :] = 0.0
            continue

        for col, (pred, gold) in enumerate(zip(counts_pred, counts_gold)):
            if use_one_vs_rest:
                table = _contingency_2x2_one_vs_rest(pred, gold, lab)
            else:
                table = _contingency_2x2_from_pred_gold(pred, gold, lab, non_entity_label)
            _, p, _ = _chi_square_stats(table)
            values[row, col] = -np.log10(np.clip(float(p), 1e-300, 1.0))

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if vmax <= vmin:
            vmax = vmin + 1.0
        else:
            lo = float(np.percentile(finite_values, 5))
            hi = float(np.percentile(finite_values, 95))
            if hi > lo:
                vmin, vmax = lo, hi

    image = ax.imshow(
        values,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(np.arange(len(test_labels)))
    ax.set_yticklabels([str(lab) for lab in test_labels])
    ax.set_xticks(np.arange(len(rates_arr)))
    ax.set_xticklabels([f"{rate:.2f}" for rate in rates_arr], rotation=45, ha="right")

    thresh = -np.log10(alpha)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$-\log_{10}(p)$")
    cbar.ax.axhline(thresh, color="white", linestyle="--", linewidth=1.5)
    cbar.ax.text(
        0.5,
        thresh,
        f" alpha={alpha}",
        color="white",
        ha="left",
        va="bottom",
        fontsize="small",
        transform=cbar.ax.get_yaxis_transform(),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_cramers_v_plot_from_pred_gold(
    counts_pred: Sequence[Any],
    counts_gold: Sequence[Any],
    selection_rates: Sequence[float],
    out_path: str,
    non_entity_label: Optional[Any] = None,
) -> None:
    """
    Effect size plot: Cramér's V vs rho for one-vs-non-entity 2x2 tables.

    For 2x2 tables:
      V = sqrt(chi2 / N)

    This removes the sample-size scaling that dominates p-values.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = sorted(counts_gold[0].data.keys(), key=_label_sort_key)
    use_one_vs_rest = False
    if non_entity_label is None:
        try:
            non_entity_label = _infer_non_entity_label(labels)
        except ValueError:
            use_one_vs_rest = True

    test_labels = labels if use_one_vs_rest else [lab for lab in labels if lab != non_entity_label]
    rates_arr = np.array(selection_rates, dtype=float)

    if use_one_vs_rest:
        ax.set_title("Cramér's V (one-vs-rest)")
    else:
        ax.set_title(f"Cramér's V vs {non_entity_label!r} (one-vs-non-entity)")
    ax.set_xlabel("Mean effective selection rate")
    ax.set_ylabel("Cramér's V")

    for lab in test_labels:
        vs: List[float] = []
        for pred, gold in zip(counts_pred, counts_gold):
            if use_one_vs_rest:
                table = _contingency_2x2_one_vs_rest(pred, gold, lab)
            else:
                table = _contingency_2x2_from_pred_gold(pred, gold, lab, non_entity_label)
            _, _, v = _chi_square_stats(table)
            vs.append(v)

        ax.plot(rates_arr, np.array(vs, dtype=float), marker="o", label=str(lab))

    ax.legend(fontsize="small")
    if len(rates_arr):
        ax.set_xlim(float(rates_arr.min()), float(rates_arr.max()))
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_loss_history(loss_history: Sequence[Mapping[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([dict(item) for item in loss_history], f, indent=2)


def load_loss_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, found {type(payload).__name__}.")

    history: list[dict[str, float]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected mapping entries in {path}, found {type(item).__name__}.")
        history.append({str(k): float(v) for k, v in item.items()})
    return history


def save_label_plots(
    counts_pred: Optional[Sequence[Any]],
    counts_gold: Optional[Sequence[Any]],
    selection_rates: Sequence[float],
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    if counts_pred is None or counts_gold is None:
        return

    if len(counts_pred) != len(counts_gold):
        raise ValueError(
            f"counts_pred and counts_gold must have same length, got "
            f"{len(counts_pred)} vs {len(counts_gold)}"
        )

    plots_dir.mkdir(parents=True, exist_ok=True)

    sel_path = plots_dir / "selection_rates.png"
    save_selection_rates_plot_from_pred_gold(counts_pred, counts_gold, selection_rates, str(sel_path))
    logger.info("Saved selection rate plot to %s", sel_path)

    chi_path = plots_dir / "chi_square.png"
    save_chi_square_plot_from_pred_gold(counts_pred, counts_gold, selection_rates, str(chi_path))
    logger.info("Saved chi-square plot to %s", chi_path)

    v_path = plots_dir / "cramers_v.png"
    save_cramers_v_plot_from_pred_gold(counts_pred, counts_gold, selection_rates, str(v_path))
    logger.info("Saved Cramer's V plot to %s", v_path)
