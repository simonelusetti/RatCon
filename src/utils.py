from __future__ import annotations

import logging, os, sys, json, torch, matplotlib.pyplot as plt, torch.nn.functional as F

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
from dora import XP
from prettytable import PrettyTable
from tqdm import tqdm

from tqdm import tqdm
tqdm._instances.clear()

# ---------------------------------------------------------------------------
# Runtime Config
# ---------------------------------------------------------------------------


def configure_runtime(runtime_cfg: Dict) -> Tuple[Dict, bool]:
    changed_device = False
    
    if "threads" in runtime_cfg and runtime_cfg["threads"] is not None:
        torch.set_num_threads(int(runtime_cfg["threads"]))
    if "interop_threads" in runtime_cfg and runtime_cfg["interop_threads"] is not None:
        torch.set_num_interop_threads(int(runtime_cfg["interop_threads"]))
    if runtime_cfg["device"] == "cuda" and not torch.cuda.is_available():
        changed_device
    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")
    runtime_cfg["device"] = device.type
    
    return runtime_cfg, changed_device
        

def to_device(device: torch.device, batch: Dict):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# I/O Utilities
# ---------------------------------------------------------------------------


def open_selection_writer(xp: XP, epoch: int):
    out_dir = xp.folder / "selections"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"eval_epoch_{epoch:03d}.json"
    return open(path, "w")


def save_final_plots(
    counts_pred_history, counts_gold_history, 
    loss_history,
    labels_present, logger, xp
):
    fig, axs = plt.subplots(2, 1, figsize=(16, 16))
    axs = axs.flatten()

    plots = [
        {
            "type": "dict",
            "data": loss_history,
            "title": "Losses Accross Epochs",
            "ylabel": "Share",
        },
        {
            "type": "dict",
            "data": [
                (pred / gold).data
                for pred, gold in zip(
                    counts_pred_history,
                    counts_gold_history,
                )
            ],
            "title": "Label Share Across Epochs",
            "ylabel": "Share",
        },
    ]

    for ax, plot_cfg in zip(axs, plots):
        ax.set_title(plot_cfg["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(plot_cfg["ylabel"])
        ax.set_yscale("log", nonpositive="clip")
        values = None

        if plot_cfg["type"] == "simple":
            values = plot_cfg["data"]
            ax.plot(values)

        elif plot_cfg["type"] == "dict" and labels_present:
            epochs = range(1, len(plot_cfg["data"]) + 1)
            labels = {k for d in plot_cfg["data"] for k in d}
            values = []

            for label in labels:
                ys = [epoch.get(label, 0) for epoch in plot_cfg["data"]]
                values.extend(ys)
                ax.plot(epochs, ys, label=label)

            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="small",
            )

        if values:
            ax.set_ylim(min(values), max(values))

    fig.tight_layout()
    out = "summary_plots.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Saved summary plots to %s for experiment %s",
        out,
        xp.sig,
    )
    
    with open(Path("histories.json"), "w") as f:
        json.dump(
            {
                "loss":loss_history,
                "labels":[
                    (pred / gold).data
                    for pred, gold in zip(
                        counts_pred_history,
                        counts_gold_history,
                    )
                ]
            },
        f, indent=2)

    if labels_present:
        full_conf_matrix(logger, counts_gold_history, counts_pred_history, labels_present)

def full_conf_matrix(logger, counts_gold_history, counts_pred_history, labels_present):
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
    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:  # pragma: no cover - logging fallback
            self.handleError(record)


def get_logger(
    logfile: str = "train.log",
) -> logging.Logger:
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

def make_table(
    fields: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> PrettyTable:
    table = PrettyTable()
    table.field_names = fields
    for row in rows:
        table.add_row(row)
    return table

def dict_to_table(
    losses: Mapping[str, float],
) -> PrettyTable:
    fields = losses.keys()
    rows = [[f"{losses.get(k, 0.0):.6f}" for k in fields]]
    return make_table(fields, rows)


def format_dict(d, new_liners=None):
    extra_newline_after = new_liners or set()
    lines = []

    for k, v in d.items():
        if isinstance(v, (int, float)):
            v = f"{v:.5f}"
        lines.append(f"{k}: {v}")
        if k in extra_newline_after:
            lines.append("") 

    return "\n".join(lines)


def tkns_to_words(
    gates: torch.Tensor,          # [B, T]
    attn_mask: torch.Tensor,      # [B, T]
    word_ids: torch.Tensor,       # [B, T], -1 for invalid
    labels: list[list[str]],      # [B, T], duplicated per word
    threshold: float = 0.0,
):
    """
    Word-level OR aggregation.

    Returns:
      word_pred: BoolTensor [B, W]   (selected words)
      word_attn: BoolTensor [B, W]   (valid words)
      word_labels: list[list[str]]  (one label per word, left-to-right)
    """
    device = gates.device
    B, T = gates.shape

    # Binary subword selection, masked by attention
    sel = (gates > threshold) & attn_mask.bool()   # [B, T]

    # Number of words in batch (max wid + 1)
    W = int(word_ids.max().item()) + 1

    # One-hot word assignment: [B, T, W]
    word_onehot = torch.zeros(B, T, W, device=device, dtype=torch.bool)
    valid = word_ids >= 0
    word_onehot[valid] = F.one_hot(word_ids[valid], num_classes=W).bool()

    # OR aggregation (max over subwords)
    word_pred = (sel.unsqueeze(-1) & word_onehot).any(dim=1)        # [B, W]
    word_attn = (attn_mask.bool().unsqueeze(-1) & word_onehot).any(dim=1)  # [B, W]

    # ---- labels: extracted once per word (Python, ordered) ----
    word_labels = []

    for b in range(B):
        label_by_wid = {}
        for t, wid in enumerate(word_ids[b].tolist()):
            if wid == -100:
                continue
            if wid not in label_by_wid:
                label_by_wid[wid] = labels[b][t]

        # ensure left-to-right order
        wids_sorted = sorted(label_by_wid.keys())
        word_labels.append([label_by_wid[wid] for wid in wids_sorted])

    return word_pred, word_attn, word_labels
