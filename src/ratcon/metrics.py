"""
Metrics and reporting utilities shared across RatCon and downstream projects.

Primary entry points:
 - evaluate: run inference and return a Report.
 - simple_report: wrap an existing metrics dict into a Report.
"""

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import torch
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

_METRIC_DISPLAY_ORDER = ("f1", "precision", "recall")


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------
@dataclass
class Metrics:
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


@dataclass
class Report:
    metrics: Metrics = field(default_factory=Metrics)
    samples: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "samples": self.samples,
        }

    def log(self, logger, report_cfg, report_name=None, show_samples=False):
        header = f"Report: {report_name}" if report_name else "Report:"
        parts = [header]

        metrics_str = format_metric_values(self.metrics)
        if report_cfg.metrics:
            parts.append(metrics_str)

        if show_samples:
            num_samples = report_cfg.samples.num
            samples = self.samples[:num_samples] if num_samples > 0 else self.samples
            samples_full = []
            for s in samples:
                orig = s.get("original", "")
                pred = s.get("predicted", "")
                stats = s.get("word_stats")
                full = f"Orig: {orig}\nPred: {pred}"
                if stats is not None and report_cfg.samples.per_sentence_stats:
                    full += f"\nStats: {stats}"
                samples_full.append(full)
            if samples_full:
                parts.append("samples:\n" + "\n".join(samples_full))

        logger.info("\n".join(parts))


@dataclass
class RankingEntry:
    name: str
    metrics: dict[str, float]
    dev_metrics: dict[str, float] | None = None


@dataclass
class RankingReport:
    title: str
    entries: list[RankingEntry] = field(default_factory=list)

    def render_table(self) -> str:
        if not self.entries:
            return "(no entries)"
        table = PrettyTable()
        table.field_names = [
            "rank",
            "leaf",
            "val_f1",
            "val_precision",
            "val_recall",
            "dev_f1",
            "dev_precision",
            "dev_recall",
        ]
        for rank, entry in enumerate(self.entries, start=1):
            stats = entry.metrics or {}
            dev_stats = entry.dev_metrics or {}
            table.add_row(
                [
                    rank,
                    entry.name,
                    f"{stats.get('f1', 0.0):.4f}",
                    f"{stats.get('precision', 0.0):.4f}",
                    f"{stats.get('recall', 0.0):.4f}",
                    f"{dev_stats.get('f1', float('nan')):.4f}" if dev_stats else "",
                    f"{dev_stats.get('precision', float('nan')):.4f}" if dev_stats else "",
                    f"{dev_stats.get('recall', float('nan')):.4f}" if dev_stats else "",
                ]
            )
        return table.get_string()


# -------------------------------------------------------------------
# Label utilities
# -------------------------------------------------------------------
def _normalize_label_name(name: str | None) -> str | None:
    if not name:
        return None
    if name.upper() == "O":
        return None
    if "-" in name:
        name = name.split("-", 1)[-1]
    return name.upper()


def build_label_groups(label_names: list[str] | None) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    if not label_names:
        return groups
    for idx, raw in enumerate(label_names):
        normalized = _normalize_label_name(raw)
        if normalized is None:
            continue
        groups.setdefault(normalized, []).append(idx)
    return groups


def build_label_masks(ner_tags, label_groups, valid_mask):
    if not label_groups:
        return None
    label_masks = {}
    for label, indices in label_groups.items():
        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for label_id in indices:
            mask |= ner_tags == label_id
        label_masks[label] = mask & valid_mask
    return label_masks


# -------------------------------------------------------------------
# Count/metric builders
# -------------------------------------------------------------------
def compute_binary_counts(pred_mask: torch.Tensor, gold_mask: torch.Tensor) -> dict:
    tp = (pred_mask & gold_mask).sum().item()
    fp = (pred_mask & (~gold_mask)).sum().item()
    fn = ((~pred_mask) & gold_mask).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn}


def compute_counts(pred_mask, gold_mask, *, label_masks=None):
    counts = compute_binary_counts(pred_mask, gold_mask)
    if label_masks:
        per_class = {}
        for label, class_mask in label_masks.items():
            per_class[label] = compute_binary_counts(pred_mask, class_mask)
        counts["per_class"] = per_class
    return counts


def merge_count_dict(dest: dict, update: dict) -> dict:
    if not update:
        return dest
    dest["tp"] = dest.get("tp", 0.0) + update.get("tp", 0.0)
    dest["fp"] = dest.get("fp", 0.0) + update.get("fp", 0.0)
    dest["fn"] = dest.get("fn", 0.0) + update.get("fn", 0.0)

    update_per_class = update.get("per_class")
    if update_per_class:
        dest_per_class = dest.setdefault("per_class", {})
        for label, cls_counts in update_per_class.items():
            dest_per_class.setdefault(label, {})
            merge_count_dict(dest_per_class[label], cls_counts)
    return dest


def merge_counts_map(base: dict, updates: dict) -> dict:
    if not updates:
        return base
    if base is None:
        base = {}
    for name, counts in updates.items():
        dest = base.setdefault(name, {})
        merge_count_dict(dest, counts)
    return base


def average_metrics(metrics_list):
    """
    Average a collection of metric dictionaries (e.g., precision/recall/f1).

    Non-numeric entries are skipped. Returns an empty dict if no metrics are provided.
    """
    if not metrics_list:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metrics_list:
        if not metrics:
            continue
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts.get(k, 0) > 0}


def compute_binary_metrics_from_counts(tp, fp, fn):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_metrics_from_counts(counts, per_class_counts=None, label_groups=None):
    metrics = compute_binary_metrics_from_counts(
        counts.get("tp", 0.0),
        counts.get("fp", 0.0),
        counts.get("fn", 0.0),
    )

    if per_class_counts:
        per_class_metrics = {}
        labels = list(label_groups.keys()) if label_groups else list(per_class_counts.keys())
        for label in labels:
            cls_counts = per_class_counts.get(label)
            if not cls_counts:
                continue
            cls_metrics = compute_binary_metrics_from_counts(
                cls_counts.get("tp", 0.0),
                cls_counts.get("fp", 0.0),
                cls_counts.get("fn", 0.0),
            )
            if (cls_metrics["tp"] + cls_metrics["fp"] + cls_metrics["fn"]) == 0:
                continue
            per_class_metrics[label] = cls_metrics
        if per_class_metrics:
            metrics["per_class"] = per_class_metrics
    return metrics


def finalize_metrics_from_counts(counts_map, label_groups=None):
    if not counts_map:
        return {}
    metrics = {}
    for name, counts in counts_map.items():
        metrics[name] = build_metrics_from_counts(counts, counts.get("per_class"), label_groups)
    return metrics


# -------------------------------------------------------------------
# Reporting and formatting
# -------------------------------------------------------------------
def format_metric_values(metrics, precision=4):
    if metrics is None:
        return "metrics=n/a"

    if isinstance(metrics, Metrics):
        data = {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        }
    elif isinstance(metrics, dict):
        data = metrics
    else:
        return "metrics=n/a"

    seen = set()
    ordered_items = []
    for key in _METRIC_DISPLAY_ORDER:
        if data.get(key) is not None:
            ordered_items.append((key, data[key]))
            seen.add(key)
    for key in sorted(data.keys()):
        if key not in seen:
            ordered_items.append((key, data[key]))

    parts = [
        f"{name}={value:.{precision}f}"
        for name, value in ordered_items
        if isinstance(value, (int, float))
    ]
    return ", ".join(parts) if parts else "metrics=n/a"


def render_reports_table(reports, precision=4):
    if not reports:
        return "Metrics: none"

    table = PrettyTable()
    table.field_names = ["model", "metrics", "details"]
    table.align["model"] = "l"
    table.align["metrics"] = "l"
    table.align["details"] = "l"

    for label in sorted(reports.keys()):
        report = reports[label]
        metrics_obj = report.metrics if isinstance(report, Report) else report.get("metrics")
        metrics_str = format_metric_values(metrics_obj, precision=precision)
        table.add_row([label, metrics_str, "-"])

    return f"Metrics:\n{table.get_string()}"


def log_report(logger, report: Report | dict, report_cfg, report_name=None, show_samples=False):
    rep = report if isinstance(report, Report) else simple_report(report.get("metrics") if isinstance(report, dict) else {})
    rep.log(logger, report_cfg, report_name=report_name, show_samples=show_samples)


def build_ranking_report(
    title: str,
    metrics_dict: dict[str, dict[str, float]],
    *,
    top_k: int | None = None,
    dev_metrics: dict[str, dict[str, float]] | None = None,
    sorted_items: list[tuple[str, dict[str, float]]] | None = None,
) -> RankingReport:
    if not metrics_dict:
        return RankingReport(title=title, entries=[])

    items = sorted_items or sorted(
        metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0), reverse=True
    )
    if top_k is not None:
        items = items[:top_k]

    entries: list[RankingEntry] = []
    for name, stats in items:
        dev_stats = dev_metrics.get(name) if dev_metrics else None
        entries.append(RankingEntry(name=name, metrics=stats, dev_metrics=dev_stats))

    return RankingReport(title=title, entries=entries)


# -------------------------------------------------------------------
# Token formatting helpers
# -------------------------------------------------------------------
def merge_subwords(ids, tokens, tokenizer):
    buf = ""
    words = []

    def flush(acc):
        if acc:
            words.append(acc)
        return ""

    for tok_id, tok_str in zip(ids, tokens):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
        else:
            buf = flush(buf)
            buf = tok_str

    buf = flush(buf)
    return words


def format_gold_spans(ids, tokens, gold_labels, tokenizer):
    buf = ""
    buf_labels = []
    words, word_labels = [], []
    for tok_id, tok_str, lab in zip(ids, tokens, gold_labels):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_labels.append(lab)
        else:
            if buf:
                words.append(buf)
                word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
            buf = tok_str
            buf_labels = [lab]
    if buf:
        words.append(buf)
        word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
    out, span = [], []
    for w, l in zip(words, word_labels):
        if l:
            span.append(w)
        else:
            if span:
                out.append(f"[[{' '.join(span)}]]")
                span = []
            out.append(w)
    if span:
        out.append(f"[[{' '.join(span)}]]")
    return " ".join(out)


def merge_spans(ids, tokens, gates, tokenizer, threshold=0.5):
    buf, buf_gs = "", []
    words, word_gates = [], []

    def flush(acc, gs):
        if acc:
            words.append(acc)
            word_gates.append(sum(gs) / len(gs))
        return "", []

    for tok_id, tok_str, g in zip(ids, tokens, gates):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_gs.append(g)
        else:
            buf, buf_gs = flush(buf, buf_gs)
            buf, buf_gs = tok_str, [g]

    buf, buf_gs = flush(buf, buf_gs)

    out_tokens, span_buf = [], []

    def flush_span(span_buf):
        if span_buf:
            out_tokens.append(f"[[{' '.join(span_buf)}]]")
        return []

    for word, g in zip(words, word_gates):
        if g >= threshold:
            span_buf.append(word)
        else:
            span_buf = flush_span(span_buf)
            out_tokens.append(word)

    span_buf = flush_span(span_buf)
    return " ".join(out_tokens)


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------
def _run_inference_examples(model, data, tok, disable_progress, threshold):
    model.eval()
    device = next(model.parameters()).device
    examples = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating", disable=disable_progress):
            embeddings = batch["embeddings"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"]
            ner_tags = batch.get("ner_tags")

            out = model(embeddings, attention_mask)
            gates_tensor = out["gates"]

            for i in range(embeddings.size(0)):
                ids = input_ids[i].cpu().tolist()
                tokens = tok.convert_ids_to_tokens(ids)
                mask = attention_mask[i].cpu().tolist()
                gold = ner_tags[i].cpu().tolist() if ner_tags is not None else None

                gates = gates_tensor[i].detach().cpu().tolist()

                examples.append(
                    {
                        "ids": ids,
                        "tokens": tokens,
                        "mask": mask,
                        "gates": gates,
                        "gold": gold,
                    }
                )
    return examples


def _compute_metrics(outputs, threshold) -> Metrics:
    y_true, y_pred = [], []

    for output in outputs:
        gold_seq = output.get("gold")
        if gold_seq is None:
            continue
        for gate, gold, mask in zip(output["gates"], gold_seq, output["mask"]):
            if mask == 0:
                continue
            y_pred.append(int(gate >= threshold))
            y_true.append(int(gold != 0))

    if not y_true:
        return Metrics()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return Metrics(precision=float(precision), recall=float(recall), f1=float(f1))


def _format_samples(outputs, tok, num_samples, threshold):
    formatted = []
    for sample in outputs:
        highlight = merge_spans(sample["ids"], sample["tokens"], sample["gates"], tok, threshold=threshold)
        if num_samples and len(formatted) >= num_samples:
            break
        if sample["gold"] is not None:
            original = format_gold_spans(sample["ids"], sample["tokens"], sample["gold"], tok)
        else:
            original = " ".join(merge_subwords(sample["ids"], sample["tokens"], tok))
        formatted.append({"original": original, "predicted": highlight})
    return formatted


def make_report(
    outputs,
    tok,
    threshold=0.5,
    num_samples=0,
) -> Report:
    metrics = _compute_metrics(outputs, threshold)
    samples = _format_samples(outputs, tok, num_samples, threshold)
    return Report(metrics=metrics, samples=samples)


def simple_report(metrics: dict) -> Report:
    metrics = metrics or {}
    return Report(
        metrics=Metrics(
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1=metrics.get("f1"),
        )
    )


def evaluate(
    model,
    data,
    tok,
    threshold=0.5,
    disable_progress=False,
    samples_num=0,
    logger=None,
) -> Report:
    outputs = _run_inference_examples(model, data, tok, disable_progress, threshold)
    report = make_report(outputs, tok, threshold=threshold, num_samples=samples_num)
    return report
