import datetime as _dt
import logging
import os
import sys
from pathlib import Path

import torch
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from .models import nt_xent

_METRIC_DISPLAY_ORDER = ("f1", "precision", "recall")


def compute_binary_counts(pred_mask: torch.Tensor, gold_mask: torch.Tensor) -> dict:
    """Return TP/FP/FN counts for boolean prediction and gold masks."""
    tp = (pred_mask & gold_mask).sum().item()
    fp = (pred_mask & (~gold_mask)).sum().item()
    fn = ((~pred_mask) & gold_mask).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn}


def build_label_masks(ner_tags, label_groups, valid_mask):
    """Construct boolean masks for each label group aligned with ner_tags."""
    if not label_groups:
        return None
    label_masks = {}
    for label, indices in label_groups.items():
        mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for label_id in indices:
            mask |= ner_tags == label_id
        label_masks[label] = mask & valid_mask
    return label_masks


def compute_counts(pred_mask, gold_mask, *, label_masks=None):
    """Compute TP/FP/FN counts plus optional per-class counts."""
    counts = compute_binary_counts(pred_mask, gold_mask)
    if label_masks:
        per_class = {}
        for label, class_mask in label_masks.items():
            per_class[label] = compute_binary_counts(pred_mask, class_mask)
        counts["per_class"] = per_class
    return counts


def merge_count_dict(dest: dict, update: dict) -> dict:
    """Accumulate TP/FP/FN counts (including nested per_class entries)."""
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
    """Accumulate counts keyed by factor/leaf name."""
    if not updates:
        return base
    if base is None:
        base = {}
    for name, counts in updates.items():
        dest = base.setdefault(name, {})
        merge_count_dict(dest, counts)
    return base


def _normalize_label_name(name: str | None) -> str | None:
    """Normalize BIO label names to a canonical key (strips BIO prefix, drops 'O')."""
    if not name:
        return None
    if name.upper() == "O":
        return None
    if "-" in name:
        name = name.split("-", 1)[-1]
    return name.upper()


def build_label_groups(label_names: list[str] | None) -> dict[str, list[int]]:
    """Map normalized label names to the indices they correspond to."""
    groups: dict[str, list[int]] = {}
    if not label_names:
        return groups
    for idx, raw in enumerate(label_names):
        normalized = _normalize_label_name(raw)
        if normalized is None:
            continue
        groups.setdefault(normalized, []).append(idx)
    return groups


def complement_margin_loss(h_anchor, h_comp, margin=0.3):
    # we want cosine(anchor, comp) to be LOW -> (1 - cos) to be HIGH
    cos = (h_anchor * h_comp).sum(dim=-1)            # [B]
    neg = 1.0 - cos
    return torch.relu(margin - neg).mean()


def sparsity_loss(gates, mask):
    # average gate value over valid tokens only (mask==1)
    valid = (mask > 0).float()
    return (gates * valid).sum() / (valid.sum() + 1e-8)


def total_variation_1d(gates, mask):
    # penalize changes across adjacent valid tokens
    valid = (mask > 0).float()
    diff = torch.abs(gates[:, 1:] - gates[:, :-1])
    # only count where both tokens are valid
    both = valid[:, 1:] * valid[:, :-1]
    return (diff * both).sum() / (both.sum() + 1e-8)


def complement_loss(h_comp, h_anchor, temperature=0.07):
    """Repel complements from anchors (no null embedding target)."""
    return -nt_xent(h_comp, h_anchor, temperature=temperature)


def compute_training_objectives(
    model,
    output,
    attention_mask,
    model_cfg,
    *,
    temperature,
):
    """Compute total loss for single-model training."""
    anchors = output["h_anchor"]
    gates = output["gates"]

    l_rat = nt_xent(output["h_rat"], anchors, temperature=temperature)
    l_comp = complement_loss(output["h_comp"], anchors, temperature=temperature)
    l_s = sparsity_loss(gates, attention_mask)
    l_tv = total_variation_1d(gates, attention_mask)

    loss_cfg = model_cfg.loss
    loss = l_rat
    loss = loss + float(loss_cfg.l_comp) * l_comp
    loss = loss + float(loss_cfg.l_s) * l_s
    loss = loss + float(loss_cfg.l_tv) * l_tv

    return loss


def compute_binary_metrics_from_counts(tp, fp, fn):
    """Return precision/recall/F1 plus counts for scalar TP/FP/FN."""
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
    """Project TP/FP/FN dictionaries into precision/recall/F1 metrics."""
    metrics = compute_binary_metrics_from_counts(
        counts.get("tp", 0.0),
        counts.get("fp", 0.0),
        counts.get("fn", 0.0),
    )

    if per_class_counts:
        per_class_metrics = {}
        if label_groups:
            labels = list(label_groups.keys())
        else:
            labels = list(per_class_counts.keys())

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
    """Convert a name->count dictionary into precision/recall/F1 metrics."""
    if not counts_map:
        return {}
    metrics = {}
    for name, counts in counts_map.items():
        metrics[name] = build_metrics_from_counts(counts, counts.get("per_class"), label_groups)
    return metrics


def summarize_word_stats(word_stats):
    word_stats = list(word_stats)
    if not word_stats:
        return {
            "total_examples": 0,
            "examples_with_highlights": 0,
            "total_highlighted_words": 0,
            "counts": {},
            "proportions": {},
        }

    counts = {}
    total_highlighted = 0.0
    examples_with_highlights = 0

    for stats in word_stats:
        total = float(stats.get("total", 0.0))
        if total > 0:
            examples_with_highlights += 1
            total_highlighted += total
        for key, value in stats.items():
            if key == "total":
                continue
            counts[key] = counts.get(key, 0.0) + float(value)

    if total_highlighted > 0:
        proportions = {key: counts[key] / total_highlighted for key in sorted(counts.keys())}
    else:
        proportions = {key: 0.0 for key in sorted(counts.keys())}

    return {
        "total_examples": len(word_stats),
        "examples_with_highlights": examples_with_highlights,
        "total_highlighted_words": int(total_highlighted),
        "counts": counts,
        "proportions": proportions,
    }


def format_metric_values(metrics, precision=4):
    if not metrics:
        return "metrics=n/a"

    seen = set()
    ordered_items = []
    for key in _METRIC_DISPLAY_ORDER:
        if key in metrics:
            ordered_items.append((key, metrics[key]))
            seen.add(key)
    for key in sorted(metrics.keys()):
        if key not in seen:
            ordered_items.append((key, metrics[key]))

    parts = [f"{name}={value:.{precision}f}" for name, value in ordered_items]
    return ", ".join(parts)


def format_word_summary(summary):
    if not summary:
        return ""
    total_words = summary.get("total_highlighted_words", 0)
    if not total_words:
        return "highlighted_words=0"

    proportions = summary.get("proportions", {})
    sorted_props = sorted(proportions.items(), key=lambda item: item[1], reverse=True)
    top_chunks = []
    for key, value in sorted_props[:4]:
        top_chunks.append(f"{key}={value * 100:.1f}%")

    return (
        f"highlighted_words={total_words}"
        + (f" ({', '.join(top_chunks)})" if top_chunks else "")
    )


def build_report(evaluation):
    metrics = evaluation.get("metrics") or {}
    metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    report = {
        "timestamp": _dt.datetime.now().isoformat(),
        "metrics": metrics,
        "samples": evaluation.get("samples") or [],
        "word_summary": evaluation.get("word_summary") or {},
    }
    return report


def log_report(logger, report, report_cfg, report_name=None, show_samples=False):
    if report_name:
        parts = [f"Report: {report_name}\n"]
    else:
        parts = ["Report: \n"]
    parts = []
    
    metrics = format_metric_values(report.get("metrics"))
    if report_cfg.metrics: parts.append(metrics+"\n")

    summary = format_word_summary(report.get("word_summary"))
    if report_cfg.summary: parts.append(summary+"\n")

    # Samples (condensed)
    if show_samples:
        num_samples = report_cfg.samples.num
        if num_samples > 0:
            samples = report.get("samples", [])[:num_samples]
        else:
            samples = report.get("samples", [])
        samples_full = []
        for s in samples:
            orig = s.get("original", "")
            pred = s.get("predicted", "")
            stats = s.get("word_stats")
            full = f"Orig: {orig}\nPred: {pred}"
            if stats is not None and report_cfg.samples.per_sentence_stats:
                full += f"\nStats: {stats}"
            full += f"\n"
            samples_full.append(full)
        parts.append("samples:\n" + "\n".join(samples_full))

    # Emit single log line
    logger.info("\n".join(parts))


def render_reports_table(reports, precision=4):    
    if not reports:
        prefix = "Metrics"
        return f"{prefix}: none"

    table = PrettyTable()
    table.field_names = ["model", "metrics", "details"]
    table.align["model"] = "l"
    table.align["metrics"] = "l"
    table.align["details"] = "l"

    for label in sorted(reports.keys()):
        report = reports[label]
        metrics_str = format_metric_values(report.get("metrics"), precision=precision)
        
        extras = []
        table.add_row([label, metrics_str, ", ".join(extras) if extras else "-"])

    prefix = "Metrics"
    return f"{prefix}:\n{table.get_string()}"


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


def merge_spans(ids, tokens, gates, tokenizer, thresh=0.5):
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
        if g >= thresh:
            span_buf.append(word)
        else:
            span_buf = flush_span(span_buf)
            out_tokens.append(word)

    span_buf = flush_span(span_buf)
    return " ".join(out_tokens)


def _run_inference_examples(model, data, tok, disable_progress, thresh):
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
                gold = None
                if ner_tags is not None:
                    gold = ner_tags[i].cpu().tolist()

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


def _collect_samples(examples, tok, thresh, num_samples):
    samples = []
    for example in examples:
        highlight = merge_spans(example["ids"], example["tokens"], example["gates"], tok, thresh=thresh)
        if num_samples and len(samples) >= num_samples:
            break
        if example["gold"] is not None:
            original = format_gold_spans(example["ids"], example["tokens"], example["gold"], tok)
        else:
            original = " ".join(merge_subwords(example["ids"], example["tokens"], tok))
        samples.append({"original": original, "predicted": highlight})
    return samples


def _compute_metrics_from_examples(examples, threshold):
    y_true, y_pred = [], []
    for example in examples:
        if example["gold"] is None:
            continue
        for gate, lab, mask in zip(example["gates"], example["gold"], example["mask"]):
            if mask == 0:
                continue
            y_pred.append(int(gate >= threshold))
            y_true.append(int(lab != 0))

    if not y_true:
        return None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(
    model,
    data,
    tok,
    tresh,
    disable_progress=False,
    thresh=0.5,
    samples_num=0,
    logger=None,
):
    examples = _run_inference_examples(model, data, tok, disable_progress, thresh)
    metrics = _compute_metrics_from_examples(examples, tresh)
    samples = _collect_samples(examples, tok, thresh, samples_num)
    return {
        "metrics": metrics,
        "samples": samples,
    }


def should_disable_tqdm(*, metrics_only=False):
    """Return True when tqdm progress bars should be disabled."""
    if metrics_only:
        return True

    override = os.environ.get("RATCON_DISABLE_TQDM")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}

    try:
        return not sys.stderr.isatty()
    except Exception:
        return True


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:  # pragma: no cover - logging fallback
            self.handleError(record)


def get_logger(logfile="train.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch_format = "%(asctime)s - %(levelname)s - %(message)s"
    ch.setFormatter(logging.Formatter(ch_format))

    fh = logging.FileHandler(Path(logfile))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
