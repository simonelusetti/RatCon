from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

_METRIC_DISPLAY_ORDER = ("f1", "precision", "recall")


@dataclass
class EvaluationResult:
    """Container returned by evaluate with metrics, samples, and word stats."""

    metrics: Optional[Dict[str, float]]
    samples: List[Dict[str, Any]]
    word_stats: List[Dict[str, float]]
    word_summary: Dict[str, Any]


def summarize_word_stats(word_stats: Iterable[Dict[str, float]]) -> Dict[str, Any]:
    word_stats = list(word_stats)
    if not word_stats:
        return {
            "total_examples": 0,
            "examples_with_highlights": 0,
            "total_highlighted_words": 0,
            "counts": {},
            "proportions": {},
        }

    counts: Dict[str, float] = {}
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

    proportions: Dict[str, float] = {}
    if total_highlighted > 0:
        proportions = {
            key: counts[key] / total_highlighted for key in sorted(counts.keys())
        }
    else:
        proportions = {key: 0.0 for key in sorted(counts.keys())}

    return {
        "total_examples": len(word_stats),
        "examples_with_highlights": examples_with_highlights,
        "total_highlighted_words": int(total_highlighted),
        "counts": counts,
        "proportions": proportions,
    }


def format_metric_values(metrics: Optional[Dict[str, float]], precision: int = 4) -> str:
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


def format_word_summary(summary: Dict[str, Any]) -> str:
    if not summary:
        return ""
    total_words = summary.get("total_highlighted_words", 0)
    if not total_words:
        return "highlighted_words=0"

    proportions: Dict[str, float] = summary.get("proportions", {})
    sorted_props = sorted(proportions.items(), key=lambda item: item[1], reverse=True)
    top_chunks = []
    for key, value in sorted_props[:4]:
        top_chunks.append(f"{key}={value * 100:.1f}%")

    return (
        f"highlighted_words={total_words}"
        + (f" ({', '.join(top_chunks)})" if top_chunks else "")
    )


def build_evaluation_payload(
    dataset: str,
    model_label: str,
    result: EvaluationResult,
    timestamp: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "dataset": dataset,
        "model": model_label,
        "metrics": result.metrics,
        "timestamp": timestamp or _dt.datetime.now().isoformat(),
        "word_stats": result.word_stats,
        "word_summary": result.word_summary,
        "num_samples": len(result.samples),
    }
    if extra:
        payload.update(extra)
    return payload


def log_evaluation_result(
    logger,
    dataset: str,
    model_label: str,
    result: EvaluationResult,
    *,
    prefix: Optional[str] = None,
    show_samples: bool = False,
    per_sentence_stats: bool = False,
) -> None:
    label = model_label or "model"

    def _log(message: str) -> None:
        if prefix:
            logger.info(f"{prefix} {message}")
        else:
            logger.info(message)

    metrics_line = format_metric_values(result.metrics)
    _log(f"[{label}] dataset={dataset} {metrics_line}")

    summary_line = format_word_summary(result.word_summary)
    if summary_line:
        _log(f"[{label}] dataset={dataset} {summary_line}")

    if show_samples and result.samples:
        for idx, sample in enumerate(result.samples, start=1):
            original = sample.get("original", "")
            predicted = sample.get("predicted", "")
            _log(f"[{label}] sample {idx} orig: {original}")
            _log(f"[{label}] sample {idx} pred: {predicted}")
            if per_sentence_stats:
                stats = sample.get("word_stats")
                if stats is not None:
                    _log(f"[{label}] sample {idx} word_stats: {stats}")


def _format_top_words(summary: Dict[str, Any], max_items: int = 3) -> str:
    proportions = summary.get("proportions", {}) if summary else {}
    items = sorted(proportions.items(), key=lambda item: item[1], reverse=True)
    chunks = []
    for key, value in items:
        if value <= 0:
            continue
        chunks.append(f"{key} {value * 100:.1f}%")
        if len(chunks) == max_items:
            break
    return ", ".join(chunks) if chunks else "-"


def render_metrics_table(
    results: Dict[str, EvaluationResult],
    *,
    dataset: Optional[str] = None,
    precision: int = 4,
) -> str:
    if not results:
        header = f"Metrics table{f' (dataset: {dataset})' if dataset else ''}"
        return f"{header}\n(no results)"

    headers = ["Model", "F1", "Precision", "Recall", "Highlighted", "Top words"]
    rows = []
    for label in sorted(results.keys()):
        result = results[label]
        metrics = result.metrics or {}
        f1 = metrics.get("f1")
        precision_val = metrics.get("precision")
        recall_val = metrics.get("recall")
        highlight_total = result.word_summary.get("total_highlighted_words", 0) if result.word_summary else 0
        top_words = _format_top_words(result.word_summary)
        def fmt(value):
            return f"{value:.{precision}f}" if value is not None else "-"
        rows.append([
            label,
            fmt(f1),
            fmt(precision_val),
            fmt(recall_val),
            str(highlight_total),
            top_words,
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def render_row(cells):
        return " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in col_widths)
    lines = []
    title = "Metrics table"
    if dataset:
        title += f" (dataset: {dataset})"
    lines.append(title)
    lines.append(render_row(headers))
    lines.append(separator)
    for row in rows:
        lines.append(render_row(row))
    return "\n".join(lines)
