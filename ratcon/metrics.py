import datetime as _dt
from prettytable import PrettyTable

_METRIC_DISPLAY_ORDER = ("f1", "precision", "recall")


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
