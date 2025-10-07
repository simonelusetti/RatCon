from dora import Explorer
import treetable as tt

class RatConExplorer(Explorer):
    metric_names = ("precision", "recall", "f1")

    def get_grid_metrics(self):
        return [
            tt.group(
                "best",
                [
                    tt.leaf("best_epoch"),
                    *(tt.leaf(name, ".4f") for name in self.metric_names),
                ],
                align=">")
        ]

    def process_history(self, history):
        best_epoch, best_metrics = summarize_best_metrics(history, self.metric_names)

        result = {"best": {"best_epoch": best_epoch}}
        result["best"].update(best_metrics)
        return result

def _extract_epoch_from_path(path):
    """Pull the first purely numeric segment out of a history path."""
    for segment in path.split("/"):
        if segment.isdigit():
            return int(segment)
    return None


def summarize_best_metrics(history, metric_names):
    """
    Return the best epoch (1-indexed when available) and metric values found in the
    provided Dora history entries.

    Prefers the explicit ``best_eval`` entries emitted by the trainer. Falls back to
    legacy summary records or per-epoch ``eval`` metrics for older runs.
    """

    def _empty_metrics():
        return {name: None for name in metric_names}

    # First choice: dedicated best_eval reports pushed at the end of training.
    best_metrics = _empty_metrics()
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for path, payload in entry.items():
            if not isinstance(path, str) or not isinstance(payload, dict):
                continue
            if not path.startswith("best_eval/"):
                continue

            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                for name in metric_names:
                    best_metrics[name] = metrics.get(name)

            best_epoch = None
            segments = path.split("/")
            if len(segments) >= 2:
                try:
                    best_epoch = int(segments[1])
                except ValueError:
                    best_epoch = None
            if best_epoch is None:
                payload_epoch = payload.get("epoch")
                if isinstance(payload_epoch, int):
                    best_epoch = payload_epoch

            return best_epoch, best_metrics

    # Fall back to the legacy summary block when present.
    best_metrics = _empty_metrics()
    for entry in history:
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary")
        if not isinstance(summary, dict):
            continue

        metrics = summary.get("best_metrics", {})
        for name in metric_names:
            if name in metrics:
                best_metrics[name] = metrics[name]
        best_epoch = summary.get("best_epoch")
        return best_epoch, best_metrics

    # As a last resort, scan per-epoch eval entries to recover the best F1.
    best_metrics = _empty_metrics()
    best_epoch = None
    best_f1 = float("-inf")

    for entry in history:
        if not isinstance(entry, dict):
            continue
        for path, payload in entry.items():
            if not isinstance(path, str) or not isinstance(payload, dict):
                continue

            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                continue

            f1 = metrics.get("f1")
            if not isinstance(f1, (int, float)):
                continue

            payload_epoch = payload.get("epoch")
            if isinstance(payload_epoch, int):
                epoch_value = payload_epoch
            else:
                epoch_value = _extract_epoch_from_path(path)

            if f1 > best_f1:
                best_f1 = float(f1)
                best_epoch = epoch_value + 1 if isinstance(epoch_value, int) else None
                for name in metric_names:
                    best_metrics[name] = metrics.get(name)

    return best_epoch, best_metrics
