import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from .metrics import SelectionLog, bonferroni_threshold, exact_null_distribution, exact_pvalue_z


_ALPHA = 0.05
_DEFAULT_SELECTION_RATE_CURVES_PATH = Path("data") / "selection_rate_curves.json"
_DEFAULT_PVALUE_CURVES_PATH = Path("data") / "pvalue_curves.json"
_DEFAULT_EFFECT_SIZE_CURVES_PATH = Path("data") / "effect_size_curves.json"
_DEFAULT_EXACT_TEST_SUMMARY_PATH = Path("data") / "exact_test_summary.json"
_DEFAULT_FREQ_BINNED_EFFECTS_PATH = Path("data") / "freq_binned_effects.json"
_DEFAULT_SELECTION_LOGREG_PATH = Path("data") / "selection_logreg.json"
_DEFAULT_SPEARMAN_CURVES_PATH = Path("data") / "spearman_curves.json"
_FREQ_BINS = 5


def _bonferroni_stats(m: int, alpha: float = _ALPHA) -> dict[str, Any]:
    """m, alpha, Bonferroni threshold, and the corresponding two-sided
    z-critical value (norm.ppf(1 - threshold/2)) — the z magnitude a cell
    must clear to be Bonferroni-significant, used as the effect-size plot's
    reference line instead of a fixed, dataset-agnostic constant."""
    threshold = bonferroni_threshold(m, alpha)
    z_critical = float(norm.ppf(1.0 - threshold / 2.0)) if threshold > 0 else None
    return {"m": m, "alpha": alpha, "threshold": threshold, "z_critical": z_critical}


def _budget_k_np(rho: float, n_values: np.ndarray) -> np.ndarray:
    """Per-sentence token budget k = round(rho * n), clamped to >=1 wherever
    n > 0 (else 0) — the exact formula selector.py's forward() uses, so the
    exact null respects the same budget the real selector does."""
    k = np.round(rho * n_values).astype(np.int64)
    return np.where(n_values > 0, np.maximum(k, 1), 0)


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


def _build_selections_payload(
    counts_pred: Sequence[Any] | None,
    counts_gold: Sequence[Any] | None,
    rhos: Sequence[float] | None,
) -> dict[str, Any] | None:
    if counts_pred is None or counts_gold is None or rhos is None:
        return None
    return {
        "selection_rates": [float(r) for r in rhos],
        "selections_by_rho": [
            {"rho": float(rho), "pred_counts": dict(pred.data), "gold_counts": dict(gold.data)}
            for rho, pred, gold in zip(rhos, counts_pred, counts_gold)
        ],
    }


def _build_exact_test_payload(
    counts_pred: Sequence[Any] | None,
    rhos: Sequence[float] | None,
    selection_log: SelectionLog | None,
    label_order: Sequence[str] | None,
) -> dict[str, Any] | None:
    """Per (label, rho) exact test: an exact closed-form null distribution
    (convolution of per-sentence Hypergeometric(N_i, c_i, k_i) distributions
    — see metrics.exact_null_distribution) replaces both Monte Carlo
    simulation and a normal approximation to it. No p-value floor, and valid
    uniformly for common and rare labels (unlike a CLT-based normal
    approximation, which breaks down for labels that occur in only a handful
    of sentences). Bonferroni-corrected across the full (label, rho) grid of
    this dataset (m = num_labels x num_rhos, threshold = alpha / m).
    """
    if counts_pred is None or rhos is None or selection_log is None or not label_order:
        return None
    if not counts_pred:
        return {"labels": [], "rows": [], **_bonferroni_stats(0)}

    labels, _log_freqs, _selected, sentence_ids, names = selection_log.arrays()
    label_to_idx = {name: i for i, name in enumerate(names)}
    num_sentences = int(sentence_ids.max()) + 1 if sentence_ids.size else 0
    n_all = np.bincount(sentence_ids, minlength=num_sentences)

    results: dict[str, list[dict[str, Any]]] = {label: [None] * len(rhos) for label in label_order}
    for label in label_order:
        li = label_to_idx.get(label)
        c_all = (
            np.bincount(sentence_ids[labels == li], minlength=num_sentences)
            if li is not None
            else np.zeros(num_sentences, dtype=np.int64)
        )
        nonzero = c_all > 0
        n_nz, c_nz = n_all[nonzero], c_all[nonzero]

        for r_idx, rate in enumerate(rhos):
            observed = int(counts_pred[r_idx].data.get(label, 0))
            k_nz = _budget_k_np(float(rate), n_nz)
            null_pmf = exact_null_distribution(n_nz.tolist(), c_nz.tolist(), k_nz.tolist())
            p_value, z = exact_pvalue_z(observed, null_pmf)
            idx = np.arange(null_pmf.size)
            mean = float(np.dot(idx, null_pmf))
            std = float(np.dot((idx - mean) ** 2, null_pmf)) ** 0.5
            results[label][r_idx] = {
                "label": str(label),
                "observed": observed,
                "null_mean": mean,
                "null_std": std,
                "p_value": float(p_value),
                "z": float(z),
            }

    rows = [
        {"rho": float(rate), "labels": [results[label][r_idx] for label in label_order]}
        for r_idx, rate in enumerate(rhos)
    ]

    cells = [cell for row in rows for cell in row["labels"]]
    stats = _bonferroni_stats(len(cells))
    for cell in cells:
        cell["significant"] = cell["p_value"] < stats["threshold"]

    return {"labels": [str(label) for label in label_order], "rows": rows, **stats}


def _extract_selection_curves(selections: Mapping[str, Any]) -> tuple[list[float], dict[str, list[float]]]:
    rows = selections.get("selections_by_rho")
    if not isinstance(rows, list) or not rows:
        return [], {}

    labels: set[str] = set()
    for row in rows:
        pred_counts = row.get("pred_counts", {}) if isinstance(row, Mapping) else {}
        gold_counts = row.get("gold_counts", {}) if isinstance(row, Mapping) else {}
        if isinstance(pred_counts, Mapping):
            labels.update(str(label) for label in pred_counts.keys())
        if isinstance(gold_counts, Mapping):
            labels.update(str(label) for label in gold_counts.keys())

    sorted_labels = sorted(labels)
    rho_values: list[float] = []
    curves: dict[str, list[float]] = {label: [] for label in sorted_labels}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rho_values.append(float(row.get("rho", 0.0)))
        pred_counts_raw = row.get("pred_counts", {})
        gold_counts_raw = row.get("gold_counts", {})
        pred_counts = {str(k): float(v) for k, v in pred_counts_raw.items()} if isinstance(pred_counts_raw, Mapping) else {}
        gold_counts = {str(k): float(v) for k, v in gold_counts_raw.items()} if isinstance(gold_counts_raw, Mapping) else {}
        for label in sorted_labels:
            kept = pred_counts.get(label, 0.0)
            total = gold_counts.get(label, 0.0)
            curves[label].append((kept / total) if total > 0.0 else 0.0)

    return rho_values, curves


def _extract_exact_test_curves(
    exact_test: Mapping[str, Any],
) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]], dict[str, list[bool]]]:
    rows = exact_test.get("rows")
    if not isinstance(rows, list) or not rows:
        return [], {}, {}, {}

    rho_values: list[float] = []
    pvalue_curves: dict[str, list[float]] = {}
    z_curves: dict[str, list[float]] = {}
    significance_curves: dict[str, list[bool]] = {}

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rho_values.append(float(row.get("rho", 0.0)))
        labels = row.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", ""))
            p_value = max(float(item.get("p_value", 1.0)), 1e-300)
            pvalue_curves.setdefault(label, []).append(-math.log10(p_value))
            z_curves.setdefault(label, []).append(float(item.get("z", 0.0)))
            significance_curves.setdefault(label, []).append(bool(item.get("significant", False)))

    return rho_values, pvalue_curves, z_curves, significance_curves


def _build_selection_rate_curves_payload(selections: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"rho": [], "curves": {}, "baseline": {"kind": "identity", "label": "y=x"}}
    if not isinstance(selections, Mapping):
        return payload

    rho_values, curves = _extract_selection_curves(selections)
    if rho_values and curves:
        payload["rho"] = rho_values
        payload["curves"] = curves
    return payload


def _build_pvalue_curves_payload(exact_test: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rho": [],
        "curves": {},
        "baseline": {
            "kind": "constant",
            "label": "p=0.05",
            "value": float(-math.log10(_ALPHA)),
        },
    }
    if not isinstance(exact_test, Mapping):
        return payload

    rho_values, pvalue_curves, _, sig_curves = _extract_exact_test_curves(exact_test)
    if rho_values and pvalue_curves:
        payload["rho"] = rho_values
        payload["curves"] = pvalue_curves
        payload["significant"] = sig_curves
    return payload


def _build_effect_size_curves_payload(exact_test: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rho": [],
        "curves": {},
        "baseline": {"kind": "constant", "label": "z-critical (Bonferroni alpha=0.05)", "value": 0.0},
    }
    if not isinstance(exact_test, Mapping):
        return payload

    z_critical = exact_test.get("z_critical")
    if isinstance(z_critical, (int, float)):
        payload["baseline"]["value"] = float(z_critical)
        payload["baseline"]["label"] = f"±{float(z_critical):.2f} (Bonferroni, m={exact_test.get('m', '?')})"

    rho_values, _, z_curves, sig_curves = _extract_exact_test_curves(exact_test)
    if rho_values and z_curves:
        payload["rho"] = rho_values
        payload["curves"] = z_curves
        payload["significant"] = sig_curves
    return payload


def _bin_indices(log_freqs: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Assign each token to a log-frequency quantile bin; returns (bin index per token, bin edges)."""
    edges = np.quantile(log_freqs, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)  # ties (very peaked distributions) collapse bins
    idx = np.clip(np.searchsorted(edges, log_freqs, side="right") - 1, 0, len(edges) - 2)
    bounds = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]
    return idx, bounds


def build_freq_binned_effects_payload(selection_log: SelectionLog | None) -> dict[str, Any]:
    """Exact-test effect size/p-value per (label, rho) within token
    log-frequency bins, so a preference that only reflects token rarity
    disappears once frequency is held (approximately) constant.

    Uses the same exact hypergeometric-convolution null as the main test
    (metrics.exact_null_distribution), keyed by the joint (label, bin)
    category per sentence instead of label alone — no simulation, no
    padded-tensor reconstruction needed (unlike the Monte Carlo approach
    this replaced), just per-sentence counts via np.bincount.
    """
    payload: dict[str, Any] = {"rho": [], "labels": [], "bins": []}
    if selection_log is None:
        return payload

    labels, log_freqs, selected, sentence_ids, names = selection_log.arrays()
    if labels.size == 0:
        return payload

    rhos = selection_log.rhos
    bin_idx, bounds = _bin_indices(log_freqs, _FREQ_BINS)

    payload["rho"] = list(rhos)
    payload["labels"] = names

    num_sentences = int(sentence_ids.max()) + 1
    n_all = np.bincount(sentence_ids, minlength=num_sentences)

    for b, (lo, hi) in enumerate(bounds):
        in_bin = bin_idx == b
        n_bin = int(in_bin.sum())
        grid: list[list[dict[str, Any]]] = [[None] * len(names) for _ in rhos]

        for li, name in enumerate(names):
            is_label_bin = in_bin & (labels == li)
            c_all = np.bincount(sentence_ids[is_label_bin], minlength=num_sentences)
            nonzero = c_all > 0
            n_nz, c_nz = n_all[nonzero], c_all[nonzero]

            lab_in_bin = labels[in_bin] == li
            n_l = int(lab_in_bin.sum())

            for r_idx, rate in enumerate(rhos):
                sel_in_bin = selected[r_idx][in_bin]
                observed = int((sel_in_bin & lab_in_bin).sum())
                k_nz = _budget_k_np(float(rate), n_nz)
                null_pmf = exact_null_distribution(n_nz.tolist(), c_nz.tolist(), k_nz.tolist())
                p_value, z = exact_pvalue_z(observed, null_pmf)
                grid[r_idx][li] = {
                    "label": name,
                    "z": float(z),
                    "p_value": float(p_value),
                    "selection_rate": float(observed / n_l) if n_l > 0 else 0.0,
                    "support": n_l,
                }

        rows = [{"rho": float(rate), "labels": grid[r_idx]} for r_idx, rate in enumerate(rhos)]
        payload["bins"].append({"lo": lo, "hi": hi, "n_tokens": n_bin, "rows": rows})

    total_cells = len(bounds) * len(rhos) * len(names)
    stats = _bonferroni_stats(total_cells)
    for bin_entry in payload["bins"]:
        for row in bin_entry["rows"]:
            for cell in row["labels"]:
                cell["significant"] = cell["p_value"] < stats["threshold"]
    payload.update(stats)

    return payload


def build_selection_logreg_payload(selection_log: SelectionLog | None) -> dict[str, Any]:
    """Per (label, rho) logistic regression: selected ~ is_label + log_freq.

    The class coefficient measures the label's association with selection
    after controlling for token frequency (Section 5, frequency control).
    """
    payload: dict[str, Any] = {"rho": [], "labels": [], "class_coef": {}, "freq_coef": {}}
    if selection_log is None:
        return payload

    labels, log_freqs, selected, _sentence_ids, names = selection_log.arrays()
    if labels.size == 0:
        return payload

    from sklearn.linear_model import LogisticRegression

    rhos = selection_log.rhos
    payload["rho"] = list(rhos)
    payload["labels"] = names

    log_freqs_std = (log_freqs - log_freqs.mean()) / (log_freqs.std() + 1e-8)

    for li, name in enumerate(names):
        is_l = (labels == li).astype(np.float32)
        class_coefs: list[float | None] = []
        freq_coefs: list[float | None] = []
        X = np.stack([is_l, log_freqs_std], axis=1)
        for r in range(len(rhos)):
            y = selected[r]
            if y.all() or not y.any() or not is_l.any():
                class_coefs.append(None)
                freq_coefs.append(None)
                continue
            clf = LogisticRegression(max_iter=200)
            clf.fit(X, y)
            class_coefs.append(float(clf.coef_[0][0]))
            freq_coefs.append(float(clf.coef_[0][1]))
        payload["class_coef"][name] = class_coefs
        payload["freq_coef"][name] = freq_coefs

    return payload


def _parse_rho_curve(values: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    items = sorted(((float(k), float(v)) for k, v in values.items()), key=lambda kv: kv[0])
    return [k for k, _ in items], [v for _, v in items]


def _build_spearman_curves_payload(stsb: Mapping[str, Any] | None) -> dict[str, Any]:
    """STS-B sufficiency-test curves (src/retrival_fun.py's run_stsb_sweep
    output): selector vs random-baseline Spearman correlation per rho, plus
    the unfiltered full-sentence baseline as a constant reference line."""
    payload: dict[str, Any] = {"rho": [], "curves": {}, "baseline": None}
    if not isinstance(stsb, Mapping):
        return payload

    ours_values = stsb.get("ours_by_rho")
    random_values = stsb.get("random_by_rho")
    if isinstance(ours_values, Mapping) and isinstance(random_values, Mapping):
        rho_ours, ours_curve = _parse_rho_curve(ours_values)
        rho_random, random_curve = _parse_rho_curve(random_values)
        if rho_ours and rho_ours == rho_random:
            payload["rho"] = rho_ours
            payload["curves"] = {
                "selector": ours_curve,
                "random": random_curve,
            }
            if "base" in stsb:
                try:
                    payload["baseline"] = {
                        "kind": "constant",
                        "label": "full sentence (no selection)",
                        "value": float(stsb["base"]),
                    }
                except (TypeError, ValueError):
                    payload["baseline"] = None
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_eval_artifacts(
    counts_pred: Sequence[Any] | None,
    counts_gold: Sequence[Any] | None,
    rhos: Sequence[float] | None,
    label_order: Sequence[str] | None = None,
    selection_log: SelectionLog | None = None,
    selection_rate_out_path: str | Path = _DEFAULT_SELECTION_RATE_CURVES_PATH,
    pvalue_out_path: str | Path = _DEFAULT_PVALUE_CURVES_PATH,
    effect_size_out_path: str | Path = _DEFAULT_EFFECT_SIZE_CURVES_PATH,
    exact_test_summary_out_path: str | Path = _DEFAULT_EXACT_TEST_SUMMARY_PATH,
    freq_binned_out_path: str | Path = _DEFAULT_FREQ_BINNED_EFFECTS_PATH,
    selection_logreg_out_path: str | Path = _DEFAULT_SELECTION_LOGREG_PATH,
    stsb: Mapping[str, Any] | None = None,
    spearman_out_path: str | Path = _DEFAULT_SPEARMAN_CURVES_PATH,
) -> dict[str, Path]:
    selections = _build_selections_payload(counts_pred, counts_gold, rhos)
    exact_test = _build_exact_test_payload(counts_pred, rhos, selection_log, label_order)

    selection_rate_payload = _build_selection_rate_curves_payload(selections)
    pvalue_payload = _build_pvalue_curves_payload(exact_test)
    effect_size_payload = _build_effect_size_curves_payload(exact_test)

    selection_rate_path = Path(selection_rate_out_path)
    pvalue_path = Path(pvalue_out_path)
    effect_size_path = Path(effect_size_out_path)

    _write_json(selection_rate_path, selection_rate_payload)
    _write_json(pvalue_path, pvalue_payload)
    _write_json(effect_size_path, effect_size_payload)

    # z is already signed (positive = over-selected, negative = under-selected), so
    # the heatmap plot reads the same effect_size artifact under a second logical key
    # instead of writing a duplicate file with identical content.
    paths = {
        "selection_rate": selection_rate_path,
        "pvalue": pvalue_path,
        "effect_size": effect_size_path,
        "signed_effect": effect_size_path,
    }

    if exact_test is not None:
        exact_test_summary_path = Path(exact_test_summary_out_path)
        _write_json(exact_test_summary_path, exact_test)
        paths["exact_test_summary"] = exact_test_summary_path

    if selection_log is not None:
        freq_binned_path = Path(freq_binned_out_path)
        selection_logreg_path = Path(selection_logreg_out_path)
        _write_json(freq_binned_path, build_freq_binned_effects_payload(selection_log))
        _write_json(selection_logreg_path, build_selection_logreg_payload(selection_log))
        paths["freq_binned_effects"] = freq_binned_path
        paths["selection_logreg"] = selection_logreg_path

    if stsb is not None:
        spearman_payload = _build_spearman_curves_payload(stsb)
        spearman_path = Path(spearman_out_path)
        _write_json(spearman_path, spearman_payload)
        paths["spearman"] = spearman_path

    return paths
