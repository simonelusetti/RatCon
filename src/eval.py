import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency


_CHI_SQUARE_PVALUE_THRESHOLD = 0.05
_CHI2_CRITICAL_DF1_P05 = 3.841458820694124
_DEFAULT_SELECTION_RATE_CURVES_PATH = Path("data") / "selection_rate_curves.json"
_DEFAULT_CHI_SQUARE_CURVES_PATH = Path("data") / "chi_square_curves.json"
_DEFAULT_CRAMERS_V_CURVES_PATH = Path("data") / "cramers_v_curves.json"
_DEFAULT_SPEARMAN_CURVES_PATH = Path("data") / "spearman_curves.json"


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
    v = count_obj.data[label]
    if isinstance(v, float):
        raise TypeError(f"Counts.data[{label!r}] is a float (rate); chi-square needs raw integer counts.")
    return int(v)


def _contingency_2x2_one_vs_rest(pred: Any, gold: Any, label: Any) -> np.ndarray:
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


def _contingency_2x2_vs_negative_label(pred: Any, gold: Any, positive_label: Any, negative_label: Any) -> np.ndarray:
    tp = _get_count(pred, positive_label)
    fn = _get_count(gold, positive_label) - tp
    fp = _get_count(pred, negative_label)
    tn = _get_count(gold, negative_label) - fp

    if fn < 0 or tn < 0:
        raise ValueError(
            "Invalid counts in pos-vs-negative table for "
            f"label={positive_label!r}, negative={negative_label!r}: "
            f"tp={tp}, fn={fn}, fp={fp}, tn={tn}"
        )

    return np.array([[tp, fn], [fp, tn]], dtype=np.int64)


def _chi_square_stats(table_2x2: np.ndarray) -> tuple[float, float, float]:
    if np.any(table_2x2.sum(axis=0) == 0) or np.any(table_2x2.sum(axis=1) == 0):
        return 0.0, 1.0, 0.0

    try:
        chi2, p, _, _ = chi2_contingency(table_2x2, correction=False)
    except ValueError:
        return 0.0, 1.0, 0.0

    n = float(table_2x2.sum())
    v = float(np.sqrt(chi2 / n)) if n > 0 else 0.0
    return float(chi2), float(p), v


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


def _build_chi_square_payload(
    counts_pred: Sequence[Any] | None,
    counts_gold: Sequence[Any] | None,
    rhos: Sequence[float] | None,
) -> dict[str, Any] | None:
    if counts_pred is None or counts_gold is None or rhos is None:
        return None
    if not counts_pred or not counts_gold:
        return {
            "mode": "one_vs_rest",
            "labels": [],
            "rows": [],
        }

    labels = sorted(
        {label for counts in counts_gold for label in counts.data.keys()},
        key=_label_sort_key,
    )

    negative_label: Any | None = None
    try:
        negative_label = _infer_non_entity_label(labels)
    except ValueError:
        negative_label = None

    if negative_label is not None:
        effective_labels = [label for label in labels if label != negative_label]
        mode = "vs_negative_label"
    else:
        effective_labels = labels
        mode = "one_vs_rest"

    rows: list[dict[str, Any]] = []
    for rate, pred, gold in zip(rhos, counts_pred, counts_gold):
        label_rows: list[dict[str, Any]] = []
        for label in effective_labels:
            if negative_label is not None:
                table = _contingency_2x2_vs_negative_label(pred, gold, label, negative_label)
            else:
                table = _contingency_2x2_one_vs_rest(pred, gold, label)
            chi2, p_value, cramers_v = _chi_square_stats(table)
            label_rows.append(
                {
                    "label": str(label),
                    "contingency": [[int(x) for x in row] for row in table.tolist()],
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "cramers_v": float(cramers_v),
                }
            )

        rows.append({"rho": float(rate), "labels": label_rows})

    return {
        "mode": mode,
        "negative_label": None if negative_label is None else str(negative_label),
        "labels": [str(label) for label in effective_labels],
        "rows": rows,
    }


def build_chi_square_payload(
    counts_pred: Sequence[Any],
    counts_gold: Sequence[Any],
    rhos: Sequence[float],
) -> dict[str, Any]:
    payload = _build_chi_square_payload(counts_pred, counts_gold, rhos)
    if payload is None:
        return {
            "mode": "one_vs_rest",
            "labels": [],
            "rows": [],
        }
    return payload


def _parse_rho_curve(values: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    items = sorted(((float(k), float(v)) for k, v in values.items()), key=lambda kv: kv[0])
    return [k for k, _ in items], [v for _, v in items]


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


def _extract_chi_square_curves(chi_square: Mapping[str, Any]) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]], float | None]:
    rows = chi_square.get("rows")
    if not isinstance(rows, list) or not rows:
        return [], {}, {}, None

    rho_values: list[float] = []
    chi_square_curves: dict[str, list[float]] = {}
    cramers_v_curves: dict[str, list[float]] = {}
    cramers_v_baseline: float | None = None

    negative_label_raw = chi_square.get("negative_label")
    negative_label = str(negative_label_raw).strip() if negative_label_raw is not None else None

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
            if negative_label is not None and label.strip() == negative_label:
                continue

            p_value = max(float(item.get("p_value", 1.0)), 1e-300)
            chi_square_curves.setdefault(label, []).append(-math.log10(p_value))
            cramers_v_curves.setdefault(label, []).append(float(item.get("cramers_v", 0.0)))

            if cramers_v_baseline is None:
                contingency = item.get("contingency")
                if isinstance(contingency, list):
                    table = np.asarray(contingency, dtype=float)
                    total = float(table.sum())
                    if total > 0.0:
                        cramers_v_baseline = float(np.sqrt(_CHI2_CRITICAL_DF1_P05 / total))

    return rho_values, chi_square_curves, cramers_v_curves, cramers_v_baseline


def _build_selection_rate_curves_payload(selections: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"rho": [], "curves": {}, "baseline": {"kind": "identity", "label": "y=x"}}
    if not isinstance(selections, Mapping):
        return payload

    rho_values, curves = _extract_selection_curves(selections)
    if rho_values and curves:
        payload["rho"] = rho_values
        payload["curves"] = curves
    return payload


def _build_chi_square_curves_payload(chi_square: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rho": [],
        "curves": {},
        "baseline": {
            "kind": "constant",
            "label": "p=0.05",
            "value": float(-math.log10(_CHI_SQUARE_PVALUE_THRESHOLD)),
        },
    }
    if not isinstance(chi_square, Mapping):
        return payload

    rho_values, chi_curves, _, _ = _extract_chi_square_curves(chi_square)
    if rho_values and chi_curves:
        payload["rho"] = rho_values
        payload["curves"] = chi_curves
    return payload


def _build_cramers_v_curves_payload(chi_square: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"rho": [], "curves": {}, "baseline": None}
    if not isinstance(chi_square, Mapping):
        return payload

    rho_values, _, v_curves, v_baseline = _extract_chi_square_curves(chi_square)
    if rho_values and v_curves:
        payload["rho"] = rho_values
        payload["curves"] = v_curves
    if v_baseline is not None:
        payload["baseline"] = {"kind": "constant", "label": "p=0.05", "value": float(v_baseline)}
    return payload


def _build_spearman_curves_payload(stsb: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"rho": [], "curves": {}, "baseline": None}
    if not isinstance(stsb, Mapping):
        return payload

    ours_values = stsb.get("ours_by_rho", stsb.get("ours"))
    random_values = stsb.get("random_by_rho", stsb.get("random"))
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
                        "label": "baseline",
                        "value": float(stsb.get("base")),
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
    stsb: Mapping[str, Any] | None,
    selection_rate_out_path: str | Path = _DEFAULT_SELECTION_RATE_CURVES_PATH,
    chi_square_out_path: str | Path = _DEFAULT_CHI_SQUARE_CURVES_PATH,
    cramers_v_out_path: str | Path = _DEFAULT_CRAMERS_V_CURVES_PATH,
    spearman_out_path: str | Path = _DEFAULT_SPEARMAN_CURVES_PATH,
) -> dict[str, Path]:
    selections = _build_selections_payload(counts_pred, counts_gold, rhos)
    chi_square = _build_chi_square_payload(counts_pred, counts_gold, rhos)

    selection_rate_payload = _build_selection_rate_curves_payload(selections)
    chi_square_payload = _build_chi_square_curves_payload(chi_square)
    cramers_v_payload = _build_cramers_v_curves_payload(chi_square)
    spearman_payload = _build_spearman_curves_payload(stsb)

    selection_rate_path = Path(selection_rate_out_path)
    chi_square_path = Path(chi_square_out_path)
    cramers_v_path = Path(cramers_v_out_path)
    spearman_path = Path(spearman_out_path)

    _write_json(selection_rate_path, selection_rate_payload)
    _write_json(chi_square_path, chi_square_payload)
    _write_json(cramers_v_path, cramers_v_payload)
    _write_json(spearman_path, spearman_payload)

    return {
        "selection_rate": selection_rate_path,
        "chi_square": chi_square_path,
        "cramers_v": cramers_v_path,
        "spearman": spearman_path,
    }
