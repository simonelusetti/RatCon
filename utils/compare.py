import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List


XPS_DIR = Path("./outputs/xps")
SIGNATURES = {"17e9488d", "db0ae496", "830d0ea8"}
REL_FILE = Path("selections/eval_epoch_030.json")


# -----------------------------
# IO
# -----------------------------

def load_selections(path: Path) -> Tuple[List[List[str]], List[List[float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["tokens"], obj["selected"]


# -----------------------------
# Stats / Metrics
# -----------------------------

def aggregate_counts(
    all_tokens: List[List[str]],
    all_selected: List[List[float]],
) -> tuple[dict[str, int], dict[str, float], int, float]:
    """
    n[t] = total occurrences of token type t
    k[t] = total selected mass of token type t (sum of s for all occurrences)
    N    = total tokens
    K    = total selected mass
    """
    n, k, N, K = defaultdict(int), defaultdict(float), 0, 0.0

    for toks, sels in zip(all_tokens, all_selected):
        for t, s in zip(toks, sels):
            n[t] += 1
            k[t] += float(s)
            N += 1
            K += float(s)

    return n, k, N, K


def weighted_l1_rate_drift(
    statsA: tuple[dict[str, int], dict[str, float], int, float],
    statsB: tuple[dict[str, int], dict[str, float], int, float],
) -> float:
    nA, kA, _, _ = statsA
    nB, kB, _, _ = statsB

    eps = 1e-12
    vocab = set(nA) | set(nB)
    total_exposure = sum(nA.get(t, 0) + nB.get(t, 0) for t in vocab) + eps

    drift = 0.0
    for t in vocab:
        na, nb = nA.get(t, 0), nB.get(t, 0)
        if na + nb == 0:
            continue

        w = (na + nb) / total_exposure
        pa = (kA.get(t, 0.0) + eps) / (na + 2 * eps) if na > 0 else 0.0
        pb = (kB.get(t, 0.0) + eps) / (nb + 2 * eps) if nb > 0 else 0.0
        drift += w * abs(pb - pa)

    return drift


def js_distance_selected_mass(
    statsA: tuple[dict[str, int], dict[str, float], int, float],
    statsB: tuple[dict[str, int], dict[str, float], int, float],
) -> float:
    _, kA, _, KA = statsA
    _, kB, _, KB = statsB

    vocab = set(kA) | set(kB)
    V = max(1, len(vocab))
    alpha = 1e-8

    def q(counter: dict[str, float], totalK: float) -> dict[str, float]:
        denom = totalK + alpha * V
        return {t: (counter.get(t, 0.0) + alpha) / denom for t in vocab}

    qA = q(kA, KA)
    qB = q(kB, KB)
    m = {t: 0.5 * (qA[t] + qB[t]) for t in vocab}

    def kl(p: dict[str, float], qd: dict[str, float]) -> float:
        return sum(p[t] * math.log(p[t] / qd[t]) for t in vocab)

    js = 0.5 * kl(qA, m) + 0.5 * kl(qB, m)
    return math.sqrt(max(js, 0.0))


def average_sentence_metrics_from_arrays(
    tokA: List[List[str]],
    selA: List[List[float]],
    tokB: List[List[str]],
    selB: List[List[float]],
) -> dict[str, float | int]:
    assert len(tokA) == len(tokB), "Mismatched number of samples"

    wl1_sum = 0.0
    jsd_sum = 0.0
    weight_sum = 0.0

    for tA, sA, tB, sB in zip(tokA, selA, tokB, selB):
        assert tA == tB, "mismatch in the base tokens"

        if not tA:
            continue

        statsA = aggregate_counts([tA], [sA])
        statsB = aggregate_counts([tB], [sB])

        wl1 = weighted_l1_rate_drift(statsA, statsB)
        jsd = js_distance_selected_mass(statsA, statsB)

        w = len(tA)
        wl1_sum += w * wl1
        jsd_sum += w * jsd
        weight_sum += w

    return {
        "avg_weighted_l1": wl1_sum / max(1.0, weight_sum),
        "avg_js_distance": jsd_sum / max(1.0, weight_sum),
        "num_samples": len(tokA),
    }


def average_sentence_metrics(pathA: Path, pathB: Path) -> dict[str, float | int]:
    tokA, selA = load_selections(pathA)
    tokB, selB = load_selections(pathB)
    return average_sentence_metrics_from_arrays(tokA, selA, tokB, selB)


def compare_pair(pathA: Path, pathB: Path) -> str:
    avg = average_sentence_metrics(pathA, pathB)
    return (
        f"Avg sentence-wise weighted L1 drift: {avg['avg_weighted_l1']:.6f}\n"
        f"Avg sentence-wise JS distance:       {avg['avg_js_distance']:.6f}\n"
        f"Num samples:                         {avg['num_samples']}\n"
    )


# -----------------------------
# Baseline: select most common tokens
# -----------------------------

def compute_selection_rate(all_tokens: List[List[str]], all_selected: List[List[float]]) -> float:
    _, _, N, K = aggregate_counts(all_tokens, all_selected)
    return float(K) / max(1.0, float(N))


def build_common_token_baseline_selected(
    all_tokens: List[List[str]],
    target_rate: float,
) -> List[List[int]]:
    """
    Build a deterministic baseline selector that selects the most frequent tokens in the dataset,
    choosing up to the same *global* selection rate as a model.

    Strategy:
    - Compute global token frequencies.
    - Choose token types in descending frequency.
    - Include entire token types while their total occurrences fit in the remaining budget.
    - For the boundary token type, select only as many occurrences as needed (deterministically
      in dataset order) to match the target selection count exactly.

    Output: baseline_selected with same shape as all_tokens, entries in {0,1}.
    """
    # Count token frequencies over the dataset
    freq = defaultdict(int)
    total_tokens = 0
    for sent in all_tokens:
        total_tokens += len(sent)
        for t in sent:
            freq[t] += 1

    if total_tokens == 0:
        return [[] for _ in all_tokens]

    # Convert target_rate to an integer selection budget
    target_rate = max(0.0, min(1.0, float(target_rate)))
    target_K = int(round(target_rate * total_tokens))
    target_K = max(0, min(total_tokens, target_K))

    # Sort token types by frequency (desc), then token string for stability
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    full_select = set()
    partial_token = None
    partial_remaining = 0

    remaining = target_K
    for t, c in items:
        if remaining <= 0:
            break
        if c <= remaining:
            full_select.add(t)
            remaining -= c
        else:
            partial_token = t
            partial_remaining = remaining
            remaining = 0
            break

    # Build per-occurrence selections deterministically in dataset order
    baseline_selected: List[List[int]] = []
    rem = partial_remaining

    for sent in all_tokens:
        out = []
        for t in sent:
            if t in full_select:
                out.append(1)
            elif partial_token is not None and t == partial_token and rem > 0:
                out.append(1)
                rem -= 1
            else:
                out.append(0)
        baseline_selected.append(out)

    return baseline_selected


def compare_model_to_common_baseline(path_model: Path) -> str:
    tokM, selM = load_selections(path_model)
    rate = compute_selection_rate(tokM, selM)
    selB = build_common_token_baseline_selected(tokM, rate)

    avg = average_sentence_metrics_from_arrays(tokM, selM, tokM, selB)

    # actual baseline rate (due to rounding)
    _, _, N, Kb = aggregate_counts(tokM, selB)
    baseline_rate = float(Kb) / max(1.0, float(N))

    return (
        f"Target selection rate (model):       {rate:.6f}\n"
        f"Actual selection rate (baseline):    {baseline_rate:.6f}\n"
        f"Avg sentence-wise weighted L1 drift: {avg['avg_weighted_l1']:.6f}\n"
        f"Avg sentence-wise JS distance:       {avg['avg_js_distance']:.6f}\n"
        f"Num samples:                         {avg['num_samples']}\n"
    )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sentence-wise averaged comparison of token selection behavior, plus common-token baseline."
    )
    ap.add_argument("--xps_dir", type=Path, default=XPS_DIR)
    ap.add_argument("--rel_file", type=Path, default=REL_FILE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--skip_model_pairs",
        action="store_true",
        help="If set, only compares each model vs the common-token baseline.",
    )
    args = ap.parse_args()

    sigs = sorted(SIGNATURES)

    if args.out is None:
        args.out = Path(f"./utils/compare/{'-'.join(sigs)}.txt")

    blocks: List[str] = []

    # Pairwise model comparisons (as before)
    if not args.skip_model_pairs:
        pairs = [(sigs[i], sigs[j]) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]

        for a, b in pairs:
            pathA = args.xps_dir / a / args.rel_file
            pathB = args.xps_dir / b / args.rel_file
            assert pathA.exists() and pathB.exists(), f"Missing selections for {a}, {b}"

            header = (
                f"\n{'='*90}\n"
                f"PAIR: {a}  vs  {b}\n"
                f"{'='*90}\n"
            )

            block = header + compare_pair(pathA, pathB)
            print(block)
            blocks.append(block)

    # Model vs common-token baseline
    for s in sigs:
        pathM = args.xps_dir / s / args.rel_file
        assert pathM.exists(), f"Missing selections for {s}"

        header = (
            f"\n{'='*90}\n"
            f"BASELINE: {s}  vs  COMMON-TOKENS@same-rate\n"
            f"{'='*90}\n"
        )
        block = header + compare_model_to_common_baseline(pathM)
        print(block)
        blocks.append(block)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(blocks), encoding="utf-8")

    print(f"\nSaved comparisons to: {args.out}")


if __name__ == "__main__":
    main()
