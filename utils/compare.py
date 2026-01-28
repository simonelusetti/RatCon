import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict


XPS_DIR = Path("./outputs/xps")
SIGNATURES = {"9c65a034","3f33fce7","e6376605"}
REL_FILE = Path("selections/eval_epoch_001.json")


def load_selections(path: Path) -> Tuple[List[List[str]], List[List[float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["tokens"], obj["selected"]


def aggregate_counts(all_tokens: List[List[str]], all_selected: List[List[float]]) -> tuple[dict[str, int], dict[str, float], int, float]:
    n, k, N, K = defaultdict(int), defaultdict(int), 0, 0

    for toks, sels in zip(all_tokens, all_selected):
        for t, s in zip(toks, sels):
            n[t] += 1
            k[t] += s
            N += 1
            K += s

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
        pa = (kA.get(t, 0) + eps) / (na + 2 * eps) if na > 0 else 0.0
        pb = (kB.get(t, 0) + eps) / (nb + 2 * eps) if nb > 0 else 0.0
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
        return {t: (counter.get(t, 0) + alpha) / denom for t in vocab}

    qA = q(kA, KA)
    qB = q(kB, KB)
    m = {t: 0.5 * (qA[t] + qB[t]) for t in vocab}

    def kl(p: dict[str, float], qd: dict[str, float]) -> float:
        return sum(p[t] * math.log(p[t] / qd[t]) for t in vocab)

    js = 0.5 * kl(qA, m) + 0.5 * kl(qB, m)
    return math.sqrt(max(js, 0.0))


def average_sentence_metrics(pathA: Path, pathB: Path) -> dict[str, float | int]:
    tokA, selA = load_selections(pathA)
    tokB, selB = load_selections(pathB)

    assert len(tokA) == len(tokB), "Mismatched number of samples"

    wl1_sum = 0.0
    jsd_sum = 0.0
    weight_sum = 0.0

    for tA, sA, tB, sB in zip(tokA, selA, tokB, selB):
        assert tA == tB, "mismatch in the base tokens"
        
        if not tA or not tB:
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


def compare_pair(pathA: Path, pathB: Path) -> str:
    avg = average_sentence_metrics(pathA, pathB)

    return (
        f"Avg sentence-wise weighted L1 drift: {avg['avg_weighted_l1']:.6f}\n"
        f"Avg sentence-wise JS distance:       {avg['avg_js_distance']:.6f}\n"
        f"Num samples:                         {avg['num_samples']}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sentence-wise averaged comparison of token selection behavior."
    )
    ap.add_argument("--xps_dir", type=Path, default=XPS_DIR)
    ap.add_argument("--rel_file", type=Path, default=REL_FILE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sigs = sorted(SIGNATURES)
    pairs = [(sigs[i], sigs[j]) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]

    if args.out is None:
        args.out = Path(f"./utils/compare/{'-'.join(sigs)}.txt")

    blocks: List[str] = []

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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(blocks), encoding="utf-8")

    print(f"\nSaved sentence-wise averaged comparisons to: {args.out}")


if __name__ == "__main__":
    main()
