from __future__ import annotations

import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Base folder where runs live
XPS_DIR = Path("../outputs/xps")

# Signatures to compare (all pairs)
SIGNATURES = {"bcd37200", "971a34f5", "ad357aeb"}

# Which selections file to use inside each run
REL_FILE = Path("selections/eval_epoch_030.jsonl")


def flatten_batch(batch_tokens, batch_selected):
    flat_tokens: List[str] = []
    flat_selected: List[Any] = []

    for sent_toks, sent_sel in zip(batch_tokens, batch_selected):
        assert len(sent_toks) == len(sent_sel), "Sentence length mismatch"
        flat_tokens.extend(sent_toks)
        flat_selected.extend(sent_sel)

    assert len(flat_tokens) == len(flat_selected), "Flattened length mismatch"
    assert all(isinstance(t, str) for t in flat_tokens), "Non-string token found"
    assert all(isinstance(s, int) for s in flat_selected), "Non-numeric selection found"

    return flat_tokens, flat_selected


def load_tokens_selected(path: Path) -> Tuple[List[List[str]], List[List[float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "tokens" in obj and "selected" in obj:
        return obj["tokens"], obj["selected"]
    raise ValueError(f"Unrecognized format in {path}")

def aggregate_counts(all_tokens, all_selected):
    n, k, N, K = defaultdict(int), defaultdict(int), 0, 0

    for batch_toks, batch_sel in zip(all_tokens, all_selected):
        toks, sels = flatten_batch(batch_toks, batch_sel)
        for t, s in zip(toks, sels):
            n[t], k[t], N, K = n[t] + 1, k[t] + s, N + 1, K + s

    return n, k, N, K


def weighted_l1_rate_drift(
    statsA: Tuple[int, int, int, int],
    statsB: Tuple[int, int, int, int],
) -> float:
    nA, kA, _, _ = statsA
    nB, kB, _, _ = statsB
    eps: float = 1e-12
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
    statsA: Tuple[int, int, int, int],
    statsB: Tuple[int, int, int, int],
) -> float:
    _, kA, _, KA = statsA
    _, kB, _, KB = statsB
    vocab = set(kA) | set(kB)
    V = max(1, len(vocab))
    alpha: float = 1e-8

    def q(counter, totalK: int) -> Dict[str, float]:
        denom = totalK + alpha * V
        return {t: (counter.get(t, 0) + alpha) / denom for t in vocab}

    qA = q(kA, KA)
    qB = q(kB, KB)
    m = {t: 0.5 * (qA[t] + qB[t]) for t in vocab}

    def kl(p: Dict[str, float], qd: Dict[str, float]) -> float:
        s = 0.0
        for t in vocab:
            pt, qt, s = p[t], qd[t], 
            qt = qd[t]
            s += pt * math.log(pt / qt)
        return s

    js = 0.5 * kl(qA, m) + 0.5 * kl(qB, m)
    return math.sqrt(max(js, 0.0))


def estimate_dirichlet_from_background(statsA: Dict[str, Any], statsB: Dict[str, Any], prior_strength: float) -> Dict[str, float]:
    nA, nB = statsA["n"], statsB["n"]
    vocab = set(nA) | set(nB)
    total = sum(nA.get(t, 0) + nB.get(t, 0) for t in vocab)
    if total <= 0:
        V = max(1, len(vocab))
        return {t: prior_strength / V for t in vocab}
    return {t: prior_strength * (nA.get(t, 0) + nB.get(t, 0)) / total for t in vocab}


def log_odds_zscores(statsA: Dict[str, Any], statsB: Dict[str, Any], prior_strength: float = 1000.0):
    """
    Informative Dirichlet prior log-odds with z-scores on selected-token counts.
    Returns list of (token, delta, z, kA, kB) sorted by |z| desc.
    """
    kA, KA = statsA["k"], statsA["K"]
    kB, KB = statsB["k"], statsB["K"]
    vocab = set(kA) | set(kB)
    if not vocab:
        return []

    alpha_t = estimate_dirichlet_from_background(statsA, statsB, prior_strength)
    alpha0 = sum(alpha_t.values())

    out = []
    for t in vocab:
        a = alpha_t.get(t, 0.0)
        if a <= 0:
            a = prior_strength / len(vocab)

        xA = kA.get(t, 0)
        xB = kB.get(t, 0)

        denomA = (KA - xA) + (alpha0 - a)
        denomB = (KB - xB) + (alpha0 - a)
        if denomA <= 0 or denomB <= 0:
            continue

        delta = math.log((xA + a) / denomA) - math.log((xB + a) / denomB)
        var = 1.0 / (xA + a) + 1.0 / (xB + a)
        z = delta / math.sqrt(var)
        out.append((t, delta, z, xA, xB))

    out.sort(key=lambda x: abs(x[2]), reverse=True)
    return out


def compare_pair(pathA: Path, pathB: Path, min_count: int, topk: int, prior_strength: float) -> str:
    tokA, selA = load_tokens_selected(pathA)
    tokB, selB = load_tokens_selected(pathB)

    statsA = aggregate_counts(tokA, selA)
    statsB = aggregate_counts(tokB, selB)

    wl1 = weighted_l1_rate_drift(statsA, statsB)
    jsd = js_distance_selected_mass(statsA, statsB)
    zlist = log_odds_zscores(statsA, statsB, prior_strength=prior_strength)

    nA, kA = statsA["n"], statsA["k"]
    nB, kB = statsB["n"], statsB["k"]
    vocab = set(nA) | set(nB)

    rows = []
    for t in vocab:
        na, nb = nA.get(t, 0), nB.get(t, 0)
        if na + nb < min_count:
            continue
        pa = (kA.get(t, 0) / na) if na > 0 else 0.0
        pb = (kB.get(t, 0) / nb) if nb > 0 else 0.0
        rows.append((t, pb - pa, pa, pb, na, nb, kA.get(t, 0), kB.get(t, 0)))
    rows.sort(key=lambda x: x[1], reverse=True)

    out = []
    out.append("=== Global distances ===")
    out.append(f"A: {pathA}")
    out.append(f"B: {pathB}")
    out.append(f"File A: total_tokens={statsA['N']} selected_tokens={statsA['K']}")
    out.append(f"File B: total_tokens={statsB['N']} selected_tokens={statsB['K']}")
    out.append(f"Weighted L1 drift over p(t)=P(sel|t): {wl1:.6f}")
    out.append(f"JS distance over q(t) selected-token mass: {jsd:.6f}")
    out.append("")

    out.append(f"=== Top {topk} tokens with largest +Δp (B selects more | token) (min_count={min_count}) ===")
    for t, dp, pa, pb, na, nb, ka, kb in rows[:topk]:
        out.append(f"{t:>15}  Δp={dp:+.4f}  pA={pa:.4f} pB={pb:.4f}  nA={na} nB={nb}  kA={ka} kB={kb}")
    out.append("")

    out.append(f"=== Top {topk} tokens with largest -Δp (A selects more | token) (min_count={min_count}) ===")
    for t, dp, pa, pb, na, nb, ka, kb in rows[-topk:][::-1]:
        out.append(f"{t:>15}  Δp={dp:+.4f}  pA={pa:.4f} pB={pb:.4f}  nA={na} nB={nb}  kA={ka} kB={kb}")
    out.append("")

    out.append(f"=== Top {topk} log-odds z-score shifts on selected-token mass (A higher z => more selected in A) ===")
    shown = 0
    for t, delta, z, ka, kb in zlist:
        if (nA.get(t, 0) + nB.get(t, 0)) < min_count:
            continue
        out.append(f"{t:>15}  z={z:+.3f}  delta={delta:+.3f}  kA={ka} kB={kb}  nA={nA.get(t,0)} nB={nB.get(t,0)}")
        shown += 1
        if shown >= topk:
            break

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Compare token selection preferences between multiple runs (all pairs).")
    ap.add_argument("--xps_dir", type=Path, default=XPS_DIR, help="Base directory containing xp signatures")
    ap.add_argument("--rel_file", type=Path, default=REL_FILE, help="Relative path to selections file within each run")
    ap.add_argument("--min_count", type=int, default=20, help="Min token exposure (nA+nB) for per-token reporting")
    ap.add_argument("--topk", type=int, default=10, help="Top-k tokens to show for shifts")
    ap.add_argument("--prior_strength", type=float, default=1000.0, help="Dirichlet prior strength for log-odds")
    ap.add_argument("--out", type=Path, default=Path("compare_all_pairs.txt"), help="Output text file")
    args = ap.parse_args()

    sigs = sorted(SIGNATURES)
    pairs = [(sigs[i], sigs[j]) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]

    blocks: List[str] = []
    for a, b in pairs:
        pathA = args.xps_dir / a / args.rel_file
        pathB = args.xps_dir / b / args.rel_file

        header = f"\n{'='*90}\nPAIR: {a}  vs  {b}\n{'='*90}\n"
        if not pathA.exists() or not pathB.exists():
            raise FileNotFoundError(f"Missing selections file for pair: {pathA}, {pathB}")
        else:
            block = header + compare_pair(
                pathA, pathB,
                min_count=args.min_count,
                topk=args.topk,
                prior_strength=args.prior_strength,
            )

        print(block)
        blocks.append(block)

    args.out.write_text("".join(blocks), encoding="utf-8")
    print(f"\nSaved all pairwise comparisons to: {args.out.resolve()}")


if __name__ == "__main__":
    main()