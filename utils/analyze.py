import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Iterable

XPS_DIR = Path("./outputs/xps")
REL_FILE = Path("selections/eval_epoch_030.jsonl")

def to01(x: bool | int | float) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x > 0.5)
    raise TypeError(f"Invalid selection value: {x} ({type(x)})")


def is_jsonl(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".ndjson"}


def load_tokens_selected(path: Path) -> Tuple[List[List[str]], List[List[int | float | bool]]]:
    if is_jsonl(path):
        tokens_all, sel_all = [], []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tokens_all.append(obj["tokens"])
                sel_all.append(obj["selected"])
        return tokens_all, sel_all

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return obj["tokens"], obj["selected"]

    raise ValueError(f"Unrecognized format in {path}")

def aggregate_to_words(tokens: List[str], selected: List[int | float | bool]) -> Tuple[List[str], List[int]]:
    assert len(tokens) == len(selected)

    words: List[str] = []
    word_sel: List[int] = []

    cur_word = ""
    cur_selected = 0

    def flush() -> None:
        nonlocal cur_word, cur_selected
        if cur_word:
            words.append(cur_word)
            word_sel.append(cur_selected)
        cur_word = ""
        cur_selected = 0

    for tok, sel in zip(tokens, selected):
        s = to01(sel)

        if tok.startswith("▁"):
            flush()
            cur_word = tok[1:]
            cur_selected = s

        elif tok.startswith("##"):
            if not cur_word:
                cur_word = tok[2:]
                cur_selected = s
            else:
                cur_word += tok[2:]
                cur_selected |= s

        else:
            flush()
            cur_word = tok
            cur_selected = s

    flush()
    return words, word_sel


def sentence_stream(
    all_tokens: List[List[List[str]]],
    all_selected: List[List[List[int | float | bool]]],
    lowercase: bool,
) -> Iterable[Tuple[List[str], List[int]]]:
    for batch_tokens, batch_selected in zip(all_tokens, all_selected):
        for sent_toks, sent_sel in zip(batch_tokens, batch_selected):
            words, sels = aggregate_to_words(sent_toks, sent_sel)
            if lowercase:
                words = [w.lower() for w in words]
            if words:
                yield words, sels

def log_odds_ratio(a: int, b: int, c: int, d: int, alpha: float = 0.5) -> float:
    return math.log((a + alpha) * (d + alpha)) - math.log((b + alpha) * (c + alpha))


def build_vocab_and_sentences(
    sent_stream: Iterable[Tuple[List[str], List[int]]],
    max_vocab: int,
    min_word_count: int,
) -> tuple[list[tuple[list[str], list[int]]], dict[str, int], set[str]]:
    sents = []
    counts = defaultdict(int)

    for words, sels in sent_stream:
        sents.append((words, sels))
        for w in words:
            counts[w] += 1

    vocab = [
        w for w, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if c >= min_word_count
    ][:max_vocab]

    return sents, counts, set(vocab)


def compute_context_dependence(
    sents: List[Tuple[List[str], List[int]]],
    vocab_set: set[str],
    min_occ: int,
    min_co: int,
    alpha: float,
    topk_ctx: int,
) -> Dict[str, dict]:
    occ = defaultdict(int)
    sel = defaultdict(int)

    a = defaultdict(lambda: defaultdict(int))
    b = defaultdict(lambda: defaultdict(int))
    c = defaultdict(lambda: defaultdict(int))
    d = defaultdict(lambda: defaultdict(int))

    for words, sels in sents:
        sent_set = set(words)

        for t, y in zip(words, sels):
            occ[t] += 1
            sel[t] += y

            ctx = sent_set - {t}
            ctx = ctx & vocab_set

            for u in ctx:
                if y:
                    a[t][u] += 1
                else:
                    b[t][u] += 1

    results = {}

    for t in occ:
        if occ[t] < min_occ:
            continue

        total_sel = sel[t]
        total_not = occ[t] - total_sel
        p_base = total_sel / occ[t]

        rows = []

        for u in set(a[t]) | set(b[t]):
            pres_sel = a[t].get(u, 0)
            pres_not = b[t].get(u, 0)
            pres = pres_sel + pres_not

            abs_sel = total_sel - pres_sel
            abs_not = total_not - pres_not
            absn = abs_sel + abs_not

            if pres < min_co or absn < min_co:
                continue

            p_pres = pres_sel / pres
            p_abs = abs_sel / absn
            delta = p_pres - p_abs
            lor = log_odds_ratio(pres_sel, pres_not, abs_sel, abs_not, alpha)

            rows.append({
                "u": u,
                "delta": delta,
                "lor": lor,
                "pres": pres,
                "abs": absn,
                "p_pres": p_pres,
                "p_abs": p_abs,
                "a": pres_sel, "b": pres_not, "c": abs_sel, "d": abs_not,
            })

        cd = sum((r["pres"] / occ[t]) * abs(r["delta"]) for r in rows)

        rows_delta = sorted(rows, key=lambda r: r["delta"], reverse=True)
        rows_lor = sorted(rows, key=lambda r: r["lor"], reverse=True)

        results[t] = {
            "occ": occ[t],
            "p_base": p_base,
            "cd": cd,
            "enforce_delta": rows_delta[:topk_ctx],
            "suppress_delta": list(reversed(rows_delta[-topk_ctx:])),
            "enforce_lor": rows_lor[:topk_ctx],
            "suppress_lor": list(reversed(rows_lor[-topk_ctx:])),
        }

    return results


def format_report(results: Dict[str, dict], top_tokens: int, topk_ctx: int) -> str:
    ranked = sorted(results.items(), key=lambda kv: kv[1]["cd"], reverse=True)
    out = []
    out.append("=== WORD-LEVEL CONTEXT DEPENDENCE REPORT ===\n")

    for t, info in ranked[:top_tokens]:
        out.append("-" * 90)
        out.append(f"WORD: {t}")
        out.append(f"occ={info['occ']}  p_base={info['p_base']:.4f}  CD={info['cd']:.6f}\n")

        out.append("Top enforcers (Δ):")
        for r in info["enforce_delta"]:
            out.append(f"  {r['u']:>15}  Δ={r['delta']:+.4f}  pres={r['pres']} abs={r['abs']}")

        out.append("Top suppressors (Δ):")
        for r in info["suppress_delta"]:
            out.append(f"  {r['u']:>15}  Δ={r['delta']:+.4f}  pres={r['pres']} abs={r['abs']}")

        out.append("Top enforcers (LOR):")
        for r in info["enforce_lor"]:
            out.append(f"  {r['u']:>15}  LOR={r['lor']:+.3f}")

        out.append("Top suppressors (LOR):")
        for r in info["suppress_lor"]:
            out.append(f"  {r['u']:>15}  LOR={r['lor']:+.3f}")

        out.append("")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser("Single-run word context dependence (sentence-level)")
    ap.add_argument("--sig", required=True)
    ap.add_argument("--xps_dir", type=Path, default=XPS_DIR)
    ap.add_argument("--rel_file", type=Path, default=REL_FILE)
    ap.add_argument("--out", type=Path, default=Path("context_dependence.txt"))

    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--max_vocab", type=int, default=5000)
    ap.add_argument("--min_word_count", type=int, default=20)
    ap.add_argument("--min_occ", type=int, default=50)
    ap.add_argument("--min_co", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top_tokens", type=int, default=50)
    ap.add_argument("--topk_ctx", type=int, default=10)

    args = ap.parse_args()

    path = args.xps_dir / args.sig / args.rel_file
    if not path.exists():
        raise FileNotFoundError(path)

    all_tokens, all_selected = load_tokens_selected(path)

    sent_stream = sentence_stream(all_tokens, all_selected, args.lowercase)
    sents, counts, vocab_set = build_vocab_and_sentences(
        sent_stream,
        max_vocab=args.max_vocab,
        min_word_count=args.min_word_count,
    )

    results = compute_context_dependence(
        sents=sents,
        vocab_set=vocab_set,
        min_occ=args.min_occ,
        min_co=args.min_co,
        alpha=args.alpha,
        topk_ctx=args.topk_ctx,
    )

    report = format_report(results, args.top_tokens, args.topk_ctx)
    print(report)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {args.out.resolve()}")


if __name__ == "__main__":
    main()
