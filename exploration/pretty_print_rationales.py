import argparse
import json
from pathlib import Path

import colorama
from colorama import Fore, Style

COLORS = [
    Fore.RED,
    Fore.LIGHTRED_EX,
    Fore.LIGHTMAGENTA_EX,
    Fore.YELLOW,
    Fore.GREEN,
    Fore.CYAN,
]


def gate_to_color(gate: float, threshold: float, mono: bool) -> str:
    if gate < threshold:
        return Style.DIM
    if mono:
        return Fore.LIGHTGREEN_EX
    if gate >= 0.9:
        return COLORS[0]
    if gate >= 0.75:
        return COLORS[1]
    if gate >= 0.6:
        return COLORS[2]
    if gate >= 0.5:
        return COLORS[3]
    if gate >= 0.4:
        return COLORS[4]
    return COLORS[5]


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def render_record(record, threshold: float, mono: bool):
    tokens = record.get("tokens") or []
    gates = record.get("gates") or []
    pieces = []
    for idx, token in enumerate(tokens):
        gate = gates[idx] if idx < len(gates) else 0.0
        color = gate_to_color(gate, threshold, mono)
        text = token or f"[{idx}]"
        if mono:
            pieces.append(f"{color}{text}{Style.RESET_ALL}")
        else:
            pieces.append(f"{color}{text}{Style.RESET_ALL}({gate:.2f})")
    return " ".join(pieces) if pieces else "(no tokens)"


def main():
    parser = argparse.ArgumentParser(description="Pretty-print rationale selections from analyze_rationales.py")
    parser.add_argument("--input", default="outputs/rationale_analysis.jsonl", help="Path to JSONL file to display.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to show.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Gate threshold for emphasis.")
    parser.add_argument("--mono", action="store_true", help="Use a single highlight color for gates above the threshold.")
    args = parser.parse_args()

    colorama.init(autoreset=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    for idx, record in enumerate(load_records(input_path)):
        if args.limit is not None and idx >= args.limit:
            break

        sentence = record.get("sentence") or "(no decoded sentence)"
        print(f"\nSentence #{record.get('index', idx)}: {sentence}")
        print(render_record(record, args.threshold, args.mono))


if __name__ == "__main__":
    main()
