import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.models import RationaleSelectorModel  # noqa: E402


def _resolve_config(signature: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = ROOT / "outputs" / "xps" / signature / ".hydra" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    return path


def _load_checkpoint(signature: str, explicit: str | None, label: str) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = ROOT / "outputs" / "xps" / signature / f"{label}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return path


def _trim_inputs(row):
    attention = torch.tensor(row["attention_mask"], dtype=torch.long)
    length = int(attention.sum().item())
    input_ids = torch.tensor(row["input_ids"], dtype=torch.long)[:length]
    embeddings = torch.tensor(row["embeddings"], dtype=torch.float32)[:length]
    mask = attention[:length]
    return input_ids, embeddings, mask


def _filter_tokens(tokens, gates):
    specials = {"[CLS]", "[SEP]", "[PAD]"}
    filtered_tokens = []
    filtered_gates = []
    for tok, gate in zip(tokens, gates):
        if tok in specials:
            continue
        filtered_tokens.append(tok)
        filtered_gates.append(gate)
    return filtered_tokens, filtered_gates


def _highlight(tokens, gates, threshold):
    pieces = []
    span = []
    for tok, gate in zip(tokens, gates):
        if gate >= threshold:
            span.append(tok)
        else:
            if span:
                pieces.append(f"[[{' '.join(span)}]]")
                span.clear()
            pieces.append(tok)
    if span:
        pieces.append(f"[[{' '.join(span)}]]")
    return " ".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="Analyze rationale selections for a trained RatCon model.")
    parser.add_argument("--xp-signature", required=True, help="Experiment signature under outputs/xps/<sig>.")
    parser.add_argument("--config", default=None, help="Optional explicit Hydra config path.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--dataset", default="data/wikiann_en_validation.pt", help="Path to cached dataset directory.")
    parser.add_argument("--output", default="outputs/rationale_analysis.jsonl", help="Where to save JSONL results.")
    parser.add_argument("--model-label", default="model", help="Checkpoint label to load (default: model.pth).")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples to process.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Gate threshold for rationale spans.")
    args = parser.parse_args()

    cfg = OmegaConf.load(_resolve_config(args.xp_signature, args.config))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = RationaleSelectorModel(cfg.model).to(device)
    ckpt_path = _load_checkpoint(args.xp_signature, args.checkpoint, args.model_label)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.sbert_name, use_fast=True)
    dataset = load_from_disk(args.dataset)
    total = args.limit if args.limit is not None else len(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        iterator = tqdm(enumerate(dataset), total=total, desc="Analyzing rationales")
        for idx, row in iterator:
            if idx >= total:
                break

            input_ids, embeddings, mask = _trim_inputs(row)
            embeddings = embeddings.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(embeddings, mask)
            gates = out["gates"].squeeze(0).detach().cpu().tolist()

            token_list = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            token_list, gate_list = _filter_tokens(token_list, gates)
            sentence = tokenizer.convert_tokens_to_string(token_list)
            highlight = _highlight(token_list, gate_list, args.threshold)

            rationales = [
                {"position": pos, "token": tok, "gate": gate}
                for pos, (tok, gate) in enumerate(zip(token_list, gate_list))
                if gate >= args.threshold
            ]

            record = {
                "index": idx,
                "sentence": sentence,
                "tokens": token_list,
                "gates": gate_list,
                "threshold": args.threshold,
                "rationales": rationales,
                "highlight": highlight,
            }
            writer.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
