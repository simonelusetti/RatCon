import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ratcon.data import collate, get_dataset  # noqa: E402
from ratcon.models import RationaleSelectorModel  # noqa: E402

LOGGER = logging.getLogger("cluster_drift")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Measure cluster-level sentence drift on CoNLL2003.")
    parser.add_argument("--exp-path", type=str, default=None, help="Hydra experiment directory to load.")
    parser.add_argument("--exp-signature", type=str, default=None, help="Experiment signature to locate under search root.")
    parser.add_argument("--search-root", type=str, default="outputs", help="Root directory to search when using --exp-signature.")
    parser.add_argument("--model-file", type=str, default=None, help="Override checkpoint file name (default: auto).")
    parser.add_argument("--num-clusters", type=int, default=None, help="Override number of clusters.")
    parser.add_argument("--cluster-threshold", type=float, default=None, help="Override cluster gate threshold.")
    parser.add_argument("--fit-subset", type=float, default=None, help="Fraction or count of fit dataset to use.")
    parser.add_argument("--eval-subset", type=float, default=None, help="Fraction or count of eval dataset to probe.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for cluster fitting.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length when encoding text.")
    parser.add_argument("--samples-per-cluster", type=int, default=3, help="Number of random baselines per cluster.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device.")
    parser.add_argument("--output", type=str, default="exploration/cluster_sentence_drift.jsonl", help="Path to save raw measurements.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_experiment_config(exp_path):
    exp_dir = Path(exp_path)
    candidates = [
        exp_dir / ".hydra" / "config.yaml",
        exp_dir / "config.yaml",
    ]
    for cfg_path in candidates:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
    raise FileNotFoundError(f"Could not find config.yaml in {exp_dir}")


def find_experiment_dir(signature, search_root):
    search_root = Path(search_root).expanduser()
    candidates = []
    if search_root.is_absolute():
        candidates.append(search_root)
    else:
        candidates.append((CURRENT_DIR / search_root).resolve())
        candidates.append((PROJECT_ROOT / search_root).resolve())
        candidates.append(search_root.resolve())

    root = None
    for candidate in candidates:
        if candidate.exists():
            root = candidate
            break
    if root is None:
        root = candidates[-1]
    if not root.exists():
        raise FileNotFoundError(f"Search root {root} does not exist.")

    direct = root / "xps" / signature
    if direct.exists():
        return direct

    matches = []
    seen = set()
    for cfg_path in root.rglob("config.yaml"):
        if cfg_path.parent.name != ".hydra":
            continue
        xp_dir = cfg_path.parent.parent
        if xp_dir in seen:
            continue
        if signature in xp_dir.name:
            matches.append(xp_dir)
            seen.add(xp_dir)
            continue
        train_log = xp_dir / "train.log"
        if train_log.exists():
            with train_log.open("r", encoding="utf-8", errors="ignore") as handle:
                head = handle.read(4096)
            if signature in head:
                matches.append(xp_dir)
                seen.add(xp_dir)

    if not matches:
        raise FileNotFoundError(f"No experiment directory matching signature '{signature}' under {root}.")
    if len(matches) > 1:
        pretty = "\n  ".join(str(p) for p in matches)
        raise ValueError(f"Multiple experiment directories match signature '{signature}':\n  {pretty}")
    return matches[0]


def build_model_cfg(cfg_dict, args):
    model_dict = dict(cfg_dict.get("model", {}))
    model_dict.setdefault("sbert_name", "sentence-transformers/all-MiniLM-L6-v2")
    model_dict.setdefault("attention_augment", False)

    loss_defaults = {
        "l_comp": 0.1,
        "l_s": 0.01,
        "l_tv": 10.0,
        "tau": 0.07,
        "use_null_target": False,
    }
    model_dict.setdefault("loss", {})
    for key, value in loss_defaults.items():
        model_dict["loss"].setdefault(key, value)

    cluster_defaults = {
        "use": True,
        "show_details": False,
        "proposal_thresh": 0.5,
        "num_clusters": 3,
        "max_tokens": -1,
        "iters": 25,
        "tol": 1e-4,
        "seed": args.seed,
    }
    model_dict.setdefault("clustering", {})
    for key, value in cluster_defaults.items():
        model_dict["clustering"].setdefault(key, value)

    if args.num_clusters is not None:
        model_dict["clustering"]["num_clusters"] = args.num_clusters
    if args.cluster_threshold is not None:
        model_dict["clustering"]["proposal_thresh"] = args.cluster_threshold
    model_dict["clustering"]["max_tokens"] = -1  # always use all tokens
    model_dict["clustering"]["use"] = True

    if "fourier" not in model_dict:
        model_dict["fourier"] = {"use": False, "mode": "lowpass", "keep_ratio": 0.5}

    return to_namespace(model_dict)


def resolve_model_path(exp_path, preferred):
    exp_dir = Path(exp_path)
    candidates = []
    if preferred:
        candidates.append(exp_dir / preferred)
    candidates.extend(
        [
            exp_dir / "model.pth",
            exp_dir / "model.pt",
            exp_dir / "model.bin",
            exp_dir / "model1.pth",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model checkpoint found in {exp_dir}")


def collate_with_tokens(batch):
    output = collate(batch)
    output["tokens"] = [item.get("tokens", []) for item in batch]
    return output


def cosine_shift(anchor, variant):
    score = torch.nn.functional.cosine_similarity(anchor.unsqueeze(0), variant.unsqueeze(0), dim=-1)
    return 1.0 - score.item()


def encode_sentence(model, tokens, device):
    text = " ".join(tokens).strip()
    if not text:
        return None
    device_name = device.type if device.type == "cuda" else "cpu"
    return model.sbert.encode(
        text,
        convert_to_tensor=True,
        device=device_name,
        show_progress_bar=False,
    )


def collect_cluster_words(raw_tokens, tokenizer, gates, assignments, attention_mask, threshold, max_length):
    encoding = tokenizer(
        raw_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )
    word_ids = encoding.word_ids()
    if word_ids is None:
        return {}

    word_info = [[] for _ in raw_tokens]
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if attention_mask[token_idx].item() == 0:
            continue
        cluster_id = assignments[token_idx].item()
        if cluster_id < 0:
            continue
        gate_val = float(gates[token_idx].item())
        word_info[word_idx].append((cluster_id, gate_val))

    cluster_words = defaultdict(list)
    for word_idx, info in enumerate(word_info):
        if not info:
            continue
        max_gate = max(g for _, g in info)
        if max_gate < threshold:
            continue
        scores = defaultdict(float)
        for cluster_id, gate_val in info:
            scores[cluster_id] += gate_val
        cluster_id = max(scores.items(), key=lambda x: x[1])[0]
        cluster_words[cluster_id].append(word_idx)
    return cluster_words


def measure_sentence(model, batch, raw_tokens, tokenizer, args, device, rng, sentence_idx):
    with torch.no_grad():
        out = model(batch["embeddings"].to(device), batch["attention_mask"].to(device))

    full_gates = out["gates"]
    token_embeddings = out["token_embeddings"]
    attention_masks = batch["attention_mask"]

    _, assignments_full = model.apply_cluster_filter(token_embeddings, full_gates, attention_masks)
    if assignments_full is None:
        return []
    assignments = assignments_full[0].cpu()

    gates = full_gates[0].cpu()
    attention_mask = attention_masks[0].cpu()

    threshold = model.cfg.clustering.proposal_thresh
    if args.cluster_threshold:
        threshold = args.cluster_threshold
    clusters = collect_cluster_words(raw_tokens, tokenizer, gates, assignments, attention_mask, threshold, args.max_length)
    if not clusters:
        return []

    eligible = sorted({idx for indices in clusters.values() for idx in indices})
    rationale_indices = eligible
    base_tokens = [raw_tokens[i] for i in rationale_indices]
    if not base_tokens:
        return []

    base_embedding = encode_sentence(model, base_tokens, device)
    if base_embedding is None:
        return []

    measurements = []

    for cluster_id, word_indices in clusters.items():
        cluster_tokens = [raw_tokens[idx] for idx in word_indices]
        cluster_set = set(word_indices)
        remaining = [raw_tokens[i] for i in rationale_indices if i not in cluster_set]
        cluster_embedding = encode_sentence(model, remaining, device)
        if cluster_embedding is None:
            continue

        cluster_shift = cosine_shift(base_embedding, cluster_embedding)
        other_indices = [idx for idx in rationale_indices if idx not in cluster_set]
        if len(other_indices) < len(word_indices):
            continue

        random_shifts = []
        for _ in range(max(1, args.samples_per_cluster)):
            sampled = rng.sample(other_indices, len(word_indices))
            sampled_set = set(sampled)
            variant_tokens = [raw_tokens[i] for i in rationale_indices if i not in sampled_set]
            variant_embedding = encode_sentence(model, variant_tokens, device)
            if variant_embedding is None:
                continue
            random_shifts.append(cosine_shift(base_embedding, variant_embedding))

        if not random_shifts:
            continue

        measurements.append(
            {
                "sentence_index": sentence_idx,
                "cluster_id": int(cluster_id),
                "removed_word_indices": word_indices,
                "removed_words": cluster_tokens,
                "cluster_cosine_shift": cluster_shift,
                "random_cosine_shifts": random_shifts,
            }
        )

    return measurements


def collect_measurements(model, loader, tokenizer, args, device):
    rng = random.Random(args.seed)
    records = []
    for idx, batch in enumerate(tqdm(loader, desc="Analyzing")):
        raw_tokens = batch["tokens"][0]
        measurements = measure_sentence(model, batch, raw_tokens, tokenizer, args, device, rng, idx)
        records.extend(measurements)
    return records


def summarize_measurements(measurements):
    bucket = defaultdict(lambda: {"cluster": [], "random": []})
    for item in measurements:
        bucket[item["cluster_id"]]["cluster"].append(item["cluster_cosine_shift"])
        bucket[item["cluster_id"]]["random"].extend(item["random_cosine_shifts"])

    summary = []
    for cluster_id in sorted(bucket):
        cluster_vals = bucket[cluster_id]["cluster"]
        random_vals = bucket[cluster_id]["random"]
        if not cluster_vals or not random_vals:
            continue
        avg_cluster = sum(cluster_vals) / len(cluster_vals)
        avg_random = sum(random_vals) / len(random_vals)
        diff = avg_random - avg_cluster
        ratio = float("inf") if avg_random == 0 else  avg_random / avg_cluster 
        summary.append(
            {
                "cluster_id": cluster_id,
                "avg_cluster_cosine": avg_cluster,
                "avg_random_cosine": avg_random,
                "diff": diff,
                "ratio": ratio,
                "events": len(cluster_vals),
                "random_samples": len(random_vals),
            }
        )
    return summary


def save_measurements(measurements, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in measurements:
            json.dump(item, handle)
            handle.write("\n")
    LOGGER.info("Wrote %d records to %s", len(measurements), output)


def build_fit_loader(cfg_dict, args, model_cfg, device):
    data_cfg = cfg_dict.get("data", {})
    train_cfg = data_cfg.get("train", {})

    dataset_name = train_cfg.get("dataset", "conll2003")
    subset = args.fit_subset if args.fit_subset is not None else train_cfg.get("subset", 1.0)
    shuffle = train_cfg.get("shuffle", True)
    num_workers = train_cfg.get("num_workers", 0)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 16)
    rebuild = data_cfg.get("rebuild_ds", False)

    dataset, tokenizer = get_dataset(
        tokenizer_name=model_cfg.sbert_name,
        name=dataset_name,
        split="train",
        subset=subset,
        rebuild=rebuild,
        shuffle=shuffle,
        max_length=args.max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    return loader, tokenizer, train_cfg


def build_eval_loader(cfg_dict, args, model_cfg, device):
    data_cfg = cfg_dict.get("data", {})
    eval_cfg = data_cfg.get("eval", {})

    dataset_name = eval_cfg.get("dataset", "conll2003")
    subset = args.eval_subset if args.eval_subset is not None else eval_cfg.get("subset", 1.0)
    num_workers = eval_cfg.get("num_workers", 0)
    rebuild = data_cfg.get("rebuild_ds", False)

    dataset, tokenizer = get_dataset(
        tokenizer_name=model_cfg.sbert_name,
        name=dataset_name,
        split="validation",
        subset=subset,
        rebuild=rebuild,
        shuffle=False,
        max_length=args.max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_tokens,
        pin_memory=device.type == "cuda",
    )
    return loader, tokenizer


def main():
    args = parse_args()
    level = logging._nameToLevel.get(args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")

    if args.exp_path:
        exp_path = Path(args.exp_path)
    elif args.exp_signature:
        exp_path = find_experiment_dir(args.exp_signature, args.search_root)
        LOGGER.info("Resolved signature %s to experiment directory %s", args.exp_signature, exp_path)
    else:
        raise ValueError("Either --exp-path or --exp-signature must be provided.")

    cfg_dict = load_experiment_config(exp_path)
    model_cfg = build_model_cfg(cfg_dict, args)
    model = RationaleSelectorModel(cfg=model_cfg).to(device)

    checkpoint_path = resolve_model_path(exp_path, args.model_file)
    LOGGER.info("Loading model weights from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    fit_loader, tokenizer, train_cfg = build_fit_loader(cfg_dict, args, model_cfg, device)
    LOGGER.info(
        "Fitting clusters on %s (subset=%s, batch=%s).",
        train_cfg.get("dataset", "unknown"),
        args.fit_subset if args.fit_subset is not None else train_cfg.get("subset", 1.0),
        fit_loader.batch_size,
    )
    model.fit_cluster_filter_from_loader(fit_loader, model.cfg.clustering, logger=LOGGER, label="model")

    eval_loader, tokenizer = build_eval_loader(cfg_dict, args, model_cfg, device)
    LOGGER.info(
        "Collecting measurements on %s (subset=%s).",
        cfg_dict.get("data", {}).get("eval", {}).get("dataset", "unknown"),
        args.eval_subset if args.eval_subset is not None else cfg_dict.get("data", {}).get("eval", {}).get("subset", 1.0),
    )

    measurements = collect_measurements(model, eval_loader, tokenizer, args, device)
    summary = summarize_measurements(measurements)
    save_measurements(measurements, args.output)

    if not summary:
        LOGGER.info("No cluster measurements collected.")
        return

    LOGGER.info("\nCluster summary (cosine distance)")
    LOGGER.info("cluster | sentences with | cluster_cos | random_cos | diff | ratio")
    for item in summary:
        LOGGER.info(
            "%7d | %15d | %11.4f | %11.4f | %5.4f | %5.2f",
            item["cluster_id"],
            item["events"],
            item["avg_cluster_cosine"],
            item["avg_random_cosine"],
            item["diff"],
            item["ratio"],
        )


if __name__ == "__main__":
    main()
