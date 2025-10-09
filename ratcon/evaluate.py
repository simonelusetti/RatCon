import torch, spacy, re, string, itertools, json
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
from .metrics import summarize_word_stats
from pathlib import Path


_PARTITION_TEMPLATE_CACHE = {}
_PARTITION_TEMPLATE_PATH = None

def format_gold_spans(ids, tokens, gold_labels, tokenizer):
    """
    Merge subwords and gold labels, then format as bracketed spans for entity words.
    """
    buf = ""
    buf_labels = []
    words, word_labels = [], []
    for tok_id, tok_str, lab in zip(ids, tokens, gold_labels):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_labels.append(lab)
        else:
            if buf:
                words.append(buf)
                word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
            buf = tok_str
            buf_labels = [lab]
    if buf:
        words.append(buf)
        word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
    # Now format with brackets for contiguous entity words
    out, span = [], []
    for w, l in zip(words, word_labels):
        if l:
            span.append(w)
        else:
            if span:
                out.append(f"[[{' '.join(span)}]]")
                span = []
            out.append(w)
    if span:
        out.append(f"[[{' '.join(span)}]]")
    return " ".join(out)
def merge_gold_labels(ids, tokens, gold_labels, tokenizer):
    """
    Merge gold labels for subwords into word-level labels.
    A word is an entity if any of its subwords is an entity (label != 0).
    """
    buf = ""
    buf_labels = []
    word_labels = []
    for tok_id, tok_str, lab in zip(ids, tokens, gold_labels):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_labels.append(lab)
        else:
            if buf_labels:
                # If any subword is entity, mark word as entity
                word_labels.append(1.0 if any(l != 0 for l in buf_labels) else 0.0)
            buf = tok_str
            buf_labels = [lab]
    if buf_labels:
        word_labels.append(1.0 if any(l != 0 for l in buf_labels) else 0.0)
    return word_labels

def merge_subwords(ids, tokens, tokenizer):
    # --- Phase 1: merge subwords ---
    buf = ""
    words = []

    def flush(buf):
        if buf:
            words.append(buf)
        return ""

    for tok_id, tok_str in zip(ids, tokens):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue

        if tok_str.startswith("##"):
            buf += tok_str[2:]
        else:
            buf = flush(buf)
            buf = tok_str

    buf = flush(buf)  # flush leftover
    return words

def merge_spans(ids, tokens, gates, tokenizer, thresh=0.2):
    """
    Merge subwords into words (avg gate over subwords), 
    then group contiguous high-gate words into rationale spans.
    """
    
    # --- Phase 1: merge subwords ---
    buf, buf_gs = "", []
    words, word_gates = [], []

    def flush(buf, buf_gs):
        if buf:
            words.append(buf)
            word_gates.append(sum(buf_gs) / len(buf_gs))
        return "", []

    for tok_id, tok_str, g in zip(ids, tokens, gates):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue

        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_gs.append(g)
        else:
            buf, buf_gs = flush(buf, buf_gs)
            buf, buf_gs = tok_str, [g]

    buf, buf_gs = flush(buf, buf_gs)  # flush leftover

    # --- Phase 2: group into rationale spans ---
    out_tokens, span_buf = [], []

    def flush_span(span_buf):
        if span_buf:
            out_tokens.append(f"[[{' '.join(span_buf)}]]")
        return []

    for word, g in zip(words, word_gates):
        if g >= thresh:
            span_buf.append(word)
        else:
            span_buf = flush_span(span_buf)
            out_tokens.append(word)

    span_buf = flush_span(span_buf)  # flush last span

    return " ".join(out_tokens) + "\n"

def _load_spacy_model(name, logger=None):
    try:
        return spacy.load(name)
    except OSError:
        if logger:
            logger.warning(f"spaCy model {name} not found. Word stats will be skipped.")
        return None


def _generate_partition_templates(length):
    indices = list(range(length))
    templates = []
    seen = set()
    for r in range(1, length):
        for combo in itertools.combinations(indices, r):
            if len(combo) == 0 or len(combo) == length:
                continue
            other = tuple(sorted(idx for idx in indices if idx not in combo))
            if not other:
                continue
            subset1 = tuple(sorted(combo))
            canonical = tuple(sorted((subset1, other)))
            if canonical in seen:
                continue
            seen.add(canonical)
            templates.append((subset1, other))
    return templates


def _ensure_partition_cache_loaded(cache_cfg):
    global _PARTITION_TEMPLATE_CACHE, _PARTITION_TEMPLATE_PATH
    if cache_cfg is None:
        return

    path = getattr(cache_cfg, "path", None)
    if not path:
        return
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if _PARTITION_TEMPLATE_PATH == path and _PARTITION_TEMPLATE_CACHE:
        return

    _PARTITION_TEMPLATE_PATH = path
    _PARTITION_TEMPLATE_CACHE = {}
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    for key, value in data.items():
        try:
            length = int(key)
        except ValueError:
            continue
        templates = []
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            first, second = pair
            templates.append((tuple(first), tuple(second)))
        if templates:
            _PARTITION_TEMPLATE_CACHE[length] = templates


def _persist_partition_cache():
    if _PARTITION_TEMPLATE_PATH is None:
        return
    data = {}
    for length, pairs in _PARTITION_TEMPLATE_CACHE.items():
        formatted = []
        for first, second in pairs:
            formatted.append([
                [int(idx) for idx in first],
                [int(idx) for idx in second],
            ])
        data[str(length)] = formatted
    _PARTITION_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PARTITION_TEMPLATE_PATH.write_text(json.dumps(data), encoding="utf-8")


def _get_partition_templates(length, partition_cfg):
    cache_cfg = getattr(partition_cfg, "cache", None) if partition_cfg is not None else None
    cache_enabled = False
    max_length = None
    if cache_cfg is not None and getattr(cache_cfg, "enable", True):
        cache_enabled = True
        max_length = getattr(cache_cfg, "max_length", None)
        _ensure_partition_cache_loaded(cache_cfg)

    if cache_enabled:
        cached = _PARTITION_TEMPLATE_CACHE.get(length)
        if cached is not None:
            return cached
    if max_length is not None and length > max_length:
        return []

    templates = _generate_partition_templates(length)
    if cache_enabled and templates:
        _PARTITION_TEMPLATE_CACHE[length] = templates
        _persist_partition_cache()
    return templates


def _run_inference_examples(model, data, tok, disable_progress, attention_augment, partition_cfg=None, thresh=0.5):
    model.eval()
    device = next(model.parameters()).device

    examples = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating: ", disable=disable_progress):
            embeddings = batch["embeddings"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"]
            incoming = outgoing = None
            if attention_augment:
                incoming = batch["incoming"].to(device, non_blocking=True)
                outgoing = batch["outgoing"].to(device, non_blocking=True)

            out = model(embeddings, attention_mask, incoming, outgoing)
            gates_tensor = out["gates"]
            use_cluster = False
            if hasattr(model, "get_cluster_info"):
                info = model.get_cluster_info()
                if info:
                    use_cluster = True
            if use_cluster:
                gates_tensor, _ = model.apply_cluster_filter(
                    out["token_embeddings"],
                    gates_tensor,
                    attention_mask,
                )
            filtered_gates = gates_tensor.cpu()

            for i in range(embeddings.size(0)):
                ids = input_ids[i].cpu().tolist()
                tokens = tok.convert_ids_to_tokens(ids)
                mask = attention_mask[i].cpu().tolist()
                gates = filtered_gates[i].tolist()
                gold = None
                if "ner_tags" in batch:
                    gold = batch["ner_tags"][i].cpu().tolist()

                partition_gates = None
                if partition_cfg is not None and getattr(partition_cfg, "use", False):
                    partition_gates = _compute_partition_gates(
                        model,
                        out["token_embeddings"][i],
                        attention_mask[i],
                        gates_tensor[i],
                        partition_cfg,
                        thresh,
                    )

                examples.append(
                    {
                        "ids": ids,
                        "tokens": tokens,
                        "mask": mask,
                        "gates": gates,
                        "gold": gold,
                        "partition_gates": partition_gates,
                    }
                )
    return examples


def _pool_embedding(model, token_embeddings, attention_mask, gate_mask):
    emb = token_embeddings
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)

    att = attention_mask.to(dtype=token_embeddings.dtype)
    if att.dim() == 1:
        att = att.unsqueeze(0)

    gates = gate_mask.to(dtype=token_embeddings.dtype)
    if gates.dim() == 1:
        gates = gates.unsqueeze(0)

    mask = att * gates

    if mask.sum() <= 0:
        return None
    pooled = model.pooler({
        "token_embeddings": emb,
        "attention_mask": mask,
    })["sentence_embedding"]
    if hasattr(model, "fourier"):
        pooled = model.fourier(pooled)
    return pooled


def _embedding_distance(anchor, other, partition_cfg):
    mode = getattr(partition_cfg, "distance", "cosine")
    if mode == "euclidean":
        return torch.norm(anchor - other, p=2)
    cos = F.cosine_similarity(anchor.unsqueeze(0), other.unsqueeze(0), dim=-1)
    return 1.0 - cos.squeeze(0)


def _compute_partition_gates(model, token_embeddings, attention_mask, gates, partition_cfg, thresh):
    if not getattr(partition_cfg, "use", False):
        return None

    device = token_embeddings.device
    att_mask = attention_mask.to(device=device, dtype=token_embeddings.dtype)
    gates_device = gates.to(device=device)
    hard_mask = (gates_device >= thresh) & (att_mask > 0)

    selected_idx = torch.nonzero(hard_mask, as_tuple=False).flatten()
    if selected_idx.numel() < 2:
        return None

    base_gate = torch.zeros_like(att_mask)
    base_gate[selected_idx] = 1.0
    original = _pool_embedding(model, token_embeddings, att_mask, base_gate)
    if original is None:
        return None

    templates = _get_partition_templates(selected_idx.numel(), partition_cfg)
    if not templates:
        return None

    selected_positions = selected_idx.tolist()

    best = None

    for subset1_rel, subset2_rel in templates:
        subset1 = [selected_positions[idx] for idx in subset1_rel]
        subset2 = [selected_positions[idx] for idx in subset2_rel]
        if not subset1 or not subset2:
            continue

        gate1 = torch.zeros_like(base_gate)
        gate2 = torch.zeros_like(base_gate)
        gate1[subset1] = 1.0
        gate2[subset2] = 1.0

        emb1 = _pool_embedding(model, token_embeddings, att_mask, gate1)
        emb2 = _pool_embedding(model, token_embeddings, att_mask, gate2)
        if emb1 is None or emb2 is None:
            continue

        dist = _embedding_distance(original, emb1, partition_cfg) + _embedding_distance(original, emb2, partition_cfg)
        if best is None or dist < best[0]:
            best = (dist, gate1.clone(), gate2.clone())

    if best is None:
        return None

    return [
        best[1].detach().cpu().tolist(),
        best[2].detach().cpu().tolist(),
    ]


def _get_gates(example, partition_idx=None):
    if partition_idx is None:
        return example.get("gates")
    partitions = example.get("partition_gates")
    if not partitions or partition_idx >= len(partitions):
        return None
    return partitions[partition_idx]


def _collect_samples(
    examples,
    tok,
    thresh,
    num_samples,
    partition_idx=None,
):
    samples = []
    highlights = []

    for example in examples:
        gates = _get_gates(example, partition_idx=partition_idx)
        if gates is None:
            continue
        highlight = merge_spans(example["ids"], example["tokens"], gates, tok, thresh=thresh)
        if num_samples == 0 or len(highlights) < num_samples:
            highlights.append(highlight)

        if num_samples == 0 or len(samples) < num_samples:
            if example["gold"] is not None:
                original = format_gold_spans(example["ids"], example["tokens"], example["gold"], tok)
            else:
                original = " ".join(merge_subwords(example["ids"], example["tokens"], tok))
            samples.append({
                "original": original,
                "predicted": highlight.strip(),
            })

    return samples, highlights


def _compute_metrics_from_examples(examples, threshold, partition_idx=None):
    y_true, y_pred = [], []
    for example in examples:
        if example["gold"] is None:
            continue
        gates = _get_gates(example, partition_idx=partition_idx)
        if gates is None:
            continue
        for gate, lab, mask in zip(gates, example["gold"], example["mask"]):
            if mask == 0:
                continue
            y_pred.append(int(gate >= threshold))
            y_true.append(int(lab != 0))

    if not y_true:
        return None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _compute_word_statistics(
    highlights,
    nlp,
):
    if nlp is None:
        return []

    stats = []
    for highlight in highlights:
        highlighted = re.findall(r"\[\[(.*?)\]\]", highlight)
        cleaned_words = []
        for span in highlighted:
            for word in span.split():
                w = word.strip(string.punctuation + string.whitespace)
                if w:
                    cleaned_words.append(w.lower())

        if cleaned_words:
            doc = nlp(" ".join(cleaned_words))
        else:
            doc = []

        noun_count = sum(1 for t in doc if getattr(t, "pos_", None) == "NOUN")
        propn_count = sum(1 for t in doc if getattr(t, "pos_", None) == "PROPN")
        verb_count = sum(1 for t in doc if getattr(t, "pos_", None) == "VERB")
        conj_count = sum(
            1 for t in doc if getattr(t, "tag_", None) in {"VBD", "VBG", "VBN", "VBP", "VBZ"}
        )
        stopword_count = sum(1 for t in doc if getattr(t, "is_stop", False))
        recognized = noun_count + propn_count + verb_count + conj_count + stopword_count
        total = len(doc)
        stats.append(
            {
                "nouns": noun_count,
                "proper_nouns": propn_count,
                "verbs": verb_count,
                "conjugations": conj_count,
                "stopwords": stopword_count,
                "other": total - recognized,
                "total": total,
            }
        )

    return stats


def evaluate(
    model,
    data,
    tok,
    tresh,
    disable_progress=False,
    attention_augment=False,
    thresh=0.5,
    samples_num=0,
    spacy_model="en_core_web_sm",
    logger=None,
    partition_cfg=None,
):
    """High-level evaluation orchestrator."""

    nlp = _load_spacy_model(spacy_model, logger=logger)

    examples = _run_inference_examples(model, data, tok, disable_progress, attention_augment, partition_cfg=partition_cfg, thresh=thresh)
    samples, highlights = _collect_samples(examples, tok, thresh, samples_num)
    word_stats = _compute_word_statistics(highlights, nlp)

    for idx, sample in enumerate(samples):
        if idx < len(word_stats):
            sample["word_stats"] = word_stats[idx]

    metrics = _compute_metrics_from_examples(examples, tresh)
    word_summary = summarize_word_stats(word_stats)

    partition_evaluations = []
    if partition_cfg is not None and getattr(partition_cfg, "use", False):
        num_partitions = 0
        for example in examples:
            partitions = example.get("partition_gates") or []
            if partitions:
                num_partitions = max(num_partitions, len(partitions))

        for idx in range(num_partitions):
            part_samples, part_highlights = _collect_samples(examples, tok, thresh, samples_num, partition_idx=idx)
            part_word_stats = _compute_word_statistics(part_highlights, nlp)
            for s_idx, sample in enumerate(part_samples):
                if s_idx < len(part_word_stats):
                    sample["word_stats"] = part_word_stats[s_idx]
            part_metrics = _compute_metrics_from_examples(examples, tresh, partition_idx=idx)
            partition_evaluations.append(
                {
                    "label": f"partition_{idx + 1}",
                    "metrics": part_metrics,
                    "samples": part_samples,
                    "word_stats": part_word_stats,
                    "word_summary": summarize_word_stats(part_word_stats),
                }
            )

    return {
        "metrics": metrics,
        "samples": samples,
        "word_stats": word_stats,
        "word_summary": word_summary,
        "partitions": partition_evaluations,
    }
