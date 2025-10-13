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


def _encode_reference_sentence(model, sentence, device, logger=None):
    if not sentence:
        return None
    if not hasattr(model, "sbert"):
        if logger:
            logger.warning("Reference sentence provided but model lacks SBERT encoder; skipping similarity step.")
        return None
    try:
        with torch.no_grad():
            encoded = model.sbert.encode([sentence], convert_to_tensor=True)
    except Exception as exc:
        if logger:
            logger.warning(f"Failed to encode reference sentence '{sentence}': {exc}")
        return None
    if encoded.dim() > 1:
        encoded = encoded[0]
    return encoded.to(device=device, dtype=torch.float32)


def _compute_reference_token_scores(
    token_embeddings,
    attention_mask,
    gates,
    tokens,
    reference_vector,
    thresh,
):
    if reference_vector is None:
        return None

    token_vecs = token_embeddings.to(device=reference_vector.device, dtype=reference_vector.dtype)
    att_mask = attention_mask.to(device=reference_vector.device)
    gate_vals = gates.to(device=reference_vector.device)

    if att_mask.dim() > 1:
        att_mask = att_mask.squeeze(0)
    if gate_vals.dim() > 1:
        gate_vals = gate_vals.squeeze(0)
    if token_vecs.dim() > 2:
        token_vecs = token_vecs.squeeze(0)

    valid_mask = (att_mask > 0) & (gate_vals >= thresh)
    if not valid_mask.any():
        return []

    indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
    scores = []
    for idx in indices:
        vec = token_vecs[idx]
        score = torch.dot(vec, reference_vector)
        scores.append(
            {
                "token": tokens[idx],
                "index": int(idx),
                "score": float(score.item()),
            }
        )
    return scores


def _generate_partition_templates(length, num_parts):
    if num_parts < 2 or length < num_parts:
        return []

    templates = []
    assignment = []

    def backtrack(position, used_parts):
        if position == length:
            if used_parts == num_parts:
                masks = []
                for part_idx in range(num_parts):
                    mask = [0.0] * length
                    for token_idx, assigned_part in enumerate(assignment):
                        if assigned_part == part_idx:
                            mask[token_idx] = 1.0
                    masks.append(mask)
                templates.append(masks)
            return

        max_part = used_parts if used_parts < num_parts else num_parts - 1
        for part_idx in range(max_part + 1):
            if part_idx == used_parts and used_parts == num_parts:
                continue
            assignment.append(part_idx)
            backtrack(position + 1, used_parts + (1 if part_idx == used_parts else 0))
            assignment.pop()

    backtrack(0, 0)
    return templates


def _ensure_partition_cache_loaded(cache_cfg):
    global _PARTITION_TEMPLATE_CACHE, _PARTITION_TEMPLATE_PATH
    if cache_cfg is None:
        return

    path = cache_cfg.path
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
        if ":" in key:
            length_str, parts_str = key.split(":", 1)
        else:
            length_str, parts_str = key, "2"
        try:
            length = int(length_str)
            num_parts = int(parts_str)
        except ValueError:
            continue
        templates = []
        for entry in value:
            if not isinstance(entry, list) or len(entry) != num_parts:
                continue
            masks = []
            valid = True
            for mask in entry:
                if not isinstance(mask, list) or len(mask) != length:
                    valid = False
                    break
                masks.append(list(map(float, mask)))
            if valid:
                templates.append(masks)
        if templates:
            _PARTITION_TEMPLATE_CACHE[(length, num_parts)] = templates


def _persist_partition_cache():
    if _PARTITION_TEMPLATE_PATH is None:
        return
    data = {}
    for (length, num_parts), templates in _PARTITION_TEMPLATE_CACHE.items():
        formatted = []
        for masks in templates:
            if len(masks) != num_parts:
                continue
            formatted.append([list(map(float, mask)) for mask in masks])
        if formatted:
            data[f"{length}:{num_parts}"] = formatted
    _PARTITION_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PARTITION_TEMPLATE_PATH.write_text(json.dumps(data), encoding="utf-8")


def _get_partition_templates(length, partition_cfg):
    cache_cfg = partition_cfg.cache if partition_cfg is not None else None
    cache_enabled = False
    max_length = None
    if cache_cfg is not None and cache_cfg.enable:
        cache_enabled = True
        max_length = cache_cfg.max_length
        _ensure_partition_cache_loaded(cache_cfg)

    num_parts = max(2, int(partition_cfg.parts))

    if cache_enabled:
        cached = _PARTITION_TEMPLATE_CACHE.get((length, num_parts))
        if cached is not None:
            return cached
    if max_length is not None and length > max_length:
        return []

    templates = _generate_partition_templates(length, num_parts)
    if cache_enabled and templates:
        _PARTITION_TEMPLATE_CACHE[(length, num_parts)] = templates
        _persist_partition_cache()
    return templates


def _run_inference_examples(
    model,
    data,
    tok,
    disable_progress,
    attention_augment,
    partition_cfg=None,
    thresh=0.5,
    reference_vector=None,
    reference_threshold=0.5,
):
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

            for i in range(embeddings.size(0)):
                ids = input_ids[i].cpu().tolist()
                tokens = tok.convert_ids_to_tokens(ids)
                mask = attention_mask[i].cpu().tolist()
                gold = None
                if "ner_tags" in batch:
                    gold = batch["ner_tags"][i].cpu().tolist()

                reference_scores = None
                if reference_vector is not None:
                    reference_scores = _compute_reference_token_scores(
                        embeddings[i],
                        attention_mask[i],
                        gates_tensor[i],
                        tokens,
                        reference_vector,
                        thresh,
                    )
                    if reference_scores:
                        keep_scores = []
                        zero_indices = []
                        for entry in reference_scores:
                            if entry["score"] > reference_threshold:
                                keep_scores.append(entry)
                            else:
                                zero_indices.append(entry["index"])
                        if zero_indices:
                            gates_tensor[i, zero_indices] = 0.0
                        reference_scores = keep_scores

                partition_gates = None
                if partition_cfg is not None and partition_cfg.use:
                    partition_gates = _compute_partition_gates(
                        model,
                        out["token_embeddings"][i],
                        attention_mask[i],
                        gates_tensor[i],
                        partition_cfg,
                        thresh,
                    )

                gates = gates_tensor[i].detach().cpu().tolist()

                example_entry = {
                    "ids": ids,
                    "tokens": tokens,
                    "mask": mask,
                    "gates": gates,
                    "gold": gold,
                    "partition_gates": partition_gates,
                }
                if reference_scores is not None:
                    example_entry["reference_dot_products"] = reference_scores

                examples.append(
                    example_entry
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
    mode = partition_cfg.distance
    anchor_flat = anchor.reshape(-1)
    other_flat = other.reshape(-1, anchor_flat.numel())
    anchor_vec = anchor_flat.unsqueeze(0).expand(other_flat.size(0), -1)
    if mode == "euclidean":
        dist = torch.norm(anchor_vec - other_flat, p=2, dim=-1)
    else:
        cos = F.cosine_similarity(anchor_vec, other_flat, dim=-1)
        dist = 1.0 - cos
    if dist.numel() == 1:
        return dist.squeeze(0)
    return dist


def _compute_partition_gates(model, token_embeddings, attention_mask, gates, partition_cfg, thresh):
    if not partition_cfg.use:
        return None

    device = token_embeddings.device
    att_mask = attention_mask.to(device=device, dtype=token_embeddings.dtype)
    gates_device = gates.to(device=device)
    hard_mask = (gates_device >= thresh) & (att_mask > 0)

    selected_idx = torch.nonzero(hard_mask, as_tuple=False).flatten()

    num_parts = max(2, int(partition_cfg.parts))
    if selected_idx.numel() < num_parts:
        return None

    base_gate = torch.zeros_like(att_mask)
    base_gate[selected_idx] = 1.0
    original = _pool_embedding(model, token_embeddings, att_mask, base_gate)
    if original is None:
        return None

    templates = _get_partition_templates(selected_idx.numel(), partition_cfg)
    if not templates:
        return None

    template_tensor = torch.tensor(
        templates,
        device=device,
        dtype=token_embeddings.dtype,
    )
    num_templates, num_parts, selected_len = template_tensor.shape
    seq_len = att_mask.size(0)

    gate_matrix = torch.zeros(
        num_templates,
        num_parts,
        seq_len,
        device=device,
        dtype=token_embeddings.dtype,
    )
    gate_matrix[:, :, selected_idx] = template_tensor

    att_mask_unsq = att_mask.unsqueeze(0).unsqueeze(0)
    masks = gate_matrix * att_mask_unsq

    valid_mask = (masks.sum(dim=2) > 0).all(dim=1)
    if not valid_mask.any():
        return None

    valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
    mask_valid = masks[valid_indices]

    num_valid = mask_valid.size(0)
    token_batch = token_embeddings.unsqueeze(0).expand(num_valid * num_parts, -1, -1)
    att_batch = mask_valid.reshape(num_valid * num_parts, seq_len)

    pooled = model.pooler({
        "token_embeddings": token_batch,
        "attention_mask": att_batch,
    })["sentence_embedding"]
    pooled = pooled.reshape(num_valid, num_parts, -1)

    dist = None
    for part_idx in range(num_parts):
        part_emb = pooled[:, part_idx, :]
        part_dist = _embedding_distance(original, part_emb, partition_cfg)
        dist = part_dist if dist is None else dist + part_dist

    best_val, best_idx = torch.min(dist, dim=0)
    if not torch.isfinite(best_val):
        return None

    best_masks = mask_valid[best_idx]
    return [best_masks[part_idx].detach().cpu().tolist() for part_idx in range(num_parts)]


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
            sample_entry = {
                "original": original,
                "predicted": highlight.strip(),
            }
            ref_scores = example.get("reference_dot_products")
            if ref_scores is not None:
                sample_entry["reference_dot_products"] = ref_scores
            samples.append(sample_entry)

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



def _compute_average_partition_length(examples, partition_idx=None):
    total_length = 0.0
    example_count = 0

    for example in examples:
        gates = _get_gates(example, partition_idx=partition_idx)
        if gates is None:
            continue

        length = 0.0
        for gate, mask in zip(gates, example["mask"]):
            if mask == 0:
                continue
            if gate > 0.0:
                length += 1.0

        total_length += length
        example_count += 1

    if example_count == 0:
        return None
    return total_length / example_count



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

        noun_count = sum(1 for t in doc if t.pos_ == "NOUN")
        propn_count = sum(1 for t in doc if t.pos_ == "PROPN")
        verb_count = sum(1 for t in doc if t.pos_ == "VERB")
        conj_count = sum(1 for t in doc if t.tag_ in {"VBD", "VBG", "VBN", "VBP", "VBZ"})
        stopword_count = sum(1 for t in doc if t.is_stop)
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
    reference_sentence=None,
    reference_threshold=0.5,
):
    """High-level evaluation orchestrator."""

    nlp = _load_spacy_model(spacy_model, logger=logger)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    reference_vector = _encode_reference_sentence(model, reference_sentence, device, logger=logger)

    examples = _run_inference_examples(
        model,
        data,
        tok,
        disable_progress,
        attention_augment,
        partition_cfg=partition_cfg,
        thresh=thresh,
        reference_vector=reference_vector,
        reference_threshold=reference_threshold,
    )
    samples, highlights = _collect_samples(examples, tok, thresh, samples_num)
    word_stats = _compute_word_statistics(highlights, nlp)

    metrics = _compute_metrics_from_examples(examples, tresh)
    word_summary = summarize_word_stats(word_stats)

    partition_evaluations = []
    if partition_cfg is not None and partition_cfg.use:
        num_partitions = 0
        for example in examples:
            partitions = example.get("partition_gates") or []
            if partitions:
                num_partitions = max(num_partitions, len(partitions))

        for idx in range(num_partitions):
            part_samples, part_highlights = _collect_samples(examples, tok, thresh, samples_num, partition_idx=idx)
            part_word_stats = _compute_word_statistics(part_highlights, nlp)
            part_metrics = _compute_metrics_from_examples(examples, tresh, partition_idx=idx)
            avg_length = _compute_average_partition_length(examples, partition_idx=idx)
            if avg_length is not None:
                if part_metrics is None:
                    part_metrics = {}
                else:
                    part_metrics = dict(part_metrics)
                part_metrics["avg_length"] = float(avg_length)
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
        "reference_sentence": reference_sentence,
        "reference_threshold": reference_threshold,
    }
