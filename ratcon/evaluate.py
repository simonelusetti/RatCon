import torch, spacy, re, string
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from .metrics import summarize_word_stats

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


def _run_inference_examples(model, data, tok, disable_progress, attention_augment):
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
                gates_tensor = model.apply_cluster_filter(
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

                examples.append(
                    {
                        "ids": ids,
                        "tokens": tokens,
                        "mask": mask,
                        "gates": gates,
                        "gold": gold,
                    }
                )
    return examples


def _collect_samples(
    examples,
    tok,
    thresh,
    num_samples,
):
    samples = []
    highlights = []

    for example in examples:
        highlight = merge_spans(example["ids"], example["tokens"], example["gates"], tok, thresh=thresh)
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


def _compute_metrics_from_examples(examples, threshold):
    y_true, y_pred = [], []
    for example in examples:
        if example["gold"] is None:
            continue
        for gate, lab, mask in zip(example["gates"], example["gold"], example["mask"]):
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
):
    """High-level evaluation orchestrator."""

    nlp = _load_spacy_model(spacy_model, logger=logger)

    examples = _run_inference_examples(model, data, tok, disable_progress, attention_augment)
    samples, highlights = _collect_samples(examples, tok, thresh, samples_num)
    word_stats = _compute_word_statistics(highlights, nlp)

    for idx, sample in enumerate(samples):
        if idx < len(word_stats):
            sample["word_stats"] = word_stats[idx]

    metrics = _compute_metrics_from_examples(examples, tresh)
    word_summary = summarize_word_stats(word_stats)

    return {
        "metrics": metrics,
        "samples": samples,
        "word_stats": word_stats,
        "word_summary": word_summary
    }
