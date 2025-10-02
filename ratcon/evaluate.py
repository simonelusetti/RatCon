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
# evaluate.py

import torch, spacy
from tqdm import tqdm
from .utils import should_disable_tqdm
from sklearn.metrics import precision_recall_fscore_support

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


from sklearn.metrics import precision_recall_fscore_support

def evaluate(model, data, tok, cfg, logger=None):
    """
    Evaluate model by comparing gate activations against gold NER labels (if present).
    Works with full batches instead of only the first example.
    """
    model.eval()

    y_true, y_pred = [], []
    highlighted_samples = []
    highlighted_word_stats = []
    thresh = cfg.eval.thresh
    num_samples = cfg.eval.samples.num

    # Load spaCy English model
    spacy_model = getattr(cfg.eval, "spacy_model", "en_core_web_sm")
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        nlp = None
        if logger:
            logger.warning(f"spaCy model {spacy_model} not found. Word stats will be skipped.")


    model_device = next(model.parameters()).device

    logging_cfg = getattr(cfg, "logging", None)
    disable_progress = should_disable_tqdm(
        metrics_only=getattr(logging_cfg, "metrics_only", False) if logging_cfg is not None else False
    )

    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating: ", disable=disable_progress):
            embeddings = batch["embeddings"].to(model_device, non_blocking=True)        # [B,L,D]
            attention_mask = batch["attention_mask"].to(model_device, non_blocking=True) # [B,L]
            input_ids = batch["input_ids"]
            incoming = outgoing = None
            if cfg.model.attention_augment:
                incoming = batch["incoming"].to(model_device, non_blocking=True)
                outgoing = batch["outgoing"].to(model_device, non_blocking=True)

            out = model(embeddings, attention_mask, incoming, outgoing)
            gates = out["gates"].cpu().numpy()      # [B,L]

            # ---- loop over batch ----
            for i in range(embeddings.size(0)):
                ids = input_ids[i].cpu().tolist()
                g   = gates[i]
                mask = attention_mask[i].cpu().tolist()

                # ---- metrics ----
                if "ner_tags" in batch:
                    gold = batch["ner_tags"][i].cpu().tolist()
                    for gi, lab, m in zip(g, gold, mask):
                        if m == 0:  # skip padding
                            continue
                        y_pred.append(int(gi >= thresh))
                        y_true.append(int(lab != 0))  # non-O = entity

                # ---- pretty printing ----
                tokens = tok.convert_ids_to_tokens(ids)
                pretty = merge_spans(ids, tokens, g, tok, thresh=thresh)
                # For original spans, merge words and gold labels, then format with brackets
                if "ner_tags" in batch:
                    gold = batch["ner_tags"][i].cpu().tolist()
                    orig_pretty = format_gold_spans(ids, tokens, gold, tok)
                else:
                    orig_pretty = " ".join(merge_subwords(ids, tokens, tok))
                if len(highlighted_samples) < num_samples:
                    highlighted_samples.append(
                        f"\nOrig: {orig_pretty}\nPred: {pretty}"
                    )

                # --- Word stats for highlighted words (for all examples) ---
                if nlp is not None:
                    import re, string
                    highlighted = re.findall(r'\[\[(.*?)\]\]', pretty)
                    cleaned_words = []
                    for span in highlighted:
                        for word in span.split():
                            w = word.strip(string.punctuation + string.whitespace)
                            if w:
                                cleaned_words.append(w.lower())
                    doc = nlp(" ".join(cleaned_words))
                    noun_count = sum(1 for t in doc if t.pos_ == "NOUN")
                    propn_count = sum(1 for t in doc if t.pos_ == "PROPN")
                    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
                    conj_count = sum(1 for t in doc if t.tag_ in ["VBD", "VBG", "VBN", "VBP", "VBZ"])
                    stopword_count = sum(1 for t in doc if t.is_stop)
                    recognized = noun_count + propn_count + verb_count + conj_count + stopword_count
                    other = len(doc) - recognized
                    highlighted_word_stats.append({
                        "nouns": noun_count,
                        "proper_nouns": propn_count,
                        "verbs": verb_count,
                        "conjugations": conj_count,
                        "stopwords": stopword_count,
                        "other": other,
                        "total": len(doc)
                    })


    # ---- metrics summary ----
    if len(y_true) == 0:
        return None, highlighted_samples, highlighted_word_stats

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    return metrics, highlighted_samples, highlighted_word_stats
