# evaluate.py

import torch, spacy
from tqdm import tqdm
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
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        if logger:
            logger.warning("spaCy model 'en_core_web_sm' not found. Word stats will be skipped.")


    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating: "):
            embeddings = batch["embeddings"]        # [B,L,D]
            attention_mask = batch["attention_mask"] # [B,L]
            input_ids = batch["input_ids"]
            incoming = outgoing = None
            if cfg.model.attention_augment:
                incoming = batch["incoming"]
                outgoing = batch["outgoing"]
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
                merged_orig = " ".join(merge_subwords(ids, tokens, tok))
                if len(highlighted_samples) < num_samples:
                    highlighted_samples.append(
                        f"\nOrig: {merged_orig}\nPred: {pretty}"
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
