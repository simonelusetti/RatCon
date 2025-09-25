# inference.py
import torch
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


def evaluate(model, ds, tok, cfg, logger=None):
    """
    Evaluate model by comparing gate activations against gold NER labels (if present).
    Returns metrics + a string with sample highlighted outputs.
    """
    model.eval()
    y_true, y_pred = [], []
    highlighted_samples = []
    device = cfg.device
    thresh = cfg.eval.thresh
    num_samples = cfg.eval.samples.num

    with torch.no_grad():
        for example in ds:
            embeddings = torch.tensor([example["embeddings"]], device=device)
            attention_mask = torch.tensor([example["attention_mask"]], device=device)
            input_ids = torch.tensor([example["input_ids"]], device=device)

            out = model(
                embeddings=embeddings,
                attention_mask=attention_mask,
                logger=logger
            )
            gates = out["gates"][0].cpu().numpy()
            ids = input_ids[0].cpu().tolist()

            # ---- metrics ----
            if "ner_tags" in example:
                gold = example["ner_tags"]
                mask = example["attention_mask"]

                for g, lab, m in zip(gates, gold, mask):
                    if m == 0:  # ignore padding
                        continue
                    y_pred.append(int(g >= thresh))
                    y_true.append(int(lab != 0))  # non-O tag = entity

            tokens = tok.convert_ids_to_tokens(ids)
            pretty = merge_spans(ids, tokens, gates, tok, thresh=thresh)
            merged_orig = " ".join(merge_subwords(ids, tokens, tok))
            highlighted_samples.append(f"\nOrig: {merged_orig}\nPred: {pretty}")

    # ---- metrics summary ----
    if len(y_true) == 0:
        if logger:
            logger.info("No gold labels found in dataset; returning empty metrics.")
        return None, highlighted_samples

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {"precision": precision, "recall": recall, "f1": f1}

    return metrics, highlighted_samples[:num_samples]