import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


def format_gold_spans(ids, tokens, gold_labels, tokenizer):
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


def merge_subwords(ids, tokens, tokenizer):
    buf = ""
    words = []

    def flush(acc):
        if acc:
            words.append(acc)
        return ""

    for tok_id, tok_str in zip(ids, tokens):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
        else:
            buf = flush(buf)
            buf = tok_str

    buf = flush(buf)
    return words


def merge_spans(ids, tokens, gates, tokenizer, thresh=0.5):
    buf, buf_gs = "", []
    words, word_gates = [], []

    def flush(acc, gs):
        if acc:
            words.append(acc)
            word_gates.append(sum(gs) / len(gs))
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

    buf, buf_gs = flush(buf, buf_gs)

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

    span_buf = flush_span(span_buf)
    return " ".join(out_tokens)


def _run_inference_examples(model, data, tok, disable_progress, thresh):
    model.eval()
    device = next(model.parameters()).device
    examples = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating", disable=disable_progress):
            embeddings = batch["embeddings"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"]
            ner_tags = batch.get("ner_tags")

            out = model(embeddings, attention_mask)
            gates_tensor = out["gates"]

            for i in range(embeddings.size(0)):
                ids = input_ids[i].cpu().tolist()
                tokens = tok.convert_ids_to_tokens(ids)
                mask = attention_mask[i].cpu().tolist()
                gold = None
                if ner_tags is not None:
                    gold = ner_tags[i].cpu().tolist()

                gates = gates_tensor[i].detach().cpu().tolist()

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


def _collect_samples(examples, tok, thresh, num_samples):
    samples = []
    for example in examples:
        highlight = merge_spans(example["ids"], example["tokens"], example["gates"], tok, thresh=thresh)
        if num_samples and len(samples) >= num_samples:
            break
        if example["gold"] is not None:
            original = format_gold_spans(example["ids"], example["tokens"], example["gold"], tok)
        else:
            original = " ".join(merge_subwords(example["ids"], example["tokens"], tok))
        samples.append({"original": original, "predicted": highlight})
    return samples


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


def evaluate(
    model,
    data,
    tok,
    tresh,
    disable_progress=False,
    thresh=0.5,
    samples_num=0,
    logger=None,
):
    examples = _run_inference_examples(model, data, tok, disable_progress, thresh)
    metrics = _compute_metrics_from_examples(examples, tresh)
    samples = _collect_samples(examples, tok, thresh, samples_num)
    return {
        "metrics": metrics,
        "samples": samples,
    }
