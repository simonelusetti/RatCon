# inference.py
import torch
from sklearn.metrics import precision_recall_fscore_support


def merge_spans(ids, gates, tokenizer, thresh=0.2):
    """
    Merge subwords and group contiguous selected tokens into spans.
    tokens: list of strings (from tokenizer)
    gates: list of floats (same length as tokens)
    """
    out_text = ""
    buffer = ""
    buffer_gate = 0.0
    tokens = tokenizer.convert_ids_to_tokens(ids)
        
    words = []
    word_gates = []
    
    for id, token, g in zip(ids, tokens, gates):
        if id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue  # skip special tokens
        if token.startswith("##"):
            buffer += token[2:]
            buffer_gate += g
        elif buffer != "":
            out_text += buffer
            words.append(buffer)
            word_gates.append(buffer_gate / len(buffer))  # avg gate for subword
            buffer = ""
        else:  
            out_text += " " + token
            words.append(token)
            word_gates.append(g)
            
    if buffer != "":
        out_text += " " + buffer
        words.append(buffer)
        word_gates.append(buffer_gate / len(buffer))  # avg gate for subword

    for token, g in zip(words, word_gates):
        if g >= thresh:
            buffer += " " + token
        elif buffer != "":
            out_text += " [[" + buffer + "]]"
            buffer = ""
        else:  
            out_text += " " + token
            
    if buffer != "":
        out_text += " [[" + buffer + "]]"

    return out_text +'\n'


def sample_inference(model, tok, ds, device, thresh=0.2, 
    verbose=False, logger=None):
    """
    Run the model on a random sentence and print rationale vs. complement.
    """
    all_pretty = ""
    for example in ds:
        
        embeddings = torch.tensor([example["embeddings"]], device=device)
        attention_mask = torch.tensor([example["attention_mask"]], device=device)
        input_ids = torch.tensor([example["input_ids"]], device=device)

        model.eval()
        with torch.no_grad():
            out = model(embeddings=embeddings,
                        attention_mask=attention_mask,
                        verbose=verbose,
                        logger=logger)
            gates = out["gates"][0].cpu().numpy()
            ids = input_ids[0].cpu().tolist()

        pretty = merge_spans(ids, gates, tok, thresh=thresh)

        all_pretty += "\n--- Sample Inference ---"
        all_pretty += "\n" + pretty
        all_pretty += "\n------------------------\n"
        
    return all_pretty

def evaluate(model, ds, tok, device, thresh=0.2, logger=None):
    """
    Evaluate model by comparing gate activations against gold NER labels (if present).
    Returns metrics + a string with sample highlighted outputs.
    """
    model.eval()
    y_true, y_pred = [], []
    highlighted_samples = []

    with torch.no_grad():
        for _, example in enumerate(ds):
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

                pretty = merge_spans(ids, gates, tok, thresh=thresh)
                highlighted_samples.append(pretty)

    # ---- metrics summary ----
    if len(y_true) == 0:
        if logger:
            logger.info("No gold labels found in dataset; returning empty metrics.")
        return None, highlighted_samples

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {"precision": precision, "recall": recall, "f1": f1}

    return metrics, highlighted_samples