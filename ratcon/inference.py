# inference.py
import torch

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

        tokens = tok.convert_ids_to_tokens(ids)
        highlighted = []
        for tok_id, tok_str, g in zip(ids, tokens, gates):
            if tok_id in [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id]:
                continue  # skip special tokens
            if g >= thresh:
                highlighted.append(f"[[{tok_str}]]")  # red
            else:
                highlighted.append(tok_str)

        # Join with spaces, handling WordPiece '##'
        out_text = []
        for _, t in enumerate(highlighted):
            if t.startswith("##"):
                out_text[-1] = out_text[-1] + t[2:]
            elif t.startswith("[[##"):
                out_text[-1] = out_text[-1] + t[4:]
            else:
                out_text.append(t)
        pretty = " ".join(out_text)

        all_pretty += "\n--- Sample Inference ---"
        all_pretty += "\n" + pretty
        all_pretty += "\n------------------------\n"
        
    return all_pretty