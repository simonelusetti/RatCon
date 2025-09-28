# data.py
import os, logging, torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from dora import to_absolute_path

logger = logging.getLogger(__name__)


# ---------- Helpers ----------

def _freeze_encoder(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def _encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels=None):
    keep_labels = keep_labels or []

    def _tokenize_and_encode(x):
        enc = tok(text_fn(x), truncation=True, max_length=max_length)
        
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"]).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"]).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs, output_attentions=True, return_dict=True)
            
            attns = out.attentions[-1].mean(1)   # last layer, avg heads [B,L,L]

            out_dict = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "embeddings": out.last_hidden_state.squeeze(0).tolist(),
                "incoming": attns.sum(-2).squeeze(0),   # [B,L]
                "outgoing": attns.sum(-1).squeeze(0),   # [B,L]
            }
            for k in keep_labels:
                out_dict[k] = x[k]
            return out_dict

    return ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)


def _build_dataset(name, split, tokenizer_name, max_length, subset=None, shuffle=False):
    """
    Generic dataset builder for CNN, WikiANN, and CoNLL.
    """
    # pick dataset + text extraction strategy
    if name == "cnn":
        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        text_fn = lambda x: x["article"]
        keep_labels = []
    elif name == "wikiann":
        ds = load_dataset("wikiann", "en", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "conll2003":
        ds = load_dataset("conll2003", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    if shuffle:
        ds = ds.shuffle(seed=42)

    if subset is not None:
        if subset <= 1.0:
            subset = int(len(ds) * subset)
        ds = ds.select(range(subset))

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = _freeze_encoder(AutoModel.from_pretrained(tokenizer_name))
    ds = _encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels)
    return ds, tok


# ---------- Collate ----------

from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    # assume batch is a list of dicts
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attention_masks = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    incoming = [torch.tensor(x["incoming"], dtype=torch.long) for x in batch]
    outgoing = [torch.tensor(x["outgoing"], dtype=torch.long) for x in batch]

    if "ner_tags" in batch[0]:
        ner_tags = [torch.tensor(x["ner_tags"], dtype=torch.long) for x in batch]

    # pad to longest sequence in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    incoming = pad_sequence(incoming, batch_first=True, padding_value=0)
    outgoing = pad_sequence(outgoing, batch_first=True, padding_value=0)
    if "ner_tags" in batch[0]:
        ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-100)  # -100 is common ignore_index

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "incoming": incoming,
        "outgoing": outgoing
    }

    if "ner_tags" in batch[0]:
        batch_out["ner_tags"] = ner_tags
        
    # add precomputed embeddings if your dataset already has them
    if "embeddings" in batch[0]:
        embeddings = [torch.tensor(x["embeddings"], dtype=torch.float) for x in batch]
        embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        batch_out["embeddings"] = embeddings

    return batch_out


# ---------- Loader ----------

def get_dataset(tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                name="cnn", split="train", max_length=256, field="highlights",
                subset=None, rebuild=False, shuffle=False):
    
    if subset is not None and subset != 1.0:
        path = f"./data/{name}_{field}_{split}_{subset}.pt"
    else:
        path = f"./data/{name}_{field}_{split}.pt"

    path = to_absolute_path(path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if os.path.exists(path) and not rebuild:
        logger.info(f"Loading cached dataset from {path}")
        ds = load_from_disk(path)
    else:
        logger.info(f"Building dataset {name} and saving to {path}")
        ds, tok = _build_dataset(name, split, tokenizer_name, max_length, subset, shuffle)
        ds.save_to_disk(path)

    if shuffle:
        ds = ds.shuffle(seed=42)

    return ds, tok
