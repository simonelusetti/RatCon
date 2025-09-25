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
            out = encoder(**inputs)

        out_dict = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "embeddings": out.last_hidden_state.squeeze(0).tolist(),
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

def collate(batch):
    max_len = max(len(x["embeddings"]) for x in batch)
    emb_dim = len(batch[0]["embeddings"][0])

    embeds, attn, input_ids, ner_tags = [], [], [], []

    for x in batch:
        e, ids = x["embeddings"], x["input_ids"]
        pad = max_len - len(e)
        embeds.append(e + [[0.0] * emb_dim] * pad)
        attn.append([1] * len(e) + [0] * pad)
        input_ids.append(ids + [0] * pad)

        if "ner_tags" in x:
            tags = x["ner_tags"]
            ner_tags.append(tags + [0] * pad)  # pad with "O" (0)

    batch_out = {
        "embeddings": torch.tensor(embeds, dtype=torch.float),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
    }
    if ner_tags:
        batch_out["ner_tags"] = torch.tensor(ner_tags, dtype=torch.long)

    return batch_out


# ---------- Loader ----------

def get_dataset(tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                name="cnn", split="train", max_length=256, field="article",
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
