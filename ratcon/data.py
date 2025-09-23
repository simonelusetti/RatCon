# data.py
import torch, os, logging, random
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from dora import to_absolute_path

logger = logging.getLogger(__name__)

from transformers import AutoModel

def build_datasets(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    split="train", 
    max_length=256,
    field="article",
    subset=None
):
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
    if subset is not None:
        if subset <= 1.0:
            subset = int(len(ds) * subset)
        ds = ds.select(range(subset))
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    encoder = AutoModel.from_pretrained(tokenizer_name)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False      

    def _tokenize_and_encode(x):
        enc = tok(x[field], truncation=True, max_length=max_length)
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"]).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"]).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "embeddings": out.last_hidden_state.squeeze(0).tolist()}

    ds = ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)

    return ds, tok


def collate(batch):
    # Find the max sequence length in this batch
    max_len = max(len(x["embeddings"]) for x in batch)
    emb_dim = len(batch[0]["embeddings"][0])

    embeds, attn, input_ids = [], [], []

    for x in batch:
        e = x["embeddings"]
        ids = x["input_ids"]

        pad = max_len - len(e)
        embeds.append(e + [[0.0] * emb_dim] * pad)
        attn.append([1] * len(e) + [0] * pad)
        input_ids.append(ids + [0] * pad)

    return {
        "embeddings": torch.tensor(embeds, dtype=torch.float),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
    }


def get_datasets(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    split="train", 
    max_length=256,
    field="article",
    subset=None,
    rebuild=False
):
    if subset is not None and subset != 1.0:
        path = f"./data/cnn_{field}_{split}_{subset}.pt"
    else:
        path = f"./data/cnn_{field}_{split}.pt"

    path = to_absolute_path(path)
        
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    if os.path.exists(path) and not rebuild:
        logger.info(f"Loading cached dataset from {path}")
        ds = load_from_disk(path)
    else:
        logger.info(f"Building dataset and saving to {path}")
        ds, tok = build_datasets(
            tokenizer_name=tokenizer_name,
            split=split,
            max_length=max_length,
            field=field,
            subset=subset,
        )
        ds.save_to_disk(path)
    return ds, tok

def build_wikiann(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    split="train",
    max_length=128,
    subset=None,
):
    """
    Build WikiANN dataset with frozen encoder embeddings.
    Each example is a sentence with NER tags.
    """
    ds = load_dataset("wikiann", "en", split=split)  # English subset
    if subset is not None:
        if subset <= 1.0:
            subset = int(len(ds) * subset)
        ds = ds.select(range(subset))

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    encoder = AutoModel.from_pretrained(tokenizer_name)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    def _tokenize_and_encode(x):
        # WikiANN provides tokens directly
        text = " ".join(x["tokens"])
        enc = tok(text, truncation=True, max_length=max_length, is_split_into_words=False)
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"]).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"]).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "embeddings": out.last_hidden_state.squeeze(0).tolist(),
            "ner_tags": x["ner_tags"],  # keep original NER labels for evaluation
            "tokens": x["tokens"],
        }

    ds = ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)
    return ds, tok


def get_wikiann(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    split="train",
    max_length=128,
    subset=None,
    rebuild=False,
):
    """
    Load cached WikiANN embeddings or build them.
    """
    if subset is not None and subset != 1.0:
        path = f"./data/wikiann_{split}_{subset}.pt"
    else:
        path = f"./data/wikiann_{split}.pt"

    path = to_absolute_path(path)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if os.path.exists(path) and not rebuild:
        logger.info(f"Loading cached WikiANN dataset from {path}")
        ds = load_from_disk(path)
    else:
        logger.info(f"Building WikiANN dataset and saving to {path}")
        ds, tok = build_wikiann(
            tokenizer_name=tokenizer_name,
            split=split,
            max_length=max_length,
            subset=subset,
        )
        ds.save_to_disk(path)
    return ds, tok
