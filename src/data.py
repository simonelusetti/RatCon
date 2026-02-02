import logging, torch
from pathlib import Path
from typing import Callable, Optional, TypedDict
from typing_extensions import NotRequired
from datasets import DatasetDict, load_dataset, load_from_disk, Value, Sequence
from dora import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .sentence import resolve_tokenizer_group, build_sentence_encoder, SentenceEncoder

from .datasets_builders import (
    build_both_parasci,
    build_conll2000,
    build_movie_reviews,
    build_shape,
    build_twitter,
    build_ud,
    build_treebank,
    build_conll2003,
    build_wikiann,
    map_conll2003_secondary_labels,
    build_wikiann_swap,
    build_ud_pos
)

ALIASES = {
    "cnn_dailymail": {"cnn", "cnn_dailymail"},
    "shape": {"shape"},
    "wikiann": {"wikiann"},
    "conll2003": {"conll2003", "conll03"},
    "conll2000": {"conll2000", "conll00"},
    "ud": {"ud"},
    "ud_pos": {"ud_raw","ud_pos"},
    "treebank": {"treebank", "tb"},
    "movie_rationales": {"movie_rationales", "mr"},
    "parasci": {"parasci", "ps"},
    "parasci_concat": {"parasci_concat", "parasci-concat", "prsc"},
    "tweet_sentiment": {"tweet_sentiment", "twitter", "twt"},
    "mpqa": {"mpqa"},
    "fever": {"fever"},
    "glue": {"glue", "stsb"},
    "wikiann_swap": {"wikiann_swap"}
}

TEXT_FIELD = {
    "cnn_dailymail": "article",
    "shape": "tokens",
    "wikiann": "tokens",
    "conll2003": "tokens",
    "conll2000": "tokens",
    "ud": "tokens",
    "treebank": "tokens",
    "movie_rationales": "tokens",
    "parasci": "tokens",
    "parasci_concat": "tokens",
    "tweet_sentiment": "tokens",
    "mpqa": "text",
    "fever": "claim",
    "glue": "sentence1",
}

PAD_TAG = "-100"

SECONDARY_LABELS_DS = {
    "conll2003": map_conll2003_secondary_labels,
}

class TokenizedExample(TypedDict):
    ids: list[int]
    attn_mask: list[int]
    word_ids: list[int | None]
    tokens: list[str]
    labels: NotRequired[list[str]]
    scnd_labels: NotRequired[list[str]]

class CollatedBatch(TypedDict, total=False):
    ids: torch.Tensor
    attn_mask: torch.Tensor
    word_ids: torch.Tensor
    tokens: list[list[str]]
    labels: list[list[str]]
    scnd_labels: list[list[str]]

def canonical_name(name: str) -> str:
    for canon, aliases in ALIASES.items():
        if name == canon or name in aliases:
            return canon
    raise ValueError(f"Unknown dataset name: {name}")


def dataset_path(name: str, tokenizer_group: str, config: dict | None) -> Path:
    suffix = f"_tok={tokenizer_group}"
    if config:
        suffix += "_" + "_".join(str(v) for v in config.values())
    return Path(to_absolute_path(f"./data/cache/{name}{suffix}"))


def shuffle_and_subset(ds: DatasetDict, subset: float | int | None, shuffle: bool) -> DatasetDict:
    if shuffle:
        ds = DatasetDict({k: v.shuffle(seed=42) for k, v in ds.items()})

    if subset is None or subset == 1.0:
        return ds

    assert "train" in ds and "test" in ds, "Expected train/test splits"
    n_train = int(len(ds["train"]) * subset) if subset <= 1 else int(subset)
    n_test = int(len(ds["test"]) * subset) if subset <= 1 else int(subset)

    ds["train"] = ds["train"].select(range(n_train))
    ds["test"] = ds["test"].select(range(n_test))
    return ds


def collate(batch: list[TokenizedExample]) -> CollatedBatch:
    tokens = [x["tokens"] for x in batch]
    tokens_max_len = max(len(x) for x in tokens)
    tokens_padded = [
        x + [PAD_TAG] * (tokens_max_len - len(x))
        for x in tokens
    ]

    out = {
        "ids": pad_sequence(
            [torch.as_tensor(x["ids"], dtype=torch.long) for x in batch],
            batch_first=True,
            padding_value=0,
        ),
        "attn_mask": pad_sequence(
            [torch.as_tensor(x["attn_mask"], dtype=torch.long) for x in batch],
            batch_first=True,
            padding_value=0,
        ),
        "word_ids": pad_sequence(
            [
                torch.as_tensor(
                    [-1 if w is None else w for w in x["word_ids"]],
                    dtype=torch.long,
                )
                for x in batch
            ],
            batch_first=True,
            padding_value=-1,
        ),
        "tokens": tokens_padded,
    }

    if "labels" in batch[0]:
        labels = [x["labels"] for x in batch]
        labels_max_len = max(len(x) for x in labels)
        out["labels"] = [
            x + [PAD_TAG] * (labels_max_len - len(x))
            for x in labels
        ]

    if "scnd_labels" in batch[0]:
        scnd_labels = [x["scnd_labels"] for x in batch]
        scnd_max_len = max(len(x) for x in scnd_labels)
        out["scnd_labels"] = [
            x + [PAD_TAG] * (scnd_max_len - len(x))
            for x in scnd_labels
        ]
        
    assert not torch.isnan(out["ids"]).any(), "Collated IDs contain NaNs"

    return out


def resolve_dataset(
    name: str,
    text_field: str = "tokens",
    logger: Optional[logging.Logger] = None,
    config: dict | None = None,
) -> DatasetDict:
    if logger:
        logger.info(f"Resolving dataset: {name}")

    if name == "conll2000":
        ds = build_conll2000()
    elif name == "movie_rationales":
        ds = build_movie_reviews()
    elif name == "tweet_sentiment":
        ds = build_twitter()
    elif name == "shape":
        assert config is not None and "shape" in config, "Shape dataset requires 'shape' config."
        original_name = canonical_name(config["shape"]["original"])
        rate = config["shape"]["rate"]
        if rate > 1.0:
            rate = float(rate) / 100.0
        original_ds = resolve_dataset(original_name, text_field=config["shape"].get("text_field", "tokens"))
        ds = build_shape(original_ds, rate=rate)
    elif name == "ud":
        ds = build_ud()
    elif name in {"parasci", "parasci_concat"}:
        a, b = build_both_parasci()
        ds = a if name == "parasci" else b
    elif name == "treebank":
        ds = build_treebank()
    elif name == "fever":
        ds = load_from_disk(to_absolute_path("./data/raw/fever"))
    elif name == "glue":
        ds = load_dataset("glue", "stsb")
    elif name == "conll2003":
        ds = build_conll2003()
    elif name == "wikiann":
        ds = build_wikiann()
    elif name == "wikiann_swap":
        ds = build_wikiann_swap()
    elif name == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0")
    elif name == "ud_pos":
        ds = build_ud_pos()
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if text_field != "tokens":
        ds = ds.rename_column(text_field, "tokens")

    assert isinstance(ds, DatasetDict), "Expected DatasetDict"
    assert "train" in ds and "test" in ds, "Dataset must have train and test splits"
    return ds

@torch.no_grad()
def encode_examples(
    data_cfg: dict,
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    scnd_labels_map: Callable | None = None,
) -> DatasetDict:
    labels_present = "labels" in ds["train"].column_names

    def _encode(example: dict) -> TokenizedExample:
        text = example["tokens"]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=data_cfg.max_length,
            is_split_into_words=isinstance(text, list),
        )
        
        out = {
            "ids": enc["input_ids"],
            "attn_mask": enc["attention_mask"],
            "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"]),
            "word_ids": enc.word_ids(),
        }

        labels = example.get("labels", None)
        if labels_present and labels is not None:
            word_ids = enc.word_ids()
            aligned_labels = []
            for wid in word_ids:
                if wid is None:
                    aligned_labels.append(PAD_TAG)
                else:
                    aligned_labels.append(str(labels[wid]))
            out["labels"] = aligned_labels
            if scnd_labels_map is not None:
                out["scnd_labels"] = scnd_labels_map(aligned_labels)

        return out

    out = DatasetDict()
    for split, d in ds.items():
        cols = ["ids", "attn_mask", "tokens", "word_ids"]
        
        if labels_present:
            cols.append("labels")
            if scnd_labels_map is not None:
                cols.append("scnd_labels")

        out[split] = d.map(_encode).select_columns(cols)

    return out


def get_dataset(
    data_cfg: dict,
    runtime_cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger | None = None,
) -> DatasetDict:
    tokenizer_group = resolve_tokenizer_group(data_cfg.encoder.family)
    name = canonical_name(data_cfg.dataset)
    path = dataset_path(name, tokenizer_group, data_cfg.get("config"))

    text_field = TEXT_FIELD.get(name, "tokens")
    if path.exists() and not runtime_cfg.rebuild:
        ds = load_from_disk(path)
        if text_field != "tokens" and "tokens" not in ds["train"].column_names and text_field in ds["train"].column_names:
            ds = ds.rename_column(text_field, "tokens")
        if logger:
            logger.info(f"Loading cached TOKENIZED dataset from {path}")
        assert "ids" in ds["train"].column_names and "attn_mask" in ds["train"].column_names
        return ds

    if logger:
        logger.info(f"Building + tokenizing dataset (NO embeddings): {name}")

    ds_dict = resolve_dataset(name, text_field, logger, config=data_cfg.get("config", None))
    
    features = ds_dict["train"].features.copy()
    if "labels" in features:
        features["labels"] = Sequence(Value("string"))
    if "scnd_labels" in features:
        features["scnd_labels"] = Sequence(Value("string"))
    ds_dict = ds_dict.cast(features)

    scnd_labels_map = SECONDARY_LABELS_DS.get(name, None)
    ds_tok = encode_examples(data_cfg, ds_dict, tokenizer, scnd_labels_map=scnd_labels_map)

    path.parent.mkdir(parents=True, exist_ok=True)
    ds_tok.save_to_disk(path)

    return ds_tok


def initialize_data(
    data_cfg: dict,
    runtime_cfg: dict,
    logger: logging.Logger | None = None,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader, SentenceEncoder, PreTrainedTokenizerBase]:
    encoder, tokenizer = build_sentence_encoder(
        family=data_cfg.encoder.family,
        encoder_name=data_cfg.encoder.name,
        device=device,
    )

    ds = get_dataset(data_cfg, runtime_cfg, tokenizer, logger)
    ds = shuffle_and_subset(ds, data_cfg.subset, data_cfg.shuffle)

    ds_train = DataLoader(
        ds["train"],
        batch_size=runtime_cfg.batch_size,
        num_workers=runtime_cfg.num_workers,
        collate_fn=collate,
        shuffle=data_cfg.shuffle,
        pin_memory=(device == "cuda"),
    )

    ds_test = DataLoader(
        ds["test"],
        batch_size=runtime_cfg.batch_size,
        num_workers=runtime_cfg.num_workers,
        collate_fn=collate,
        shuffle=data_cfg.shuffle,
        pin_memory=(device == "cuda"),
    )

    return ds_train, ds_test, encoder, tokenizer
