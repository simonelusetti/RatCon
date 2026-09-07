import logging, torch
from pathlib import Path
from typing import TypedDict
from typing_extensions import NotRequired
from datasets import DatasetDict, load_from_disk, Value, Sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .sentence import ALIAS_TO_CANON, build_sentence_encoder, SentenceEncoder
from .utils import to_absolute_path

from .datasets_builders import (
    build_conll2000,
    build_movie_reviews,
    build_conll2003,
    build_wikiann,
    build_stsb,
)

ALIASES = {
    "wikiann": {"wikiann"},
    "conll2003": {"conll2003", "conll03"},
    "conll2000": {"conll2000", "conll00"},
    "movie_rationales": {"movie_rationales", "mr"},
    "stsb": {"stsb", "sts-b", "stsb-en"},
}

BUILDERS = {
    "wikiann": build_wikiann,
    "conll2003": build_conll2003,
    "conll2000": build_conll2000,
    "movie_rationales": build_movie_reviews,
    "stsb": build_stsb,
}

PAD_TAG = "-100"
SPECIAL_TAG = "special"

# Fixed across every run so dataset-level example ordering never varies with
# runtime.seed (see shuffle_and_subset) -- only weight init/dropout/batch order do.
DATASET_SHUFFLE_SEED = 42

# ---------------------------------------------------------------------------
# Human-readable label display names, keyed by canonical dataset name.
# Inner dict: raw label string (as stored after the ClassLabel->string cast
# in get_dataset) -> real tag name. For wikiann/conll03 these are the actual
# BIO entity tags (needed for seqeval, not just display); other datasets are
# shown/used with their raw label unchanged.
# ---------------------------------------------------------------------------
LABEL_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "conll03": {
        "0": "O",
        "1": "B-PER",
        "2": "I-PER",
        "3": "B-ORG",
        "4": "I-ORG",
        "5": "B-LOC",
        "6": "I-LOC",
        "7": "B-MISC",
        "8": "I-MISC",
    },
    "wikiann": {
        "0": "O",
        "1": "B-PER",
        "2": "I-PER",
        "3": "B-ORG",
        "4": "I-ORG",
        "5": "B-LOC",
        "6": "I-LOC",
    },
    "mr": {
        "0": "not rationale",
        "1": "rationale",
    },
    # CoNLL-2000 has no ClassLabel encoding upstream (NLTK's iob_sents()
    # yields raw tag strings, not integer ids) -- this fixes a canonical
    # index order for the 23 tags actually observed across train+test
    # (11 phrase types x {B,I}, plus O), needed by the NER probe's output
    # layer. The rationale/bias-test pipeline doesn't need this at all: it
    # keys everything by the raw tag string directly.
    "conll2000": {
        "0": "O",
        "1": "B-ADJP", "2": "I-ADJP",
        "3": "B-ADVP", "4": "I-ADVP",
        "5": "B-CONJP", "6": "I-CONJP",
        "7": "B-INTJ", "8": "I-INTJ",
        "9": "B-LST", "10": "I-LST",
        "11": "B-NP", "12": "I-NP",
        "13": "B-PP", "14": "I-PP",
        "15": "B-PRT", "16": "I-PRT",
        "17": "B-SBAR", "18": "I-SBAR",
        "19": "B-UCP", "20": "I-UCP",
        "21": "B-VP", "22": "I-VP",
    },
}
# Also index by canonical_name() output, so lookups keyed on the canonical
# name (e.g. NER scoring) work regardless of which alias the user passed in
# data.dataset (view.py's plotting lookups key on the raw override string, so
# both spellings are kept).
LABEL_DISPLAY_NAMES["conll2003"] = LABEL_DISPLAY_NAMES["conll03"]
LABEL_DISPLAY_NAMES["movie_rationales"] = LABEL_DISPLAY_NAMES["mr"]

class TokenizedExample(TypedDict):
    ids: list[int]
    attn_mask: list[int]
    word_ids: list[int | None]
    tokens: list[str]
    labels: NotRequired[list[str]]

class CollatedBatch(TypedDict, total=False):
    ids: torch.Tensor
    attn_mask: torch.Tensor
    word_ids: torch.Tensor
    tokens: list[list[str]]
    labels: list[list[str]]

def canonical_name(name: str) -> str:
    for canon, aliases in ALIASES.items():
        if name == canon or name in aliases:
            return canon
    raise ValueError(f"Unknown dataset name: {name}")


def dataset_path(name: str, family: str, max_length: int) -> Path:
    # Anchored to the launch directory, not the cwd: forge chdirs into the
    # per-run output directory before training starts, and the tokenized
    # dataset cache is shared across every run, not per-run state.
    return to_absolute_path(f"./data/cache/{name}_tok={family}_len={max_length}")


def shuffle_and_subset(ds: DatasetDict, subset: float | int | None, shuffle: bool) -> DatasetDict:
    if shuffle:
        ds = DatasetDict({k: v.shuffle(seed=DATASET_SHUFFLE_SEED) for k, v in ds.items()})

    if subset is None or subset == 1.0:
        return ds

    ds["train"] = subset_split(ds["train"], subset)
    ds["test"] = subset_split(ds["test"], subset)
    return ds


def subset_split(split, subset: float | int | None):
    if subset is None or subset == 1.0:
        return split

    n = int(len(split) * subset) if subset <= 1 else int(subset)
    n = max(0, min(len(split), n))
    return split.select(range(n))


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

    return out


def resolve_dataset(name: str, logger: logging.Logger | None = None) -> DatasetDict:
    if logger:
        logger.info(f"Resolving dataset: {name}")

    ds = BUILDERS[name]()

    assert isinstance(ds, DatasetDict), "Expected DatasetDict"
    assert "train" in ds and "test" in ds, "Dataset must have train and test splits"
    return ds


def encode_examples(
    data_cfg: dict,
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    # E5 is trained with a "query: " prefix on every input -- the model card
    # documents unprefixed text as a known way to degrade embedding quality.
    # "query: " (rather than "passage: ") is E5's own recommended choice for
    # symmetric tasks (similarity/clustering), which is what both the
    # selector's cosine-similarity loss and the downstream probes are.
    # Prepended at the id level (not the word list) so it reuses the existing
    # None-word_id -> PAD_TAG handling below, same as any other added token,
    # with no offset bookkeeping needed.
    family = ALIAS_TO_CANON[data_cfg.encoder.family.lower()]
    prefix_ids = tokenizer.encode("query: ", add_special_tokens=False) if family == "e5" else []

    def _encode(example: dict) -> TokenizedExample:
        budget = max(1, data_cfg.max_length - len(prefix_ids))
        enc = tokenizer(
            example["tokens"],
            truncation=True,
            max_length=budget,
            is_split_into_words=True,
        )

        ids = prefix_ids + enc["input_ids"]
        attn_mask = [1] * len(prefix_ids) + enc["attention_mask"]
        word_ids = [None] * len(prefix_ids) + enc.word_ids()

        out = {
            "ids": ids,
            "attn_mask": attn_mask,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "word_ids": word_ids,
        }

        labels = example.get("labels")
        if labels is not None:
            out["labels"] = [
                PAD_TAG if wid is None else str(labels[wid])
                for wid in word_ids
            ]

        return out

    return DatasetDict({
        split: d.map(
            _encode,
            desc=f"Encoding {split} split",
            load_from_cache_file=False,
            remove_columns=d.column_names,
        )
        for split, d in ds.items()
    })


def get_dataset(
    data_cfg: dict,
    runtime_cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger | None = None,
) -> DatasetDict:
    # Keyed by family, not tokenizer_group: e5 shares its underlying
    # tokenizer with sbert (both "bert-base") but now encodes different ids
    # (the "query: " prefix), so they can no longer share one cache file.
    family = ALIAS_TO_CANON[data_cfg.encoder.family.lower()]
    name = canonical_name(data_cfg.dataset)
    path = dataset_path(name, family, int(data_cfg.max_length))

    if path.exists() and not runtime_cfg.rebuild:
        if logger:
            logger.info(f"Loading cached tokenized dataset from {path}")
        return load_from_disk(path)

    if logger:
        logger.info(f"Building + tokenizing dataset: {name}")

    ds_dict = resolve_dataset(name, logger)

    features = ds_dict["train"].features.copy()
    if "labels" in features:
        features["labels"] = Sequence(Value("string"))
    ds_dict = ds_dict.cast(features)

    ds_tok = encode_examples(data_cfg, ds_dict, tokenizer)

    path.parent.mkdir(parents=True, exist_ok=True)
    ds_tok.save_to_disk(path)

    return ds_tok


def strip_special_tokens(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger | None = None,
) -> DatasetDict:
    stripped = {}
    for split, dataset in ds.items():
        stripped_dataset = dataset.map(
            lambda ex: {
                key: [value[i] for i, is_special in enumerate(tokenizer.get_special_tokens_mask(ex["ids"], already_has_special_tokens=True)) if not is_special]
                for key, value in ex.items()
            },
            desc=f"Stripping {split}",
        )
        if logger is not None:
            logger.info("Stripped special tokens from %s split", split)
        stripped[split] = stripped_dataset

    return DatasetDict(stripped)


def initialize_data(
    data_cfg: dict,
    runtime_cfg: dict,
    logger: logging.Logger | None = None,
    device: str = "cpu",
    keep_special: bool = True,
    encoder: SentenceEncoder | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[DataLoader, DataLoader, SentenceEncoder, PreTrainedTokenizerBase, set[str] | None, DatasetDict]:
    if encoder is None or tokenizer is None:
        encoder, tokenizer = build_sentence_encoder(
            family=data_cfg.encoder.family,
            encoder_name=data_cfg.encoder.name,
            device=device,
            pooling=str(data_cfg.encoder.get("pooling", "mean")),
        )

    ds = get_dataset(data_cfg, runtime_cfg, tokenizer, logger)
    ds = shuffle_and_subset(ds, data_cfg.subset, data_cfg.shuffle)

    test_subset = runtime_cfg.get("test_subset", None)
    if test_subset is not None and "test" in ds:
        old_test_len = len(ds["test"])
        ds["test"] = subset_split(ds["test"], test_subset)
        if logger is not None:
            logger.info(
                "Applied runtime.data.test_subset=%s: test split %d -> %d samples",
                test_subset,
                old_test_len,
                len(ds["test"]),
            )

    if not keep_special:
        ds = strip_special_tokens(ds, tokenizer, logger)

    ds_train, ds_test = build_dataloaders(
        ds,
        batch_size=int(runtime_cfg.batch_size),
        num_workers=int(runtime_cfg.num_workers),
        shuffle=bool(data_cfg.shuffle),
        device=device,
    )

    labels_set = None
    if "labels" in ds["train"].column_names:
        labels_set = set(label for sample in ds["train"]["labels"] for label in sample)

    return ds_train, ds_test, encoder, tokenizer, labels_set, ds


def build_dataloaders(
    ds: DatasetDict,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    batch_size = max(1, int(batch_size))

    persistent = num_workers > 0

    ds_train = DataLoader(
        ds["train"],
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        shuffle=shuffle,
        pin_memory=(device == "cuda"),
        persistent_workers=persistent,
    )

    ds_test = DataLoader(
        ds["test"],
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        shuffle=shuffle,
        pin_memory=(device == "cuda"),
        persistent_workers=persistent,
    )

    return ds_train, ds_test
