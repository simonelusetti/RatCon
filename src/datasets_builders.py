from __future__ import annotations

import gzip, csv, re, nltk, random
from collections import defaultdict
from pathlib import Path, List, Tuple, Dict
from typing import Callable
from dora import to_absolute_path
from urllib.request import urlretrieve
from conllu import parse_incr
from tqdm import tqdm
from nltk.corpus import treebank
from transformers import AutoTokenizer

from datasets import (
    Dataset,
    DatasetDict,
    load_from_disk,
    concatenate_datasets,
    load_dataset,
)

RATIONALE_DS_FIELD = {
    "mr": {"tokens": "review", "rationale": "evidences"},
    "twitter": {"tokens": "text", "rationale": "selected_text"},
}

# ============================================================
# Helpers
# ============================================================

def extract_sentences(s: str):
    return [m[0] or m[1] for m in re.findall(r"'(.*?)'|\"(.*?)\"", s)]

    
def find_sublist(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i, i + len(needle)
    return None, None


def _read_csv(
    path: Path, 
    message: str ="Reading rationale csv",
) -> Dataset:
    examples = []
    dataset_name = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=message):
            if not dataset_name:
                if "evidences" in row:
                    dataset_name = "mr"
                elif "selected_text" in row:
                    dataset_name = "twitter"
            words = row[RATIONALE_DS_FIELD[dataset_name]["tokens"]].split()
            rationale = [0] * len(words)
            iter = row[RATIONALE_DS_FIELD[dataset_name]["rationale"]].split()
            if dataset_name == "mr":
                iter = extract_sentences(row[RATIONALE_DS_FIELD[dataset_name]["rationale"]])
            for phrase in iter:
                s, e = find_sublist(words, phrase.split())
                if s is not None:
                    rationale[s:e] = [1] * (e - s)
            examples.append({"tokens": words, "labels": rationale})
    return Dataset.from_list(examples)


# ============================================================
# NLTK POS
# ============================================================

def load_nltk_pos_corpus(
    corpus_name: str,
    corpus_loader: Callable[[], list[list[tuple[str, str]]]],
) -> DatasetDict:
    try:
        tagged = corpus_loader()
        if not tagged:
            raise LookupError
    except LookupError:
        nltk.download(corpus_name)
        tagged = corpus_loader()

    ds = Dataset.from_list([
        (lambda tokens, labels: {"tokens": tokens, "labels": labels})(
            *zip(*sent)
        )
        for sent in tagged
    ])

    return DatasetDict({
        "train": ds,
        "test": ds.select([]),
    })
    
def build_treebank() -> DatasetDict:
    return load_nltk_pos_corpus("treebank", treebank.tagged_sents)

# ============================================================
# CoNLL-2003
# ============================================================

def build_conll2003() -> DatasetDict:
    ds = load_dataset("conll2003").rename_column("ner_tags", "labels")\
        .remove_columns(["id", "pos_tags", "chunk_tags"])
    train_ds = ds["train"]
    test_ds = concatenate_datasets([ds["validation"], ds["test"]])
    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

def map_conll2003_secondary_labels(labels):
    return ["0" if lbl == "0" else "1" for lbl in labels]


# ============================================================
# wikiann
# ============================================================

def build_wikiann() -> DatasetDict:
    ds = load_dataset("wikiann","en").rename_column("ner_tags", "labels")\
        .remove_columns(["spans", "langs"])
    train_ds = ds["train"]
    test_ds = concatenate_datasets([ds["validation"], ds["test"]])
    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

# ============================================================
# ParaSCI
# ============================================================

BASE = Path(to_absolute_path("data/raw/parasci/ParaSCI-master/Data"))
SUBSETS = ["ParaSCI-ACL", "ParaSCI-arXiv"]
SPLITS = ["train", "val", "test"]


def load_pairs(folder: Path, split: str):
    src = folder / split / f"{split}.src"
    tgt = folder / split / f"{split}.tgt"
    if not src.exists():
        return []
    with open(src) as fs, open(tgt) as ft:
        return [(s.strip(), t.strip()) for s, t in zip(fs, ft)]


def build_parasci() -> DatasetDict:
    split_data = {}

    for split in SPLITS:
        rows = []
        for subset in SUBSETS:
            for src, tgt in load_pairs(BASE / subset, split):
                rows.append({"tokens": src.split()})
                rows.append({"tokens": tgt.split(), "labels": [1] * len(tgt.split())})
        split_data[split] = Dataset.from_list(rows)

    return DatasetDict({
        "train": split_data["train"],
        "test": concatenate_datasets([split_data["val"], split_data["test"]]),
    })


def build_parasci_concat() -> DatasetDict:
    split_data = {}

    for split in SPLITS:
        clusters = defaultdict(list)
        for subset in SUBSETS:
            for src, tgt in load_pairs(BASE / subset, split):
                clusters[src].append(tgt)

        split_data[split] = Dataset.from_list([
            {"tokens": " || ".join([src] + tgts).split()}
            for src, tgts in clusters.items()
        ])

    return DatasetDict({
        "train": split_data["train"],
        "test": concatenate_datasets([split_data["val"], split_data["test"]]),
    })


def build_both_parasci() -> tuple[DatasetDict, DatasetDict]:
    return build_parasci(), build_parasci_concat()


# ============================================================
# CoNLL-2000
# ============================================================

def _parse_conll2000(path: Path) -> Dataset:
    sentences, tokens, labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                tokens, labels = [], []
                continue
            word, _, label = line.split()
            tokens.append(word)
            labels.append(label)
    if tokens:
        sentences.append((tokens, labels))
    final_sentences = {
        "tokens": [x[0] for x in sentences],
        "labels": [x[1] for x in sentences],
    }
    return Dataset.from_dict(final_sentences)


def build_conll2000() -> DatasetDict:
    try:
        dataset = load_dataset("conll2000").rename_column("chunk_tags", "labels")\
            .remove_columns(["id", "pos_tags", "ner_tags"])
        train_ds = dataset["train"]
        test_ds = dataset["test"]
    except Exception as e:
        base_url = "https://www.clips.uantwerpen.be/conll2000/chunking/"
        raw_root = Path(to_absolute_path("./data/raw/conll2000"))
        raw_root.mkdir(parents=True, exist_ok=True)
        def ensure(fname):
            gz = raw_root / fname
            txt = raw_root / fname.replace(".gz", "")
            if not gz.exists():
                urlretrieve(base_url + fname, gz)
            if not txt.exists():
                with gzip.open(gz, "rt") as f_in, open(txt, "w") as f_out:
                    f_out.write(f_in.read())
            return txt
        train_ds = _parse_conll2000(ensure("train.txt.gz"))
        test_ds = _parse_conll2000(ensure("test.txt.gz"))

    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })


# ============================================================
# Movie Rationales (labels = rationale)
# ============================================================


def build_movie_reviews() -> DatasetDict:
    root = Path(to_absolute_path("./data/raw/movie_rationales"))

    train_ds = _read_csv(root / "train.csv", message="Reading train rationale csv")
    val_ds = _read_csv(root / "validation.csv", message="Reading validation rationale csv")
    test_ds = _read_csv(root / "test.csv", message="Reading test rationale csv")

    return DatasetDict({
        "train": train_ds,
        "test": concatenate_datasets([val_ds, test_ds]),
    })


# ============================================================
# Twitter / Sentiment
# ============================================================

def build_twitter() -> DatasetDict:
    root = Path(to_absolute_path("./data/raw/twitter"))

    full_ds = _read_csv(root / "train.csv", message="Reading train twitter csv")
    
    ds_split = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = ds_split["train"]
    test_ds  = ds_split["test"]
    
    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })


# ============================================================
# UD English
# ============================================================

def _is_core_np(upos, deprel):
    # Core NP heads: NOUN, PROPN, some PRON
    if upos in {"NOUN", "PROPN"}:
        return True
    if upos == "PRON" and deprel not in {"nsubj:relcl", "obj:relcl"}:
        return True
    return False


def _chunk_ud_labels(tokens, upos, heads, deprel):
    n = len(tokens)
    labels = ["O"] * n

    # build children index
    children = {i: [] for i in range(n)}
    for i, h in enumerate(heads):
        if h > 0:
            children[h - 1].append(i)

    # collect NP spans as sets of indices
    np_spans = []

    for i in range(n):
        if not _is_core_np(upos[i], deprel[i]):
            continue
        span = {i}
        queue = [i]
        while queue:
            h = queue.pop()
            for c in children.get(h, []):
                rel = deprel[c]
                c_upos = upos[c]
                h_upos = upos[h]

                attach = False

                if rel in {
                    "compound", "compound:prt", "flat", "flat:name",
                    "goeswith", "fixed", "nummod"
                }:
                    attach = True
                elif rel == "det" and c < h:
                    attach = True
                elif rel == "amod" and c_upos != "ADV":
                    attach = True
                elif rel == "conj" and c_upos == "ADJ" and h_upos == "ADJ":
                    attach = True
                elif rel == "appos" and abs(c - h) == 1:
                    attach = True
                elif rel == "advmod" and c_upos not in {"PART", "VERB"} and h_upos == "ADJ":
                    attach = True
                elif rel == "nmod:poss" and c_upos not in {"NOUN", "PROPN"}:
                    attach = True
                elif rel in {"obl:npmod", "obl:tmod"}:
                    attach = True
                elif rel == "obl":
                    # head must itself be adjectival modifier
                    if deprel[h] == "amod":
                        attach = True
                if attach and c not in span:
                    span.add(c)
                    queue.append(c)

        # attach intervening tokens
        min_i, max_i = min(span), max(span)
        for j in range(min_i + 1, max_i):
            span.add(j)
        np_spans.append(span)

    # merge overlapping spans (safety)
    merged = []
    for span in sorted(np_spans, key=lambda s: min(s)):
        if not merged or min(span) > max(merged[-1]):
            merged.append(span)
        else:
            merged[-1] |= span

    # BIO encoding
    for span in merged:
        indices = sorted(span)
        labels[indices[0]] = "B-NP"
        for j in indices[1:]:
            labels[j] = "I-NP"

    return labels


def build_ud() -> DatasetDict:
    base_url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"
    raw_root = Path("./data/raw/ud")
    raw_root.mkdir(parents=True, exist_ok=True)

    def load(fname):
        path = raw_root / fname
        if not path.exists():
            urlretrieve(base_url + fname, path)

        rows = []
        with open(path, encoding="utf-8") as f:
            for sent in parse_incr(f):
                cleaned_sent = [
                    tok for tok in sent
                    if isinstance(tok["id"], int)   # removes MWTs like "1-2"
                ]
                
                assert all(isinstance(tok["id"], int) for tok in cleaned_sent)
                assert all(tok["head"] is None or isinstance(tok["head"], int) for tok in cleaned_sent)

                tokens  = [tok["form"]   for tok in cleaned_sent]
                upos    = [tok["upos"]   for tok in cleaned_sent]
                heads   = [tok["head"]   for tok in cleaned_sent]
                deprel  = [tok["deprel"] for tok in cleaned_sent]

                chunk_lables = _chunk_ud_labels(tokens, upos, heads, deprel)

                rows.append({
                    "tokens": tokens,
                    "labels": chunk_lables,
                })

        return Dataset.from_list(rows)

    train_ds = load("en_ewt-ud-train.conllu")
    dev_ds   = load("en_ewt-ud-dev.conllu")
    test_ds  = load("en_ewt-ud-test.conllu")
    test_ds = concatenate_datasets([dev_ds, test_ds])

    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })


# ============================================================
# Shape dataset
# ============================================================

def _word_shape(w,r):
    if random.random() < r:
        return w, "True"
    return "".join("X" if c.isupper() else "x" if c.isalpha() else c for c in w), "False"

def _apply_shape(example, rate):
    tokens = example["tokens"]
    new_tokens, changed = [], []
    for w in tokens.split():
        w_new, c = _word_shape(w, rate)
        new_tokens.append(w_new)
        changed.append(c)
        
    example["tokens"] = new_tokens
    example["labels"] = changed
    return example

def build_shape(cfg: dict, tokenizer: AutoTokenizer | None = None) -> DatasetDict:
    random.seed(cfg.get("seed", 42))

    original = cfg["original"]
    mask_rate = cfg["rate"]
    if mask_rate > 1.0:
        mask_rate = mask_rate = float(cfg["rate"]) / 100.0
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    ds = load_from_disk(to_absolute_path(f"./data/cache/{original}"))
    ds = ds.map(
        lambda x: {
            "tokens": tokenizer.convert_tokens_to_string(x["tokens"])
            .replace("[CLS] ", "")
            .replace(" [SEP]", "")
        }
    )
    ds = ds.map(lambda ex: _apply_shape(ex, mask_rate))
    ds = ds["train"].train_test_split(test_size=0.2, seed=42)
                
    return ds

# ============================================================
# WikiANN with entity swapping
# ============================================================

def extract_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    i = 0
    n = len(labels)

    while i < n:
        lbl = labels[i]
        if lbl.startswith("B-"):
            typ = lbl[2:]
            j = i + 1
            while j < n and labels[j] == f"I-{typ}":
                j += 1
            spans.append((i, j, typ))
            i = j
        else:
            i += 1

    return spans


def build_entity_bank(dataset: Dataset) -> Dict[str, List[List[str]]]:
    bank = defaultdict(list)

    for ex in dataset:
        tokens = ex["tokens"]
        labels = ex["labels"]

        for i, j, typ in extract_spans(tokens, labels):
            span_tokens = tokens[i:j]
            if span_tokens:  # safety
                bank[typ].append(span_tokens)

    return {t: mentions for t, mentions in bank.items() if len(mentions) >= 2}


def choose_replacement(candidates: List[List[str]],original: List[str],rng: \
    random.Random, max_tries: int = 20,) -> List[str]:
    if len(candidates) == 0:
        return original
    if len(candidates) == 1:
        return candidates[0]
    repl = candidates[rng.randrange(len(candidates))]
    i = 0
    while repl == original and i < max_tries:
        repl = candidates[rng.randrange(len(candidates))]
        i += 1
    return repl


def swap_entities(example: dict, bank: Dict[str, List[List[str]]], rng: random.Random) -> dict:
    tokens = example["tokens"]
    labels = example["labels"]

    spans = extract_spans(tokens, labels)
    if not spans:
        return example

    new_tokens = []
    new_labels = []
    cursor = 0

    for i, j, typ in spans:
        new_tokens.extend(tokens[cursor:i])
        new_labels.extend(labels[cursor:i])

        candidates = bank.get(typ)
        if not candidates:
            replacement = tokens[i:j]
        else:
            replacement = choose_replacement(candidates, tokens[i:j], rng)

        new_tokens.extend(replacement)
        new_labels.append(f"B-{typ}")
        for _ in range(1, len(replacement)):
            new_labels.append(f"I-{typ}")

        cursor = j

    new_tokens.extend(tokens[cursor:])
    new_labels.extend(labels[cursor:])

    return {"tokens": new_tokens, "labels": new_labels}


def build_wikiann_swap(seed: int = 67) -> DatasetDict:
    rng = random.Random(seed)
    
    ds = load_dataset("wikiann", "en") \
        .rename_column("ner_tags", "labels") \
        .remove_columns(["spans", "langs"])
    train_ds = ds["train"]
    
    test_ds = concatenate_datasets([ds["validation"], ds["test"]])
    base = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })
    
    bank = build_entity_bank(base["train"])

    def _swap(ex):
        return swap_entities(ex, bank, rng)

    swapped = DatasetDict({
        "train": base["train"].map(_swap),
        "test": base["test"].map(_swap),
    })

    return swapped
