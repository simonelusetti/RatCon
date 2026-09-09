from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)


def find_sublist(haystack: list[str], needle: list[str]) -> tuple[int | None, int | None]:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i, i + len(needle)
    return None, None


def build_conll2003() -> DatasetDict:
    # datasets>=4 no longer executes Hub dataset scripts, which made the old
    # load_dataset("conll2003") path fail before an experiment could start.
    # The Hub's auto-converted parquet revision carries the same three splits
    # and ClassLabel metadata without requiring script execution.
    from huggingface_hub import hf_hub_download

    files = {
        split: hf_hub_download(
            "eriktks/conll2003",
            f"conll2003/{split}/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
        )
        for split in ("train", "validation", "test")
    }
    ds = load_dataset("parquet", data_files=files).rename_column("ner_tags", "labels")\
        .remove_columns(["id", "pos_tags", "chunk_tags"])
    train_ds = ds["train"]
    test_ds = concatenate_datasets([ds["validation"], ds["test"]])
    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })


def build_wikiann() -> DatasetDict:
    ds = load_dataset("wikiann", "en", trust_remote_code=True).rename_column("ner_tags", "labels")\
        .remove_columns(["spans", "langs"])
    train_ds = ds["train"]
    test_ds = concatenate_datasets([ds["validation"], ds["test"]])
    return DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })


def build_conll2000() -> DatasetDict:
    # nltk ships the CoNLL-2000 shared-task data; the HF hub script and the
    # original clips.uantwerpen.be download URL have both gone stale.
    import nltk
    from nltk.corpus import conll2000

    try:
        conll2000.fileids()
    except LookupError:
        nltk.download("conll2000")

    def build(fileid: str) -> Dataset:
        return Dataset.from_list([
            {
                "tokens": [w for w, _, _ in sent],
                "labels": [tag for _, _, tag in sent],
            }
            for sent in conll2000.iob_sents(fileid)
        ])

    return DatasetDict({
        "train": build("train.txt"),
        "test": build("test.txt"),
    })


def build_stsb() -> DatasetDict:
    # sentence-transformers/stsb (not raw GLUE stsb): GLUE hides its test-split
    # gold labels for leaderboard submission, so its "test" is unusable here;
    # this HF mirror carries real scores in all three original splits.
    ds = load_dataset("sentence-transformers/stsb")
    train_ds = ds["train"]
    test_ds = concatenate_datasets([ds["validation"], ds["test"]])

    def melt(split: Dataset) -> Dataset:
        # Selector training is label-free and needs nothing but a bag of
        # sentences, so each pair becomes two rows (one per side). pair_id/
        # sentence_role/score are carried through as extra columns for a
        # future STS-B evaluation script to reconstruct pairs from -- note
        # get_dataset()'s tokenization step (encode_examples) currently drops
        # any column it doesn't itself produce, so these do not yet survive
        # into the cached tokenized dataset; wiring that through is separate
        # follow-up work, not needed for training alone.
        rows = [
            {"tokens": sentence.split(), "pair_id": i, "sentence_role": role, "score": example["score"]}
            for i, example in enumerate(split)
            for role, sentence in enumerate((example["sentence1"], example["sentence2"]))
        ]
        return Dataset.from_list(rows)

    return DatasetDict({
        "train": melt(train_ds),
        "test": melt(test_ds),
    })


def build_movie_reviews() -> DatasetDict:
    # The script-based movie_rationales repo no longer loads with pinned
    # datasets<3; its auto-converted parquet export carries the same data.
    from huggingface_hub import hf_hub_download

    files = {
        split: hf_hub_download(
            "eraser-benchmark/movie_rationales",
            f"default/{split}/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
        )
        for split in ("train", "validation", "test")
    }
    raw = load_dataset("parquet", data_files=files)

    def to_token_rationales(example: dict) -> dict:
        words = example["review"].split()
        rationale = [0] * len(words)
        for phrase in example["evidences"]:
            s, e = find_sublist(words, phrase.split())
            if s is not None:
                rationale[s:e] = [1] * (e - s)
        return {"tokens": words, "labels": rationale}

    mapped = raw.map(
        to_token_rationales,
        remove_columns=["review", "label", "evidences"],
        desc="Aligning rationales",
    )

    return DatasetDict({
        "train": mapped["train"],
        "test": concatenate_datasets([mapped["validation"], mapped["test"]]),
    })


# Universal Dependencies English EWT. Two label sets over the SAME sentences:
# upos (17 universal POS tags) and deprel (the syntactic relation each word
# bears to its head). Holding the corpus fixed and varying only the labelling
# is what isolates the abstraction axis -- measured on the training split,
# H(label | word) / H(label) is 0.08 for upos and 0.26 for deprel, so upos is
# almost entirely determined by the word itself while deprel needs context.
#
# Both are far better balanced than this repo's NER corpora: H/log2(tags) is
# 0.89 and 0.84, against wikiann's 0.79 over only 7 tags, and neither has an
# "O" class swallowing half the tokens. That matters because the grounding
# correlation runs over tags, so usable tag count sets its statistical power
# (14 and 29 usable tags here vs wikiann's 6).
#
# The Hub repo ships native parquet, so this needs no script execution and is
# unaffected by the datasets>=4 change that broke the conll2003 loader.
UD_EWT_REPO = "universal-dependencies/universal_dependencies"


def _load_ud_ewt() -> DatasetDict:
    from huggingface_hub import hf_hub_download

    files = {
        split: hf_hub_download(UD_EWT_REPO, f"parquet/en_ewt/{split}.parquet", repo_type="dataset")
        for split in ("train", "dev", "test")
    }
    return load_dataset("parquet", data_files=files)


def _ud_with_labels(column: str) -> DatasetDict:
    """One UD annotation layer, verbatim: {tokens, labels} and nothing else.

    Deliberately no relabelling. An earlier version collapsed deprel subtypes
    (nsubj:pass -> nsubj) to shrink the label set, which turned out to be
    strictly worse: measured on the training split it drops usable tags (>=0.5%
    of tokens) from 29 to 24 and raises the |r| needed for p<0.05 from 0.367 to
    0.404, while leaving H(label|word)/H(label) unchanged at 0.26. It cost
    statistical power and bought nothing, so the raw UD labels are used.
    """
    ds = _load_ud_ewt()

    def to_labels(example: dict) -> dict:
        return {"tokens": example["tokens"], "labels": example[column]}

    mapped = ds.map(to_labels, remove_columns=ds["train"].column_names,
                    desc=f"Extracting UD {column}")
    return DatasetDict({
        "train": mapped["train"],
        "test": concatenate_datasets([mapped["dev"], mapped["test"]]),
    })


def build_ud_upos() -> DatasetDict:
    return _ud_with_labels("upos")


def build_ud_deprel() -> DatasetDict:
    return _ud_with_labels("deprel")


# UD English GUM, RST discourse relations. GUM is multilayer: the same tokens
# carry syntax, entities/coreference and Rhetorical Structure Theory, and UD's
# parquet packs the non-syntactic layers into the `misc` column. This reads the
# RST layer out of it.
#
# Under RST a text is cut into elementary discourse units (EDUs, roughly
# clauses) and each is labelled with the rhetorical job it does relative to
# what it attaches to -- conceding, justifying, giving evidence, evaluating.
# That is a property of the span's role in the argument, not of its words, so
# it sits at the far end of the abstraction axis: H(label|word)/H(label) is
# 0.86, against 0.26 for deprel and 0.08 for upos.
#
# It also happens to be the best balanced corpus here (H/log2(tags) = 0.90,
# largest class 11.3%) because every token belongs to some EDU -- there is no
# "O" class swallowing the distribution the way there is in NER.
def _gum_discourse_labels(example: dict) -> list[str]:
    """Forward-fill each EDU's relation across the tokens it covers.

    `Discourse=` is written once, on the EDU's first token, as
    `Discourse=<relation>:<edu>-><parent>:...`; the tokens after it belong to
    the same EDU until the next marker. Verified across all three splits:
    every sentence carries a marker on its first token, so no token is left
    without a relation and nothing has to be carried across sentences.

    The value is taken verbatim -- `_m` marks a multinuclear relation (joint,
    contrast: coordinate units rather than nucleus/satellite) and `ROOT` is the
    top of a document's tree. Both are real distinctions, and an earlier
    version of this parser silently dropped them with a too-narrow regex.
    """
    labels: list[str] = []
    current: str | None = None
    for misc in example["misc"]:
        if misc and "Discourse=" in misc:
            current = misc.split("Discourse=")[1].split("|")[0].split(":")[0]
        labels.append(current)
    return labels


def build_ud_discourse() -> DatasetDict:
    from huggingface_hub import hf_hub_download

    files = {
        split: hf_hub_download(UD_EWT_REPO, f"parquet/en_gum/{split}.parquet", repo_type="dataset")
        for split in ("train", "dev", "test")
    }
    ds = load_dataset("parquet", data_files=files)
    mapped = ds.map(
        lambda ex: {"tokens": ex["tokens"], "labels": _gum_discourse_labels(ex)},
        remove_columns=ds["train"].column_names,
        desc="Extracting GUM RST relations",
    )
    return DatasetDict({
        "train": mapped["train"],
        "test": concatenate_datasets([mapped["dev"], mapped["test"]]),
    })
