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
