import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
from numpy import linspace

from .utils import should_disable_tqdm
from .retrival_fun import build_non_special_mask, get_rhos, build_selector_mask_generator, build_random_mask_generator


# entailment=1.0, neutral=0.5, contradiction=0.0 — mirrors STS-B's continuous scores
_NLI_LABEL_TO_SCORE = {0: 1.0, 1: 0.5, 2: 0.0}


class NLIDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = [
            item for item in hf_dataset
            if item["label"] in _NLI_LABEL_TO_SCORE
        ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "premise": item["premise"],
            "hypothesis": item["hypothesis"],
            "label": _NLI_LABEL_TO_SCORE[item["label"]],
        }


def build_nli_collate(tokenizer, device, max_length, keep_special: bool = True):
    def strip_special(toks):
        if keep_special:
            return toks

        ids = toks["input_ids"]
        attn = toks["attention_mask"]
        kept_ids, kept_attn = [], []
        for ids_i, attn_i in zip(ids, attn):
            keep = ~torch.tensor(
                tokenizer.get_special_tokens_mask(ids_i.tolist(), already_has_special_tokens=True),
                dtype=torch.bool,
            )
            keep &= attn_i.bool()
            kept_ids.append(ids_i[keep])
            kept_attn.append(attn_i[keep])

        return {
            "input_ids": pad_sequence(kept_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": pad_sequence(kept_attn, batch_first=True, padding_value=0),
        }

    def collate_fn(batch):
        premises = [x["premise"] for x in batch]
        hypotheses = [x["hypothesis"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)

        t1 = tokenizer(
            premises,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        t2 = tokenizer(
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        t1 = strip_special(t1)
        t2 = strip_special(t2)

        return {
            "t1": {k: v.to(device) for k, v in t1.items()},
            "t2": {k: v.to(device) for k, v in t2.items()},
            "labels": labels,
        }

    return collate_fn


@torch.no_grad()
def eval_nli_baseline(loader, encoder):
    sims, labels = [], []

    for batch in tqdm(
        loader,
        desc="NLI baseline",
        leave=False,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
        file=sys.stderr,
    ):
        t1, t2 = batch["t1"], batch["t2"]

        e1 = encoder.token_embeddings(t1["input_ids"], t1["attention_mask"])
        e2 = encoder.token_embeddings(t2["input_ids"], t2["attention_mask"])

        z1 = encoder.pool(e1, t1["attention_mask"])
        z2 = encoder.pool(e2, t2["attention_mask"])

        sims.extend(F.cosine_similarity(z1, z2).cpu().tolist())
        labels.extend(batch["labels"].tolist())

    return float(spearmanr(sims, labels)[0])


@torch.no_grad()
def eval_nli_sweep(
    loader,
    encoder,
    mask_generator,
    eval_cfg,
    desc: str = "NLI selector",
    progress_bar=None,
):
    rhos = get_rhos(eval_cfg)

    sims = {rho: [] for rho in rhos}
    labels = []

    iterator = loader
    if progress_bar is None:
        iterator = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=should_disable_tqdm(),
            file=sys.stderr,
        )

    for batch in iterator:
        t1, t2 = batch["t1"], batch["t2"]

        a2 = t2["attention_mask"]
        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        z2 = encoder.pool(e2_full, a2)

        a1 = t1["attention_mask"]
        new_a1_sweep = mask_generator(t1, a1, rhos)

        for rho, new_a1 in zip(rhos, new_a1_sweep):
            e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
            z1 = encoder.pool(e1_masked, new_a1)
            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["labels"].tolist())
        if progress_bar is not None:
            progress_bar.update(1)

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}


@torch.no_grad()
def eval_nli_random_sweep(loader, encoder, tokenizer, eval_cfg, device, keep_special: bool = True):
    rhos = get_rhos(eval_cfg)
    runs = eval_cfg.random_selector.runs

    acc = {rho: [] for rho in rhos}
    cached = []

    for batch in loader:
        t1, t2 = batch["t1"], batch["t2"]
        a2 = t2["attention_mask"]
        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        cached.append((t1, encoder.pool(e2_full, a2), batch["labels"]))

    with tqdm(
        total=runs * len(cached),
        desc="NLI random selector",
        leave=False,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
        file=sys.stderr,
    ) as pbar:
        for run in range(runs):
            torch.manual_seed(eval_cfg.random_selector.seed + run)
            rand_mask_gen = build_random_mask_generator(
                eval_cfg,
                tokenizer,
                device,
                keep_special=keep_special,
            )
            sims = {rho: [] for rho in rhos}
            labels = []

            for t1, z2, batch_labels in cached:
                a1 = t1["attention_mask"]
                new_a1_sweep = rand_mask_gen(t1, a1, rhos)

                for rho, new_a1 in zip(rhos, new_a1_sweep):
                    e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
                    z1 = encoder.pool(e1_masked, new_a1)
                    sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

                labels.extend(batch_labels.tolist())
                pbar.update(1)

            out = {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}
            for rho in rhos:
                acc[rho].append(out[rho])

    return {rho: sum(v) / len(v) for rho, v in acc.items()}


@torch.no_grad()
def run_nli_sweep(cfg, device, encoder, tokenizer, selector):
    hf_ds = load_dataset("stanfordnlp/snli", split="validation")
    ds = NLIDataset(hf_ds)

    collate_fn = build_nli_collate(
        tokenizer,
        device,
        cfg.data.max_length,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.runtime.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    base = eval_nli_baseline(loader, encoder)

    keep_special = bool(cfg.model.get("keep_special", True))
    selector_mask_gen = build_selector_mask_generator(
        selector,
        encoder,
        tokenizer,
        device,
        hard=bool(cfg.runtime.eval.get("hard", False)),
        keep_special=keep_special,
    )

    ours = eval_nli_sweep(loader, encoder, selector_mask_gen, cfg.runtime.eval)
    rand = eval_nli_random_sweep(
        loader,
        encoder,
        tokenizer,
        cfg.runtime.eval,
        device,
        keep_special=keep_special,
    )

    return base, ours, rand
