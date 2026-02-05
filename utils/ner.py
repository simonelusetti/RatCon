import sys
import torch
import numpy as np
from typing import Callable, Dict, List
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from pathlib import Path
from dora import hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selector import RationaleSelectorModel
from src.sentence import (
    build_sentence_encoder,
    SentenceEncoder,
    FrozenLLMEncoder,
)

def is_entity_label(label_id: int, id2label: Dict[int, str]) -> int:
    return 0 if id2label[int(label_id)] == "O" else 1


class NERDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length: int = 256):
        self.data = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        ner_feat = self.data.features["ner_tags"].feature
        self.id2label = {i: n for i, n in enumerate(ner_feat.names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ex = self.data[idx]
        words = ex["tokens"]
        word_labels = ex["ner_tags"]

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0).float()
        word_ids = enc.word_ids(batch_index=0)

        labels_tok = []
        for wi in word_ids:
            if wi is None:
                labels_tok.append(-100)
            else:
                labels_tok.append(is_entity_label(word_labels[wi], self.id2label))

        labels_tok = torch.tensor(labels_tok, dtype=torch.long)

        return {
            "ids": input_ids,
            "attn": attn,
            "labels": labels_tok,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    def pad(key: str, dtype: torch.dtype, padding_value):
        return pad_sequence(
            [x[key].to(dtype) for x in batch],
            batch_first=True,
            padding_value=padding_value,
        )

    return {
        "ids": pad("ids", torch.long, 0),
        "attn": pad("attn", torch.float, 0.0),
        "labels": pad("labels", torch.long, -100),
    }


@torch.no_grad()
def compute_gates(
    encoder: SentenceEncoder,
    selector: RationaleSelectorModel | Callable,
    ids: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    token_emb = encoder.token_embeddings(ids, attn)
    if isinstance(selector, RationaleSelectorModel):
        _, g, _ = selector(token_emb, attn)
        return g
    else:
        return selector(token_emb, attn)


@torch.no_grad()
def evaluate_unsupervised_ner(
    encoder: SentenceEncoder,
    selector: RationaleSelectorModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    tp = fp = fn = tn = 0

    for batch in tqdm(dataloader):
        ids = batch["ids"].to(device)
        attn = batch["attn"].to(device)
        labels = batch["labels"].to(device)

        g = compute_gates(encoder, selector, ids, attn)
        preds = g.long()

        valid = (labels != -100) & (attn > 0)
        y = labels[valid]
        yhat = preds[valid]

        tp += int(((yhat == 1) & (y == 1)).sum())
        fp += int(((yhat == 1) & (y == 0)).sum())
        fn += int(((yhat == 0) & (y == 1)).sum())
        tn += int(((yhat == 0) & (y == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@hydra_main(config_path="retrieval_conf", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    device = cfg.runtime.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    encoder, tokenizer = build_sentence_encoder(
        family=cfg.model.encoder.family,
        encoder_name=cfg.model.encoder.name,
        device=str(device),
    )

    dataset_name = getattr(cfg, "data", {}).get("dataset_name", "wikiann")
    split_name = getattr(cfg, "data", {}).get("split", "test")
    max_length = int(cfg.eval.get("max_length", 256))

    ds = load_dataset(dataset_name, "en", split=split_name)
    ner_ds = NERDataset(ds, tokenizer, max_length=max_length)

    loader = DataLoader(
        ner_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        sample = ner_ds[0]
        ids0 = sample["ids"].unsqueeze(0).to(device)
        attn0 = sample["attn"].unsqueeze(0).to(device)
        model_dim = encoder.token_embeddings(ids0, attn0).shape[-1]

    selector = RationaleSelectorModel(model_dim).to(device)
    state = torch.load(cfg.eval.checkpoint, map_location=device)
    selector.load_state_dict(state["model"], strict=False)
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False

    metrics = evaluate_unsupervised_ner(
        encoder=encoder,
        selector=selector,
        dataloader=loader,
        device=device,
    )

    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} tn={metrics['tn']}")


if __name__ == "__main__":
    main()
