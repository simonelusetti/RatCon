"""Train-once, load-thereafter NER probe over a frozen encoder.

Why this is a cache and not an experiment: the probe reads *word-level token
embeddings* and never pools, so neither the pooling strategy nor rho nor the
selector itself can change its answer. One probe therefore serves every
selector built on the same token encoder, and the only thing that identifies
it is (dataset, encoder family). Seed is in the key purely so the grounding
figures can still show a spread across independently trained probes.

    outputs/ner/<dataset>/<family>/seed<k>/{model.pth,report.json}

report.json holds the three views the plot scripts read -- span level
(seqeval), token level (per-tag), and a binary entity/O collapse -- plus the
per-epoch `history` the B/I convergence figure needs. There is one model per
key and no per-epoch checkpoints: the probe is cheap to retrain and nothing
downstream resumes it.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import torch
from seqeval.metrics import classification_report as span_classification_report
from sklearn.metrics import classification_report as flat_classification_report
from tqdm import tqdm

from src.data import LABEL_DISPLAY_NAMES, PAD_TAG, canonical_name, initialize_data
from src.utils import to_device

from .model import MLPTagger, gather_word_level

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "outputs" / "ner"

TAGGED_DATASETS = {"wikiann", "conll2003", "conll2000", "movie_rationales"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_dir(dataset: str, family: str, seed: int) -> Path:
    return STORE / canonical_name(dataset) / family / f"seed{seed}"


def load(dataset: str, family: str) -> list[dict]:
    """Every cached report for this key, seed order. Empty if none exist."""
    root = STORE / canonical_name(dataset) / family
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("seed*/report.json"),
                           key=lambda p: int(p.parent.name.removeprefix("seed")))
    ]


def tag_names(dataset: str) -> list[str]:
    name = canonical_name(dataset)
    if name not in TAGGED_DATASETS:
        raise ValueError(
            f"The NER probe needs a per-token-labeled dataset "
            f"({', '.join(sorted(TAGGED_DATASETS))}), got {dataset!r}."
        )
    tag_map = LABEL_DISPLAY_NAMES[name]
    return [tag_map[str(i)] for i in range(len(tag_map))]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def build_reports(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    flat_true = [tag for seq in y_true for tag in seq]
    flat_pred = [tag for seq in y_pred for tag in seq]
    # The binary view needs the joint confusion matrix -- a B-PER predicted
    # as B-ORG is correct here but wrong per-tag -- so it cannot be derived
    # from the other two after the fact.
    binarize = lambda tags: ["entity" if t != "O" else "O" for t in tags]
    return {
        "span_level": span_classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "token_level": flat_classification_report(flat_true, flat_pred, output_dict=True, zero_division=0),
        "binary_entity_level": flat_classification_report(
            binarize(flat_true), binarize(flat_pred), output_dict=True, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _Probe:
    """One MLP over one frozen encoder. Lives only as long as a train call."""

    def __init__(self, cfg, encoder, tags: list[str], train_dl, test_dl, device: str):
        self.encoder, self.tags, self.device = encoder, tags, device
        self.train_dl, self.test_dl = train_dl, test_dl
        self.quiet = bool(cfg.runtime.eval.get("short_log", False)) or bool(cfg.runtime.get("grid", False))

        # Raw label values are stringified indices for ClassLabel datasets
        # (wikiann/conll2003) and the tag itself for conll2000, whose NLTK
        # loader has no ClassLabel. Accept both rather than assuming int().
        self.label_to_idx = {str(i): i for i in range(len(tags))}
        self.label_to_idx.update({name: i for i, name in enumerate(tags)})

        encoder.to(device).eval().requires_grad_(False)
        with torch.no_grad():
            first = next(iter(train_dl))
            dim = encoder.token_embeddings(
                first["ids"].to(device), first["attn_mask"].to(device)).shape[-1]

        ner_cfg = cfg.get("ner", {})
        self.model = MLPTagger(dim, num_tags=len(tags),
                               hidden=int(ner_cfg.get("hidden", 256)),
                               dropout=float(ner_cfg.get("dropout", 0.1))).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(cfg.model.optim.lr),
            weight_decay=float(cfg.model.optim.weight_decay),
            betas=tuple(cfg.model.optim.betas))
        self.weights = self._class_weights() if bool(ner_cfg.get("class_weighted", False)) else None

    def _class_weights(self) -> torch.Tensor:
        """sklearn's "balanced" weights: total / (num_classes * count[c]).

        Guards against a skewed corpus (movie_rationales is ~5.4:1)
        collapsing the probe onto the majority class.
        """
        counts = Counter()
        for seq in self.train_dl.dataset["labels"]:
            for value in seq:
                if value != PAD_TAG:
                    counts[self.label_to_idx[value]] += 1
        total = sum(counts.values())
        return torch.tensor(
            [total / (len(self.tags) * counts.get(i, 1)) for i in range(len(self.tags))],
            dtype=torch.float32, device=self.device)

    def _forward(self, batch: dict):
        label_ids = torch.tensor(
            [[-1 if v == PAD_TAG else self.label_to_idx[v] for v in seq] for seq in batch["labels"]],
            dtype=torch.long, device=self.device)
        with torch.no_grad():
            token_emb = self.encoder.token_embeddings(batch["ids"], batch["attn_mask"])
        word_emb, word_mask, word_labels = gather_word_level(token_emb, batch["word_ids"], label_ids)
        emissions = self.model(word_emb, word_mask)
        return emissions, word_mask, word_labels, self.model.loss(
            emissions, word_labels, word_mask, weight=self.weights)

    @torch.no_grad()
    def evaluate(self) -> tuple[list[list[str]], list[list[str]]]:
        self.model.eval()
        y_true: list[list[str]] = []
        y_pred: list[list[str]] = []
        for batch in tqdm(self.test_dl, desc="eval", leave=False, dynamic_ncols=True,
                          disable=self.quiet, file=sys.stderr):
            batch = to_device(self.device, batch)
            emissions, word_mask, word_labels, _ = self._forward(batch)
            decoded = self.model.decode(emissions, word_mask)
            gold = word_labels.cpu().tolist()
            for i, length in enumerate(word_mask.sum(dim=1).tolist()):
                y_true.append([self.tags[gold[i][t]] for t in range(length)])
                y_pred.append([self.tags[tag] for tag in decoded[i]])
        return y_true, y_pred

    def fit(self, epochs: int) -> list[dict]:
        """Train for `epochs`, returning the per-epoch report history."""
        history = []
        for epoch in tqdm(range(1, epochs + 1), desc="probe", dynamic_ncols=True,
                          disable=self.quiet, file=sys.stderr):
            self.model.train()
            total, seen = 0.0, 0
            for batch in tqdm(self.train_dl, desc=f"epoch {epoch}", leave=False,
                              dynamic_ncols=True, disable=self.quiet, file=sys.stderr):
                batch = to_device(self.device, batch)
                self.optimizer.zero_grad(set_to_none=True)
                *_, loss = self._forward(batch)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * batch["ids"].size(0)
                seen += batch["ids"].size(0)
            reports = build_reports(*self.evaluate())
            history.append({"epoch": epoch, "train_loss": total / max(1, seen), **reports})
            log.info("epoch %d/%d train_loss=%.4f entity_f1=%.4f", epoch, epochs,
                     history[-1]["train_loss"],
                     reports["binary_entity_level"]["entity"]["f1-score"])
        return history


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def performances(cfg, dataset: str | None = None, seeds=(0,), retrain: bool = False) -> list[dict]:
    """Probe performance on `dataset`, training and saving only what is missing.

    Returns one report per seed. `cfg` supplies the encoder, optimiser and
    runtime settings; `dataset` overrides cfg.data.dataset so the caller can
    ask about a corpus it is not itself training on.
    """
    dataset = canonical_name(dataset or cfg.data.dataset)
    family = str(cfg.data.encoder.family)
    tags = tag_names(dataset)

    wanted = [s for s in seeds
              if retrain or not (store_dir(dataset, family, s) / "report.json").exists()]
    if wanted:
        # Built once and shared: loading the encoder and tokenising the corpus
        # dominates the cost of an epoch, and neither depends on the seed.
        data_cfg = cfg.data.copy()
        data_cfg.dataset = dataset
        # Pooling cannot reach a probe (it reads token embeddings and never
        # pools), so it is pinned rather than inherited -- otherwise the same
        # cached probe would appear to depend on the caller's strategy.
        data_cfg.encoder.pooling = "mean"
        train_dl, test_dl, encoder, *_ = initialize_data(
            data_cfg, cfg.runtime.data, None, device=cfg.runtime.device,
            keep_special=bool(cfg.model.get("keep_special", True)))

        for seed in wanted:
            torch.manual_seed(int(seed))
            log.info("training NER probe: %s/%s seed=%s", dataset, family, seed)
            probe = _Probe(cfg, encoder, tags, train_dl, test_dl, cfg.runtime.device)
            history = probe.fit(int(cfg.train.epochs))
            out = store_dir(dataset, family, seed)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(probe.model.state_dict(), out / "model.pth")
            report = {
                "dataset": dataset, "family": family, "seed": int(seed),
                "epochs": int(cfg.train.epochs), "tags": tags,
                **{k: v for k, v in history[-1].items() if k not in ("epoch", "train_loss")},
                "history": history,
            }
            (out / "report.json").write_text(
                json.dumps(report, indent=2, default=lambda o: o.item()), encoding="utf-8")
            log.info("saved %s", out / "report.json")

    return [json.loads((store_dir(dataset, family, s) / "report.json").read_text(encoding="utf-8"))
            for s in seeds]
