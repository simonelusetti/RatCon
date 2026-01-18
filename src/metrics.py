from __future__ import annotations

import torch

from typing import Tuple
from .data import PAD_TAG
from .utils import dict_to_table, format_dict

class Counts(dict):
    def __init__(
        self, labels: list | None = None, mask: torch.Tensor | None = None, \
        pred_mask: torch.Tensor | None = None, pad: str = PAD_TAG,
    ) -> None:
        super().__init__()
        self.data = {}
        self.pad = pad
        
        if labels is None:
            return
        self.data = dict.fromkeys(set(labels) - {self.pad}, 0)
        
        if pred_mask is not None:
            mask = mask.bool() & pred_mask.bool()
        mask = mask.tolist()
        for c in self.data.keys():
            self.data[c] = sum([1 if yi and xi == c else 0 for xi, yi in zip(labels, mask)])
                        
    def __add__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only add Counts to Counts.")

        result = Counts()
        result.data = dict.fromkeys(self.data.keys() | other.data.keys(), 0)
        for c in self.data.keys():
            if c in other.data:
                result.data[c] = self.data[c] + other.data[c]
            else:
                result.data[c] = self.data[c]
        only_in_other = set(other.data.keys()) - set(self.data.keys())
        for c in only_in_other:
            result.data[c] = other.data[c]
        return result
    
    def __truediv__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only divide Counts by Counts.")
        
        result = Counts()
        result.data = dict.fromkeys(self.data.keys() | other.data.keys(), 0)
        for c in self.data.keys():
            if c in other.data:
                result.data[c] = self.data[c] / other.data[c] if other.data[c] != 0 else 0.0
            else:
                result.data[c] = self.data[c]
        only_in_other = set(other.data.keys()) - set(self.data.keys())
        for c in only_in_other:
            result.data[c] = other.data[c]
        return result
    
    def __str__(self) -> str:
        data_dict = {
            **{f"{k}": v for k, v in self.data.items()},
        }
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))
        return format_dict(sorted_dict)
    
    def to_table(self) -> str:
        data_dict = {
            **{f"{k}": v for k, v in self.data.items()},
        }
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))
        return dict_to_table(sorted_dict)
    
    def preferences_over_total(self, total: int) -> Counts:
        result = Counts()
        result.data = dict.fromkeys(self.data.keys(), 0)
        for c in self.data.keys():
            if c != self.pad:
                result.data[c] = self.data[c] / total if total > 0 else 0.0
        return result
    
    def preferences(self) -> Counts:
        result = Counts()
        result.data = dict.fromkeys(self.data.keys(), 0)
        total = sum(self.data[c] for c in self.data.keys() if c != self.pad)
        for c in self.data.keys():
            if c != self.pad:
                result.data[c] = self.data[c] / total if total > 0 else 0.0
        return result
    
    def confusion_with(self, pred: Counts, positive_label: str | None = None) -> tuple[int, int, int, int, str]:
        """
        Compute confusion matrix assuming:
        - self = gold Counts
        - pred = predicted Counts
        - binary labels only
        """

        if not isinstance(pred, Counts):
            raise ValueError("pred must be a Counts instance")

        labels = set(self.data.keys()) | set(pred.data.keys())

        if self.pad in labels:
            labels.discard(self.pad)

        if len(labels) != 2:
            return -1, -1, -1, -1, ""

        labels = list(labels)

        # infer positive label if not provided
        if positive_label is None:
            # common conventions
            for candidate in (1, "1", "True", "true", "POS", "pos", "positive"):
                if candidate in labels:
                    positive_label = candidate
                    break
            if positive_label is None:
                raise ValueError(
                    f"Cannot infer positive label automatically from {labels}. "
                    "Please specify positive_label explicitly."
                )
                
        if positive_label not in labels:
            raise ValueError(f"positive_label {positive_label} not in labels {labels}")

        negative_label = labels[0] if labels[1] == positive_label else labels[1]

        # gold counts
        gold_pos = self.data.get(positive_label, 0)
        gold_neg = self.data.get(negative_label, 0)

        # predicted counts
        pred_pos = pred.data.get(positive_label, 0)
        pred_neg = pred.data.get(negative_label, 0)

        # intersections (these are the key quantities)
        TP = min(gold_pos, pred_pos)
        TN = min(gold_neg, pred_neg)

        FP = pred_pos - TP
        FN = gold_pos - TP

        return int(TP), int(FP), int(FN), int(TN), positive_label
    
    def conf_matrix(self, counts_pred: Counts, epoch: int | None = None, positive_label: str | None = None) -> str | None:
        TP, FP, FN, TN, positive_label = self.confusion_with(
            counts_pred,
            positive_label=positive_label,
        )
                
        if TP == -1:
            return None
        
        header = f"Confusion Matrix for Epoch {epoch}" if epoch is not None else "Confusion Matrix"
        
        return (
            header + "\n"
            f"Positive label: {positive_label}\n\n"
            f"               Pred +     Pred -\n"
            f"Gold +     |   {TP:<8}   {FN:<8}\n"
            f"Gold -     |   {FP:<8}   {TN:<8}\n\n"
        )

def counts(
    pred_mask: torch.Tensor,
    gold_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    tp = (pred_mask & gold_mask).sum().item()
    fp = (pred_mask & (~gold_mask)).sum().item()
    fn = ((~pred_mask) & gold_mask).sum().item()
    return tp, fp, fn