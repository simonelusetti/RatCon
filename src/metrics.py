from __future__ import annotations

import torch

from .data import PAD_TAG
from .utils import dict_to_table, format_dict


class Counts:
    def __init__(
        self,
        labels: list | None = None,
        mask: torch.Tensor | None = None,
        pred_mask: torch.Tensor | None = None,
        pad: str = PAD_TAG,
    ) -> None:
        self.data = {}
        self.pad = pad

        if labels is None:
            return  # Fully empty Counts

        self.data = dict.fromkeys(set(labels) - {self.pad}, 0)

        if mask is None:
            return  # Labels but all set to 0

        if pred_mask is not None:
            mask = mask.bool() & pred_mask.bool()  # Pred counts

        mask = mask.tolist()
        for c in self.data.keys():
            self.data[c] = sum(1 if yi and xi == c else 0 for xi, yi in zip(labels, mask))

    def __add__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only add Counts to Counts.")
        result = Counts()
        result.data = {
            c: self.data.get(c, 0) + other.data.get(c, 0)
            for c in self.data.keys() | other.data.keys()
        }
        return result

    def __truediv__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only divide Counts by Counts.")
        result = Counts()
        result.data = {}
        for c in self.data.keys() | other.data.keys():
            if c in self.data and c in other.data:
                result.data[c] = self.data[c] / other.data[c] if other.data[c] != 0 else 0.0
            elif c in self.data:
                result.data[c] = self.data[c]
            else:
                result.data[c] = other.data[c]
        return result

    def _sorted_data(self) -> dict:
        return dict(sorted(self.data.items(), key=lambda x: x[1], reverse=True))

    def __str__(self) -> str:
        return format_dict(self._sorted_data())

    def to_table(self) -> str:
        return dict_to_table(self._sorted_data())

    def preferences_over_total(self, total: int) -> Counts:
        result = Counts()
        result.data = {
            c: self.data[c] / total if total > 0 else 0.0
            for c in self.data.keys()
            if c != self.pad
        }
        return result

    def preferences(self) -> Counts:
        result = Counts()
        total = sum(v for c, v in self.data.items() if c != self.pad)
        result.data = {
            c: self.data[c] / total if total > 0 else 0.0
            for c in self.data.keys()
            if c != self.pad
        }
        return result

    def confusion_with(self, pred: Counts, positive_label: str | None = None) -> tuple[int, int, int, int, str]:
        if not isinstance(pred, Counts):
            raise ValueError("pred must be a Counts instance")

        labels = set(self.data.keys()) | set(pred.data.keys())

        if self.pad in labels:
            labels.discard(self.pad)

        if len(labels) != 2:
            return -1, -1, -1, -1, ""

        labels = list(labels)

        if positive_label is None:
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

        gold_pos = self.data.get(positive_label, 0)
        gold_neg = self.data.get(negative_label, 0)

        pred_pos = pred.data.get(positive_label, 0)
        pred_neg = pred.data.get(negative_label, 0)

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
