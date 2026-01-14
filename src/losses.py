import torch
from torch.nn import functional as F

def recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # returns [B]
    return 1.0 - cos_sim.mean()


def sparsity_loss(z: torch.Tensor, budget=0.5):
    rate = z.mean(dim=1)
    return ((rate - budget) ** 2).mean()


def certainty_loss(z: torch.Tensor):
    return (z * (1-z)).mean()