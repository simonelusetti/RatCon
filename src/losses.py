import torch
from torch.nn import functional as F

def recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    return 1.0 - cos_sim.mean()


def sparsity_loss(z: torch.Tensor, attn: torch.Tensor, target_rate: float):
    rate = (z * attn).sum() / attn.sum().clamp(min=1)
    return F.relu(rate - target_rate) ** 2
 

def certainty_loss(z: torch.Tensor, attn: torch.Tensor):
    total = attn.sum().clamp(min=1)
    return (z * (1 - z) * attn).sum() / total
