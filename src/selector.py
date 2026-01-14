import torch, torch.nn as nn

def kuma_sample(alpha: torch.Tensor, beta: torch.Tensor, u_min: float = 1e-4, eps: float = 1e-6):
    """
    Reparameterized Kumaraswamy sample via inverse CDF.
    alpha, beta > 0, same shape.
    """
    u = torch.rand_like(alpha).clamp(u_min, 1.0 - u_min)
    inv_alpha = 1.0 / (alpha + eps)
    inv_beta = 1.0 / (beta + eps)
    inner = 1.0 - torch.pow(1.0 - u, inv_beta)
    z = torch.pow(inner.clamp(min=eps, max=1.0), inv_alpha)
    return z.clamp(eps, 1.0 - eps)


def kuma_mean(alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-8):
    """
    Mean of Kumaraswamy(alpha, beta):
    E[z] = beta * Gamma(1 + 1/alpha) * Gamma(beta) / Gamma(1 + beta + 1/alpha)
    """
    inv_alpha = 1.0 / (alpha + eps)
    log_mean = (
        torch.log(beta + eps)
        + torch.lgamma(1.0 + inv_alpha)
        + torch.lgamma(beta)
        - torch.lgamma(1.0 + beta + inv_alpha)
    )
    return torch.exp(log_mean).clamp(0.0, 1.0)
    

class Selector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.out = nn.Linear(d_model, 2)
        self.softplus = nn.Softplus()

    def forward(self, token_emb: torch.Tensor):
        a, b = self.out(token_emb).unbind(-1)
        alpha = self.softplus(a) + 1e-3
        beta = self.softplus(b) + 1e-3
        return alpha, beta


class RationaleSelectorModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.selector = Selector(int(embedding_dim))

    def forward(self, embeddings: torch.Tensor, attn: torch.Tensor, deterministic: bool = False):
        """
        embeddings:      [B, T, D]
        attention_mask:  [B, T]   (0/1)
        returns gates g: [B, T]   (train: hard with ST; eval: hard deterministic by default)
        """
        emb = embeddings * attn.unsqueeze(-1)

        alpha, beta = self.selector(emb)
        
        if deterministic:
            z = kuma_mean(alpha, beta)
            g = (z > 0.5).float()
        else:
            z = kuma_sample(alpha, beta)
            h = (z > 0.5)
            g = (h + (z - z.detach()))
            
        return g * attn, z
