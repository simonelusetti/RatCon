# models.py
import torch, logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

EPS = 1e-8

logger = logging.getLogger(__name__)

class HardKumaSampler(nn.Module):
    """
    Differentiable HardKuma gate per token.
    We use Kumaraswamy reparameterization and clamp to (0,1).
    Log-space version.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def sample(alpha, beta, eps=1e-6, u_min=1e-4):
        # u in (u_min, 1-u_min)
        u = torch.rand_like(alpha)
        u = u.clamp(u_min, 1.0 - u_min)

        # log-space version:
        # t = (1 - u)^(1/beta)  = exp( (1/beta) * log(1-u) )
        log1m_u = torch.log1p(-u)                              # safe log(1-u)
        inv_beta = 1.0 / (beta + eps)
        t = torch.exp(inv_beta * log1m_u)                      # (1-u)^(1/beta)

        # x = (1 - t)^(1/alpha)  = exp( (1/alpha) * log(1 - t) )
        # clamp argument to log1p to avoid log(<=0) numerics
        one_minus_t = (1.0 - t).clamp(min=eps, max=1.0)
        log1m_t = torch.log(one_minus_t)
        inv_alpha = 1.0 / (alpha + eps)
        x = torch.exp(inv_alpha * log1m_t)

        # keep away from exact 0/1 to stabilize backward
        return x.clamp(eps, 1.0 - eps)

class Selector(nn.Module):
    """
    Tiny 1-layer conv + FF to produce (alpha, beta) per token from token embeddings.
    """
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, 2)  # alpha, beta
        self.softplus = nn.Softplus()
        
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, token_emb):  # [B, L, D]
        h = F.gelu(self.proj(token_emb))
        params = self.out(h)                 # [B, L, 2]
        alpha, beta = params[..., 0], params[..., 1]
        # map to (0, +inf)
        alpha = (self.softplus(alpha) + 1.0).clamp(1.0, 10.0)
        beta  = (self.softplus(beta)  + 1.0).clamp(1.0, 10.0)   
        return alpha, beta


class RationaleSelectorModel(nn.Module):
    """
    Shared encoder (SBERT backbone as HF AutoModel) + selector + pooling + projection heads.
    """
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)  # returns last_hidden_state
        d_model = self.encoder.config.hidden_size
        self.selector = Selector(d_model)
        self.kuma = HardKumaSampler()

        self.proj_anchor = nn.Sequential(nn.Linear(d_model, proj_dim), nn.GELU(),
                                            nn.Linear(proj_dim, proj_dim))
        self.proj_view   = nn.Sequential(nn.Linear(d_model, proj_dim), nn.GELU(),
                                            nn.Linear(proj_dim, proj_dim))

    @staticmethod
    def weighted_mean_pool(last_hidden, attn_weights):  # [B,L,D], [B,L]
        w = attn_weights.unsqueeze(-1)                  # [B,L,1]
        s = (last_hidden * w).sum(dim=1)                # [B,D]
        z = w.sum(dim=1).clamp_min(EPS)                 # [B,1]
        return s / z

    def forward(self, embeddings, attention_mask, verbose=False, logger=None, input_ids=None):
        if verbose:
            msg = (f"tok: nan={torch.isnan(embeddings).any().item()} "
                f"min={embeddings.min().item():.4f} max={embeddings.max().item():.4f}")
            (logger.debug if logger else print)(msg)

        # HardKuma params & gates
        alpha, beta = self.selector(embeddings)

        if verbose:
            msg = (f"alpha: nan={torch.isnan(alpha).any().item()} "
                f"min={alpha.min().item():.4f} max={alpha.max().item():.4f} | "
                f"beta: nan={torch.isnan(beta).any().item()} "
                f"min={beta.min().item():.4f} max={beta.max().item():.4f}")
            (logger.debug if logger else print)(msg)

        gates = self.kuma.sample(alpha, beta)  # [B,L]
        gates = gates.clamp(1e-6, 1.0 - 1e-6)

        if verbose:
            msg = (f"gates: nan={torch.isnan(gates).any().item()} "
                f"min={gates.min().item():.4f} max={gates.max().item():.4f}")
            (logger.debug if logger else print)(msg)

        # Build views
        h_anchor = self.weighted_mean_pool(embeddings, attention_mask.float())                  # [B,D]
        h_rat    = self.weighted_mean_pool(embeddings, attention_mask.float() * gates)          # [B,D]
        h_comp   = self.weighted_mean_pool(embeddings, attention_mask.float() * (1.0 - gates))  # [B,D]

        h_anchor = self.proj_anchor(h_anchor)
        h_rat    = self.proj_view(h_rat)
        h_comp   = self.proj_view(h_comp)

        # Normalize for cosine
        h_anchor = F.normalize(h_anchor, dim=-1, eps=1e-6)
        h_rat    = F.normalize(h_rat, dim=-1, eps=1e-6)
        h_comp   = F.normalize(h_comp, dim=-1, eps=1e-6)

        return {
            "h_anchor": h_anchor,
            "h_rat": h_rat,
            "h_comp": h_comp,
            "gates": gates,
            "alpha": alpha,
            "beta": beta,
        }


def nt_xent(anchor, positive, temperature=0.07):
    """
    InfoNCE with in-batch negatives: anchors vs. positives (one-to-one).
    """
    temperature = max(float(temperature), 1e-3)
    B = anchor.size(0)
    logits = anchor @ positive.t() / temperature              # [B,B]
    labels = torch.arange(B, device=anchor.device)
    return F.cross_entropy(logits, labels)
