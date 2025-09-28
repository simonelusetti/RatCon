# models.py
import torch, logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

EPS = 1e-8

logger = logging.getLogger(__name__)


class HardKumaSampler(nn.Module):
    @staticmethod
    def sample(alpha, beta, eps=1e-6, u_min=1e-4):
        u = torch.rand_like(alpha).clamp(u_min, 1.0 - u_min)
        log1m_u = torch.log1p(-u)
        t = torch.exp((1.0 / (beta + eps)) * log1m_u)
        one_minus_t = (1.0 - t).clamp(min=eps, max=1.0)
        x = torch.exp((1.0 / (alpha + eps)) * torch.log(one_minus_t))
        return x.clamp(eps, 1.0 - eps)

class Selector(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out  = nn.Linear(hidden, 2)
        self.softplus = nn.Softplus()

    def forward(self, token_emb):  # token_emb: [B,L,D] precomputed SBERT embeddings
        h = F.gelu(self.proj(token_emb))
        alpha, beta = self.out(h).unbind(-1)
        alpha = (self.softplus(alpha) + 1.0).clamp(1.0, 10.0)
        beta  = (self.softplus(beta)  + 1.0).clamp(1.0, 10.0)
        return alpha, beta

class RationaleSelectorModel(nn.Module):
    """
    Inputs are already SBERT token embeddings.
    We only do selection + SBERT pooling (mean/cls/max).
    """
    def __init__(self, sbert_name="sentence-transformers/all-MiniLM-L6-v2", attention_augment=False):
        super().__init__()
        self.sbert = SentenceTransformer(sbert_name)
        self.pooler = self.sbert[1]  # SentenceTransformer pooling module
        self.attention_augment = attention_augment
        d = self.sbert[0].auto_model.config.hidden_size
        if self.attention_augment: d = d + 2
        self.selector = Selector(d)
        self.kuma = HardKumaSampler()
        

    def forward(self, embeddings, attention_mask, incoming=None, outgoing=None):
        if self.attention_augment:
            embeddings = torch.cat([
                embeddings,
                incoming.unsqueeze(-1),
                outgoing.unsqueeze(-1)
            ], dim=-1)  # [B,L,D+2]
        
        alpha, beta = self.selector(embeddings)
        g = self.kuma.sample(alpha, beta).clamp(1e-6, 1.0 - 1e-6)

        # Anchor = pool all tokens
        h_anchor = self.pooler({"token_embeddings": embeddings,
                                "attention_mask": attention_mask})["sentence_embedding"]

        # Rationale = pool gated tokens
        h_rat = self.pooler({"token_embeddings": embeddings,
                             "attention_mask": attention_mask * g})["sentence_embedding"]

        # Complement = pool complement tokens
        h_comp = self.pooler({"token_embeddings": embeddings,
                              "attention_mask": attention_mask * (1.0 - g)})["sentence_embedding"]

        return {
            "h_anchor": h_anchor, "h_rat": h_rat, "h_comp": h_comp,
            "gates": g, "alpha": alpha, "beta": beta
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
