# models.py
import torch, logging
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import should_disable_tqdm

EPS = 1e-8

logger = logging.getLogger(__name__)


class FourierFilter(nn.Module):
    def __init__(self, mode="lowpass", keep_ratio=0.5):
        """
        mode: 'lowpass', 'highpass', or 'bandpass'
        keep_ratio: fraction of frequencies to keep (0 < keep_ratio <= 1)
        """
        super().__init__()
        self.mode = mode
        self.keep_ratio = keep_ratio

    def forward(self, x, mask=None):
        """
        x: [B, D] token embeddings
        mask: optional attention mask [B,L] (ignored here but can zero out padded tokens)
        """
        B, D = x.shape

        # FFT along sequence length
        Xf = torch.fft.rfft(x, dim=1)  # [B, L//2+1, D]

        # build frequency mask
        freqs = Xf.size(1)
        k = int(self.keep_ratio * freqs)

        if self.mode == "lowpass":
            m = torch.zeros(freqs, device=x.device)
            m[:k] = 1.0
        elif self.mode == "highpass":
            m = torch.zeros(freqs, device=x.device)
            m[-k:] = 1.0
        elif self.mode == "bandpass":
            m = torch.zeros(freqs, device=x.device)
            start, end = freqs//4, freqs//4 + k
            m[start:end] = 1.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # apply mask in frequency domain
        Xf_filtered = Xf * m

        # inverse FFT to time domain
        x_filtered = torch.fft.irfft(Xf_filtered, dim=1)  # [B,D]

        return x_filtered

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

class ClusterFilter:
    def __init__(self):
        self.centroids = None
        self.counts = None
        self.entity_cluster = None
        self.token_count = None
        self.metadata = {}

    def info(self):
        if self.centroids is None or self.entity_cluster is None:
            return None
        counts_list = self.counts.detach().cpu().tolist() if self.counts is not None else None
        payload = {
            "tokens": self.token_count,
            "cluster_counts": counts_list,
            "entity_cluster": self.entity_cluster,
        }
        payload.update(self.metadata)
        return payload

    def fit_from_loader(
        self,
        model,
        loader,
        cluster_cfg,
        *,
        logger=None,
        label="model",
    ):
        if cluster_cfg is None or not getattr(cluster_cfg, "use", False):
            self.centroids = None
            self.counts = None
            self.entity_cluster = None
            self.token_count = None
            self.metadata = {}
            return None

        threshold = float(cluster_cfg.proposal_thresh)
        max_tokens = int(cluster_cfg.max_tokens)
        num_clusters = int(cluster_cfg.num_clusters)
        num_iters = int(cluster_cfg.iters)
        tol = float(cluster_cfg.tol)
        seed = int(cluster_cfg.seed)

        if num_clusters <= 0:
            if logger:
                logger.warning("Cluster config: num_clusters <= 0; skipping filter")
            return None

        collected = []
        total = 0
        device = next(model.parameters()).device
        model.eval()

        disable_progress = should_disable_tqdm()
        desc = f"Collecting gates for {label}"
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, disable=disable_progress):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                embeddings = batch["embeddings"]
                attention_mask = batch["attention_mask"]
                incoming = batch.get("incoming") if model.cfg.attention_augment else None
                outgoing = batch.get("outgoing") if model.cfg.attention_augment else None

                out = model(embeddings, attention_mask, incoming, outgoing)
                gates = out["gates"]
                valid = attention_mask > 0
                mask = (gates >= threshold) & valid

                if not mask.any():
                    continue

                selected = out["token_embeddings"][mask]
                if selected.ndim == 1:
                    selected = selected.unsqueeze(0)

                collected.append(selected.cpu())
                total += selected.size(0)
                if total >= max_tokens and max_tokens > 0:
                    break

        if not collected:
            if logger:
                logger.warning(
                    f"Insufficient gated tokens to fit clusters for {label}; skipping filter"
                )
            return None

        features = torch.cat(collected, dim=0)
        if features.size(0) > max_tokens and max_tokens > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            perm = torch.randperm(features.size(0), generator=generator)[:max_tokens]
            features = features[perm]

        features = features.to(dtype=torch.float32)
        num_clusters = min(num_clusters, features.size(0))
        if num_clusters < 1:
            if logger:
                logger.warning(f"Need at least one cluster for {label}; skipping filter")
            return None

        generator = torch.Generator(device=features.device)
        generator.manual_seed(seed)
        perm_init = torch.randperm(features.size(0), generator=generator)
        centroids = features[perm_init[:num_clusters]].clone()
        for _ in range(max(1, num_iters)):
            distances = torch.cdist(features, centroids)
            assignments = distances.argmin(dim=1)
            new_centroids = []
            for idx in range(num_clusters):
                mask = assignments == idx
                if mask.any():
                    new_centroids.append(features[mask].mean(dim=0))
                else:
                    new_centroids.append(centroids[idx])
            new_centroids = torch.stack(new_centroids)
            shift = (new_centroids - centroids).abs().max().item()
            centroids = new_centroids
            if shift <= tol:
                break

        distances = torch.cdist(features, centroids)
        assignments = distances.argmin(dim=1)
        counts = torch.bincount(assignments, minlength=num_clusters)
        entity_cluster = int(counts.argmax().item())

        self.centroids = centroids.detach().to(device)
        self.counts = counts.detach().to(device)
        self.entity_cluster = entity_cluster
        self.token_count = int(features.size(0))
        self.metadata = {
            "proposal_thresh": threshold,
            "num_clusters": int(cluster_cfg.num_clusters),
        }

        if logger:
            logger.info(
                f"Cluster filter for {label}: tokens={self.token_count}, "
                f"counts={self.counts.cpu().tolist()}, entity_cluster={entity_cluster}"
            )

        return self.info()

    def apply(
        self,
        token_embeddings,
        gates,
        attention_mask,
    ):
        if self.centroids is None or self.entity_cluster is None:
            return gates

        centroids = self.centroids.to(token_embeddings.device)
        flat = token_embeddings.reshape(-1, token_embeddings.size(-1))
        distances = torch.cdist(flat, centroids)
        assignments = distances.argmin(dim=1)
        assignments = assignments.view(token_embeddings.shape[:-1])
        entity_mask = assignments.eq(self.entity_cluster).float()
        entity_mask = entity_mask * attention_mask.float()
        return gates * entity_mask


class RationaleSelectorModel(nn.Module):
    """
    Inputs are already SBERT token embeddings.
    We only do selection + SBERT pooling (mean/cls/max).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sbert = SentenceTransformer(cfg.sbert_name)
        self.pooler = self.sbert[1]  # SentenceTransformer pooling module
        
        d = self.sbert[0].auto_model.config.hidden_size
        
        if cfg.attention_augment: d = d + 2
        
        self.selector = Selector(d)
        
        self.kuma = HardKumaSampler()
        
        if cfg.fourier.use:
            self.fourier = FourierFilter(mode=cfg.fourier.mode, keep_ratio=cfg.fourier.keep_ratio)

        with torch.no_grad():
            null_emb = self.sbert.encode([""], convert_to_tensor=True)
        self.register_buffer("null_embedding", null_emb.squeeze(0))

        self.cluster_filter = ClusterFilter()


    def forward(self, embeddings, attention_mask, incoming=None, outgoing=None):
        if self.cfg.attention_augment:
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
        
        if self.cfg.fourier.use:
            h_anchor = self.fourier(h_anchor)
            h_rat = self.fourier(h_rat)
            h_comp = self.fourier(h_comp)

        null_vec = self.null_embedding.to(embeddings.device).unsqueeze(0).expand(h_comp.size(0), -1)

        return {
            "h_anchor": h_anchor, "h_rat": h_rat, "h_comp": h_comp,
            "gates": g, "alpha": alpha, "beta": beta,
            "null": null_vec,
            "token_embeddings": embeddings,
        }

    def apply_cluster_filter(
        self,
        token_embeddings,
        gates,
        attention_mask,
    ):
        return self.cluster_filter.apply(token_embeddings, gates, attention_mask)

    def get_cluster_info(self):
        return self.cluster_filter.info()

    def fit_cluster_filter_from_loader(
        self,
        loader,
        cluster_cfg,
        *,
        logger=None,
        label="model",
    ):
        return self.cluster_filter.fit_from_loader(
            self,
            loader,
            cluster_cfg,
            logger=logger,
            label=label,
        )




def nt_xent(anchor, positive, temperature=0.07):
    """
    InfoNCE with in-batch negatives: anchors vs. positives (one-to-one).
    """
    temperature = max(float(temperature), 1e-3)
    B = anchor.size(0)
    logits = anchor @ positive.t() / temperature              # [B,B]
    labels = torch.arange(B, device=anchor.device)
    return F.cross_entropy(logits, labels)
