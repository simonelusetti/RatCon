import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# -----------------------------------------------------------------------------
# Aliases / defaults
# -----------------------------------------------------------------------------

ALIASES = {
    "sbert": {"sbert"},
    "e5": {"e5", "retrieval", "gte"},
    "bge": {"bge", "late"},
    "llm": {"llm"},
}

ALIAS_TO_CANON = {
    alias: canon
    for canon, aliases in ALIASES.items()
    for alias in aliases
}

DEFAULT_MODEL_NAMES = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "llm": "EleutherAI/pythia-410m",
}

TOKENIZER_GROUPS = {
    "bert-base": {"sbert", "e5", "retrieval", "gte"},
    "bge": {"bge", "late"},
    "gpt": {"llm"},
}

CANONICAL_TOKENIZERS = {
    "bert-base": "bert-base-uncased",
    "bge": "BAAI/bge-base-en-v1.5",
    "gpt": "EleutherAI/pythia-410m",
}


def resolve_tokenizer_group(family: str) -> str:
    family = family.lower()
    for group, families in TOKENIZER_GROUPS.items():
        if family in families:
            return group
    raise ValueError(f"Unknown encoder family: {family}")


def resolve_tokenizer(family: str) -> AutoTokenizer:
    group = resolve_tokenizer_group(family)
    return AutoTokenizer.from_pretrained(CANONICAL_TOKENIZERS[group], use_fast=True)


# -----------------------------------------------------------------------------
# Token embedding backends (NO fractional attention; attention_mask is padding mask)
# -----------------------------------------------------------------------------

def bert_token_embeddings(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last hidden states [B, T, D] for BERT-style encoders.
    Uses attention_mask only as a standard padding mask (0/1).
    """
    hidden_states = model.embeddings(input_ids)
    key_mask = attention_mask[:, None, None, :].type_as(hidden_states)  # [B,1,1,T]

    for layer in model.encoder.layer:
        attn = layer.attention.self
        bsz, seq_len, _ = hidden_states.size()

        q = attn.query(hidden_states)
        k = attn.key(hidden_states)
        v = attn.value(hidden_states)

        q = q.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
        k = k.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
        v = v.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attn.attention_head_size)
        scores = scores + torch.log(key_mask.clamp(min=1e-9))  # mask padding

        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)

        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, attn.all_head_size)
        attn_out = layer.attention.output(context, hidden_states)
        hidden_states = layer.output(layer.intermediate(attn_out), attn_out)

    return hidden_states


def gpt_token_embeddings(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last hidden states [B, T, D] for GPT-style encoders.
    Uses attention_mask only as a standard padding mask (0/1).
    """
    hidden_states = model.embed_in(input_ids)
    bsz, seq_len, _ = hidden_states.size()

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    key_mask = attention_mask[:, None, None, :].clamp(min=0.0)  # [B,1,1,T]

    for block in model.layers:
        ln_out = block.input_layernorm(hidden_states)

        qkv = block.attention.query_key_value(ln_out)
        qkv = qkv.view(bsz, seq_len, num_heads, 3 * head_dim).permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv, head_dim, dim=-1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores + torch.log(key_mask.clamp(min=1e-9))  # mask padding

        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(bsz, seq_len, num_heads * head_dim)

        hidden_states = hidden_states + block.attention.dense(context)
        hidden_states = hidden_states + block.mlp(block.post_attention_layernorm(hidden_states))
        hidden_states = block.post_attention_layernorm(hidden_states)

    return hidden_states


# -----------------------------------------------------------------------------
# Encoder interface: token_embeddings (expensive, once) + pool (cheap, many times)
# -----------------------------------------------------------------------------

class SentenceEncoder(nn.Module):
    """
    NO fractional attention.
    - token_embeddings(ids, attn) does the heavy transformer forward once.
    - pool(token_emb, pool_mask) computes sentence repr for an arbitrary pool_mask.
      pool_mask is where your selector's g lives (e.g., pool_mask = attn * g).
    """
    def __init__(self, normalize: bool) -> None:
        super().__init__()
        self.normalize = normalize

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pool(self, token_emb: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Concrete encoders
# -----------------------------------------------------------------------------

class FrozenSBERT(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, device: str) -> None:
        super().__init__(normalize)
        self.model = SentenceTransformer(model_name, device=device)
        self.backbone = self.model[0].auto_model

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.backbone, input_ids, attention_mask)

    def pool(self, token_emb: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        mask = pool_mask.unsqueeze(-1).type_as(token_emb)
        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb


class _FrozenHFEncoder(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool) -> None:
        super().__init__(normalize)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False


class FrozenE5(_FrozenHFEncoder):
    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.model, input_ids, attention_mask)

    def pool(self, token_emb: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        mask = pool_mask.unsqueeze(-1).type_as(token_emb)
        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb


class FrozenBGE(_FrozenHFEncoder):
    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.model, input_ids, attention_mask)

    def pool(self, token_emb: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        # BGE convention: use CLS token. pool_mask is irrelevant here.
        sent_emb = token_emb[:, 0]
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb


class FrozenLLMEncoder(_FrozenHFEncoder):
    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return gpt_token_embeddings(self.model, input_ids, attention_mask)

    def pool(self, token_emb: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        mask = pool_mask.unsqueeze(-1).type_as(token_emb)
        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

def build_sentence_encoder(
    family: str,
    encoder_name: str | None,
    device: str | None = None,
) -> tuple[SentenceEncoder, AutoTokenizer]:
    family = ALIAS_TO_CANON[family.lower()]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_name in {None, "None", "null", "NULL"}:
        encoder_name = DEFAULT_MODEL_NAMES[family]

    tokenizer = resolve_tokenizer(family)

    if family == "sbert":
        encoder = FrozenSBERT(encoder_name, normalize=False, device=device)
    elif family in {"e5", "retrieval", "gte"}:
        encoder = FrozenE5(encoder_name, normalize=False)
    elif family in {"bge", "late"}:
        encoder = FrozenBGE(encoder_name, normalize=False)
    elif family == "llm":
        encoder = FrozenLLMEncoder(encoder_name, normalize=False)
    else:
        raise ValueError(f"Unknown encoder family: {family}")

    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
