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
    tokenizer = AutoTokenizer.from_pretrained(CANONICAL_TOKENIZERS[group], use_fast=True)

    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
        if fallback is None:
            raise ValueError(
                f"Tokenizer {tokenizer.name_or_path} has no pad/eos/bos/unk token available for padding."
            )
        tokenizer.pad_token = fallback

    tokenizer.padding_side = "right"
    return tokenizer


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

        attn_bias = torch.log(key_mask.clamp(min=1e-9))  # 0 for valid, ≈-inf for masked
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

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
    return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


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
        mask = pool_mask.unsqueeze(-1).type_as(token_emb)
        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb


# -----------------------------------------------------------------------------
# Concrete encoders
# -----------------------------------------------------------------------------

class FrozenSBERT(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool) -> None:
        super().__init__(normalize)
        self.model = SentenceTransformer(model_name)
        self.backbone = self.model[0].auto_model

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.backbone, input_ids, attention_mask)


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
        encoder = FrozenSBERT(encoder_name, normalize=False)
    elif family == "e5":
        encoder = FrozenE5(encoder_name, normalize=False)
    elif family == "bge":
        encoder = FrozenBGE(encoder_name, normalize=False)
    elif family == "llm":
        encoder = FrozenLLMEncoder(encoder_name, normalize=False)
    else:
        raise ValueError(f"Unknown encoder family: {family}")

    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
