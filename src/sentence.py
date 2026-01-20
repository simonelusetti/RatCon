from __future__ import annotations

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALIASES = {
    "sbert": {"sbert"},
    "e5": {"e5", "retrieval", "gte"},
    "bge": {"bge", "late"},
    "llm": {"llm"}
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


def tkns_embedding_bert(model, input_ids, attention_mask):
        hidden_states = model.embeddings(input_ids)
        key_mask = attention_mask[:, None, None, :].type_as(hidden_states)

        for layer in model.encoder.layer:
            attn = layer.attention.self
            bsz, seq_len, _ = hidden_states.size()

            query = attn.query(hidden_states)
            key = attn.key(hidden_states)
            value = attn.value(hidden_states)

            query = query.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
            key = key.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
            value = value.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(attn.attention_head_size)

            probs = torch.softmax(scores, dim=-1)
            probs = probs * key_mask
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            context = torch.matmul(probs, value)
            context = context.transpose(1, 2).contiguous()
            context = context.view(bsz, seq_len, attn.all_head_size)

            attn_output = layer.attention.output(context, hidden_states)
            intermediate = layer.intermediate(attn_output)
            hidden_states = layer.output(intermediate, attn_output)

        return hidden_states
        
def tkns_embedding_gpt(model, input_ids, attention_mask):
    hidden_states = model.embed_in(input_ids)  # [B,T,H]
    bsz, seq_len, _ = hidden_states.size()
    key_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)  # [B,1,1,T]

    num_heads = int(model.config.num_attention_heads)

    for block in model.layers:
        attn = block.attention
        ln_out = block.input_layernorm(hidden_states)

        qkv = attn.query_key_value(ln_out)
        head_dim = int(getattr(attn, "head_size", None))
        qkv = qkv.view(bsz, seq_len, num_heads, 3 * head_dim).permute(0, 2, 1, 3)
        query, key, value = torch.split(qkv, head_dim, dim=-1)  # each [B,nh,T,hd]

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)  # [B,nh,T,T]
        probs = torch.softmax(scores, dim=-1)

        probs = probs * key_mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        context = torch.matmul(probs, value)  # [B,nh,T,hd]
        context = context.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, num_heads * head_dim)  # [B,T,H]

        attn_out = attn.dense(context)
        hidden_states = hidden_states + attn_out

        mlp_in = block.post_attention_layernorm(hidden_states)
        mlp_out = block.mlp(mlp_in)
        hidden_states = hidden_states + mlp_out
        hidden_states = model.final_layer_norm(hidden_states)

    return hidden_states


class SentenceEncoder(nn.Module):
    def __init__(self, normalize: bool):
        super().__init__()
        self.normalize = normalize

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if isinstance(self, FrozenSBERT):
            return tkns_embedding_bert(self.model, input_ids, attention_mask)
        elif isinstance(self, _FrozenHFEncoder):
            if isinstance(self, FrozenE5) or isinstance(self, FrozenBGE):
                return tkns_embedding_bert(self.model, input_ids, attention_mask)
            elif isinstance(self, FrozenLLMEncoder):
                return tkns_embedding_gpt(self.model, input_ids, attention_mask)
            raise ValueError(f"Unknown tokenizer type: {type}")

    def encode(self, input_ids, attention_mask):
        raise NotImplementedError()

# ---------------------------------------------------------------------
# SBERT (kept separate, uses SentenceTransformer internals)
# ---------------------------------------------------------------------

class FrozenSBERT(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, device: str):
        super().__init__(normalize)
        self.model = SentenceTransformer(model_name, device=device)
        self.transformer = self.model[0]

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, input_ids, attention_mask):
        token_emb = tkns_embedding_bert(self.transformer.auto_model, input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).type_as(token_emb)

        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb

# ---------------------------------------------------------------------
# Generic frozen HF encoder with shared mask injection
# ---------------------------------------------------------------------

class _FrozenHFEncoder(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool):
        super().__init__(normalize)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

# ---------------------------------------------------------------------
# E5 / retrieval-style mean pooling
# ---------------------------------------------------------------------

class FrozenE5(_FrozenHFEncoder):
    def encode(self, input_ids, attention_mask):
        token_emb = tkns_embedding_bert(self.model, input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).type_as(token_emb)

        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb

# ---------------------------------------------------------------------
# BGE (CLS pooling)
# ---------------------------------------------------------------------

class FrozenBGE(_FrozenHFEncoder):
    def encode(self, input_ids, attention_mask):
        token_emb = tkns_embedding_bert(self.model, input_ids, attention_mask)
        sent_emb = token_emb[:, 0]
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb

# ---------------------------------------------------------------------
# LLM-style last-token pooling
# ---------------------------------------------------------------------

class FrozenLLMEncoder(_FrozenHFEncoder):
    def encode(self, input_ids, attention_mask):
        token_emb = tkns_embedding_gpt(self.model, input_ids, attention_mask)
        hard_mask = attention_mask > 0
        idx = hard_mask.long().sum(dim=1).clamp(min=1) - 1
        sent_emb = token_emb[torch.arange(token_emb.size(0)), idx]
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb

# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

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
        encoder = FrozenSBERT(encoder_name, normalize=True, device=device)
    elif family in {"e5", "retrieval", "gte"}:
        encoder = FrozenE5(encoder_name, normalize=True)
    elif family in {"bge", "late"}:
        encoder = FrozenBGE(encoder_name, normalize=True)
    elif family == "llm":
        encoder = FrozenLLMEncoder(encoder_name, normalize=True)
    else:
        raise ValueError(f"Unknown encoder family: {family}")

    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
