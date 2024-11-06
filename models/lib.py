# Trying to stitch together a simpler version of Microsofts phi-2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
    from flash_attn import flash_attn_func


def gelu(x):
    # Copying this version of gelu that's used everywhere for some reason
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.expand(x)
        x = gelu(x)
        x = self.proj(x)

        return x


class Attention(nn.Module):
    def __init__(self, config, layer_idx, rotary=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.rotary = rotary

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.n_embd = config.n_embd

        if not config.use_flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, config.block_size, config.block_size, 1), persistent=False)
        else:
            print('Using Flash Attention')
        # Whether to have rotary embeddings or not (Phi does, GPT2 does not)
        self.rotary_emb = RotaryEmbedding(config) if rotary else None
        self.queries_linear = nn.Linear(config.n_embd, config.n_embd)
        self.keys_linear = nn.Linear(config.n_embd, config.n_embd)
        self.values_linear = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    from flash_attn import flash_attn_func

    def forward(self, x, cache=None, attn_mask=None):
        # Initial dimensions
        bsz, query_len, n_embd = x.shape
        queries = self.queries_linear(x)
        keys = self.keys_linear(x)
        values = self.values_linear(x)

        # Get starting position for positional embedding
        start_pos = 0 if cache is None or not cache.use_caching else cache.cur_seq_len[self.layer_idx]

        # Reshape for multi-head attention
        queries = queries.view(bsz, query_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, query_len, self.n_heads, self.head_dim)
        values = values.view(bsz, query_len, self.n_heads, self.head_dim)

        # Apply rotary embeddings if enabled
        if self.rotary:
            queries, keys = self.rotary_emb(queries, keys, start_pos)

        # Update cache if enabled
        if cache is not None and cache.use_caching:
            if cache.cur_seq_len[self.layer_idx] > 0:
                keys_, values_ = cache.get(self.layer_idx)
                keys = torch.cat([keys_, keys], dim=1)
                values = torch.cat([values_, values], dim=1)
            cache.update(keys, values, self.layer_idx)
            seq_len = values.shape[1]
        else:
            seq_len = query_len

        # Apply padding mask by zeroing out positions in queries, keys, values where attn_mask is 0
        if attn_mask is not None:
            # Reshape attn_mask for broadcasting: [batch_size, seq_len, 1, 1]
            attn_mask = attn_mask[:, :, None, None]
            queries = queries * attn_mask
            keys = keys * attn_mask
            values = values * attn_mask

        # Use flash attention if enabled, otherwise fall back to standard attention
        if self.config.use_flash:
            # flash_attn_func with causal masking
            out = flash_attn_func(queries, keys, values, causal=True)
        else:
            # Standard attention mechanism
            att = torch.einsum('bmhd,bnhd->bmnh', queries, keys)  # [batch_size, query_len, seq_len, num_heads]
            
            # Apply attn_mask if provided to ignore padding tokens
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None, :]  # Shape for broadcasting: [batch_size, 1, 1, seq_len]
                att = att.masked_fill(attn_mask == 0, float('-inf'))

            # Apply causal mask if required
            if cache is None or not cache.use_caching:
                att = att.masked_fill(self.bias[:, :query_len, :seq_len, :] == 0, float('-inf'))
            else:
                att = att.masked_fill(self.bias[:, seq_len - query_len:seq_len, :seq_len, :] == 0, float('-inf'))

            att = F.softmax(att / math.sqrt(self.head_dim), dim=2)
            out = torch.einsum('bmnh,bnhd->bmhd', att, values)  # [batch_size, query_len, n_heads, head_dim]

        # Final projection to embedding dimension
        out = self.proj(out.contiguous().view(bsz, query_len, self.n_embd))
        return out



class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.base = config.base
        self.dim = config.rope_dim
        self.dtype = config.dtype

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
        t = torch.arange(self.block_size, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Reshape in order to make multiplication go through in forward()
        emb = emb.view((1, emb.shape[0], 1, emb.shape[1]))
        self.register_buffer("cos_cached", emb.cos().to(self.dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(self.dtype), persistent=False)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, queries, keys, start_pos=0):
        seq_len = keys.shape[1]
        cos = self.cos_cached[:, start_pos:seq_len+start_pos]
        sin = self.sin_cached[:, start_pos:seq_len+start_pos]

        # Partial rotary embedding, we might only rotate some part of the embedding, "pass" will be left as is
        query_rot, query_pass = (
            queries[..., :self.dim],
            queries[..., self.dim:],
        )
        key_rot, key_pass = (
            keys[..., :self.dim],
            keys[..., self.dim:],
        )

        query_rot = (query_rot * cos) + (self.rotate_half(query_rot) * sin)
        key_rot = (key_rot * cos) + (self.rotate_half(key_rot) * sin)

        queries = torch.cat((query_rot, query_pass), dim=-1)
        keys = torch.cat((key_rot, key_pass), dim=-1)

        return queries, keys
