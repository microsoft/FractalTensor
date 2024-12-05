import torch
from torch import Tensor
import torch.nn.functional as F

import math

from einops import rearrange

from flash_attn.flash_attn_interface import _flash_attn_forward

__all__ = [
    'MultiHeadAttention',
    'MultilHeadFlashAttention',
    'GenerateQKV',
    'AttentionRef',
]


def GenerateQKV(x: Tensor, Wqkv, nheads: int, qkvpacked: bool):
    """
    Arguments:
        x: (batch_size, seqlen, nheads * d)
        Wqkv: nn.Linear(nheads * d, 3 * nheads * d)
    """
    batch_size, seqlen, _ = x.shape
    q, k, v = Wqkv(x).chunk(3, dim=-1)

    q_unpad = rearrange(q, 'b s (h d) -> (b s) h d', h=nheads)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=q_unpad.device)
    max_seqlen = seqlen

    k_unpad = rearrange(k, 'b s (h d) -> (b s) h d', h=nheads)
    v_unpad = rearrange(v, 'b s (h d) -> (b s) h d', h=nheads)

    if qkvpacked:
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        return (qkv_unpad, cu_seqlens, max_seqlen)

    else:
        q, k, v = [
            rearrange(z, 'b s (h d) -> b s h d', h=nheads).detach()
            for z in [q, k, v]
        ]

        return (q_unpad, k_unpad, v_unpad, cu_seqlens, max_seqlen)


def MultiHeadAttention(query: Tensor,
                       key: Tensor,
                       value: Tensor,
                       num_heads: int,
                       dropout_p: float = 0.) -> Tensor:
    """
    Arguments:
        query: (batch_size, seqlen, model_dim)
        wq: (model_dim, model_dim)
    """

    # Transpose. After transposed, the layout is:
    # [batch, num_heads, length, head_dim]
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).permute(
        0, 2, 1, 3)
    # [batch, num_heads, head_dim, length]
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).permute(
        0, 2, 3, 1)
    # [batch, num_heads, length, head_dim]
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).permute(
        0, 2, 1, 3)

    d = query.shape[-1]
    # MHA
    scores = torch.matmul(query, key / math.sqrt(d))
    attn = torch.softmax(scores, dim=-1)
    attn = F.dropout(attn, dropout_p)
    out = torch.matmul(attn, value)

    # transpose
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out


def AttentionRef(qkv: Tensor):
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]
    d = q.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(d), k)
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum('bhts,bshd->bthd', attention, v)

    return output
