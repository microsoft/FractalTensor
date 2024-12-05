import torch

import triton
import triton.language as tl

from time import time

import os
__all__ = ['global_qk']


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_HIDDEN': 32
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_HIDDEN': 32
        }, num_stages=2, num_warps=2),
        triton.Config({
            'BLOCK_HIDDEN': 64
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_HIDDEN': 64
        }, num_stages=2, num_warps=2),
        triton.Config({
            'BLOCK_HIDDEN': 128
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_HIDDEN': 128
        }, num_stages=2, num_warps=2),
        triton.Config({
            'BLOCK_HIDDEN': 256
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_HIDDEN': 256
        }, num_stages=2, num_warps=2),
    ],
    key=['hidden_size'],
)
@triton.jit
def global_qk_kernel(
        Q_ptr,
        K_ptr,
        QK_ptr,
        window_size,
        batch_size,
        hidden_size,
        seq_len,
        block_num,
        stride_Qb,
        stride_Qs,
        stride_Qh,
        stride_Kb,
        stride_Kh,
        stride_Ks,
        stride_QKb,
        stride_QKs,
        stride_QKh,
        block_size: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block_m = tl.program_id(1)
    pid_block_n = tl.program_id(2)
    offset_batch = pid_batch * stride_Qb
    # offset_block = diff_block * block_size * stride_Qs
    # diff_block: the first or the last block
    diff_block = 0
    if pid_block_m == 1:
        diff_block = (block_num - 1)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + offset_batch,
        shape=(seq_len, hidden_size),
        strides=(stride_Qs, stride_Qh),
        offsets=(diff_block * block_size, 0),
        block_shape=(block_size, BLOCK_HIDDEN),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_batch * stride_Kb,
        shape=(hidden_size, seq_len),
        strides=(stride_Kh, stride_Ks),
        # padding
        offsets=(0, (window_size // 2 * block_size)),
        block_shape=(BLOCK_HIDDEN, block_size),
        order=(1, 0),
    )
    qk = tl.zeros([block_size, block_size], dtype=tl.float32)
    for _ in range(0, hidden_size, BLOCK_HIDDEN):
        q = tl.load(Q_block_ptr)
        k = tl.load(K_block_ptr)
        qk += tl.dot(q, k)
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BLOCK_HIDDEN))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_HIDDEN, 0))

    offset_qk_batch = pid_batch * stride_QKb
    diff_qk_m = diff_block * block_size + tl.arange(0, block_size)
    diff_qk_n = pid_block_n * block_size + tl.arange(0, block_size)
    QK_ptrs = (QK_ptr + offset_qk_batch + diff_qk_m[:, None] * stride_QKs +
               diff_qk_n[None, :] * stride_QKh)
    tl.store(QK_ptrs, qk)


def global_qk(Q, K, QK, para):
    (batch_size, global_size, random_size, window_size, hidden_size, seq_len,
     block_num, block_size) = para

    def grid(META):
        return (
            # batch, Q(global row), K
            batch_size,
            2,
            triton.cdiv(seq_len, block_size),
        )

    global_qk_kernel[grid](
        Q_ptr=Q,
        K_ptr=K,
        QK_ptr=QK,
        window_size=window_size,
        batch_size=batch_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        block_num=block_num,
        stride_Qb=Q.stride(0),
        stride_Qs=Q.stride(1),
        stride_Qh=Q.stride(2),
        stride_Kb=K.stride(0),
        stride_Kh=K.stride(1),
        stride_Ks=K.stride(2),
        stride_QKb=QK.stride(0),
        stride_QKs=QK.stride(1),
        stride_QKh=QK.stride(2),
        block_size=block_size,
    )
    return
