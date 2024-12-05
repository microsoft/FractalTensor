import torch

import triton
import triton.language as tl

from time import time

import os
__all__ = ['sparse_qk']


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
def sparse_qk_kernel(
        Q_ptr,
        K_ptr,
        QK_ptr,
        random_ptr,
        window_size,
        batch_size,
        global_size,
        hidden_size,
        seq_len,
        block_num,
        stride_randomb,
        stride_randomr,
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

    # random_ptr: [block_num, random_size]
    random_ptr = (random_ptr + pid_block_m * stride_randomb +
                  (pid_block_n - global_size * 2) * stride_randomr)
    random_index = tl.load(random_ptr)

    diff_block_q = pid_block_m
    if pid_block_n < global_size * 2:
        # global_size == 1, but in fast (global_size * 2)
        diff_block_k = tl.where(pid_block_n == 0, 0, block_num - 1)
    elif pid_block_n >= global_size * 2 + window_size:
        diff_block_k = random_index
    else:
        diff_block_k = pid_block_m - 1

    offset_global_row = global_size * block_size * stride_Qs
    # offset_batch = pid_batch * BLOCK_BATCH * stride_b
    # offset_block = diff_block * block_size * stride_s

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_batch * stride_Qb + offset_global_row,
        shape=(seq_len - global_size * block_size, hidden_size),
        strides=(stride_Qs, stride_Qh),
        offsets=(diff_block_q * block_size, 0),
        block_shape=(block_size, BLOCK_HIDDEN),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_batch * stride_Kb,
        shape=(hidden_size, seq_len + (window_size // 2 * 2 * block_size)),
        strides=(stride_Kh, stride_Ks),
        # padding + global/window/random index
        offsets=(0,
                 (window_size // 2 * block_size) + diff_block_k * block_size),
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

    offset_global_row = 1 * block_size * stride_QKs
    offset_qk_batch = pid_batch * stride_QKb
    diff_qk_m = diff_block_q * block_size + tl.arange(0, block_size)
    diff_qk_n = pid_block_n * block_size + tl.arange(0, block_size)
    QK_ptrs = (
        QK_ptr + offset_qk_batch + offset_global_row +
        diff_qk_m[:, None] * stride_QKs + diff_qk_n[None, :] * stride_QKh)
    tl.store(QK_ptrs, qk)


def sparse_qk(Q, K, QK, random_index, para):
    (batch_size, global_size, random_size, window_size, hidden_size, seq_len,
     block_num, block_size) = para

    def grid(META):
        return (
            # batch, Q, K
            batch_size,
            (triton.cdiv(seq_len, block_size) - 2),
            (global_size * 2 + window_size + random_size),
        )

    sparse_qk_kernel[grid](
        Q_ptr=Q,
        K_ptr=K,
        QK_ptr=QK,
        random_ptr=random_index,
        window_size=window_size,
        batch_size=batch_size,
        global_size=global_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        block_num=block_num,
        stride_randomb=random_index.stride(0),
        stride_randomr=random_index.stride(1),
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
