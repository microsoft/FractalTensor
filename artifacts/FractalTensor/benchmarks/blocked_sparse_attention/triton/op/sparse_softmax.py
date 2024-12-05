import torch

import triton
import triton.language as tl

from time import time

import os
__all__ = ['sparse_softmax']


@triton.jit
def sparse_softmax_kernel(
        QK_ptr,
        softmax_QK_ptr,
        window_size,
        batch_size,
        global_size,
        hidden_size,
        seq_len,
        block_num,
        stride_QKb,
        stride_QKs,
        stride_QKh,
        block_size: tl.constexpr,
        col_size: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block_m = tl.program_id(1)
    offset_batch = pid_batch * stride_QKb
    offset_global_row = 1 * block_size * stride_QKs

    QK_block_ptr = tl.make_block_ptr(
        base=QK_ptr + offset_batch + offset_global_row,
        shape=(seq_len, seq_len),
        strides=(stride_QKs, stride_QKh),
        offsets=(pid_block_m * block_size, 0),
        block_shape=(block_size, col_size),
        order=(1, 0),
    )
    qk = tl.load(QK_block_ptr)
    qk_minus_max = qk - tl.max(qk, axis=1)[:, None]
    numerator = tl.exp(qk_minus_max)
    denominator = tl.sum(numerator, axis=1)
    softmax_output = numerator / denominator[:, None]

    offset_qk_batch = pid_batch * stride_QKb
    diff_qk_m = pid_block_m * block_size + tl.arange(0, block_size)
    diff_qk_n = tl.arange(0, col_size)
    softmax_QK_ptrs = (softmax_QK_ptr + offset_qk_batch + offset_global_row +
                       diff_qk_m[:, None] * stride_QKs + diff_qk_n[None, :])
    tl.store(softmax_QK_ptrs, softmax_output)


def sparse_softmax(QK, softmax_QK, para):
    (batch_size, global_size, random_size, window_size, hidden_size, seq_len,
     block_num, block_size) = para

    def grid(META):
        return (
            # batch, Q(global_row exception),
            batch_size,
            triton.cdiv(seq_len, block_size) - 2,
        )

    sparse_softmax_kernel[grid](
        QK_ptr=QK,
        softmax_QK_ptr=softmax_QK,
        window_size=window_size,
        batch_size=batch_size,
        global_size=global_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        block_num=block_num,
        stride_QKb=QK.stride(0),
        stride_QKs=QK.stride(1),
        stride_QKh=QK.stride(2),
        block_size=block_size,
        col_size=(global_size * 2 + window_size + random_size),
    )
    return
