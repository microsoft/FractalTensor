import torch

import triton
import triton.language as tl

from time import time

import os
__all__ = ['global_softmax']


@triton.jit
def global_softmax_kernel(
        QK_ptr,
        softmax_QK_ptr,
        window_size,
        batch_size,
        hidden_size,
        seq_len,
        block_num,
        stride_QKb,
        stride_QKs,
        stride_QKh,
        block_size: tl.constexpr,
        col_size: tl.constexpr,
        col_load: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block_m = tl.program_id(1)
    offset_batch = pid_batch * stride_QKb
    # offset_block = diff_block * block_size * stride_Qs
    # diff_block: the first or the last block
    diff_block = 0
    if pid_block_m == 1:
        diff_block = block_num - 1

    col_offsets = tl.arange(0, col_load)
    row_offsets = diff_block * block_size + tl.arange(0, block_size)
    QK_ptrs = QK_ptr + offset_batch + row_offsets[:,
                                                  None] * stride_QKs + col_offsets[None, :] * stride_QKh
    qk = tl.load(
        QK_ptrs, mask=col_offsets[None, :] < col_size, other=-float('inf'))
    # qk = tl.load(QK_block_ptr)
    qk_minus_max = qk - tl.max(qk, axis=1)[:, None]
    numerator = tl.exp(qk_minus_max)
    denominator = tl.sum(numerator, axis=1)
    softmax_output = numerator / denominator[:, None]

    offset_qk_batch = pid_batch * stride_QKb
    diff_qk_m = diff_block * block_size + tl.arange(0, block_size)
    diff_qk_n = tl.arange(0, col_load)
    softmax_QK_ptrs = (softmax_QK_ptr + offset_qk_batch +
                       diff_qk_m[:, None] * stride_QKs + diff_qk_n[None, :])
    tl.store(
        softmax_QK_ptrs, softmax_output, mask=col_offsets[None, :] < col_size)


def global_softmax(QK, softmax_QK, para):
    (batch_size, global_size, random_size, window_size, hidden_size, seq_len,
     block_num, block_size) = para

    def grid(META):
        return (
            # batch, Q(global_row),
            batch_size,
            2,
        )

    global_softmax_kernel[grid](
        QK_ptr=QK,
        softmax_QK_ptr=softmax_QK,
        window_size=window_size,
        batch_size=batch_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        block_num=block_num,
        stride_QKb=QK.stride(0),
        stride_QKs=QK.stride(1),
        stride_QKh=QK.stride(2),
        block_size=block_size,
        col_size=seq_len,
        col_load=triton.next_power_of_2(seq_len),
    )
    return
