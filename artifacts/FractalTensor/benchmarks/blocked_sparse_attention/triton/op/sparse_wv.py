import torch

import triton
import triton.language as tl

from time import time

import os
__all__ = ['sparse_wv']


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
def sparse_wv_kernel(
        W_ptr,
        V_ptr,
        O_ptr,
        random_ptr,
        window_size,
        batch_size,
        global_size,
        hidden_size,
        seq_len,
        block_num,
        random_size,
        stride_randomb,
        stride_randomr,
        stride_Wb,
        stride_Ws,
        stride_Wh,
        stride_Vb,
        stride_Vh,
        stride_Vs,
        stride_Ob,
        stride_Os,
        stride_Oh,
        block_size: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block_m = tl.program_id(1)
    pid_block_n = tl.program_id(2)

    offset_global_row = global_size * block_size * stride_Ws

    # offset_batch = pid_batch * BLOCK_BATCH * stride_b
    # offset_block = diff_block * block_size * stride_s

    sparse_seq_len = (global_size * 2 + window_size + random_size)

    W_block_ptr = tl.make_block_ptr(
        base=W_ptr + pid_batch * stride_Wb + offset_global_row,
        shape=(seq_len - global_size * 2, sparse_seq_len),
        strides=(stride_Ws, stride_Wh),
        offsets=(pid_block_m * block_size, 0),
        block_shape=(block_size, block_size),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_batch * stride_Vb,
        shape=(seq_len + (window_size // 2 * 2 * block_size), hidden_size),
        strides=(stride_Vs, stride_Vh),
        # padding
        offsets=((window_size // 2 * block_size), pid_block_n * BLOCK_HIDDEN),
        block_shape=(block_size, BLOCK_HIDDEN),
        order=(1, 0),
    )

    o = tl.zeros([block_size, BLOCK_HIDDEN], dtype=tl.float32)

    # global_size: first
    w = tl.load(W_block_ptr)
    v = tl.load(V_block_ptr)
    o += tl.dot(w, v)
    W_block_ptr = tl.advance(W_block_ptr, (0, block_size))
    V_block_ptr = tl.advance(V_block_ptr, (block_size * (block_num - 1), 0))

    # global_size: last
    w = tl.load(W_block_ptr)
    v = tl.load(V_block_ptr)
    o += tl.dot(w, v)
    W_block_ptr = tl.advance(W_block_ptr, (0, block_size))
    V_block_ptr = tl.advance(V_block_ptr, (-block_size * (block_num - 1), 0))

    # window_size
    diff_block = pid_block_m - 1
    V_block_ptr = tl.advance(V_block_ptr, (diff_block * block_size, 0))
    for i in range(window_size):
        w = tl.load(W_block_ptr)
        v = tl.load(V_block_ptr)
        o += tl.dot(w, v)
        W_block_ptr = tl.advance(W_block_ptr, (0, block_size))
        V_block_ptr = tl.advance(V_block_ptr, (block_size, 0))

    # random_size
    V_block_ptr = tl.advance(V_block_ptr,
                             ((-diff_block - window_size) * block_size, 0))
    random_ptr = random_ptr + pid_block_m * stride_randomb
    for i in range(random_size):
        random_ptr += i * stride_randomr
        random_index = tl.load(random_ptr)
        V_block_ptr = tl.advance(V_block_ptr, (random_index * block_size, 0))
        w = tl.load(W_block_ptr)
        v = tl.load(V_block_ptr)
        o += tl.dot(w, v)
        W_block_ptr = tl.advance(W_block_ptr, (0, block_size))
        V_block_ptr = tl.advance(V_block_ptr, (-random_index * block_size, 0))

    offset_qk_batch = pid_batch * stride_Ob
    offset_global_row = global_size * block_size * stride_Os
    diff_qk_m = pid_block_m * block_size + tl.arange(0, block_size)
    diff_qk_n = pid_block_n * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    O_ptrs = (O_ptr + offset_qk_batch + offset_global_row +
              diff_qk_m[:, None] * stride_Os + diff_qk_n[None, :] * stride_Oh)
    tl.store(O_ptrs, o)


def sparse_wv(W, V, O, random_index, para):
    (batch_size, global_size, random_size, window_size, hidden_size, seq_len,
     block_num, block_size) = para

    def grid(META):
        return (
            # batch, W(global_row exception), hidden
            batch_size,
            (triton.cdiv(seq_len, block_size) - 2),
            triton.cdiv(hidden_size, META['BLOCK_HIDDEN']),
        )

    sparse_wv_kernel[grid](
        W_ptr=W,
        V_ptr=V,
        O_ptr=O,
        random_ptr=random_index,
        window_size=window_size,
        batch_size=batch_size,
        global_size=global_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        block_num=block_num,
        random_size=random_size,
        stride_randomb=random_index.stride(0),
        stride_randomr=random_index.stride(1),
        stride_Wb=W.stride(0),
        stride_Ws=W.stride(1),
        stride_Wh=W.stride(2),
        stride_Vb=V.stride(0),
        stride_Vh=V.stride(1),
        stride_Vs=V.stride(2),
        stride_Ob=O.stride(0),
        stride_Os=O.stride(1),
        stride_Oh=O.stride(2),
        block_size=block_size,
    )
    return
