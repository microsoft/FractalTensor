import torch

import triton
import triton.language as tl

from time import time

import os
__all__ = ['global_wv']


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 32
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 32
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 64
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 64
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 64,
                'BLOCK_HIDDEN': 32
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 64,
                'BLOCK_HIDDEN': 32
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 64,
                'BLOCK_HIDDEN': 64
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 64,
                'BLOCK_HIDDEN': 64
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 128
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 128
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 128,
                'BLOCK_HIDDEN': 32
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 128,
                'BLOCK_HIDDEN': 32
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 256,
                'BLOCK_HIDDEN': 32
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 256,
                'BLOCK_HIDDEN': 32
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 256
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 32,
                'BLOCK_HIDDEN': 256
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 128,
                'BLOCK_HIDDEN': 128
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 128,
                'BLOCK_HIDDEN': 128
            }, num_stages=2, num_warps=2),
        triton.Config(
            {
                'BLOCK_SEQ': 256,
                'BLOCK_HIDDEN': 256
            }, num_stages=4, num_warps=4),
        triton.Config(
            {
                'BLOCK_SEQ': 256,
                'BLOCK_HIDDEN': 256
            }, num_stages=2, num_warps=2),
    ],
    key=['seq_len', 'hidden_size'],
)
@triton.jit
def global_wv_kernel(
        W_ptr,
        V_ptr,
        O_ptr,
        window_size,
        batch_size,
        hidden_size,
        seq_len,
        block_num,
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
        BLOCK_SEQ: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block_m = tl.program_id(1)
    pid_block_n = tl.program_id(2)
    # offset_batch = pid_batch * global_size * stride_b
    # offset_block = diff_block * block_size * stride_s
    # diff_block: the first or the last block
    diff_block = 0
    if pid_block_m == 1:
        diff_block = block_num - 1
    W_block_ptr = tl.make_block_ptr(
        base=W_ptr + pid_batch * stride_Wb,
        shape=(hidden_size, seq_len),
        strides=(stride_Ws, stride_Wh),
        offsets=(diff_block * block_size, 0),
        block_shape=(block_size, BLOCK_SEQ),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_batch * stride_Vb,
        shape=(seq_len, hidden_size),
        strides=(stride_Vs, stride_Vh),
        offsets=((window_size // 2 * block_size), pid_block_n * BLOCK_HIDDEN),
        block_shape=(BLOCK_SEQ, BLOCK_HIDDEN),
        order=(1, 0),
    )

    o = tl.zeros([block_size, BLOCK_HIDDEN], dtype=tl.float32)
    for _ in range(0, seq_len, BLOCK_SEQ):
        w = tl.load(W_block_ptr)
        v = tl.load(V_block_ptr)
        o += tl.dot(w, v)
        W_block_ptr = tl.advance(W_block_ptr, (0, BLOCK_SEQ))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SEQ, 0))

    offset_qk_batch = pid_batch * stride_Ob
    diff_qk_m = diff_block * block_size + tl.arange(0, block_size)
    diff_qk_n = pid_block_n * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    O_ptrs = (O_ptr + offset_qk_batch + diff_qk_m[:, None] * stride_Os +
              diff_qk_n[None, :] * stride_Oh)
    tl.store(O_ptrs, o)


def global_wv(W, V, O, para):
    (batch_size, global_size, random_size, window_size, hidden_size, seq_len,
     block_num, block_size) = para

    def grid(META):
        return (
            # batch, global_row, hidden
            batch_size,
            2,
            triton.cdiv(hidden_size, META['BLOCK_HIDDEN']),
        )

    global_wv_kernel[grid](
        W_ptr=W,
        V_ptr=V,
        O_ptr=O,
        window_size=window_size,
        batch_size=batch_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        block_num=block_num,
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
