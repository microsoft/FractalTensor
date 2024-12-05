import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_ as init
from torch import Tensor

import triton
import triton.language as tl

from time import time

import os
__all__ = ['LSTMscan']


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 16,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 256,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 16,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 16
            },
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 16
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 64,
                'BLOCK_SIZE_K': 16
            },
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 16,
                'BLOCK_SIZE_K': 16
            },
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 128,
                'BLOCK_SIZE_K': 16
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 256,
                'BLOCK_SIZE_K': 16
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 16,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 16
            },
            num_stages=3,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 64,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 16,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 128,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 256,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 16,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 128
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 64,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 16,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 128,
                'BLOCK_SIZE_K': 128
            },
            num_stages=2,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 256,
                'BLOCK_SIZE_K': 128
            },
            num_stages=2,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 16,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 128
            },
            num_stages=2,
            num_warps=2),
    ],
    key=['hidden_size', 'batch_size'],
)
@triton.jit
def LSTMscan_kernel(
        Wi_ptr,
        Ui_ptr,
        bi_ptr,
        Wf_ptr,
        Uf_ptr,
        bf_ptr,
        Wo_ptr,
        Uo_ptr,
        bo_ptr,
        Wg_ptr,
        Ug_ptr,
        bg_ptr,
        h_prev_ptr,
        c_prev_ptr,
        input_ptr,
        h_ptr,
        c_ptr,
        input_size,
        hidden_size,
        batch_size,
        stride_hm,
        stride_hk,
        stride_wk,
        stride_wn,
        BLOCK_SIZE_B: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    Wi_block_ptr = tl.make_block_ptr(
        base=Wi_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Wf_block_ptr = tl.make_block_ptr(
        base=Wf_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Wo_block_ptr = tl.make_block_ptr(
        base=Wo_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Wg_block_ptr = tl.make_block_ptr(
        base=Wg_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Ui_block_ptr = tl.make_block_ptr(
        base=Ui_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Uf_block_ptr = tl.make_block_ptr(
        base=Uf_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Uo_block_ptr = tl.make_block_ptr(
        base=Uo_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    Ug_block_ptr = tl.make_block_ptr(
        base=Ug_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    h_prev_block_ptr = tl.make_block_ptr(
        base=h_prev_ptr,
        shape=(batch_size, hidden_size),
        strides=(stride_hm, stride_hk),
        offsets=(pid_m * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K),
        order=(1, 0),
    )
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(batch_size, hidden_size),
        strides=(stride_hm, stride_hk),
        offsets=(pid_m * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K),
        order=(1, 0),
    )
    c_prev_block_ptr = tl.make_block_ptr(
        base=c_prev_ptr,
        shape=(batch_size, hidden_size),
        strides=(stride_hm, stride_hk),
        offsets=(pid_m * BLOCK_SIZE_B, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H),
        order=(1, 0),
    )
    offset_batch = (
        pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % batch_size
    offset_hidden = (
        pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)) % hidden_size
    bi_ptrs = bi_ptr + offset_hidden[None, :]
    bf_ptrs = bf_ptr + offset_hidden[None, :]
    bo_ptrs = bo_ptr + offset_hidden[None, :]
    bg_ptrs = bg_ptr + offset_hidden[None, :]
    bi, bf, bo, bg = tl.load(bi_ptrs), tl.load(bf_ptrs), tl.load(
        bo_ptrs), tl.load(bg_ptrs)
    bi_ = tl.broadcast_to(bi, (BLOCK_SIZE_B, BLOCK_SIZE_H))
    bf_ = tl.broadcast_to(bf, (BLOCK_SIZE_B, BLOCK_SIZE_H))
    bo_ = tl.broadcast_to(bo, (BLOCK_SIZE_B, BLOCK_SIZE_H))
    bg_ = tl.broadcast_to(bg, (BLOCK_SIZE_B, BLOCK_SIZE_H))

    ig_ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    fg_ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    og_ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    c_candidate_ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    for k in range(hidden_size // BLOCK_SIZE_K):
        input = tl.load(input_block_ptr)
        h_prew = tl.load(h_prev_block_ptr)
        Wi, Wf, Wo, Wg = tl.load(Wi_block_ptr), tl.load(Wf_block_ptr), tl.load(
            Wo_block_ptr), tl.load(Wg_block_ptr)
        Ui, Uf, Uo, Ug = tl.load(Ui_block_ptr), tl.load(Ui_block_ptr), tl.load(
            Uo_block_ptr), tl.load(Ug_block_ptr)

        ig_ += tl.dot(input, Wi) + tl.dot(h_prew, Ui)
        fg_ += tl.dot(input, Wf) + tl.dot(h_prew, Uf)
        og_ += tl.dot(input, Wo) + tl.dot(h_prew, Uo)
        c_candidate_ += tl.dot(input, Wg) + tl.dot(h_prew, Ug)

        Wi_block_ptr = tl.advance(Wi_block_ptr, (BLOCK_SIZE_K, 0))
        Wf_block_ptr = tl.advance(Wf_block_ptr, (BLOCK_SIZE_K, 0))
        Wo_block_ptr = tl.advance(Wo_block_ptr, (BLOCK_SIZE_K, 0))
        Wg_block_ptr = tl.advance(Wg_block_ptr, (BLOCK_SIZE_K, 0))

        Ui_block_ptr = tl.advance(Ui_block_ptr, (BLOCK_SIZE_K, 0))
        Uf_block_ptr = tl.advance(Uf_block_ptr, (BLOCK_SIZE_K, 0))
        Uo_block_ptr = tl.advance(Uo_block_ptr, (BLOCK_SIZE_K, 0))
        Ug_block_ptr = tl.advance(Ug_block_ptr, (BLOCK_SIZE_K, 0))

        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_SIZE_K))
        h_prev_block_ptr = tl.advance(h_prev_block_ptr, (0, BLOCK_SIZE_K))

    ig = ig_ + bi_
    fg = fg_ + bf_
    og = og_ + bo_
    c_candidate = c_candidate_ + bg_

    ig = _sigmoid(ig)
    fg = _sigmoid(fg)
    og = _sigmoid(og)
    c_candidate = _tanh(c_candidate)

    c_prev = tl.load(c_prev_block_ptr)
    c = fg * c_prev + ig * c_candidate

    c_ptrs = c_ptr + offset_batch[:, None] * \
        stride_hm + offset_hidden[None, :] * stride_hk
    tl.store(c_ptrs, c)

    c = _tanh(c)
    h = og * c
    h_ptrs = h_ptr + offset_batch[:, None] * \
        stride_hm + offset_hidden[None, :] * stride_hk
    tl.store(h_ptrs, h)


@triton.jit
def _dot(a, b):
    return tl.sum(a[:, :, None] * b[None, :, :], axis=1)


@triton.jit
def _sigmoid(x):
    # \sigma(x) = \frac{1}{1 + 2^{-x \cdot \log_2(e)}}
    log2_e = 1.4426950408889634  # log2(e)
    neg_log2_e_x = -x * log2_e
    exp_neg_log2_e_x = tl.math.exp2(neg_log2_e_x)
    return 1 / (1 + exp_neg_log2_e_x)


@triton.jit
def _tanh(x):
    return 2 * _sigmoid(2 * x) - 1


def LSTMscan(input_,
             weight_,
             blas_,
             state_,
             resident_,
             size_,
             device_='cuda',
             dtype_=torch.float16):
    Wi, Wf, Wo, Wg, Ui, Uf, Uo, Ug = weight_
    bi, bf, bo, bg = blas_
    h_prew, c_prew = state_
    input_size, hidden_size, batch_size = size_
    h_resident, c_resident = resident_

    def grid(META):
        return (
            triton.cdiv(batch_size, META['BLOCK_SIZE_B']),
            triton.cdiv(hidden_size, META['BLOCK_SIZE_H']),
        )

    LSTMscan_kernel[grid](
        Wi_ptr=Wi,
        Ui_ptr=Ui,
        bi_ptr=bi,
        Wf_ptr=Wf,
        Uf_ptr=Uf,
        bf_ptr=bf,
        Wo_ptr=Wo,
        Uo_ptr=Uo,
        bo_ptr=bo,
        Wg_ptr=Wg,
        Ug_ptr=Ug,
        bg_ptr=bg,
        h_prev_ptr=h_prew,
        c_prev_ptr=c_prew,
        input_ptr=input_,
        h_ptr=h_resident,
        c_ptr=c_resident,
        input_size=input_size,
        hidden_size=hidden_size,
        batch_size=batch_size,
        stride_hm=h_resident.stride(0),
        stride_hk=h_resident.stride(1),
        stride_wk=Wi.stride(0),
        stride_wn=Wi.stride(1))
    return h_resident, c_resident
