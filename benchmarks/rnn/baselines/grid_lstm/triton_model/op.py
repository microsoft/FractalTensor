import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_ as init
from torch import Tensor

import triton
import triton.language as tl

from time import time

import os
__all__ = ['Vanilla_scan']


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
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 64,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 16,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 128,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 256,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 16,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
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
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_B': 32,
                'BLOCK_SIZE_H': 256,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_B': 16,
                'BLOCK_SIZE_H': 32,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
            num_warps=2),
    ],
    key=['hidden_size', 'batch_size'],
)
@triton.jit
def Vanilla_scan_kernel(
        W_ptr,
        U_ptr,
        b_ptr,
        x_ptr,
        y_ptr,
        state_ptr,
        h_x_ptr,
        h_y_ptr,
        hidden_size,
        batch_size,
        grid_dim,
        stride_wk,
        stride_wn,
        stride_uk,
        stride_un,
        stride_xm,
        stride_xk,
        stride_sm,
        stride_sk,
        BLOCK_SIZE_B: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    W_block_ptr = tl.make_block_ptr(
        base=W_ptr,
        shape=(hidden_size, hidden_size),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    U_block_ptr = tl.make_block_ptr(
        base=U_ptr,
        shape=(hidden_size * grid_dim, hidden_size),
        strides=(stride_uk, stride_un),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_H),
        order=(1, 0),
    )
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(batch_size, hidden_size),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(batch_size, hidden_size),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K),
        order=(1, 0),
    )
    state_block_ptr = tl.make_block_ptr(
        base=state_ptr,
        shape=(batch_size, hidden_size * grid_dim),
        strides=(stride_sm, stride_sk),
        offsets=(pid_m * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_K),
        order=(1, 0),
    )
    offset_batch = (
        pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % batch_size
    offset_hidden = (
        pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)) % hidden_size
    b_ptrs = b_ptr + offset_hidden[None, :]
    b = tl.load(b_ptrs)
    b_ = tl.broadcast_to(b, (BLOCK_SIZE_B, BLOCK_SIZE_H))
    temp = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    h_x = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    h_y = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    for k in range(hidden_size * grid_dim // BLOCK_SIZE_K):
        state = tl.load(state_block_ptr)
        U = tl.load(U_block_ptr)
        temp += tl.dot(state, U)
        U_block_ptr = tl.advance(U_block_ptr, (BLOCK_SIZE_K, 0))
        state_block_ptr = tl.advance(state_block_ptr, (0, BLOCK_SIZE_K))
    temp = temp + b_
    for k in range(hidden_size // BLOCK_SIZE_K):
        x = tl.load(x_block_ptr)
        y = tl.load(y_block_ptr)
        W = tl.load(W_block_ptr)
        h_x += tl.dot(x, W)
        h_y += tl.dot(y, W)
        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_SIZE_K, 0))
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_K))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_SIZE_K))
    h_x = _tanh(h_x + temp)
    h_y = _tanh(h_y + temp)

    h_x_ptrs = h_x_ptr + offset_batch[:,
                                      None] * stride_xm + offset_hidden[None, :] * stride_xk
    h_y_ptrs = h_y_ptr + offset_batch[:,
                                      None] * stride_xm + offset_hidden[None, :] * stride_xk
    tl.store(h_x_ptrs, h_x)
    tl.store(h_y_ptrs, h_y)


@triton.jit
def _dot(a, b):
    return tl.sum(a[:, :, None] * b[None, :, :], axis=1)


@triton.jit
def _sigmoid(x):
    #\sigma(x) = \frac{1}{1 + 2^{-x \cdot \log_2(e)}}
    log2_e = 1.4426950408889634  # log2(e)
    neg_log2_e_x = -x * log2_e
    exp_neg_log2_e_x = tl.math.exp2(neg_log2_e_x)
    return 1 / (1 + exp_neg_log2_e_x)


@triton.jit
def _tanh(x):
    return 2 * _sigmoid(2 * x) - 1


def Vanilla_scan(weight_,
                 input_,
                 blas_,
                 state_,
                 resident_,
                 size_,
                 device_='cuda',
                 dtype_=torch.float32):
    W, U = weight_
    x_t, y_t = input_
    h_x, h_y = resident_
    hidden_size, batch_size, grid_dim = size_
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_SIZE_B']), triton.cdiv(hidden_size, META['BLOCK_SIZE_H']),
    )
    Vanilla_scan_kernel[grid](
        W_ptr=W,
        U_ptr=U,
        b_ptr=blas_,
        x_ptr=x_t,
        y_ptr=y_t,
        state_ptr=state_,
        h_x_ptr=h_x,
        h_y_ptr=h_y,
        hidden_size=hidden_size,
        batch_size=batch_size,
        grid_dim=grid_dim,
        stride_wk=W.stride(0),
        stride_wn=W.stride(1),
        stride_uk=U.stride(0),
        stride_un=U.stride(1),
        stride_xm=x_t.stride(0),
        stride_xk=x_t.stride(1),
        stride_sm=state_.stride(0),
        stride_sk=state_.stride(1),
    )
    return h_x, h_y
