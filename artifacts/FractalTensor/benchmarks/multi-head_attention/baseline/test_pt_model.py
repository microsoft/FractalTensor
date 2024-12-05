import os
import warnings

import torch
from torch import Tensor
import flash_attn

is_sm80 = torch.cuda.get_device_capability('cuda') >= (8, 0)

try:
    from flash_attn.flash_attn_triton import flash_attn_func
except (ImportError, AttributeError):
    flash_attn_func = None

try:
    # flash_attn2
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as ft_fwd_fun
except (ImportError, AttributeError):

    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func as ft_fwd_fun

from einops import rearrange
from pt_model import pt_attn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')


@torch.no_grad()
def test_pt_mha(qkv: Tensor, batch_size: int, warmup=10, iters=50):
    nnz, n, nheads, d = qkv.shape
    qkv = qkv.view(batch_size, nnz // batch_size, n, nheads, d)

    qkv = qkv.detach().requires_grad_(False)
    for _ in range(warmup):  # warmup
        output_ref = pt_attn.AttentionRef(qkv)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        output_ref = pt_attn.AttentionRef(qkv)
    end_event.record()
    torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(end_event) / iters
    return elapsed, output_ref


@torch.no_grad()
def test_flash_attention(qkv: Tensor,
                         cu_seqlens: Tensor,
                         max_seqlen: Tensor,
                         batch_size: int,
                         dropout_p=0.,
                         warmup=10,
                         iters=50):
    qkv = qkv.detach().requires_grad_(False)
    for _ in range(warmup):  # warmup
        output, _, _ = ft_fwd_fun(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            return_attn_probs=True,
            causal=False)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        output, _, _ = ft_fwd_fun(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            return_attn_probs=True,
            causal=False)
    end_event.record()
    torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(end_event) / iters

    output = rearrange(output, '(b s) h d -> b s h d', b=batch_size)
    return elapsed, output


def make_test(
        batch_size: int,
        nheads: int,
        hidden: int,  # this is head_dim
        seqlen: int,
        device='cuda',
        dtype=torch.float16):

    x = torch.randn(
        batch_size,
        seqlen,
        nheads * hidden,
        device=device,
        dtype=dtype,
        requires_grad=False)
    Wqkv = torch.nn.Linear(
        nheads * hidden, 3 * nheads * hidden, device=device, dtype=dtype)

    qkv, cu_seqlens, max_seqlen = pt_attn.GenerateQKV(
        x, Wqkv, nheads, qkvpacked=True)

    time, _ = test_flash_attention(qkv, cu_seqlens, max_seqlen, batch_size)
    print(f"{seqlen}\t{hidden}\t{time}")


if __name__ == '__main__':
    torch.random.manual_seed(0)
    print('hidden\tseqlen\tflash_attn')
    batch = 32
    nheads = 8
    for hidden in [
            128,
            256,
    ]:
        for seqlen in [
                128,
                256,
                512,
                768,
                1024,
                1536,
                2048,
                4096,
        ]:
            make_test(batch, nheads, hidden, seqlen)
