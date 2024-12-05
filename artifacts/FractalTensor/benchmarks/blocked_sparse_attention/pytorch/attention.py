import os
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from time import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import types
import utils

MAX_SEQ_LEN = 4096  # DO NOT modify this.

__all__ = [
    'BigbirdBlockSpareAttention',
]


def output_file_func(OUTPUT_FILE, cmd_args, run_time):
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'a') as fout:
            fout.write(
                f"{cmd_args.batch_size}\t{cmd_args.seq_len}\t{cmd_args.hidden_size}\t{cmd_args.block_size}\t"
                f"{run_time}\n")


def bigbird_block_rand_mask(from_seq_length: int,
                            to_seq_length: int,
                            from_block_size: int,
                            to_block_size: int,
                            num_rand_blocks: int,
                            last_idx=-1):
    """Create adjacency list of random attention.
    Args:
        from_seq_length: int. length of from sequence.
        to_seq_length: int. length of to sequence.
        from_block_size: int. size of block in from sequence.
        to_block_size: int. size of block in to sequence.
        num_rand_blocks: int. Number of random chunks per row.
        last_idx: if -1 then num_rand_blocks blocks chosen anywhere in
          to sequence, if positive then num_rand_blocks blocks choosen
          only upto last_idx.
    Returns:
        adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
    """
    from_block_num = from_seq_length // from_block_size
    to_block_num = to_seq_length // to_block_size

    if (from_block_num != to_block_num):
        raise ValueError("Error!. The number of blocks needs to be same!")

    # the magic number 2 is the global attention
    # `rand_attn` has a shape of [number_from_blocks, num_rand_blocks]
    rand_attn = np.zeros((from_block_num - 2, num_rand_blocks), dtype=np.int32)
    middle_seq = np.arange(1, to_block_num - 1, dtype=np.int32)
    last = to_block_num - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_block_num - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_block_num - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_block_num - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(
                    middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(
                    middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start],
                                    middle_seq[end + 1:last])))[:r]
    return rand_attn


def create_rand_mask_from_inputs(
        from_blocked_mask: Tensor, to_blocked_mask: Tensor, rand_attn: Tensor,
        num_attention_heads: int, num_rand_blocks: int, batch_size: int,
        from_seq_length: int, from_block_size: int):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
        from_blocked_mask: 2D Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
        to_blocked_mask: int32 Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].
        rand_attn: [batch_size, num_attention_heads,
        from_seq_length//from_block_size-2, num_rand_blocks]
        num_attention_heads: int. Number of attention heads.
        num_rand_blocks: int. Number of random chunks per row.
        batch_size: int. Batch size for computation.
        from_seq_length: int. length of from sequence.
        from_block_size: int. size of block in from sequence.
    Returns:
        float Tensor of shape [batch_size, num_attention_heads,
                            from_seq_length//from_block_size-2,
                            from_block_size, num_rand_blocks*to_block_size].
    """
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = utils.torch_gather4d(to_blocked_mask, rand_attn)
    rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows,
                               num_rand_blocks * from_block_size)
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1],
                             rand_mask)
    return rand_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
    mask = torch.einsum("bf, bt->bft", from_mask, to_mask)

    # expand to create a slot for heads.
    mask = torch.unsqueeze(mask, 1)

    return mask


class BigbirdBlockSpareAttention(nn.Module):
    def __init__(self,
                 num_attention_heads: int,
                 size_per_head: int,
                 num_rand_blocks: int,
                 from_block_size: int,
                 to_block_size: int,
                 seed=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head

        self.num_rand_blocks = num_rand_blocks
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size

        self.seed = seed

    def _attn_func(
            self,
            query_layer: Tensor,  # [batch, head, seq_length, hidden]
            key_layer: Tensor,
            value_layer: Tensor,
            rand_attn: Tensor,
            from_mask: Tensor,
            to_mask: Tensor,
            rand_mask: Tensor,
            band_mask: Tensor,
            batch_size: int,
            from_seq_length: int,
            to_seq_length: int,
            device="cuda:0"):
        # Define shorthands
        h = self.num_attention_heads
        r = self.num_rand_blocks
        d = self.size_per_head
        b = batch_size
        m = from_seq_length
        n = to_seq_length
        wm = self.from_block_size
        wn = self.to_block_size

        # blocked q, k, v are 5D tensors:
        # [batch_size, head_num, from_block_num, from_block_size, size_per_head]
        blocked_query_matrix = query_layer.view((b, h, m // wm, wm, -1))
        # [batch_size, head_num, to_block_num, to_block_size, size_per_head]
        blocked_key_matrix = key_layer.view((b, h, n // wn, wn, -1))
        # [batch_size, head_num, to_block_num, to_block_size, size_per_head]
        blocked_value_matrix = value_layer.view((b, h, n // wn, wn, -1))
        """`gathered_key` and `gathered_value` have a shape of:
                [
                    batch_size,
                    head_num,
                    from_block_num - global_attn_num,
                    rand_attn_num * block_size,
                    size_per_head
                ]
        """
        gathered_key = utils.torch_gather5d(blocked_key_matrix,
                                            rand_attn).view((b, h, n // wn - 2,
                                                             r * wn, -1))
        gathered_value = utils.torch_gather5d(
            blocked_value_matrix, rand_attn).view((b, h, n // wn - 2, r * wn,
                                                   -1))

        # ============== Compute the first component ===================
        #                 the pure global attention
        # ==============================================================
        """
        Q: [batch_size, head_num, block_size, size_per_head]
        K: [batch_size, head_num, to_seq_length, size_per_head]

        The Einsum is equivalent to:
        for (int i = 0; i < batch_size; ++i)
          for (int j = 0; j < head_num; ++j)
            for (m = 0; m < block_size; ++m)
              for (m = 0; n < to_seq_length; ++n)
                out[i, j, m, n] = Q[i, j, m, :] * K[i, j, n, :]   
        """
        first_product = torch.einsum(
            "bhqd,bhkd->bhqk", blocked_query_matrix[:, :, 0, :, :], key_layer)
        first_product = first_product * (1. / np.sqrt(d))
        first_product += (1.0 - to_mask) * -10000.0
        first_attn_weights = F.softmax(first_product, -1)  # [b, h, wm, n]
        """
        Attn_W: [batch_size, head_num, block_size, to_seq_length]
        V: [batch_size, head_num, to_seq_length, size_per_head]

        The Einsum is equivalent to:

        for (int i = 0; i < batch_size, ++i)
          for (int j = 0; j < head_num, ++j)
            for (int m = 0; m < block_size; ++m)
              for (int n = 0; n < size_per_head; ++n)
                out[i, j, m, n] = Attn_W[i, j, m, :] * V[i, j, :, n]
        """
        first_context_layer = torch.einsum("bhqk,bhkd->bhqd",
                                           first_attn_weights, value_layer)
        first_context_layer = torch.unsqueeze(first_context_layer, 2)

        # ================== Compute the second component  ==================
        #      windowed attention is overlapped with global attention
        # ==================================================================
        second_key_mat = torch.cat(
            (
                blocked_key_matrix[:, :, 0, :, :],
                blocked_key_matrix[:, :, 1, :, :],
                blocked_key_matrix[:, :, 2, :, :],
                blocked_key_matrix[:, :, -1, :, :],  #
                gathered_key[:, :, 0, :, :]),
            2)
        second_value_mat = torch.cat(
            (
                blocked_value_matrix[:, :, 0, :, :],
                blocked_value_matrix[:, :, 1, :, :],
                blocked_value_matrix[:, :, 2, :, :],
                blocked_value_matrix[:, :, -1, :, :],  #
                gathered_value[:, :, 0, :, :]),
            2)  # [b, h, (4+r)*wn, -1]
        second_product = torch.einsum("bhqd,bhkd->bhqk",
                                      blocked_query_matrix[:, :, 1, :, :],
                                      second_key_mat)
        second_seq_pad = torch.cat(
            (to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
             torch.ones(b, 1, 1, r * wn, device=device).long()), 3)
        second_rand_pad = torch.cat((torch.ones(
            b, h, wm, 4 * wn, device=device).long(), rand_mask[:, :, 0]), 3)
        second_product = second_product * (1.0 / np.sqrt(d))
        second_product += (
            1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * -10000.0
        second_attn_weights = F.softmax(second_product, -1)
        second_context_layer = torch.einsum(
            "bhqk,bhkd->bhqd", second_attn_weights, second_value_mat)
        second_context_layer = torch.unsqueeze(second_context_layer, 2)

        # =============== Compute the third component  =======================
        #              make windowed attention continuous
        # ====================================================================
        exp_blocked_key_matrix = torch.cat(
            (blocked_key_matrix[:, :, 1:-3, :, :],
             blocked_key_matrix[:, :, 2:-2, :, :],
             blocked_key_matrix[:, :, 3:-1, :, :]), 3)
        exp_blocked_value_matrix = torch.cat(
            (blocked_value_matrix[:, :, 1:-3, :, :],
             blocked_value_matrix[:, :, 2:-2, :, :],
             blocked_value_matrix[:, :, 3:-1, :, :]), 3)
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2, :]
        inner_band_product = torch.einsum(
            "bhlqd,bhlkd->bhlqk", middle_query_matrix,
            exp_blocked_key_matrix)  # windowd attention
        inner_band_product = inner_band_product * (1.0 / np.sqrt(d))

        rand_band_product = torch.einsum(
            "bhlqd,bhlkd->bhlqk", middle_query_matrix,
            gathered_key[:, :, 1:-1, :])  # random attention
        rand_band_product = rand_band_product * (1.0 / np.sqrt(d))

        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix,
            blocked_key_matrix[:, :, 0, :, :])  # global attention
        first_band_product = first_band_product * (1.0 / np.sqrt(d))
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix,
            blocked_key_matrix[:, :, -1, :, :])  # global attention
        last_band_product = last_band_product * (1.0 / np.sqrt(d))

        inner_band_product += (1.0 - band_mask) * -10000.0
        first_band_product += (
            1.0 - torch.unsqueeze(to_mask[:, :, :, :wn], 3)) * -10000.0
        last_band_product += (
            1.0 - torch.unsqueeze(to_mask[:, :, :, -wn:], 3)) * -10000.0
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0

        band_product = torch.cat((first_band_product, inner_band_product,
                                  rand_band_product, last_band_product), -1)
        attn_weights = F.softmax(band_product, -1)

        context_layer = torch.einsum(
            "bhlqk,bhlkd->bhlqd", attn_weights[:, :, :, :, wn:4 * wn],
            exp_blocked_value_matrix)  # windowed attention
        context_layer += torch.einsum(
            "bhlqk,bhlkd->bhlqd", attn_weights[:, :, :, :, 4 * wn:-wn],
            gathered_value[:, :, 1:-1, :])  # random attention
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :wn],
            blocked_value_matrix[:, :, 0, :, :])  # global attention
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -wn:],
            blocked_value_matrix[:, :, -1, :, :])  # global attention

        # ================= Compute the forth component ======================
        #       windowd attention is overlapped with the global attention
        # ====================================================================
        second_last_key_mat = torch.cat(
            (
                blocked_key_matrix[:, :, 0, :, :],
                blocked_key_matrix[:, :, -3, :, :],
                blocked_key_matrix[:, :, -2, :, :],
                blocked_key_matrix[:, :, -1, :, :],  #
                gathered_key[:, :, -1, :, :]),
            2)
        second_last_value_mat = torch.cat(
            (blocked_value_matrix[:, :, 0, :, :],
             blocked_value_matrix[:, :, -3, :, :],
             blocked_value_matrix[:, :, -2, :, :],
             blocked_value_matrix[:, :, -1, :, :],
             gathered_value[:, :, -1, :, :]), 2)
        second_last_product = torch.einsum(
            "bhqd,bhkd->bhqk", blocked_query_matrix[:, :, -2, :, :],
            second_last_key_mat)
        second_last_seq_pad = torch.cat(
            (to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
             torch.ones(b, 1, 1, r * wn, device=device).long()), 3)
        second_last_rand_pad = torch.cat((torch.ones(
            b, h, wm, 4 * wn, device=device).long(), rand_mask[:, :, -1]), 3)
        second_last_product = second_last_product * (1.0 / np.sqrt(d))
        second_last_product += (1.0 - torch.minimum(
            second_last_seq_pad, second_last_rand_pad)) * -10000.0
        second_last_attn_weights = F.softmax(second_last_product, -1)
        second_last_context_layer = torch.einsum(
            "bhqk,bhkd->bhqd", second_last_attn_weights, second_last_value_mat)
        second_last_context_layer = torch.unsqueeze(second_last_context_layer,
                                                    2)

        # ========== Compute the last component ==============
        #               pure global attention
        # ====================================================
        last_product = torch.einsum(
            "bhqd,bhkd->bhqk", blocked_query_matrix[:, :, -1, :, :], key_layer)
        last_product = last_product * (1.0 / np.sqrt(d))
        last_product += (1.0 - to_mask) * -10000.0
        last_attn_weights = F.softmax(last_product, -1)
        last_context_layer = torch.einsum("bhqk,bhkd->bhqd", last_attn_weights,
                                          value_layer)
        last_context_layer = torch.unsqueeze(last_context_layer, 2)

        #=========================== Adjust layout =============================
        context_layer = torch.cat(
            (first_context_layer, second_context_layer, context_layer,
             second_last_context_layer, last_context_layer), 2)
        context_layer = context_layer.view((b, h, m, -1)) * from_mask
        context_layer = context_layer.permute(0, 2, 1, 3)

        return context_layer

    def forward(self,
                query_layer: Tensor,
                key_layer: Tensor,
                value_layer: Tensor,
                band_mask: Tensor,
                from_mask: Tensor,
                to_mask: Tensor,
                from_blocked_mask: Tensor,
                to_blocked_mask: Tensor,
                batch_size: int,
                from_seq_length: int,
                to_seq_length: int,
                output_file=None,
                plan_from_length=None,
                plan_num_rand_blocks=None):
        """BigBird attention sparse calculation using blocks in linear time.

        Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.

        Args:
          query_layer: float Tensor of shape [batch_size, num_attention_heads,
            from_seq_length, size_per_head]
          key_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
          value_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
          band_mask: float32 Tensor of shape:
            [batch_size, 1, from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
            The values should be 1 or 0. The attention scores will
            effectively be set to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          from_mask: float32 Tensor of shape:
            [batch_size, 1, from_seq_length, 1].
            The values should be 1 or 0. The attention scores will effectively
            be set to -infinity for any positions in the mask that are 0, and will be unchanged for positions that are 1.
          to_mask: float32 Tensor of shape:
            [batch_size, 1, 1, to_seq_length].
            The values should be 1 or 0. The attention scores will effectively
            be set to -infinity for any positions in the mask that are 0,
            and will be unchanged for positions that are 1.
          from_blocked_mask: float32 Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            Same as from_mask, just reshaped.
          to_blocked_mask: float32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            Same as to_mask, just reshaped.
          rand_attn: int32 Tensor of shape:
            [num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks],
            specifying which blocks to attend to for each from sequence block (except 2 global ones).
          num_attention_heads: int. Number of attention heads.
          size_per_head: int. Size of each attention head.
          num_rand_blocks: int. Number of random chunks per row.
          from_seq_length: int. length of from sequence.
          to_seq_length: int. length of to sequence.
          from_block_size: int. size of block in from sequence.
          to_block_size: int. size of block in to sequence.

        Returns:
          float Tensor of shape:
            [batch_size, from_seq_length, num_attention_heads, size_per_head].
        """

        # runtime error check
        if (from_seq_length // self.from_block_size !=
                to_seq_length // self.to_block_size):
            raise ValueError("Error! The number of blocks needs to be same!")

        # cast masks to float
        from_mask = from_mask.float()
        to_mask = to_mask.float()
        band_mask = band_mask.float()
        from_blocked_mask = from_blocked_mask.float()
        to_blocked_mask = to_blocked_mask.float()

        # generate random attention and corresponding masks
        np.random.seed(self.seed)
        if from_seq_length in [1024, 3072, 4096]:  # old plans used in paper
            rand_attn = [
                bigbird_block_rand_mask(
                    MAX_SEQ_LEN,
                    MAX_SEQ_LEN,
                    self.from_block_size,
                    self.to_block_size,
                    self.num_rand_blocks,
                    last_idx=1024)[:(
                        from_seq_length // self.from_block_size - 2)]
                for _ in range(self.num_attention_heads)
            ]
        else:
            raise NotImplementedError()

        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.from_numpy(rand_attn).long()
        rand_attn = torch.unsqueeze(rand_attn, 0)
        """`rand_attn` has a shape of: [
           batch_size,
           attn_head_num,
           num_blocks - global_attn_num,
           rand_attn_num
        ]
        dtype = torch.int64
        """
        rand_attn = torch.repeat_interleave(rand_attn, batch_size, 0)

        rand_mask = create_rand_mask_from_inputs(
            from_blocked_mask,
            to_blocked_mask,
            rand_attn,
            self.num_attention_heads,
            self.num_rand_blocks,
            batch_size,
            from_seq_length,
            self.from_block_size,
        )

        # warmup execution
        for i in range(5):
            self._attn_func(query_layer, key_layer, value_layer, rand_attn,
                            from_mask, to_mask, rand_mask, band_mask,
                            batch_size, from_seq_length, to_seq_length)

        #============== Core computation begins from here =====================
        dir_name = 'bs%d_seq%d_block%d' % (batch_size, from_seq_length,
                                           self.from_block_size)
        save_path = os.path.join('log', dir_name)

        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=1, warmup=2, active=5, repeat=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    save_path),
                with_stack=True) as prof:
            for _ in range(3):
                self._attn_func(query_layer, key_layer, value_layer, rand_attn,
                                from_mask, to_mask, rand_mask, band_mask,
                                batch_size, from_seq_length, to_seq_length)
                prof.step()

        key_averages = prof.key_averages()
        total_cpu_time = 0
        total_cuda_time = 0

        for avg in key_averages:
            total_cpu_time += avg.self_cpu_time_total
            total_cuda_time += avg.self_cuda_time_total

        # print(prof.key_averages().table(sort_by='cuda_time_total'))

        print(
            f"block_size: {self.to_block_size}, seq_length: {to_seq_length}, batch_size: {batch_size}, "
            f"hidden_size: {self.size_per_head}, PyTorch(ms): {total_cuda_time / 1000}ms"
        )
        cmd_args = types.SimpleNamespace(
            seq_len=to_seq_length,
            batch_size=batch_size,
            hidden_size=self.size_per_head,
            block_size=self.to_block_size)
        output_file_func(output_file, cmd_args, total_cuda_time / 1000)
