#!/usr/bin/env bash

SEED=1234


for seq_len in {128..1024..128}
do
  echo $seq_len

  ./build/cudnn_mha \
  -attn_train0 \
  -attn_data_type0 \
  -attn_res_link1 \
  -attn_data_layout3 \
  -attn_num_heads16 \
  -attn_beam_size1 \
  -attn_batch_size32 \
  -attn_q_size512 \
  -attn_k_size512 \
  -attn_v_size512 \
  -attn_proj_q_size512 \
  -attn_proj_k_size512 \
  -attn_proj_v_size512 \
  -attn_proj_o_size512 \
  -attn_res_link0 \
  -attn_seq_len_q$seq_len \
  -attn_seq_len_k$seq_len \
  -attn_sm_scaler1.0 \
  -attn_sweep1 \
  -attn_rand_seed$SEED

done
