# gridLSTM

## Hyper-parameters

1. `batch_size`=20
2. `seq_len`=64
3. `hidden_size`=`input_size`=128
4. `rnn_cell`=`LSTM`
5. `iters` = 20, `warmup` = 10

## Result
  
|Name|PyTorch Average Time| TF_graph Average Time|
|:--|:--|:--|
|gridlstm_gpu:0_forward| 2.6266 |2.5567|
|gridlstm_cpu_forward| 8.6012 |3.7226|

> tf_graph: using tf.compat.v1.session
