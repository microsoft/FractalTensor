# Hyper-parameters

1.  `num_layers` = 8, 8 LSTM layers are stacked
1.  LSTM's `hidden_dim` = `output_dim` = 512
1.  All training samples have a fixed length: `seq_len_` = 100
1.  `batch_size` = 64
1.  `warm_up` = 10, `iteration` = 30

Explanation for some implementations:

|Name|Explanation|
|:--|:--|
|Fine-grained Lstm Cell V1|Compute LSTM's Four gates separatedly.|
|Fine-grained Lstm Cell V2|Manually batch GEMMs in LSTM's four gates into a large GEMM.|
|Static LSTM cell in TensorFlow|LSTM cell as a single operator.|

<p align="center">
<img src="figures/stacked_lstm_perf_with_depth.png" width=50%>
</p>
