#!/bin/bash
set -e

# RNN
make benchmark CUDNN_HOME=$CUDNN_HOME BENCHMARK_MODEL_CLASS=rnn BENCHMARK_PROJ=fractaltensor BENCHMARK_MODEL=stacked_lstm
make benchmark CUDNN_HOME=$CUDNN_HOME BENCHMARK_MODEL_CLASS=rnn BENCHMARK_PROJ=fractaltensor BENCHMARK_MODEL=grid_lstm
make benchmark CUDNN_HOME=$CUDNN_HOME BENCHMARK_MODEL_CLASS=rnn BENCHMARK_PROJ=fractaltensor BENCHMARK_MODEL=dilated_lstm

# MultiHead Attention
make benchmark CUDNN_HOME=$CUDNN_HOME BENCHMARK_MODEL_CLASS=multi-head_attention BENCHMARK_PROJ=fractaltensor BENCHMARK_MODEL=mha
