# SOSP'24 FractalTensor Artifact Evaluation

## Overview

This branch contains the artifact evaluation for SOSP '24, Paper #227, titled "Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor".

This paper introduces the design and implementation of the FractalTensor programming framework. The framework aims to uncover various optimization opportunities for compiler analysis by encouraging users to explicitly program nested data parallelism and fine-grained data access. It also allows for automatic analysis to exploit this parallelism. We have developed three key components to implement the FractalTensor framework.

1. the [programming interface](FractalTensor/kaleido/), including the front-end implementation of the FracatlTensor ADT and parser (an example for the [parsed result](FractalTensor/examples/stacked_rnn/example.gv.pdf));
2. the dataflow analysis component: [ThrillerFlow](https://github.com/TiledTensor/ThrillerFlow);
3. the header-only macro kernel library: [TiledCUDA](https://github.com/TiledTensor/TiledCUDA) on Nvidia CUDA devices for code generation.

The FractalTensor project will be open-souced at https://github.com/microsoft/FractalTensor. This code branch and document describe the steps to reproduce the benchmark experiments reported in the paper.

### Workloads in benchmark tests

The benchmark tests include six DNN workloads. The links below show where the benchmarks are located and their corresponding FractalTensor programs:

- Stacked LSTM: [[benchmark]](FractalTensor/benchmarks/rnn/fractaltensor/cute_stacked_lstm/), [[fractal tensor program]](FractalTensor/examples/stacked_rnn/stacked_rnn.py)
- Stacked Dilated RNNs: [[benchmark]](FractalTensor/benchmarks/rnn/fractaltensor/cute_dilated_lstm/),[[fractal tensor program]](FractalTensor/examples/dilated_rnn/dilated_rnn.py)
- Stacked Grid RNNs: [[benchmark]](FractalTensor/benchmarks/rnn/fractaltensor/grid_lstm/README.md), [[fractal tensor program]](FractalTensor/examples/grid_rnn/grid_rnn.py)
- Back-to-Back GEMMs: [[benchmark]](FractalTensor/benchmarks/fused_two_hgemms/README.md)
- Flash Attention: [[benchmark]](FractalTensor/benchmarks/multi-head_attention/README.md), [[fractal tensor program]](FractalTensor/examples/flash_attention/flash_attention.py)
- BigBird: [[benchmark]](https://github.com/FractalTensor/artifacts/blob/master/FractalTensor/benchmarks/blocked_sparse_attention/README.md), [[fractal tensor program]](FractalTensor/examples/sparse_attention/bigbird.py)

## Hardware pre-requisite

The benchmark results reported in the paper were tested on an NVIDIA A100 GPU and require an NVIDIA GPU with Tensor Cores.

## Software pre-requisite

To run the baselines and build the FractalTensor, the following software package are required.

- [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)
- [PyTorch 2.2.1 JIT](https://pytorch.org/get-started/previous-versions/)
- [TensorFlow 2.2.0](https://www.tensorflow.org/install/pip)
- [Triton 2.2.0](https://triton-lang.org/main/getting-started/installation.html#binary-distributions)
- [TVM commit ef46f4e (Mar 24, 2024)](https://github.com/apache/tvm)
- [cuDNN 8.9.7.29-12](https://developer.nvidia.com/cudnn-archive)
- [CUTLASS commit f9ece1b (Mar 28, 2024)](https://github.com/NVIDIA/cutlass)
- [FlashAttention-2 official implementation](https://github.com/Dao-AILab/flash-attention)
- [CMake 3.18+](https://cmake.org/download/)

If you want to run the code on your own server, CUDA 12.4 and cuDNN 8.9.729-12 should be installed first.

We have included hyperlinks to the official installation documentation for each package. In our tests, we found that the software packages required to run different baselines have varying versions of their dependencies, and they may not be compatible within the same environment. Specifically, TensorFlow has strict requirements for Python and other dependencies. Therefore, it is recommended to install them in isolated Python environments.

> [!NOTE]
> We found that PyTorch and Triton have good compatibility. However, it is recommended to install TVM and TensorFlow in isolated environments.

## Installation

If you are using the server we provided, there is no need to set up the environment. We have pre-installed the baselines in an isolated environment on the AE test server.

### Activate the environment for baselines

To activate the environment before executing the corresponding baselines, follow these steps:

```bash
## before running PyTorch/Triton baselines
## Execute the following command to activate the environment.
source ~/env/torch_env.sh

## before running the TVM baselines
## Execute the following command to activate the environment.
source ~/env/tvm_env.sh

## before running TensorFlow baselines
## Execute the following command to activate the environment.
source ~/env/tensorflow_env.sh
```

We also compare our results with cutlass's fused multi-head attention, which requires building the cutlass project. To facilitate this, we provide the script [build_cutlass.sh](build_cutlass.sh) to download and build cutass.

**However, building CUTLASS takes approximately 3 hours. If you are using the server we provided, CUTLASS has already been pre-built for you and can be found in `~/cutlass`.**

### Build FractalTensor

To build the FractalTensor code, CMake version 3.18 or higher is required. We provide a single shell script, [run_build.sh](run_build.sh), for the building process of FractalTensor and all benchmark tests.

```bash
sh run_build.sh
```

> [!NOTE]
> *The `run_build.sh` script should be executed from the root directory of the codebase.*

## Run all the benchmark tests

Since the benchmarks reported in the paper involve running the worklos we listed at the beginning of this document under various settings for different baselines, we provide scripts to run all the benchmarks and collect the data reported in the paper.

### 1. Run all baselines

1. Run all the PyTorch and Triton baselines:

    ```bash
    source ~/env/torch_env.sh
    ./run_all_tests_pt.sh
    ./run_all_tests_triton.sh
    ```
    The execution will take approximately 3 minutes.

1. Run all the Tensorflow baseline:

    ```base
    source ~/env/tensorflow_env.sh
    ./run_all_tests_tf.sh
    ```
    The execution will take approximately 2 minutes.

1. Run cutlass fused multi-head attention.

    Before running this script, please ensure that the pre-built cutlass is located in the root directory of the codebase. If you are using the server we provided, create a symlink to link CUTLASS into the current directory.

    ```bash
    ln -s ~/cutlass cutlass

    ./run_cutlass_mha.sh
    ```
    The execution will take approximately 1 minutes.

1. Run all the TVM baselines

    ```bash
    source ~/env/tvm_env.sh
    ./run_all_tests_tvm.sh
    ```

    However, running the TVM baselines takes several days. We provide the code and scripts needed to run these tests.

### 2. Run all FractalTensor benchmarks

Once the FractalTensor is successfully built, running [run_all_tests_ft.sh](run_all_tests_ft.sh) as shown below will execute all the benchmark tests reported in the paper.

```bash
sh run_all_tests_ft.sh
```

> [!NOTE]
> *The `run_all_tests_ft.sh` script should be executed from the root directory of the codebase.*

The execution will take approximately 3 minutes. Once completed, all the benchmark logs can be found in the `logs` directory with the following structure. You can review them one by one:

```text
logs/
├── ft_attention.tsv
├── ft_b2b_gemm.tsv
├── ft_bigbird.tsv
├── ft_dilated_lstm.tsv
├── ft_grid_lstm.tsv
└── ft_stacked_lstm.tsv
```

## Reproducing individual experiment results

We provided scripts to post-process the original logs to obtain the results presented in our paper, including:

|Experiments|Figure # in Paper|Script Location|Instructions|
|:--|:--|:--|:--|
|#1 Data parallelism is under-utilization in the DAG approach|figure2|[run_all.sh](figure2/run_all.sh)|[README.md](figure2/README.md)|
|#2 Overall performance|figure7|[run_all.sh](figure7/run_all.sh)|[README.md](figure7/README.md)|
|#3 Data parallelism exploitation when aggregation patterns are nested|figure9|[run_all.sh](figure9/run_all.sh)|[README.md](figure9/README.md)|
|#4 Memory performance|table 6||[README.md](table6/README.md)|

Results can be processed by running the `run_all.sh` script within each experiment's directory. 

> [!NOTE]
> *The `run_all.sh` script should be executed within each experiment's directory, not from the root directory of the codebase.*

Once executed, the processed logs will be generated in their respective directories. We have included a `README.md` file in each experiment's directory to provide more specific explanations about each experiment.

For Table 6, we need to generate profiling data interactively using NVIDIA Nsight Compute (ncu), CUDA's profiler tool. A README file is included with instructions on how to do this.
