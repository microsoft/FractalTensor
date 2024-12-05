## Profile about memory behavior

The profiling results shown in Table 6 are based on [NVIDIA Nsight Compute (ncu)](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), which offers a non-interactive method to analyze NVIDIA CUDA kernels via the command line. Below, we describe the steps to use ncu to reproduce the results presented in Table 6.

### Environment preparation

1. Ensure that your test cluster account has root privileges, as ncu tools require these permissions.
2. Use the root account to install FractalTensor and the other baseline tools on the cluster.

### Usage

1. Locate the executable file for the ncu tool: it is typically found in the `bin` directory of the CUDA toolkit.
   
   - For instance, the ncu executable is usually located in `$CUDA_HOME/bin/ncu`.

2. Profile the kernel: To measure the memory traffic through the kernel, use ncu's `--section "MemoryWorkloadAnalysis"` parameter.
   
   - For example, to test the memory behavior of FractalTensor on the FlashAttention2 benchmark, the following command can be used:
  
      ```bash
      mha_dir="$benchmark_dir/multi-head_attention/fractaltensor/build"
      mha_exe="$mha_dir/main"
      ncu --section "MemoryWorkloadAnalysis" --csv --set full $mha_exe > profile_ft.csv
      ```
      
   - For example, to test the memory behavior of the Triton baseline on the FlashAttention2 benchmark, the following command can be used:
   
      ```bash
      benchmark_dir="FractalTensor/benchmarks"
      mha_dir="$benchmark_dir/multi-head_attention/baseline"
      ncu --section "MemoryWorkloadAnalysis" \
          --csv --set full python3 $mha_dir/test_triton_model.py > profile_triton.csv
      ```

3. Analyze the profile results

   In the output file of the profile results, you will find the memory traffic behavior of the kernel of interest. You can then further process and analyze these results.
   
   A problem is that we cannot use pre-assigned names in a general script to identify kernels we monitored since benchmarks not implemented by FractalTensor, such as Triton and PyTorch, have internal implementations that call extra kernels, and ncu will monitor them all. These kernels should not all be measure. We have to address this problem through manual observation of the logs first as following:
   
   During the profiling process, we know that the monitored kernel will be executed for 'warm-up' plus 'actual runs' times specified by us. Suppose this number is `N`. We then identify the names of the kernels in the log that executed `N` times. Often, these kernels are the ones we are concerned with. We then calculate the metrics for the kernels invoked after the warm-up phase, as these are the ones we need to measure.

### Run the test

We have prepared a testing environment on the provided server to run the tests.

>The following command should be executed in the `artifacts` directory of the project, instead of in the `table6` directory.

1. The script [run_all_ncu_cutlass.sh](../run_all_ncu_cutlass.sh) is used to run the test for Flash Attention 2, implemented in CuTlass.

  ```bash
   sudo -i # Switch to root account
   cd /home/sosp/nnfusion/artifacts
   ./run_all_ncu_cutlass.sh
   ```

2. The script [run_all_ncu_flash2.sh](../run_all_ncu_flash2.sh) is used to run the test for Flash Attention 2, from the author's official implementation.

   ```bash
   sudo -i # Switch to root account
   cd /home/sosp/nnfusion/artifacts
   # Choose the environment you want to test
   source /home/sosp/env/torch_env.sh
   ./run_all_ncu_flash2.sh
   ```

3. The script [run_all_ncu_ft.sh](../run_all_ncu_ft.sh) is used to run the test for BigBird and Flash Attention, implemented in FractalTensor.

   ```bash
   sudo -i # Switch to root account
   cd /home/sosp/nnfusion/artifacts
   ./run_all_ncu_ft.sh
   ```

4. The script [run_all_ncu_pt.sh](../run_all_ncu_pt.sh) is used to run the test for BigBird, implemented in PyTorch.

   ```bash
   sudo -i # Switch to root account
   cd /home/sosp/nnfusion/artifacts
   # Choose the environment you want to test
   source /home/sosp/env/torch_env.sh
   ./run_all_ncu_pt.sh
   ```

5. The script [run_all_ncu_triton.sh](../run_all_ncu_triton.sh) is used to run the test for BigBird and Flash Attention, implemented in Triton.

   ```bash
   sudo -i # Switch to root account
   cd /home/sosp/nnfusion/artifacts
   # Choose the environment you want to test
   source /home/sosp/env/torch_env.sh
   ./run_all_ncu_triton.sh
   ```
