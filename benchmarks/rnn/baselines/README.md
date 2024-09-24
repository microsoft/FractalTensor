# Test Environment

``` {.text}
OS: Ubuntu 16.04.7 LTS
TensorFlow version: 2.2.3, compiled by gcc 5.0
PyTorch v1.9.0
CUDA Version 10.2
CUDNN Version 7.6.5
```
## CPU information

```bash
lscpu
```

``` {.text}
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                12   # virtual CPU
On-line CPU(s) list:   0-11
Thread(s) per core:    2
Core(s) per socket:    6
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 63
Model name:            Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz
Stepping:              2
CPU MHz:               1200.117
CPU max MHz:           3700.0000
CPU min MHz:           1200.0000
BogoMIPS:              7000.36
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              15360K
NUMA node0 CPU(s):     0-11
```

### GPU information

GeForce RTX 2080 Ti, Compute Capability 7.5
