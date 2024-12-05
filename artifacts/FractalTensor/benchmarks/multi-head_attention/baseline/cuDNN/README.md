If there is any question, refer to the [original implementation](https://github.com/johnpzh/cudnn_samples_v8/tree/master/multiHeadAttention)

```text
CUDA Version 11.6
CUDNN Version 8.4.1

GeForce RTX 2080 Ti, Compute Capability 7.5
```


|Hyper Parameter|value|
|:--|:--|
|batch_size|32|
|num_heads|16|
|q/k/v_size|512|

|q/k/v seq length|Elapsed Time(ms)|
|:--|:--|
|128|16.178|
|256|30.283|
|384|47.769|
|512|70.292|
|640|95.845|
|768|124.774|
|896|162.018|
|1024|199.508|
