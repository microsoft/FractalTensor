The second column `Shape` stands for `[batch_size, hidden_size, length, depth]`.

Variable sequence length in batch:  normal distribution (`mean`=`seq_length/2`, `stddev`=`seq_length/8`)

## Fixed sequence length in batch(256)

### Vary in depth

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[256, 256, 100, 2]|||||7.70831|
|CuDNN|[256, 256, 100, 4]|||||13.3942|
|CuDNN|[256, 256, 100, 6]|||||18.7314|
|CuDNN|[256, 256, 100, 8]|||||28.6231|
|CuDNN|[256, 256, 100, 10]|||||36.1529|
|CuDNN|[256, 256, 100, 12]|||||45.2314|
|CuDNN|[256, 256, 100, 14]|||||55.5315|
|CuDNN|[256, 256, 100, 16]|||||64.4784|
|CuDNN|[256, 256, 100, 18]|||||73.7362|
|CuDNN|[256, 256, 100, 20]|||||77.9639|
|CuDNN|[256, 256, 100, 22]|||||87.4949|

### Vary in sequence length

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[256, 256, 50, 10]|||||17.9963|
|CuDNN|[256, 256, 75, 10]|||||28.2362|
|CuDNN|[256, 256, 100, 10]|||||36.75|
|CuDNN|[256, 256, 125, 10]|||||44.5773|
|CuDNN|[256, 256, 150, 10]|||||50.1302|
|CuDNN|[256, 256, 175, 10]|||||59.9653|
|CuDNN|[256, 256, 200, 10]|||||68.3289|

## Variable sequence length in batch(256)

### Vary in depth

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[256, 256, 100, 2]|||||5.72042|
|CuDNN|[256, 256, 100, 4]|||||9.48156|
|CuDNN|[256, 256, 100, 6]|||||16.9558|
|CuDNN|[256, 256, 100, 8]|||||24.1965|
|CuDNN|[256, 256, 100, 10]|||||32.0736|
|CuDNN|[256, 256, 100, 12]|||||38.5646|
|CuDNN|[256, 256, 100, 14]|||||46.5616|
|CuDNN|[256, 256, 100, 16]|||||56.7443|
|CuDNN|[256, 256, 100, 18]|||||63.9878|
|CuDNN|[256, 256, 100, 20]|||||68.5947|
|CuDNN|[256, 256, 100, 22]|||||76.7222|

### Vary in sequence length

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[256, 256, 50, 10]|||||16.0103|
|CuDNN|[256, 256, 75, 10]|||||25.2088|
|CuDNN|[256, 256, 100, 10]|||||33.9192|
|CuDNN|[256, 256, 125, 10]|||||40.1468|
|CuDNN|[256, 256, 150, 10]|||||45.2206|
|CuDNN|[256, 256, 175, 10]|||||54.8865|
|CuDNN|[256, 256, 200, 10]|||||61.0839|

## Fixed sequence length in batch(64)

### Vary in depth

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[64, 256, 100, 2]|||||3.64703|
|CuDNN|[64, 256, 100, 4]|||||5.62296|
|CuDNN|[64, 256, 100, 6]|||||10.4035|
|CuDNN|[64, 256, 100, 8]|||||13.8126|
|CuDNN|[64, 256, 100, 10]|||||17.115|
|CuDNN|[64, 256, 100, 12]|||||22.0281|
|CuDNN|[64, 256, 100, 14]|||||26.3998|
|CuDNN|[64, 256, 100, 16]|||||29.0647|
|CuDNN|[64, 256, 100, 18]|||||36.4702|
|CuDNN|[64, 256, 100, 20]|||||41.028|
|CuDNN|[64, 256, 100, 22]|||||42.3192|

### Vary in sequence length

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[64, 256, 50, 10]|||||8.34408|
|CuDNN|[64, 256, 75, 10]|||||13.3728|
|CuDNN|[64, 256, 100, 10]|||||16.3981|
|CuDNN|[64, 256, 125, 10]|||||22.0347|
|CuDNN|[64, 256, 150, 10]|||||26.8824|
|CuDNN|[64, 256, 175, 10]|||||30.6395|
|CuDNN|[64, 256, 200, 10]|||||31.9782|

## Variable sequence length in batch(64)

### Vary in depth

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[64, 256, 100, 2]|||||2.89689|
|CuDNN|[64, 256, 100, 4]|||||5.9606|
|CuDNN|[64, 256, 100, 6]|||||9.18842|
|CuDNN|[64, 256, 100, 8]|||||12.3405|
|CuDNN|[64, 256, 100, 10]|||||19.093|
|CuDNN|[64, 256, 100, 12]|||||23.3191|
|CuDNN|[64, 256, 100, 14]|||||22.5437|
|CuDNN|[64, 256, 100, 16]|||||27.4492|
|CuDNN|[64, 256, 100, 18]|||||38.8575|
|CuDNN|[64, 256, 100, 20]|||||40.549|
|CuDNN|[64, 256, 100, 22]|||||45.7558|

### Vary in sequence length

||Shape|Gather(ms)|Cell-GEMM(ms)|Cell-Elementwise(ms)|Scatter(ms)|Total(ms)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CuDNN|[64, 256, 50, 10]|||||8.22669|
|CuDNN|[64, 256, 75, 10]|||||11.778|
|CuDNN|[64, 256, 100, 10]|||||15.9492|
|CuDNN|[64, 256, 125, 10]|||||19.4848|
|CuDNN|[64, 256, 150, 10]|||||28.9545|
|CuDNN|[64, 256, 175, 10]|||||33.4161|
|CuDNN|[64, 256, 200, 10]|||||37.8032|
