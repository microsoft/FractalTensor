# Methods

Below is a small test case to illustrate the idea:

- batch_size = 2
- length = 7
- depth = 4
- `ysss` is laid out in `[depth, length, batch_size]`

<p align="center">
<img src="images/access_to_generate_ysss.png" width=60%>
</p>

In this small, it is easy to observe that, within a hyperplance, parallel iterations that access `ysss`, `xs` and `hs` **follows a fixed stride**, thus `xs @ W` and `hs @ U` can be translated into `stridedBMM`.

|No.|ysss|xs|hs|
|:-:|:-:|:-:|:-:|
|3-0|[16]|[2]|[14]|
|3-1|[18,30]|[4,16]|[16,28]|
|3-2|[20,32,44]|[6,18,30]|[18,30,42]|
|3-3|[22,32,44]|[8,20,32]|[20,32,44]|
|3-4|[24,36,48]|[10,22,34]|[22,34,46]|
|3-5|[26,38,50]|[12,24,36]|[24,36,48]|
|3-6|[40,52]|[26,38]|[38,50]|
|3-7|[54]|[40]|[52]|

<p align="center">
<img src="images/lstm.png" width=50%><br>
Fig. The dataflow graph representation for the stacked LSTM network.
</p>
