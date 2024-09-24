# Test Environment

Test with TVM v0.9.0

# LSTM Cell

- hidden size = 512
- input size = 512
- batch size = 128

| TVM auto-generated kernel|CuDNN LSTM|
|:--|:--|
|0.8607 ms|0.2115 ms|

# A single LSTM layer.

- `(M, N, K) = (batch_size, seq_len, input_size)`
- `hidden_dize` = `input_size`
- For tuner, `num_measure_trials`=200
- `exe_time` in **milliseconds**, else in **seconds**. 
- `*` means tuning in progress, `√` means to be test when GPU is idle

| M | N | K | exe_time_ori (ms)| exe_time_tune (ms)| schedule time (s)|tuning time(s)|build time (s)|
|--|--|--|--|--|--|--|--|
|32|8|32|0.4493|0.182 (0.001)|18.512|259.080|11.191|
|32|32|32||1.219(0.127)|
|32|128|128||4.215 (0.242)|721.718|313.246|798.423|
|64|8|32|0.4516|0.167 (0.002)|17.227|269.304|12.199|
|64|128|128||3.929 (0.004)|684.012|285.584|833.774|
|128|8|32|0.4601|0.186 (0.001)|23.480|249.230|10.436|
|128|8|128||0.316 (0.004)|20.583|338.304|11.355|
|128|10|128||0.378 (0.001)|23.968|281.737|22.522|
|128|16|128||0.627 (0.001)|36.117|311.692|25.102|
|128|32|32|1.4229| 0.728 (0.004)|66.403|249.148|50.899|
|128|32|128|4.0095|1.236 (0.006)|113.846|274.225|45.124|
|128|64|128|8.0108|2.734 (0.035)|233.892|330.043|158.368|
|128|100|128||4.018 (0.046)|436.000|353.194|375.611|
|128|128|32||2.991 (0.007)|838.539|297.875|724.869|
|128|128|64||3.601 (0.008)|657.601|270.664|596.579|
|128|128|128|16.0620|5.406 (0.014)|775.652|311.214|650.439|
|128(2)|128|128||10.202(0.7623)|2577.036|297.362|657.159|
|128|128|256||9.019 (0.956)|616.219|295.275|712.355|
|128|128|512||20.135 (0.737)|702.173|381.202|690.625|
|128|128|1024||[1]|623.378|465.903|644.258|
|128|256|128|30.4828|10.535(0.540)|3369.937|298.461|2801.295|
|128|256|256||15.921 (1.032)|3186.220|302.805|2869.141|
|128|512|128||-|stack overflow|-|-|
|256|128|128||7.567 (0.087)|729.661|303.439|579.785|
|256|128|256||11.728 (1.578)|700.704|307.382|573.924|
|512|128|128||8.077 (1.315)|758.194|307.967|609.049|
|1024|128|128||10.568 (1.685) |745.107|326.712|642.105|

[1]: change `N` in `ulimit -s N` to solve this problem, however, `stack overflow` and `OpenBLAS errors: max RLIMIT_NPROC=XXX, current=XXX` always happen one.

# Stacked LSTM network

[input_size, hidden_size, length, depth]|mean (ms)|median (ms)|max (ms) |min (ms)|std (ms)|
|:--|:--|:--|:--|:--|:--|
|[64, 64, 100, 1]|2.9751|2.9105|4.0721|2.9089|0.2564|
|[64, 64, 100, 2]|5.7520|5.6468|7.9043|5.6449|0.4511|
|[64, 64, 100, 4]|11.1599|11.0088|15.4327|11.0050|0.7496|
|[64, 64, 100, 6]|16.6433|16.5516|20.7199|16.5452|0.5833|
|[64, 64, 100, 8]|22.1412|22.0441|24.5396|22.0253|0.3491|
|[64, 64, 100, 10]|27.5771|27.4451|31.9805|27.4303|0.6327|
|[64, 64, 100, 12]|32.8588|32.7153|36.4737|32.6627|0.5268|
|[64, 64, 100, 14]|38.5312|38.5054|41.4401|38.3107|0.4400|
|[64, 64, 100, 16]|44.2967|44.2726|47.7058|43.9830|0.5154|
|[64, 64, 100, 18]|49.7128|49.7378|52.4634|49.3582|0.4571|
|[64, 64, 100, 20]|54.7555|54.6511|59.7667|54.2759|0.7533|

```text
d=1,bs=32, seq=32, hs=32
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
   1.0292       0.9687       1.1352       0.9444       0.0813
```