import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
from time import perf_counter
from pt_model import pt_attn
import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
import logging
import csv
logging.basicConfig(level=logging.DEBUG)


def load_pt_trace_model_to_tvm(traced_module, input_infos, device, query_shape,
                               key_shape, value_shape, output_shape):
    mod, params = relay.frontend.from_pytorch(traced_module, input_infos)

    target = tvm.target.Target(target=device, host='llvm')

    # compile on CUDA
    print("############################")
    print("Deploy on CUDA, build the relay.")

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)

    module = graph_executor.GraphModule(lib["default"](dev))

    ######################################
    # TVM runtime
    print("#############################")
    print("TVM runtime")
    tvm_dtype = "float32"
    # print(mod)
    query = tvm.nd.array(np.random.uniform(size=query_shape).astype(tvm_dtype))
    key = tvm.nd.array(np.random.uniform(size=key_shape).astype(tvm_dtype))
    value = tvm.nd.array(np.random.uniform(size=value_shape).astype(tvm_dtype))

    module.set_input("query", query)
    module.set_input("key", key)
    module.set_input("value", value)
    # warmup execution
    for i in range(10):
        module.run()
        tvm_output = module.get_output(
            0, tvm.nd.empty(output_shape, dtype=tvm_dtype)).numpy()

    num = 10  # number of times we run module for a single measurement
    rep = 3  # number of measurements (we derive std dev from this)
    timer = module.module.time_evaluator("run", dev, number=num, repeat=rep)

    tcost = timer()
    start = perf_counter()

    module.run()
    output_shape = output_shape

    tvm_output = module.get_output(0,
                                   tvm.nd.empty(output_shape,
                                                dtype=tvm_dtype)).numpy()

    mean = tcost.mean * 1000
    print("TVM Average per sample inference time: %.2fms" % (mean))
    return mean


def load_onnx_model_to_tvm(onnx_model, shape_dict, device, query_shape,
                           key_shape, value_shape, output_shape):

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    target = tvm.target.Target(target=device, host='llvm')

    # compile on CUDA
    print("############################")
    print("Deploy on CUDA, build the relay.")

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)

    module = graph_executor.GraphModule(lib["default"](dev))

    ######################################
    # TVM runtime
    print("#############################")
    print("TVM runtime")
    tvm_dtype = "float32"
    # print(mod)
    query = tvm.nd.array(np.random.uniform(size=query_shape).astype(tvm_dtype))
    key = tvm.nd.array(np.random.uniform(size=key_shape).astype(tvm_dtype))
    value = tvm.nd.array(np.random.uniform(size=value_shape).astype(tvm_dtype))

    module.set_input("query", query)
    module.set_input("key", key)
    module.set_input("value", value)
    # warmup execution
    for i in range(10):
        module.run()
        tvm_output = module.get_output(
            0, tvm.nd.empty(output_shape, dtype=tvm_dtype)).numpy()

    num = 10  # number of times we run module for a single measurement
    rep = 3  # number of measurements (we derive std dev from this)
    timer = module.module.time_evaluator("run", dev, number=num, repeat=rep)

    tcost = timer()
    start = perf_counter()

    module.run()
    output_shape = output_shape

    tvm_output = module.get_output(0,
                                   tvm.nd.empty(output_shape,
                                                dtype=tvm_dtype)).numpy()

    mean = tcost.mean * 1000
    print("TVM Average per sample inference time: %.2fms" % (mean))
    return mean


def onnx_runtime(path, batch_size, query_len, kv_len, query_size, value_size):
    ort_session = ort.InferenceSession(path)

    # warmup
    for i in range(5):
        outputs = ort_session.run(
            None, {
                "query":
                np.random.randn(batch_size, query_len, query_size).astype(
                    np.float32),
                "key":
                np.random.randn(batch_size, kv_len, query_size).astype(
                    np.float32),
                "value":
                np.random.randn(batch_size, kv_len, value_size).astype(
                    np.float32)
            })
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(10):
        outputs = ort_session.run(
            None, {
                "query":
                np.random.randn(batch_size, query_len, query_size).astype(
                    np.float32),
                "key":
                np.random.randn(batch_size, kv_len, query_size).astype(
                    np.float32),
                "value":
                np.random.randn(batch_size, kv_len, value_size).astype(
                    np.float32)
            })
    torch.cuda.synchronize()
    t1 = time.time()
    return round(((t1 - t0) / 10.0) * 1000, 3)


def test_run(len):
    batch_size = 32
    num_heads = 16
    query_len = len
    kv_len = len
    query_size = 512
    value_size = 512
    pt_dtype = torch.float32
    device = 'cuda'
    onnx_path = "pt.onnx"

    query = torch.randn(
        (batch_size, query_len, query_size),
        device=device,
        dtype=pt_dtype,
    )
    key = torch.randn(
        (batch_size, kv_len, query_size),
        device=device,
        dtype=pt_dtype,
    )
    value = torch.randn(
        (batch_size, kv_len, value_size),
        device=device,
        dtype=pt_dtype,
    )

    model = pt_attn.MutilHeadAttention(num_heads=num_heads, d_model=query_size)
    model.eval()

    torch.onnx.export(
        model,
        (query, key, value),
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=['query', 'key', 'value'],
        output_names=['output'],
    )

    traced_module = torch.jit.trace(model.to(device),
                                    [query, key, value]).eval()

    query_shape = (batch_size, query_len, query_size)
    key_shape = (batch_size, kv_len, query_size)
    value_shape = (batch_size, kv_len, value_size)
    output_shape = (batch_size, query_len, value_size)

    input_infos = [("query", (query_shape, "float32")),
                   ("key", (key_shape, "float32")), ("value", (value_shape,
                                                               "float32"))]

    shape_dict = {"query": query_shape, "key": key_shape, "value": value_shape}
    onnx_model = onnx.load(onnx_path)
    pt_model_time = load_pt_trace_model_to_tvm(traced_module, input_infos,
                                               device, query_shape, key_shape,
                                               value_shape, output_shape)
    onnx_model_time = load_onnx_model_to_tvm(onnx_model, shape_dict, device,
                                             query_shape, key_shape,
                                             value_shape, output_shape)
    onnx_model_runtime = onnx_runtime(onnx_path, batch_size, query_len, kv_len,
                                      query_size, value_size)
    return format(pt_model_time, ".3f"), format(onnx_model_time,
                                                ".3f"), format(
                                                    onnx_model_runtime, ".3f")


if __name__ == "__main__":
    with open('figures/tvm_data.tsv', 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow(["model name", 'query len', "total time(ms)"])
        for len in [128, 256, 384, 512, 640, 768, 896]:
            pt_model_time, onnx_model_time, onnx_model_runtime = test_run(len)
            tsv_w.writerow(["pt_model_time", len, pt_model_time])
            tsv_w.writerow(["onnx_model_time", len, onnx_model_time])
            tsv_w.writerow(["onnx_model_runtime", len, onnx_model_runtime])
