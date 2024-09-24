import json
import numpy as np

import tvm
from tvm.relay.frontend.pytorch import from_pytorch
import tvm.contrib.graph_executor as runtime
from utils import get_logger

import torch
import lstm

seq_len = 100
batch_size = 128
input_size = 512
hidden_size = 512
num_layers = 3

device = 'cuda'

logger = get_logger()


def lstm_cell():
    state_tensor_shape = (batch_size, hidden_size)
    model = lstm.LSTMCell(input_size, hidden_size).to(device)
    logger.info('torch script graph:\n\n{}\n\n'.format(model.graph))

    inp = torch.randn(batch_size, input_size, device=device)
    h_prev = torch.randn(state_tensor_shape, device=device)
    c_prev = torch.randn(state_tensor_shape, device=device)

    with torch.no_grad():
        y = model(inp, h_prev, c_prev)

    input_shapes = [
        ('input', (batch_size, input_size)),
        ('h_prev', state_tensor_shape),  #
        ('c_prev', state_tensor_shape)
    ]

    relay_model, params = from_pytorch(torch.jit.script(model), input_shapes)
    logger.info('relay text form:\n\n{}\n\n'.format(relay_model))

    target = tvm.target.Target(target='cuda', host='llvm')
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(relay_model, target=target, params=params)
        lib.export_library('lstm_cell.so')

        with open('lstm_cell_relay_build.json', 'w', encoding='utf-8') as f:
            graph_str = lib.get_graph_json()
            data = json.dumps(
                json.loads(graph_str), ensure_ascii=False, indent=4)
            f.write(data)


def test_lstm_cell():
    lib = tvm.runtime.load_module('lstm_cell.so')

    x = np.random.uniform(
        -1, 1, size=(batch_size, input_size)).astype('float32')
    h_prev = np.random.uniform(
        -1, 1, size=(batch_size, hidden_size)).astype('float32')
    c_prev = np.random.uniform(
        -1, 1, size=(batch_size, hidden_size)).astype('float32')

    dev = tvm.cuda()
    module = runtime.GraphModule(lib["default"](dev))  # cast into GraphModule

    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=100)

    module.set_input('input', tvm.nd.array(x))
    module.set_input('h_prev', tvm.nd.array(h_prev))
    module.set_input('c_prev', tvm.nd.array(c_prev))
    prof_res = np.array(
        ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print("%-20s %-19s" % ("%.4f ms" % np.mean(prof_res),
                           "%.4f ms" % np.std(prof_res)))


def lstm_layer():
    state_tensor_shape = (batch_size, hidden_size)
    input_shapes = [
        ('inputs', (seq_len, batch_size, input_size)),
        ('state_c', state_tensor_shape),  # cell
        ('state_h', state_tensor_shape)  # hidden
    ]

    inp = torch.randn(seq_len, batch_size, input_size, device=device)
    state_c = torch.randn(state_tensor_shape, device=device)
    state_h = torch.randn(state_tensor_shape, device=device)

    model = lstm.LSTMLayer(input_size, hidden_size).to(device)
    print(model.graph)

    with torch.no_grad():
        pt_rv = model(inp, state_c, state_h)

    relay_model, params = from_pytorch(torch.jit.script(model), input_shapes)
    print(relay_model)

    target = tvm.target.Target(target='cuda', host='llvm')
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(relay_model, target=target, params=params)
        lib.export_library('lstm_layer.so')


def stacked_lstm():
    state_tensor_shape = (batch_size, hidden_size)
    input_shapes_stacked = [
        ('input', (seq_len, batch_size, input_size)),
        ('h_inits',
         state_tensor_shape),  # TODO(ying): how to fill the shape information?
        ('c_inits', state_tensor_shape),
    ]

    inp = torch.randn(seq_len, batch_size, input_size, device=device)
    h_inits = [
        torch.randn(state_tensor_shape, device=device)
        for _ in range(num_layers)
    ]
    c_inits = [
        torch.randn(state_tensor_shape, device=device)
        for _ in range(num_layers)
    ]

    model = lstm.StackedLSTM(input_size, hidden_size).to(device)
    logger.info('Stacked LSTM, torch script graph:\n\n{}\n\n'.format(
        model.graph))

    with torch.no_grad():
        pt_rv = model(inp, h_inits, c_inits)

    relay_model, params = from_pytorch(
        torch.jit.script(model), input_shapes_stacked)
    logger.info('Stacked LSTM, relay text form:\n\n{}\n\n'.format(relay))


if __name__ == '__main__':
    lstm_cell()
    # test_lstm_cell()
