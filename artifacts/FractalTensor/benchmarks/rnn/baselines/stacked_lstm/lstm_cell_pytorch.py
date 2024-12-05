import os
from time import time
from collections import namedtuple
import sys

import torch
from torch.profiler import profiler, record_function, ProfilerActivity

import pt_model as model
torch.manual_seed(1234)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def GetTestName(cell_type):
    if cell_type == 'cudnn_lstm':
        return 'CuDNN'
    elif cell_type == 'v2':
        return 'PT_JITed'


def run_lstm_cell_pytorch_cudnn(batch_size, seq_len, hidden, depth, cell_type):
    input_shape = [seq_len, batch_size, hidden]
    torch.backends.cudnn.enabled = True
    device = 'cuda:0'

    x = torch.randn(*input_shape, device=device)

    m = model.small_model(
        batch_size=batch_size,
        cell_type=cell_type,
        max_seq_length=seq_len,
        hidden_size=hidden,
        num_layers=depth).to(device)
    m = torch.jit.script(m)
    m.eval()

    torch.cuda.synchronize()
    for i in range(10):  # warmup
        output = m(x)

    iter_count = 1000

    torch.cuda.synchronize()
    start = time()
    for i in range(iter_count):
        output = m(x)
    total_time = time() - start
    return total_time / iter_count  # count in seconds


def run_lstm_cell_pytorch_cudnn_profiler(batch_size, seq_len, hidden, depth,
                                         cell_type):
    input_shape = [seq_len, batch_size, hidden]
    torch.backends.cudnn.enabled = True
    device = 'cuda:0'

    x = torch.randn(*input_shape, device=device)
    m = model.small_model(
        batch_size=batch_size,
        cell_type=cell_type,
        max_seq_length=seq_len,
        hidden_size=hidden,
        num_layers=depth).to(device)
    m = torch.jit.script(m)
    m.eval()

    warmup = 10
    for _ in range(warmup):
        m(x)
        torch.cuda.synchronize()

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("lstm_cell"):
            m(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    event_list = prof.key_averages()
    cuda_time = None

    for event in event_list:
        if event.key == "lstm_cell":
            cuda_time = event.cuda_time_total
            break

    return cuda_time


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python lstm_cell_pytorch.py [tsv_file_path]")
        sys.exit(1)

    tsv = sys.argv[1]

    max_depth = 20
    print("Pytorch LstmCell Benchmark......")

    hidden_sizes = [128, 256, 512, 1024]
    batch_sizes = [32, 64, 128, 256]

    depth = 1
    seq_length = 1
    with open(tsv, 'w') as f:
        f.write(
            "[depth, seq_length, batch_size, hidden_size]\tTestName\tAvgTime(ms)\n"
        )
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                t = run_lstm_cell_pytorch_cudnn(
                    batch_size, seq_length, hidden_size, depth, 'cudnn_lstm')
                f.write('[%d, %d, %d, %d]\t%s\t%.5f\n' %
                        (depth, seq_length, batch_size, hidden_size, 'CuDNN',
                         t * 1000))
