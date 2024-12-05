import gc
import unittest
import time
import sys
import math
import os
import logging
import datetime
import argparse

import test_utils as tu
import tensorflow as tf

from tf_model.rnn2 import StaticRNN
from tf_model.rnn2 import FineGrainedOpLstmNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only print error information.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def force_gpu_sync():
    tf.constant(1).gpu().cpu()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ('True'):
        return True
    elif v in ('False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_test_args():
    parser = argparse.ArgumentParser(description='Girdlstm')
    parser.add_argument(
        '--seq_len', type=int, help='Sequence length', default=32)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=256)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=256)
    parser.add_argument('--depth', type=int, help='Depth size', default=8)
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    parser.add_argument(
        '--default_test',
        type=str2bool,
        help='Whether to run the default test',
        default=False)
    return parser.parse_args()


class TFEagerStackedLSTM(unittest.TestCase):
    WARM_UP = 5
    ITERS = 10

    cmd_args = parse_test_args()
    SEQ_LEN = cmd_args.seq_len
    BATCH_SIZE = cmd_args.batch_size
    HIDDEN = cmd_args.hidden_size
    NUM_LAYERS = cmd_args.depth
    OUTPUT_FILE = cmd_args.output_file
    DEFAULT_TEST = cmd_args.default_test

    PROFILER_ENABLE = False

    def setUp(self):
        tf.compat.v2.random.set_seed(1234)

        self.stddev = 1.0 / math.sqrt(TFEagerStackedLSTM.HIDDEN)
        self.shape = (TFEagerStackedLSTM.BATCH_SIZE,
                      TFEagerStackedLSTM.SEQ_LEN, TFEagerStackedLSTM.HIDDEN)

    def _report(self, test_name, test_case, start):
        '''
        Args:
            test_name (String): Name of the test.
            start (String): Timestamp of the start time.
        '''
        seq_len, batch_size, hidden, num_layers = test_case
        elapsed_time = time.time() - start
        average_time = elapsed_time / TFEagerStackedLSTM.ITERS
        seq_per_sec = (TFEagerStackedLSTM.ITERS *
                       TFEagerStackedLSTM.BATCH_SIZE) / elapsed_time\

        print(
            f"depth: {num_layers}, seq_length: {seq_len}, batch_size: {batch_size}, "
            f"hidden_size: {hidden}, Tensorflow(ms): {average_time * 1000}ms")

        if self.OUTPUT_FILE:
            with open(self.OUTPUT_FILE, 'a') as fout:
                fout.write(
                    f"{num_layers}\t[{seq_len}, {batch_size}, {hidden}]\t"
                    f"{average_time * 1000}\n")

    def _apply_forward(self, test_case, dev, test_name, model):
        '''Only Test the forward computation.
        Args:
            dev, String: Device that on which the test is running. cpu or gpu.
            test_name, String: Name of the test.
            model, Callable: The tested model. It should be a callable object.
        '''
        shape = (test_case[0], test_case[1], test_case[2])
        with tf.device(tu.device(dev)):
            data = tf.random.uniform(
                shape, minval=-self.stddev, maxval=self.stddev)

            for i in range(TFEagerStackedLSTM.WARM_UP):
                y = model(data)
            gc.collect()

            if TFEagerStackedLSTM.PROFILER_ENABLE:
                log_dir = 'logs/' + datetime.datetime.now().strftime(
                    '%Y%m%d-%H%M%S') + '_' + test_name
                tf.profiler.experimental.start(log_dir)

            start = time.time()
            for i in range(TFEagerStackedLSTM.ITERS):
                y = model(data)
            self._report(test_name, test_case, start)

            if TFEagerStackedLSTM.PROFILER_ENABLE:
                profiler_result = tf.python.eager.profiler.stop()
                tf.profiler.experimental.stop()

    def test_fine_grained_op_lstm_forward(self):
        if not self.DEFAULT_TEST:
            for device in [
                    # 'cpu',
                    'gpu',
            ]:
                for cell_type in [
                        # 'v1',
                        'v2',
                ]:
                    model = FineGrainedOpLstmNet(
                        input_size=TFEagerStackedLSTM.HIDDEN,
                        hidden_size=TFEagerStackedLSTM.HIDDEN,
                        num_layers=TFEagerStackedLSTM.NUM_LAYERS,
                        cell_type=cell_type)
                    test_case = (self.SEQ_LEN, self.BATCH_SIZE, self.HIDDEN,
                                 self.NUM_LAYERS)
                    self._apply_forward(
                        test_case, device,
                        f'tf_eager_fine_grained_op_lstm_{cell_type}_{device}',
                        model)

    def test_staticlstm_forward(self):
        if not self.DEFAULT_TEST:
            for device in [
                    # 'cpu',
                    'gpu',
            ]:
                model = StaticRNN(
                    hidden_size=TFEagerStackedLSTM.HIDDEN,
                    num_layers=TFEagerStackedLSTM.NUM_LAYERS,
                    use_cudnn_rnn=False)
                test_case = (self.SEQ_LEN, self.BATCH_SIZE, self.HIDDEN,
                             self.NUM_LAYERS)
                self._apply_forward(test_case, device,
                                    f'tf_eager_static_lstm_cell_{device}',
                                    model)

    def test_default_data(self):
        if self.DEFAULT_TEST:

            def build_model(test_case):
                seq_len, batch_size, hidden, num_layers = test_case
                GraphModeModel = StaticRNN(
                    hidden_size=hidden,
                    num_layers=num_layers,
                    use_cudnn_rnn=False)
                return GraphModeModel

            test_cases = [
                [64, 256, 256, 1],
                [64, 256, 256, 4],
                [64, 256, 256, 8],
                [64, 256, 256, 12],
                [64, 256, 256, 16],
                [64, 256, 256, 20],
            ]

            if self.OUTPUT_FILE:
                with open(self.OUTPUT_FILE, 'w') as fout:
                    fout.write(
                        "depth\t[seq_length, batch_size, hidden_size]\tTensorflow-autograph(ms)\n"
                    )
            print('default-tf_autograph_gpu')
            for test_case in test_cases:
                model = build_model(test_case)
                self._apply_forward(test_case, 'gpu', f'tf_autograph_gpu',
                                    model)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution(tu.get_config())
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    unittest.main(argv=['first-arg-is-ignored'])
