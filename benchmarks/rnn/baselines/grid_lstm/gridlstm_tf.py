# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import datetime
import logging
import math
import os
import sys
import time
import unittest

import tensorflow as tf
import test_utils as tu
from tf_model import (BaseWhileOpGridLSTMNet, FineGrainedOpGridLSTMNet,
                      WhileOpGridLSTMNet)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only print error information.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_test_args():
    parser = argparse.ArgumentParser(description='Girdlstm')
    parser.add_argument('--seq_len',
                        type=int,
                        help='Sequence length',
                        default=10)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size',
                        default=32)
    parser.add_argument('--hidden_size',
                        type=int,
                        help='Hidden size',
                        default=256)
    parser.add_argument('--depth', type=int, help='Depth size', default=4)
    return parser.parse_args()


class TFGraphGridLSTM(unittest.TestCase):
    WARM_UP = 5
    ITERS = 10

    cmd_args = parse_test_args()
    SEQ_LEN = cmd_args.seq_len
    BATCH_SIZE = cmd_args.batch_size
    HIDDEN_SIZE = cmd_args.hidden_size
    DEPTH = cmd_args.depth

    LOG_DEBUG_INFO = 1
    PROFILER_ENABLE = 0

    def setUp(self):
        tf.compat.v2.random.set_seed(1234)
        self._init_logger()

        self.stddev = 1.0 / math.sqrt(TFGraphGridLSTM.HIDDEN)
        self.shape = (TFGraphGridLSTM.SEQ_LEN, TFGraphGridLSTM.BATCH_SIZE,
                      TFGraphGridLSTM.HIDDEN)

    def _init_logger(self):
        self.logger = logging.getLogger()
        logging.basicConfig(
            level=(logging.DEBUG
                   if TFGraphGridLSTM.LOG_DEBUG_INFO else logging.INFO),
            filename="grid_lstm_results_tensorflow_graph.txt",
            filemode="w",
            format="%(message)s")

    def _report(self, test_name, start):
        """
        Args:
            test_name (String): Name of the test.
            start (String): Timestamp of the start time.
        """
        elapsed_time = time.time() - start
        average_time = elapsed_time / TFGraphGridLSTM.ITERS
        seq_per_sec = (TFGraphGridLSTM.ITERS *
                       TFGraphGridLSTM.BATCH_SIZE) / elapsed_time
        self.logger.info(("|%s|%.4f\t|%.4f\t|%.4f|") %
                         (test_name, average_time, elapsed_time, seq_per_sec))
        print((
            "|test_name = %s|average_time = %.4f s|elapsed_time = %.4f s|seq_per_sec = %.4f|"
        ) % (test_name, average_time, elapsed_time, seq_per_sec))

    def _apply_forward(self, dev, test_name, model):
        """Only Test the forward computation.
        Args:
            dev, String: Device that on which the test is running. cpu or gpu.
            test_name, String: Name of the test.
            model, Callable: The tested model. It should be a callable object.
        """

        with tf.device(tu.device(dev)):
            source = tf.random.uniform(self.shape,
                                       minval=-self.stddev,
                                       maxval=self.stddev)
            target = tf.random.uniform(self.shape,
                                       minval=-self.stddev,
                                       maxval=self.stddev)

            output = model(source, target)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())

                for _ in range(TFGraphGridLSTM.WARM_UP):
                    sess.run(output)

                if TFGraphGridLSTM.PROFILER_ENABLE:
                    log_dir = "logs/" + datetime.datetime.now().strftime(
                        "%Y%m%d-%H%M%S") + "_" + test_name
                    tf.profiler.experimental.start(log_dir)

                start = time.time()
                for _ in range(TFGraphGridLSTM.ITERS):
                    sess.run(output)

                if TFGraphGridLSTM.PROFILER_ENABLE:
                    tf.profiler.experimental.stop()

            self._report(test_name, start)

    def test_fine_grained_op_lstm_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            model = FineGrainedOpGridLSTMNet(TFGraphGridLSTM.NUM_LAYERS,
                                             TFGraphGridLSTM.SEQ_LEN,
                                             TFGraphGridLSTM.SEQ_LEN,
                                             TFGraphGridLSTM.BATCH_SIZE,
                                             TFGraphGridLSTM.HIDDEN)
            self._apply_forward(device,
                                f"graph_finegrained_op_lstm_{device}_forward",
                                model)

    def test_while_op_lstm_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            model = WhileOpGridLSTMNet(TFGraphGridLSTM.NUM_LAYERS,
                                       TFGraphGridLSTM.SEQ_LEN,
                                       TFGraphGridLSTM.SEQ_LEN,
                                       TFGraphGridLSTM.BATCH_SIZE,
                                       TFGraphGridLSTM.HIDDEN)
            self._apply_forward(device,
                                f"graph_while_op_lstm_{device}_forward", model)

    def test_base_while_op_lstm_forward(self):
        for device in [
                "cpu",
                "gpu",
        ]:
            model = BaseWhileOpGridLSTMNet(TFGraphGridLSTM.HIDDEN)
            self._apply_forward(device,
                                f"graph_base_while_op_lstm_{device}_forward",
                                model)


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=['first-arg-is-ignored'])
