import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
import unittest
from time import time
import tensorflow as tf

from tf_model import StackedDRNN
from utils import *


class TFGraphDRNN(unittest.TestCase):
    def setUp(self):
        self.shape = (SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        self.stddev = 1.0 / math.sqrt(HIDDEN_SIZE)

        self.log_dir = ''
        self.logger = init_logger(self.log_dir, 'tensorflow_drnn.txt')

    def _apply_forward(self, test_name, model, *inputs):
        for i in range(WARMUP):
            output = model(*inputs)

        start = time()

        for i in range(ITERS):
            output = model(*inputs)
        report(test_name, start, self.logger)

    def test_drnn_forward(self):
        shape = (SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        stddev = 1.0 / math.sqrt(HIDDEN_SIZE)

        gpus = tf.config.list_physical_devices('GPU')
        for device in [
                # 'cpu',
                '/device:GPU:0',
        ]:
            with tf.device(device):
                model = StackedDRNN(
                    batch_size=BATCH_SIZE,
                    seq_len=SEQ_LEN,
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    dilation=DILATION)

                x = tf.random.uniform(shape, minval=-stddev, maxval=stddev)
                rate = DILATION[-1]
                padding_data = tf.zeros(
                    ((rate - (SEQ_LEN % rate)) % rate, BATCH_SIZE, INPUT_SIZE),
                    dtype=tf.dtypes.float32)
                test_name = f'TensorFlow_Stacked_DLSTM_{device}'
                self._apply_forward(test_name, model, x, padding_data)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])
