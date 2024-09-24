import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
import unittest
from time import time
import tensorflow as tf

from tf_model import StackedDRNN
from utils import *

tf.compat.v1.disable_eager_execution()


class TFGraphDRNN(unittest.TestCase):
    def setUp(self):
        self.shape = (SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        self.stddev = 1.0 / math.sqrt(HIDDEN_SIZE)

        self.log_dir = ''
        self.logger = init_logger(self.log_dir, 'tensorflow_drnn_graph.txt')

    def test_drnn_forward(self):
        shape = (SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        rate = DILATION[-1]
        pad_shape = ((rate - (SEQ_LEN % rate)) % rate, BATCH_SIZE, INPUT_SIZE)

        stddev = 1.0 / math.sqrt(HIDDEN_SIZE)

        with tf.compat.v1.Session() as sess:
            for device in [
                    'cpu',
                    '/device:GPU:0',
            ]:
                with tf.device(device):
                    model = StackedDRNN(
                        batch_size=BATCH_SIZE,
                        seq_len=SEQ_LEN,
                        input_size=INPUT_SIZE,
                        hidden_size=HIDDEN_SIZE,
                        dilation=DILATION)

                    inputs = tf.compat.v1.placeholder(tf.float32, shape=shape)
                    pads = tf.compat.v1.placeholder(
                        tf.float32, shape=pad_shape)
                    res = model(inputs, pads)

                    sess.run(tf.compat.v1.global_variables_initializer())

                    gen_x = tf.random.uniform(
                        shape, minval=-stddev, maxval=stddev)
                    gen_padding = tf.zeros(pad_shape, dtype=tf.dtypes.float32)

                    x_data = sess.run(gen_x)
                    padding_data = sess.run(gen_padding)

                    for i in range(WARMUP):
                        output = sess.run(
                            res,
                            feed_dict={
                                inputs: x_data,
                                pads: padding_data
                            })

                    start = time()
                    for i in range(ITERS):
                        sess.run(
                            res,
                            feed_dict={
                                inputs: x_data,
                                pads: padding_data
                            })
                    test_name = f'TensorFlow_Stacked_DLSTM_graph_{device}'
                    report(test_name, start, self.logger)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])
