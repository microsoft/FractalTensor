import os
import tensorflow as tf
from tf_model import tf_attn
import tensorflow as tf

from model_config import model_settings

from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    device = "cuda"

    batch_size = 1
    length = 512
    for setting in model_settings:
        _, num_heads, model_dim = setting

        query_len = length
        kv_len = length
        query_size = model_dim
        value_size = model_dim

        with tf.device('/GPU:0'):
            model = tf_attn.MutilHeadAttention(
                num_heads=num_heads, d_model=query_size)
            query = tf.random.uniform((batch_size, query_len, query_size))
            key = tf.random.uniform((batch_size, kv_len, query_size))
            value = tf.random.uniform((batch_size, kv_len, value_size))

            for _ in range(10):  # warmup
                out = model(query, key, value)

            start = time()
            for _ in range(100):
                model(query, key, value)
            elapsed_time = time() - start
            print('%.6f' % (elapsed_time / 100.))
