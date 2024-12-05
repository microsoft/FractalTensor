import tensorflow as tf
import time

__all__ = [
    'MultiHeadAttention',
]


class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, d_model: int):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.xo = tf.keras.layers.Dense(d_model)

    def call(self, value: tf.Tensor, key: tf.Tensor, query: tf.Tensor):
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        query = tf.reshape(
            query, [query.shape[0], query.shape[1], self.num_heads, -1])
        key = tf.reshape(key, [key.shape[0], key.shape[1], self.num_heads, -1])
        value = tf.reshape(
            value, [value.shape[0], value.shape[1], self.num_heads, -1])

        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 3, 1])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        scores = tf.matmul(query, key)
        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, value)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, [out.shape[0], out.shape[1], -1])
        out = self.xo(out)
        return out
