from typing import List
import tensorflow as tf

__all__ = [
    'StackedDRNN',
]


class StackedDRNN(tf.keras.Model):
    def __init__(self, batch_size: int, seq_len: int, input_size: int,
                 hidden_size: int, dilation: List[int]):
        super(StackedDRNN, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dilation = dilation
        self.num_layers = len(dilation)

        rate = dilation[-1]
        self.padded_length = (rate - (seq_len % rate)) % rate + self.seq_len

        self.cells = []
        for i in range(self.num_layers):
            self.cells.append(
                tf.compat.v1.keras.layers.CuDNNLSTM(
                    hidden_size, return_sequences=False))

    # uncomment the following line to enable auto-graph.
    # @tf.function
    def call(self, input, padding_data):
        # step 0: pad the input
        input_x = tf.concat((input, padding_data), axis=0)

        # no special treatment for the first layer.
        xs = self.cells[0](input_x)

        for i, cell in enumerate(self.cells[1:]):
            # for layers above the frist layer.
            # step 1: pre-process: form a new batch
            num_split = self.padded_length // self.dilation[i + 1]

            xs_ = [
                tf.reshape(x, (-1, self.hidden_size))
                for x in tf.split(xs, num_or_size_splits=num_split, axis=0)
            ]
            dilated_input = tf.stack(xs_)

            # step 2: call LSTM layer
            xs = cell(dilated_input)

            # step 3: post-processing, revert to the original layout
            xss = [
                tf.split(x, self.dilation[i + 1], axis=0)
                for x in tf.unstack(xs, axis=0)
            ]

            xs = tf.stack([x for sublist in xss for x in sublist])
        return xs
