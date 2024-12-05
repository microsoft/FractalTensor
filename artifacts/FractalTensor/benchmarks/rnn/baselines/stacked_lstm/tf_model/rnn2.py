from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
import math
import tensorflow as tf

layers = tf.keras.layers


class FineGrainedOpLstmCellV1(layers.Layer):
    def __init__(self, input_size, hidden_size):
        super(FineGrainedOpLstmCellV1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        stddev = 1.0 / math.sqrt(self.hidden_size)
        self.igx = tf.Variable(
            tf.random.uniform(
                [self.input_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.igu = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.ib = tf.Variable(
            tf.random.uniform(
                [self.hidden_size], minval=-stddev, maxval=stddev))

        self.fgx = tf.Variable(
            tf.random.uniform(
                [self.input_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.fgu = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.fb = tf.Variable(
            tf.random.uniform(
                [self.hidden_size], minval=-stddev, maxval=stddev))

        self.ogx = tf.Variable(
            tf.random.uniform(
                [self.input_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.ogu = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.ob = tf.Variable(
            tf.random.uniform(
                [self.hidden_size], minval=-stddev, maxval=stddev))

        self.cgx = tf.Variable(
            tf.random.uniform(
                [self.input_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.cgu = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev))
        self.cb = tf.Variable(
            tf.random.uniform(
                [self.hidden_size], minval=-stddev, maxval=stddev))

    # uncomment the following line to enable auto-graph.
    @tf.function
    def call(self, x, h_prev, c_prev):
        ig = tf.sigmoid(x @ self.igx + h_prev @ self.igu + self.ib)
        fg = tf.sigmoid(x @ self.fgx + h_prev @ self.fgu + self.fb)
        og = tf.sigmoid(x @ self.ogx + h_prev @ self.ogu + self.ob)
        c_candidate = tf.tanh(x @ self.cgx + h_prev @ self.cgu + self.cb)

        c = fg * c_prev + ig * c_candidate
        h = og * tf.tanh(c)
        return h, c


class FineGrainedOpLstmCellV2(layers.Layer):
    def __init__(self, input_size, hidden_size):
        super(FineGrainedOpLstmCellV2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        stddev = 1.0 / math.sqrt(self.hidden_size)
        self.w = tf.Variable(
            tf.random.uniform(
                [self.input_size, self.hidden_size * 4],
                minval=-stddev,
                maxval=stddev))
        self.u = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.hidden_size * 4],
                minval=-stddev,
                maxval=stddev))
        self.b = tf.Variable(
            tf.random.uniform(
                [self.hidden_size * 4], minval=-stddev, maxval=stddev))

    # uncomment the following line to enable auto-graph.
    @tf.function
    def call(self, x, h_prev, c_prev):
        g = x @ self.w + h_prev @ self.u + self.b
        g_act = tf.sigmoid(g[:, :self.hidden_size * 3])
        c_candidate = tf.tanh(g[:, self.hidden_size * 3:])

        ig, fg, og = (
            g_act[:, :self.hidden_size],  # input
            g_act[:, self.hidden_size:self.hidden_size * 2],  # forget
            g_act[:, self.hidden_size * 2:],  # output
        )

        c = fg * c_prev + ig * c_candidate
        h = og * tf.tanh(c)
        return h, c


class FineGrainedOpLstmNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers, cell_type):
        super(FineGrainedOpLstmNet, self).__init__()
        self.hidden_size = hidden_size

        if cell_type == 'v1':
            self.cells = [
                FineGrainedOpLstmCellV1(input_size if i == 0 else hidden_size,
                                        hidden_size) for i in range(num_layers)
            ]
        elif cell_type == 'v2':
            self.cells = [
                FineGrainedOpLstmCellV2(input_size if i == 0 else hidden_size,
                                        hidden_size) for i in range(num_layers)
            ]
        else:
            raise ValueError('Unknow cell type.')

    # uncomment the following line to enable auto-graph.
    @tf.function
    def call(self, input_seq):
        batch_size = int(input_seq.shape[1])

        for rnncell in self.cells:  # iterate over depth
            outputs = []
            input_seq = tf.unstack(
                input_seq, num=int(input_seq.shape[0]), axis=0)
            h = tf.zeros((batch_size, self.hidden_size))
            c = tf.zeros((batch_size, self.hidden_size))
            for inp in input_seq:  # iterate over time step
                h, c = rnncell(inp, h, c)
                outputs.append(h)

            input_seq = tf.stack(outputs, axis=0)

        return [input_seq]


class WhileOpLstmLayer(tf.keras.Model):
    """Lstm implemented in fine-grained operators via symbolic while-ops.
    Only works in graph-mode.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: int,
            hidden_size: int,
        Return:
            A Tensor with a shape [batch_size, sequence_length, hidden_dim]
        """
        super(WhileOpLstmLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        stddev = 1.0 / math.sqrt(self.hidden_size)

        self.w = tf.Variable(
            tf.random.uniform(
                [self.input_size, self.hidden_size * 4],
                minval=-stddev,
                maxval=stddev))

        self.u = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.hidden_size * 4],
                minval=-stddev,
                maxval=stddev))
        self.bias = tf.Variable(
            tf.random.uniform(
                [self.hidden_size * 4], minval=-stddev, maxval=stddev))

    def _while_op_lstm(self, input):
        shape = tf.shape(input)
        seq_len = shape[0]
        batch_size = shape[1]

        def body(t, step):
            """The Lstm cell.
            For some TF implementation constrains, we cannot reuse LstmCell
            defined in utils.py, but implement in the body function.
            """
            h_prev, c_prev, output_array = step

            x_t = input[t, :]
            g = x_t @ self.w + h_prev @ self.u + self.bias
            g_act = tf.sigmoid(g[:, :self.hidden_size * 3])
            c_candidate = tf.tanh(g[:, self.hidden_size * 3:])

            ig, fg, og = (
                g_act[:, :self.hidden_size],  # input
                g_act[:, self.hidden_size:self.hidden_size * 2],  # forget
                g_act[:, self.hidden_size * 2:],  # output
            )

            c = fg * c_prev + ig * c_candidate
            h = og * tf.tanh(c)

            return t + 1, (h, c, output_array.write(t, h))

        init_h = tf.zeros([batch_size, self.hidden_size])
        init_c = tf.zeros([batch_size, self.hidden_size])

        init_t = tf.constant(0)
        output_array = tf.TensorArray(
            dtype=tf.float32, size=seq_len, dynamic_size=False)
        cond = lambda i, _: tf.less(i, seq_len)
        _, step = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(init_t, (init_h, init_c, output_array)))
        _, _, output_array = step

        return output_array.stack()

    def __call__(self, input_seq):
        """Stacked Lstm network implemented by TF's symbolic while loop operator.
        Args:
            input_seq, Tensor, input sequence batch. The layout must be
                batch_size major: [seq_len, batch_size, input_dim].
        """
        return self._while_op_lstm(input_seq)


class WhileOpLstmNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers):
        super(WhileOpLstmNet, self).__init__()
        self.hidden_size = hidden_size

        self.rnns = [
            WhileOpLstmLayer(input_size
                             if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def call(self, input_seq):
        outputs = []
        xs = input_seq
        for rnn in self.rnns:  # iterate over depth
            xs = rnn(xs)
            outputs.append(xs)
        return outputs


class StaticRNN(tf.keras.Model):
    """A static RNN.
    """

    def __init__(self, hidden_size, num_layers, use_cudnn_rnn=True):
        """
        hidden_size: Int, hidden dimension of the RNN unit.
        num_layers: Int, the number of stacked RNN unit, namely depth of the RNN
            network.
        """
        super(StaticRNN, self).__init__()

        if use_cudnn_rnn:
            self.cells = [
                tf.compat.v1.keras.layers.CuDNNLSTM(
                    hidden_size, return_state=True, return_sequences=True)
                for _ in range(num_layers)
            ]
        else:
            # About layers.LstmCell's `implementation` argument, either 1 or 2.
            # Mode 1 will structure its operations as a larger number of smaller
            # dot products and additions, whereas mode 2 will batch them into
            # fewer, larger operations. These modes will have different
            # performance profiles on different hardware and for different
            # applications.
            self.cells = [
                layers.LSTMCell(
                    units=hidden_size,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    unit_forget_bias=True,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    implementation=2) for _ in range(num_layers)
            ]

        self.hidden_size = hidden_size
        self.use_cudnn_rnn = use_cudnn_rnn

    def _cudnn_lstm_call(self, input_seq):
        # A workaround to stack CuDNNLstm in TF 2.0.
        # https://stackoverflow.com/questions/55324307/how-to-implement-a-stacked-rnns-in-tensorflow
        x = input_seq
        for rnn in self.cells:
            x = rnn(x)
        return x

    # uncomment the following line to enable auto-graph.
    @tf.function
    def call(self, input_seq):
        """Define computations in a single time step.

        input_seq: Tensor, the layout is
            [batch_size, max_sequence_length, embedding_dim].
        """
        if self.use_cudnn_rnn:
            return self._cudnn_lstm_call(input_seq)

        batch_size = int(input_seq.shape[1])

        hiddens = []
        for cell in self.cells:  # iterate over depth
            state = (tf.zeros((batch_size, self.hidden_size)),
                     tf.zeros((batch_size, self.hidden_size)))

            # unpack the input 3D tensors along the `max_sequence_length` axis
            # to get input tensors for each time step.
            input_seq = tf.unstack(
                input_seq, num=int(input_seq.shape[0]), axis=0)
            outputs = []
            for inp in input_seq:  # iterate over time step
                output, state = cell(inp, state)
                outputs.append(output)

            input_seq = tf.stack(outputs, axis=0)
            hiddens.append(input_seq)

        return hiddens
