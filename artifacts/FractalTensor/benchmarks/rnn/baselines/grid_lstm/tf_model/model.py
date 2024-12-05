from typing import NamedTuple, Tuple
import math

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import Tensor, TensorArray
from tensorflow.keras.layers import Dense, Layer, LSTMCell


class DimArg(NamedTuple):
    step: Tensor
    h: Tensor
    m: Tensor


class DimArrayArg(NamedTuple):
    step_array: TensorArray
    h_array: TensorArray
    m_array: TensorArray


class StepArg(NamedTuple):
    target_step: DimArg
    source_step: DimArg


class InnerLoopArg(NamedTuple):
    source: DimArg
    target_array: DimArrayArg


class OuterLoopArg(NamedTuple):
    source_array: DimArrayArg
    target_array: DimArrayArg


class VanillaRNNCell(Layer):
    def __init__(self, hidden_size, grid_dim=2):
        """
        Args:
            hidden_size(int): hidden dimension
            grid_dim(int): grid dimension
        """
        self.hidden_size = hidden_size
        self.grid_dim = grid_dim
        super(VanillaRNNCell, self).__init__()

    def build(self, _):
        stddev = 1.0 / math.sqrt(self.hidden_size)
        with tf.name_scope('weight'):
            self.W = tf.random.uniform(
                [self.hidden_size, self.hidden_size],
                minval=-stddev,
                maxval=stddev)
            self.U = tf.random.uniform(
                [self.hidden_size * self.grid_dim, self.hidden_size],
                minval=-stddev,
                maxval=stddev)
            self.b = tf.random.uniform(
                [1, self.hidden_size], minval=-stddev, maxval=stddev)

    def call(self, x_t: Tensor, y_t: Tensor,
             state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x_t(Tensor):
                the shape is (batch_size, hidden_size)
            y_t(Tensor):
                the shape is (batch_size, hidden_size)   
            state(Tensor):
                the shape is (batch_size, grid_dim * hidden_size)   
        Returns:
            (h_x, h_y): Tuple[Tensor, Tensor]
                h_x:
                    the shape is (batch_size, hidden_size)   
                h_y:
                    the shape is (batch_size, hidden_size)
        """
        temp = tf.matmul(state, self.U) + self.b

        h_x = tf.tanh(tf.matmul(x_t, self.W) + temp)
        h_y = tf.tanh(tf.matmul(y_t, self.W) + temp)
        return h_x, h_y


class FineGrainedOpGridLSTMNet(tf.keras.Model):
    def __init__(self, depth: int, src_len: int, trg_len: int, batch_size: int,
                 hidden_size: int):
        super(FineGrainedOpGridLSTMNet, self).__init__()

        self.depth = depth
        self.src_len = src_len
        self.trg_len = trg_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.cells = [VanillaRNNCell(hidden_size) for i in range(depth)]

        stddev = 1.0 / math.sqrt(self.hidden_size)
        with tf.name_scope('output'):
            self.h_output = tf.random.uniform(
                [self.depth, src_len, trg_len, 2, batch_size, hidden_size],
                minval=-stddev,
                maxval=stddev)

    def call(self, src_input_seq, trg_input_seq):

        # dim 1: stack Grid LSTM Cell to form depth.
        for d in range(0, self.depth, 1):
            # dim 2: iterate over source sequence length.
            for i in range(0, self.src_len, 1):
                # dim 3: iterate over target sequence length.
                for j in range(0, self.trg_len, 1):

                    # print("depth:", d, " src:", i, " trg:", j)
                    if d == 0:
                        x_t = src_input_seq[i]
                        y_t = trg_input_seq[j]
                    else:
                        x_t = self.h_output[d - 1][i][j][0]
                        y_t = self.h_output[d - 1][i][j][1]

                    if i == 0:
                        state_x = tf.zeros([self.batch_size, self.hidden_size])
                    else:
                        state_x = self.h_output[d][i - 1][j][0]

                    if j == 0:
                        state_y = tf.zeros([self.batch_size, self.hidden_size])
                    else:
                        state_y = self.h_output[d][i][j - 1][0]

                    state = tf.concat([state_x, state_y], 1)

                    h_x, h_y = self.cells[d](x_t, y_t, state)
                    temp = tf.stack([h_x, h_y], 0)
                    tf.tensor_scatter_nd_update(
                        self.h_output, [[d, i, j, 0], [d, i, j, 1]], temp)

        return self.h_output


class WhileOpGridLSTMNet(Layer):
    def __init__(self, depth: int, src_len: int, trg_len: int, batch_size: int,
                 hidden_size: int):
        super(WhileOpGridLSTMNet, self).__init__()
        self.depth = depth
        self.src_len = src_len
        self.trg_len = trg_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.cells = [VanillaRNNCell(hidden_size) for i in range(depth)]

        stddev = 1.0 / math.sqrt(self.hidden_size)
        with tf.name_scope('output'):
            self.h_output = tf.random.uniform(
                [self.depth, src_len, trg_len, 2, batch_size, hidden_size],
                minval=-stddev,
                maxval=stddev)

        self.cur_d = 0
        self.cur_i = 0
        self.cur_j = 0

    # dim 3: iterate over target sequence length.
    def inner_loop2(self):
        init_j = tf.constant(0)

        def cond(j: int):
            return tf.less(j, self.trg_len)

        def body(j: int) -> int:
            if self.cur_d == 0:
                x_t = self.source[self.cur_i]
                y_t = self.target[self.cur_j]
            else:
                x_t = self.h_output[self.cur_d - 1][self.cur_i][self.cur_j][0]
                y_t = self.h_output[self.cur_d - 1][self.cur_i][self.cur_j][1]

            if self.cur_i == 0:
                state_x = tf.zeros([self.batch_size, self.hidden_size])
            else:
                state_x = self.h_output[self.cur_d][self.cur_i -
                                                    1][self.cur_j][0]

            if self.cur_j == 0:
                state_y = tf.zeros([self.batch_size, self.hidden_size])
            else:
                state_y = self.h_output[self.cur_d][self.cur_i][self.cur_j -
                                                                1][0]

            state = tf.concat([state_x, state_y], 1)
            h_x, h_y = self.cells[self.cur_d](x_t, y_t, state)
            temp = tf.stack([h_x, h_y], 0)
            tf.tensor_scatter_nd_update(
                self.h_output, [[self.cur_d, self.cur_i, self.cur_j, 0],
                                [self.cur_d, self.cur_i, self.cur_j, 1]], temp)

            self.cur_j += 1
            return j + 1

        return tf.while_loop(cond, body, [init_j])

    # dim 2: iterate over source sequence length.
    def inner_loop1(self):
        init_i = tf.constant(0)

        def cond(i: int):
            return tf.less(i, self.src_len)

        def body(i: int) -> int:
            print("test")
            self.inner_loop2()
            self.cur_i += 1
            return i + 1

        return tf.while_loop(cond, body, [init_i])

    # dim 1: stack Grid LSTM Cell to form depth.
    def call(self, source: Tensor, target: Tensor) -> Tensor:
        init_d = tf.constant(0)

        self.source = source
        self.target = target

        def cond(d: int):
            return tf.less(d, self.depth)

        def body(d: int) -> int:
            self.inner_loop1()
            self.cur_d += 1
            return tf.add(d, 1)

        tf.while_loop(cond, body, [init_d])

        return self.h_output


class GridLSTMBlock(Layer):
    def __init__(self, hidden_size: int):
        super(GridLSTMBlock, self).__init__()
        self.hidden_size = hidden_size

        self.lstm_cell = LSTMCell(self.hidden_size)
        self.H2h = Dense(self.hidden_size)

    def call(self, arg: StepArg) -> StepArg:
        source_step = arg.source_step
        target_step = arg.target_step

        # shape: (batch_size, hidden_size)
        s_h = source_step.h
        # shape: (batch_size, hidden_size)
        t_h = target_step.h

        # shape: (batch_size, hidden_size)
        s_m = source_step.m
        # shape: (batch_size, hidden_size)
        t_m = target_step.m

        H = tf.concat([s_h, t_h], 1)
        h = self.H2h(H)

        # shape: (batch_size, hidden_size), (batch_size, hidden_size)
        _, [next_s_h, next_s_m] = self.lstm_cell(source_step.step, (h, s_m))
        # shape: (batch_size, hidden_size), (batch_size, hidden_size)
        _, [next_t_h, next_t_m] = self.lstm_cell(target_step.step, (h, t_m))

        return StepArg(
            DimArg(next_t_h, next_t_h, next_t_m),
            DimArg(next_s_h, next_s_h, next_s_m))


class GridLSTM(Layer):
    def __init__(self, hidden_size: int):
        super(GridLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.block = GridLSTMBlock(hidden_size)

    def inner_loop(self, input: InnerLoopArg) -> InnerLoopArg:
        init_i = tf.constant(0)

        target_seq_len = input.target_array.step_array.size()

        def cond(i: int, x):
            return tf.less(i, target_seq_len)

        def body(i: int, acc: InnerLoopArg) -> Tuple[int, InnerLoopArg]:
            step_array: TensorArray
            h_array: TensorArray
            m_array: TensorArray
            step_array, h_array, m_array = acc.target_array
            t_step: Tensor = step_array.read(i)
            t_h: Tensor = h_array.read(i)
            t_m: Tensor = m_array.read(i)

            t_step_arg: DimArg = DimArg(t_step, t_h, t_m)
            s_step_arg: DimArg = acc.source

            step_arg: StepArg = self.block(StepArg(t_step_arg, s_step_arg))
            source_step = step_arg.source_step
            target_step = step_arg.target_step
            return i + 1, InnerLoopArg(
                source_step,
                DimArrayArg(
                    step_array.write(i, target_step.step),
                    h_array.write(i, target_step.h),
                    m_array.write(i, target_step.m),
                ))

        return tf.while_loop(cond, body, (init_i, input))[1]

    def call(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        init_i = tf.constant(0)

        batch_size = tf.shape(source)[1]

        source_seq_len = tf.shape(source)[0]
        target_seq_len = tf.shape(target)[0]

        def cond(i: int, x):
            return tf.less(i, source_seq_len)

        def body(i: int, acc: OuterLoopArg) -> Tuple[int, OuterLoopArg]:
            source_array = acc.source_array
            target_array = acc.target_array

            step_array: TensorArray
            h_array: TensorArray
            m_array: TensorArray
            step_array, h_array, m_array = source_array

            s_step: Tensor = step_array.read(i)
            s_h: Tensor = h_array.read(i)
            s_m: Tensor = m_array.read(i)

            source_arg = DimArg(s_step, s_h, s_m)
            inner_loop_arg: InnerLoopArg = InnerLoopArg(
                source_arg, target_array)
            inner_loop_arg = self.inner_loop(inner_loop_arg)

            source = inner_loop_arg.source
            s_step, s_h, s_m = source

            step_array = step_array.write(i, s_step)
            h_array = h_array.write(i, s_h)
            m_array = m_array.write(i, s_m)

            source_array: DimArrayArg = DimArrayArg(step_array, h_array,
                                                    m_array)
            target_array: DimArrayArg = inner_loop_arg.target_array

            outer_loop_arg: OuterLoopArg = OuterLoopArg(
                source_array, target_array)
            return i + 1, outer_loop_arg

        s_step_array = tf.TensorArray(tf.float32, size=source_seq_len)
        s_h_array = tf.TensorArray(tf.float32, size=source_seq_len)
        s_m_array = tf.TensorArray(tf.float32, size=source_seq_len)

        t_step_array = tf.TensorArray(tf.float32, size=source_seq_len)
        t_h_array = tf.TensorArray(tf.float32, size=source_seq_len)
        t_m_array = tf.TensorArray(tf.float32, size=source_seq_len)

        init_outer_loop_arg: OuterLoopArg = OuterLoopArg(
            DimArrayArg(
                step_array=s_step_array.unstack(source),
                h_array=s_h_array.unstack(
                    tf.zeros([source_seq_len, batch_size, self.hidden_size])),
                m_array=s_m_array.unstack(
                    tf.zeros([source_seq_len, batch_size, self.hidden_size]))),
            DimArrayArg(
                step_array=t_step_array.unstack(target),
                h_array=t_h_array.unstack(
                    tf.zeros([target_seq_len, batch_size, self.hidden_size])),
                m_array=t_m_array.unstack(
                    tf.zeros([target_seq_len, batch_size, self.hidden_size]))))

        outer_loop_arg: OuterLoopArg = tf.while_loop(
            cond, body, (init_i, init_outer_loop_arg))[1]

        source: Tensor = outer_loop_arg.source_array.step_array.stack()
        target: Tensor = outer_loop_arg.target_array.step_array.stack()

        return source, target


class BaseWhileOpGridLSTMNet(Layer):
    def __init__(self, hidden_size: int):
        super(BaseWhileOpGridLSTMNet, self).__init__()

        # As stated in the Section 4.4, hierarchy grows along the third
        # dimension.
        # We stack two 2d GridLSTM to get 3d GridLSTM.
        self.gridLSTM_1 = GridLSTM(hidden_size)
        self.gridLSTM_2 = GridLSTM(hidden_size)
        self.gridLSTM_3 = GridLSTM(hidden_size)

    def call(self, source_input: Tensor, target_input: Tensor):

        source_output, target_output = self.gridLSTM_1(source_input,
                                                       target_input)

        source_output, target_output = self.gridLSTM_2(source_output,
                                                       target_output)

        source_output, target_output = self.gridLSTM_3(source_output,
                                                       target_output)

        return target_output
