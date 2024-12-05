from __future__ import print_function

from context import *
import torch


class TestTensor(unittest.TestCase):
    def test1(self):
        """Test create Tensor from torch's tensor."""

        def _test_row_major(shape):
            for d in ['cpu', 'cuda']:
                t = Tensor(shape, kaleido.float32, device=d)

                numel = 1
                for s in t.shape:
                    numel *= s

                t.initialize(torch.arange, numel, dtype=torch.float32)
                t.data = t.data.view(shape)
                self.assertEqual(t.data.ndim, t.ndim)

                indices = [
                    random.randint(0, shape[i] - 1) for i in range(t.ndim)
                ]
                strides = t.strides

                offset = 0
                for i, s in zip(indices, strides):
                    offset += i * s

                self.assertEqual(t.data.reshape(-1)[offset], offset)

        for s in [
            [13],
            [3, 7],
            [7, 11, 3],
            [3, 7, 2, 11, 5],
        ]:
            _test_row_major(s)


class TestFractalTensor(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

    def test_FractalTensor_from_tensors1(self):
        """Test create FractalTensor from tensors."""

        device = ['cpu', 'cuda']
        for d in device:

            tensors = []
            for i in range(5):
                t = Tensor([3, 7, 5], kaleido.float32, device=d)
                init = torch.rand
                t.initialize(torch.rand, *t.shape, device=d)
                tensors.append(t)
            ta = FractalTensor.from_tensors(*tensors)

            for t in ta:
                for s1, s2 in zip(t.shape, t.data.shape):
                    self.assertEqual(s1, s2)

    def test_FractalTensor_nested(self):
        """Test create nested FractalTensor from FractalTensors."""

        device = ['cpu', 'cuda']

        for d in device:
            x_indices = []
            x1 = FractalTensor(
                TensorStorage((3, 4), kaleido.float32, device=d, order='row'))
            x_indices.append(list(range(7)))

            x2 = FractalTensor(
                TensorStorage((3, 4), kaleido.float32, device=d, order='row'))
            x_indices.append(list(range(11)))

            x3 = FractalTensor(
                TensorStorage((3, 4), kaleido.float32, device=d, order='row'))
            x_indices.append(list(range(2)))

            x = FractalTensor.from_fractaltensors(x1, x2, x3)

            y_indices = []
            y1 = FractalTensor(
                TensorStorage((3, 4), kaleido.float32, device=d, order='row'))
            y_indices.append(list(range(9)))
            y2 = FractalTensor(
                TensorStorage((3, 4), kaleido.float32, device=d, order='row'))
            y_indices.append(list(range(1)))
            y = FractalTensor.from_fractaltensors(y1, y2)

            z = FractalTensor.from_fractaltensors(x, y)
            z.indices = [x_indices, y_indices]
            self.assertEqual(z.depth, 3)

            z.initialize(torch.rand, *z.flatten_shape)

    def test_FractalTensor_pylist1(self):
        """Test create FractalTensor from Python built-in list if ints."""

        data = list(range(0, 13))
        x = FractalTensor.from_pylist(data)
        self.assertEqual(x.depth, 1)

        for i in range(len(x)):
            self.assertEqual(x[i].data, i)

    def test_FractalTensor_pylist2(self):
        """Test create FractalTensor from Pythoni's built-in nested list of ints."""

        MAX_N = 23

        data = []
        start = 0
        for i in range(random.randint(1, MAX_N)):
            j_out = []
            for j in range(random.randint(1, MAX_N)):
                end = start + random.randint(1, MAX_N)
                j_out.append(list(range(start, end, 1)))
                start = end
            data.append(j_out)
        x = FractalTensor.from_pylist(data)
        self.assertEqual(x.depth, 3)

        v = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                for k in range(len(x[i][j])):
                    self.assertEqual(x[i][j][k].data[0], v)
                    v += 1

    def test_FractalTensor_pylist3(self):
        """Test create FractalTensor from Pythoni's built-in nested list of ints."""

        data = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        x = FractalTensor.from_pylist(data)
        self.assertEqual(x.depth, 2)

        widths = x.width_by_depth(-1)
        self.assertListEqual(widths, [3])

        widths = x.width_by_depth(-2)
        self.assertListEqual(widths, [3, 2, 4])

        with self.assertRaises(ValueError):
            depths = x.width_by_depth(-3)

        data = [[[0, 1], [2, 3, 4], [5]], [[6, 7, 8, 9]], [[10]]]
        x = FractalTensor.from_pylist(data)
        self.assertEqual(x.depth, 3)

        widths = x.width_by_depth(-1)
        self.assertListEqual(widths, [3])

        widths = x.width_by_depth(-2)
        self.assertListEqual(widths, [3, 1, 1])

        widths = x.width_by_depth(-3)
        self.assertListEqual(widths, [2, 3, 1, 4, 1])

        with self.assertRaises(ValueError):
            depths = x.width_by_depth(-4)


if __name__ == '__main__':
    unittest.main()
