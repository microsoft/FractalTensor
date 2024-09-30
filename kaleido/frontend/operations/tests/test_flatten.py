from __future__ import print_function

from context import *
import torch

from kaleido import FractalTensor


class TestFlatten(unittest.TestCase):
    def create_data(self):
        shape = [3, 4]
        dtype = kaleido.TensorStorage(
            shape, kaleido.float32, device='cpu', order='row')

        count = 0

        xs = []
        x_indices = []
        for i in range(5):
            xs.append(FractalTensor(dtype))
            n = random.randint(13, 27)
            count += n
            x_indices.append(list(range(n)))

        x = FractalTensor.from_fractaltensors(*xs)
        x.indices = x_indices

        x.initialize(torch.rand, *x.flatten_shape)
        return x, count

    def test1(self):
        x, count = self.create_data()
        dim = 0
        y = kaleido.operations.flatten(x, dim)
        self.assertTrue(isinstance(y, kaleido.Tensor))

        new_shape = x.element_type.shape
        new_shape[dim] = new_shape[dim] * count
        for s1, s2 in zip(new_shape, y.shape):
            self.assertEqual(s1, s2)

    def test2(self):
        x, count = self.create_data()
        dim = 1
        y = kaleido.operations.flatten(x, dim)
        self.assertTrue(isinstance(y, kaleido.Tensor))

        new_shape = x.element_type.shape
        new_shape[dim] = new_shape[dim] * count
        for s1, s2 in zip(new_shape, y.shape):
            self.assertEqual(s1, s2)


if __name__ == '__main__':
    unittest.main()
