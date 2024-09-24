from __future__ import print_function

from context import *

import torch


class TestJoin(unittest.TestCase):
    def create_depth1_fractaltensor(self, length, device='cpu'):
        shape = [3, 7]
        xs = kaleido.FractalTensor(
            kaleido.TensorStorage(shape, kaleido.float32, device=device))
        xs.indices = list(range(length))
        xs.initialize(torch.rand, *xs.flatten_shape, device=device)
        return xs

    def create_depth2_fractaltensor(self, length, device='cpu'):
        shape = [3, 7]
        xss = kaleido.FractalTensor(
            kaleido.FractalTensorStorage(
                kaleido.TensorStorage(shape, kaleido.float32, device=device)))
        xss.indices = [
            list(range(random.randint(5, 17))) for _ in range(length)
        ]
        xss.initialize(torch.rand, *xss.flatten_shape, device=device)
        return xss

    def setUp(self):
        random.seed(12345)

    def test_join1(self):
        xs = self.create_depth1_fractaltensor(19)
        ys = self.create_depth1_fractaltensor(3)
        zs = kaleido.operations.join(xs, ys)

        self.assertTrue(isinstance(zs, kaleido.FractalTensor))
        self.assertEqual(zs.depth, xs.depth)
        self.assertEqual(len(xs) + len(ys), len(zs))
        self.assertEqual(xs.numel + ys.numel, zs.numel)

    def test_join2(self):
        xss = self.create_depth2_fractaltensor(11)
        yss = self.create_depth2_fractaltensor(7)
        zss = kaleido.operations.join(xss, yss)

        self.assertTrue(isinstance(zss, kaleido.FractalTensor))
        self.assertEqual(zss.depth, xss.depth)
        self.assertEqual(len(xss) + len(yss), len(zss))
        self.assertEqual(xss.numel + yss.numel, zss.numel)


if __name__ == '__main__':
    unittest.main()
