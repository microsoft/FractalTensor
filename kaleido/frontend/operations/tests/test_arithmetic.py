from __future__ import print_function

from context import *

import torch


# FIXME(ying): All the tests are not meaningful in checking the correctness,
# but only to make sure the codes are able to run.
class TestConstantsOp(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

        shape = [3, 11, 7]
        device = 'cpu'

        self.x = kaleido.Tensor(shape, kaleido.float32, device=device)
        self.x.initialize(
            torch.rand,
            *self.x.shape,
            dtype=torch.float32,
            device=self.x.device)

        self.y = kaleido.Tensor(shape, kaleido.float32, device=device)
        self.y.initialize(
            torch.rand,
            *self.y.shape,
            dtype=torch.float32,
            device=self.y.device)

    def test_elementwise(self):
        self.x + self.y
        self.x - self.y
        self.x * self.y
        self.x / self.y
        kaleido.operations.exp(self.x)
        kaleido.operations.log(self.x)
        kaleido.operations.pow(self.x, 2)
        kaleido.operations.sin(self.x)
        kaleido.operations.cos(self.x)
        kaleido.operations.tanh(self.x)
        kaleido.operations.sqrt(self.x)
        kaleido.operations.abs(self.x)
        kaleido.operations.sigmoid(self.x)
        kaleido.operations.dropout(self.x, drop_rate=0.1)
        kaleido.operations.relu(self.x)

    def test_reduction(self):
        kaleido.operations.softmax(self.x)
        kaleido.operations.max(self.x)
        kaleido.operations.sum(self.x)
        kaleido.operations.mean(self.x)

        x = kaleido.Tensor([17, 1], kaleido.float32, device='cpu')
        x.initialize(
            torch.rand, *x.shape, dtype=torch.float32, device=x.device)

        y = kaleido.Tensor([17, 1], kaleido.float32, device='cpu')
        y.initialize(
            torch.rand, *y.shape, dtype=torch.float32, device=x.device)
        kaleido.operations.dot(x, y)

        kaleido.operations.dot(self.x, self.y)

    def test_contraction(self):
        x = kaleido.Tensor([3, 7], kaleido.float32, device='cpu')
        x.initialize(
            torch.rand, *x.shape, dtype=torch.float32, device=x.device)

        y = kaleido.Tensor([7, 5], kaleido.float32, device='cpu')
        y.initialize(
            torch.rand, *y.shape, dtype=torch.float32, device=x.device)

        z = x @ y


if __name__ == '__main__':
    unittest.main()
