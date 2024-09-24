from __future__ import print_function

from context import *

import operator
import itertools
import torch


class TestScan(unittest.TestCase):
    MAX = 193
    N = 17

    def setUp(self):
        random.seed(12345)

        self.data = [
            random.randint(0, TestScan.MAX) for _ in range(TestScan.N)
        ]
        self.xs = kaleido.FractalTensor.from_pylist(self.data)

    def test1(self):
        """Test single-level scan."""

        expected_results = list(itertools.accumulate(self.data, operator.add))

        ys = kaleido.operations.scan(lambda s, x: kaleido.operations.add(s, x),
                                     self.xs)
        self.assertTrue(isinstance(ys, kaleido.FractalTensor))
        self.assertEqual(len(ys), len(self.xs))

        init = kaleido.Tensor((1, ), kaleido.int32)
        init.data = torch.LongTensor([5])
        ys = kaleido.operations.scan(lambda s, x: kaleido.operations.add(s, x),
                                     self.xs, init)
        self.assertTrue(isinstance(ys, kaleido.FractalTensor))
        self.assertEqual(len(ys), len(self.xs))

        expected_results = list(
            itertools.accumulate([5] + self.data, operator.add))
        for i, y in enumerate(ys):
            self.assertEqual(y.data.item(), expected_results[i + 1])

    def test2(self):
        init = kaleido.Tensor((1, ), kaleido.int32)
        init.data = torch.LongTensor([5])

        ys, zs = kaleido.operations.scan(lambda s, x: (x, x), self.xs, init)

        for x, y, z in kaleido.operations.zip(self.xs, ys, zs):
            self.assertEqual(y.data.item(), x.data.item())
            self.assertEqual(y.data.item(), x.data.item())


if __name__ == '__main__':
    unittest.main()
