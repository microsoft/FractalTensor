from __future__ import print_function

from context import *


class TestProduct(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

        data1 = list(range(11))
        self.x = kaleido.FractalTensor.from_pylist(data1)

        data2 = list(range(7))
        self.y = kaleido.FractalTensor.from_pylist(data2)

    def test_product(self):
        xss, yss = kaleido.operations.product(self.x, self.y)
        self.assertTrue(isinstance(xss, kaleido.FractalTensor))
        self.assertTrue(isinstance(yss, kaleido.FractalTensor))

        for i, (xs, ys) in enumerate(kaleido.operations.zip(xss, yss)):
            for j, (x, y) in enumerate(kaleido.operations.zip(xs, ys)):
                self.assertEqual(x.data.item(), j)
                self.assertEqual(y.data.item(), i)


if __name__ == '__main__':
    unittest.main()
