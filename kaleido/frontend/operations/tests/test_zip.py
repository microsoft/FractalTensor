from __future__ import print_function

from context import *


class TestZip(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

        data = list(range(7))
        self.xs1 = kaleido.FractalTensor.from_pylist(data)

        data = list(range(7, 14, 1))
        self.xs2 = kaleido.FractalTensor.from_pylist(data)

    def test_zipped_ta(self):
        zipped = kaleido.operations.zip(self.xs1, self.xs2)
        self.assertTrue(isinstance(zipped, kaleido.Iterative))

        for x1, x2 in zipped:
            self.assertEqual(x2.data - x1.data, 7)

    def test_nested_zip(self):
        zipped = kaleido.operations.zip(
            kaleido.operations.zip(self.xs1, self.xs2), self.xs1)
        self.assertTrue(isinstance(zipped, kaleido.Iterative))

        for xz, z in zipped:
            x, y = xz
            self.assertEqual(y.data - x.data, 7)
            self.assertEqual(y.data - z.data, 7)


if __name__ == '__main__':
    unittest.main()
