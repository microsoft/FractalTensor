from __future__ import print_function

from context import *


class TestConstantsOp(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

    def test_arange(self):
        x1 = kaleido.operations.slices(kaleido.operations.arange(11), dim=0)
        self.assertTrue(isinstance(x1, kaleido.FractalTensor))
        self.assertTrue(
            isinstance(x1.element_type.element_type,
                       kaleido.frontend.types.Int))
        y = list(range(11))
        for i, x in enumerate(x1):
            self.assertEqual(y[i], x.data.item())

        x2 = kaleido.operations.slices(kaleido.operations.arange(5, 7), dim=0)
        y = list(range(5, 7))
        for i, x in enumerate(x2):
            self.assertEqual(y[i], x.data.item())

        x3 = kaleido.operations.slices(
            kaleido.operations.arange(5, 24, 3), dim=0)
        y = list(range(5, 24, 3))
        for i, x in enumerate(x3):
            self.assertEqual(y[i], x.data.item())

    def test_constants(self):
        for d in ['cpu', 'cuda']:
            x = kaleido.operations.zeros(shape=(3, 7), dtype='float', device=d)
            y = kaleido.operations.ones(shape=(3, 7), dtype='float', device=d)


if __name__ == '__main__':
    unittest.main()
