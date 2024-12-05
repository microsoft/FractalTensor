from __future__ import print_function

from context import *

from kaleido.frontend.types import Real, Int, Bool

device = 'cpu'


class Test1(unittest.TestCase):
    def test_basic_type(self):
        self.assertFalse(Real(64).is_equal_type(Int(32)))
        self.assertFalse(Real(64).is_equal_type(Real(32)))
        self.assertFalse(Bool().is_equal_type(Real(16)))

        self.assertTrue(Real(64).is_equal_type(Real(64)))
        self.assertTrue(Int(16).is_equal_type(Int(16)))

    def test_tensor_type(self):
        x = TensorStorage((2, 64), kaleido.float32, device=device)

        self.assertTrue(x.is_equal_type(x))
        self.assertFalse(
            x.is_equal_type(TensorStorage((1, 3), kaleido.float32, device)))

        y = FractalTensorStorage(
            TensorStorage((2, 64), kaleido.float32, device=device))
        self.assertTrue(y.is_equal_type(y))
        self.assertTrue(
            y.is_equal_type(
                FractalTensorStorage(
                    TensorStorage((2, 64), kaleido.float32, device=device))))

        self.assertFalse(
            y.is_equal_type(
                FractalTensorStorage(
                    TensorStorage((4, 7), kaleido.float32, device=device))))


if __name__ == '__main__':
    unittest.main()
