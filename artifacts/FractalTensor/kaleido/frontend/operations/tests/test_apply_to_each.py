from __future__ import print_function

from context import *


class TestMap(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

        data = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        self.xss = kaleido.FractalTensor.from_pylist(data)
        self.assertEqual(self.xss.depth, 2)

    def test1(self):
        """Test raising Error if invalid input type is given."""

        with self.assertRaises(TypeError):
            kaleido.operations.map(lambda x: x, [1, 2, 3])

        with self.assertRaises(TypeError):
            kaleido.operations.fold(lambda x: x, [1, 2, 3])

        with self.assertRaises(TypeError):
            kaleido.operations.scan(lambda x: x, [1, 2, 3])

    def test2(self):
        """Test nested map."""

        def f1(xs):
            return kaleido.operations.map(lambda x: x, xs)

        yss1 = kaleido.operations.map(lambda xs: f1(xs), self.xss)

        for i, d1 in enumerate(self.xss):
            for j, x in enumerate(d1):
                self.assertEqual(x.data, yss1[i][j].data)

    def test3(self):
        """Test lambda function returns more than one results."""

        def f2(xs):
            return kaleido.operations.map(lambda x: (x, x), xs)

        yss2, yss3 = kaleido.operations.map(lambda xs: f2(xs), self.xss)

        for i, d1 in enumerate(self.xss):
            for j, x in enumerate(d1):
                self.assertEqual(x.data, yss2[i][j].data)
                self.assertEqual(x.data, yss3[i][j].data)

    def test4(self):
        """Test map with nested Iterative as its input."""

        data = list(range(7))
        xs1 = kaleido.FractalTensor.from_pylist(data)
        self.assertEqual(xs1.depth, 1)

        data = list(range(7, 14, 1))
        xs2 = kaleido.FractalTensor.from_pylist(data)
        self.assertEqual(xs2.depth, 1)

        xs = kaleido.operations.zip(xs1, xs2)

        def f(x, y):
            return x, y

        ys, zs = kaleido.operations.map(lambda x: f(*x), xs)

        for y, z, x1, x2 in kaleido.operations.zip(ys, zs, xs1, xs2):
            self.assertTrue(y.data.eq(x1.data).all().tolist())

            self.assertTrue(z.data.eq(x2.data).all().tolist())

    def test5(self):
        def f1(xs) -> kaleido.Iterative:
            v1, v2 = kaleido.operations.map(lambda x: (x, x), xs)
            return kaleido.operations.zip(v1, v2)

        yss, zss = kaleido.operations.map(lambda xs: f1(xs), self.xss)

        for i, ys in enumerate(yss):
            for j, y in enumerate(ys):
                self.assertEqual(yss[i][j].data.item(),
                                 self.xss[i][j].data.item())
                self.assertEqual(zss[i][j].data.item(),
                                 self.xss[i][j].data.item())


if __name__ == '__main__':
    unittest.main()
