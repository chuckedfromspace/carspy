import unittest
from carspy import LineStrength


class TestLineStrength(unittest.TestCase):

    def test_init(self):
        self.assertEqual(LineStrength().Const_D, 7.162e-07)

        with self.assertRaises(ValueError):
            LineStrength('O2')


if __name__ == '__main__':
    unittest.main()
