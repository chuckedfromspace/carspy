import unittest
from carspy.utils import downsample, comp_normalize


class TestUtils(unittest.TestCase):

    def test_normalize(self):
        _dict = {'N2': 0.6, 'O2': 0.3}
        _corr = comp_normalize(_dict)
        self.assertEqual(sum(list(_corr.values())), 1.0)


if __name__ == '__main__':
    unittest.main()
