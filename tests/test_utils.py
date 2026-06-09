import unittest

from rosmaster_a1_e2e_vision.utils import SteeringFilter


class TestSteeringFilter(unittest.TestCase):
    def test_clamp_and_rate_limit(self):
        filt = SteeringFilter(max_abs=0.4, low_pass_alpha=1.0, max_rate_per_sec=0.5)
        first = filt.apply(1.0, 0.1)
        second = filt.apply(-1.0, 0.1)

        self.assertAlmostEqual(first, 0.4)
        self.assertAlmostEqual(second, 0.35)


if __name__ == "__main__":
    unittest.main()
