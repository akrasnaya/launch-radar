import unittest

from launch_radar.calibration import best_f1_point, reaches_precision


class CalibrationTest(unittest.TestCase):
    def test_safety_precision_target_is_not_met_at_60_percent_coverage(self) -> None:
        self.assertFalse(reaches_precision(0.90, 0.60))

    def test_best_safety_point_stays_below_the_precision_target(self) -> None:
        point = best_f1_point()
        self.assertLess(point.precision, 0.50)
        self.assertGreater(point.recall, 0.70)
        self.assertEqual(point.autoclear_missed_safety, 0)
        self.assertLess(point.autoclear_coverage, 0.10)


if __name__ == "__main__":
    unittest.main()
