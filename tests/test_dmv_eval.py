import unittest

from launch_radar.dmv_eval import evaluate


class DmvEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluate()

    def test_keyword_safety_flags_are_mostly_wrong_on_report_volume(self) -> None:
        score = self.report.by_report.keyword_safety
        self.assertEqual(score.support, 71)
        self.assertGreater(score.precision, 0.2)
        self.assertLess(score.precision, 0.50)
        self.assertGreater(score.recall, 0.5)
        self.assertLess(score.recall, 0.95)

    def test_safety_recall_stays_below_the_gate_target(self) -> None:
        wording = self.report.by_formulation
        volume = self.report.by_report
        self.assertEqual(wording.keyword_safety.support, 38)
        self.assertLess(volume.keyword_safety.recall, 0.95)
        self.assertLess(volume.keyword_safety.precision, 0.50)
        self.assertLess(wording.model_safety.precision, 0.50)

    def test_keyword_has_no_cause_advantage_to_claim(self) -> None:
        wording = self.report.by_formulation
        self.assertAlmostEqual(wording.model_cause.accuracy, wording.majority_cause.accuracy, places=2)


if __name__ == "__main__":
    unittest.main()
