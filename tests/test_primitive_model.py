import unittest

from launch_radar.pipeline import run
from launch_radar.primitive_model import evaluate
from launch_radar.sample_data import sample_records


class PrimitiveModelTest(unittest.TestCase):
    def test_keyword_classifier_is_the_baseline(self) -> None:
        report = evaluate(sample_records())
        self.assertAlmostEqual(report.keyword_mean, 0.92, places=2)
        self.assertGreater(report.model_mean, report.keyword_mean * 0.75)

    def test_release_path_still_uses_keywords(self) -> None:
        records = sample_records()
        classifications, _, cards = run(records)
        self.assertTrue(classifications["fb-011"].safety_flag)
        self.assertTrue(any(card.recommendation == "HOLD" for card in cards))


if __name__ == "__main__":
    unittest.main()
