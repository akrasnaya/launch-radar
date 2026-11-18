import unittest

from launch_radar.baseline import classify
from launch_radar.models import FeedbackRecord
from launch_radar.sample_data import sample_records


def _record(text: str) -> FeedbackRecord:
    return FeedbackRecord("x", text, "test", "pilot", "1", True)


class BaselineBoundaryTest(unittest.TestCase):
    def test_negated_takeover_is_not_safety(self) -> None:
        item = classify(_record("One lane drift on the previous build, no takeover needed."))
        self.assertFalse(item.safety_flag)
        self.assertNotIn("perception_safety", item.topics)

    def test_took_over_counts_as_takeover(self) -> None:
        item = classify(_record("Localization drifted near the barrier and the driver took over."))
        self.assertTrue(item.safety_flag)
        self.assertEqual(item.severity, "critical")

    def test_precautionary_takeover_is_not_safety(self) -> None:
        item = classify(_record("Precautionary Takeover: Disengaged due to a software issue."))
        self.assertFalse(item.safety_flag)

    def test_disengagement_is_not_a_navigation_term(self) -> None:
        item = classify(_record("The ADS initiated a disengagement and the driver took control."))
        self.assertNotIn("navigation", item.topics)
        self.assertFalse(item.safety_flag)

    def test_appeared_does_not_create_a_ux_topic(self) -> None:
        by_id = {record.record_id: record for record in sample_records()}
        item = classify(by_id["fb-011"])
        self.assertNotIn("ux", item.topics)
        self.assertTrue(item.safety_flag)
        self.assertEqual(item.severity, "critical")


if __name__ == "__main__":
    unittest.main()
