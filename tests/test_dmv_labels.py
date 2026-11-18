import unittest

from launch_radar.dmv_labels import SCHEME, label_disengagement, labeled_rows


class DmvLabelTest(unittest.TestCase):
    def test_precautionary_takeover_is_not_a_safety_event(self) -> None:
        label = label_disengagement(
            "Precautionary Takeover: Disengaged due to a vehicle platform software issue "
            "within the Autonomous Vehicle's (AV) controller interface sub-system"
        )
        self.assertEqual(label.cause, "platform")
        self.assertEqual(label.severity, "low")
        self.assertFalse(label.safety)

    def test_failed_pedestrian_yield_blocks_release(self) -> None:
        label = label_disengagement(
            "Driver took over due to failure to yield to pedestrian at crosswalk."
        )
        self.assertEqual(label.cause, "conflict")
        self.assertEqual(label.severity, "critical")
        self.assertTrue(label.safety)

    def test_lane_marker_contact_is_not_a_crash(self) -> None:
        label = label_disengagement(
            "The AV Test Driver took control when the vehicle made contact with a lane marker."
        )
        self.assertEqual(label.cause, "planning")
        self.assertFalse(label.safety)

    def test_every_distinct_report_has_one_label(self) -> None:
        rows = labeled_rows()
        self.assertEqual(len(rows), 304)
        self.assertEqual(sum(int(row["reports"]) for row in rows), 6562)
        self.assertTrue(all(row["scheme"] == SCHEME for row in rows))
        self.assertEqual(sum(int(row["reports"]) for row in rows if row["rule"] == "fallback"), 0)


if __name__ == "__main__":
    unittest.main()
