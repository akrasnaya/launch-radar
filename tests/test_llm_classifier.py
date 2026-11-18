import json
import unittest

from launch_radar.llm_classifier import guard_safety, parse_label


class LlmParseTest(unittest.TestCase):
    def test_parse_extracts_json_label(self) -> None:
        label = parse_label('{"cause":"prediction","severity":"medium","safety":false}')
        self.assertEqual(label.cause, "prediction")
        self.assertEqual(label.severity, "medium")
        self.assertFalse(label.safety)

    def test_parse_accepts_string_bool(self) -> None:
        label = parse_label('{"cause":"conflict","severity":"high","safety":"false"}')
        self.assertEqual(label.cause, "conflict")
        self.assertFalse(label.safety)

    def test_guard_drops_hardware_handback(self) -> None:
        text = (
            "The ADS initiated a disengagement due to a diagnosed hardware issue. "
            "The Safety Driver was able to safely take control of the vehicle."
        )
        self.assertFalse(guard_safety(text, True))

    def test_guard_keeps_yield_to_pedestrian(self) -> None:
        text = "Driver took over for improper yielding to a pedestrian near the roadway."
        self.assertTrue(guard_safety(text, False))

    def test_guard_leaves_lane_boundary_alone(self) -> None:
        text = "Safety Driver disengaged upon judging that vehicle was too close to road/lane boundary."
        self.assertFalse(guard_safety(text, False))

    def test_guard_does_not_treat_merge_yield_as_safety(self) -> None:
        text = "Driver disengagement due to planning discrepancy in yielding to merging vehicle."
        self.assertFalse(guard_safety(text, False))

    def test_guard_drops_give_more_space(self) -> None:
        text = "Driver took over to yield to pedestrians at a crosswalk and give more space."
        self.assertFalse(guard_safety(text, True))

    def test_guard_drops_detector_timeout(self) -> None:
        text = "Pedestrian detection timeout while the module state error was open."
        self.assertFalse(guard_safety(text, True))

    def test_parse_rejects_unknown_cause(self) -> None:
        with self.assertRaises(ValueError):
            parse_label(json.dumps({"cause": "magic", "severity": "low", "safety": False}))


if __name__ == "__main__":
    unittest.main()
