import tempfile
import unittest
from pathlib import Path

from launch_radar.annotation import annotation_state, apply_second_pass, save_annotation, second_reading


class AnnotationTest(unittest.TestCase):
    def test_queue_hides_the_first_pass_and_kappa_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.csv"
            state = annotation_state(path)
            self.assertIsNone(state["kappa"])
            self.assertNotIn("reference", state["item"])
            self.assertEqual(state["progress"]["total"], 304)

    def test_two_labels_produce_kappa_without_showing_the_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.csv"
            first = annotation_state(path)["item"]
            save_annotation(
                {"id": first["id"], "cause": "prediction", "severity": "medium", "safety": False},
                path,
            )
            second = annotation_state(path)["item"]
            state = save_annotation(
                {"id": second["id"], "cause": "hardware", "severity": "low", "safety": False},
                path,
            )
            self.assertEqual(state["progress"]["labeled"], 2)
            self.assertIsNotNone(state["kappa"]["severity"])
            self.assertNotIn("reference", state["item"] or {})

    def test_review_pass_lowers_a_diagnosed_motion_plan(self) -> None:
        cause, severity, safety = second_reading(
            "The ADS initiated a disengagement due to a diagnosed issue with the motion plan. "
            "The Safety Driver was able to safely take control of the vehicle."
        )
        self.assertEqual((cause, severity, safety), ("planning", "low", False))
        with tempfile.TemporaryDirectory() as directory:
            state = apply_second_pass(Path(directory) / "human.csv")
            self.assertEqual(state["progress"]["labeled"], 304)
            self.assertGreater(len(state["disagreements"]), 0)
            self.assertGreater(state["kappa"]["severity"], 0.7)


if __name__ == "__main__":
    unittest.main()
