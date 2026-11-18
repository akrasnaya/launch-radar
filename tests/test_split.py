import unittest

from launch_radar.annotation import text_id
from launch_radar.dmv_labels import labeled_rows
from launch_radar.split import TRAIN_ONLY_REPORTS, build_test_ids, load_test_ids, test_rows, train_rows


class SplitTest(unittest.TestCase):
    def test_frozen_file_matches_the_seed(self) -> None:
        self.assertEqual(load_test_ids(), set(build_test_ids()))

    def test_holdout_is_a_small_partition(self) -> None:
        rows = labeled_rows()
        held = test_rows()
        train = train_rows()
        held_ids = {text_id(str(row["text"])) for row in held}
        train_ids = {text_id(str(row["text"])) for row in train}
        self.assertEqual(len(held_ids & train_ids), 0)
        self.assertEqual(len(held) + len(train), len(rows))
        self.assertLess(len(held), len(train))
        self.assertGreaterEqual(len(held), 40)
        self.assertLessEqual(len(held), 80)
        self.assertTrue(any(row["safety"] == "true" for row in held))
        self.assertTrue(any(row["severity"] == "critical" for row in held))
        self.assertTrue(all(int(row["reports"]) < TRAIN_ONLY_REPORTS for row in held))
        self.assertTrue(any(row["safety"] == "true" and int(row["reports"]) >= 2 for row in held))


if __name__ == "__main__":
    unittest.main()
