import unittest

from launch_radar.annotation import text_id
from launch_radar.finetune import dev_rows, oversample
from launch_radar.split import load_test_ids, train_rows


class FinetuneSplitTest(unittest.TestCase):
    def test_dev_split_stays_inside_train(self) -> None:
        fit, valid = dev_rows()
        test_ids = load_test_ids()
        train_ids = {text_id(str(row["text"])) for row in train_rows()}
        fit_ids = {text_id(str(row["text"])) for row in fit}
        valid_ids = {text_id(str(row["text"])) for row in valid}
        self.assertEqual(len(fit_ids & valid_ids), 0)
        self.assertEqual(fit_ids | valid_ids, train_ids)
        self.assertTrue(fit_ids.isdisjoint(test_ids))
        self.assertTrue(valid_ids.isdisjoint(test_ids))
        self.assertGreater(len(valid), 0)
        self.assertLess(len(valid), len(fit))

    def test_oversample_repeats_rare_severity_only(self) -> None:
        fit, _valid = dev_rows()
        expanded = oversample(fit)
        self.assertGreater(len(expanded), len(fit))
        originals = {text_id(str(row["text"])) for row in fit}
        self.assertEqual({text_id(str(row["text"])) for row in expanded}, originals)


if __name__ == "__main__":
    unittest.main()
