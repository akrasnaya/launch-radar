import csv
import tempfile
import unittest
from pathlib import Path

from launch_radar.clustering import text_overlap
from launch_radar.dmv_corpus import cluster_view, load_texts, summarize


class DmvCorpusTest(unittest.TestCase):
    def test_distinct_descriptions_keep_report_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dmv.csv"
            with path.open("w", newline="", encoding="latin-1") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Manufacturer",
                        "DATE",
                        "DESCRIPTION OF FACTS CAUSING DISENGAGEMENT",
                    ]
                )
                writer.writerow(["Waymo LLC", "1/2/2023", "Driver took over for a brake failure near a pedestrian."])
                writer.writerow(["Waymo LLC", "1/3/2023", "Driver took over for a brake failure near a pedestrian."])
                writer.writerow(["Zoox, Inc", "2/2/2023", "System software failure in the planning module."])
                writer.writerow(["", "", "   "])

            texts = load_texts(path)
            self.assertEqual(len(texts), 2)
            self.assertEqual(sum(item.reports for item in texts), 3)
            payload = summarize(path, limit=5)
            self.assertEqual(payload["reports"], 3)
            self.assertEqual(payload["distinct_texts"], 2)
            safety = [item for item in texts if "brake" in item.record.text]
            self.assertEqual(safety[0].reports, 2)

    def test_shared_closing_sentence_does_not_glue_different_causes(self) -> None:
        tail = " The Safety Driver was able to safely take control of the vehicle."
        predicted = "The ADS incorrectly predicted the behavior of another road user." + tail
        hardware = "The ADS initiated a disengagement due to a diagnosed hardware issue." + tail
        self.assertGreaterEqual(text_overlap(predicted, hardware)[0], 0.34)
        self.assertLess(text_overlap(cluster_view(predicted), cluster_view(hardware))[0], 0.34)


if __name__ == "__main__":
    unittest.main()
