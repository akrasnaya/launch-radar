import unittest

from launch_radar.pipeline import run
from launch_radar.sample_data import sample_records


class PipelineTest(unittest.TestCase):
    def test_safety_issue_is_held_despite_lower_volume(self) -> None:
        records = sample_records()
        classifications, _, cards = run(records)

        self.assertTrue(classifications["fb-011"].safety_flag)
        self.assertEqual(classifications["fb-011"].severity, "critical")
        safety_cards = [card for card in cards if card.safety_flag]
        self.assertTrue(safety_cards)
        self.assertTrue(all(card.recommendation == "HOLD" for card in safety_cards))
        self.assertLess(cards.index(safety_cards[0]), len(cards) - 1)

    def test_reworded_interface_complaints_share_one_cluster(self) -> None:
        records = sample_records()
        _, clusters, _ = run(records)
        grouped = [
            cluster
            for cluster in clusters
            if {"fb-005", "fb-006"}.issubset(cluster.record_ids)
        ]
        self.assertEqual(len(grouped), 1)

    def test_every_sample_card_is_marked_synthetic(self) -> None:
        records = sample_records()
        _, _, cards = run(records)
        self.assertTrue(cards)
        self.assertTrue(all(card.synthetic for card in cards))


if __name__ == "__main__":
    unittest.main()
