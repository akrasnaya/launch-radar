"""Run the baseline path from feedback text to a synthetic risk readout."""

from __future__ import annotations

from launch_radar.baseline import classify
from launch_radar.clustering import cluster_records
from launch_radar.models import Classification, FeedbackRecord, IssueCluster, RiskCard
from launch_radar.risk import build_risk_cards


def run(records: list[FeedbackRecord]) -> tuple[
    dict[str, Classification],
    list[IssueCluster],
    list[RiskCard],
]:
    classifications = {record.record_id: classify(record) for record in records}
    clusters = cluster_records(records, classifications)
    cards = build_risk_cards(records, clusters)
    return classifications, clusters, cards


def render(cards: list[RiskCard]) -> str:
    lines = [
        "LaunchRadar prototype",
        "All cohort comparisons below are synthetic.",
        "",
        f"{'Issue':<12} {'Label':<22} {'Sev':<10} {'Ratio':>7}  Decision",
        "-" * 72,
    ]
    for card in cards:
        ratio = "-" if card.rate_ratio is None else f"{card.rate_ratio:.2f}"
        lines.append(
            f"{card.cluster_id:<12} {card.label:<22} {card.severity:<10} {ratio:>7}  {card.recommendation}"
        )
        lines.append(f"  {card.rationale}")
    return "\n".join(lines)
