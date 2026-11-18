"""Synthetic pilot-versus-holdout comparison. This is not causal evidence."""

from __future__ import annotations

from launch_radar.models import FeedbackRecord, IssueCluster, RiskCard

PILOT_DEVICES = 1_240
HOLDOUT_DEVICES = 1_180


def build_risk_cards(
    records: list[FeedbackRecord],
    clusters: list[IssueCluster],
) -> list[RiskCard]:
    by_id = {record.record_id: record for record in records}
    cards = [_card(cluster, by_id) for cluster in clusters]
    return sorted(cards, key=_sort_key)


def _card(cluster: IssueCluster, by_id: dict[str, FeedbackRecord]) -> RiskCard:
    members = [by_id[record_id] for record_id in cluster.record_ids]
    pilot_count = sum(record.cohort == "pilot" for record in members)
    holdout_count = sum(record.cohort == "holdout" for record in members)
    pilot_rate = _rate(pilot_count, PILOT_DEVICES)
    holdout_rate = _rate(holdout_count, HOLDOUT_DEVICES)
    ratio = pilot_rate / holdout_rate if holdout_rate else None
    recommendation, rationale = _recommend(cluster, ratio, members)
    return RiskCard(
        cluster_id=cluster.cluster_id,
        label=cluster.label,
        pilot_count=pilot_count,
        holdout_count=holdout_count,
        pilot_rate=round(pilot_rate, 2),
        holdout_rate=round(holdout_rate, 2),
        rate_ratio=round(ratio, 2) if ratio is not None else None,
        severity=cluster.severity,
        safety_flag=cluster.safety_flag,
        confidence=cluster.confidence,
        recommendation=recommendation,
        synthetic=all(record.synthetic for record in members),
        rationale=rationale,
    )


def _recommend(
    cluster: IssueCluster,
    ratio: float | None,
    members: list[FeedbackRecord],
) -> tuple[str, str]:
    if cluster.safety_flag:
        return (
            "HOLD",
            "Safety candidate bypasses frequency ranking and requires human review.",
        )
    environments = {record.build for record in members}
    if len({record.cohort for record in members}) > 1 and len(environments) > 1 and ratio is None:
        return ("REVIEW", "Cohort comparison is incomplete.")
    if ratio is not None and ratio >= 2 and cluster.severity in {"high", "critical"}:
        return ("HOLD", "Synthetic pilot rate is materially higher than holdout.")
    if cluster.severity == "low":
        return ("BACKLOG", "Low-severity issue stays in the backlog.")
    return ("REVIEW", "Evidence is not sufficient for an automatic release verdict.")


def _rate(count: int, devices: int) -> float:
    return 1_000 * count / devices


def _sort_key(card: RiskCard) -> tuple[int, float]:
    safety_first = 0 if card.safety_flag else 1
    magnitude = -(card.rate_ratio or 0)
    return (safety_first, magnitude)
