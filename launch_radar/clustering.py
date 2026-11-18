"""Group differently worded records that describe the same issue."""

from __future__ import annotations

from launch_radar.baseline import tokenize
from launch_radar.models import Classification, FeedbackRecord, IssueCluster

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
THRESHOLD = 0.34


def cluster_records(
    records: list[FeedbackRecord],
    classifications: dict[str, Classification],
    threshold: float = THRESHOLD,
) -> list[IssueCluster]:
    pending = list(records)
    clusters: list[IssueCluster] = []

    while pending:
        seed = pending.pop(0)
        members = [seed]
        seed_tokens = tokenize(seed.text)
        remaining: list[FeedbackRecord] = []
        for candidate in pending:
            score = _jaccard(seed_tokens, tokenize(candidate.text))
            if score >= threshold:
                members.append(candidate)
            else:
                remaining.append(candidate)
        pending = remaining
        clusters.append(_make_cluster(len(clusters) + 1, members, classifications))

    return clusters


def _make_cluster(
    number: int,
    members: list[FeedbackRecord],
    classifications: dict[str, Classification],
) -> IssueCluster:
    labels = [
        classifications[member.record_id]
        for member in members
        if member.record_id in classifications
    ]
    top_topic = _most_common_topic(labels)
    severity = max((item.severity for item in labels), key=_SEVERITY_RANK.get, default="low")
    confidence = sum(item.confidence for item in labels) / len(labels) if labels else 0.0
    return IssueCluster(
        cluster_id=f"ISSUE-{number:03d}",
        label=top_topic.replace("_", " "),
        record_ids=[member.record_id for member in members],
        severity=severity,
        safety_flag=any(item.safety_flag for item in labels),
        confidence=round(confidence, 2),
    )


def _most_common_topic(labels: list[Classification]) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        for topic in label.topics:
            counts[topic] = counts.get(topic, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def text_overlap(left: str, right: str) -> tuple[float, tuple[str, ...]]:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    shared = tuple(sorted(left_tokens & right_tokens))
    return _jaccard(left_tokens, right_tokens), shared
