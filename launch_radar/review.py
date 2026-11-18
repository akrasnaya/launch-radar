"""Human review and the release signature. The model recommendation stays put."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from launch_radar.models import Classification, FeedbackRecord, RiskCard

REVIEW_ACTIONS = frozenset({"accept", "correct", "split", "abstain"})
GATE_DECISIONS = frozenset({"Hold", "Rollback", "Expand"})


@dataclass(frozen=True)
class Review:
    record_id: str
    action: str
    note: str = ""


def parse_review(payload: Mapping[str, object], known_ids: set[str]) -> Review:
    record_id = str(payload.get("record_id", ""))
    action = str(payload.get("action", ""))
    note = str(payload.get("note", "")).strip()
    if record_id not in known_ids:
        raise ValueError("Нет такого сообщения.")
    if action not in REVIEW_ACTIONS:
        raise ValueError("Неизвестное действие.")
    return Review(record_id=record_id, action=action, note=note)


def pending_safety_ids(
    records: Sequence[FeedbackRecord],
    classifications: Mapping[str, Classification],
    reviews: Mapping[str, Review],
) -> tuple[str, ...]:
    return tuple(
        record.record_id
        for record in records
        if classifications[record.record_id].safety_flag and record.record_id not in reviews
    )


def advisory_recommendation(cards: Sequence[RiskCard]) -> str:
    if any(card.safety_flag or card.recommendation == "HOLD" for card in cards):
        return "HOLD"
    if any(card.recommendation == "REVIEW" for card in cards):
        return "REVIEW"
    return "BACKLOG"


def validate_signature(
    decision: str,
    signer: str,
    records: Sequence[FeedbackRecord],
    classifications: Mapping[str, Classification],
    reviews: Mapping[str, Review],
) -> None:
    if decision not in GATE_DECISIONS:
        raise ValueError("Неизвестное решение.")
    if not signer.strip():
        raise ValueError("Укажите подписанта.")
    if decision == "Expand" and pending_safety_ids(records, classifications, reviews):
        raise ValueError("Expand недоступен, пока есть safety без разбора.")
