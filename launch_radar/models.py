"""Records produced by the prototype. Synthetic fields stay explicitly marked."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeedbackRecord:
    record_id: str
    text: str
    source: str
    cohort: str
    build: str
    synthetic: bool = True


@dataclass(frozen=True)
class Classification:
    record_id: str
    topics: tuple[str, ...]
    topic_sentiment: dict[str, str]
    severity: str
    safety_flag: bool
    confidence: float
    abstain: bool
    matched_terms: tuple[str, ...] = ()


@dataclass
class IssueCluster:
    cluster_id: str
    label: str
    record_ids: list[str] = field(default_factory=list)
    severity: str = "low"
    safety_flag: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class RiskCard:
    cluster_id: str
    label: str
    pilot_count: int
    holdout_count: int
    pilot_rate: float
    holdout_rate: float
    rate_ratio: float | None
    severity: str
    safety_flag: bool
    confidence: float
    recommendation: str
    synthetic: bool
    rationale: str
