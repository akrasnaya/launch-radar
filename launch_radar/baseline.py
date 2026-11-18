"""Keyword baseline. A later model must beat this on weighted error."""

from __future__ import annotations

import re

from launch_radar.models import Classification, FeedbackRecord
from launch_radar.taxonomy import SAFETY_TERMS, TOPICS

_TOKEN = re.compile(r"[a-z0-9]+")
_NEGATIVE = {"fail", "failed", "lost", "unable", "not", "no", "wrong", "unsafe", "cannot"}
_POSITIVE = {"works", "good", "clear", "fixed", "improved"}
_NEGATORS = {"no", "not", "without", "never"}
_SAFETY_PHRASES = (
    ("collision",),
    ("crash",),
    ("crashed",),
    ("pedestrian",),
    ("unsafe",),
    ("emergency",),
    ("takeover",),
    ("take", "over"),
    ("took", "over"),
    ("taken", "over"),
    ("brake", "failure"),
)


def ordered_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def tokenize(text: str) -> set[str]:
    return set(ordered_tokens(text))


def _phrase_spans(tokens: list[str], phrase: tuple[str, ...]) -> list[int]:
    width = len(phrase)
    return [index for index in range(len(tokens) - width + 1) if tuple(tokens[index : index + width]) == phrase]


def _negated(tokens: list[str], index: int, phrase: tuple[str, ...]) -> bool:
    window = tokens[max(0, index - 3) : index]
    if any(token in _NEGATORS for token in window):
        return True
    return phrase == ("takeover",) and any(token in {"precautionary", "accidental"} for token in window)


def _active_phrases(tokens: list[str], phrases: tuple[tuple[str, ...], ...]) -> list[str]:
    hits: list[str] = []
    for phrase in phrases:
        spans = [index for index in _phrase_spans(tokens, phrase) if not _negated(tokens, index, phrase)]
        if spans:
            hits.append(" ".join(phrase))
    return hits


def classify(record: FeedbackRecord) -> Classification:
    tokens = ordered_tokens(record.text)
    topics: list[str] = []
    sentiment: dict[str, str] = {}
    matched: list[str] = []
    tone = _aspect_sentiment(tokens)

    for topic, terms in TOPICS.items():
        phrases = tuple(tuple(term.split()) for term in terms)
        hits = _active_phrases(tokens, phrases)
        if not hits:
            continue
        topics.append(topic)
        matched.extend(hits)
        sentiment[topic] = tone

    safety_hits = _active_phrases(tokens, _SAFETY_PHRASES)
    safety_flag = bool(safety_hits)
    matched.extend(safety_hits)
    severity = _severity(safety_flag, topics, tokens)
    coverage = len(set(matched))
    confidence = min(0.95, 0.35 + 0.12 * coverage)
    abstain = not topics or confidence < 0.5

    return Classification(
        record_id=record.record_id,
        topics=tuple(topics),
        topic_sentiment=sentiment,
        severity="medium" if abstain else severity,
        safety_flag=safety_flag,
        confidence=round(confidence, 2),
        abstain=abstain,
        matched_terms=tuple(sorted(set(matched))),
    )


def _aspect_sentiment(tokens: list[str]) -> str:
    negative = any(token in _NEGATIVE for token in tokens)
    positive = any(token in _POSITIVE for token in tokens)
    if negative and positive:
        return "mixed"
    if negative:
        return "negative"
    if positive:
        return "positive"
    return "neutral"


def _severity(safety_flag: bool, topics: list[str], tokens: list[str]) -> str:
    if safety_flag or _phrase_spans(tokens, ("brake", "failure")):
        return "critical"
    if "navigation" in topics or "hardware" in topics:
        return "high"
    if "connectivity" in topics or "support" in topics:
        return "medium"
    return "low"
