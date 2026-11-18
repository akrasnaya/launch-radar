"""Leave-one-out bag-of-words model. It does not replace classify()."""

from __future__ import annotations

import math
from dataclasses import dataclass

from launch_radar.baseline import _TOKEN, _aspect_sentiment, _severity, classify, ordered_tokens
from launch_radar.models import FeedbackRecord
from launch_radar.sample_data import sample_records

TOPICS = ("navigation", "perception_safety", "connectivity", "hardware", "ux", "support")
_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_STOP = frozenset({"a", "an", "and", "for", "in", "is", "it", "its", "of", "on", "or", "the", "to"})

# Single-author labels for the 12 synthetic lines. Not a production gold set.
_GOLD: dict[str, tuple[frozenset[str], str, str, bool]] = {
    "fb-001": (frozenset({"navigation", "perception_safety"}), "negative", "critical", True),
    "fb-002": (frozenset({"navigation", "perception_safety"}), "negative", "critical", True),
    "fb-003": (frozenset({"navigation", "perception_safety"}), "negative", "critical", True),
    "fb-004": (frozenset({"navigation"}), "negative", "low", False),
    "fb-005": (frozenset({"ux"}), "negative", "low", False),
    "fb-006": (frozenset({"ux"}), "negative", "low", False),
    "fb-007": (frozenset({"ux"}), "negative", "low", False),
    "fb-008": (frozenset({"connectivity"}), "negative", "medium", False),
    "fb-009": (frozenset({"hardware"}), "negative", "medium", False),
    "fb-010": (frozenset({"support"}), "negative", "medium", False),
    "fb-011": (frozenset({"perception_safety"}), "negative", "critical", True),
    "fb-012": (frozenset({"navigation"}), "positive", "low", False),
}


@dataclass(frozen=True)
class Prediction:
    topics: frozenset[str]
    sentiment: str
    severity: str
    safety: bool


@dataclass(frozen=True)
class Comparison:
    keyword_mean: float
    model_mean: float
    rows: tuple[tuple[str, int, int], ...]


def features(text: str) -> set[str]:
    tokens = [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]
    found = set(tokens)
    found.update(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return found


def _weighted_error(record_id: str, prediction: Prediction) -> int:
    topics, sentiment, severity, safety = _GOLD[record_id]
    error = 0
    if safety != prediction.safety:
        error += 5
    error += len(topics ^ prediction.topics)
    if sentiment != prediction.sentiment:
        error += 1
    error += abs(_RANK[severity] - _RANK[prediction.severity])
    return error


def _keyword_prediction(record: FeedbackRecord) -> Prediction:
    item = classify(record)
    tones = set(item.topic_sentiment.values())
    sentiment = next(iter(tones)) if len(tones) == 1 else "mixed" if tones else "neutral"
    return Prediction(frozenset(item.topics), sentiment, item.severity, item.safety_flag)


class _BinaryNB:
    def __init__(self, positive: list[set[str]], negative: list[set[str]]) -> None:
        self._positive = positive
        self._negative = negative
        self._vocab: set[str] = set()
        for feats in positive + negative:
            self._vocab.update(feats)
        total = len(positive) + len(negative)
        self._prior = (len(positive) + 1) / (total + 2)

    def predicts(self, feats: set[str]) -> bool:
        log_pos = math.log(self._prior)
        log_neg = math.log(1 - self._prior)
        pos_n = len(self._positive)
        neg_n = len(self._negative)
        # Short texts: score tokens that occur, not the long tail of absences.
        for token in feats & self._vocab:
            pos_p = (sum(token in item for item in self._positive) + 1) / (pos_n + 2)
            neg_p = (sum(token in item for item in self._negative) + 1) / (neg_n + 2)
            log_pos += math.log(pos_p)
            log_neg += math.log(neg_p)
        return log_pos > log_neg


def _predict_label(feats: set[str], rows: list[tuple[FeedbackRecord, set[str]]], label_of) -> bool:
    positive = [item_feats for record, item_feats in rows if label_of(record)]
    negative = [item_feats for record, item_feats in rows if not label_of(record)]
    return _BinaryNB(positive, negative).predicts(feats)


def predict_loo(records: list[FeedbackRecord]) -> dict[str, Prediction]:
    """Train on every record except the one being scored."""
    featured = [(record, features(record.text)) for record in records]
    predictions: dict[str, Prediction] = {}
    for held, held_feats in featured:
        train = [(record, feats) for record, feats in featured if record.record_id != held.record_id]
        topics = [
            topic
            for topic in TOPICS
            if _predict_label(held_feats, train, lambda record, topic=topic: topic in _GOLD[record.record_id][0])
        ]
        safety = _predict_label(held_feats, train, lambda record: _GOLD[record.record_id][3])
        tokens = ordered_tokens(held.text)
        tone = _aspect_sentiment(tokens)
        severity = "medium" if not topics else _severity(safety, topics, tokens)
        predictions[held.record_id] = Prediction(frozenset(topics), tone, severity, safety)
    return predictions


def evaluate(records: list[FeedbackRecord] | None = None) -> Comparison:
    rows = records if records is not None else sample_records()
    model = predict_loo(rows)
    scored = []
    keyword_total = 0
    model_total = 0
    for record in rows:
        keyword_error = _weighted_error(record.record_id, _keyword_prediction(record))
        model_error = _weighted_error(record.record_id, model[record.record_id])
        keyword_total += keyword_error
        model_total += model_error
        scored.append((record.record_id, keyword_error, model_error))
    count = len(rows)
    return Comparison(keyword_total / count, model_total / count, tuple(scored))


def main() -> None:
    report = evaluate()
    print("Baseline: keyword classify()")
    print(f"Keyword mean weighted error: {report.keyword_mean:.2f}")
    print(f"Primitive NB mean weighted error: {report.model_mean:.2f}")
    print(f"{'id':<8} {'keyword':>8} {'model':>8}")
    for record_id, keyword_error, model_error in report.rows:
        print(f"{record_id:<8} {keyword_error:8d} {model_error:8d}")


if __name__ == "__main__":
    main()
