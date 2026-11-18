"""Quality of the keyword baseline and a bag-of-words model on DMV labels.

Folds are split by distinct description, so copies of one sentence stay together.
The labels were written by one rule pass, so the learned model is recovering that
scheme from wording. It is not a second annotator.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from launch_radar.baseline import classify
from launch_radar.dmv_labels import labeled_rows
from launch_radar.models import FeedbackRecord
from launch_radar.primitive_model import features

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_FOLDS = 5


@dataclass(frozen=True)
class BinaryScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class TaskScore:
    name: str
    accuracy: float
    macro_f1: float
    safety: BinaryScore | None = None


@dataclass(frozen=True)
class Slice:
    name: str
    keyword_safety: BinaryScore
    keyword_severity: TaskScore
    majority_safety: BinaryScore
    majority_severity: TaskScore
    majority_cause: TaskScore
    model_safety: BinaryScore
    model_severity: TaskScore
    model_cause: TaskScore


@dataclass(frozen=True)
class QualityReport:
    formulations: int
    reports: int
    by_report: Slice
    by_formulation: Slice


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _binary(truth: list[str], predicted: list[str], weights: list[int], positive: str) -> BinaryScore:
    true_positive = false_positive = false_negative = support = 0
    for actual, guess, weight in zip(truth, predicted, weights, strict=True):
        if actual == positive:
            support += weight
        if actual == positive and guess == positive:
            true_positive += weight
        elif guess == positive:
            false_positive += weight
        elif actual == positive:
            false_negative += weight
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return BinaryScore(precision, recall, _f1(precision, recall), support)


def _accuracy(truth: list[str], predicted: list[str], weights: list[int]) -> float:
    total = sum(weights)
    correct = sum(weight for actual, guess, weight in zip(truth, predicted, weights, strict=True) if actual == guess)
    return correct / total if total else 0.0


def _macro_f1(truth: list[str], predicted: list[str], weights: list[int]) -> float:
    labels = sorted(set(truth))
    scores = []
    for label in labels:
        binary = _binary(truth, predicted, weights, label)
        scores.append(binary.f1)
    return sum(scores) / len(scores) if scores else 0.0


def _task(name: str, truth: list[str], predicted: list[str], weights: list[int]) -> TaskScore:
    safety = _binary(truth, predicted, weights, "true") if "true" in truth or "false" in truth else None
    return TaskScore(name, _accuracy(truth, predicted, weights), _macro_f1(truth, predicted, weights), safety)


class _MulticlassNB:
    def __init__(self, samples: list[tuple[set[str], str]]) -> None:
        grouped: dict[str, list[set[str]]] = defaultdict(list)
        vocabulary: set[str] = set()
        for feats, label in samples:
            grouped[label].append(feats)
            vocabulary.update(feats)
        self._labels = sorted(grouped)
        total = len(samples)
        self._prior = {label: (len(grouped[label]) + 1) / (total + len(self._labels)) for label in self._labels}
        self._token = {}
        for label, documents in grouped.items():
            count = len(documents)
            self._token[label] = {
                token: (sum(token in document for document in documents) + 1) / (count + 2)
                for token in vocabulary
            }
        self._vocabulary = vocabulary

    def predict(self, feats: set[str]) -> str:
        best_label = self._labels[0]
        best_score = float("-inf")
        for label in self._labels:
            score = math.log(self._prior[label])
            probabilities = self._token[label]
            for token in feats & self._vocabulary:
                score += math.log(probabilities[token])
            if score > best_score:
                best_label = label
                best_score = score
        return best_label


def _folds(size: int, labels: list[str], folds: int) -> list[set[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[label].append(index)
    held: list[set[int]] = [set() for _ in range(folds)]
    for indexes in buckets.values():
        for offset, index in enumerate(indexes):
            held[offset % folds].add(index)
    missing = set(range(size)) - set().union(*held)
    if missing:
        raise RuntimeError("fold split dropped a formulation")
    return held


def _predict_fold(samples: list[tuple[set[str], str]], held: set[int]) -> list[str]:
    train = [sample for index, sample in enumerate(samples) if index not in held]
    model = _MulticlassNB(train)
    return [model.predict(samples[index][0]) for index in sorted(held)]


def _slice(
    name: str,
    weights: list[int],
    safety: list[str],
    severity: list[str],
    cause: list[str],
    keyword_safety: list[str],
    keyword_severity: list[str],
    model_safety: list[str],
    model_severity: list[str],
    model_cause: list[str],
) -> Slice:
    def mode(labels: list[str]) -> str:
        counts: Counter[str] = Counter()
        for label, weight in zip(labels, weights, strict=True):
            counts[label] += weight
        return counts.most_common(1)[0][0]

    majority_safety = ["false"] * len(safety)
    majority_severity = [mode(severity)] * len(severity)
    majority_cause = [mode(cause)] * len(cause)
    return Slice(
        name,
        _binary(safety, keyword_safety, weights, "true"),
        _task("severity", severity, keyword_severity, weights),
        _binary(safety, majority_safety, weights, "true"),
        _task("severity", severity, majority_severity, weights),
        _task("cause", cause, majority_cause, weights),
        _binary(safety, model_safety, weights, "true"),
        _task("severity", severity, model_severity, weights),
        _task("cause", cause, model_cause, weights),
    )


def evaluate() -> QualityReport:
    rows = labeled_rows()
    report_weights = [int(row["reports"]) for row in rows]
    formulation_weights = [1] * len(rows)
    texts = [str(row["text"]) for row in rows]
    safety = [str(row["safety"]) for row in rows]
    severity = [str(row["severity"]) for row in rows]
    cause = [str(row["cause"]) for row in rows]
    keyword_safety: list[str] = []
    keyword_severity: list[str] = []
    for text in texts:
        item = classify(FeedbackRecord("eval", text, "dmv-ca-2023", "disengagement", "2023", False))
        keyword_safety.append("true" if item.safety_flag else "false")
        keyword_severity.append(item.severity)

    safety_folds = _folds(len(rows), safety, _FOLDS)
    model_safety = _oof([(features(text), label) for text, label in zip(texts, safety, strict=True)], safety_folds)
    model_severity = _oof(
        [(features(text), label) for text, label in zip(texts, severity, strict=True)],
        safety_folds,
    )
    model_cause = _oof([(features(text), label) for text, label in zip(texts, cause, strict=True)], safety_folds)
    shared = (
        safety,
        severity,
        cause,
        keyword_safety,
        keyword_severity,
        model_safety,
        model_severity,
        model_cause,
    )
    return QualityReport(
        formulations=len(rows),
        reports=sum(report_weights),
        by_report=_slice("по отчётам", report_weights, *shared),
        by_formulation=_slice("по формулировкам", formulation_weights, *shared),
    )


def _oof(samples: list[tuple[set[str], str]], folds: list[set[int]]) -> list[str]:
    predicted = [""] * len(samples)
    for held in folds:
        guesses = _predict_fold(samples, held)
        for index, guess in zip(sorted(held), guesses, strict=True):
            predicted[index] = guess
    return predicted


def _percent(score: BinaryScore) -> str:
    return f"P {score.precision:.2f}  R {score.recall:.2f}  F1 {score.f1:.2f}  n {score.support}"


def _print_slice(slice_: Slice) -> None:
    print(slice_.name)
    print("  safety F1")
    print("    majority always false   ", _percent(slice_.majority_safety))
    print("    keyword baseline        ", _percent(slice_.keyword_safety))
    print("    naive Bayes, out of fold", _percent(slice_.model_safety))
    print("  severity accuracy / macro-F1")
    print(f"    majority {slice_.majority_severity.accuracy:.2f} / {slice_.majority_severity.macro_f1:.2f}")
    print(f"    keyword  {slice_.keyword_severity.accuracy:.2f} / {slice_.keyword_severity.macro_f1:.2f}")
    print(f"    Bayes    {slice_.model_severity.accuracy:.2f} / {slice_.model_severity.macro_f1:.2f}")
    print("  cause accuracy / macro-F1")
    print(f"    majority {slice_.majority_cause.accuracy:.2f} / {slice_.majority_cause.macro_f1:.2f}")
    print("    keyword  no cause head")
    print(f"    Bayes    {slice_.model_cause.accuracy:.2f} / {slice_.model_cause.macro_f1:.2f}")


def main() -> None:
    report = evaluate()
    print(f"formulations {report.formulations} reports {report.reports} folds {_FOLDS}")
    _print_slice(report.by_report)
    _print_slice(report.by_formulation)


if __name__ == "__main__":
    main()
