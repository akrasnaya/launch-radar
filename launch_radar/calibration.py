"""Selective threshold for the keyword safety flag on DMV labels.

Confidence stays 0.35 + 0.12 × matched terms. This module only reads that score
and reports the precision-coverage curve. It does not claim a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass

from launch_radar.baseline import classify
from launch_radar.dmv_labels import labeled_rows
from launch_radar.models import FeedbackRecord


@dataclass(frozen=True)
class SafetyPoint:
    minimum_confidence: float
    coverage: float
    precision: float
    recall: float
    f1: float
    autoclear_coverage: float
    autoclear_missed_safety: int


def safety_curve() -> list[SafetyPoint]:
    scored = []
    for row in labeled_rows():
        item = classify(FeedbackRecord("cal", str(row["text"]), "dmv-ca-2023", "disengagement", "2023", False))
        scored.append((item.confidence, item.safety_flag, str(row["safety"]) == "true", int(row["reports"])))
    total = sum(weight for _, _, _, weight in scored)
    gold = sum(weight for _, _, actual, weight in scored if actual)
    points = []
    for threshold in sorted({confidence for confidence, _, _, _ in scored}):
        selected = [row for row in scored if row[0] >= threshold]
        weight = sum(item[3] for item in selected)
        true_positive = sum(item[3] for item in selected if item[1] and item[2])
        false_positive = sum(item[3] for item in selected if item[1] and not item[2])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        cleared = [row for row in scored if row[0] >= threshold and not row[1]]
        cleared_weight = sum(item[3] for item in cleared)
        missed = sum(item[3] for item in cleared if item[2])
        points.append(
            SafetyPoint(
                minimum_confidence=threshold,
                coverage=weight / total,
                precision=precision,
                recall=recall,
                f1=f1,
                autoclear_coverage=cleared_weight / total,
                autoclear_missed_safety=missed,
            )
        )
    return points


def best_f1_point() -> SafetyPoint:
    return max(safety_curve(), key=lambda point: (point.f1, point.recall))


def reaches_precision(target: float = 0.90, minimum_coverage: float = 0.60) -> bool:
    return any(point.precision >= target and point.coverage >= minimum_coverage for point in safety_curve())


def operating_point() -> dict[str, object]:
    curve = safety_curve()
    best = max(curve, key=lambda point: (point.f1, point.recall))
    precise = [point for point in curve if point.precision >= 0.90]
    safest_clear = min(
        (point for point in curve if point.autoclear_missed_safety == 0 and point.autoclear_coverage > 0),
        key=lambda point: point.minimum_confidence,
        default=None,
    )
    return {
        "target_precision": 0.90,
        "target_coverage": 0.60,
        "target_met": reaches_precision(),
        "best_f1_confidence": best.minimum_confidence,
        "best_precision": round(best.precision, 2),
        "best_recall": round(best.recall, 2),
        "best_f1": round(best.f1, 2),
        "precise_coverage": round(max((point.coverage for point in precise), default=0.0), 3),
        "clear_without_miss_coverage": round(safest_clear.autoclear_coverage, 3) if safest_clear else 0.0,
        "clear_without_miss_confidence": safest_clear.minimum_confidence if safest_clear else None,
    }
