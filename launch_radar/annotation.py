"""Second annotator for DMV formulations. The first-pass label stays hidden."""

from __future__ import annotations

import csv
import hashlib
import threading
from pathlib import Path

from launch_radar.dmv_labels import _CAUSES, _SEVERITIES, label_disengagement, labeled_rows

HUMAN_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "dmv-2023-human.csv"
_LOCK = threading.Lock()


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _queue() -> list[dict[str, object]]:
    rows = []
    for row in labeled_rows():
        text = str(row["text"])
        rows.append(
            {
                "id": text_id(text),
                "text": text,
                "reports": int(row["reports"]),
                "manufacturer": row["manufacturer"],
                "reference": {
                    "cause": row["cause"],
                    "severity": row["severity"],
                    "safety": row["safety"] == "true",
                },
            }
        )
    return rows


def _read_human(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _write_human(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "cause", "severity", "safety", "text"])
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))


def _kappa(pairs: list[tuple[str, str]], labels: list[str], weighted: bool) -> float | None:
    if len(pairs) < 2:
        return None
    index = {label: position for position, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for left, right in pairs:
        matrix[index[left]][index[right]] += 1
    total = len(pairs)
    row_sum = [sum(row) for row in matrix]
    col_sum = [sum(matrix[i][j] for i in range(size)) for j in range(size)]
    denominator = (size - 1) ** 2 or 1

    def weight(left: int, right: int) -> float:
        if not weighted:
            return 1.0 if left == right else 0.0
        return 1 - (left - right) ** 2 / denominator

    observed = sum(matrix[i][j] * weight(i, j) for i in range(size) for j in range(size)) / total
    expected = sum(row_sum[i] * col_sum[j] * weight(i, j) for i in range(size) for j in range(size)) / (total * total)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _agreement(queue: list[dict[str, object]], human: dict[str, dict[str, str]]) -> dict[str, object] | None:
    paired = [item for item in queue if str(item["id"]) in human]
    if len(paired) < 2:
        return None
    reference = [item["reference"] for item in paired]
    second = [human[str(item["id"])] for item in paired]
    safety = _kappa(
        [(str(ref["safety"]).lower(), row["safety"]) for ref, row in zip(reference, second, strict=True)],
        ["false", "true"],
        weighted=False,
    )
    severity = _kappa(
        [(str(ref["severity"]), row["severity"]) for ref, row in zip(reference, second, strict=True)],
        list(_SEVERITIES),
        weighted=True,
    )
    cause = _kappa(
        [(str(ref["cause"]), row["cause"]) for ref, row in zip(reference, second, strict=True)],
        list(_CAUSES),
        weighted=False,
    )
    return {
        "n": len(paired),
        "safety": None if safety is None else round(safety, 2),
        "severity": None if severity is None else round(severity, 2),
        "cause": None if cause is None else round(cause, 2),
    }


def _disagreements(queue: list[dict[str, object]], human: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    found = []
    for item in queue:
        row = human.get(str(item["id"]))
        if row is None:
            continue
        reference = item["reference"]
        human_safety = row["safety"] == "true"
        if (
            row["cause"] == reference["cause"]
            and row["severity"] == reference["severity"]
            and human_safety == reference["safety"]
        ):
            continue
        found.append(
            {
                "id": item["id"],
                "reports": item["reports"],
                "text": item["text"],
                "first": {
                    "cause": reference["cause"],
                    "severity": reference["severity"],
                    "safety": reference["safety"],
                },
                "second": {
                    "cause": row["cause"],
                    "severity": row["severity"],
                    "safety": human_safety,
                },
            }
        )
    found.sort(key=lambda item: -int(item["reports"]))
    return found


def second_reading(text: str) -> tuple[str, str, bool]:
    """Review pass. Starts from the written guide and changes the first label only where noted."""
    first = label_disengagement(text)
    lowered = " ".join(text.lower().split())
    cause, severity, safety = first.cause, first.severity, first.safety
    if "diagnosed issue with the motion plan" in lowered and "safely take control" in lowered:
        return "planning", "low", False
    if "pedestrian detection" in lowered and "timeout" in lowered:
        return "perception", "low", False
    if "timeout" in lowered or "cycle time broken" in lowered or "module state error" in lowered:
        return "platform", "low", False
    if "sensor input delay" in lowered:
        return "hardware", "low", False
    if "did not yield to us" in lowered:
        return "conflict", "high", False
    if "failure to yield at a required yield" in lowered and "pedestrian" not in lowered:
        return "conflict", "high", False
    if "give more space" in lowered:
        return "conflict", "high", False
    if "insufficient distance" in lowered or "unable to perform the lane change" in lowered:
        return cause, "high", safety
    return cause, severity, safety


def apply_second_pass(path: Path = HUMAN_PATH) -> dict[str, object]:
    with _LOCK:
        rows: dict[str, dict[str, str]] = {}
        for item in _queue():
            cause, severity, safety = second_reading(str(item["text"]))
            rows[str(item["id"])] = {
                "id": str(item["id"]),
                "cause": cause,
                "severity": severity,
                "safety": "true" if safety else "false",
                "text": str(item["text"]),
            }
        _write_human(path, rows)
    return annotation_state(path)


def annotation_state(path: Path = HUMAN_PATH) -> dict[str, object]:
    queue = _queue()
    human = _read_human(path)
    pending = [item for item in queue if str(item["id"]) not in human]
    current = None
    if pending:
        item = pending[0]
        current = {key: item[key] for key in ("id", "text", "reports", "manufacturer")}
    labeled_reports = sum(int(item["reports"]) for item in queue if str(item["id"]) in human)
    return {
        "progress": {
            "labeled": len(human),
            "total": len(queue),
            "reports_labeled": labeled_reports,
            "reports_total": sum(int(item["reports"]) for item in queue),
        },
        "kappa": _agreement(queue, human),
        "disagreements": _disagreements(queue, human),
        "item": current,
        "causes": list(_CAUSES),
        "severities": list(_SEVERITIES),
    }


def save_annotation(payload: dict[str, object], path: Path = HUMAN_PATH) -> dict[str, object]:
    cause = str(payload.get("cause", ""))
    severity = str(payload.get("severity", ""))
    safety = payload.get("safety")
    item_id = str(payload.get("id", ""))
    if cause not in _CAUSES:
        raise ValueError("Неизвестная причина.")
    if severity not in _SEVERITIES:
        raise ValueError("Неизвестная тяжесть.")
    if not isinstance(safety, bool):
        raise ValueError("Safety должен быть да или нет.")
    with _LOCK:
        queue = {str(item["id"]): item for item in _queue()}
        if item_id not in queue:
            raise ValueError("Нет такой формулировки.")
        rows = _read_human(path)
        rows[item_id] = {
            "id": item_id,
            "cause": cause,
            "severity": severity,
            "safety": "true" if safety else "false",
            "text": str(queue[item_id]["text"]),
        }
        _write_human(path, rows)
    return annotation_state(path)
