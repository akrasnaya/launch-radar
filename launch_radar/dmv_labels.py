"""First-pass labels for DMV 2023 disengagement texts.

Severity is a target, not a source column. Each distinct description also gets a
cause and a safety flag. safety is the release block: a person, animal, contact,
leaving the roadway, or a stated failure to yield. A precautionary software
takeover is not safety, even when the text says takeover.

One annotator, scheme dmv-cause-severity-safety-v1. Agreement is not measured.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from launch_radar.dmv_corpus import load_texts

LABELS_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "dmv-2023.csv"
SCHEME = "dmv-cause-severity-safety-v1"

_CAUSES = (
    "prediction",
    "perception",
    "planning",
    "localization",
    "hardware",
    "platform",
    "controls",
    "conflict",
    "conditions",
    "unknown",
)
_SEVERITIES = ("low", "medium", "high", "critical")


@dataclass(frozen=True)
class DisengagementLabel:
    cause: str
    severity: str
    safety: bool
    rule: str


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def label_disengagement(text: str) -> DisengagementLabel:
    lowered = " ".join(text.lower().split())
    rule = _match(lowered)
    if rule.cause not in _CAUSES or rule.severity not in _SEVERITIES:
        raise ValueError(f"bad label {rule}")
    return rule


def _focus(text: str) -> str:
    """Keep the stated reason when a company repeats a perception preamble."""
    if " in this case, " in text:
        return text.rsplit(" in this case, ", 1)[1]
    marker = "ghost av test drivers are trained to disengage proactively. "
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def _match(text: str) -> DisengagementLabel:
    text = _focus(text)
    reason = text
    if _has(reason, r"pedestrian detection") and _has(reason, r"timeout|state error|module"):
        return DisengagementLabel("perception", "low", False, "detector-timeout")
    if _has(reason, r"false positive collision|collision prediction"):
        return DisengagementLabel("perception", "medium", False, "false-collision-prediction")
    if _has(reason, r"stuck after yielding|overly conservative|drive past pedestrians|braking approaching cyclist|tailgating cyclist"):
        return DisengagementLabel("conflict", "high", False, "conservative-near-vulnerable")
    if _has(reason, r"pedestrian|cyclist|scooter|animal in the roadway"):
        failed = _has(
            reason,
            r"fail|improper|inadequa|inappropriate|suddenly|too quickly|on the road|yield point|creeping into|made contact",
        ) and not _has(reason, r"give more space")
        if failed:
            return DisengagementLabel("conflict", "critical", True, "vulnerable-road-user")
        return DisengagementLabel("conflict", "high", False, "vulnerable-precaution")
    if _has(reason, r"scooter made contact|car crash|crash in front"):
        return DisengagementLabel("conflict", "critical", True, "contact-or-crash")
    if _has(reason, r"made contact") and not _has(reason, r"lane marker"):
        return DisengagementLabel("conflict", "critical", True, "contact-or-crash")
    if _has(reason, r"onto curb|onto the curb"):
        return DisengagementLabel("localization", "critical", True, "left-roadway")
    if _has(reason, r"wrong way"):
        return DisengagementLabel("conflict", "critical", True, "wrong-way")
    if _has(reason, r"driver-side door"):
        return DisengagementLabel("conflict", "critical", True, "door-opening")
    if _has(reason, r"safety critical"):
        return DisengagementLabel("conflict", "critical", True, "stated-safety-critical")
    if _has(reason, r"not giving enough space for cross traffic"):
        return DisengagementLabel("conflict", "high", True, "failed-yield")

    if _has(reason, r"failure to yield|failed to yield|fails to yield|inadequately yields|inadequate yield"):
        return DisengagementLabel("conflict", "high", True, "failed-yield")
    if _has(reason, r"did not yield|didn't yield|not yielding enough"):
        return DisengagementLabel("conflict", "high", True, "failed-yield")

    if _has(text, r"precautionary takeover|precautionary disengagement"):
        return DisengagementLabel("platform", "low", False, "precautionary-software")
    if _has(text, r"diagnosed hardware|hardware issue|hardware irregularity|hardware discrepancy"):
        return DisengagementLabel("hardware", "low", False, "hardware-handback")
    if _has(text, r"seat belt"):
        return DisengagementLabel("platform", "low", False, "seatbelt")
    if _has(text, r"accidental takeover|took over accidentally|disengaged accidentally"):
        return DisengagementLabel("unknown", "low", False, "accidental")
    if _has(text, r"manual lane change") and not _has(text, r"fail|improper|insufficient"):
        return DisengagementLabel("unknown", "low", False, "manual-convenience")

    if _has(text, r"incorrectly predicted|prediction discrepancy|behaviour prediction|behavior prediction|inaccurate oncoming traffic predictions|actor vehicle predictions"):
        return DisengagementLabel("prediction", "medium", False, "prediction")
    if _has(text, r"incorrect prediction"):
        return DisengagementLabel("prediction", "medium", False, "prediction")

    if _has(text, r"perception|perceived|object detection|lane detection|traffic light|stop line|field of view|camera is blocked|camera perception|green light not detected"):
        severity = "high" if _has(text, r"too close|cross traffic|not stopping|ensure stopping") else "medium"
        return DisengagementLabel("perception", severity, False, "perception")

    if _has(text, r"\bmaps?\b|localization|localizer|positioning error|delocalization|drive lines"):
        severity = "high" if _has(text, r"too close|curb|boundary") else "medium"
        return DisengagementLabel("localization", severity, False, "localization")

    if _has(text, r"motion plan|planning|planner|trajectory|path planning|lane-keeping|lane keeping|lane change|lane placement|lane marker|overrode a lane|lane-switching|within the lane|lateral movement"):
        severity = "high" if _has(text, r"too close|insufficient|incorrect location|contact|marker|clearance|curb") else "medium"
        return DisengagementLabel("planning", severity, False, "planning")

    if _has(text, r"sensor input delay|sensor field|camera|radar"):
        return DisengagementLabel("hardware", "medium", False, "sensor")

    if _has(text, r"sun glare|debris|\bodd\b|construction|double parked|sun glare") or (
        _has(text, r"weather") and not _has(text, r"non-inclement weather")
    ):
        return DisengagementLabel("conditions", "medium", False, "conditions")

    if _has(text, r"software issue|software discrepancy|software bug|software failure|compute infrastructure|controller interface|module timeout|module state|module incident|diagnostics|system load|buffer congestion|recording module|cycle time broken"):
        return DisengagementLabel("platform", "low", False, "platform-fault")

    if _has(text, r"too close|insufficient distance|lane boundary|road boundary"):
        return DisengagementLabel("planning", "high", False, "proximity")

    if _has(text, r"brak|accel|steering|swerv|speed profile|velocity"):
        severity = "high" if _has(text, r"too close|cut off|encroach") else "medium"
        return DisengagementLabel("controls", severity, False, "controls")

    if _has(text, r"cut-in|cut in|merging|encroach|tailgat|preceding vehicle|lead vehicle|oncoming"):
        return DisengagementLabel("conflict", "high", False, "vehicle-conflict")

    if _has(text, r"unwanted maneuver|undesirable under the circumstances|unexpected behavior|undesirable performance"):
        return DisengagementLabel("unknown", "medium", False, "unspecified")
    if _has(text, r"vision received|sensor fusion|fusion dynamic|fusion grid"):
        return DisengagementLabel("perception", "medium", False, "fusion")
    if _has(text, r"re-center vehicle in lane|reduce speed|green traffic signal|motion control system"):
        return DisengagementLabel("controls", "medium", False, "driver-correction")
    if _has(text, r"recklessly behaving road user"):
        return DisengagementLabel("conflict", "high", False, "other-road-user")
    if _has(text, r"too old"):
        return DisengagementLabel("perception", "medium", False, "stale-input")
    if _has(text, r"derbis|debris"):
        return DisengagementLabel("conditions", "medium", False, "debris")
    if _has(text, r"minor system error|giving back control"):
        return DisengagementLabel("platform", "low", False, "minor-software")
    if _has(text, r"precaution") and _has(text, r"parked"):
        return DisengagementLabel("planning", "medium", False, "parked-vehicle-precaution")

    return DisengagementLabel("unknown", "medium", False, "fallback")


def labeled_rows() -> list[dict[str, object]]:
    rows = []
    for item in load_texts():
        label = label_disengagement(item.record.text)
        rows.append(
            {
                "reports": item.reports,
                "manufacturer": item.manufacturer,
                "cause": label.cause,
                "severity": label.severity,
                "safety": "true" if label.safety else "false",
                "rule": label.rule,
                "scheme": SCHEME,
                "text": item.record.text,
            }
        )
    rows.sort(key=lambda row: (-int(row["reports"]), str(row["text"])))
    return rows


def write_labels(path: Path = LABELS_PATH) -> Path:
    rows = labeled_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reports", "manufacturer", "cause", "severity", "safety", "rule", "scheme", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    path = write_labels()
    rows = labeled_rows()
    reports = sum(int(row["reports"]) for row in rows)
    print(f"wrote {path}")
    print(f"formulations {len(rows)} reports {reports}")
    for field in ("cause", "severity", "safety"):
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row[field])] = counts.get(str(row[field]), 0) + int(row["reports"])
        print(field, dict(sorted(counts.items(), key=lambda item: -item[1])))


if __name__ == "__main__":
    main()
