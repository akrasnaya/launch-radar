"""California DMV 2023 disengagement reports. Not a pilot/holdout release."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from launch_radar.baseline import classify
from launch_radar.clustering import cluster_records
from launch_radar.models import FeedbackRecord

SOURCE_URL = "https://www.dmv.ca.gov/portal/file/2023-autonomous-vehicle-disengagement-reports-csv/"
SOURCE_NAME = "California DMV Autonomous Vehicle Disengagement Reports, 2023"


@dataclass(frozen=True)
class DmvText:
    record: FeedbackRecord
    reports: int
    manufacturer: str


def corpus_path() -> Path:
    name = "dmv-2023-disengagements.csv"
    candidates = (
        Path("data/raw") / name,
        Path(__file__).resolve().parents[1] / "data" / "raw" / name,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Нет файла data/raw/dmv-2023-disengagements.csv. "
        f"Скачайте CSV California DMV: {SOURCE_URL}"
    )


def load_texts(path: Path | None = None) -> list[DmvText]:
    source = path if path is not None else corpus_path()
    counts: Counter[tuple[str, str]] = Counter()
    with source.open(newline="", encoding="latin-1") as handle:
        for raw in csv.DictReader(handle):
            fields = {_header(key): value for key, value in raw.items() if key}
            text = _clean(fields.get("DESCRIPTION OF FACTS CAUSING DISENGAGEMENT", ""))
            if not text:
                continue
            manufacturer = _clean(fields.get("MANUFACTURER", "")) or "не указан"
            counts[(text, manufacturer)] += 1

    grouped: dict[str, tuple[int, Counter[str]]] = {}
    for (text, manufacturer), count in counts.items():
        total, makers = grouped.setdefault(text, (0, Counter()))
        grouped[text] = (total + count, makers)
        makers[manufacturer] += count

    texts: list[DmvText] = []
    for index, (text, (total, makers)) in enumerate(grouped.items(), start=1):
        manufacturer = makers.most_common(1)[0][0]
        if len(makers) > 1:
            manufacturer = f"{manufacturer} и ещё {len(makers) - 1}"
        texts.append(
            DmvText(
                record=FeedbackRecord(
                    record_id=f"dmv-{index:04d}",
                    text=text,
                    source="dmv-ca-2023",
                    cohort="disengagement",
                    build="2023",
                    synthetic=False,
                ),
                reports=total,
                manufacturer=manufacturer,
            )
        )
    return texts


def summarize(path: Path | None = None, limit: int = 12) -> dict[str, object]:
    texts = load_texts(path)
    records = [item.record for item in texts]
    by_id = {item.record.record_id: item for item in texts}
    classifications = {record.record_id: classify(record) for record in records}
    cluster_inputs = [
        FeedbackRecord(
            record.record_id,
            cluster_view(record.text),
            record.source,
            record.cohort,
            record.build,
            record.synthetic,
        )
        for record in records
    ]
    clusters = cluster_records(cluster_inputs, classifications)

    reports = sum(item.reports for item in texts)
    topic_reports: Counter[str] = Counter()
    abstain_reports = 0
    safety_reports = 0
    no_topic_reports = 0
    for item in texts:
        item_class = classifications[item.record.record_id]
        if item_class.abstain:
            abstain_reports += item.reports
        if item_class.safety_flag:
            safety_reports += item.reports
        if not item_class.topics:
            no_topic_reports += item.reports
        for topic in item_class.topics:
            topic_reports[topic] += item.reports

    ranked = []
    for cluster in clusters:
        members = [by_id[record_id] for record_id in cluster.record_ids]
        dominant = max(members, key=lambda item: item.reports)
        dominant_class = classifications[dominant.record.record_id]
        member_safety = sum(
            item.reports for item in members if classifications[item.record.record_id].safety_flag
        )
        ranked.append(
            {
                "cluster_id": cluster.cluster_id,
                "reports": sum(item.reports for item in members),
                "forms": len(members),
                "topic": ", ".join(dominant_class.topics) if dominant_class.topics else "unknown",
                "severity": dominant_class.severity,
                "safety": dominant_class.safety_flag,
                "safety_reports": member_safety,
                "abstain": dominant_class.abstain,
                "manufacturer": dominant.manufacturer,
                "text": dominant.record.text,
                "matched_terms": list(dominant_class.matched_terms),
            }
        )
    ranked.sort(key=lambda item: (-int(item["reports"]), str(item["cluster_id"])))

    return {
        "source": SOURCE_NAME,
        "url": SOURCE_URL,
        "reports": reports,
        "distinct_texts": len(texts),
        "abstain_reports": abstain_reports,
        "safety_reports": safety_reports,
        "no_topic_reports": no_topic_reports,
        "topics": [{"topic": topic, "reports": count} for topic, count in topic_reports.most_common()],
        "clusters": ranked[:limit],
    }


def render_summary(path: Path | None = None) -> str:
    payload = summarize(path)
    lines = [
        payload["source"],
        "Это не сравнение pilot/holdout. Каждая строка — формулировка отчёта о disengagement.",
        "",
        (
            f"Отчётов: {payload['reports']}. Разных формулировок: {payload['distinct_texts']}. "
            f"Без уверенной темы: {payload['abstain_reports']}. Safety-флаг: {payload['safety_reports']}."
        ),
        "",
        f"{'Отчётов':>8} {'Форм':>5} {'Тема':<20} {'Sev':<10}  Текст",
        "-" * 88,
    ]
    for cluster in payload["clusters"]:
        preview = str(cluster["text"]).replace("\n", " ")
        if len(preview) > 70:
            preview = preview[:67] + "..."
        topic = "не уверена" if cluster["abstain"] else str(cluster["topic"])
        lines.append(
            f"{cluster['reports']:8} {cluster['forms']:5} {topic:<20} "
            f"{str(cluster['severity']):<10}  {preview}"
        )
    return "\n".join(lines)


def _header(name: str) -> str:
    return " ".join(name.replace("\xa0", " ").split()).upper()


def _clean(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


_TAILS = (
    "the safety driver was able to safely take control of the vehicle.",
    "conditions: non-inclement weather, dry roads, no other factors involved.",
)


def cluster_view(text: str) -> str:
    """Drop the repeated closing sentence before Jaccard, not before classification."""
    cleaned = " ".join(text.split())
    lowered = cleaned.lower()
    for tail in _TAILS:
        if lowered.endswith(tail):
            cleaned = cleaned[: -len(tail)].rstrip(" .")
            lowered = cleaned.lower()
    return cleaned or text


_CACHE: dict[str, object] | None = None


def cached_summary() -> dict[str, object]:
    global _CACHE
    if _CACHE is None:
        _CACHE = summarize()
    return _CACHE
