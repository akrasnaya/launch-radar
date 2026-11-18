"""Frozen formulation holdout for DMV labels.

The split is by distinct description, not by report copy. Texts that repeat
50 times or more stay in train: twenty of them cover most reports, and one of
them would dominate a report-weighted test. Inside a stratum, formulations
that repeat at least twice keep a share of the test slots.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from launch_radar.annotation import text_id
from launch_radar.dmv_labels import labeled_rows

SPLIT_PATH = Path(__file__).resolve().parents[1] / "data" / "splits" / "dmv-2023-test.csv"
SEED = 20261113
TEST_FRACTION = 0.2
TRAIN_ONLY_REPORTS = 50


def build_test_ids() -> list[str]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in labeled_rows():
        grouped.setdefault((str(row["safety"]), str(row["severity"])), []).append(row)
    rng = random.Random(SEED)
    chosen: list[str] = []
    for key in sorted(grouped):
        rows = grouped[key]
        eligible = [row for row in rows if int(row["reports"]) < TRAIN_ONLY_REPORTS]
        count = min(round(len(rows) * TEST_FRACTION), len(eligible))
        chosen.extend(text_id(str(row["text"])) for row in _sample(eligible, count, rng))
    return sorted(chosen)


def _sample(rows: list[dict[str, object]], count: int, rng: random.Random) -> list[dict[str, object]]:
    if count <= 0:
        return []
    repeated = sorted((row for row in rows if int(row["reports"]) >= 2), key=lambda row: text_id(str(row["text"])))
    single = sorted((row for row in rows if int(row["reports"]) < 2), key=lambda row: text_id(str(row["text"])))
    rng.shuffle(repeated)
    rng.shuffle(single)
    repeated_count = round(len(repeated) / len(rows) * count)
    if repeated and count >= 3 and repeated_count == 0:
        repeated_count = 1
    repeated_count = min(repeated_count, len(repeated), count)
    single_count = min(count - repeated_count, len(single))
    return repeated[:repeated_count] + single[:single_count]


def write_split(path: Path = SPLIT_PATH) -> None:
    ids = set(build_test_ids())
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_split_row(row) for row in labeled_rows() if text_id(str(row["text"])) in ids]
    rows.sort(key=lambda row: row["id"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "reports", "cause", "severity", "safety"])
        writer.writeheader()
        writer.writerows(rows)


def load_test_ids(path: Path = SPLIT_PATH) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"] for row in csv.DictReader(handle)}


def test_rows(path: Path = SPLIT_PATH) -> list[dict[str, object]]:
    ids = load_test_ids(path)
    return [row for row in labeled_rows() if text_id(str(row["text"])) in ids]


def train_rows(path: Path = SPLIT_PATH) -> list[dict[str, object]]:
    ids = load_test_ids(path)
    return [row for row in labeled_rows() if text_id(str(row["text"])) not in ids]


def _split_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "id": text_id(str(row["text"])),
        "reports": row["reports"],
        "cause": row["cause"],
        "severity": row["severity"],
        "safety": row["safety"],
    }


def main() -> None:
    write_split()
    held = test_rows()
    train = train_rows()
    held_reports = sum(int(row["reports"]) for row in held)
    safety = sum(1 for row in held if row["safety"] == "true")
    print(f"test formulations {len(held)} reports {held_reports} safety texts {safety}")
    print(f"train formulations {len(train)} reports {sum(int(row['reports']) for row in train)}")
    print(SPLIT_PATH)


if __name__ == "__main__":
    main()
