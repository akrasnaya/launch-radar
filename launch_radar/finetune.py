"""LoRA examples for the DMV training split. The frozen test split stays out."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from launch_radar.annotation import text_id
from launch_radar.dmv_labels import _CAUSES, _SEVERITIES
from launch_radar.split import load_test_ids, train_rows

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "finetune"
ADAPTER_PATH = Path(__file__).resolve().parents[1] / "data" / "adapters" / "dmv-qwen"
MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
VALID_SEED = 20261114
VALID_FRACTION = 0.15
MAX_REPEATS = 4

INSTRUCTION = (
    "Label this autonomous-vehicle disengagement description. "
    "Reply with JSON only, with keys cause, severity, and safety. "
    f"cause is one of {', '.join(_CAUSES)}. "
    f"severity is one of {', '.join(_SEVERITIES)}. "
    "safety is true or false."
)


def dev_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split the training formulations into fit and validation. The test ids stay out."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in train_rows():
        grouped.setdefault((str(row["safety"]), str(row["severity"])), []).append(row)
    rng = random.Random(VALID_SEED)
    valid: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: text_id(str(row["text"])))
        count = round(len(rows) * VALID_FRACTION)
        if count == 0 and len(rows) >= 6:
            count = 1
        rng.shuffle(rows)
        valid.extend(rows[:count])
    valid_ids = {text_id(str(row["text"])) for row in valid}
    fit = [row for row in train_rows() if text_id(str(row["text"])) not in valid_ids]
    return fit, valid


def oversample(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["severity"]), []).append(row)
    largest = max(len(items) for items in buckets.values())
    expanded: list[dict[str, object]] = []
    for severity in sorted(buckets):
        items = buckets[severity]
        repeats = min(MAX_REPEATS, max(1, round(largest / len(items))))
        expanded.extend(items * repeats)
    return expanded


def example(row: dict[str, object]) -> dict[str, object]:
    safety = str(row["safety"]) == "true"
    answer = json.dumps(
        {"cause": row["cause"], "severity": row["severity"], "safety": safety},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": str(row["text"])},
            {"role": "assistant", "content": answer},
        ]
    }


def write_datasets(directory: Path = DATA_DIR) -> dict[str, int]:
    fit, valid = dev_rows()
    _reject_test_leak(fit + valid)
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "train.jsonl", [example(row) for row in oversample(fit)])
    _write_jsonl(directory / "valid.jsonl", [example(row) for row in valid])
    return {
        "fit": len(fit),
        "fit_oversampled": len(oversample(fit)),
        "valid": len(valid),
    }


def _reject_test_leak(rows: list[dict[str, object]]) -> None:
    leaked = {text_id(str(row["text"])) for row in rows} & load_test_ids()
    if leaked:
        raise RuntimeError(f"Тестовые формулировки попали в обучение: {len(leaked)}")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(adapter_path: Path = ADAPTER_PATH) -> None:
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    from launch_radar.annotation import _kappa
    from launch_radar.dmv_labels import _SEVERITIES
    from launch_radar.llm_classifier import LlmLabel, guard_safety, parse_label, score_labels
    from launch_radar.split import test_rows

    if not (adapter_path / "adapters.safetensors").is_file():
        raise SystemExit(f"Адаптер не найден: {adapter_path}")
    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    sampler = make_sampler(temp=0.0)
    rows = test_rows()
    predicted: dict[str, LlmLabel] = {}
    checklist: dict[str, LlmLabel] = {}
    failed = 0
    for index, row in enumerate(rows, start=1):
        text = str(row["text"])
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=80, sampler=sampler, verbose=False)
        try:
            predicted[text] = parse_label(str(raw), strict=False)
        except (ValueError, json.JSONDecodeError):
            failed += 1
            predicted[text] = LlmLabel("unknown", "medium", False, str(raw))
        checklist[text] = LlmLabel(
            str(row["cause"]),
            str(row["severity"]),
            guard_safety(text, False),
            "",
        )
        if index % 10 == 0 or index == len(rows):
            print(f"scored {index}/{len(rows)}", flush=True)
    scores = score_labels(predicted)
    rules = score_labels(checklist)
    pairs = [(str(row["severity"]), predicted[str(row["text"])].severity) for row in rows]
    cause_hits = sum(str(row["cause"]) == predicted[str(row["text"])].cause for row in rows)
    weights = [int(row["reports"]) for row in rows]
    weighted_pairs = [pair for pair, weight in zip(pairs, weights) for _ in range(weight)]
    print(f"adapter {adapter_path}")
    print(f"unparsed {failed}")
    print(
        "LoRA safety by reports "
        f"P {scores['reports_precision']:.2f} R {scores['reports_recall']:.2f} F1 {scores['reports_f1']:.2f}"
    )
    print(
        "LoRA safety by formulations "
        f"P {scores['forms_precision']:.2f} R {scores['forms_recall']:.2f} F1 {scores['forms_f1']:.2f}"
    )
    print(
        "Checklist safety by reports "
        f"P {rules['reports_precision']:.2f} R {rules['reports_recall']:.2f} F1 {rules['reports_f1']:.2f}"
    )
    print(f"LoRA severity quadratic kappa forms {_kappa(pairs, list(_SEVERITIES), True):.2f}")
    print(f"LoRA severity quadratic kappa reports {_kappa(weighted_pairs, list(_SEVERITIES), True):.2f}")
    print(f"LoRA cause accuracy forms {cause_hits / len(rows):.2f}")
    print("Ship only if severity kappa is at least 0.70 and safety F1 is not below the checklist.")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if command == "eval":
        path = Path(sys.argv[2]) if len(sys.argv) > 2 else ADAPTER_PATH
        evaluate(path)
        return
    counts = write_datasets()
    print(
        f"fit {counts['fit']} oversampled {counts['fit_oversampled']} valid {counts['valid']}"
    )
    print(DATA_DIR)


if __name__ == "__main__":
    main()
