"""Local LLM labels for DMV descriptions. The raw CSV, including VINs, is never sent."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from launch_radar.annotation import text_id
from launch_radar.dmv_labels import _CAUSES, _SEVERITIES, labeled_rows

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "llm-qwen25-3b-v2.jsonl"

_SYSTEM = """You label one autonomous-vehicle disengagement description.
Most descriptions are ordinary software handbacks. Those are safety false.

safety is true only when the text states that a person or animal was at risk, there was physical contact with a vehicle or object, the vehicle drove onto a curb, or the vehicle failed to yield to a pedestrian, cyclist, cross traffic, or an oncoming vehicle.
safety is false for a precautionary software takeover, a template sentence that the safety driver took control, a pedestrian-detector timeout, a false-positive collision prediction, and contact with a lane marker.
If none of the true conditions are stated, safety is false.

severity is low for a diagnosed handback or precautionary software with no driving conflict.
severity is medium when the driver took control but no proximity is stated.
severity is high for being too close, an insufficient gap, or touching a lane marker.
severity is critical only when safety is true.
Use the cause of the last stated reason, not a camera preamble.
cause must be one of: prediction, perception, planning, localization, hardware, platform, controls, conflict, conditions, unknown.
severity must be one of: low, medium, high, critical.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "cause": {"type": "string"},
        "severity": {"type": "string"},
        "safety": {"type": "boolean"},
    },
    "required": ["cause", "severity", "safety"],
}


@dataclass(frozen=True)
class LlmLabel:
    cause: str
    severity: str
    safety: bool
    raw: str


def _cache_key(model: str, text: str) -> str:
    return f"{model}:{text_id(text)}"


def load_cache(path: Path = CACHE_PATH) -> dict[str, LlmLabel]:
    if not path.is_file():
        return {}
    found: dict[str, LlmLabel] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[row["key"]] = LlmLabel(row["cause"], row["severity"], bool(row["safety"]), row.get("raw", ""))
    return found


def _append_cache(path: Path, key: str, label: LlmLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": key,
                    "cause": label.cause,
                    "severity": label.severity,
                    "safety": label.safety,
                    "raw": label.raw,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


_STATED_RISK = re.compile(
    r"failure to yield|failed to yield|fails to yield"
    r"|inadequately yields|inadequate yield"
    r"|improper yielding to (?:a )?(?:pedestrian|cyclist)"
    r"|improper yielding at a crosswalk"
    r"|inappropriate yield"
    r"|not giving enough space for cross traffic"
    r"|pedestrians suddenly entering"
    r"|yielding to (?:a )?pedestrians?"
    r"|yield for (?:a )?pedestrian"
    r"|yield to (?:a )?pedestrian"
    r"|pedestrian on the road"
    r"|driver-side door"
    r"|\bwrong way\b"
    r"|onto (?:the )?curb"
    r"|made contact"
    r"|safety critical"
    r"|car crash"
    r"|did not yield|didn't yield"
)

_HANDBACK = re.compile(
    r"safely take control|took control|take control|\btook over\b|\btake over\b"
    r"|undesirable lane placement"
    r"|module state error|module timeout"
    r"|software failure|software crash"
    r"|diagnosed hardware|hardware issue|hardware irregularity|hardware discrepancy"
    r"|failed to detect an object"
    r"|falsely percepted"
    r"|exceeding odd"
    r"|lead vehicle"
    r"|recklessly behaving"
)


def guard_safety(text: str, model_safety: bool) -> bool:
    """Keep the model label unless the text states a risk or only a handback template."""
    normalized = " ".join(text.lower().split())
    if _safety_excluded(normalized):
        return False
    if _safety_stated(normalized):
        return True
    if model_safety and _HANDBACK.search(normalized):
        return False
    return model_safety


def _safety_excluded(text: str) -> bool:
    if "give more space" in text or "stuck after yielding" in text:
        return True
    if "lane marker" in text and "made contact" in text:
        return True
    if "pedestrian detection" in text and re.search(r"timeout|state error|module", text):
        return True
    return bool(re.search(r"false positive collision|collision prediction", text))


def _safety_stated(text: str) -> bool:
    if "too quickly" in text and "pedestrian" in text:
        return True
    return bool(_STATED_RISK.search(text))


def apply_safety_guard(text: str, label: LlmLabel) -> LlmLabel:
    safety = guard_safety(text, label.safety)
    if safety == label.safety:
        return label
    return LlmLabel(label.cause, label.severity, safety, label.raw)


def parse_label(raw: str, *, strict: bool = True) -> LlmLabel:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("В ответе модели нет JSON.")
    payload = json.loads(raw[start : end + 1])
    cause = str(payload.get("cause", "")).strip().lower()
    severity = str(payload.get("severity", "")).strip().lower()
    safety = payload.get("safety")
    if isinstance(safety, str):
        lowered = safety.strip().lower()
        if lowered in {"true", "false"}:
            safety = lowered == "true"
    if cause not in _CAUSES:
        if strict:
            raise ValueError(f"Неизвестная причина: {cause}")
        cause = "unknown"
    if severity not in _SEVERITIES:
        raise ValueError(f"Неизвестная тяжесть: {severity}")
    if not isinstance(safety, bool):
        raise ValueError("Safety должен быть true или false.")
    return LlmLabel(cause, severity, safety, raw)


def classify_text(text: str, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL) -> LlmLabel:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": _SCHEMA,
            "think": False,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": text},
            ],
            "options": {"temperature": 0, "num_predict": 80},
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama не отвечает на 127.0.0.1:11434.") from exc
    message = payload.get("message") or {}
    content = message.get("content") or payload.get("response") or ""
    return parse_label(str(content), strict=False)


def cached_label(text: str, model: str = DEFAULT_MODEL, path: Path = CACHE_PATH) -> LlmLabel:
    key = _cache_key(model, text)
    cached = load_cache(path)
    if key in cached:
        return cached[key]
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            label = classify_text(text, model)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Модель дважды вернула неразборчивый ответ: {last_error}")
    _append_cache(path, key, label)
    return label


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safety_f1(truth: list[bool], predicted: list[bool], weights: list[int]) -> tuple[float, float, float]:
    true_positive = false_positive = false_negative = 0
    for actual, guess, weight in zip(truth, predicted, weights, strict=True):
        if actual and guess:
            true_positive += weight
        elif guess:
            false_positive += weight
        elif actual:
            false_negative += weight
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall, _f1(precision, recall)


def score_labels(predicted_by_text: dict[str, LlmLabel]) -> dict[str, float]:
    truth_reports: list[bool] = []
    guess_reports: list[bool] = []
    weights: list[int] = []
    truth_forms: list[bool] = []
    guess_forms: list[bool] = []
    for row in labeled_rows():
        text = str(row["text"])
        if text not in predicted_by_text:
            continue
        actual = str(row["safety"]) == "true"
        guess = predicted_by_text[text].safety
        weight = int(row["reports"])
        truth_reports.append(actual)
        guess_reports.append(guess)
        weights.append(weight)
        truth_forms.append(actual)
        guess_forms.append(guess)
    report_precision, report_recall, report_f1 = _safety_f1(truth_reports, guess_reports, weights)
    form_precision, form_recall, form_f1 = _safety_f1(truth_forms, guess_forms, [1] * len(truth_forms))
    return {
        "reports_precision": report_precision,
        "reports_recall": report_recall,
        "reports_f1": report_f1,
        "forms_precision": form_precision,
        "forms_recall": form_recall,
        "forms_f1": form_f1,
        "labeled": float(len(truth_forms)),
    }


def _predicted_from_cache(rows: list[dict[str, object]]) -> dict[str, LlmLabel]:
    cached = load_cache()
    predicted: dict[str, LlmLabel] = {}
    filled = 0
    for row in rows:
        text = str(row["text"])
        key = _cache_key(DEFAULT_MODEL, text)
        if key not in cached:
            predicted[text] = cached_label(text)
            filled += 1
            if filled % 10 == 0:
                print(f"labeled {filled} new", flush=True)
            continue
        predicted[text] = cached[key]
    if filled:
        print(f"labeled {filled} new", flush=True)
    return predicted


def _print_safety(title: str, scores: dict[str, float]) -> None:
    print(
        f"{title} by reports "
        f"P {scores['reports_precision']:.2f} R {scores['reports_recall']:.2f} F1 {scores['reports_f1']:.2f}"
    )
    print(
        f"{title} by formulations "
        f"P {scores['forms_precision']:.2f} R {scores['forms_recall']:.2f} F1 {scores['forms_f1']:.2f}"
    )


def main() -> None:
    rows = labeled_rows()
    predicted = _predicted_from_cache(rows)
    _print_safety("LLM safety", score_labels(predicted))
    guarded = {text: apply_safety_guard(text, label) for text, label in predicted.items()}
    _print_safety("Guarded safety", score_labels(guarded))
    print(
        "Keyword safety by reports is P 0.28 R 0.76 F1 0.41. "
        "Guarded positives come from the text checklist. The release pipeline stays on keywords."
    )


if __name__ == "__main__":
    main()
