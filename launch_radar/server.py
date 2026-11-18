"""Local interface over the sample pipeline. Session state lives in this process."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from launch_radar.clustering import THRESHOLD, text_overlap
from launch_radar.pipeline import run
from launch_radar.review import Review, advisory_recommendation, parse_review, pending_safety_ids, validate_signature
from launch_radar.risk import HOLDOUT_DEVICES, PILOT_DEVICES
from launch_radar.sample_data import sample_records

INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
_MAX_BODY = 16_384


@dataclass
class AppState:
    reviews: dict[str, Review] = field(default_factory=dict)
    decision: str | None = None
    signer: str = ""
    note: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def readout(self) -> dict[str, Any]:
        with self.lock:
            return build_readout(self)

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            records = sample_records()
            review = parse_review(payload, {record.record_id for record in records})
            self.reviews[review.record_id] = review
            return build_readout(self)

    def save_signature(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            records = sample_records()
            classifications, _, _ = run(records)
            decision = str(payload.get("decision", ""))
            signer = str(payload.get("signer", ""))
            note = str(payload.get("note", ""))
            validate_signature(decision, signer, records, classifications, self.reviews)
            self.decision = decision
            self.signer = signer.strip()
            self.note = note.strip()
            return build_readout(self)

    def clear_signature(self) -> dict[str, Any]:
        with self.lock:
            self.decision = None
            return build_readout(self)

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.reviews.clear()
            self.decision = None
            self.signer = ""
            self.note = ""
            return build_readout(self)


def build_readout(state: AppState) -> dict[str, Any]:
    records = sample_records()
    classifications, clusters, cards = run(records)
    by_id = {record.record_id: record for record in records}
    cluster_of = {
        record_id: cluster.cluster_id
        for cluster in clusters
        for record_id in cluster.record_ids
    }
    card_of = {card.cluster_id: card for card in cards}
    pending = pending_safety_ids(records, classifications, state.reviews)
    items = []
    for record in records:
        classification = classifications[record.record_id]
        cluster_id = cluster_of[record.record_id]
        review = state.reviews.get(record.record_id)
        items.append(
            {
                "record_id": record.record_id,
                "text": record.text,
                "cohort": record.cohort,
                "build": record.build,
                "synthetic": record.synthetic,
                "topics": list(classification.topics),
                "sentiment": classification.topic_sentiment,
                "severity": classification.severity,
                "safety_flag": classification.safety_flag,
                "confidence": classification.confidence,
                "abstain": classification.abstain,
                "matched_terms": list(classification.matched_terms),
                "cluster_id": cluster_id,
                "recommendation": card_of[cluster_id].recommendation,
                "review": None if review is None else {"action": review.action, "note": review.note},
            }
        )
    return {
        "meta": {
            "pilot_devices": PILOT_DEVICES,
            "holdout_devices": HOLDOUT_DEVICES,
            "pilot_build": _build(records, "pilot"),
            "holdout_build": _build(records, "holdout"),
            "cluster_threshold": THRESHOLD,
            "synthetic": all(record.synthetic for record in records),
            "advisory": advisory_recommendation(cards),
        },
        "items": items,
        "clusters": [_cluster_payload(cluster, by_id) for cluster in clusters],
        "cards": [asdict(card) for card in cards],
        "gate": {
            "recommendation": advisory_recommendation(cards),
            "decision": state.decision,
            "signer": state.signer,
            "note": state.note,
            "expand_allowed": not pending,
            "pending_safety": list(pending),
        },
    }


def _build(records: list[Any], cohort: str) -> str:
    builds = sorted({record.build for record in records if record.cohort == cohort})
    return ", ".join(builds)


def _cluster_payload(cluster: Any, by_id: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    record_ids = list(cluster.record_ids)
    if len(record_ids) > 1:
        for index, left in enumerate(record_ids):
            for right in record_ids[index + 1 :]:
                score, shared = text_overlap(by_id[left].text, by_id[right].text)
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "score": round(score, 2),
                        "shared": list(shared),
                    }
                )
    return {
        "cluster_id": cluster.cluster_id,
        "label": cluster.label,
        "record_ids": record_ids,
        "severity": cluster.severity,
        "safety_flag": cluster.safety_flag,
        "confidence": cluster.confidence,
        "pairs": pairs,
    }


class Handler(BaseHTTPRequestHandler):
    state = AppState()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._bytes(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/readout":
            self._json(200, self.state.readout())
            return
        if path == "/api/annotate":
            from launch_radar.annotation import annotation_state

            try:
                self._json(200, annotation_state())
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
            return
        if path == "/api/dmv":
            from launch_radar.calibration import operating_point
            from launch_radar.dmv_corpus import cached_summary

            try:
                payload = dict(cached_summary())
                payload["operating_point"] = operating_point()
                self._json(200, payload)
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
            return
        self._json(404, {"error": "Нет такого адреса."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/reviews":
                self._json(200, self.state.save_review(self._read_json()))
                return
            if path == "/api/gate":
                self._json(200, self.state.save_signature(self._read_json()))
                return
            if path == "/api/gate/clear":
                self._json(200, self.state.clear_signature())
                return
            if path == "/api/reset":
                self._json(200, self.state.reset())
                return
            if path == "/api/annotate":
                from launch_radar.annotation import save_annotation

                self._json(200, save_annotation(self._read_json()))
                return
            self._json(404, {"error": "Нет такого адреса."})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except json.JSONDecodeError:
            self._json(400, {"error": "Нужен JSON."})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[launch-radar] {self.address_string()} {format % args}", flush=True)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > _MAX_BODY:
            raise ValueError("Слишком большой запрос.")
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Нужен JSON-объект.")
        return payload

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class LaunchRadarServer(ThreadingHTTPServer):
    allow_reuse_address = True


def make_server(host: str = "127.0.0.1", port: int = 8765) -> LaunchRadarServer:
    return LaunchRadarServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    Handler.state.reset()
    server = make_server(host, port)
    print(f"LaunchRadar http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.", flush=True)
    finally:
        server.server_close()
