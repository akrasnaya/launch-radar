import json
import threading
import unittest
import urllib.error
import urllib.request

from launch_radar.pipeline import run
from launch_radar.review import Review, pending_safety_ids, validate_signature
from launch_radar.sample_data import sample_records
from launch_radar.server import Handler, make_server


class ReviewRuleTest(unittest.TestCase):
    def test_expand_waits_for_every_safety_record(self) -> None:
        records = sample_records()
        classifications, _, _ = run(records)
        reviews: dict[str, Review] = {}

        self.assertEqual(
            pending_safety_ids(records, classifications, reviews),
            ("fb-001", "fb-002", "fb-003", "fb-011"),
        )
        with self.assertRaisesRegex(ValueError, "Expand"):
            validate_signature("Expand", "Аня", records, classifications, reviews)

        validate_signature("Hold", "Аня", records, classifications, reviews)
        for record_id in ("fb-001", "fb-002", "fb-003"):
            reviews[record_id] = Review(record_id, "accept")
        with self.assertRaisesRegex(ValueError, "Expand"):
            validate_signature("Expand", "Аня", records, classifications, reviews)

        reviews["fb-011"] = Review("fb-011", "correct", "тема brake, не ux")
        self.assertEqual(pending_safety_ids(records, classifications, reviews), ())
        validate_signature("Expand", "Аня", records, classifications, reviews)

    def test_signature_needs_a_signer(self) -> None:
        records = sample_records()
        classifications, _, _ = run(records)
        with self.assertRaisesRegex(ValueError, "подписанта"):
            validate_signature("Hold", "  ", records, classifications, {})


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Handler.state.reset()
        cls.httpd = make_server("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.httpd.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=3)

    def setUp(self) -> None:
        status, _ = self.request("POST", "/api/reset", {})
        self.assertEqual(status, 200)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode())
            finally:
                exc.close()

    def test_page_and_live_pipeline(self) -> None:
        with urllib.request.urlopen(self.base + "/") as response:
            page = response.read().decode()
        self.assertIn("Карточка риска", page)
        status, payload = self.request("GET", "/api/readout")
        self.assertEqual(status, 200)
        self.assertEqual(payload["gate"]["recommendation"], "HOLD")
        ux = next(card for card in payload["cards"] if card["cluster_id"] == "ISSUE-005")
        self.assertEqual(ux["rate_ratio"], 1.9)
        self.assertEqual(ux["recommendation"], "BACKLOG")
        self.assertEqual(payload["gate"]["pending_safety"], ["fb-001", "fb-002", "fb-003", "fb-011"])

    def test_expand_is_rejected_until_safety_is_reviewed(self) -> None:
        status, payload = self.request(
            "POST",
            "/api/gate",
            {"decision": "Expand", "signer": "Аня", "note": ""},
        )
        self.assertEqual(status, 400)
        self.assertIn("Expand", payload["error"])

        for record_id in ("fb-001", "fb-002", "fb-003", "fb-011"):
            status, _ = self.request(
                "POST",
                "/api/reviews",
                {"record_id": record_id, "action": "accept", "note": ""},
            )
            self.assertEqual(status, 200)

        status, payload = self.request(
            "POST",
            "/api/gate",
            {"decision": "Expand", "signer": "Аня", "note": "safety разобран"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["gate"]["decision"], "Expand")
        self.assertEqual(payload["gate"]["recommendation"], "HOLD")
        self.assertTrue(payload["gate"]["expand_allowed"])


if __name__ == "__main__":
    unittest.main()
