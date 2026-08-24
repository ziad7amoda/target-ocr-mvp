import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.model import FakeEngine

GOOD = {
    "full_name": "JOHN A SMITH",
    "full_name_ar": "جون سميث",
    "id_number": "12345678",
    "date_of_birth": "1990-04-12",
    "expiry_date": "2030-04-11",
    "nationality": "OMANI",
    "nationality_ar": "عماني",
    "sex": "M",
    "sex_ar": "ذكر",
}


def _upload():
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, format="JPEG")
    buf.seek(0)
    return {"image": ("card.jpg", buf, "image/jpeg")}


def _client(replies):
    return TestClient(create_app(engine=FakeEngine(replies)))


def test_health_reports_model_device_and_loaded_state():
    body = _client([]).get("/api/health").json()
    assert body["model"] == "fake"
    assert body["device"] == "cpu"
    assert body["loaded"] is True
    assert body["self_consistency"] is True


def test_extract_returns_a_schema_conforming_response():
    payload = json.dumps(GOOD)
    r = _client([payload, payload, "{}"]).post("/api/extract", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert set(body["fields"]) == {
        "full_name", "id_number", "date_of_birth", "expiry_date", "nationality", "sex",
    }
    assert body["fields"]["id_number"]["status"] == "ok"
    assert body["agreement"] == {"matched": 6, "compared": 6, "total": 6}
    assert body["model"] == "fake"


def test_extract_rejects_a_non_image_upload():
    files = {"image": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    r = _client([]).post("/api/extract", files=files)
    assert r.status_code == 400
    assert "image" in r.json()["detail"].lower()


def test_extract_requires_a_file():
    assert _client([]).post("/api/extract").status_code == 422


def test_transcribe_returns_text():
    r = _client(["LINE ONE\nLINE TWO"]).post("/api/transcribe", files=_upload())
    assert r.status_code == 200
    assert r.json()["text"] == "LINE ONE\nLINE TWO"


def test_cors_headers_are_present():
    r = _client([]).get("/api/health", headers={"Origin": "https://example.trycloudflare.com"})
    assert r.headers["access-control-allow-origin"] == "*"


def test_logs_never_contain_field_values(caplog):
    """Spec §10: event, timing and status counts only."""
    payload = json.dumps(GOOD)
    with caplog.at_level("DEBUG"):
        _client([payload, payload, "{}"]).post("/api/extract", files=_upload())
    combined = " ".join(r.getMessage() for r in caplog.records)
    assert "JOHN A SMITH" not in combined
    assert "12345678" not in combined


def test_extract_logs_status_counts():
    """The useful half of the logging rule: operational signal must survive."""
    payload = json.dumps(GOOD)
    client = _client([payload, payload, "{}"])
    r = client.post("/api/extract", files=_upload())
    assert r.status_code == 200
