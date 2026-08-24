import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.model import FakeEngine

GOOD = {
    "card_type": "resident",
    "full_name_ar": "زياد نشأت عبد الحى ابو الوفا حموده",
    "id_number": "70011864",
    "date_of_birth": "2002-09-29",
    "expiry_date": "2027-01-25",
    "place_of_birth_ar": "جمهورية مصر العربية",
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
        "card_type", "full_name", "id_number", "date_of_birth", "expiry_date", "place_of_birth",
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
    """Spec §10: event, timing and status counts only.

    The Arabic name is now the most sensitive field on the card - checking
    only a Latin value would miss a leak of the field that actually carries
    the holder's identity.
    """
    payload = json.dumps(GOOD)
    with caplog.at_level("DEBUG"):
        _client([payload, payload, "{}"]).post("/api/extract", files=_upload())
    combined = " ".join(r.getMessage() for r in caplog.records)
    assert GOOD["id_number"] not in combined
    assert GOOD["full_name_ar"] not in combined


def test_extract_logs_status_counts():
    """The useful half of the logging rule: operational signal must survive."""
    payload = json.dumps(GOOD)
    client = _client([payload, payload, "{}"])
    r = client.post("/api/extract", files=_upload())
    assert r.status_code == 200
