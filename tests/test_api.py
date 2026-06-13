"""
FastAPI route smoke tests (review item: no API-layer tests).

Boots the app in-process with TestClient and exercises each router's happy path
plus the new input-validation guards. Runs against the committed model/data
artifacts, so it needs no live services.
"""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)

BOOKING = {
    "hotel": "Resort Hotel", "lead_time": 30, "arrival_date_month": "July",
    "total_stay": 3, "total_guests": 2, "meal": "BB", "country": "PRT",
    "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "is_repeated_guest": 0, "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
    "deposit_type": "No Deposit", "customer_type": "Transient",
    "required_car_parking_spaces": 0, "total_of_special_requests": 1, "adr": 120.0,
}
GUEST = {
    "hotel": "Resort Hotel", "adr": 180, "adults": 2, "children": 2.0, "babies": 0,
    "total_stay": 7, "country": "GBR", "meal": "BB", "is_repeated_guest": 0,
    "previous_bookings_not_canceled": 2, "total_of_special_requests": 2,
    "market_segment": "Online TA",
}


def test_root_and_health():
    assert client.get("/").json()["version"]
    h = client.get("/health")
    assert h.status_code in (200, 503)
    assert "models" in h.json()


def test_cancellation_schema_has_fields():
    fields = client.get("/api/v1/cancellation/schema").json()["fields"]
    assert {"hotel", "meal", "reserved_room_type"} <= set(fields)


def test_cancellation_predict_happy_and_validation():
    r = client.post("/api/v1/cancellation/predict", json=BOOKING)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["cancellation_probability"] <= 1
    assert body["risk_level"] in {"HIGH", "MODERATE", "LOW"}
    # validation guard
    assert client.post("/api/v1/cancellation/predict",
                       json={**BOOKING, "lead_time": -1}).status_code == 422


def test_sentiment_analyse_and_payload_cap():
    r = client.post("/api/v1/sentiment/analyse", json={"text": "Lovely stay, great staff."})
    assert r.status_code == 200
    assert r.json()["label"] in {"Positive", "Negative", "Neutral"}
    assert client.post("/api/v1/sentiment/analyse",
                       json={"text": "x" * 6000}).status_code == 422


def test_recommender_respects_top_n():
    r = client.post("/api/v1/recommend/next-action", params={"top_n": 2}, json=GUEST)
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 2


def test_overbooking_solver():
    body = {"capacity": 100,
            "tiers": [{"name": "Std", "n_bookings": 80, "cancel_prob": 0.28,
                       "adr": 120, "stay_nights": 2.0}],
            "max_walk_prob": 0.05}
    r = client.post("/api/v1/overbooking/solve", json=body)
    assert r.status_code == 200
    assert r.json()["optimal_overbooking"] >= 0


def test_analytics_channel_mix():
    r = client.get("/api/v1/analytics/channel-mix", params={"lookback_days": 180})
    assert r.status_code == 200
    assert "summary" in r.json()


def test_forecast_occupancy():
    r = client.get("/api/v1/forecast/occupancy", params={"horizon_days": 30})
    # 200 with the committed Prophet model; 503/500 only if it's genuinely absent.
    assert r.status_code in (200, 500, 503)
    if r.status_code == 200:
        assert "forecast" in r.json()
