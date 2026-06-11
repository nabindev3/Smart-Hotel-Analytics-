"""
Tests for backend/registry.py — the optional MLflow registry loader.

The default (no env) path needs only joblib + the committed model, so it runs in
the slim CI. The registry round-trip needs mlflow, so it's skipped where mlflow
isn't installed (it's intentionally not a serving dependency).
"""
import os

import pandas as pd
import pytest

from backend.registry import load_cancellation_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE = pd.DataFrame([{
    "hotel": "Resort Hotel", "lead_time": 30, "arrival_date_month": "July",
    "total_stay": 3, "total_guests": 2, "meal": "BB", "country": "PRT",
    "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "is_repeated_guest": 0, "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
    "deposit_type": "No Deposit", "customer_type": "Transient",
    "required_car_parking_spaces": 0, "total_of_special_requests": 1, "adr": 120.0,
}])


def test_default_loads_local_joblib(monkeypatch):
    # No registry configured → local joblib pipeline that predicts.
    monkeypatch.delenv("MLFLOW_CANCELLATION_MODEL", raising=False)
    if not os.path.exists(os.path.join(ROOT, "models", "cancellation_model.joblib")):
        pytest.skip("model not trained")
    model = load_cancellation_pipeline()
    proba = model.predict_proba(SAMPLE)[0]
    assert len(proba) == 2 and abs(proba.sum() - 1.0) < 1e-6


def test_bad_registry_name_falls_back_to_joblib(monkeypatch):
    # A configured-but-broken registry must not break serving — fall back.
    pytest.importorskip("mlflow")
    if not os.path.exists(os.path.join(ROOT, "models", "cancellation_model.joblib")):
        pytest.skip("model not trained")
    monkeypatch.setenv("MLFLOW_CANCELLATION_MODEL", "does_not_exist_model")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + os.devnull)
    model = load_cancellation_pipeline()  # should log an error and fall back
    assert model.predict_proba(SAMPLE).shape == (1, 2)


def test_registry_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import joblib
    import mlflow
    import mlflow.sklearn

    model_path = os.path.join(ROOT, "models", "cancellation_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("model not trained")

    uri = f"sqlite:///{tmp_path / 'reg.db'}"
    mlflow.set_tracking_uri(uri)
    pipe = joblib.load(model_path)
    with mlflow.start_run():
        mlflow.sklearn.log_model(pipe, name="cancellation_model",
                                 registered_model_name="hotel_cancellation_test")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_CANCELLATION_MODEL", "hotel_cancellation_test")
    monkeypatch.setenv("MLFLOW_CANCELLATION_STAGE", "latest")

    model = load_cancellation_pipeline()
    # The registry model must agree with the local one it was logged from.
    local = joblib.load(model_path)
    assert abs(model.predict_proba(SAMPLE)[0][1] - local.predict_proba(SAMPLE)[0][1]) < 1e-9
