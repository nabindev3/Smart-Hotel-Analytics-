"""
backend/routers/cancellation.py
"""
import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.artifacts import artifact_path
from backend.registry import load_cancellation_pipeline

router = APIRouter()

@lru_cache(maxsize=1)
def _load_cancel_model():
    # Registry when configured (MLFLOW_CANCELLATION_MODEL), else local joblib.
    return load_cancellation_pipeline()


@lru_cache(maxsize=1)
def _load_threshold() -> float:
    """Load the F1-optimal decision threshold saved by training. Falls
    back to 0.5 if the field is absent (older joblib files)."""
    cfg_path = artifact_path("models", "feature_config.joblib")
    if os.path.exists(cfg_path):
        cfg = joblib.load(cfg_path)
        if isinstance(cfg, dict):
            return float(cfg.get("best_threshold", 0.5))
    return 0.5

class BookingInput(BaseModel):
    # Bounds mirror the cleaning rules in src/train_models_ts.py, so the API
    # rejects negatives / absurd values up front instead of silently scoring them.
    hotel:                          str   = "Resort Hotel"
    lead_time:                      int   = Field(30, ge=0, le=700)
    arrival_date_month:             str   = "July"
    total_stay:                     int   = Field(3, ge=0, le=60)
    total_guests:                   int   = Field(2, ge=1, le=20)
    meal:                           str   = "BB"
    country:                        str   = Field("PRT", max_length=8)
    market_segment:                 str   = "Online TA"
    distribution_channel:           str   = "TA/TO"
    is_repeated_guest:              int   = Field(0, ge=0, le=1)
    previous_cancellations:         int   = Field(0, ge=0, le=100)
    previous_bookings_not_canceled: int   = Field(0, ge=0, le=200)
    reserved_room_type:             str   = "A"
    deposit_type:                   str   = "No Deposit"
    customer_type:                  str   = "Transient"
    required_car_parking_spaces:    int   = Field(0, ge=0, le=10)
    total_of_special_requests:      int   = Field(1, ge=0, le=10)
    adr:                            float = Field(120.0, ge=0, le=5000)

# Must match the training schema in src/train_models_ts.py (the leakage columns
# booking_changes and days_in_waiting_list are intentionally excluded there).
FEATURES = [
    "hotel","lead_time","arrival_date_month","total_stay","total_guests",
    "meal","country","market_segment","distribution_channel",
    "is_repeated_guest","previous_cancellations","previous_bookings_not_canceled",
    "reserved_room_type","deposit_type","customer_type",
    "required_car_parking_spaces","total_of_special_requests","adr",
]

# Allowed values for the categorical inputs — the single source of truth the
# frontend dropdowns read from (GET /schema), so the UI can't drift out of sync
# with what the model expects. The model itself tolerates unseen categories
# (OneHotEncoder handle_unknown="ignore"); these are the curated, sensible options.
SCHEMA_FIELDS = {
    "hotel":                ["Resort Hotel", "City Hotel"],
    "arrival_date_month":   ["January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"],
    "meal":                 ["BB", "HB", "FB", "SC", "Undefined"],
    "market_segment":       ["Online TA", "Direct", "Corporate", "Offline TA/TO",
                             "Groups", "Complementary", "Aviation"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
    "reserved_room_type":   ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
    "deposit_type":         ["No Deposit", "Non Refund", "Refundable"],
    "customer_type":        ["Transient", "Contract", "Transient-Party", "Group"],
}


@router.get("/schema")
def cancellation_schema():
    """Categorical form options for the cancellation/recommender UIs (item 10)."""
    return {"fields": SCHEMA_FIELDS}

@router.post("/predict")
def predict_cancellation(booking: BookingInput):
    model     = _load_cancel_model()
    threshold = _load_threshold()
    df        = pd.DataFrame([booking.model_dump()])[FEATURES]
    cp        = float(model.predict_proba(df)[0][1])
    op        = 1 - cp

    # Risk bands centred on the F1-tuned decision threshold rather than
    # a hard-coded 0.5 split.
    high_cut = max(threshold + 0.20, 0.55)
    mod_cut  = max(threshold,        0.30)
    risk = "HIGH" if cp > high_cut else "MODERATE" if cp > mod_cut else "LOW"

    return {
        "cancellation_probability": round(cp, 4),
        "occupancy_probability":    round(op, 4),
        "risk_level":               risk,
        "decision_threshold":       round(threshold, 4),
        "recommended_action": (
            "Request non-refundable deposit or apply overbooking." if risk=="HIGH" else
            "Courtesy confirmation call 48–72 hrs pre-arrival."    if risk=="MODERATE" else
            "Standard processing. No action required."
        ),
    }
