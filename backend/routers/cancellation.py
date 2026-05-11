"""
backend/routers/cancellation.py
"""
import os, sys
import pandas as pd, joblib
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
router = APIRouter()

@lru_cache(maxsize=1)
def _load_cancel_model():
    return joblib.load(os.path.join(ROOT,"models","cancellation_model.joblib"))


@lru_cache(maxsize=1)
def _load_threshold() -> float:
    """Load the F1-optimal decision threshold saved by training. Falls
    back to 0.5 if the field is absent (older joblib files)."""
    cfg_path = os.path.join(ROOT, "models", "feature_config.joblib")
    if os.path.exists(cfg_path):
        cfg = joblib.load(cfg_path)
        if isinstance(cfg, dict):
            return float(cfg.get("best_threshold", 0.5))
    return 0.5

class BookingInput(BaseModel):
    hotel:                          str   = "Resort Hotel"
    lead_time:                      int   = 30
    arrival_date_month:             str   = "July"
    total_stay:                     int   = 3
    total_guests:                   int   = 2
    meal:                           str   = "BB"
    country:                        str   = "PRT"
    market_segment:                 str   = "Online TA"
    distribution_channel:           str   = "TA/TO"
    is_repeated_guest:              int   = 0
    previous_cancellations:         int   = 0
    previous_bookings_not_canceled: int   = 0
    reserved_room_type:             str   = "A"
    booking_changes:                int   = 0
    deposit_type:                   str   = "No Deposit"
    days_in_waiting_list:           float = 0
    customer_type:                  str   = "Transient"
    required_car_parking_spaces:    int   = 0
    total_of_special_requests:      int   = 1
    adr:                            float = 120.0

FEATURES = [
    "hotel","lead_time","arrival_date_month","total_stay","total_guests",
    "meal","country","market_segment","distribution_channel",
    "is_repeated_guest","previous_cancellations","previous_bookings_not_canceled",
    "reserved_room_type","booking_changes","deposit_type",
    "days_in_waiting_list","customer_type",
    "required_car_parking_spaces","total_of_special_requests","adr",
]

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
