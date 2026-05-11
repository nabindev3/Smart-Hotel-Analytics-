"""backend/routers/xai.py"""
import os, sys
import pandas as pd
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.shap_explainer import CancellationExplainer, FEATURES

router = APIRouter()

@lru_cache(maxsize=1)
def _load_explainer():
    return CancellationExplainer(os.path.join(ROOT,"models","cancellation_model.joblib"))

class BookingForXAI(BaseModel):
    hotel:                          str   = "Resort Hotel"
    lead_time:                      int   = 120
    arrival_date_month:             str   = "August"
    total_stay:                     int   = 3
    total_guests:                   int   = 2
    meal:                           str   = "BB"
    country:                        str   = "PRT"
    market_segment:                 str   = "Online TA"
    distribution_channel:           str   = "TA/TO"
    is_repeated_guest:              int   = 0
    previous_cancellations:         int   = 1
    previous_bookings_not_canceled: int   = 0
    reserved_room_type:             str   = "A"
    booking_changes:                int   = 0
    deposit_type:                   str   = "No Deposit"
    days_in_waiting_list:           float = 0
    customer_type:                  str   = "Transient"
    required_car_parking_spaces:    int   = 0
    total_of_special_requests:      int   = 0
    adr:                            float = 80.0

@router.post("/explain")
def explain_booking(booking: BookingForXAI):
    """SHAP waterfall explanation for a single booking."""
    exp = _load_explainer()
    df  = pd.DataFrame([booking.model_dump()])[FEATURES]
    try:
        result = exp.explain_instance(df)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/global-importance")
def global_importance(n_samples: int = 300):
    """Top-20 global feature importances from SHAP."""
    exp = _load_explainer()
    bk  = pd.read_csv(os.path.join(ROOT,"data","bookings.csv"))
    try:
        result = exp.explain_global(bk[FEATURES], n_samples=n_samples)
        return {
            "feature_names":  result["feature_names"],
            "mean_abs_shap":  result["mean_abs_shap"],
            "base_value":     result["base_value"],
            "n_samples":      result["n_samples"],
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/ablation")
def get_ablation_results():
    """Return pre-computed ablation study results."""
    import json
    path = os.path.join(ROOT,"models","ablation_results.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Run src/ablation_study.py first.")
    with open(path) as f:
        return json.load(f)
