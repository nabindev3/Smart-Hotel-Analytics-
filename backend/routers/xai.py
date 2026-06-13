"""backend/routers/xai.py"""
import logging
import os
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("hotel_api.xai")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.artifacts import artifact_path
from backend.registry import load_cancellation_pipeline
from src.data_io import read_table
from src.shap_explainer import FEATURES, CancellationExplainer

router = APIRouter()

@lru_cache(maxsize=1)
def _load_explainer():
    # Share the predictor's model source (registry or joblib) so SHAP explains
    # exactly the model that's serving predictions.
    return CancellationExplainer(model=load_cancellation_pipeline())


@lru_cache(maxsize=1)
def _background() -> pd.DataFrame:
    # A representative sample used as the SHAP reference for single-booking
    # explanations (so the explanation isn't taken against the row itself).
    bk = read_table(artifact_path("data", "bookings.csv"))
    return bk[FEATURES].sample(min(100, len(bk)), random_state=42)

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
    deposit_type:                   str   = "No Deposit"
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
        return exp.explain_instance(df, background_raw=_background())
    except Exception:
        logger.exception("SHAP explain_instance failed")
        raise HTTPException(503, "explanation temporarily unavailable")

@lru_cache(maxsize=8)
def _global_importance(n_samples: int) -> dict:
    # Expensive: reads the full bookings frame and runs SHAP. The sample is fixed
    # (random_state=42), so the result is deterministic for a given n_samples —
    # cache it instead of recomputing on every request.
    exp = _load_explainer()
    bk  = read_table(artifact_path("data", "bookings.csv"))
    return exp.explain_global(bk[FEATURES], n_samples=n_samples)


@router.get("/global-importance")
def global_importance(n_samples: int = 300):
    """Top-20 global feature importances from SHAP."""
    try:
        result = _global_importance(n_samples)
        return {
            "feature_names":  result["feature_names"],
            "mean_abs_shap":  result["mean_abs_shap"],
            "base_value":     result["base_value"],
            "n_samples":      result["n_samples"],
        }
    except Exception:
        logger.exception("SHAP global-importance failed")
        raise HTTPException(503, "explanation temporarily unavailable")

@router.get("/ablation")
def get_ablation_results():
    """Return pre-computed ablation study results."""
    import json
    path = artifact_path("models", "ablation_results.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Run src/ablation_study.py first.")
    with open(path) as f:
        return json.load(f)
