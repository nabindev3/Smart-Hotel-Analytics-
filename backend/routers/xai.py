"""backend/routers/xai.py"""
import json
import logging
import os
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("hotel_api.xai")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.artifacts import artifact_path
from backend.routers.cancellation import _load_cancel_model
from src.data_io import read_table
from src.shap_explainer import CancellationExplainer

router = APIRouter()

# Live SHAP is brutal on a small box (Render free tier = 512 MB RAM + a heavily
# throttled CPU): a full-frame global run OOM-kills the worker (502), and the
# first call pays a one-time numba JIT compile that can take >100 s — past the
# client timeout, surfacing as a 503. Two mitigations live in this file:
#   • global importance is served from a precomputed artifact (memory-safe,
#     instant); live compute is only a *capped* fallback if it's missing.
#   • warmup() pays the cold JIT cost at startup, not on a user's first request.
_BACKGROUND_ROWS = 50    # SHAP reference sample size — smaller = less RAM/CPU/call
_LIVE_GLOBAL_CAP = 150   # ceiling for the fallback live global run (avoid OOM)


@lru_cache(maxsize=1)
def _load_explainer():
    # Reuse the predictor's cached pipeline so SHAP explains exactly the model
    # that's serving predictions (and only one copy sits in memory).
    return CancellationExplainer(model=_load_cancel_model())


@lru_cache(maxsize=1)
def _background() -> pd.DataFrame:
    # A representative sample used as the SHAP reference for single-booking
    # explanations (so the explanation isn't taken against the row itself).
    bk = read_table(artifact_path("data", "bookings.csv"))
    cols = _load_explainer().features
    return bk[cols].sample(min(_BACKGROUND_ROWS, len(bk)), random_state=42)


def warmup() -> None:
    """Build the explainer and run one throwaway explanation at startup.

    The first SHAP call triggers a one-time numba JIT compilation. On the
    free-tier CPU that cold compile can take well over a minute — long enough to
    blow the client timeout and surface to the user as a 503. Doing it here (from
    the background warm-up thread) means real requests hit an already-warm
    explainer. Failures are non-fatal: the lazy first-call path still works."""
    exp = _load_explainer()
    exp.explain_instance(_background().head(1), background_raw=_background())
    logger.info("XAI explainer warmed up.")


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
    df  = pd.DataFrame([booking.model_dump()])[exp.features]
    try:
        return exp.explain_instance(df, background_raw=_background())
    except Exception:
        logger.exception("SHAP explain_instance failed")
        raise HTTPException(503, "explanation temporarily unavailable")


@lru_cache(maxsize=8)
def _global_importance_live(n_samples: int) -> dict:
    # Fallback path only (see _precomputed_global). Expensive: reads the bookings
    # frame and runs SHAP. Deterministic for a given n_samples (random_state=42),
    # so cache it instead of recomputing on every request.
    exp = _load_explainer()
    bk  = read_table(artifact_path("data", "bookings.csv"))
    return exp.explain_global(bk[exp.features], n_samples=n_samples)


def _precomputed_global() -> dict | None:
    """Load the precomputed global-importance artifact, if present. Regenerate it
    with `python -m src.shap_explainer` after retraining the cancellation model."""
    path = artifact_path("models", "global_importance.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@router.get("/global-importance")
def global_importance(n_samples: int = 300):
    """Top-20 global feature importances from SHAP.

    Served from the precomputed artifact (instant, memory-safe). The full-frame
    live run OOM-kills the 512 MB free-tier worker, so it's used only as a capped
    fallback when the artifact is missing."""
    pre = _precomputed_global()
    if pre is not None:
        return {
            "feature_names": pre["feature_names"],
            "mean_abs_shap": pre["mean_abs_shap"],
            "base_value":    pre["base_value"],
            "n_samples":     pre["n_samples"],
            "source":        "precomputed",
        }
    try:
        result = _global_importance_live(min(n_samples, _LIVE_GLOBAL_CAP))
        return {
            "feature_names": result["feature_names"],
            "mean_abs_shap": result["mean_abs_shap"],
            "base_value":    result["base_value"],
            "n_samples":     result["n_samples"],
            "source":        "live",
        }
    except Exception:
        logger.exception("SHAP global-importance failed")
        raise HTTPException(503, "explanation temporarily unavailable")

@router.get("/ablation")
def get_ablation_results():
    """Return pre-computed ablation study results."""
    path = artifact_path("models", "ablation_results.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Run src/ablation_study.py first.")
    with open(path) as f:
        return json.load(f)
