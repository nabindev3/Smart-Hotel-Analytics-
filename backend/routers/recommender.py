"""backend/routers/recommender.py"""
import logging
import os
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.artifacts import artifact_path
from src.data_io import read_table
from src.recommender import GuestRecommender

logger = logging.getLogger("hotel_api.recommender")
router = APIRouter()

@lru_cache(maxsize=1)
def _load_recommender():
    """
    Load (or re-train) the recommender. Re-trains on first load if the saved
    pickle was produced by a different module path (the historical
    `python src/recommender.py` flow pickles as `__main__.GuestRecommender`).
    """
    path = artifact_path("models", "recommender.joblib")

    def _train_and_save() -> "GuestRecommender":
        logger.warning("Training a fresh recommender on the request path "
                       "(no usable cached model).")
        bk = read_table(artifact_path("data", "bookings.csv"))
        rec = GuestRecommender()
        rec.fit(bk)
        rec.save(path)
        return rec

    if not os.path.exists(path):
        return _train_and_save()

    try:
        return GuestRecommender.load(path)
    except (AttributeError, ModuleNotFoundError, ImportError) as e:
        # Pickled with a different module path (e.g., __main__) — re-train.
        logger.warning("Cached recommender incompatible (%s); retraining.", e)
        return _train_and_save()

class GuestProfile(BaseModel):
    hotel:                          str   = "Resort Hotel"
    adr:                            float = 180.0
    adults:                         int   = 2
    children:                       float = 0.0
    babies:                         int   = 0
    total_stay:                     int   = 4
    country:                        str   = "GBR"
    meal:                           str   = "BB"
    is_repeated_guest:              int   = 0
    previous_bookings_not_canceled: int   = 0
    total_of_special_requests:      int   = 1
    market_segment:                 str   = "Online TA"

@router.post("/next-action")
def recommend(profile: GuestProfile, top_n: int = 3):
    rec = _load_recommender()
    result = rec.predict_guest(profile.model_dump(), top_n=top_n)
    return {
        "loyalty_tier":       result.loyalty_tier,
        "next_best_action":   result.next_best_action,
        "estimated_upsell":   result.estimated_upsell,
        "recommendations": [
            {
                "service":    r["service"],
                "label":      r["label"],
                "score":      r["score"],
                "revenue":    r["revenue"],
                "email_copy": r["email_copy"],
            }
            for r in result.top_recommendations
        ],
    }
