"""backend/routers/recommender.py"""
import os, sys
import pandas as pd
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.recommender import GuestRecommender

router = APIRouter()

@lru_cache(maxsize=1)
def _load_recommender():
    """
    Load (or re-train) the recommender. Re-trains on first load if the saved
    pickle was produced by a different module path (the historical
    `python src/recommender.py` flow pickles as `__main__.GuestRecommender`).
    """
    path = os.path.join(ROOT, "models", "recommender.joblib")

    def _train_and_save() -> "GuestRecommender":
        print("[recommender] Training fresh recommender…")
        bk = pd.read_csv(os.path.join(ROOT, "data", "bookings.csv"))
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
        print(f"[recommender] Cached model incompatible ({e}); retraining…")
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
