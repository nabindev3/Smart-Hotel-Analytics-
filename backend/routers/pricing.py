"""backend/routers/pricing.py"""
import os, sys
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.pricing_engine import DynamicPricingEngine
router  = APIRouter()
_engine = DynamicPricingEngine()

import joblib
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_forecast():
    import joblib
    b = joblib.load(os.path.join(ROOT,"models","prophet_occupancy.joblib"))
    return b["forecast"]

@router.get("/recommendation")
def get_pricing(
    current_adr:  float = Query(120.0, description="Current ADR in EUR"),
    horizon_days: int   = Query(30,    ge=7, le=90),
):
    """Dynamic pricing recommendation based on Prophet demand forecast."""
    forecast_df = _load_forecast()
    daily       = pd.read_csv(os.path.join(ROOT,"data","daily_kpis.csv"), parse_dates=["ds"])
    ext         = pd.read_csv(os.path.join(ROOT,"data","external_regs.csv"), parse_dates=["ds"])

    rec = _engine.recommend(forecast_df, daily, ext,
                             horizon_days=horizon_days, current_adr=current_adr)
    return {
        "date_range":           rec.date_range,
        "current_adr":          rec.current_adr,
        "recommended_adr":      rec.recommended_adr,
        "price_change_pct":     rec.price_change_pct,
        "demand_index":         rec.demand_index,
        "forecast_occupancy":   rec.forecast_occupancy,
        "historical_occupancy": rec.historical_occupancy,
        "revpar_uplift_est":    rec.revpar_uplift_est,
        "strategy":             rec.strategy,
        "room_tier_prices":     rec.room_tier_prices,
        "reasoning":            rec.reasoning,
    }
