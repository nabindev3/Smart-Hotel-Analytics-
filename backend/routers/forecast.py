"""
backend/routers/forecast.py — Prophet forecast endpoints
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

router = APIRouter()

@lru_cache(maxsize=3)
def _load_prophet(name: str):
    path = os.path.join(ROOT, "models", f"prophet_{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def _load_daily():
    return pd.read_csv(os.path.join(ROOT,"data","daily_kpis.csv"), parse_dates=["ds"])

def _load_ext():
    return pd.read_csv(os.path.join(ROOT,"data","external_regs.csv"), parse_dates=["ds"])


@router.get("/kpis/summary")
def get_kpi_summary():
    """Return latest 30-day KPI snapshot.

    NOTE: This route MUST be declared before /{metric} so FastAPI's wildcard
    matcher does not greedy-match the static path "/kpis" as a metric name.
    Even though FastAPI's longest-prefix resolver currently handles this
    correctly, ordering it explicitly makes the intent unambiguous and
    immune to refactors.
    """
    daily = _load_daily()
    last30 = daily.tail(30)
    return {
        "avg_occupancy":      round(float(last30["occupancy_rate"].mean()), 4),
        "avg_adr":            round(float(last30["avg_adr"].mean()), 2),
        "avg_revpar":         round(float(last30["revpar"].mean()), 2),
        "avg_cancel_rate":    round(float(last30["cancellation_rate"].mean()), 4),
        "total_revenue_30d":  round(float(last30["revenue"].sum()), 2),
        "period_start":       str(last30["ds"].iloc[0].date()),
        "period_end":         str(last30["ds"].iloc[-1].date()),
    }


@router.get("/{metric}")
def get_forecast(
    metric:       str,
    horizon_days: int = Query(90, ge=1, le=365),
    include_components: bool = False,
):
    """Get Prophet forecast for occupancy | adr | revenue"""
    if metric not in ["occupancy","adr","revenue"]:
        raise HTTPException(400, "metric must be occupancy | adr | revenue")

    bundle   = _load_prophet(metric)
    forecast = bundle["forecast"]
    model    = bundle["model"]
    mape     = bundle["mape"]

    daily  = _load_daily()
    col_map= {"occupancy":"occupancy_rate","adr":"avg_adr","revenue":"revenue"}
    col    = col_map[metric]
    actual = daily[["ds",col]].rename(columns={col:"y"})
    actual = actual[actual["y"]>0]

    today    = actual["ds"].max()
    fut_fc   = forecast[forecast["ds"] > today].head(horizon_days)

    result = {
        "metric":      metric,
        "mape":        round(float(mape), 4) if not np.isnan(mape) else None,
        "horizon_days":horizon_days,
        "forecast": [
            {
                "date":        str(row["ds"].date()),
                "yhat":        round(float(row["yhat"]),  4),
                "yhat_lower":  round(float(row["yhat_lower"]), 4),
                "yhat_upper":  round(float(row["yhat_upper"]), 4),
            }
            for _, row in fut_fc.iterrows()
        ],
        "actual_tail": [
            {"date": str(row["ds"].date()), "value": round(float(row["y"]),4)}
            for _, row in actual.tail(30).iterrows()
        ],
    }

    if include_components:
        yr_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=365, freq="D")})
        ext   = _load_ext()
        for reg in bundle.get("regressors",[]):
            yr_df[reg] = ext[reg].mean()
        yr_fc = model.predict(yr_df)
        result["components"] = {
            "trend_sample":  [round(float(v),4) for v in forecast["trend"].tail(90).tolist()],
            "yearly":        [round(float(v),4) for v in yr_fc["yearly"].tolist()],
            "weekly":        [round(float(v),4) for v in yr_fc["weekly"].head(7).tolist()],
        }
    return result
