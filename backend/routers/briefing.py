"""
backend/routers/briefing.py — Today's Briefing
================================================
A single endpoint that aggregates everything a hotel manager needs to see
when they sit down at their desk in the morning. Designed for the new
'Today's Briefing' home tab.

GET  /api/v1/briefing/today
"""
from __future__ import annotations
import os, sys, json
from functools import lru_cache
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

router = APIRouter()


# ─── Caching helpers ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_daily() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, "data", "daily_kpis.csv"),
                       parse_dates=["ds"])


@lru_cache(maxsize=1)
def _load_bookings() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, "data", "bookings.csv"),
                       parse_dates=["arrival_date"])


@lru_cache(maxsize=3)
def _load_prophet(name: str):
    return joblib.load(os.path.join(ROOT, "models", f"prophet_{name}.joblib"))


# ─── Helpers ────────────────────────────────────────────────────────────────
def _trend(curr: float, prev: float) -> dict:
    if prev == 0 or pd.isna(prev):
        return {"direction": "flat", "delta_pct": 0.0}
    delta = (curr - prev) / abs(prev)
    direction = "up" if delta > 0.02 else "down" if delta < -0.02 else "flat"
    return {"direction": direction, "delta_pct": round(float(delta) * 100, 1)}


def _alerts(daily: pd.DataFrame, fc_occ: pd.DataFrame) -> list[dict]:
    """Generate plain-English alerts for the manager."""
    out = []
    last30 = daily.tail(30)
    last7  = daily.tail(7)
    prev7  = daily.iloc[-14:-7] if len(daily) >= 14 else last7

    cancel_now  = last7["cancellation_rate"].mean()
    cancel_prev = prev7["cancellation_rate"].mean()
    if cancel_now > 0.35:
        out.append({
            "level":   "warning",
            "title":   "High cancellation rate this week",
            "detail":  f"{cancel_now:.0%} of bookings are cancelling — "
                       f"normal range is 20–30%. Consider tightening deposit policy.",
        })
    elif cancel_now > cancel_prev * 1.3 and cancel_now > 0.20:
        out.append({
            "level":   "info",
            "title":   "Cancellations trending up",
            "detail":  f"Up {(cancel_now - cancel_prev) * 100:+.1f} pts vs last week.",
        })

    occ_now = last7["occupancy_rate"].mean()
    if occ_now < 0.50:
        out.append({
            "level":   "warning",
            "title":   "Occupancy below 50%",
            "detail":  f"7-day average is {occ_now:.0%}. "
                       f"Consider a flash promotion or lower OTA rates.",
        })
    elif occ_now > 0.85:
        out.append({
            "level":   "good",
            "title":   "Strong occupancy",
            "detail":  f"7-day average is {occ_now:.0%}. "
                       f"Room to push prices up on remaining inventory.",
        })

    # Forecast spike detection
    if fc_occ is not None and len(fc_occ) >= 30:
        next30 = fc_occ.head(30)
        peak   = next30["yhat"].max()
        if peak > occ_now * 1.20 and peak > 0.80:
            peak_date = next30.loc[next30["yhat"].idxmax(), "ds"]
            out.append({
                "level":  "good",
                "title":  "Demand spike coming",
                "detail": f"Forecast occupancy peaks at {peak:.0%} around "
                          f"{peak_date.strftime('%a %d %b')}. Hold inventory; raise rates.",
            })

    if not out:
        out.append({
            "level":  "info",
            "title":  "Operations are stable",
            "detail": "No anomalies detected in the last 7 days.",
        })
    return out


def _top_actions(daily: pd.DataFrame, bookings: pd.DataFrame) -> list[str]:
    """Three things the manager should consider doing today."""
    actions = []
    last7   = daily.tail(7)
    cancel  = last7["cancellation_rate"].mean()
    occ     = last7["occupancy_rate"].mean()

    if cancel > 0.30:
        actions.append("Review deposit policy for online travel agency bookings.")
    if occ < 0.55:
        actions.append("Send a 7-day flash discount to past guests to fill rooms.")
    if occ > 0.80:
        actions.append("Increase rates on remaining inventory by 10–15%.")

    # Channel mix
    recent = bookings[bookings["arrival_date"] > bookings["arrival_date"].max() - pd.Timedelta(days=60)]
    if len(recent) > 0:
        ota_share = (recent["distribution_channel"] == "TA/TO").mean()
        if ota_share > 0.55:
            actions.append(f"Online travel agency mix is {ota_share:.0%} — "
                           "consider 'Book Direct' incentives to cut commissions.")

    while len(actions) < 3:
        actions.append("Stable day — focus on guest experience and review responses.")
    return actions[:3]


# ─── Endpoint ───────────────────────────────────────────────────────────────
@router.get("/today")
def today_briefing(horizon_days: int = Query(7, ge=1, le=30)):
    """
    Manager's morning briefing — single call, returns everything for the home tab.
    """
    daily    = _load_daily()
    bookings = _load_bookings()

    last_n   = daily.tail(horizon_days)
    prev_n   = daily.iloc[-horizon_days * 2:-horizon_days] if len(daily) >= horizon_days * 2 else last_n

    # Prophet forecast — prefer future-of-today, but fall back to the
    # latest forecast points if the cached model's horizon doesn't
    # extend past `today`.
    fc_occ = pd.DataFrame()
    try:
        full_fc = _load_prophet("occupancy")["forecast"]
        today   = pd.Timestamp.now().normalize()
        fwd     = full_fc[full_fc["ds"] > today].head(30).reset_index(drop=True)
        if len(fwd) > 0:
            fc_occ = fwd
        elif len(full_fc) > 0:
            # Cached forecast doesn't reach today — use the last 30 points.
            fc_occ = full_fc.tail(30).reset_index(drop=True)
    except Exception:
        fc_occ = pd.DataFrame()

    # Quality
    quality_path = os.path.join(ROOT, "data", "data_quality.json")
    quality = {}
    if os.path.exists(quality_path):
        try:
            with open(quality_path) as f:
                quality = json.load(f)
        except Exception:
            quality = {}

    return {
        "as_of":   datetime.now().isoformat(timespec="seconds"),
        "period_days": horizon_days,
        "headline": {
            "occupancy":   round(float(last_n["occupancy_rate"].mean()), 4),
            "adr":         round(float(last_n["avg_adr"].mean()), 2),
            "revpar":      round(float(last_n["revpar"].mean()), 2),
            "cancel_rate": round(float(last_n["cancellation_rate"].mean()), 4),
            "revenue":     round(float(last_n["revenue"].sum()), 2),
            "bookings":    int(last_n["total_bookings"].sum()),
        },
        "trend": {
            "occupancy":   _trend(last_n["occupancy_rate"].mean(),    prev_n["occupancy_rate"].mean()),
            "adr":         _trend(last_n["avg_adr"].mean(),            prev_n["avg_adr"].mean()),
            "revpar":      _trend(last_n["revpar"].mean(),             prev_n["revpar"].mean()),
            "cancel_rate": _trend(last_n["cancellation_rate"].mean(),  prev_n["cancellation_rate"].mean()),
        },
        "next_7_days_outlook": [
            {
                "date":     str(row["ds"].date()),
                "expected_occupancy": round(float(row["yhat"]), 4),
                "low":      round(float(row["yhat_lower"]), 4),
                "high":     round(float(row["yhat_upper"]), 4),
            }
            for _, row in fc_occ.head(7).iterrows()
        ] if len(fc_occ) else [],
        "alerts":         _alerts(daily, fc_occ if len(fc_occ) else None),
        "suggested_actions": _top_actions(daily, bookings),
        "data_quality":   quality,
    }
