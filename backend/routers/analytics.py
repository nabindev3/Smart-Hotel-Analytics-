"""
backend/routers/analytics.py — Operational analytics
======================================================
Endpoints designed for the GM rather than the data scientist.

  GET  /api/v1/analytics/channel-mix
  GET  /api/v1/analytics/no-show-heatmap
  GET  /api/v1/analytics/guest-mix
  GET  /api/v1/analytics/revenue-trend
"""
from __future__ import annotations
import os, sys
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

router = APIRouter()


# ── Industry-standard commission rates by channel (after negotiation) ──
COMMISSION = {
    "TA/TO":     0.18,    # OTA (Booking, Expedia)
    "Direct":    0.00,
    "Corporate": 0.05,
    "GDS":       0.10,    # Global Distribution System
    "Undefined": 0.12,
}

CHANNEL_LABEL = {
    "TA/TO":     "Online Travel Agency (OTA)",
    "Direct":    "Direct (website / phone)",
    "Corporate": "Corporate accounts",
    "GDS":       "Travel agent (GDS)",
    "Undefined": "Other",
}


@lru_cache(maxsize=1)
def _bookings() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, "data", "bookings.csv"),
                     parse_dates=["arrival_date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Channel mix profitability
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/channel-mix")
def channel_mix(lookback_days: int = Query(180, ge=30, le=730)):
    """
    Show booking volume, gross revenue, commission cost, and net revenue by
    distribution channel. The numbers a GM cares about — not the data scientist.
    """
    df = _bookings()
    cutoff = df["arrival_date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[(df["arrival_date"] >= cutoff) & (df["is_canceled"] == 0)].copy()

    if len(recent) == 0:
        return {"channels": [], "summary": {}}

    rows = []
    total_gross = total_net = total_book = 0
    for chan, grp in recent.groupby("distribution_channel"):
        rate     = COMMISSION.get(chan, 0.12)
        gross    = float(grp["revenue"].sum())
        commish  = gross * rate
        net      = gross - commish
        nb       = int(len(grp))
        avg_adr  = float(grp["adr"].mean())
        cancel_rate_chan = float(
            df[(df["arrival_date"] >= cutoff) &
               (df["distribution_channel"] == chan)]["is_canceled"].mean()
        )

        rows.append({
            "channel":           chan,
            "label":             CHANNEL_LABEL.get(chan, chan),
            "bookings":          nb,
            "gross_revenue":     round(gross, 2),
            "commission_rate":   rate,
            "commission_cost":   round(commish, 2),
            "net_revenue":       round(net, 2),
            "avg_adr":           round(avg_adr, 2),
            "cancellation_rate": round(cancel_rate_chan, 4),
        })
        total_gross += gross
        total_net   += net
        total_book  += nb

    rows.sort(key=lambda r: r["net_revenue"], reverse=True)

    return {
        "lookback_days": lookback_days,
        "channels":      rows,
        "summary": {
            "total_gross_revenue": round(total_gross, 2),
            "total_net_revenue":   round(total_net, 2),
            "total_commission":    round(total_gross - total_net, 2),
            "blended_take_rate":   round((total_gross - total_net) / total_gross, 4) if total_gross else 0,
            "total_bookings":      total_book,
        },
        "explanation": (
            "Net revenue is what stays in the hotel's pocket after commissions. "
            "Channels with the highest gross revenue aren't always the best — "
            "an OTA booking at $200 with 18% commission nets less than a direct "
            "booking at $180 with no commission."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. No-show / cancel heatmap
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/no-show-heatmap")
def no_show_heatmap(lookback_days: int = Query(365, ge=60, le=2000)):
    """
    Day-of-week × month heatmap of cancellation rate. Helps the front office
    spot which weekdays in which seasons need extra attention.
    """
    df = _bookings()
    cutoff = df["arrival_date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["arrival_date"] >= cutoff].copy()

    recent["day"]   = recent["arrival_date"].dt.day_name()
    recent["month"] = recent["arrival_date"].dt.month_name()

    months_order = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]
    days_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    pivot = (recent.pivot_table(index="day", columns="month",
                                values="is_canceled", aggfunc="mean")
                   .reindex(index=days_order, columns=months_order))
    counts = (recent.pivot_table(index="day", columns="month",
                                values="is_canceled", aggfunc="count")
                   .reindex(index=days_order, columns=months_order))

    return {
        "lookback_days": lookback_days,
        "days":          days_order,
        "months":        months_order,
        "rate_matrix":   pivot.fillna(0).round(4).values.tolist(),
        "count_matrix":  counts.fillna(0).astype(int).values.tolist(),
        "overall_rate":  round(float(recent["is_canceled"].mean()), 4),
        "explanation": (
            "Each cell is the share of bookings that cancelled or didn't show, "
            "broken down by day-of-week and month. Hot spots tell the front "
            "office where to add reminder calls or stricter deposits."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Guest mix
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/guest-mix")
def guest_mix(lookback_days: int = Query(365, ge=30, le=2000), top_n: int = 10):
    """Where guests come from + segment breakdown."""
    df = _bookings()
    cutoff = df["arrival_date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[(df["arrival_date"] >= cutoff) & (df["is_canceled"] == 0)]

    countries = (recent.groupby("country")
                       .agg(bookings=("country","count"),
                            revenue=("revenue","sum"))
                       .sort_values("revenue", ascending=False)
                       .head(top_n)
                       .reset_index())

    segments = (recent.groupby("market_segment")
                      .agg(bookings=("market_segment","count"),
                           revenue=("revenue","sum"),
                           avg_adr=("adr","mean"))
                      .sort_values("revenue", ascending=False)
                      .reset_index())

    return {
        "lookback_days": lookback_days,
        "top_countries": [
            {"country": r["country"], "bookings": int(r["bookings"]),
             "revenue": round(float(r["revenue"]), 2)}
            for _, r in countries.iterrows()
        ],
        "segments": [
            {"segment": r["market_segment"], "bookings": int(r["bookings"]),
             "revenue": round(float(r["revenue"]), 2),
             "avg_adr": round(float(r["avg_adr"]), 2)}
            for _, r in segments.iterrows()
        ],
        "repeat_share": round(float(recent["is_repeated_guest"].mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Revenue trend (rolling)
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/revenue-trend")
def revenue_trend(lookback_days: int = Query(180, ge=30, le=2000),
                  rolling: int = Query(7, ge=1, le=30)):
    """Daily revenue + rolling average, ready to chart."""
    daily = pd.read_csv(os.path.join(ROOT, "data", "daily_kpis.csv"),
                        parse_dates=["ds"])
    cutoff = daily["ds"].max() - pd.Timedelta(days=lookback_days)
    recent = daily[daily["ds"] >= cutoff].copy()
    recent["revenue_smooth"] = recent["revenue"].rolling(rolling, min_periods=1).mean()
    recent["occupancy_smooth"] = recent["occupancy_rate"].rolling(rolling, min_periods=1).mean()

    return {
        "lookback_days": lookback_days,
        "rolling_window": rolling,
        "series": [
            {
                "date":      str(r["ds"].date()),
                "revenue":   round(float(r["revenue"]), 2),
                "smoothed":  round(float(r["revenue_smooth"]), 2),
                "occupancy": round(float(r["occupancy_rate"]), 4),
                "occupancy_smooth": round(float(r["occupancy_smooth"]), 4),
            }
            for _, r in recent.iterrows()
        ],
    }
