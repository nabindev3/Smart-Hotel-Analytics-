"""
pricing_engine.py — Dynamic Pricing Engine
==========================================
Connects Prophet forecasts to a revenue-management pricing algorithm.

Rules:
  1. Compare 30-day forecast occupancy vs historical average
  2. Adjust recommended ADR using price elasticity curves
  3. Apply floor (cost + margin) and ceiling (market rate) constraints
  4. Output tier-specific price recommendations

Industry standard: RevPAR = Occupancy × ADR
Goal:              Maximise RevPAR subject to market constraints
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PricingRecommendation:
    date_range:            str
    current_adr:           float
    recommended_adr:       float
    price_change_pct:      float
    demand_index:          float      # >1 = high demand, <1 = low demand
    forecast_occupancy:    float
    historical_occupancy:  float
    revpar_uplift_est:     float
    strategy:              str
    room_tier_prices:      dict
    reasoning:             str


class DynamicPricingEngine:
    """
    Multi-lever pricing engine combining:
      • Demand-based pricing   (Prophet occupancy forecast)
      • Competitor-indexed     (external regressor competitor_adr)
      • Event-based surcharge  (local_events flag)
      • Elasticity dampening   (avoid over-pricing in shoulder seasons)
    """

    # Price elasticity: % ADR change per % demand deviation
    ELASTICITY = 0.40

    # Hard bounds on price moves per recommendation cycle
    MAX_INCREASE_PCT = 0.25
    MAX_DECREASE_PCT = 0.15

    # RevPAR floor assumption (cost coverage)
    MIN_ADR = 60.0

    # Room tier multipliers relative to base ADR
    TIER_MULTIPLIERS = {
        "Standard":         1.00,
        "Superior":         1.25,
        "Deluxe":           1.55,
        "Junior Suite":     2.10,
        "Executive Suite":  3.20,
        "Presidential":     6.50,
    }

    def recommend(
        self,
        forecast_df:         pd.DataFrame,      # Prophet forecast with yhat
        historical_daily:    pd.DataFrame,       # daily_kpis.csv
        external_regs:       pd.DataFrame,       # external_regs.csv
        horizon_days:        int = 30,
        current_adr:         float = 120.0,
        competitor_premium:  float = 0.0,        # manual competitor offset
    ) -> PricingRecommendation:

        today = pd.Timestamp.now().normalize()

        # ── 1. Forecast occupancy for next `horizon_days` ─────────────────
        future_fc = forecast_df[forecast_df["ds"] >= today].head(horizon_days)
        if len(future_fc) == 0:
            future_fc = forecast_df.tail(horizon_days)

        forecast_occ = future_fc["yhat"].clip(0, 1).mean()

        # ── 2. Historical average occupancy (same period, prior years) ────
        hist = historical_daily.copy()
        hist["ds"] = pd.to_datetime(hist["ds"])
        same_period_hist = hist[
            (hist["ds"].dt.month.isin(future_fc["ds"].dt.month.unique())) &
            (hist["year"] < today.year)
        ]
        hist_occ = (same_period_hist["occupancy_rate"].mean()
                    if len(same_period_hist) > 0 else 0.65)

        # ── 3. Demand index ───────────────────────────────────────────────
        demand_index = forecast_occ / max(hist_occ, 0.01)

        # ── 4. Base price adjustment via elasticity ───────────────────────
        demand_deviation = demand_index - 1.0              # e.g. +0.15 = 15% above hist
        raw_price_change = demand_deviation * self.ELASTICITY

        # Clamp
        price_change = np.clip(raw_price_change,
                                -self.MAX_DECREASE_PCT,
                                 self.MAX_INCREASE_PCT)

        # ── 5. Event surcharge ────────────────────────────────────────────
        if "local_events" in external_regs.columns:
            future_ext = external_regs[external_regs["ds"] >= today].head(horizon_days)
            event_days = future_ext["local_events"].mean() if len(future_ext) > 0 else 0
            if event_days > 0.20:
                price_change = min(price_change + 0.05, self.MAX_INCREASE_PCT)

        # ── 6. Competitor adjustment ──────────────────────────────────────
        if "competitor_adr" in external_regs.columns:
            future_ext = external_regs[external_regs["ds"] >= today].head(horizon_days)
            comp_adr = (future_ext["competitor_adr"].mean()
                        if len(future_ext) > 0 else current_adr)
            comp_gap = (current_adr * (1 + price_change) - comp_adr) / comp_adr
            # If we're going to be >15% above competitor, moderate increase
            if comp_gap > 0.15:
                price_change = min(price_change, 0.08)
            # If we're below competitor, floor the decrease
            elif comp_gap < -0.10:
                price_change = max(price_change, 0.0)

        # ── 7. Compute final ADR recommendation ───────────────────────────
        recommended_adr = max(current_adr * (1 + price_change), self.MIN_ADR)
        actual_change   = (recommended_adr - current_adr) / current_adr

        # EstimatedRevPAR uplift
        current_revpar  = current_adr    * hist_occ
        forecast_revpar = recommended_adr * forecast_occ
        revpar_uplift   = forecast_revpar - current_revpar

        # ── 8. Strategy label ─────────────────────────────────────────────
        if demand_index >= 1.15:
            strategy = "YIELD MAXIMISATION — High Demand"
        elif demand_index >= 1.05:
            strategy = "RATE GROWTH — Above-Average Demand"
        elif demand_index >= 0.95:
            strategy = "HOLD — Stable Demand"
        elif demand_index >= 0.85:
            strategy = "STIMULATE — Below-Average Demand"
        else:
            strategy = "FIRE SALE — Low Demand Period"

        # ── 9. Per-tier prices ────────────────────────────────────────────
        tier_prices = {
            tier: round(recommended_adr * mult, 2)
            for tier, mult in self.TIER_MULTIPLIERS.items()
        }

        # ── 10. Reasoning ─────────────────────────────────────────────────
        direction = "increase" if actual_change > 0 else "decrease"
        reasoning = (
            f"Forecast occupancy ({forecast_occ:.1%}) is {demand_index:.2f}× "
            f"the historical baseline ({hist_occ:.1%}). "
            f"Elasticity model recommends a {abs(actual_change):.1%} "
            f"{direction} in ADR "
            f"(${current_adr:.0f} → ${recommended_adr:.0f}). "
            f"Estimated RevPAR change: {'+' if revpar_uplift>=0 else ''}${revpar_uplift:.0f}."
        )

        date_end = (today + pd.Timedelta(days=horizon_days)).strftime("%d %b %Y")
        return PricingRecommendation(
            date_range           = f"{today.strftime('%d %b %Y')} → {date_end}",
            current_adr          = round(current_adr, 2),
            recommended_adr      = round(recommended_adr, 2),
            price_change_pct     = round(actual_change * 100, 2),
            demand_index         = round(demand_index, 4),
            forecast_occupancy   = round(forecast_occ, 4),
            historical_occupancy = round(hist_occ, 4),
            revpar_uplift_est    = round(revpar_uplift, 2),
            strategy             = strategy,
            room_tier_prices     = tier_prices,
            reasoning            = reasoning,
        )


if __name__ == "__main__":
    # Quick smoke test with dummy data
    engine = DynamicPricingEngine()
    dates  = pd.date_range("2025-01-01", periods=60, freq="D")
    fc_df  = pd.DataFrame({"ds": dates, "yhat": np.random.uniform(0.70, 0.90, 60)})
    hist   = pd.DataFrame({"ds": dates[:30], "occupancy_rate": [0.65]*30,
                            "year": [2024]*30})
    hist["ds"] = pd.to_datetime(hist["ds"])
    ext    = pd.DataFrame({"ds": dates, "local_events": [0]*60, "competitor_adr": [115]*60})
    ext["ds"] = pd.to_datetime(ext["ds"])

    rec = engine.recommend(fc_df, hist, ext, current_adr=120.0)
    print(f"Strategy  : {rec.strategy}")
    print(f"ADR       : ${rec.current_adr} → ${rec.recommended_adr} ({rec.price_change_pct:+.1f}%)")
    print(f"RevPAR ↑  : ${rec.revpar_uplift_est:+.0f}")
    print(f"Reasoning : {rec.reasoning}")
