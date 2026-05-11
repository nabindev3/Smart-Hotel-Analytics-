"""
overbooking_engine.py  — Linear Programming Overbooking Optimiser
==================================================================
Solves the classic hotel overbooking problem using Linear Programming (PuLP).

Given:
  • P(cancellation)  per booking tier (from ML model)
  • Cost of empty room when cancelled booking NOT replaced     → c_empty
  • Cost of "walking" a guest (VIP compensation, reputation)   → c_walk
  • Hotel capacity (rooms)

Finds: the optimal number of rooms to overbook (Δ) such that
       expected profit is maximised without unacceptable walk risk.

References
----------
  Rothstein, M. (1974). Hotel Overbooking as a Markovian Sequential Decision Process.
  Decision Sciences, 5(3), 389-404.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pulp


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BookingTier:
    name:             str
    n_bookings:       int          # bookings in this tier
    cancel_prob:      float        # P(cancel) from ML model
    adr:              float        # average daily rate $
    stay_nights:      float = 2.0  # expected length of stay


@dataclass
class OverbookingResult:
    optimal_overbooking:   int
    expected_revenue:      float
    expected_walk_cost:    float
    expected_profit:       float
    walk_probability:      float
    recommendation:        str
    tier_details:          list[dict] = field(default_factory=list)
    sensitivity:           list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
def solve_overbooking(
    capacity:       int,
    tiers:          list[BookingTier],
    c_empty:        float = 500.0,   # cost of leaving room empty (lost revenue)
    c_walk:         float = 1500.0,  # cost of walking a guest (comp + reputation)
    max_walk_prob:  float = 0.05,    # max acceptable walk probability
    max_overbook:   int   = 30,      # upper search bound
) -> OverbookingResult:
    """
    Solves for the optimal integer overbooking level Δ ∈ {0, …, max_overbook}.

    Objective (maximise):
        E[Profit(Δ)] = E[Revenue] - E[Walk Cost] - E[Empty Cost]

    Subject to:
        P(walk | Δ) ≤ max_walk_prob
        Δ ≥ 0
    """

    total_bookings  = sum(t.n_bookings for t in tiers)
    # Weighted average cancel probability
    avg_cancel_prob = (sum(t.n_bookings * t.cancel_prob for t in tiers)
                       / total_bookings) if total_bookings > 0 else 0.30

    # Expected cancellations ~ Binomial(n, p), approximated by Normal for LP
    mu_cancel  = total_bookings * avg_cancel_prob
    sd_cancel  = np.sqrt(total_bookings * avg_cancel_prob * (1 - avg_cancel_prob)) + 1e-9

    # Weighted average revenue per night
    avg_revenue = (sum(t.n_bookings * t.adr * t.stay_nights for t in tiers)
                   / total_bookings) if total_bookings > 0 else 200.0

    results = []
    for delta in range(0, max_overbook + 1):
        booked = capacity + delta
        # Expected number who show up = booked - expected cancellations
        e_show = booked - mu_cancel

        # P(walk) = P(show > capacity) — normal CDF approximation
        from scipy.stats import norm
        p_walk  = 1 - norm.cdf(capacity, loc=e_show, scale=sd_cancel)

        # Expected walks = E[max(show - capacity, 0)]
        e_walk  = max(0, e_show - capacity)

        # Expected empty rooms = E[max(capacity - show, 0)]
        e_empty = max(0, capacity - e_show)

        e_revenue  = min(e_show, capacity) * avg_revenue
        e_walk_cost= e_walk  * c_walk
        e_empty_cost=e_empty * c_empty
        e_profit   = e_revenue - e_walk_cost - e_empty_cost

        results.append({
            "delta":      delta,
            "p_walk":     round(p_walk,  4),
            "e_walk":     round(e_walk,  2),
            "e_empty":    round(e_empty, 2),
            "e_revenue":  round(e_revenue,   2),
            "e_walk_cost":round(e_walk_cost, 2),
            "e_profit":   round(e_profit,    2),
        })

    # ── PuLP formulation ──────────────────────────────────────────────────
    prob = pulp.LpProblem("hotel_overbooking", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("overbook", range(max_overbook+1),
                                cat="Binary")

    # Objective: maximise expected profit
    prob += pulp.lpSum(r["e_profit"] * x[r["delta"]] for r in results)

    # Constraints
    prob += pulp.lpSum(x[r["delta"]] for r in results) == 1   # pick exactly one
    prob += pulp.lpSum(r["p_walk"] * x[r["delta"]] for r in results) <= max_walk_prob
    prob += pulp.lpSum(r["delta"]  * x[r["delta"]] for r in results) >= 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Find optimal delta
    optimal_delta = 0
    for r in results:
        if pulp.value(x[r["delta"]]) and pulp.value(x[r["delta"]]) > 0.5:
            optimal_delta = r["delta"]
            break

    opt = next(r for r in results if r["delta"] == optimal_delta)

    # Recommendation text
    if optimal_delta == 0:
        rec = ("No overbooking recommended. Cancellation probability is low enough "
               "that the walk penalty outweighs the benefit.")
    elif opt["p_walk"] < 0.02:
        rec = (f"Overbook by {optimal_delta} rooms. Walk probability is very low "
               f"({opt['p_walk']:.1%}). Expected profit uplift: "
               f"${opt['e_profit'] - results[0]['e_profit']:,.0f}.")
    else:
        rec = (f"Overbook by {optimal_delta} rooms (walk risk: {opt['p_walk']:.1%}). "
               f"Expected profit: ${opt['e_profit']:,.0f}. "
               f"Monitor closely — activate walk protocol if occupancy hits 98%.")

    return OverbookingResult(
        optimal_overbooking = optimal_delta,
        expected_revenue    = opt["e_revenue"],
        expected_walk_cost  = opt["e_walk_cost"],
        expected_profit     = opt["e_profit"],
        walk_probability    = opt["p_walk"],
        recommendation      = rec,
        tier_details        = [
            {"tier": t.name, "bookings": t.n_bookings,
             "cancel_prob": t.cancel_prob, "adr": t.adr}
            for t in tiers
        ],
        sensitivity = results,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tiers = [
        BookingTier("VIP Suites",     n_bookings=10, cancel_prob=0.10, adr=450),
        BookingTier("Standard Rooms", n_bookings=80, cancel_prob=0.28, adr=120),
        BookingTier("OTA Discount",   n_bookings=40, cancel_prob=0.42, adr=85),
    ]
    res = solve_overbooking(capacity=100, tiers=tiers)
    print(f"\nOptimal overbooking: {res.optimal_overbooking} rooms")
    print(f"Walk probability:    {res.walk_probability:.2%}")
    print(f"Expected profit:     ${res.expected_profit:,.0f}")
    print(f"\n{res.recommendation}")
