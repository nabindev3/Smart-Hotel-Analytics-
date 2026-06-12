"""
overbooking_engine.py  — Overbooking Optimiser
==================================================================
Solves the classic hotel overbooking problem by enumerating every integer
overbooking level Δ ∈ {0, …, max_overbook}, scoring expected profit for each,
and selecting the most profitable level whose walk probability stays within the
risk tolerance. This is a small 1-D search, so it is a direct filter + argmax —
no LP solver is required (an earlier version used PuLP/CBC purely to express a
"pick exactly one row" selection, which added a heavy solver dependency for a
one-line operation).

Modelling assumption
--------------------
The number of guests who cancel is treated as Binomial(n, p) and approximated by
a Normal distribution to derive the walk probability (`norm.cdf`). This Gaussian
approximation is convenient and accurate for large n with moderate p, but it is
*crude for small booking counts or extreme cancellation rates* (p near 0 or 1),
where the exact Binomial CDF should be preferred. Treat `walk_probability` as an
estimate, not an exact figure.

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

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


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

        # P(walk) = P(show > capacity) — Normal approximation to the Binomial
        # (see module docstring for the limitations of this approximation).
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

    # ── Selection: max-profit level within the walk-risk tolerance ─────────
    # The decision is "pick the single Δ with the highest expected profit among
    # those whose walk probability is acceptable" — a filter followed by an
    # argmax. No optimisation solver is needed for a 1-D enumerated search.
    feasible = [r for r in results if r["p_walk"] <= max_walk_prob]
    if feasible:
        opt = max(feasible, key=lambda r: r["e_profit"])
    else:
        # Even Δ=0 breaches the walk tolerance (high no-show variance). Fall back
        # to the lowest-risk option, which is always Δ=0 (walk risk is monotone
        # increasing in Δ). This mirrors the previous behaviour, where the LP
        # became infeasible and the code defaulted to Δ=0.
        opt = min(results, key=lambda r: r["p_walk"])
    optimal_delta = opt["delta"]

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
