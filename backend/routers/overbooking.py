"""backend/routers/overbooking.py"""
import os, sys
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.overbooking_engine import solve_overbooking, BookingTier

router = APIRouter()

class TierInput(BaseModel):
    name:        str   = "Standard"
    n_bookings:  int   = 80
    cancel_prob: float = 0.28
    adr:         float = 120.0
    stay_nights: float = 2.0

class OverbookingRequest(BaseModel):
    capacity:      int         = Field(100, ge=1)
    tiers:         List[TierInput]
    c_empty:       float       = 500.0
    c_walk:        float       = 1500.0
    max_walk_prob: float       = Field(0.05, ge=0.01, le=0.20)

@router.post("/solve")
def solve(req: OverbookingRequest):
    tiers = [BookingTier(t.name, t.n_bookings, t.cancel_prob, t.adr, t.stay_nights)
             for t in req.tiers]
    res = solve_overbooking(req.capacity, tiers, req.c_empty, req.c_walk, req.max_walk_prob)
    return {
        "optimal_overbooking": res.optimal_overbooking,
        "expected_revenue":    res.expected_revenue,
        "expected_walk_cost":  res.expected_walk_cost,
        "expected_profit":     res.expected_profit,
        "walk_probability":    res.walk_probability,
        "recommendation":      res.recommendation,
        "tier_details":        res.tier_details,
        "sensitivity":         res.sensitivity[:20],
    }
