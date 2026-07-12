"""
recommender.py — Hyper-Personalised Guest Recommender
======================================================

⚠️  SYNTHETIC / DEMONSTRATION MODEL — NOT REAL COLLABORATIVE FILTERING  ⚠️
─────────────────────────────────────────────────────────────────────────────
There is **no real guest×service interaction data** behind this model. Service
scores come from the hand-written rules in `_build_interaction_matrix` (plus a
little Gaussian noise), ranked directly. An earlier version fitted TruncatedSVD
over a matrix fabricated from these same rules and averaged the top-20
cosine-similar fabricated rows — which could only re-derive the rules with
extra steps, so that layer was removed.

To make this real, source an actual guest×service usage matrix from the PMS
(Property Management System) / POS — e.g. spa bookings, upgrades purchased,
restaurant covers — and fit genuine collaborative filtering on observed
interactions.
─────────────────────────────────────────────────────────────────────────────

Rule-based scorer that suggests the guest's Next Best Action (NBA).

Guest features used:
  • room type booked   → proxy for spend level
  • travel party       → adults + children + babies
  • meal plan          → dining preference
  • special requests   → engagement level
  • previous bookings  → loyalty tier
  • market segment     → booking behaviour
  • arrival month      → seasonality

Services modelled:
  Spa, Family Package, Airport Transfer, Room Upgrade,
  Late Checkout, Early Check-in, Restaurant Reservation,
  Wine Welcome, Couples Package, Kids Club
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

SERVICES = [
    "spa_treatment", "family_package", "airport_transfer",
    "room_upgrade", "late_checkout", "early_checkin",
    "restaurant_reservation", "wine_welcome",
    "couples_package", "kids_club",
]

SERVICE_LABELS = {
    "spa_treatment":          "🧖 Spa & Wellness Treatment",
    "family_package":         "👨‍👩‍👧 Family Fun Package",
    "airport_transfer":       "🚗 Private Airport Transfer",
    "room_upgrade":           "🏨 Room Upgrade",
    "late_checkout":          "⏰ Late Checkout (2 PM)",
    "early_checkin":          "🌅 Early Check-in (10 AM)",
    "restaurant_reservation": "🍽️  Fine Dining Reservation",
    "wine_welcome":           "🍾 Champagne Welcome",
    "couples_package":        "💑 Couples Escape Package",
    "kids_club":              "🎠 Kids Club Access",
}

# Loyalty tiering thresholds (minimum count of previous non-cancelled bookings).
# Ordered best → worst; the first tier whose threshold is met wins. Centralised
# here rather than hard-coded inline so the cut-offs are configurable in one place.
LOYALTY_TIERS = [
    ("Gold",   5),
    ("Silver", 2),
    ("Bronze", 0),
]

# Revenue per upsell (for ROI reporting)
SERVICE_REVENUE = {
    "spa_treatment": 180,   "family_package": 250,  "airport_transfer": 95,
    "room_upgrade": 150,    "late_checkout": 45,     "early_checkin": 45,
    "restaurant_reservation": 80, "wine_welcome": 65,
    "couples_package": 320, "kids_club": 120,
}


@dataclass
class GuestRecommendation:
    guest_profile:       dict
    top_recommendations: list[dict]    # [{service, score, revenue, email_copy}]
    loyalty_tier:        str
    next_best_action:    str
    estimated_upsell:    float


class GuestRecommender:
    """
    Rule-based service scorer (see the module banner: the SVD/CF layer was
    removed because it was fitted on a matrix fabricated from these rules).
    """

    def __init__(self):
        self.is_fitted = False

    def _build_interaction_matrix(self, bookings: pd.DataFrame) -> np.ndarray:
        """
        Simulate a guest×service matrix from booking features.
        In production this comes from a PMS (Property Management System).
        """
        n = len(bookings)
        rng = np.random.default_rng(42)
        matrix = np.zeros((n, len(SERVICES)))

        # Rule-based interaction simulation (proxy for real PMS data)
        for i, (_, row) in enumerate(bookings.iterrows()):
            # Spa — high value guests and couples without kids
            if (row.get("adr", 100) > 150 and row.get("total_guests", 2) <= 2
                    and row.get("children", 0) == 0):
                matrix[i, SERVICES.index("spa_treatment")]    = rng.uniform(0.6, 1.0)
                matrix[i, SERVICES.index("couples_package")]  = rng.uniform(0.5, 0.9)

            # Family — guests with children
            if row.get("children", 0) > 0 or row.get("babies", 0) > 0:
                matrix[i, SERVICES.index("family_package")]   = rng.uniform(0.7, 1.0)
                matrix[i, SERVICES.index("kids_club")]        = rng.uniform(0.6, 0.9)

            # Transfers — long-haul countries
            if row.get("country", "PRT") in ["USA","AUS","JPN","SGP","BRA","CAN"]:
                matrix[i, SERVICES.index("airport_transfer")] = rng.uniform(0.5, 0.9)

            # Upgrade — special requests signal upgrade propensity
            if row.get("total_of_special_requests", 0) >= 2:
                matrix[i, SERVICES.index("room_upgrade")]     = rng.uniform(0.4, 0.8)

            # Restaurant — HB/FB meal plan = already dining
            if row.get("meal", "BB") in ["HB","FB"]:
                matrix[i, SERVICES.index("restaurant_reservation")] = rng.uniform(0.3, 0.7)

            # Late checkout — long-stay guests
            if row.get("total_stay", 2) >= 5:
                matrix[i, SERVICES.index("late_checkout")]    = rng.uniform(0.4, 0.7)

            # Wine welcome — repeat guests or resort
            if row.get("is_repeated_guest", 0) == 1 or row.get("hotel","") == "Resort Hotel":
                matrix[i, SERVICES.index("wine_welcome")]     = rng.uniform(0.5, 0.8)

            # Add noise to all services
            matrix[i] += rng.normal(0, 0.05, len(SERVICES)).clip(-0.1, 0.1)
            matrix[i]  = matrix[i].clip(0, 1)

        return matrix

    def fit(self, bookings: pd.DataFrame):
        # The rules need no fitting; kept so the train → save → load flow
        # (bootstrap, backend fallback retrain) stays unchanged.
        self.is_fitted = True
        return self

    def predict_guest(self, guest: dict, top_n: int = 3) -> GuestRecommendation:
        """
        Predict top-N service recommendations for a new guest profile.
        guest: dict with booking features (same schema as bookings.csv row)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        # Score services for this guest straight from the rules.
        dummy = pd.DataFrame([guest])
        dummy["children"] = dummy.get("children", 0)
        predicted = self._build_interaction_matrix(dummy)[0]

        # Rank
        ranked = sorted(zip(SERVICES, predicted), key=lambda x: -x[1])

        top_recs = []
        for svc, score in ranked[:top_n]:
            email_copy = _generate_email_snippet(svc, guest)
            top_recs.append({
                "service":   svc,
                "label":     SERVICE_LABELS[svc],
                "score":     round(float(score), 3),
                "revenue":   SERVICE_REVENUE[svc],
                "email_copy": email_copy,
            })

        # Loyalty tier — first tier (best → worst) whose threshold is met.
        prev_bookings = guest.get("previous_bookings_not_canceled", 0)
        loyalty = next(name for name, floor in LOYALTY_TIERS if prev_bookings >= floor)

        nba = top_recs[0]["label"] if top_recs else "Standard welcome"
        est_upsell = sum(r["revenue"] * r["score"] for r in top_recs)

        return GuestRecommendation(
            guest_profile       = guest,
            top_recommendations = top_recs,
            loyalty_tier        = loyalty,
            next_best_action    = nba,
            estimated_upsell    = round(est_upsell, 2),
        )

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "GuestRecommender":
        return joblib.load(path)


def _generate_email_snippet(service: str, guest: dict) -> str:
    """Generate personalised pre-arrival email copy for a service."""
    first = "Dear Valued Guest"
    nights = guest.get("total_stay", guest.get("stays_in_week_nights", 2))
    snips  = {
        "spa_treatment":
            f"{first}, ahead of your {nights}-night stay we'd love to reserve "
            "your preferred treatment at our award-winning spa. Book now and "
            "receive 15% off with complimentary aromatherapy.",
        "family_package":
            f"{first}, we've prepared our exclusive Family Adventure Package "
            "for your stay. Complimentary kids' breakfast, supervised activities, "
            "and a family portrait session — all included.",
        "airport_transfer":
            f"{first}, let us take care of your journey from the very first moment. "
            "Our private chauffeur service is available for $95 one-way.",
        "room_upgrade":
            f"{first}, a superior room with ocean view is available at a special "
            f"pre-arrival rate for you. Upgrade from ${int(guest.get('adr',120))} "
            f"to just ${int(guest.get('adr',120))*1.3:.0f}/night.",
        "late_checkout":
            f"{first}, extend your stay with our 2 PM Late Checkout for just $45. "
            "No need to rush — enjoy a leisurely final morning.",
        "early_checkin":
            f"{first}, arrive early and begin your stay immediately. "
            "10 AM Early Check-in available for $45.",
        "restaurant_reservation":
            f"{first}, our sommelier-curated tasting menu is reserved for hotel "
            "guests only. Secure your table for an unforgettable evening.",
        "wine_welcome":
            f"{first}, welcome back. A bottle of estate Champagne will be "
            "chilled and waiting in your room upon arrival.",
        "couples_package":
            f"{first}, celebrate in style with our Couples Escape — private "
            "candlelit dinner, couples massage, and rose-petal turndown.",
        "kids_club":
            f"{first}, our supervised Kids Club (ages 3–12) ensures a relaxing "
            "holiday for you and unforgettable adventures for the little ones.",
    }
    return snips.get(service, f"{first}, we look forward to welcoming you.")


if __name__ == "__main__":
    bk = pd.read_csv("data/bookings.csv")
    rec = GuestRecommender()
    rec.fit(bk)
    rec.save("models/recommender.joblib")
    print("\nTest prediction:")
    result = rec.predict_guest({
        "hotel": "Resort Hotel", "adr": 200, "children": 2,
        "adults": 2, "babies": 0, "total_stay": 7,
        "country": "GBR", "meal": "BB", "is_repeated_guest": 1,
        "previous_bookings_not_canceled": 3, "total_of_special_requests": 2,
    })
    print(f"Loyalty: {result.loyalty_tier}")
    print(f"NBA: {result.next_best_action}")
    print(f"Est. upsell: ${result.estimated_upsell:.0f}")
    for r in result.top_recommendations:
        print(f"  {r['label']} (score={r['score']:.3f}, rev=${r['revenue']})")
