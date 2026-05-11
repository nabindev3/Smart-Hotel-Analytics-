"""
tests/test_pipeline.py — CI/CD Unit Tests
==========================================
Runs automatically on every git push via GitHub Actions.
Tests: API health · model loading · data quality · business logic
"""

import os, sys, json
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def bookings():
    path = os.path.join(ROOT,"data","bookings.csv")
    if not os.path.exists(path):
        pytest.skip("bookings.csv not found — run generate_data.py first")
    return pd.read_csv(path)

@pytest.fixture(scope="session")
def daily_kpis():
    path = os.path.join(ROOT,"data","daily_kpis.csv")
    if not os.path.exists(path):
        pytest.skip("daily_kpis.csv not found")
    return pd.read_csv(path, parse_dates=["ds"])

@pytest.fixture(scope="session")
def cancel_model():
    import joblib
    path = os.path.join(ROOT,"models","cancellation_model.joblib")
    if not os.path.exists(path):
        pytest.skip("Model not trained yet")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# DATA TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestData:
    def test_bookings_schema(self, bookings):
        required = ["booking_id","arrival_date","hotel","is_canceled","adr","lead_time"]
        for col in required:
            assert col in bookings.columns, f"Missing column: {col}"

    def test_bookings_row_count(self, bookings):
        assert len(bookings) >= 10_000, "Expected at least 10k bookings"

    def test_cancellation_rate_reasonable(self, bookings):
        cr = bookings["is_canceled"].mean()
        assert 0.10 <= cr <= 0.60, f"Cancellation rate {cr:.2%} outside expected range"

    def test_daily_kpis_date_continuity(self, daily_kpis):
        dates = pd.to_datetime(daily_kpis["ds"])
        delta = (dates.max() - dates.min()).days
        assert delta >= 365, "Daily KPI series too short"

    def test_external_regs_all_present(self):
        path = os.path.join(ROOT,"data","external_regs.csv")
        if not os.path.exists(path): pytest.skip()
        ext = pd.read_csv(path)
        expected = ["temperature_c","precipitation_mm","local_events",
                    "holiday_flag","competitor_adr","cpi_yoy",
                    "consumer_confidence","search_trend"]
        for col in expected:
            assert col in ext.columns, f"Missing regressor: {col}"

    def test_data_quality_report(self):
        path = os.path.join(ROOT,"data","data_quality.json")
        if not os.path.exists(path): pytest.skip()
        with open(path) as f:
            report = json.load(f)
        assert report["data_quality_grade"] in ["A","B","C"]
        assert report["total_missing_pct"] < 20.0, "Missing rate too high"

    def test_no_negative_adr(self, bookings):
        adr_clean = bookings["adr"].dropna()
        n_neg = (adr_clean < 0).sum()
        assert n_neg == 0, f"{n_neg} negative ADR values after cleaning"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestModels:
    def test_cancellation_model_loads(self, cancel_model):
        assert cancel_model is not None

    def test_cancellation_model_predicts(self, cancel_model):
        import pandas as pd
        row = pd.DataFrame([{
            "hotel":"Resort Hotel","lead_time":30,"arrival_date_month":"July",
            "total_stay":3,"total_guests":2,"meal":"BB","country":"PRT",
            "market_segment":"Online TA","distribution_channel":"TA/TO",
            "is_repeated_guest":0,"previous_cancellations":0,
            "previous_bookings_not_canceled":0,"reserved_room_type":"A",
            "booking_changes":0,"deposit_type":"No Deposit",
            "days_in_waiting_list":0.0,"customer_type":"Transient",
            "required_car_parking_spaces":0,"total_of_special_requests":1,"adr":120.0,
        }])
        proba = cancel_model.predict_proba(row)[0]
        assert len(proba) == 2, "Expected binary output"
        assert 0 <= proba[0] <= 1 and 0 <= proba[1] <= 1
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_cancellation_probability_range(self, cancel_model, bookings):
        bookings = bookings.dropna(subset=["adr"]).head(100)
        bookings["children"] = bookings["children"].fillna(0)
        bookings["days_in_waiting_list"] = bookings["days_in_waiting_list"].fillna(0)
        bookings["meal"] = bookings["meal"].fillna("BB")
        bookings["country"] = bookings["country"].fillna("PRT")
        bookings["total_stay"]   = bookings.get("total_stay", 
            bookings["stays_in_weekend_nights"] + bookings["stays_in_week_nights"])
        bookings["total_guests"] = bookings["adults"] + bookings["children"] + bookings["babies"]

        FEATURES = [
            "hotel","lead_time","arrival_date_month","total_stay","total_guests",
            "meal","country","market_segment","distribution_channel",
            "is_repeated_guest","previous_cancellations","previous_bookings_not_canceled",
            "reserved_room_type","booking_changes","deposit_type",
            "days_in_waiting_list","customer_type",
            "required_car_parking_spaces","total_of_special_requests","adr",
        ]
        X = bookings[FEATURES].copy()
        proba = cancel_model.predict_proba(X)[:,1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_prophet_models_load(self):
        import joblib
        for name in ["occupancy","adr","revenue"]:
            path = os.path.join(ROOT,"models",f"prophet_{name}.joblib")
            if not os.path.exists(path):
                pytest.skip(f"Prophet {name} not trained")
            bundle = joblib.load(path)
            assert "model"    in bundle
            assert "forecast" in bundle
            assert "mape"     in bundle

    def test_recommender_loads(self):
        """
        Use the resilient backend loader rather than raw joblib.load — it
        re-trains on the fly if the cached pickle was saved under a stale
        module path (a known failure mode when src/recommender.py is run
        as __main__).
        """
        path = os.path.join(ROOT, "models", "recommender.joblib")
        if not os.path.exists(path):
            pytest.skip()
        from backend.routers.recommender import _load_recommender
        _load_recommender.cache_clear()
        rec = _load_recommender()
        assert rec.is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestBusinessLogic:
    def test_overbooking_solver(self):
        from src.overbooking_engine import solve_overbooking, BookingTier
        tiers = [BookingTier("Standard", 80, 0.30, 120.0)]
        result = solve_overbooking(capacity=100, tiers=tiers)
        assert result.optimal_overbooking >= 0
        assert 0 <= result.walk_probability <= 1.0
        assert isinstance(result.recommendation, str)
        assert len(result.sensitivity) > 0

    def test_overbooking_walk_prob_constraint(self):
        from src.overbooking_engine import solve_overbooking, BookingTier
        tiers = [BookingTier("Standard", 120, 0.40, 100.0)]
        result = solve_overbooking(capacity=100, tiers=tiers, max_walk_prob=0.05)
        assert result.walk_probability <= 0.06  # small tolerance

    def test_pricing_engine(self):
        from src.pricing_engine import DynamicPricingEngine
        import pandas as pd, numpy as np
        engine = DynamicPricingEngine()
        dates  = pd.date_range("2025-01-01", periods=60, freq="D")
        fc_df  = pd.DataFrame({"ds":dates,"yhat":np.full(60, 0.80)})
        hist   = pd.DataFrame({"ds":dates[:30],"occupancy_rate":[0.65]*30,"year":[2024]*30})
        hist["ds"] = pd.to_datetime(hist["ds"])
        ext    = pd.DataFrame({"ds":dates,"local_events":[0]*60,"competitor_adr":[115.0]*60})
        ext["ds"] = pd.to_datetime(ext["ds"])
        rec = engine.recommend(fc_df, hist, ext, current_adr=120.0)
        assert rec.recommended_adr >= engine.MIN_ADR
        assert rec.demand_index > 0
        assert isinstance(rec.strategy, str)
        assert len(rec.room_tier_prices) == 6

    def test_recommender_predict(self):
        path = os.path.join(ROOT,"models","recommender.joblib")
        if not os.path.exists(path): pytest.skip()
        import joblib
        from src.recommender import GuestRecommender
        rec = GuestRecommender.load(path)
        result = rec.predict_guest({
            "hotel":"Resort Hotel","adr":200,"children":2,"adults":2,
            "babies":0,"total_stay":5,"country":"GBR","meal":"BB",
            "is_repeated_guest":0,"previous_bookings_not_canceled":0,
            "total_of_special_requests":2,
        })
        assert len(result.top_recommendations) > 0
        assert result.loyalty_tier in ["Gold","Silver","Bronze"]
        for r in result.top_recommendations:
            assert 0 <= r["score"] <= 1


# ─────────────────────────────────────────────────────────────────────────────
# API INTEGRATION TEST (requires running backend)
# ─────────────────────────────────────────────────────────────────────────────
class TestAPI:
    """Integration tests — skipped if backend is not running."""
    API = os.environ.get("API_BASE","http://localhost:8000")

    def _get(self, path):
        import requests
        try:
            return requests.get(f"{self.API}{path}", timeout=5)
        except Exception:
            return None

    def test_api_health(self):
        r = self._get("/health")
        if r is None: pytest.skip("Backend not running")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_forecast_endpoint(self):
        r = self._get("/api/v1/forecast/occupancy?horizon_days=30")
        if r is None: pytest.skip("Backend not running")
        assert r.status_code == 200
        data = r.json()
        assert "forecast" in data
        assert len(data["forecast"]) == 30

    def test_kpi_summary(self):
        r = self._get("/api/v1/forecast/kpis/summary")
        if r is None: pytest.skip("Backend not running")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# IN-PROCESS TESTS for new analytics + briefing routers (no live backend)
# ─────────────────────────────────────────────────────────────────────────────
class TestNewEndpoints:
    """Boots FastAPI in-process and exercises the new manager-friendly endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
        except Exception as e:
            pytest.skip(f"Could not boot FastAPI app: {e}")
        return TestClient(app)

    def test_briefing_today(self, client):
        r = client.get("/api/v1/briefing/today", params={"horizon_days": 7})
        assert r.status_code == 200
        d = r.json()
        for k in ("headline", "trend", "alerts", "suggested_actions", "data_quality"):
            assert k in d
        assert isinstance(d["alerts"], list) and len(d["alerts"]) >= 1
        assert len(d["suggested_actions"]) == 3
        # Headline must include core KPIs
        for k in ("occupancy", "adr", "revpar", "cancel_rate", "revenue", "bookings"):
            assert k in d["headline"]

    def test_channel_mix(self, client):
        r = client.get("/api/v1/analytics/channel-mix", params={"lookback_days": 180})
        assert r.status_code == 200
        d = r.json()
        assert "channels" in d and "summary" in d
        # Net revenue should never exceed gross
        s = d["summary"]
        assert s["total_net_revenue"] <= s["total_gross_revenue"] + 1e-6

    def test_no_show_heatmap(self, client):
        r = client.get("/api/v1/analytics/no-show-heatmap", params={"lookback_days": 365})
        assert r.status_code == 200
        d = r.json()
        assert len(d["days"]) == 7
        assert len(d["months"]) == 12
        assert len(d["rate_matrix"]) == 7
        assert all(0 <= v <= 1 for row in d["rate_matrix"] for v in row)

    def test_guest_mix(self, client):
        r = client.get("/api/v1/analytics/guest-mix",
                       params={"lookback_days": 365, "top_n": 10})
        assert r.status_code == 200
        d = r.json()
        assert "top_countries" in d and "segments" in d

    def test_revenue_trend(self, client):
        r = client.get("/api/v1/analytics/revenue-trend",
                       params={"lookback_days": 90, "rolling": 7})
        assert r.status_code == 200
        d = r.json()
        assert len(d["series"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
