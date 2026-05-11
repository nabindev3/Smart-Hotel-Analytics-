"""
backend/main.py — Smart Hotel Analytics API
=============================================
FastAPI microservice exposing all ML models as REST endpoints.

Run (development):
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Run (production):
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os, sys, time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.routers import (
    forecast, cancellation, pricing, overbooking, recommender, sentiment, xai,
    briefing, analytics,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan — warm up models on startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🏨  Warming up ML models…")
    try:
        # Pre-load all models into module-level caches
        from backend.routers.cancellation import _load_cancel_model
        from backend.routers.forecast     import _load_prophet
        from backend.routers.recommender  import _load_recommender
        from backend.routers.xai          import _load_explainer

        _load_cancel_model()
        for name in ["occupancy","adr","revenue"]:
            _load_prophet(name)
        _load_recommender()
        _load_explainer()
        print("✅  All models loaded.")
    except Exception as e:
        print(f"⚠️  Model warm-up warning: {e}")
    yield
    print("🏨  Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Smart Hotel Analytics API",
    description = (
        "Production ML microservice for hotel revenue management.\n\n"
        "Endpoints: forecasting · cancellation risk · dynamic pricing · "
        "LP overbooking · guest recommender · XAI (SHAP) · sentiment NLP."
    ),
    version  = "2.0.0",
    docs_url = "/docs",
    lifespan = lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],     # restrict in prod
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(forecast.router,     prefix="/api/v1/forecast",    tags=["Forecasting"])
app.include_router(cancellation.router, prefix="/api/v1/cancellation",tags=["Cancellation"])
app.include_router(pricing.router,      prefix="/api/v1/pricing",     tags=["Pricing"])
app.include_router(overbooking.router,  prefix="/api/v1/overbooking", tags=["Overbooking"])
app.include_router(recommender.router,  prefix="/api/v1/recommend",   tags=["Recommender"])
app.include_router(sentiment.router,    prefix="/api/v1/sentiment",   tags=["Sentiment"])
app.include_router(xai.router,          prefix="/api/v1/xai",         tags=["XAI"])
app.include_router(briefing.router,     prefix="/api/v1/briefing",    tags=["Briefing"])
app.include_router(analytics.router,    prefix="/api/v1/analytics",   tags=["Analytics"])

# ── Health & root ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"service": "Smart Hotel Analytics API", "version": "2.0.0", "status": "ok"}

@app.get("/health", tags=["Health"])
def health():
    return {
        "status":    "healthy",
        "timestamp": time.time(),
        "models": {
            "prophet":      os.path.exists(os.path.join(ROOT,"models","prophet_occupancy.joblib")),
            "cancellation": os.path.exists(os.path.join(ROOT,"models","cancellation_model.joblib")),
            "recommender":  os.path.exists(os.path.join(ROOT,"models","recommender.joblib")),
        },
    }

@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse({"error": "endpoint not found", "docs": "/docs"}, status_code=404)

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse({"error": str(exc)}, status_code=500)
