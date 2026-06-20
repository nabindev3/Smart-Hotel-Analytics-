"""
backend/main.py — Smart Hotel Analytics API
=============================================
FastAPI microservice exposing all ML models as REST endpoints.

Run (development):
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Run (production):
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from backend.routers import (
    analytics,
    briefing,
    cancellation,
    forecast,
    overbooking,
    pricing,
    recommender,
    sentiment,
    xai,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("hotel_api")

# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan — warm up models on startup
# ─────────────────────────────────────────────────────────────────────────────
# Tracks the actual load outcome of each model so /health can report the real
# runtime state instead of merely checking whether files exist on disk.
# Values: None = not attempted yet, True = loaded OK, str = error message.
_MODEL_STATUS: dict[str, object] = {
    "cancellation":     None,
    "prophet_occupancy": None,
    "prophet_adr":       None,
    "prophet_revenue":   None,
    "recommender":      None,
    "explainer":        None,
}
_WARMUP_DONE = threading.Event()


def _try_load(name: str, loader) -> None:
    """Run a model loader, recording success or the failure reason."""
    try:
        loader()
        _MODEL_STATUS[name] = True
    except Exception as e:
        _MODEL_STATUS[name] = str(e)
        logger.exception("Model load failed: %s", name)


def _warm_models():
    from backend.routers.cancellation import _load_cancel_model
    from backend.routers.forecast import _load_prophet
    from backend.routers.recommender import _load_recommender
    from backend.routers.xai import _load_explainer
    from backend.routers.xai import warmup as _warmup_xai

    _try_load("cancellation", _load_cancel_model)
    for n in ["occupancy", "adr", "revenue"]:
        _try_load(f"prophet_{n}", lambda n=n: _load_prophet(n))
    _try_load("recommender", _load_recommender)
    _try_load("explainer", _load_explainer)
    # Pay SHAP's one-time numba JIT cost here (background thread) so the first
    # /xai/explain request isn't the one that waits >100 s and times out as a 503.
    if _MODEL_STATUS["explainer"] is True:
        try:
            _warmup_xai()
        except Exception:
            logger.exception("XAI warm-up failed (non-fatal)")

    failed = [k for k, v in _MODEL_STATUS.items() if v not in (True, None)]
    if failed:
        logger.warning("Model warm-up finished with failures: %s", ", ".join(failed))
    else:
        logger.info("All models loaded.")
    _WARMUP_DONE.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm models in a background thread so the port binds immediately.
    # On Render's free tier, blocking startup on heavy model loads
    # (Prophet + SHAP) can exceed the proxy timeout and surface as 502.
    # Router-level @lru_cache loaders make lazy first-call loading safe.
    logger.info("Starting; warming models in background…")
    threading.Thread(target=_warm_models, daemon=True).start()
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────
VERSION = "2.0.0"   # single source — referenced by the app, root, and /health

app = FastAPI(
    title       = "Smart Hotel Analytics API",
    description = (
        "ML microservice for hotel revenue management.\n\n"
        "Endpoints: forecasting · cancellation risk · dynamic pricing · "
        "overbooking · guest recommender · XAI (SHAP) · sentiment NLP."
    ),
    version  = VERSION,
    docs_url = "/docs",
    lifespan = lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# Lock CORS to the frontend origin(s) via CORS_ORIGINS (comma-separated). Defaults
# to "*" so local dev / the demo keep working, but a real deploy should set it to
# the dashboard's origin. Only the methods the API actually uses are allowed.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins   = _origins or ["*"],
    allow_credentials = bool(_origins),     # credentials can't be combined with "*"
    allow_methods   = ["GET", "POST", "OPTIONS"],
    allow_headers   = ["*"],
)

# ── Optional API-key auth ─────────────────────────────────────────────────────
# When API_KEY is set, every /api/v1/* route requires a matching X-API-Key header
# (the paid sentiment/XAI endpoints especially shouldn't be open). Unset → no auth,
# so the local demo keeps working. /health, /, and /docs stay public.
API_KEY = os.environ.get("API_KEY", "")


def require_api_key(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


_auth = [Depends(require_api_key)]

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(forecast.router,     prefix="/api/v1/forecast",    tags=["Forecasting"], dependencies=_auth)
app.include_router(cancellation.router, prefix="/api/v1/cancellation",tags=["Cancellation"], dependencies=_auth)
app.include_router(pricing.router,      prefix="/api/v1/pricing",     tags=["Pricing"],      dependencies=_auth)
app.include_router(overbooking.router,  prefix="/api/v1/overbooking", tags=["Overbooking"],  dependencies=_auth)
app.include_router(recommender.router,  prefix="/api/v1/recommend",   tags=["Recommender"],  dependencies=_auth)
app.include_router(sentiment.router,    prefix="/api/v1/sentiment",   tags=["Sentiment"],    dependencies=_auth)
app.include_router(xai.router,          prefix="/api/v1/xai",         tags=["XAI"],          dependencies=_auth)
app.include_router(briefing.router,     prefix="/api/v1/briefing",    tags=["Briefing"],     dependencies=_auth)
app.include_router(analytics.router,    prefix="/api/v1/analytics",   tags=["Analytics"],    dependencies=_auth)

# ── Health & root ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"service": "Smart Hotel Analytics API", "version": VERSION, "status": "ok"}

@app.get("/health", tags=["Health"])
def health():
    """Report the real load state of every warmed model.

    Reflects whether models actually loaded into memory (not just whether the
    files exist on disk). Returns:
      • starting — warm-up still running
      • healthy  — every model loaded
      • degraded — some models loaded, some failed (HTTP 200, but flagged)
      • unhealthy — nothing loaded yet failures occurred (HTTP 503)
    """
    # Map each tracked model to a JSON-friendly state. Failures include the
    # underlying error message to aid debugging on this internal endpoint.
    models = {}
    for name, state in _MODEL_STATUS.items():
        if state is True:
            models[name] = "loaded"
        elif state is None:
            models[name] = "pending"
        else:
            models[name] = {"status": "failed", "error": str(state)}

    loaded = [k for k, v in _MODEL_STATUS.items() if v is True]
    failed = [k for k, v in _MODEL_STATUS.items() if v not in (True, None)]

    if not _WARMUP_DONE.is_set() and not failed:
        status, code = "starting", 200
    elif not failed:
        status, code = "healthy", 200
    elif loaded:
        status, code = "degraded", 200
    else:
        status, code = "unhealthy", 503

    return JSONResponse(
        {
            "status":    status,
            "timestamp": time.time(),
            "models":    models,
        },
        status_code=code,
    )

@app.exception_handler(FileNotFoundError)
async def model_not_available(request: Request, exc):
    # A missing model/data artifact (e.g. joblib.load on an untrained model) is a
    # "not ready" condition, not a server bug — surface a clear 503 instead of a
    # 500 with a leaked path.
    logger.warning("Artifact not available on %s %s: %s",
                   request.method, request.url.path, exc)
    return JSONResponse(
        {"error": "model or data not available — has training run?"},
        status_code=503,
    )

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse({"error": "endpoint not found", "docs": "/docs"}, status_code=404)

@app.exception_handler(500)
async def server_error(request: Request, exc):
    # Log the real detail server-side; return an opaque message to the client so
    # stack traces, file paths, and internals are never exposed.
    logger.exception("Unhandled server error on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"error": "internal server error"},
        status_code=500,
    )
