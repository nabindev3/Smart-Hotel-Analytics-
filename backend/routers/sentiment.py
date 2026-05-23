"""backend/routers/sentiment.py — NLP endpoints with engine info"""
import os, sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.sentiment_engine import (analyse, analyse_batch, get_active_engine,
                                    _textblob_analyse)
from src.hf_sentiment_engine import MODELS as HF_MODELS

router = APIRouter()

# Endpoint-level latency cap. Even with the new in-engine budgets, we never
# want this endpoint to spend more than this before returning *something* —
# Render's proxy starts emitting 502 around 30s of upstream silence.
_REQUEST_DEADLINE_S = 12
_EXEC = ThreadPoolExecutor(max_workers=8)

class ReviewText(BaseModel):
    text: str

class BatchReviews(BaseModel):
    reviews: List[str]

@router.get("/engine-info")
def engine_info():
    """Returns which NLP engine is active and which models are configured."""
    # HuggingFace works with or without a token (free tier)
    hf_available = True
    claude_key= bool(os.environ.get("ANTHROPIC_API_KEY"))
    tier = 1
    return {
        "active_engine":   get_active_engine(),
        "active_tier":     tier,
        "tier_1": {
            "name":      "HuggingFace Inference API",
            "available": hf_available,
            "models": {
                "sentiment": HF_MODELS["sentiment"],
                "irony":     HF_MODELS["irony"],
                "aspect":    HF_MODELS["zero_shot"],
            },
            "capabilities": ["sentiment","sarcasm","aspect-based","confidence"],
            "latency_ms":   "200–800 (cold) / 50–200 (warm)",
            "cost":         "Free with token / rate-limited without",
        },
        "tier_2": {
            "name":      "Anthropic Claude API",
            "available": claude_key,
            "model":     "claude-sonnet-4-20250514",
            "capabilities": ["sentiment","sarcasm","aspect-based","themes","nuance"],
            "latency_ms":   "500–2000",
            "cost":         "$0.003/1k tokens",
        },
        "tier_3": {
            "name":      "TextBlob",
            "available": True,
            "capabilities": ["sentiment","polarity"],
            "latency_ms":   "<1",
            "cost":         "Free (local)",
        },
        "setup": {
            "huggingface": "export HF_API_TOKEN=hf_... (free at huggingface.co/settings/tokens)",
            "claude":      "export ANTHROPIC_API_KEY=sk-ant-...",
        },
    }

def _textblob_fallback(text: str, reason: str) -> dict:
    r = _textblob_analyse(text)
    r["engine"] = f"TextBlob (fallback: {reason})"
    return r

@router.post("/analyse")
def analyse_single(body: ReviewText):
    """
    Single-review analysis with a hard deadline. If the active engine
    (HF / Claude) doesn't return within the request budget, we return a
    TextBlob result instead of letting the request hang until Render's
    proxy returns 502 Bad Gateway.
    """
    import logging, traceback
    log = logging.getLogger("sentiment")
    fut = _EXEC.submit(analyse, body.text)
    try:
        return fut.result(timeout=_REQUEST_DEADLINE_S)
    except FutTimeout:
        # The underlying call keeps running and will populate the in-memory
        # cache; the user just doesn't wait for it on this request.
        log.warning(f"sentiment analyse exceeded {_REQUEST_DEADLINE_S}s; falling back to TextBlob")
        return _textblob_fallback(body.text, f"deadline {_REQUEST_DEADLINE_S}s exceeded")
    except Exception as e:
        log.warning(f"sentiment analyse failed: {e}\n{traceback.format_exc()}")
        return _textblob_fallback(body.text, f"{type(e).__name__}")

@router.post("/analyse-batch")
def analyse_batch_endpoint(body: BatchReviews):
    # Same deadline pattern. Scale budget mildly with batch size but cap it
    # so the request can never block the proxy.
    deadline = min(_REQUEST_DEADLINE_S + 2 * len(body.reviews), 45)
    fut = _EXEC.submit(analyse_batch, body.reviews)
    try:
        results = fut.result(timeout=deadline)
    except FutTimeout:
        # Per-item TextBlob so the client never sees a 502 on /analyse-batch.
        results = [_textblob_fallback(t, f"batch deadline {deadline}s exceeded")
                   for t in body.reviews]
    except Exception as e:
        return {"count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}
    return {"count": len(results), "results": results}
