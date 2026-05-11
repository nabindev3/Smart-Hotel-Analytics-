"""backend/routers/sentiment.py — NLP endpoints with engine info"""
import os, sys
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.sentiment_engine import analyse, analyse_batch, get_active_engine
from src.hf_sentiment_engine import MODELS as HF_MODELS

router = APIRouter()

class ReviewText(BaseModel):
    text: str

class BatchReviews(BaseModel):
    reviews: List[str]

@router.get("/engine-info")
def engine_info():
    """Returns which NLP engine is active and which models are configured."""
    hf_token  = bool(os.environ.get("HF_API_TOKEN"))
    claude_key= bool(os.environ.get("ANTHROPIC_API_KEY"))
    tier = (1 if hf_token else 2 if claude_key else 3)
    return {
        "active_engine":   get_active_engine(),
        "active_tier":     tier,
        "tier_1": {
            "name":      "HuggingFace Inference API",
            "available": hf_token,
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

@router.post("/analyse")
def analyse_single(body: ReviewText):
    """
    Resilient single-review analysis. If the active engine throws (timeout,
    cold start, parse error, read-only cache, etc.), we walk down the tier
    fallback rather than returning a 500.
    """
    import logging, traceback
    log = logging.getLogger("sentiment")
    try:
        return analyse(body.text)
    except Exception as e:
        log.warning(f"sentiment analyse failed: {e}\n{traceback.format_exc()}")
        # Last-ditch TextBlob fallback so the dashboard never sees a 500.
        try:
            from src.sentiment_engine import _textblob_analyse
            r = _textblob_analyse(body.text)
            r["engine"] = f"TextBlob (fallback after error: {type(e).__name__})"
            return r
        except Exception:
            return {
                "label": "Neutral", "polarity": 0.0, "confidence": 0.0,
                "sarcasm_flag": False, "aspects": {}, "themes": [],
                "engine": f"error: {type(e).__name__}: {e}",
            }

@router.post("/analyse-batch")
def analyse_batch_endpoint(body: BatchReviews):
    try:
        results = analyse_batch(body.reviews)
    except Exception as e:
        return {"count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}
    return {"count": len(results), "results": results}
