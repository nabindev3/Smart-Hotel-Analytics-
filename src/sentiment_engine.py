"""
sentiment_engine.py — Unified Sentiment Pipeline (3-Tier Fallback)
Tier 1: HuggingFace Inference API (RoBERTa + BART, no torch)
Tier 2: Anthropic Claude API
Tier 3: TextBlob
"""
from __future__ import annotations
import os, json, re, time, hashlib, logging, threading
from typing import Optional
from pathlib import Path
import pandas as pd
from textblob import TextBlob

logger = logging.getLogger(__name__)

try:
    from src.hf_sentiment_engine import HuggingFaceSentimentEngine
    _HF_MODULE_OK = True
except ImportError:
    try:
        from hf_sentiment_engine import HuggingFaceSentimentEngine
        _HF_MODULE_OK = True
    except ImportError:
        _HF_MODULE_OK = False

try:
    from anthropic import Anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# ─── In-memory cache ─────────────────────────────────────────────────────────
# Was: load+rewrite data/sentiment_cache.json on every request (O(N) per call,
# burns Render's ephemeral disk). Now: bounded in-process dict, thread-safe.
# We also drop the previous "result cached under one key regardless of which
# engine actually produced it" — a TextBlob fallback no longer poisons later
# warm-engine lookups, because failures are never cached.
_MEM_CACHE: dict[str, dict] = {}
_MEM_CACHE_MAX = 2048
_MEM_CACHE_LOCK = threading.Lock()

def _cache_get(key: str) -> Optional[dict]:
    with _MEM_CACHE_LOCK:
        return _MEM_CACHE.get(key)

def _cache_put(key: str, value: dict) -> None:
    if not value or value.get("engine", "").startswith("error:"):
        return
    with _MEM_CACHE_LOCK:
        if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
            _MEM_CACHE.pop(next(iter(_MEM_CACHE)))
        _MEM_CACHE[key] = value

def _key(text): return hashlib.md5(text.strip().lower().encode()).hexdigest()[:20]

def _textblob_analyse(text):
    blob = TextBlob(str(text))
    p, s = blob.sentiment.polarity, blob.sentiment.subjectivity
    return {
        "label": "Positive" if p>0.1 else "Negative" if p<-0.1 else "Neutral",
        "polarity": round(p,4), "confidence": round(min(abs(p)*s+0.35,1.0),4),
        "sarcasm_flag": False,
        "aspects": {"room":None,"service":None,"food":None,"value":None,"location":None},
        "themes": [], "engine": "TextBlob (offline fallback)",
    }

_CLAUDE_SYSTEM = """Hotel sentiment analyst. Return ONLY valid JSON:
{"label":"Positive"|"Neutral"|"Negative","polarity":float,"confidence":float,
"aspects":{"room":float_or_null,"service":float_or_null,"food":float_or_null,"value":float_or_null,"location":float_or_null},
"themes":["theme1"],"sarcasm_flag":boolean}
Handle sarcasm: "Oh wonderful" when AC broke = Negative. "merely adequate" = mildly negative."""

def _claude_analyse(text, client):
    resp = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400,
        system=_CLAUDE_SYSTEM, messages=[{"role":"user","content":f"Review: {text}"}])
    raw = re.sub(r"```json|```","", resp.content[0].text.strip()).strip()
    r = json.loads(raw)
    r["engine"] = "Claude claude-sonnet-4-20250514"
    return r

_hf_engine = None
_claude_client = None

def _get_hf():
    global _hf_engine
    if not _HF_MODULE_OK: return None
    if _hf_engine is None: _hf_engine = HuggingFaceSentimentEngine()
    return _hf_engine

def _get_claude():
    global _claude_client
    if not _ANTHROPIC_OK: return None
    key = os.environ.get("ANTHROPIC_API_KEY","")
    if not key: return None
    if _claude_client is None: _claude_client = Anthropic(api_key=key)
    return _claude_client

def analyse(text: str, use_cache: bool = True) -> dict:
    key = _key(text)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit

    result = None
    # Tier 1: HuggingFace (its own cache is in-memory now too)
    hf = _get_hf()
    if hf:
        try:
            r = hf.analyse(text, use_cache=True)
            if not r.get("_hf_failed"): result = r
        except Exception as e: logger.warning(f"HF failed: {e}")
    # Tier 2: Claude
    if result is None:
        claude = _get_claude()
        if claude:
            try: result = _claude_analyse(text, claude)
            except Exception as e: logger.warning(f"Claude failed: {e}")
    # Tier 3: TextBlob (always succeeds)
    if result is None: result = _textblob_analyse(text)

    if use_cache:
        _cache_put(key, result)
    return result

def analyse_batch(texts, max_workers: int = 4):
    # Was: sequential with a 0.3s sleep between every call — 30 reviews
    # spent ~9s asleep before any work, which alone exceeded Render's
    # proxy timeout. Now: bounded concurrency, no artificial sleep.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if not texts:
        return []
    results: list[Optional[dict]] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyse, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                logger.warning(f"sentiment batch item {i} failed: {e}")
                results[i] = _textblob_analyse(texts[i])
    return results

def enrich_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    results = analyse_batch(df[text_col].tolist())
    df = df.copy()
    df["sentiment_label"]      = [r.get("label","Neutral")   for r in results]
    df["sentiment_polarity"]   = [r.get("polarity",0.0)      for r in results]
    df["sentiment_confidence"] = [r.get("confidence",0.5)    for r in results]
    df["sarcasm_flag"]         = [r.get("sarcasm_flag",False) for r in results]
    df["themes"]               = [", ".join(r.get("themes",[]))  for r in results]
    df["aspect_room"]          = [r.get("aspects",{}).get("room")      for r in results]
    df["aspect_service"]       = [r.get("aspects",{}).get("service")   for r in results]
    df["aspect_food"]          = [r.get("aspects",{}).get("food")      for r in results]
    df["aspect_value"]         = [r.get("aspects",{}).get("value")     for r in results]
    df["aspect_location"]      = [r.get("aspects",{}).get("location")  for r in results]
    df["nlp_engine"]           = [r.get("engine","")          for r in results]
    return df

def get_active_engine() -> str:
    if _get_hf() is not None: return "HuggingFace (RoBERTa + BART)"
    if _get_claude() is not None: return "Claude API (Anthropic)"
    return "TextBlob (offline fallback)"

if __name__ == "__main__":
    print(f"Active engine: {get_active_engine()}")
    r = analyse("Absolutely wonderful stay. Spa transcendent.")
    print(r)
