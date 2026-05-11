"""
sentiment_engine.py — Unified Sentiment Pipeline (3-Tier Fallback)
Tier 1: HuggingFace Inference API (RoBERTa + BART, no torch)
Tier 2: Anthropic Claude API
Tier 3: TextBlob
"""
from __future__ import annotations
import os, json, re, time, hashlib, logging
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

CACHE_PATH = Path("data/sentiment_cache.json")

def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f: return json.load(f)
        except: pass
    return {}

def _save_cache(cache):
    """Best-effort cache save. Never raises — read-only mounts (e.g. Docker
    `./data:/app/data:ro`) and missing dirs just disable caching for the
    current request."""
    try:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except (OSError, PermissionError) as e:
        logger.debug(f"sentiment cache disabled (cannot write {CACHE_PATH}): {e}")

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
    if not _HF_MODULE_OK or not os.environ.get("HF_API_TOKEN"): return None
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
    cache = _load_cache() if use_cache else {}
    if key in cache: return cache[key]
    result = None
    # Tier 1: HuggingFace
    hf = _get_hf()
    if hf:
        try:
            r = hf.analyse(text, use_cache=False)
            if not r.get("_hf_failed"): result = r
        except Exception as e: logger.warning(f"HF failed: {e}")
    # Tier 2: Claude
    if result is None:
        claude = _get_claude()
        if claude:
            try: result = _claude_analyse(text, claude)
            except Exception as e: logger.warning(f"Claude failed: {e}")
    # Tier 3: TextBlob
    if result is None: result = _textblob_analyse(text)
    if use_cache:
        cache[key] = result
        _save_cache(cache)
    return result

def analyse_batch(texts, delay=0.3):
    results = []
    for i, text in enumerate(texts):
        r = analyse(text)
        results.append(r)
        if ("HuggingFace" in r.get("engine","") or "Claude" in r.get("engine","")) and i < len(texts)-1:
            time.sleep(delay)
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
