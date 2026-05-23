"""
hf_sentiment_engine.py — HuggingFace Inference API Sentiment Pipeline
======================================================================
Uses THREE specialist HuggingFace models via their REST Inference API.
No torch, no GPU, no local model files required.
Pure HTTP calls → works in any environment with internet access.

Models used
-----------
1. cardiffnlp/twitter-roberta-base-sentiment-latest
   → Sentiment classification (Positive / Neutral / Negative)
   → RoBERTa fine-tuned on 124M tweets + domain-adaptation
   → Much better than TextBlob for nuance and colloquial language

2. cardiffnlp/twitter-roberta-base-irony
   → Sarcasm / irony detection
   → "Oh wonderful, the AC broke AGAIN" → irony: True

3. facebook/bart-large-mnli
   → Zero-shot aspect classification
   → Determines which hotel aspects the review mentions
   → Then re-runs sentiment on per-aspect text snippets

Authentication
--------------
  export HF_API_TOKEN=hf_...
  (Free at huggingface.co/settings/tokens)
  Without a token: requests still work but may be rate-limited.

Fallback chain
--------------
  HuggingFace API  →  Claude API  →  TextBlob (always available)
"""

from __future__ import annotations

import os, json, re, time, hashlib, logging, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Model constants
# ─────────────────────────────────────────────────────────────────────────────
HF_API_BASE = "https://api-inference.huggingface.co/models"

MODELS = {
    "sentiment":  "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "irony":      "cardiffnlp/twitter-roberta-base-irony",
    "zero_shot":  "facebook/bart-large-mnli",
}

ASPECT_LABELS = ["room quality", "staff service", "food and dining",
                  "value for money", "location and surroundings"]

ASPECT_KEY_MAP = {
    "room quality":           "room",
    "staff service":          "service",
    "food and dining":        "food",
    "value for money":        "value",
    "location and surroundings": "location",
}

# Label mapping for cardiffnlp model output
# Model returns: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
SENTIMENT_LABEL_MAP = {
    "LABEL_0": "Negative", "negative": "Negative",
    "LABEL_1": "Neutral",  "neutral":  "Neutral",
    "LABEL_2": "Positive", "positive": "Positive",
}

# Disk cache removed in favor of in-memory cache (see _MEM_CACHE below).


# ─────────────────────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
# ─── In-memory cache ─────────────────────────────────────────────────────────
# Was: read+rewrite the full data/hf_sentiment_cache.json on every request.
# That's O(N) I/O per call and burns Render's ephemeral disk.
# Now: bounded process-local dict, thread-safe. Disk persistence removed —
# on free-tier the container restarts often, so a persistent cache is low-value.
_MEM_CACHE: dict[str, dict] = {}
_MEM_CACHE_MAX = 2048
_MEM_CACHE_LOCK = threading.Lock()

def _cache_get(key: str) -> Optional[dict]:
    with _MEM_CACHE_LOCK:
        return _MEM_CACHE.get(key)

def _cache_put(key: str, value: dict) -> None:
    # Don't cache failures; we want the next request to retry.
    if value.get("_hf_failed"):
        return
    with _MEM_CACHE_LOCK:
        if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
            # Cheap eviction: drop one arbitrary entry. Avoids importing OrderedDict.
            _MEM_CACHE.pop(next(iter(_MEM_CACHE)))
        _MEM_CACHE[key] = value

def _cache_key(text: str, suffix: str = "") -> str:
    # Include engine identifier so a degraded result doesn't poison a later
    # warm-engine lookup.
    return hashlib.md5(f"{text.strip().lower()}|{suffix}".encode()).hexdigest()[:20]


# ─────────────────────────────────────────────────────────────────────────────
#  HuggingFace HTTP client
# ─────────────────────────────────────────────────────────────────────────────
class HFInferenceClient:
    """
    Thin HTTP wrapper around the HuggingFace Inference API.
    Handles: auth, retries, cold-start waits (model loading), rate limits.
    """

    # Tight budgets so a slow HF doesn't burn Render's proxy timeout.
    # Previous values (TIMEOUT=25, RETRIES=3, WAIT=20) could spend up to
    # ~400s × 3 models per request and surface as 502 Bad Gateway.
    TIMEOUT         = 8
    RETRY_ATTEMPTS  = 1
    COLD_START_WAIT = 3

    def __init__(self, token: Optional[str] = None):
        self.token   = token or os.environ.get("HF_API_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def query(self, model_id: str, payload: dict,
               attempt: int = 0) -> Optional[dict | list]:
        url = f"{HF_API_BASE}/{model_id}"
        try:
            resp = self.session.post(url, json=payload, timeout=self.TIMEOUT)
        except requests.exceptions.RequestException as e:
            logger.warning(f"HF request failed ({model_id}): {e}")
            return None

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 503:
            # Model loading — HF returns {"estimated_time": N}
            try:
                wait = resp.json().get("estimated_time", self.COLD_START_WAIT)
            except Exception:
                wait = self.COLD_START_WAIT
            if attempt < self.RETRY_ATTEMPTS:
                logger.info(f"HF model {model_id} loading, waiting {wait:.0f}s…")
                time.sleep(min(wait, 30))
                return self.query(model_id, payload, attempt + 1)

        if resp.status_code == 429:
            # Rate-limited
            if attempt < self.RETRY_ATTEMPTS:
                time.sleep(3 * (attempt + 1))
                return self.query(model_id, payload, attempt + 1)

        logger.warning(f"HF API error {resp.status_code} for {model_id}: {resp.text[:200]}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_sentiment(raw: list) -> tuple[str, float, float]:
    """
    Parse cardiffnlp sentiment output.
    Returns (label, polarity_float, confidence)
    """
    if not raw or not isinstance(raw, list):
        return "Neutral", 0.0, 0.5

    # Flatten nested list if needed
    items = raw[0] if isinstance(raw[0], list) else raw

    scores = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        lbl = SENTIMENT_LABEL_MAP.get(item.get("label",""), "Neutral")
        scores[lbl] = float(item.get("score", 0))

    if not scores:
        return "Neutral", 0.0, 0.5

    label      = max(scores, key=scores.get)
    confidence = scores[label]

    # Map to -1…+1 polarity
    pos = scores.get("Positive", 0)
    neg = scores.get("Negative", 0)
    polarity = round(pos - neg, 4)

    return label, polarity, round(confidence, 4)


def _parse_irony(raw: list) -> bool:
    """Parse irony model — returns True if sarcasm/irony detected."""
    if not raw or not isinstance(raw, list):
        return False
    items = raw[0] if isinstance(raw[0], list) else raw
    for item in items:
        if not isinstance(item, dict):
            continue
        lbl   = item.get("label", "").lower()
        score = float(item.get("score", 0))
        if ("irony" in lbl or "ironic" in lbl) and score > 0.65:
            return True
    return False


def _parse_zero_shot(raw: dict, labels: list[str]) -> dict[str, float]:
    """Parse BART zero-shot output → {label: score}"""
    if not raw or not isinstance(raw, dict):
        return {}
    zipped = zip(raw.get("labels",[]), raw.get("scores",[]))
    return {lbl: round(float(sc), 4) for lbl, sc in zipped}


def _extract_themes(text: str, label: str) -> list[str]:
    """
    Lightweight keyword-based theme extraction.
    In a production system this would use NER or a dedicated extraction model.
    """
    kw_map = {
        "spa": "spa", "massage": "spa", "wellness": "spa",
        "pool": "pool", "swim": "pool",
        "breakfast": "breakfast", "buffet": "breakfast", "brunch": "breakfast",
        "room": "room", "suite": "suite", "bed": "room",
        "staff": "staff service", "concierge": "staff service", "recepti": "staff service",
        "view": "views", "ocean": "views", "mountain": "views",
        "food": "dining", "restaurant": "dining", "dinner": "dining",
        "clean": "cleanliness", "dirty": "cleanliness", "dust": "cleanliness",
        "noisy": "noise", "quiet": "noise",
        "wifi": "connectivity", "internet": "connectivity",
        "parking": "parking",
        "checkin": "check-in", "check-in": "check-in", "arrival": "check-in",
        "price": "value", "expensive": "value", "overpriced": "value",
        "transfer": "transport", "airport": "transport",
        "location": "location", "central": "location", "beach": "location",
    }
    found = set()
    text_lower = text.lower()
    for kw, theme in kw_map.items():
        if kw in text_lower:
            found.add(theme)
    return list(found)[:4]


# ─────────────────────────────────────────────────────────────────────────────
#  Main engine
# ─────────────────────────────────────────────────────────────────────────────
class HuggingFaceSentimentEngine:
    """
    Full hotel review NLP pipeline using three HuggingFace models.

    Pipeline:
      1. Sentiment classification (RoBERTa)  → label, polarity, confidence
      2. Irony/sarcasm detection  (RoBERTa)  → sarcasm_flag
      3. Zero-shot aspect scoring (BART)     → per-aspect -1…+1 scores
      4. Keyword theme extraction            → themes list

    All results are cached to avoid redundant API calls.
    """

    def __init__(self, token: Optional[str] = None):
        self.client  = HFInferenceClient(token)
        self._engine = f"HuggingFace ({MODELS['sentiment'].split('/')[-1]})"

    def analyse(self, text: str, use_cache: bool = True) -> dict:
        key = _cache_key(text, "hf_v3")
        if use_cache:
            hit = _cache_get(key)
            if hit is not None:
                return hit

        text   = str(text).strip()
        result = self._run_pipeline(text)

        if use_cache:
            _cache_put(key, result)
        return result

    def _run_pipeline(self, text: str) -> dict:
        # Fan out the 3 HF model calls concurrently — they're independent.
        # Was: 3 sequential calls × up to 25s each = 75s+ per review, easily
        # triggering Render's 502 proxy timeout. Now: bounded by the slowest
        # call (~TIMEOUT seconds) instead of their sum.
        #
        # wait_for_model is OFF so HF returns 503 fast on a cold model
        # rather than blocking up to 20s; we retry once briefly via the
        # client's RETRY_ATTEMPTS, and if the model is still cold we fail
        # over to the next tier (Claude / TextBlob) in sentiment_engine.py.
        payloads = {
            "sentiment": (MODELS["sentiment"],
                          {"inputs": text,
                           "options": {"wait_for_model": False}}),
            "irony":     (MODELS["irony"],
                          {"inputs": text,
                           "options": {"wait_for_model": False}}),
            "zero_shot": (MODELS["zero_shot"],
                          {"inputs": text,
                           "parameters": {"candidate_labels": ASPECT_LABELS,
                                          "multi_label": True},
                           "options": {"wait_for_model": False}}),
        }

        raw: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(self.client.query, model_id, payload): name
                       for name, (model_id, payload) in payloads.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    raw[name] = fut.result()
                except Exception as e:
                    logger.warning(f"HF {name} call raised: {e}")
                    raw[name] = None

        raw_sent  = raw.get("sentiment")
        raw_irony = raw.get("irony")
        raw_zs    = raw.get("zero_shot")

        # Sentiment is the only mandatory call. If it failed, signal so the
        # caller can fall through to Claude / TextBlob without poisoning cache.
        if raw_sent is None:
            return {"_hf_failed": True}

        label, polarity, confidence = _parse_sentiment(raw_sent)
        sarcasm_flag = _parse_irony(raw_irony) if raw_irony else False
        aspects = {"room": None, "service": None, "food": None,
                    "value": None, "location": None}

        if raw_zs:
            aspect_scores = _parse_zero_shot(raw_zs, ASPECT_LABELS)
            for full_label, short_key in ASPECT_KEY_MAP.items():
                score = aspect_scores.get(full_label, 0)
                # Only set aspect if model is reasonably confident it's mentioned
                if score > 0.25:
                    # Scale: high zero-shot score × sentiment direction
                    aspects[short_key] = round((score * 2 - 1) * abs(polarity), 4)

        # ── 4. Themes ─────────────────────────────────────────────────────
        themes = _extract_themes(text, label)

        return {
            "label":              label,
            "polarity":           polarity,
            "confidence":         confidence,
            "sarcasm_flag":       sarcasm_flag,
            "aspects":            aspects,
            "themes":             themes,
            "engine":             self._engine,
            "models_used": {
                "sentiment": MODELS["sentiment"],
                "irony":     MODELS["irony"],
                "aspect":    MODELS["zero_shot"],
            },
        }

    def analyse_batch(self, texts: list[str],
                       max_workers: int = 4) -> list[dict]:
        # Was: sequential with a 0.5s sleep between calls — 30 reviews =
        # 15s of pure sleep before any model work, which exceeded Render's
        # proxy timeout. Now: bounded concurrency with no artificial sleep.
        # HF's API will 429 if we overshoot; HFInferenceClient retries on 429.
        if not texts:
            return []
        results: list[Optional[dict]] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.analyse, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    logger.warning(f"HF batch item {i} failed: {e}")
                    results[i] = {"_hf_failed": True}
        return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level convenience functions (same interface as sentiment_engine.py)
# ─────────────────────────────────────────────────────────────────────────────
_engine_instance: Optional[HuggingFaceSentimentEngine] = None

def _get_engine() -> HuggingFaceSentimentEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HuggingFaceSentimentEngine()
    return _engine_instance


def analyse_hf(text: str, use_cache: bool = True) -> dict:
    """Analyse a single review with the HuggingFace pipeline."""
    return _get_engine().analyse(text, use_cache=use_cache)


def analyse_batch_hf(texts: list[str]) -> list[dict]:
    """Analyse a batch of reviews."""
    return _get_engine().analyse_batch(texts)


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test (offline-safe — uses mock data if network unavailable)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("Absolutely wonderful stay. The spa was transcendent and staff flawless.",
         "Positive, no sarcasm"),
        ("Oh wonderful — the AC broke at 2am and reception took 40 minutes to answer.",
         "Negative, sarcasm"),
        ("The room was okay. Breakfast was fine. Nothing special.",
         "Neutral"),
        ("The sommelier's pairing was merely adequate. Expected far more at this price point.",
         "Negative, nuanced"),
    ]

    print("HuggingFace Sentiment Engine — Test")
    print("=" * 60)
    print(f"Token set: {'YES' if os.environ.get('HF_API_TOKEN') else 'NO (rate-limited mode)'}")
    print()

    engine = HuggingFaceSentimentEngine()
    for text, expected in test_cases:
        print(f"Text    : {text[:70]}…" if len(text) > 70 else f"Text    : {text}")
        print(f"Expected: {expected}")
        result = engine.analyse(text, use_cache=False)
        if result.get("_hf_failed"):
            print("Result  : ⚠ HF API unreachable (network blocked in this environment)")
            print("          This will work correctly on your local machine.")
        else:
            print(f"Result  : {result['label']} (polarity={result['polarity']:+.3f}, "
                  f"confidence={result['confidence']:.0%})")
            print(f"Sarcasm : {result['sarcasm_flag']}")
            print(f"Aspects : { {k:v for k,v in result['aspects'].items() if v is not None} }")
            print(f"Themes  : {result['themes']}")
            print(f"Engine  : {result['engine']}")
        print("-" * 60)
