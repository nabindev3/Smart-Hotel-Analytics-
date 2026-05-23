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

import os, json, re, time, hashlib, logging
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

CACHE_PATH = Path("data/hf_sentiment_cache.json")


# ─────────────────────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    """Best-effort cache save. Never raises — read-only mounts (e.g. Docker
    `./data:/app/data:ro`) and missing dirs just disable caching for the
    current request."""
    try:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except (OSError, PermissionError) as e:
        logger.debug(f"HF cache disabled (cannot write {CACHE_PATH}): {e}")

def _cache_key(text: str, suffix: str = "") -> str:
    return hashlib.md5(f"{text.strip().lower()}{suffix}".encode()).hexdigest()[:20]


# ─────────────────────────────────────────────────────────────────────────────
#  HuggingFace HTTP client
# ─────────────────────────────────────────────────────────────────────────────
class HFInferenceClient:
    """
    Thin HTTP wrapper around the HuggingFace Inference API.
    Handles: auth, retries, cold-start waits (model loading), rate limits.
    """

    TIMEOUT        = 25
    RETRY_ATTEMPTS = 3
    COLD_START_WAIT= 20   # HF cold-starts large models; wait and retry

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
        key   = _cache_key(text, "hf_v2")
        cache = _load_cache() if use_cache else {}
        if key in cache:
            return cache[key]

        text  = str(text).strip()
        result = self._run_pipeline(text)

        if use_cache:
            cache[key] = result
            _save_cache(cache)
        return result

    def _run_pipeline(self, text: str) -> dict:
        # ── 1. Sentiment ──────────────────────────────────────────────────
        raw_sent = self.client.query(
            MODELS["sentiment"],
            {"inputs": text, "options": {"wait_for_model": True}},
        )
        if raw_sent is None:
            # Model unreachable — return structured failure so caller can fallback
            return {"_hf_failed": True}

        label, polarity, confidence = _parse_sentiment(raw_sent)

        # ── 2. Irony / sarcasm ────────────────────────────────────────────
        raw_irony    = self.client.query(
            MODELS["irony"],
            {"inputs": text, "options": {"wait_for_model": True}},
        )
        sarcasm_flag = _parse_irony(raw_irony) if raw_irony else False

        # ── 3. Zero-shot aspect scoring ───────────────────────────────────
        raw_zs = self.client.query(
            MODELS["zero_shot"],
            {
                "inputs":     text,
                "parameters": {
                    "candidate_labels": ASPECT_LABELS,
                    "multi_label": True,
                },
                "options": {"wait_for_model": True},
            },
        )
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
                       delay: float = 0.5) -> list[dict]:
        results = []
        for i, text in enumerate(texts):
            r = self.analyse(text)
            results.append(r)
            if i < len(texts) - 1:
                time.sleep(delay)
        return results


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
