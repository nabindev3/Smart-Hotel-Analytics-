"""
frontend/api.py — thin client for the analytics backend
=======================================================
One place for the base URL, the cold-start retry policy, and the GET/POST
helpers. ``get_schema`` (item 10) pulls the categorical form options from the
backend so the UI dropdowns can't drift out of sync with the model's enums.
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

# API_BASE is where the dashboard makes its calls; PUBLIC_API_URL is only used to
# build user-facing links (e.g. the /docs link) that must work from the browser.
# In a single-host deploy they're the same; behind a proxy they can differ.
API_BASE       = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", "http://localhost:8000").rstrip("/")

# Short-lived cache so reruns (every slider move / tab switch) don't re-fire the
# same read-only GETs. Tunable; 0 disables.
CACHE_TTL = int(os.environ.get("API_CACHE_TTL", "60"))

# Render's free tier sleeps after ~15 min idle; the first hit often returns
# 502/503/504 while the container restarts. Retry with backoff. Configurable so
# tests (and impatient local runs) can fail fast: API_RETRIES=1 → no waiting.
_RETRIES = int(os.environ.get("API_RETRIES", "4"))
_COLD_START_STATUSES = {502, 503, 504}


def _request_with_retry(method, url, *, retries=None, timeout=60, **kwargs):
    retries = _RETRIES if retries is None else retries
    delay, last_err = 2, None
    for attempt in range(retries):
        try:
            r = requests.request(method, url, timeout=timeout, **kwargs)
            if r.status_code in _COLD_START_STATUSES and attempt < retries - 1:
                time.sleep(delay); delay *= 2
                continue
            r.raise_for_status()
            return r.json(), None
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2
                continue
            return None, f"Service is waking up, please retry in a moment ({e})"
        except Exception as e:
            return None, str(e)
    return None, str(last_err) if last_err else "Service unavailable"


def api_get(path, params=None, timeout=60):
    return _request_with_retry("GET", f"{API_BASE}{path}", params=params, timeout=timeout)


def api_post(path, body, params=None, timeout=60):
    return _request_with_retry("POST", f"{API_BASE}{path}", json=body, params=params, timeout=timeout)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def _cached_get(path: str, params_items, timeout: int):
    params = dict(params_items) if params_items else None
    return api_get(path, params=params, timeout=timeout)


def cached_get(path, params=None, timeout=60):
    """GET with a short-lived cache keyed on path+params. Use for read-only data
    that doesn't need sub-minute freshness, so reruns don't re-hit the backend."""
    items = tuple(sorted(params.items())) if params else None
    return _cached_get(path, items, timeout)


# Categorical form options come from the backend so the dropdowns stay in lockstep
# with the model's expected enums (item 10). Falls back to a baked-in schema if
# the endpoint is unavailable, so the form still renders offline.
_SCHEMA_FALLBACK = {
    "hotel": ["Resort Hotel", "City Hotel"],
    "arrival_date_month": ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"],
    "meal": ["BB", "HB", "FB", "SC", "Undefined"],
    "market_segment": ["Online TA", "Direct", "Corporate", "Offline TA/TO",
                       "Groups", "Complementary", "Aviation"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
    "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Transient", "Contract", "Transient-Party", "Group"],
}


def get_schema() -> dict:
    """Return categorical field options from the backend (cached), falling back
    to a baked-in copy if the endpoint can't be reached."""
    data, err = cached_get("/api/v1/cancellation/schema", timeout=10)
    if err or not isinstance(data, dict) or "fields" not in data:
        return _SCHEMA_FALLBACK
    return {**_SCHEMA_FALLBACK, **data["fields"]}
