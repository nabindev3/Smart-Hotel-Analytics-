"""
Smoke test for the Streamlit dashboard.

Runs the whole app headlessly via Streamlit's AppTest against a dead backend
(API_RETRIES=1 + an unreachable host, so every call fails instantly and the app
renders its error states). This catches import/wiring regressions across the
refactored frontend modules without needing a live backend or a browser.

streamlit/plotly are frontend-only deps; skip cleanly if they aren't installed.
"""
import os

import pytest

# Make every backend call fail fast, before the app module is imported by AppTest.
os.environ.setdefault("API_RETRIES", "1")
os.environ.setdefault("API_BASE", "http://127.0.0.1:9")
os.environ.setdefault("PUBLIC_API_URL", "http://127.0.0.1:9")

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "frontend", "app.py")


def test_dashboard_runs_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    # 8 top-level tabs + 3 sub-tabs in the "origins" tab = 11.
    assert len(at.tabs) == 11
    # It should surface errors (backend is down) rather than crash.
    assert len(at.error) >= 1
