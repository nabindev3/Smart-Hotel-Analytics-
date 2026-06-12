"""
frontend/app.py — Smart Hotel Analytics Dashboard (entry point)
===============================================================
Hotel-manager-friendly Streamlit front end. All data is fetched from the FastAPI
backend at $API_BASE.

This file is intentionally thin: it wires together the shared modules
(``api``, ``theme``, ``ui``) and the per-tab render modules under ``sections/``.
Each tab's logic lives in its own module so this file stays a table of contents.

Run:
  export API_BASE=http://localhost:8000
  streamlit run frontend/app.py
"""
import os
import sys

# `streamlit run frontend/app.py` puts this folder on sys.path automatically;
# add it explicitly so the sibling modules (theme/api/ui/sections) also resolve
# under other launchers (tests, `python -m`, etc.).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from sections import (
    briefing,
    cancellation,
    diagnostics,
    forecast,
    header,
    offers,
    origins,
    pricing,
    reviews,
    sidebar,
)
from theme import load_css

st.set_page_config(
    page_title="Smart Hotel Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# Sidebar fetches the 30-day KPI summary once and shares it with the header.
kpi_s = sidebar.render()
header.render(kpi_s)

# One module per tab — same order and labels as before.
TABS = [
    ("🌅  Today's Briefing",        briefing.render),
    ("📈  Demand Outlook",           forecast.render),
    ("🚪  Will They Show Up?",       cancellation.render),
    ("💰  Pricing & Inventory",      pricing.render),
    ("🎯  Guest Offers",             offers.render),
    ("💬  Guest Reviews",            reviews.render),
    ("🌍  Where Guests Come From",   origins.render),
    ("🔬  Diagnostics",              diagnostics.render),
]

for tab, render in zip(st.tabs([label for label, _ in TABS]), (fn for _, fn in TABS)):
    with tab:
        render()
