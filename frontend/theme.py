"""
frontend/theme.py — single source of visual truth
==================================================
Colours, KPI threshold bands, the shared Plotly layout + ``themed_figure``
helper, and the one-shot stylesheet loader. Everything that used to be a magic
hex constant or a copy-pasted ``fig.update_layout(**LAYOUT, ...)`` lives here so
the sidebar, header, and every chart agree on one palette and one look.
"""
from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────
GOLD  = "#BF9740"
BLUE  = "#3A7BD5"
GREEN = "#4CAF50"
RED   = "#F44336"
AMBER = "#FFC107"
CREAM = "#EDD98A"   # headings / emphasis
TEXT  = "#C5C5BF"   # body text
MUTED = "#767670"   # secondary text

LOYALTY_COLORS = {"Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32"}

# ── KPI threshold bands (single source for the sidebar + header colouring) ─────
# Each entry: value >= good → GREEN, >= warn → AMBER, else RED.
# `inverse=True` flips it (lower is better, e.g. cancellation rate).
_THRESHOLDS = {
    "occupancy":   dict(good=0.70, warn=0.50),
    "adr":         dict(good=100,  warn=70),
    "revpar":      dict(good=0,    warn=0),          # informational — always cream
    "cancel_rate": dict(good=0.20, warn=0.35, inverse=True),
}


def kpi_color(kind: str, value: float) -> str:
    """Return the band colour for a KPI value, per the thresholds above."""
    t = _THRESHOLDS.get(kind)
    if not t:
        return CREAM
    if kind == "revpar":
        return CREAM
    good, warn, inverse = t["good"], t["warn"], t.get("inverse", False)
    if inverse:
        return GREEN if value <= good else AMBER if value <= warn else RED
    return GREEN if value >= good else AMBER if value >= warn else RED


# ── Plotly layout + helper (item 7: centralised themed_figure) ─────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,11,17,.75)",
    font=dict(color=TEXT, family="DM Sans", size=12),
    title_font=dict(color=CREAM, family="Cormorant Garamond", size=17),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=48, b=35, l=38, r=18),
    xaxis=dict(gridcolor="rgba(196,155,60,.07)"),
    yaxis=dict(gridcolor="rgba(196,155,60,.07)"),
)


def themed_figure(fig: go.Figure | None = None, **overrides) -> go.Figure:
    """Apply the shared theme to a figure, with per-chart overrides merged on
    top. Use this everywhere instead of spreading ``**LAYOUT`` by hand, so no
    chart can silently drift (e.g. a gauge overriding paper_bgcolor)."""
    fig = fig if fig is not None else go.Figure()
    fig.update_layout(**{**LAYOUT, **overrides})
    return fig


# ── Stylesheet (item 6: load static CSS once instead of ~50 inline blocks) ─────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")


def load_css() -> None:
    with open(_CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def gold_rule() -> None:
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
