"""
frontend/ui.py — reusable presentational components
===================================================
The dashboard repeats the same handful of HTML snippets (metric cards, alerts,
bordered result cards, sidebar cells). They live here as functions instead of
being pasted inline ~50 times, so there's one place to change the markup — and
one place that escapes free text.

``esc`` is applied to anything that originates as free text (review topics,
recommendation copy, alert detail, model strategy strings). Those come from the
backend today, but escaping them means a stray ``<`` in a future real-data field
can't break the layout or inject markup.
"""
from __future__ import annotations

from html import escape as esc

import streamlit as st
from theme import GREEN, MUTED, RED


def help_box(text: str) -> None:
    st.markdown(f"<div class='help-box'>💡 {esc(str(text))}</div>", unsafe_allow_html=True)


def clean_country(raw: str) -> str:
    """Normalise a country code to 3 upper-case letters, warning the user when
    the input clearly isn't a valid code instead of silently garbling it."""
    cc = (raw or "").strip().upper()[:3]
    if len(cc) != 3 or not cc.isalpha():
        st.warning(f"Country should be a 3-letter code (e.g. GBR, USA). Using '{cc or '—'}'.")
    return cc


def trend_arrow(trend: dict | None) -> str:
    """{direction, delta_pct} → a coloured arrow span (safe HTML, numbers only)."""
    if not trend:
        return ""
    d, p = trend.get("direction", "flat"), trend.get("delta_pct", 0)
    if d == "up":
        return f"<span style='color:{GREEN}'>▲ {p:+.1f}%</span>"
    if d == "down":
        return f"<span style='color:{RED}'>▼ {p:+.1f}%</span>"
    return f"<span style='color:{MUTED}'>● flat</span>"


def metric_card(col, label: str, value: str, trend: dict | None = None) -> None:
    """The KPI card with a trend arrow — used 4× on the briefing (item 9)."""
    trend_html = (f"<div class='mc-trend'>{trend_arrow(trend)} vs prior period</div>"
                  if trend is not None else "")
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='mc-label'>{esc(label)}</div>"
        f"<div class='mc-value'>{esc(value)}</div>"
        f"{trend_html}</div>",
        unsafe_allow_html=True,
    )


_ALERT = {"warning": ("alert-warning", "⚠️"), "good": ("alert-good", "✅")}


def alert(level: str, title: str, detail: str) -> None:
    cls, icon = _ALERT.get(level, ("alert-info", "ℹ️"))
    st.markdown(
        f"<div class='{cls}'><b>{icon} {esc(title)}</b><br>"
        f"<span style='font-size:.85rem'>{esc(detail)}</span></div>",
        unsafe_allow_html=True,
    )


def action_item(idx: int, text: str) -> None:
    from theme import GOLD
    st.markdown(
        f"<div class='action-card'><b style='color:{GOLD}'>{idx}.</b> {esc(text)}</div>",
        unsafe_allow_html=True,
    )


def card(body_html: str, *, border: str, bg: str = "rgba(14,16,24,.9)",
         radius: str = "10px", pad: str = "1rem", margin: str = ".5rem 0") -> str:
    """Generic bordered result card. ``body_html`` is trusted (built by the caller
    from numbers/colours); escape any free text before passing it in."""
    return (f"<div style='background:{bg};border:1px solid {border};"
            f"border-radius:{radius};padding:{pad};margin:{margin}'>{body_html}</div>")


def render_card(target, body_html: str, **kw) -> None:
    """Render a `card` into a Streamlit container (st, a column, etc.)."""
    target.markdown(card(body_html, **kw), unsafe_allow_html=True)


def kpi_cell(label: str, value: str, color: str) -> str:
    """One cell of the sidebar 30-day KPI grid (returns HTML for grid assembly)."""
    return (f"<div style='background:rgba(14,16,24,.9);"
            f"border:1px solid rgba(196,155,60,.15);border-radius:8px;"
            f"padding:.55rem;text-align:center'>"
            f"<div style='color:{MUTED};font-size:.58rem;letter-spacing:.1em'>{esc(label)}</div>"
            f"<div style='color:{color};font-size:1.15rem;"
            f"font-family:Cormorant Garamond,serif;font-weight:600'>{esc(value)}</div></div>")


def svc_row(icon: str, name: str, port: str, ok: bool, url: str) -> str:
    """One row of the sidebar system-status panel (returns HTML)."""
    dot   = f"<span style='color:{'#4CAF50' if ok else '#F44336'}'>●</span>"
    label = f"<span style='color:{'#C5C5BF' if ok else '#555'};font-size:.72rem'> {icon} {esc(name)}</span>"
    link  = (f"<a href='{esc(url)}' target='_blank' style='color:#767670;font-size:.62rem;"
             f"float:right;text-decoration:none'>:{esc(port)} ↗</a>")
    return f"<div style='margin:.28rem 0'>{dot}{label}{link}</div>"


def section_label(icon: str, text: str) -> None:
    """A small gold uppercase label used for sidebar sections."""
    st.markdown(
        f"<div style='color:#EDD98A;font-size:.7rem;letter-spacing:.12em;font-weight:600'>"
        f"{icon} {esc(text)}</div>", unsafe_allow_html=True)
