"""Sidebar: brand, live 30-day KPIs, system status, data quality."""
from __future__ import annotations

import os

import requests
import streamlit as st
from api import PUBLIC_API_URL, cached_get
from theme import CREAM, TEXT, gold_rule, kpi_color
from ui import kpi_cell, section_label, svc_row

# Optional MLflow UI to show in the status panel. Unset by default so the
# dashboard never blocks on a localhost:5001 ping in a cloud deploy (where it
# always times out). Set MLFLOW_UI_URL to a reachable MLflow server to enable.
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "").rstrip("/")


@st.cache_data(show_spinner=False, ttl=30)
def _mlflow_ok(url: str):
    """Cached health ping for the MLflow UI. Returns None when not configured."""
    if not url:
        return None
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def render() -> dict | None:
    """Draw the sidebar and return the 30-day KPI summary (shared with header)."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:.4rem 0'>"
            "<span style='font-size:2rem'>🏨</span>"
            "<div style='font-family:Cormorant Garamond,serif;font-size:1.2rem;"
            "color:#EDD98A;font-weight:600;letter-spacing:.05em'>Smart Hotel Analytics</div>"
            "<div style='color:#767670;font-size:.65rem;letter-spacing:.12em'>MANAGER DASHBOARD</div>"
            "</div>", unsafe_allow_html=True)
        gold_rule()

        # ── Live 30-day KPI grid ──────────────────────────────────────────────
        section_label("📊", "LAST 30 DAYS")
        kpi_s, _ = cached_get("/api/v1/forecast/kpis/summary")
        if kpi_s:
            occ  = kpi_s["avg_occupancy"]
            adr  = kpi_s["avg_adr"]
            rev  = kpi_s["avg_revpar"]
            canc = kpi_s.get("avg_cancel_rate", 0)
            st.markdown(
                "<div style='display:grid;grid-template-columns:1fr 1fr;gap:.35rem;margin:.5rem 0'>"
                + kpi_cell("ROOMS FILLED",     f"{occ:.0%}",  kpi_color("occupancy", occ))
                + kpi_cell("AVG NIGHTLY RATE", f"${adr:.0f}", kpi_color("adr", adr))
                + kpi_cell("$ PER ROOM",       f"${rev:.0f}", CREAM)
                + kpi_cell("NO-SHOW RATE",     f"{canc:.0%}", kpi_color("cancel_rate", canc))
                + "</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:#444440;font-size:.62rem;text-align:right'>"
                f"{kpi_s.get('period_start', '?')} → {kpi_s.get('period_end', '?')}</div>",
                unsafe_allow_html=True)
        else:
            st.caption("Loading…")
        gold_rule()

        # ── System health ─────────────────────────────────────────────────────
        section_label("🖥️", "SYSTEM STATUS")
        health, _ = cached_get("/health")
        backend_ok = bool(health and health.get("status") in ("healthy", "degraded"))
        rows = [
            svc_row("☁️", "Analytics engine", "API", backend_ok, f"{PUBLIC_API_URL}/docs"),
            svc_row("🎨", "Dashboard", "UI", True, "#"),  # you're here; no external link
        ]
        # Only show the model-registry row when an MLflow UI is configured —
        # otherwise we'd block every render pinging a localhost URL that isn't there.
        mlflow_ok = _mlflow_ok(MLFLOW_UI_URL)
        if mlflow_ok is not None:
            rows.append(svc_row("📊", "Model registry", "MLflow", mlflow_ok, MLFLOW_UI_URL))
        st.markdown("".join(rows), unsafe_allow_html=True)
        gold_rule()

        # ── Data quality ──────────────────────────────────────────────────────
        section_label("📦", "DATA QUALITY")
        dq, _ = cached_get("/api/v1/briefing/today")
        if dq and dq.get("data_quality"):
            q = dq["data_quality"]
            real_pct = q.get("real_share", 0) * 100
            st.markdown(
                f"<div style='font-size:.7rem;color:{TEXT};line-height:1.7'>"
                f"<b>{q.get('rows', '—'):,}</b> bookings<br>"
                f"<b>{real_pct:.0f}%</b> real-world data<br>"
                f"<b>{q.get('date_min', '?')}</b> → <b>{q.get('date_max', '?')}</b><br>"
                f"<b>{q.get('n_countries', '?')}</b> countries"
                f"</div>", unsafe_allow_html=True)
        else:
            st.caption("Run `python src/load_real_data.py` to add real bookings")
        gold_rule()

        st.markdown(
            f"<div style='font-size:.72rem;line-height:2'>"
            f"<a href='{PUBLIC_API_URL}/docs' target='_blank' style='color:#C5C5BF;text-decoration:none'>"
            f"📖 Developer docs</a></div>", unsafe_allow_html=True)

    return kpi_s
