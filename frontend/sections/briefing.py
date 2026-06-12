"""Tab: Today's Briefing — headline KPIs, alerts, suggested actions, 7-day outlook."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api import api_get
from theme import GOLD, gold_rule, themed_figure
from ui import action_item, alert, help_box, metric_card


def render() -> None:
    st.markdown("### 🌅  Your Morning Briefing")
    help_box("Everything you need to know before your shift starts. "
             "Numbers cover the last 7 days, with arrows showing how each one is moving.")

    period = st.slider("How many days to summarise?", 1, 30, 7, key="brf_period")
    brief, err = api_get("/api/v1/briefing/today", params={"horizon_days": period})

    if err:
        st.error(f"Could not reach the analytics engine: {err}")
        return
    if not brief:
        return

    h, tr = brief["headline"], brief["trend"]

    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "ROOMS FILLED",     f"{h['occupancy']:.0%}",    tr["occupancy"])
    metric_card(m2, "AVG NIGHTLY RATE", f"${h['adr']:.0f}",         tr["adr"])
    metric_card(m3, "REVENUE / ROOM",   f"${h['revpar']:.0f}",      tr["revpar"])
    metric_card(m4, "NO-SHOW RATE",     f"{h['cancel_rate']:.0%}",  tr["cancel_rate"])

    gold_rule()

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### Things to look at today")
        for a in brief["alerts"]:
            alert(a["level"], a["title"], a["detail"])

        st.markdown("#### Three things you could do today")
        for i, action in enumerate(brief["suggested_actions"], 1):
            action_item(i, action)

    with c2:
        st.markdown("#### Next 7 days — expected occupancy")
        outlook = brief.get("next_7_days_outlook", [])
        if outlook:
            fc = pd.DataFrame(outlook)
            fc["date"] = pd.to_datetime(fc["date"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["date"], fc["date"].iloc[::-1]]),
                y=pd.concat([fc["high"], fc["low"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(191,151,64,.10)",
                line=dict(color="rgba(0,0,0,0)"), name="Range", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc["expected_occupancy"], mode="lines+markers",
                line=dict(color=GOLD, width=2.5), marker=dict(size=8, color=GOLD),
                name="Expected"))
            themed_figure(fig, height=300, showlegend=False)
            fig.update_yaxes(tickformat=".0%", title="Rooms filled")
            st.plotly_chart(fig, use_container_width=True)
            help_box("The shaded band is the range of likely outcomes. "
                     "If the line is rising, more guests are booked or expected.")
        else:
            st.info("Forecast model not loaded yet. Run `python src/train_models_ts.py`.")
