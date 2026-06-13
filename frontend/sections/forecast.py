"""Tab: Demand Outlook — Prophet forecast for occupancy / ADR / revenue."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api import cached_get
from theme import BLUE, GOLD, RED, gold_rule, themed_figure
from ui import help_box

_METRIC_MAP = {"Rooms Filled": "occupancy", "Avg Nightly Rate": "adr", "Total Revenue": "revenue"}


def render() -> None:
    st.markdown("### 📈  Demand Outlook")
    help_box("Forecast for the next 30–180 days, broken down by what you want to see: "
             "rooms filled, average rate, or revenue. Useful for staffing, "
             "purchasing, and rate-setting decisions.")

    metric_choice = st.radio("What do you want to forecast?",
                             options=list(_METRIC_MAP), horizontal=True)
    metric = _METRIC_MAP[metric_choice]
    horizon = st.slider("How far ahead?", 30, 180, 90)

    with st.spinner("Loading forecast…"):
        data, err = cached_get(f"/api/v1/forecast/{metric}",
                            params={"horizon_days": horizon, "include_components": True})

    if err:
        st.error(f"Could not load forecast: {err}")
        return
    if not data:
        return

    fc_df = pd.DataFrame(data["forecast"]); ac_df = pd.DataFrame(data["actual_tail"])
    fc_df["date"] = pd.to_datetime(fc_df["date"]); ac_df["date"] = pd.to_datetime(ac_df["date"])
    # Plotly accepts d3 format strings ("$,.0f"); Python f-strings do not, so the
    # "$" prefix is handled separately for the metric cards below.
    is_pct       = (metric == "occupancy")
    plotly_fmt   = ".1%" if is_pct else "$,.0f"
    py_fmt       = ".1%" if is_pct else ",.0f"
    money_prefix = ""    if is_pct else "$"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"], fc_df["date"].iloc[::-1]]),
        y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(191,151,64,.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Likely range"))
    fig.add_trace(go.Scatter(x=ac_df["date"], y=ac_df["value"],
        mode="lines", line=dict(color=BLUE, width=1.5), name="Recent actuals"))
    fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["yhat"],
        mode="lines", line=dict(color=GOLD, width=2.5), name="Forecast"))
    if len(ac_df):
        vline_x = ac_df["date"].max().timestamp() * 1000
        fig.add_vline(x=vline_x, line_dash="dash", line_color="rgba(196,155,60,.4)",
                      annotation_text="Today", annotation_font_color=GOLD)

    accuracy = (1 - data.get("mape", 0)) * 100 if data.get("mape") else None
    title = f"{metric_choice} — next {horizon} days"
    if accuracy is not None:
        title += f"  (forecast accuracy on past data: {accuracy:.1f}%)"
    themed_figure(fig, title=title, height=420)
    fig.update_yaxes(tickformat=plotly_fmt)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg over next 30 days", f"{money_prefix}{fc_df.head(30)['yhat'].mean():{py_fmt}}")
    c2.metric("Avg over next 90 days", f"{money_prefix}{fc_df.head(90)['yhat'].mean():{py_fmt}}")
    if accuracy is not None:
        c3.metric("Forecast accuracy", f"{accuracy:.1f}%",
                  help="How close past forecasts were to what actually happened. "
                       "90% means the forecast was within 10% on average.")

    if data.get("components"):
        gold_rule()
        st.markdown("#### Patterns the model has learned")
        help_box("These show recurring patterns in your data. "
                 "Use them to plan staffing and promotions around natural peaks and dips.")
        comp = data["components"]
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_y = go.Figure(go.Scatter(
                y=comp["yearly"], mode="lines",
                line=dict(color=GOLD, width=2), fill="tozeroy",
                fillcolor="rgba(191,151,64,.10)"))
            themed_figure(fig_y, title="Time-of-year pattern", height=240)
            fig_y.update_xaxes(title="Day of year")
            fig_y.update_yaxes(title="Above/below average")
            st.plotly_chart(fig_y, use_container_width=True)
        with cc2:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig_w = go.Figure(go.Bar(
                x=days, y=comp["weekly"],
                marker_color=[GOLD if v >= 0 else RED for v in comp["weekly"]]))
            themed_figure(fig_w, title="Day-of-week pattern", height=240)
            fig_w.update_yaxes(title="Above/below average")
            st.plotly_chart(fig_w, use_container_width=True)
