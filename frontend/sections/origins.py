"""Tab: Where Guests Come From — channel profitability, no-show patterns, origins."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api import api_get
from theme import AMBER, BLUE, GOLD, GREEN, RED, themed_figure
from ui import help_box


def render() -> None:
    st.markdown("### 🌍  Where Guests Come From")
    help_box("Three views: which booking channels actually make us money "
             "(after commissions); when no-shows tend to happen; and "
             "where in the world your guests are travelling from.")

    sub_a, sub_b, sub_c = st.tabs(["💸 Channel Profitability", "📅 No-Show Patterns", "🌎 Guest Origins"])
    with sub_a:
        _channel_mix()
    with sub_b:
        _no_show_heatmap()
    with sub_c:
        _guest_mix()


def _channel_mix() -> None:
    st.markdown("#### Which booking channels actually make us money?")
    days = st.slider("Look back how many days?", 30, 730, 180, key="cm_days")
    cm, err = api_get("/api/v1/analytics/channel-mix", params={"lookback_days": days})
    if err:
        st.error(err)
        return
    if not cm:
        return

    s = cm["summary"]
    help_box(cm.get("explanation", ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Gross revenue",       f"${s['total_gross_revenue']:,.0f}")
    c2.metric("Commissions paid",    f"${s['total_commission']:,.0f}")
    c3.metric("Net (yours to keep)", f"${s['total_net_revenue']:,.0f}",
              f"-{s['blended_take_rate']:.1%} avg take-rate")

    df = pd.DataFrame(cm["channels"])
    if not len(df):
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Gross", x=df["label"], y=df["gross_revenue"],
                         marker_color="rgba(191,151,64,.4)"))
    fig.add_trace(go.Bar(name="Net (after commission)", x=df["label"],
                         y=df["net_revenue"], marker_color=GOLD))
    themed_figure(fig, barmode="overlay", height=380, title="Revenue by channel — gross vs net")
    fig.update_yaxes(title="Revenue ($)")
    st.plotly_chart(fig, use_container_width=True)

    tbl = df[["label", "bookings", "gross_revenue", "commission_rate",
              "commission_cost", "net_revenue", "avg_adr", "cancellation_rate"]].copy()
    tbl.columns = ["Channel", "Bookings", "Gross $", "Commission %",
                   "Commission $", "Net $", "Avg rate", "Cancel rate"]
    for col, fmt in [("Gross $", "${:,.0f}"), ("Commission $", "${:,.0f}"), ("Net $", "${:,.0f}"),
                     ("Avg rate", "${:,.0f}"), ("Commission %", "{:.0%}"), ("Cancel rate", "{:.0%}")]:
        tbl[col] = tbl[col].map(fmt.format)
    st.dataframe(tbl, use_container_width=True, hide_index=True)


def _no_show_heatmap() -> None:
    st.markdown("#### When are no-shows worst?")
    days = st.slider("Look back how many days?", 60, 1500, 365, key="ns_days")
    ns, err = api_get("/api/v1/analytics/no-show-heatmap", params={"lookback_days": days})
    if err:
        st.error(err)
        return
    if not ns:
        return

    help_box(ns.get("explanation", ""))
    mat = np.array(ns["rate_matrix"]) * 100
    fig = go.Figure(go.Heatmap(
        z=mat, x=ns["months"], y=ns["days"],
        colorscale=[[0, "#1a3a1a"], [0.4, "#5b9b3b"], [0.7, "#FFC107"], [1, "#F44336"]],
        colorbar=dict(title=dict(text="Cancel %", font=dict(color="#C5C5BF")),
                      tickfont=dict(color="#C5C5BF")),
        text=[[f"{v:.0f}%" for v in row] for row in mat],
        texttemplate="%{text}", textfont={"size": 10, "color": "white"}))
    themed_figure(fig, height=350,
                  title=f"No-show rate by day-of-week × month (overall avg: {ns['overall_rate']:.0%})")
    st.plotly_chart(fig, use_container_width=True)


def _guest_mix() -> None:
    st.markdown("#### Where are your guests coming from?")
    days = st.slider("Look back how many days?", 60, 1500, 365, key="gm_days")
    gm, err = api_get("/api/v1/analytics/guest-mix", params={"lookback_days": days, "top_n": 12})
    if err:
        st.error(err)
        return
    if not gm:
        return

    c1, c2 = st.columns(2)
    with c1:
        if gm["top_countries"]:
            df = pd.DataFrame(gm["top_countries"])
            fig = go.Figure(go.Bar(
                x=df["revenue"], y=df["country"], orientation="h", marker_color=GOLD,
                text=df["revenue"].map("${:,.0f}".format), textposition="outside"))
            themed_figure(fig, title="Top countries by revenue", height=400, xaxis_title="Revenue ($)")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if gm["segments"]:
            df = pd.DataFrame(gm["segments"])
            fig = go.Figure(go.Pie(
                labels=df["segment"], values=df["revenue"], hole=0.5,
                marker=dict(colors=[GOLD, BLUE, GREEN, AMBER, RED, "#A8B5C8", "#767670"])))
            themed_figure(fig, title="Revenue by booking type", height=400)
            st.plotly_chart(fig, use_container_width=True)
    st.metric("Repeat-guest share", f"{gm['repeat_share']:.1%}",
              help="The share of recent bookings from guests who've stayed before. "
                   "Higher is generally better — they cost less to acquire and cancel less.")
