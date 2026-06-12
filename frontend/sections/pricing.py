"""Tab: Pricing & Inventory — dynamic pricing + overbooking, side by side."""
from __future__ import annotations

from html import escape as esc

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api import api_get, api_post
from theme import AMBER, GOLD, GREEN, RED, themed_figure
from ui import help_box, render_card

_TIER_DEFAULTS = [
    ("VIP / Suites",   10, 0.10, 450),
    ("Standard rooms", 80, 0.28, 120),
    ("OTA / Discount", 40, 0.42,  85),
]


def render() -> None:
    st.markdown("### 💰  Pricing & Inventory")
    help_box("Two tools side by side. **Left:** what should we charge tonight given expected "
             "demand? **Right:** how many extra reservations should we accept knowing some will "
             "cancel? The math weighs the cost of an empty room against the cost of having "
             "to walk a guest.")

    p1, p2 = st.columns(2)
    with p1:
        _pricing()
    with p2:
        _overbooking()


def _pricing() -> None:
    st.markdown("#### What should we charge?")
    curr_adr = st.number_input("Today's nightly rate ($)", 50., 1000., 120.)
    horizon  = st.slider("Looking ahead how many days?", 7, 90, 30, key="p_hz")
    if not st.button("Get pricing recommendation", key="btn_price"):
        return
    with st.spinner("Calculating…"):
        rec, err = api_get("/api/v1/pricing/recommendation",
                           params={"current_adr": curr_adr, "horizon_days": horizon})
    if err:
        st.error(err)
        return
    if not rec:
        return

    col = GREEN if rec["price_change_pct"] > 0 else RED if rec["price_change_pct"] < 0 else AMBER
    demand_label = ("strong demand — push prices up" if rec["demand_index"] > 1.10 else
                    "soft demand — discount to fill rooms" if rec["demand_index"] < 0.90 else
                    "demand is in line with normal")
    render_card(
        st,
        f"<div style='color:#EDD98A;font-size:1rem;font-weight:600'>{esc(rec['strategy'])}</div>"
        f"<div style='font-size:1.7rem;color:{col};font-family:Cormorant Garamond,serif;margin:.4rem 0'>"
        f"${rec['current_adr']:.0f} → ${rec['recommended_adr']:.0f} "
        f"<span style='font-size:.95rem'>({rec['price_change_pct']:+.1f}%)</span></div>"
        f"<div style='color:#767670;font-size:.78rem'>"
        f"Estimated lift in revenue per room: <b>${rec['revpar_uplift_est']:+.0f}</b><br>"
        f"Demand level: <b>{demand_label}</b></div>"
        f"<div style='color:#C5C5BF;font-size:.82rem;margin-top:.6rem'>{esc(rec['reasoning'])}</div>",
        border="rgba(196,155,60,.2)",
    )

    tier_df = pd.DataFrame([{"Room Type": k, "Suggested rate": f"${v:,.0f}"}
                            for k, v in rec["room_tier_prices"].items()])
    st.markdown("**Suggested rates by room type**")
    st.dataframe(tier_df, use_container_width=True, hide_index=True)


def _overbooking() -> None:
    st.markdown("#### How many extra reservations to take?")
    capacity = st.number_input("Total rooms in hotel", 10, 500, 100)
    max_walk = st.slider("Max acceptable walk risk", 0.01, 0.10, 0.05, format="%.0f%%",
                         help="The chance more guests show up than rooms available, "
                              "forcing us to relocate someone. Most hotels target 3-5%.")

    st.markdown("**Booking sources expected:**")
    tiers = []
    for name, nb, cp, ad in _TIER_DEFAULTS:
        tc1, tc2, tc3 = st.columns(3)
        nb2 = tc1.number_input(f"{name} — bookings", 0, 300, nb, key=f"nb_{name}")
        cp2 = tc2.number_input("Cancel rate", 0.0, 1.0, cp, key=f"cp_{name}", step=0.01)
        ad2 = tc3.number_input("Rate $", 0., 2000., float(ad), key=f"ad_{name}")
        tiers.append({"name": name, "n_bookings": nb2, "cancel_prob": cp2,
                      "adr": ad2, "stay_nights": 2.0})

    if not st.button("Solve", key="btn_ob"):
        return
    body = {"capacity": capacity, "tiers": tiers, "c_empty": 500,
            "c_walk": 1500, "max_walk_prob": max_walk}
    with st.spinner("Solving…"):
        res, err = api_post("/api/v1/overbooking/solve", body)
    if err:
        st.error(err)
        return
    if not res:
        return

    ov = res["optimal_overbooking"]
    clr = GREEN if ov < 5 else AMBER if ov < 12 else RED
    render_card(
        st,
        f"<div style='color:{clr};font-size:1.8rem;font-family:Cormorant Garamond,serif'>"
        f"Take {ov} extra reservations</div>"
        f"<div style='color:#767670;font-size:.78rem'>"
        f"Walk risk: {res['walk_probability']:.1%} · "
        f"Expected profit: ${res['expected_profit']:,.0f}</div>"
        f"<div style='color:#C5C5BF;font-size:.82rem;margin-top:.5rem'>"
        f"{esc(res['recommendation'])}</div>",
        border=clr,
    )

    if res.get("sensitivity"):
        sens = pd.DataFrame(res["sensitivity"])
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sens["delta"], y=sens["e_profit"],
            mode="lines+markers", name="Expected profit", line=dict(color=GOLD, width=2)))
        fig_s.add_vline(x=ov, line_dash="dash", line_color=GREEN,
                        annotation_text=f"Best: {ov}", annotation_font_color=GREEN)
        themed_figure(fig_s, title="Profit at different overbooking levels", height=260,
                      xaxis_title="Extra reservations taken", yaxis_title="Expected profit ($)")
        st.plotly_chart(fig_s, use_container_width=True)
