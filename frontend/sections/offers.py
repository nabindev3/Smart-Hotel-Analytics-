"""Tab: Guest Offers — personalised upsell recommendations for a guest profile."""
from __future__ import annotations

from html import escape as esc

import streamlit as st
from api import api_post, get_schema
from theme import GOLD, LOYALTY_COLORS
from ui import help_box, render_card


def render() -> None:
    st.markdown("### 🎯  Personalised Offers for a Guest")
    help_box("Type a guest profile and we'll suggest what to upsell them — spa, room upgrade, "
             "private transfer, etc. Each suggestion comes with a ready-to-send email and an "
             "expected revenue impact. Higher score = more likely they'll say yes.")

    schema = get_schema()

    with st.form("rec_form"):
        g1, g2 = st.columns(2)
        with g1:
            g_hotel    = st.selectbox("Hotel", schema["hotel"], key="rec_hotel")
            g_adr      = st.number_input("Nightly rate ($)", 0., 5000., 180., key="rec_adr")
            g_adults   = st.number_input("Adults",   1, 10, 2, key="rec_a")
            g_children = st.number_input("Children", 0, 10, 2, key="rec_c")
            g_stay     = st.number_input("Total nights", 1, 30, 7, key="rec_s")
            g_country  = st.text_input("Country code (e.g., GBR)", "GBR", key="rec_co")
        with g2:
            g_meal = st.selectbox("Meal plan", schema["meal"], key="rec_m")
            g_seg  = st.selectbox("How they booked", schema["market_segment"], key="rec_seg")
            g_rep  = st.selectbox("Has stayed before?", [0, 1], key="rec_rep",
                                  format_func=lambda x: "Yes" if x else "No")
            g_prev = st.number_input("Past completed stays", 0, 50, 2, key="rec_pv")
            g_spec = st.number_input("Special requests", 0, 5, 2, key="rec_sp")
            top_n  = st.slider("How many offers to show?", 1, 5, 3, key="rec_n")
        rec_submit = st.form_submit_button("Get offers")

    if not rec_submit:
        return

    body = {
        "hotel": g_hotel, "adr": g_adr, "adults": g_adults, "children": float(g_children),
        "babies": 0, "total_stay": g_stay, "country": g_country.upper()[:3],
        "meal": g_meal, "is_repeated_guest": g_rep,
        "previous_bookings_not_canceled": g_prev,
        "total_of_special_requests": g_spec, "market_segment": g_seg,
    }
    with st.spinner("Generating offers…"):
        result, err = api_post("/api/v1/recommend/next-action", body,
                               params={"top_n": top_n}, timeout=30)
    if err:
        st.error(err)
        return
    if not result:
        return

    loy = result["loyalty_tier"]
    loy_col = LOYALTY_COLORS.get(loy, GOLD)
    render_card(
        st,
        f"<span style='color:{loy_col};font-weight:600'>● {esc(loy)} guest</span>"
        f"&nbsp;&nbsp;<span style='color:#767670;font-size:.78rem'>"
        f"Estimated extra revenue if they accept: <b>${result['estimated_upsell']:.0f}</b></span>",
        border=loy_col, margin="0 0 1rem",
    )
    st.markdown(f"**Top suggestion:** {esc(result['next_best_action'])}")

    for rec in result.get("recommendations", []):
        filled = int(rec["score"] * 10)
        bar = "█" * filled + "░" * (10 - filled)
        render_card(
            st,
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<b style='color:#EDD98A'>{esc(rec['label'])}</b>"
            f"<span style='color:#767670;font-size:.75rem'>"
            f"Match: {rec['score']:.0%} · Revenue: ${rec['revenue']}</span></div>"
            f"<div style='color:#BF9740;font-family:monospace;font-size:.75rem;margin:.3rem 0'>{bar}</div>"
            f"<div style='color:#C5C5BF;font-size:.85rem;font-style:italic;margin-top:.5rem'>"
            f"📧 {esc(rec['email_copy'])}</div>",
            border="rgba(196,155,60,.15)", bg="rgba(14,16,24,.85)",
        )
