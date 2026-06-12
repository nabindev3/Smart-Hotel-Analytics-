"""Tab: Will They Show Up? — cancellation risk score + SHAP explanation."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from api import api_post, get_schema
from theme import AMBER, GREEN, RED, gold_rule, themed_figure
from ui import help_box, render_card

# Human-readable labels for coded values. The *values* come from the backend
# schema (item 10); these are presentation only and degrade gracefully via .get.
_MEAL_LABELS = {"BB": "Bed & Breakfast", "HB": "Half board", "FB": "Full board",
                "SC": "Self-catering", "Undefined": "Not specified"}
_CHAN_LABELS = {"TA/TO": "Travel agency / OTA", "Direct": "Direct", "Corporate": "Corporate",
                "GDS": "Travel agent (GDS)", "Undefined": "Other"}
_RISK_LABELS = {"HIGH": "High risk", "MODERATE": "Medium risk", "LOW": "Low risk"}


def render() -> None:
    st.markdown("### 🚪  Will This Guest Actually Show Up?")
    help_box("Enter a booking and we'll predict the chance the guest will cancel or no-show. "
             "We also explain *why* — so you can decide if it's worth a confirmation call, "
             "a stricter deposit, or letting it go. No jargon: red bars push risk up, "
             "green bars push it down.")

    schema = get_schema()

    with st.form("cancel_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hotel = st.selectbox("Hotel", schema["hotel"])
            month = st.selectbox("Arrival month", schema["arrival_date_month"])
            lead  = st.number_input("Days booked in advance", 0, 700, 60,
                                    help="Bookings made far in advance tend to cancel more often.")
            wknd  = st.number_input("Weekend nights", 0, 10, 1)
            wkk   = st.number_input("Weekday nights",  0, 20, 3)
            adr   = st.number_input("Nightly rate ($)", 0., 5000., 100.)
        with c2:
            adults   = st.number_input("Adults",   1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            country  = st.text_input("Country (3-letter code)", "PRT", help="e.g., USA, GBR, FRA, DEU")
            is_rep   = st.selectbox("Has stayed before?", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")
            prev_can = st.number_input("Past cancellations", 0, 20, 0)
            prev_ok  = st.number_input("Past completed stays", 0, 50, 0)
        with c3:
            meal = st.selectbox("Meal plan", schema["meal"],
                                format_func=lambda x: _MEAL_LABELS.get(x, x))
            seg  = st.selectbox("How was it booked?", schema["market_segment"])
            chan = st.selectbox("Channel", schema["distribution_channel"],
                                format_func=lambda x: _CHAN_LABELS.get(x, x))
            room = st.selectbox("Room type", schema["reserved_room_type"])
            dep  = st.selectbox("Deposit", schema["deposit_type"])
            cust = st.selectbox("Guest type", schema["customer_type"])
            wait = st.number_input("Days on waiting list", 0, 300, 0)
            spec = st.number_input("Special requests", 0, 5, 1)
        submitted = st.form_submit_button("Score this booking")

    if not submitted:
        return

    body = {
        "hotel": hotel, "lead_time": lead, "arrival_date_month": month,
        "total_stay": wknd + wkk, "total_guests": adults + children,
        "meal": meal, "country": country.upper()[:3],
        "market_segment": seg, "distribution_channel": chan,
        "is_repeated_guest": is_rep, "previous_cancellations": prev_can,
        "previous_bookings_not_canceled": prev_ok,
        "reserved_room_type": room, "booking_changes": 0,
        "deposit_type": dep, "days_in_waiting_list": float(wait),
        "customer_type": cust, "required_car_parking_spaces": 0,
        "total_of_special_requests": spec, "adr": adr,
    }
    with st.spinner("Scoring…"):
        pred, e1 = api_post("/api/v1/cancellation/predict", body)
        xai,  e2 = api_post("/api/v1/xai/explain", body)

    if e1:
        st.error(f"Could not score: {e1}")
    if pred:
        _render_score(pred)
    if e2:
        st.warning(f"Could not load explanation: {e2}")
    if xai and "waterfall" in xai:
        _render_explanation(xai)


def _render_score(pred: dict) -> None:
    cp = pred["cancellation_probability"]
    level_label = _RISK_LABELS.get(pred["risk_level"], pred["risk_level"])
    color = RED if cp > 0.6 else AMBER if cp > 0.35 else GREEN

    gold_rule()
    r1, r2, r3 = st.columns([1, 1, 1.5])
    r1.metric("Chance they show", f"{(1 - cp):.0%}")
    r2.metric("Chance they cancel/no-show", f"{cp:.0%}")
    from html import escape as esc
    render_card(
        r3,
        f"<div style='color:{color};font-weight:600;font-size:1rem'>{esc(level_label)}</div>"
        f"<div style='color:#C5C5BF;font-size:.85rem;margin-top:.3rem'>"
        f"<b>What to do:</b> {esc(pred['recommended_action'])}</div>",
        border=color, pad=".85rem 1rem",
    )

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=cp * 100,
        title={"text": "Cancel / no-show risk", "font": {"color": "#EDD98A", "size": 13}},
        number={"suffix": "%", "font": {"color": "#EDD98A", "size": 34}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [
                   {"range": [0, 35],   "color": "rgba(76,175,80,.12)"},
                   {"range": [35, 60],  "color": "rgba(255,193,7,.12)"},
                   {"range": [60, 100], "color": "rgba(244,67,54,.12)"}]},
    ))
    # Through themed_figure so the gauge can't drift from the shared theme.
    themed_figure(fig_g, height=240, margin=dict(t=50, b=5, l=25, r=25))
    st.plotly_chart(fig_g, use_container_width=True)


def _render_explanation(xai: dict) -> None:
    st.markdown("#### Why this score?")
    help_box("Each bar is a piece of the booking. **Red** bars push the risk up "
             "(reasons to worry). **Green** bars push it down (reasons to relax). "
             "Bigger bar = bigger effect.")
    wf = xai["waterfall"]
    names  = [w["feature"] for w in wf]
    values = [w["shap_value"] for w in wf]
    fig_shap = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=[RED if v > 0 else GREEN for v in values],
        text=[f"{v:+.3f}" for v in values], textposition="outside"))
    themed_figure(fig_shap, title="What drove the score (red = ↑ risk, green = ↓ risk)",
                  height=350, xaxis_title="Impact on risk score")
    st.plotly_chart(fig_shap, use_container_width=True)
    st.info(f"🔍 **Biggest reason:** `{xai['top_risk_factor']}`")
