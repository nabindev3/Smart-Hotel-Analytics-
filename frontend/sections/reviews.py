"""Tab: Guest Reviews — sentiment + aspect analysis of free-text reviews."""
from __future__ import annotations

from html import escape as esc

import plotly.graph_objects as go
import streamlit as st
from api import api_post, cached_get
from theme import AMBER, GOLD, GREEN, RED, gold_rule, themed_figure
from ui import help_box, render_card

_TIER_MAP = {
    1: ("🤗", "Best — using HuggingFace AI", GREEN),
    2: ("🤖", "Good — using Claude",         GOLD),
    3: ("📝", "Basic — TextBlob (free)",     AMBER),
}
_MOOD = {"Positive": "Happy guest", "Negative": "Unhappy guest", "Neutral": "Neutral feedback"}


def render() -> None:
    st.markdown("### 💬  What Guests Are Saying")
    help_box("Paste a review (or anything a guest said) and we'll tell you if it's positive, "
             "negative, or sarcastic — plus what they're talking about (food, room, service, etc.).")

    eng, _ = cached_get("/api/v1/sentiment/engine-info")
    if eng:
        tier = eng.get("active_tier", 3)
        ico, lbl, clr = _TIER_MAP.get(tier, ("📝", "Basic", AMBER))
        render_card(
            st,
            f"<span style='font-size:1.2rem'>{ico}</span>"
            f" <b style='color:{clr}'>Currently active: {esc(lbl)}</b>",
            border=clr, pad=".7rem 1rem", margin="0 0 .8rem",
        )
        if tier == 3:
            st.info("To unlock sarcasm detection and topic analysis, set the "
                    "`HF_API_TOKEN` environment variable (free at huggingface.co/settings/tokens).")

    gold_rule()

    user_text = st.text_area("Paste a guest review:", height=100,
        placeholder="e.g., 'Oh wonderful — the AC broke at 2am. Truly a five-star experience.'")
    if not (st.button("Analyse this review") and user_text):
        return

    with st.spinner("Reading…"):
        result, err = api_post("/api/v1/sentiment/analyse", {"text": user_text})
    if err:
        st.error(err)
        return
    if not result:
        return

    _render_result(result)


def _render_result(result: dict) -> None:
    label  = result.get("label", "Neutral")
    pol    = result.get("polarity", 0)
    conf   = result.get("confidence", 0)
    sarc   = result.get("sarcasm_flag", False)
    engine = result.get("engine", "")
    label_clr = GREEN if label == "Positive" else RED if label == "Negative" else AMBER
    mood = _MOOD.get(label, label)

    lc, rc = st.columns([1, 2])
    with lc:
        badge = "🤗" if "HuggingFace" in engine else "🤖" if "Claude" in engine else "📝"
        sarc_bit = ("<div style='color:#F44336;font-size:.78rem;margin-top:.3rem'>"
                    "⚠️ Sarcasm detected — read carefully</div>") if sarc else ""
        render_card(
            st,
            f"<div style='color:{label_clr};font-size:1.3rem;font-weight:600'>{esc(mood)}</div>"
            f"<div style='color:#767670;font-size:.78rem;margin:.3rem 0'>"
            f"How positive: <b>{pol:+.2f}</b> (range -1 to +1)<br>"
            f"How sure: <b>{conf:.0%}</b></div>"
            f"{sarc_bit}"
            f"<div style='color:#444440;font-size:.65rem;margin-top:.5rem'>{badge} {esc(engine)}</div>",
            border=label_clr, pad="1.1rem",
        )
    with rc:
        aspects = {k: v for k, v in result.get("aspects", {}).items() if v is not None}
        if aspects:
            fig_a = go.Figure(go.Bar(
                x=list(aspects.values()), y=list(aspects.keys()), orientation="h",
                marker_color=[GREEN if v >= 0.1 else RED if v <= -0.1 else AMBER
                              for v in aspects.values()]))
            themed_figure(fig_a, title="What they're talking about",
                          height=220, margin=dict(t=35, b=15, l=10, r=10))
            fig_a.update_xaxes(range=[-1, 1], title="Negative ← → Positive")
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("Topic-level analysis needs a smarter engine. "
                    "Set `HF_API_TOKEN` or `ANTHROPIC_API_KEY` to enable it.")
        themes = result.get("themes", [])
        if themes:
            st.markdown("**Topics mentioned:** " + "  ".join(f"`{esc(str(t))}`" for t in themes))
