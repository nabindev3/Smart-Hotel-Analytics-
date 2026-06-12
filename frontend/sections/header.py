"""Page header: title, tagline, and the top-right KPI metrics."""
from __future__ import annotations

import streamlit as st
from theme import gold_rule


def render(kpi_s: dict | None) -> None:
    hl, hr = st.columns([3, 1])
    with hl:
        st.markdown("# Smart Hotel Analytics")
        st.markdown('<div class="tagline">"7 years in hotels showed me exactly '
                    'what data was being wasted."</div>', unsafe_allow_html=True)
        st.write("Plain-English insights for the front desk, sales, and the GM. "
                 "Forecasts, no-show risk, smart pricing, guest offers, review sentiment — "
                 "powered by machine learning behind the scenes.")
    with hr:
        if kpi_s:
            st.metric("Rooms Filled (30d)", f"{kpi_s['avg_occupancy']:.0%}")
            st.metric("Avg Nightly Rate",    f"${kpi_s['avg_adr']:.0f}")
            st.metric("Revenue / Room",      f"${kpi_s['avg_revpar']:.0f}")
    gold_rule()
