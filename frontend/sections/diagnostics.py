"""Tab: Diagnostics — global SHAP importance and forecast-input ablation."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api import cached_get
from theme import BLUE, GOLD, themed_figure
from ui import help_box


def render() -> None:
    st.markdown("### 🔬  Model Diagnostics")
    help_box("Technical view for whoever maintains the system. Shows which inputs the "
             "no-show model relies on most, and how much each forecast input matters. "
             "Skip this tab unless you're debugging.")

    xa1, xa2 = st.columns(2)
    with xa1:
        _global_importance()
    with xa2:
        _ablation()


def _global_importance() -> None:
    st.markdown("#### What drives the no-show predictions?")
    st.caption("Precomputed with SHAP over a 500-booking sample.")
    if not st.button("Compute"):
        return
    with st.spinner("Loading…"):
        gi, err = cached_get("/api/v1/xai/global-importance", timeout=30)
    if err:
        st.error(err)
        return
    if not gi:
        return
    names, vals = gi["feature_names"], gi["mean_abs_shap"]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=[GOLD if i < 3 else BLUE for i in range(len(names))],
        text=[f"{v:.4f}" for v in vals], textposition="outside"))
    themed_figure(fig, height=480, title="Most influential booking attributes",
                  xaxis_title="Average impact on prediction")
    st.plotly_chart(fig, use_container_width=True)


def _ablation() -> None:
    st.markdown("#### Forecast input significance")
    if not st.button("Load results"):
        return
    with st.spinner("Loading…"):
        abl, err = cached_get("/api/v1/xai/ablation", timeout=10)
    if err:
        st.warning(f"{err} — run `python src/ablation_study.py` first.")
        return
    if not abl:
        return
    for target_name, result in abl.items():
        st.markdown(f"**{target_name}** — base accuracy {(1 - result['baseline']['mape']):.1%}")
        rows = [{
            "Removed":       a["removed_regressor"],
            "Accuracy lost": f"{(a['mape_delta'] * 100):+.2f}pts",
            "p-value":       f"{a['p_value']:.3f}",
            "Matters?":      "✅ yes" if a["significant_p05"] else "❌ no",
        } for a in result.get("ablations", [])]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=240)
        st.markdown("---")
