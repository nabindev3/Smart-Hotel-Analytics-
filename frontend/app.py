"""
frontend/app.py — Smart Hotel Analytics Dashboard
==================================================
Hotel-manager friendly Streamlit frontend.
All data fetched from FastAPI backend at $API_BASE.

Run:
  export API_BASE=http://localhost:8000
  streamlit run frontend/app.py
"""

import os, json, time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

API_BASE       = os.environ.get("API_BASE", "http://localhost:8000")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Smart Hotel Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS — kept luxury aesthetic, softened palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');
html,body,.stApp{background:#07090D!important;color:#C5C5BF!important;}
.stApp{font-family:'DM Sans',sans-serif;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0C0E16,#07090D)!important;
  border-right:1px solid rgba(196,155,60,.12);}
section[data-testid="stSidebar"] *{color:#C5C5BF!important;}
h1{font-family:'Cormorant Garamond',serif!important;font-size:2.6rem!important;
   color:#EDD98A!important;font-weight:600!important;letter-spacing:.04em;}
h2,h3{font-family:'Cormorant Garamond',serif!important;color:#EDD98A!important;}
.tagline{color:#BF9740;font-style:italic;font-family:'Cormorant Garamond',serif;
  font-size:1.05rem;border-left:3px solid #BF9740;padding:.45rem 1rem;
  background:rgba(191,151,64,.06);border-radius:0 8px 8px 0;margin:0 0 1.4rem;}
.gold-rule{height:1px;background:linear-gradient(90deg,transparent,#BF9740 30%,#BF9740 70%,transparent);
  margin:1.2rem 0;border:none;}
.help-box{background:rgba(58,123,213,.06);border-left:3px solid #3A7BD5;
  border-radius:0 6px 6px 0;padding:.55rem .9rem;margin:.4rem 0 1rem;
  color:#A8B5C8;font-size:.82rem;line-height:1.55;}
.action-card{background:rgba(14,16,24,.92);border:1px solid rgba(196,155,60,.18);
  border-radius:10px;padding:.85rem 1rem;margin:.4rem 0;}
.alert-warning{background:rgba(255,193,7,.08);border-left:4px solid #FFC107;
  border-radius:0 8px 8px 0;padding:.6rem .9rem;margin:.4rem 0;}
.alert-good{background:rgba(76,175,80,.08);border-left:4px solid #4CAF50;
  border-radius:0 8px 8px 0;padding:.6rem .9rem;margin:.4rem 0;}
.alert-info{background:rgba(58,123,213,.08);border-left:4px solid #3A7BD5;
  border-radius:0 8px 8px 0;padding:.6rem .9rem;margin:.4rem 0;}
div[data-testid="metric-container"]{
  background:linear-gradient(135deg,rgba(14,16,24,.95),rgba(10,12,19,.98))!important;
  border:1px solid rgba(196,155,60,.18)!important;border-radius:12px!important;
  padding:1.1rem!important;box-shadow:0 4px 20px rgba(0,0,0,.5)!important;}
div[data-testid="metric-container"] label{color:#767670!important;font-size:.67rem!important;
  letter-spacing:.14em!important;text-transform:uppercase;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color:#EDD98A!important;font-family:'Cormorant Garamond',serif!important;font-size:1.85rem!important;}
.stTabs [data-baseweb="tab-list"]{background:#0C0E16;border-bottom:1px solid rgba(196,155,60,.12);}
.stTabs [data-baseweb="tab"]{color:#767670!important;background:transparent;border:none;
  font-size:.79rem;letter-spacing:.05em;padding:.55rem 1.1rem;}
.stTabs [aria-selected="true"]{color:#EDD98A!important;border-bottom:2px solid #BF9740!important;}
.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextInput>div>div>input{
  background:#111320!important;color:#C5C5BF!important;
  border:1px solid rgba(196,155,60,.18)!important;border-radius:8px!important;}
label{color:#767670!important;font-size:.74rem!important;letter-spacing:.05em!important;}
.stButton>button{background:linear-gradient(135deg,#BF9740,#8C6A20)!important;
  color:#07090D!important;border:none!important;border-radius:40px!important;
  padding:.55rem 1.8rem!important;font-weight:500!important;letter-spacing:.08em!important;
  text-transform:uppercase!important;font-size:.78rem!important;
  box-shadow:0 4px 16px rgba(191,151,64,.22)!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:#BF9740;border-radius:2px;}
</style>
""", unsafe_allow_html=True)

GOLD, BLUE, GREEN, RED, AMBER = "#BF9740", "#3A7BD5", "#4CAF50", "#F44336", "#FFC107"
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,11,17,.75)",
    font=dict(color="#C5C5BF", family="DM Sans", size=12),
    title_font=dict(color="#EDD98A", family="Cormorant Garamond", size=17),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=48, b=35, l=38, r=18),
    xaxis=dict(gridcolor="rgba(196,155,60,.07)"),
    yaxis=dict(gridcolor="rgba(196,155,60,.07)"),
)


# ─────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────
def api_get(path, params=None, timeout=15):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status(); return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(path, body, timeout=20):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status(); return r.json(), None
    except Exception as e:
        return None, str(e)

def help_box(text):
    st.markdown(f"<div class='help-box'>💡 {text}</div>", unsafe_allow_html=True)

def trend_arrow(trend):
    """Convert {direction, delta_pct} → coloured arrow string."""
    if not trend: return ""
    d, p = trend.get("direction","flat"), trend.get("delta_pct",0)
    if d == "up":   return f"<span style='color:{GREEN}'>▲ {p:+.1f}%</span>"
    if d == "down": return f"<span style='color:{RED}'>▼ {p:+.1f}%</span>"
    return f"<span style='color:#767670'>● flat</span>"


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:.4rem 0'>"
        "<span style='font-size:2rem'>🏨</span>"
        "<div style='font-family:Cormorant Garamond,serif;font-size:1.2rem;"
        "color:#EDD98A;font-weight:600;letter-spacing:.05em'>Smart Hotel Analytics</div>"
        "<div style='color:#767670;font-size:.65rem;letter-spacing:.12em'>MANAGER DASHBOARD</div>"
        "</div>", unsafe_allow_html=True)
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── Live 30-day KPI grid ────────────────────────────────────────────
    st.markdown("<div style='color:#EDD98A;font-size:.7rem;letter-spacing:.12em;font-weight:600'>"
                "📊 LAST 30 DAYS</div>", unsafe_allow_html=True)
    kpi_s, _ = api_get("/api/v1/forecast/kpis/summary")
    if kpi_s:
        occ, adr, rev, canc = (kpi_s["avg_occupancy"], kpi_s["avg_adr"],
                                kpi_s["avg_revpar"], kpi_s.get("avg_cancel_rate",0))
        occ_clr  = GREEN if occ  >= 0.70 else AMBER if occ  >= 0.50 else RED
        adr_clr  = GREEN if adr  >= 100  else AMBER if adr  >= 70   else RED
        canc_clr = GREEN if canc <= 0.20 else AMBER if canc <= 0.35 else RED

        def cell(lbl, val, clr):
            return (f"<div style='background:rgba(14,16,24,.9);"
                    f"border:1px solid rgba(196,155,60,.15);border-radius:8px;"
                    f"padding:.55rem;text-align:center'>"
                    f"<div style='color:#767670;font-size:.58rem;letter-spacing:.1em'>{lbl}</div>"
                    f"<div style='color:{clr};font-size:1.15rem;"
                    f"font-family:Cormorant Garamond,serif;font-weight:600'>{val}</div></div>")
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:.35rem;margin:.5rem 0'>"
            + cell("ROOMS FILLED",   f"{occ:.0%}",  occ_clr)
            + cell("AVG NIGHTLY RATE", f"${adr:.0f}", adr_clr)
            + cell("$ PER ROOM",     f"${rev:.0f}", "#EDD98A")
            + cell("NO-SHOW RATE",   f"{canc:.0%}", canc_clr)
            + "</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#444440;font-size:.62rem;text-align:right'>"
                    f"{kpi_s.get('period_start','?')} → {kpi_s.get('period_end','?')}</div>",
                    unsafe_allow_html=True)
    else:
        st.caption("Loading…")

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── System health ───────────────────────────────────────────────────
    st.markdown("<div style='color:#EDD98A;font-size:.7rem;letter-spacing:.12em;font-weight:600'>"
                "🖥️ SYSTEM STATUS</div>", unsafe_allow_html=True)
    health, _ = api_get("/health")
    backend_ok = bool(health and health.get("status") == "healthy")
    try:
        mlflow_ok = requests.get("http://localhost:5001/health", timeout=2).status_code == 200
    except Exception:
        mlflow_ok = False

    def svc_row(icon, name, port, ok, url):
        dot   = f"<span style='color:{'#4CAF50' if ok else '#F44336'}'>●</span>"
        label = f"<span style='color:{'#C5C5BF' if ok else '#555'};font-size:.72rem'> {icon} {name}</span>"
        link  = (f"<a href='{url}' target='_blank' style='color:#767670;font-size:.62rem;"
                 f"float:right;text-decoration:none'>:{port} ↗</a>")
        return f"<div style='margin:.28rem 0'>{dot}{label}{link}</div>"

    st.markdown(
        svc_row("☁️", "Analytics engine", "8000", backend_ok, f"{PUBLIC_API_URL}/docs")
        + svc_row("🎨", "Dashboard",        "8501", True,       "http://localhost:8501")
        + svc_row("📊", "Model registry",   "5001", mlflow_ok,  "http://localhost:5001"),
        unsafe_allow_html=True)

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── Data freshness ──────────────────────────────────────────────────
    st.markdown("<div style='color:#EDD98A;font-size:.7rem;letter-spacing:.12em;font-weight:600'>"
                "📦 DATA QUALITY</div>", unsafe_allow_html=True)
    dq, _ = api_get("/api/v1/briefing/today")
    if dq and dq.get("data_quality"):
        q = dq["data_quality"]
        real_pct = q.get("real_share", 0) * 100
        st.markdown(
            f"<div style='font-size:.7rem;color:#C5C5BF;line-height:1.7'>"
            f"<b>{q.get('rows','—'):,}</b> bookings<br>"
            f"<b>{real_pct:.0f}%</b> real-world data<br>"
            f"<b>{q.get('date_min','?')}</b> → <b>{q.get('date_max','?')}</b><br>"
            f"<b>{q.get('n_countries','?')}</b> countries"
            f"</div>", unsafe_allow_html=True)
    else:
        st.caption("Run `python src/load_real_data.py` to add real bookings")

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:.72rem;line-height:2'>"
        f"<a href='{PUBLIC_API_URL}/docs' target='_blank' style='color:#C5C5BF;text-decoration:none'>"
        f"📖 Developer docs</a></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
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
        st.metric("Rooms Filled (30d)",  f"{kpi_s['avg_occupancy']:.0%}")
        st.metric("Avg Nightly Rate",     f"${kpi_s['avg_adr']:.0f}")
        st.metric("Revenue / Room",       f"${kpi_s['avg_revpar']:.0f}")

st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🌅  Today's Briefing",
    "📈  Demand Outlook",
    "🚪  Will They Show Up?",
    "💰  Pricing & Inventory",
    "🎯  Guest Offers",
    "💬  Guest Reviews",
    "🌍  Where Guests Come From",
    "🔬  Diagnostics",
])
t0, t1, t2, t3, t4, t5, t6, t7 = tabs


# ╔════════════════════════════════════╗
# ║  TAB 0 — TODAY'S BRIEFING        ║
# ╚════════════════════════════════════╝
with t0:
    st.markdown("### 🌅  Your Morning Briefing")
    help_box("Everything you need to know before your shift starts. "
             "Numbers cover the last 7 days, with arrows showing how each one is moving.")

    period = st.slider("How many days to summarise?", 1, 30, 7, key="brf_period")
    brief, err = api_get("/api/v1/briefing/today", params={"horizon_days": period})

    if err:
        st.error(f"Could not reach the analytics engine: {err}")
    elif brief:
        h = brief["headline"]; tr = brief["trend"]

        # Headline metrics with trend arrows
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div data-testid='metric-container'>"
                    f"<label>ROOMS FILLED</label>"
                    f"<div data-testid='stMetricValue'>{h['occupancy']:.0%}</div>"
                    f"<div style='font-size:.7rem'>{trend_arrow(tr['occupancy'])} vs prior period</div>"
                    f"</div>", unsafe_allow_html=True)
        m2.markdown(f"<div data-testid='metric-container'>"
                    f"<label>AVG NIGHTLY RATE</label>"
                    f"<div data-testid='stMetricValue'>${h['adr']:.0f}</div>"
                    f"<div style='font-size:.7rem'>{trend_arrow(tr['adr'])} vs prior period</div>"
                    f"</div>", unsafe_allow_html=True)
        m3.markdown(f"<div data-testid='metric-container'>"
                    f"<label>REVENUE / ROOM</label>"
                    f"<div data-testid='stMetricValue'>${h['revpar']:.0f}</div>"
                    f"<div style='font-size:.7rem'>{trend_arrow(tr['revpar'])} vs prior period</div>"
                    f"</div>", unsafe_allow_html=True)
        m4.markdown(f"<div data-testid='metric-container'>"
                    f"<label>NO-SHOW RATE</label>"
                    f"<div data-testid='stMetricValue'>{h['cancel_rate']:.0%}</div>"
                    f"<div style='font-size:.7rem'>{trend_arrow(tr['cancel_rate'])} vs prior period</div>"
                    f"</div>", unsafe_allow_html=True)

        st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown("#### Things to look at today")
            for a in brief["alerts"]:
                cls = {"warning": "alert-warning", "good": "alert-good"}.get(a["level"], "alert-info")
                icon = {"warning": "⚠️", "good": "✅"}.get(a["level"], "ℹ️")
                st.markdown(f"<div class='{cls}'><b>{icon} {a['title']}</b><br>"
                            f"<span style='font-size:.85rem'>{a['detail']}</span></div>",
                            unsafe_allow_html=True)

            st.markdown("#### Three things you could do today")
            for i, action in enumerate(brief["suggested_actions"], 1):
                st.markdown(f"<div class='action-card'>"
                            f"<b style='color:{GOLD}'>{i}.</b> {action}</div>",
                            unsafe_allow_html=True)

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
                    line=dict(color="rgba(0,0,0,0)"), name="Range",
                    hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=fc["date"], y=fc["expected_occupancy"],
                    mode="lines+markers",
                    line=dict(color=GOLD, width=2.5),
                    marker=dict(size=8, color=GOLD),
                    name="Expected"))
                fig.update_layout(**LAYOUT, height=300, showlegend=False)
                fig.update_yaxes(tickformat=".0%", title="Rooms filled")
                st.plotly_chart(fig, use_container_width=True)
                help_box("The shaded band is the range of likely outcomes. "
                         "If the line is rising, more guests are booked or expected.")
            else:
                st.info("Forecast model not loaded yet. Run `python src/train_models_ts.py`.")


# ╔══════════════════════════════════╗
# ║  TAB 1 — DEMAND OUTLOOK        ║
# ╚══════════════════════════════════╝
with t1:
    st.markdown("### 📈  Demand Outlook")
    help_box("Forecast for the next 30–180 days, broken down by what you want to see: "
             "rooms filled, average rate, or revenue. Useful for staffing, "
             "purchasing, and rate-setting decisions.")

    metric_choice = st.radio(
        "What do you want to forecast?",
        options=["Rooms Filled", "Avg Nightly Rate", "Total Revenue"],
        horizontal=True)
    metric_map = {"Rooms Filled": "occupancy", "Avg Nightly Rate": "adr", "Total Revenue": "revenue"}
    metric = metric_map[metric_choice]
    horizon = st.slider("How far ahead?", 30, 180, 90)

    with st.spinner("Loading forecast…"):
        data, err = api_get(f"/api/v1/forecast/{metric}",
                            params={"horizon_days": horizon, "include_components": True})

    if err:
        st.error(f"Could not load forecast: {err}")
    elif data:
        fc_df = pd.DataFrame(data["forecast"]); ac_df = pd.DataFrame(data["actual_tail"])
        fc_df["date"] = pd.to_datetime(fc_df["date"]); ac_df["date"] = pd.to_datetime(ac_df["date"])
        # Plotly accepts d3 format strings (`$,.0f` is valid).
        # Python f-strings do NOT — `$` must be a literal prefix outside the spec.
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
            mode="lines", line=dict(color=GOLD, width=2.5), name=f"Forecast"))
        if len(ac_df):
            vline_x = ac_df["date"].max().timestamp() * 1000
            fig.add_vline(x=vline_x, line_dash="dash", line_color="rgba(196,155,60,.4)",
                          annotation_text="Today", annotation_font_color=GOLD)

        accuracy = (1 - data.get("mape", 0)) * 100 if data.get("mape") else None
        title = f"{metric_choice} — next {horizon} days"
        if accuracy is not None:
            title += f"  (forecast accuracy on past data: {accuracy:.1f}%)"
        fig.update_layout(**LAYOUT, title=title, height=420)
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
            st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
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
                fig_y.update_layout(**LAYOUT, title="Time-of-year pattern", height=240)
                fig_y.update_xaxes(title="Day of year")
                fig_y.update_yaxes(title="Above/below average")
                st.plotly_chart(fig_y, use_container_width=True)
            with cc2:
                days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                fig_w = go.Figure(go.Bar(
                    x=days, y=comp["weekly"],
                    marker_color=[GOLD if v >= 0 else RED for v in comp["weekly"]]))
                fig_w.update_layout(**LAYOUT, title="Day-of-week pattern", height=240)
                fig_w.update_yaxes(title="Above/below average")
                st.plotly_chart(fig_w, use_container_width=True)


# ╔════════════════════════════════════════╗
# ║  TAB 2 — WILL THEY SHOW UP?         ║
# ╚════════════════════════════════════════╝
with t2:
    st.markdown("### 🚪  Will This Guest Actually Show Up?")
    help_box("Enter a booking and we'll predict the chance the guest will cancel or no-show. "
             "We also explain *why* — so you can decide if it's worth a confirmation call, "
             "a stricter deposit, or letting it go. No jargon: red bars push risk up, "
             "green bars push it down.")

    with st.form("cancel_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hotel    = st.selectbox("Hotel",  ["Resort Hotel","City Hotel"])
            month    = st.selectbox("Arrival month",  ["January","February","March","April","May","June",
                                                       "July","August","September","October","November","December"])
            lead     = st.number_input("Days booked in advance", 0, 700, 60,
                                       help="Bookings made far in advance tend to cancel more often.")
            wknd     = st.number_input("Weekend nights", 0, 10, 1)
            wkk      = st.number_input("Weekday nights",  0, 20, 3)
            adr      = st.number_input("Nightly rate ($)", 0., 5000., 100.)
        with c2:
            adults   = st.number_input("Adults",   1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            country  = st.text_input("Country (3-letter code)","PRT",
                                     help="e.g., USA, GBR, FRA, DEU")
            is_rep   = st.selectbox("Has stayed before?", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")
            prev_can = st.number_input("Past cancellations", 0, 20, 0)
            prev_ok  = st.number_input("Past completed stays", 0, 50, 0)
        with c3:
            meal    = st.selectbox("Meal plan",
                ["BB","HB","FB","SC","Undefined"],
                format_func=lambda x: {"BB":"Bed & Breakfast","HB":"Half board",
                                        "FB":"Full board","SC":"Self-catering",
                                        "Undefined":"Not specified"}[x])
            seg     = st.selectbox("How was it booked?",
                ["Online TA","Direct","Corporate","Offline TA/TO","Groups","Complementary","Aviation"])
            chan    = st.selectbox("Channel", ["TA/TO","Direct","Corporate","GDS","Undefined"],
                format_func=lambda x: {"TA/TO":"Travel agency / OTA","Direct":"Direct",
                                        "Corporate":"Corporate","GDS":"Travel agent (GDS)",
                                        "Undefined":"Other"}[x])
            room    = st.selectbox("Room type", ["A","B","C","D","E","F","G","H","L","P"])
            dep     = st.selectbox("Deposit", ["No Deposit","Non Refund","Refundable"])
            cust    = st.selectbox("Guest type", ["Transient","Contract","Transient-Party","Group"])
            wait    = st.number_input("Days on waiting list", 0, 300, 0)
            spec    = st.number_input("Special requests", 0, 5, 1)
        submitted = st.form_submit_button("Score this booking")

    if submitted:
        body = {
            "hotel":hotel, "lead_time":lead, "arrival_date_month":month,
            "total_stay":wknd+wkk, "total_guests":adults+children,
            "meal":meal, "country":country.upper()[:3],
            "market_segment":seg, "distribution_channel":chan,
            "is_repeated_guest":is_rep, "previous_cancellations":prev_can,
            "previous_bookings_not_canceled":prev_ok,
            "reserved_room_type":room, "booking_changes":0,
            "deposit_type":dep, "days_in_waiting_list":float(wait),
            "customer_type":cust, "required_car_parking_spaces":0,
            "total_of_special_requests":spec, "adr":adr,
        }
        with st.spinner("Scoring…"):
            pred, e1 = api_post("/api/v1/cancellation/predict", body)
            xai,  e2 = api_post("/api/v1/xai/explain", body)

        if e1: st.error(f"Could not score: {e1}")
        if pred:
            cp = pred["cancellation_probability"]
            level_label = {"HIGH":"High risk","MODERATE":"Medium risk","LOW":"Low risk"}.get(
                pred["risk_level"], pred["risk_level"])
            color = RED if cp > 0.6 else AMBER if cp > 0.35 else GREEN

            st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
            r1, r2, r3 = st.columns([1, 1, 1.5])
            r1.metric("Chance they show", f"{(1-cp):.0%}")
            r2.metric("Chance they cancel/no-show", f"{cp:.0%}")
            r3.markdown(f"<div style='background:rgba(14,16,24,.9);"
                        f"border:1px solid {color};border-radius:10px;padding:.85rem 1rem'>"
                        f"<div style='color:{color};font-weight:600;font-size:1rem'>{level_label}</div>"
                        f"<div style='color:#C5C5BF;font-size:.85rem;margin-top:.3rem'>"
                        f"<b>What to do:</b> {pred['recommended_action']}</div></div>",
                        unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=cp*100,
                title={"text":"Cancel / no-show risk", "font":{"color":"#EDD98A","size":13}},
                number={"suffix":"%", "font":{"color":"#EDD98A","size":34}},
                gauge={"axis":{"range":[0, 100]},
                       "bar":{"color":color},
                       "steps":[
                           {"range":[0, 35],   "color":"rgba(76,175,80,.12)"},
                           {"range":[35, 60],  "color":"rgba(255,193,7,.12)"},
                           {"range":[60, 100], "color":"rgba(244,67,54,.12)"}],
                       },
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#C5C5BF"),
                                height=240, margin=dict(t=50, b=5, l=25, r=25))
            st.plotly_chart(fig_g, use_container_width=True)

        if e2:
            st.warning(f"Could not load explanation: {e2}")
        if xai and "waterfall" in xai:
            st.markdown("#### Why this score?")
            help_box("Each bar is a piece of the booking. **Red** bars push the risk up "
                     "(reasons to worry). **Green** bars push it down (reasons to relax). "
                     "Bigger bar = bigger effect.")
            wf = xai["waterfall"]
            names  = [w["feature"] for w in wf]
            values = [w["shap_value"] for w in wf]
            colors = [RED if v > 0 else GREEN for v in values]
            fig_shap = go.Figure(go.Bar(
                x=values, y=names, orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside"))
            fig_shap.update_layout(**LAYOUT, title="What drove the score (red = ↑ risk, green = ↓ risk)",
                                   height=350, xaxis_title="Impact on risk score")
            st.plotly_chart(fig_shap, use_container_width=True)
            st.info(f"🔍 **Biggest reason:** `{xai['top_risk_factor']}`")


# ╔══════════════════════════════════════╗
# ║  TAB 3 — PRICING & INVENTORY     ║
# ╚══════════════════════════════════════╝
with t3:
    st.markdown("### 💰  Pricing & Inventory")
    help_box("Two tools side by side. **Left:** what should we charge tonight given expected "
             "demand? **Right:** how many extra reservations should we accept knowing some will "
             "cancel? The math weighs the cost of an empty room against the cost of having "
             "to walk a guest.")

    p1, p2 = st.columns(2)

    # ── Smart pricing ────────────────────────────────────────────────────
    with p1:
        st.markdown("#### What should we charge?")
        curr_adr = st.number_input("Today's nightly rate ($)", 50., 1000., 120.)
        horizon  = st.slider("Looking ahead how many days?", 7, 90, 30, key="p_hz")
        if st.button("Get pricing recommendation", key="btn_price"):
            with st.spinner("Calculating…"):
                rec, err = api_get("/api/v1/pricing/recommendation",
                                   params={"current_adr":curr_adr, "horizon_days":horizon})
            if err:
                st.error(err)
            elif rec:
                col = GREEN if rec["price_change_pct"] > 0 else RED if rec["price_change_pct"] < 0 else AMBER
                demand_label = ("strong demand — push prices up"
                                if rec["demand_index"] > 1.10 else
                                "soft demand — discount to fill rooms"
                                if rec["demand_index"] < 0.90 else
                                "demand is in line with normal")

                st.markdown(
                    f"<div style='background:rgba(14,16,24,.9);border:1px solid rgba(196,155,60,.2);"
                    f"border-radius:10px;padding:1rem;margin:.5rem 0'>"
                    f"<div style='color:#EDD98A;font-size:1rem;font-weight:600'>{rec['strategy']}</div>"
                    f"<div style='font-size:1.7rem;color:{col};font-family:Cormorant Garamond,serif;margin:.4rem 0'>"
                    f"${rec['current_adr']:.0f} → ${rec['recommended_adr']:.0f} "
                    f"<span style='font-size:.95rem'>({rec['price_change_pct']:+.1f}%)</span></div>"
                    f"<div style='color:#767670;font-size:.78rem'>"
                    f"Estimated lift in revenue per room: <b>${rec['revpar_uplift_est']:+.0f}</b><br>"
                    f"Demand level: <b>{demand_label}</b></div>"
                    f"<div style='color:#C5C5BF;font-size:.82rem;margin-top:.6rem'>{rec['reasoning']}</div>"
                    f"</div>", unsafe_allow_html=True)

                tier_df = pd.DataFrame([
                    {"Room Type":k, "Suggested rate":f"${v:,.0f}"}
                    for k, v in rec["room_tier_prices"].items()
                ])
                st.markdown("**Suggested rates by room type**")
                st.dataframe(tier_df, use_container_width=True, hide_index=True)

    # ── Smart overbooking ────────────────────────────────────────────────
    with p2:
        st.markdown("#### How many extra reservations to take?")
        capacity = st.number_input("Total rooms in hotel", 10, 500, 100)
        max_walk = st.slider("Max acceptable walk risk",
                             0.01, 0.10, 0.05, format="%.0f%%",
                             help="The chance more guests show up than rooms available, "
                                  "forcing us to relocate someone. Most hotels target 3-5%.")

        st.markdown("**Booking sources expected:**")
        tier_defaults = [
            ("VIP / Suites",   10, 0.10, 450),
            ("Standard rooms", 80, 0.28, 120),
            ("OTA / Discount", 40, 0.42,  85),
        ]
        tiers = []
        for name, nb, cp, ad in tier_defaults:
            tc1, tc2, tc3 = st.columns(3)
            nb2 = tc1.number_input(f"{name} — bookings", 0, 300, nb, key=f"nb_{name}")
            cp2 = tc2.number_input(f"Cancel rate", 0.0, 1.0, cp, key=f"cp_{name}", step=0.01)
            ad2 = tc3.number_input(f"Rate $", 0., 2000., float(ad), key=f"ad_{name}")
            tiers.append({"name":name, "n_bookings":nb2, "cancel_prob":cp2,
                          "adr":ad2, "stay_nights":2.0})

        if st.button("Solve", key="btn_ob"):
            body = {"capacity":capacity, "tiers":tiers, "c_empty":500,
                    "c_walk":1500, "max_walk_prob":max_walk}
            with st.spinner("Solving…"):
                res, err = api_post("/api/v1/overbooking/solve", body)
            if err:
                st.error(err)
            elif res:
                ov = res["optimal_overbooking"]
                clr = GREEN if ov < 5 else AMBER if ov < 12 else RED
                st.markdown(
                    f"<div style='background:rgba(14,16,24,.9);border:1px solid {clr};"
                    f"border-radius:10px;padding:1rem;margin:.5rem 0'>"
                    f"<div style='color:{clr};font-size:1.8rem;font-family:Cormorant Garamond,serif'>"
                    f"Take {ov} extra reservations</div>"
                    f"<div style='color:#767670;font-size:.78rem'>"
                    f"Walk risk: {res['walk_probability']:.1%} · "
                    f"Expected profit: ${res['expected_profit']:,.0f}</div>"
                    f"<div style='color:#C5C5BF;font-size:.82rem;margin-top:.5rem'>"
                    f"{res['recommendation']}</div></div>", unsafe_allow_html=True)

                if res.get("sensitivity"):
                    sens = pd.DataFrame(res["sensitivity"])
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=sens["delta"], y=sens["e_profit"],
                        mode="lines+markers", name="Expected profit",
                        line=dict(color=GOLD, width=2)))
                    fig_s.add_vline(x=ov, line_dash="dash", line_color=GREEN,
                                    annotation_text=f"Best: {ov}",
                                    annotation_font_color=GREEN)
                    fig_s.update_layout(**LAYOUT, title="Profit at different overbooking levels",
                                        height=260,
                                        xaxis_title="Extra reservations taken",
                                        yaxis_title="Expected profit ($)")
                    st.plotly_chart(fig_s, use_container_width=True)


# ╔══════════════════════════════════╗
# ║  TAB 4 — GUEST OFFERS           ║
# ╚══════════════════════════════════╝
with t4:
    st.markdown("### 🎯  Personalised Offers for a Guest")
    help_box("Type a guest profile and we'll suggest what to upsell them — spa, room upgrade, "
             "private transfer, etc. Each suggestion comes with a ready-to-send email and an "
             "expected revenue impact. Higher score = more likely they'll say yes.")

    with st.form("rec_form"):
        g1, g2 = st.columns(2)
        with g1:
            g_hotel    = st.selectbox("Hotel", ["Resort Hotel","City Hotel"], key="rec_hotel")
            g_adr      = st.number_input("Nightly rate ($)", 0., 5000., 180., key="rec_adr")
            g_adults   = st.number_input("Adults",   1, 10, 2, key="rec_a")
            g_children = st.number_input("Children", 0, 10, 2, key="rec_c")
            g_stay     = st.number_input("Total nights", 1, 30, 7, key="rec_s")
            g_country  = st.text_input("Country code (e.g., GBR)","GBR", key="rec_co")
        with g2:
            g_meal     = st.selectbox("Meal plan",["BB","HB","FB","SC","Undefined"], key="rec_m")
            g_seg      = st.selectbox("How they booked",
                ["Online TA","Direct","Corporate","Offline TA/TO","Groups","Complementary","Aviation"],
                key="rec_seg")
            g_rep      = st.selectbox("Has stayed before?", [0, 1], key="rec_rep",
                                       format_func=lambda x: "Yes" if x else "No")
            g_prev     = st.number_input("Past completed stays", 0, 50, 2, key="rec_pv")
            g_spec     = st.number_input("Special requests", 0, 5, 2, key="rec_sp")
            top_n      = st.slider("How many offers to show?", 1, 5, 3, key="rec_n")
        rec_submit = st.form_submit_button("Get offers")

    if rec_submit:
        body = {
            "hotel":g_hotel, "adr":g_adr, "adults":g_adults, "children":float(g_children),
            "babies":0, "total_stay":g_stay, "country":g_country.upper()[:3],
            "meal":g_meal, "is_repeated_guest":g_rep,
            "previous_bookings_not_canceled":g_prev,
            "total_of_special_requests":g_spec, "market_segment":g_seg,
        }
        with st.spinner("Generating offers…"):
            result, err = api_post("/api/v1/recommend/next-action", body, timeout=30)
        if err:
            st.error(err)
        elif result:
            loy_col = {"Gold":"#FFD700","Silver":"#C0C0C0","Bronze":"#CD7F32"}.get(
                result["loyalty_tier"], GOLD)
            st.markdown(
                f"<div style='background:rgba(14,16,24,.9);border:1px solid {loy_col};"
                f"border-radius:10px;padding:1rem;margin-bottom:1rem;'>"
                f"<span style='color:{loy_col};font-weight:600'>● {result['loyalty_tier']} guest</span>"
                f"&nbsp;&nbsp;<span style='color:#767670;font-size:.78rem'>"
                f"Estimated extra revenue if they accept: <b>${result['estimated_upsell']:.0f}</b></span></div>",
                unsafe_allow_html=True)
            st.markdown(f"**Top suggestion:** {result['next_best_action']}")

            for rec in result.get("recommendations", []):
                bar = "█" * int(rec["score"] * 10) + "░" * (10 - int(rec["score"] * 10))
                st.markdown(
                    f"<div style='background:rgba(14,16,24,.85);border:1px solid rgba(196,155,60,.15);"
                    f"border-radius:10px;padding:1rem;margin:.5rem 0'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<b style='color:#EDD98A'>{rec['label']}</b>"
                    f"<span style='color:#767670;font-size:.75rem'>"
                    f"Match: {rec['score']:.0%} · Revenue: ${rec['revenue']}</span></div>"
                    f"<div style='color:#BF9740;font-family:monospace;font-size:.75rem;margin:.3rem 0'>"
                    f"{bar}</div>"
                    f"<div style='color:#C5C5BF;font-size:.85rem;font-style:italic;margin-top:.5rem'>"
                    f"📧 {rec['email_copy']}</div></div>",
                    unsafe_allow_html=True)


# ╔══════════════════════════════╗
# ║  TAB 5 — GUEST REVIEWS    ║
# ╚══════════════════════════════╝
with t5:
    st.markdown("### 💬  What Guests Are Saying")
    help_box("Paste a review (or anything a guest said) and we'll tell you if it's positive, "
             "negative, or sarcastic — plus what they're talking about (food, room, service, etc.).")

    eng, _ = api_get("/api/v1/sentiment/engine-info")
    if eng:
        tier = eng.get("active_tier", 3)
        tier_map = {
            1: ("🤗", "Best — using HuggingFace AI",  GREEN),
            2: ("🤖", "Good — using Claude",         GOLD),
            3: ("📝", "Basic — TextBlob (free)",      AMBER),
        }
        ico, lbl, clr = tier_map.get(tier, ("📝", "Basic", AMBER))
        st.markdown(
            f"<div style='background:rgba(14,16,24,.9);border:1px solid {clr};"
            f"border-radius:10px;padding:.7rem 1rem;margin-bottom:.8rem;'>"
            f"<span style='font-size:1.2rem'>{ico}</span>"
            f" <b style='color:{clr}'>Currently active: {lbl}</b>"
            f"</div>", unsafe_allow_html=True)
        if tier == 3:
            st.info("To unlock sarcasm detection and topic analysis, set the "
                    "`HF_API_TOKEN` environment variable (free at huggingface.co/settings/tokens).")

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    user_text = st.text_area("Paste a guest review:", height=100,
        placeholder="e.g., 'Oh wonderful — the AC broke at 2am. Truly a five-star experience.'")
    if st.button("Analyse this review") and user_text:
        with st.spinner("Reading…"):
            result, err = api_post("/api/v1/sentiment/analyse", {"text":user_text})
        if err:
            st.error(err)
        elif result:
            label = result.get("label", "Neutral")
            pol   = result.get("polarity", 0)
            conf  = result.get("confidence", 0)
            sarc  = result.get("sarcasm_flag", False)
            engine = result.get("engine", "")
            label_clr = GREEN if label == "Positive" else RED if label == "Negative" else AMBER
            mood = {"Positive":"Happy guest", "Negative":"Unhappy guest",
                    "Neutral":"Neutral feedback"}.get(label, label)

            lc, rc = st.columns([1, 2])
            with lc:
                badge = "🤗" if "HuggingFace" in engine else "🤖" if "Claude" in engine else "📝"
                sarc_bit = ("<div style='color:#F44336;font-size:.78rem;margin-top:.3rem'>"
                            "⚠️ Sarcasm detected — read carefully</div>") if sarc else ""
                st.markdown(
                    f"<div style='background:rgba(14,16,24,.9);border:1px solid {label_clr};"
                    f"border-radius:10px;padding:1.1rem;'>"
                    f"<div style='color:{label_clr};font-size:1.3rem;font-weight:600'>{mood}</div>"
                    f"<div style='color:#767670;font-size:.78rem;margin:.3rem 0'>"
                    f"How positive: <b>{pol:+.2f}</b> (range -1 to +1)<br>"
                    f"How sure: <b>{conf:.0%}</b></div>"
                    f"{sarc_bit}"
                    f"<div style='color:#444440;font-size:.65rem;margin-top:.5rem'>"
                    f"{badge} {engine}</div></div>",
                    unsafe_allow_html=True)
            with rc:
                aspects = {k:v for k, v in result.get("aspects", {}).items() if v is not None}
                if aspects:
                    fig_a = go.Figure(go.Bar(
                        x=list(aspects.values()), y=list(aspects.keys()),
                        orientation="h",
                        marker_color=[GREEN if v >= 0.1 else RED if v <= -0.1 else AMBER
                                      for v in aspects.values()]))
                    fig_a.update_layout(**LAYOUT, title="What they're talking about",
                                        height=220, margin=dict(t=35, b=15, l=10, r=10))
                    fig_a.update_xaxes(range=[-1, 1],
                                       title="Negative ← → Positive")
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    st.info("Topic-level analysis needs a smarter engine. "
                            "Set `HF_API_TOKEN` or `ANTHROPIC_API_KEY` to enable it.")
                themes = result.get("themes", [])
                if themes:
                    st.markdown("**Topics mentioned:** " + "  ".join([f"`{t}`" for t in themes]))


# ╔════════════════════════════════════════════╗
# ║  TAB 6 — WHERE GUESTS COME FROM          ║
# ╚════════════════════════════════════════════╝
with t6:
    st.markdown("### 🌍  Where Guests Come From")
    help_box("Three views: which booking channels actually make us money "
             "(after commissions); when no-shows tend to happen; and "
             "where in the world your guests are travelling from.")

    sub_a, sub_b, sub_c = st.tabs(["💸 Channel Profitability",
                                   "📅 No-Show Patterns",
                                   "🌎 Guest Origins"])

    # ── Channel mix ─────────────────────────────────────────────────────
    with sub_a:
        st.markdown("#### Which booking channels actually make us money?")
        days = st.slider("Look back how many days?", 30, 730, 180, key="cm_days")
        cm, err = api_get("/api/v1/analytics/channel-mix",
                          params={"lookback_days": days})
        if err:
            st.error(err)
        elif cm:
            s = cm["summary"]
            help_box(cm.get("explanation", ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("Gross revenue",       f"${s['total_gross_revenue']:,.0f}")
            c2.metric("Commissions paid",   f"${s['total_commission']:,.0f}")
            c3.metric("Net (yours to keep)", f"${s['total_net_revenue']:,.0f}",
                      f"-{s['blended_take_rate']:.1%} avg take-rate")

            df = pd.DataFrame(cm["channels"])
            if len(df):
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Gross", x=df["label"], y=df["gross_revenue"],
                                     marker_color="rgba(191,151,64,.4)"))
                fig.add_trace(go.Bar(name="Net (after commission)", x=df["label"],
                                     y=df["net_revenue"], marker_color=GOLD))
                fig.update_layout(**LAYOUT, barmode="overlay", height=380,
                                  title="Revenue by channel — gross vs net")
                fig.update_yaxes(title="Revenue ($)")
                st.plotly_chart(fig, use_container_width=True)

                tbl = df[["label","bookings","gross_revenue","commission_rate",
                          "commission_cost","net_revenue","avg_adr","cancellation_rate"]].copy()
                tbl.columns = ["Channel","Bookings","Gross $","Commission %",
                               "Commission $","Net $","Avg rate","Cancel rate"]
                tbl["Gross $"]        = tbl["Gross $"].map("${:,.0f}".format)
                tbl["Commission $"]   = tbl["Commission $"].map("${:,.0f}".format)
                tbl["Net $"]          = tbl["Net $"].map("${:,.0f}".format)
                tbl["Avg rate"]       = tbl["Avg rate"].map("${:,.0f}".format)
                tbl["Commission %"]   = tbl["Commission %"].map("{:.0%}".format)
                tbl["Cancel rate"]    = tbl["Cancel rate"].map("{:.0%}".format)
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── No-show heatmap ─────────────────────────────────────────────────
    with sub_b:
        st.markdown("#### When are no-shows worst?")
        days = st.slider("Look back how many days?", 60, 1500, 365, key="ns_days")
        ns, err = api_get("/api/v1/analytics/no-show-heatmap",
                          params={"lookback_days": days})
        if err:
            st.error(err)
        elif ns:
            help_box(ns.get("explanation", ""))
            mat = np.array(ns["rate_matrix"]) * 100
            fig = go.Figure(go.Heatmap(
                z=mat, x=ns["months"], y=ns["days"],
                colorscale=[[0, "#1a3a1a"], [0.4, "#5b9b3b"], [0.7, "#FFC107"], [1, "#F44336"]],
                colorbar=dict(title=dict(text="Cancel %",
                                          font=dict(color="#C5C5BF")),
                              tickfont=dict(color="#C5C5BF")),
                text=[[f"{v:.0f}%" for v in row] for row in mat],
                texttemplate="%{text}", textfont={"size":10, "color":"white"}))
            fig.update_layout(**LAYOUT, height=350,
                              title=f"No-show rate by day-of-week × month "
                                    f"(overall avg: {ns['overall_rate']:.0%})")
            st.plotly_chart(fig, use_container_width=True)

    # ── Guest mix ──────────────────────────────────────────────────────
    with sub_c:
        st.markdown("#### Where are your guests coming from?")
        days = st.slider("Look back how many days?", 60, 1500, 365, key="gm_days")
        gm, err = api_get("/api/v1/analytics/guest-mix",
                          params={"lookback_days": days, "top_n": 12})
        if err:
            st.error(err)
        elif gm:
            c1, c2 = st.columns(2)
            with c1:
                if gm["top_countries"]:
                    df = pd.DataFrame(gm["top_countries"])
                    fig = go.Figure(go.Bar(
                        x=df["revenue"], y=df["country"], orientation="h",
                        marker_color=GOLD,
                        text=df["revenue"].map("${:,.0f}".format),
                        textposition="outside"))
                    fig.update_layout(**LAYOUT, title="Top countries by revenue",
                                      height=400, xaxis_title="Revenue ($)")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if gm["segments"]:
                    df = pd.DataFrame(gm["segments"])
                    fig = go.Figure(go.Pie(
                        labels=df["segment"], values=df["revenue"],
                        hole=0.5,
                        marker=dict(colors=[GOLD, BLUE, GREEN, AMBER, RED, "#A8B5C8", "#767670"])))
                    fig.update_layout(**LAYOUT, title="Revenue by booking type", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            st.metric("Repeat-guest share", f"{gm['repeat_share']:.1%}",
                      help="The share of recent bookings from guests who've stayed before. "
                           "Higher is generally better — they cost less to acquire and cancel less.")


# ╔══════════════════════════════════╗
# ║  TAB 7 — DIAGNOSTICS          ║
# ╚══════════════════════════════════╝
with t7:
    st.markdown("### 🔬  Model Diagnostics")
    help_box("Technical view for whoever maintains the system. Shows which inputs the "
             "no-show model relies on most, and how much each forecast input matters. "
             "Skip this tab unless you're debugging.")

    xa1, xa2 = st.columns(2)

    with xa1:
        st.markdown("#### What drives the no-show predictions?")
        n_samp = st.slider("Sample size", 100, 500, 200)
        if st.button("Compute"):
            with st.spinner("Computing (may take 30s)…"):
                gi, err = api_get("/api/v1/xai/global-importance",
                                  params={"n_samples": n_samp}, timeout=90)
            if err:
                st.error(err)
            elif gi:
                names, vals = gi["feature_names"], gi["mean_abs_shap"]
                fig = go.Figure(go.Bar(
                    x=vals, y=names, orientation="h",
                    marker_color=[GOLD if i < 3 else BLUE for i in range(len(names))],
                    text=[f"{v:.4f}" for v in vals], textposition="outside"))
                fig.update_layout(**LAYOUT, height=480,
                                  title="Most influential booking attributes",
                                  xaxis_title="Average impact on prediction")
                st.plotly_chart(fig, use_container_width=True)

    with xa2:
        st.markdown("#### Forecast input significance")
        if st.button("Load results"):
            with st.spinner("Loading…"):
                abl, err = api_get("/api/v1/xai/ablation", timeout=10)
            if err:
                st.warning(f"{err} — run `python src/ablation_study.py` first.")
            elif abl:
                for target_name, result in abl.items():
                    st.markdown(f"**{target_name}** — base accuracy "
                                f"{(1-result['baseline']['mape']):.1%}")
                    rows = []
                    for a in result.get("ablations", []):
                        rows.append({
                            "Removed":      a["removed_regressor"],
                            "Accuracy lost":f"{(a['mape_delta'] * 100):+.2f}pts",
                            "p-value":      f"{a['p_value']:.3f}",
                            "Matters?":     "✅ yes" if a["significant_p05"] else "❌ no",
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                     hide_index=True, height=240)
                    st.markdown("---")
