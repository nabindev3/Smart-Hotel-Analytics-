"""
generate_data.py — Production-Grade Hotel Data Generator
=========================================================
Simulates the REAL-WORLD messiness a data pipeline must defend against:
  • ~6% missing values (MCAR + MAR patterns)
  • ~2% outlier injections (fat-finger ADR, impossible lead times)
  • Concept drift: booking behaviour shifts 2022→2024 (post-COVID recovery)
  • Seasonal noise: heterogeneous variance (weekends noisier than weekdays)
  • External regressors: weather, local events, macro-economics, competitor rates

Outputs
-------
  data/bookings.csv       — 60,000 messy bookings (2019–2024)
  data/daily_kpis.csv     — 2,192-day daily KPI series
  data/external_regs.csv  — daily external regressor series (for Prophet)
  data/reviews.csv        — 30 timestamped reviews
  data/data_quality.json  — quality report (missing%, outlier%, drift score)
"""

import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(2024)

# ── Constants ─────────────────────────────────────────────────────────────
START  = date(2019, 1, 1)
END    = date(2024, 12, 31)
N      = 60_000
CAPACITY = 120

MONTHS   = ["January","February","March","April","May","June",
             "July","August","September","October","November","December"]
COUNTRIES = ["PRT","GBR","FRA","ESP","DEU","ITA","USA","BRA",
              "NLD","BEL","CHE","AUT","IRL","AUS","JPN","SGP","CAN","POL"]
MEALS    = ["BB","HB","FB","SC","Undefined"]
SEGS     = ["Online TA","Direct","Corporate","Offline TA/TO",
             "Groups","Complementary","Aviation"]
CHANS    = ["TA/TO","Direct","Corporate","GDS","Undefined"]
ROOMS    = ["A","B","C","D","E","F","G","H","L","P"]
DEPS     = ["No Deposit","Non Refund","Refundable"]
CUSTS    = ["Transient","Contract","Transient-Party","Group"]


# ── 1. SEASONAL WEIGHT ────────────────────────────────────────────────────
def seasonal_weight(month_idx: int, year: int) -> float:
    """Summer peak + Xmas bump + post-COVID recovery ramp."""
    base_summer = 0.35 * np.sin(2 * np.pi * (month_idx - 5) / 12)
    xmas_bump   = 0.15 * np.exp(-((month_idx - 11)**2) / 2)
    recovery    = 1.0 + max(0, (year - 2021)) * 0.07  # +7%/yr post-COVID
    return (0.50 + base_summer + xmas_bump) * recovery


# ── 2. CONCEPT DRIFT ─────────────────────────────────────────────────────
def drift_cancel_prob(base_prob: np.ndarray, year: np.ndarray,
                      month: np.ndarray) -> np.ndarray:
    """
    Simulate concept drift:
      2019     → stable patterns
      2020     → COVID spike in Q1-Q3
      2021     → gradual recovery; high uncertainty
      2022-24  → shift to shorter lead-times, lower cancellation from
                  direct bookings (OTA market share changes)
    """
    p = base_prob.copy()
    covid_mask = (year == 2020) & (month <= 9)
    p[covid_mask] = (p[covid_mask] + 0.35).clip(0, 0.97)
    drift_mask = year >= 2022
    p[drift_mask] = (p[drift_mask] * 0.88).clip(0.02, 0.95)
    return p


# ── 3. INJECT MISSINGNESS ─────────────────────────────────────────────────
def inject_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    MCAR: 'country', 'children', 'meal' missing completely at random (~3%)
    MAR:  'adr' missing when 'market_segment' == Complementary (~25% of those)
          'days_in_waiting_list' missing when 'deposit_type' == Non Refund
    """
    rng = np.random.default_rng(42)

    # MCAR
    for col, rate in [("country", 0.028), ("children", 0.032), ("meal", 0.018)]:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan

    # MAR — ADR missing for Complementary bookings
    comp_mask = df["market_segment"] == "Complementary"
    missing_mask = comp_mask & (rng.random(len(df)) < 0.22)
    df.loc[missing_mask, "adr"] = np.nan

    # MAR — waiting list missing for Non Refund deposits
    nr_mask = df["deposit_type"] == "Non Refund"
    df.loc[nr_mask & (rng.random(len(df)) < 0.15), "days_in_waiting_list"] = np.nan

    return df


# ── 4. INJECT OUTLIERS ────────────────────────────────────────────────────
def inject_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    ~2% of rows get one of:
      • Fat-finger ADR (10× or 0.1×)
      • Impossible lead time (1000+ days)
      • Negative children count (data entry error)
    """
    rng = np.random.default_rng(99)
    n_outliers = int(len(df) * 0.02)
    idx = rng.choice(len(df), n_outliers, replace=False)

    fat_adr  = idx[:n_outliers // 3]
    bad_lead = idx[n_outliers // 3: 2 * n_outliers // 3]

    df.loc[fat_adr,  "adr"]       = df.loc[fat_adr,  "adr"] * np.random.choice([10, 0.1], len(fat_adr))
    df.loc[bad_lead, "lead_time"] = rng.integers(800, 1500, len(bad_lead))

    return df


# ── 5. GENERATE BOOKINGS ─────────────────────────────────────────────────
def generate_bookings() -> pd.DataFrame:
    all_dates = [START + timedelta(d) for d in range((END - START).days + 1)]
    weights   = np.array([seasonal_weight(d.month-1, d.year) for d in all_dates],
                          dtype=float)
    weights  /= weights.sum()

    arrival_dates = np.random.choice(all_dates, N, p=weights, replace=True)
    years   = np.array([d.year  for d in arrival_dates])
    months  = np.array([d.month for d in arrival_dates])

    lead_time = np.random.exponential(72, N).clip(0, 700).astype(int)
    deposit   = np.random.choice([0, 1, 2], N, p=[0.87, 0.10, 0.03])
    repeated  = np.random.choice([0, 1],    N, p=[0.97, 0.03])
    prev_can  = np.random.choice(range(10), N,
                    p=[0.84,0.06,0.03,0.02,0.01,0.01,0.01,0.01,0.005,0.005])
    spec_req  = np.random.choice([0,1,2,3,4,5], N, p=[0.54,0.25,0.11,0.05,0.03,0.02])
    seg_arr   = np.random.choice(SEGS, N, p=[0.44,0.16,0.12,0.10,0.08,0.05,0.05])

    # Base cancellation probability
    base_p = (
        0.05
        + 0.28 * (lead_time / 700)
        + 0.18 * (deposit == 0)
        - 0.12 * repeated
        + 0.10 * np.minimum(prev_can / 5, 1)
        - 0.07 * np.minimum(spec_req / 3, 1)
        + 0.08 * np.isin(seg_arr, ["Online TA","Groups"])
        + 0.05 * np.isin(seg_arr, ["Complementary"])
        + 0.03 * np.random.randn(N)
    ).clip(0.02, 0.96)

    # Apply concept drift
    cancel_prob = drift_cancel_prob(base_p, years, months)
    is_canceled = (np.random.rand(N) < cancel_prob).astype(int)

    # ADR: YoY growth + seasonality + heterogeneous noise
    year_trend  = 1 + 0.045 * (years - 2019)
    season_mod  = 1 + 0.22 * np.array([seasonal_weight(m-1, y) - 0.5
                                         for m, y in zip(months, years)])
    covid_dip   = np.where((years == 2020) & (months <= 9), 0.52, 1.0)
    weekend_noise = np.where(np.array([d.weekday() for d in arrival_dates]) >= 5,
                              1 + 0.06 * np.abs(np.random.randn(N)), 1.0)
    adr = (np.random.normal(118, 42, N).clip(20, 4000)
           * year_trend * season_mod * covid_dip * weekend_noise).round(2)

    adults   = np.random.choice([1,2,3,4], N, p=[0.25,0.55,0.15,0.05])
    children = np.random.choice([0,1,2,3], N, p=[0.75,0.15,0.08,0.02]).astype(float)
    babies   = np.random.choice([0,1,2],   N, p=[0.95,0.04,0.01])
    wkend_n  = np.random.randint(0, 5, N)
    week_n   = np.random.randint(0, 10, N)

    df = pd.DataFrame({
        "booking_id":                     range(1, N+1),
        "arrival_date":                   [d.isoformat() for d in arrival_dates],
        "arrival_date_year":              years,
        "arrival_date_month":             [MONTHS[m-1] for m in months],
        "arrival_date_week":              [d.isocalendar()[1] for d in arrival_dates],
        "hotel":                          np.random.choice(["Resort Hotel","City Hotel"], N, p=[0.4,0.6]),
        "is_canceled":                    is_canceled,
        "lead_time":                      lead_time,
        "stays_in_weekend_nights":        wkend_n,
        "stays_in_week_nights":           week_n,
        "adults":                         adults,
        "children":                       children,
        "babies":                         babies,
        "meal":                           np.random.choice(MEALS, N, p=[0.60,0.15,0.10,0.10,0.05]),
        "country":                        np.random.choice(COUNTRIES, N),
        "market_segment":                 seg_arr,
        "distribution_channel":           np.random.choice(CHANS, N, p=[0.50,0.20,0.15,0.10,0.05]),
        "is_repeated_guest":              repeated,
        "previous_cancellations":         prev_can,
        "previous_bookings_not_canceled": np.random.randint(0, 10, N),
        "reserved_room_type":             np.random.choice(ROOMS, N,
                                              p=[0.45,0.10,0.08,0.12,0.08,0.06,0.04,0.03,0.02,0.02]),
        "booking_changes":                np.random.randint(0, 5, N),
        "deposit_type":                   np.where(deposit==0,"No Deposit",
                                              np.where(deposit==1,"Non Refund","Refundable")),
        "days_in_waiting_list":           np.random.exponential(3, N).clip(0, 200).astype(int).astype(float),
        "customer_type":                  np.random.choice(CUSTS, N, p=[0.70,0.10,0.15,0.05]),
        "adr":                            adr,
        "required_car_parking_spaces":    np.random.choice([0,1,2], N, p=[0.92,0.07,0.01]),
        "total_of_special_requests":      spec_req,
    })

    # Derived
    df["total_stay"]   = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"].fillna(0) + df["babies"]

    # Inject real-world messiness
    df = inject_missing(df)
    df = inject_outliers(df)

    # Revenue only for fulfilled bookings (NaN for cancelled)
    df["revenue"] = (df["adr"].fillna(df["adr"].median()) *
                     df["total_stay"].clip(1) *
                     (1 - df["is_canceled"])).round(2)

    return df


# ── 6. DAILY KPIs ─────────────────────────────────────────────────────────
def generate_daily_kpis(bookings: pd.DataFrame) -> pd.DataFrame:
    bk = bookings.copy()
    bk["ds"]  = pd.to_datetime(bk["arrival_date"])
    bk["adr"] = bk["adr"].fillna(bk["adr"].median())

    all_days = pd.date_range(START.isoformat(), END.isoformat(), freq="D")
    base = pd.DataFrame({"ds": all_days})

    grp = bk.groupby("ds").agg(
        total_bookings   =("booking_id",  "count"),
        cancellations    =("is_canceled", "sum"),
        revenue          =("revenue",     "sum"),
        avg_adr          =("adr",         "mean"),
        avg_lead_time    =("lead_time",   "mean"),
        avg_stay         =("total_stay",  "mean"),
        avg_special_req  =("total_of_special_requests","mean"),
        missing_rate     =("adr",         lambda x: x.isna().mean()),
    ).reset_index()

    daily = base.merge(grp, on="ds", how="left").fillna({"total_bookings":0,
                                                           "cancellations":0,"revenue":0})

    # Interpolate sparse metrics
    for col in ["avg_adr","avg_lead_time","avg_stay","avg_special_req"]:
        daily[col] = daily[col].interpolate(method="linear").round(2)
    daily["missing_rate"] = daily["missing_rate"].fillna(0)

    daily["fulfilled_bookings"] = daily["total_bookings"] - daily["cancellations"]
    daily["occupancy_rate"]     = (daily["fulfilled_bookings"] / CAPACITY).clip(0,1).round(4)
    daily["cancellation_rate"]  = np.where(daily["total_bookings"]>0,
                                             daily["cancellations"]/daily["total_bookings"],0).round(4)
    daily["revpar"]             = (daily["avg_adr"] * daily["occupancy_rate"]).round(2)
    daily["day_of_week"]        = daily["ds"].dt.dayofweek
    daily["month"]              = daily["ds"].dt.month
    daily["year"]               = daily["ds"].dt.year
    daily["is_weekend"]         = (daily["day_of_week"] >= 5).astype(int)

    return daily.reset_index(drop=True)


# ── 7. EXTERNAL REGRESSORS ────────────────────────────────────────────────
def generate_external_regressors(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add external signals that are NOT in the bookings data:
      temperature_c       — avg daily temp at hotel location (coastal Portugal)
      precipitation_mm    — daily precipitation
      local_events        — binary: major conference / concert / festival nearby
      holiday_flag        — national public holiday
      competitor_adr      — competitor hotel average daily rate
      cpi_yoy             — Consumer Price Index (YoY %)
      consumer_confidence — monthly consumer confidence index (Eurostat-style)
      search_trend        — relative Google Trends search volume proxy
    """
    rng = np.random.default_rng(77)
    n   = len(daily)
    ds  = daily["ds"]

    # Temperature (Portugal coastal): peaks ~28°C in July, min ~12°C in Jan
    day_of_year = ds.dt.dayofyear
    temperature = (
        20.0 + 8.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + rng.normal(0, 1.5, n)
    ).round(1)

    # Precipitation: inverse of temperature season, higher winter
    precip_base = (
        5.0 - 4.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ).clip(0)
    precipitation = (precip_base * rng.exponential(1.0, n)).clip(0, 80).round(1)

    # Local events (conferences, concerts, festivals): ~8% of days
    local_events = (rng.random(n) < 0.08).astype(int)
    # Events cluster in summer and autumn
    event_boost  = (np.sin(2 * np.pi * (day_of_year - 150) / 365) + 1) / 2
    local_events = np.where(rng.random(n) < 0.06 * event_boost, 1, local_events)

    # Holiday flag: ~12 per year
    holiday_dates = set()
    for yr in range(2019, 2025):
        for (m, d) in [(1,1),(4,25),(5,1),(6,10),(8,15),(10,5),
                        (11,1),(12,1),(12,8),(12,25)]:
            holiday_dates.add(date(yr, m, d).isoformat())
        # Easter (approximate)
        holiday_dates.add(date(yr, 4, 10).isoformat())
        holiday_dates.add(date(yr, 4, 11).isoformat())

    holiday_flag = ds.apply(lambda x: 1 if x.date().isoformat()
                             in holiday_dates else 0).values

    # Competitor ADR: correlated with own ADR but with own noise
    own_adr     = daily["avg_adr"].values
    competitor  = (own_adr * rng.uniform(0.88, 1.12, n)
                   + rng.normal(0, 8, n)).clip(50, 600).round(2)

    # CPI YoY (%): stable 2019, low 2020-21, spiking 2022, easing 2023-24
    year_arr = ds.dt.year.values
    cpi_base = {2019: 0.9, 2020: 0.2, 2021: 1.5, 2022: 8.1, 2023: 5.4, 2024: 2.8}
    cpi = (np.array([cpi_base.get(y, 2.0) for y in year_arr])
           + rng.normal(0, 0.3, n)).clip(-0.5, 12).round(2)

    # Consumer confidence (monthly, higher = better): 0–120 scale
    # 2020: crashed; 2021-22: recovery with inflation dip; 2023-24: normalise
    month_arr  = ds.dt.month.values
    conf_trend = {2019:100, 2020:72, 2021:88, 2022:82, 2023:95, 2024:101}
    conf_monthly = (np.array([conf_trend.get(y, 95) for y in year_arr])
                     + 4 * np.sin(2 * np.pi * (month_arr - 5) / 12)
                     + rng.normal(0, 2, n)).clip(50, 120).round(1)

    # Search trend (Google Trends proxy): seasonal + growing YoY
    search_trend = (
        60 + 20 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        + 3 * (year_arr - 2019)
        + rng.normal(0, 5, n)
    ).clip(10, 100).round(1)

    ext = pd.DataFrame({
        "ds":                 ds,
        "temperature_c":      temperature,
        "precipitation_mm":   precipitation,
        "local_events":       local_events,
        "holiday_flag":       holiday_flag,
        "competitor_adr":     competitor,
        "cpi_yoy":            cpi,
        "consumer_confidence":conf_monthly,
        "search_trend":       search_trend,
    })
    return ext


# ── 8. DATA QUALITY REPORT ────────────────────────────────────────────────
def build_quality_report(bookings: pd.DataFrame) -> dict:
    missing_pct = (bookings.isna().sum() / len(bookings) * 100).round(2).to_dict()
    total_missing = bookings.isna().mean().mean() * 100

    # Outlier detection via IQR
    outlier_counts = {}
    for col in ["adr","lead_time","days_in_waiting_list"]:
        s = bookings[col].dropna()
        q1,q3 = s.quantile(0.25), s.quantile(0.75)
        iqr   = q3 - q1
        outs  = ((s < q1-3*iqr) | (s > q3+3*iqr)).sum()
        outlier_counts[col] = int(outs)

    # Drift score: compare cancel rate 2019 vs 2023
    cr_2019 = bookings[bookings["arrival_date_year"]==2019]["is_canceled"].mean()
    cr_2023 = bookings[bookings["arrival_date_year"]==2023]["is_canceled"].mean()
    drift_score = abs(cr_2019 - cr_2023)

    return {
        "total_rows":      len(bookings),
        "missing_pct":     missing_pct,
        "total_missing_pct": round(total_missing, 2),
        "outlier_counts":  outlier_counts,
        "drift_score":     round(float(drift_score), 4),
        "cancel_rate_by_year": {
            str(yr): round(float(bookings[bookings["arrival_date_year"]==yr]["is_canceled"].mean()), 3)
            for yr in sorted(bookings["arrival_date_year"].unique())
        },
        "data_quality_grade": (
            "A" if total_missing < 3 and drift_score < 0.05 else
            "B" if total_missing < 8 and drift_score < 0.10 else "C"
        ),
    }


# ── 9. REVIEWS ────────────────────────────────────────────────────────────
def generate_reviews() -> pd.DataFrame:
    return pd.DataFrame([
        {"date":"2024-01-10","nationality":"United Kingdom","score":9.8,
         "text":"Absolutely stunning property. The lobby alone takes your breath away. Staff were impeccable — every request met with extraordinary professionalism. The spa treatment was transcendent."},
        {"date":"2024-01-12","nationality":"United States","score":9.5,
         "text":"The rooftop pool and ocean view exceeded every expectation. Concierge arranged a private yacht tour — unforgettable experience."},
        {"date":"2024-01-15","nationality":"France","score":8.2,
         "text":"Magnifique! The suite was immaculate. Restaurant service was slightly slow on Friday, though the cuisine was exceptional. Would absolutely return."},
        {"date":"2024-01-18","nationality":"Germany","score":4.5,
         "text":"Room was far smaller than photos suggested. Noise from the bar kept us awake until 2am. Price point entirely unjustifiable."},
        {"date":"2024-01-20","nationality":"Japan","score":10.0,
         "text":"Perfect stay from arrival to departure. The pillow menu was a thoughtful touch. Housekeeping was outstanding and discreet."},
        {"date":"2024-01-22","nationality":"Australia","score":9.7,
         "text":"Truly world-class hospitality. Butler service was remarkable — anticipating our needs before we voiced them."},
        {"date":"2024-01-25","nationality":"Canada","score":2.0,
         "text":"Terrible check-in. Waited over an hour despite a confirmed reservation. Room smelled musty. Management completely dismissive of our complaints."},
        {"date":"2024-01-28","nationality":"Spain","score":5.5,
         "text":"Good location but the property is showing its age. Bar team seemed undertrained. Gym equipment badly outdated."},
        {"date":"2024-02-01","nationality":"Italy","score":9.3,
         "text":"The afternoon tea service is an absolute must-do. Our suite had a private terrace with jaw-dropping mountain views."},
        {"date":"2024-02-03","nationality":"Brazil","score":7.0,
         "text":"Wonderful ambiance and beautiful architecture. Pool is stunning. Breakfast decent but nothing extraordinary. Parking was a hassle."},
        {"date":"2024-02-05","nationality":"Netherlands","score":9.9,
         "text":"Outstanding in every regard. The sommelier's wine pairings during our anniversary dinner were inspired. Personalised turndown service was a lovely touch."},
        {"date":"2024-02-08","nationality":"Sweden","score":7.5,
         "text":"Clean modern rooms. Staff professional if a bit formal. Sauna and hydrotherapy pool were excellent. Solid reliable stay."},
        {"date":"2024-02-11","nationality":"China","score":9.6,
         "text":"The penthouse suite view was absolutely spectacular. Every detail considered — locally-sourced bath products, handwritten welcome note. Exceptional service."},
        {"date":"2024-02-14","nationality":"Russia","score":3.0,
         "text":"Overpriced and underwhelming. Restaurant closed two nights of our stay with no prior notice. Front desk unhelpful and unapologetic."},
        {"date":"2024-02-17","nationality":"Mexico","score":8.0,
         "text":"Lovely stay overall. Beachfront location is superb. Kids club kept our children wonderfully entertained. Minor maintenance issues quickly resolved."},
        {"date":"2024-02-20","nationality":"India","score":9.4,
         "text":"Flawless experience. The cultural sensitivity of the team was impressive — dietary requirements accommodated without question."},
        {"date":"2024-02-22","nationality":"South Korea","score":8.8,
         "text":"Well-equipped gym for a luxury property. Staff courteous and multilingual. Rooftop bar has wonderful sunset views."},
        {"date":"2024-02-25","nationality":"Switzerland","score":4.0,
         "text":"Mediocre experience for the price. WiFi was intermittent throughout despite multiple reports to reception. Room décor felt dated."},
        {"date":"2024-03-01","nationality":"UAE","score":10.0,
         "text":"Extraordinary service and opulent surroundings. Private airport transfer set the tone perfectly. Every staff member knew our names by day two."},
        {"date":"2024-03-04","nationality":"Singapore","score":7.8,
         "text":"Good hotel in prime location. Rooms modern and comfortable. Checkout process was unnecessarily slow."},
        {"date":"2024-03-10","nationality":"United Kingdom","score":9.1,
         "text":"Second visit and standards have only improved. The new garden terrace is a beautiful addition. Exceptional value at this level."},
        {"date":"2024-03-15","nationality":"Germany","score":6.5,
         "text":"Average stay for the price. Breakfast was good but room service took 45 minutes. Would not rush back."},
        {"date":"2024-03-20","nationality":"France","score":9.0,
         "text":"Le service était irréprochable. Pool heated perfectly, food excellent. Will definitely return in the summer."},
        {"date":"2024-04-02","nationality":"USA","score":8.5,
         "text":"Excellent business hotel. Meeting rooms well-equipped, concierge incredibly helpful with last-minute arrangements."},
        {"date":"2024-04-10","nationality":"Japan","score":9.8,
         "text":"Meticulous attention to detail throughout. The omakase dinner arranged by the concierge was extraordinary. Five stars unequivocally."},
        {"date":"2024-04-18","nationality":"Australia","score":5.0,
         "text":"Disappointed. Advertised spa closed for renovation with zero notification. Promised partial refund still has not arrived three weeks later."},
        {"date":"2024-05-05","nationality":"Brazil","score":8.3,
         "text":"Great stay overall. The cocktail bar on the 14th floor has sensational views. Rooms are beautifully appointed."},
        {"date":"2024-05-20","nationality":"Canada","score":9.2,
         "text":"Came for the food and left obsessed with everything. The pastry chef deserves their own Michelin star."},
        {"date":"2024-06-01","nationality":"Italy","score":7.2,
         "text":"Beautiful property but slightly overcrowded in summer. Private pool access is absolutely worth the upgrade."},
        {"date":"2024-06-15","nationality":"Spain","score":9.5,
         "text":"The honeymoon package was flawlessly executed. Champagne, rose petals, private dinner on the terrace — genuinely magical."},
    ])


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating 60,000 messy hotel bookings (2019–2024)…")
    bookings = generate_bookings()
    bookings.to_csv("data/bookings.csv", index=False)
    print(f"  ✓ bookings.csv  — {len(bookings):,} rows | "
          f"cancel rate: {bookings['is_canceled'].mean():.1%} | "
          f"missing: {bookings.isna().mean().mean():.1%}")

    print("Aggregating daily KPI series…")
    daily = generate_daily_kpis(bookings)
    daily.to_csv("data/daily_kpis.csv", index=False)
    print(f"  ✓ daily_kpis.csv — {len(daily):,} days")

    print("Generating external regressors…")
    ext = generate_external_regressors(daily)
    ext.to_csv("data/external_regs.csv", index=False)
    print(f"  ✓ external_regs.csv — {ext.shape[1]-1} regressors over {len(ext):,} days")

    print("Building data quality report…")
    report = build_quality_report(bookings)
    with open("data/data_quality.json","w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ data_quality.json — grade: {report['data_quality_grade']} | "
          f"drift score: {report['drift_score']:.4f}")

    print("Generating reviews…")
    reviews = generate_reviews()
    reviews.to_csv("data/reviews.csv", index=False)
    print(f"  ✓ reviews.csv — {len(reviews)} reviews")

    print("\n✅  Data generation complete.\n")
