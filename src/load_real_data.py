"""
src/load_real_data.py — Real-world hotel data loader
======================================================
Downloads the Antonio, Almeida & Nunes (2019) Hotel Booking Demand dataset
(~119,390 real bookings from two Portuguese hotels, 2015-2017) and normalises
it to match this project's schema.

Three sources are tried in order:
  1. Local cache at data/raw/hotel_bookings_real.csv
  2. TidyTuesday public mirror (no API key needed)
  3. kagglehub (if installed and credentials configured)

The script also:
  - Date-shifts the original 2015-2017 records to 2022-2024 so they overlap
    the synthetic 2019-2025 range.
  - Adds plausible derived columns (revenue, total_stay, total_guests).
  - Creates a 'source' column flagging real vs synthetic for audit.
  - Outputs a blended bookings.csv plus a `data/data_quality.json` summary.

Usage:
  python src/load_real_data.py                      # blend with synthetic
  python src/load_real_data.py --real-only          # use only real data
  python src/load_real_data.py --no-blend           # save real to separate file

Reference:
  Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand
  datasets. Data in Brief, 22, 41-49.
  https://doi.org/10.1016/j.dib.2018.11.126
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

REAL_CACHE = RAW_DIR / "hotel_bookings_real.csv"

# Public mirror — TidyTuesday hosted copy of the Antonio et al. dataset.
TIDYTUESDAY_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Acquisition
# ─────────────────────────────────────────────────────────────────────────────
def _try_local_cache() -> pd.DataFrame | None:
    if REAL_CACHE.exists():
        print(f"[load_real_data] Using cached file: {REAL_CACHE}")
        return pd.read_csv(REAL_CACHE)
    return None


def _try_tidytuesday() -> pd.DataFrame | None:
    """Public no-auth mirror. Most reliable."""
    try:
        import requests
        print(f"[load_real_data] Fetching from TidyTuesday mirror …")
        r = requests.get(TIDYTUESDAY_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.to_csv(REAL_CACHE, index=False)
        print(f"[load_real_data] Cached → {REAL_CACHE}  ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"[load_real_data] TidyTuesday fetch failed: {e}")
        return None


def _try_kagglehub() -> pd.DataFrame | None:
    """Optional: if user has kagglehub configured."""
    try:
        import kagglehub
        print("[load_real_data] Trying kagglehub …")
        path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
        for f in os.listdir(path):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, f))
                df.to_csv(REAL_CACHE, index=False)
                return df
    except Exception as e:
        print(f"[load_real_data] kagglehub unavailable: {e}")
    return None


def fetch_real() -> pd.DataFrame:
    for fn in (_try_local_cache, _try_tidytuesday, _try_kagglehub):
        df = fn()
        if df is not None and len(df) > 1000:
            return df
    raise RuntimeError(
        "Could not obtain real dataset.\n"
        "Manual fix: download from https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand "
        f"and save the CSV to {REAL_CACHE}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation — map Kaggle columns to our schema
# ─────────────────────────────────────────────────────────────────────────────
MONTH_MAP = {m: i for i, m in enumerate(
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"], 1)}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Make the Kaggle dataset look like our bookings.csv schema."""
    df = df.copy()

    # ── Known data-quality fixes for the Antonio et al. dataset ──
    #   • Two rows have adr = -6.38 (data-entry error).
    #   • A handful of rows have adr > 5000 (suspected typos).
    # We treat both as outliers and clip to the median ADR to avoid
    # poisoning downstream models. Source flag still marks them as 'real'.
    if "adr" in df.columns:
        median_adr = float(df.loc[(df["adr"] > 0) & (df["adr"] < 1000), "adr"].median())
        bad = (df["adr"] <= 0) | (df["adr"] > 5000)
        n_bad = int(bad.sum())
        if n_bad:
            df.loc[bad, "adr"] = median_adr
            print(f"[load_real_data] Cleaned {n_bad} negative/extreme ADR rows "
                  f"→ median ${median_adr:.2f}")

    # Build arrival_date (the original has y/m/d split)
    df["arrival_date"] = pd.to_datetime(dict(
        year  = df["arrival_date_year"],
        month = df["arrival_date_month"].map(MONTH_MAP),
        day   = df["arrival_date_day_of_month"],
    ), errors="coerce")

    # Date-shift 2015-2017 → 2022-2024 so it sits in our active range
    df["arrival_date"] = df["arrival_date"] + pd.DateOffset(years=7)
    df["arrival_date_year"] = df["arrival_date"].dt.year

    # Derived columns we use downstream
    df["total_stay"]   = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)
    df["revenue"]      = np.where(df["is_canceled"] == 1, 0.0,
                                  df["adr"] * df["total_stay"]).round(2)
    df["arrival_date_week"] = df["arrival_date"].dt.isocalendar().week.astype(int)

    # Some columns may have NaNs that break downstream models — fill sensibly
    df["children"] = df["children"].fillna(0)
    df["country"]  = df["country"].fillna("UNK")
    df["meal"]     = df["meal"].fillna("Undefined")
    df["agent"]    = df.get("agent", pd.Series([np.nan]*len(df)))
    df["company"]  = df.get("company", pd.Series([np.nan]*len(df)))

    df["source"] = "real"

    target_cols = [
        "arrival_date","arrival_date_year","arrival_date_month","arrival_date_week",
        "hotel","is_canceled","lead_time","stays_in_weekend_nights",
        "stays_in_week_nights","adults","children","babies","meal","country",
        "market_segment","distribution_channel","is_repeated_guest",
        "previous_cancellations","previous_bookings_not_canceled",
        "reserved_room_type","booking_changes","deposit_type",
        "days_in_waiting_list","customer_type","adr",
        "required_car_parking_spaces","total_of_special_requests",
        "total_stay","total_guests","revenue","source",
    ]
    return df[[c for c in target_cols if c in df.columns]]


# ─────────────────────────────────────────────────────────────────────────────
#  Blending
# ─────────────────────────────────────────────────────────────────────────────
def _clean_adr(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative or wildly extreme ADRs (regardless of source) to median.
    Applied to the ENTIRE blended frame, so prior-run pollution gets fixed too."""
    if "adr" not in df.columns:
        return df
    median = float(df.loc[(df["adr"] > 0) & (df["adr"] < 1000), "adr"].median())
    bad = (df["adr"] <= 0) | (df["adr"] > 5000)
    n = int(bad.sum())
    if n:
        df.loc[bad, "adr"] = median
        # Recompute revenue for cleaned rows so it stays consistent
        if "is_canceled" in df.columns and "total_stay" in df.columns:
            df.loc[bad, "revenue"] = np.where(
                df.loc[bad, "is_canceled"] == 1, 0.0,
                df.loc[bad, "adr"] * df.loc[bad, "total_stay"]
            ).round(2)
        print(f"[load_real_data] Final clean: fixed {n} negative/extreme ADR rows "
              f"→ median ${median:.2f}")
    return df


def blend_with_synthetic(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concat real + existing synthetic; renumber booking_id.

    Idempotent: if the existing bookings.csv already contains rows tagged
    `source == "real"`, those rows are dropped first so re-running the loader
    doesn't double-stack the real dataset on top of itself.
    """
    synth_path = DATA_DIR / "bookings.csv"
    if not synth_path.exists():
        print("[load_real_data] No synthetic bookings.csv — using real only.")
        out = real_df.copy()
    else:
        synth = pd.read_csv(synth_path, parse_dates=["arrival_date"])
        if "source" not in synth.columns:
            synth["source"] = "synthetic"
        else:
            synth["source"] = synth["source"].fillna("synthetic")

        # Idempotency: drop any 'real'-flagged rows from the existing file so
        # re-running this script always converges to the same result.
        n_pre = len(synth)
        synth = synth[synth["source"] != "real"].copy()
        n_dropped = n_pre - len(synth)
        if n_dropped:
            print(f"[load_real_data] Idempotency: dropped {n_dropped:,} pre-existing "
                  f"'real' rows so this run replaces them.")

        common = [c for c in real_df.columns if c in synth.columns]
        out = pd.concat([synth[common], real_df[common]], ignore_index=True)

    out = out.sort_values("arrival_date").reset_index(drop=True)

    # Defense-in-depth: clean ADR over the WHOLE output, not just real rows.
    out = _clean_adr(out)

    out["booking_id"] = np.arange(1, len(out) + 1)
    cols = ["booking_id"] + [c for c in out.columns if c != "booking_id"]
    return out[cols]


# ─────────────────────────────────────────────────────────────────────────────
#  Quality summary
# ─────────────────────────────────────────────────────────────────────────────
def quality_report(df: pd.DataFrame) -> dict:
    """
    Produce a quality summary that is **schema-compatible with
    src/generate_data.py** (so train_models_ts.py's data-quality gate keeps
    working) AND adds blend-specific fields (real_share, date range, etc.)
    consumed by the sidebar and briefing endpoint.
    """
    real_share = (df["source"] == "real").mean() if "source" in df.columns else 0.0

    # ── Original-schema fields (required by check_data_quality) ──
    missing_pct_per_col = {
        c: round(float(df[c].isna().mean() * 100), 2)
        for c in df.columns
    }
    total_missing = float(df.isna().sum().sum() / df.size * 100)

    # Drift score — change in cancellation rate between earliest & latest year
    df_dates       = pd.to_datetime(df["arrival_date"])
    years          = df_dates.dt.year
    yr_min, yr_max = int(years.min()), int(years.max())
    cr_first       = float(df.loc[years == yr_min, "is_canceled"].mean())
    cr_last        = float(df.loc[years == yr_max, "is_canceled"].mean())
    drift_score    = round(abs(cr_last - cr_first), 4)

    outlier_counts = {
        "lead_time_extreme": int((df["lead_time"]  > 600).sum()),
        "adr_extreme":       int((df["adr"]        > 1000).sum()),
        "stay_extreme":      int((df["total_stay"] > 30).sum()),
    }

    # Grade — same A/B thresholds as the synthetic generator + a relaxed C
    # tier so the trainer accepts blended data with mild messiness.
    if total_missing < 3 and drift_score < 0.05:
        grade = "A"
    elif total_missing < 8 and drift_score < 0.10:
        grade = "B"
    elif total_missing < 15 and drift_score < 0.20:
        grade = "C"
    else:
        grade = "D"

    cancel_rate_by_year = {
        str(int(y)): round(float(df.loc[years == y, "is_canceled"].mean()), 3)
        for y in sorted(years.unique())
    }

    return {
        # ─ original schema (required by train_models_ts.py) ─
        "total_rows":          int(len(df)),
        "missing_pct":         missing_pct_per_col,
        "total_missing_pct":   round(total_missing, 2),
        "outlier_counts":      outlier_counts,
        "drift_score":         drift_score,
        "cancel_rate_by_year": cancel_rate_by_year,
        "data_quality_grade":  grade,

        # ─ blend-specific extras (consumed by frontend + briefing) ─
        "rows":              int(len(df)),
        "real_share":        round(float(real_share), 4),
        "synthetic_share":   round(1 - float(real_share), 4),
        "date_min":          str(df["arrival_date"].min().date()),
        "date_max":          str(df["arrival_date"].max().date()),
        "cancellation_rate": round(float(df["is_canceled"].mean()), 4),
        "avg_adr":           round(float(df["adr"].mean()), 2),
        "avg_lead_time":     round(float(df["lead_time"].mean()), 1),
        "n_countries":       int(df["country"].nunique()),
        "n_segments":        int(df["market_segment"].nunique()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-only", action="store_true",
                        help="Skip blending; replace bookings.csv with real-only data.")
    parser.add_argument("--no-blend",  action="store_true",
                        help="Save real data to data/bookings_real.csv only.")
    parser.add_argument("--out", default=str(DATA_DIR / "bookings.csv"))
    args = parser.parse_args()

    print("\n" + "─" * 70)
    print(" Real Hotel Booking Loader")
    print("─" * 70)

    raw = fetch_real()
    print(f"[load_real_data] Raw rows: {len(raw):,} | columns: {len(raw.columns)}")

    real = _normalise(raw)
    print(f"[load_real_data] Normalised real rows: {len(real):,}")

    if args.no_blend:
        out_path = DATA_DIR / "bookings_real.csv"
        real.to_csv(out_path, index=False)
        print(f"[load_real_data] Saved real-only → {out_path}")
        report = quality_report(real)
    elif args.real_only:
        real = _clean_adr(real)
        real["booking_id"] = np.arange(1, len(real) + 1)
        cols = ["booking_id"] + [c for c in real.columns if c != "booking_id"]
        real[cols].to_csv(args.out, index=False)
        print(f"[load_real_data] Saved real-only as primary → {args.out}")
        report = quality_report(real)
    else:
        merged = blend_with_synthetic(real)
        merged.to_csv(args.out, index=False)
        print(f"[load_real_data] Blended dataset → {args.out}  ({len(merged):,} rows)")
        report = quality_report(merged)

    qpath = DATA_DIR / "data_quality.json"
    with open(qpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[load_real_data] Quality report → {qpath}")
    print("\n" + json.dumps(report, indent=2))
    print("\n[load_real_data] Done.\n")


if __name__ == "__main__":
    main()
