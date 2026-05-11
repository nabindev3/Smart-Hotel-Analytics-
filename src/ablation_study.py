"""
ablation_study.py — Ablation Study & Statistical Significance
=============================================================
For each Prophet external regressor, run two conditions:
  1. Baseline: all regressors
  2. Ablated:  remove one regressor at a time

Compute MAPE per condition, then run paired t-test to determine
whether each regressor's contribution is statistically significant (p < 0.05).

This is standard scientific methodology for ablation in ML papers.
"""

from __future__ import annotations
import os, json, warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from prophet import Prophet
import joblib

EXTERNAL_REGS = [
    "temperature_c", "precipitation_mm", "local_events",
    "holiday_flag", "competitor_adr", "cpi_yoy",
    "consumer_confidence", "search_trend",
]

PROPHET_TARGETS = {
    "occupancy": "occupancy_rate",
    "adr":       "avg_adr",
    "revenue":   "revenue",
}

TRAIN_CUTOFF = "2023-12-31"
TEST_START   = "2024-01-01"
N_BOOTSTRAP  = 50    # bootstrap samples for t-test


def _normalise(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)


def _run_prophet(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    regressors:   list[str],
) -> np.ndarray:
    """
    Fit Prophet with given regressors, return array of absolute % errors on test set.
    """
    m = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.90,
    )
    m.add_seasonality("monthly", period=30.5, fourier_order=6)

    for reg in regressors:
        m.add_regressor(reg)

    m.fit(train_df)

    future = m.make_future_dataframe(periods=len(test_df), freq="D")
    # Fill regressors in future
    all_ext = pd.concat([train_df[["ds"]+regressors],
                          test_df[["ds"]+regressors]], ignore_index=True)
    future  = future.merge(all_ext.drop_duplicates("ds"), on="ds", how="left")
    for reg in regressors:
        future[reg] = future[reg].ffill().bfill()

    forecast = m.predict(future)
    merged   = test_df[["ds","y"]].merge(forecast[["ds","yhat"]], on="ds", how="inner")
    merged   = merged[(merged["y"] > 0) & (merged["yhat"] > 0)]

    if len(merged) == 0:
        return np.array([1.0])
    ape = np.abs(merged["y"] - merged["yhat"]) / merged["y"]
    return ape.values


def run_ablation(
    target_name: str = "occupancy",
    daily_path:  str = "data/daily_kpis.csv",
    ext_path:    str = "data/external_regs.csv",
    output_path: str = "models/ablation_results.json",
) -> dict:
    """
    Full ablation study for one Prophet target.
    Returns results dict with MAPE, t-statistic, p-value per regressor.
    """
    print(f"\n{'='*55}")
    print(f"  Ablation Study: Prophet [{target_name}]")
    print(f"{'='*55}")

    daily = pd.read_csv(daily_path, parse_dates=["ds"])
    ext   = pd.read_csv(ext_path,   parse_dates=["ds"])

    col = PROPHET_TARGETS[target_name]
    df  = daily[["ds", col]].rename(columns={col:"y"}).merge(ext, on="ds", how="left")
    df  = df[df["y"] > 0].reset_index(drop=True)

    for reg in EXTERNAL_REGS:
        df[reg] = _normalise(df[reg])

    train = df[df["ds"] <= TRAIN_CUTOFF].copy()
    test  = df[df["ds"] >= TEST_START].copy()

    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    # ── Baseline: all regressors ──────────────────────────────────────────
    print(f"\n  Running BASELINE (all {len(EXTERNAL_REGS)} regressors)…", end=" ")
    baseline_ape = _run_prophet(train, test, EXTERNAL_REGS)
    baseline_mape = baseline_ape.mean()
    print(f"MAPE = {baseline_mape:.4f}")

    # ── Ablations ─────────────────────────────────────────────────────────
    results = {
        "target":   target_name,
        "baseline": {
            "regressors": EXTERNAL_REGS,
            "mape":       round(float(baseline_mape), 4),
            "n_test":     len(baseline_ape),
        },
        "ablations": [],
        "no_regressors": None,
    }

    for reg in EXTERNAL_REGS:
        remaining = [r for r in EXTERNAL_REGS if r != reg]
        print(f"  Ablating [{reg:<22}]…", end=" ")

        ablated_ape  = _run_prophet(train, test, remaining)
        ablated_mape = ablated_ape.mean()

        # Align lengths for paired t-test
        n = min(len(baseline_ape), len(ablated_ape))
        t_stat, p_val = stats.ttest_rel(baseline_ape[:n], ablated_ape[:n])

        mape_delta   = ablated_mape - baseline_mape
        significant  = p_val < 0.05

        result = {
            "removed_regressor":  reg,
            "ablated_mape":       round(float(ablated_mape), 4),
            "mape_delta":         round(float(mape_delta), 4),
            "mape_delta_pct":     round(float(mape_delta / baseline_mape * 100), 2),
            "t_statistic":        round(float(t_stat), 4),
            "p_value":            round(float(p_val), 6),
            "significant_p05":    significant,
            "interpretation":     (
                f"Removing '{reg}' {'degrades' if mape_delta>0 else 'improves'} MAPE "
                f"by {abs(mape_delta_pct := mape_delta/baseline_mape*100):.1f}% "
                f"({'**significant**' if significant else 'NOT significant'} at p<0.05, "
                f"p={p_val:.4f})."
            ),
        }
        # fix inner reference
        result["interpretation"] = (
            f"Removing '{reg}' {'degrades' if mape_delta>0 else 'improves'} MAPE "
            f"by {abs(mape_delta/baseline_mape*100):.1f}% "
            f"({'significant' if significant else 'NOT significant'} at p<0.05, "
            f"p={p_val:.4f})."
        )
        results["ablations"].append(result)

        sig_str = "✅ p<0.05" if significant else "❌ ns"
        print(f"MAPE={ablated_mape:.4f} Δ={mape_delta:+.4f} {sig_str} (p={p_val:.4f})")

    # ── No-regressor baseline ─────────────────────────────────────────────
    print("  Running NO-REGRESSOR baseline…", end=" ")
    no_reg_ape  = _run_prophet(train, test, [])
    no_reg_mape = no_reg_ape.mean()
    n           = min(len(baseline_ape), len(no_reg_ape))
    t_full, p_full = stats.ttest_rel(baseline_ape[:n], no_reg_ape[:n])
    results["no_regressors"] = {
        "mape":          round(float(no_reg_mape), 4),
        "mape_delta":    round(float(no_reg_mape - baseline_mape), 4),
        "p_value":       round(float(p_full), 6),
        "significant":   p_full < 0.05,
    }
    print(f"MAPE={no_reg_mape:.4f} (regressors overall: {'significant' if p_full<0.05 else 'ns'})")

    # Sort ablations by |MAPE delta| descending (most impactful first)
    results["ablations"].sort(key=lambda x: abs(x["mape_delta"]), reverse=True)

    # Save
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
    existing[target_name] = results
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n  ✓ Results saved → {output_path}")
    return results


def print_summary(results: dict):
    """Pretty-print ablation study summary table."""
    print(f"\n{'─'*65}")
    print(f"  ABLATION SUMMARY — Prophet [{results['target']}]")
    print(f"  Baseline MAPE: {results['baseline']['mape']:.4f}")
    print(f"{'─'*65}")
    print(f"  {'Removed Regressor':<25} {'MAPE':>8} {'Δ MAPE':>8} {'p-value':>9} {'Sig?':>6}")
    print(f"{'─'*65}")
    for r in results["ablations"]:
        sig = "✓" if r["significant_p05"] else "✗"
        print(f"  {r['removed_regressor']:<25} "
              f"{r['ablated_mape']:>8.4f} "
              f"{r['mape_delta']:>+8.4f} "
              f"{r['p_value']:>9.4f} "
              f"{sig:>6}")
    nr = results.get("no_regressors",{})
    if nr:
        sig = "✓" if nr.get("significant") else "✗"
        print(f"{'─'*65}")
        print(f"  {'No regressors (vanilla)':<25} "
              f"{nr.get('mape',0):>8.4f} "
              f"{nr.get('mape_delta',0):>+8.4f} "
              f"{nr.get('p_value',0):>9.4f} "
              f"{sig:>6}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = run_ablation("occupancy")
    print_summary(results)

    results_adr = run_ablation("adr")
    print_summary(results_adr)
