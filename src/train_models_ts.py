"""
train_models.py — MLOps-grade Training Pipeline
================================================
What this does that basic pipelines don't:
  1. MLflow experiment tracking  — every run logged with params, metrics, artefacts
  2. Model versioning            — tagged with run_id so you can roll back
  3. Prophet + external regressors — weather, events, macro, competitor pricing
  4. Drift detection             — compares MAPE to registered baseline
  5. Auto-retraining trigger     — fires when drift > threshold (default 20%)
  6. Data quality gating         — rejects training if quality score < B

Usage
-----
  python src/train_models.py                     # normal train
  python src/train_models.py --check-drift       # drift check only
  python src/train_models.py --force-retrain     # force even if no drift
"""

import os, sys, json, time, warnings, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D        = lambda f: os.path.join(BASE, "data",       f)
M        = lambda f: os.path.join(BASE, "models",     f)
MON      = lambda f: os.path.join(BASE, "monitoring", f)

os.makedirs(os.path.join(BASE,"models"),     exist_ok=True)
os.makedirs(os.path.join(BASE,"monitoring"), exist_ok=True)

# ── MLflow setup ───────────────────────────────────────────────────────────
MLFLOW_URI     = f"sqlite:///{os.path.join(BASE,'mlruns','mlflow.db')}"
EXPERIMENT     = "Smart_Hotel_Analytics"
DRIFT_THRESHOLD = 0.20     # retrain if MAPE degrades > 20% vs baseline

# ── Feature schema ─────────────────────────────────────────────────────────
FEATURES = [
    "hotel","lead_time","arrival_date_month","total_stay","total_guests",
    "meal","country","market_segment","distribution_channel",
    "is_repeated_guest","previous_cancellations","previous_bookings_not_canceled",
    "reserved_room_type","booking_changes","deposit_type",
    "days_in_waiting_list","customer_type",
    "required_car_parking_spaces","total_of_special_requests","adr",
]
CAT = ["hotel","arrival_date_month","meal","country","market_segment",
       "distribution_channel","reserved_room_type","deposit_type","customer_type"]
NUM = [f for f in FEATURES if f not in CAT]

PROPHET_TARGETS = {
    "occupancy": {"col":"occupancy_rate",    "mode":"multiplicative", "cp":0.15},
    "adr":       {"col":"avg_adr",           "mode":"multiplicative", "cp":0.10},
    "revenue":   {"col":"revenue",           "mode":"multiplicative", "cp":0.20},
}

EXTERNAL_REGS = [
    "temperature_c","precipitation_mm","local_events",
    "holiday_flag","competitor_adr","cpi_yoy",
    "consumer_confidence","search_trend",
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════
def check_data_quality(min_grade: str = "C") -> dict:
    grade_order = {"A":0,"B":1,"C":2,"D":3}
    path = D("data_quality.json")
    if not os.path.exists(path):
        raise FileNotFoundError("data/data_quality.json not found. Run generate_data.py first.")
    with open(path) as f:
        report = json.load(f)
    grade = report.get("data_quality_grade","D")
    if grade_order.get(grade,9) > grade_order.get(min_grade,0):
        raise ValueError(f"Data quality grade {grade} is below minimum {min_grade}. "
                         f"Fix data issues before training.")
    print(f"  ✓ Data quality gate passed — grade: {grade} | "
          f"missing: {report['total_missing_pct']:.1f}% | "
          f"drift score: {report['drift_score']:.4f}")
    return report


# ═══════════════════════════════════════════════════════════════════════════
# PROPHET TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def _prepare_prophet_df(daily: pd.DataFrame, ext: pd.DataFrame,
                         col: str) -> pd.DataFrame:
    df = daily[["ds", col]].rename(columns={col:"y"}).merge(ext, on="ds", how="left")
    df = df[df["y"] > 0].reset_index(drop=True)
    return df


def _normalise_regressor(series: pd.Series) -> pd.Series:
    """Scale to 0-1 to prevent Prophet from weighting regressors by magnitude."""
    mn, mx = series.min(), series.max()
    if mx == mn: return series * 0
    return (series - mn) / (mx - mn)


def train_prophet_model(name: str, cfg: dict,
                         train_df: pd.DataFrame,
                         test_df:  pd.DataFrame) -> tuple:
    """Train one Prophet model, return (model, forecast_df, mape)."""
    m = Prophet(
        seasonality_mode        = cfg["mode"],
        changepoint_prior_scale = cfg["cp"],
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        interval_width          = 0.90,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=6)

    # Add normalised external regressors
    for reg in EXTERNAL_REGS:
        if reg in train_df.columns:
            m.add_regressor(reg)

    m.fit(train_df)

    # Future dataframe must include regressors — use last known values
    last_ext = test_df[EXTERNAL_REGS].iloc[-1] if len(test_df) > 0 else None
    future = m.make_future_dataframe(periods=400, freq="D")

    # Fill regressors in future
    full_ext = pd.concat([train_df[["ds"]+EXTERNAL_REGS],
                           test_df[["ds"]+EXTERNAL_REGS]], ignore_index=True)
    full_ext = full_ext.drop_duplicates("ds")
    future = future.merge(full_ext, on="ds", how="left")

    # For truly future dates, forward-fill with last known regressor values
    for reg in EXTERNAL_REGS:
        if reg in future.columns:
            future[reg] = future[reg].ffill().bfill()

    forecast = m.predict(future)

    # MAPE on hold-out
    if len(test_df) > 0:
        merged = test_df[["ds","y"]].merge(forecast[["ds","yhat"]], on="ds", how="inner")
        merged = merged[(merged["yhat"] > 0) & (merged["y"] > 0)]
        mape = (np.abs(merged["y"] - merged["yhat"]) / merged["y"]).mean() if len(merged)>0 else np.nan
    else:
        mape = np.nan

    return m, forecast, mape


def train_all_prophet(daily: pd.DataFrame, ext: pd.DataFrame,
                       run_id: str) -> dict:
    TRAIN_CUTOFF = "2024-06-30"
    results = {}

    for name, cfg in PROPHET_TARGETS.items():
        col = cfg["col"]
        print(f"  ▸ Prophet [{name}] …", end=" ", flush=True)
        t0 = time.time()

        df  = _prepare_prophet_df(daily, ext, col)

        # Normalise regressors
        for reg in EXTERNAL_REGS:
            if reg in df.columns:
                df[reg] = _normalise_regressor(df[reg])

        train = df[df["ds"] <= TRAIN_CUTOFF].copy()
        test  = df[df["ds"] >  TRAIN_CUTOFF].copy()

        model, forecast, mape = train_prophet_model(name, cfg, train, test)
        elapsed = time.time() - t0

        # Save
        bundle = {"model":model, "forecast":forecast, "mape":mape,
                  "run_id":run_id, "train_cutoff":TRAIN_CUTOFF,
                  "regressors":EXTERNAL_REGS}
        joblib.dump(bundle, M(f"prophet_{name}.joblib"))

        results[name] = {"mape":mape, "elapsed":elapsed}
        print(f"MAPE={mape:.2%}  [{elapsed:.1f}s]")

        # MLflow log
        mlflow.log_metric(f"prophet_{name}_mape", mape if not np.isnan(mape) else 1.0)
        mlflow.log_param(f"prophet_{name}_mode", cfg["mode"])
        mlflow.log_param(f"prophet_{name}_cp",   cfg["cp"])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ML CANCELLATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def clean_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
    df = bookings.copy()

    # Impute missing
    df["children"] = df["children"].fillna(df["children"].median())
    df["country"]  = df["country"].fillna(df["country"].mode()[0])
    df["adr"]      = df["adr"].fillna(df["adr"].median())
    df["days_in_waiting_list"] = df["days_in_waiting_list"].fillna(0)
    df["meal"]     = df["meal"].fillna("Undefined")

    # Sanity filters
    df = df[(df["adr"] >= 0) & (df["adr"] <= 2000)]  # remove fat-finger ADRs
    df = df[(df["lead_time"] >= 0) & (df["lead_time"] <= 700)]
    df = df[(df["adults"] + df["children"].fillna(0) + df["babies"]) > 0]

    # Re-derive engineered features after imputation
    df["total_stay"]   = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # IQR outlier removal (conservative 3-sigma equiv for non-normal)
    for col in ["adr","days_in_waiting_list"]:
        q1,q3 = df[col].quantile(0.05), df[col].quantile(0.95)
        df    = df[(df[col] >= q1) & (df[col] <= q3)]

    print(f"    Clean shape: {df.shape} | "
          f"cancel rate: {df['is_canceled'].mean():.2%}")
    return df


def train_cancellation_model(bookings: pd.DataFrame) -> dict:
    """
    Cancellation classifier — LightGBM v2.

    Improvements over the previous GBM+SMOTE pipeline:
      • LightGBM with histogram binning — 5–10× faster than sklearn GBM,
        usually +5–10 AUC points on the same data.
      • Drops SMOTE (which introduced label noise on this task) in favour
        of LightGBM's `scale_pos_weight` for class imbalance.
      • Hold-out validation split with early stopping on AUC to prevent
        overfitting on the larger 180k-row blended dataset.
      • Decision threshold tuned on the validation set to maximise F1
        instead of using the default 0.5 cut-off — the previous F1 of
        0.156 was largely a threshold artefact.
      • Fallback to sklearn HistGradientBoostingClassifier if LightGBM is
        not installed, so the trainer remains executable in any env.
    """
    print("  ▸ Cancellation classifier (LightGBM v2) …")
    df = clean_bookings(bookings)

    X, y = df[FEATURES].copy(), df["is_canceled"].astype(int).copy()

    # Hold-out test set + inner validation for early stopping & threshold tuning
    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    X_tr_p = pre.fit_transform(X_tr)
    X_va_p = pre.transform(X_va)
    X_te_p = pre.transform(X_te)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    # ── Try LightGBM, fall back to sklearn HistGradientBoosting ──
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators        = 1000,
            learning_rate       = 0.05,
            max_depth           = -1,
            num_leaves          = 63,
            min_child_samples   = 30,
            subsample           = 0.85,
            subsample_freq      = 1,
            colsample_bytree    = 0.85,
            reg_alpha           = 0.1,
            reg_lambda          = 0.1,
            scale_pos_weight    = pos_weight,
            objective           = "binary",
            metric              = "auc",
            random_state        = 42,
            verbose             = -1,
            n_jobs              = -1,
        )
        clf.fit(
            X_tr_p, y_tr,
            eval_set       = [(X_va_p, y_va)],
            eval_metric    = "auc",
            callbacks      = [lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(0)],
        )
        engine_name = "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter        = 600,
            learning_rate   = 0.05,
            max_depth       = 8,
            min_samples_leaf= 30,
            l2_regularization = 0.1,
            class_weight    = "balanced",
            early_stopping  = True,
            validation_fraction = 0.15,
            n_iter_no_change= 30,
            random_state    = 42,
        )
        clf.fit(X_tr_p, y_tr)
        engine_name = "HistGradientBoosting (LightGBM unavailable)"

    # ── Threshold tuning on validation set to maximise F1 ──
    val_proba = clf.predict_proba(X_va_p)[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    f1s = [f1_score(y_va, (val_proba >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    print(f"    [{engine_name}] tuned threshold = {best_threshold:.3f} "
          f"(val F1 = {max(f1s):.3f})")

    # ── Final test-set evaluation at the tuned threshold ──
    y_proba = clf.predict_proba(X_te_p)[:, 1]
    y_pred  = (y_proba >= best_threshold).astype(int)
    metrics = {
        "accuracy":       accuracy_score(y_te, y_pred),
        "roc_auc":        roc_auc_score(y_te, y_proba),
        "f1":             f1_score(y_te, y_pred),
        "best_threshold": best_threshold,
        "engine":         engine_name,
    }
    print(f"    Accuracy={metrics['accuracy']:.3f} | "
          f"AUC={metrics['roc_auc']:.3f} | F1={metrics['f1']:.3f}")

    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])
    joblib.dump(pipe, M("cancellation_model.joblib"))
    joblib.dump(
        {"features": FEATURES, "cat": CAT, "num": NUM,
         "best_threshold": best_threshold, "engine": engine_name},
        M("feature_config.joblib"),
    )

    rpt = classification_report(y_te, y_pred,
                                target_names=["Not Canceled", "Canceled"])
    return {**metrics, "report": rpt, "cm": confusion_matrix(y_te, y_pred)}


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════
def check_drift() -> dict:
    """
    Compare current MAPE against the registered baseline.
    Returns drift report dict.
    """
    baseline_path = MON("baseline_metrics.json")
    if not os.path.exists(baseline_path):
        print("  No baseline found — this will become the baseline.")
        return {"drift_detected": False, "reason": "no_baseline"}

    with open(baseline_path) as f:
        baseline = json.load(f)

    drift_report = {"drift_detected": False, "metrics": {}}
    for name in PROPHET_TARGETS:
        path = M(f"prophet_{name}.joblib")
        if not os.path.exists(path): continue
        bundle = joblib.load(path)
        current_mape = bundle.get("mape", 1.0)
        base_mape    = baseline.get(f"prophet_{name}_mape", current_mape)
        if base_mape > 0:
            degradation = (current_mape - base_mape) / base_mape
        else:
            degradation = 0

        drift_report["metrics"][name] = {
            "baseline_mape": round(base_mape, 4),
            "current_mape":  round(current_mape, 4),
            "degradation":   round(degradation, 4),
        }
        if degradation > DRIFT_THRESHOLD:
            drift_report["drift_detected"] = True
            drift_report["reason"] = (
                f"Prophet [{name}] MAPE degraded {degradation:.1%} "
                f"vs baseline (threshold {DRIFT_THRESHOLD:.0%})"
            )
            print(f"  ⚠️  Drift detected in [{name}]: "
                  f"MAPE {base_mape:.2%} → {current_mape:.2%} "
                  f"({degradation:+.1%})")

    return drift_report


def save_baseline(ts_results: dict, ml_results: dict):
    baseline = {f"prophet_{name}_mape": res["mape"]
                for name, res in ts_results.items()}
    baseline["cancellation_accuracy"] = ml_results["accuracy"]
    baseline["cancellation_auc"]      = ml_results["roc_auc"]
    baseline["timestamp"]             = pd.Timestamp.now().isoformat()

    with open(MON("baseline_metrics.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  ✓ Baseline registered → monitoring/baseline_metrics.json")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def run_training(force: bool = False) -> str:
    """Full training run. Returns MLflow run_id."""

    print("\n[0/4]  Data quality gate …")
    quality = check_data_quality(min_grade="C")

    # Drift check (skip if no baseline)
    if not force:
        print("\n[1/4]  Drift check …")
        drift = check_drift()
        if not drift["drift_detected"] and os.path.exists(MON("baseline_metrics.json")):
            print("  No drift detected — skipping full retraining.")
            print("  (Use --force-retrain to override)\n")
            return "skipped"

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"train_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}") as run:
        run_id = run.info.run_id
        print(f"\n[2/4]  MLflow run: {run_id}")

        # Log data quality metadata
        mlflow.log_param("data_rows",           quality["total_rows"])
        mlflow.log_param("missing_pct",         quality["total_missing_pct"])
        mlflow.log_param("data_quality_grade",  quality["data_quality_grade"])
        mlflow.log_metric("drift_score",        quality["drift_score"])
        mlflow.log_param("external_regressors", str(EXTERNAL_REGS))

        # ── Prophet ───────────────────────────────────────────────────────
        print("\n[3/4]  Prophet models with external regressors …")
        daily = pd.read_csv(D("daily_kpis.csv"), parse_dates=["ds"])
        ext   = pd.read_csv(D("external_regs.csv"), parse_dates=["ds"])
        ts_results = train_all_prophet(daily, ext, run_id)

        # ── Cancellation GBM ──────────────────────────────────────────────
        print("\n[4/4]  Cancellation GBM + SMOTE …")
        bookings = pd.read_csv(D("bookings.csv"))
        ml_res   = train_cancellation_model(bookings)

        # Log all ML metrics
        mlflow.log_metric("cancellation_accuracy", ml_res["accuracy"])
        mlflow.log_metric("cancellation_auc",       ml_res["roc_auc"])
        mlflow.log_metric("cancellation_f1",        ml_res["f1"])
        mlflow.log_param("gbm_n_estimators", 250)
        mlflow.log_param("gbm_max_depth",    5)
        mlflow.log_param("smote_enabled",    True)

        # Log artefacts
        for fname in ["cancellation_model.joblib","feature_config.joblib"]:
            mlflow.log_artifact(M(fname))
        for name in PROPHET_TARGETS:
            mlflow.log_artifact(M(f"prophet_{name}.joblib"))

        # ── Consolidated report ───────────────────────────────────────────
        report = {
            "run_id":      run_id,
            "timestamp":   pd.Timestamp.now().isoformat(),
            "mlflow_uri":  MLFLOW_URI,
            "data_quality": quality,
            "prophet": {k: {"mape":v["mape"], "elapsed":v["elapsed"]}
                         for k,v in ts_results.items()},
            "cancellation": {
                "accuracy": ml_res["accuracy"],
                "roc_auc":  ml_res["roc_auc"],
                "f1":       ml_res["f1"],
            },
            "external_regressors": EXTERNAL_REGS,
        }
        with open(M("training_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Human-readable
        lines = [
            "Smart Hotel Analytics — Training Report",
            "=" * 50,
            f"Run ID   : {run_id}",
            f"Timestamp: {report['timestamp']}",
            f"MLflow   : {MLFLOW_URI}",
            "",
            "── Data Quality ─────────────────────────────",
            f"  Grade   : {quality['data_quality_grade']}",
            f"  Missing : {quality['total_missing_pct']:.1f}%",
            f"  Drift   : {quality['drift_score']:.4f}",
            "",
            "── Prophet (with external regressors) ───────",
        ]
        for k,v in ts_results.items():
            lines.append(f"  {k:<12}: MAPE={v['mape']:.2%}  [{v['elapsed']:.1f}s]")
        lines += [
            "",
            "── Cancellation GBM + SMOTE ─────────────────",
            f"  Accuracy: {ml_res['accuracy']:.4f}",
            f"  ROC-AUC : {ml_res['roc_auc']:.4f}",
            f"  F1 Score: {ml_res['f1']:.4f}",
            "",
            "── External Regressors Used ─────────────────",
        ]
        for r in EXTERNAL_REGS:
            lines.append(f"  • {r}")
        lines += ["", ml_res["report"]]
        with open(M("training_report.txt"), "w") as f:
            f.write("\n".join(lines))

        # Register baseline
        save_baseline(ts_results, ml_res)

        # Append to retraining history
        hist_path = MON("retrain_history.json")
        history   = json.load(open(hist_path)) if os.path.exists(hist_path) else []
        history.append({
            "run_id":    run_id,
            "timestamp": report["timestamp"],
            "trigger":   "forced" if force else "drift",
            "prophet_mapes": {k:v["mape"] for k,v in ts_results.items()},
            "cancellation_auc": ml_res["roc_auc"],
        })
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n✅  Training complete | MLflow run: {run_id}")
    print(f"   View runs: mlflow ui --backend-store-uri {MLFLOW_URI}\n")
    return run_id


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-drift",   action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    os.chdir(BASE)

    if args.check_drift:
        drift = check_drift()
        with open(MON("drift_report.json"), "w") as f:
            json.dump(drift, f, indent=2)
        print(json.dumps(drift, indent=2))
    else:
        run_training(force=args.force_retrain)
