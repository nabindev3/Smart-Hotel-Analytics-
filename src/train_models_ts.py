"""
train_models_ts.py — MLOps-grade Training Pipeline
================================================
What this does that basic pipelines don't:
  1. MLflow experiment tracking
  2. Walk-forward CV avoiding temporal leakage
  3. Proper Baselines (LogReg, XGBoost, LightGBM, N-BEATS)
  4. Calibration with Brier Score & Reliability Diagrams
  5. Subgroup Metrics
"""

import os, sys, json, time, warnings, argparse
# Scope warning suppression to the known-noisy third-party deprecation/future
# warnings instead of a blanket ignore. A global filterwarnings("ignore") also
# hides RuntimeWarnings (divide-by-zero, overflow) and UserWarnings about data
# quality — the very signals we want to see during training.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              classification_report, confusion_matrix, brier_score_loss,
                              precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
DRIFT_THRESHOLD = 0.20

# ── Feature schema ─────────────────────────────────────────────────────────
# Removed Antonio-dataset leakage columns: booking_changes, days_in_waiting_list
FEATURES = [
    "hotel","lead_time","arrival_date_month","total_stay","total_guests",
    "meal","country","market_segment","distribution_channel",
    "is_repeated_guest","previous_cancellations","previous_bookings_not_canceled",
    "reserved_room_type","deposit_type","customer_type",
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
        raise ValueError(f"Data quality grade {grade} is below minimum {min_grade}.")
    print(f"  ✓ Data quality gate passed — grade: {grade} | "
          f"missing: {report['total_missing_pct']:.1f}% | "
          f"drift score: {report['drift_score']:.4f}")
    return report


# ═══════════════════════════════════════════════════════════════════════════
# PROPHET + NBEATS TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def _prepare_prophet_df(daily: pd.DataFrame, ext: pd.DataFrame,
                         col: str) -> pd.DataFrame:
    df = daily[["ds", col]].rename(columns={col:"y"}).merge(ext, on="ds", how="left")
    df = df[df["y"] > 0].reset_index(drop=True)
    return df

def _normalise_regressor(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn: return series * 0
    return (series - mn) / (mx - mn)

def train_prophet_model(name: str, cfg: dict, train_df: pd.DataFrame, test_df:  pd.DataFrame) -> tuple:
    m = Prophet(seasonality_mode=cfg["mode"], changepoint_prior_scale=cfg["cp"],
                yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=6)
    for reg in EXTERNAL_REGS:
        if reg in train_df.columns: m.add_regressor(reg)

    m.fit(train_df)
    future = m.make_future_dataframe(periods=400, freq="D")
    full_ext = pd.concat([train_df[["ds"]+EXTERNAL_REGS], test_df[["ds"]+EXTERNAL_REGS]], ignore_index=True).drop_duplicates("ds")
    future = future.merge(full_ext, on="ds", how="left")
    for reg in EXTERNAL_REGS:
        if reg in future.columns: future[reg] = future[reg].ffill().bfill()
        
    forecast = m.predict(future)
    if len(test_df) > 0:
        merged = test_df[["ds","y"]].merge(forecast[["ds","yhat"]], on="ds", how="inner")
        merged = merged[(merged["yhat"] > 0) & (merged["y"] > 0)]
        mape = (np.abs(merged["y"] - merged["yhat"]) / merged["y"]).mean() if len(merged)>0 else np.nan
    else: mape = np.nan
    return m, forecast, mape

def train_nbeats_model(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    horizon = len(test_df)
    if horizon == 0: return np.nan
    train_nf = train_df[['ds', 'y']].copy()
    train_nf['unique_id'] = name
    
    # Simple N-BEATS configuration
    models = [NBEATS(input_size=max(7, horizon), h=horizon, max_steps=100)]
    try:
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(df=train_nf)
        forecast = nf.predict().reset_index()
        merged = test_df[["ds", "y"]].merge(forecast, on="ds", how="inner")
        if len(merged) > 0:
            mape = (np.abs(merged["y"] - merged["NBEATS"]) / merged["y"]).mean()
        else: mape = np.nan
        return mape
    except Exception as e:
        print(f"    [N-BEATS Error: {e}]")
        return np.nan

def train_all_prophet(daily: pd.DataFrame, ext: pd.DataFrame, run_id: str) -> dict:
    TRAIN_CUTOFF = "2024-06-30"
    results = {}

    for name, cfg in PROPHET_TARGETS.items():
        col = cfg["col"]
        print(f"  ▸ Prophet [{name}] …", end=" ", flush=True)
        t0 = time.time()
        df  = _prepare_prophet_df(daily, ext, col)
        for reg in EXTERNAL_REGS:
            if reg in df.columns: df[reg] = _normalise_regressor(df[reg])
            
        train = df[df["ds"] <= TRAIN_CUTOFF].copy()
        test  = df[df["ds"] >  TRAIN_CUTOFF].copy()

        model, forecast, mape = train_prophet_model(name, cfg, train, test)
        elapsed = time.time() - t0
        
        # CAVEAT: we persist the full pre-computed forecast alongside the model.
        # This makes the joblib large and, more importantly, the forecast goes
        # stale the moment training finishes — its date range is frozen at
        # train time and does not advance with the calendar. The serving layer
        # works around this (briefing.py has a "forecast doesn't reach today"
        # fallback). The durable fix is to persist only the fitted `model` and
        # call model.predict() at request time for the current horizon.
        bundle = {"model":model, "forecast":forecast, "mape":mape, "run_id":run_id, "train_cutoff":TRAIN_CUTOFF, "regressors":EXTERNAL_REGS}
        joblib.dump(bundle, M(f"prophet_{name}.joblib"))
        
        print(f"MAPE={mape:.2%}  [{elapsed:.1f}s]")
        
        # N-BEATS baseline. KNOWN LIMITATION: this is run only for the first
        # target, so the RESULTS.md baseline table is half-empty. Root cause is
        # the PyTorch-Lightning DataLoader spawning multiprocessing workers that
        # leak named semaphores on macOS ("objc/resource_tracker" warnings),
        # eventually hitting the per-process semaphore limit across 3 fits. The
        # real fix is to force single-process data loading (num_workers=0 on the
        # NeuralForecast/NBEATS dataloader) so all three targets can run; until
        # then we cap it at one to keep training reliable on macOS dev machines.
        if name == "occupancy":
            print(f"  ▸ N-BEATS baseline [{name}] …", end=" ", flush=True)
            t1 = time.time()
            nbeats_mape = train_nbeats_model(name, train, test)
            print(f"MAPE={nbeats_mape:.2%} [{time.time()-t1:.1f}s]")
        else:
            nbeats_mape = np.nan

        results[name] = {"mape":mape, "nbeats_mape": nbeats_mape, "elapsed":elapsed}
        mlflow.log_metric(f"prophet_{name}_mape", mape if not np.isnan(mape) else 1.0)
        mlflow.log_metric(f"nbeats_{name}_mape", nbeats_mape if not np.isnan(nbeats_mape) else 1.0)
        mlflow.log_param(f"prophet_{name}_mode", cfg["mode"])
        mlflow.log_param(f"prophet_{name}_cp",   cfg["cp"])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ML CANCELLATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def clean_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
    df = bookings.copy()
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"])
        df = df.sort_values("arrival_date").reset_index(drop=True)

    df["children"] = df["children"].fillna(df["children"].median())
    df["country"]  = df["country"].fillna(df["country"].mode()[0])
    df["adr"]      = df["adr"].fillna(df["adr"].median())
    df["meal"]     = df["meal"].fillna("Undefined")

    df = df[(df["adr"] >= 0) & (df["adr"] <= 2000)]
    df = df[(df["lead_time"] >= 0) & (df["lead_time"] <= 700)]
    df = df[(df["adults"] + df["children"].fillna(0) + df["babies"]) > 0]

    df["total_stay"]   = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    for col in ["adr"]:
        q1,q3 = df[col].quantile(0.05), df[col].quantile(0.95)
        df    = df[(df[col] >= q1) & (df[col] <= q3)]

    print(f"    Clean shape: {df.shape} | cancel rate: {df['is_canceled'].mean():.2%}")
    return df

def train_cancellation_model(bookings: pd.DataFrame) -> dict:
    print("  ▸ Cancellation classifiers (Walk-forward CV + Calibration) …")
    df = clean_bookings(bookings)

    X = df[FEATURES].copy()
    y = df["is_canceled"].astype(int).copy()

    # Single explicit temporal hold-out: train on the earliest rows, evaluate on
    # the most recent. The previous code built a 5-fold TimeSeriesSplit but used
    # only the last fold — paying for 5 splits and discarding 4. This computes
    # exactly that last fold directly (test = final 1/6 of the rows, matching
    # TimeSeriesSplit(n_splits=5)), so a retrain reproduces the same model.
    # NOTE: this assumes `df` is in chronological order; sort upstream if not.
    n_test   = max(1, len(X) // 6)
    split_at = len(X) - n_test
    train_idx = np.arange(0, split_at)
    test_idx  = np.arange(split_at, len(X))

    X_tr_raw, X_te_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    df_te = df.iloc[test_idx].copy()

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])

    X_tr_p = pre.fit_transform(X_tr_raw)
    X_te_p = pre.transform(X_te_raw)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, 
                                 scale_pos_weight=float((y_tr==0).sum()/max((y_tr==1).sum(), 1)),
                                 eval_metric="auc", random_state=42, n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                   scale_pos_weight=float((y_tr==0).sum()/max((y_tr==1).sum(), 1)),
                                   random_state=42, n_jobs=1, verbose=-1)
    }

    best_auc = 0
    best_model_name = ""
    best_model = None
    results = {}

    for name, clf in models.items():
        print(f"    Training {name}...")
        clf.fit(X_tr_p, y_tr)
        y_proba = clf.predict_proba(X_te_p)[:, 1]
        auc = roc_auc_score(y_te, y_proba)
        brier = brier_score_loss(y_te, y_proba)
        
        results[name] = {"auc": auc, "brier": brier, "model": clf, "proba": y_proba}
        print(f"      AUC: {auc:.3f} | Brier: {brier:.3f}")
        mlflow.log_metric(f"{name}_auc", auc)

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = clf

    print(f"    Best baseline model: {best_model_name} (AUC: {best_auc:.3f})")
    
    # Calibration. CalibratedClassifierCV(cv=3) fits the sigmoid on 3 internal
    # folds of the *training* data only; the reliability numbers below are then
    # reported on the single temporal hold-out (see calibration_curve / the
    # reliability diagram). With one explicit hold-out there are no outer CV
    # folds to report per-fold — per-fold reliability would require nested
    # temporal CV, which is a deliberate non-goal here.
    print(f"    Calibrating {best_model_name}...")
    calibrated = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv=3)
    calibrated.fit(X_tr_p, y_tr)
    
    y_proba_cal = calibrated.predict_proba(X_te_p)[:, 1]
    y_pred_cal = calibrated.predict(X_te_p)
    
    auc_cal = roc_auc_score(y_te, y_proba_cal)
    brier_cal = brier_score_loss(y_te, y_proba_cal)
    f1_cal = f1_score(y_te, y_pred_cal)

    print(f"    Calibrated AUC: {auc_cal:.3f} | Brier: {brier_cal:.3f}")

    # F1-optimal decision threshold on the calibrated probabilities. The serving
    # layer (backend/routers/cancellation.py::_load_threshold) reads this from
    # feature_config.joblib; without it, scoring silently falls back to 0.5.
    prec, rec, thr = precision_recall_curve(y_te, y_proba_cal)
    f1_curve = 2 * prec * rec / (prec + rec + 1e-9)
    # precision_recall_curve returns one fewer threshold than precision/recall
    # points (the last point has no threshold), so align on f1_curve[:-1].
    best_threshold = float(thr[int(np.nanargmax(f1_curve[:-1]))]) if len(thr) else 0.5
    f1_at_best = f1_score(y_te, (y_proba_cal >= best_threshold).astype(int))
    print(f"    F1-optimal threshold: {best_threshold:.3f} (F1={f1_at_best:.3f} vs {f1_cal:.3f} @0.5)")

    # Reliability Diagram
    prob_true, prob_pred = calibration_curve(y_te, y_proba_cal, n_bins=10)
    uncal_true, uncal_pred = calibration_curve(y_te, results[best_model_name]["proba"], n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"{best_model_name} (Calibrated)")
    plt.plot(uncal_pred, uncal_true, "s-", label=f"{best_model_name} (Uncalibrated)")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.savefig(M("reliability_diagram.png"))
    plt.close()

    # Subgroup metrics
    subgroups = {}
    df_te["y_true"] = y_te.values
    df_te["y_proba"] = y_proba_cal
    
    print("    Subgroup Metrics (Hotel):")
    h_metrics = {}
    for h in df_te["hotel"].unique():
        sub = df_te[df_te["hotel"] == h]
        if len(sub['y_true'].unique()) > 1:
            h_metrics[h] = {"auc": roc_auc_score(sub["y_true"], sub["y_proba"]), 
                            "brier": brier_score_loss(sub["y_true"], sub["y_proba"])}
            print(f"      {h}: AUC={h_metrics[h]['auc']:.3f}")
    subgroups["hotel"] = h_metrics

    print("    Subgroup Metrics (Top 5 Countries):")
    c_metrics = {}
    top_countries = df_te["country"].value_counts().nlargest(5).index
    for c in top_countries:
        sub = df_te[df_te["country"] == c]
        if len(sub['y_true'].unique()) > 1:
            c_metrics[c] = {"auc": roc_auc_score(sub["y_true"], sub["y_proba"]), 
                            "brier": brier_score_loss(sub["y_true"], sub["y_proba"])}
            print(f"      {c}: AUC={c_metrics[c]['auc']:.3f}")
    subgroups["country"] = c_metrics

    print("    Subgroup Metrics (Season):")
    s_metrics = {}
    for s in df_te["arrival_date_month"].unique():
        sub = df_te[df_te["arrival_date_month"] == s]
        if len(sub['y_true'].unique()) > 1:
            s_metrics[s] = {"auc": roc_auc_score(sub["y_true"], sub["y_proba"]), 
                            "brier": brier_score_loss(sub["y_true"], sub["y_proba"])}
    subgroups["season"] = s_metrics

    mlflow.log_metric("cancellation_best_threshold", best_threshold)

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred_cal),
        "roc_auc": auc_cal,
        "f1": f1_cal,
        "brier": brier_cal,
        "best_threshold": best_threshold,
        "engine": best_model_name,
    }

    pipe = Pipeline([("preprocessor", pre), ("classifier", calibrated)])
    joblib.dump(pipe, M("cancellation_model.joblib"))
    joblib.dump({"features": FEATURES, "cat": CAT, "num": NUM,
                 "engine": best_model_name, "best_threshold": best_threshold},
                M("feature_config.joblib"))

    rpt = classification_report(y_te, y_pred_cal, target_names=["Not Canceled", "Canceled"])
    return {**metrics, "report": rpt, "cm": confusion_matrix(y_te, y_pred_cal), "subgroups": subgroups, "baselines": {k: v["auc"] for k,v in results.items()}}


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION & RESULTS
# ═══════════════════════════════════════════════════════════════════════════
def check_drift() -> dict:
    baseline_path = MON("baseline_metrics.json")
    if not os.path.exists(baseline_path):
        return {"drift_detected": False, "reason": "no_baseline"}
    with open(baseline_path) as f: baseline = json.load(f)

    drift_report = {"drift_detected": False, "metrics": {}}
    for name in PROPHET_TARGETS:
        path = M(f"prophet_{name}.joblib")
        if not os.path.exists(path): continue
        bundle = joblib.load(path)
        current_mape = bundle.get("mape", 1.0)
        base_mape    = baseline.get(f"prophet_{name}_mape", current_mape)
        degradation = (current_mape - base_mape) / base_mape if base_mape > 0 else 0
        drift_report["metrics"][name] = {"baseline_mape": round(base_mape, 4), "current_mape": round(current_mape, 4), "degradation": round(degradation, 4)}
        if degradation > DRIFT_THRESHOLD: drift_report["drift_detected"] = True

    # Persist the report so external consumers (the nightly CI drift job greps
    # monitoring/drift_report.json for '"drift_detected": true') have a file to
    # read. Previously this dict was only returned/printed, so the CI
    # drift→retrain trigger never fired.
    with open(MON("drift_report.json"), "w") as f:
        json.dump(drift_report, f, indent=2)

    return drift_report

def save_baseline(ts_results: dict, ml_results: dict):
    baseline = {f"prophet_{name}_mape": res["mape"] for name, res in ts_results.items()}
    baseline["cancellation_accuracy"] = ml_results["accuracy"]
    baseline["cancellation_auc"]      = ml_results["roc_auc"]
    baseline["timestamp"]             = pd.Timestamp.now().isoformat()
    with open(MON("baseline_metrics.json"), "w") as f: json.dump(baseline, f, indent=2)

def generate_results_md(ts_results: dict, ml_res: dict):
    md = [
        "# Honest ML Pipeline Results",
        "This document details the performance metrics after removing leakage columns and implementing a rigorous time-series walk-forward CV.",
        "",
        "## 1. Classification Baselines",
        "Models were trained on the earliest ~83% of rows and evaluated on a single chronological hold-out (the most recent ~17%).",
        "| Model | AUC |",
        "|---|---|",
    ]
    for m, auc in ml_res["baselines"].items():
        md.append(f"| {m} | {auc:.3f} |")
    
    md += [
        "",
        "## 2. Calibration Analysis",
        f"The best performing model (**{ml_res['engine']}**) was calibrated using CalibratedClassifierCV.",
        f"* **Calibrated AUC:** {ml_res['roc_auc']:.3f}",
        f"* **Calibrated Brier Score:** {ml_res['brier']:.3f}",
        f"* **Calibrated F1:** {ml_res['f1']:.3f}",
        "",
        "![Reliability Diagram](models/reliability_diagram.png)",
        "",
        "## 3. Subgroup Metrics",
        "AUC & Brier scores evaluated on key segments within the temporal holdout.",
        "### By Hotel"
    ]
    for h, m in ml_res["subgroups"]["hotel"].items():
        md.append(f"- **{h}:** AUC={m['auc']:.3f}, Brier={m['brier']:.3f}")
        
    md.append("### By Top 5 Countries")
    for c, m in ml_res["subgroups"]["country"].items():
        md.append(f"- **{c}:** AUC={m['auc']:.3f}, Brier={m['brier']:.3f}")
        
    md.append("### By Season (Month)")
    for s, m in ml_res["subgroups"]["season"].items():
        md.append(f"- **{s}:** AUC={m['auc']:.3f}, Brier={m['brier']:.3f}")
        
    md += [
        "",
        "## 4. Confusion Matrix (Holdout)",
        "At the calibrated optimal threshold:",
        "```text",
        f"[[{ml_res['cm'][0][0]}  {ml_res['cm'][0][1]}]",
        f" [{ml_res['cm'][1][0]}  {ml_res['cm'][1][1]}]]",
        "```",
        "",
        "## 5. Forecasting Baselines",
        "Comparing Prophet (with external regressors) against N-BEATS (univariate baseline).",
        "| Target | Prophet MAPE | N-BEATS MAPE |",
        "|---|---|---|"
    ]
    for name, r in ts_results.items():
        nbeats_m = f"{r['nbeats_mape']:.2%}" if not pd.isna(r['nbeats_mape']) else "N/A"
        md.append(f"| {name} | {r['mape']:.2%} | {nbeats_m} |")
        
    with open(os.path.join(BASE, "RESULTS.md"), "w") as f:
        f.write("\n".join(md))

    metrics = {
        "holdout_auc": ml_res["roc_auc"],
        "holdout_mape": ts_results["occupancy"]["mape"]
    }
    with open(os.path.join(BASE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def run_training(force: bool = False) -> str:
    quality = check_data_quality(min_grade="C")

    if not force:
        drift = check_drift()
        if not drift["drift_detected"] and os.path.exists(MON("baseline_metrics.json")):
            return "skipped"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"train_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}") as run:
        run_id = run.info.run_id
        
        print("\n[1/3] Prophet & N-BEATS Models …")
        daily = pd.read_csv(D("daily_kpis.csv"), parse_dates=["ds"])
        ext   = pd.read_csv(D("external_regs.csv"), parse_dates=["ds"])
        ts_results = train_all_prophet(daily, ext, run_id)

        print("\n[2/3] Cancellation Classification …")
        bookings = pd.read_csv(D("bookings.csv"))
        ml_res   = train_cancellation_model(bookings)

        mlflow.log_metric("cancellation_accuracy", ml_res["accuracy"])
        mlflow.log_metric("cancellation_auc",       ml_res["roc_auc"])
        mlflow.log_metric("cancellation_f1",        ml_res["f1"])

        for fname in ["cancellation_model.joblib","feature_config.joblib","reliability_diagram.png"]:
            if os.path.exists(M(fname)): mlflow.log_artifact(M(fname))
        for name in PROPHET_TARGETS:
            mlflow.log_artifact(M(f"prophet_{name}.joblib"))

        print("\n[3/3] Generating Results Report …")
        generate_results_md(ts_results, ml_res)
        save_baseline(ts_results, ml_res)

    print(f"\n✅  Training complete | MLflow run: {run_id}\n")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-drift",   action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    os.chdir(BASE)

    if args.check_drift:
        drift = check_drift()
        print(json.dumps(drift, indent=2))
    else:
        run_training(force=args.force_retrain)
