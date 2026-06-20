"""
shap_explainer.py — XAI via SHAP (SHapley Additive exPlanations)
=================================================================
Wraps the trained GBM cancellation model in SHAP to produce:
  • Global feature importance (beeswarm + bar charts)
  • Local explanation per prediction ("why is this booking high risk?")
  • Waterfall chart for individual predictions
  • Feature interaction heatmap

Uses TreeExplainer (O(TLD²) complexity vs O(2^d) brute force).
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

# Must match the training schema in src/train_models_ts.py exactly. The two
# leakage columns (booking_changes, days_in_waiting_list) were dropped there,
# so they must NOT appear here — otherwise NUM has more entries than the fitted
# preprocessor produces and SHAP feature names get shifted/mislabeled.
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


class CancellationExplainer:
    """
    Wraps the trained sklearn Pipeline to produce SHAP explanations
    on the transformed feature space, then maps back to readable names.
    """

    def __init__(self, model_path: str = "models/cancellation_model.joblib", model=None):
        # Accept either a path (joblib) or an already-loaded pipeline (e.g. one
        # pulled from the MLflow registry), so the explainer and the predictor
        # can share a single model source.
        pipeline = model if model is not None else joblib.load(model_path)
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.clf           = pipeline.named_steps["classifier"]
        self._explainer:   Optional[shap.Explainer] = None
        self._feature_names: list[str] = []

    def _get_feature_names(self) -> list[str]:
        if self._feature_names:
            return self._feature_names
        try:
            num_names = NUM.copy()
            cat_names = (self.preprocessor
                         .named_transformers_["cat"]
                         .get_feature_names_out(CAT).tolist())
            self._feature_names = num_names + cat_names
        except Exception:
            n = self.clf.n_features_in_
            self._feature_names = [f"feature_{i}" for i in range(n)]
        return self._feature_names

    def _build_explainer(self, background: np.ndarray):
        # TreeExplainer can't explain a CalibratedClassifierCV wrapper (and the
        # unwrapped XGBoost trips a shap/xgboost base_score parse bug), so we
        # explain the *served, calibrated* model directly with the modern
        # model-agnostic Explainer over its class-1 probability. `background`
        # should be a representative sample, not the row(s) being explained.
        if self._explainer is None:
            # Keep the reference set small — on a 512 MB box a large background
            # multiplies both the per-call RAM and the (already throttled) CPU
            # cost of the model-agnostic explainer.
            bg = shap.sample(background, min(50, len(background)), random_state=0)
            self._explainer = shap.Explainer(
                lambda d: self.clf.predict_proba(d)[:, 1], bg)

    @staticmethod
    def _fillna(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["children", "adr", "meal", "country"]:
            if col not in df.columns:
                continue
            if df[col].dtype in ["float64", "int64", "float32"]:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if len(mode) else "BB")
        return df

    def explain_global(self, X_raw: pd.DataFrame, n_samples: int = 500) -> dict:
        """Compute global SHAP values on a sample (JSON-serialisable)."""
        sample = self._fillna(X_raw).sample(min(n_samples, len(X_raw)), random_state=42)
        X_t = self.preprocessor.transform(sample)

        self._build_explainer(X_t)
        exp = self._explainer(X_t)
        sv = np.asarray(exp.values)              # (n, dims) — single class-1 output
        base = float(np.mean(exp.base_values))

        names     = self._get_feature_names()
        mean_abs  = np.abs(sv).mean(axis=0)
        top20_idx = np.argsort(mean_abs)[::-1][:20]

        return {
            "feature_names":  [names[i] if i < len(names) else f"f{i}" for i in top20_idx],
            "mean_abs_shap":  [round(float(mean_abs[i]), 5) for i in top20_idx],
            "shap_matrix":    sv[:, top20_idx].tolist(),
            "base_value":     round(base, 5),
            "n_samples":      len(sample),
        }

    def explain_instance(self, X_raw_row: pd.DataFrame,
                         background_raw: Optional[pd.DataFrame] = None) -> dict:
        """SHAP waterfall explanation for a single booking. `background_raw` is a
        representative sample used as the SHAP reference; without it the
        explanation is taken against the row itself (degraded)."""
        row = self._fillna(X_raw_row)
        X_t = self.preprocessor.transform(row)
        bg_t = (self.preprocessor.transform(self._fillna(background_raw))
                if background_raw is not None else X_t)

        self._build_explainer(bg_t)
        exp  = self._explainer(X_t)
        vals = np.asarray(exp.values)[0]
        base_val = float(np.asarray(exp.base_values)[0])
        names    = self._get_feature_names()

        # Map back to readable names (just top features)
        top_idx    = np.argsort(np.abs(vals))[::-1][:10]
        waterfall  = []
        for i in top_idx:
            name = names[i] if i < len(names) else f"f{i}"
            # Shorten OHE feature names
            if "__" in name:
                name = name.split("__", 1)[1]
            waterfall.append({
                "feature":    name,
                "shap_value": round(float(vals[i]), 5),
                "direction":  "increases_risk" if vals[i] > 0 else "decreases_risk",
            })

        pred_prob = float(self.clf.predict_proba(X_t)[0][1])
        return {
            "base_value":      round(float(base_val), 4),
            "prediction_prob": round(pred_prob, 4),
            "waterfall":       waterfall,
            "top_risk_factor": waterfall[0]["feature"] if waterfall else "N/A",
        }


if __name__ == "__main__":
    import json
    import os

    bk = pd.read_csv("data/bookings.csv")
    exp = CancellationExplainer("models/cancellation_model.joblib")

    print("Computing global SHAP values (500 samples)…")
    g = exp.explain_global(bk[FEATURES])
    print("Top 5 features:")
    for name, val in zip(g["feature_names"][:5], g["mean_abs_shap"][:5]):
        print(f"  {name:<40} {val:.5f}")

    # Persist the slim artifact the API serves from GET /api/v1/xai/global-importance.
    # Serving this precomputed file avoids running full-frame SHAP at request time,
    # which OOM-kills the 512 MB free-tier worker. Rerun this after retraining.
    out = os.path.join("models", "global_importance.json")
    with open(out, "w") as f:
        json.dump({k: g[k] for k in ("feature_names", "mean_abs_shap",
                                     "base_value", "n_samples")}, f, indent=2)
    print(f"Saved {out} (n_samples={g['n_samples']}).")

    print("\nInstance explanation (booking #1):")
    row = bk[FEATURES].head(1)
    inst = exp.explain_instance(row)
    print(f"  Predicted cancel prob: {inst['prediction_prob']:.3f}")
    print(f"  Base value:            {inst['base_value']:.3f}")
    print(f"  Top risk factor:       {inst['top_risk_factor']}")
    for w in inst["waterfall"][:3]:
        arrow = "↑" if w["shap_value"]>0 else "↓"
        print(f"  {arrow} {w['feature']}: {w['shap_value']:+.4f}")
