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
import numpy as np
import pandas as pd
import shap
import joblib, os
from typing import Optional


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


class CancellationExplainer:
    """
    Wraps the trained sklearn Pipeline to produce SHAP explanations
    on the transformed feature space, then maps back to readable names.
    """

    def __init__(self, model_path: str = "models/cancellation_model.joblib"):
        pipeline = joblib.load(model_path)
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.clf           = pipeline.named_steps["classifier"]
        self._explainer:   Optional[shap.TreeExplainer] = None
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

    def _build_explainer(self, X_transformed: np.ndarray):
        if self._explainer is None:
            # Subsample background for efficiency
            bg = shap.sample(X_transformed, min(100, len(X_transformed)))
            self._explainer = shap.TreeExplainer(
                self.clf,
                data=bg,
                feature_perturbation="interventional",
            )

    def explain_global(
        self,
        X_raw: pd.DataFrame,
        n_samples: int = 500,
    ) -> dict:
        """
        Compute global SHAP values on a sample.
        Returns dict suitable for JSON serialisation and Plotly charts.
        """
        X_raw = X_raw.copy()
        for col in ["children","adr","days_in_waiting_list","meal","country"]:
            if col in X_raw.columns:
                if X_raw[col].dtype in ["float64","int64","float32"]:
                    X_raw[col] = X_raw[col].fillna(X_raw[col].median())
                else:
                    X_raw[col] = X_raw[col].fillna(X_raw[col].mode()[0]
                                                   if len(X_raw[col].mode())>0 else "BB")

        sample = X_raw.sample(min(n_samples, len(X_raw)), random_state=42)
        X_t    = self.preprocessor.transform(sample)

        self._build_explainer(X_t)
        shap_vals = self._explainer.shap_values(X_t)

        # For binary classifiers shap_values returns list[2] or 2D array
        if isinstance(shap_vals, list):
            sv = shap_vals[1]   # class 1 = Canceled
        else:
            sv = shap_vals

        names     = self._get_feature_names()
        mean_abs  = np.abs(sv).mean(axis=0)
        top20_idx = np.argsort(mean_abs)[::-1][:20]

        return {
            "feature_names":  [names[i] if i < len(names) else f"f{i}" for i in top20_idx],
            "mean_abs_shap":  [round(float(mean_abs[i]), 5) for i in top20_idx],
            "shap_matrix":    sv[:, top20_idx].tolist(),
            "base_value":     float(self._explainer.expected_value[1]
                                    if isinstance(self._explainer.expected_value, (list,np.ndarray))
                                    else self._explainer.expected_value),
            "n_samples":      len(sample),
        }

    def explain_instance(self, X_raw_row: pd.DataFrame) -> dict:
        """
        SHAP waterfall explanation for a single booking.
        Returns top contributing features with direction and magnitude.
        """
        row = X_raw_row.copy()
        for col in ["children","adr","days_in_waiting_list","meal","country"]:
            if col in row.columns:
                if row[col].dtype in ["float64","int64","float32"]:
                    row[col] = row[col].fillna(row[col].median() if row[col].notna().any() else 0)
                else:
                    row[col] = row[col].fillna("BB" if col=="meal" else "PRT")

        X_t = self.preprocessor.transform(row)
        self._build_explainer(X_t)
        sv  = self._explainer.shap_values(X_t)

        if isinstance(sv, list):
            vals = sv[1][0]
        else:
            vals = sv[0]

        base_val   = (self._explainer.expected_value[1]
                      if isinstance(self._explainer.expected_value, (list,np.ndarray))
                      else self._explainer.expected_value)
        names      = self._get_feature_names()

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
    bk = pd.read_csv("data/bookings.csv")
    exp = CancellationExplainer("models/cancellation_model.joblib")

    print("Computing global SHAP values (500 samples)…")
    g = exp.explain_global(bk[FEATURES])
    print("Top 5 features:")
    for name, val in zip(g["feature_names"][:5], g["mean_abs_shap"][:5]):
        print(f"  {name:<40} {val:.5f}")

    print("\nInstance explanation (booking #1):")
    row = bk[FEATURES].head(1)
    inst = exp.explain_instance(row)
    print(f"  Predicted cancel prob: {inst['prediction_prob']:.3f}")
    print(f"  Base value:            {inst['base_value']:.3f}")
    print(f"  Top risk factor:       {inst['top_risk_factor']}")
    for w in inst["waterfall"][:3]:
        arrow = "↑" if w["shap_value"]>0 else "↓"
        print(f"  {arrow} {w['feature']}: {w['shap_value']:+.4f}")
