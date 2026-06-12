#!/usr/bin/env python3
"""
check_prod_deps.py — guard the serving image against engine/dependency drift
============================================================================
The cancellation model is whichever engine won the training bake-off (XGBoost /
LightGBM / LogisticRegression). The serving image must carry that engine's
library or it can't unpickle the model. This exact mismatch bit us once
(LightGBM-trained model, lightgbm absent from the slim image), so this script
makes it a checkable invariant instead of a runtime surprise.

It reads the saved engine from models/feature_config.joblib and asserts:
  1. the engine's library is importable here, and
  2. its pip package is listed in requirements.prod.txt.

Exits non-zero with an actionable message on any mismatch. Wired into CI.
"""
from __future__ import annotations

import os
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# engine name (as saved by train_models_ts) → (import module, pip package)
ENGINE_DEPS = {
    "XGBoost":            ("xgboost", "xgboost"),
    "LightGBM":          ("lightgbm", "lightgbm"),
    "LogisticRegression": ("sklearn", "scikit-learn"),
}


def main() -> None:
    cfg_path = os.path.join(ROOT, "models", "feature_config.joblib")
    if not os.path.exists(cfg_path):
        sys.exit(f"feature_config.joblib not found at {cfg_path} — train a model first.")

    cfg = joblib.load(cfg_path)
    engine = cfg.get("engine") if isinstance(cfg, dict) else None
    if not engine:
        sys.exit("No 'engine' recorded in feature_config.joblib.")

    if engine not in ENGINE_DEPS:
        sys.exit(f"Unknown engine '{engine}' — add it to ENGINE_DEPS in this script.")

    module, pip_name = ENGINE_DEPS[engine]

    # 1. Importable in this environment?
    try:
        __import__(module)
    except ImportError:
        sys.exit(f"Saved model engine is '{engine}', but `import {module}` failed here. "
                 f"The serving environment cannot unpickle the model.")

    # 2. Listed in the production requirements?
    prod_req = os.path.join(ROOT, "requirements.prod.txt")
    with open(prod_req) as f:
        listed = {line.strip().split("==")[0].split()[0].lower()
                  for line in f if line.strip() and not line.startswith("#")}
    if pip_name.lower() not in listed:
        sys.exit(f"Saved model engine is '{engine}', which needs `{pip_name}`, but it is "
                 f"NOT in requirements.prod.txt. Add it or the serving image will fail to "
                 f"load the model.")

    print(f"OK: engine '{engine}' → {pip_name} is importable and in requirements.prod.txt.")


if __name__ == "__main__":
    main()
