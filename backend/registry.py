"""
backend/registry.py — optional MLflow Model Registry loader for serving
=======================================================================
By default the backend loads the cancellation model from a local ``.joblib``
file (via the artifact resolver). That makes the MLflow tracking the training
pipeline does effectively decorative at serve time.

This module closes that loop *opt-in*. If ``MLFLOW_CANCELLATION_MODEL`` is set,
``load_cancellation_pipeline()`` loads that registered model from the MLflow
Model Registry instead — so promoting a new version in MLflow is all it takes to
roll the served model forward (or back), no redeploy. If the env var is unset,
mlflow isn't installed, or the registry load fails for any reason, it falls back
to the local joblib. So this is safe to ship: default behaviour is unchanged.

Configuration (all optional):
  MLFLOW_CANCELLATION_MODEL   registered model name, e.g. "hotel_cancellation"
  MLFLOW_CANCELLATION_STAGE   version / alias / stage to load (default: "latest")
  MLFLOW_TRACKING_URI         registry backend (e.g. a hosted MLflow server)

mlflow is intentionally NOT in requirements.prod.txt — registry-at-serve is an
opt-in feature; add mlflow to the image only if you turn it on.
"""
from __future__ import annotations

import logging
import os

import joblib

from backend.artifacts import artifact_path

logger = logging.getLogger("hotel_api.registry")


def load_cancellation_pipeline():
    """Return the cancellation sklearn Pipeline — from the MLflow registry when
    configured, otherwise from the local joblib artifact."""
    name = os.environ.get("MLFLOW_CANCELLATION_MODEL", "").strip()
    if name:
        try:
            import mlflow.sklearn

            uri = os.environ.get("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
            ref = os.environ.get("MLFLOW_CANCELLATION_STAGE", "latest")
            model_uri = f"models:/{name}/{ref}"
            logger.info("Loading cancellation model from MLflow registry: %s", model_uri)
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error(
                "Registry load failed for '%s'; falling back to local joblib: %s",
                name, e,
            )
    return joblib.load(artifact_path("models", "cancellation_model.joblib"))
