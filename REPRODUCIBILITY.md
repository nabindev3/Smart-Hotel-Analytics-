# Reproducibility

This project is designed to be reproducible end-to-end from a clean checkout.
Everything downstream of data generation is deterministic given the fixed seeds
below; the one caveat is noted under *Known sources of non-determinism*.

> Reminder: all data is **synthetic** (see [`src/generate_data.py`](src/generate_data.py)).
> Reproducing the pipeline reproduces a *simulation*, not real-world results.

## Reproduce from scratch

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # full train+serve deps
python src/generate_data.py            # writes data/*.csv (synthetic)
python src/train_models_ts.py          # writes models/*.joblib, metrics.json, RESULTS.md
pytest tests/ -v                        # enforces metric floors (auc>0.80, mape<0.16)
```

The backend then serves the artifacts in `models/` with no retraining
(`backend/routers/*.py` load `models/*.joblib` directly).

## Fixed seeds

| Stage | Location | Seed |
|-------|----------|------|
| Synthetic data generation | `src/generate_data.py` | `np.random.seed(2024)`, plus `default_rng(42/77/99)` for sub-streams |
| Train/test temporal split | `src/train_models_ts.py` | deterministic chronological cut (last 1/6 of rows) — no RNG |
| Classifiers (LogReg / XGBoost / LightGBM) | `src/train_models_ts.py` | `random_state=42` |
| Calibration (`CalibratedClassifierCV`) | `src/train_models_ts.py` | inherits estimator seed; `cv=3` is deterministic |
| SHAP background sampling | `src/shap_explainer.py` | `random_state=42` |
| Recommender SVD + synthetic interactions | `src/recommender.py` | `default_rng(42)`, `TruncatedSVD(random_state=42)` |

## Pinned environment

`requirements.txt` is fully version-pinned (a `pip freeze`-style lock).
`requirements.prod.txt` is the slim runtime subset used by the backend Docker
image. Key versions: Python 3.11, scikit-learn 1.7.2, xgboost 3.2.0,
prophet 1.3.0, shap 0.49.1.

## Verifying the data you trained on

The trained artifacts are derived from `data/bookings.csv`. To confirm a model
matches the data it was trained on, hash the input:

```bash
shasum -a 256 data/bookings.csv
```

## Known sources of non-determinism

- **N-BEATS baseline** (`neuralforecast`/PyTorch-Lightning) is **not** seeded and
  runs only for the `occupancy` target on macOS (see the semaphore-leak note in
  `src/train_models_ts.py`). Its MAPE may vary run to run; the Prophet forecasts
  and the cancellation model are deterministic.

## Not yet automated (future work)

- A run manifest emitted at train time (data hash + library versions + git SHA +
  resulting metrics) — today `MODEL_CARD.md` is maintained by hand.
- A model registry as the serving source of truth (the backend currently loads
  raw `models/*.joblib` paths; MLflow tracking is recorded but not consulted at
  serve time).
