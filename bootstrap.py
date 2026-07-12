#!/usr/bin/env python3
"""
bootstrap.py — One-click project bootstrap (data + model training)
==================================================================
NOTE: This is a task runner, not a setuptools script. Packaging metadata lives
in pyproject.toml; `pip install -e .` installs the `backend`/`src` packages.

python bootstrap.py             # generate data + train all models
python bootstrap.py --ablation  # also run ablation study (takes ~5 min)
python bootstrap.py --distil    # generate NLP training data via Claude API

Then:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
  streamlit run frontend/app.py

Or with Docker:
  docker-compose up --build
"""
import subprocess, sys, os, argparse

def run(cmd, desc):
    print(f"\n▶  {desc}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"❌  Failed: {cmd}")
        sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--ablation", action="store_true")
parser.add_argument("--distil",   action="store_true")
parser.add_argument("--n-distil", type=int, default=500)
args = parser.parse_args()

print("=" * 62)
print("  Smart Hotel Analytics — Enterprise Edition")
print("  Full Setup (10-Point Upgrade)")
print("=" * 62)

run(f"{sys.executable} -m pip install -r requirements.txt -q",  "Installing dependencies…")
run(f"{sys.executable} src/generate_data.py",   "Generating hotel datasets (messy + drift)…")
run(f"{sys.executable} src/train_models_ts.py", "Training Prophet + GBM + MLflow…")
# Import form (not `python src/recommender.py`): run as a script, the class
# pickles as __main__.GuestRecommender and the backend loader must retrain.
run(f"{sys.executable} -c 'from src.recommender import GuestRecommender; import pandas as pd; "
    "GuestRecommender().fit(pd.read_csv(\"data/bookings.csv\")).save(\"models/recommender.joblib\"); "
    "print(\"Recommender saved\")'",
    "Training rule-based recommender…")

if args.ablation:
    run(f"{sys.executable} src/ablation_study.py", "Running ablation study (p-value tests)…")

if args.distil:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set — skipping knowledge distillation.")
    else:
        run(f"{sys.executable} src/knowledge_distillation.py --generate --n {args.n_distil}",
            f"Generating {args.n_distil} NLP training samples via Claude…")

print("\n" + "=" * 62)
print("  ✅  Setup complete!")
print()
print("  Option A — Local development:")
print("    uvicorn backend.main:app --port 8000 &")
print("    streamlit run frontend/app.py")
print()
print("  Option B — Docker (recommended):")
print("    docker-compose up --build")
print()
print("  Option C — Run tests:")
print("    pytest tests/ -v")
print()
print("  MLflow UI:")
print("    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db")
print("=" * 62 + "\n")
