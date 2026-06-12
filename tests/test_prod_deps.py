"""Tests for scripts/check_prod_deps.py — the engine/dependency guard."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_prod_deps", os.path.join(ROOT, "scripts", "check_prod_deps.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cpd = _load()


def test_engine_map_covers_the_bake_off_engines():
    assert set(cpd.ENGINE_DEPS) == {"XGBoost", "LightGBM", "LogisticRegression"}


def test_passes_for_the_committed_model(capsys):
    if not os.path.exists(os.path.join(ROOT, "models", "feature_config.joblib")):
        pytest.skip("no trained model present")
    cpd.main()  # must not raise SystemExit
    assert "OK:" in capsys.readouterr().out


def test_unknown_engine_is_rejected(monkeypatch):
    monkeypatch.setattr(cpd.joblib, "load", lambda *_: {"engine": "BogusEngine"})
    monkeypatch.setattr(cpd.os.path, "exists", lambda *_: True)
    with pytest.raises(SystemExit):
        cpd.main()


def test_engine_lib_missing_is_rejected(monkeypatch):
    # Engine maps to an import that doesn't exist → should fail fast.
    monkeypatch.setattr(cpd.joblib, "load", lambda *_: {"engine": "XGBoost"})
    monkeypatch.setattr(cpd.os.path, "exists", lambda *_: True)
    monkeypatch.setitem(cpd.ENGINE_DEPS, "XGBoost", ("no_such_module_xyz", "xgboost"))
    with pytest.raises(SystemExit):
        cpd.main()
