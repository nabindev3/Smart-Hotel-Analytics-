"""
Tests for backend/artifacts.py — the serve-time artifact resolver.

These are hermetic: the remote cases use a local file:// "store" via tmp_path,
so there's no network and they run in the slim CI job.
"""
import os
import tarfile

import backend.artifacts as art


def test_local_file_resolves_to_repo_path():
    # A real committed artifact should resolve to its on-disk location.
    p = art.artifact_path("models", "feature_config.joblib")
    assert p.endswith(os.path.join("models", "feature_config.joblib"))
    assert os.path.exists(p)


def test_missing_without_store_returns_local_path_without_raising():
    # No store configured + file absent → hand back the local path and let the
    # caller's own existence check fail, rather than raising in this layer.
    p = art.artifact_path("models", "does_not_exist_42.joblib")
    assert p.endswith("does_not_exist_42.joblib")
    assert not os.path.exists(p)


def test_per_file_remote_download_and_cache(tmp_path, monkeypatch):
    store = tmp_path / "store" / "models"
    store.mkdir(parents=True)
    (store / "remote_only.joblib").write_text("REMOTE")
    cache = tmp_path / "cache"

    monkeypatch.setattr(art, "ARTIFACTS_BASE_URL", (tmp_path / "store").as_uri())
    monkeypatch.setattr(art, "ARTIFACTS_CACHE_DIR", str(cache))

    p = art.artifact_path("models", "remote_only.joblib")
    assert os.path.exists(p)
    assert str(cache) in p
    assert open(p).read() == "REMOTE"

    # Second call must reuse the cached copy (same path), not re-download.
    assert art.artifact_path("models", "remote_only.joblib") == p


def test_bundle_download_and_extract(tmp_path, monkeypatch):
    data_dir = tmp_path / "src" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "bundle_only.csv").write_text("a,b\n1,2\n")
    bundle = tmp_path / "artifacts.tar.gz"
    with tarfile.open(bundle, "w:gz") as t:
        t.add(data_dir / "bundle_only.csv", arcname="data/bundle_only.csv")

    cache = tmp_path / "bcache"
    monkeypatch.setattr(art, "ARTIFACTS_BUNDLE_URL", bundle.as_uri())
    monkeypatch.setattr(art, "ARTIFACTS_CACHE_DIR", str(cache))
    monkeypatch.setattr(art, "_bundle_ready", False)  # reset one-shot guard

    p = art.artifact_path("data", "bundle_only.csv")
    assert os.path.exists(p)
    assert str(cache) in p
