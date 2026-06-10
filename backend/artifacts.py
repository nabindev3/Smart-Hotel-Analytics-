"""
backend/artifacts.py — serve-time artifact resolution
=====================================================
Resolves model/data artifacts (e.g. ``models/cancellation_model.joblib``) to a
local filesystem path that the loaders can open.

Two modes, chosen automatically:

* **Local (default)** — if the artifact exists in the repo working tree, that
  path is returned. Dev and CI keep working with zero configuration, and the
  committed artifacts remain the source of truth until you flip to a store.

* **Remote** — when ``ARTIFACTS_BASE_URL`` is set, a missing artifact is
  downloaded on first use from ``${ARTIFACTS_BASE_URL}/<relative/path>`` into a
  local cache (``ARTIFACTS_CACHE_DIR``, default: the repo root) and reused
  thereafter. This lets the deploy image *stop baking artifacts in* — see
  ``Dockerfile.backend`` and ``scripts/publish_artifacts.py``.

The base URL is intentionally backend-agnostic: anything HTTP-reachable works —
GitHub Release assets, an S3/GCS public or presigned URL, a CDN, etc. The store
is chosen entirely by what URL you point it at; no provider-specific code lives
here.

Download failures are non-fatal: the function falls back to the (possibly
missing) local path, so the caller's existing "file not found" handling still
applies instead of this layer raising.
"""
from __future__ import annotations

import os
import tarfile
import logging
import threading
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-file store: a missing artifact is fetched from ${ARTIFACTS_BASE_URL}/<rel>.
# Works with any path-preserving HTTP store (S3/GCS/Azure/CDN/nginx). Empty
# string disables remote resolution (the default).
ARTIFACTS_BASE_URL = os.environ.get("ARTIFACTS_BASE_URL", "").rstrip("/")
# Bundle store: a single .tar.gz containing models/… and data/…, fetched and
# extracted once into the cache dir on first use. Use this for single-asset
# stores that flatten paths (e.g. GitHub Release assets).
ARTIFACTS_BUNDLE_URL = os.environ.get("ARTIFACTS_BUNDLE_URL", "")
# Where downloaded artifacts are cached. Defaults to the repo root so cached
# files land next to the local ones (models/…, data/…).
ARTIFACTS_CACHE_DIR = os.environ.get("ARTIFACTS_CACHE_DIR", ROOT)

logger = logging.getLogger("hotel_api.artifacts")
_download_lock = threading.Lock()
_bundle_ready = False


def _ensure_bundle() -> None:
    """If a bundle URL is configured, download and extract it once into the
    cache dir. Idempotent and thread-safe; failures are logged, not raised."""
    global _bundle_ready
    if not ARTIFACTS_BUNDLE_URL or _bundle_ready:
        return
    with _download_lock:
        if _bundle_ready:
            return
        tmp = os.path.join(ARTIFACTS_CACHE_DIR, "_artifacts_bundle.tar.gz")
        try:
            os.makedirs(ARTIFACTS_CACHE_DIR, exist_ok=True)
            logger.info("Downloading artifact bundle %s", ARTIFACTS_BUNDLE_URL)
            urllib.request.urlretrieve(ARTIFACTS_BUNDLE_URL, tmp)
            with tarfile.open(tmp, "r:gz") as tar:
                # Guard against path traversal in the archive.
                members = [m for m in tar.getmembers()
                           if not (m.name.startswith("/") or ".." in m.name.split("/"))]
                tar.extractall(ARTIFACTS_CACHE_DIR, members=members)
            _bundle_ready = True
            logger.info("Artifact bundle extracted into %s", ARTIFACTS_CACHE_DIR)
        except (urllib.error.URLError, OSError, tarfile.TarError) as e:
            logger.error("Artifact bundle fetch failed (%s): %s", ARTIFACTS_BUNDLE_URL, e)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


def artifact_path(*parts: str) -> str:
    """Return a local path for an artifact, downloading it on first use if a
    remote store is configured and the file is not already present.

    ``parts`` are path components relative to the project root, e.g.
    ``artifact_path("models", "cancellation_model.joblib")``.
    """
    rel = os.path.join(*parts)
    local = os.path.join(ROOT, rel)
    if os.path.exists(local):
        return local

    # Bundle mode populates the cache dir up-front; fall through to the per-file
    # cache lookup below once extracted.
    if ARTIFACTS_BUNDLE_URL:
        _ensure_bundle()
        bundled = os.path.join(ARTIFACTS_CACHE_DIR, rel)
        if os.path.exists(bundled):
            return bundled

    if not ARTIFACTS_BASE_URL:
        # No store configured — hand back the local path and let the caller's
        # existence check / loader raise as it does today.
        return local

    cached = os.path.join(ARTIFACTS_CACHE_DIR, rel)
    if os.path.exists(cached):
        return cached

    with _download_lock:
        # Re-check inside the lock in case a concurrent warm-up fetched it.
        if os.path.exists(cached):
            return cached
        url = f"{ARTIFACTS_BASE_URL}/{rel.replace(os.sep, '/')}"
        tmp = cached + ".part"
        try:
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            logger.info("Downloading artifact %s → %s", url, cached)
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, cached)
            return cached
        except (urllib.error.URLError, OSError) as e:
            logger.error("Artifact download failed (%s): %s", url, e)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return local  # fall back; caller surfaces the missing file
