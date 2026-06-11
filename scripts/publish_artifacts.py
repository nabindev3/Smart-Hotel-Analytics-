#!/usr/bin/env python3
"""
publish_artifacts.py — bundle serve-time model/data artifacts for an external store
===================================================================================
Packages exactly the artifacts the backend loads at serve time (see
``backend/artifacts.py``) into a manifest + tarball so they can live outside the
git repo and outside the Docker image. This is the producer side of the
artifact-store mechanism that unblocks untracking the binaries (review #18/#24/#22).

Usage
-----
  # Build dist/artifacts.tar.gz + dist/manifest.json (for ARTIFACTS_BUNDLE_URL):
  python scripts/publish_artifacts.py

  # Also lay the files out path-preserved (sync this dir to S3/GCS/CDN, then
  # point ARTIFACTS_BASE_URL at its public/base URL):
  python scripts/publish_artifacts.py --dest-dir dist/artifacts

  # Upload the bundle to a GitHub release (requires `gh auth login`); the asset
  # URL becomes the value for ARTIFACTS_BUNDLE_URL:
  python scripts/publish_artifacts.py --gh-release v-artifacts-2026-06-09

Then on the deploy (e.g. Render) set ONE of:
  ARTIFACTS_BUNDLE_URL=https://github.com/<owner>/<repo>/releases/download/<tag>/artifacts.tar.gz
  ARTIFACTS_BASE_URL=https://<your-bucket-or-cdn>/<prefix>
…and the backend downloads artifacts on startup instead of needing them baked in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The exact artifacts the backend resolves via artifact_path(). Required ones
# must exist; optional ones are skipped with a warning if absent.
REQUIRED = [
    "models/cancellation_model.joblib",
    "models/feature_config.joblib",
    "models/recommender.joblib",
    "models/prophet_occupancy.joblib",
    "models/prophet_adr.joblib",
    "models/prophet_revenue.joblib",
    "data/bookings.csv",
    "data/daily_kpis.csv",
    "data/external_regs.csv",
]
OPTIONAL = [
    "models/ablation_results.json",
    "data/data_quality.json",
    # Parquet siblings (preferred fast path at serve time; CSV is the fallback).
    "data/bookings.parquet",
    "data/daily_kpis.parquet",
    "data/external_regs.parquet",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[str]:
    files: list[str] = []
    for rel in REQUIRED:
        if not os.path.exists(os.path.join(ROOT, rel)):
            sys.exit(f"ERROR: required artifact missing: {rel} "
                     f"(run src/generate_data.py + src/train_models_ts.py first)")
        files.append(rel)
    for rel in OPTIONAL:
        if os.path.exists(os.path.join(ROOT, rel)):
            files.append(rel)
        else:
            print(f"  (skipping optional, not found: {rel})")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="dist", help="output directory (default: dist/)")
    ap.add_argument("--dest-dir", help="also copy artifacts here, path-preserved (for ARTIFACTS_BASE_URL stores)")
    ap.add_argument("--gh-release", metavar="TAG", help="upload the bundle to this GitHub release tag via `gh`")
    args = ap.parse_args()

    files = collect()
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Manifest (sha256 + size) — lets consumers verify integrity.
    manifest = {rel: {"sha256": _sha256(os.path.join(ROOT, rel)),
                      "bytes": os.path.getsize(os.path.join(ROOT, rel))}
                for rel in files}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Bundle tarball (for ARTIFACTS_BUNDLE_URL).
    bundle_path = os.path.join(out_dir, "artifacts.tar.gz")
    with tarfile.open(bundle_path, "w:gz") as tar:
        for rel in files:
            tar.add(os.path.join(ROOT, rel), arcname=rel)
        tar.add(manifest_path, arcname="manifest.json")
    total = sum(m["bytes"] for m in manifest.values())
    print(f"Wrote {bundle_path} ({len(files)} files, {total/1e6:.1f} MB uncompressed)")

    # Optional path-preserved copy (for ARTIFACTS_BASE_URL stores).
    if args.dest_dir:
        dest = os.path.join(ROOT, args.dest_dir)
        for rel in files:
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, rel), target)
        shutil.copy2(manifest_path, os.path.join(dest, "manifest.json"))
        print(f"Copied path-preserved artifacts to {dest}")

    # Optional upload to a GitHub release (outward-facing — opt-in only).
    if args.gh_release:
        if not shutil.which("gh"):
            sys.exit("ERROR: --gh-release requires the GitHub CLI (`gh`) on PATH.")
        subprocess.run(["gh", "release", "create", args.gh_release,
                        bundle_path, manifest_path, "--title", args.gh_release,
                        "--notes", "Serve-time model/data artifacts."], check=False)
        subprocess.run(["gh", "release", "upload", args.gh_release,
                        bundle_path, manifest_path, "--clobber"], check=True)
        print(f"Uploaded bundle to GitHub release {args.gh_release}")


if __name__ == "__main__":
    main()
