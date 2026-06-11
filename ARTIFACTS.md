# Model & Data Artifacts

The backend needs trained models (`models/*.joblib`) and reference data
(`data/*.csv`, `*.json`) at serve time. This document explains how those
artifacts are resolved and how to move them out of the git repo / Docker image.

## How resolution works

All serve-time loads go through [`backend/artifacts.py`](backend/artifacts.py)
`artifact_path("models", "x.joblib")`, which resolves in this order:

1. **Local file** in the repo working tree → used directly (default; zero config).
2. **Bundle** — if `ARTIFACTS_BUNDLE_URL` is set, a single `artifacts.tar.gz` is
   downloaded once and extracted into the cache dir.
3. **Per-file** — if `ARTIFACTS_BASE_URL` is set, each missing file is fetched
   from `${ARTIFACTS_BASE_URL}/<relative/path>`.
4. **Fallback** — the (missing) local path is returned and the loader raises as
   before. Download failures are logged, never fatal.

| Env var | Use for | Example |
|---------|---------|---------|
| `ARTIFACTS_BUNDLE_URL` | single-asset stores that flatten paths | a GitHub Release asset `…/releases/download/<tag>/artifacts.tar.gz` |
| `ARTIFACTS_BASE_URL` | path-preserving stores | `https://my-bucket.s3.amazonaws.com/hotel-artifacts` |
| `ARTIFACTS_CACHE_DIR` | where downloads land (default: repo root) | `/app` |

The startup warm-up loads every model, so artifacts download during boot, not on
the first user request. `/health` reflects the actual load/download outcome.

## Publishing artifacts

```bash
python src/generate_data.py && python src/train_models_ts.py   # produce artifacts
python scripts/publish_artifacts.py                            # → dist/artifacts.tar.gz + manifest.json
```

- **GitHub Releases (free, no new infra):**
  `python scripts/publish_artifacts.py --gh-release v-artifacts-YYYY-MM-DD`
  then set `ARTIFACTS_BUNDLE_URL` to the uploaded asset URL.
- **S3 / GCS / CDN:**
  `python scripts/publish_artifacts.py --dest-dir dist/artifacts`, sync that
  tree to your bucket, and set `ARTIFACTS_BASE_URL` to its base URL.

## The flip: stop baking artifacts in & untrack them (review #18/#24/#22)

The mechanism above is built and tested, but the artifacts are still committed
to git and baked into the image (the safe default). Completing the untracking is
**deploy-breaking until a store is live**, so it is a deliberate, ordered manual
step:

1. **Publish** the artifacts to a store (above) and verify the URL is reachable.
2. **Configure** the deploy: set `ARTIFACTS_BUNDLE_URL` (or `ARTIFACTS_BASE_URL`)
   in the Render service env.
3. **Slim the image:** delete the `COPY data/` and `COPY models/` lines in
   [`Dockerfile.backend`](Dockerfile.backend) (marked "LEAN mode").
4. **Untrack the binaries** (only now safe):
   ```bash
   git rm -r --cached models/ data/*.csv data/*.json
   echo -e "models/\ndata/*.csv\ndata/*.json" >> .gitignore
   ```
5. **Verify** a fresh deploy boots and `/health` is `healthy` (artifacts came
   from the store, not the image).

Do **not** do steps 3–4 before steps 1–2 — the service would start with no
models. Until then, keeping the artifacts tracked is correct.
