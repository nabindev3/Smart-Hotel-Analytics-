# Retrospective

What went wrong, what I learned, and what I'd do differently. Written for myself,
but also because the bugs here are more interesting than the features.

## The metric that was too good

The first cancellation model scored an AUC in the low 0.9s and I was briefly
pleased with myself. It was leakage. `reservation_status` literally encodes the
outcome, and `booking_changes` / `days_in_waiting_list` are near-deterministic
proxies that mostly get populated *around* a cancellation. Pulling them dropped
the AUC to ~0.81.

**Lesson:** when a tabular model looks suspiciously good, my first move now is to
rank feature importances and ask "could I actually know this at prediction
time?" for the top few. A leaked feature usually sits right at the top.

## The threshold feature that did nothing for weeks

The API advertised an "F1-optimal decision threshold" and the serving code read
`best_threshold` from `feature_config.joblib`. Problem: training never *wrote*
that key. So `cfg.get("best_threshold", 0.5)` silently fell back to 0.5 on every
request. The feature was real in the code review and fictional at runtime.

**Lesson:** a serving path that reads an artifact is only as correct as the code
that *writes* the artifact. Now I check the persisted file actually contains the
keys the reader expects — `joblib.load(...).keys()` is a five-second check that
would have caught this. There's a CI artifact-presence gate now for the same
reason.

## 502s on deploy (the boring kind: out of memory)

Render kept returning 502s on cold start. It wasn't the code — it was four
Uvicorn workers each loading Prophet + SHAP into a 512 MB box, plus a startup
that blocked the port bind on those heavy loads long enough to trip the proxy
timeout. Fix was unglamorous: one worker, slim runtime deps, and warm the models
in a daemon thread so the port opens immediately.

**Lesson:** "works on my machine" hides the memory ceiling. I now think about the
deploy target's RAM as a hard constraint, not an afterthought, and I separated
`requirements.prod.txt` from the full training requirements so the serving image
isn't dragging TensorFlow and JupyterLab along.

## The build break I caused by being "safe"

I added `lightgbm` to the production requirements defensively — "in case the
model needs it." The served model is XGBoost; LightGBM is only used in the
training bake-off. The extra install weight contributed to a failed build, and
the package was dead weight at serve time anyway.

**Lesson:** production dependencies should match what you actually load, not what
training touched. "Might need it" is how slim images get fat. The honest fix
(deferred) is to derive prod deps from the saved engine name.

## Artifacts baked into the image

Models and CSVs were `COPY`'d into the Docker image and committed to git. That
couples "ship new models" to "make a git commit of binaries," and bloats both the
repo and the image with things that change on a different cadence than the code.

**Lesson / action:** I built an artifact-resolution layer (`backend/artifacts.py`)
so serve-time loads can come from an external store. I deliberately *didn't*
untrack the binaries yet — doing that before a store is live would break the
deploy. The mechanism is in; the flip is a documented, ordered step
(`ARTIFACTS.md`). Knowing where to stop mattered as much as the code.

## What I'd do differently next time

- **Write the data contract first.** A `pandera` schema on the raw frame would
  have caught half the cleaning special-cases before they became scattered
  `fillna`s. It's on the roadmap, not in the repo, and that's the wrong order.
- **Stand up the registry earlier.** MLflow tracks training runs but the backend
  loads raw `.joblib` paths, so the tracking is decorative at serve time. If I'd
  wired serving to the registry from the start, the artifact-store work would
  have been mostly free.
- **Lint from commit one.** I added a ruff gate late, scoped to `backend/` because
  `src/` had accumulated enough import-ordering and unused-import drift that
  fixing it safely is its own task. Cheap to enforce early, annoying to
  retrofit.
- **Stop gold-plating comments.** Some of my early comments explain the obvious.
  The ones worth keeping say *why* (the leakage rationale, the walk-risk
  fallback); the ones narrating *what* the next line does are noise.
