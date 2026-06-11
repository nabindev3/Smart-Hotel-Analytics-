# Design decisions

A running log of the calls I made on this project and why. These are the
questions I'd expect to get grilled on, so I'm writing down the reasoning while
it's fresh. Format is loosely ADR-style: context → decision → trade-off.

Newest first.

---

## Serve from an artifact resolver, not hardcoded paths

**Context.** Every router loaded models and CSVs with
`os.path.join(ROOT, "models", ...)`. That works, but it bakes a hard dependency
on the files sitting in the repo and in the Docker image. When I retrained, the
deploy only picked up new models if I committed ~9 MB of binaries to git.

**Decision.** Route every load through `backend/artifacts.py::artifact_path()`.
Default behaviour is unchanged (use the local file), but if `ARTIFACTS_BASE_URL`
or `ARTIFACTS_BUNDLE_URL` is set it downloads the artifact on first use and
caches it. So the *mechanism* to stop shipping binaries exists, even though I
haven't flipped the switch yet (see limitations).

**Trade-off.** It's an indirection layer that earns its keep only once there's a
real store behind it. I kept it dead-simple — stdlib `urllib`, no boto3, no
provider SDKs — so it works against S3, a CDN, or a GitHub release asset without
committing to any of them. If this never gets a real store it's mild
over-engineering; I think the option value is worth it.

## XGBoost is the served model; LightGBM is only a training dependency

**Context.** Training does a small bake-off (LogReg / XGBoost / LightGBM) and
keeps the best by AUC. XGBoost wins on this data. Early on I had `lightgbm` in
the *production* requirements "to be safe," and a Render build fell over partly
because of the extra install weight.

**Decision.** Production deps (`requirements.prod.txt`) contain only what's
needed to *load and serve* the chosen model — XGBoost, not LightGBM. Training
keeps both in the full `requirements.txt`.

**Trade-off.** If LightGBM ever wins the bake-off, the prod image won't be able
to unpickle the model until I add it back. That's a real footgun, so it's called
out in the retrospective. The right long-term fix is to key prod deps off the
saved engine, but I didn't want to over-build that yet.

## Drop the LP solver for overbooking; it was a one-line argmax in disguise

**Context.** `overbooking_engine.py` built a PuLP/CBC linear program whose only
job was "pick the single Δ with max expected profit subject to a walk-risk
cap." That's a filter + argmax over ~30 precomputed rows, not an optimisation
problem — and CBC is a heavy native dependency to drag in for it.

**Decision.** Replaced the LP with `max(feasible, key=profit)` plus an explicit
fallback to Δ=0 when even zero overbooking breaches the risk cap (which is what
the LP did implicitly by going infeasible). Dropped `pulp` from both
requirements files.

**Trade-off.** None I can see — same answer, fewer dependencies, easier to read.
If the model ever grew real coupling between decisions (multi-day, multi-room
LP), the solver would come back. It doesn't have that today.

## Single chronological hold-out, not a 5-fold split I throw away

**Context.** The code spun up `TimeSeriesSplit(n_splits=5)` and then used only
the last fold. So it paid for five splits, discarded four, and the comment
called it "walk-forward CV" — which it wasn't.

**Decision.** Made it an explicit single temporal split: train on the earliest
~83% of rows, evaluate on the most recent ~17%. The boundary is identical to the
old last-fold, so retraining reproduces the same model.

**Trade-off.** I lose any cross-fold variance estimate on the cancellation
metrics. For a leaderboard I'd want aggregated walk-forward folds; for a single
deployable model evaluated honestly out-of-time, one clean split is more honest
than pretending five folds mattered. Noted as future work.

## Calibrate probabilities and persist an F1-optimal threshold

**Context.** A raw GBM's `predict_proba` is not well-calibrated, and the risk
bands ("HIGH/MODERATE/LOW") in the API need a meaningful decision boundary. A
hard 0.5 cut is arbitrary for an imbalanced target.

**Decision.** Wrap the classifier in `CalibratedClassifierCV` (sigmoid) and, at
train time, compute the F1-optimal threshold on the calibrated probabilities and
**save it into `feature_config.joblib`**. The serving layer reads it back.

**Trade-off.** F1-optimal isn't business-optimal — the real threshold should
come from the cost of a walked guest vs. an empty room. F1 is a reasonable,
data-only default until those costs are known. (This is also where a real bug
lived for a while — see the retrospective.)

## Drop leakage features even though it tanks the AUC

**Context.** The public hotel-bookings dataset has columns that are only known
*after* the cancellation outcome (`reservation_status`) or are near-deterministic
proxies for it (`booking_changes`, `days_in_waiting_list`). Leaving them in
gives a gorgeous AUC and a useless model.

**Decision.** Remove them from the feature schema everywhere — training, the API
input model, and the SHAP explainer — and accept ~0.81 AUC instead of the
inflated number.

**Trade-off.** The headline metric looks worse. That's the point: it's the
honest ceiling. I'd rather defend 0.81 than explain why 0.93 evaporates in
production.

## Three-tier sentiment with graceful fallback

**Context.** Review sentiment can come from a hosted HF model, Claude, or local
TextBlob. Any of the first two can be unavailable (no key, rate limit, network).

**Decision.** Try HuggingFace → Claude → TextBlob, in that order, and always
return *something*. The model id is centralised in one constant so it's not
sprinkled across files.

**Trade-off.** The tiers have different quality, so the same review can score
differently depending on what's available. For a dashboard that's acceptable; for
anything downstream-critical it would need pinning to one engine.

## Synthetic data, loudly labelled

**Context.** I don't have access to a real PMS, and a portfolio project
shouldn't pretend otherwise.

**Decision.** Generate the data (`src/generate_data.py`) with deliberate
messiness — missing values, outliers, post-COVID drift — so the cleaning and
validation code has something real to do, and slap an unmissable "this is
synthetic" banner on the generator and the README.

**Trade-off.** Every metric in this repo measures fit-to-simulation, not
real-world skill. I'd rather be upfront about that than have someone assume the
numbers transfer.

## Slim prod image, one Uvicorn worker (Render free tier)

**Context.** Early deploys 502'd. The free tier has ~512 MB RAM, and Prophet +
SHAP + four workers blew past it.

**Decision.** One worker, a slim `requirements.prod.txt`, and warm the models in
a background thread so the port binds before the heavy loads finish.

**Trade-off.** One worker caps throughput. For a demo that's fine; a real deploy
would scale horizontally behind a load balancer rather than add workers to a
memory-starved box.
