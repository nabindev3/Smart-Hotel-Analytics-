# Roadmap

Roughly priority-ordered. Most of this comes out of the retrospective and the
review items I consciously deferred rather than half-build. "Status" is honest —
some of these are designed-but-not-done, not just ideas.

## Near term — pay down the known debt

- **Flip the artifact store on.** The resolver (`backend/artifacts.py`) and
  publish script exist; what's left is to publish a bundle, point
  `ARTIFACTS_BUNDLE_URL` at it, drop the `COPY` lines, and untrack the binaries.
  Ordered steps are in `ARTIFACTS.md`. *Status: mechanism done, switch not
  thrown (it breaks the deploy until a store is live).*
- **Data contract on ingestion.** Add a `pandera` (or Great Expectations) schema
  for the raw bookings frame — types, ranges, nullability — and fail fast with a
  readable error instead of letting bad rows trickle into scattered `fillna`s.
  *Status: not started.*
- **Lint `src/` too.** The CI ruff gate currently covers `backend scripts tests`.
  `src/` needs an import-ordering + unused-import cleanup (carefully — there's a
  `warnings.filterwarnings` ordering trap) before it can join the gate. *Status:
  scoped, not done.*
- **Derive prod deps from the saved engine.** So that if LightGBM ever wins the
  bake-off, the production image installs it automatically instead of failing to
  unpickle. *Status: idea.*

## Medium term — make the serving real

- **Model registry at serve time.** Training already logs to MLflow; wire the
  backend to load the *registered/promoted* model version instead of a raw
  `.joblib` path. Closes the loop and makes rollbacks a version bump.
- **Parquet over CSV.** `bookings.csv` etc. are re-read and re-parsed on every
  cold cache. Typed, columnar parquet is faster and removes the `parse_dates`
  guesswork.
- **Cost-based cancellation threshold.** Replace the F1-optimal default with a
  threshold derived from the actual cost of walking a guest vs. an empty room.
  This is the one that turns the risk score into a business decision.
- **Refit Prophet at request time.** Today the 400-day forecast is pickled with
  the model and goes stale immediately (there's a "doesn't reach today" fallback
  hack in the briefing endpoint). Persist only the fitted model and predict for
  the live horizon.

## Longer term — productisation

- **Real PMS/POS connector.** Replace the synthetic generator with an adapter
  that pulls real bookings/folios on the same schema. This is what makes every
  metric in the repo mean something.
- **Real recommender signal.** The current guest recommender factorises a
  rule-based synthetic interaction matrix — it can only re-derive its own rules.
  Swap in observed guest×service usage (spa bookings, upgrades, covers) before
  trusting any recommendation.
- **AuthN/AuthZ + per-tenant isolation.** The API is open and CORS is `*`. A real
  deployment needs API keys/OAuth and per-property data scoping.
- **Empirical price elasticity.** The pricing engine's elasticity (0.40) and
  bounds are hand-chosen assumptions, flagged as such in code. Estimate them from
  data (log-occupancy on log-ADR, controlling for season/events).

## Explicitly out of scope (for now)

- A full feature store. Overkill for this dataset size.
- Online/streaming inference. The use cases here are batch/daily.
- Multi-region HA. It's a portfolio project on a free tier, not a 99.99% SLA.
