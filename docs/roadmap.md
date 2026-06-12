# Roadmap

Roughly priority-ordered. Most of this comes out of the retrospective and the
review items I consciously deferred rather than half-build. "Status" is honest —
some of these are designed-but-not-done, not just ideas.

## Recently done

- **Data contract on ingestion.** ✅ `src/schemas.py` — a pandera contract wraps
  `clean_bookings` (structural check on raw, invariant check on clean).
- **Parquet for the analytical frames.** ✅ `src/data_io.py` reads parquet when
  present (CSV fallback); committed parquet siblings, ~7x smaller bookings.
- **Model registry at serve time.** ✅ Training registers `hotel_cancellation`;
  `backend/registry.py` serves it from the registry when configured, else joblib.
- **Lint the whole repo.** ✅ The ruff gate now covers `backend scripts tests
  frontend src` — cleaning `src/` also caught a real `UnboundLocalError` (a
  shadowing local `mlflow` import in training).

## Near term — pay down the known debt

- **Flip the artifact store on.** The resolver (`backend/artifacts.py`) and
  publish script exist; what's left is to publish a bundle, point
  `ARTIFACTS_BUNDLE_URL` at it, drop the `COPY` lines, and untrack the binaries.
  Ordered steps are in `ARTIFACTS.md`. *Status: mechanism done, switch not
  thrown (it breaks the deploy until a store is live, and needs the Render env
  set — neither of which can be done from the codebase alone).*
- **Derive prod deps from the saved engine.** So that if LightGBM ever wins the
  bake-off, the production image installs it automatically instead of failing to
  unpickle. *Status: idea.*

## Medium term — make the serving real

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
