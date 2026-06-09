# Contributing to Smart Hotel Analytics

Thanks for taking the time to contribute. This document describes the workflow
for proposing changes, the conventions the codebase follows, and what to expect
during review.

## Quick start

1. Fork the repository and clone your fork locally.
2. Create a branch off `main` with a descriptive name:
   ```
   git checkout -b fix/short-description
   ```
   Branch prefixes: `fix/`, `feat/`, `chore/`, `docs/`, `refactor/`.
3. Set up the dev environment:
   ```
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run the test suite before committing:
   ```
   pytest tests/ -q
   ```

## What kinds of contributions are welcome

- Bug fixes with a regression test or a clear reproduction.
- Performance improvements with a before/after benchmark.
- Documentation improvements, including clarifications to the README.
- New routers or model heads, provided they include tests.

Please open an issue before starting a large change so we can agree on scope.

## Code style

- Python 3.10+. Follow PEP 8; prefer explicit names over abbreviations.
- New endpoints live under `backend/routers/` and must use `@lru_cache` for
  any model or dataframe loaded from disk on the request hot path.
- Frontend changes go in `frontend/app.py`; avoid blocking calls without a
  timeout and a retry path.
- Keep diffs surgical. Don't reformat unrelated code.

## Tests

- Unit tests go in `tests/`.
- A test should fail before your fix and pass after it.
- Tests must not depend on a running backend unless guarded by `pytest.skip`
  when the service is unreachable.
- Avoid asserting only on shape (`0 <= p <= 1`); assert on behaviour
  (`accuracy > X` on a held-out fixture) when the change touches a model.

## Commit messages

- Use the imperative mood: "Fix sentiment 502" not "Fixed" or "Fixes".
- First line ≤ 72 characters; explain the *why*, not just the *what*, in the
  body when the change is non-obvious.
- Reference issues by number where applicable: `Fixes #42`.

## Pull requests

- Open the PR against `main`.
- Fill in the PR template; the checklist matters.
- Keep PRs small and focused. A 200-line PR will be reviewed faster than a
  1,000-line one.
- CI must be green before review.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the disclosure process.

## Code of conduct

This project adheres to the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to abide by its terms.

## Questions

Open a discussion or a low-priority issue. There is no Slack or chat for this
project; written, searchable channels only.
