"""
Tests for the bookings data contract (src/schemas.py).

pandera is a training/ingestion dependency, not a serving one, so skip cleanly if
it isn't installed in this environment.
"""
import os

import pandas as pd
import pytest

pytest.importorskip("pandera")

from src.schemas import validate_clean_bookings, validate_raw_bookings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def raw_bookings():
    path = os.path.join(ROOT, "data", "bookings.csv")
    if not os.path.exists(path):
        pytest.skip("bookings.csv not found")
    return pd.read_csv(path)


def test_real_data_passes_raw_contract(raw_bookings):
    out = validate_raw_bookings(raw_bookings)
    assert len(out) == len(raw_bookings)
    # arrival_date must have been coerced to a real datetime.
    assert pd.api.types.is_datetime64_any_dtype(out["arrival_date"])


def test_missing_required_column_is_rejected(raw_bookings):
    import pandera as pa
    bad = raw_bookings.drop(columns=["is_canceled"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_bookings(bad)


def test_non_binary_target_is_rejected(raw_bookings):
    import pandera as pa
    bad = raw_bookings.copy()
    bad.loc[bad.index[0], "is_canceled"] = 7  # not in {0, 1}
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_bookings(bad)


def test_clean_invariants_catch_bad_ranges():
    import pandera as pa
    # A frame that violates the post-cleaning invariants (negative ADR, zero
    # guests, out-of-range lead time) must be rejected.
    bad = pd.DataFrame({
        "is_canceled":  [0, 1],
        "lead_time":    [10, 9999],   # 9999 > 700
        "adr":          [100.0, -5.0],  # negative ADR
        "total_guests": [2.0, 0.0],     # zero guests
        "total_stay":   [3.0, 2.0],
    })
    with pytest.raises(pa.errors.SchemaErrors):
        validate_clean_bookings(bad)
