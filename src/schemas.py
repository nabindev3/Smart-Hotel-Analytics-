"""
schemas.py — data contracts for the bookings pipeline
=====================================================
Two pandera schemas, used as a contract around the cleaning step in
``train_models_ts.clean_bookings``:

* ``RAW_BOOKINGS_SCHEMA`` — a *structural* contract on the ingested frame:
  required columns exist and coerce to the expected types, ``is_canceled`` is
  binary. It is intentionally lenient on ranges, because the synthetic generator
  deliberately injects messy values (impossible lead times, fat-finger ADRs) —
  catching those is the cleaning step's job, not the schema's.

* ``CLEAN_BOOKINGS_SCHEMA`` — an *invariant* contract on the frame that leaves
  cleaning: ranges are now tight (lead time bounded, non-negative ADR, at least
  one guest). This turns "the cleaner did its job" into an explicit, checkable
  postcondition instead of an assumption.

Validation is lazy (collects all violations) and raises a readable
``SchemaErrors`` on failure, so a bad input surfaces as one clear error up front
rather than a cascade of downstream ``KeyError``/``NaN`` surprises.
"""
from __future__ import annotations

import logging

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# Columns the rest of the pipeline relies on. Extra columns are allowed
# (strict=False) so upstream schema additions don't break ingestion.
RAW_BOOKINGS_SCHEMA = DataFrameSchema(
    {
        "booking_id":                     Column(int,   coerce=True),
        "arrival_date":                   Column("datetime64[ns]", coerce=True),
        "hotel":                          Column(str,   coerce=True, nullable=True),
        "is_canceled":                    Column(int,   Check.isin([0, 1]), coerce=True),
        "lead_time":                      Column(int,   coerce=True),
        "adults":                         Column(float, coerce=True, nullable=True),
        "children":                       Column(float, coerce=True, nullable=True),
        "babies":                         Column(float, coerce=True, nullable=True),
        "stays_in_weekend_nights":        Column(int,   coerce=True),
        "stays_in_week_nights":           Column(int,   coerce=True),
        "adr":                            Column(float, coerce=True, nullable=True),
        "country":                        Column(str,   coerce=True, nullable=True),
        "meal":                           Column(str,   coerce=True, nullable=True),
        "market_segment":                 Column(str,   coerce=True, nullable=True),
        "distribution_channel":           Column(str,   coerce=True, nullable=True),
        "reserved_room_type":             Column(str,   coerce=True, nullable=True),
        "deposit_type":                   Column(str,   coerce=True, nullable=True),
        "customer_type":                  Column(str,   coerce=True, nullable=True),
        "is_repeated_guest":              Column(int,   coerce=True),
        "previous_cancellations":         Column(int,   coerce=True),
        "previous_bookings_not_canceled": Column(int,   coerce=True),
        "required_car_parking_spaces":    Column(int,   coerce=True),
        "total_of_special_requests":      Column(int,   coerce=True),
    },
    strict=False,
    name="raw_bookings",
)

# Invariants that must hold *after* clean_bookings(). These mirror the filters in
# that function; if they ever fail, cleaning and the schema have drifted apart.
CLEAN_BOOKINGS_SCHEMA = DataFrameSchema(
    {
        "is_canceled":  Column(int,   Check.isin([0, 1])),
        "lead_time":    Column(int,   Check.in_range(0, 700)),
        "adr":          Column(float, Check.ge(0.0)),
        "total_guests": Column(float, Check.gt(0.0), coerce=True),
        "total_stay":   Column(float, Check.ge(0.0), coerce=True),
    },
    strict=False,
    name="clean_bookings",
)


def validate_raw_bookings(df):
    """Validate + coerce the raw bookings frame. Raises pandera SchemaErrors
    (with every violation listed) if the structural contract is broken."""
    try:
        out = RAW_BOOKINGS_SCHEMA.validate(df, lazy=True)
        logger.info("Raw bookings passed schema contract (%d rows).", len(out))
        return out
    except pa.errors.SchemaErrors as err:
        logger.error("Raw bookings failed the data contract:\n%s", err.failure_cases)
        raise


def validate_clean_bookings(df):
    """Assert the post-cleaning invariants. Raises on violation."""
    out = CLEAN_BOOKINGS_SCHEMA.validate(df, lazy=True)
    logger.info("Clean bookings satisfied invariant contract (%d rows).", len(out))
    return out
