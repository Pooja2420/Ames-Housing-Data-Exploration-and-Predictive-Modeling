"""Pandera schema contracts for the raw Ames Housing dataset.

Any CSV loaded through the data pipeline is validated against this schema
before any transformation is applied. This catches data drift and corrupt
files early rather than causing silent errors downstream.
"""

from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema

# ── Raw dataset schema ─────────────────────────────────────────────────────────
# Only the columns we actively use are declared; extras are allowed via
# `strict=False` so the schema does not break when columns are added upstream.

RAW_SCHEMA = DataFrameSchema(
    columns={
        # ── Target ────────────────────────────────────────────────────────────
        "SalePrice": Column(
            float,
            checks=[
                pa.Check.greater_than(0, error="SalePrice must be positive"),
                pa.Check.less_than(2_000_000, error="SalePrice > $2M — possible data error"),
            ],
            nullable=False,
            coerce=True,
        ),
        # ── Key numericals ────────────────────────────────────────────────────
        "Gr Liv Area": Column(
            float,
            checks=pa.Check.greater_than(0),
            nullable=False,
            coerce=True,
        ),
        "Overall Qual": Column(
            int,
            checks=pa.Check.isin(list(range(1, 11))),
            nullable=False,
            coerce=True,
        ),
        "Overall Cond": Column(
            int,
            checks=pa.Check.isin(list(range(1, 11))),
            nullable=False,
            coerce=True,
        ),
        "Year Built": Column(
            int,
            checks=[
                pa.Check.greater_than_or_equal_to(1800),
                pa.Check.less_than_or_equal_to(2025),
            ],
            nullable=False,
            coerce=True,
        ),
        "Yr Sold": Column(
            int,
            checks=pa.Check.isin([2006, 2007, 2008, 2009, 2010]),
            nullable=False,
            coerce=True,
        ),
        "Mo Sold": Column(
            int,
            checks=pa.Check.isin(list(range(1, 13))),
            nullable=False,
            coerce=True,
        ),
        "Lot Area": Column(
            float,
            checks=pa.Check.greater_than(0),
            nullable=True,    # allow null; imputed in preprocessor
            coerce=True,
        ),
        "Garage Cars": Column(
            float,
            checks=pa.Check.in_range(0, 10),
            nullable=True,
            coerce=True,
        ),
        "Garage Area": Column(
            float,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
        "Total Bsmt SF": Column(
            float,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
        # ── Key categoricals ──────────────────────────────────────────────────
        "MS Zoning": Column(
            str,
            checks=pa.Check.isin(["A", "C (all)", "FV", "I (all)", "RH", "RL", "RP", "RM"]),
            nullable=True,
            coerce=True,
        ),
        "Sale Condition": Column(
            str,
            checks=pa.Check.isin(
                ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"]
            ),
            nullable=True,
            coerce=True,
        ),
    },
    strict=False,       # allow extra columns
    coerce=True,        # attempt type coercion before failing
    name="AmesHousingRaw",
)


# ── Processed dataset schema (post-preprocessing) ─────────────────────────────
PROCESSED_SCHEMA = DataFrameSchema(
    columns={
        "SalePrice": Column(float, checks=pa.Check.greater_than(0), nullable=False),
        "Overall Qual": Column(float, nullable=False),
        "Gr Liv Area": Column(float, nullable=False),
    },
    strict=False,
    name="AmesHousingProcessed",
)
