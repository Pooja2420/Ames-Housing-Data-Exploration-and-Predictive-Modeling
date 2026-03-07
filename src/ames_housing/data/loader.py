"""Data loader — reads the raw CSV and validates it against the Pandera schema.

Usage
-----
    from ames_housing.data.loader import load_raw

    df = load_raw()          # loads from path in config
    df = load_raw("custom/path.csv")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
from loguru import logger

from ames_housing.config import settings
from ames_housing.data.schema import RAW_SCHEMA
from ames_housing.utils.helpers import memory_usage_mb, reduce_memory


# ── Column name mapping ────────────────────────────────────────────────────────
# The OpenML version uses underscores and slightly different names vs the
# original Kaggle/DePaul CSV that uses spaces. This map normalises both
# formats to the canonical spaced format used throughout the codebase.
_OPENML_RENAME: dict[str, str] = {
    # Special cases that can't be handled by simple underscore→space replace
    "Sale Price":        "SalePrice",
    "Year Sold":         "Yr Sold",
    "First Flr SF":      "1st Flr SF",
    "Second Flr SF":     "2nd Flr SF",
    "Three season porch":"3Ssn Porch",
    "Year Remod Add":    "Year Remod/Add",
    "MS SubClass":       "MS SubClass",   # already correct after replace
    "Bedroom AbvGr":     "Bedroom AbvGr", # already correct
    "TotRms AbvGrd":     "TotRms AbvGrd", # already correct
}

# Extra columns present in OpenML version not needed for modelling
_DROP_OPENML_EXTRAS = {"Longitude", "Latitude"}

# ── Ordinal quality/condition label mapping ────────────────────────────────────
# OpenML stores Overall Qual & Overall Cond as English strings rather than the
# 1-10 integers used in the Kaggle/DePaul version.  Values may still have
# underscores at point of mapping (cell values are not touched by the column-
# name normalisation step, so we handle both underscore and space variants).
_QUAL_ORDINAL_MAP: dict[str, int] = {
    "Very_Poor":      1,  "Very Poor":      1,
    "Poor":           2,
    "Fair":           3,
    "Below_Average":  4,  "Below Average":  4,
    "Average":        5,
    "Above_Average":  6,  "Above Average":  6,
    "Good":           7,
    "Very_Good":      8,  "Very Good":      8,
    "Excellent":      9,
    "Very_Excellent": 10, "Very Excellent": 10,
}
_ORDINAL_COLS = {"Overall Qual", "Overall Cond"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to the canonical Ames Housing (Kaggle) format.

    Handles both:
    - Original Kaggle format  : spaces  (``Gr Liv Area``, ``SalePrice``)
    - OpenML / underscore format: underscores (``Gr_Liv_Area``, ``Sale_Price``)
    """
    # Detect OpenML format by checking for underscores in column names
    if any("_" in col for col in df.columns):
        logger.info("OpenML column format detected — normalising column names.")
        # Step 1: replace underscores with spaces
        df.columns = df.columns.str.replace("_", " ", regex=False)
        # Step 2: apply specific overrides
        df = df.rename(columns=_OPENML_RENAME)
        # Step 3: drop extra OpenML-only columns
        extras = _DROP_OPENML_EXTRAS & set(df.columns)
        if extras:
            df = df.drop(columns=list(extras))
            logger.debug("Dropped OpenML-only columns: {}", extras)

        # Step 4: map text quality/condition labels → integers (OpenML only)
        for col in _ORDINAL_COLS & set(df.columns):
            if df[col].dtype == object:
                df[col] = df[col].map(_QUAL_ORDINAL_MAP)
                unmapped = df[col].isna().sum()
                if unmapped:
                    logger.warning(
                        "Column '{}': {} values could not be mapped to ordinal int",
                        col, unmapped,
                    )
                else:
                    logger.debug("Column '{}': text labels mapped to 1-10 integers.", col)

        logger.info("Column normalisation complete.")
    return df


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Ames Housing CSV and validate its schema.

    Parameters
    ----------
    path:
        Path to ``AmesHousing.csv``. Defaults to ``settings.paths.data_raw``.

    Returns
    -------
    pd.DataFrame
        Validated raw DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    pandera.errors.SchemaError
        If the DataFrame fails schema validation.
    """
    csv_path = Path(path) if path else settings.paths.data_raw

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{csv_path}'.\n"
            "Please place AmesHousing.csv in data/raw/ or set AMES_PATHS__DATA_RAW."
        )

    logger.info("Loading raw data from '{}'", csv_path)
    df = pd.read_csv(csv_path)
    logger.info(
        "Loaded {} rows × {} columns | {:.2f} MB",
        len(df),
        df.shape[1],
        memory_usage_mb(df),
    )

    # ── Column normalisation (handles OpenML underscore format) ────────────────
    df = _normalize_columns(df)

    # ── Schema validation ──────────────────────────────────────────────────────
    logger.info("Validating raw data schema ...")
    try:
        df = RAW_SCHEMA.validate(df, lazy=True)   # lazy=True collects ALL errors
        logger.success("Schema validation passed.")
    except pa.errors.SchemaErrors as exc:
        # Log every failure then re-raise so CI catches it
        logger.error("Schema validation FAILED with {} error(s):", len(exc.failure_cases))
        for _, row in exc.failure_cases.iterrows():
            logger.error("  column={!r}  check={!r}  value={!r}", row["column"], row["check"], row["failure_case"])
        raise

    # ── Memory optimisation ────────────────────────────────────────────────────
    df = reduce_memory(df)
    logger.debug("After memory reduction: {:.2f} MB", memory_usage_mb(df))

    return df


def load_processed(split: str = "train") -> pd.DataFrame:
    """Load a previously saved processed split (train / val / test).

    Parameters
    ----------
    split:
        One of ``'train'``, ``'val'``, ``'test'``.
    """
    valid_splits = ("train", "val", "test")
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got {split!r}")

    path = settings.paths.data_processed / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split '{split}' not found at '{path}'.\n"
            "Run the preprocessing pipeline first."
        )

    logger.info("Loading processed '{}' split from '{}'", split, path)
    df = pd.read_parquet(path)
    logger.info("Loaded {} rows × {} columns", *df.shape)
    return df
