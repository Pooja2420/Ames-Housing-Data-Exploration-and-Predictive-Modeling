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
