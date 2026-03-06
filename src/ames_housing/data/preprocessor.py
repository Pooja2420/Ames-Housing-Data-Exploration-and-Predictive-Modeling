"""Data preprocessing pipeline for the Ames Housing dataset.

Steps (in order)
----------------
1.  Drop the PID identifier column
2.  Drop columns with > ``missing_threshold`` fraction of nulls
3.  Impute remaining nulls  (median for numerics, mode for categoricals)
4.  Remove sale-price outliers (IQR-based)
5.  Log-transform highly skewed numerical features
6.  One-hot encode all categorical columns
7.  Train / Validation / Test stratified split
8.  Persist splits as Parquet files

Usage
-----
    from ames_housing.data.preprocessor import run_preprocessing

    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from ames_housing.config import settings
from ames_housing.data.loader import load_raw
from ames_housing.data.schema import PROCESSED_SCHEMA
from ames_housing.utils.helpers import save_json


# ── Constants ─────────────────────────────────────────────────────────────────
_ID_COLS = ["PID", "Order"]
_TARGET = settings.data.target_column


# ── Individual steps ──────────────────────────────────────────────────────────

def _drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier columns that carry no predictive value."""
    to_drop = [c for c in _ID_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.debug("Dropped ID columns: {}", to_drop)
    return df


def _drop_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns whose missing-value fraction exceeds *threshold*."""
    missing_frac = df.isnull().mean()
    high_missing = missing_frac[missing_frac > threshold].index.tolist()

    if high_missing:
        logger.info(
            "Dropping {} columns with >{:.0%} missing: {}",
            len(high_missing),
            threshold,
            high_missing,
        )
        df = df.drop(columns=high_missing)
    else:
        logger.debug("No columns exceed the {:.0%} missing threshold.", threshold)

    return df


def _log_missing_summary(df: pd.DataFrame) -> None:
    """Log a tidy table of remaining columns that still have nulls."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.debug("No remaining null values after high-missing drop.")
        return
    logger.info("Remaining nulls (will be imputed):")
    for col, cnt in missing.items():
        logger.info("  {:<30} {:>4} ({:.1%})", col, cnt, cnt / len(df))


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute nulls: median for numerics, mode for categoricals."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    logger.debug("Imputation complete. Remaining nulls: {}", df.isnull().sum().sum())
    return df


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme SalePrice outliers using the 1.5 × IQR rule."""
    q1 = df[_TARGET].quantile(0.25)
    q3 = df[_TARGET].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    before = len(df)
    df = df[(df[_TARGET] >= lower) & (df[_TARGET] <= upper)]
    removed = before - len(df)

    logger.info(
        "Outlier removal: kept {}/{} rows (removed {} with SalePrice outside [{:.0f}, {:.0f}])",
        len(df),
        before,
        removed,
        lower,
        upper,
    )
    return df


def _log_transform_skewed(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Apply log1p to numerical columns whose |skewness| exceeds *threshold*.

    The target column is handled separately and always transformed when
    ``settings.features.log_transform_target`` is True.
    """
    # ── Target ────────────────────────────────────────────────────────────────
    if settings.features.log_transform_target and _TARGET in df.columns:
        df[_TARGET] = np.log1p(df[_TARGET])
        logger.info("Log-transformed target column '{}'", _TARGET)

    # ── Numerical features ────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.difference([_TARGET])
    skewness = df[num_cols].skew().abs()
    skewed_cols = skewness[skewness > threshold].index.tolist()

    if skewed_cols:
        df[skewed_cols] = np.log1p(df[skewed_cols].clip(lower=0))
        logger.info(
            "Log-transformed {} skewed feature(s) (threshold={:.2f}): {}",
            len(skewed_cols),
            threshold,
            skewed_cols,
        )

    return df, skewed_cols


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all object/category columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("One-hot encoding {} categorical column(s)", len(cat_cols))
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=np.float32)
    logger.info("Shape after encoding: {} rows × {} columns", *df.shape)
    return df


def _split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / val / test split on SalePrice quintiles."""
    X = df.drop(columns=[_TARGET])
    y = df[_TARGET]

    # Bin SalePrice into quintiles for stratification
    strat_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    # First split off test
    X_temp, X_test, y_temp, y_test, strat_temp, _ = train_test_split(
        X, y, strat_bins,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_bins,
    )

    # Then split remaining into train / val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        random_state=random_state,
        stratify=strat_temp,
    )

    logger.info(
        "Split → train:{} | val:{} | test:{}",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _persist(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save each split as Parquet and store feature names as JSON."""
    out = settings.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)

    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_df = X.copy()
        split_df[_TARGET] = y.values
        path = out / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info("Saved '{}' split → {} ({} rows)", name, path, len(split_df))

    # Persist feature names for the API layer
    feature_names_path = settings.paths.models / "feature_names.json"
    settings.paths.models.mkdir(parents=True, exist_ok=True)
    save_json({"features": X_train.columns.tolist()}, feature_names_path)
    logger.info("Feature names saved → {}", feature_names_path)


# ── Public entry point ────────────────────────────────────────────────────────

def run_preprocessing(
    raw_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Run the full preprocessing pipeline end-to-end.

    Parameters
    ----------
    raw_path:
        Override for the raw CSV path (useful in tests).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    cfg_data = settings.data
    cfg_feat = settings.features

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    # 1. Load + validate
    df = load_raw(raw_path)

    # 2. Drop ID cols
    df = _drop_id_columns(df)

    # 3. Drop high-missing columns
    df = _drop_high_missing(df, cfg_data.missing_threshold)
    _log_missing_summary(df)

    # 4. Impute remaining nulls
    df = _impute(df)

    # 5. Remove extreme outliers
    df = _remove_outliers(df)

    # 6. Log-transform skewed features
    df, skewed_cols = _log_transform_skewed(df, cfg_feat.skewness_threshold)

    # 7. One-hot encode categoricals
    df = _encode_categoricals(df)

    # 8. Validate processed schema
    logger.info("Validating processed schema ...")
    PROCESSED_SCHEMA.validate(df, lazy=True)
    logger.success("Processed schema OK.")

    # 9. Split
    X_train, X_val, X_test, y_train, y_val, y_test = _split(
        df,
        test_size=cfg_data.test_size,
        val_size=cfg_data.val_size,
        random_state=cfg_data.random_state,
    )

    # 10. Persist to disk
    _persist(X_train, X_val, X_test, y_train, y_val, y_test)

    logger.success("Preprocessing pipeline complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test
