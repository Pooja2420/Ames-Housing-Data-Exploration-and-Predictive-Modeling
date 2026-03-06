"""Tests for the data pipeline: schema validation, loader, preprocessor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from ames_housing.data.schema import PROCESSED_SCHEMA, RAW_SCHEMA
from tests.conftest import make_synthetic_df


# ── Schema validation ──────────────────────────────────────────────────────────

class TestRawSchema:
    def test_valid_dataframe_passes(self, raw_df: pd.DataFrame):
        """A well-formed DataFrame should pass RAW_SCHEMA validation."""
        validated = RAW_SCHEMA.validate(raw_df, lazy=True)
        assert len(validated) == len(raw_df)

    def test_negative_saleprice_fails(self, raw_df: pd.DataFrame):
        """SalePrice ≤ 0 must be rejected."""
        bad_df = raw_df.copy()
        bad_df.loc[0, "SalePrice"] = -1.0
        with pytest.raises(pa.errors.SchemaErrors):
            RAW_SCHEMA.validate(bad_df, lazy=True)

    def test_invalid_overall_qual_fails(self, raw_df: pd.DataFrame):
        """OverallQual outside [1, 10] must be rejected."""
        bad_df = raw_df.copy()
        bad_df.loc[0, "Overall Qual"] = 99
        with pytest.raises(pa.errors.SchemaErrors):
            RAW_SCHEMA.validate(bad_df, lazy=True)

    def test_invalid_yr_sold_fails(self, raw_df: pd.DataFrame):
        """YrSold outside dataset range must be rejected."""
        bad_df = raw_df.copy()
        bad_df.loc[0, "Yr Sold"] = 2099
        with pytest.raises(pa.errors.SchemaErrors):
            RAW_SCHEMA.validate(bad_df, lazy=True)

    def test_extra_columns_allowed(self, raw_df: pd.DataFrame):
        """Schema is non-strict — extra columns should not raise."""
        df_extra = raw_df.copy()
        df_extra["UnknownColumn"] = 999
        validated = RAW_SCHEMA.validate(df_extra, lazy=True)
        assert "UnknownColumn" in validated.columns

    def test_nullable_columns_pass_with_nulls(self, raw_df: pd.DataFrame):
        """Columns marked nullable=True should not fail when null."""
        df_null = raw_df.copy()
        df_null.loc[:5, "Lot Area"] = np.nan
        validated = RAW_SCHEMA.validate(df_null, lazy=True)
        assert validated["Lot Area"].isnull().sum() == 6


class TestProcessedSchema:
    def test_valid_processed_df_passes(self, raw_df: pd.DataFrame):
        """A processed DataFrame with target + key features should pass."""
        PROCESSED_SCHEMA.validate(raw_df, lazy=True)


# ── Loader ─────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_raw_file_not_found(self):
        """load_raw should raise FileNotFoundError for missing CSV."""
        from ames_housing.data.loader import load_raw
        with pytest.raises(FileNotFoundError, match="Raw data not found"):
            load_raw(path="/nonexistent/path/AmesHousing.csv")

    def test_load_raw_returns_dataframe(self, tmp_path: Path, raw_df: pd.DataFrame):
        """load_raw should return a DataFrame when given a valid CSV."""
        from ames_housing.data.loader import load_raw
        csv = tmp_path / "AmesHousing.csv"
        raw_df.to_csv(csv, index=False)
        # Patch schema validation so synthetic data passes all checks
        with patch("ames_housing.data.loader.RAW_SCHEMA") as mock_schema:
            mock_schema.validate.return_value = raw_df
            result = load_raw(path=csv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(raw_df)

    def test_load_processed_bad_split(self):
        """load_processed should raise ValueError for unknown split names."""
        from ames_housing.data.loader import load_processed
        with pytest.raises(ValueError, match="split must be one of"):
            load_processed(split="banana")

    def test_load_processed_missing_file(self, tmp_path: Path):
        """load_processed should raise FileNotFoundError if parquet not saved yet."""
        from ames_housing.data.loader import load_processed
        with patch("ames_housing.data.loader.settings") as mock_cfg:
            mock_cfg.paths.data_processed = tmp_path
            with pytest.raises(FileNotFoundError, match="not found"):
                load_processed(split="train")


# ── Preprocessor ──────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_drop_id_columns(self, raw_df: pd.DataFrame):
        """PID and Order columns must be removed."""
        from ames_housing.data.preprocessor import _drop_id_columns
        df = raw_df.copy()
        df["PID"] = range(len(df))
        df["Order"] = range(len(df))
        result = _drop_id_columns(df)
        assert "PID" not in result.columns
        assert "Order" not in result.columns

    def test_drop_high_missing_respects_threshold(self, raw_df: pd.DataFrame):
        """Columns above the missing threshold should be dropped."""
        from ames_housing.data.preprocessor import _drop_high_missing
        df = raw_df.copy()
        # Make 'Pool Area' 80% null → above threshold
        df.loc[:int(len(df) * 0.8), "Pool Area"] = np.nan
        result = _drop_high_missing(df, threshold=0.30)
        assert "Pool Area" not in result.columns

    def test_drop_high_missing_keeps_below_threshold(self, raw_df: pd.DataFrame):
        """Columns below the missing threshold should be kept."""
        from ames_housing.data.preprocessor import _drop_high_missing
        df = raw_df.copy()
        # Make 'Lot Frontage' 10% null → below threshold
        df.loc[:int(len(df) * 0.10), "Lot Frontage"] = np.nan
        result = _drop_high_missing(df, threshold=0.30)
        assert "Lot Frontage" in result.columns

    def test_impute_fills_all_nulls(self, raw_df: pd.DataFrame):
        """After imputation, no nulls should remain."""
        from ames_housing.data.preprocessor import _impute
        df = raw_df.copy()
        df.loc[:10, "Lot Frontage"] = np.nan
        df.loc[:10, "Neighborhood"] = np.nan
        result = _impute(df)
        assert result.isnull().sum().sum() == 0

    def test_impute_numeric_uses_median(self, small_df: pd.DataFrame):
        """Numeric imputation should use the column median."""
        from ames_housing.data.preprocessor import _impute
        df = small_df.copy()
        median_val = df["Gr Liv Area"].median()
        df.loc[0, "Gr Liv Area"] = np.nan
        result = _impute(df)
        assert result.loc[0, "Gr Liv Area"] == pytest.approx(median_val)

    def test_remove_outliers_reduces_rows(self, raw_df: pd.DataFrame):
        """Outlier removal should decrease (or equal) the row count."""
        from ames_housing.data.preprocessor import _remove_outliers
        df = raw_df.copy()
        # Inject extreme outliers
        df.loc[0, "SalePrice"] = 5_000_000.0
        df.loc[1, "SalePrice"] = 1.0
        result = _remove_outliers(df)
        assert len(result) < len(df)

    def test_log_transform_reduces_skewness(self, raw_df: pd.DataFrame):
        """Log transformation should reduce skewness of highly skewed columns."""
        from ames_housing.data.preprocessor import _log_transform_skewed
        df = raw_df.copy()
        # Misc Val is heavily right-skewed
        before_skew = abs(df["Misc Val"].skew())
        result, transformed = _log_transform_skewed(df, threshold=0.75)
        after_skew = abs(result["Misc Val"].skew())
        if "Misc Val" in transformed:
            assert after_skew < before_skew

    def test_encode_categoricals_produces_numerics(self, raw_df: pd.DataFrame):
        """After OHE, all columns should be numeric."""
        from ames_housing.data.preprocessor import _encode_categoricals
        result = _encode_categoricals(raw_df.copy())
        cat_cols = result.select_dtypes(include=["object", "category"]).columns
        assert len(cat_cols) == 0

    def test_split_returns_correct_sizes(self, raw_df: pd.DataFrame):
        """Train/val/test split sizes should approximately match config ratios."""
        from ames_housing.data.preprocessor import _split
        result = _split(raw_df, test_size=0.15, val_size=0.15, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = result
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(raw_df)
        # Test set should be ~15% of total
        assert abs(len(X_test) / total - 0.15) < 0.05
