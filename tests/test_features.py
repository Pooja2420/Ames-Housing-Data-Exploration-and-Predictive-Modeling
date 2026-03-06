"""Tests for the feature engineering module: transformers and pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from ames_housing.features.transformers import (
    AmesFeatureEngineer,
    HighMissingDropper,
    RareLabelEncoder,
    SkewnessCorrector,
)
from tests.conftest import make_synthetic_df


# ── AmesFeatureEngineer ────────────────────────────────────────────────────────

class TestAmesFeatureEngineer:
    @pytest.fixture()
    def transformer(self) -> AmesFeatureEngineer:
        return AmesFeatureEngineer()

    @pytest.fixture()
    def df(self) -> pd.DataFrame:
        return make_synthetic_df(50)

    def test_fit_returns_self(self, transformer, df):
        result = transformer.fit(df)
        assert result is transformer

    def test_adds_total_sf(self, transformer, df):
        """TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF."""
        out = transformer.fit_transform(df)
        assert "TotalSF" in out.columns
        expected = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
        pd.testing.assert_series_equal(
            out["TotalSF"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_adds_total_bath(self, transformer, df):
        """TotalBath counts full + 0.5 × half baths."""
        out = transformer.fit_transform(df)
        assert "TotalBath" in out.columns
        expected = (
            df["Full Bath"]
            + 0.5 * df["Half Bath"]
            + df["Bsmt Full Bath"]
            + 0.5 * df["Bsmt Half Bath"]
        )
        pd.testing.assert_series_equal(
            out["TotalBath"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_house_age_non_negative(self, transformer, df):
        """HouseAge must always be ≥ 0."""
        out = transformer.fit_transform(df)
        assert (out["HouseAge"] >= 0).all()

    def test_binary_flags_are_zero_or_one(self, transformer, df):
        """HasPool, HasFireplace, HasGarage, HasBasement must be binary."""
        out = transformer.fit_transform(df)
        for col in ("HasPool", "HasFireplace", "HasGarage", "HasBasement"):
            assert out[col].isin([0.0, 1.0]).all(), f"{col} has non-binary values"

    def test_has_pool_matches_pool_area(self, transformer, df):
        """HasPool should be 1 iff Pool Area > 0."""
        out = transformer.fit_transform(df)
        expected = (df["Pool Area"] > 0).astype(float)
        pd.testing.assert_series_equal(
            out["HasPool"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_missing_columns_graceful(self, transformer):
        """If source columns are absent, engineered features should default to 0."""
        minimal_df = pd.DataFrame({"SalePrice": [150000, 200000]})
        out = transformer.fit_transform(minimal_df)
        assert "TotalSF" in out.columns
        assert (out["TotalSF"] == 0).all()

    def test_does_not_drop_existing_columns(self, transformer, df):
        """The transformer must not remove any existing columns."""
        out = transformer.fit_transform(df)
        for col in df.columns:
            assert col in out.columns

    def test_transform_is_deterministic(self, transformer, df):
        """Calling transform twice should produce identical results."""
        transformer.fit(df)
        out1 = transformer.transform(df)
        out2 = transformer.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


# ── HighMissingDropper ─────────────────────────────────────────────────────────

class TestHighMissingDropper:
    def test_drops_high_missing_column(self):
        df = pd.DataFrame({
            "good_col": [1, 2, 3, 4, 5],
            "bad_col":  [np.nan, np.nan, np.nan, np.nan, 1],  # 80% missing
        })
        dropper = HighMissingDropper(threshold=0.30)
        dropper.fit(df)
        out = dropper.transform(df)
        assert "bad_col" not in out.columns
        assert "good_col" in out.columns

    def test_keeps_low_missing_column(self):
        df = pd.DataFrame({
            "col_a": [1, 2, np.nan, 4, 5],   # 20% missing
        })
        dropper = HighMissingDropper(threshold=0.30)
        dropper.fit(df)
        out = dropper.transform(df)
        assert "col_a" in out.columns

    def test_learned_columns_stored(self):
        df = pd.DataFrame({"high": [np.nan] * 9 + [1], "low": range(10)})
        dropper = HighMissingDropper(threshold=0.5)
        dropper.fit(df)
        assert "high" in dropper.cols_to_drop_
        assert "low" not in dropper.cols_to_drop_

    def test_transform_uses_fit_columns(self):
        """Transform should apply columns learned on fit data, not re-compute."""
        train_df = pd.DataFrame({"drop_me": [np.nan] * 9 + [1], "keep_me": range(10)})
        test_df  = pd.DataFrame({"drop_me": range(10), "keep_me": range(10)})
        dropper  = HighMissingDropper(threshold=0.5).fit(train_df)
        out = dropper.transform(test_df)
        assert "drop_me" not in out.columns


# ── SkewnessCorrector ──────────────────────────────────────────────────────────

class TestSkewnessCorrector:
    def test_reduces_skewness(self):
        """log1p should reduce skewness on a right-skewed column."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"skewed": rng.exponential(scale=10, size=500)})
        corrector = SkewnessCorrector(threshold=0.75).fit(df)
        out = corrector.transform(df)
        assert abs(out["skewed"].skew()) < abs(df["skewed"].skew())

    def test_exclude_columns_not_transformed(self):
        """Excluded columns must not be log-transformed."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "target":  rng.exponential(5, 100),
            "feature": rng.exponential(5, 100),
        })
        corrector = SkewnessCorrector(threshold=0.0, exclude=["target"]).fit(df)
        out = corrector.transform(df)
        pd.testing.assert_series_equal(out["target"], df["target"])

    def test_no_negative_values_after_transform(self):
        """log1p on clipped values must produce no negatives."""
        df = pd.DataFrame({"col": [-100, -1, 0, 1, 100, 10000]})
        corrector = SkewnessCorrector(threshold=0.0).fit(df)
        out = corrector.transform(df)
        assert (out["col"] >= 0).all()

    def test_applies_same_columns_on_unseen_data(self):
        """Columns learned during fit must be applied consistently to new data."""
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"a": rng.exponential(5, 100), "b": rng.normal(0, 1, 100)})
        test  = pd.DataFrame({"a": rng.exponential(5, 20),  "b": rng.normal(0, 1, 20)})
        corrector = SkewnessCorrector(threshold=0.75).fit(train)
        out = corrector.transform(test)
        assert out.shape == test.shape


# ── RareLabelEncoder ───────────────────────────────────────────────────────────

class TestRareLabelEncoder:
    def test_replaces_rare_categories(self):
        """Categories below the threshold should become 'Other'."""
        df = pd.DataFrame({
            "city": ["NYC"] * 90 + ["LA"] * 8 + ["Tiny Town"] * 2
        })
        enc = RareLabelEncoder(threshold=0.05).fit(df)
        out = enc.transform(df)
        assert "Tiny Town" not in out["city"].values
        assert "Other" in out["city"].values

    def test_keeps_frequent_categories(self):
        """Frequent categories must be preserved."""
        df = pd.DataFrame({"city": ["NYC"] * 90 + ["LA"] * 10})
        enc = RareLabelEncoder(threshold=0.05).fit(df)
        out = enc.transform(df)
        assert "NYC" in out["city"].values
        assert "LA"  in out["city"].values

    def test_unseen_labels_become_other(self):
        """Labels not seen during fit must map to 'Other' on transform."""
        train = pd.DataFrame({"col": ["A"] * 80 + ["B"] * 20})
        test  = pd.DataFrame({"col": ["A", "B", "C", "UNSEEN"]})
        enc = RareLabelEncoder(threshold=0.05).fit(train)
        out = enc.transform(test)
        assert out.loc[2, "col"] == "Other"
        assert out.loc[3, "col"] == "Other"


# ── Full feature pipeline ──────────────────────────────────────────────────────

class TestBuildPipeline:
    def test_returns_pipeline(self):
        from ames_housing.features.engineering import build_pipeline
        p = build_pipeline()
        assert isinstance(p, Pipeline)

    def test_fit_transform_produces_numeric_array(self, raw_df):
        from ames_housing.features.engineering import build_pipeline
        X = raw_df.drop(columns=["SalePrice"])
        pipeline = build_pipeline(scale=False)
        out = pipeline.fit_transform(X)
        assert out.dtype in (np.float32, np.float64)
        assert np.isfinite(out).all()

    def test_transform_same_shape_as_fit(self, raw_df):
        """Transform on new data must produce the same column count as fit."""
        from ames_housing.features.engineering import build_pipeline
        X = raw_df.drop(columns=["SalePrice"])
        split = len(X) // 2
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        pipeline = build_pipeline(scale=False)
        train_out = pipeline.fit_transform(X_train)
        test_out  = pipeline.transform(X_test)
        assert train_out.shape[1] == test_out.shape[1]

    def test_no_nan_in_output(self, raw_df):
        """Pipeline output must not contain any NaN values."""
        from ames_housing.features.engineering import build_pipeline
        X = raw_df.drop(columns=["SalePrice"])
        out = build_pipeline(scale=False).fit_transform(X)
        assert not np.isnan(out).any()

    def test_scale_false_vs_true_shape(self, raw_df):
        """Scaled and unscaled pipelines should output the same number of columns."""
        from ames_housing.features.engineering import build_pipeline
        X = raw_df.drop(columns=["SalePrice"])
        out_no_scale = build_pipeline(scale=False).fit_transform(X)
        out_scaled   = build_pipeline(scale=True).fit_transform(X)
        assert out_no_scale.shape[1] == out_scaled.shape[1]
