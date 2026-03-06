"""Tests for model evaluation, registry (save/load), and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


# ── Metrics ────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    """Unit tests for compute_metrics — verifies formula correctness."""

    def _compute(self, y_true, y_pred, log_transform=False):
        with patch("ames_housing.models.evaluator.settings") as mock_cfg:
            mock_cfg.features.log_transform_target = log_transform
            from ames_housing.models.evaluator import compute_metrics
            return compute_metrics(np.array(y_true), np.array(y_pred))

    def test_perfect_prediction_r2_is_one(self):
        y = np.array([100_000, 200_000, 300_000], dtype=float)
        m = self._compute(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["mae"]  == pytest.approx(0.0, abs=1e-6)

    def test_r2_decreases_with_noise(self):
        rng = np.random.default_rng(0)
        y   = rng.uniform(100_000, 500_000, 100)
        noise = rng.normal(0, 30_000, 100)
        m_clean = self._compute(y, y)
        m_noisy = self._compute(y, y + noise)
        assert m_clean["r2"] > m_noisy["r2"]

    def test_rmse_units_are_dollars(self):
        """RMSE should equal constant error when all predictions are off by same amount."""
        y    = np.array([200_000.0] * 10)
        pred = np.array([210_000.0] * 10)
        m = self._compute(y, pred)
        assert m["rmse"] == pytest.approx(10_000.0)

    def test_mae_lower_bound_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        m = self._compute(y, y)
        assert m["mae"] >= 0.0

    def test_metrics_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        m = self._compute(y, y)
        assert {"r2", "rmse", "mae", "mape"} == set(m.keys())

    def test_log_transform_inverse_applied(self):
        """When log_transform_target=True, predictions should be expm1'd."""
        y_log  = np.log1p(np.array([100_000.0, 200_000.0]))
        pred_log = np.log1p(np.array([100_000.0, 200_000.0]))
        m = self._compute(y_log, pred_log, log_transform=True)
        assert m["r2"] == pytest.approx(1.0)


# ── Registry: save / load ──────────────────────────────────────────────────────

class TestRegistry:
    def test_save_creates_pkl_file(
        self, fitted_full_pipeline: Pipeline, tmp_models_dir: Path
    ):
        from ames_housing.models.registry import save_pipeline

        with patch("ames_housing.models.registry.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            path = save_pipeline(
                pipeline      = fitted_full_pipeline,
                feature_names = ["f1", "f2", "f3"],
                metrics       = {"test": {"r2": 0.9, "rmse": 20000, "mae": 13000, "mape": 7.0}},
                params        = {"n_estimators": 100, "learning_rate": 0.1},
                path          = tmp_models_dir,
            )

        assert path.exists()
        assert path.suffix == ".pkl"

    def test_save_creates_feature_names_json(
        self, fitted_full_pipeline: Pipeline, tmp_models_dir: Path
    ):
        from ames_housing.models.registry import save_pipeline

        feature_names = ["feat_a", "feat_b", "feat_c"]
        with patch("ames_housing.models.registry.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            save_pipeline(
                pipeline      = fitted_full_pipeline,
                feature_names = feature_names,
                metrics       = {},
                params        = {},
                path          = tmp_models_dir,
            )

        fn_path = tmp_models_dir / "feature_names.json"
        assert fn_path.exists()
        loaded = json.loads(fn_path.read_text())
        assert loaded["features"] == feature_names

    def test_save_creates_model_meta_json(
        self, fitted_full_pipeline: Pipeline, tmp_models_dir: Path
    ):
        from ames_housing.models.registry import save_pipeline

        params  = {"n_estimators": 200}
        metrics = {"test": {"r2": 0.94}}

        with patch("ames_housing.models.registry.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            save_pipeline(
                pipeline      = fitted_full_pipeline,
                feature_names = [],
                metrics       = metrics,
                params        = params,
                run_id        = "abc123",
                path          = tmp_models_dir,
            )

        meta = json.loads((tmp_models_dir / "model_meta.json").read_text())
        assert meta["mlflow_run_id"] == "abc123"
        assert meta["best_params"] == params
        assert meta["metrics"] == metrics

    def test_load_pipeline_returns_pipeline(
        self, fitted_full_pipeline: Pipeline, tmp_models_dir: Path
    ):
        from ames_housing.models.registry import load_pipeline, save_pipeline

        with patch("ames_housing.models.registry.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            pkl_path = save_pipeline(
                pipeline=fitted_full_pipeline,
                feature_names=[],
                metrics={},
                params={},
                path=tmp_models_dir,
            )

        loaded = load_pipeline(path=pkl_path)
        assert isinstance(loaded, Pipeline)

    def test_load_pipeline_missing_raises(self, tmp_path: Path):
        from ames_housing.models.registry import load_pipeline
        with pytest.raises(FileNotFoundError, match="No saved model"):
            load_pipeline(path=tmp_path / "nonexistent.pkl")

    def test_saved_pipeline_predicts_same(
        self, fitted_full_pipeline: Pipeline, tmp_models_dir: Path, raw_df: pd.DataFrame
    ):
        """A pipeline saved and reloaded should produce identical predictions."""
        from ames_housing.models.registry import load_pipeline, save_pipeline

        X = raw_df.drop(columns=["SalePrice"]).head(10)
        original_preds = fitted_full_pipeline.predict(X)

        with patch("ames_housing.models.registry.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            pkl_path = save_pipeline(
                pipeline=fitted_full_pipeline,
                feature_names=[],
                metrics={},
                params={},
                path=tmp_models_dir,
            )

        loaded = load_pipeline(path=pkl_path)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)


# ── End-to-end prediction sanity check ────────────────────────────────────────

class TestPredictionSanity:
    def test_predictions_in_realistic_range(
        self, fitted_full_pipeline: Pipeline, raw_df: pd.DataFrame
    ):
        """Model predictions (after expm1) should be in a realistic price range."""
        X = raw_df.drop(columns=["SalePrice"])
        raw_preds = fitted_full_pipeline.predict(X)
        prices = np.expm1(raw_preds)
        assert (prices > 10_000).all(),  "Some predictions are unrealistically low"
        assert (prices < 2_000_000).all(), "Some predictions are unrealistically high"

    def test_higher_qual_predicts_higher_price(
        self, fitted_full_pipeline: Pipeline, raw_df: pd.DataFrame
    ):
        """Higher OverallQual should generally predict higher prices."""
        X = raw_df.drop(columns=["SalePrice"]).copy()
        X_low  = X.copy(); X_low["Overall Qual"]  = 2
        X_high = X.copy(); X_high["Overall Qual"] = 9
        pred_low  = np.expm1(fitted_full_pipeline.predict(X_low)).mean()
        pred_high = np.expm1(fitted_full_pipeline.predict(X_high)).mean()
        assert pred_high > pred_low, "Higher quality should predict higher price"

    def test_larger_area_predicts_higher_price(
        self, fitted_full_pipeline: Pipeline, raw_df: pd.DataFrame
    ):
        """Larger GrLivArea should generally predict higher prices."""
        X = raw_df.drop(columns=["SalePrice"]).copy()
        X_small = X.copy(); X_small["Gr Liv Area"] = 800.0
        X_large = X.copy(); X_large["Gr Liv Area"] = 3500.0
        pred_small = np.expm1(fitted_full_pipeline.predict(X_small)).mean()
        pred_large = np.expm1(fitted_full_pipeline.predict(X_large)).mean()
        assert pred_large > pred_small, "Larger area should predict higher price"
