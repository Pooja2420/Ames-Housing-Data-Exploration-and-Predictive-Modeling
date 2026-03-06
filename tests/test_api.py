"""Integration tests for the FastAPI prediction API.

Uses FastAPI's TestClient (synchronous) with the fitted pipeline injected
directly into app.state — no disk I/O required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, api_client: TestClient):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, api_client: TestClient):
        body = api_client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_model_loaded_true(self, api_client: TestClient):
        """Model should be loaded because we injected the pipeline in fixture."""
        body = api_client.get("/health").json()
        assert body["model_loaded"] is True

    def test_health_has_version(self, api_client: TestClient):
        body = api_client.get("/health").json()
        assert "version" in body
        assert body["version"] != ""

    def test_health_has_timestamp(self, api_client: TestClient):
        body = api_client.get("/health").json()
        assert "timestamp" in body

    def test_health_model_not_loaded(self):
        """When no pipeline is loaded, model_loaded should be False."""
        from ames_housing.api.main import create_app
        app = create_app()
        app.state.pipeline = None
        with TestClient(app, raise_server_exceptions=True) as client:
            body = client.get("/health").json()
        assert body["model_loaded"] is False


# ── Root endpoint ──────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, api_client: TestClient):
        response = api_client.get("/")
        assert response.status_code == 200

    def test_root_contains_docs_link(self, api_client: TestClient):
        body = api_client.get("/").json()
        assert "docs" in body


# ── /predict endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        response = api_client.post("/predict", json=valid_prediction_payload)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        body = api_client.post("/predict", json=valid_prediction_payload).json()
        for field in ("predicted_price", "lower_bound", "upper_bound",
                      "model_version", "prediction_id", "timestamp"):
            assert field in body, f"Missing field: {field}"

    def test_predict_price_is_positive(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        body = api_client.post("/predict", json=valid_prediction_payload).json()
        assert body["predicted_price"] > 0

    def test_predict_confidence_interval_ordered(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """lower_bound < predicted_price < upper_bound."""
        body = api_client.post("/predict", json=valid_prediction_payload).json()
        assert body["lower_bound"] < body["predicted_price"] < body["upper_bound"]

    def test_predict_unique_ids_per_request(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """Each prediction should get a unique ID."""
        id1 = api_client.post("/predict", json=valid_prediction_payload).json()["prediction_id"]
        id2 = api_client.post("/predict", json=valid_prediction_payload).json()["prediction_id"]
        assert id1 != id2

    def test_predict_missing_required_field_returns_422(self, api_client: TestClient):
        """Omitting a required field should return HTTP 422."""
        payload = {"GrLivArea": 1710, "YearBuilt": 2003}   # missing OverallQual
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_overall_qual_returns_422(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """OverallQual outside [1, 10] should return HTTP 422."""
        payload = {**valid_prediction_payload, "OverallQual": 99}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_year_built_returns_422(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """YearBuilt in the distant future should return HTTP 422."""
        payload = {**valid_prediction_payload, "YearBuilt": 2999}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_living_area_returns_422(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """Negative GrLivArea must be rejected."""
        payload = {**valid_prediction_payload, "GrLivArea": -100}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_higher_quality_gives_higher_price(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """OverallQual=9 should predict a higher price than OverallQual=3."""
        low_qual  = {**valid_prediction_payload, "OverallQual": 3}
        high_qual = {**valid_prediction_payload, "OverallQual": 9}
        price_low  = api_client.post("/predict", json=low_qual).json()["predicted_price"]
        price_high = api_client.post("/predict", json=high_qual).json()["predicted_price"]
        assert price_high > price_low

    def test_no_model_returns_503(self, valid_prediction_payload: dict):
        """If no model is loaded, /predict should return 503."""
        from ames_housing.api.main import create_app
        app = create_app()
        app.state.pipeline = None
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict", json=valid_prediction_payload)
        assert response.status_code == 503


# ── /predict/batch endpoint ────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_returns_list(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        payload = [valid_prediction_payload, valid_prediction_payload]
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2

    def test_batch_each_has_prediction_id(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        payload = [valid_prediction_payload] * 3
        body = api_client.post("/predict/batch", json=payload).json()
        ids = [item["prediction_id"] for item in body]
        assert len(set(ids)) == 3, "Each batch item must have a unique ID"

    def test_batch_exceeds_limit_returns_422(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        """Batch of > 100 items should be rejected with 422."""
        payload = [valid_prediction_payload] * 101
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_single_item(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        response = api_client.post("/predict/batch", json=[valid_prediction_payload])
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_batch_prices_are_positive(
        self, api_client: TestClient, valid_prediction_payload: dict
    ):
        payload = [valid_prediction_payload] * 5
        body = api_client.post("/predict/batch", json=payload).json()
        assert all(item["predicted_price"] > 0 for item in body)


# ── Schema validation ──────────────────────────────────────────────────────────

class TestPredictionRequestSchema:
    def test_year_remod_defaults_to_year_built(self):
        """YearRemodAdd should default to YearBuilt when not provided."""
        from ames_housing.api.schemas import PredictionRequest
        req = PredictionRequest(OverallQual=7, GrLivArea=1500, YearBuilt=2000)
        assert req.YearRemodAdd == 2000

    def test_first_floor_defaults_to_gr_liv_area(self):
        """1stFlrSF should default to GrLivArea when not provided."""
        from ames_housing.api.schemas import PredictionRequest
        req = PredictionRequest(OverallQual=7, GrLivArea=1800, YearBuilt=2000)
        assert req.FirstFlrSF == pytest.approx(1800)

    def test_to_dataframe_returns_single_row(self):
        """to_dataframe() should always return exactly one row."""
        from ames_housing.api.schemas import PredictionRequest
        req = PredictionRequest(OverallQual=7, GrLivArea=1500, YearBuilt=2000)
        df = req.to_dataframe()
        assert len(df) == 1

    def test_to_dataframe_uses_ames_column_names(self):
        """Column names in the DataFrame should match Ames dataset conventions."""
        from ames_housing.api.schemas import PredictionRequest
        req = PredictionRequest(OverallQual=7, GrLivArea=1500, YearBuilt=2000)
        df = req.to_dataframe()
        assert "Overall Qual" in df.columns
        assert "Gr Liv Area" in df.columns
        assert "Year Built" in df.columns
