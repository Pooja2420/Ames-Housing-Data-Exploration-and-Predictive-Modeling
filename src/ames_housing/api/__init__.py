"""FastAPI serving layer — prediction API for the Ames Housing model."""

from ames_housing.api.main import app, create_app

__all__ = ["app", "create_app"]
