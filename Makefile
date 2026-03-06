.PHONY: install install-dev lint format test train serve mlflow clean all

PYTHON := python
UV := uv

install:
	$(UV) venv
	$(UV) pip install -e .

install-dev:
	$(UV) venv
	$(UV) pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/

test:
	pytest tests/ -v --cov=src/ames_housing --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

train:
	$(PYTHON) -m ames_housing.models.trainer

serve:
	uvicorn ames_housing.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf dist/ build/ htmlcov/ .coverage

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

all: lint test
