# Makefile for langchain-healthcare-assistant
# Use `make help` to list targets.

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 5000

.PHONY: help venv install install-dev run dev prod test lint fmt typecheck clean docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create local virtualenv (.venv)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: venv ## Install runtime dependencies
	$(BIN)/pip install -r requirements.txt

install-dev: venv ## Install runtime + dev dependencies
	$(BIN)/pip install -e ".[dev]"

run: ## Run the Flask dev server (defaults to LLM_PROVIDER=none)
	LLM_PROVIDER=$${LLM_PROVIDER:-none} PORT=$(PORT) $(BIN)/python -m app.api

dev: run ## Alias for `run`

prod: ## Run with gunicorn (production)
	$(BIN)/pip install gunicorn >/dev/null
	$(BIN)/gunicorn --bind 0.0.0.0:$(PORT) --workers 2 app.api:app

test: ## Run the test suite
	LLM_PROVIDER=none $(BIN)/pytest -v

lint: ## Lint with ruff
	$(BIN)/ruff check .

fmt: ## Auto-fix with ruff
	$(BIN)/ruff check . --fix
	$(BIN)/ruff format .

typecheck: ## Run mypy
	$(BIN)/mypy app

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build: ## Build the production Docker image
	docker build -t langchain-healthcare-assistant .

docker-run: ## Run the production Docker image
	docker run --rm -p $(PORT):5000 \
		-e LLM_PROVIDER=$${LLM_PROVIDER:-none} \
		-e ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY \
		-e OPENAI_API_KEY=$$OPENAI_API_KEY \
		langchain-healthcare-assistant
