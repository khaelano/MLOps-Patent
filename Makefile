#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLOps-Patent
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Type-check the project with ty (Astral's type checker — never mypy)
.PHONY: typecheck
typecheck:
	ty check



## Run tests (excludes integration tests needing external services)
.PHONY: test
test:
	python -m pytest tests --ignore=tests/test_data_ingestion.py --ignore=tests/test_registry.py


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Initialize raw dataset from Kaggle snapshot (run once)
.PHONY: data-init
data-init: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py data init

## Update raw dataset incrementally via OAI-PMH
.PHONY: data-update
data-update: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py data update $(if $(FROM),--from-date $(FROM)) $(if $(TO),--to-date $(TO))

## Reserialize XML/JSON data to Parquet
.PHONY: data-reserialize
data-reserialize: requirements
	@if [ -z "$(INPUT)" ]; then echo "Error: INPUT is not set. Use make data-reserialize INPUT=<path>"; exit 1; fi
	uv run $(PYTHON_INTERPRETER) patent/cli.py data reserialize $(INPUT) $(if $(IS_JSON),--json)

## Clean reserialized data
.PHONY: data-clean
data-clean: requirements
	@if [ -z "$(INPUT)" ]; then echo "Error: INPUT is not set. Use make data-clean INPUT=<path>"; exit 1; fi
	uv run $(PYTHON_INTERPRETER) patent/cli.py data clean $(INPUT)

## Embed cleaned data
.PHONY: data-embed
data-embed: requirements
	@if [ -z "$(INPUT)" ]; then echo "Error: INPUT is not set. Use make data-embed INPUT=<path>"; exit 1; fi
	uv run $(PYTHON_INTERPRETER) patent/cli.py data embed $(INPUT)

## Reduce embedding dimensionality via Temporal Incremental PCA
.PHONY: data-reduce
data-reduce: requirements
	@if [ -z "$(INPUT)" ]; then echo "Error: INPUT is not set. Use make data-reduce INPUT=<path>"; exit 1; fi
	uv run $(PYTHON_INTERPRETER) patent/cli.py data reduce $(INPUT)

## Train LSHiForest model on processed embeddings (default: data/processed/ -> models/)
.PHONY: model-train
model-train: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py model train $(DATA) $(OUTPUT)

## Evaluate model stability (default: models/model.lshif + data/processed/)
.PHONY: model-evaluate
model-evaluate: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py model evaluate $(MODEL) $(DATA) $(OUTPUT)

## Build Docker image from production MLflow model (requires .env for registry access)
.PHONY: docker-build
docker-build:
	@test -f .env || { echo "Error: .env file not found."; exit 1; }
	set -a && . ./.env && set +a && \
		uv run python scripts/download_model.py -o docker/app/model && \
		docker build \
			-t ghcr.io/khaelano/mlops-patent:latest \
			-f docker/app/Dockerfile \
			.

## Push the built Docker image to GHCR (requires `docker login ghcr.io`)
.PHONY: docker-push
docker-push:
	docker push ghcr.io/khaelano/mlops-patent:latest

## Build and push: docker-build → docker-push
.PHONY: docker-release
docker-release: docker-build docker-push

## Run the full pipeline: reserialize -> clean -> embed -> reduce -> train -> evaluate
.PHONY: pipeline
pipeline: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py pipeline $(if $(RAW),--raw $(RAW)) $(if $(SKIP_INIT),--skip-init) $(if $(FORCE),--force)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
