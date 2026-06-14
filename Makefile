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
	uv run pytest tests --ignore=tests/test_data_ingestion.py --ignore=tests/test_registry.py --ignore=tests/test_mlflow_integration.py


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

## Train LSHiForest model on processed embeddings (default: data/processed/ -> models/)
.PHONY: model-train
model-train: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py model train $(DATA) $(OUTPUT)

## Evaluate model stability (default: models/model.lshif + data/processed/)
.PHONY: model-evaluate
model-evaluate: requirements
	uv run $(PYTHON_INTERPRETER) patent/cli.py model evaluate $(MODEL) $(DATA) $(OUTPUT)

## Download model + baseline from MLflow, then build self-contained image.
## Requires .env with MLFLOW_TRACKING_URI and AWS credentials.
## Override model: make docker-build MODEL_STAGE=Staging  or  MODEL_VERSION=5
.PHONY: docker-build
docker-build:
	@test -f .env || { echo "Error: .env file not found."; exit 1; }
	$(eval MODEL_NAME ?= patent-lshiforest)
	$(eval MODEL_STAGE ?= Production)
	rm -rf model && mkdir model && \
		export MLFLOW_TRACKING_URI=$$(grep '^MLFLOW_TRACKING_URI=' .env | head -1 | cut -d= -f2-) && \
		export AWS_ACCESS_KEY_ID=$$(grep '^AWS_ACCESS_KEY_ID=' .env | head -1 | cut -d= -f2-) && \
		export AWS_SECRET_ACCESS_KEY=$$(grep '^AWS_SECRET_ACCESS_KEY=' .env | head -1 | cut -d= -f2-) && \
		export MLFLOW_S3_ENDPOINT_URL=$$(grep '^MLFLOW_S3_ENDPOINT_URL=' .env | head -1 | cut -d= -f2-) && \
		if [ -n "$(MODEL_VERSION)" ]; then \
			VERSION="$(MODEL_VERSION)"; \
		else \
			VERSION=$$(uv run python -c "from mlflow.tracking import MlflowClient; print(MlflowClient().get_latest_versions('$(MODEL_NAME)', stages=['$(MODEL_STAGE)'])[0].version)"); \
		fi && \
		echo "Downloading $(MODEL_NAME) v$$VERSION ..." && \
		uv run python docker/app/download_model.py \
			--model-name $(MODEL_NAME) \
			--version $$VERSION \
			--output-dir model && \
		echo "Building image for $(MODEL_NAME) v$$VERSION ..." && \
		docker build \
			-f docker/app/Dockerfile \
			--build-arg MODEL_SOURCE=./model \
			-t "mlops-patent:model-v$$VERSION" \
			-t "mlops-patent:latest" \
			. && \
		rm -rf model && \
		echo "Built mlops-patent:model-v$$VERSION"

## Push the built Docker image to GHCR (requires `docker login ghcr.io`)
.PHONY: docker-push
docker-push:
	docker push ghcr.io/khaelano/mlops-patent:latest
	docker push ghcr.io/khaelano/mlops-patent:model-v$(MODEL_VERSION)

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
