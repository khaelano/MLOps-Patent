# MLOps-Patent

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A paper novelty assessor based on anomaly detection

## Background
Research novelty is one of the important aspects in the development of science; this characteristic serves as a driving force for scientific development by investigating problems that have never been researched before (Zhao & Zhang, 2025). Therefore, evaluating novelty in research is crucial.

One method to assess the novelty of a study is by using keywords. This method works by identifying keyword combinations that are uncommon within an article. If an article contains many keyword combinations that are unusual in its field, then it can be said that the article has a high novelty value (Zhao & Zhang, 2025). However, this method only considers keywords, thereby ignoring other important aspects such as semantics and citation relationships between articles.

Therefore, an anomaly detection-based novelty assessment method has emerged to evaluate the novelty of an article solely through its title. This method works by detecting anomalies (outliers) in a semantic space created using vector representations of existing research titles, where outliers are considered indicative of novelty (Jeon et al., 2023). Thus, the model maps relationships between articles based on the semantic similarity of titles. Through this approach, the model is expected to assess the novelty of a research title reliably and accurately.

## Usage

### Prerequisites

- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/khaelano/MLOps-Patent.git
   cd MLOps-Patent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   make create_environment
   source .venv/bin/activate
   make requirements
   ```

### Pipeline

The project follows a sequential data science pipeline. Each step is a Typer CLI app that can be run independently using `uv run`. For the data pipeline, use the `make` utility.

1. **Process raw data (Integrated Data Pipeline)**
   The integrated data pipeline will pull incremental arXiv text data and generate semantic SentenceTransformer numerical vectors natively:
   ```bash
   # Make update data starting from a specific date natively through Make:
   make data-update FROM=2026-03-01 TO=2026-03-15

   # Preprocess sequentially:
   make data-reserialize INPUT=data/raw/updates/
   make data-clean INPUT=data/interim/serialized/updates.parquet
   make data-embed INPUT=data/interim/cleaned/updates.parquet
   ```
   *(For full details, see the architecture in [docs/data-pipeline.md](docs/docs/data-pipeline.md)).*

2. **Reduce dimensions (Optional)**
   Dramatically shrink the embedding vectors from 384d down to a computationally manageable size natively via iterative processing:
   ```bash
   make data-reduce INPUT=data/processed/updates.parquet
   ```

3. **Train Model (LSHiForest)**
   Dynamically build and train an LSHiForest (Locality-Sensitive Hashing isolation Forest) model over the semantic vectors. The pipeline natively tracks granular metrics (execution time, query latency, peak memory footprint, baseline C(n)) and serializes the memory-mapped artifacts (`.lshif`) using **MLflow**. Models, artifacts, and metrics are automatically tracked for robust life-cycle versioning.
   ```bash
   make model-tune INPUT=data/processed/updates.parquet
   ```

 Use `--help` on any Typer CLI command (`uv run patent/cli.py <command> --help`) to see advanced invocation options.

### Make Commands

| Command                  | Description                                |
| ------------------------ | ------------------------------------------ |
| `make requirements`      | Install/sync Python dependencies with uv   |
| `make data-update FROM=YYYY-MM-DD` | Run incremental data ingestion |
| `make data-reserialize INPUT=<path>`| Run XML/JSON DataFrame conversion |
| `make data-clean INPUT=<path>`| Run text preprocessing logic |
| `make data-embed INPUT=<path>`| Run sequence embedding logic |
| `make data-reduce INPUT=<path>`| Run dimensionality reduction via PCA |
| `make model-tune INPUT=<path>` | Train and serialize LSHiForest model via MLflow |
| `make test`              | Run tests with pytest                      |
| `make lint`              | Check code style with ruff                 |
| `make format`            | Auto-format source code with ruff          |
| `make clean`             | Delete compiled Python files and caches    |
| `make create_environment`| Create a new uv virtual environment        |

### Docker Deployment

The project includes a Docker Compose stack with three services:

```bash
# Required environment variables (MLflow external PostgreSQL + S3)
export MLFLOW_BACKEND_STORE_URI="postgresql://user:pass@host:5432/mlflow"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket/mlflow-artifacts"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export MLFLOW_S3_ENDPOINT_URL="https://s3.your-provider.com"  # if not AWS

# Start the stack (3 inference replicas + load balancer)
docker compose up -d
```

| Service | Port | Replicas | Description |
|---------|------|----------|-------------|
| `mlflow-server` | 5000 | 1 | MLflow tracking server (UI + REST API) |
| `inference-server` | — (internal) | 3 | LSHiForest model served via MLflow pyfunc |
| `inference-lb` | 8000 | 1 | Nginx load balancer → inference-server replicas |

#### Scaling replicas dynamically

Add or remove inference-server replicas at runtime **without downtime**:

```bash
# Scale up to 5 replicas
docker compose up -d --scale inference-server=5

# Scale down to 2 replicas
docker compose up -d --scale inference-server=2
```

The load balancer automatically detects new replicas via Docker's internal DNS.
The embedder ONNX model cache is shared across replicas (`embedder-cache` volume)
so only the first replica downloads the model from HuggingFace.

### Inference API

Once the stack is running, the inference server loads the latest **Production**
model from the MLflow Model Registry (baked into the Docker image at build time).

```bash
# Health check (model liveness)
curl http://localhost:8000/ping
# → 200 (empty body)

# LB health check
curl http://localhost:8000/health
# → OK

# Score texts for anomaly
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"texts": "A novel approach to graph neural networks"},
      {"texts": "Standard survey of existing NLP methods"}
    ]
  }'
# → {"predictions":[
#     {"scores":0.6448,"rescaled_scores":0.0},
#     {"scores":0.6949,"rescaled_scores":1.0}
#   ]}
```

For the full API reference (both request formats, Python example), see
[docs/api.md](docs/api.md).

| Field | Type | Range | Description |
|---|---|---|---|
| `scores` | float | [0, 1] | Raw LSHiForest anomaly score — higher = more anomalous |
| `rescaled_scores` | float | [0, 1] | Percentile-rescaled for interpretability |

Configurable via `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server address |
| `MLFLOW_MODEL_NAME` | `patent-lshiforest` | Registered model name |
| `INFERENCE_PORT` | `8000` | Host port for the inference API |

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── docker-compose.yml <- Docker Compose orchestration for MLflow + inference server
├── docker             <- Dockerfiles and configs (one per service)
│   ├── mlflow         <-   MLflow tracking server (extends official image)
│   └── nginx          <-   Nginx load-balancer config for inference replicas
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- MkDocs-based documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata and tool settings
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── patent             <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes patent a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── api.py                  <- FastAPI inference server
    │
    ├── dataset/                <- Data ingestion and preprocessing
    │   └── embedders.py        <-   Pluggable embedder abstraction
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── lshiforest/             <- LSHiForest anomaly detection model
    │
    ├── modeling/ 
    │   ├── __init__.py 
    │   ├── evaluate.py         <- Model evaluation logic
    │   ├── registry.py         <- MLflow model registry helpers
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## References
Zhao, Y., & Zhang, C. (2025). A review on the novelty measurements of  academic papers. https://doi.org/10.48550/arXiv.2501.17456 