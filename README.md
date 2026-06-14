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

The project uses a DVC pipeline (`dvc.yaml`) for reproducible data processing
and model training. Each stage can also be run independently via the Typer CLI.

1. **Fetch raw data**
   ```bash
   uv run python patent/cli.py data init          # Bootstrap from Kaggle
   uv run python patent/cli.py data update        # Incremental arXiv updates
   ```

2. **Run the DVC pipeline** (recommended)
   ```bash
   dvc repro                  # Run all stages (preprocess → training)
   dvc repro preprocess       # reserialize → clean → embed only
   dvc repro training         # train + evaluate only
   ```
   *(For full details, see [docs/data-pipeline.md](docs/docs/data-pipeline.md)).*

3. **Or run individual CLI steps**
   ```bash
   uv run python patent/cli.py data reserialize data/raw/updates/
   uv run python patent/cli.py data clean data/interim/serialized/updates.parquet
   uv run python patent/cli.py data embed data/interim/cleaned/updates.parquet
   uv run python patent/cli.py model train
   ```

 Use `--help` on any Typer CLI command (`uv run patent/cli.py <command> --help`) to see advanced invocation options.

### Make Commands

| Command                  | Description                                |
| ------------------------ | ------------------------------------------ |
| `make requirements`      | Install/sync Python dependencies with uv   |
| `make test`              | Run tests with pytest                      |
| `make lint`              | Check code style with ruff                 |
| `make format`            | Auto-format source code with ruff          |
| `make typecheck`         | Run static type checking with ty           |
| `make clean`             | Delete compiled Python files and caches    |
| `make create_environment`| Create a new uv virtual environment        |
| `make docker-build`      | Build production inference Docker image    |

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
| `inference-server` | — (internal) | 3 | FastAPI inference with LSHiForest model |
| `inference-lb` | 8000 | 1 | Nginx load balancer → inference-server replicas |
| `prometheus` | 9090 | 1 | Metrics scraper (5s interval) |
| `grafana` | 3000 | 1 | Dashboards + alerting for drift monitoring |

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
model from the MLflow Model Registry at startup.

```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok","model_name":"patent-lshiforest","model_version":"5",...}

# Score texts for anomaly
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["A novel approach to graph neural networks", "Standard survey of existing NLP methods"]}'
# → {"scores":[0.6448,0.6949],"rescaled_scores":[0.0,1.0],"model_version":"5"}
```

For the full API reference (all endpoints, Python example), see
[docs/api.md](docs/api.md).

| Field | Type | Range | Description |
|---|---|---|---|
| `scores` | float | [0, 1] | Raw LSHiForest anomaly score — higher = more anomalous |
| `rescaled_scores` | float | [0, 1] | Percentile-rescaled for interpretability |
| `model_version` | string | — | MLflow model version that produced the scores |

### Monitoring

The stack includes Prometheus and Grafana for drift monitoring:

- **Prometheus** scrapes `/metrics` from the inference server every 5s
- **Grafana** dashboard shows drift gauges, score distribution, throughput, and latency
- **Grafana alert** fires when drift is sustained for 5 minutes, triggering the CT pipeline
- See [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md) for full details

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
├── pipelines          <- DVC pipeline stage entry points (preprocess.py, train.py)
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