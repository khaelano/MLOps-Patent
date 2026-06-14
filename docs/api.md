# Inference API

Served on **port 8000** through an nginx load balancer in front of 3 LSHiForest replicas.

## Endpoints

### `GET /health`

Liveness / readiness probe.  Returns model version, model name, and embedder
metadata.

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# → {"status": "ok", "model_name": "patent-lshiforest", "model_version": "5", "embedder": "embed-anything-onnx:AllMiniLML6V2Q"}
```

### `GET /metrics`

Prometheus metrics endpoint.  Exposes drift detection gauges, score distribution
histogram, and model info for Grafana dashboards.

```bash
curl -s http://localhost:8000/metrics | head -20
```

### `POST /predict`

Score texts for anomaly.  Each text should be a **paper title** (the same
format used during training).

---

#### Request

Single JSON format — a list of raw text strings:

```json
{
  "texts": [
    "A novel deep learning approach to anomaly detection in documents",
    "Standard survey of existing classification methods"
  ]
}
```

#### Response

```json
{
  "scores": [0.6448, 0.6949],
  "rescaled_scores": [0.0, 1.0],
  "model_version": "5"
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `scores` | float[] | [0, 1] | Raw LSHiForest anomaly scores. Higher = more anomalous. |
| `rescaled_scores` | float[] | [0, 1] | Percentile-rescaled scores. 1.0 = most anomalous in the batch. |
| `model_version` | string | — | MLflow model version that produced the scores. |

---

### `POST /dispatch`

Internal relay endpoint that receives the Grafana drift alert webhook and
forwards it as a `repository_dispatch` event to the GitHub Actions API.
Requires `GITHUB_DISPATCH_URL` and `GITHUB_DISPATCH_TOKEN` environment
variables.

The Grafana contact point sends `POST http://inference-lb/dispatch` when the
drift alert fires.  This endpoint transforms the request into a properly
formatted `repository_dispatch` call with event type `drift-detected`,
triggering the Model CT workflow.

Returns the GitHub API response status for debugging.

---

## curl examples

### Health check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Single text

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["A novel method for anomaly detection using isolation forests"]}' \
  | python3 -m json.tool
```

### Multiple texts

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Quantum computing approach to solve NP-complete problems",
      "A survey of existing methods for text classification",
      "Revolutionary battery technology enables 1000x energy density"
    ]
  }' | python3 -m json.tool
```

### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    json={
        "texts": [
            "Novel AI architecture achieves human-level reasoning",
            "Comparative study of existing optimization algorithms",
        ]
    },
)
print(resp.json())
# {
#   "scores": [0.72, 0.35],
#   "rescaled_scores": [1.0, 0.0],
#   "model_version": "5"
# }
```

## Architecture

```
client → :8000 → nginx (inference-lb)
                    ├─ replica-1 (inference-server :8000)
                    ├─ replica-2 (inference-server :8000)
                    └─ replica-3 (inference-server :8000)
```

The load balancer uses Docker's internal DNS to distribute requests across
replicas.  Each replica runs a FastAPI application that loads the Production
model from the MLflow Model Registry and uses an ONNX embedder
(`AllMiniLML6V2Q`, 384-dimensional) to vectorise input text before
scoring.
