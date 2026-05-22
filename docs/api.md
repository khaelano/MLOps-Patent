# Inference API

Served on **port 8000** through an nginx load balancer in front of 3 LSHiForest replicas.

## Endpoints

### `GET /ping`

Model liveness check.  Returns **200** with empty body when the model and
embedder are loaded.

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ping
# → 200
```

### `GET /health`

Load balancer liveness check.  Returns `OK`.

```bash
curl -s http://localhost:8000/health
# → OK
```

### `POST /invocations`

Score texts for anomaly.  Each text should be a **title + abstract**
concatenation (the same format used during training).

---

#### Request

Two JSON formats are accepted.

**`dataframe_records`** (list of objects):

```json
{
  "dataframe_records": [
    {"texts": "A novel deep learning approach to anomaly detection in documents"},
    {"texts": "Standard survey of existing classification methods"}
  ]
}
```

**`dataframe_split`** (column-oriented, better for large batches):

```json
{
  "dataframe_split": {
    "columns": ["texts"],
    "data": [
      ["A novel deep learning approach to anomaly detection in documents"],
      ["Standard survey of existing classification methods"]
    ]
  }
}
```

#### Response

```json
{
  "predictions": [
    {"scores": 0.6448, "rescaled_scores": 0.0},
    {"scores": 0.6949, "rescaled_scores": 1.0}
  ]
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `scores` | float | [0, 1] | Raw LSHiForest anomaly score. Higher = more anomalous. |
| `rescaled_scores` | float | [0, 1] | Percentile-rescaled score. 1.0 = most anomalous in the batch. |

---

## curl examples

### Health check

```bash
curl -s http://localhost:8000/ping && echo "model OK"
curl -s http://localhost:8000/health
```

### Single text

```bash
curl -s -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"texts": "A novel method for anomaly detection using isolation forests"}
    ]
  }' | python3 -m json.tool
```

### Multiple texts

```bash
curl -s -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"texts": "Quantum computing approach to solve NP-complete problems"},
      {"texts": "A survey of existing methods for text classification"},
      {"texts": "Revolutionary battery technology enables 1000x energy density"}
    ]
  }' | python3 -m json.tool
```

### Large batches (compact format)

```bash
curl -s -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["texts"],
      "data": [
        ["Quantum computing breakthrough in error correction"],
        ["Review of recent advances in natural language processing"],
        ["Graphene-based room-temperature superconductor discovered"],
        ["Standard benchmark results for image classification"],
        ["CRISPR gene editing cures genetic disease in human trials"]
      ]
    }
  }' | python3 -m json.tool
```

### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/invocations",
    json={
        "dataframe_records": [
            {"texts": "Novel AI architecture achieves human-level reasoning"},
            {"texts": "Comparative study of existing optimization algorithms"},
        ]
    },
)
print(resp.json())
# {
#   "predictions": [
#     {"scores": 0.72, "rescaled_scores": 1.0},
#     {"scores": 0.35, "rescaled_scores": 0.0}
#   ]
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
replicas.  Each replica runs a self-contained MLflow pyfunc model with an
embedded ONNX text embedder (`AllMiniLML6V2Q`, 384-dimensional).
