mlflow server \
    --backend-store-uri "sqlite:///.mlflow/backend.db" \
    --registry-store-uri "sqlite:///.mlflow/registry.db" \
    --default-artifact-root ".mlflow/mlruns/" \
    --artifacts-destination ".mlflow/mlartifacts/" \
    --port 5000
