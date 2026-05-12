mlflow server \
    --backend-store-uri "sqlite:///.project_temp/mlflow/backend.db" \
    --registry-store-uri "sqlite:///.project_temp/mlflow/registry.db" \
    --default-artifact-root ".project_temp/mlflow/mlruns/" \
    --artifacts-destination ".project_temp/mlflow/mlartifacts/" \
    --port 5000
