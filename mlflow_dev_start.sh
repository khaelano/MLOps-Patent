mlflow server --backend-store-uri "${MLFLOW_DB_URI}" --default-artifact-root "s3://${MLFLOW_S3_BUCKET}/artifacts/" --host 127.0.0.1 --port 5000
