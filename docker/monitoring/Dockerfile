FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps . 2>/dev/null || true
RUN pip install --no-cache-dir loguru requests pyyaml

COPY patent/ patent/
COPY config/alertmanager/webhook_bridge.py .

ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=9099
EXPOSE 9099

CMD ["python", "webhook_bridge.py"]
