FROM python:3.10-slim

WORKDIR /app

COPY scripts/alertmanager_webhook.py /app/

ENV WEBHOOK_HOST=0.0.0.0
ENV WEBHOOK_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "/app/alertmanager_webhook.py"]
