FROM python:3.10-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY patent/ ./patent/

RUN uv venv /app/.venv \
    && uv sync --frozen --no-dev --python /app/.venv/bin/python

FROM python:3.10-slim AS inference
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

COPY --from=builder /app/.venv /app/.venv
COPY patent/ ./patent/
COPY README.md LICENSE ./

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "patent.api:app", "--host", "0.0.0.0", "--port", "8000"]
