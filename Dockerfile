FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["sh", "-c", ".venv/bin/uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
