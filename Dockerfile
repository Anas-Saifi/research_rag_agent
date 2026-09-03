FROM python

WORKDIR /myapp

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . .

EXPOSE 8080

CMD [".venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]