FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,geo]"

EXPOSE 8000
VOLUME /data
CMD ["pharos", "serve", "--host", "0.0.0.0", "--data", "/data"]
