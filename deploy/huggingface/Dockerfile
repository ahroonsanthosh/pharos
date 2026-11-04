# Pharos as a public Hugging Face Space (Docker SDK). The deploy workflow copies this
# next to the source; Spaces run the container as user 1000 and route port 7860.
FROM python:3.12-slim
LABEL org.opencontainers.image.title="Pharos" \
      org.opencontainers.image.description="GPS-free drone navigation from video and free maps" \
      org.opencontainers.image.source="https://github.com/ahroonsanthosh/pharos" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.authors="Ahroon Santhosh"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server,geo]"

USER user
ENV PHAROS_PUBLIC=1 PHAROS_CACHE=/home/user/.cache/pharos
EXPOSE 7860
CMD ["pharos", "serve", "--host", "0.0.0.0", "--port", "7860", "--data", "/home/user/data"]
