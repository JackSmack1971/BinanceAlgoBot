# syntax=docker/dockerfile:1

FROM python:3.11-slim as base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip

FROM base as builder
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

FROM base as audit
RUN pip install pip-audit && pip-audit -r requirements.txt

FROM base as runtime
COPY --from=builder /usr/local /usr/local
COPY . .
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser
EXPOSE 8000 8080
HEALTHCHECK CMD curl -f http://localhost:8000/healthz || exit 1
CMD ["python", "-m", "src.api.run"]
