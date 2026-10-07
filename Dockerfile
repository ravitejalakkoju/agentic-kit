FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && useradd --system --home-dir /app app \
    && mkdir /data \
    && chown app:app /data

USER app

EXPOSE 3200

CMD ["python", "-m", "uvicorn", "agentic_kit.main:app", "--host", "0.0.0.0", "--port", "3200"]
