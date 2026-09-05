FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Bakes chromadb's ~80MB default embedding model into the image at build
# time instead of downloading it on the first live request — Render's free
# tier spins the container down when idle, so without this every wake-up
# would re-pay that download inside a user-facing request.
RUN python -c "import chromadb; chromadb.Client().get_or_create_collection('warmup', metadata={'hnsw:space': 'cosine'})"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Shell form so ${PORT} expands: Render assigns its own port at runtime,
# defaulting to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
