FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Bakes chromadb's ~80MB default embedding model into the image at build
# time instead of downloading it on the first live request — Render's free
# tier spins the container down when idle, so without this every wake-up
# would re-pay that download inside a user-facing request.
#
# Pre-fetches the archive with a plain urlretrieve first: chromadb's own
# httpx-based downloader uses a short default timeout that a slow or
# jittery connection to a large file can miss, even though the transfer
# is still progressing. Creating the collection alone doesn't trigger the
# download at all; an actual embedding call (.add here) does.
COPY docker/fetch_embedding_model.py ./
RUN python fetch_embedding_model.py && rm fetch_embedding_model.py
RUN python -c "\
import chromadb; \
c = chromadb.Client().get_or_create_collection('warmup', metadata={'hnsw:space': 'cosine'}); \
c.add(ids=['1'], documents=['warmup'])"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Shell form so ${PORT} expands: Render assigns its own port at runtime,
# defaulting to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
