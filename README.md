# RAGSearch Engine

A from-scratch approximate nearest neighbor search engine with three indexing strategies, built on FastAPI for RAG retrieval workloads.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/Tests-150%2B-passing-brightgreen.svg)

---

## What this is

A vector search engine that implements three index structures from scratch — Naive (brute-force), LSH (locality-sensitive hashing), and VP-Tree (vantage-point tree) — each with different trade-offs in latency, recall, and memory. The system exposes a REST API for document ingestion, chunking, embedding (via Cohere), and similarity search, organized around a domain-driven architecture with dependency injection, full type checking, and 150+ tests across unit, integration, and semantic quality layers.

### Index strategies

| Index | Algorithm | Search complexity | Results | Use case |
|-------|-----------|-------------------|---------|----------|
| **Naive** | Exhaustive linear scan | O(n * d) | Exact | Baseline, small corpora (<1K vectors) |
| **LSH** | Multi-table random hyperplane hashing | O(d * L * k + candidates) | Approximate | Large corpora, latency-sensitive retrieval |
| **VP-Tree** | Binary metric tree with triangle inequality pruning | O(log n * d) expected | Exact | Medium corpora, exact results with sublinear search |

Each library can be configured with a different index type and tunable parameters (e.g., `num_tables`/`num_hyperplanes` for LSH, `leaf_size` for VP-Tree). Indexes are library-scoped, thread-safe (`RLock`), and use deterministic seeding for reproducibility.

---

## Architecture

The project follows domain-driven design with strict layer separation and dependency injection throughout.

```
src/vector_db/
├── api/             # FastAPI routers, Pydantic schemas, DI wiring
├── application/     # Service layer — orchestrates indexing, search, CRUD
├── domain/          # Models (Library, Document, Chunk, Metadata), interfaces
└── infrastructure/  # Index implementations, repositories, Cohere client, logging
    └── indexes/     # Naive, LSH, VP-Tree (all from scratch)
```

**Design decisions:**
- Repository pattern with in-memory stores behind abstract interfaces — swappable to any persistence backend without touching business logic
- Immutable domain models with `.model_copy()` for updates
- Cosine similarity with L2 normalization and edge-case handling (zero vectors, NaN/inf clamping)
- Structured logging via `structlog` with JSON output for production and console output for development
- Library-scoped indexes: each library maintains its own index instance, enabling per-library index type selection and clean isolation

---

## Engineering practices

- **Testing**: 150+ tests across three layers — unit (mocked dependencies), integration (component interactions), and semantic quality (real Cohere API calls validating search relevance). Coverage at 80%+.
- **Type safety**: Full type annotations validated by mypy with strict configuration.
- **Linting and formatting**: Ruff for both, enforced via pre-commit hooks alongside mypy.
- **Concurrency**: Thread-safe index operations and repository access via reentrant locks.
- **Containerization**: Docker image on Python 3.11-slim with Poetry-based dependency resolution.

---

## Roadmap

Features not yet implemented that would strengthen this as a production retrieval system:

- **Benchmarks**: Latency vs. recall vs. dataset size comparison across the three index types. The implementations exist — the quantitative evaluation does not yet.
- **Hybrid search**: Combining BM25 lexical matching with vector similarity for better recall on keyword-heavy queries.
- **Reranking**: A second-stage reranker (e.g., cross-encoder) to refine candidate sets returned by the ANN index.

---

## Quick start

### Prerequisites

- Python 3.11+
- Poetry
- Cohere API key ([cohere.com](https://cohere.com/))
- Docker (optional)

### Installation

```bash
git clone https://github.com/mbenavente/ragsearch-engine
cd ragsearch-engine
poetry install
```

### Configuration

```bash
cp env.example .env
# Add your COHERE_API_KEY to .env
```

### Run

```bash
poetry run uvicorn src.vector_db.api.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `/docs`.

### Docker

```bash
docker build -f docker/Dockerfile -t ragsearch-engine .
docker run -p 8000:8000 --env-file .env ragsearch-engine
```

### Production notes

- Set `CORS_ORIGINS` in `.env` to restrict allowed origins (defaults to `*` for development).
- The API does not include authentication — add middleware for production use.
- Use HTTPS to protect API keys and data in transit.

---

## API

Full OpenAPI documentation is available at `/docs` and `/redoc` when the server is running.

### Libraries

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/libraries/` | Create a library (specify index type and params) |
| GET | `/api/v1/libraries/` | List all libraries |
| GET | `/api/v1/libraries/{id}` | Get library details |
| PUT | `/api/v1/libraries/{id}` | Update library |
| DELETE | `/api/v1/libraries/{id}` | Delete library |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/libraries/{id}/documents/` | Ingest a document (auto-chunks and embeds) |
| GET | `/api/v1/libraries/{id}/documents/` | List documents |
| GET | `/api/v1/libraries/{id}/documents/{doc_id}` | Get document |
| PUT | `/api/v1/libraries/{id}/documents/{doc_id}` | Update document |
| DELETE | `/api/v1/libraries/{id}/documents/{doc_id}` | Delete document |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/libraries/{id}/search` | Search across all chunks in a library |
| POST | `/api/v1/libraries/{id}/documents/{doc_id}/search` | Search within a specific document |

**Request:**
```json
{
  "query_text": "machine learning algorithms",
  "k": 10,
  "min_similarity": 0.5
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk": {
        "id": "chunk-uuid",
        "document_id": "doc-uuid",
        "text": "chunk content...",
        "metadata": { "creation_time": "...", "tags": ["tag1"] }
      },
      "similarity_score": 0.95
    }
  ],
  "total_chunks_searched": 42,
  "query_time_ms": 15.3
}
```

---

## Example usage

```bash
# Create a library with LSH index
curl -X POST "http://localhost:8000/api/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Papers",
    "username": "researcher",
    "index_type": "lsh",
    "index_params": { "num_tables": 12, "num_hyperplanes": 10 }
  }'

# Ingest a document
curl -X POST "http://localhost:8000/api/v1/libraries/${LIBRARY_ID}/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Document content to index...",
    "chunk_size": 500
  }'

# Search
curl -X POST "http://localhost:8000/api/v1/libraries/${LIBRARY_ID}/search" \
  -H "Content-Type: application/json" \
  -d '{ "query_text": "neural networks", "k": 5 }'
```

### Demo data

```bash
poetry run python scripts/create_demo_data.py --index-type vptree
poetry run python scripts/create_demo_data.py --index-type lsh --index-params '{"num_tables": 4, "num_hyperplanes": 2}'
```

---

## Development

### Tests

```bash
# Fast tests (unit + integration)
poetry run pytest -m "not semantic_quality"

# Full suite (requires COHERE_API_KEY)
poetry run pytest

# By layer
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest -m semantic_quality

# With coverage
poetry run pytest tests/unit/ --cov=src/vector_db --cov-report=html
```

See [tests/README.md](tests/README.md) for the full testing strategy.

### Code quality

```bash
poetry run ruff format src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/
```

Pre-commit hooks run ruff and mypy automatically on each commit.

---

## Tech stack

| Category | Tool |
|----------|------|
| Framework | FastAPI, Pydantic, Uvicorn |
| Embeddings | Cohere API |
| Numerics | NumPy |
| Logging | structlog |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Quality | Ruff, mypy, pre-commit |
| Packaging | Poetry |
| Deployment | Docker |

---

## License

MIT. See [LICENSE](LICENSE).
