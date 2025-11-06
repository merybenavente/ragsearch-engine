# Vector Database API

A FastAPI-based REST API for vector database operations with document indexing and similarity search capabilities.

## Features

- **Document Management**: Create, read, update, and delete documents within libraries
- **Automatic Chunking**: Documents are automatically split into searchable chunks
- **Vector Embeddings**: Automatic generation of vector embeddings for text chunks
- **Library Organization**: Organize documents into libraries with metadata
- **Thread-Safe Operations**: Concurrent access support with proper locking
- **Structured Logging**: Comprehensive logging with performance metrics
- **Comprehensive Testing**: 100+ tests covering all functionality
- **Docker Support**: Containerized deployment ready

## Architecture

The project follows Domain-Driven Design (DDD) principles with clean architecture and dependency injection:

```
src/vector_db/
├── api/           # API layer (FastAPI routers, schemas, dependencies)
├── application/   # Application services (business logic)
├── domain/        # Domain models, interfaces, and exceptions
└── infrastructure/ # Infrastructure (repositories, logging, external services)
```

The architecture implements a comprehensive dependency injection pattern where all services are provided through abstract interfaces, enabling easy testing and flexible embedding provider swapping. The `EmbeddingService` interface abstracts vector generation from the business logic, while the application layer orchestrates document chunking and embedding generation to ensure chunks are always created with their corresponding vector embeddings. This design separates concerns cleanly: domain models focus on business rules, application services handle use cases and orchestration, infrastructure services manage external dependencies (like the Cohere API), and the API layer provides a consistent RESTful interface with proper error handling and dependency injection throughout.

### Key Components

- **Libraries**: Top-level organizational units containing documents
- **Documents**: Text content that gets automatically chunked
- **Chunks**: Searchable pieces of text with vector embeddings
- **Metadata**: Timestamps, tags, and user information

### Index Types

The system supports multiple vector index types for different performance characteristics:

- **Naive Index** (`naive`): Simple linear search, good for small datasets
- **LSH Index** (`lsh`): Locality-Sensitive Hashing for approximate nearest neighbor search
  - `num_tables`: Number of hash tables (default: 8)
  - `num_hyperplanes`: Number of hyperplanes per table (default: 6)
- **VPTree Index** (`vptree`): Vantage Point Tree for exact nearest neighbor search
  - `leaf_size`: Maximum number of points in leaf nodes (default: varies by implementation)

Configure index type and parameters when creating a library to optimize for your specific use case.

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vector-db
```

2. Install dependencies:
```bash
poetry install
```

3. Run the application:
```bash
poetry run uvicorn src.vector_db.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Using Docker

1. Build the image:
```bash
docker build -f docker/Dockerfile -t vector-db .
```

2. Set up environment variables:
```bash
# Create .env file with your API key
echo "COHERE_API_KEY=your_api_key_here" > .env
```

3. Run the container:
```bash
docker run -p 8000:8000 --env-file .env vector-db
```

**Note**: The COHERE_API_KEY is required for document creation as it generates embeddings for text chunks.

## API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`
- **Health check**: `http://localhost:8000/health`

### Core Endpoints

#### Libraries
- `POST /api/v1/libraries/` - Create a new library
- `GET /api/v1/libraries/` - List all libraries
- `GET /api/v1/libraries/{library_id}` - Get library details
- `PUT /api/v1/libraries/{library_id}` - Update library
- `DELETE /api/v1/libraries/{library_id}` - Delete library

#### Documents
- `POST /api/v1/libraries/{library_id}/documents/` - Create document
- `GET /api/v1/libraries/{library_id}/documents/` - List documents
- `GET /api/v1/libraries/{library_id}/documents/{document_id}` - Get document
- `PUT /api/v1/libraries/{library_id}/documents/{document_id}` - Update document
- `DELETE /api/v1/libraries/{library_id}/documents/{document_id}` - Delete document

#### Search
- `POST /api/v1/libraries/{library_id}/search` - Perform text-based similarity search across chunks in a library
- `POST /api/v1/libraries/{library_id}/documents/{document_id}/search` - Perform text-based similarity search within a specific document

**Request Body**:
```json
{
  "query_text": "machine learning algorithms",
  "k": 10
}
```

**Response**:
```json
{
  "results": [
    {
      "chunk": {
        "id": "chunk-uuid",
        "document_id": "doc-uuid",
        "text": "chunk content...",
        "metadata": {
          "creation_time": "2023-01-01T00:00:00Z",
          "last_update": "2023-01-01T00:00:00Z",
          "username": "user",
          "tags": ["tag1"]
        }
      },
      "similarity_score": 0.95
    }
  ],
  "total_chunks_searched": 42,
  "query_time_ms": 15.3
}
```


## Example Usage

### Create a Library
```bash
curl -X POST "http://localhost:8000/api/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Research Library",
    "username": "researcher",
    "tags": ["research", "papers"]
  }'
```

### Create a Library with Custom Index Parameters
```bash
# Create a library with LSH index and custom parameters
curl -X POST "http://localhost:8000/api/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "High-Performance Library",
    "username": "researcher",
    "tags": ["research", "papers"],
    "index_type": "lsh",
    "index_params": {
      "num_tables": 12,
      "num_hyperplanes": 10
    }
  }'
```

### Add a Document
```bash
curl -X POST "http://localhost:8000/api/v1/libraries/${LIBRARY_ID}/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is my research document content...",
    "chunk_size": 500,
    "tags": ["important"]
  }'
```

### Search for Similar Content
```bash
# Search within a library
curl -X POST "http://localhost:8000/api/v1/libraries/${LIBRARY_ID}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning neural networks",
    "k": 5
  }'

# Search within a specific document
curl -X POST "http://localhost:8000/api/v1/libraries/${LIBRARY_ID}/documents/${DOCUMENT_ID}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning neural networks",
    "k": 5
  }'
```

## Development

### Running Tests

Run the recommended test suite (excludes slow semantic quality tests):
```bash
poetry run pytest -m "not semantic_quality"
```

Run the full test suite (including slow semantic quality tests, and which needs a `COHERE_API_KEY`):
```bash
poetry run pytest
```


Run specific test categories:
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/

# Semantic quality tests (requires COHERE_API_KEY)
poetry run pytest -m semantic_quality

# Specific test file
poetry run pytest tests/unit/test_domain_models.py
```

More information in the [Testing Strategy README](tests/README.md).

### Code Quality

Format and lint code:
```bash
poetry run ruff format src/ tests/
poetry run ruff check src/ tests/
```

Type checking:
```bash
poetry run mypy src/
```

Pre-commit hooks are configured to run ruff formatting, linting, and mypy type checking automatically on each commit.

### Logging

The application uses structured logging with `structlog`:

- **Production**: JSON format for log aggregation
- **Development**: Human-readable console format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

Configure logging in your environment:
```python
from src.vector_db.infrastructure.logging import configure_logging, LogLevel

# Configure for production
configure_logging(level=LogLevel.INFO, json_format=True)
```

Logs include:
- Request/response information
- Performance metrics
- Error context and stack traces
- Thread safety operations
- Business logic events
