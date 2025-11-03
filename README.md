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

2. Run the container:
```bash
docker run -p 8000:8000 vector-db
```

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
        "embedding": [0.1, 0.2, ...],
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
curl -X POST "http://localhost:8000/api/v1/libraries/" \
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
curl -X POST "http://localhost:8000/api/v1/libraries/" \
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
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/documents/" \
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
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning neural networks",
    "k": 5
  }'

# Search within a specific document
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/documents/{document_id}/search" \
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

Run with coverage:
```bash
poetry run pytest --cov=src/vector_db --cov-report=html -m "not semantic_quality"
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

### Test Structure

- **Unit Tests** (`tests/unit/`):
  - `test_domain_models.py` - Domain model functionality
  - `test_document_service.py` - Document service operations
  - `test_library_service.py` - Library service operations
  - `test_chunk_service.py` - Chunk service operations
  - `test_search_service_integration.py` - Search service integration
  - `test_repositories.py` - Repository operations and thread safety
  - `test_search_service.py` - Search service unit tests
  - `test_api_functions.py` - API function tests

- **Integration Tests** (`tests/integration/`):
  - `test_api_endpoints.py` - Full API endpoint testing
  - `test_main.py` - Main application testing
  - `test_semantic_search_quality.py` - Semantic search quality (slow, requires API key)

### Code Quality

Format code:
```bash
poetry run black src/ tests/
```

Lint code:
```bash
poetry run flake8 src/ tests/
```

Type checking:
```bash
poetry run mypy src/
```

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

## Technical Details

### Domain Models

- **Immutable Design**: Models use Pydantic's `model_copy()` for updates
- **Automatic Chunking**: Documents automatically split text into chunks
- **Embedding Generation**: Each chunk gets a vector embedding.
- **Metadata Tracking**: Creation time, last update, tags, and user information

### Thread Safety

- **Repository Locking**: Thread-safe operations using `RLock`
- **Concurrent Testing**: Comprehensive concurrent access testing
- **Performance Monitoring**: Lock contention and operation timing

### Data Flow

1. **Document Creation**: Text → Automatic Chunking → Embedding Generation
2. **Storage**: Thread-safe in-memory storage with proper locking
3. **Retrieval**: Fast lookup by ID with O(1) access patterns
4. **Updates**: Immutable updates with timestamp tracking

## Future Enhancements

- **Vector Search**: k-nearest neighbor search implementation
- **Multiple Index Types**: LSH, hierarchical clustering, etc.
- **Real Embeddings**: Integration with embedding models (OpenAI, Sentence Transformers)
- **Persistent Storage**: Database integration (PostgreSQL with pgvector)
- **Search API**: Similarity search endpoints
- **Authentication**: User authentication and authorization
- **Rate Limiting**: API rate limiting and quotas
