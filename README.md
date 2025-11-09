# Vector Database API

A FastAPI-based REST API for vector database operations with document indexing and similarity search capabilities.

## Features

- **Document Management**: Create, read, update, and delete documents within libraries
- **Automatic Chunking**: Documents are automatically split into searchable chunks
- **Vector Embeddings**: Automatic generation of vector embeddings for text chunks
- **Library Organization**: Organize documents into libraries with metadata
- **Thread-Safe Operations**: Concurrent access support with proper locking
- **Structured Logging**: Comprehensive logging with performance metrics
- **Comprehensive Testing**: 150+ tests covering all functionality
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

The architecture implements clean separation of concerns through a layered design with comprehensive dependency injection. All services use abstract interfaces to enable easy testing and flexible provider swapping. The domain layer defines core business entities (Libraries, Documents, Chunks) and their relationships, while the application layer orchestrates complex workflows like document processing, chunking, and search operations. The infrastructure layer handles external integrations (embedding providers, vector indices) and data persistence, supporting multiple index types (naive, LSH, VPTree) with configurable parameters for different performance characteristics. The API layer provides a consistent RESTful interface with proper error handling, request validation, and response formatting, all connected through dependency injection for maintainability and testability.

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

The project has 150+ tests organized into three categories:

**Quick Test Run (Recommended)**
```bash
# Run fast tests (unit + integration, excludes semantic quality tests)
poetry run pytest -m "not semantic_quality"
```

**Full Test Suite**
```bash
# Run all tests including semantic quality (requires COHERE_API_KEY)
poetry run pytest
```

**Test Categories**
```bash
# Unit tests (fast, isolated components with mocked dependencies)
poetry run pytest tests/unit/

# Integration tests (medium speed, component interactions with mocked external services)
poetry run pytest tests/integration/

# Semantic quality tests (slow, real Cohere API calls for search quality validation)
export COHERE_API_KEY="your-api-key"
poetry run pytest -m semantic_quality
```

**Additional Options**
```bash
# Run with coverage report
poetry run pytest tests/unit/ --cov=src/vector_db --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_domain_models.py

# Verbose output for debugging
poetry run pytest -v tests/integration/test_complete_workflow.py
```

For detailed testing strategy, architecture, and best practices, see the [Testing Strategy README](tests/README.md).

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
