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

The project follows Domain-Driven Design (DDD) principles with clean architecture:

```
src/vector_db/
├── api/           # API layer (FastAPI routers, schemas, dependencies)
├── application/   # Application services (business logic)
├── domain/        # Domain models and exceptions
└── infrastructure/ # Infrastructure (repositories, logging)
```

### Key Components

- **Libraries**: Top-level organizational units containing documents
- **Documents**: Text content that gets automatically chunked
- **Chunks**: Searchable pieces of text with vector embeddings
- **Metadata**: Timestamps, tags, and user information

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

#### Chunks
- `GET /api/v1/libraries/{library_id}/chunks/` - List all chunks in library
- `GET /api/v1/libraries/{library_id}/chunks/{chunk_id}` - Get specific chunk
- `GET /api/v1/libraries/{library_id}/chunks/documents/{document_id}` - Get document chunks

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

## Development

### Running Tests

Run the full test suite:
```bash
poetry run pytest
```

Run with coverage:
```bash
poetry run pytest --cov=src/vector_db --cov-report=html
```

Run specific test categories:
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/

# Specific test file
poetry run pytest tests/unit/test_domain_models.py
```

### Test Structure

- **Unit Tests** (`tests/unit/`):
  - `test_domain_models.py` - Domain model functionality
  - `test_services.py` - Application service logic
  - `test_repositories.py` - Repository operations and thread safety

- **Integration Tests** (`tests/integration/`):
  - `test_api_endpoints.py` - Full API endpoint testing

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
- **Embedding Generation**: Each chunk gets a vector embedding (currently random, placeholder for real embeddings)
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
