# 🚀 RAGSearch Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/Tests-150%2B-passing-brightgreen.svg)

**A personal project built for fun - a semantic search engine for RAG applications with multiple vector index implementations**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [Tech Stack](#-tech-stack)

</div>

---

## 📋 Overview

RAGSearch Engine is a personal project exploring semantic search concepts with FastAPI, designed for RAG (Retrieval Augmented Generation) applications. It enables semantic search, document indexing, and similarity matching with multiple vector index implementations. Built for fun and learning, this project demonstrates clean architecture principles, domain-driven design, and software engineering best practices.

### Key Highlights

- 🎯 **Multiple Index Types**: Naive, LSH (Locality-Sensitive Hashing), and VPTree implementations
- 🔍 **Semantic Search**: Powered by Cohere embeddings for intelligent text similarity
- 🏗️ **Clean Architecture**: Domain-Driven Design with clear separation of concerns
- 🧪 **Comprehensive Testing**: 150+ tests covering unit, integration, and semantic quality
- 🚀 **Well Engineered**: Docker support, structured logging, and error handling
- ⚡ **High Performance**: Configurable index parameters for optimal performance

---

## ✨ Features

### Core Functionality
- **Document Management**: Full CRUD operations for documents within organized libraries
- **Automatic Chunking**: Intelligent text splitting for optimal search performance
- **Vector Embeddings**: Automatic generation using Cohere's embedding API
- **Library Organization**: Hierarchical structure for managing document collections
- **Similarity Search**: Fast semantic search across documents and libraries

### Technical Excellence
- **Thread-Safe Operations**: Concurrent access support with proper locking mechanisms
- **Structured Logging**: Comprehensive logging with performance metrics using `structlog`
- **Type Safety**: Full type hints with mypy validation
- **Docker Support**: Containerized deployment ready
- **RESTful API**: OpenAPI/Swagger documentation included

### Index Types

The system supports multiple vector index implementations, each optimized for different use cases:

| Index Type | Best For | Characteristics |
|------------|----------|----------------|
| **Naive** | Small datasets (< 1K vectors) | Simple linear search, exact results |
| **LSH** | Large datasets, approximate search | Fast approximate nearest neighbor search |
| **VPTree** | Medium datasets, exact search | Balanced performance with exact results |

---

## 🏗️ Architecture

This project follows **Domain-Driven Design (DDD)** principles with clean architecture and dependency injection:

```
src/vector_db/
├── api/           # API layer (FastAPI routers, schemas, dependencies)
├── application/   # Application services (business logic)
├── domain/        # Domain models, interfaces, and exceptions
└── infrastructure/ # Infrastructure (repositories, logging, external services)
```

### Architecture Highlights

- **Layered Design**: Clear separation between API, application, domain, and infrastructure layers
- **Dependency Injection**: All services use abstract interfaces for testability and flexibility
- **Domain Models**: Core business entities (Libraries, Documents, Chunks) with rich domain logic
- **Repository Pattern**: Abstracted data access for easy testing and provider swapping
- **Service Layer**: Orchestrates complex workflows like document processing and search

### Key Components

- **Libraries**: Top-level organizational units containing documents
- **Documents**: Text content that gets automatically chunked and indexed
- **Chunks**: Searchable pieces of text with vector embeddings
- **Metadata**: Timestamps, tags, and user information for all entities

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Pydantic** - Data validation using Python type annotations
- **Uvicorn** - Lightning-fast ASGI server

### Vector Search & ML
- **Cohere** - Embedding generation for semantic search
- **NumPy** - Efficient vector operations and mathematical computations

### Infrastructure
- **Poetry** - Dependency management and packaging
- **Docker** - Containerization for deployment
- **Structlog** - Structured logging for observability

### Testing & Quality
- **Pytest** - Comprehensive testing framework
- **Ruff** - Fast Python linter and formatter
- **MyPy** - Static type checking

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Docker (optional)
- Cohere API key (for embedding generation)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mbenavente/ragsearch-engine
cd ragsearch-engine
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your Cohere API key
# Get your API key from https://cohere.com/
```

Or set the environment variable directly:
```bash
export COHERE_API_KEY="your_api_key_here"
```

4. **Run the application:**
```bash
poetry run uvicorn src.vector_db.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Using Docker

1. **Build the image:**
```bash
docker build -f docker/Dockerfile -t vector-db .
```

2. **Set up environment variables:**
```bash
# Copy the example environment file and edit it
cp env.example .env
# Edit .env and add your Cohere API key
```

3. **Run the container:**
```bash
docker run -p 8000:8000 --env-file .env vector-db
```

### Production Deployment

**CORS Configuration:**
For production deployments, configure CORS to restrict allowed origins:

```bash
# In your .env file, specify allowed origins (comma-separated)
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

**Security Notes:**
- Default CORS allows all origins (`*`) for development convenience
- **Always restrict CORS origins in production** to prevent unauthorized access
- The API does not include authentication/authorization - consider adding middleware for production use
- Use HTTPS in production to protect API keys and data in transit

---

## 📚 API Documentation

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
- `POST /api/v1/libraries/{library_id}/search` - Search across all chunks in a library
- `POST /api/v1/libraries/{library_id}/documents/{document_id}/search` - Search within a specific document

### Example Request

**Search Request:**
```json
{
  "query_text": "machine learning algorithms",
  "k": 10,
  "min_similarity": 0.5
}
```

**Search Response:**
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

---

## 💡 Example Usage

### Create a Library with Custom Index

```bash
curl -X POST "http://localhost:8000/api/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Research Library",
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
curl -X POST "http://localhost:8000/api/v1/libraries/${LIBRARY_ID}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning neural networks",
    "k": 5
  }'
```

### Demo Data Script

Populate the database with sample data:

```bash
# Using VPTree index (default)
poetry run python scripts/create_demo_data.py --index-type vptree

# Using LSH index with custom parameters
poetry run python scripts/create_demo_data.py --index-type lsh --index-params '{"num_tables": 4, "num_hyperplanes": 2}'
```

---

## 🧪 Development

### Running Tests

The project has **150+ tests** organized into three categories:

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

# Integration tests (component interactions with mocked external services)
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

For detailed testing strategy, see the [Testing Strategy README](tests/README.md).

### Code Quality

**Format and lint code:**
```bash
poetry run ruff format src/ tests/
poetry run ruff check src/ tests/
```

**Type checking:**
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

---

## 📊 Project Statistics

- **Test Coverage**: 80%+ across all layers
- **Total Tests**: 150+ test cases
- **Code Quality**: Type-checked with mypy, linted with ruff
- **Architecture**: Clean architecture with DDD principles
- **Documentation**: Comprehensive API docs with OpenAPI/Swagger

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<div align="center">

**Built with ❤️ using Python and FastAPI**

</div>
