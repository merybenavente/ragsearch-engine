# Vector Database Testing Strategy

This document outlines the testing strategy, organization, and best practices for the Vector Database project.

## Test Organization

### Directory Structure
```
tests/
├── README.md                           # This file - testing strategy and documentation
├── __init__.py                         # Makes tests a Python package
├── conftest.py                         # Global pytest configuration and fixtures
├── utils.py                            # Shared test utilities and helpers
├── unit/                               # Unit tests for isolated components
│   ├── __init__.py                     # Package marker
│   ├── test_domain_models.py           # Domain model tests (Chunk, Document, Library)
│   ├── test_repositories.py            # Repository layer tests
│   ├── test_vector_db_service_document.py  # Document service tests
│   ├── test_vector_db_service_library.py   # Library service tests
│   ├── test_vector_db_service_search.py    # Search service tests
│   ├── test_api_functions.py           # Basic API function tests
│   ├── test_index_factory.py           # Index factory tests
│   ├── test_index_base.py              # BaseVectorIndex shared functionality tests
│   ├── test_index_naive.py             # NaiveIndex linear search tests
│   ├── test_index_lsh.py               # LSH index algorithm-specific tests
│   └── test_index_vptree.py            # VPTree index algorithm-specific tests
├── integration/                        # Integration and end-to-end tests
│   ├── __init__.py                     # Package marker
│   ├── test_api_endpoints.py           # API endpoint integration tests
│   ├── test_complete_workflow.py       # End-to-end workflow tests
│   ├── test_vector_db_service_integration.py # Service integration tests
│   └── test_main.py                    # Main application tests
└── semantic/                           # Semantic search quality tests (real Cohere API)
    └── test_semantic_search_quality.py # Real API semantic validation tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
**Purpose**: Test individual components in isolation with fast feedback

- **Scope**: Single classes, functions, or modules
- **Dependencies**: Mocked external services for deterministic results
- **Speed**: Fast (< 100ms per test)
- **Coverage**: Aim for 90%+ code coverage
- **Focus**: Edge cases, error conditions, algorithm correctness

**Key Components:**
- **Domain Models**: Chunk, Document, Library, Metadata validation
- **Service Layer**: VectorDBService business logic
- **Repository Layer**: Data access patterns
- **Vector Indexes**: Algorithm-specific behavior testing

### Integration Tests (`tests/integration/`)
**Purpose**: Test component interactions and complete workflows with mocked external services

- **Scope**: Multiple components working together
- **Dependencies**: Real internal components + mocked external services (Cohere API)
- **Speed**: Medium (100ms - 1s per test)
- **Coverage**: Critical user journeys and cross-component interactions
- **Focus**: API endpoints, complete workflows, system integration

**Key Scenarios:**
- **API Layer**: REST endpoint validation with proper HTTP responses
- **Complete Workflows**: End-to-end user journeys (create → ingest → search → update → delete)
- **Service Integration**: Cross-component interactions with realistic data flows
- **Error Recovery**: System resilience and error handling

### Semantic Quality Tests (`tests/semantic/`)
**Purpose**: Test search quality and semantic understanding using real embedding API

- **Scope**: Semantic search accuracy with real embeddings
- **Dependencies**: **Real Cohere API** (requires COHERE_API_KEY)
- **Speed**: Slow (1s+ per test due to API calls)
- **Coverage**: Search quality, semantic relationships, multilingual support
- **Focus**: Relative ranking validation, context understanding, synonyms

**Quality Validation:**
- **Semantic Similarity**: Related concepts score higher than unrelated ones
- **Context Awareness**: Domain-specific understanding (e.g., "Python" as programming language vs. snake)
- **Multilingual**: Cross-language semantic understanding
- **Ranking Quality**: Results properly ordered by relevance
- **Index Performance**: Semantic quality across different index types (LSH, VPTree, Naive)

## Test Architecture

### Test Fixtures and Mocking Strategy

#### Global Fixtures (`conftest.py`)
- **`reset_repositories`**: Clears all repository data between tests
- **`sample_library_data`**: Standard library test data
- **`sample_document_data`**: Standard document test data
- **`mock_cohere_deterministic`**: Deterministic embeddings for consistent results

#### Shared Test Utilities (`utils.py`)
- **`create_deterministic_embedding(text, dimension)`**: Creates reproducible embeddings based on text content
  - Uses hash-based seeding for consistent results across test runs
  - Normalizes embeddings to unit length
  - Supports configurable embedding dimensions

#### Mock Embedding Service
```python
class MockEmbeddingService(EmbeddingService):
    def create_embedding(self, text: str, input_type: str = "search_document") -> list[float]:
        # Creates deterministic embeddings based on text hash
        # Ensures reproducible test results
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed % (2**32))
        embedding = np.random.randn(EMBEDDING_DIMENSION)
        return (embedding / np.linalg.norm(embedding)).tolist()
```

**Why deterministic embeddings?**
- Ensures test reproducibility
- Allows testing of relative similarity relationships
- Eliminates flaky tests due to random embeddings
- Still enables testing of search functionality

### Test Data Management

#### Test Isolation
- Each test runs in isolation with clean state
- Repository data is cleared between tests
- No shared state between test cases

#### Test Data Patterns
- Use factories for creating test objects
- Provide sensible defaults with ability to override
- Use descriptive test data that reflects real usage

## Testing Best Practices

### Test Naming Convention
```python
def test_[component]_[action]_[expected_result]():
    """Test that [component] [action] [expected_result]"""
```

Examples:
- `test_library_creation_success()`: Test successful library creation
- `test_document_search_empty_library()`: Test searching in empty library
- `test_embedding_service_failure_handling()`: Test error handling

### Test Structure (Arrange-Act-Assert)
```python
def test_create_document_success(self):
    """Test successful document creation"""
    # Arrange
    library = self.vector_db_service.create_library("Test Library")
    document_data = {"text": "Test content", "username": "testuser"}

    # Act
    document = self.vector_db_service.create_document(
        library_id=library.id,
        **document_data
    )

    # Assert
    assert document.library_id == library.id
    assert document.metadata.username == "testuser"
    assert len(document.chunks) > 0
```

### Error Testing
Every major function should have corresponding error tests:
```python
def test_create_document_library_not_found(self):
    """Test creating document in non-existent library raises error"""
    fake_library_id = str(uuid4())

    with pytest.raises(ValueError, match="Library .* not found"):
        self.vector_db_service.create_document(
            library_id=fake_library_id,
            text="Test text"
        )
```

### Search Testing Strategy
- Use `min_similarity=0.0` for basic functionality tests
- Test relative similarity relationships rather than absolute scores
- Focus on ranking correctness over specific similarity values
- Test edge cases (empty queries, no results, etc.)

## Running Tests

### Basic Test Execution

#### Unit Tests (Fast, Mocked)
```bash
# Run all unit tests
poetry run pytest tests/unit/

# Run specific unit test components
poetry run pytest tests/unit/test_domain_models.py      # Domain models only
poetry run pytest tests/unit/test_index_*.py            # All index tests
poetry run pytest tests/unit/test_vector_db_service_*.py  # Service layer only

# Run with coverage
poetry run pytest tests/unit/ --cov=src/vector_db --cov-report=html
```

#### Integration Tests (Medium, Mocked External Services)
```bash
# Run all integration tests (mocked Cohere API)
poetry run pytest tests/integration/

# Run specific integration scenarios
poetry run pytest tests/integration/test_api_endpoints.py     # API layer only
poetry run pytest tests/integration/test_complete_workflow.py # End-to-end workflows
poetry run pytest tests/integration/test_vector_db_service_integration.py  # Service integration

# Run integration tests with coverage
poetry run pytest tests/integration/ --cov=src/vector_db
```

#### Semantic Quality Tests (Slow, Real API)
```bash
# Set up Cohere API key first
export COHERE_API_KEY="your-api-key-here"

# Run all semantic quality tests
poetry run pytest -m semantic_quality
# OR run the semantic directory directly
poetry run pytest tests/semantic/

# Run specific semantic tests
poetry run pytest tests/semantic/test_semantic_search_quality.py::TestSemanticSearchQuality::test_semantic_similarity_basic

# Run with verbose output (recommended for debugging)
poetry run pytest -m semantic_quality -v

# Skip if no API key (tests will be skipped automatically)
poetry run pytest tests/semantic/test_semantic_search_quality.py
```

#### Combined Test Runs
```bash
# Run all tests except semantic quality (default CI behavior)
poetry run pytest -m "not semantic_quality"

# Run everything including semantic tests (if API key available)
poetry run pytest

# Run fast tests only (unit + integration without semantic)
poetry run pytest -m "not slow"
```

### Test Markers
- `@pytest.mark.integration`: Integration tests with mocked external services
- `@pytest.mark.semantic_quality`: Semantic search quality tests (real Cohere API)
- `@pytest.mark.slow`: Slow-running tests (require API calls or heavy computation)

## Test Coverage Goals

### Coverage Targets
- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: 80%+ critical path coverage
- **Overall**: 80%+ combined coverage

### Coverage Areas
- ✅ Domain models (Chunk, Document, Library, Metadata)
- ✅ Service layer (VectorDBService)
- ✅ Repository layer (DocumentRepository, LibraryRepository)
- ✅ API endpoints
- ✅ Search functionality
- ✅ Index implementations (BaseVectorIndex, NaiveIndex, LSH, VPTree, Index Factory)


### Test Debugging
```bash
# Run single test with verbose output
poetry run pytest -xvs tests/unit/test_domain_models.py::TestChunk::test_chunk_creation

# Run with pdb debugger
poetry run pytest --pdb tests/unit/test_domain_models.py::TestChunk::test_chunk_creation

# Show test output
poetry run pytest -s tests/unit/test_domain_models.py
```
