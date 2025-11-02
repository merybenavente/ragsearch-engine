# Vector Database Testing Strategy

This document outlines the testing strategy, organization, and best practices for the Vector Database project.

## 📁 Test Organization

### Directory Structure
```
tests/
├── README.md                           # This file - testing strategy and documentation
├── conftest.py                         # Global pytest configuration and fixtures
├── unit/                               # Unit tests for isolated components
│   ├── test_domain_models.py           # Domain model tests (Chunk, Document, Library)
│   ├── test_repositories.py            # Repository layer tests
│   ├── test_vector_db_service_*.py     # Service layer tests (split by functionality)
│   └── test_api_functions.py           # Basic API function tests
└── integration/                        # Integration and end-to-end tests
    ├── test_api_endpoints.py           # API endpoint integration tests
    ├── test_complete_workflow.py       # End-to-end workflow tests
    ├── test_semantic_search_quality.py # Semantic search quality tests
    └── test_main.py                    # Main application tests
```

### Test Categories

#### 🔧 Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Single classes, functions, or modules
- **Dependencies**: Use mocks and stubs to isolate system under test
- **Speed**: Fast (< 100ms per test)
- **Coverage**: Aim for 90%+ code coverage

#### 🔗 Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Scope**: Multiple components working together
- **Dependencies**: Real components with mocked external services
- **Speed**: Medium (100ms - 1s per test)
- **Coverage**: Focus on critical user paths

#### 📊 Semantic Quality Tests (`tests/integration/test_semantic_search_quality.py`)
- **Purpose**: Test search quality and relevance
- **Scope**: Real embedding API integration
- **Dependencies**: Actual Cohere API (optional, skipped if no API key)
- **Speed**: Slow (1s+ per test)
- **Coverage**: Search quality and semantic understanding

## 🏗️ Test Architecture

### Test Fixtures and Mocking Strategy

#### Global Fixtures (`conftest.py`)
- **`reset_repositories`**: Clears all repository data between tests
- **`sample_library_data`**: Standard library test data
- **`sample_document_data`**: Standard document test data
- **`mock_cohere_deterministic`**: Deterministic embeddings for consistent results

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

## 🧪 Testing Best Practices

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

## 🚀 Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                      # Unit tests only
pytest tests/integration/               # Integration tests only
pytest -m "not slow"                    # Skip slow tests

# Run with coverage
pytest --cov=src/vector_db --cov-report=html

# Run specific test file
pytest tests/unit/test_domain_models.py
```

### Test Markers
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.semantic_quality`: Semantic search quality tests
- `@pytest.mark.slow`: Slow-running tests (require API calls)

### Environment Setup
```bash
# Optional: Set Cohere API key for semantic quality tests
export COHERE_API_KEY="your-api-key-here"

# Run without API key (semantic tests will be skipped)
pytest tests/integration/test_semantic_search_quality.py
```

## 📈 Test Coverage Goals

### Coverage Targets
- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: 80%+ critical path coverage
- **Overall**: 85%+ combined coverage

### Coverage Areas
- ✅ Domain models (Chunk, Document, Library, Metadata)
- ✅ Service layer (VectorDBService)
- ✅ Repository layer (DocumentRepository, LibraryRepository)
- ✅ API endpoints
- ✅ Search functionality
- ⚠️ Index implementations (LSH, VPTree) - **WIP**
- ⚠️ Error recovery scenarios - **WIP**
- ⚠️ Performance characteristics - **WIP**


### Test Debugging
```bash
# Run single test with verbose output
pytest -xvs tests/unit/test_domain_models.py::TestChunk::test_chunk_creation

# Run with pdb debugger
pytest --pdb tests/unit/test_domain_models.py::TestChunk::test_chunk_creation

# Show test output
pytest -s tests/unit/test_domain_models.py
```