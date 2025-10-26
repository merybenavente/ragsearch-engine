# Semantic Search Quality Tests

These tests evaluate the quality of semantic search using the real Cohere API (no mocks).

## Requirements

- `COHERE_API_KEY` must be set as an environment variable
- Internet connection to access the Cohere API

## Running the tests

### All semantic quality tests:
```bash
poetry run pytest -m semantic_quality
```

### Only the semantic search quality test file:
```bash
poetry run pytest tests/integration/test_semantic_search_quality.py
```

### With verbose output:
```bash
poetry run pytest -m semantic_quality -v
```

### With coverage:
```bash
poetry run pytest -m semantic_quality --cov=src.vector_db --cov-report=term-missing
```

### A specific test:
```bash
poetry run pytest tests/integration/test_semantic_search_quality.py::TestSemanticSearchQuality::test_semantic_similarity_basic
```

### Run all tests EXCEPT semantic quality (default):
```bash
poetry run pytest -m "not semantic_quality"
```

## What these tests evaluate

1. **Basic semantic similarity**: Verifies that related queries have higher similarity than unrelated queries
2. **Content relevance**: Checks that search results are ranked correctly by semantic relevance
3. **Edge cases**: Tests behavior with edge cases and boundary conditions
4. **Consistency**: Ensures that identical embeddings produce consistent results

## Important notes

- These tests are **slow** because they use the real Cohere API
- They require a valid Cohere API key
- Similarity thresholds are tuned for the `embed-v4.0` model
- If they fail, it may indicate issues with the API, embedding quality, or similarity calculations
- Tests are automatically skipped if `COHERE_API_KEY` is not set
- By default, these tests are excluded from normal test runs (see `pytest.ini`) 