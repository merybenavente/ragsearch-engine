# Vector Index Algorithms Comparison

This directory contains three different vector indexing algorithms, each optimized for different use cases and performance characteristics.

## Algorithm Overview

### 1. Naive Index (`naive.py`)
**Linear search with exact cosine similarity**

**Time Complexity:**
- Indexing: O(1) - no preprocessing required
- Search: O(n×d) where n = number of chunks, d = embedding dimension
- Space: O(n×d) for storing embeddings

**When to use:**
- **Small datasets** (< 1K chunks)
- **Guarantee exact results** required
- **Development/testing** baseline
- **Real-time indexing** with immediate search needs

**Best for document types:**
- Personal note collections
- Small knowledge bases
- Development environments
- Legal documents where precision is critical

### 2. LSH Index (`lsh.py`)
**Locality-Sensitive Hashing for approximate nearest neighbor search**

**Time Complexity:**
- Indexing: O(n×L×k×d) where L = tables, k = hyperplanes, d = dimensions
- Search: O(L×k×d + c) where c = candidates found
- Space: O(n×L) for hash tables

**Parameters:**
- `num_tables`: More tables = better recall, more memory
- `num_hyperplanes`: More hyperplanes = finer bucketing, fewer collisions

**When to use:**
- **Large datasets** (> 10,000 chunks)
- **High-dimensional embeddings** (> 512 dimensions)
- **Fast approximate search** acceptable
- **Memory-constrained** environments

**Best for document types:**
- Large document collections (Wikipedia, news articles)
- Real-time search systems
- Content recommendation engines
- Social media posts
- Research paper databases

### 3. VP-Tree Index (`vptree.py`)
**Vantage Point Tree for exact nearest neighbor search in metric spaces**

**Time Complexity:**
- Indexing: O(n log n) average case, O(n²) worst case
- Search: O(log n) average case, O(n) worst case
- Space: O(n) for tree structure

**Parameters:**
- `leaf_size`: Minimum number of points in leaf nodes (unused in current implementation)
- `random_seed`: Seed for deterministic vantage point selection

**When to use:**
- **Medium to large datasets** (1,000 - 100,000 chunks)
- **Exact results required** but faster than naive
- **Balanced performance** needs
- **Metric space guarantees** important

**Best for document types:**
- Technical documentation
- Academic papers
- Legal case databases
- Medical literature
- Enterprise knowledge bases

## Performance Comparison

| Algorithm | Dataset Size | Search Speed | Memory Usage | Accuracy | Use Case |
|-----------|-------------|--------------|--------------|----------|----------|
| Naive     | < 1K        | Slow O(n)    | Low (~50MB/1K vectors) | 100%     | Small, exact |
| LSH       | > 10K       | Fast O(L×k×d + c) | High (~500MB/10K vectors) | ~90-95%* | Large, approximate |
| VP-Tree   | 1K-100K     | Medium O(log n) | Medium (~200MB/10K vectors) | 100%     | Medium, exact |

*LSH accuracy depends heavily on parameter tuning and data distribution

## Choosing the Right Index

### For Different Document Types:

**Personal Documents & Notes:**
- **Naive Index** - Small scale, exact results needed
- Example: Personal journal, meeting notes, todo lists

**Enterprise Knowledge Base:**
- **VP-Tree** - Balanced performance, exact results
- Example: Company wikis, technical documentation, training materials

**Content Platforms:**
- **LSH Index** - Large scale, fast approximate search
- Example: Blog platforms, news sites, social media

**Research & Academic:**
- **VP-Tree** or **Naive** - Exact results critical
- Example: Scientific papers, legal documents, medical literature

**Real-time Applications:**
- **LSH Index** - Speed over perfect accuracy
- Example: Chatbots, recommendation systems, auto-complete

### Configuration Guidelines:

**High-Precision Requirements:**
```python
# Use Naive for small datasets
library = vector_db.create_library(name="Legal Docs", index_type="naive")

# Use VP-Tree for medium datasets
library = vector_db.create_library(name="Research Papers", index_type="vptree",
                                  index_params={"leaf_size": 20, "random_seed": 42})
```

**High-Performance Requirements:**
```python
# LSH with aggressive parameters for speed
library = vector_db.create_library(name="News Articles", index_type="lsh",
                                  index_params={"num_tables": 4, "num_hyperplanes": 3})

# LSH with conservative parameters for accuracy
library = vector_db.create_library(name="Product Catalog", index_type="lsh",
                                  index_params={"num_tables": 8, "num_hyperplanes": 6})
```

## Implementation Notes

All indexes inherit from `BaseVectorIndex` and provide:
- **Thread safety** with RLock
- **Consistent interface** for add/remove/search operations
- **Cosine similarity** as the distance metric
- **Proper error handling** and logging

Choose based on your specific requirements for accuracy, speed, and dataset size.
