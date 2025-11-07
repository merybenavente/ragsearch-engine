# Vector Index Algorithms Comparison

This directory contains three different vector indexing algorithms, each optimized for different use cases and performance characteristics.
- Naive Index (`naive.py`): Linear search with exact cosine similarity.
- LSH Index (`lsh.py`): Locality-Sensitive Hashing for approximate nearest neighbor search.
- VP-Tree Index (`vptree.py`): Vantage Point Tree for exact nearest neighbor search in metric spaces.

## Algorithm Comparison

| Algorithm | Type | Exactness | Configuration | Memory Usage | Indexing Speed&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Search Speed&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Update Cost&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Dataset Size |
|-----------|------|-----------|---------------|--------------|----------------|--------------|-------------|--------------|
| [Naive](naive.py)     | Brute Force | Exact | None | • Low<br>• O(n×d) | • Fast&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(1) | • Slow&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(n×d) | • Fast&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(1) | < 10K |
| [LSH](lsh.py)       | Hashing | Approximate (~90-95%**) | High | • High<br>• O(n×L) | • Slow&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(n×L×k×d) | • Fast&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(L×k×d + c) | • Medium&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(L×k×d) | > 10K |
| [VP-Tree](vptree.py)   | Tree-based | Exact | Low | • Medium<br>• O(n) | • Medium&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(n log n×d) | • Medium&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>• O(log n×d) | • Slow&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;<br>• O(n log n×d) | 1K-100K |

*L = num_tables, k = num_hyperplanes, d = embedding dimension, c = candidates found

**LSH accuracy depends heavily on parameter tuning and data distribution
