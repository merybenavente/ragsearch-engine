# Vector Index Algorithms Comparison

This directory contains three different vector indexing algorithms, each optimized for different use cases and performance characteristics.
- Naive Index (`naive.py`): Linear search with exact cosine similarity.
- LSH Index (`lsh.py`): Locality-Sensitive Hashing for approximate nearest neighbor search.
- VP-Tree Index (`vptree.py`): Vantage Point Tree for exact nearest neighbor search in metric spaces.

## Algorithm Comparison

| Algorithm |   Type   | Result Guarantee | Config |   Memory Usage   |  Indexing Speed   |   Search Speed   |  Update Cost  | Dataset Size |
|:---------:|:--------:|:---------:|:------:|:---------------:|:-----------------:|:----------------:|:-------------:|:------------:|
| [Naive](naive.py)   | Brute Force | Exact   | None  | • Baseline<br>• O(n×d) | O(n×d) | O(n×d) | O(1) | < 10 K |
| [LSH](lsh.py)      | Hashing     | Approx.<br>(tunable**) | High  | • High<br>• O(n×d + n×L) | O(n×L×k×d) | O(L×k×d + c) | O(L×k×d) | > 10 K |
| [VP‑Tree](vptree.py)| Tree‑based  | Exact   | Low   | • Medium<br>• O(n) | O(n log n×d) | O(log n×d)† | O(n log n×d) | 1 K–100 K |



*L = num_tables, k = num_hyperplanes, d = embedding dimension, c = candidates found

**LSH accuracy depends heavily on parameter tuning and data distribution

† In high‑dimensional spaces (>≈15 dims) VP‑Tree search tends toward **O(n×d)**.
