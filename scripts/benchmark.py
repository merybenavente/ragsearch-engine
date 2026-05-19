"""
Benchmark script for ANN index implementations.

Downloads standard datasets from ann-benchmarks.com, runs each index
configuration, and reports build time, QPS, recall@10, and memory usage.

Usage:
    poetry run python scripts/benchmark.py --datasets glove-25-angular --subset-sizes 5000 10000
    poetry run python scripts/benchmark.py --datasets glove-25-angular glove-100-angular --subset-sizes 5000
"""

import argparse
import os
import sys
import time
import tracemalloc
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_db.domain.models import Chunk
from src.vector_db.infrastructure.indexes.lsh import LSHIndex
from src.vector_db.infrastructure.indexes.naive import NaiveIndex
from src.vector_db.infrastructure.indexes.vptree import VPTreeIndex

DATASET_BASE_URL = "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main"

DATASETS = {
    "glove-25-angular": f"{DATASET_BASE_URL}/glove-25-angular.hdf5",
    "glove-100-angular": f"{DATASET_BASE_URL}/glove-100-angular.hdf5",
}


@dataclass
class BenchmarkResult:
    dataset: str
    index_name: str
    params: str
    n_vectors: int
    build_time_s: float
    qps: float
    recall_at_10: float
    memory_mb: float


def download_dataset(name: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / f"{name}.hdf5"
    if filepath.exists():
        print(f"  Dataset {name} already downloaded.")
        return filepath

    url = DATASETS[name]
    print(f"  Downloading {name} from {url} ...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    print()
    return filepath


def compute_ground_truth(corpus: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute brute-force cosine similarity ground truth."""
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_norms = np.maximum(corpus_norms, 1e-10)
    corpus_norm = corpus / corpus_norms

    queries_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries_norms = np.maximum(queries_norms, 1e-10)
    queries_norm = queries / queries_norms

    gt = np.empty((len(queries), k), dtype=np.int64)
    batch_size = 100
    for i in range(0, len(queries), batch_size):
        batch = queries_norm[i : i + batch_size]
        sims = batch @ corpus_norm.T
        top_k_indices = np.argpartition(-sims, k, axis=1)[:, :k]
        for j in range(len(batch)):
            sorted_idx = top_k_indices[j][np.argsort(-sims[j, top_k_indices[j]])]
            gt[i + j] = sorted_idx
    return gt


def load_and_subsample(
    filepath: Path, subset_size: int, num_queries: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load HDF5 dataset, subsample corpus and queries, recompute ground truth."""
    print(f"  Loading {filepath.name} ...")
    with h5py.File(filepath, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])

    rng = np.random.RandomState(seed)

    actual_subset = min(subset_size, len(train))
    corpus_indices = rng.choice(len(train), size=actual_subset, replace=False)
    corpus = train[corpus_indices]

    actual_queries = min(num_queries, len(test))
    query_indices = rng.choice(len(test), size=actual_queries, replace=False)
    queries = test[query_indices]

    print(f"  Corpus: {len(corpus)} vectors, Queries: {len(queries)}, Dims: {corpus.shape[1]}")
    print(f"  Computing ground truth ...")
    ground_truth = compute_ground_truth(corpus, queries, k=10)

    return corpus, queries, ground_truth


def create_chunks(corpus: np.ndarray) -> List[Chunk]:
    return [
        Chunk(
            id=str(i),
            document_id="benchmark-doc",
            text=f"v{i}",
            embedding=row.tolist(),
        )
        for i, row in enumerate(corpus)
    ]


def get_index_configs():
    """Return list of (index_class, kwargs, name, params_string)."""
    configs = []

    configs.append((NaiveIndex, {}, "Naive", "-"))

    for tables in [4, 8, 12]:
        for planes in [4, 6, 8]:
            configs.append((
                LSHIndex,
                {"num_tables": tables, "num_hyperplanes": planes},
                "LSH",
                f"t={tables},h={planes}",
            ))

    for leaf in [10, 20, 40]:
        configs.append((
            VPTreeIndex,
            {"leaf_size": leaf},
            "VPTree",
            f"leaf={leaf}",
        ))

    return configs


def run_single_benchmark(
    index_class,
    index_kwargs: dict,
    index_name: str,
    params_str: str,
    chunks: List[Chunk],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    dataset_name: str,
) -> BenchmarkResult:
    index = index_class(**index_kwargs)

    # Build phase
    tracemalloc.start()
    t0 = time.perf_counter()
    index.add_chunks("benchmark-doc", chunks)
    build_time = time.perf_counter() - t0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak_memory / (1024 * 1024)

    # Query phase
    n_queries = len(queries)
    total_recall = 0.0
    t0 = time.perf_counter()
    for qi in range(n_queries):
        results = index.search(queries[qi].tolist(), k=10, min_similarity=0.0)
        retrieved_ids = {int(chunk.id) for chunk, _ in results}
        gt_ids = set(ground_truth[qi].tolist())
        total_recall += len(retrieved_ids & gt_ids) / 10.0
    query_time = time.perf_counter() - t0

    qps = n_queries / query_time if query_time > 0 else float("inf")
    recall = total_recall / n_queries

    return BenchmarkResult(
        dataset=dataset_name,
        index_name=index_name,
        params=params_str,
        n_vectors=len(chunks),
        build_time_s=build_time,
        qps=qps,
        recall_at_10=recall,
        memory_mb=memory_mb,
    )


def print_results_table(results: List[BenchmarkResult]):
    print()
    print("| Dataset | Index | Params | N | Build(s) | QPS | Recall@10 | Mem(MB) |")
    print("|---------|-------|--------|---|----------|-----|-----------|---------|")
    for r in results:
        print(
            f"| {r.dataset} | {r.index_name} | {r.params} | "
            f"{r.n_vectors} | {r.build_time_s:.2f} | {r.qps:.1f} | "
            f"{r.recall_at_10:.3f} | {r.memory_mb:.1f} |"
        )
    print()


def save_plots(results: List[BenchmarkResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    markers = {"Naive": "s", "LSH": "o", "VPTree": "^"}
    colors = {"Naive": "#2196F3", "LSH": "#FF9800", "VPTree": "#4CAF50"}

    # Group results by (dataset, n_vectors)
    groups: dict[tuple[str, int], List[BenchmarkResult]] = {}
    for r in results:
        key = (r.dataset, r.n_vectors)
        groups.setdefault(key, []).append(r)

    for (dataset, n_vectors), group in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_labels = set()

        for r in group:
            label = r.index_name if r.index_name not in plotted_labels else None
            ax.scatter(
                r.recall_at_10,
                r.qps,
                marker=markers[r.index_name],
                color=colors[r.index_name],
                s=80,
                label=label,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )
            plotted_labels.add(r.index_name)
            ax.annotate(
                r.params,
                (r.recall_at_10, r.qps),
                fontsize=7,
                textcoords="offset points",
                xytext=(5, 5),
                alpha=0.7,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS (log scale)")
        ax.set_title(f"QPS vs Recall@10 — {dataset} (N={n_vectors:,})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"qps_vs_recall_{dataset}_n{n_vectors}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ANN index implementations")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["glove-25-angular"],
        choices=list(DATASETS.keys()),
        help="Datasets to benchmark against",
    )
    parser.add_argument(
        "--subset-sizes",
        nargs="+",
        type=int,
        default=[10000],
        help="Corpus sizes to test (subsampled from train set)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=200,
        help="Number of queries to run per configuration",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/benchmarks",
        help="Directory for dataset downloads and output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_results: List[BenchmarkResult] = []
    configs = get_index_configs()
    total_runs = len(args.datasets) * len(args.subset_sizes) * len(configs)

    print(f"Benchmark: {len(args.datasets)} dataset(s), {len(args.subset_sizes)} size(s), {len(configs)} index configs = {total_runs} runs\n")

    run_num = 0
    for dataset_name in args.datasets:
        print(f"=== Dataset: {dataset_name} ===")
        filepath = download_dataset(dataset_name, data_dir)

        for subset_size in args.subset_sizes:
            print(f"\n--- Subset size: {subset_size} ---")
            corpus, queries, ground_truth = load_and_subsample(
                filepath, subset_size, args.num_queries, args.seed
            )
            chunks = create_chunks(corpus)

            for index_class, index_kwargs, index_name, params_str in configs:
                run_num += 1
                print(
                    f"[{run_num}/{total_runs}] {index_name} ({params_str}) "
                    f"on {dataset_name} [N={len(chunks)}] ...",
                    end=" ",
                    flush=True,
                )

                result = run_single_benchmark(
                    index_class,
                    index_kwargs,
                    index_name,
                    params_str,
                    chunks,
                    queries,
                    ground_truth,
                    dataset_name,
                )
                all_results.append(result)

                print(
                    f"Build: {result.build_time_s:.2f}s | "
                    f"QPS: {result.qps:.1f} | "
                    f"Recall@10: {result.recall_at_10:.3f} | "
                    f"Mem: {result.memory_mb:.1f}MB"
                )

    print_results_table(all_results)
    save_plots(all_results, data_dir)


if __name__ == "__main__":
    main()
