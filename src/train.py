import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Existing arguments
    ...
    parser.add_argument(
        "--faiss_index_type",
        type=str,
        default="IVFFlat",
        choices=["IVFFlat", "IVFPQ", "HNSWFlat"],
        help="Type of Faiss index to use for the memory (IVFFlat, IVFPQ, or HNSWFlat)",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1024,
        help="Number of clusters for Faiss index (used for IVFFlat and IVFPQ index types)",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=50000,
        help="Size of the memory buffer to store embeddings in the memory",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Dimensionality of the embeddings in the memory",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=8,
        help="Number of subquantizers for the Product Quantization (used for IVFPQ index type)",
    )
    parser.add_argument(
        "--ef_construction",
        type=int,
        default=100,
        help="Size of the dynamic list for HNSW index construction (used for HNSWFlat index type)",
    )
    parser.add_argument(
        "--ef_search",
        type=int,
        default=64,
        help="Size of the dynamic list for HNSW index search (used for HNSWFlat index type)",
    )
    return parser.parse_args()
