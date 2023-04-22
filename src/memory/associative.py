import faiss
import numpy as np
from typing import Any

from . import Memory


class AssociativeMemory(Memory):
    def __init__(
        self,
        memory_size: int,
        embedding_dim: int,
        faiss_index_type: str = "IVFFlat",
        num_clusters: int = 1024,
        m: int = 8,
        ef_construction: int = 100,
        ef_search: int = 64,
        use_gpu: bool = False,
        gpu_device: int = 0,
    ):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        if faiss_index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, num_clusters, faiss.METRIC_L2)
        elif faiss_index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, num_clusters, m, 8)
        elif faiss_index_type == "HNSWFlat":
            index = faiss.IndexHNSWFlat(embedding_dim, ef_construction, faiss.METRIC_L2)
            index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"Invalid faiss_index_type: {faiss_index_type}")

        if use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, gpu_device, index)
        else:
            self.index = index

        self.index.train(np.zeros((memory_size, embedding_dim), dtype=np.float32))

    def add(self, embeddings: np.typing.NDArray[Any]) -> None:
        self.index.add(embeddings)

    def remove(self, ids: np.typing.NDArray[Any]) -> None:
        self.index.remove_ids(ids)

    def update(self, ids: np.typing.NDArray[Any], updated_embeddings: np.typing.NDArray[Any]) -> None:
        self.remove(ids)
        self.add(updated_embeddings)

    def search(self, query_vectors: np.typing.NDArray[Any], k: int = 10) -> Any:
        distances, indices = self.index.search(query_vectors, k)
        return indices
