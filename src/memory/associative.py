import faiss
import numpy as np
import numpy.typing as npt
from typing import Any

from . import Memory


class AssociativeMemory(Memory):
    def __init__(
        self,
        memory_size: int,
        embedding_dim: int,
        index_type: str = "flat",
        num_clusters: int = 1024,
        m: int = 8,
        ef_construction: int = 100,
        ef_search: int = 64,
        use_gpu: bool = False,
        gpu_device: int = 0,
        forgetfulness_factor: float = 1,
    ):
        # Initialize memory parameters
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.forgetfulness_factor = forgetfulness_factor

        # Create the appropriate Faiss index based on the specified type
        if index_type == "flat":
            # Inverted File with Flat index - a compressed index with an inverted file structure
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, num_clusters, faiss.METRIC_L2)
            index.make_direct_map()
            index.set_direct_map_type(faiss.DirectMap.Hashtable)
        elif index_type == "compressed":
            # Inverted File with Product Quantization index - a compressed index with a product quantization compression
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, num_clusters, m, 8)
        elif index_type == "graph":
            # Hierarchical Navigable Small World index - a graph-based index with a flat storage
            index = faiss.IndexHNSWFlat(embedding_dim, ef_construction, faiss.METRIC_L2)
            index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"Invalid index_type: {index_type}")
        self.index_type = index_type

        # Enable GPU support if specified
        if use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, gpu_device, index)
        else:
            self.index = index

        # Train the index with empty data
        self.index.train(np.zeros((max(memory_size, num_clusters), embedding_dim), dtype=np.float32))

    def add(self, embeddings: npt.NDArray[Any]) -> None:
        """Add embeddings to the memory."""
        self.index.add(embeddings)

    def remove(self, ids: npt.NDArray[Any]) -> None:
        """Remove embeddings with the specified IDs from the memory."""
        if self.index_type != "flat":
            raise ValueError(
                f"Update is not implemented in FAISS this type of index, use flat instad of: {self.index_type}"
            )
        self.index.remove_ids(ids)

    def update(self, ids: npt.NDArray[Any], updated_embeddings: npt.NDArray[Any]) -> None:
        """Update embeddings with the specified IDs in the memory."""
        if self.index_type != "flat":
            raise ValueError(
                f"Update is not implemented in FAISS this type of index, use flat instad of: {self.index_type}"
            )
        self.remove(ids)
        self.add(updated_embeddings)

    def search(self, query_vectors: npt.NDArray[Any], k: int = 10) -> Any:
        """Search the memory for the top k closest embeddings to the query vectors."""
        distances, indices = self.index.search(query_vectors, k)
        return indices, distances

    def age_memory(self, decay_factor: float = 0.99) -> None:
        """
        Age the memory embeddings by multiplying them with a decay factor.

        :param decay_factor: float, factor to multiply the embeddings with (default: 0.99)
        """
        assert 0 <= decay_factor <= 1, "Decay factor should be between 0 and 1."

        # Get the current embeddings from the memory
        current_embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
        for idx in range(self.index.ntotal):
            current_embeddings[idx] = self.index.reconstruct(idx)

        # Apply the decay factor
        aged_embeddings = current_embeddings * decay_factor

        # Update the memory with the aged embeddings
        ids = np.arange(self.index.ntotal, dtype=np.int64)
        self.update(ids, aged_embeddings)

    def forget_randomly(self) -> None:
        """
        Remove a random subset of embeddings from the memory based on the forgetfulness factor.
        """
        assert 0 <= self.forgetfulness_factor <= 1, "Forgetfulness factor should be between 0 and 1."

        total_embeddings = self.index.ntotal
        num_embeddings_to_forget = int(total_embeddings * self.forgetfulness_factor)

        # Select random embeddings to forget
        ids_to_forget = np.random.choice(total_embeddings, size=num_embeddings_to_forget, replace=False)

        # Remove the selected embeddings from the memory
        self.remove(ids_to_forget)

    def garbage_collect(self, threshold: float = 1e-6) -> npt.NDArray[Any]:
        """
        Remove nearly zero vectors from the memory and return the indices of the empty vectors.

        Parameters:
        threshold (float): Threshold value to consider a vector nearly zero.

        Returns:
        npt.NDArray[Any]: Indices of the empty vectors.
        """
        # Fetch all embeddings from the memory
        embeddings = self.get_all_embeddings()

        # Calculate the L2 norms of the embeddings
        norms = np.linalg.norm(embeddings, axis=1)

        # Identify nearly zero vectors based on the threshold
        nearly_zero_vectors = np.where(norms < threshold)[0]

        # Remove nearly zero vectors from the memory
        self.remove(nearly_zero_vectors)

        return nearly_zero_vectors

    def get_all_embeddings(self) -> npt.NDArray[Any]:
        """Retrieve all embeddings stored in the memory."""
        if isinstance(self.index, faiss.IndexIVFFlat):
            all_embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
            for idx in range(self.index.ntotal):
                all_embeddings[idx] = self.index.reconstruct(idx)
            return all_embeddings
        else:
            raise NotImplementedError("get_all_embeddings is not implemented for this index type")
