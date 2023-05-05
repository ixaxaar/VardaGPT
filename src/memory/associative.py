from typing import Any, Union

import faiss
import numpy as np
import numpy.typing as npt
import torch

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
        forgetfulness_factor: float = 0.0001,
    ):
        """
        Initialize the associative memory.

        :param memory_size: The maximum number of items the memory can store.
        :param embedding_dim: The dimensionality of the input embeddings.
        :param index_type: The type of FAISS index to use (default: 'flat').
        :param num_clusters: The number of clusters to use for an IVF index (default: 1024).
        :param m: The number of product quantization codes to use for an IVFPQ index (default: 8).
        :param ef_construction: The size of the entry list for an HNSW index (default: 100).
        :param ef_search: The search list size for an HNSW index (default: 64).
        :param use_gpu: Whether to use GPU acceleration for the FAISS index (default: False).
        :param gpu_device: The ID of the GPU device to use (default: 0).
        :param forgetfulness_factor: The percentage of items to remove during random forgetting (default: 1).
        """
        # Initialize memory parameters
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.forgetfulness_factor = forgetfulness_factor
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

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

        # Initialize an empty array to store the input vectors
        self.input_vectors = np.zeros((memory_size, embedding_dim), dtype=np.float32)

    def _to_numpy(self, data: Union[npt.NDArray[Any], torch.Tensor]) -> Union[npt.NDArray[Any], torch.Tensor]:
        """
        Convert input data to a NumPy array if it's a PyTorch tensor and not using GPU.

        :param data: Input data to be converted. Can be either a NumPy array or a PyTorch tensor.
        :return: Converted data as a NumPy array or a PyTorch tensor.
        """
        if isinstance(data, torch.Tensor) and not self.use_gpu:
            return data.detach().cpu().numpy()  # type: ignore
        return data

    def add(self, embeddings: Union[npt.NDArray[Any], torch.Tensor]) -> None:
        """
        Add embeddings to the memory.

        :param embeddings: A 2D array of shape (n, embedding_dim) containing the embeddings to be added,
                           where n is the number of items to add. Can be either a NumPy array or a PyTorch tensor.
        """
        embeddings = self._to_numpy(embeddings)
        n_added = self.index.ntotal  # Existing number of added items
        n_to_add = embeddings.shape[0]  # Number of items to add

        # Update the input_vectors array with the new embeddings
        self.input_vectors[n_added : n_added + n_to_add] = embeddings

        # Add embeddings to the index
        if self.use_gpu:
            ptr = faiss.torch_utils.swig_ptr_from_FloatTensor(embeddings)
            self.index.add_c(n_to_add, ptr)
        else:
            self.index.add(embeddings)

    def remove(self, ids: Union[npt.NDArray[Any], torch.Tensor]) -> None:
        """
        Remove embeddings with the specified IDs from the memory.

        :param ids: A 1D array of shape (n,) containing the indices of the items to be removed,
                    where n is the number of items to remove. Can be either a NumPy array or a PyTorch tensor.
        """
        ids = self._to_numpy(ids)

        if self.index_type != "flat":
            raise ValueError(
                f"Update is not implemented in FAISS this type of index, use flat instad of: {self.index_type}"
            )
        self.index.remove_ids(ids)

        # Remove input vectors from the input_vectors array
        self.input_vectors = np.delete(self.input_vectors, ids, axis=0)

    def update(
        self, ids: Union[npt.NDArray[Any], torch.Tensor], updated_embeddings: Union[npt.NDArray[Any], torch.Tensor]
    ) -> None:
        """
        Update embeddings with the specified IDs in the memory.

        :param ids: A 1D array of shape (n,) containing the indices of the items to be updated,
                    where n is the number of items to update. Can be either a NumPy array or a PyTorch tensor.
        :param updated_embeddings: A 2D array of shape (n, embedding_dim) containing the updated embeddings,
                    where n is the number of items to update. Can be either a NumPy array or a PyTorch tensor.
        """
        ids = self._to_numpy(ids)
        updated_embeddings = self._to_numpy(updated_embeddings)

        if self.index_type != "flat":
            raise ValueError(
                f"Update is not implemented in FAISS this type of index, use flat instad of: {self.index_type}"
            )
        self.remove(ids)
        self.add(updated_embeddings)

    def search(self, query_vectors: Any, k: int = 10) -> Any:
        """
        Search the memory for the top k closest embeddings to the query vectors.

        :param query_vectors: A 2D array or tensor of shape (n, embedding_dim) containing the query vectors,
                            where n is the number of query vectors.
        :param k: The number of nearest neighbors to return for each query.
        :return: A tuple containing two 2D arrays or tensors for indices and distances, both of shape (n, k).
        """
        if self.use_gpu and isinstance(query_vectors, torch.Tensor):
            n_query = query_vectors.shape[0]
            distances = torch.empty((n_query, k), device=self.gpu_device)  # type: ignore
            indices = torch.empty((n_query, k), dtype=torch.long, device=self.gpu_device)  # type: ignore

            ptr_query = faiss.torch_utils.swig_ptr_from_FloatTensor(query_vectors)
            ptr_distances = faiss.torch_utils.swig_ptr_from_FloatTensor(distances)
            ptr_indices = faiss.torch_utils.swig_ptr_from_LongTensor(indices)
            self.index.search_c(n_query, ptr_query, k, ptr_distances, ptr_indices)

            return indices, distances
        else:
            if isinstance(query_vectors, torch.Tensor):
                query_vectors = query_vectors.numpy()
            distances, indices = self.index.search(query_vectors, k)
            if isinstance(query_vectors, torch.Tensor):
                return torch.from_numpy(indices), torch.from_numpy(distances)
            else:
                return indices, distances

    def age_memory(self, decay_factor: float = 0.999) -> None:
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
        """
        Retrieve all embeddings stored in the memory.

        Returns:
        npt.NDArray[Any]: A 2D array of shape (n, embedding_dim) containing all stored embeddings,
                          where n is the number of stored items.
        """
        # Return the stored input_vectors directly
        return self.input_vectors[: self.index.ntotal]  # Use slicing to get only the added items

    def __getitem__(self, index: int) -> Any:
        """
        Retrieve the input vector at the specified index.

        :param index: The index of the input vector to retrieve.
        :return: A 1D array of shape (embedding_dim,) containing the input vector.
        """
        if index >= self.index.ntotal:
            raise IndexError("Index out of range.")
        return self.input_vectors[index]

    def size(self) -> int:
        """
        Get the number of items stored in the memory.

        :return: The number of items stored in the memory.
        """
        return int(self.index.ntotal)
