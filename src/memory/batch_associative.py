import numpy as np
import numpy.typing as npt
from typing import Any, List, Tuple, Union

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .associative import AssociativeMemory


class BatchAssociativeMemory:
    def __init__(
        self,
        num_batches: int,
        memory_size: int,
        embedding_dim: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a batch associative memory.

        :param num_batches: Number of separate associative memories in this batch.
        :param memory_size: The maximum number of items each memory can store.
        :param embedding_dim: The dimensionality of the input embeddings.
        :param kwargs: Additional arguments to be passed to the AssociativeMemory constructor.
        """
        self.num_batches = num_batches
        self.embedding_dim = embedding_dim
        self.memories = [AssociativeMemory(memory_size, embedding_dim, **kwargs) for _ in range(num_batches)]

    def _ensure_numpy(self, array: Union[npt.NDArray[Any], "torch.Tensor"]) -> npt.NDArray[Any]:
        if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()  # type: ignore
        return array  # type: ignore

    def batch_add(self, embeddings: npt.NDArray[np.float32]) -> None:
        """
        Perform a batch of add operations.

        :param embeddings: A 2D array of shape (num_batches, embedding_dim) containing the embeddings to be added.
        """
        embeddings = self._ensure_numpy(embeddings)
        for memory, batch_embeddings in zip(self.memories, embeddings):
            memory.add(batch_embeddings.reshape(1, -1))

    def batch_remove(self, ids: npt.NDArray[np.int64]) -> None:
        """
        Perform a batch of remove operations.

        :param ids: A 1D array of shape (num_batches,) containing the indices of the items to be removed.
        """
        ids = self._ensure_numpy(ids)
        for memory, batch_id in zip(self.memories, ids):
            memory.remove(np.array([batch_id], dtype=np.int64))

    def batch_search(
        self, query_vectors: npt.NDArray[np.float32], k: int = 10
    ) -> List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]]:
        """
        Perform a batch of search operations.

        :param query_vectors: A 2D array of shape (num_batches, embedding_dim) containing the query vectors.
        :param k: The number of nearest neighbors to return for each query.
        :return: A list of tuples, each containing two 1D arrays of shape (k,) for indices and distances.
        """
        query_vectors = self._ensure_numpy(query_vectors)
        results = []
        for memory, batch_query_vector in zip(self.memories, query_vectors.reshape(-1, 1, self.embedding_dim)):
            indices, distances = memory.search(batch_query_vector, k)
            results.append((indices, distances))
        return results