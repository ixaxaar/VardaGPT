from typing import Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt

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

    def batch_add(self, embeddings: Union[npt.NDArray[Any], "torch.Tensor"]) -> None:
        """
        Perform a batch of add operations. Assumes a single token per batch.

        :param embeddings: A 2D array of shape (num_batches, embedding_dim) containing the embeddings
                           of single tokens to be added for each batch.
        """
        for memory, batch_embeddings in zip(self.memories, embeddings):
            memory.add(batch_embeddings.reshape(1, -1))

    def batch_remove(self, ids: Union[npt.NDArray[Any], "torch.Tensor"]) -> None:
        """
        Perform a batch of remove operations. Assumes a single token per batch.

        :param ids: A 1D array of shape (num_batches,) containing the indices of the items to be removed.
        """
        for memory, batch_id in zip(self.memories, ids):
            memory.remove(np.array([batch_id], dtype=np.int64))

    def batch_search(
        self, query_vectors: Union[npt.NDArray[Any], "torch.Tensor"], k: int = 10
    ) -> List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]]:
        """
        Perform a batch of search operations. Assumes a single token per batch.

        :param query_vectors: A 2D array of shape (num_batches, embedding_dim) containing the query vectors
                              of single tokens for each batch.
        :param k: The number of nearest neighbors to return for each query.
        :return: A list of tuples, each containing two 1D arrays of shape (1,k) for indices and distances.
        """
        results = []
        for memory, batch_query_vector in zip(self.memories, query_vectors.reshape(-1, 1, self.embedding_dim)):
            indices, distances = memory.search(batch_query_vector, k)
            results.append((indices, distances))
        return results
