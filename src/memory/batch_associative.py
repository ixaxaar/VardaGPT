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
    def __init__(self, num_batches: int, memory_size: int, embedding_dim: int, **kwargs):
        self.memories = [AssociativeMemory(memory_size, embedding_dim, **kwargs) for _ in range(num_batches)]

    def _ensure_numpy(self, array: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return array

    def batch_add(self, embeddings: npt.NDArray[Any]) -> None:
        """Perform a batch of add operations."""
        embeddings = self._ensure_numpy(embeddings)
        for memory, batch_embeddings in zip(self.memories, embeddings):
            memory.add(batch_embeddings)

    def batch_remove(self, ids: npt.NDArray[Any]) -> None:
        """Perform a batch of remove operations."""
        ids = self._ensure_numpy(ids)
        for memory, batch_ids in zip(self.memories, ids):
            memory.remove(batch_ids)

    def batch_search(
        self, query_vectors: npt.NDArray[Any], k: int = 10
    ) -> List[Tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """Perform a batch of search operations."""
        query_vectors = self._ensure_numpy(query_vectors)
        results = []
        for memory, batch_query_vectors in zip(self.memories, query_vectors):
            indices, distances = memory.search(batch_query_vectors, k)
            results.append((indices, distances))
        return results
