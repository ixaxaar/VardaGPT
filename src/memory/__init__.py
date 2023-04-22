from typing import Optional, Any

from faiss import IndexIVF
import numpy as np


class Memory:
    memory_size: int = 0
    embedding_dim: int = 0
    index: IndexIVF

    def add(self, embeddings: np.typing.NDArray[Any]) -> None:
        self.index.add(embeddings)

    def remove(self, ids: np.typing.NDArray[Any]) -> None:
        self.index.remove_ids(ids)

    def update(self, ids: np.typing.NDArray[Any], updated_embeddings: np.typing.NDArray[Any]) -> None:
        self.remove(ids)
        self.add(updated_embeddings)

    def search(self, query_vectors: np.typing.NDArray[Any], k: int = 10) -> Any:
        raise NotImplementedError()
