from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from faiss import IndexIVF


class Memory:
    memory_size: int = 0
    embedding_dim: int = 0
    index: IndexIVF

    def add(self, embeddings: npt.NDArray[Any]) -> None:
        self.index.add(embeddings)

    def remove(self, ids: npt.NDArray[Any]) -> None:
        self.index.remove_ids(ids)

    def update(self, ids: npt.NDArray[Any], updated_embeddings: npt.NDArray[Any]) -> None:
        self.remove(ids)
        self.add(updated_embeddings)

    def search(self, query_vectors: npt.NDArray[Any], k: int = 10) -> Any:
        raise NotImplementedError()

    def refresh(self) -> None:
        self.index.reset()
