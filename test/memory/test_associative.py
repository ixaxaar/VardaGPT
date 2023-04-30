# type: ignore
import numpy as np
import torch
import pytest

from src.memory.associative import AssociativeMemory


@pytest.fixture(params=["numpy", "torch"])
def embeddings(request):
    embeddings_np = np.random.rand(1000, 768).astype(np.float32)
    if request.param == "numpy":
        return embeddings_np
    else:
        return torch.from_numpy(embeddings_np)


def test_add_and_search(embeddings):
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    memory.add(embeddings)

    query_vectors = embeddings[:5]
    indices, distances = memory.search(query_vectors)

    assert indices.shape == (5, 10)


def test_update(embeddings):
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    memory.add(embeddings)

    element_id = np.array([0], dtype=np.int64)
    updated_embedding = embeddings[:1]
    memory.update(element_id, updated_embedding)

    query_vectors = embeddings[1:2]
    indices, distances = memory.search(query_vectors)

    assert 0 not in indices[0]


def test_refresh_memory(embeddings):
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    memory.add(embeddings)

    memory.refresh()

    query_vectors = embeddings[:5]
    indices, distances = memory.search(query_vectors)

    assert indices.shape == (5, 10)


def test_getitem(embeddings):
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    memory.add(embeddings)

    # Test __getitem__ method
    retrieved_vector = memory[5]
    assert np.allclose(retrieved_vector, embeddings[5])
