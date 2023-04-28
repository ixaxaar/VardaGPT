# type: ignore
import numpy as np

from src.memory.associative import AssociativeMemory


def test_add_and_search():
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    embeddings = np.random.rand(1000, 768).astype(np.float32)
    memory.add(embeddings)

    query_vectors = np.random.rand(5, 768).astype(np.float32)
    indices, distances = memory.search(query_vectors)

    assert indices.shape == (5, 10)


def test_update():
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    embeddings = np.random.rand(1000, 768).astype(np.float32)
    memory.add(embeddings)

    element_id = np.array([0], dtype=np.int64)
    updated_embedding = np.random.rand(1, 768).astype(np.float32)
    memory.update(element_id, updated_embedding)

    query_vectors = np.random.rand(1, 768).astype(np.float32)
    indices, distances = memory.search(query_vectors)

    assert 0 not in indices[0]


def test_refresh_memory():
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    embeddings = np.random.rand(1000, 768).astype(np.float32)
    memory.add(embeddings)

    memory.refresh()

    query_vectors = np.random.rand(5, 768).astype(np.float32)
    indices, distances = memory.search(query_vectors)

    assert indices.shape == (5, 10)


def test_getitem():
    memory = AssociativeMemory(memory_size=50000, embedding_dim=768)

    embeddings = np.random.rand(1000, 768).astype(np.float32)
    memory.add(embeddings)

    # Test __getitem__ method
    retrieved_vector = memory[5]
    assert np.allclose(retrieved_vector, embeddings[5])
