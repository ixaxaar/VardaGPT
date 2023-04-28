# type: ignore
import numpy as np
import pytest
from src.memory.batch_associative import BatchAssociativeMemory


@pytest.fixture
def batch_memory():
    num_batches = 3
    memory_size = 1000
    embedding_dim = 128
    return BatchAssociativeMemory(num_batches, memory_size, embedding_dim)


def test_batch_add(batch_memory):
    # Create batched embeddings
    embeddings = np.random.random((batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Check if the embeddings are added to the corresponding memories
    for i, memory in enumerate(batch_memory.memories):
        all_embeddings = memory.get_all_embeddings()
        assert all_embeddings.shape == (1, batch_memory.embedding_dim)
        assert np.allclose(embeddings[i], all_embeddings[0])


def test_batch_remove(batch_memory):
    # Create batched embeddings
    embeddings = np.random.random((batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Remove embeddings by index
    indices_to_remove = np.arange(batch_memory.num_batches)
    batch_memory.batch_remove(indices_to_remove)

    # Check if the embeddings are removed from the corresponding memories
    for memory in batch_memory.memories:
        all_embeddings = memory.get_all_embeddings()
        assert all_embeddings.shape == (0, batch_memory.embedding_dim)


def test_batch_search(batch_memory):
    # Create batched embeddings
    embeddings = np.random.random((batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Create batched query vectors
    query_vectors = np.random.random((batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)

    # Perform batched search
    k = 1
    search_results = batch_memory.batch_search(query_vectors, k)

    # Check if the search results have the correct shape
    for indices, distances in search_results:
        assert indices.shape == (1, k)
        assert distances.shape == (1, k)
