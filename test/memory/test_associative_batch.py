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
    # Create integer batched embeddings
    embeddings = np.random.randint(0, 10, (batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Check if the embeddings are added to the corresponding memories
    for i, memory in enumerate(batch_memory.memories):
        all_embeddings = memory.get_all_embeddings()
        assert all_embeddings.shape == (1, batch_memory.embedding_dim)
        assert np.array_equal(embeddings[i], all_embeddings[0])


def test_batch_remove(batch_memory):
    # Create integer batched embeddings
    embeddings = np.random.randint(0, 10, (batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)
    # batch_memory.batch_add(embeddings)

    # Remove embeddings by index
    indices_to_remove = np.zeros(batch_memory.num_batches)
    batch_memory.batch_remove(indices_to_remove)

    # Check if the embeddings are removed from the corresponding memories
    for _, memory in enumerate(batch_memory.memories):
        all_embeddings = memory.get_all_embeddings()

        all_embeddings.shape = (0, batch_memory.embedding_dim)


def test_batch_search(batch_memory):
    # Create a specific vector
    specific_vector = np.ones((1, batch_memory.embedding_dim), dtype=np.float32)

    # Create batched embeddings
    embeddings = np.random.randint(0, 10, size=(batch_memory.num_batches - 1, batch_memory.embedding_dim)).astype(
        np.float32
    )

    # Combine the specific vector with random embeddings
    embeddings = np.vstack((specific_vector, embeddings))

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Perform batched search for the specific vector
    k = 1
    search_results = batch_memory.batch_search(specific_vector, k)

    # Check if the first search result is the same as the specific vector
    indices, distances = search_results[0]
    found_vector = batch_memory.memories[0].get_all_embeddings()[indices[0][0]]
    assert np.allclose(specific_vector, found_vector)
