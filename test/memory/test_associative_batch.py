# type: ignore
import numpy as np
import pytest

try:
    import torch
except ImportError:
    pass

from src.memory.batch_associative import BatchAssociativeMemory


@pytest.fixture
def batch_memory():
    num_batches = 3
    memory_size = 1000
    embedding_dim = 128
    return BatchAssociativeMemory(num_batches, memory_size, embedding_dim)


@pytest.fixture(params=["numpy", "torch"])
def tensor_type(request):
    return request.param


def create_tensor(tensor_type, data):
    if tensor_type == "numpy":
        return data
    else:
        return torch.from_numpy(data)


def test_batch_add(batch_memory, tensor_type):
    # Create integer batched embeddings
    embeddings_np = np.random.randint(0, 10, (batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)
    embeddings = create_tensor(tensor_type, embeddings_np)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Check if the embeddings are added to the corresponding memories
    for i, memory in enumerate(batch_memory.memories):
        all_embeddings = memory.get_all_embeddings()
        assert all_embeddings.shape == (1, batch_memory.embedding_dim)
        assert np.array_equal(embeddings_np[i], all_embeddings[0])


def test_batch_remove(batch_memory, tensor_type):
    # Create integer batched embeddings
    embeddings_np = np.random.randint(0, 10, (batch_memory.num_batches, batch_memory.embedding_dim)).astype(np.float32)
    embeddings = create_tensor(tensor_type, embeddings_np)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Remove embeddings by index
    indices_to_remove_np = np.zeros(batch_memory.num_batches)
    indices_to_remove = create_tensor(tensor_type, indices_to_remove_np)
    batch_memory.batch_remove(indices_to_remove)

    # Check if the embeddings are removed from the corresponding memories
    for _, memory in enumerate(batch_memory.memories):
        all_embeddings = memory.get_all_embeddings()
        all_embeddings.shape = (0, batch_memory.embedding_dim)


def test_batch_search(batch_memory, tensor_type):
    # Create a specific vector
    specific_vector_np = np.ones((1, batch_memory.embedding_dim), dtype=np.float32)
    specific_vector = create_tensor(tensor_type, specific_vector_np)

    # Create batched embeddings
    embeddings_np = np.random.randint(0, 10, size=(batch_memory.num_batches - 1, batch_memory.embedding_dim)).astype(
        np.float32
    )

    # Combine the specific vector with random embeddings
    embeddings_np = np.vstack((specific_vector_np, embeddings_np))
    embeddings = create_tensor(tensor_type, embeddings_np)

    # Add embeddings to the batch memory
    batch_memory.batch_add(embeddings)

    # Perform batched search for the specific vector
    k = 1
    search_results = batch_memory.batch_search(specific_vector, k)

    # Check if the first search result is the same as the specific vector
    indices, distances = search_results[0]
    found_vector = batch_memory.memories[0].get_all_embeddings()[indices[0][0]]
    assert np.allclose(specific_vector_np, found_vector)
