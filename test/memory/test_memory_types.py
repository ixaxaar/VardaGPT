# type: ignore
import numpy as np
import torch
import pytest

from src.memory.associative import AssociativeMemory


@pytest.fixture(params=["flat", "compressed", "graph"])
def index_type(request):
    return request.param


@pytest.fixture(params=["numpy", "torch"])
def tensor_type(request):
    return request.param


def to_tensor(tensor_type, array):
    if tensor_type == "numpy":
        return array
    elif tensor_type == "torch":
        return torch.from_numpy(array)
    else:
        raise ValueError("Unsupported tensor_type")


def test_associative_memory_add_remove_update_search(index_type, tensor_type):
    memory_size = 1000
    embedding_dim = 128
    num_test_vectors = 100
    k = 10

    memory = AssociativeMemory(memory_size, embedding_dim, index_type=index_type)

    # Add test embeddings to memory
    test_embeddings = to_tensor(tensor_type, np.random.random((num_test_vectors, embedding_dim)).astype(np.float32))
    memory.add(test_embeddings)

    # Search for closest embeddings in memory
    query_vectors = to_tensor(tensor_type, np.random.random((5, embedding_dim)).astype(np.float32))
    search_results, search_distances = memory.search(query_vectors, k)

    assert search_results.shape == (query_vectors.shape[0], k)

    if index_type == "flat":
        # Remove some embeddings from memory
        ids_to_remove = np.array([2, 5, 10, 30, 50])
        memory.remove(ids_to_remove)

        # Update some embeddings in memory
        ids_to_update = np.array([0, 1, 3, 4])
        updated_embeddings = to_tensor(
            tensor_type, np.random.random((len(ids_to_update), embedding_dim)).astype(np.float32)
        )

        memory.update(ids_to_update, updated_embeddings)

        # Check updated embeddings
        updated_search_results, updated_distances = memory.search(updated_embeddings, k=1)
        for i, _ in enumerate(ids_to_update):
            assert np.isclose(updated_distances[i, 0], 0, atol=1e-6)

    elif index_type in ["compressed", "graph"]:
        with pytest.raises(ValueError):
            memory.remove(np.array([0]))

        with pytest.raises(ValueError):
            memory.update(
                np.array([0]), to_tensor(tensor_type, np.random.random((1, embedding_dim)).astype(np.float32))
            )
