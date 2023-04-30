# type: ignore
import numpy as np
import torch
import pytest

from src.memory.associative import AssociativeMemory


@pytest.fixture(params=["numpy", "torch"])
def embeddings(request):
    embeddings_np = np.random.rand(1000, 128).astype(np.float32)
    if request.param == "numpy":
        return embeddings_np
    else:
        return torch.from_numpy(embeddings_np)


@pytest.fixture
def forgetful_memory():
    memory_size = 1000
    embedding_dim = 128
    forgetfulness_factor = 0.9
    return AssociativeMemory(memory_size, embedding_dim, forgetfulness_factor=forgetfulness_factor)


@pytest.fixture
def memory():
    memory_size = 1000
    embedding_dim = 128
    return AssociativeMemory(memory_size, embedding_dim)


def test_age_memory(embeddings, forgetful_memory):
    # Add some embeddings to the memory
    forgetful_memory.add(embeddings[:10])

    # Age the memory
    forgetful_memory.age_memory(0.5)

    # Check if the embeddings have been aged
    aged_embeddings = forgetful_memory.get_all_embeddings()
    assert aged_embeddings.shape == embeddings[:10].shape
    assert np.allclose(embeddings[:10] * 0.5, aged_embeddings)


def test_forget_randomly(embeddings, forgetful_memory):
    # Add some embeddings to the memory
    forgetful_memory.add(embeddings[:100])

    # Forget randomly
    forgetful_memory.forget_randomly()

    # Check if the number of embeddings in memory has decreased
    remaining_embeddings = forgetful_memory.get_all_embeddings()
    expected_remaining_embeddings = int(embeddings[:100].shape[0] * (1 - forgetful_memory.forgetfulness_factor))
    assert (
        remaining_embeddings.shape[0] == expected_remaining_embeddings
        or remaining_embeddings.shape[0] == expected_remaining_embeddings + 1
    )


def test_garbage_collect(embeddings, memory):
    # Add some nearly zero embeddings to the memory
    nearly_zero_embeddings = embeddings[:10] * 1e-7
    memory.add(nearly_zero_embeddings)

    # Garbage collect
    removed_indices = memory.garbage_collect(threshold=1e-6)

    # Check if the nearly zero embeddings have been removed
    assert len(removed_indices) == len(nearly_zero_embeddings)
    remaining_embeddings = memory.get_all_embeddings()
    assert remaining_embeddings.shape[0] == 0
    assert np.array_equal(removed_indices, np.arange(len(nearly_zero_embeddings)))
