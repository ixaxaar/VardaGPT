# type: ignore
import numpy as np
import torch
from src.memory.multitoken_batch_associative import MultiTokenBatchAssociativeMemory


def test_batch_add():
    num_batches = 2
    memory_size = 5
    embedding_dim = 3
    num_tokens = 4

    memory = MultiTokenBatchAssociativeMemory(num_batches, memory_size, embedding_dim)

    embeddings = np.random.randn(num_batches, num_tokens, embedding_dim)
    memory.batch_add(embeddings)

    for mem in memory.memories:
        assert mem.size() == num_tokens


def test_batch_remove():
    num_batches = 2
    memory_size = 5
    embedding_dim = 3
    num_tokens = 4

    memory = MultiTokenBatchAssociativeMemory(num_batches, memory_size, embedding_dim)

    embeddings = np.random.randn(num_batches, num_tokens, embedding_dim)
    memory.batch_add(embeddings)

    ids_to_remove = np.array([[0, 1], [2, 3]])
    memory.batch_remove(ids_to_remove)

    for mem in memory.memories:
        assert mem.size() == num_tokens - 2


def test_batch_search():
    num_batches = 2
    memory_size = 5
    embedding_dim = 3
    num_tokens = 4

    memory = MultiTokenBatchAssociativeMemory(num_batches, memory_size, embedding_dim)

    embeddings = np.random.randn(num_batches, num_tokens, embedding_dim)
    memory.batch_add(embeddings)

    query_vectors = np.random.randn(num_batches, num_tokens, embedding_dim)
    k = 3
    search_results = memory.batch_search(query_vectors, k)

    for indices, distances in search_results:
        assert indices.shape == (num_tokens, k)
        assert distances.shape == (num_tokens, k)


def test_torch_tensors():
    num_batches = 2
    memory_size = 5
    embedding_dim = 3
    num_tokens = 4

    memory = MultiTokenBatchAssociativeMemory(num_batches, memory_size, embedding_dim)

    embeddings = torch.randn(num_batches, num_tokens, embedding_dim)
    memory.batch_add(embeddings)

    query_vectors = torch.randn(num_batches, num_tokens, embedding_dim)
    k = 3
    search_results = memory.batch_search(query_vectors, k)

    for indices, distances in search_results:
        assert indices.shape == (num_tokens, k)
        assert distances.shape == (num_tokens, k)
