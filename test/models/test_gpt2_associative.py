# type: ignore
import torch
from varda_gpt import VardaGPT
from memory.associative import AssociativeMemory


# Test the VardaGPT initialization
def test_varda_gpt_initialization():
    model = VardaGPT()
    assert isinstance(model, VardaGPT)


# Test the forward function with no memory_input
def test_varda_gpt_forward_no_memory_input():
    model = VardaGPT(use_gpu=False)
    input_vectors = torch.randn(1, 5, model.gpt2_config.n_embd)
    logits = model(input_vectors)
    assert logits.shape == (1, 5, model.output_dim)


# Test the forward function with memory_input
def test_varda_gpt_forward_with_memory_input():
    model = VardaGPT(use_gpu=False)
    input_vectors = torch.randn(1, 5, model.gpt2_config.n_embd)
    memory_input = torch.randn(1, model.memory.embedding_dim)
    logits = model(input_vectors, memory_input)
    assert logits.shape == (1, 5, model.output_dim)


# Test the addition and deletion of embeddings in the memory
def test_memory_operations():
    model = VardaGPT(use_gpu=False)
    memory = AssociativeMemory(memory_size=10, embedding_dim=3)

    # Test addition
    initial_size = memory.get_all_embeddings().shape[0]
    memory.add(torch.tensor([[1, 2, 3]]))
    new_size = memory.get_all_embeddings().shape[0]
    assert new_size == initial_size + 1

    # Test deletion
    memory.remove([0])
    final_size = memory.get_all_embeddings().shape[0]
    assert final_size == new_size - 1
