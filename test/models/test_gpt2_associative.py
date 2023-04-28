# type: ignore
import torch

# from src.memory.associative import AssociativeMemory
from src.models.gpt2_associative import VardaGPTAssociative


# Test the VardaGPT initialization
def test_varda_gpt_initialization():
    model = VardaGPTAssociative()
    assert isinstance(model, VardaGPTAssociative)


# Test the forward function with no memory_input
def test_varda_gpt_forward_no_memory_input():
    model = VardaGPTAssociative(use_gpu=False)
    input_vectors = torch.randn(1, 5, model.gpt2_config.n_embd)
    logits = model(input_vectors)
    assert logits.shape == (1, 5, model.gpt2_config.vocab_size)


# Test the forward function with memory_input
def test_varda_gpt_forward_with_memory_input():
    model = VardaGPTAssociative(use_gpu=False)
    input_vectors = torch.randn(1, 5, model.gpt2_config.n_embd)

    # Add some embeddings to the memory
    dummy_embeddings = torch.randn(10, model.memory.embedding_dim)
    model.memory.add(dummy_embeddings.numpy())

    memory_input = torch.randn(1, model.memory.embedding_dim)
    logits = model(input_vectors, memory_input)
    assert logits.shape == (1, 5, model.gpt2_config.vocab_size)  # use vocab_size instead of output_dim


# # Test the addition and deletion of embeddings in the memory
# def test_memory_operations():
#     model = VardaGPTAssociative(use_gpu=False)
#     memory = AssociativeMemory(memory_size=10, embedding_dim=3)

#     # Test addition
#     initial_size = memory.get_all_embeddings().shape[0]
#     memory.add(torch.tensor([[1, 2, 3]]))
#     new_size = memory.get_all_embeddings().shape[0]
#     assert new_size == initial_size + 1

#     # Test deletion
#     memory.remove([0])
#     final_size = memory.get_all_embeddings().shape[0]
#     assert final_size == new_size - 1
