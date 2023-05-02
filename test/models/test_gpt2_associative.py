# type: ignore
import pytest
import torch
from src.models.gpt2_associative import VardaGPTAssociative
from transformers import GPT2Config


@pytest.fixture
def varda_gpt_associative():
    return VardaGPTAssociative(gpt2_model_name="gpt2", use_gpu=False, batch_size=2)


def test_initialization(varda_gpt_associative):
    assert isinstance(varda_gpt_associative, VardaGPTAssociative)
    assert varda_gpt_associative.device.type == "cpu"
    assert isinstance(varda_gpt_associative.gpt2_config, GPT2Config)
    assert varda_gpt_associative.num_search_results == 5
    assert varda_gpt_associative.forgetfulness_factor == 0.001


def test_forward_pass_no_memory(varda_gpt_associative):
    batch_size = 2
    sequence_length = 4
    input_dim = varda_gpt_associative.gpt2_config.n_embd

    input_vectors = torch.randn(batch_size, sequence_length, input_dim)
    logits = varda_gpt_associative.forward(input_vectors)

    assert logits.shape == (batch_size, sequence_length, varda_gpt_associative.gpt2_config.vocab_size)


def test_forward_pass_with_memory(varda_gpt_associative):
    batch_size = 2
    sequence_length = 4
    input_dim = varda_gpt_associative.gpt2_config.n_embd

    input_vectors = torch.randn(batch_size, sequence_length, input_dim)
    logits = varda_gpt_associative.forward(input_vectors)

    assert logits.shape == (batch_size, sequence_length, varda_gpt_associative.gpt2_config.vocab_size)
