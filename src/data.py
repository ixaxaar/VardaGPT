from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.datasets import WikiText2
from transformers import GPT2Tokenizer


def load_wikitext2() -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """
    Load the WikiText-2 dataset for training, validation, and testing.

    :return: A tuple of three DataLoaders for train, valid, and test sets.
    """

    # Define a tokenizer using the GPT2Tokenizer from the transformers library
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Define a tokenize function using the GPT2Tokenizer
    def tokenize(text: str) -> torch.Tensor:
        return tokenizer.encode(text, return_tensors="pt").squeeze(0)  # type: ignore

    # Use torchtext's get_tokenizer function to create a tokenizer pipeline
    text_pipeline = get_tokenizer(tokenize)

    # Load the WikiText-2 dataset
    train_data, valid_data, test_data = WikiText2()

    # Create DataLoaders for the train, valid, and test sets
    train_loader = DataLoader([text_pipeline(text) for text in train_data], batch_size=1, shuffle=True)  # type: ignore
    valid_loader = DataLoader([text_pipeline(text) for text in valid_data], batch_size=1, shuffle=True)  # type: ignore
    test_loader = DataLoader([text_pipeline(text) for text in test_data], batch_size=1, shuffle=True)  # type: ignore

    return train_loader, valid_loader, test_loader
