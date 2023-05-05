from typing import Any

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from memory.batch_associative import BatchAssociativeMemory


class VardaGPTAssociative(nn.Module):
    def __init__(
        self,
        gpt2_model_name: str = "gpt2",
        memory_size: int = 10000,
        memory_dim: int = 768,
        index_type: str = "flat",
        num_clusters: int = 1024,
        num_search_results: int = 5,
        use_gpu: bool = False,
        batch_size: int = 1,
        forgetfulness_factor: float = 0.001,
    ):
        """
        Initialize a GPT-2 model with associative memory.

        :param gpt2_model_name: The name of the GPT-2 model to load. Default is "gpt2".
        :param memory_size: The maximum number of items the associative memory can store. Default is 10000.
        :param memory_dim: The dimensionality of the embeddings stored in the associative memory. Default is 768.
        :param index_type: The type of index used for the associative memory. Default is "flat".
        :param num_clusters: The number of clusters to use for the memory if the index type is "ivf". Default is 1024.
        :param num_search_results: The number of search results to return from the associative memory. Default is 5.
        :param use_gpu: Whether to use the GPU for the model if available. Default is False.
        """
        super(VardaGPTAssociative, self).__init__()

        # Set up the device for the model
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Load the GPT-2 model and configuration
        self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        # Initialize the BatchAssociativeMemory module
        self.memory = BatchAssociativeMemory(
            num_batches=batch_size,
            memory_size=memory_size,
            embedding_dim=memory_dim,
            index_type=index_type,
            num_clusters=num_clusters,
            use_gpu=use_gpu,
            forgetfulness_factor=forgetfulness_factor,
        )

        # Define dimensions for search results and output
        self.search_results_dim = memory_dim * num_search_results

        # Linear layers for concatenated input, storable vector, and store decision
        self.fc = nn.Linear(self.gpt2_config.n_embd + self.search_results_dim, self.gpt2_config.n_embd)
        self.fc_storable_vector = nn.Linear(self.gpt2_config.n_embd, memory_dim)
        self.fc_store_decision = nn.Linear(self.gpt2_config.n_embd, 1)

        # Move all layers to the device
        self.to(self.device)

        self.memory_dim = memory_dim
        self.num_search_results = num_search_results
        self.forgetfulness_factor = forgetfulness_factor

    def forward(self, input_vectors: torch.Tensor) -> Any:
        """
        Perform a forward pass through the GPT-2 model with associative memory.

        :param input_vectors: A 3D tensor of shape (batch_size, sequence_length, input_dim) containing
            the input vectors for each token in the batch.
        :param memory_input: A 2D tensor of shape (batch_size, memory_dim) containing the memory input for each item
            in the batch. If not provided, memory will not be used.
        :return: A 3D tensor of shape (batch_size, sequence_length, vocab_size) containing
        the logits from the GPT-2 model.
        """
        input_vectors = input_vectors.to(self.device)
        batch_size, seq_len, _ = input_vectors.shape

        # Initialize search_results tensor with the correct shape
        search_results = torch.zeros((batch_size, seq_len, self.search_results_dim), device=self.device)

        # Search for relevant results for each item in the batch
        for t in range(seq_len):
            search_results_list = self.memory.batch_search(input_vectors[:, t, :].squeeze(1), self.num_search_results)
            retrieved_embeddings_list = []
            # Retrieve and concatenate search results with input vectors
            for ctr, (indices, _) in enumerate(search_results_list):
                retrieved_embeddings = torch.cat(
                    [
                        self.memory.memories[ctr][i].unsqueeze(0)
                        if i >= 0
                        else torch.zeros(self.memory_dim).unsqueeze(0)
                        for i in indices.squeeze()
                    ],
                    dim=0,
                )
                # Update the corresponding search_results tensor
                retrieved_embeddings_list.append(retrieved_embeddings)
            search_results[:, t, :] = torch.cat(retrieved_embeddings_list, dim=0).view(batch_size, -1)

        concatenated_input = torch.cat([input_vectors, search_results], dim=-1)

        input_vectors = self.fc(concatenated_input)

        # Pass input_vectors through GPT-2 model's transformer and obtain hidden states
        transformer_outputs = self.gpt2_model.transformer(inputs_embeds=input_vectors)
        hidden_states = transformer_outputs.last_hidden_state

        # Get logits from hidden states
        logits = self.gpt2_model.lm_head(hidden_states)

        # Calculate storable vector and store decision
        storable_vector = self.fc_storable_vector(hidden_states)
        store_decision = self.fc_store_decision(hidden_states)

        # Store the storable_vector in the associative memory if the store_decision is affirmative
        store_threshold = 0.5  # Define a threshold for store decision
        store_mask = (store_decision > store_threshold).float()
        storable_vector_to_store = storable_vector * store_mask

        for i in range(seq_len):
            storable_vector_to_store_i = storable_vector_to_store[:, i, :].view(batch_size, -1).detach()
            self.memory.batch_add(storable_vector_to_store_i)

        # Randomly forget items from the memory with a specified probability
        for memory in self.memory.memories:
            memory.forget_randomly()

        return logits
