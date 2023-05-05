# VardaGPT

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [VardaGPT](#vardagpt)
  - [TLDR - Training](#tldr---training)
    - [Requirements](#requirements)
    - [Usage](#usage)
  - [Overview](#overview)
  - [Models](#models)
  - [Training, Evaluation, and Fine-tuning Process](#training-evaluation-and-fine-tuning-process)
    - [1. Data Preparation](#1-data-preparation)
    - [2. GPT-2 Model Adaptation](#2-gpt-2-model-adaptation)
    - [3. Training](#3-training)
    - [4. Evaluation](#4-evaluation)
    - [5. Fine-tuning (if necessary)](#5-fine-tuning-if-necessary)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage-1)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [Contributing](#contributing)
  - [Code Formatting and Pre-commit](#code-formatting-and-pre-commit)
    - [Setup](#setup-1)
    - [Using Pre-commit](#using-pre-commit)
  - [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

VardaGPT is a memory-enhanced GPT-2 model powered by Hugging Face Transformers
and FAISS. Inspired by J.R.R. Tolkien's Silmarillion, VardaGPT aims to provide
guidance and knowledge through its memory-augmented text generation
capabilities.

## TLDR - Training

The `VardaGPTAssociative` model combines GPT-2 with an associative memory to
improve context retrieval. This repository includes a script to train this model
on the WikiText-2 dataset.

### Requirements

- Python 3.7+
- PyTorch 1.8.1+
- torchtext 0.9.1
- transformers 4.10.0
- rich 10.3.0
- faiss-cpu 1.7.1

To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

### Usage

To train the `VardaGPTAssociative` model on the WikiText-2 dataset, use the
provided training script (`train_varda_gpt_associative.py`). You can customize
the training settings by passing command-line arguments. Here's a basic example:

```bash
python train_varda_gpt_associative.py --epochs 5 --learning_rate 1e-4 --use_gpu
```

Available command-line arguments:

- `--epochs`: Number of epochs to train the model (default: 5).
- `--learning_rate`: Learning rate for the optimizer (default: 1e-4).
- `--memory_size`: Maximum number of items the associative memory can store
  (default: 10000).
- `--memory_dim`: Dimensionality of the embeddings stored in the associative
  memory (default: 768).
- `--index_type`: Type of index used for the associative memory (default:
  "flat").
- `--num_clusters`: Number of clusters to use for the memory if the index type
  is "ivf" (default: 1024).
- `--num_search_results`: Number of search results to return from the
  associative memory (default: 5).
- `--use_gpu`: Whether to use the GPU for the model if available (default:
  False).
- `--batch_size`: Batch size for training (default: 1).
- `--forgetfulness_factor`: Forgetfulness factor for the associative memory
  (default: 0.001).

During training, the script will periodically print the training loss,
validation loss, and elapsed time for each epoch, along with a progress bar for
each training step.

After training, you can use the trained model for your specific use case, such
as text generation or fine-tuning for a particular task.

## Overview

<details>
  <summary>Click me</summary>

```plantuml
@startuml
!define AWSPUML https://raw.githubusercontent.com/awslabs/aws-icons-for-plantuml/v14.0

actor User

skinparam component {
  BackgroundColor<<Data Preparation>> LightSkyBlue
  BackgroundColor<<FAISS Memory>> Plum
  BackgroundColor<<GPT-2 Adaptation>> LightGreen
  BackgroundColor<<Training>> LightSalmon
  BackgroundColor<<Inference>> LightCoral
  BorderColor Black
  FontName Arial
}

package "VardaGPT" {
  [Data Preparation]<<Data Preparation>> --> [FAISS Memory]<<FAISS Memory>>
  [Data Preparation]<<Data Preparation>> --> [GPT-2 Adaptation]<<GPT-2 Adaptation>>

  [FAISS Memory]<<FAISS Memory>> --> [GPT-2 Adaptation]<<GPT-2 Adaptation>>
  [GPT-2 Adaptation]<<GPT-2 Adaptation>> --> [Training]<<Training>>

  [Training]<<Training>> --> [Inference]<<Inference>>
  [FAISS Memory]<<FAISS Memory>> --> [Inference]<<Inference>>

  User --> [Data Preparation]<<Data Preparation>> : Dataset
  User --> [Inference]<<Inference>> : Prompts
}

@enduml
```

</details>

![overview](./assets/README.svg)

This diagram shows the main components of the VardaGPT project and their
interactions. The Data Preparation component processes the dataset and feeds it
to both the FAISS Memory Model and the GPT-2 Model Adaptation component. The
FAISS Memory Model generates embeddings, which are used by the GPT-2 Model
Adaptation component to create a modified GPT-2 model. The modified GPT-2 model
is then trained and evaluated, and the final trained model is used in the
Inference and Application component. The user provides the dataset and prompts
for text generation.

## Models

The associative memory model:

<details>
  <summary>Click me</summary>

```plantuml
@startuml

rectangle "Input Vectors" as input #b3e0ff
rectangle "Memory" as memory #f2d7b9
rectangle "Concatenated Input" as concatenated_input #f6e3c6
rectangle "Fully Connected Layer (fc)" as fc #e5ebf0
rectangle "GPT-2 Transformer" as transformer #c6e0b4
rectangle "GPT-2 LM Head" as lm_head #c9daf8
rectangle "Fully Connected Layer\n(fc_storable_vector)" as fc_storable_vector #c9daf8
rectangle "Fully Connected Layer\n(fc_store_decision)" as fc_store_decision #c9daf8

input -down-> memory : Perform search in memory
memory -down-> concatenated_input : Concatenate search results with input vectors
concatenated_input -down-> fc : Apply fully connected layer (fc)
fc -down-> transformer : Pass through GPT-2 transformer
transformer -down-> lm_head : Apply GPT-2 lm_head
transformer -right-> fc_storable_vector : Apply fully connected layer (fc_storable_vector)
transformer -right-> fc_store_decision : Apply fully connected layer (fc_store_decision)

note right of fc_storable_vector: Calculate storable vector\n and store decision
note right of fc_store_decision: Store the storable_vector in\n the associative memory if\n the store_decision is affirmative
note bottom of lm_head: Return logits

@enduml

```

</details>

![model1](./assets/README_001.svg)

<details>
  <summary>Click me</summary>

```plantuml
@startuml
title Forward Function

!define Tensor(t,d) t + " (" + d + ")"
!define DEVICE "device"

actor "input_vectors" as input_vectors
actor "memory_input" as memory_input

note right of input_vectors
  Tensor:
  (batch_size, seq_len, embedding_dim)
end note

note right of memory_input
  Tensor (optional):
  (batch_size, seq_len, embedding_dim)
end note

input_vectors -> DEVICE
memory_input -> DEVICE

DEVICE -> "search(memory_input)" as search
search --> "indices, distances" as search_result
note right of search_result
  Tensors:
  indices: (batch_size, seq_len, num_search_results)
  distances: (batch_size, seq_len, num_search_results)
end note

search_result -> "get_all_embeddings()" as all_embeddings
note right of all_embeddings
  Tensor:
  (memory_size, embedding_dim)
end note

all_embeddings -> "search_results" as search_results
note right of search_results
  Tensor:
  (batch_size, seq_len, search_results_dim)
end note

search_results --> "concatenate(input_vectors, search_results)" as concatenated_input
note right of concatenated_input
  Tensor:
  (batch_size, seq_len, embedding_dim + search_results_dim)
end note

concatenated_input --> "self.fc(concatenated_input)" as fc_output
note right of fc_output
  Tensor:
  (batch_size, seq_len, embedding_dim)
end note

fc_output --> "self.gpt2_model.transformer(inputs_embeds=input_vectors)" as transformer_outputs
transformer_outputs --> "hidden_states" as hidden_states
note right of hidden_states
  Tensor:
  (batch_size, seq_len, embedding_dim)
end note

hidden_states --> "self.gpt2_model.lm_head(hidden_states)" as logits
note right of logits
  Tensor:
  (batch_size, seq_len, vocab_size)
end note

hidden_states --> "self.fc_storable_vector(hidden_states)" as storable_vector
note right of storable_vector
  Tensor:
  (batch_size, seq_len, memory_dim)
end note

hidden_states --> "self.fc_store_decision(hidden_states)" as store_decision
note right of store_decision
  Tensor:
  (batch_size, seq_len, 1)
end note

hidden_states --> "self.fc_delete_decision(hidden_states)" as delete_decision
note right of delete_decision
  Tensor:
  (batch_size, seq_len, num_search_results)
end note

hidden_states --> "self.fc_deletable_vector(hidden_states)" as deletable_vector
note right of deletable_vector
  Tensor:
  (batch_size, seq_len, memory_dim)
end note

storable_vector --> "self.memory.add(storable_vector_to_store)" as add_memory

deletable_vector --> "calculate L2 distances" as l2_distances
note right of l2_distances
  Tensor:
  (batch_size, num_search_results)
end note

l2_distances --> "threshold comparison" as threshold_comparison
note right of threshold_comparison
  Tensor (bool):
  (batch_size, num_search_results)
end note

threshold_comparison --> "self.memory.remove(indices_to_delete_flat)" as remove_memory

logits --> "return logits" as return_logits

@enduml
```

</details>

![model](./assets/README_002.svg)

## Training, Evaluation, and Fine-tuning Process

<details>
  <summary>Click me</summary>

```plantuml
@startuml

skinparam activity {
  BackgroundColor LightSkyBlue
  BorderColor Black
  FontName Arial
}

start

:Data Preparation;

partition "FAISS Memory Model" {
  :Create FAISS Index;
  :Encode and Decode Text Data;
  :Test FAISS Index;
}

partition "GPT-2 Model Adaptation" {
  :Load Pre-trained GPT-2 Model;
  :Modify GPT-2 Architecture;
  :Define Custom Loss Function;
}

partition "Training" {
  :Train Adapted GPT-2 Model;
  :Save Model Checkpoints;
}

partition "Evaluation" {
  :Evaluate Model on Testing Set;
  :Calculate Metrics;
}

if (Fine-tuning needed?) then (Yes)
  partition "Fine-tuning" {
    :Adjust Hyperparameters;
    :Iterate Training and Evaluation;
  }
endif

partition "Inference and Application" {
  :Inference Function;
  :API or Interface;
}

stop

@enduml
```

</details>

![process](./assets/README_003.svg)

### 1. Data Preparation

- Collect and preprocess a dataset for training, evaluation, and fine-tuning.
- Split the dataset into training, validation, and testing sets.
- Create data loaders for handling data.

### 2. GPT-2 Model Adaptation

- Load a pre-trained GPT-2 model from Hugging Face Transformers.
- Modify the GPT-2 model architecture to incorporate the FAISS memory model.
- Define a custom loss function that considers both the GPT-2 model's output and
  the memory model.

### 3. Training

- Set up the training loop and train the adapted GPT-2 model.
- Save model checkpoints and track training metrics (loss, perplexity, etc.).
- Monitor the training progress, validate the model on the validation set, and
  perform early stopping if necessary.

### 4. Evaluation

- Evaluate the trained model on the testing set.
- Calculate evaluation metrics (e.g., perplexity, accuracy, F1-score).

### 5. Fine-tuning (if necessary)

- If the model's performance on the testing set is not satisfactory, fine-tune
  the model with different hyperparameters, learning rates, or architectures.
- Iterate through the training and evaluation steps until the desired
  performance is achieved.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- FAISS (CPU or GPU version)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/VardaGPT.git
cd VardaGPT
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Directory Structure

- `src/`: Contains the Python source code for the project.
- `data/`: Stores the datasets used for training and evaluation.
- `models/`: Holds the trained models and their checkpoints.

## Usage

### Data Preparation

1. Place your dataset in the `data/` directory.
2. Preprocess and split your dataset into training, validation, and testing sets
   using the provided scripts in `src/`.

### Training

1. Configure the training settings and model hyperparameters in the
   `src/config.py` file.
2. Run the training script:

```bash
python src/train.py
```

3. Monitor the training progress and save model checkpoints in the `models/`
   directory.

### Evaluation

1. Evaluate the trained model on the validation and testing sets using the
   provided evaluation script:

```bash
python src/evaluate.py
```

### Inference

1. Use the provided inference script to generate text with the memory-enhanced
   GPT-2 model:

```bash
python src/inference.py --prompt "Your prompt text here"
```

## Contributing

Feel free to contribute to this project by submitting pull requests or opening
issues for bug reports and feature requests.

## Code Formatting and Pre-commit

This project uses `black`, `flake8`, and `mypy` for Python code formatting and
linting. We also use `prettier` to format JSON and Markdown files. The
configuration for these tools is in the `.pre-commit-config.yaml` file.

### Setup

1. Install `pre-commit` if you haven't already:

```bash
pip install pre-commit
```

2. Set up the git hooks:

```bash
pre-commit install
```

### Using Pre-commit

Whenever you commit changes, the pre-commit hooks will automatically format your
code and check for issues. If the hooks detect any problems, the commit will be
aborted, and you'll see a list of issues that need to be fixed. Once you've
resolved the issues, you can try committing again.

You can also run the pre-commit hooks manually on all files:

```bash
pre-commit run --all-files
```

Or run the hooks on specific files:

```bash
pre-commit run --files <file1> <file2>
```

By following this setup and using pre-commit hooks, you can ensure that the code
in the repository remains consistently formatted and adheres to the project's
coding standards.

## License

This project is licensed under the [MIT License](LICENSE).
