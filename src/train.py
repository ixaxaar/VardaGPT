import argparse
import time
from typing import Any

import torch
import torch.optim as optim
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.theme import Theme
from torch.utils.data import DataLoader

from data import load_wikitext2
from models.gpt2_associative import VardaGPTAssociative


def train(
    model: VardaGPTAssociative,
    train_loader: DataLoader[Any],
    valid_loader: DataLoader[Any],
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """
    Train the model with the given training and validation data.

    :param model: The VardaGPTAssociative model to be trained.
    :param train_loader: DataLoader for the training data.
    :param valid_loader: DataLoader for the validation data.
    :param epochs: Number of epochs to train the model.
    :param lr: Learning rate for the optimizer.
    :param device: The device to use for training (CPU or GPU).
    """

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # Create a console object for printing colorful prompts
    theme = Theme({"info": "dim green", "warning": "yellow", "error": "bold red"})
    console = Console(theme=theme)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for _, batch in enumerate(track(train_loader, description=f"[bold][info]Epoch {epoch + 1}")):
            # Move the input to the device
            input_vectors = batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_vectors)

            # Calculate loss
            loss = loss_function(logits.view(-1, logits.shape[-1]), input_vectors.view(-1))

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average epoch loss
        average_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for _, batch in enumerate(valid_loader):
                input_vectors = batch.to(device)
                logits = model(input_vectors)
                loss = loss_function(logits.view(-1, logits.shape[-1]), input_vectors.view(-1))
                valid_loss += loss.item()

        # Calculate average validation loss
        average_valid_loss = valid_loss / len(valid_loader)

        # Print epoch summary
        table = Table(title=f"Epoch {epoch + 1} Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="bold")
        table.add_row("Training Loss", f"{average_epoch_loss:.4f}")
        table.add_row("Validation Loss", f"{average_valid_loss:.4f}")
        table.add_row("Time", f"{epoch_time:.2f} seconds")
        console.print(table)


if __name__ == "__main__":
    console = Console()

    console.print(Panel.fit("[bold blue]VardaGPTAssociative Training Script[/bold blue]"))

    description = """\
This script trains a VardaGPTAssociative model on the WikiText-2 dataset. The model combines GPT-2 with an associative memory to improve context retrieval.
"""
    console.print(Markdown(description))

    parser = argparse.ArgumentParser(description="Train VardaGPTAssociative model on WikiText-2 dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument(
        "--memory_size", type=int, default=10000, help="Maximum number of items the associative memory can store"
    )
    parser.add_argument(
        "--memory_dim", type=int, default=768, help="Dimensionality of the embeddings stored in the associative memory"
    )
    parser.add_argument("--index_type", type=str, default="flat", help="Type of index used for the associative memory")
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1024,
        help="Number of clusters to use for the memory if the index type is 'ivf'",
    )
    parser.add_argument(
        "--num_search_results",
        type=int,
        default=5,
        help="Number of search results to return from the associative memory",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use the GPU for the model if available")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--forgetfulness_factor", type=float, default=0.001, help="Forgetfulness factor for the associative memory"
    )

    args = parser.parse_args()

    console.print("[bold green]Training settings:[/bold green]")
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  Learning rate: {args.learning_rate}")

    model = VardaGPTAssociative(
        gpt2_model_name="gpt2",
        memory_size=args.memory_size,
        memory_dim=args.memory_dim,
        index_type=args.index_type,
        num_clusters=args.num_clusters,
        num_search_results=args.num_search_results,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        forgetfulness_factor=args.forgetfulness_factor,
    )

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model.to(device)

    train_loader, valid_loader, test_loader = load_wikitext2()

    # Train the model
    train(model, train_loader, valid_loader, args.epochs, args.learning_rate, device)
