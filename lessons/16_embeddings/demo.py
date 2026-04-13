from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core import find_nearest_words
from core import lookup_embedding
from core import prepare_dataset
from core import train_skipgram_embeddings


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "vietnamese_food"
VOCAB_PATH = DATA_DIR / "vocab.json"
SENTENCES_PATH = DATA_DIR / "sentences.txt"

RNG_SEED = 7
EMBEDDING_DIM = 8
LEARNING_RATE = 0.1
EPOCHS = 20
WINDOW_SIZE = 1
TOP_K = 5


def print_dataset_summary(
    tokens: list[str],
    sentences: list[str],
    training_pairs: list[tuple[int, int]],
) -> None:
    """Print basic dataset information."""
    print("Dataset summary")
    print(f"- Vocabulary size: {len(tokens)}")
    print(f"- Number of sentences: {len(sentences)}")
    print(f"- Number of training pairs: {len(training_pairs)}")
    print()


def print_loss_history(loss_history: list[float]) -> None:
    """Print average loss for each epoch."""
    print("Training loss")
    for epoch_idx, loss_value in enumerate(loss_history, start=1):
        print(f"Epoch {epoch_idx:02d} | avg_loss={loss_value:.4f}")
    print()


def print_lookup_demo(
    words: list[str],
    token_to_id: dict[str, int],
    embedding,
) -> None:
    """Print learned vectors for selected words."""
    print("Embedding lookup demo")
    for word in words:
        word_id = token_to_id[word]
        vector = lookup_embedding(
            word=word,
            token_to_id=token_to_id,
            embedding=embedding,
        )
        print(f"Word: {word}")
        print(f"ID: {word_id}")
        print(f"Vector: {np.round(vector, 4)}")
        print()


def print_nearest_neighbors_demo(
    query_words: list[str],
    token_to_id: dict[str, int],
    tokens: list[str],
    embedding,
) -> None:
    """Print nearest words for selected query words."""
    print("Nearest neighbors")
    for query_word in query_words:
        print(f"Nearest words to '{query_word}':")
        neighbors = find_nearest_words(
            query_word=query_word,
            token_to_id=token_to_id,
            tokens=tokens,
            embedding=embedding,
            top_k=TOP_K,
        )
        for word, score in neighbors:
            print(f"  {word:12s} cosine={score:.4f}")
        print()


def plot_loss_curve(loss_history: list[float]) -> None:
    """Plot training loss by epoch."""
    epochs = np.arange(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Skip-gram training loss")
    plt.tight_layout()
    plt.show()


def plot_selected_embeddings(
    words: list[str],
    token_to_id: dict[str, int],
    embedding,
) -> None:
    """Plot selected 2D embeddings using the first two dimensions."""
    plt.figure(figsize=(7, 7))

    for word in words:
        vector = lookup_embedding(
            word=word,
            token_to_id=token_to_id,
            embedding=embedding,
        )
        x_coord = vector[0]
        y_coord = vector[1]
        plt.scatter(x_coord, y_coord)
        plt.text(x_coord + 0.01, y_coord + 0.01, word, fontsize=10)

    plt.xlabel("Embedding dim 1")
    plt.ylabel("Embedding dim 2")
    plt.title("Selected learned word embeddings")
    plt.tight_layout()
    plt.show()


def main() -> None:
    tokens, token_to_id, sentences, training_pairs = prepare_dataset(
        vocab_path=VOCAB_PATH,
        sentences_path=SENTENCES_PATH,
        window_size=WINDOW_SIZE,
    )

    embedding, _, loss_history = train_skipgram_embeddings(
        training_pairs=training_pairs,
        vocab_size=len(tokens),
        embedding_dim=EMBEDDING_DIM,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        seed=RNG_SEED,
    )

    print_dataset_summary(
        tokens=tokens,
        sentences=sentences,
        training_pairs=training_pairs,
    )
    print_loss_history(loss_history)

    print_lookup_demo(
        words=["phở", "bún", "cà phê sữa", "nước mắm"],
        token_to_id=token_to_id,
        embedding=embedding,
    )

    print_nearest_neighbors_demo(
        query_words=["phở", "cà phê sữa", "bò"],
        token_to_id=token_to_id,
        tokens=tokens,
        embedding=embedding,
    )

    plot_loss_curve(loss_history)

    plot_selected_embeddings(
        words=[
            "phở",
            "bún",
            "hủ tiếu",
            "cà phê",
            "trà đá",
            "nước mắm",
            "chanh",
            "bò",
            "gà",
        ],
        token_to_id=token_to_id,
        embedding=embedding,
    )


if __name__ == "__main__":
    main()