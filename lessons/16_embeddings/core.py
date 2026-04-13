from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from utils import cross_entropy, softmax, cosine_similarity


class Embedding:
    """A minimal trainable embedding table."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = (
            np.random.randn(vocab_size, embedding_dim) * 0.01
        )
        self.grad_weight = np.zeros_like(self.weight)
        self.last_token_ids: np.ndarray | None = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embedding vectors for token IDs."""
        self.last_token_ids = token_ids
        return self.weight[token_ids]

    def get_vector(self, token_id: int) -> np.ndarray:
        """Return embedding vector for a single token ID."""
        return self.weight[token_id]

    def backward(
        self,
        grad_output: np.ndarray,
        token_ids: np.ndarray | None = None,
    ) -> None:
        """Accumulate gradients for the embedding table."""
        if token_ids is None:
            if self.last_token_ids is None:
                raise ValueError(
                    "token_ids must be provided before backward."
                )
            token_ids = self.last_token_ids

        self.zero_grad()
        np.add.at(self.grad_weight, token_ids, grad_output)

    def step(self, learning_rate: float) -> None:
        """Update embedding weights with gradient descent."""
        self.weight -= learning_rate * self.grad_weight

    def zero_grad(self) -> None:
        """Reset stored gradients to zero."""
        self.grad_weight.fill(0.0)


class Tokenizer:
    """Store tokens and provide tokenization utilities."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.token_to_id = {
            token: idx for idx, token in enumerate(tokens)
        }

    @classmethod
    def from_grouped_vocab(
        cls,
        grouped_vocab: dict[str, list[str]],
    ) -> "Tokenizer":
        """Build a tokenizer from grouped tokens."""
        tokens: list[str] = []
        seen: set[str] = set()

        for words in grouped_vocab.values():
            for word in words:
                if word not in seen:
                    tokens.append(word)
                    seen.add(word)

        return cls(tokens)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a sentence using longest-match lookup."""
        sorted_tokens = sorted(
            self.tokens,
            key=len,
            reverse=True,
        )
        remaining = text
        output: list[str] = []

        while remaining:
            remaining = remaining.strip()
            if not remaining:
                break

            matched_token: str | None = None
            for token in sorted_tokens:
                if remaining.startswith(token):
                    matched_token = token
                    break

            if matched_token is None:
                next_space = remaining.find(" ")
                if next_space == -1:
                    output.append(remaining)
                    break
                output.append(remaining[:next_space])
                remaining = remaining[next_space + 1 :]
                continue

            output.append(matched_token)
            remaining = remaining[len(matched_token) :]

        return output

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize(text)
        return [
            self.token_to_id[token]
            for token in tokens
            if token in self.token_to_id
        ]

    @property
    def vocab_size(self) -> int:
        """Return the number of known tokens."""
        return len(self.tokens)


class Trainer:
    """Train an embedding model with a simple skip-gram objective."""

    def __init__(
        self,
        embedding: Embedding,
        learning_rate: float,
        epochs: int,
        seed: int,
    ) -> None:
        self.embedding = embedding
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.output_weight: np.ndarray | None = None
        self.loss_history: list[float] = []

    def train(
        self,
        training_pairs: list[tuple[int, int]],
    ) -> tuple[Embedding, np.ndarray, list[float]]:
        """Train the embedding model on skip-gram pairs."""
        np.random.seed(self.seed)
        self.embedding.weight = (
            np.random.randn(
                self.embedding.vocab_size,
                self.embedding.embedding_dim,
            ) * 0.01
        )
        self.embedding.zero_grad()
        self.output_weight = np.random.randn(
            self.embedding.embedding_dim,
            self.embedding.vocab_size,
        ) * 0.01
        self.loss_history = []

        for _ in range(self.epochs):
            total_loss = 0.0

            for center_id, target_id in training_pairs:
                center_array = np.array(center_id, dtype=np.int64)
                center_vector = self.embedding.forward(center_array)
                logits = center_vector @ self.output_weight
                probs = softmax(logits)
                loss = cross_entropy(probs, target_id)
                total_loss += loss

                grad_logits = probs.copy()
                grad_logits[target_id] -= 1.0

                grad_output_weight = np.outer(center_vector, grad_logits)
                grad_center_vector = self.output_weight @ grad_logits

                self.embedding.backward(
                    grad_output=grad_center_vector,
                    token_ids=center_array,
                )
                self.embedding.step(self.learning_rate)
                self.output_weight -= (
                    self.learning_rate * grad_output_weight
                )

            avg_loss = total_loss / len(training_pairs)
            self.loss_history.append(avg_loss)

        return self.embedding, self.output_weight, self.loss_history
    
    
def load_grouped_vocab(path: Path) -> dict[str, list[str]]:
    """Load grouped vocabulary from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_sentences(path: Path) -> list[str]:
    """Load one sentence per line."""
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def build_training_pairs(
    sentences: list[str],
    tokenizer: Tokenizer,
    window_size: int,
) -> list[tuple[int, int]]:
    """Create skip-gram center-context pairs from sentences."""
    pairs: list[tuple[int, int]] = []

    for sentence in sentences:
        ids = tokenizer.encode(sentence)

        for center_idx, center_id in enumerate(ids):
            left = max(0, center_idx - window_size)
            right = min(
                len(ids),
                center_idx + window_size + 1,
            )

            for context_idx in range(left, right):
                if context_idx == center_idx:
                    continue
                context_id = ids[context_idx]
                pairs.append((center_id, context_id))

    return pairs


def find_nearest_words(
    query_word: str,
    tokenizer: Tokenizer,
    embedding: Embedding,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find nearest words using cosine similarity."""
    word_id = tokenizer.token_to_id.get(query_word)
    if word_id is None:
        raise ValueError(
            f"Query word '{query_word}' not found in vocabulary."
        )

    query_vector = embedding.get_vector(word_id)
    scores: list[tuple[str, float]] = []

    for token in tokenizer.tokens:
        if token == query_word:
            continue
        candidate_vector = embedding.get_vector(
            tokenizer.token_to_id[token]
        )
        score = cosine_similarity(query_vector, candidate_vector)
        scores.append((token, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]

def build_dataset(
    vocab_path: Path,
    sentences_path: Path,
    window_size: int,
) -> tuple[Tokenizer, list[str], list[tuple[int, int]]]:
    """Load files and prepare tokenizer and training pairs."""
    grouped_vocab = load_grouped_vocab(vocab_path)
    sentences = load_sentences(sentences_path)
    tokenizer = Tokenizer.from_grouped_vocab(grouped_vocab)
    training_pairs = build_training_pairs(
        sentences=sentences,
        tokenizer=tokenizer,
        window_size=window_size,
    )
    return tokenizer, sentences, training_pairs
