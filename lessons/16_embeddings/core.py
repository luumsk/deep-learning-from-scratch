
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


def load_grouped_vocab(path: Path) -> dict[str, list[str]]:
    """Load grouped vocabulary from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_sentences(path: Path) -> list[str]:
    """Load one sentence per line."""
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def flatten_vocab(grouped_vocab: dict[str, list[str]]) -> list[str]:
    """Flatten grouped vocabulary into a unique ordered token list."""
    tokens: list[str] = []
    seen: set[str] = set()

    for words in grouped_vocab.values():
        for word in words:
            if word not in seen:
                tokens.append(word)
                seen.add(word)

    return tokens


def build_token_to_id(tokens: list[str]) -> dict[str, int]:
    """Map each token to a unique integer ID."""
    return {token: idx for idx, token in enumerate(tokens)}


def tokenize_sentence(sentence: str, tokens: list[str]) -> list[str]:
    """Tokenize a generated sentence using longest-match lookup."""
    sorted_tokens = sorted(tokens, key=len, reverse=True)
    remaining = sentence
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


def build_training_pairs(
    sentences: list[str],
    tokens: list[str],
    token_to_id: dict[str, int],
    window_size: int,
) -> list[tuple[int, int]]:
    """Create skip-gram center-context pairs from tokenized sentences."""
    pairs: list[tuple[int, int]] = []

    for sentence in sentences:
        words = tokenize_sentence(sentence, tokens)
        ids = [token_to_id[word] for word in words if word in token_to_id]

        for center_idx, center_id in enumerate(ids):
            left = max(0, center_idx - window_size)
            right = min(len(ids), center_idx + window_size + 1)

            for context_idx in range(left, right):
                if context_idx == center_idx:
                    continue
                context_id = ids[context_idx]
                pairs.append((center_id, context_id))

    return pairs


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def cross_entropy(probs: np.ndarray, target_id: int) -> float:
    """Return the negative log-likelihood for the target class."""
    eps = 1e-12
    return float(-np.log(probs[target_id] + eps))


def train_skipgram_embeddings(
    training_pairs: list[tuple[int, int]],
    vocab_size: int,
    embedding_dim: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> tuple[Embedding, np.ndarray, list[float]]:
    """Train a tiny skip-gram model with full softmax."""
    np.random.seed(seed)
    embedding = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    output_weight = np.random.randn(embedding_dim, vocab_size) * 0.01
    loss_history: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0

        for center_id, target_id in training_pairs:
            center_array = np.array(center_id, dtype=np.int64)
            center_vector = embedding.forward(center_array)
            logits = center_vector @ output_weight
            probs = softmax(logits)
            loss = cross_entropy(probs, target_id)
            total_loss += loss

            grad_logits = probs.copy()
            grad_logits[target_id] -= 1.0

            grad_output_weight = np.outer(center_vector, grad_logits)
            grad_center_vector = output_weight @ grad_logits

            embedding.backward(
                grad_output=grad_center_vector,
                token_ids=center_array,
            )
            embedding.step(learning_rate)
            output_weight -= learning_rate * grad_output_weight

        avg_loss = total_loss / len(training_pairs)
        loss_history.append(avg_loss)

    return embedding, output_weight, loss_history


def lookup_embedding(
    word: str,
    token_to_id: dict[str, int],
    embedding: Embedding,
) -> np.ndarray:
    """Return the learned embedding vector for one word."""
    word_id = token_to_id[word]
    return embedding.weight[word_id]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def find_nearest_words(
    query_word: str,
    token_to_id: dict[str, int],
    tokens: list[str],
    embedding: Embedding,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find nearest words using learned embedding vectors."""
    query_vector = lookup_embedding(query_word, token_to_id, embedding)
    scores: list[tuple[str, float]] = []

    for token in tokens:
        if token == query_word:
            continue
        candidate_vector = lookup_embedding(token, token_to_id, embedding)
        score = cosine_similarity(query_vector, candidate_vector)
        scores.append((token, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def prepare_dataset(
    vocab_path: Path,
    sentences_path: Path,
    window_size: int,
) -> tuple[list[str], dict[str, int], list[str], list[tuple[int, int]]]:
    """Load files and prepare vocabulary and training pairs."""
    grouped_vocab = load_grouped_vocab(vocab_path)
    sentences = load_sentences(sentences_path)
    tokens = flatten_vocab(grouped_vocab)
    token_to_id = build_token_to_id(tokens)
    training_pairs = build_training_pairs(
        sentences=sentences,
        tokens=tokens,
        token_to_id=token_to_id,
        window_size=window_size,
    )
    return tokens, token_to_id, sentences, training_pairs
