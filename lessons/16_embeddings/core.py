from __future__ import annotations

import numpy as np


class Embedding:
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
    
    def backward(self,
                 grad_output: np.ndarray,
                 token_ids: np.ndarray | None = None) -> None:
        
        # If token_ids is not provided, use the last token IDs from the forward pass.
        if token_ids is None:
            if self.last_token_ids is None:
                raise ValueError(
                    "token_ids must be provided before backward."
                )
            token_ids = self.last_token_ids
        
        # Reset gradients to zero before accumulation
        self.zero_grad()

        # Accumulate gradients for the embedding table
        # if the same words appear multiple times in the batch,
        # their gradients will be summed together.
        np.add.at(self.grad_weight, token_ids, grad_output)

    def step(self, learning_rate: float) -> None:
        self.weight -= learning_rate * self.grad_weight

    def zero_grad(self) -> None:
        self.grad_weight.fill(0.0)
