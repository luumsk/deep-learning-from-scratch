import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def cross_entropy(probs: np.ndarray, target_id: int) -> float:
    """Return the negative log-likelihood for the target class."""
    eps = 1e-12
    return float(-np.log(probs[target_id] + eps))

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))