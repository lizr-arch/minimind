"""
OddsMind Evaluation Metrics (P0.4 / P0.5A)

Pure PyTorch implementations — no sklearn, no numpy dependency.

Functions:
    accuracy_from_logits / accuracy_from_probs
    logloss_from_logits / logloss_from_probs
    brier_from_logits / brier_from_probs
    class_counts
    prediction_counts / prediction_counts_from_probs
"""

import math

import torch
import torch.nn.functional as F
from typing import Dict


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy from logits.

    Args:
        logits: [N, C] float tensor.
        labels: [N] long tensor with class indices.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if logits.shape[0] == 0:
        raise ValueError("Cannot compute accuracy on empty input")
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    return correct / logits.shape[0]


def logloss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Compute multi-class log-loss (cross-entropy) from logits.

    Args:
        logits: [N, C] float tensor.
        labels: [N] long tensor.
        eps: clamp threshold for numerical stability.

    Returns:
        Average log-loss as a float.
    """
    if logits.shape[0] == 0:
        raise ValueError("Cannot compute logloss on empty input")
    log_probs = F.log_softmax(logits, dim=-1)
    # Clamp for safety
    log_probs = log_probs.clamp(min=math.log(eps))
    nll = F.nll_loss(log_probs, labels, reduction="mean")
    return nll.item()


def brier_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Compute multi-class Brier score from logits.

    Brier = (1/N) * sum_i sum_c (p_ic - y_ic)^2

    Args:
        logits: [N, C] float tensor.
        labels: [N] long tensor.
        num_classes: number of classes.

    Returns:
        Brier score as a float (lower is better, 0 = perfect).
    """
    if logits.shape[0] == 0:
        raise ValueError("Cannot compute Brier on empty input")
    probs = F.softmax(logits, dim=-1)
    # One-hot encode labels
    y_onehot = F.one_hot(labels, num_classes=num_classes).float()
    squared_error = (probs - y_onehot).pow(2).sum(dim=-1)
    return squared_error.mean().item()


def class_counts(labels: torch.Tensor, num_classes: int) -> Dict[str, int]:
    """
    Count occurrences of each class.

    Args:
        labels: [N] long tensor.
        num_classes: number of classes.

    Returns:
        Dict mapping "class_0", "class_1", ... to counts.
    """
    counts = {}
    for c in range(num_classes):
        counts[f"class_{c}"] = int((labels == c).sum().item())
    counts["total"] = int(labels.shape[0])
    return counts


def prediction_counts(logits: torch.Tensor, num_classes: int) -> Dict[str, int]:
    """
    Count how often each class is predicted.

    Args:
        logits: [N, C] float tensor.
        num_classes: number of classes.

    Returns:
        Dict mapping "class_0", "class_1", ... to prediction counts.
    """
    preds = torch.argmax(logits, dim=-1)
    counts = {}
    for c in range(num_classes):
        counts[f"class_{c}"] = int((preds == c).sum().item())
    counts["total"] = int(preds.shape[0])
    return counts


# ── Probability-based metrics (P0.5A) ──────────────────────────────────

def accuracy_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy from probability tensor (not logits).

    Args:
        probs: [N, C] float tensor, each row sums to ~1.
        labels: [N] long tensor.
    """
    if probs.shape[0] == 0:
        raise ValueError("Cannot compute accuracy on empty input")
    preds = torch.argmax(probs, dim=-1)
    correct = (preds == labels).sum().item()
    return correct / probs.shape[0]


def logloss_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Compute multi-class log-loss directly from probabilities.

    logloss = -(1/N) * sum_i log(p_i[y_i])

    Args:
        probs: [N, C] float tensor (softmaxed).
        labels: [N] long tensor.
        eps: clamp threshold.
    """
    if probs.shape[0] == 0:
        raise ValueError("Cannot compute logloss on empty input")
    probs = probs.clamp(min=eps, max=1.0 - eps)
    # Gather the probability of the true class
    gathered = probs[torch.arange(probs.shape[0]), labels]
    return -torch.log(gathered).mean().item()


def brier_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Compute multi-class Brier score from probability tensor.

    Args:
        probs: [N, C] float tensor.
        labels: [N] long tensor.
        num_classes: number of classes.
    """
    if probs.shape[0] == 0:
        raise ValueError("Cannot compute Brier on empty input")
    y_onehot = F.one_hot(labels, num_classes=num_classes).float()
    squared_error = (probs - y_onehot).pow(2).sum(dim=-1)
    return squared_error.mean().item()


def prediction_counts_from_probs(probs: torch.Tensor, num_classes: int) -> Dict[str, int]:
    """Count predictions from probability tensor (same as logits version)."""
    preds = torch.argmax(probs, dim=-1)
    counts = {}
    for c in range(num_classes):
        counts[f"class_{c}"] = int((preds == c).sum().item())
    counts["total"] = int(preds.shape[0])
    return counts
