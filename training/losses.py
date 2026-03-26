"""Loss functions for VQA training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQALoss(nn.Module):
    """Combined loss for VQA experiments.

    Supports two complementary objectives:

    * **Soft-target BCE** (``"bce"``): binary cross-entropy against the
      per-answer soft scores produced by VQA annotation aggregation
      (``min(count / 3, 1)``).  Commonly used when training with a fixed
      answer vocabulary.
    * **Cross-entropy** (``"ce"``): standard multi-class cross-entropy
      against a single hard label.  Useful when fine-tuning a generative
      decoder.
    * **KL divergence** (``"kl"``): treats the soft scores as a proper
      probability distribution and minimises the KL divergence.

    Args:
        loss_type: One of ``"bce"``, ``"ce"``, or ``"kl"``.
        label_smoothing: Label smoothing for ``"ce"`` loss (0.0 = disabled).
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        loss_type: str = "bce",
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if loss_type not in {"bce", "ce", "kl"}:
            raise ValueError(
                f"loss_type must be one of 'bce', 'ce', 'kl'; got '{loss_type}'"
            )
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "ce":
            self.ce = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction=reduction,
            )

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Model output logits.
                Shape ``[B, num_answers]`` for ``"bce"`` / ``"kl"``, or
                ``[B, num_answers]`` for ``"ce"``.
            labels: Ground-truth targets.
                * For ``"bce"`` / ``"kl"``: soft score tensor ``[B, num_answers]``
                  with values in ``[0, 1]``.
                * For ``"ce"``: long integer tensor ``[B]`` with class indices.

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == "ce":
            return self.ce(logits, labels)

        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                reduction=self.reduction,
            )

        # KL divergence  ─  normalise soft scores to a valid distribution
        eps = 1e-8
        target_probs = labels.float()
        target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + eps)
        log_probs = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1)
        return kl.mean() if self.reduction == "mean" else kl.sum()


class GenerativeLoss(nn.Module):
    """Cross-entropy loss over generated token sequences.

    Wraps :class:`torch.nn.CrossEntropyLoss` with optional label smoothing and
    ignores padding positions by default (``ignore_index=-100``).

    Args:
        ignore_index: Token id to ignore in the loss computation.
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: ``[B, L, vocab_size]``
            labels: ``[B, L]`` with integer token ids; pad positions must be
                set to ``ignore_index``.

        Returns:
            Scalar loss.
        """
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
