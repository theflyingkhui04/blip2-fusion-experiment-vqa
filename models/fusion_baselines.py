"""Fusion baseline models for VQA.

Each baseline takes a visual feature vector and a text (question) feature
vector and fuses them into a single representation used for answer prediction.

Baselines
---------
- :class:`ConcatFusion`     – plain concatenation followed by an MLP.
- :class:`BilinearFusion`   – element-wise product of projected features.
- :class:`AttentionFusion`  – question-guided attention over visual tokens.
- :class:`MLBFusion`        – Multi-modal Low-rank Bilinear pooling (MLB).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    dropout: float = 0.1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_features, out_features),
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class ConcatFusion(nn.Module):
    """Concatenation-based fusion.

    Concatenates the visual and text feature vectors then passes them through
    a two-layer MLP classifier.

    Args:
        visual_dim: Dimensionality of the visual feature vector.
        text_dim: Dimensionality of the text feature vector.
        fusion_dim: Hidden dimensionality of the MLP.
        num_answers: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 1024,
        num_answers: int = 3129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = _mlp(
            visual_dim + text_dim, fusion_dim, num_answers, dropout
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: ``[B, visual_dim]``
            text_features:   ``[B, text_dim]``

        Returns:
            Logits ``[B, num_answers]``.
        """
        fused = torch.cat([visual_features, text_features], dim=-1)
        return self.classifier(fused)


class BilinearFusion(nn.Module):
    """Bilinear fusion via element-wise product of projected features.

    Both modalities are projected to the same ``fusion_dim`` and then fused
    with element-wise multiplication (Hadamard product).

    Args:
        visual_dim: Dimensionality of the visual feature vector.
        text_dim: Dimensionality of the text feature vector.
        fusion_dim: Shared projection dimensionality.
        num_answers: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 1024,
        num_answers: int = 3129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_answers)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        v = self.dropout(F.relu(self.visual_proj(visual_features)))
        t = self.dropout(F.relu(self.text_proj(text_features)))
        fused = v * t                   # element-wise product
        return self.classifier(fused)


class AttentionFusion(nn.Module):
    """Question-guided attention over visual tokens.

    For each question, a scalar attention weight is computed for every visual
    token.  The attended visual summary is concatenated with the question
    representation and classified.

    Args:
        visual_dim: Dimensionality of each visual token (e.g. 768 for Q-Former).
        text_dim: Dimensionality of the question feature vector.
        fusion_dim: Hidden size for attention and final MLP.
        num_answers: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 1024,
        num_answers: int = 3129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.attn_fc = nn.Linear(fusion_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = _mlp(visual_dim + text_dim, fusion_dim, num_answers, dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: ``[B, N_tokens, visual_dim]`` – sequence of visual tokens.
            text_features:   ``[B, text_dim]``.
            visual_mask:     Boolean mask ``[B, N_tokens]``; ``True`` = keep.

        Returns:
            Logits ``[B, num_answers]``.
        """
        B, N, _ = visual_features.shape

        v_proj = self.dropout(F.relu(self.visual_proj(visual_features)))  # [B, N, fusion_dim]
        t_proj = self.dropout(F.relu(self.text_proj(text_features)))      # [B, fusion_dim]

        # Compute attention scores
        attn_in = v_proj + t_proj.unsqueeze(1)                            # broadcast over N
        attn_scores = self.attn_fc(torch.tanh(attn_in)).squeeze(-1)       # [B, N]

        if visual_mask is not None:
            attn_scores = attn_scores.masked_fill(~visual_mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)                     # [B, N]
        attended = (attn_weights.unsqueeze(-1) * visual_features).sum(1)  # [B, visual_dim]

        fused = torch.cat([attended, text_features], dim=-1)
        return self.classifier(fused)


class MLBFusion(nn.Module):
    """Multi-modal Low-rank Bilinear (MLB) fusion.

    Projects each modality into ``num_glimpses`` separate sub-spaces and
    combines them with element-wise multiplication, effectively approximating
    the full bilinear interaction at much lower parameter cost.

    Reference:
        Kim et al., "Hadamard Product for Low-rank Bilinear Pooling", 2017.

    Args:
        visual_dim: Dimensionality of the visual feature vector.
        text_dim: Dimensionality of the text feature vector.
        fusion_dim: Low-rank projection dimensionality.
        num_glimpses: Number of bilinear heads (parallel low-rank components).
        num_answers: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 1024,
        num_glimpses: int = 2,
        num_answers: int = 3129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_glimpses = num_glimpses
        proj_dim = fusion_dim * num_glimpses

        self.visual_proj = nn.Linear(visual_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = _mlp(proj_dim, fusion_dim, num_answers, dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        v = self.dropout(torch.tanh(self.visual_proj(visual_features)))
        t = self.dropout(torch.tanh(self.text_proj(text_features)))
        fused = v * t                   # element-wise product (Hadamard)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "bilinear": BilinearFusion,
    "attention": AttentionFusion,
    "mlb": MLBFusion,
}


def build_fusion_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a fusion baseline by name.

    Args:
        name: One of ``"concat"``, ``"bilinear"``, ``"attention"``, ``"mlb"``.
        **kwargs: Constructor keyword arguments forwarded to the model class.

    Returns:
        An ``nn.Module`` instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion model '{name}'. "
            f"Available: {list(_FUSION_REGISTRY.keys())}"
        )
    return _FUSION_REGISTRY[name](**kwargs)
