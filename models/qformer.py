"""Query Transformer (Q-Former) implementation for BLIP-2.

The Q-Former bridges a frozen vision encoder and a language model using a
small set of learnable query tokens.  During each forward pass the query
tokens interact with the visual patch tokens through cross-attention and with
each other through self-attention, distilling task-relevant visual information
into a compact representation that can be fed to the language model.

Reference:
    Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
    Image Encoders and Large Language Models", ICML 2023.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QFormerConfig:
    """Hyper-parameters for the Q-Former."""

    num_query_tokens: int = 32
    vision_width: int = 1408          # ViT-L/14 output width used in BLIP-2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    cross_attention_freq: int = 2      # insert cross-attn every N layers
    query_length: int = field(init=False)

    def __post_init__(self) -> None:
        self.query_length = self.num_query_tokens
        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, _ = query.shape
        Nkv = key.shape[1]

        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)          # [B, heads, Nq, head_dim]
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward sub-layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class QFormerLayer(nn.Module):
    """One layer of the Q-Former.

    Query tokens perform:
    1. Self-attention among themselves.
    2. Cross-attention over the visual patch tokens (every ``cross_attention_freq``
       layers, otherwise omitted to reduce compute).
    3. Position-wise feed-forward.
    """

    def __init__(self, config: QFormerConfig, has_cross_attention: bool) -> None:
        super().__init__()
        self.has_cross_attention = has_cross_attention

        self.self_attn = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)

        if has_cross_attention:
            self.cross_attn = MultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
            self.norm_cross = nn.LayerNorm(config.hidden_size)
            self.drop_cross = nn.Dropout(config.hidden_dropout_prob)

        self.ffn = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_dropout_prob,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.drop2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        query: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-attention
        residual = query
        query = self.norm1(query)
        query = residual + self.drop1(self.self_attn(query, query, query, query_mask))

        # 2. Cross-attention (only in designated layers)
        if self.has_cross_attention and visual_features is not None:
            residual = query
            query = self.norm_cross(query)
            query = residual + self.drop_cross(
                self.cross_attn(query, visual_features, visual_features, visual_mask)
            )

        # 3. Feed-forward
        residual = query
        query = self.norm2(query)
        query = residual + self.drop2(self.ffn(query))

        return query


# ---------------------------------------------------------------------------
# Q-Former
# ---------------------------------------------------------------------------


class QFormer(nn.Module):
    """Query Transformer used in BLIP-2.

    Extracts a fixed-length visual representation from variable-length
    visual patch tokens using ``num_query_tokens`` learnable query vectors.

    Example::

        config = QFormerConfig(num_query_tokens=32, vision_width=1408)
        qformer = QFormer(config)
        visual_feats = torch.randn(2, 256, 1408)   # [B, patches, vision_width]
        out = qformer(visual_feats)                # [B, 32, 768]
    """

    def __init__(self, config: Optional[QFormerConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = QFormerConfig(**kwargs)
        self.config = config

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.hidden_size)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Project visual features into Q-Former's hidden dimension
        self.visual_proj = nn.Linear(config.vision_width, config.hidden_size)
        self.visual_norm = nn.LayerNorm(config.hidden_size)

        # Stacked Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                config,
                has_cross_attention=((i % config.cross_attention_freq) == 0),
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        visual_features: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract query representations from visual features.

        Args:
            visual_features: Patch features from the vision encoder.
                Shape ``[B, N_patches, vision_width]``.
            visual_attention_mask: Boolean / float mask over patches.
                Shape ``[B, N_patches]``; ``True`` / ``1`` means *keep*.

        Returns:
            Query token representations, shape ``[B, num_query_tokens, hidden_size]``.
        """
        B = visual_features.shape[0]

        # Project visual features
        visual_features = self.visual_norm(self.visual_proj(visual_features))

        # Build 4-D attention bias for cross-attention if mask is given
        visual_mask_bias: Optional[torch.Tensor] = None
        if visual_attention_mask is not None:
            # [B, 1, 1, N_patches] — broadcast over heads and queries
            visual_mask_bias = (
                (1.0 - visual_attention_mask.float())
                .unsqueeze(1)
                .unsqueeze(2)
                .mul(-1e4)
            )

        # Expand query tokens to batch
        query = self.query_tokens.expand(B, -1, -1)

        # Forward through all Q-Former layers
        for layer in self.layers:
            query = layer(
                query,
                visual_features=visual_features,
                visual_mask=visual_mask_bias,
            )

        return self.norm(query)

    # ------------------------------------------------------------------

    @property
    def num_query_tokens(self) -> int:
        return self.config.num_query_tokens

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size
