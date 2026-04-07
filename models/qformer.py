"""Query Transformer (Q-Former) — module dùng chung cho EXP-06 và BLIP2VQA.

Q-Former là "cầu nối" giữa vision encoder bị đóng băng và language model,
dùng một tập query token có thể học để cô đọng thông tin thị giác.

Cơ chế hoạt động trong mỗi forward pass:
  - Query token self-attend với nhau (học quan hệ giữa các query).
  - Query token cross-attend vào patch token của vision encoder
    (mỗi ``cross_attention_freq`` lớp — mặc định mỗi 2 lớp).
  - Kết quả: biểu diễn thị giác compact [B, num_query_tokens, hidden_size].

Sự khác biệt so với BLIP-2 gốc:
  - ``vision_width`` mặc định là ``VISION_ENCODER_WIDTH`` (1024, CLIP ViT-L/14)
    thay vì 1408 (EVA-CLIP) — tuân theo contracts.py.
  - Tất cả giá trị mặc định lấy từ ``configs.contracts``.

Tham khảo:
    Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
    Image Encoders and Large Language Models", ICML 2023.
    https://arxiv.org/abs/2301.12597
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.contracts import (
    NUM_QUERY_TOKENS,       # 32
    QFORMER_HIDDEN_SIZE,    # 768
    VISION_ENCODER_WIDTH,   # 1024 — CLIP ViT-L/14
)


# ---------------------------------------------------------------------------
# Cấu hình Q-Former
# ---------------------------------------------------------------------------


@dataclass
class QFormerConfig:
    """Siêu tham số cấu hình Q-Former.

    Tất cả giá trị mặc định được lấy từ ``configs.contracts`` để đảm bảo
    đồng bộ với phần còn lại của codebase.

    Attributes:
        num_query_tokens:           Số query token có thể học. Mặc định 32.
        vision_width:               Chiều đầu ra của vision encoder.
                                    Mặc định ``VISION_ENCODER_WIDTH`` = 1024 (CLIP ViT-L/14).
                                    ⚠️ BLIP-2 gốc dùng 1408 (EVA-CLIP) — dự án này dùng 1024.
        hidden_size:                Chiều ẩn Q-Former. Mặc định 768 (chuẩn BERT-base).
        num_hidden_layers:          Số lớp QFormerLayer. Mặc định 12.
        num_attention_heads:        Số đầu attention. Mặc định 12.
        intermediate_size:          Chiều ẩn FFN (= 4 × hidden_size). Mặc định 3072.
        hidden_dropout_prob:        Dropout trong residual sau self/cross-attn và FFN.
        attention_probs_dropout_prob: Dropout trong attention weights.
        cross_attention_freq:       Chèn cross-attention mỗi N lớp. Mặc định 2
                                    (lớp chẵn: 0, 2, 4, ... có cross-attn;
                                     lớp lẻ: 1, 3, 5, ... chỉ self-attn + FFN).
    """

    num_query_tokens:             int   = NUM_QUERY_TOKENS      # 32
    vision_width:                 int   = VISION_ENCODER_WIDTH  # 1024
    hidden_size:                  int   = QFORMER_HIDDEN_SIZE   # 768
    num_hidden_layers:            int   = 12
    num_attention_heads:          int   = 12
    intermediate_size:            int   = 3072
    hidden_dropout_prob:          float = 0.1
    attention_probs_dropout_prob: float = 0.1
    cross_attention_freq:         int   = 2
    # query_length là alias của num_query_tokens — tự động gán trong __post_init__
    query_length: int = field(init=False)

    def __post_init__(self) -> None:
        self.query_length = self.num_query_tokens
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) phải chia hết cho "
            f"num_attention_heads ({self.num_attention_heads})"
        )


# ---------------------------------------------------------------------------
# Khối cơ bản: Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention tích vô hướng có tỉ lệ (Scaled Dot-Product).

    Dùng cho cả self-attention (Q=K=V=queries) và cross-attention
    (Q=queries, K=V=visual_features) trong QFormerLayer.

    Args:
        hidden_size: Chiều embedding. Phải chia hết cho ``num_heads``.
        num_heads:   Số đầu attention.
        dropout:     Xác suất dropout trên attention weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) phải chia hết cho num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        # Hệ số tỉ lệ: 1/√d_k — tránh dot product quá lớn gây vanishing gradient
        self.scale     = self.head_dim ** -0.5

        # Chiếu Q, K, V và output
        self.q_proj   = nn.Linear(hidden_size, hidden_size)
        self.k_proj   = nn.Linear(hidden_size, hidden_size)
        self.v_proj   = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính multi-head attention.

        Args:
            query:          Shape ``[B, Nq, hidden_size]``.
            key:            Shape ``[B, Nkv, hidden_size]``.
            value:          Shape ``[B, Nkv, hidden_size]``.
            attention_mask: Additive bias ``[B, heads, Nq, Nkv]`` hoặc broadcast-compatible.
                            Thường là tensor âm lớn (-1e4) ở vị trí cần mask.

        Returns:
            Shape ``[B, Nq, hidden_size]``.
        """
        B, Nq, _  = query.shape
        Nkv       = key.shape[1]

        # Chiếu và reshape thành [B, num_heads, seq_len, head_dim]
        q = self.q_proj(query).view(B, Nq,  self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key)  .view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        # Tích vô hướng có tỉ lệ: [B, heads, Nq, Nkv]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask  # cộng bias mask (vị trí mask → -inf → softmax ≈ 0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Tổng hợp value theo trọng số attention: [B, heads, Nq, head_dim] → [B, Nq, hidden_size]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Khối cơ bản: Feed-Forward Network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """FFN position-wise: Linear(hidden→intermediate) → GELU → Linear(intermediate→hidden).

    Áp dụng độc lập tại mỗi vị trí (position-wise) theo chuẩn Transformer.

    Args:
        hidden_size:       Chiều đầu vào và đầu ra.
        intermediate_size: Chiều ẩn (thường = 4 × hidden_size = 3072).
        dropout:           Dropout sau lớp Linear cuối.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1     = nn.Linear(hidden_size, intermediate_size)
        self.fc2     = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear → GELU (kích hoạt mượt, ưa dùng trong BERT/GPT) → Linear → Dropout
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Một lớp Q-Former
# ---------------------------------------------------------------------------


class QFormerLayer(nn.Module):
    """Một lớp của Q-Former.

    Mỗi lớp thực hiện theo thứ tự (pre-norm style):
      1. Self-attention: queries học quan hệ lẫn nhau.
      2. Cross-attention (tùy chọn): queries lấy thông tin từ visual patch token.
         Chỉ có ở lớp ``has_cross_attention=True`` (được quyết định bởi ``cross_attention_freq``).
      3. FFN: biến đổi phi tuyến position-wise.

    Tất cả sub-layers dùng kiến trúc **pre-norm** (LayerNorm trước, residual sau).

    Args:
        config:              Cấu hình Q-Former.
        has_cross_attention: Nếu True, lớp này có thêm cross-attention block.
    """

    def __init__(self, config: QFormerConfig, has_cross_attention: bool) -> None:
        super().__init__()
        self.has_cross_attention = has_cross_attention

        # --- Self-attention block ---
        self.self_attn = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)

        # --- Cross-attention block (chỉ tạo nếu lớp này có cross-attn) ---
        if has_cross_attention:
            self.cross_attn = MultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
            self.norm_cross = nn.LayerNorm(config.hidden_size)
            self.drop_cross = nn.Dropout(config.hidden_dropout_prob)

        # --- FFN block ---
        self.ffn   = FeedForward(
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
        """Forward một lớp Q-Former.

        Args:
            query:           Query token hiện tại. Shape ``[B, num_queries, hidden_size]``.
            visual_features: Patch token visual đã chiếu. Shape ``[B, N_patches, hidden_size]``.
                             Chỉ dùng nếu ``has_cross_attention=True``.
            query_mask:      Attention bias cho self-attention. Thường là None.
            visual_mask:     Attention bias cho cross-attention. Shape ``[B, 1, 1, N_patches]``
                             với giá trị âm lớn ở vị trí cần bỏ qua.

        Returns:
            Query token sau khi qua lớp. Shape ``[B, num_queries, hidden_size]``.
        """
        # 1. Self-attention (pre-norm): queries ↔ queries
        residual = query
        query = self.norm1(query)
        query = residual + self.drop1(self.self_attn(query, query, query, query_mask))

        # 2. Cross-attention (pre-norm): queries ← visual patch tokens
        #    Chỉ thực hiện ở lớp chẵn (has_cross_attention=True)
        if self.has_cross_attention and visual_features is not None:
            residual = query
            query = self.norm_cross(query)
            query = residual + self.drop_cross(
                self.cross_attn(query, visual_features, visual_features, visual_mask)
            )

        # 3. FFN (pre-norm): biến đổi phi tuyến position-wise
        residual = query
        query = self.norm2(query)
        query = residual + self.drop2(self.ffn(query))

        return query


# ---------------------------------------------------------------------------
# Q-Former — module chính
# ---------------------------------------------------------------------------


class QFormer(nn.Module):
    """Query Transformer (Q-Former) dùng trong BLIP-2.

    Trích xuất biểu diễn thị giác có độ dài cố định từ chuỗi patch token
    có độ dài biến đổi bằng ``num_query_tokens`` query vector học được.

    Luồng xử lý:
        visual: [B, N_patches, vision_width]
                → visual_proj (Linear)
                → visual_norm (LayerNorm)
                → [B, N_patches, hidden_size]

        queries: [1, num_query_tokens, hidden_size]  (parameter học được)
                → expand(B, ...)
                → 12 × QFormerLayer (self-attn + cross-attn xen kẽ + FFN)
                → norm
                → [B, num_query_tokens, hidden_size]

    Ví dụ::

        config  = QFormerConfig(num_query_tokens=32, vision_width=1024)
        qformer = QFormer(config)
        visual  = torch.randn(2, 257, 1024)   # [B, N_patches, vision_width]
        out     = qformer(visual)              # [B, 32, 768]
    """

    def __init__(self, config: Optional[QFormerConfig] = None, **kwargs) -> None:
        super().__init__()
        # Cho phép khởi tạo bằng keyword args thay vì truyền config object
        if config is None:
            config = QFormerConfig(**kwargs)
        self.config = config

        # --- Query token học được ---
        # Shape [1, num_query_tokens, hidden_size]; sẽ expand theo batch khi forward
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.hidden_size)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # --- Chiếu visual: vision_width → hidden_size ---
        # CLIP ViT-L/14: 1024 → 768
        self.visual_proj = nn.Linear(config.vision_width, config.hidden_size)
        self.visual_norm = nn.LayerNorm(config.hidden_size)

        # --- Stack num_hidden_layers lớp QFormerLayer ---
        # Lớp chẵn (0, 2, 4, ...): has_cross_attention=True
        # Lớp lẻ  (1, 3, 5, ...): has_cross_attention=False
        self.layers = nn.ModuleList([
            QFormerLayer(
                config,
                has_cross_attention=((i % config.cross_attention_freq) == 0),
            )
            for i in range(config.num_hidden_layers)
        ])

        # LayerNorm cuối chuỗi — chuẩn hóa output trước khi dùng
        self.norm = nn.LayerNorm(config.hidden_size)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Khởi tạo trọng số:
        - Linear: trunc_normal (std=0.02) theo chuẩn ViT/BERT.
        - LayerNorm: weight=1, bias=0 (giá trị mặc định nhưng tường minh).
        """
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
        """Trích xuất biểu diễn query từ patch token visual.

        Args:
            visual_features:       Patch token từ vision encoder.
                                   Shape ``[B, N_patches, vision_width]``.
            visual_attention_mask: Mask boolean/float trên patch token.
                                   Shape ``[B, N_patches]``; True/1 = giữ lại token.
                                   Nếu None, tất cả patch đều được attend.

        Returns:
            Biểu diễn query token. Shape ``[B, num_query_tokens, hidden_size]``.
        """
        B = visual_features.shape[0]

        # Chiếu và normalize visual features: [B, N, vision_width] → [B, N, hidden_size]
        visual_features = self.visual_norm(self.visual_proj(visual_features))

        # Xây dựng attention bias 4D cho cross-attention từ mask
        # Vị trí bị mask nhận bias = -1e4 → sau softmax ≈ 0 → bị bỏ qua
        visual_mask_bias: Optional[torch.Tensor] = None
        if visual_attention_mask is not None:
            # [B, N] → [B, 1, 1, N] để broadcast qua (heads, query_positions)
            visual_mask_bias = (
                (1.0 - visual_attention_mask.float())
                .unsqueeze(1)
                .unsqueeze(2)
                .mul(-1e4)
            )

        # Expand query tokens theo batch: [1, Q, D] → [B, Q, D]
        query = self.query_tokens.expand(B, -1, -1)

        # Qua tất cả QFormerLayer — mỗi lớp cập nhật query
        for layer in self.layers:
            query = layer(
                query,
                visual_features=visual_features,
                visual_mask=visual_mask_bias,
            )

        # LayerNorm cuối: [B, num_query_tokens, hidden_size]
        return self.norm(query)

    # ------------------------------------------------------------------

    @property
    def num_query_tokens(self) -> int:
        """Số query token (alias truy cập nhanh từ config)."""
        return self.config.num_query_tokens

    @property
    def hidden_size(self) -> int:
        """Chiều ẩn Q-Former (alias truy cập nhanh từ config)."""
        return self.config.hidden_size
