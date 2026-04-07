"""EXP-05: Query Cross-Attention Bridge cho VQA.

Ý tưởng cốt lõi: **Thay vì mean-pool toàn bộ 257 patch token** (EXP-01→04),
dùng một tập **32 query token có thể học được** để *chủ động truy vấn*
thông tin không gian từ patch visual qua cơ chế **cross-attention**.

So sánh với các phương pháp trước:
    EXP-01→04: visual[B,257,1024] → mean() → [B,1024]  — mất thông tin phân bố không gian
    EXP-05:   queries[B,32,768] attend vào visual[B,257,768] → khai thác không gian ảnh có chọn lọc

Kiến trúc một lớp Cross-Attention Bridge:
    ┌─ LayerNorm(queries) ──────────────────────────────────────────────────────┐
    │                                                                            │
    │  CrossAttn(Q=queries_norm, K=visual_proj, V=visual_proj)                 │
    │         ↓ residual                                                        │
    │  queries = queries + attn_output         → [B, 32, 768]                  │
    │                                                                            │
    │  FFN(LayerNorm(queries)):                                                 │
    │    Linear(768 → 3072) → GELU → Dropout → Linear(3072 → 768)             │
    │         ↓ residual                                                        │
    │  queries = queries + ffn_output          → [B, 32, 768]                  │
    └───────────────────────────────────────────────────────────────────────────┘
    Lặp 3 lần (3 lớp cross-attention).

Luồng xử lý đầy đủ:
    visual:  [B, 257, 1024]  →  Linear(1024, 768)   →  [B, 257, 768]
    queries: [1, 32, 768]    →  expand(B, 32, 768)  →  [B, 32, 768]
    text:    [B, 768]

    Qua 3 lớp:
        queries = CrossAttnLayer(queries, visual_proj)   →  [B, 32, 768]

    pooled = queries.mean(dim=1)                         →  [B, 768]
    fused  = concat(pooled, text)                        →  [B, 1536]
           → MLP(1536, 1024, 3129)                       →  logits [B, 3129]

Ghi chú về đầu vào:
    - Model này yêu cầu TOÀN BỘ patch token [B, 257, 1024], KHÔNG chỉ CLS.
    - Nếu cache đặc trưng, cần cache dim-3 tensor (patch sequence), không phải CLS.
    - Ưu tiên on-the-fly ViT forward trên Colab để tiết kiệm disk.

Tham khảo:
    Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
    Image Encoders and Large Language Models", ICML 2023.
    https://arxiv.org/abs/2301.12597
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_CROSS_ATTN_FUSION,
    NUM_QUERY_TOKENS,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)

# Số lớp cross-attention Bridge (theo experiment-cases.md: 3 lớp)
_DEFAULT_NUM_LAYERS: int = 3
# Số đầu attention (12 đầu × 64 = 768, chuẩn BERT-base)
_DEFAULT_NUM_HEADS: int = 12
# Tỉ lệ mở rộng FFN — 4x so với hidden_dim (768 × 4 = 3072, chuẩn Transformer)
_DEFAULT_FFN_RATIO: int = 4


# ---------------------------------------------------------------------------
# Sub-module: Một lớp Cross-Attention Bridge
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """Một lớp cross-attention kiểu pre-norm với FFN.

    Queries attend vào visual projected features theo cơ chế cross-attention:
        queries_norm = LayerNorm(queries)
        queries = queries + MultiHeadAttn(Q=queries_norm, K=visual, V=visual)
        queries = queries + FFN(LayerNorm(queries))

    Args:
        hidden_dim:  Chiều của queries và visual. Mặc định 768.
        num_heads:   Số đầu attention. Mặc định 12.
        ffn_dim:     Chiều ẩn FFN. Mặc định 3072 (= 4 × hidden_dim).
        dropout:     Xác suất dropout trong attn và FFN.
    """

    def __init__(
        self,
        hidden_dim: int = QFORMER_HIDDEN_SIZE,  # 768
        num_heads: int = _DEFAULT_NUM_HEADS,     # 12
        ffn_dim: int = QFORMER_HIDDEN_SIZE * _DEFAULT_FFN_RATIO,  # 3072
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Pre-norm trước cross-attention
        self.norm_queries = nn.LayerNorm(hidden_dim)
        # Pre-norm trước FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        # Multi-head cross-attention:
        #   Q = queries, K = V = visual_projected
        #   batch_first=True → input [B, seq, dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN: Linear → GELU → Dropout → Linear
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        visual: torch.Tensor,
        visual_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Một bước forward của lớp cross-attention.

        Args:
            queries:  Query tokens. Shape ``[B, num_queries, hidden_dim]``.
            visual:   Visual key/value tokens (đã chiếu). Shape ``[B, N, hidden_dim]``.
            visual_key_padding_mask:  Mask ``[B, N]`` — True = vị trí bị mask (bỏ qua).
                                      Thường là None vì không có padding trong patch tokens.

        Returns:
            queries đã được cập nhật. Shape ``[B, num_queries, hidden_dim]``.
        """
        # --- Cross-Attention với pre-norm ---
        # queries_norm: [B, num_queries, hidden_dim]
        queries_norm = self.norm_queries(queries)

        # attn_output: [B, num_queries, hidden_dim]
        # Q = queries_norm, K = V = visual (patch tokens đã chiếu)
        attn_output, _ = self.cross_attn(
            query=queries_norm,
            key=visual,
            value=visual,
            key_padding_mask=visual_key_padding_mask,
        )
        # Residual connection
        queries = queries + attn_output  # [B, num_queries, hidden_dim]

        # --- FFN với pre-norm ---
        queries = queries + self.ffn(self.norm_ffn(queries))  # [B, num_queries, hidden_dim]

        return queries


# ---------------------------------------------------------------------------
# Main model: CrossAttnFusion (EXP-05)
# ---------------------------------------------------------------------------

class CrossAttnFusion(nn.Module):
    """Baseline Query Cross-Attention Bridge (EXP-05).

    32 query token có thể học được attend vào patch token visual qua 3 lớp
    cross-attention. Visual patch sequence được chiếu về hidden_dim trước
    khi đưa vào attention. Sau khi attend, mean-pool queries rồi concat
    với text CLS token và đưa qua MLP để phân loại.

    Args:
        visual_dim:   Chiều patch token visual.   Mặc định 1024 (CLIP ViT-L/14).
        text_dim:     Chiều text CLS token.        Mặc định 768  (BERT-base).
        hidden_dim:   Chiều queries và attn.       Mặc định 768.
        num_queries:  Số query token học được.     Mặc định 32.
        num_layers:   Số lớp cross-attention.      Mặc định 3.
        num_heads:    Số đầu attention.             Mặc định 12.
        num_answers:  Kích thước vocab đầu ra.     Mặc định 3129 (VQAv2).
        dropout:      Xác suất dropout.            Mặc định 0.1.
    """

    # Tên mô hình — phải khớp với contracts.MODEL_CROSS_ATTN_FUSION
    MODEL_NAME: str = MODEL_CROSS_ATTN_FUSION

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,   # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,       # 768
        hidden_dim: int = QFORMER_HIDDEN_SIZE,     # 768
        num_queries: int = NUM_QUERY_TOKENS,       # 32
        num_layers: int = _DEFAULT_NUM_LAYERS,     # 3
        num_heads: int = _DEFAULT_NUM_HEADS,       # 12
        num_answers: int = ANSWER_VOCAB_SIZE,      # 3129
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_dim  = visual_dim
        self.text_dim    = text_dim
        self.hidden_dim  = hidden_dim
        self.num_queries = num_queries
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.num_answers = num_answers

        # --- Query token học được ---
        # Khởi tạo shape [1, num_queries, hidden_dim]; sẽ expand theo batch khi forward
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, hidden_dim)
        )
        # Khởi tạo query tokens theo phân phối chuẩn nhỏ để tránh symmetry breaking
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

        # --- Chiếu visual từ visual_dim → hidden_dim ---
        # (CLIP ViT-L/14: 1024 → 768)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        # LayerNorm sau khi chiếu visual (ổn định đặc trưng trước khi vào attn)
        self.visual_norm = nn.LayerNorm(hidden_dim)

        # --- Stack 3 lớp Cross-Attention Bridge ---
        ffn_dim = hidden_dim * _DEFAULT_FFN_RATIO  # 768 × 4 = 3072
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # LayerNorm cuối sau tất cả cross-attn layers
        self.norm_out = nn.LayerNorm(hidden_dim)

        # --- Classifier MLP ---
        # Input: concat(pooled_queries [B, hidden_dim], text [B, text_dim]) → [B, hidden_dim + text_dim]
        # Kiến trúc: (hidden_dim + text_dim) → 1024 → num_answers
        fusion_input_dim = hidden_dim + text_dim  # 768 + 768 = 1536
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_answers),
        )

        # Khởi tạo trọng số Linear
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo tất cả Linear bằng xavier_uniform; LayerNorm giữ mặc định."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # LayerNorm được init mặc định (weight=1, bias=0) — không cần override

    # ------------------------------------------------------------------

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính logits dự đoán câu trả lời.

        Args:
            visual_features: Patch token sequence từ CLIP ViT-L/14.
                **Ưu tiên:** ``[B, 257, 1024]`` — CLS + 256 patch tokens.
                Nếu đưa vào shape ``[B, visual_dim]`` (đã pool), model vẫn hoạt động
                bằng cách unsqueeze thành sequence 1 token — nhưng mất lợi thế không gian.
            text_features:   CLS token từ BERT-base. Shape ``[B, text_dim]``.
            visual_mask:     Mask boolean ``[B, N]`` — True = token **hợp lệ** (giữ lại).
                             Được chuyển thành key_padding_mask cho MultiheadAttention.
                             Nếu None, tất cả patch token đều được attend.

        Returns:
            Logits chưa qua softmax. Shape ``[B, num_answers]``.
        """
        # --- Bước 1: Chuẩn bị visual key/value ---
        if visual_features.dim() == 2:
            # Trường hợp visual đã bị pool bên ngoài → unsqueeze thành 1 token
            # Ghi chú: trường hợp này mất lợi thế không gian của cross-attention
            visual_features = visual_features.unsqueeze(1)  # [B, 1, visual_dim]

        # visual_features: [B, N, visual_dim]
        # Chiếu về hidden_dim và normalize
        visual_proj = self.visual_norm(
            self.visual_proj(visual_features)
        )  # [B, N, hidden_dim]

        # Chuyển visual_mask sang key_padding_mask cho nn.MultiheadAttention
        # nn.MultiheadAttention dùng: True = vị trí BỊ bỏ qua (ngược với convention của ta)
        key_padding_mask: Optional[torch.Tensor] = None
        if visual_mask is not None:
            # visual_mask: True = hợp lệ → key_padding_mask: True = bỏ qua
            key_padding_mask = ~visual_mask  # [B, N]

        # --- Bước 2: Expand query tokens theo batch ---
        B = visual_features.size(0)
        # self.query_tokens: [1, num_queries, hidden_dim] → [B, num_queries, hidden_dim]
        queries = self.query_tokens.expand(B, -1, -1)  # [B, 32, hidden_dim]

        # --- Bước 3: Qua num_layers lớp Cross-Attention Bridge ---
        # Mỗi lớp: queries attend vào visual patch tokens → cập nhật queries
        for layer in self.layers:
            queries = layer(
                queries=queries,
                visual=visual_proj,
                visual_key_padding_mask=key_padding_mask,
            )  # [B, 32, hidden_dim]

        # LayerNorm cuối chuỗi cross-attn
        queries = self.norm_out(queries)  # [B, 32, hidden_dim]

        # --- Bước 4: Mean-pool queries → vector đại diện ---
        # Tổng hợp thông tin từ 32 query token thành 1 vector [B, hidden_dim]
        pooled = queries.mean(dim=1)  # [B, hidden_dim]

        # --- Bước 5: Concat với text CLS và phân loại ---
        # fused: [B, hidden_dim + text_dim] = [B, 1536]
        fused = torch.cat([pooled, text_features], dim=-1)

        # logits: [B, num_answers]
        return self.classifier(fused)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Thông tin bổ sung khi print(model)."""
        return (
            f"visual_dim={self.visual_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_queries={self.num_queries}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"num_answers={self.num_answers}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_cross_attn(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    hidden_dim: int = QFORMER_HIDDEN_SIZE,
    num_queries: int = NUM_QUERY_TOKENS,
    num_layers: int = _DEFAULT_NUM_LAYERS,
    num_heads: int = _DEFAULT_NUM_HEADS,
    num_answers: int = ANSWER_VOCAB_SIZE,
    dropout: float = 0.1,
) -> CrossAttnFusion:
    """Khởi tạo model CrossAttnFusion với các tham số tuỳ chỉnh.

    Args:
        visual_dim:  Chiều patch token visual.  Mặc định 1024 (CLIP ViT-L/14).
        text_dim:    Chiều text CLS token.       Mặc định 768  (BERT-base).
        hidden_dim:  Chiều query và attention.   Mặc định 768.
        num_queries: Số query token học được.    Mặc định 32.
        num_layers:  Số lớp cross-attention.     Mặc định 3.
        num_heads:   Số đầu attention.            Mặc định 12.
        num_answers: Kích thước bộ từ vựng.      Mặc định 3129 (VQAv2).
        dropout:     Xác suất dropout.            Mặc định 0.1.

    Returns:
        Một instance của :class:`CrossAttnFusion`.

    Ví dụ::

        model = build_cross_attn()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return CrossAttnFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
        num_answers=num_answers,
        dropout=dropout,
    )
