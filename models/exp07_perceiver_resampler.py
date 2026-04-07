"""EXP-07: Perceiver Resampler cho VQA.

Perceiver Resampler (từ Flamingo) dùng tập **M latent vector học được** để
truy vấn thông tin từ patch token visual qua cross-attention — tương tự Q-Former
(EXP-06) nhưng với kiến trúc đơn giản hơn và hiệu quả tham số hơn.

So sánh với Q-Former (EXP-06):
    Q-Former (EXP-06):
      - 12 lớp transformer đầy đủ
      - Self-attention giữa queries + cross-attention xen kẽ (mỗi 2 lớp)
      - ~100M params (chỉ Q-Former)
      - Queries trao đổi thông tin với nhau qua self-attn

    Perceiver Resampler (EXP-07):
      - 4 lớp, MỖI lớp đều có cross-attention (không xen kẽ)
      - KHÔNG có self-attention thuần giữa latents
      - Thay vào đó: K và V = concat(latents, visual) — latents vừa là Q vừa là context
      - ~58M params — hiệu quả hơn Q-Former

Điểm đặc biệt của Perceiver Resampler so với Cross-Attn Bridge (EXP-05):
    EXP-05: Q=queries, K=V=visual                   → queries chỉ xem visual
    EXP-07: Q=latents, K=V=concat(latents, visual)  → latents còn "xem" lẫn nhau
             Đây là trick cho phép latents chia sẻ thông tin mà không cần self-attn block riêng

Kiến trúc mỗi lớp Perceiver:
    Đầu vào: latents [B, M, D], visual_proj [B, N, D]

    ┌─ pre-norm ──────────────────────────────────────────────────────────────┐
    │  latents_norm = LayerNorm(latents)                                      │
    │  context = concat(latents_norm, visual_proj)   → [B, M+N, D]           │
    │  attn = MultiHeadAttn(Q=latents_norm, K=context, V=context)            │
    │  latents = latents + Dropout(attn)             → [B, M, D]             │
    └─────────────────────────────────────────────────────────────────────────┘
    ┌─ pre-norm FFN ──────────────────────────────────────────────────────────┐
    │  latents = latents + FFN(LayerNorm(latents))   → [B, M, D]             │
    └─────────────────────────────────────────────────────────────────────────┘
    Lặp 4 lần (4 lớp Perceiver).

Luồng xử lý đầy đủ:
    visual:  [B, 257, 1024]  →  visual_proj (Linear 1024→768)
                             →  visual_norm (LayerNorm)
                             →  [B, 257, 768]

    latents: [1, M, 768]     →  expand(B, M, 768)   →  [B, M, 768]
    text:    [B, 768]

    Qua 4 lớp Perceiver:
        context = concat(latents_norm, visual_proj)  →  [B, M+257, 768]
        latents = latents + CrossAttn(Q=latents, K=context, V=context)
        latents = latents + FFN(latents)

    pooled = latents.mean(dim=1)           →  [B, 768]
    fused  = concat(pooled, text)          →  [B, 1536]
    logits = MLP(1536, 1024, num_answers)  →  [B, 3129]

Ghi chú về đầu vào:
    - Model yêu cầu toàn bộ patch token [B, 257, 1024], KHÔNG chỉ CLS.
    - Ưu tiên on-the-fly ViT forward trên Colab để tiết kiệm disk.

Tham khảo:
    Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning",
    NeurIPS 2022. https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_PERCEIVER_RESAMPLER,
    NUM_QUERY_TOKENS,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)

# Số latent vector (experiment-cases.md: M=64, lớn hơn Q-Former 32 để bù thiếu self-attn)
_DEFAULT_NUM_LATENTS: int = 64
# Số lớp Perceiver (experiment-cases.md: 4 lớp)
_DEFAULT_NUM_LAYERS: int = 4
# Số đầu attention (chuẩn BERT-base)
_DEFAULT_NUM_HEADS: int = 12
# Tỉ lệ mở rộng FFN (4x so với hidden_dim)
_DEFAULT_FFN_RATIO: int = 4


# ---------------------------------------------------------------------------
# Sub-module: Một lớp Perceiver Resampler
# ---------------------------------------------------------------------------

class PerceiverLayer(nn.Module):
    """Một lớp của Perceiver Resampler.

    Khác với cross-attention thông thường (Q=latents, K=V=visual), ở đây
    K và V là **concat(latents_norm, visual_proj)** — cho phép latents
    vừa attend vào visual vừa "nhìn thấy" giá trị của nhau mà không cần
    một self-attention block riêng biệt.

    Thứ tự thực hiện (pre-norm style):
        1. latents_norm = LayerNorm(latents)
        2. context = concat(latents_norm, visual_proj)  →  [B, M+N, D]
        3. attn = MultiHeadAttn(Q=latents_norm, K=context, V=context)
        4. latents = latents + Dropout(attn)
        5. latents = latents + FFN(LayerNorm(latents))

    Args:
        hidden_dim: Chiều latents và visual. Mặc định 768.
        num_heads:  Số đầu attention. Mặc định 12.
        ffn_dim:    Chiều ẩn FFN. Mặc định 3072 (= 4 × hidden_dim).
        dropout:    Xác suất dropout.
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
        self.norm_latents = nn.LayerNorm(hidden_dim)
        # Pre-norm trước FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        # Cross-attention: Q = latents_norm, K = V = concat(latents_norm, visual_proj)
        # batch_first=True → input [B, seq, dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN: Linear → GELU → Dropout → Linear → Dropout
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        visual: torch.Tensor,
        visual_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Một bước forward của lớp Perceiver.

        Args:
            latents: Latent vector hiện tại. Shape ``[B, M, hidden_dim]``.
            visual:  Visual patch token đã chiếu. Shape ``[B, N, hidden_dim]``.
            visual_key_padding_mask: Mask ``[B, N]`` — True = bỏ qua.
                Được mở rộng thành ``[B, M+N]`` (latents không bao giờ bị mask).

        Returns:
            Latents đã cập nhật. Shape ``[B, M, hidden_dim]``.
        """
        B, M, _ = latents.shape

        # --- Bước 1: Pre-norm latents ---
        latents_norm = self.norm_latents(latents)  # [B, M, D]

        # --- Bước 2: Xây dựng context = concat(latents_norm, visual_proj) ---
        # Đây là điểm mấu chốt của Perceiver Resampler:
        # Bằng cách đưa latents_norm vào cả K và V, latents vừa trao đổi
        # thông tin với nhau vừa attend vào visual — không cần self-attn riêng
        context = torch.cat([latents_norm, visual], dim=1)  # [B, M+N, D]

        # --- Bước 3: Mở rộng key_padding_mask nếu có ---
        # Latents luôn hợp lệ (không bị mask) → thêm M cột False vào đầu
        key_padding_mask: Optional[torch.Tensor] = None
        if visual_key_padding_mask is not None:
            # visual_key_padding_mask: [B, N] (True = bỏ qua visual token)
            # Tạo mask False cho phần latents: [B, M]
            latent_mask = torch.zeros(
                B, M, dtype=torch.bool, device=latents.device
            )
            # Ghép: [B, M] + [B, N] → [B, M+N]
            key_padding_mask = torch.cat([latent_mask, visual_key_padding_mask], dim=1)

        # --- Bước 4: Cross-attention ---
        # Q = latents_norm  → [B, M, D]
        # K = V = context   → [B, M+N, D]
        # output             → [B, M, D]
        attn_output, _ = self.cross_attn(
            query=latents_norm,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
        )
        # Residual connection
        latents = latents + attn_output  # [B, M, D]

        # --- Bước 5: FFN với pre-norm ---
        latents = latents + self.ffn(self.norm_ffn(latents))  # [B, M, D]

        return latents


# ---------------------------------------------------------------------------
# Main model: PerceiverResampler (EXP-07)
# ---------------------------------------------------------------------------

class PerceiverResampler(nn.Module):
    """Baseline Perceiver Resampler (EXP-07).

    M latent vector học được attend vào patch token visual qua 4 lớp
    Perceiver. Trong mỗi lớp, K và V là concat(latents, visual_proj) — cho
    phép latents trao đổi thông tin với nhau mà không cần self-attention
    block riêng biệt.

    So sánh tổng quan:
        EXP-05 (CrossAttn Bridge):     3 lớp, 32 queries, không self-attn,  ~27M params
        EXP-06 (Q-Former Scratch):    12 lớp, 32 queries, có self-attn,   ~105M params
        EXP-07 (Perceiver Resampler):  4 lớp, 64 latents, latent-as-context, ~58M params

    Args:
        visual_dim:   Chiều patch token visual.   Mặc định 1024 (CLIP ViT-L/14).
        text_dim:     Chiều text CLS token.        Mặc định 768  (BERT-base).
        hidden_dim:   Chiều latents và attention.  Mặc định 768.
        num_latents:  Số latent vector học được.   Mặc định 64.
        num_layers:   Số lớp Perceiver.            Mặc định 4.
        num_heads:    Số đầu attention.             Mặc định 12.
        num_answers:  Kích thước vocab đầu ra.     Mặc định 3129 (VQAv2).
        dropout:      Xác suất dropout.            Mặc định 0.1.
    """

    # Tên mô hình — phải khớp với contracts.MODEL_PERCEIVER_RESAMPLER
    MODEL_NAME: str = MODEL_PERCEIVER_RESAMPLER

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,   # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,       # 768
        hidden_dim: int = QFORMER_HIDDEN_SIZE,     # 768
        num_latents: int = _DEFAULT_NUM_LATENTS,   # 64
        num_layers: int = _DEFAULT_NUM_LAYERS,     # 4
        num_heads: int = _DEFAULT_NUM_HEADS,       # 12
        num_answers: int = ANSWER_VOCAB_SIZE,      # 3129
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_dim  = visual_dim
        self.text_dim    = text_dim
        self.hidden_dim  = hidden_dim
        self.num_latents = num_latents
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.num_answers = num_answers

        # --- Latent vector học được ---
        # Shape [1, num_latents, hidden_dim]; sẽ expand theo batch khi forward
        # Khởi tạo theo phân phối chuẩn nhỏ (std=0.02) theo chuẩn ViT/BERT
        self.latents = nn.Parameter(
            torch.zeros(1, num_latents, hidden_dim)
        )
        nn.init.normal_(self.latents, mean=0.0, std=0.02)

        # --- Chiếu visual: visual_dim → hidden_dim ---
        # CLIP ViT-L/14: 1024 → 768
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)

        # --- Stack num_layers lớp Perceiver ---
        ffn_dim = hidden_dim * _DEFAULT_FFN_RATIO  # 768 × 4 = 3072
        self.layers = nn.ModuleList([
            PerceiverLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # LayerNorm cuối chuỗi Perceiver — chuẩn hóa output trước khi pool
        self.norm_out = nn.LayerNorm(hidden_dim)

        # --- Classifier MLP ---
        # Input: concat(pooled_latents [B, hidden_dim], text [B, text_dim])
        # = [B, 768 + 768] = [B, 1536]
        fusion_input_dim = hidden_dim + text_dim  # 1536
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
                **Ưu tiên:** ``[B, N_patches, visual_dim]`` để tận dụng
                cross-attention của Perceiver.
                Nếu đưa vào ``[B, visual_dim]`` (đã pool), unsqueeze thành 1 token.
            text_features:   CLS token từ BERT-base. Shape ``[B, text_dim]``.
            visual_mask:     Mask boolean ``[B, N_patches]`` — True = token hợp lệ.
                             Chuyển thành key_padding_mask (True = bỏ qua) khi truyền
                             vào PerceiverLayer.

        Returns:
            Logits chưa qua softmax. Shape ``[B, num_answers]``.
        """
        # --- Bước 1: Chuẩn bị visual token ---
        if visual_features.dim() == 2:
            # [B, visual_dim] → [B, 1, visual_dim]
            visual_features = visual_features.unsqueeze(1)

        # visual_features: [B, N, visual_dim]
        # Chiếu về hidden_dim và normalize: [B, N, hidden_dim]
        visual_proj = self.visual_norm(
            self.visual_proj(visual_features)
        )  # [B, N, hidden_dim]

        # Chuyển visual_mask → key_padding_mask cho nn.MultiheadAttention
        # Convention: key_padding_mask True = bỏ qua (ngược với visual_mask)
        key_padding_mask: Optional[torch.Tensor] = None
        if visual_mask is not None:
            key_padding_mask = ~visual_mask  # [B, N]

        # --- Bước 2: Expand latents theo batch ---
        B = visual_features.size(0)
        # self.latents: [1, M, D] → [B, M, D]
        latents = self.latents.expand(B, -1, -1)  # [B, num_latents, hidden_dim]

        # --- Bước 3: Qua num_layers lớp Perceiver ---
        # Mỗi lớp: latents attend vào concat(latents, visual_proj)
        for layer in self.layers:
            latents = layer(
                latents=latents,
                visual=visual_proj,
                visual_key_padding_mask=key_padding_mask,
            )  # [B, num_latents, hidden_dim]

        # LayerNorm cuối
        latents = self.norm_out(latents)  # [B, num_latents, hidden_dim]

        # --- Bước 4: Mean-pool latents → vector đại diện ---
        # Tổng hợp thông tin từ 64 latent thành 1 vector [B, hidden_dim]
        pooled = latents.mean(dim=1)  # [B, hidden_dim]

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
            f"num_latents={self.num_latents}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"num_answers={self.num_answers}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_perceiver_resampler(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    hidden_dim: int = QFORMER_HIDDEN_SIZE,
    num_latents: int = _DEFAULT_NUM_LATENTS,
    num_layers: int = _DEFAULT_NUM_LAYERS,
    num_heads: int = _DEFAULT_NUM_HEADS,
    num_answers: int = ANSWER_VOCAB_SIZE,
    dropout: float = 0.1,
) -> PerceiverResampler:
    """Khởi tạo model PerceiverResampler với các tham số tuỳ chỉnh.

    Args:
        visual_dim:  Chiều patch token visual.  Mặc định 1024 (CLIP ViT-L/14).
        text_dim:    Chiều text CLS token.       Mặc định 768  (BERT-base).
        hidden_dim:  Chiều latents và attention. Mặc định 768.
        num_latents: Số latent vector học được.  Mặc định 64.
        num_layers:  Số lớp Perceiver.           Mặc định 4.
        num_heads:   Số đầu attention.            Mặc định 12.
        num_answers: Kích thước bộ từ vựng.      Mặc định 3129 (VQAv2).
        dropout:     Xác suất dropout.            Mặc định 0.1.

    Returns:
        Một instance của :class:`PerceiverResampler`.

    Ví dụ::

        model = build_perceiver_resampler()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return PerceiverResampler(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_latents=num_latents,
        num_layers=num_layers,
        num_heads=num_heads,
        num_answers=num_answers,
        dropout=dropout,
    )
