"""EXP-06: Q-Former from Scratch cho VQA.

Đây là "chuẩn vàng" (gold standard) của bộ thí nghiệm — dùng kiến trúc
Q-Former đầy đủ từ BLIP-2 được train hoàn toàn từ đầu (không load pretrained
weight), kết hợp với text CLS token để dự đoán câu trả lời VQA.

So sánh với EXP-05 (Cross-Attention Bridge):
    EXP-05: 3 lớp cross-attention thuần túy — queries chỉ attend vào visual
    EXP-06: 12 lớp Q-Former — queries vừa self-attend với nhau (học inter-query
            relation), vừa cross-attend visual mỗi 2 lớp (cross_attention_freq=2)

Kiến trúc Q-Former (từ `models/qformer.py`):
    queries: [B, 32, 768]   (learnable, train từ đầu)
    visual:  [B, 257, 1024] → visual_proj(Linear 1024→768) → visual_norm → [B, 257, 768]

    12 khối QFormerLayer:
      lớp chẵn (0, 2, 4, 6, 8, 10):
        self-attn(queries ↔ queries)
        cross-attn(queries ← visual)   ← query học thông tin từ image patch
        FFN
      lớp lẻ  (1, 3, 5, 7, 9, 11):
        self-attn(queries ↔ queries)   ← queries trao đổi thông tin với nhau
        FFN

    → output: [B, 32, 768]

Fusion với text:
    pooled = output.mean(dim=1)            → [B, 768]
    fused  = concat(pooled, text_cls)      → [B, 1536]
    logits = MLP(1536, 1024, num_answers)  → [B, 3129]

Lưu ý then chốt:
    - Reuse hoàn toàn `models.qformer.QFormer` và `models.qformer.QFormerConfig`.
    - QFormerConfig.vision_width phải được set đúng VISION_ENCODER_WIDTH (1024)
      chứ không phải default 1408 của BLIP-2 paper.
    - Model này cần patch token đầy đủ [B, 257, 1024] — không dùng CLS-only cache.
    - Ước tính ~190M params — tốn VRAM nhất trong bộ 7 EXP.

Tham khảo:
    Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
    Image Encoders and Large Language Models", ICML 2023.
    https://arxiv.org/abs/2301.12597
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_QFORMER_SCRATCH,
    NUM_QUERY_TOKENS,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)
from models.qformer import QFormer, QFormerConfig

# Số lớp Q-Former theo experiment-cases.md (12 lớp — giống BLIP-2 gốc)
_DEFAULT_NUM_LAYERS: int = 12
# Cross-attention được chèn mỗi 2 lớp (lớp chẵn: 0,2,4,...,10)
_DEFAULT_CROSS_ATTN_FREQ: int = 2
# Số đầu attention (chuẩn BERT-base)
_DEFAULT_NUM_HEADS: int = 12
# Chiều ẩn FFN (768 × 4 = 3072)
_DEFAULT_INTERMEDIATE_SIZE: int = 3072


class QFormerScratch(nn.Module):
    """Baseline Q-Former from Scratch (EXP-06).

    Wrap ``models.qformer.QFormer`` với classifier MLP phía sau để tạo thành
    pipeline VQA end-to-end. Toàn bộ trọng số được train từ đầu — không load
    pretrained weight từ BLIP-2.

    So sánh với EXP-05:
        - EXP-05 có 3 lớp cross-attn, không có self-attention giữa queries.
        - EXP-06 có 12 lớp Q-Former với cả self-attn + cross-attn xen kẽ.
        - Queries trong EXP-06 học được quan hệ *giữa chúng với nhau* qua
          self-attention — tạo ra biểu diễn phong phú hơn nhiều.

    Args:
        visual_dim:    Chiều patch token visual.    Mặc định 1024 (CLIP ViT-L/14).
        text_dim:      Chiều text CLS token.         Mặc định 768  (BERT-base).
        hidden_size:   Chiều ẩn Q-Former.            Mặc định 768.
        num_queries:   Số query token học được.      Mặc định 32.
        num_layers:    Số lớp Q-Former.              Mặc định 12.
        num_heads:     Số đầu attention.              Mặc định 12.
        intermediate_size: Chiều ẩn FFN trong Q-Former. Mặc định 3072.
        cross_attn_freq: Tần suất chèn cross-attn.  Mặc định 2 (lớp chẵn).
        num_answers:   Kích thước vocab câu trả lời. Mặc định 3129 (VQAv2).
        dropout:       Xác suất dropout.             Mặc định 0.1.
    """

    # Tên mô hình — phải khớp với contracts.MODEL_QFORMER_SCRATCH
    MODEL_NAME: str = MODEL_QFORMER_SCRATCH

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,      # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,          # 768
        hidden_size: int = QFORMER_HIDDEN_SIZE,       # 768
        num_queries: int = NUM_QUERY_TOKENS,          # 32
        num_layers: int = _DEFAULT_NUM_LAYERS,        # 12
        num_heads: int = _DEFAULT_NUM_HEADS,          # 12
        intermediate_size: int = _DEFAULT_INTERMEDIATE_SIZE,  # 3072
        cross_attn_freq: int = _DEFAULT_CROSS_ATTN_FREQ,      # 2
        num_answers: int = ANSWER_VOCAB_SIZE,         # 3129
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_dim    = visual_dim
        self.text_dim      = text_dim
        self.hidden_size   = hidden_size
        self.num_queries   = num_queries
        self.num_layers    = num_layers
        self.num_answers   = num_answers

        # --- Khởi tạo Q-Former với config tùy chỉnh ---
        # Quan trọng: vision_width phải khớp với VISION_ENCODER_WIDTH (1024)
        # không phải 1408 như trong QFormerConfig default
        qformer_config = QFormerConfig(
            num_query_tokens=num_queries,
            vision_width=visual_dim,          # 1024 — CLIP ViT-L/14
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            cross_attention_freq=cross_attn_freq,
        )

        # QFormer bao gồm:
        #   - query_tokens (learnable): [1, 32, 768]
        #   - visual_proj + visual_norm
        #   - 12 QFormerLayer (6 lớp có cross-attn + 6 lớp chỉ self-attn + FFN)
        #   - norm cuối
        self.qformer = QFormer(config=qformer_config)

        # --- Classifier MLP ---
        # Input: concat(pooled_queries [B, hidden_size], text_cls [B, text_dim])
        # hidden_size + text_dim = 768 + 768 = 1536
        fusion_dim = hidden_size + text_dim  # 1536
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_answers),
        )

        # Khởi tạo classifier (qformer._init_weights() đã được gọi trong QFormer.__init__)
        self._init_classifier_weights()

    def _init_classifier_weights(self) -> None:
        """Khởi tạo riêng classifier MLP bằng xavier_uniform."""
        for module in self.classifier.modules():
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
                **Bắt buộc shape** ``[B, N_patches, visual_dim]`` để tận dụng
                cross-attention trong Q-Former.
                Nếu đưa vào ``[B, visual_dim]`` (đã pool), vẫn hoạt động nhưng
                mất lợi thế không gian (unsqueeze thành 1 patch token).
            text_features:   CLS token từ BERT-base. Shape ``[B, text_dim]``.
            visual_mask:     Mask boolean ``[B, N_patches]`` — True = token hợp lệ.
                             Khớp với interface ``visual_attention_mask`` của QFormer.

        Returns:
            Logits chưa qua softmax. Shape ``[B, num_answers]``.
        """
        # --- Bước 1: Xử lý trường hợp visual đã bị pool ---
        if visual_features.dim() == 2:
            # [B, visual_dim] → [B, 1, visual_dim]
            # Ghi chú: cross-attn vẫn được tính nhưng chỉ attend vào 1 "patch token"
            visual_features = visual_features.unsqueeze(1)

        # --- Bước 2: Forward qua Q-Former ---
        # QFormer nhận:
        #   visual_features: [B, N, visual_dim]
        #   visual_attention_mask: [B, N]  (True = giữ)
        # QFormer trả về:
        #   query_output: [B, num_queries, hidden_size] = [B, 32, 768]
        query_output = self.qformer(
            visual_features=visual_features,
            visual_attention_mask=visual_mask,
        )  # [B, 32, 768]

        # --- Bước 3: Mean-pool 32 query token → vector đại diện ---
        # Tổng hợp thông tin từ tất cả query token thành 1 vector toàn cục
        pooled = query_output.mean(dim=1)  # [B, 768]

        # --- Bước 4: Concat với text CLS và phân loại ---
        # fused: [B, hidden_size + text_dim] = [B, 1536]
        fused = torch.cat([pooled, text_features], dim=-1)

        # logits: [B, num_answers]
        return self.classifier(fused)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Thông tin bổ sung khi print(model)."""
        total_params = sum(p.numel() for p in self.parameters())
        qformer_params = sum(p.numel() for p in self.qformer.parameters())
        return (
            f"visual_dim={self.visual_dim}, "
            f"hidden_size={self.hidden_size}, "
            f"num_queries={self.num_queries}, "
            f"num_layers={self.num_layers}, "
            f"num_answers={self.num_answers}, "
            f"qformer_params={qformer_params:,}, "
            f"total_params={total_params:,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_qformer_scratch(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    hidden_size: int = QFORMER_HIDDEN_SIZE,
    num_queries: int = NUM_QUERY_TOKENS,
    num_layers: int = _DEFAULT_NUM_LAYERS,
    num_heads: int = _DEFAULT_NUM_HEADS,
    intermediate_size: int = _DEFAULT_INTERMEDIATE_SIZE,
    cross_attn_freq: int = _DEFAULT_CROSS_ATTN_FREQ,
    num_answers: int = ANSWER_VOCAB_SIZE,
    dropout: float = 0.1,
) -> QFormerScratch:
    """Khởi tạo model QFormerScratch với các tham số tuỳ chỉnh.

    Args:
        visual_dim:        Chiều patch token visual.  Mặc định 1024 (CLIP ViT-L/14).
        text_dim:          Chiều text CLS token.       Mặc định 768  (BERT-base).
        hidden_size:       Chiều ẩn Q-Former.          Mặc định 768.
        num_queries:       Số query token học được.    Mặc định 32.
        num_layers:        Số lớp Q-Former.            Mặc định 12.
        num_heads:         Số đầu attention.            Mặc định 12.
        intermediate_size: Chiều FFN ẩn.               Mặc định 3072.
        cross_attn_freq:   Tần suất cross-attn.        Mặc định 2.
        num_answers:       Kích thước bộ từ vựng.      Mặc định 3129 (VQAv2).
        dropout:           Xác suất dropout.            Mặc định 0.1.

    Returns:
        Một instance của :class:`QFormerScratch`.

    Ví dụ::

        model = build_qformer_scratch()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return QFormerScratch(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_size=hidden_size,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        cross_attn_freq=cross_attn_freq,
        num_answers=num_answers,
        dropout=dropout,
    )
