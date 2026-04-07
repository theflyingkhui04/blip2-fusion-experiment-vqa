"""EXP-02: Concat + MLP Fusion cho VQA.

Mở rộng EXP-01 bằng cách thêm một lớp ẩn phi tuyến vào sau phép ghép nối.
MLP có khả năng học các tương tác phức tạp hơn giữa visual và text features
so với lớp linear đơn thuần.

Giả thuyết: Phi tuyến tính (ReLU + Dropout) giúp mô hình biểu diễn tốt hơn
và đạt accuracy cao hơn EXP-01 Mean Pooling + Linear.

Luồng xử lý:
    visual: [B, N, visual_dim]  →  mean(dim=1)  →  [B, visual_dim]
    text:   [B, text_dim]       →  (giữ nguyên)
    fused:  concat              →  [B, visual_dim + text_dim]
                                →  Linear(visual_dim + text_dim, fusion_dim)
                                →  ReLU
                                →  Dropout(dropout)
                                →  Linear(fusion_dim, num_answers)
                                →  logits [B, num_answers]

Tham khảo:
    Antol et al., "VQA: Visual Question Answering", ICCV 2015 (baseline ban đầu).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_CONCAT_FUSION,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)


class ConcatMLPFusion(nn.Module):
    """Baseline Concat + MLP (EXP-02).

    Gộp visual patch tokens bằng mean pooling, ghép với text CLS token,
    sau đó đưa qua MLP hai lớp với ReLU và Dropout ở giữa.

    So với EXP-01 (linear đơn), lớp ẩn ``fusion_dim`` cho phép mô hình học
    các tổ hợp phi tuyến của visual và text features trước khi phân loại.

    Args:
        visual_dim:  Chiều của mỗi visual patch token.
                     Mặc định 1024 — CLIP ViT-L/14 (contracts.VISION_ENCODER_WIDTH).
        text_dim:    Chiều của text feature vector (CLS token BERT-base).
                     Mặc định 768 — BERT-base (contracts.QFORMER_HIDDEN_SIZE).
        fusion_dim:  Số unit của lớp ẩn trong MLP. Mặc định 1024.
        num_answers: Số lớp đầu ra (kích thước bộ từ vựng câu trả lời).
                     Mặc định 3129 — VQAv2 (contracts.ANSWER_VOCAB_SIZE).
        dropout:     Xác suất dropout sau lớp ẩn. Mặc định 0.1.
    """

    # Tên mô hình — phải khớp với contracts.MODEL_CONCAT_FUSION
    MODEL_NAME: str = MODEL_CONCAT_FUSION

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,   # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,       # 768
        fusion_dim: int = 1024,
        num_answers: int = ANSWER_VOCAB_SIZE,      # 3129
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_dim  = visual_dim
        self.text_dim    = text_dim
        self.fusion_dim  = fusion_dim
        self.num_answers = num_answers

        # MLP hai lớp:
        #   Lớp 1: chiếu từ không gian ghép nối → lớp ẩn (có phi tuyến)
        #   Lớp 2: chiếu từ lớp ẩn → không gian câu trả lời (logits)
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + text_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_answers),
        )

        # Khởi tạo trọng số tất cả các lớp Linear
        for module in self.mlp.modules():
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
            visual_features: Đặc trưng thị giác từ CLIP ViT-L/14.
                Chấp nhận hai dạng:
                  - Patch sequence: ``[B, N_patches, visual_dim]``  — sẽ tự mean pool.
                  - Đã pool sẵn:    ``[B, visual_dim]``             — dùng trực tiếp.
            text_features:   CLS token từ BERT-base. Shape ``[B, text_dim]``.
            visual_mask:     Mask boolean ``[B, N_patches]`` — True = giữ token.
                             Chỉ có tác dụng khi visual_features là patch sequence
                             (dim == 3). Nếu None, mean pool toàn bộ token.

        Returns:
            Logits chưa qua softmax. Shape ``[B, num_answers]``.
        """
        # --- Bước 1: Mean pooling visual patch tokens ---
        if visual_features.dim() == 3:
            # visual_features: [B, N, D]
            if visual_mask is not None:
                # Chỉ tính trung bình trên các token không phải padding
                mask_float = visual_mask.float().unsqueeze(-1)          # [B, N, 1]
                visual_sum = (visual_features * mask_float).sum(dim=1)  # [B, D]
                token_count = mask_float.sum(dim=1).clamp(min=1.0)      # [B, 1]
                visual_pooled = visual_sum / token_count                 # [B, D]
            else:
                # Không có mask → mean toàn bộ 257 token (1 CLS + 256 patches)
                visual_pooled = visual_features.mean(dim=1)             # [B, D]
        else:
            # visual_features đã được pool bên ngoài: [B, D]
            visual_pooled = visual_features

        # --- Bước 2: Ghép visual và text ---
        # fused: [B, visual_dim + text_dim]  =  [B, 1792]
        fused = torch.cat([visual_pooled, text_features], dim=-1)

        # --- Bước 3: Qua MLP (Linear → ReLU → Dropout → Linear) ---
        # logits: [B, num_answers]  =  [B, 3129]
        return self.mlp(fused)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Thông tin bổ sung khi print(model)."""
        return (
            f"visual_dim={self.visual_dim}, "
            f"text_dim={self.text_dim}, "
            f"fusion_dim={self.fusion_dim}, "
            f"num_answers={self.num_answers}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_concat_mlp(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    fusion_dim: int = 1024,
    num_answers: int = ANSWER_VOCAB_SIZE,
    dropout: float = 0.1,
) -> ConcatMLPFusion:
    """Khởi tạo model ConcatMLPFusion với các tham số tuỳ chỉnh.

    Args:
        visual_dim:  Chiều visual patch token. Mặc định 1024 (CLIP ViT-L/14).
        text_dim:    Chiều text CLS token.     Mặc định 768  (BERT-base).
        fusion_dim:  Kích thước lớp ẩn MLP.   Mặc định 1024.
        num_answers: Kích thước bộ từ vựng.   Mặc định 3129 (VQAv2).
        dropout:     Xác suất dropout.         Mặc định 0.1.

    Returns:
        Một instance của :class:`ConcatMLPFusion`.

    Ví dụ::

        model = build_concat_mlp()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return ConcatMLPFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        fusion_dim=fusion_dim,
        num_answers=num_answers,
        dropout=dropout,
    )
