"""EXP-01: Mean Pooling + Linear Fusion cho VQA.

Đây là baseline đơn giản nhất trong bộ thí nghiệm — thiết lập điểm sàn để
so sánh với các phương pháp fusion phức tạp hơn (EXP-02 đến EXP-07).

Luồng xử lý:
    visual: [B, N, visual_dim]  →  mean(dim=1)  →  [B, visual_dim]
    text:   [B, text_dim]       →  (giữ nguyên)
    fused:  concat              →  [B, visual_dim + text_dim]
                                →  Linear(visual_dim + text_dim, num_answers)
                                →  logits [B, num_answers]

Không có lớp ẩn, không có phi tuyến — chỉ một phép chiếu tuyến tính.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_MEAN_LINEAR,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)


class MeanLinearFusion(nn.Module):
    """Baseline Mean Pooling + Linear (EXP-01).

    Gộp toàn bộ visual patch tokens bằng phép trung bình (mean pooling),
    ghép với text CLS token, sau đó dự đoán câu trả lời qua một lớp Linear.

    Đây là fusion yếu nhất có thể, dùng làm mốc tham chiếu điểm sàn.
    Không có tham số ẩn, không có phi tuyến.

    Args:
        visual_dim: Chiều của mỗi visual patch token.
                    Mặc định 1024 — CLIP ViT-L/14 (contracts.VISION_ENCODER_WIDTH).
        text_dim:   Chiều của text feature vector (CLS token BERT-base).
                    Mặc định 768 — BERT-base (contracts.QFORMER_HIDDEN_SIZE).
        num_answers: Số lớp đầu ra (kích thước bộ từ vựng câu trả lời).
                    Mặc định 3129 — VQAv2 (contracts.ANSWER_VOCAB_SIZE).
    """

    # Tên mô hình — phải khớp với contracts.MODEL_MEAN_LINEAR
    MODEL_NAME: str = MODEL_MEAN_LINEAR

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,   # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,       # 768
        num_answers: int = ANSWER_VOCAB_SIZE,      # 3129
    ) -> None:
        super().__init__()

        self.visual_dim  = visual_dim
        self.text_dim    = text_dim
        self.num_answers = num_answers

        # Lớp phân loại tuyến tính duy nhất
        # Đầu vào:  [B, visual_dim + text_dim]  =  [B, 1792]
        # Đầu ra:   [B, num_answers]             =  [B, 3129]
        self.classifier = nn.Linear(visual_dim + text_dim, num_answers)

        # Khởi tạo trọng số theo xavier_uniform để training ổn định
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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
                # visual_mask: [B, N] — True = token hợp lệ
                mask_float = visual_mask.float().unsqueeze(-1)          # [B, N, 1]
                visual_sum = (visual_features * mask_float).sum(dim=1)  # [B, D]
                token_count = mask_float.sum(dim=1).clamp(min=1.0)      # [B, 1]
                visual_pooled = visual_sum / token_count                 # [B, D]
            else:
                # Không có mask → mean toàn bộ 257 token
                visual_pooled = visual_features.mean(dim=1)             # [B, D]
        else:
            # visual_features đã được pool bên ngoài: [B, D]
            visual_pooled = visual_features

        # --- Bước 2: Ghép visual và text ---
        # fused: [B, visual_dim + text_dim]
        fused = torch.cat([visual_pooled, text_features], dim=-1)

        # --- Bước 3: Phân loại tuyến tính ---
        # logits: [B, num_answers]
        return self.classifier(fused)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Thông tin bổ sung khi print(model)."""
        return (
            f"visual_dim={self.visual_dim}, "
            f"text_dim={self.text_dim}, "
            f"num_answers={self.num_answers}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_mean_linear(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    num_answers: int = ANSWER_VOCAB_SIZE,
) -> MeanLinearFusion:
    """Khởi tạo model MeanLinearFusion với các tham số tuỳ chỉnh.

    Args:
        visual_dim:  Chiều visual patch token. Mặc định 1024 (CLIP ViT-L/14).
        text_dim:    Chiều text CLS token.     Mặc định 768  (BERT-base).
        num_answers: Kích thước bộ từ vựng.   Mặc định 3129 (VQAv2).

    Returns:
        Một instance của :class:`MeanLinearFusion`.

    Ví dụ::

        model = build_mean_linear()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return MeanLinearFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        num_answers=num_answers,
    )
