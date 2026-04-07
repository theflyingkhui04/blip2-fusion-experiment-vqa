"""EXP-03: MLB — Multi-modal Low-rank Bilinear Fusion cho VQA.

Thay vì ghép nối (concat) hai vector rồi chiếu tuyến tính, MLB xấp xỉ tương
tác bilinear đầy đủ bằng **tích Hadamard** (element-wise product) sau khi chiếu
cả hai modaliti về cùng một không gian ẩn.

Tích Hadamard mô hình hóa tương tác NHÂN giữa từng chiều của visual và text —
bắt được cross-modal correlation mà concat/cộng đơn thuần bỏ qua.

Ý tưởng toán học:
    Bilinear đầy đủ:  f = v^T · W · t   →   O(D_v × D_t × D_out) tham số  (quá lớn)
    MLB xấp xỉ:       f = tanh(W_v · v)  ⊙  tanh(W_t · t)         →  O(D × (D_v + D_t)) tham số

Luồng xử lý:
    visual: [B, N, visual_dim]  →  mean(dim=1)  →  [B, visual_dim]
                                →  Linear(visual_dim, fusion_dim)
                                →  tanh  →  Dropout
                                →  [B, fusion_dim]

    text:   [B, text_dim]       →  Linear(text_dim, fusion_dim)
                                →  tanh  →  Dropout
                                →  [B, fusion_dim]

    fused:  v ⊙ t               →  [B, fusion_dim]
                                →  MLP(fusion_dim, fusion_dim, num_answers)
                                →  logits [B, num_answers]

Tham khảo:
    Kim et al., "Hadamard Product for Low-rank Bilinear Pooling", ICLR 2017.
    https://arxiv.org/abs/1610.04325
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_MLB_FUSION,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)


class MLBFusion(nn.Module):
    """Baseline MLB — Multi-modal Low-rank Bilinear (EXP-03).

    Chiếu visual và text features về cùng chiều ``fusion_dim``, áp dụng
    tanh + Dropout, sau đó kết hợp bằng tích Hadamard (element-wise product).
    Vector fused được đưa qua MLP để dự đoán câu trả lời.

    Args:
        visual_dim:  Chiều của mỗi visual patch token.
                     Mặc định 1024 — CLIP ViT-L/14 (contracts.VISION_ENCODER_WIDTH).
        text_dim:    Chiều của text feature vector (CLS token BERT-base).
                     Mặc định 768 — BERT-base (contracts.QFORMER_HIDDEN_SIZE).
        fusion_dim:  Chiều không gian ẩn chung sau khi chiếu.
                     Paper MLB gốc dùng 1200, ở đây mặc định 2048 cho VQAv2.
        num_answers: Số lớp đầu ra.
                     Mặc định 3129 — VQAv2 (contracts.ANSWER_VOCAB_SIZE).
        dropout:     Xác suất dropout sau tanh ở mỗi nhánh. Mặc định 0.1.
    """

    # Tên mô hình — phải khớp với contracts.MODEL_MLB_FUSION
    MODEL_NAME: str = MODEL_MLB_FUSION

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,   # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,       # 768
        fusion_dim: int = 2048,
        num_answers: int = ANSWER_VOCAB_SIZE,      # 3129
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_dim  = visual_dim
        self.text_dim    = text_dim
        self.fusion_dim  = fusion_dim
        self.num_answers = num_answers

        # Nhánh visual: chiếu về fusion_dim rồi tanh + dropout
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)

        # Nhánh text: chiếu về fusion_dim rồi tanh + dropout
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.dropout = nn.Dropout(p=dropout)

        # Classifier MLP sau tích Hadamard:
        #   fusion_dim → fusion_dim → num_answers
        # Thêm một lớp ẩn để tăng khả năng biểu diễn
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_answers),
        )

        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo tất cả các lớp Linear bằng xavier_uniform."""
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
                # Tính trung bình có trọng số — bỏ qua padding token
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

        # --- Bước 2: Chiếu và kích hoạt từng nhánh ---
        # Nhánh visual: Linear → tanh → Dropout   →   [B, fusion_dim]
        v = self.dropout(torch.tanh(self.visual_proj(visual_pooled)))

        # Nhánh text:   Linear → tanh → Dropout   →   [B, fusion_dim]
        t = self.dropout(torch.tanh(self.text_proj(text_features)))

        # --- Bước 3: Tích Hadamard (element-wise product) ---
        # f = v ⊙ t  →  [B, fusion_dim]
        # Đây là bước quan trọng: mỗi chiều thứ i của fused = v[i] × t[i]
        # → mô hình hóa tương tác multiplicative giữa visual và text
        fused = v * t

        # --- Bước 4: Phân loại qua MLP ---
        # logits: [B, num_answers]
        return self.classifier(fused)

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

def build_mlb(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    fusion_dim: int = 2048,
    num_answers: int = ANSWER_VOCAB_SIZE,
    dropout: float = 0.1,
) -> MLBFusion:
    """Khởi tạo model MLBFusion với các tham số tuỳ chỉnh.

    Args:
        visual_dim:  Chiều visual patch token. Mặc định 1024 (CLIP ViT-L/14).
        text_dim:    Chiều text CLS token.     Mặc định 768  (BERT-base).
        fusion_dim:  Chiều không gian ẩn MLB.  Mặc định 2048.
        num_answers: Kích thước bộ từ vựng.   Mặc định 3129 (VQAv2).
        dropout:     Xác suất dropout.         Mặc định 0.1.

    Returns:
        Một instance của :class:`MLBFusion`.

    Ví dụ::

        model = build_mlb()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return MLBFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        fusion_dim=fusion_dim,
        num_answers=num_answers,
        dropout=dropout,
    )
