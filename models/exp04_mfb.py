"""EXP-04: MFB — Multi-modal Factorized Bilinear Pooling cho VQA.

MFB là bước tiến hóa từ MLB: thay vì dùng **một tập** chiếu bilinear, MFB
dùng **k tập song song** (gọi là k "factors"), mỗi tập tính tích Hadamard
độc lập, sau đó **sum-pool** qua k factors để thu vector bilinear compressed.
Cuối cùng áp dụng **sqrt (power norm) + L2-norm** giúp ổn định gradient.

Ý tưởng toán học (so sánh với MLB):
    MLB: f = tanh(W_v · v) ⊙ tanh(W_t · t)           →  [B, fusion_dim]

    MFB: chiếu v → [B, fusion_dim × k] và t → [B, fusion_dim × k]
         Hadamard từng cặp factor: z = v̂ ⊙ t̂         →  [B, fusion_dim × k]
         Sum-pool qua k factors:   z = z.view(B, -1, k).sum(-1)  →  [B, fusion_dim]
         Power norm + L2 norm:     z = sign(z)·sqrt(|z|) / ||z||₂

Giải thích trực quan về sum-pool qua k factors:
    - Mỗi factor học một "góc nhìn" bilinear khác nhau về quan hệ v–t.
    - Sum-pool tổng hợp tất cả k góc nhìn → biểu diễn phong phú hơn.
    - Tương tự multi-head attention nhưng ở cấp pooling.

Luồng xử lý:
    visual: [B, N, visual_dim]  →  mean(dim=1)  →  [B, visual_dim]
                                →  Linear(visual_dim, proj_dim)  →  Dropout
                                →  [B, proj_dim]   (proj_dim = fusion_dim × k)

    text:   [B, text_dim]       →  Linear(text_dim, proj_dim)    →  Dropout
                                →  [B, proj_dim]

    Hadamard:  z = v ⊙ t        →  [B, proj_dim]
    Sum-pool:  z = z.view(B, fusion_dim, k).sum(-1)  →  [B, fusion_dim]
    Norm:      z = sign(z)·sqrt(|z|)                 →  [B, fusion_dim]
               z = F.normalize(z, p=2, dim=-1)        →  [B, fusion_dim]

    Classifier: z → Linear(fusion_dim, num_answers)  →  logits [B, num_answers]

Tham khảo:
    Zhou et al., "MFB: Multi-modal Factorized Bilinear Pooling with Co-Attention
    Learning for Visual Question Answering", ICCV 2017.
    https://arxiv.org/abs/1708.01471
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_MFB_FUSION,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)


class MFBFusion(nn.Module):
    """Baseline MFB — Multi-modal Factorized Bilinear Pooling (EXP-04).

    Hai nhánh visual và text được chiếu về không gian chiều cao (``proj_dim =
    fusion_dim × k``), kết hợp bằng tích Hadamard, rồi **sum-pool** qua k
    factors để nén về ``fusion_dim``. Power norm (sqrt) + L2-norm ổn định
    phân phối đặc trưng trước khi phân loại.

    Args:
        visual_dim:  Chiều của visual patch token.
                     Mặc định 1024 — CLIP ViT-L/14 (contracts.VISION_ENCODER_WIDTH).
        text_dim:    Chiều của text CLS token.
                     Mặc định 768 — BERT-base (contracts.QFORMER_HIDDEN_SIZE).
        fusion_dim:  Chiều output sau sum-pool (sau khi nén k factors).
                     Mặc định 1024. Không gian proj_dim = fusion_dim × k.
        k:           Số factors của MFB. Paper gốc dùng k=5. Mặc định 5.
        num_answers: Số lớp đầu ra.
                     Mặc định 3129 — VQAv2 (contracts.ANSWER_VOCAB_SIZE).
        dropout:     Xác suất dropout sau chiếu. Mặc định 0.1.
    """

    # Tên mô hình — phải khớp với contracts.MODEL_MFB_FUSION
    MODEL_NAME: str = MODEL_MFB_FUSION

    def __init__(
        self,
        visual_dim: int = VISION_ENCODER_WIDTH,   # 1024
        text_dim: int = QFORMER_HIDDEN_SIZE,       # 768
        fusion_dim: int = 1024,
        k: int = 5,
        num_answers: int = ANSWER_VOCAB_SIZE,      # 3129
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_dim  = visual_dim
        self.text_dim    = text_dim
        self.fusion_dim  = fusion_dim
        self.k           = k
        self.num_answers = num_answers

        # proj_dim = fusion_dim × k — mỗi factor chiếm fusion_dim chiều
        proj_dim = fusion_dim * k  # ví dụ: 1024 × 5 = 5120

        # Nhánh visual: Linear → Dropout
        self.visual_proj = nn.Linear(visual_dim, proj_dim)

        # Nhánh text: Linear → Dropout
        self.text_proj = nn.Linear(text_dim, proj_dim)

        self.dropout = nn.Dropout(p=dropout)

        # Classifier đơn giản sau khi đã có bilinear feature
        # (fusion_dim → num_answers); thêm phi tuyến nhẹ
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_answers),
        )

        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo tất cả Linear bằng xavier_uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Hàm chuẩn hóa đặc trưng bilinear
    # ------------------------------------------------------------------

    @staticmethod
    def _power_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Áp dụng power norm (signed sqrt) + L2 norm theo chiều cuối.

        Power norm giảm ảnh hưởng của các giá trị ngoại lệ lớn — quan trọng
        khi đặc trưng là tích của hai phép chiếu (có thể bùng nổ về magnitude).

        Công thức:  z = sign(x) · sqrt(|x| + eps) / ||z||₂
        """
        # Signed square root (còn gọi là "signed sqrt" hay "element-wise power 0.5")
        x_sqrt = x.sign() * (x.abs() + eps).sqrt()
        # L2 normalize theo chiều feature (dim=-1)
        return F.normalize(x_sqrt, p=2, dim=-1)

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
            visual_mask:     Mask boolean ``[B, N_patches]`` (True = giữ).
                             Chỉ có tác dụng khi visual_features là patch sequence.
                             Nếu None, mean pool không có trọng số.

        Returns:
            Logits chưa qua softmax. Shape ``[B, num_answers]``.
        """
        # --- Bước 1: Mean pooling visual patch tokens ---
        if visual_features.dim() == 3:
            # visual_features: [B, N, D]
            if visual_mask is not None:
                # Mean pool có trọng số — bỏ qua token padding
                mask_float = visual_mask.float().unsqueeze(-1)          # [B, N, 1]
                visual_sum = (visual_features * mask_float).sum(dim=1)  # [B, D]
                token_count = mask_float.sum(dim=1).clamp(min=1.0)      # [B, 1]
                visual_pooled = visual_sum / token_count                 # [B, D]
            else:
                # Mean toàn bộ 257 token (CLS + 256 patch)
                visual_pooled = visual_features.mean(dim=1)             # [B, D]
        else:
            # visual_features đã được pool: [B, D]
            visual_pooled = visual_features

        # --- Bước 2: Chiếu lên không gian proj_dim ---
        # v_proj: [B, proj_dim]  (= fusion_dim × k)
        v_proj = self.dropout(self.visual_proj(visual_pooled))
        # t_proj: [B, proj_dim]
        t_proj = self.dropout(self.text_proj(text_features))

        # --- Bước 3: Tích Hadamard toàn bộ proj_dim chiều ---
        # z: [B, proj_dim]  — mỗi cặp (v[i], t[i]) tương tác multiplicative
        z = v_proj * t_proj

        # --- Bước 4: Sum-pool qua k factors ---
        # Reshape: [B, proj_dim] → [B, fusion_dim, k]
        # Mỗi nhóm k chiều liên tiếp tương ứng 1 "tập factor" của bilinear
        B = z.size(0)
        z = z.view(B, self.fusion_dim, self.k)  # [B, fusion_dim, k]

        # Sum theo chiều k (dim=-1): tổng hợp k factors
        z = z.sum(dim=-1)                        # [B, fusion_dim]

        # --- Bước 5: Power norm + L2 norm ---
        # Ổn định phân phối đặc trưng bilinear trước khi đưa vào classifier
        z = self._power_norm(z)                  # [B, fusion_dim]

        # --- Bước 6: Phân loại ---
        # logits: [B, num_answers]
        return self.classifier(z)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Thông tin bổ sung khi print(model)."""
        return (
            f"visual_dim={self.visual_dim}, "
            f"text_dim={self.text_dim}, "
            f"fusion_dim={self.fusion_dim}, "
            f"k={self.k}, "
            f"proj_dim={self.fusion_dim * self.k}, "
            f"num_answers={self.num_answers}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_mfb(
    visual_dim: int = VISION_ENCODER_WIDTH,
    text_dim: int = QFORMER_HIDDEN_SIZE,
    fusion_dim: int = 1024,
    k: int = 5,
    num_answers: int = ANSWER_VOCAB_SIZE,
    dropout: float = 0.1,
) -> MFBFusion:
    """Khởi tạo model MFBFusion với các tham số tuỳ chỉnh.

    Args:
        visual_dim:  Chiều visual patch token.  Mặc định 1024 (CLIP ViT-L/14).
        text_dim:    Chiều text CLS token.      Mặc định 768  (BERT-base).
        fusion_dim:  Chiều output sau sum-pool. Mặc định 1024.
        k:           Số factors MFB.            Mặc định 5 (giống paper gốc).
        num_answers: Kích thước bộ từ vựng.    Mặc định 3129 (VQAv2).
        dropout:     Xác suất dropout.          Mặc định 0.1.

    Returns:
        Một instance của :class:`MFBFusion`.

    Ví dụ::

        model = build_mfb()
        visual = torch.randn(4, 257, 1024)  # 4 ảnh, 257 patch tokens
        text   = torch.randn(4, 768)         # 4 câu hỏi, CLS embedding
        logits = model(visual, text)         # [4, 3129]
    """
    return MFBFusion(
        visual_dim=visual_dim,
        text_dim=text_dim,
        fusion_dim=fusion_dim,
        k=k,
        num_answers=num_answers,
        dropout=dropout,
    )
