"""Text encoder đóng băng (Frozen BERT-base) cho pipeline VQA.

Phương án B: BERT encoder sống trong trainer (một instance duy nhất cho cả
pipeline), không nằm trong từng EXP model. Điều này tránh load BERT 7 lần
cho 7 thí nghiệm.

Luồng xử lý:
    collate_fn  → input_ids [B, 50], attention_mask [B, 50]
    FrozenTextEncoder.forward()
    → BERT last_hidden_state[:, 0, :]   # CLS token
    → text_features [B, 768]
    → EXP model(visual_features, text_features)

Ghi chú:
    - Tất cả tham số BERT bị đóng băng (requires_grad=False) — không train.
    - Forward luôn chạy trong torch.no_grad() để tiết kiệm bộ nhớ.
    - Không dùng AMP autocast cho BERT để tránh lỗi precision với BertLayerNorm.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from configs.contracts import QFORMER_HIDDEN_SIZE  # 768 — chiều CLS token BERT-base


class FrozenTextEncoder(nn.Module):
    """BERT-base-uncased với toàn bộ tham số bị đóng băng.

    Nhận token id câu hỏi, trả về CLS embedding làm biểu diễn text.
    Model chỉ dùng để inference — không backprop qua BERT.

    Args:
        model_name: Tên HuggingFace model. Mặc định ``"bert-base-uncased"``.

    Ví dụ::

        encoder = FrozenTextEncoder()
        input_ids      = torch.randint(0, 30522, (4, 50))  # [B, L]
        attention_mask = torch.ones(4, 50, dtype=torch.long)
        text_features  = encoder(input_ids, attention_mask)  # [B, 768]
    """

    # Chiều output CLS token BERT-base (khớp với contracts.QFORMER_HIDDEN_SIZE)
    OUTPUT_DIM: int = QFORMER_HIDDEN_SIZE  # 768

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.model_name = model_name

        # Load BERT-base — chỉ lấy encoder (không cần pooler/head)
        self.bert = AutoModel.from_pretrained(model_name)

        # Đóng băng toàn bộ tham số — không cần grad, tiết kiệm VRAM
        for param in self.bert.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mã hóa câu hỏi → CLS token embedding.

        Luôn chạy trong ``torch.no_grad()`` vì BERT đóng băng — không cần
        lưu activation để tính gradient.

        Args:
            input_ids:      Token id từ BERT tokenizer. Shape ``[B, L]``.
            attention_mask: Mask padding ``[B, L]``; 1 = token thật, 0 = padding.
                            Nếu None, mặc định tất cả là 1.

        Returns:
            CLS token embedding. Shape ``[B, 768]``.
        """
        # BERT forward — lấy last_hidden_state [B, L, 768]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token là vị trí đầu tiên (index 0) của sequence
        # Đây là biểu diễn tổng hợp toàn câu — chuẩn cho classification tasks
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        return cls_embedding

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"model={self.model_name}, "
            f"output_dim={self.OUTPUT_DIM}, "
            f"frozen=True"
        )
