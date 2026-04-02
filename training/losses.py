"""Các hàm loss cho huấn luyện mô hình VQA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.contracts import (
    LABEL_IGNORE_INDEX,
    LOSS_BCE,
    LOSS_CE,
    LOSS_KL,
    VALID_LOSS_TYPES,
)


class VQALoss(nn.Module):
    """Hàm loss kết hợp cho các thí nghiệm VQA.

    Hỗ trợ ba mục tiêu huấn luyện:

    * **Soft-target BCE** (``"bce"``): binary cross-entropy so với soft scores
      của từng đáp án, được tính từ annotation VQA (``min(count / 3, 1)``).
      Thường dùng khi huấn luyện với bộ answer vocabulary cố định.
    * **Cross-entropy** (``"ce"``): cross-entropy đa lớp chuẩn so với một
      hard label duy nhất. Phù hợp khi fine-tune generative decoder.
    * **KL divergence** (``"kl"``): coi soft scores như một phân phối xác
      suất hợp lệ và tối thiểu hoá KL divergence.

    Args:
        loss_type: Một trong ``"bce"``, ``"ce"``, hoặc ``"kl"``.
        label_smoothing: Label smoothing cho loss ``"ce"`` (0.0 = tắt).
        reduction: ``"mean"`` hoặc ``"sum"``.
    """

    def __init__(
        self,
        loss_type: str = LOSS_BCE,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type phải là một trong {sorted(VALID_LOSS_TYPES)}; nhận được '{loss_type}'"
            )
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == LOSS_CE:
            self.ce = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction=reduction,
            )

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Tính giá trị loss.

        Args:
            logits: Logits đầu ra của mô hình.
                Shape ``[B, num_answers]`` cho ``"bce"`` / ``"kl"`` và ``"ce"``.
            labels: Nhãn ground-truth.
                * Với ``"bce"`` / ``"kl"``: soft score tensor ``[B, num_answers]``
                  có giá trị trong khoảng ``[0, 1]``.
                * Với ``"ce"``: long integer tensor ``[B]`` chứa chỉ số class.

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == LOSS_CE:
            return self.ce(logits, labels)

        if self.loss_type == LOSS_BCE:
            return F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                reduction=self.reduction,
            )

        # KL divergence — chuẩn hoá soft scores thành phân phối xác suất hợp lệ
        eps = 1e-8
        target_probs = labels.float()
        target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + eps)
        log_probs = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1)
        return kl.mean() if self.reduction == "mean" else kl.sum()


class GenerativeLoss(nn.Module):
    """Cross-entropy loss trên chuỗi token sinh ra (generative mode).

    Bọc :class:`torch.nn.CrossEntropyLoss` với label smoothing tuỳ chọn và
    bỏ qua các vị trí padding mặc định
    (``ignore_index=LABEL_IGNORE_INDEX`` = ``-100``).

    Args:
        ignore_index: Token id bị bỏ qua khi tính loss.
        label_smoothing: Hệ số label smoothing.
    """

    def __init__(
        self,
        ignore_index: int = LABEL_IGNORE_INDEX,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Tính cross-entropy loss cho sequence generation.

        Args:
            logits: ``[B, L, vocab_size]`` — logits của từng bước thời gian.
            labels: ``[B, L]`` — token id nguyên; vị trí padding phải được
                đặt thành ``ignore_index`` (mặc định ``-100``).

        Returns:
            Scalar loss.
        """
        # Dịch chuyển 1 bước để dự đoán token tiếp theo (next-token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
