from models.qformer import QFormer, QFormerConfig
from models.blip2_vqa import BLIP2VQA
from models.text_encoder import FrozenTextEncoder

# --- Fusion baselines (mỗi file tương ứng một thí nghiệm) ---
from models.exp01_mean_linear import MeanLinearFusion, build_mean_linear
from models.exp02_concat_mlp import ConcatMLPFusion, build_concat_mlp
from models.exp03_mlb import MLBFusion, build_mlb
from models.exp04_mfb import MFBFusion, build_mfb
from models.exp05_cross_attn import CrossAttnFusion, build_cross_attn
from models.exp06_qformer_scratch import QFormerScratch, build_qformer_scratch
from models.exp07_perceiver_resampler import PerceiverResampler, build_perceiver_resampler

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_MEAN_LINEAR,
    MODEL_CONCAT_FUSION,
    MODEL_MLB_FUSION,
    MODEL_MFB_FUSION,
    MODEL_CROSS_ATTN_FUSION,
    MODEL_QFORMER_SCRATCH,
    MODEL_PERCEIVER_RESAMPLER,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
    VALID_MODEL_NAMES,
)

import torch.nn as nn

# ---------------------------------------------------------------------------
# Registry: ánh xạ tên model → factory function
# ---------------------------------------------------------------------------

_EXP_REGISTRY = {
    MODEL_MEAN_LINEAR:       build_mean_linear,
    MODEL_CONCAT_FUSION:     build_concat_mlp,
    MODEL_MLB_FUSION:        build_mlb,
    MODEL_MFB_FUSION:        build_mfb,
    MODEL_CROSS_ATTN_FUSION: build_cross_attn,
    MODEL_QFORMER_SCRATCH:   build_qformer_scratch,
    MODEL_PERCEIVER_RESAMPLER: build_perceiver_resampler,
}


def build_model(config) -> nn.Module:
    """Khởi tạo EXP fusion model từ config object.

    Args:
        config: OmegaConf config — phải có trường ``config.model.name``
                khớp với một trong ``VALID_MODEL_NAMES``.

    Returns:
        Instance của EXP model tương ứng.

    Raises:
        ValueError: Nếu ``model.name`` không hợp lệ hoặc là ``blip2_vqa``
                    (BLIP2VQA dùng pipeline riêng).
    """
    name = config.model.name
    if name not in _EXP_REGISTRY:
        if name == "blip2_vqa":
            raise ValueError(
                "blip2_vqa dùng BLIP2VQA pipeline riêng — không dùng build_model()."
            )
        raise ValueError(
            f"Tên model '{name}' không hợp lệ. "
            f"Các tên hợp lệ: {sorted(_EXP_REGISTRY.keys())}"
        )
    factory = _EXP_REGISTRY[name]
    return factory(
        visual_dim=getattr(config.model, "vision_width", VISION_ENCODER_WIDTH),
        text_dim=getattr(config.model, "text_dim", QFORMER_HIDDEN_SIZE),
        num_answers=getattr(config.model, "num_answers", ANSWER_VOCAB_SIZE),
    )


__all__ = [
    # Kiến trúc nền
    "QFormer",
    "QFormerConfig",
    "BLIP2VQA",
    # Text encoder (Phương án B)
    "FrozenTextEncoder",
    # Factory tổng hợp
    "build_model",
    "BLIP2VQA",
    # EXP-01
    "MeanLinearFusion",
    "build_mean_linear",
    # EXP-02
    "ConcatMLPFusion",
    "build_concat_mlp",
    # EXP-03
    "MLBFusion",
    "build_mlb",
    # EXP-04
    "MFBFusion",
    "build_mfb",
    # EXP-05
    "CrossAttnFusion",
    "build_cross_attn",
    # EXP-06
    "QFormerScratch",
    "build_qformer_scratch",
    # EXP-07
    "PerceiverResampler",
    "build_perceiver_resampler",
]
