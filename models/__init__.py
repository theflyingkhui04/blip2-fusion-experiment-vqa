import torch.nn as nn

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    MODEL_CONCAT_FUSION,
    MODEL_CROSS_ATTN_FUSION,
    MODEL_MEAN_LINEAR,
    MODEL_MFB_FUSION,
    MODEL_MLB_FUSION,
    MODEL_PERCEIVER_RESAMPLER,
    MODEL_QFORMER_SCRATCH,
    QFORMER_HIDDEN_SIZE,
    VISION_ENCODER_WIDTH,
)
from models.blip2_vqa import BLIP2VQA
from models.exp01_mean_linear import MeanLinearFusion, build_mean_linear
from models.exp02_concat_mlp import ConcatMLPFusion, build_concat_mlp
from models.exp03_mlb import MLBFusion, build_mlb
from models.exp04_mfb import MFBFusion, build_mfb
from models.exp05_cross_attn import CrossAttnFusion, build_cross_attn
from models.exp06_qformer_scratch import QFormerScratch, build_qformer_scratch
from models.exp07_perceiver_resampler import PerceiverResampler, build_perceiver_resampler
from models.qformer import QFormer, QFormerConfig
from models.text_encoder import FrozenTextEncoder

_EXP_REGISTRY = {
    MODEL_MEAN_LINEAR: build_mean_linear,
    MODEL_CONCAT_FUSION: build_concat_mlp,
    MODEL_MLB_FUSION: build_mlb,
    MODEL_MFB_FUSION: build_mfb,
    MODEL_CROSS_ATTN_FUSION: build_cross_attn,
    MODEL_QFORMER_SCRATCH: build_qformer_scratch,
    MODEL_PERCEIVER_RESAMPLER: build_perceiver_resampler,
}


def _model_kwargs(config, name: str) -> dict:
    model = config.model
    kwargs = {
        "visual_dim": getattr(model, "vision_width", VISION_ENCODER_WIDTH),
        "text_dim": getattr(model, "text_dim", QFORMER_HIDDEN_SIZE),
        "num_answers": getattr(model, "num_answers", ANSWER_VOCAB_SIZE),
    }

    if name in {MODEL_CONCAT_FUSION, MODEL_MLB_FUSION, MODEL_MFB_FUSION}:
        fusion_dim_default = 2048 if name == MODEL_MLB_FUSION else 1024
        kwargs.update(
            fusion_dim=getattr(model, "fusion_output_size", fusion_dim_default),
            dropout=getattr(model, "dropout", 0.1),
        )
    elif name == MODEL_CROSS_ATTN_FUSION:
        kwargs.update(
            hidden_dim=getattr(model, "hidden_size", QFORMER_HIDDEN_SIZE),
            num_queries=getattr(model, "num_query_tokens", 32),
            num_layers=getattr(model, "num_layers", 3),
            num_heads=getattr(model, "num_heads", 12),
            dropout=getattr(model, "dropout", 0.1),
        )
    elif name == MODEL_QFORMER_SCRATCH:
        kwargs.update(
            hidden_size=getattr(model, "hidden_size", QFORMER_HIDDEN_SIZE),
            num_queries=getattr(model, "num_query_tokens", 32),
            num_layers=getattr(model, "num_layers", 12),
            num_heads=getattr(model, "num_heads", 12),
            intermediate_size=getattr(model, "intermediate_size", 3072),
            dropout=getattr(model, "dropout", 0.1),
        )
    elif name == MODEL_PERCEIVER_RESAMPLER:
        kwargs.update(
            hidden_dim=getattr(model, "hidden_size", QFORMER_HIDDEN_SIZE),
            num_latents=getattr(
                model, "num_latents", getattr(model, "num_query_tokens", 64)
            ),
            num_layers=getattr(model, "num_layers", 4),
            num_heads=getattr(model, "num_heads", 12),
            dropout=getattr(model, "dropout", 0.1),
        )

    return kwargs


def build_model(config) -> nn.Module:
    name = config.model.name
    if name not in _EXP_REGISTRY:
        if name == "blip2_vqa":
            raise ValueError("blip2_vqa uses the BLIP2VQA pipeline.")
        raise ValueError(
            f"Unknown model '{name}'. Valid EXP models: {sorted(_EXP_REGISTRY.keys())}"
        )
    return _EXP_REGISTRY[name](**_model_kwargs(config, name))


__all__ = [
    "QFormer",
    "QFormerConfig",
    "BLIP2VQA",
    "FrozenTextEncoder",
    "build_model",
    "MeanLinearFusion",
    "build_mean_linear",
    "ConcatMLPFusion",
    "build_concat_mlp",
    "MLBFusion",
    "build_mlb",
    "MFBFusion",
    "build_mfb",
    "CrossAttnFusion",
    "build_cross_attn",
    "QFormerScratch",
    "build_qformer_scratch",
    "PerceiverResampler",
    "build_perceiver_resampler",
]
