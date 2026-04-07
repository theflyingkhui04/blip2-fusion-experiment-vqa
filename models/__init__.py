from models.qformer import QFormer, QFormerConfig
from models.blip2_vqa import BLIP2VQA

# --- Fusion baselines (mỗi file tương ứng một thí nghiệm) ---
from models.exp01_mean_linear import MeanLinearFusion, build_mean_linear
from models.exp02_concat_mlp import ConcatMLPFusion, build_concat_mlp
from models.exp03_mlb import MLBFusion, build_mlb
from models.exp04_mfb import MFBFusion, build_mfb
from models.exp05_cross_attn import CrossAttnFusion, build_cross_attn
from models.exp06_qformer_scratch import QFormerScratch, build_qformer_scratch
from models.exp07_perceiver_resampler import PerceiverResampler, build_perceiver_resampler

__all__ = [
    # Kiến trúc nền
    "QFormer",
    "QFormerConfig",
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
