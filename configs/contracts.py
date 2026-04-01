"""
contracts.py — Unified I/O contracts for the BLIP-2 VQA experiment repo.

PURPOSE
-------
This file is the single source of truth for every tensor shape, dictionary key,
and type that crosses a module boundary.  Any code (model, dataset, trainer,
evaluation) that produces or consumes structured data MUST conform to the
TypedDicts and constants defined here.

RULE: if you add a new key to a batch, a model output, or a prediction record,
you add it here first, then implement it everywhere else.

CONTENTS
--------
1.  Canonical key constants          – string literals re-used across modules.
2.  Batch contract (DataLoader out)  – VQABatch
3.  Model output contracts           – ModelOutput, GenerateOutput
4.  Prediction record contract       – PredictionRecord
5.  Evaluation output contract       – EvalResult
6.  Checkpoint contract              – CheckpointDict
7.  Config sub-contracts             – ModelConfig, DataConfig, TrainingConfig,
                                       OptimizerConfig, SchedulerConfig, LoggingConfig
8.  Fusion model contracts           – FusionInput, FusionOutput
9.  Runtime constants                – ANSWER_VOCAB_SIZE, IMAGE_SIZE, …
"""

from __future__ import annotations

from typing import List, Optional

# ---------------------------------------------------------------------------
# mypy / runtime-safe TypedDict (works on Python 3.8+)
# ---------------------------------------------------------------------------
try:
    from typing import TypedDict, Literal
except ImportError:  # Python 3.7 fallback
    from typing_extensions import TypedDict, Literal  # type: ignore[assignment]

import torch

# ===========================================================================
# 1. Canonical key constants
# ===========================================================================
# Use these constants instead of bare string literals so that a typo becomes
# a NameError instead of a silent bug.

# ── Batch keys ──────────────────────────────────────────────────────────────
KEY_PIXEL_VALUES    = "pixel_values"       # torch.Tensor [B, 3, H, W] float32
KEY_INPUT_IDS       = "input_ids"          # torch.Tensor [B, L]       int64
KEY_ATTENTION_MASK  = "attention_mask"     # torch.Tensor [B, L]       int64 0/1
KEY_ANSWER_SCORES   = "answer_scores"      # torch.Tensor [B, V]       float32  ∈ [0,1]
KEY_LABELS          = "labels"             # torch.Tensor [B, L]       int64   (-100 = ignore)
KEY_QUESTION_IDS    = "question_ids"       # List[int]
KEY_QUESTION_TEXT   = "question_text"      # List[str]
KEY_IMAGE_IDS       = "image_ids"          # List[int]

# ── Model output keys ────────────────────────────────────────────────────────
KEY_LOSS            = "loss"               # torch.Tensor scalar
KEY_LOGITS          = "logits"             # torch.Tensor [B, V]       float32 (classify)
                                           #           or [B, L, vocab] float32 (generate)
KEY_VISUAL_FEATURES = "visual_features"    # torch.Tensor [B, D]       float32

# ── Prediction / evaluation keys ────────────────────────────────────────────
KEY_QUESTION_ID     = "question_id"        # int
KEY_ANSWER          = "answer"             # str  (predicted answer string)

# ── Checkpoint keys ─────────────────────────────────────────────────────────
KEY_EPOCH           = "epoch"              # int
KEY_GLOBAL_STEP     = "global_step"        # int
KEY_MODEL_STATE     = "model_state_dict"   # OrderedDict
KEY_OPTIM_STATE     = "optimizer_state_dict"
KEY_SCHED_STATE     = "scheduler_state_dict"
KEY_BEST_VAL_METRIC = "best_val_metric"    # float
KEY_CONFIG          = "config"             # dict (full YAML config)

# ── Evaluation result keys ───────────────────────────────────────────────────
KEY_OVERALL_ACC     = "overall"            # float  VQA accuracy 0-1
KEY_YESNO_ACC       = "yes/no"             # float
KEY_NUMBER_ACC      = "number"             # float
KEY_OTHER_ACC       = "other"              # float


# ===========================================================================
# 2. Batch contract  (output of DataLoader / VQADataset.__getitem__)
# ===========================================================================

class VQABatch(TypedDict, total=False):
    """Structured batch produced by :class:`data.vqa_dataset.VQADataset`.

    Mandatory fields (``total=False`` so optional fields don't break older
    code, but the fields below marked ★ are always present):

    ★ pixel_values   : torch.Tensor  shape [B, 3, IMAGE_SIZE, IMAGE_SIZE], float32
                       Pre-processed image tensor (normalized with ImageNet stats).
    ★ input_ids      : torch.Tensor  shape [B, MAX_QUESTION_LENGTH], int64
                       Tokenised question padded / truncated to MAX_QUESTION_LENGTH.
    ★ attention_mask : torch.Tensor  shape [B, MAX_QUESTION_LENGTH], int64, values {0, 1}
                       0 = padding, 1 = real token.
    ★ question_ids   : List[int]     length B
                       VQAv2 question_id for each sample.
      question_text  : List[str]     length B
                       Raw question strings (present in eval mode).
      image_ids      : List[int]     length B
                       COCO image_id for each sample.
      answer_scores  : torch.Tensor  shape [B, ANSWER_VOCAB_SIZE], float32 ∈ [0, 1]
                       Soft answer scores: min(human_count / 3, 1.0).
                       Only present when annotation file is provided (train / val).
      labels         : torch.Tensor  shape [B, MAX_QUESTION_LENGTH], int64
                       Token ids for teacher-forcing; positions to ignore = -100.
                       Only present in generative training.
    """

    pixel_values:   torch.Tensor
    input_ids:      torch.Tensor
    attention_mask: torch.Tensor
    question_ids:   List[int]
    question_text:  List[str]
    image_ids:      List[int]
    answer_scores:  torch.Tensor
    labels:         torch.Tensor


# ===========================================================================
# 3. Model output contracts
# ===========================================================================

class ModelOutput(TypedDict, total=False):
    """Dictionary returned by :meth:`models.blip2_vqa.BLIP2VQA.forward`.

    Mode "classify"
    ---------------
    ★ logits          : torch.Tensor  [B, ANSWER_VOCAB_SIZE], float32 (raw logits)
      loss            : torch.Tensor  scalar  (only when answer_scores provided)
      visual_features : torch.Tensor  [B, HIDDEN_SIZE], float32  (Q-Former mean pool)

    Mode "generate"
    ---------------
    ★ logits          : torch.Tensor  [B, SEQ_LEN, LM_VOCAB_SIZE], float32
      loss            : torch.Tensor  scalar  (only when labels provided)

    ★ = always present in that mode.
    """

    loss:            torch.Tensor
    logits:          torch.Tensor
    visual_features: torch.Tensor


class GenerateOutput(TypedDict):
    """Return type of :meth:`models.blip2_vqa.BLIP2VQA.generate_answers`.

    Produced per-batch; the caller should collect and flatten over batches.

    question_ids : List[int]  — VQAv2 question ids (same order as input)
    answers      : List[str]  — decoded answer strings (post-processed, lowercased)
    """

    question_ids: List[int]
    answers:      List[str]


# ===========================================================================
# 4. Prediction record  (one element in the predictions list)
# ===========================================================================

class PredictionRecord(TypedDict):
    """One prediction entry passed to :meth:`evaluation.vqa_eval.VQAEvaluator.compute_accuracy`.

    question_id : int — VQAv2 question_id.
    answer      : str — Predicted answer string.  Must be the *raw* model
                        output before normalisation; :class:`VQAEvaluator`
                        applies ``normalize_answer`` internally.
    """

    question_id: int
    answer:      str


# ===========================================================================
# 5. Evaluation output contract
# ===========================================================================

class EvalResult(TypedDict, total=False):
    """Dictionary returned by :meth:`evaluation.vqa_eval.VQAEvaluator.compute_accuracy`
    and by :meth:`training.trainer.VQATrainer.evaluate`.

    ★ overall  : float  — Mean VQA accuracy over all questions (0.0 – 1.0 scale,
                           i.e. NOT percentage).
      yes/no   : float  — Accuracy for yes/no question type.
      number   : float  — Accuracy for number question type.
      other    : float  — Accuracy for all other question types.
      loss     : float  — Validation loss (average over batches).
      metric   : float  — Primary scalar metric used for checkpoint selection;
                           equals ``overall`` when available, else ``-loss``.
    """

    overall: float
    yesno:   float
    number:  float
    other:   float
    loss:    float
    metric:  float


# ===========================================================================
# 6. Checkpoint contract
# ===========================================================================

class CheckpointDict(TypedDict, total=False):
    """Schema for ``.pth`` checkpoint files saved by
    :meth:`training.trainer.VQATrainer._save_checkpoint`.

    ★ epoch                  : int
    ★ global_step            : int
    ★ model_state_dict       : dict  — output of ``model.state_dict()``
    ★ optimizer_state_dict   : dict  — output of ``optimizer.state_dict()``
    ★ best_val_metric        : float
    ★ config                 : dict  — full YAML config dict
      scheduler_state_dict   : dict  — output of ``scheduler.state_dict()``
                                        (absent when no scheduler is used)
    """

    epoch:                 int
    global_step:           int
    model_state_dict:      dict
    optimizer_state_dict:  dict
    best_val_metric:       float
    config:                dict
    scheduler_state_dict:  dict


# ===========================================================================
# 7. Config sub-contracts  (mirrors default.yaml structure)
# ===========================================================================

class ModelConfig(TypedDict, total=False):
    """``cfg["model"]`` block in default.yaml."""

    name:               str    # "blip2_vqa" | "concat_fusion" | "bilinear_fusion"
                               # | "attention_fusion" | "mlb_fusion"
    blip2_model_name:   str    # HuggingFace model id, e.g. "Salesforce/blip2-opt-2.7b"
    num_query_tokens:   int    # Q-Former query token count (default 32)
    vision_width:       int    # ViT output dim (default 1408 for EVA-CLIP ViT-g)
    hidden_size:        int    # Q-Former hidden dim (default 768)
    num_layers:         int    # Q-Former transformer layers (default 12)
    num_heads:          int    # Attention heads (default 12)
    intermediate_size:  int    # FFN intermediate dim (default 3072)
    dropout:            float  # Dropout probability (default 0.1)
    max_answer_length:  int    # Max new tokens in generate mode (default 10)
    fusion_output_size: int    # Hidden size for fusion baselines (default 1024)
    num_answers:        int    # Answer vocab size (default 3129 for VQAv2)
    mode:               str    # "generate" | "classify"


class DataConfig(TypedDict, total=False):
    """``cfg["data"]`` block in default.yaml."""

    train_annotation:    str   # path to v2_OpenEnded_mscoco_train2014_questions.json
    val_annotation:      str   # path to v2_OpenEnded_mscoco_val2014_questions.json
    train_answers:       str   # path to v2_mscoco_train2014_annotations.json
    val_answers:         str   # path to v2_mscoco_val2014_annotations.json
    train_image_dir:     str   # path to train2014/ directory
    val_image_dir:       str   # path to val2014/ directory
    answer_list:         str   # path to answer_list.json
    max_question_length: int   # token truncation (default 50)
    image_size:          int   # pixels (default 224; images resized to square)
    num_workers:         int   # DataLoader worker count (default 4)


class TrainingConfig(TypedDict, total=False):
    """``cfg["training"]`` block in default.yaml."""

    output_dir:                   str    # checkpoint save directory
    log_dir:                      str    # TensorBoard / WandB log directory
    num_epochs:                   int
    batch_size:                   int    # training batch size
    eval_batch_size:              int    # validation batch size
    learning_rate:                float
    weight_decay:                 float
    warmup_steps:                 int
    gradient_clip:                float  # max-norm; 0 = disabled
    gradient_accumulation_steps:  int
    save_every:                   int    # save checkpoint every N epochs
    eval_every:                   int    # evaluate every N epochs
    seed:                         int
    mixed_precision:              bool
    resume_from:                  Optional[str]   # checkpoint path or null


class OptimizerConfig(TypedDict, total=False):
    """``cfg["optimizer"]`` block."""

    name:  str    # "adamw" | "adam" | "sgd"
    betas: List[float]
    eps:   float


class SchedulerConfig(TypedDict, total=False):
    """``cfg["scheduler"]`` block."""

    name:   str    # "cosine" | "linear" | "constant"
    min_lr: float


class LoggingConfig(TypedDict, total=False):
    """``cfg["logging"]`` block."""

    use_wandb:  bool
    project:    str
    run_name:   Optional[str]


# ===========================================================================
# 8. Fusion model contracts
# ===========================================================================

class FusionInput(TypedDict, total=False):
    """Inputs to any fusion baseline forward() method.

    ConcatFusion / BilinearFusion / MLBFusion
    -----------------------------------------
    ★ visual_features : torch.Tensor  [B, visual_dim]    – pooled visual rep.
    ★ text_features   : torch.Tensor  [B, text_dim]      – pooled text rep.

    AttentionFusion
    ---------------
    ★ visual_features : torch.Tensor  [B, N_tokens, visual_dim]
    ★ text_features   : torch.Tensor  [B, text_dim]
      visual_mask     : torch.Tensor  [B, N_tokens]  bool  True = keep
    """

    visual_features: torch.Tensor
    text_features:   torch.Tensor
    visual_mask:     torch.Tensor


class FusionOutput(TypedDict):
    """Return value of all fusion baseline forward() methods.

    logits : torch.Tensor  [B, num_answers], float32  — raw (pre-softmax) logits.
    """

    logits: torch.Tensor


# ===========================================================================
# 9. Runtime constants
# ===========================================================================

# VQAv2 answer vocabulary size (3,129 most frequent answers in training set)
ANSWER_VOCAB_SIZE: int = 3129

# Default image resolution fed to the vision encoder
IMAGE_SIZE: int = 224

# Q-Former defaults (must match QFormerConfig defaults)
NUM_QUERY_TOKENS: int = 32
QFORMER_HIDDEN_SIZE: int = 768
VISION_ENCODER_WIDTH: int = 1408   # EVA-CLIP ViT-g/14

# Question tokenisation
MAX_QUESTION_LENGTH: int = 50

# Answer generation
MAX_ANSWER_LENGTH: int = 10        # maximum new tokens in generate mode

# VQA soft-score normalisation denominator
VQA_SCORE_DENOMINATOR: int = 3     # min(count / 3, 1.0)

# Padding / ignore index for labels tensor
LABEL_IGNORE_INDEX: int = -100

# Model name keys (must match ModelConfig.name values and fusion registry keys)
MODEL_BLIP2_VQA:      str = "blip2_vqa"
MODEL_CONCAT_FUSION:  str = "concat_fusion"
MODEL_BILINEAR_FUSION: str = "bilinear_fusion"
MODEL_ATTENTION_FUSION: str = "attention_fusion"
MODEL_MLB_FUSION:     str = "mlb_fusion"

VALID_MODEL_NAMES = frozenset({
    MODEL_BLIP2_VQA,
    MODEL_CONCAT_FUSION,
    MODEL_BILINEAR_FUSION,
    MODEL_ATTENTION_FUSION,
    MODEL_MLB_FUSION,
})

# Operating modes for BLIP2VQA
MODE_GENERATE: str = "generate"
MODE_CLASSIFY: str = "classify"
VALID_MODES = frozenset({MODE_GENERATE, MODE_CLASSIFY})

# Loss types for VQALoss
LOSS_BCE: str = "bce"
LOSS_CE:  str = "ce"
LOSS_KL:  str = "kl"
VALID_LOSS_TYPES = frozenset({LOSS_BCE, LOSS_CE, LOSS_KL})

# Optimizer names
OPTIM_ADAMW: str = "adamw"
OPTIM_ADAM:  str = "adam"
OPTIM_SGD:   str = "sgd"
VALID_OPTIMIZER_NAMES = frozenset({OPTIM_ADAMW, OPTIM_ADAM, OPTIM_SGD})

# Scheduler names
SCHED_COSINE:   str = "cosine"
SCHED_LINEAR:   str = "linear"
SCHED_CONSTANT: str = "constant"
VALID_SCHEDULER_NAMES = frozenset({SCHED_COSINE, SCHED_LINEAR, SCHED_CONSTANT})
