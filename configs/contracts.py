"""
contracts.py — Quy uoc I/O chung cho toan bo repo BLIP-2 VQA.

MUC DICH
File nay la nguon chan ly duy nhat (single source of truth) cho moi:
    - Ten bien (key) trong batch dict
    - Kieu du lieu va shape cua Tensor
    - Hang so duoc dung chung giua cac module

QUY TAC BAT BUOC
Neu ban them key moi vao batch, model output, hoac prediction:
    1. Dinh nghia key/hang so tai day TRUOC
    2. Sau do moi implement o cac file khac

NOI DUNG
    1. Hang so ten bien (batch key, checkpoint key, eval key...)
    2. VQABatch        — cau truc du lieu tra ra tu DataLoader
    3. ModelOutput     — cau truc du lieu tra ra tu model.forward()
    4. GenerateOutput  — cau truc du lieu tra ra tu model.generate_answers()
    5. PredictionRecord — mot dong du doan (de truyen vao VQAEvaluator)
    6. EvalResult      — ket qua danh gia tra ra tu trainer.evaluate()
    7. CheckpointDict  — cau truc file checkpoint .pth
    8. Config classes  — tuong ung tung block trong file YAML
    9. FusionInput / FusionOutput — hop dong I/O cho cac fusion model EXP-01..07
    10. Hang so runtime — kich thuoc vocab, dim tensor, ten model...
"""

from __future__ import annotations

from typing import List, Optional

# Tuong thich Python 3.8+; neu Python 3.7 thi fallback sang typing_extensions
try:
    from typing import TypedDict, Literal
except ImportError:
    from typing_extensions import TypedDict, Literal  # type: ignore[assignment]

import torch


# =============================================================================
# 1. Hang so ten bien
#
# Dung hang so thay vi viet chuoi truc tiep (vd: batch["image_features"])
# de neu go sai ten thi Python bao NameError ngay — thay vi bug am tham.
# =============================================================================

# Ten bien trong batch (output cua DataLoader)
KEY_PIXEL_VALUES    = "pixel_values"
# Tensor anh goc: [B, 3, H, W] float32 — dung khi khong co cache HDF5

KEY_IMAGE_FEATURES  = "image_features"
# Tensor dac trung anh da trich xuat: [B, 257, 1024] float32
# (1 CLS token + 256 patch tokens tu CLIP ViT-L/14)
# Chi co khi use_cache=True

KEY_INPUT_IDS       = "input_ids"
# Tensor ID token cau hoi (sau khi tokenize bang BERT): [B, 50] int64

KEY_ATTENTION_MASK  = "attention_mask"
# Mask cho padding: [B, 50] int64, gia tri 0 (padding) hoac 1 (token that)

KEY_ANSWER_SCORES   = "answer_scores"
# Soft label cua cau tra loi: [B, 3129] float32, gia tri trong [0, 1]
# Cong thuc: score = min(so_nguoi_chon / 3, 1.0)

KEY_ANSWER_LABEL    = "answer_label"
# Hard label: [B] int64 — index cua cau tra loi pho bien nhat
# -1 neu cau tra loi khong nam trong vocab (OOV)

KEY_LABELS          = "labels"
# Tensor label cho huan luyen generative: [B, L] int64
# Vi tri bi ignore = -100 (tieu chuan PyTorch)

KEY_QUESTION_IDS    = "question_ids"
# List[int] — ID cau hoi trong VQAv2, dung khi danh gia chinh thuc

KEY_QUESTION_TEXT   = "question_text"
# List[str] — Chuoi cau hoi goc (chua qua tokenize)

KEY_IMAGE_IDS       = "image_ids"
# List[int] — ID anh COCO tuong ung voi tung mau trong batch

KEY_ANSWER_TEXT     = "answer"
# List[str] — Cau tra loi pho bien nhat (mot chuoi duy nhat)

KEY_ANSWERS         = "answers"
# List[List[str]] — Toan bo 10 cau tra loi cua 10 nguoi chu thich

KEY_ANSWER_TYPE     = "answer_type"
# List[str] — Loai cau hoi: "yes/no" | "number" | "other"

# Ten bien trong output cua model
KEY_LOSS            = "loss"
# Tensor scalar — gia tri loss cua batch hien tai

KEY_LOGITS          = "logits"
# Tensor logit chua qua softmax:
#   classify mode: [B, 3129] float32
#   generate mode: [B, seq_len, lm_vocab] float32

KEY_VISUAL_FEATURES = "visual_features"
# Tensor dac trung anh sau khi qua fusion module: [B, D] float32

# Ten bien cho ket qua du doan va danh gia
KEY_QUESTION_ID     = "question_id"
# int — ID cua mot cau hoi cu the (dung trong PredictionRecord)

KEY_ANSWER          = "answer"
# str — Cau tra loi duoc du doan (mot chuoi)

# Ten bien trong file checkpoint .pth
KEY_EPOCH           = "epoch"
# int — Epoch tai thoi diem luu checkpoint

KEY_GLOBAL_STEP     = "global_step"
# int — Tong so buoc optimizer da chay

KEY_MODEL_STATE     = "model_state_dict"
# OrderedDict — ket qua cua model.state_dict()

KEY_OPTIM_STATE     = "optimizer_state_dict"
# dict — ket qua cua optimizer.state_dict()

KEY_SCHED_STATE     = "scheduler_state_dict"
# dict — ket qua cua scheduler.state_dict()

KEY_BEST_VAL_METRIC = "best_val_metric"
# float — gia tri metric tot nhat tu truoc den gio (dung de luu best_model.pth)

KEY_EARLY_STOP_BAD_EPOCHS = "early_stop_bad_epochs"
# int — so epoch validation lien tiep khong du cai thien de reset early stopping

KEY_CONFIG          = "config"
# dict — toan bo config YAML tai thoi diem train

# Ten bien trong EvalResult
KEY_OVERALL_ACC     = "overall"
# float — accuracy toan bo (khong phan loai cau hoi)

KEY_YESNO_ACC       = "yes/no"
# float — accuracy rieng cho cau hoi Yes/No (~38% tap val)

KEY_NUMBER_ACC      = "number"
# float — accuracy rieng cho cau hoi dem so (~12% tap val)

KEY_OTHER_ACC       = "other"
# float — accuracy rieng cho cau hoi mo (~50% tap val)


# =============================================================================
# 2. VQABatch — Cau truc du lieu tra ra tu DataLoader
#
# TypedDict giup IDE va type checker biet chinh xac cac key nao co trong batch.
# total=False: tat ca cac truong deu la "co the vang mat" —
#   vi batch train co answer_scores nhung batch test thi khong.
# =============================================================================

class VQABatch(TypedDict, total=False):
    """
    Cau truc batch du lieu tra ra tu VQADataset.__getitem__() va DataLoader.

    Truong luon co mat (trong ca train va val):
        pixel_values   HOAC image_features  (mot trong hai, tuy use_cache)
        input_ids
        attention_mask
        question_ids

    Truong chi co khi co annotation (train / val, khong co o test):
        answer_scores, answer_label, answer, answers, answer_type

    Truong chi co khi huan luyen theo kieu generative:
        labels
    """

    # Anh — chi co mot trong hai tuy che do
    pixel_values:   torch.Tensor    # [B, 3, IMAGE_SIZE, IMAGE_SIZE] — anh goc
    image_features: torch.Tensor    # [B, 257, 1024] — dac trung CLIP da trich xuat

    # Cau hoi
    input_ids:      torch.Tensor    # [B, 50] — token ID sau khi tokenize
    attention_mask: torch.Tensor    # [B, 50] — mask padding
    question_ids:   List[int]       # ID cau hoi VQAv2
    question_text:  List[str]       # Chuoi cau hoi goc
    image_ids:      List[int]       # ID anh COCO

    # Cau tra loi (vang mat o tap test)
    answer_scores:  torch.Tensor    # [B, 3129] — soft label tu 10 nguoi chu thich
    answer_label:   torch.Tensor    # [B] — hard label (index cau tra loi pho bien nhat)
    answer:         List            # List[str] — cau tra loi pho bien nhat
    answers:        List            # List[List[str]] — toan bo 10 cau tra loi
    answer_type:    List            # List[str] — "yes/no" | "number" | "other"

    # Chi dung cho huan luyen generative
    labels:         torch.Tensor    # [B, L] — teacher-forcing labels, -100 = ignore


# =============================================================================
# 3. Model output contracts
#
# ModelOutput: tra ra tu BLIP2VQA.forward()
# GenerateOutput: tra ra tu BLIP2VQA.generate_answers()
# =============================================================================

class ModelOutput(TypedDict, total=False):
    """
    Dict tra ra tu BLIP2VQA.forward().

    Che do classify (dung cho EXP-01 den EXP-07):
        logits          : [B, 3129] — logit chua qua softmax
        loss            : scalar    — chi co khi truyen vao answer_scores
        visual_features : [B, 768]  — dac trung anh sau Q-Former (mean pool)

    Che do generate:
        logits : [B, seq_len, lm_vocab] — logit cua language model
        loss   : scalar                 — chi co khi truyen vao labels
    """

    loss:            torch.Tensor   # gia tri loss (scalar)
    logits:          torch.Tensor   # logit chinh cua model
    visual_features: torch.Tensor   # dac trung trung gian (tuy chon)


class GenerateOutput(TypedDict):
    """
    Dict tra ra tu BLIP2VQA.generate_answers() — mot batch.

    question_ids : List[int] — ID cua tung cau hoi trong batch
    answers      : List[str] — Cau tra loi da giai ma (lowercase, normalized)
    """

    question_ids: List[int]
    answers:      List[str]


# =============================================================================
# 4. PredictionRecord — Mot dong du doan de truyen vao VQAEvaluator
# =============================================================================

class PredictionRecord(TypedDict):
    """
    Mot ban ghi du doan — element trong list predictions
    duoc truyen vao VQAEvaluator.compute_accuracy().

    question_id : ID cau hoi VQAv2
    answer      : Chuoi cau tra loi du doan (chua normalize —
                  VQAEvaluator se tu normalize khi tinh accuracy)
    """

    question_id: int
    answer:      str


# =============================================================================
# 5. EvalResult — Ket qua danh gia
#
# Tra ra tu VQAEvaluator.compute_accuracy() va VQATrainer.evaluate().
# total=False vi khong phai luc nao cung co du cac truong
# (vd: khi khong co annotation thi khong co overall/yes_no/number/other).
# =============================================================================

class EvalResult(TypedDict, total=False):
    """
    Dict ket qua danh gia.

    overall : accuracy tong — thang do chinh, tu 0.0 den 1.0 (khong phai %)
    yes/no  : accuracy cho cau hoi Yes/No
    number  : accuracy cho cau hoi dem so
    other   : accuracy cho cau hoi mo
    loss    : val loss trung binh tren toan bo tap val
    metric  : gia tri scalar chinh de chon best checkpoint
              (= overall neu co, nguoc lai = -loss)
    """

    overall: float
    yesno:   float
    number:  float
    other:   float
    loss:    float
    metric:  float


# =============================================================================
# 6. CheckpointDict — Cau truc file checkpoint .pth
#
# Moi file checkpoint duoc luu bang torch.save() phai co dung cac key nay.
# =============================================================================

class CheckpointDict(TypedDict, total=False):
    """
    Cau truc dict ben trong file checkpoint .pth.

    Truong bat buoc (luon co mat):
        epoch, global_step, model_state_dict,
        optimizer_state_dict, best_val_metric, config

    Truong tuy chon:
        scheduler_state_dict — vang mat neu khong dung scheduler
    """

    epoch:                 int    # epoch tai thoi diem luu
    global_step:           int    # tong so buoc optimizer
    model_state_dict:      dict   # trong so model (tu model.state_dict())
    optimizer_state_dict:  dict   # trang thai optimizer
    best_val_metric:       float  # metric tot nhat de so sanh
    early_stop_bad_epochs: int    # so epoch validation khong du cai thien
    config:                dict   # toan bo config YAML
    scheduler_state_dict:  dict   # trang thai scheduler (neu co)


# =============================================================================
# 7. Config sub-contracts — Tuong ung tung block trong file YAML
#
# Cac class nay mo ta dung cau truc cua dict cfg["model"], cfg["data"]...
# Giup IDE tu dong goi y khi truy cap config.
# =============================================================================

class ModelConfig(TypedDict, total=False):
    """
    Block cfg["model"] trong file YAML (default.yaml / expXX.yaml).

    Truong name chon model nao se duoc dung:
        "blip2_vqa"          — BLIP-2 pretrained (legacy)
        "mean_linear"        — EXP-01: Mean Pool + Linear
        "concat_fusion"      — EXP-02: Concat + MLP
        "mlb_fusion"         — EXP-03: Hadamard Bilinear
        "mfb_fusion"         — EXP-04: Factorized Bilinear
        "cross_attn_fusion"  — EXP-05: Cross-Attention Bridge
        "qformer_scratch"    — EXP-06: Q-Former tu dau
        "perceiver_resampler"— EXP-07: Perceiver Resampler
    """

    name:               str    # ten model, xem danh sach o tren
    blip2_model_name:   str    # HuggingFace model id cho BLIP-2 legacy
    num_query_tokens:   int    # so query token cua Q-Former (mac dinh 32)
    vision_width:       int    # chieu output cua ViT (CLIP ViT-L/14 = 1024)
    hidden_size:        int    # chieu an cua Q-Former / BERT (mac dinh 768)
    num_layers:         int    # so lop transformer (mac dinh 12)
    num_heads:          int    # so dau attention (mac dinh 12)
    intermediate_size:  int    # chieu FFN = 4 * hidden_size (mac dinh 3072)
    dropout:            float  # xac suat dropout (mac dinh 0.1)
    max_answer_length:  int    # so token toi da khi sinh cau tra loi (mac dinh 10)
    fusion_output_size: int    # chieu an cua cac fusion baseline (mac dinh 1024)
    num_answers:        int    # kich thuoc vocab cau tra loi VQAv2 (= 3129)
    mode:               str    # "generate" hoac "classify"


class DataConfig(TypedDict, total=False):
    """
    Block cfg["data"] trong file YAML.

    Duong dan du lieu duoc ghep tu:
        {data_root}/{vqav2_dir}/  — chua cac file JSON cua VQAv2
        {data_root}/{coco_dir}/   — chua anh COCO (train2014/, val2014/)
        {data_root}/{cache_dir}/  — chua file HDF5 dac trung da trich xuat

    Cac truong legacy (giu lai de tuong thich nguoc):
        train_annotation, val_annotation... — duong dan day du den tung file
    """

    # Duong dan chinh (khuyen nghi dung)
    data_root:           str   # thu muc goc chua toan bo data
    vqav2_dir:           str   # ten thu muc chua JSON VQAv2 (tuong doi voi data_root)
    coco_dir:            str   # ten thu muc chua anh COCO (tuong doi voi data_root)
    cache_dir:           str   # ten thu muc chua HDF5 cache (tuong doi voi data_root)
    train_size:          int   # so mau train sau khi lay subset (stratified)
    val_size:            int   # so mau val sau khi lay subset
    seed:                int   # random seed cho stratified sampling
    batch_size:          int   # batch size cho DataLoader

    # Dung chung
    answer_list:         str   # duong dan den file vocab cau tra loi (ans2idx.json)
    max_question_length: int   # do dai toi da cua cau hoi (so token, mac dinh 50)
    image_size:          int   # kich thuoc anh resize truoc khi vao CLIP (mac dinh 224)
    num_workers:         int   # so worker cua DataLoader (mac dinh 2 tren Colab)

    # Truong legacy — duong dan day du (backward compat)
    train_annotation:    str   # duong dan den file JSON cau hoi train
    val_annotation:      str   # duong dan den file JSON cau hoi val
    train_answers:       str   # duong dan den file JSON annotation train
    val_answers:         str   # duong dan den file JSON annotation val
    train_image_dir:     str   # duong dan den thu muc anh train2014/
    val_image_dir:       str   # duong dan den thu muc anh val2014/


class TrainingConfig(TypedDict, total=False):
    """Block cfg["training"] trong file YAML."""

    output_dir:                   str    # thu muc luu checkpoint
    log_dir:                      str    # thu muc luu log W&B / TensorBoard
    num_epochs:                   int    # tong so epoch huan luyen
    batch_size:                   int    # batch size huan luyen
    eval_batch_size:              int    # batch size khi danh gia
    learning_rate:                float  # learning rate ban dau
    weight_decay:                 float  # he so weight decay cho AdamW
    warmup_steps:                 int    # so buoc warmup LR
    gradient_clip:                float  # nguong clip gradient (0 = tat)
    gradient_accumulation_steps:  int    # so buoc tich luy gradient truoc moi optimizer step
    save_every:                   int    # luu checkpoint moi N epoch
    eval_every:                   int    # danh gia moi N epoch
    seed:                         int    # random seed
    mixed_precision:              bool   # dung fp16 AMP hay khong
    early_stopping_patience:       int    # so epoch khong cai thien truoc khi dung
    early_stopping_min_delta:      float  # muc tang metric toi thieu de reset patience
    resume_from:                  Optional[str]   # duong dan checkpoint de resume, hoac null


class OptimizerConfig(TypedDict, total=False):
    """Block cfg["optimizer"] trong file YAML."""

    name:  str          # "adamw" | "adam" | "sgd"
    betas: List[float]  # he so beta cho Adam (mac dinh [0.9, 0.999])
    eps:   float        # epsilon tranh chia zero (mac dinh 1e-8)


class SchedulerConfig(TypedDict, total=False):
    """Block cfg["scheduler"] trong file YAML."""

    name:   str    # "cosine" | "linear" | "constant"
    min_lr: float  # learning rate toi thieu o cuoi cosine decay (mac dinh 1e-6)


class LoggingConfig(TypedDict, total=False):
    """Block cfg["logging"] trong file YAML."""

    use_wandb:  bool         # bat/tat W&B logging
    project:    str          # ten project tren W&B
    run_name:   Optional[str]  # ten run tren W&B (vd: "exp01_lan1_khoa")


# =============================================================================
# 8. FusionInput / FusionOutput — Hop dong I/O cho cac fusion model
#
# Tat ca 7 EXP model deu nhan FusionInput va tra ra FusionOutput.
# =============================================================================

class FusionInput(TypedDict, total=False):
    """
    Input cho phuong thuc forward() cua cac fusion model (EXP-01 den EXP-07).

    EXP-01 den EXP-04 (dung pooled features):
        visual_features : [B, 1024]  — dac trung anh da mean-pool
        text_features   : [B, 768]   — CLS token cua BERT

    EXP-05 den EXP-07 (dung patch-level features):
        visual_features : [B, 257, 1024]  — toan bo patch token tu CLIP
        text_features   : [B, 768]
        visual_mask     : [B, 257] bool   — True = giu lai token nay (tuy chon)
    """

    visual_features: torch.Tensor   # dac trung anh (pooled hoac patch-level)
    text_features:   torch.Tensor   # dac trung cau hoi tu BERT
    visual_mask:     torch.Tensor   # mask cho patch tokens (tuy chon, EXP-05..07)


class FusionOutput(TypedDict):
    """
    Output cua forward() cho tat ca fusion model.

    logits : [B, 3129] float32 — raw logit chua qua softmax
             Trainer se tinh loss tu day bang VQALoss.
    """

    logits: torch.Tensor


# =============================================================================
# 9. Hang so runtime
#
# Tat ca hang so duoc dung trong nhieu file deu phai khai bao tai day.
# KHONG hardcode con so nay o bat ky file nao khac.
# =============================================================================

# Kich thuoc vocab cau tra loi — 3129 cau tra loi pho bien nhat trong VQAv2
ANSWER_VOCAB_SIZE: int = 3129

# Kich thuoc anh dau vao CLIP ViT-L/14
IMAGE_SIZE: int = 224

# Cau hinh mac dinh cho Q-Former (EXP-06)
NUM_QUERY_TOKENS: int = 32        # so learnable query token
QFORMER_HIDDEN_SIZE: int = 768    # chieu an — khop voi BERT-base
VISION_ENCODER_WIDTH: int = 1024  # chieu output cua CLIP ViT-L/14

# Thong so CLIP ViT-L/14 — phai khop voi cache HDF5 da trich xuat
CLIP_FEATURE_DIM: int = 1024      # chieu output moi token
CLIP_PATCH_TOKENS: int = 257      # 1 CLS + 256 patch token (anh 224x224)

# Cau hoi
MAX_QUESTION_LENGTH: int = 50     # do dai toi da sau khi tokenize (so token)

# Sinh cau tra loi (generative mode)
MAX_ANSWER_LENGTH: int = 10       # so token toi da khi generate

# Cong thuc tinh soft score: score = min(so_nguoi_chon / 3, 1.0)
VQA_SCORE_DENOMINATOR: int = 3

# Index bi bo qua khi tinh CE loss (tieu chuan PyTorch)
LABEL_IGNORE_INDEX: int = -100

# Ten model (phai khop voi gia tri cua ModelConfig.name va registry trong models/__init__.py)
MODEL_BLIP2_VQA:           str = "blip2_vqa"
MODEL_MEAN_LINEAR:         str = "mean_linear"           # EXP-01
MODEL_CONCAT_FUSION:       str = "concat_fusion"          # EXP-02
MODEL_MLB_FUSION:          str = "mlb_fusion"             # EXP-03
MODEL_MFB_FUSION:          str = "mfb_fusion"             # EXP-04
MODEL_CROSS_ATTN_FUSION:   str = "cross_attn_fusion"      # EXP-05
MODEL_QFORMER_SCRATCH:     str = "qformer_scratch"        # EXP-06
MODEL_PERCEIVER_RESAMPLER: str = "perceiver_resampler"    # EXP-07
MODEL_BILINEAR_FUSION:     str = "bilinear_fusion"        # giu lai de tuong thich nguoc
MODEL_ATTENTION_FUSION:    str = "attention_fusion"       # giu lai de tuong thich nguoc

# Tap hop tat ca ten model hop le — dung de validate trong build_model()
VALID_MODEL_NAMES = frozenset({
    MODEL_BLIP2_VQA,
    MODEL_MEAN_LINEAR,
    MODEL_CONCAT_FUSION,
    MODEL_MLB_FUSION,
    MODEL_MFB_FUSION,
    MODEL_CROSS_ATTN_FUSION,
    MODEL_QFORMER_SCRATCH,
    MODEL_PERCEIVER_RESAMPLER,
    MODEL_BILINEAR_FUSION,
    MODEL_ATTENTION_FUSION,
})

# Che do hoat dong cua BLIP2VQA
MODE_GENERATE: str = "generate"   # sinh van ban tu do (generative)
MODE_CLASSIFY: str = "classify"   # phan loai trong 3129 cau tra loi (classification)
VALID_MODES = frozenset({MODE_GENERATE, MODE_CLASSIFY})

# Loai ham loss cho VQALoss
LOSS_BCE:       str = "bce"        # Binary Cross-Entropy voi soft label (mac dinh)
LOSS_CE:        str = "ce"         # Cross-Entropy voi hard label
LOSS_KL:        str = "kl"         # KL Divergence
LOSS_FOCAL_BCE: str = "focal_bce"  # Focal BCE — tang trong so mau kho hoc
VALID_LOSS_TYPES = frozenset({LOSS_BCE, LOSS_CE, LOSS_KL, LOSS_FOCAL_BCE})

# Ten optimizer
OPTIM_ADAMW: str = "adamw"   # AdamW (mac dinh, phu hop voi transformer)
OPTIM_ADAM:  str = "adam"    # Adam thong thuong
OPTIM_SGD:   str = "sgd"     # SGD voi momentum
VALID_OPTIMIZER_NAMES = frozenset({OPTIM_ADAMW, OPTIM_ADAM, OPTIM_SGD})

# Ten scheduler
SCHED_COSINE:   str = "cosine"    # Cosine annealing (mac dinh)
SCHED_LINEAR:   str = "linear"    # Giam tuyen tinh
SCHED_CONSTANT: str = "constant"  # Giu nguyen LR
VALID_SCHEDULER_NAMES = frozenset({SCHED_COSINE, SCHED_LINEAR, SCHED_CONSTANT})
