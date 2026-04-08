"""Huấn luyện một EXP fusion model hoặc BLIP2VQA trên VQAv2.

Sử dụng
-------
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --resume checkpoints/best_model.pth

Ghi chú pipeline
----------------
- Nếu ``model.name == "blip2_vqa"``: dùng BLIP2VQA legacy pipeline (pixel_values).
- Ngược lại (EXP-01 → EXP-07): dùng EXP pipeline:
    HDF5 image_features + FrozenTextEncoder (BERT) → fusion model → logits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf

from data.vqa_dataset import build_dataloader
from models import build_model, FrozenTextEncoder
from models.blip2_vqa import BLIP2VQA
from training.losses import VQALoss
from training.trainer import VQATrainer
from utils.helpers import build_optimizer, build_scheduler, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện EXP fusion model hoặc BLIP2VQA trên VQAv2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Đường dẫn file YAML config.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint để tiếp tục huấn luyện.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override thư mục lưu checkpoint.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu). Mặc định: tự phát hiện.")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override số epoch huấn luyện.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # load_config trả dict; convert sang OmegaConf để dùng dot-notation
    cfg_dict = load_config(args.config)

    # Áp dụng CLI overrides vào dict trước khi tạo OmegaConf
    if args.output_dir:
        cfg_dict.setdefault("training", {})["output_dir"] = args.output_dir
    if args.num_epochs:
        cfg_dict.setdefault("training", {})["num_epochs"] = args.num_epochs
    if args.batch_size:
        cfg_dict.setdefault("data", {})["batch_size"] = args.batch_size
    if args.seed:
        cfg_dict.setdefault("training", {})["seed"] = args.seed

    config = OmegaConf.create(cfg_dict)

    seed = int(getattr(config.training, "seed", 42))
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Sử dụng device: %s", device)

    # ------------------------------------------------------------------
    # Data loaders — dùng HDF5 cache cho EXP models
    # ------------------------------------------------------------------
    model_name = config.model.name
    use_cache = (model_name != "blip2_vqa")
    logger.info("Xây dựng DataLoader (use_cache=%s) …", use_cache)

    train_loader = build_dataloader("train", config, use_cache=use_cache)
    val_loader   = build_dataloader("val",   config, use_cache=use_cache)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ------------------------------------------------------------------
    # Model + text encoder
    # ------------------------------------------------------------------
    logger.info("Xây dựng model: %s", model_name)
    text_encoder = None

    if model_name == "blip2_vqa":
        model = BLIP2VQA(
            model_name=getattr(config.model, "blip2_model_name", "Salesforce/blip2-opt-2.7b"),
            num_answers=int(getattr(config.model, "num_answers", 3129)),
            mode="classify",
            freeze_vision_encoder=True,
            freeze_qformer=False,
            max_answer_length=int(getattr(config.model, "max_answer_length", 10)),
        )
    else:
        # EXP fusion model: cần FrozenTextEncoder chạy trong trainer
        model = build_model(config)
        text_encoder = FrozenTextEncoder()
        logger.info("FrozenTextEncoder (BERT-base-uncased) đã khởi tạo và đóng băng.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Số tham số huấn luyện được: %s", f"{trainable:,}")

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, cfg_dict)
    num_training_steps = len(train_loader) * int(config.training.num_epochs)
    scheduler = build_scheduler(optimizer, cfg_dict, num_training_steps)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=VQALoss(loss_type="bce"),
        device=device,
        output_dir=str(config.training.output_dir),
        gradient_accumulation_steps=int(getattr(config.training, "gradient_accumulation_steps", 1)),
        gradient_clip=float(getattr(config.training, "gradient_clip", 1.0)),
        mixed_precision=bool(getattr(config.training, "mixed_precision", True)),
        text_encoder=text_encoder,
    )

    if args.resume:
        logger.info("Tiếp tục từ checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    logger.info("Bắt đầu huấn luyện …")
    trainer.train(num_epochs=int(config.training.num_epochs))
    logger.info("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
