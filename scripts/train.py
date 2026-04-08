"""Huấn luyện một EXP fusion model hoặc BLIP2VQA trên VQAv2.

Sử dụng
-------
python scripts/train.py --config configs/exp06.yaml
python scripts/train.py --config configs/exp06.yaml --resume auto        # tự tìm checkpoint mới nhất
python scripts/train.py --config configs/exp06.yaml --resume path/to/ck.pth

Ghi chú pipeline
----------------
- Nếu ``model.name == "blip2_vqa"``: dùng BLIP2VQA legacy pipeline (pixel_values).
- Ngược lại (EXP-01 → EXP-07): dùng EXP pipeline:
    HDF5 image_features + FrozenTextEncoder (BERT) → fusion model → logits.

Auto-resume
-----------
  Truyền ``--resume auto`` hoặc đặt ``training.resume_from: auto`` trong YAML.
  Script sẽ tự tìm checkpoint_epoch_*.pth mới nhất trong output_dir.

W&B
---
  Đặt ``logging.use_wandb: true`` trong YAML và chạy ``wandb login`` trước.
  Hoặc đặt biến môi trường WANDB_API_KEY trên Colab.
"""

from __future__ import annotations

import argparse
import logging
import os
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
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Tìm checkpoint_epoch_*.pth mới nhất trong output_dir."""
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    return str(checkpoints[-1]) if checkpoints else None


def _resolve_resume(args_resume: str | None, config) -> str | None:
    """Xử lý logic auto-resume: trả về đường dẫn checkpoint hoặc None."""
    resume_val = args_resume or str(getattr(config.training, "resume_from", "") or "")
    if not resume_val or resume_val.lower() == "null":
        return None
    if resume_val.lower() == "auto":
        found = _find_latest_checkpoint(str(config.training.output_dir))
        if found:
            logger.info("Auto-resume: tìm thấy checkpoint %s", found)
        else:
            logger.info("Auto-resume: chưa có checkpoint, bắt đầu từ đầu.")
        return found
    return resume_val


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
                        help="Checkpoint để resume: đường dẫn cụ thể hoặc 'auto'.")
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
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override tên W&B run.")
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
    if args.run_name:
        cfg_dict.setdefault("logging", {})["run_name"] = args.run_name

    config = OmegaConf.create(cfg_dict)

    seed = int(getattr(config.training, "seed", 42))
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Sử dụng device: %s", device)

    # ------------------------------------------------------------------
    # Auto-resume: xác định checkpoint trước khi build model
    # ------------------------------------------------------------------
    resume_path = _resolve_resume(args.resume, config)

    # ------------------------------------------------------------------
    # W&B init (tùy chọn)
    # ------------------------------------------------------------------
    wandb_run = None
    log_cfg = cfg_dict.get("logging", {})
    if log_cfg.get("use_wandb", False):
        try:
            import wandb  # noqa: F401
            run_name = log_cfg.get("run_name") or f"{config.model.name}"
            wandb_run = wandb.init(
                project=log_cfg.get("project", "blip2-vqa-experiment"),
                name=run_name,
                config=OmegaConf.to_container(config, resolve=True),
                resume="allow",                   # cho phép resume run cũ (nếu cùng id)
                id=log_cfg.get("wandb_run_id"),   # set trong YAML khi muốn resume run cụ thể
            )
            logger.info("W&B run: %s  (url: %s)", wandb_run.name, wandb_run.url)
        except ImportError:
            logger.warning("wandb chưa cài — bỏ qua W&B logging. `pip install wandb`")

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
    num_training_steps = (
        len(train_loader)
        // int(getattr(config.training, "gradient_accumulation_steps", 1))
        * int(config.training.num_epochs)
    )
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
        wandb_run=wandb_run,
    )

    # ------------------------------------------------------------------
    # Load checkpoint & tính start_epoch
    # ------------------------------------------------------------------
    start_epoch = 0
    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path)
        logger.info("Tiếp tục từ epoch %d.", start_epoch + 1)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Bắt đầu huấn luyện (epoch %d → %d) …",
                start_epoch + 1, int(config.training.num_epochs))
    try:
        trainer.train(
            num_epochs=int(config.training.num_epochs),
            start_epoch=start_epoch,
        )
    finally:
        # Đảm bảo W&B finish ngay cả khi crash / Colab ngắt
        if wandb_run is not None:
            wandb_run.finish()
            logger.info("W&B run đã kết thúc.")

    logger.info("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
