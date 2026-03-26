"""Train a BLIP-2 VQA model.

Usage
-----
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_005.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.vqa_dataset import build_vqa_dataloader
from models.blip2_vqa import BLIP2VQA
from models.fusion_baselines import build_fusion_model
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
        description="Train a BLIP-2 VQA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu). Defaults to auto-detect.")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override training batch size.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.num_epochs:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.seed:
        cfg["training"]["seed"] = args.seed

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    data_cfg = cfg["data"]
    logger.info("Building data loaders …")

    train_loader = build_vqa_dataloader(
        question_file=data_cfg["train_annotation"],
        annotation_file=data_cfg["train_answers"],
        image_dir=data_cfg["train_image_dir"],
        answer_list_file=data_cfg.get("answer_list"),
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        is_train=True,
    )
    val_loader = build_vqa_dataloader(
        question_file=data_cfg["val_annotation"],
        annotation_file=data_cfg["val_answers"],
        image_dir=data_cfg["val_image_dir"],
        answer_list_file=data_cfg.get("answer_list"),
        batch_size=cfg["training"].get("eval_batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
        is_train=False,
    )
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    logger.info("Building model: %s", model_name)

    if model_name == "blip2_vqa":
        model = BLIP2VQA(
            model_name=model_cfg.get("blip2_model_name", "Salesforce/blip2-opt-2.7b"),
            num_answers=model_cfg.get("num_answers", 3129),
            mode="classify",
            freeze_vision_encoder=True,
            freeze_qformer=False,
            max_answer_length=model_cfg.get("max_answer_length", 10),
        )
    else:
        model = build_fusion_model(
            model_name,
            visual_dim=model_cfg.get("hidden_size", 768),
            text_dim=model_cfg.get("hidden_size", 768),
            fusion_dim=model_cfg.get("fusion_output_size", 1024),
            num_answers=model_cfg.get("num_answers", 3129),
        )

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, cfg)
    num_training_steps = len(train_loader) * cfg["training"]["num_epochs"]
    scheduler = build_scheduler(optimizer, cfg, num_training_steps)

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
        output_dir=cfg["training"]["output_dir"],
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        gradient_clip=cfg["training"].get("gradient_clip", 1.0),
        mixed_precision=cfg["training"].get("mixed_precision", True),
    )

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training …")
    trainer.train(num_epochs=cfg["training"]["num_epochs"])
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
