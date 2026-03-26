"""Evaluate a trained BLIP-2 VQA model on a validation or test split.

Usage
-----
python scripts/evaluate.py \\
    --config configs/default.yaml \\
    --checkpoint checkpoints/best_model.pth \\
    --split val \\
    --output results/val_predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from data.vqa_dataset import VQADataset, build_vqa_dataloader
from evaluation.vqa_eval import VQAEvaluator
from models.blip2_vqa import BLIP2VQA
from models.fusion_baselines import build_fusion_model
from utils.helpers import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VQA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save prediction JSON.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    is_val = args.split == "val"
    question_file = data_cfg["val_annotation"] if is_val else data_cfg.get("test_annotation", data_cfg["val_annotation"])
    annotation_file = data_cfg["val_answers"] if is_val else None
    image_dir = data_cfg["val_image_dir"] if is_val else data_cfg.get("test_image_dir", data_cfg["val_image_dir"])
    answer_list_file = data_cfg.get("answer_list")

    batch_size = args.batch_size or cfg["training"].get("eval_batch_size", 64)
    loader = build_vqa_dataloader(
        question_file=question_file,
        annotation_file=annotation_file,
        image_dir=image_dir,
        answer_list_file=answer_list_file,
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        is_train=False,
    )
    logger.info("Loaded %d batches for split '%s'", len(loader), args.split)

    # Build answer vocab list from dataset
    idx_to_answer = loader.dataset.idx_to_answer

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_name = model_cfg["name"]
    logger.info("Loading model: %s from %s", model_name, args.checkpoint)

    if model_name == "blip2_vqa":
        model = BLIP2VQA(
            model_name=model_cfg.get("blip2_model_name", "Salesforce/blip2-opt-2.7b"),
            num_answers=model_cfg.get("num_answers", 3129),
            mode="classify",
        )
    else:
        model = build_fusion_model(
            model_name,
            visual_dim=model_cfg.get("hidden_size", 768),
            text_dim=model_cfg.get("hidden_size", 768),
            fusion_dim=model_cfg.get("fusion_output_size", 1024),
            num_answers=model_cfg.get("num_answers", 3129),
        )

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    all_question_ids: list = []
    all_predicted_answers: list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch["image"].to(device)
            out = model(pixel_values=pixel_values)

            if "logits" in out:
                pred_ids = out["logits"].argmax(dim=-1).tolist()
                preds = [idx_to_answer[i] if idx_to_answer else str(i) for i in pred_ids]
            else:
                preds = [""] * pixel_values.shape[0]

            all_question_ids.extend(batch["question_id"])
            all_predicted_answers.extend(preds)

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    if args.output:
        evaluator_tmp = VQAEvaluator.__new__(VQAEvaluator)
        evaluator_tmp.save_predictions(all_question_ids, all_predicted_answers, args.output)
        logger.info("Saved predictions to %s", args.output)

    # ------------------------------------------------------------------
    # Compute accuracy (val split only)
    # ------------------------------------------------------------------
    if is_val and annotation_file:
        evaluator = VQAEvaluator(
            annotation_file=annotation_file,
            question_file=question_file,
        )
        predictions = [
            {"question_id": qid, "answer": ans}
            for qid, ans in zip(all_question_ids, all_predicted_answers)
        ]
        results = evaluator.compute_accuracy(predictions)
        logger.info("Evaluation results: %s", json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    else:
        logger.info("No annotations available; skipping accuracy computation.")


if __name__ == "__main__":
    main()
