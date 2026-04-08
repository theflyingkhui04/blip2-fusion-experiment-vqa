"""Đánh giá model VQA đã huấn luyện trên tập val hoặc test.

Sử dụng
-------
python scripts/evaluate.py \\
    --config configs/default.yaml \\
    --checkpoint checkpoints/best_model.pth \\
    --split val \\
    --output results/val_predictions.json

Ghi chú pipeline
----------------
- Nếu ``model.name == "blip2_vqa"``: dùng BLIP2VQA legacy pipeline.
- Ngược lại (EXP-01 → EXP-07): dùng EXP pipeline:
    HDF5 image_features + FrozenTextEncoder (BERT) → fusion model → logits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from configs.contracts import (
    KEY_ANSWER_TEXT,
    KEY_ATTENTION_MASK,
    KEY_IMAGE_FEATURES,
    KEY_INPUT_IDS,
    KEY_LOGITS,
    KEY_MODEL_STATE,
    KEY_PIXEL_VALUES,
    KEY_QUESTION_IDS,
)
from data.vqa_dataset import build_dataloader
from evaluation.vqa_eval import VQAEvaluator
from models import build_model, FrozenTextEncoder
from models.blip2_vqa import BLIP2VQA
from utils.helpers import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá model VQA trên tập val/test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Đường dẫn checkpoint (.pth).")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Split cần đánh giá.")
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn lưu file JSON dự đoán.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_dict = load_config(args.config)
    if args.batch_size:
        cfg_dict.setdefault("data", {})["batch_size"] = args.batch_size
    config = OmegaConf.create(cfg_dict)

    set_seed(int(getattr(config.training, "seed", 42)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    model_name = config.model.name

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    use_cache = (model_name != "blip2_vqa")
    logger.info("Xây dựng DataLoader cho split '%s' (use_cache=%s) …", args.split, use_cache)
    loader = build_dataloader(args.split, config, use_cache=use_cache)
    logger.info("Số batch: %d", len(loader))

    # Lấy ánh xạ index → câu trả lời từ vocab của dataset
    idx_to_answer: list = loader.dataset.idx_to_answer

    # ------------------------------------------------------------------
    # Khởi tạo model
    # ------------------------------------------------------------------
    logger.info("Tải model '%s' từ %s …", model_name, args.checkpoint)
    text_encoder = None

    if model_name == "blip2_vqa":
        model = BLIP2VQA(
            model_name=getattr(config.model, "blip2_model_name", "Salesforce/blip2-opt-2.7b"),
            num_answers=int(getattr(config.model, "num_answers", 3129)),
            mode="classify",
        )
    else:
        model = build_model(config)
        text_encoder = FrozenTextEncoder()
        text_encoder = text_encoder.to(device)
        text_encoder.eval()

    state = torch.load(args.checkpoint, map_location=device)
    # Checkpoint có thể là dict với key KEY_MODEL_STATE hoặc raw state_dict
    state_dict = state.get(KEY_MODEL_STATE, state)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    all_question_ids: list = []
    all_predicted_answers: list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if text_encoder is not None:
                # EXP pipeline
                visual_features = batch[KEY_IMAGE_FEATURES].to(device)
                input_ids       = batch[KEY_INPUT_IDS].to(device)
                attention_mask  = batch.get(KEY_ATTENTION_MASK)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                text_features = text_encoder(input_ids, attention_mask)
                logits = model(visual_features, text_features)
                pred_ids = logits.argmax(dim=-1).tolist()
            else:
                # Legacy BLIP2VQA pipeline
                pixel_values = batch[KEY_PIXEL_VALUES].to(device)
                input_ids    = batch.get(KEY_INPUT_IDS)
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                attention_mask = batch.get(KEY_ATTENTION_MASK)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                out = model(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask)
                if KEY_LOGITS in out:
                    pred_ids = out[KEY_LOGITS].argmax(dim=-1).tolist()
                else:
                    pred_ids = [0] * pixel_values.shape[0]

            preds = [
                idx_to_answer[i] if idx_to_answer and i < len(idx_to_answer) else str(i)
                for i in pred_ids
            ]
            all_question_ids.extend(batch[KEY_QUESTION_IDS])
            all_predicted_answers.extend(preds)

    # ------------------------------------------------------------------
    # Lưu predictions
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions = [
            {"question_id": int(qid), "answer": ans}
            for qid, ans in zip(all_question_ids, all_predicted_answers)
        ]
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info("Đã lưu dự đoán tại %s", out_path)

    # ------------------------------------------------------------------
    # Tính accuracy (chỉ cho split 'val' khi có annotation)
    # ------------------------------------------------------------------
    if args.split == "val":
        ann_file = getattr(config.data, "val_annotation_file", None)
        q_file   = getattr(config.data, "val_question_file", None)
        if ann_file and q_file:
            evaluator = VQAEvaluator(annotation_file=ann_file, question_file=q_file)
            predictions = [
                {"question_id": int(qid), "answer": ans}
                for qid, ans in zip(all_question_ids, all_predicted_answers)
            ]
            results = evaluator.compute_accuracy(predictions)
            logger.info("Kết quả đánh giá: %s", json.dumps(results, indent=2))
            print(json.dumps(results, indent=2))
        else:
            logger.info("Không tìm thấy annotation file; bỏ qua tính accuracy.")
    else:
        logger.info("Split '%s': không có annotation; bỏ qua tính accuracy.", args.split)


if __name__ == "__main__":
    main()
