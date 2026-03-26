"""Interactive VQA demo.

Ask a question about any image using a trained BLIP-2 VQA model.

Usage
-----
# Answer a single question:
python scripts/demo.py \\
    --checkpoint checkpoints/best_model.pth \\
    --image /path/to/image.jpg \\
    --question "What color is the car?"

# Interactive REPL (enter questions one by one):
python scripts/demo.py --checkpoint checkpoints/best_model.pth --image /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision import transforms

from models.blip2_vqa import BLIP2VQA
from models.fusion_baselines import build_fusion_model
from utils.helpers import load_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive VQA demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to an image file.")
    parser.add_argument("--question", type=str, default=None,
                        help="Question string. If omitted, launches interactive mode.")
    parser.add_argument("--answer_list", type=str, default=None,
                        help="Path to answer vocabulary JSON (classify mode).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Show top-K predicted answers (classify mode).")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


def load_image(path: str) -> torch.Tensor:
    """Load and preprocess a single image.

    Args:
        path: Filesystem path to the image.

    Returns:
        Preprocessed tensor with batch dimension ``[1, 3, H, W]``.
    """
    img = Image.open(path).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    cfg: dict,
    device: torch.device,
) -> tuple:
    """Load the model and answer vocabulary from a checkpoint.

    Returns:
        ``(model, idx_to_answer)`` tuple.
    """
    import json

    state = torch.load(checkpoint_path, map_location=device)
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "blip2_vqa")
    num_answers = model_cfg.get("num_answers", 3129)

    if model_name == "blip2_vqa":
        model = BLIP2VQA(
            model_name=model_cfg.get("blip2_model_name", "Salesforce/blip2-opt-2.7b"),
            num_answers=num_answers,
            mode="classify",
        )
    else:
        model = build_fusion_model(
            model_name,
            visual_dim=model_cfg.get("hidden_size", 768),
            text_dim=model_cfg.get("hidden_size", 768),
            fusion_dim=model_cfg.get("fusion_output_size", 1024),
            num_answers=num_answers,
        )

    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()

    # Load answer vocabulary if available
    idx_to_answer: list = []
    answer_list_path = cfg.get("data", {}).get("answer_list")
    if answer_list_path and Path(answer_list_path).exists():
        with open(answer_list_path) as f:
            idx_to_answer = json.load(f)

    return model, idx_to_answer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def answer_question(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    idx_to_answer: list,
    top_k: int = 5,
) -> list:
    """Run a forward pass and return top-K answers with scores.

    Args:
        model: Loaded VQA model.
        pixel_values: Preprocessed image tensor ``[1, 3, H, W]``.
        idx_to_answer: Answer vocabulary list.
        top_k: Number of top answers to return.

    Returns:
        List of ``(answer_str, probability_float)`` tuples.
    """
    out = model(pixel_values=pixel_values)
    logits = out.get("logits")
    if logits is None:
        return [("(no prediction)", 0.0)]

    probs = torch.softmax(logits[0], dim=-1)
    top_vals, top_idxs = probs.topk(min(top_k, len(probs)))

    results = []
    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        ans = idx_to_answer[idx] if idx < len(idx_to_answer) else str(idx)
        results.append((ans, round(val * 100, 2)))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    cfg = load_config(args.config)
    model, idx_to_answer = load_model(args.checkpoint, cfg, device)
    pixel_values = load_image(args.image).to(device)

    def _ask(question: str) -> None:
        predictions = answer_question(model, pixel_values, idx_to_answer, top_k=args.top_k)
        print(f"\nQ: {question}")
        print("Top predictions:")
        for rank, (ans, prob) in enumerate(predictions, 1):
            print(f"  {rank}. {ans}  ({prob:.1f}%)")

    if args.question:
        _ask(args.question)
    else:
        print(f"Image: {args.image}")
        print("Type a question and press Enter.  Type 'quit' or Ctrl-C to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("Bye!")
                break
            _ask(question)


if __name__ == "__main__":
    main()
