"""Extract and save visual features from a BLIP-2 / ViT model.

Features are saved as a numpy .npz archive with keys:
    - ``image_ids``  : array of COCO image ids (int64)
    - ``features``   : float32 feature matrix [N, feature_dim]

Usage
-----
python scripts/extract_features.py \\
    --config configs/default.yaml \\
    --image_dir /path/to/coco/images \\
    --output_dir data/features/ \\
    --split val
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.helpers import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract visual features from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing COCO images.")
    parser.add_argument("--output_dir", type=str, default="data/features/",
                        help="Directory to save feature files.")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Dataset split name (used for output filename).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional model checkpoint path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class FeatureExtractor(torch.nn.Module):
    """Lightweight wrapper that exposes vision-encoder features from BLIP-2.

    If the HuggingFace ``transformers`` library is available, a pretrained
    BLIP-2 ViT is used.  Otherwise a random ViT-style projection is used as
    a placeholder.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b") -> None:
        super().__init__()
        try:
            from transformers import Blip2ForConditionalGeneration
            blip2 = Blip2ForConditionalGeneration.from_pretrained(
                model_name, ignore_mismatched_sizes=True
            )
            self.vision_model = blip2.vision_model
            self.feature_dim = blip2.config.vision_config.hidden_size
            self._use_hf = True
        except Exception as exc:
            logger.warning("HuggingFace backend unavailable (%s). Using random projections.", exc)
            self.vision_model = torch.nn.Identity()
            self.feature_dim = 768
            self._use_hf = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled patch features.

        Args:
            pixel_values: ``[B, 3, H, W]``

        Returns:
            ``[B, feature_dim]``
        """
        if self._use_hf:
            out = self.vision_model(pixel_values=pixel_values)
            return out.last_hidden_state.mean(dim=1)
        # Fallback: global average pool of pixel values as dummy features
        return pixel_values.flatten(1).mean(dim=-1, keepdim=True).expand(-1, self.feature_dim)


# ---------------------------------------------------------------------------
# Dataset of raw images (no annotations needed)
# ---------------------------------------------------------------------------


class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple dataset over all .jpg / .png files in a directory."""

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, image_dir: str, image_size: int = 224) -> None:
        self.image_paths = [
            p for p in sorted(Path(image_dir).iterdir())
            if p.suffix.lower() in self._IMAGE_EXTS
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        # Extract COCO image id: use the last contiguous numeric run in the stem.
        # e.g. "COCO_val2014_000000123456" → 123456
        import re
        numeric_parts = re.findall(r"\d+", path.stem)
        image_id = int(numeric_parts[-1]) if numeric_parts else idx
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 128)
        return {"image_id": image_id, "image": self.transform(img)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("blip2_model_name", "Salesforce/blip2-opt-2.7b")

    # Load extractor
    logger.info("Loading feature extractor …")
    extractor = FeatureExtractor(model_name).to(device).eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        extractor.load_state_dict(state.get("model_state_dict", state), strict=False)
        logger.info("Loaded weights from %s", args.checkpoint)

    # Dataset
    dataset = ImageFolderDataset(args.image_dir, image_size=args.image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info("Found %d images in %s", len(dataset), args.image_dir)

    # Extraction loop
    all_ids: list = []
    all_features: list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            pixel_values = batch["image"].to(device)
            feats = extractor(pixel_values)
            all_ids.extend(batch["image_id"].tolist())
            all_features.append(feats.cpu().numpy())

    image_ids = np.array(all_ids, dtype=np.int64)
    features = np.concatenate(all_features, axis=0).astype(np.float32)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_features.npz"
    np.savez_compressed(output_path, image_ids=image_ids, features=features)
    logger.info(
        "Saved %d features (dim=%d) to %s",
        len(image_ids), features.shape[1], output_path,
    )


if __name__ == "__main__":
    main()
