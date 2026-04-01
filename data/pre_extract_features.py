"""Pre-extract ViT-L/14 (CLIP) image features for all COCO 2014 images.
Outputs: data/cache/train_features.h5 and val_features.h5
         Key: str(image_id), Value: float16 array (257, 1024)
Supports resume: skips image_ids already in the cache.
"""

import json
import os

import h5py
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel
from omegaconf import OmegaConf

from data.vqa_dataset import VQAv2Dataset


def pre_extract_features(
    config,
    split: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> None:
    """Run ViT-L/14 on all COCO images for a split and save features to HDF5.

    Each image is saved as float16 with shape (257, 1024):
      - 1 CLS token + 256 patch tokens (ViT-L/14 @ 224)
      - dim = 1024

    Args:
        config:     OmegaConf config (data.* and model.image_encoder)
        split:      "train" or "val"
        device:     "cuda" or "cpu"
        batch_size: images per forward pass (reduce if OOM)
    """
    cfg       = config.data
    data_root = cfg.data_root
    meta      = VQAv2Dataset.SPLIT_FILES[split]

    img_dir    = os.path.join(data_root, cfg.coco_dir,    meta["img_dir"])
    ann_path   = os.path.join(data_root, cfg.vqav2_dir,   meta["ann"])
    cache_dir  = os.path.join(data_root, cfg.cache_dir)
    cache_path = os.path.join(cache_dir, meta["cache"])
    os.makedirs(cache_dir, exist_ok=True)

    # All unique image_ids referenced in VQAv2 annotations
    with open(ann_path) as f:
        ann_data = json.load(f)
    image_ids = sorted({ann["image_id"] for ann in ann_data["annotations"]})
    print(f"[pre_extract/{split}] {len(image_ids):,} unique images → {cache_path}")

    # Load CLIP vision model
    model_name = config.model.image_encoder
    print(f"Loading {model_name} ...")
    processor  = CLIPImageProcessor.from_pretrained(model_name)
    model      = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
    print(f"Model loaded on {device} ✅")

    with h5py.File(cache_path, "a") as h5f:
        existing   = set(h5f.keys())
        to_process = [iid for iid in image_ids if str(iid) not in existing]
        print(f"  Already cached: {len(existing):,} | Remaining: {len(to_process):,}")

        for i in tqdm(range(0, len(to_process), batch_size), desc=f"Extracting {split}"):
            ids    = to_process[i : i + batch_size]
            imgs, valid_ids = [], []

            for iid in ids:
                p = os.path.join(img_dir, f"{meta['img_prefix']}{iid:012d}.jpg")
                if not os.path.exists(p):
                    continue
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid_ids.append(iid)
                except Exception:
                    continue

            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model(**inputs).last_hidden_state  # (B, 257, 1024)
                feats = feats.cpu().numpy().astype("float16")

            for j, iid in enumerate(valid_ids):
                h5f.create_dataset(
                    str(iid), data=feats[j],
                    compression="gzip", compression_opts=1,
                )

    print(f"✅ Done! Cache: {cache_path}")


if __name__ == "__main__":
    cfg = OmegaConf.create({
        "data": {
            "data_root": "/content/data",
            "vqav2_dir": "vqav2", "coco_dir": "coco", "cache_dir": "cache",
            "train_size": 50000, "val_size": 10000,
            "image_size": 224, "seed": 42, "batch_size": 32,
        },
        "model": {"image_encoder": "openai/clip-vit-large-patch14", "query_dim": 768},
    })
    for s in ["train", "val"]:
        pre_extract_features(cfg, s, device="cuda", batch_size=64)
