"""Pre-extract ViT-L/14 (CLIP) image features for all COCO 2014 images.
Outputs: data/cache/train_features.h5 and val_features.h5
         Key: str(image_id), Value: float16 array (257, 1024)

Resume strategy:
  - Checkpoint JSON (<cache>.ckpt.json) records all processed image_ids.
  - HDF5 is flushed to disk every `checkpoint_interval` batches.
  - On restart: merges checkpoint + HDF5 keys → no duplicate work.

Notes:
  - Does NOT call text_encoder.py — that encoder (BERT) runs inside the
    trainer at training time; pre-extraction only handles frozen ViT.
  - Uses VQAv2Dataset.SPLIT_FILES (class-level dict) for file/dir metadata;
    does NOT instantiate VQAv2Dataset itself.
  - Pass max_images to cap the number of images (useful for smoke-tests).
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
    checkpoint_interval: int = 10,
    max_images: int | None = None,
    input_dir: str | None = None,
) -> None:
    """Run ViT-L/14 on all (or a capped subset of) COCO images and save to HDF5.

    Each image is saved as float16 with shape (257, 1024):
      - 1 CLS token + 256 patch tokens (ViT-L/14 @ 224)
      - dim = 1024

    Args:
        config:               OmegaConf config (data.* and model.image_encoder)
        split:                "train" or "val"
        device:               "cuda" or "cpu"
        batch_size:           images per forward pass (reduce if OOM)
        checkpoint_interval:  flush HDF5 + write .ckpt.json every N batches
        max_images:           cap total images processed (None = all).
                              Set a small number (e.g. 50) to smoke-test the
                              full pipeline without running on all 82k images.
    """
    cfg       = config.data
    data_root = cfg.data_root
    meta      = VQAv2Dataset.SPLIT_FILES[split]

    img_dir    = os.path.join(data_root, cfg.coco_dir,  meta["img_dir"])
    ann_path   = os.path.join(data_root, cfg.vqav2_dir, meta["ann"])
    cache_dir  = os.path.join(data_root, cfg.cache_dir)
    cache_path = os.path.join(cache_dir, meta["cache"])
    ckpt_path  = cache_path + ".ckpt.json"
    os.makedirs(cache_dir, exist_ok=True)

    # All unique image_ids referenced in VQAv2 annotations
    with open(ann_path) as f:
        ann_data = json.load(f)
    image_ids = sorted({ann["image_id"] for ann in ann_data["annotations"]})

    if max_images is not None:
        image_ids = image_ids[:max_images]
        print(f"[pre_extract/{split}] smoke-test mode: capped at {max_images} images")

    print(f"[pre_extract/{split}] {len(image_ids):,} unique images → {cache_path}")

    # ── Determine already-processed ids (checkpoint merge strategy) ──────────
    done_ids: set = set()

    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            done_ids = set(json.load(f).get("done", []))
        print(f"  Checkpoint: {len(done_ids):,} ids from {os.path.basename(ckpt_path)}")

    if os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as h5f:
            h5_done = {int(k) for k in h5f.keys()}
        extra = h5_done - done_ids
        if extra:
            print(f"  HDF5 has {len(extra):,} extra ids not in checkpoint — merging")
        done_ids |= h5_done

    # ── Seed done_ids from a separate input_dir (cloned source) ──────────────
    if input_dir is not None:
        in_cache = os.path.join(input_dir, meta["cache"])
        in_ckpt  = in_cache + ".ckpt.json"
        if os.path.exists(in_ckpt):
            with open(in_ckpt) as f:
                done_ids |= set(json.load(f).get("done", []))
            print(f"  Input ckpt: merged from {os.path.basename(in_ckpt)}")
        if os.path.exists(in_cache):
            with h5py.File(in_cache, "r") as h5f:
                in_done = {int(k) for k in h5f.keys()}
            print(f"  Input HDF5: {len(in_done):,} ids from {os.path.basename(in_cache)}")
            done_ids |= in_done

    to_process = [iid for iid in image_ids if iid not in done_ids]
    print(f"  Already cached: {len(done_ids):,} | Remaining: {len(to_process):,}")

    if not to_process:
        print("  All images already cached. Nothing to do.")
        return

    # ── Load CLIP vision model ────────────────────────────────────────────────
    model_name = config.model.image_encoder
    print(f"Loading {model_name} ...")
    processor  = CLIPImageProcessor.from_pretrained(model_name)
    model      = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {device} ✅")

    # ── Extraction loop with periodic checkpoint ──────────────────────────────
    saved_this_run = 0
    n_batches = (len(to_process) + batch_size - 1) // batch_size

    with h5py.File(cache_path, "a") as h5f:
        pbar = tqdm(
            range(0, len(to_process), batch_size),
            total=n_batches,
            desc=f"Extracting {split}",
        )
        for batch_num, i in enumerate(pbar):
            ids = to_process[i : i + batch_size]
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
                done_ids.update(ids)  # mark missing as done so they aren't retried
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model(**inputs).last_hidden_state  # (B, 257, 1024)
                feats = feats.cpu().numpy().astype("float16")

            for j, iid in enumerate(valid_ids):
                if str(iid) not in h5f:   # guard against duplicate on resume
                    h5f.create_dataset(
                        str(iid), data=feats[j],
                        compression="gzip", compression_opts=1,
                    )

            done_ids.update(valid_ids)
            saved_this_run += len(valid_ids)

            # Flush HDF5 + write checkpoint every checkpoint_interval batches
            if (batch_num + 1) % checkpoint_interval == 0:
                h5f.flush()
                with open(ckpt_path, "w") as f:
                    json.dump({"done": list(done_ids)}, f)
                pbar.set_postfix(saved=saved_this_run, total_cached=len(done_ids))

        # Final flush + checkpoint for the last partial interval
        h5f.flush()
        with open(ckpt_path, "w") as f:
            json.dump({"done": list(done_ids)}, f)

    print(f"✅ Done! Saved {saved_this_run:,} images this run. Cache: {cache_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-extract CLIP image features")
    parser.add_argument("--data_root",   default="/content/drive/MyDrive/blip2_project/data",
                        help="Thư mục gốc chứa ảnh và annotation VQAv2.")
    parser.add_argument("--output_dir",  default=None,
                        help="Thư mục đích để lưu file HDF5 (mặc định: {data_root}/cache). "
                             "Override hoàn toàn data_root/cache_dir.")
    parser.add_argument("--vqav2_dir",   default="vqav2",
                        help="Tên thư mục JSON annotation bên trong data_root.")
    parser.add_argument("--coco_dir",    default="coco",
                        help="Tên thư mục ảnh COCO bên trong data_root.")
    parser.add_argument("--model",       default="openai/clip-vit-large-patch14",
                        help="HuggingFace model id của CLIP vision encoder.")
    parser.add_argument("--split",       default="train", choices=["train", "val", "both"])
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--ckpt_every",  type=int, default=10,
                        help="Flush + save checkpoint every N batches")
    parser.add_argument("--max_images",  type=int, default=None,
                        help="Cap images for a smoke-test (e.g. --max_images 50)")
    parser.add_argument("--input_dir",   default=None,
                        help="Thư mục chứa HDF5 đã extract sẵn (cloned source). "
                             "Dùng để biết image nào đã xong, KHÔNG ghi vào đây. "
                             "Kết hợp với --output_dir để extract tiếp mà ko động cái gốc.")
    args = parser.parse_args()

    # --output_dir overrides data_root/cache_dir khi được truyền vào
    # Nếu không truyền, dùng data_root/cache (hành vi cũ)
    import os as _os
    if args.output_dir:
        # data_root vẫn dùng args.data_root để đọc annotation + ảnh
        # cache_dir dùng absolute path → os.path.join sẽ bỏ qua data_root
        _data_root_eff = args.data_root
        cache_dir_cfg  = _os.path.abspath(args.output_dir)
    else:
        _data_root_eff = args.data_root
        cache_dir_cfg  = "cache"

    cfg = OmegaConf.create({
        "data": {
            "data_root": _data_root_eff,
            "vqav2_dir": args.vqav2_dir,
            "coco_dir":  args.coco_dir,
            "cache_dir": cache_dir_cfg,
            "train_size": 50000, "val_size": 10000,
            "image_size": 224, "seed": 42, "batch_size": 32,
        },
        "model": {"image_encoder": args.model, "query_dim": 768},
    })

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        pre_extract_features(
            cfg,
            split=s,
            device=args.device,
            batch_size=args.batch_size,
            checkpoint_interval=args.ckpt_every,
            max_images=args.max_images,
            input_dir=args.input_dir,
        )
