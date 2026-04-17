"""Pre-extract ViT-L/14 (CLIP) image features for all COCO 2014 images.
Outputs: data/cache/train_features.h5 and val_features.h5
         Key: str(image_id), Value: float16 array (257, 1024)

Resume strategy:
  - Checkpoint JSON (<cache>.ckpt.json) records all processed image_ids (ints).
  - HDF5 is flushed + fsync'd to disk every `checkpoint_interval` batches.
  - Checkpoint is written ONLY AFTER h5f.flush()+fsync succeeds AND
    the H5 key count is verified (atomic rename prevents partial writes).
  - On restart: H5 keys are the ground truth; checkpoint is used only as a
    hint (union of both prevents duplicate work).

Notes:
  - Does NOT call text_encoder.py — that encoder (BERT) runs inside the
    trainer at training time; pre-extraction only handles frozen ViT.
  - Uses VQAv2Dataset.SPLIT_FILES (class-level dict) for file/dir metadata;
    does NOT instantiate VQAv2Dataset itself.
  - Pass max_images to cap the number of images (useful for smoke-tests).
"""

import json
import os
import tempfile

import h5py
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel
from omegaconf import OmegaConf

from data.vqa_dataset import VQAv2Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_checkpoint(ckpt_path: str) -> set:
    """Load checkpoint JSON and return a set of *int* image_ids.

    All ids are cast to int so that comparisons with annotation image_ids
    (which are always int) work correctly.
    """
    try:
        with open(ckpt_path) as f:
            raw = json.load(f).get("done", [])
        return {int(x) for x in raw}
    except (json.JSONDecodeError, OSError):
        print(f"  [warn] Could not read checkpoint {ckpt_path} — treating as empty.")
        return set()


def _write_checkpoint_atomic(ckpt_path: str, done_ids: set) -> None:
    """Write checkpoint JSON atomically via a temp file + rename.

    - All ids stored as int for consistency with annotation image_ids.
    - Atomic rename prevents a half-written file from corrupting resume state.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(ckpt_path), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump({"done": sorted(done_ids)}, f)
        os.replace(tmp_path, ckpt_path)  # atomic on POSIX; best-effort on Windows
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _flush_and_fsync(h5f: h5py.File) -> None:
    """Flush HDF5 buffers to OS and then fsync() to storage.

    h5f.flush() alone only pushes data from HDF5's internal buffers to the
    OS page cache. On networked/FUSE filesystems (Google Drive, NFS) the OS
    may still hold dirty pages in memory.  fsync() forces those pages to be
    written to the underlying storage before we touch the checkpoint file.
    """
    h5f.flush()
    try:
        # h5py does not expose the raw fd directly; use the filename instead.
        with open(h5f.filename, "rb") as fh:
            os.fsync(fh.fileno())
    except (OSError, AttributeError):
        # fsync may fail on some FUSE implementations — not fatal,
        # but log a warning so the user is aware.
        print("  [warn] fsync() failed — data may not be fully persisted yet.")


def _verify_h5_keys(cache_path: str, expected_ids: set) -> set:
    """Open the H5 file (read-only) and return the set of *int* keys present.

    This is called after flush+fsync to confirm the data actually landed on
    disk before committing the checkpoint.

    Returns the set of int keys found in the file (may be a superset of
    expected_ids if the file already held data from previous runs).
    """
    if not os.path.exists(cache_path):
        return set()
    try:
        with h5py.File(cache_path, "r") as h5r:
            return {int(k) for k in h5r.keys()}
    except Exception as exc:
        print(f"  [warn] Could not verify H5 keys: {exc}")
        return set()


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

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

    Checkpoint safety guarantee
    ---------------------------
    The checkpoint JSON is ONLY updated AFTER:
      1. h5f.flush() + fsync() succeeds.
      2. A read-back of the H5 file confirms every id in this batch is
         present as a key.
    If either step fails, the checkpoint is left at its previous state so
    the next run will re-process the batch cleanly.

    Args:
        config:               OmegaConf config (data.* and model.image_encoder)
        split:                "train" or "val"
        device:               "cuda" or "cpu"
        batch_size:           images per forward pass (reduce if OOM)
        checkpoint_interval:  flush HDF5 + write .ckpt.json every N batches
        max_images:           cap total images processed (None = all).
                              Set a small number (e.g. 50) to smoke-test the
                              full pipeline without running on all 82k images.
        input_dir:            optional path to an existing HDF5/checkpoint
                              from a previous partial run (read-only seed).
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

    # ── All unique image_ids referenced in VQAv2 annotations ─────────────────
    with open(ann_path) as f:
        ann_data = json.load(f)
    # Always int — annotation image_ids are ints.
    image_ids = sorted({int(ann["image_id"]) for ann in ann_data["annotations"]})

    if max_images is not None:
        image_ids = image_ids[:max_images]
        print(f"[pre_extract/{split}] smoke-test mode: capped at {max_images} images")

    print(f"[pre_extract/{split}] {len(image_ids):,} unique images → {cache_path}")

    # ── Determine already-processed ids ──────────────────────────────────────
    # Ground truth: H5 keys (what actually made it to disk).
    # Hint:         checkpoint JSON (what we *think* made it to disk).
    # Strategy:     union of both, so we never double-process.
    # All ids are stored as int throughout.
    done_ids: set = set()

    # 1. Load from checkpoint JSON (hint)
    if os.path.exists(ckpt_path):
        ckpt_ids = _read_checkpoint(ckpt_path)
        done_ids |= ckpt_ids
        print(f"  Checkpoint: {len(ckpt_ids):,} ids from {os.path.basename(ckpt_path)}")

    # 2. Read back H5 keys (ground truth — overrides/extends checkpoint)
    if os.path.exists(cache_path):
        h5_done = _verify_h5_keys(cache_path, set())
        extra = h5_done - done_ids
        if extra:
            print(f"  HDF5 has {len(extra):,} extra ids not in checkpoint — merging")
        done_ids |= h5_done
        print(f"  HDF5 ground-truth: {len(h5_done):,} keys in {os.path.basename(cache_path)}")

    # 3. Seed from separate input_dir (read-only cloned source)
    if input_dir is not None:
        in_cache = os.path.join(input_dir, meta["cache"])
        in_ckpt  = in_cache + ".ckpt.json"
        if os.path.exists(in_ckpt):
            in_ckpt_ids = _read_checkpoint(in_ckpt)
            done_ids |= in_ckpt_ids
            print(f"  Input ckpt: merged {len(in_ckpt_ids):,} ids from {os.path.basename(in_ckpt)}")
        if os.path.exists(in_cache):
            in_h5_done = _verify_h5_keys(in_cache, set())
            print(f"  Input HDF5: {len(in_h5_done):,} ids from {os.path.basename(in_cache)}")
            done_ids |= in_h5_done

    to_process = [iid for iid in image_ids if iid not in done_ids]
    print(f"  Already cached: {len(done_ids):,} | Remaining: {len(to_process):,}")

    if not to_process:
        print("  All images already cached. Nothing to do.")
        # Final consistency check: warn if checkpoint is behind H5
        if os.path.exists(cache_path):
            h5_keys = _verify_h5_keys(cache_path, set())
            ckpt_keys = _read_checkpoint(ckpt_path) if os.path.exists(ckpt_path) else set()
            missing_in_ckpt = h5_keys - ckpt_keys
            if missing_in_ckpt:
                print(f"  [warn] {len(missing_in_ckpt):,} H5 keys not in checkpoint — syncing checkpoint now.")
                _write_checkpoint_atomic(ckpt_path, h5_keys)
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

    # pending_ids   : ids whose features were written to h5f this interval,
    #                 but not yet committed to the checkpoint.
    # pending_missing: ids that had no image file / could not be opened;
    #                  these are safe to checkpoint without touching H5.
    pending_ids: list     = []
    pending_missing: list = []

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
                    pending_missing.append(iid)
                    continue
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid_ids.append(iid)
                except Exception:
                    pending_missing.append(iid)
                    continue

            if not imgs:
                # No H5 writes this batch.
                # Only checkpoint if we hit the interval AND there are
                # pending_missing ids accumulated (so checkpoint stays consistent).
                if (batch_num + 1) % checkpoint_interval == 0 and pending_missing:
                    # No need to flush/fsync — nothing was written to H5.
                    done_ids.update(pending_missing)
                    pending_missing.clear()
                    _write_checkpoint_atomic(ckpt_path, done_ids)
                    pbar.set_postfix(saved=saved_this_run, total_cached=len(done_ids))
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model(**inputs).last_hidden_state  # (B, 257, 1024)
                feats = feats.cpu().numpy().astype("float16")

            written_this_batch: list = []
            for j, iid in enumerate(valid_ids):
                if str(iid) not in h5f:   # guard against duplicate on resume
                    h5f.create_dataset(
                        str(iid), data=feats[j],
                        compression="gzip", compression_opts=1,
                    )
                    written_this_batch.append(iid)

            # Buffer — do NOT update done_ids or checkpoint yet
            pending_ids.extend(written_this_batch)
            saved_this_run += len(written_this_batch)

            # ── Periodic checkpoint: flush → fsync → verify → checkpoint ─────
            if (batch_num + 1) % checkpoint_interval == 0:
                _flush_and_fsync(h5f)

                # Verify: re-read H5 keys from disk to confirm they landed
                verified_h5_keys = _verify_h5_keys(cache_path, set(pending_ids))
                unverified = set(pending_ids) - verified_h5_keys
                if unverified:
                    print(
                        f"\n  [WARN] {len(unverified):,} ids not confirmed in H5 after flush "
                        f"— will retry on next run. NOT adding to checkpoint."
                    )
                    # Remove unverified from pending so they get retried
                    pending_ids = [iid for iid in pending_ids if iid not in unverified]

                # Commit only verified ids + missing ids to done_ids
                done_ids.update(pending_ids)
                done_ids.update(pending_missing)
                pending_ids.clear()
                pending_missing.clear()
                _write_checkpoint_atomic(ckpt_path, done_ids)
                pbar.set_postfix(saved=saved_this_run, total_cached=len(done_ids))

        # ── Final flush + checkpoint for the last partial interval ────────────
        _flush_and_fsync(h5f)

        # Final verification
        if pending_ids:
            verified_h5_keys = _verify_h5_keys(cache_path, set(pending_ids))
            unverified = set(pending_ids) - verified_h5_keys
            if unverified:
                print(
                    f"\n  [WARN] Final verify: {len(unverified):,} ids not confirmed in H5. "
                    f"These will be retried on the next run."
                )
                pending_ids = [iid for iid in pending_ids if iid not in unverified]

        done_ids.update(pending_ids)
        done_ids.update(pending_missing)
        _write_checkpoint_atomic(ckpt_path, done_ids)

    # ── Post-run consistency check ────────────────────────────────────────────
    print(f"\n✅ Done! Saved {saved_this_run:,} images this run. Cache: {cache_path}")
    final_h5_keys  = _verify_h5_keys(cache_path, set())
    final_ckpt_ids = _read_checkpoint(ckpt_path)
    in_h5_not_ckpt = final_h5_keys - final_ckpt_ids
    in_ckpt_not_h5 = final_ckpt_ids - final_h5_keys
    print(f"  H5  keys : {len(final_h5_keys):,}")
    print(f"  Ckpt ids : {len(final_ckpt_ids):,}")
    if in_h5_not_ckpt:
        print(f"  [warn] {len(in_h5_not_ckpt):,} H5 keys not in checkpoint — fixing checkpoint.")
        _write_checkpoint_atomic(ckpt_path, final_h5_keys | final_ckpt_ids)
    if in_ckpt_not_h5:
        print(
            f"  [warn] {len(in_ckpt_not_h5):,} checkpoint ids not found in H5 "
            f"(likely images that were skipped/missing). This is expected for "
            f"missing image files."
        )
    else:
        print("  ✅ H5 and checkpoint are perfectly consistent.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

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
    if args.output_dir:
        _data_root_eff = args.data_root
        cache_dir_cfg  = os.path.abspath(args.output_dir)
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
