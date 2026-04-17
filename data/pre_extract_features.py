"""Pre-extract ViT-L/14 (CLIP) image features for all COCO 2014 images.
Outputs: data/cache/train_features.h5 and val_features.h5
         Key: str(image_id), Value: float16 array (257, 1024)

Drive-safe write strategy (Colab + Google Drive FUSE)
=====================================================
Writing HDF5 directly to Drive FUSE is UNSAFE: HDF5 performs random-access
writes (superblock at byte 0 + data at end). Drive FUSE may flush these out
of order, leaving a corrupt superblock when the runtime dies.

Safe strategy used here:
  1. ALL H5 writes go to a LOCAL directory (/content/h5_cache by default).
     Local disk is a fast SSD and POSIX-compliant — no FUSE involved.
  2. At each checkpoint interval:
       a. Flush + fsync local H5.
       b. Verify local H5 has all expected keys.
       c. Copy local H5 → Drive H5 (sequential write, FUSE handles OK).
       d. Verify Drive H5 has expected keys.
       e. Write checkpoint JSON to Drive (atomic rename).
  3. If the Drive copy fails at any step, skip updating the checkpoint and
     retry at the next interval — no data loss.

Resume strategy:
  - On start: copy Drive H5 → local (if it exists and is valid).
  - If Drive H5 is corrupted: warn, ignore it, start fresh locally.
  - Checkpoint JSON + Drive H5 are always consistent.

Notes:
  - Does NOT call text_encoder.py — BERT runs inside the trainer at
    training time; pre-extraction only handles frozen ViT.
  - Uses VQAv2Dataset.SPLIT_FILES for file/dir metadata.
  - Pass max_images to cap images (useful for smoke-tests).
"""

import json
import os
import shutil
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
    """Load checkpoint JSON → set of *int* image_ids."""
    try:
        with open(ckpt_path) as f:
            raw = json.load(f).get("done", [])
        return {int(x) for x in raw}
    except (json.JSONDecodeError, OSError):
        print(f"  [warn] Could not read checkpoint {ckpt_path} — treating as empty.")
        return set()


def _write_checkpoint_atomic(ckpt_path: str, done_ids: set) -> None:
    """Write checkpoint JSON atomically (temp file → rename).

    Stores ids as sorted ints for consistency and readability.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(ckpt_path), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump({"done": sorted(done_ids)}, f)
        os.replace(tmp_path, ckpt_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _flush_and_fsync(h5f: h5py.File) -> None:
    """Flush HDF5 internal buffers and fsync to storage."""
    h5f.flush()
    try:
        with open(h5f.filename, "rb") as fh:
            os.fsync(fh.fileno())
    except (OSError, AttributeError):
        pass  # fsync failure is non-fatal for local SSD


def _verify_h5_keys(path: str) -> set:
    """Open H5 read-only and return set of *int* keys. Returns empty on error."""
    if not os.path.exists(path):
        return set()
    try:
        with h5py.File(path, "r") as h5r:
            return {int(k) for k in h5r.keys()}
    except Exception as exc:
        print(f"  [warn] Could not read H5 keys from {os.path.basename(path)}: {exc}")
        return set()


def _sync_to_drive(
    local_path: str,
    drive_path: str,
    must_have_ids: set,
) -> bool:
    """Copy local H5 → Drive H5 and verify. Returns True on success.

    This is a plain sequential file copy — Drive FUSE handles sequential
    writes reliably, unlike random-access H5 writes.
    """
    try:
        print(f"\n  Syncing to Drive ({os.path.getsize(local_path)/1e6:.1f} MB)…", end=" ", flush=True)
        shutil.copy2(local_path, drive_path)
        # Flush Drive FUSE page cache for this file
        try:
            with open(drive_path, "rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass

        # Verify Drive copy has all required keys
        drive_keys = _verify_h5_keys(drive_path)
        missing = must_have_ids - drive_keys
        if missing:
            print(f"WARN — {len(missing):,} keys missing in Drive copy!")
            return False

        print(f"OK ({len(drive_keys):,} keys on Drive) ✅")
        return True

    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def pre_extract_features(
    config,
    split: str,
    device: str = "cuda",
    batch_size: int = 64,
    checkpoint_interval: int = 20,
    max_images: int | None = None,
    input_dir: str | None = None,
    local_work_dir: str = "/content/h5_cache",
) -> None:
    """Run ViT-L/14 on COCO images and save features to HDF5.

    H5 is written locally to ``local_work_dir`` and synced to Drive only
    at each checkpoint — this is the only safe way to use HDF5 on Drive FUSE.

    Args:
        config:              OmegaConf config (data.* and model.image_encoder)
        split:               "train" or "val"
        device:              "cuda" or "cpu"
        batch_size:          Images per forward pass (reduce if OOM)
        checkpoint_interval: Sync to Drive every N batches (default 20 =
                             ~1,280 images with batch_size=64)
        max_images:          Cap total images (None = all)
        input_dir:           Read-only source of already-extracted features
        local_work_dir:      Local directory to write H5 to (default
                             /content/h5_cache); must be on local disk, NOT
                             on a FUSE mount like Drive
    """
    cfg       = config.data
    data_root = cfg.data_root
    meta      = VQAv2Dataset.SPLIT_FILES[split]

    img_dir    = os.path.join(data_root, cfg.coco_dir,  meta["img_dir"])
    ann_path   = os.path.join(data_root, cfg.vqav2_dir, meta["ann"])
    cache_dir  = os.path.join(data_root, cfg.cache_dir)  # Drive output dir
    cache_path = os.path.join(cache_dir, meta["cache"])   # Drive H5
    ckpt_path  = cache_path + ".ckpt.json"                # Drive checkpoint
    os.makedirs(cache_dir, exist_ok=True)

    # Local working copy — all H5 writes go here
    os.makedirs(local_work_dir, exist_ok=True)
    local_cache_path = os.path.join(local_work_dir, meta["cache"])

    print(f"[pre_extract/{split}]")
    print(f"  Drive H5 : {cache_path}")
    print(f"  Local H5 : {local_cache_path}  ← writes go here")

    # ── Load annotation image_ids ─────────────────────────────────────────────
    with open(ann_path) as f:
        ann_data = json.load(f)
    image_ids = sorted({int(ann["image_id"]) for ann in ann_data["annotations"]})

    if max_images is not None:
        image_ids = image_ids[:max_images]
        print(f"  smoke-test: capped at {max_images:,} images")

    print(f"  Total unique images in annotation: {len(image_ids):,}")

    # ── Determine already-done ids ────────────────────────────────────────────
    done_ids: set = set()

    # 1. Checkpoint JSON on Drive (hint — may be ahead of reality)
    if os.path.exists(ckpt_path):
        ckpt_ids = _read_checkpoint(ckpt_path)
        if ckpt_ids:
            done_ids |= ckpt_ids
            print(f"  Checkpoint JSON: {len(ckpt_ids):,} ids")

    # 2. Drive H5 (ground truth) — copy to local if valid
    if os.path.exists(cache_path):
        drive_keys = _verify_h5_keys(cache_path)
        if drive_keys:
            print(f"  Drive H5 has {len(drive_keys):,} valid keys — copying to local…")
            try:
                shutil.copy2(cache_path, local_cache_path)
                local_keys = _verify_h5_keys(local_cache_path)
                print(f"  Local copy verified: {len(local_keys):,} keys ✅")
                done_ids |= local_keys
                # Heal checkpoint if behind H5
                orphan_in_ckpt = done_ids - local_keys - {
                    iid for iid in done_ids if iid not in image_ids
                }
            except Exception as exc:
                print(f"  [warn] Could not copy Drive H5 to local: {exc}")
                print(f"         Will start fresh locally (Drive H5 kept as-is).")
        else:
            print(f"  [warn] Drive H5 exists but is corrupted — ignoring it.")
            print(f"         Discarding checkpoint ids that reference it.")
            done_ids.clear()  # checkpoint is untrustworthy without a valid H5
            # Rename corrupted file instead of deleting (safety backup)
            bkp = cache_path + ".corrupted_bkp"
            try:
                shutil.move(cache_path, bkp)
                print(f"         Corrupted H5 renamed → {os.path.basename(bkp)}")
            except Exception:
                pass

    # 3. Seed from input_dir (read-only; usually same as cache_dir)
    if input_dir is not None:
        in_cache = os.path.join(input_dir, meta["cache"])
        in_ckpt  = in_cache + ".ckpt.json"
        # Only use input_dir if it's a DIFFERENT path from Drive output
        if os.path.abspath(in_cache) != os.path.abspath(cache_path):
            if os.path.exists(in_ckpt):
                in_ckpt_ids = _read_checkpoint(in_ckpt)
                done_ids |= in_ckpt_ids
                print(f"  Input ckpt: +{len(in_ckpt_ids):,} ids")
            if os.path.exists(in_cache):
                in_keys = _verify_h5_keys(in_cache)
                done_ids |= in_keys
                print(f"  Input H5  : +{len(in_keys):,} ids")

    # Also check local cache (if a previous session wrote to local but crashed
    # before syncing to Drive)
    if os.path.exists(local_cache_path) and local_cache_path != cache_path:
        local_existing = _verify_h5_keys(local_cache_path)
        new_in_local = local_existing - done_ids
        if new_in_local:
            print(f"  Local H5 has {len(new_in_local):,} extra ids not on Drive — will sync at next ckpt")
            done_ids |= local_existing

    to_process = [iid for iid in image_ids if iid not in done_ids]
    print(f"  Already cached: {len(done_ids):,} | Remaining: {len(to_process):,}")

    if not to_process:
        print("  All images already cached. Nothing to do.")
        # Consistency check
        local_keys = _verify_h5_keys(local_cache_path)
        ckpt_ids   = _read_checkpoint(ckpt_path) if os.path.exists(ckpt_path) else set()
        behind = local_keys - ckpt_ids
        if behind:
            print(f"  Checkpoint behind H5 by {len(behind):,} ids — syncing…")
            _sync_to_drive(local_cache_path, cache_path, local_keys)
            _write_checkpoint_atomic(ckpt_path, local_keys)
        return

    # ── Load CLIP vision model ────────────────────────────────────────────────
    model_name = config.model.image_encoder
    print(f"\nLoading {model_name} …")
    processor  = CLIPImageProcessor.from_pretrained(model_name)
    model      = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded on {device} ✅\n")

    # ── Extraction loop ───────────────────────────────────────────────────────
    saved_this_run = 0
    n_batches      = (len(to_process) + batch_size - 1) // batch_size
    pending_ids:    list = []   # written to local H5, not yet synced to Drive
    pending_missing: list = []  # image files absent — safe to checkpoint w/o H5

    with h5py.File(local_cache_path, "a") as h5f:
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
                if (batch_num + 1) % checkpoint_interval == 0 and pending_missing:
                    done_ids.update(pending_missing)
                    pending_missing.clear()
                    _write_checkpoint_atomic(ckpt_path, done_ids)
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model(**inputs).last_hidden_state  # (B, 257, 1024)
                feats = feats.cpu().numpy().astype("float16")

            written_this_batch: list = []
            for j, iid in enumerate(valid_ids):
                if str(iid) not in h5f:
                    h5f.create_dataset(
                        str(iid), data=feats[j],
                        compression="gzip", compression_opts=1,
                    )
                    written_this_batch.append(iid)

            pending_ids.extend(written_this_batch)
            saved_this_run += len(written_this_batch)

            # ── Checkpoint: local flush → Drive copy → verify → JSON ──────────
            if (batch_num + 1) % checkpoint_interval == 0:
                _flush_and_fsync(h5f)

                # Verify local H5 first
                local_verified = _verify_h5_keys(local_cache_path)
                unverified_local = set(pending_ids) - local_verified
                if unverified_local:
                    print(f"\n  [WARN] {len(unverified_local):,} ids missing in local H5 "
                          f"— will retry.")
                    pending_ids = [x for x in pending_ids if x not in unverified_local]

                # Sync local → Drive (sequential copy, FUSE-safe)
                all_ids_to_commit = set(pending_ids) | set(pending_missing)
                expected_on_drive = done_ids | set(pending_ids)
                sync_ok = _sync_to_drive(local_cache_path, cache_path, expected_on_drive)

                if sync_ok:
                    done_ids.update(pending_ids)
                    done_ids.update(pending_missing)
                    pending_ids.clear()
                    pending_missing.clear()
                    _write_checkpoint_atomic(ckpt_path, done_ids)
                else:
                    print(f"  Checkpoint skipped — will retry at next interval.")

                pbar.set_postfix(saved=saved_this_run, on_drive=len(done_ids))

        # ── Final sync ────────────────────────────────────────────────────────
        _flush_and_fsync(h5f)

        if pending_ids:
            local_verified = _verify_h5_keys(local_cache_path)
            unverified = set(pending_ids) - local_verified
            if unverified:
                print(f"\n  [WARN] Final: {len(unverified):,} ids not in local H5 — will retry.")
                pending_ids = [x for x in pending_ids if x not in unverified]

        expected_final = done_ids | set(pending_ids)
        sync_ok = _sync_to_drive(local_cache_path, cache_path, expected_final)
        if sync_ok:
            done_ids.update(pending_ids)
            done_ids.update(pending_missing)
            _write_checkpoint_atomic(ckpt_path, done_ids)
        else:
            print("[ERROR] Final Drive sync FAILED — run again to retry remaining images.")

    # ── Post-run report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Extraction done!  Saved this run: {saved_this_run:,}")
    drive_keys = _verify_h5_keys(cache_path)
    ckpt_ids   = _read_checkpoint(ckpt_path)
    print(f"  Drive H5  : {len(drive_keys):,} images")
    print(f"  Checkpoint: {len(ckpt_ids):,} ids")
    ghost = ckpt_ids - drive_keys
    miss  = drive_keys - ckpt_ids
    if ghost:
        print(f"  [warn] {len(ghost):,} checkpoint ids not in Drive H5 "
              f"(missing images — expected).")
    if miss:
        print(f"  [warn] {len(miss):,} Drive H5 keys not in checkpoint — fixing…")
        _write_checkpoint_atomic(ckpt_path, ckpt_ids | miss)
    if not ghost and not miss:
        print(f"  ✅ Drive H5 and checkpoint are perfectly consistent.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-extract CLIP image features")
    parser.add_argument("--data_root",      default="/content/drive/MyDrive/blip2_project/data")
    parser.add_argument("--output_dir",     default=None,
                        help="Drive output dir for H5 (default: {data_root}/cache).")
    parser.add_argument("--local_work_dir", default="/content/h5_cache",
                        help="LOCAL directory for H5 writes. Must NOT be on Drive. "
                             "Default: /content/h5_cache")
    parser.add_argument("--vqav2_dir",      default="vqav2")
    parser.add_argument("--coco_dir",       default="coco")
    parser.add_argument("--model",          default="openai/clip-vit-large-patch14")
    parser.add_argument("--split",          default="train", choices=["train", "val", "both"])
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--ckpt_every",     type=int, default=20,
                        help="Sync to Drive every N batches (default 20 ≈ 1,280 images "
                             "at batch_size=64). Larger = faster overall, less frequent saves.")
    parser.add_argument("--max_images",     type=int, default=None)
    parser.add_argument("--input_dir",      default=None)
    args = parser.parse_args()

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
            "image_size": 224,   "seed": 42, "batch_size": 32,
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
            local_work_dir=args.local_work_dir,
        )
