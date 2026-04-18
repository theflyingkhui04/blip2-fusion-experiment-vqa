"""Trích xuất trước các đặc trưng hình ảnh ViT-L/14 (CLIP) cho tất cả các hình ảnh COCO 2014.
Đầu ra: data/cache/train_features.h5 và val_features.h5
Khóa: str(image_id), Giá trị: mảng float16 (257, 1024)
Chiến lược ghi an toàn (Colab + Google Drive FUSE)
======================================================
Ghi trực tiếp HDF5 vào Drive FUSE là KHÔNG AN TOÀN: HDF5 thực hiện ghi truy cập ngẫu nhiên
(superblock ở byte 0 + dữ liệu ở cuối). Drive FUSE có thể ghi các thao tác này không theo thứ tự, dẫn đến superblock bị lỗi khi chương trình dừng hoạt động.

Chiến lược an toàn được sử dụng ở đây:
1. TẤT CẢ các thao tác ghi H5 đều được ghi vào thư mục CỤC BỘ (/content/h5_cache) (mặc định).
Ổ đĩa cục bộ là SSD tốc độ cao và tuân thủ POSIX — không liên quan đến FUSE.

2. Tại mỗi khoảng thời gian kiểm tra:
a. ĐÓNG file H5 cục bộ (đảm bảo metadata + data đều flush xuống disk).
b. Xác minh H5 cục bộ có tất cả các khóa dự kiến.
c. Sao chép H5 cục bộ → Drive (ghi chunked → fsync → atomic rename).
d. Xác minh kích thước file + số lượng key trên Drive.
e. Ghi JSON điểm kiểm tra vào Drive (đổi tên nguyên tử).
f. Mở lại H5 cục bộ để tiếp tục ghi.

3. Nếu việc sao chép Drive thất bại ở bất kỳ bước nào, hãy bỏ qua việc cập nhật điểm kiểm tra và
thử lại ở khoảng thời gian tiếp theo — không mất dữ liệu.

Chiến lược tiếp tục:
- Khi bắt đầu: sao chép Drive H5 → cục bộ (nếu tồn tại và hợp lệ).
- Nếu Drive H5 bị hỏng: cảnh báo, bỏ qua, bắt đầu lại cục bộ.
- JSON điểm kiểm tra + Drive H5 luôn nhất quán.

Ghi chú:
- KHÔNG gọi text_encoder.py — BERT chạy bên trong trình huấn luyện tại
thời điểm huấn luyện; trích xuất trước chỉ xử lý ViT bị đóng băng.
- Sử dụng VQAv2Dataset.SPLIT_FILES dùng để lấy siêu dữ liệu tệp/thư mục.
- Truyền max_images để giới hạn số lượng ảnh (hữu ích cho các bài kiểm tra sơ bộ).
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
    """Copy local H5 → Drive H5 via chunked write + atomic rename.

    Fixes cho Google Drive FUSE:
    1. Ghi vào file TẠM trên Drive (tránh overwrite partial).
    2. Chunked read/write (64 MB) — tránh FUSE buffering issues.
    3. fsync file tạm để buộc FUSE flush xuống storage.
    4. os.replace() atomic để swap tmp → final.
    5. os.sync() buộc TẤT CẢ FUSE pending writes flush xuống storage.
    6. Verify kích thước file (bắt lỗi truncated upload mà FUSE cache giấu).
    7. Verify H5 key count.
    """
    local_size = os.path.getsize(local_path)
    tmp_drive = drive_path + ".tmp_sync"

    try:
        print(f"\n  Syncing to Drive ({local_size / 1e6:.1f} MB)…", end=" ", flush=True)

        # Step 1: Chunked copy → temp file on Drive
        chunk_size = 64 * 1024 * 1024  # 64 MB
        with open(local_path, "rb") as fsrc, open(tmp_drive, "wb") as fdst:
            while True:
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break
                fdst.write(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())

        # Step 2: Verify temp file size before rename
        tmp_size = os.path.getsize(tmp_drive)
        if tmp_size != local_size:
            print(f"WARN — tmp size mismatch! local={local_size:,} tmp={tmp_size:,}")
            try:
                os.unlink(tmp_drive)
            except OSError:
                pass
            return False

        # Step 3: Atomic rename tmp → final
        os.replace(tmp_drive, drive_path)

        # Step 4: Force ALL pending FUSE writes to storage
        try:
            os.sync()
        except AttributeError:
            pass  # os.sync() not available on this platform (e.g. Windows)

        # Step 5: Re-open and fsync the final file to be absolutely sure
        try:
            with open(drive_path, "rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass

        # Step 6: File size verification (catches truncated uploads
        # that FUSE page cache hides from h5py reads)
        drive_size = os.path.getsize(drive_path)
        if drive_size != local_size:
            print(f"WARN — drive size mismatch! "
                  f"local={local_size:,} drive={drive_size:,}")
            return False

        # Step 7: H5 key verification
        drive_keys = _verify_h5_keys(drive_path)
        if len(must_have_ids) > 0 and len(drive_keys) < len(must_have_ids):
            missing = must_have_ids - drive_keys
            print(f"WARN — {len(missing):,} keys missing "
                  f"(expected≥{len(must_have_ids):,}, got {len(drive_keys):,})")
            return False

        print(f"OK ({len(drive_keys):,} keys, "
              f"{drive_size / 1e6:.1f} MB on Drive) ✅")
        return True

    except Exception as exc:
        print(f"FAILED: {exc}")
        try:
            os.unlink(tmp_drive)
        except OSError:
            pass
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

    # NOTE: Không dùng `with h5py.File(...)` vì cần ĐÓNG file trước mỗi lần
    # sync lên Drive. h5py.flush() KHÔNG đảm bảo file consistent trên disk —
    # chỉ có close() mới đảm bảo metadata HDF5 (superblock, B-tree) được ghi
    # đầy đủ. Đây là nguyên nhân shutil.copy2 tạo ra file bị truncate.
    h5f = h5py.File(local_cache_path, "a")
    try:
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

            # ── Checkpoint: close H5 → verify → Drive copy → JSON → reopen ──
            if (batch_num + 1) % checkpoint_interval == 0:
                # CLOSE H5 to finalize ALL internal metadata + data blocks.
                # h5py.flush() alone does NOT guarantee a consistent file —
                # only close() writes the final superblock + B-tree indices.
                h5f.close()

                # Verify local H5 (file is now fully consistent on disk)
                local_verified = _verify_h5_keys(local_cache_path)
                unverified_local = set(pending_ids) - local_verified
                if unverified_local:
                    print(f"\n  [WARN] {len(unverified_local):,} ids missing in local H5 "
                          f"— will retry.")
                    pending_ids = [x for x in pending_ids if x not in unverified_local]

                # Sync local → Drive.
                # FIX BUG: Dùng local_verified (actual H5 keys) để verify,
                # KHÔNG dùng done_ids vì done_ids chứa ghost IDs (missing
                # images) — những ID này không bao giờ nằm trong file H5.
                sync_ok = _sync_to_drive(
                    local_cache_path, cache_path, local_verified
                )

                if sync_ok:
                    done_ids.update(pending_ids)
                    done_ids.update(pending_missing)
                    pending_ids.clear()
                    pending_missing.clear()
                    _write_checkpoint_atomic(ckpt_path, done_ids)
                else:
                    print(f"  Checkpoint skipped — will retry at next interval.")

                pbar.set_postfix(saved=saved_this_run, on_drive=len(done_ids))

                # Reopen H5 for the next batch of writes
                h5f = h5py.File(local_cache_path, "a")

    finally:
        # Ensure H5 file is always closed, even on unhandled exceptions
        try:
            if h5f.id.valid:
                h5f.close()
        except Exception:
            pass

    # ── Final sync (H5 is closed — file is fully consistent) ──────────────
    local_verified = _verify_h5_keys(local_cache_path)

    if pending_ids or pending_missing:
        if pending_ids:
            unverified = set(pending_ids) - local_verified
            if unverified:
                print(f"\n  [WARN] Final: {len(unverified):,} ids not in local H5 — dropping.")
                pending_ids = [x for x in pending_ids if x not in unverified]

        # FIX BUG: Dùng local_verified thay vì done_ids | set(pending_ids)
        sync_ok = _sync_to_drive(local_cache_path, cache_path, local_verified)
        if sync_ok:
            done_ids.update(pending_ids)
            done_ids.update(pending_missing)
            _write_checkpoint_atomic(ckpt_path, done_ids)
        else:
            print("[ERROR] Final Drive sync FAILED — run again to retry remaining images.")
    else:
        print("  All data synced at last checkpoint — no final sync needed.")

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
