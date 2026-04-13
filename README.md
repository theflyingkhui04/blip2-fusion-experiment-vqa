# BLIP-2 Fusion Experiment — VQA

**Deep Learning Assignment** — Khám phá kiến trúc Q-Former của BLIP-2 và so sánh nhiều chiến lược fusion đa phương thức trên tập VQAv2.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Danh sách 7 thí nghiệm](#2-danh-sách-7-thí-nghiệm)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Chuẩn bị dữ liệu trên Google Drive](#4-chuẩn-bị-dữ-liệu-trên-google-drive)
5. [Hướng dẫn train trên Google Colab (từng bước)](#5-hướng-dẫn-train-trên-google-colab-từng-bước)
6. [Quản lý quá trình training bằng W&B](#6-quản-lý-quá-trình-training-bằng-wb)
7. [Đánh giá mô hình](#7-đánh-giá-mô-hình)
8. [Bảng so sánh kết quả](#8-bảng-so-sánh-kết-quả)

---

## 1. Giới thiệu

Dự án nghiên cứu Visual Question Answering (VQA) kết hợp:
- **Frozen CLIP ViT-L/14** — trích xuất đặc trưng ảnh (patch features `[B, 257, 1024]`)
- **Frozen BERT-base** — trích xuất đặc trưng câu hỏi (CLS token `[B, 768]`)
- **7 kiến trúc fusion** khác nhau để so sánh hiệu quả từ đơn giản → phức tạp

Tập dữ liệu: **VQAv2** — 443,757 câu hỏi train, 214,354 câu hỏi val.

---

## 2. Danh sách 7 thí nghiệm

| ID | Tên (`model.name`) | Mô tả | Tham số |
|----|-----|-------|---------|
| EXP-01 | `mean_linear` | Mean Pooling + Linear (baseline tuyến tính) | ~5.6M |
| EXP-02 | `concat_fusion` | Concat + MLP (1 lớp ẩn ReLU, fusion_dim=1024) | ~5M |
| EXP-03 | `mlb_fusion` | MLB — Hadamard Bilinear (fusion_dim=2048) | ~14M |
| EXP-04 | `mfb_fusion` | MFB — Factorized Bilinear (fusion_dim=1024, k=5) | ~13M |
| EXP-05 | `cross_attn_fusion` | Cross-Attention Bridge (3 lớp, 32 query tokens) | ~27M |
| EXP-06 | `qformer_scratch` | **Q-Former từ đầu** (12 lớp, 32 query tokens) | ~105M |
| EXP-07 | `perceiver_resampler` | Perceiver Resampler — Flamingo style (4 lớp, 64 latents) | ~58M |

> **EXP-06** là thí nghiệm chính — tái hiện kiến trúc Q-Former của BLIP-2 từ đầu.

---

## 3. Cấu trúc thư mục

```
blip2-fusion-experiment-vqa/
├── configs/
│   ├── contracts.py          # dataclass cho config validation
│   ├── default.yaml          # config mặc định (dùng cho local)
│   └── exp06.yaml            # config EXP-06 (đường dẫn Colab + Drive)
├── data/
│   ├── vqa_dataset.py        # VQAv2Dataset — đọc HDF5 cache
│   └── pre_extract_features.py  # trích xuất CLIP features → HDF5
├── models/
│   ├── exp01_mean_linear.py          # EXP-01: MeanLinearFusion
│   ├── exp02_concat_mlp.py           # EXP-02: ConcatMLPFusion
│   ├── exp03_mlb.py                  # EXP-03: MLBFusion (Hadamard Bilinear)
│   ├── exp04_mfb.py                  # EXP-04: MFBFusion (Factorized Bilinear)
│   ├── exp05_cross_attn.py           # EXP-05: CrossAttnFusion (Bridge)
│   ├── exp06_qformer_scratch.py      # EXP-06: QFormerScratch
│   ├── exp07_perceiver_resampler.py  # EXP-07: PerceiverResampler (Flamingo)
│   ├── blip2_vqa.py                  # BLIP-2 pretrained wrapper
│   ├── qformer.py                    # Q-Former core (dùng bởi EXP-06)
│   └── text_encoder.py               # FrozenTextEncoder (BERT-base)
├── training/
│   ├── trainer.py            # VQATrainer — vòng lặp train + checkpoint + W&B
│   └── losses.py
├── evaluation/
│   └── vqa_eval.py           # VQA accuracy metric
├── scripts/
│   ├── train.py              # script train chính (auto-resume + W&B)
│   └── evaluate.py           # script đánh giá
├── notebooks/
│   ├── eda_vqav2.ipynb       # EDA dữ liệu VQAv2
│   ├── pre-extracted + cached img.ipynb  # cache img to .H5 file
│   ├── Dataset + Dataloader.ipynb        # viết hàm dataset và hàm dataloader
│   └── test_ViT_pretrained.ipynb        # test model ViT
├── requirements.txt
├── experiment-cases.md       # tất cả các case để thử nghiệm
└── knowledge.md              # tài liệu kỹ thuật BLIP-2 + VQAv2
```

---

## 4. Chuẩn bị dữ liệu trên Google Drive

Tổ chức thư mục trong Google Drive theo cấu trúc sau:

```
MyDrive/VQAv2/
├── annotations/
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│   └── answer_list.json          ← danh sách 3129 câu trả lời
├── images/
│   ├── train2014/                ← ~83k ảnh (~13GB)
│   └── val2014/                  ← ~41k ảnh (~6GB)
└── features/                     ← HDF5 cache (tạo ở bước 5)
    ├── train_features.h5         ← ~7GB sau khi extract
    └── val_features.h5           ← ~3.5GB sau khi extract
```

> **Lưu ý:** Thư mục `features/` sẽ được tạo tự động khi chạy bước pre-extract.  
> Sau khi có HDF5 cache, pipeline **không cần ảnh gốc** nữa — tiết kiệm VRAM.

---

## 5. Hướng dẫn train trên Google Colab (từng bước)

### Bước 0: Mở Colab và chọn GPU

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU** (hoặc A100 nếu có Colab Pro)
3. Kiểm tra GPU:
   ```python
   !nvidia-smi
   ```

---

### Bước 1: Mount Google Drive và clone repo

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Clone repo vào Colab
%cd /content
!git clone https://github.com/<your-username>/blip2-fusion-experiment-vqa.git
%cd blip2-fusion-experiment-vqa
```

> Thay `<your-username>` bằng tên tài khoản GitHub của bạn.

---

### Bước 2: Cài đặt dependencies

```bash
!pip install -r requirements.txt -q
```

Kiểm tra các gói quan trọng đã cài đúng:

```python
import torch, transformers, wandb
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print(f"W&B: {wandb.__version__}")
```

---

### Bước 3: Đăng nhập W&B

```python
import wandb
wandb.login()
# Nhập API key khi được hỏi
# Lấy API key tại: https://wandb.ai/authorize
```

Hoặc dùng biến môi trường (không cần nhập tay):

```python
import os
os.environ["WANDB_API_KEY"] = "your_api_key_here"  # dán API key vào đây
```

---

### Bước 4: Kiểm tra và điều chỉnh config

```bash
!cat configs/exp06.yaml
```

Nếu đường dẫn Drive của bạn khác, sửa nhanh bằng Python:

```python
import yaml, pathlib

cfg_path = pathlib.Path("configs/exp06.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["data"]["data_root"]   = "/content/drive/MyDrive/VQAv2"
cfg["data"]["answer_list"] = "/content/drive/MyDrive/VQAv2/annotations/answer_list.json"
cfg["training"]["output_dir"] = "/content/drive/MyDrive/VQAv2/checkpoints/exp06"

cfg_path.write_text(yaml.dump(cfg, allow_unicode=True))
print("Config đã cập nhật.")
```

---

### Bước 5: Pre-extract CLIP features (chỉ cần làm 1 lần)

> **Bỏ qua bước này nếu đã có `train_features.h5` và `val_features.h5` trong Drive.**

```bash
# Extract cả train + val (~2-3h trên T4 GPU)
!python data/pre_extract_features.py \
    --split       both \
    --data_root   "/content/drive/MyDrive/blip2_project/data" \
    --output_dir  "/content/drive/MyDrive/blip2_project/data/cache" \
    --vqav2_dir   "vqav2" \
    --coco_dir    "coco"
```

**Tham số đầy đủ:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data_root` | *(bắt buộc)* | Thư mục gốc chứa ảnh COCO và annotation VQAv2 |
| `--output_dir` | `{data_root}/cache` | **Thư mục lưu HDF5** — override hoàn toàn `data_root/cache_dir`. Dùng khi muốn lưu sang ổ khác hoặc đường dẫn tùy ý |
| `--vqav2_dir` | `vqav2` | Tên thư mục JSON annotation bên trong `data_root` |
| `--coco_dir` | `coco` | Tên thư mục ảnh COCO bên trong `data_root` |
| `--model` | `openai/clip-vit-large-patch14` | HuggingFace model id của CLIP vision encoder |
| `--split` | `train` | `train` \| `val` \| `both` |
| `--batch_size` | `64` | Số ảnh mỗi forward pass (giảm nếu OOM) |
| `--ckpt_every` | `10` | Flush HDF5 + lưu `.ckpt.json` mỗi N batch (resume-safe) |
| `--max_images` | `None` | Giới hạn số ảnh — dùng để smoke-test nhanh (ví dụ `50`) |

**Ví dụ smoke-test (~30 giây) trước khi chạy full:**

```bash
!python data/pre_extract_features.py \
    --split       train \
    --data_root   "/content/drive/MyDrive/blip2_project/data" \
    --output_dir  "/content/drive/MyDrive/blip2_project/data/cache" \
    --max_images  50
```

Sau khi xong, kiểm tra file đã tạo:

```python
import h5py
with h5py.File("/content/drive/MyDrive/blip2_project/data/cache/train_features.h5", "r") as f:
    print(f"Số ảnh đã cache: {len(f.keys()):,}")
    sample_key = list(f.keys())[0]
    print(f"Shape mẫu [{sample_key}]: {f[sample_key].shape}")
    # Kỳ vọng: (257, 1024) — 1 CLS + 256 patch tokens × dim 1024
```

---

### Bước 6: Train EXP-06

```bash
# Train từ đầu, ví dụ EXP06
!python scripts/train.py \
    --config configs/exp06.yaml \
    --run_name "exp06_qformer_scratch_run1" \
    --data_root "/content/drive/MyDrive/blip2_project/data" \
    --vqav2_dir "vqav2" \
    --coco_dir  "coco" \
    --cache_dir "cache" \
    --answer_list "/content/drive/MyDrive/blip2_project/data/ans2idx.json" \
    --output_dir "/content/drive/MyDrive/blip2_project/data/checkpoints/exp06"
```

Script sẽ tự động:
- Khởi tạo W&B run với tên `exp06_qformer_scratch_run1`
- Lưu checkpoint sau mỗi epoch vào `output_dir`
- Log metrics lên W&B dashboard

**Output mẫu:**
```
[2024-01-15 10:30:00] INFO | Bắt đầu huấn luyện EXP-06 từ epoch 1/10
[2024-01-15 10:30:05] INFO | Epoch 1/10 | Step 100/13867 | Loss: 4.2315 | LR: 1.2e-05
...
[2024-01-15 11:45:00] INFO | Epoch 1 hoàn tất | Train Loss: 3.8421 | Val Acc: 42.31%
[2024-01-15 11:45:01] INFO | Checkpoint lưu → .../checkpoint_epoch_1.pth
```

---

### Bước 7: Resume sau khi Colab ngắt kết nối

Khi Colab bị disconnect, chỉ cần chạy lại với flag `--resume auto`:

```bash
!python scripts/train.py \
    --config configs/exp06.yaml \
    --resume auto \
    --run_name "exp06_qformer_scratch_run1"
```

Script tự động tìm checkpoint mới nhất trong `output_dir` và tiếp tục từ epoch đó.

> **Quan trọng:** Giữ nguyên `--run_name` để W&B nhận ra run ID và tiếp tục cùng một run, không tạo run mới.

Nếu muốn resume từ checkpoint cụ thể:

```bash
!python scripts/train.py \
    --config configs/exp06.yaml \
    --resume /content/drive/MyDrive/VQAv2/checkpoints/exp06/checkpoint_epoch_3.pth \
    --run_name "exp06_qformer_scratch_run1"
```

---

### Bước 8: Giữ kết nối Colab không bị timeout

Dán đoạn JavaScript sau vào **Console** của trình duyệt (F12 → Console):

```javascript
function ClickConnect(){
  console.log("Giữ kết nối Colab...");
  document.querySelector("#top-toolbar > colab-connect-button")
    .shadowRoot.querySelector("#connect").click();
}
setInterval(ClickConnect, 60000);
```

---

## 6. Quản lý quá trình training bằng W&B

### 6.1 Đăng ký tài khoản

1. Vào [wandb.ai](https://wandb.ai) → Sign up (miễn phí)
2. Vào [wandb.ai/authorize](https://wandb.ai/authorize) → copy API key
3. Dùng API key ở Bước 3 bên trên

### 6.2 Các metric được log

| Metric | Ý nghĩa | Kỳ vọng tốt |
|--------|---------|-------------|
| `train/loss` | Loss theo từng bước | Giảm đều |
| `train/lr` | Learning rate (warmup → cosine decay) | Tăng → giảm dần |
| `train/steps_per_sec` | Tốc độ xử lý | T4: ~8-12 steps/s |
| `train/loss_epoch` | Loss trung bình mỗi epoch | Giảm qua các epoch |
| `val/loss` | Loss trên tập val | Giảm theo epoch |
| `val/accuracy` | VQA accuracy (%) | EXP-06: mục tiêu ≥55% |

### 6.3 Xem Dashboard

Sau khi training bắt đầu, vào **[wandb.ai/home](https://wandb.ai/home)** → chọn project **`blip2-vqa-experiment`**:

- Tab **Charts**: biểu đồ loss + accuracy theo epoch/step
- Tab **System**: GPU utilization, memory usage trong Colab
- Tab **Overview**: hyperparameters, model config

### 6.4 So sánh nhiều run

1. Vào **Runs** trong project → chọn nhiều run → **Compare**
2. Tạo biểu đồ `val/accuracy` vs epoch cho tất cả 7 EXP
3. Tab **Table** → so sánh `best_val_metric` và `best_epoch`

### 6.5 Cài đặt Alerts

Vào **Settings → Notifications** trong W&B để nhận email khi:
- Run kết thúc (succeeded / failed)
- `val/accuracy` đạt ngưỡng cụ thể

### 6.6 Resume cùng W&B run sau khi Colab crash

Script `train.py` tự động lưu `wandb_run_id` vào `output_dir/wandb_run_id.txt`.  
Khi resume với cùng `--run_name`, script đọc file này và gọi:

```python
wandb.init(resume="allow", id=<saved_run_id>)
```

Kết quả: biểu đồ W&B **liên tục, không bị gián đoạn** dù Colab crash nhiều lần.

---

## 7. Đánh giá mô hình

```bash
!python scripts/evaluate.py \
    --config configs/exp06.yaml \
    --checkpoint /content/drive/MyDrive/VQAv2/checkpoints/exp06/best_model.pth
```

Kết quả được lưu vào `output_dir/eval_results.json`:

```json
{
  "overall_accuracy": 57.84,
  "yes_no_accuracy": 84.21,
  "number_accuracy": 46.37,
  "other_accuracy": 51.09,
  "num_samples": 214354
}
```

---

## 8. Bảng so sánh kết quả

> Điền kết quả sau khi train xong từng EXP.

| EXP | Mô hình | Params | Val Accuracy | Best Epoch | Ghi chú |
|-----|---------|--------|-------------|------------|---------|
| EXP-01 | mean_linear | ~5.6M | — | — | Baseline tuyến tính |
| EXP-02 | concat_fusion | ~5M | — | — | Concat + MLP |
| EXP-03 | mlb_fusion | ~14M | — | — | Hadamard Bilinear |
| EXP-04 | mfb_fusion | ~13M | — | — | Factorized Bilinear |
| EXP-05 | cross_attn_fusion | ~27M | — | — | Cross-Attn Bridge |
| EXP-06 | qformer_scratch | ~105M | — | — | Q-Former từ đầu |
| EXP-07 | perceiver_resampler | ~58M | — | — | Flamingo Perceiver |

---

## Tài liệu tham khảo

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597) — Bootstrapping Language-Image Pre-training
- [VQAv2 Dataset](https://visualqa.org/) — Making the V in VQA Matter
- [W&B Docs](https://docs.wandb.ai/) — Weights & Biases documentation
- `knowledge.md` — Tài liệu kỹ thuật chi tiết về BLIP-2 và VQAv2 trong repo này
