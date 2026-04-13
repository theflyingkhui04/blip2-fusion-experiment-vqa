# Báo cáo kỹ thuật: BLIP-2 Fusion Experiment — VQA

---

## 1. Tổng quan bài toán

### 1.1 Bài toán VQA (Visual Question Answering)

Bài toán nhận đầu vào là một ảnh và một câu hỏi bằng ngôn ngữ tự nhiên, yêu cầu mô hình trả lời câu hỏi đó dựa trên nội dung ảnh.

**Thiết kế trong project này là VQA dạng classification (đóng)** — không phải generative. Model không tạo ra văn bản tự do mà chọn 1 trong 3.129 câu trả lời định sẵn (closed-set). Đây là tiêu chuẩn đánh giá của benchmark VQAv2.

Ví dụ minh họa 3 loại câu hỏi trong VQAv2:

| Loại | Ảnh | Câu hỏi | Câu trả lời |
|---|---|---|---|
| Yes/No (~38%) | Người cầm ô | "Is the person holding an umbrella?" | **"yes"** |
| Number (~12%) | Bàn có 3 ghế | "How many chairs are at the table?" | **"3"** |
| Other (~50%) | Chó trên cỏ | "What color is the dog?" | **"brown"** |

Cơ chế dự đoán:
```
logits [B, 3129] → argmax → index 47 → tra bảng vocab → "brown"
```

Vocab 3.129 từ được build từ **tần suất câu trả lời** trong tập train — chỉ giữ các câu trả lời phổ biến nhất sau khi normalize.

---

### 1.2 Dataset: VQAv2 (COCO 2014)

| Tập | Số cặp QA | Số ảnh duy nhất |
|---|---|---|
| Train | 443.757 | 82.783 |
| Val | 214.354 | 40.504 |

Ảnh lấy từ COCO 2014. Mỗi câu hỏi có 10 người chú thích (annotator) trả lời độc lập — được dùng để tính soft-target score cho loss.

---

## 2. Kiến trúc tổng quan

Pipeline xử lý:

```
COCO image  →  CLIP ViT-L/14 (frozen)  →  [B, 257, 1024]  ─┐
                                                              ├─→  Fusion Module  →  logits [B, 3129]
Question    →  BERT tokenizer  →  BERT-base (frozen)  →  [B, 768]  ─┘
                                   (CLS token)
```

### 2.1 Vision Encoder — CLIP ViT-L/14 (frozen)

- Model: `openai/clip-vit-large-patch14`
- Input: ảnh RGB resize về 224×224
- Output: `last_hidden_state` — shape `[B, 257, 1024]`
  - 257 tokens = 1 CLS token + 256 patch tokens (grid 16×16, patch size 14px)
  - 1024 = hidden dim ViT-L/14
- **Hoàn toàn đóng băng (frozen)** — không cập nhật trọng số trong suốt training
- Không cần LLM vì đây là classification, không phải generation

### 2.2 Text Encoder — FrozenTextEncoder (BERT-base-uncased)

File: `models/text_encoder.py`

```python
class FrozenTextEncoder(nn.Module):
    OUTPUT_DIM: int = 768  # CLS token dimension

    # forward: input_ids [B, L] → CLS embedding [B, 768]
```

- Model: `bert-base-uncased` (110M params)
- Nhận `input_ids [B, 50]` + `attention_mask [B, 50]` → CLS token `[B, 768]`
- **Hoàn toàn đóng băng** (`requires_grad=False` cho toàn bộ params)
- `forward()` luôn chạy trong `torch.no_grad()` để tiết kiệm VRAM
- **Không dùng AMP autocast** cho BERT để tránh precision issues với BertLayerNorm
- BERT sống trong **trainer** (không phải trong từng EXP model) — chỉ load 1 lần dùng chung cho 7 thí nghiệm

---

## 3. Pre-extraction đặc trưng ảnh

### 3.1 Lý do

ViT-L/14 bị frozen hoàn toàn trong training. Nếu chạy forward pass ViT mỗi training step sẽ lãng phí GPU vì:
- ViT không được cập nhật → output luôn giống nhau với cùng ảnh đó
- Mỗi ảnh được xem nhiều lần qua nhiều epoch

Giải pháp: chạy ViT **một lần duy nhất**, lưu kết quả vào HDF5 cache, training chỉ đọc từ file.

### 3.2 Cấu trúc cache HDF5

File: `data/pre_extract_features.py`

```
train_features.h5:
  key: str(image_id)   ← ví dụ "391895"
  value: float16 array shape (257, 1024)

val_features.h5:
  (tương tự)
```

- Dtype: `float16` — tiết kiệm 50% so với float32
- Compression: `gzip level 1` — nhanh, tiết kiệm thêm ~30% dung lượng
- Mỗi ảnh là **1 dataset riêng** với key = `str(image_id)` → random access O(1) theo image_id
- **Không** lưu dạng array `(N, 257, 1024)` vì không cho phép truy cập ngẫu nhiên hiệu quả

Ước tính dung lượng:
| Split | Số ảnh | Raw float16 | Sau gzip |
|---|---|---|---|
| train2014 | 82.783 | ~43 GB | ~30 GB |
| val2014 | 40.504 | ~21 GB | ~15 GB |

### 3.3 Resume strategy & checkpoint

Phiên bản trước chỉ scan HDF5 keys để detect resume — chậm với cache lớn và mất data nếu crash. Đã nâng cấp thêm:

- **Checkpoint JSON** (`train_features.h5.ckpt.json`): ghi danh sách image_id đã xử lý
- **Flush HDF5** sau mỗi `checkpoint_interval` batch (mặc định 10 batch) → data được ghi xuống disk
- **Khi restart**: đọc JSON trước → merge với HDF5 keys → chỉ xử lý phần còn thiếu
- Guard `if str(iid) not in h5f` khi ghi → tránh lỗi duplicate key khi resume

```python
# Merge strategy
done_ids = set(ckpt_json["done"])          # từ JSON
done_ids |= {int(k) for k in h5f.keys()}  # từ HDF5 (safety net)
to_process = [iid for iid in image_ids if iid not in done_ids]
```

### 3.4 CLI và smoke-test

```bash
# Smoke-test: chỉ extract 100 ảnh để test luồng
python -m data.pre_extract_features --split both --batch_size 32 --max_images 100

# Production: toàn bộ dataset
python -m data.pre_extract_features --split both --batch_size 64 --ckpt_every 10
```

Tham số `--max_images N` cắt `image_ids[:N]` trước khi bắt đầu, toàn bộ logic checkpoint/resume vẫn chạy bình thường.

---

## 4. Dữ liệu — VQAv2Dataset

File: `data/vqa_dataset.py`

### 4.1 Chế độ hoạt động

Dataset hỗ trợ 2 chế độ:

| Chế độ | `use_cache` | Input ảnh | Dùng khi |
|---|---|---|---|
| Cache mode | `True` | `image_features [B, 257, 1024]` từ HDF5 | EXP-01 → EXP-07 |
| Raw mode | `False` | `pixel_values [B, 3, 224, 224]` từ JPEG | BLIP2VQA legacy |

### 4.2 Answer vocabulary

- Ưu tiên 1: đọc từ file `ans2idx.json` (dict `{answer: index}`)
- Ưu tiên 2: build dynamically từ annotation file (top-K frequent answers)
- **Lỗi đã fix**: file `ans2idx.json` là dict `{answer: index}` với index gốc non-contiguous (lên tới 3111), nhưng code cũ re-enumerate lại từ 0 làm sai mapping. Fix: giữ nguyên giá trị index từ dict, không re-enumerate.
- `_get_answer_scores()` luôn dùng `ANSWER_VOCAB_SIZE = 3129` làm kích thước tensor → khớp với model output

### 4.3 Partial cache handling

Khi cache chưa đầy đủ (đặc biệt sau smoke-test extraction), dataset **tự lọc samples** chỉ giữ những sample có `image_id` tồn tại trong HDF5, thay vì crash mid-training:

```python
with h5py.File(self._h5_path, "r") as _h5:
    cached_ids = set(_h5.keys())
before = len(self.samples)
self.samples = [s for s in self.samples if str(s["image_id"]) in cached_ids]
dropped = before - len(self.samples)
if dropped:
    print(f"WARNING: {dropped:,} samples dropped. Cache coverage: {len(self.samples):,}/{before:,}.")
```

### 4.4 Soft-target answer scores

Công thức VQAv2 chính thức: $\text{score}(a) = \min\!\left(\frac{\text{count}(a)}{3}, 1.0\right)$

Với mỗi sample, 10 annotators trả lời độc lập. Câu trả lời nào được ≥3 người chọn → score đủ 1.0. Dùng làm soft label cho BCE loss.

### 4.5 Stratified sampling

Khi dùng subset (ví dụ `train_size=50000`), dataset dùng **stratified sampling theo answer_type** (yes/no / number / other) để giữ phân phối cân bằng, thay vì random sampling đơn thuần.

---

## 5. Training pipeline

### 5.1 Cấu hình chung cho tất cả EXP

| Hyperparameter | Giá trị |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=0.05) |
| Learning rate | 1e-4, cosine decay → min 1e-6 |
| Warmup | 1.000 gradient steps |
| Gradient clip | max-norm = 1.0 |
| Num epochs | 10 |
| Mixed precision | fp16 AMP |
| Loss function | Soft-target Binary Cross-Entropy |

### 5.2 Loss function — Soft-target BCE

File: `training/losses.py`

Thay vì CrossEntropy với 1 label cứng, project dùng **BCE với soft target**:

$$\mathcal{L} = -\frac{1}{V}\sum_{v=1}^{V}\left[s_v \log\sigma(l_v) + (1-s_v)\log(1-\sigma(l_v))\right]$$

Với $s_v = \min(\text{count}_v/3, 1.0)$ là soft score của answer $v$. Cách này reward model khi dự đoán bất kỳ câu trả lời mà nhiều annotator đồng ý, không chỉ câu trả lời phổ biến nhất.

### 5.3 EXP Pipeline trong Trainer

File: `training/trainer.py`

```
Batch[image_features, input_ids, attention_mask, answer_scores]
         │                   │
         │            FrozenTextEncoder (BERT) — no_grad, no AMP
         │                   │
         │            text_features [B, 768]
         │                   │
         └──── EXP model(visual_features, text_features) ────→ logits [B, 3129]
                                                                       │
                                                             VQALoss(logits, answer_scores)
```

- BERT chạy **ngoài** AMP context manager (tránh BertLayerNorm precision issue)
- EXP model chạy **trong** AMP (`torch.cuda.amp.autocast`)
- `gradient_accumulation_steps`: tích lũy gradient N bước trước khi optimizer step — giữ effective batch size ổn định khi giảm batch_size do VRAM

### 5.4 Checkpoint

Lưu checkpoint sau mỗi epoch:
```python
{
    "epoch": int,
    "global_step": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_val_metric": float,
    "config": dict,
}
```

Auto-resume: truyền `--resume auto` → script tự tìm `checkpoint_epoch_*.pth` mới nhất trong `output_dir`.

### 5.5 Wandb logging

| Metric | Scope |
|---|---|
| `train/loss` | Mỗi `log_every` gradient steps |
| `train/lr` | Mỗi `log_every` gradient steps |
| `train/steps_per_sec` | Mỗi `log_every` gradient steps |
| `train/loss_epoch` | Mỗi epoch |
| `val/loss` | Mỗi epoch |
| `val/acc` | Mỗi epoch — VQA soft accuracy tổng |
| `val/acc_yesno` | Mỗi epoch — accuracy câu yes/no |
| `val/acc_number` | Mỗi epoch — accuracy câu đếm |
| `val/acc_other` | Mỗi epoch — accuracy câu mở |

---

## 6. Evaluation metric

### 6.1 VQA Accuracy (metric chính thức)

$$\text{acc}(q) = \min\!\left(\frac{\text{số annotator chọn đáp án đó}}{3},\ 1.0\right)$$

Tính trên toàn dataset rồi lấy trung bình. Xử lý sự mơ hồ của ngôn ngữ tự nhiên — 10 annotator có thể trả lời khác nhau, cần ≥3 người đồng ý mới đạt điểm tối đa.

Đây là metric duy nhất để so sánh với paper khác trên VQAv2 benchmark.

### 6.2 Per-type breakdown

Được tính trong cùng 1 validation pass bên trong `_val_epoch()`:

- **Soft accuracy** mỗi sample = `answer_scores[b, argmax(logits[b])]`
- Classify theo `answer_type` (yes/no / number / other) và tính mean riêng từng nhóm
- `number` accuracy thường thấp nhất và là điểm yếu của Q-Former — cần theo dõi để phân tích từng EXP

### 6.3 Các metric không dùng và lý do

| Metric | Lý do không phù hợp |
|---|---|
| BLEU / ROUGE | Dùng cho generation, không phải classification |
| F1 token overlap | Phù hợp cho extractive QA (SQuAD), không phải VQA |
| Perplexity | Chỉ cho language model |
| Hard accuracy (exact match) | Bỏ qua partial credit của annotator agreement |

---

## 7. Các thí nghiệm (7 EXP)

Tất cả EXP cùng dùng CLIP ViT-L/14 (frozen) + BERT-base (frozen). Chỉ khác nhau ở **fusion module** — phần duy nhất được train.

### EXP-01 — Mean Pooling + Linear (Baseline)

```
visual: [B, 257, 1024] → mean theo patch dim → [B, 1024]
text:   [B, 768]
fused:  concat [B, 1792] → Linear(1792, 3129) → logits
```

- Trainable params: ~5.7M
- Đây là điểm sàn tham chiếu — fusion yếu nhất có thể

### EXP-02 — Concat + MLP

```
visual: [B, 257, 1024] → mean → [B, 1024]
text:   [B, 768]
fused:  concat [B, 1792] → Linear(1792, 1024) → ReLU → Dropout → Linear(1024, 3129)
```

- Trainable params: ~7.2M
- Fusion phi tuyến đơn giản

### EXP-03 — MLB (Multi-modal Low-rank Bilinear)

Dựa trên: Kim et al., *Hadamard Product for Low-rank Bilinear Pooling*, ICLR 2017

```
visual: [B, 1024] → Linear(1024, 2048) → tanh → Dropout → [B, 2048]
text:   [B, 768]  → Linear(768, 2048)  → tanh → Dropout → [B, 2048]
fused:  v ⊙ t → [B, 2048] → MLP(2048, 1024, 3129)
```

- Tích Hadamard $v \odot t$ bắt được tương quan giữa hai modality mà concat/cộng bỏ qua
- Trainable params: ~10.5M

### EXP-04 — MFB (Multi-modal Factorized Bilinear)

Dựa trên: Zhou et al., *MFB and Co-Attention for VQA*, ICCV 2017

```
visual: [B, 1024] → Linear(1024, k×d) → [B, 5120]   (k=5, d=1024)
text:   [B, 768]  → Linear(768, k×d)  → [B, 5120]
fused:  v ⊙ t → reshape [B, 5, 1024] → sum theo k → [B, 1024] → L2-norm → MLP(1024, 1024, 3129)
```

- Phân tích bilinear thành k=5 nhân tố → biểu diễn ổn định hơn MLB
- Trainable params: ~17M

### EXP-05 — Query Cross-Attention Bridge

```
queries:  [B, 32, 768]   (learnable query tokens)
visual:   [B, 257, 1024] → Linear(1024, 768) → [B, 257, 768]
text:     [B, 768]

Lặp 3 lớp:
  queries = LayerNorm(queries)
  queries += MultiHeadCrossAttn(Q=queries, K=visual, V=visual)
  queries += FFN(LayerNorm(queries))

pooled = queries.mean(1) → [B, 768]
fused  = concat(pooled, text) [B, 1536] → MLP(1536, 1024, 3129)
```

- Khai thác được thông tin **không gian** của patch tokens — các EXP trước chỉ dùng mean-pool
- Trainable params: ~22M
- Cần toàn bộ patch tokens [B, 257, 1024] — không dùng được CLS-only cache

### EXP-06 — Q-Former from Scratch

Dựa trên: Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training*, ICML 2023

```
queries:  [B, 32, 768]   (learnable)
visual:   [B, 257, 1024] → visual_proj → visual_norm → [B, 257, 768]

12 khối QFormerLayer:
  lớp chẵn (0,2,4,6,8,10):  self-attn(Q↔Q) → cross-attn(Q←V) → FFN
  lớp lẻ  (1,3,5,7,9,11):   self-attn(Q↔Q) → FFN

output: [B, 32, 768] → mean pool → [B, 768]
fused:  concat(pooled, text_cls) [B, 1536] → MLP(1536, 1024, 3129)
```

- Queries vừa **self-attend** với nhau (học inter-query relation), vừa **cross-attend** visual mỗi 2 lớp
- Train từ đầu (không load pretrained BLIP-2 weights)
- Trainable params: ~190M — tốn VRAM nhất
- `QFormerConfig.vision_width = 1024` (CLIP ViT-L/14), không phải default 1408 của BLIP-2 paper

### EXP-07 — Perceiver Resampler

Dựa trên: Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022

```
latents: [B, 64, 768]   (learnable, M=64 latent vectors)
visual:  [B, 257, 1024] → Linear(1024, 768) → [B, 257, 768]
text:    [B, 768]

4 lớp Perceiver:
  x = concat(latents, visual) [B, 321, 768]
  latents += MultiHeadCrossAttn(Q=latents, K=x, V=x)
  latents += FFN(LayerNorm(latents))

pooled = latents.mean(1) → [B, 768]
fused  = concat(pooled, text) [B, 1536] → MLP(1536, 1024, 3129)
```

- Latent attend vào **combined context** (latents + visual) thay vì chỉ visual cross-attend
- Trainable params: ~58M — hiệu quả tham số hơn Q-Former

---

## 8. Cấu trúc dữ liệu & contracts

File: `configs/contracts.py` — single source of truth cho mọi tensor shape và dict key.

### Batch keys chính

| Key | Type | Shape | Mô tả |
|---|---|---|---|
| `image_features` | Tensor | `[B, 257, 1024]` | CLIP features từ HDF5 cache |
| `pixel_values` | Tensor | `[B, 3, 224, 224]` | Raw ảnh (legacy mode) |
| `input_ids` | Tensor | `[B, 50]` | BERT tokenized question |
| `attention_mask` | Tensor | `[B, 50]` | Padding mask |
| `answer_scores` | Tensor | `[B, 3129]` | Soft VQA targets |
| `answer_label` | Tensor | `[B]` | Hard label index |
| `answer_type` | List[str] | `B` | "yes/no" / "number" / "other" |

### Runtime constants

| Constant | Giá trị | Ý nghĩa |
|---|---|---|
| `ANSWER_VOCAB_SIZE` | 3129 | Số lớp phân loại |
| `IMAGE_SIZE` | 224 | Kích thước ảnh input CLIP |
| `CLIP_PATCH_TOKENS` | 257 | 1 CLS + 256 patches |
| `CLIP_FEATURE_DIM` | 1024 | Hidden dim ViT-L/14 |
| `QFORMER_HIDDEN_SIZE` | 768 | Hidden dim BERT/Q-Former |
| `NUM_QUERY_TOKENS` | 32 | Số query tokens Q-Former |
| `MAX_QUESTION_LENGTH` | 50 | Max token câu hỏi |
| `VQA_SCORE_DENOMINATOR` | 3 | Mẫu soft-target: count/3 |

---

## 9. Scripts

### train.py

```bash
python scripts/train.py \
    --config configs/exp06.yaml \
    --run_name "exp06_run1" \
    --data_root "/content/drive/MyDrive/blip2_project/data" \
    --vqav2_dir "vqav2" \
    --coco_dir  "coco" \
    --cache_dir "cache" \
    --answer_list "/content/.../ans2idx.json" \
    --output_dir "/content/.../checkpoints/exp06" \
    --num_epochs 10 \
    --batch_size 8
```

CLI args path override (ưu tiên hơn YAML):

| Arg | Override config key |
|---|---|
| `--data_root` | `config.data.data_root` |
| `--vqav2_dir` | `config.data.vqav2_dir` |
| `--coco_dir` | `config.data.coco_dir` |
| `--cache_dir` | `config.data.cache_dir` |
| `--answer_list` | `config.data.answer_list` |
| `--output_dir` | `config.training.output_dir` |
| `--num_epochs` | `config.training.num_epochs` |
| `--batch_size` | `config.data.batch_size` |
| `--resume` | `auto` hoặc đường dẫn checkpoint |

### Quy trình đầy đủ

```bash
# Bước 1: Pre-extract features (chạy 1 lần)
python -m data.pre_extract_features --split both --batch_size 64

# Bước 2: Train
python scripts/train.py --config configs/exp06.yaml [args...]

# Bước 3: Evaluate
python scripts/evaluate.py --config configs/exp06.yaml \
    --checkpoint .../best_model.pth --split val
```

---

## 10. Các bug đã phát hiện và fix

| Bug | Nguyên nhân | Fix |
|---|---|---|
| `FileNotFoundError` annotation JSON | `vqav2_dir: annotations` sai, đúng phải là `vqav2` | Sửa `configs/exp06.yaml` |
| `KeyError: "features"` trong HDF5 | Code dùng `f["features"]` nhưng HDF5 lưu theo key = `str(image_id)` | Dùng `f[sample_key][:]` |
| `KeyError: image_id not in cache` khi training | Cache partial nhưng dataset load toàn bộ samples | Filter samples theo cached_ids khi `use_cache=True` |
| `IndexError: index 3111 out of bounds for size 3107` | `ans2idx.json` là dict với index gốc, code cũ re-enumerate về 0..3106 | Giữ nguyên index từ dict, dùng `ANSWER_VOCAB_SIZE=3129` cho tensor size |
| `FutureWarning: GradScaler deprecated` | API cũ `torch.cuda.amp.GradScaler` | Cập nhật `torch.amp.GradScaler('cuda', ...)` |
| Không có VQA accuracy trong log | `eval_metric_fn` không được truyền vào trainer | Tính soft VQA accuracy trực tiếp trong `_val_epoch()` |
| `val_acc` không phân tích theo loại câu hỏi | Chỉ log loss | Thêm per-type breakdown (yes/no / number / other) vào W&B |

---

## 11. Sau khi train xong

1. **Chạy evaluation chính thức** trên val set lấy VQA accuracy cuối cùng
2. **So sánh 7 EXP** theo bảng:
   - `val/acc` tổng — xem fusion nào tốt nhất
   - `val/acc_number` — thường thấp nhất, phân tích điểm yếu
   - `val/acc_yesno` — thường cao nhất, ít phân biệt được giữa các EXP
3. **Viết phân tích**: EXP-06 Q-Former có ~190M trainable params — liệu có overfitting trên subset nhỏ không? So sánh cost-accuracy tradeoff với EXP-05 và EXP-07.
