# Các trường hợp thí nghiệm — BLIP-2 Fusion Baselines cho VQAv2

## 1. Cấu hình chung

| Tham số | Giá trị |
|---|---|
| Dataset | VQAv2 (COCO 2014) |
| Mẫu train | 443.757 cặp QA · 82.783 ảnh duy nhất |
| Mẫu val | 214.354 cặp QA · 40.504 ảnh duy nhất |
| Bộ từ vựng câu trả lời | 3.129 lớp |
| Vision encoder | CLIP ViT-L/14 (đóng băng) — output: `[B, 257, 1024]` (1 CLS + 256 patch) |
| Text encoder | BERT-base tokenizer + embedding (đóng băng) — CLS: `[B, 768]` |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 1e-4 với cosine decay, min 1e-6 |
| Warmup | 1.000 bước |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Số epoch | 10 |
| Độ chính xác hỗn hợp | fp16 (AMP) |
| Hàm loss | Soft-target BCE (điểm đồng thuận người chú thích VQAv2) |
| Metric đánh giá | VQAv2 accuracy |

### Quy đổi compute unit (CU)
| Phần cứng | CU/giờ (Colab xấp xỉ) |
|---|---|
| T4 (16 GB VRAM) | ~1 CU/giờ |
| A100 (40 GB VRAM) | ~4 CU/giờ |

---

## 2. Bước 0 — Trích xuất đặc trưng (thực hiện một lần, dùng chung cho tất cả thí nghiệm)

Chạy CLIP ViT-L/14 trước trên toàn bộ 122k ảnh COCO 2014 và lưu cache vào file HDF5.

### Dữ liệu được lưu

| Chế độ | Shape mỗi ảnh | Kích thước file (float16) |
|---|---|---|
| Chỉ CLS token | `[1, 1024]` | ~0.24 GB |
| Toàn bộ patch | `[257, 1024]` | ~63 GB |

> **Khuyến nghị:** Cache chỉ CLS token cho EXP-01 đến EXP-04. Với EXP-05 đến EXP-07, nên tính on-the-fly hoặc cache lên Google Drive nếu đủ dung lượng.

### Ước tính tài nguyên

| | Batch size | Thời gian | CU |
|---|---|---|---|
| T4 | 64 | 12–18 phút | ~0.3 CU |
| A100 | 128 | 3–6 phút | ~0.4 CU |

---

## 3. Các thí nghiệm

---

### EXP-01 — Mean Pooling + Linear

**Giả thuyết:** Một lớp Linear đơn lẻ đặt trên visual features đã mean-pool kết hợp với CLS text là fusion yếu nhất có thể — dùng làm baseline tham chiếu điểm sàn.

**Kiến trúc:**
```
visual: [B, 257, 1024] → mean theo patch → [B, 1024]
text:   [B, 768]       → CLS token
fused:  concat → [B, 1792] → Linear(1792, 3129) → logits
```

**Tham số có thể train:** ~5.7 M (chỉ lớp linear)  
**Batch size:** 32 (T4) / 64 (A100)  
**Nguồn đầu vào:** Cache CLS-only (rất nhỏ)

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 12–18 phút | 2–3 giờ | **2–3 CU** |
| A100 | 3–5 phút | 30–50 phút | **2–3.5 CU** |

> Lưu ý: A100 tốn CU/giờ cao hơn nhưng xong nhanh hơn nhiều — tổng CU tương đương với model nhẹ.

---

### EXP-02 — Concat + MLP

**Giả thuyết:** Fusion phi tuyến (MLP) trên features pooled ghép lại sẽ tốt hơn lớp linear đơn.

**Kiến trúc:**
```
visual: [B, 257, 1024] → mean → [B, 1024]
text:   [B, 768]
fused:  concat → [B, 1792] → Linear(1792, 1024) → ReLU → Dropout → Linear(1024, 3129)
```

**Tham số có thể train:** ~7.2 M  
**Batch size:** 32 (T4) / 64 (A100)  
**Nguồn đầu vào:** Cache CLS-only

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 15–22 phút | 2.5–3.7 giờ | **2.5–4 CU** |
| A100 | 4–7 phút | 40–70 phút | **2.7–4.7 CU** |

---

### EXP-03 — MLB (Multi-modal Low-rank Bilinear)

**Bài báo:** Kim et al., *Hadamard Product for Low-rank Bilinear Pooling*, ICLR 2017.

**Giả thuyết:** Tương tác bilinear qua tích Hadamard bắt được tương quan giữa hai modaliti mà concat và cộng đơn thuần bỏ qua.

**Kiến trúc:**
```
visual: [B, 1024] → Linear(1024, 2048) → tanh → Dropout → [B, 2048]
text:   [B, 768]  → Linear(768,  2048) → tanh → Dropout → [B, 2048]
fused:  v ⊙ t  →  [B, 2048] → MLP(2048, 1024, 3129)
```

**Tham số có thể train:** ~10.5 M  
**Batch size:** 32 (T4) / 64 (A100)  
**Nguồn đầu vào:** Cache CLS-only

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 18–28 phút | 3–4.7 giờ | **3–5 CU** |
| A100 | 5–9 phút | 50–90 phút | **3.3–6 CU** |

---

### EXP-04 — MFB (Multi-modal Factorized Bilinear)

**Bài báo:** Zhou et al., *MFB and Co-Attention for VQA*, ICCV 2017.

**Giả thuyết:** Phân tích tương tác bilinear thành k=5 nhân tố, sau đó sum-pool + L2-norm tạo ra biểu diễn phong phú và ổn định hơn MLB.

**Kiến trúc:**
```
visual: [B, 1024] → Linear(1024, k×d) → [B, k×d]
text:   [B, 768]  → Linear(768,  k×d) → [B, k×d]
  với k=5, d=1024  →  proj_dim = 5120

fused:  v ⊙ t → reshape → [B, k, d] → sum theo k → [B, 1024] → L2-norm
→ MLP(1024, 1024, 3129)
```

**Tham số có thể train:** ~17 M  
**Batch size:** 32 (T4) / 64 (A100)  
**Nguồn đầu vào:** Cache CLS-only

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 22–35 phút | 3.7–5.8 giờ | **3.7–6 CU** |
| A100 | 6–11 phút | 60–110 phút | **4–7.3 CU** |

---

### EXP-05 — Query Cross-Attention Bridge

**Giả thuyết:** Các query token có thể học được, trực tiếp attend vào toàn bộ patch token thị giác (thay vì chỉ CLS pooled), sẽ khai thác được thông tin không gian ảnh phong phú hơn.

**Kiến trúc:**
```
queries:  [B, 32, 768]   (learnable, mở rộng từ [1, 32, 768])
visual:   [B, 257, 1024] → Linear(1024, 768) → [B, 257, 768]
text:     [B, 768]

Lặp qua 3 lớp cross-attention:
  queries = LayerNorm(queries)
  queries = queries + MultiHeadCrossAttn(Q=queries, K=visual, V=visual)
  queries = queries + FFN(LayerNorm(queries))

pooled = queries.mean(dim=1) → [B, 768]
fused  = concat(pooled, text) → [B, 1536] → MLP(1536, 1024, 3129)
```

**Tham số có thể train:** ~22 M  
**Batch size:** 16 (T4, giới hạn VRAM) / 32 (A100)  
**Nguồn đầu vào:** Toàn bộ patch token — cache hoặc on-the-fly ViT forward

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 45–70 phút | 7.5–12 giờ | **7.5–12 CU** |
| A100 | 12–20 phút | 2–3.3 giờ | **8–13 CU** |

> ⚠️ T4: giảm batch xuống 16, bật `gradient_accumulation_steps=2` để giữ effective batch size = 32.  
> ⚠️ Cache đầy đủ patch (63 GB) không vừa ổ đĩa Colab — ưu tiên dùng on-the-fly ViT inference.

---

### EXP-06 — Q-Former from Scratch

**Bài báo:** Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training...*, ICML 2023.

**Giả thuyết:** Q-Former đầy đủ với 12 lớp transformer và cross-attention xen kẽ (mỗi 2 lớp) cung cấp sự căn chỉnh visual-language mạnh nhất — đây là "chuẩn vàng" trong bộ thí nghiệm này.

**Kiến trúc:** Xem `models/qformer.py`
```
queries:  [B, 32, 768]   (learnable)
visual:   [B, 257, 1024] → visual_proj → visual_norm → [B, 257, 768]
text:     [B, 768]

12 khối QFormerLayer:
  lớp chẵn (0,2,4,...): self-attn(queries) → cross-attn(queries→visual) → FFN
  lớp lẻ  (1,3,5,...): self-attn(queries) → FFN

output: [B, 32, 768] → mean pool → [B, 768]
classifier: MLP(768+768, 1024, 3129)   (concat với text)
```

**Tham số có thể train:** ~190 M  
**Batch size:** 8 (T4, giới hạn VRAM) / 24 (A100)  
**`gradient_accumulation_steps`:** 4 (T4) / 2 (A100) → effective batch 32  
**Nguồn đầu vào:** Toàn bộ patch token — on-the-fly ViT forward được khuyến nghị mạnh

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 90–130 phút | 15–22 giờ | **15–22 CU** |
| A100 | 22–35 phút | 3.7–5.8 giờ | **14.8–23 CU** |

> ⚠️ Thí nghiệm tốn tài nguyên nhất. Ưu tiên chạy trên A100 nếu còn CU.  
> ⚠️ T4 VRAM: bắt buộc batch_size=8. Theo dõi bằng `nvidia-smi`.  
> ⚠️ Bật checkpoint mỗi epoch — session Colab có thể bị ngắt.

---

### EXP-07 — Perceiver Resampler

**Bài báo:** Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.

**Giả thuyết:** Kiến trúc Perceiver Resampler — latent query attend vào patch ảnh qua cross-attention — hiệu quả tham số hơn Q-Former trong khi đạt năng lực biểu diễn tương đương.

**Kiến trúc:**
```
latents: [B, 64, 768]   (learnable, M=64 vector latent)
visual:  [B, 257, 1024] → Linear(1024, 768) → [B, 257, 768]
text:    [B, 768]

Lặp qua 4 lớp Perceiver:
  latents = LayerNorm(latents)
  x = concat(latents, visual) → [B, 257+64, 768]   # visual làm context
  latents = latents + MultiHeadCrossAttn(Q=latents, K=x, V=x)
  latents = latents + FFN(LayerNorm(latents))

pooled = latents.mean(dim=1) → [B, 768]
fused  = concat(pooled, text) → [B, 1536] → MLP(1536, 1024, 3129)
```

**Tham số có thể train:** ~58 M  
**Batch size:** 12 (T4) / 32 (A100)  
**`gradient_accumulation_steps`:** 3 (T4) → effective batch 36  
**Nguồn đầu vào:** Toàn bộ patch token — on-the-fly ViT forward được khuyến nghị

| | Thời gian / epoch | Tổng (10 epoch) | CU |
|---|---|---|---|
| T4 | 55–85 phút | 9–14 giờ | **9–14 CU** |
| A100 | 14–22 phút | 2.3–3.7 giờ | **9–15 CU** |

> ⚠️ T4: chạy với batch=12, accum=3.

---

## 4. Bảng tổng hợp

| Mã | Mô hình | Params | Đầu vào | Tổng T4 | CU T4 | Tổng A100 | CU A100 |
|---|---|---|---|---|---|---|---|
| EXP-01 | Mean Pool + Linear | 5.7 M | pooled | 2–3 giờ | 2–3 | 30–50 phút | 2–3.5 |
| EXP-02 | Concat + MLP | 7.2 M | pooled | 2.5–3.7 giờ | 2.5–4 | 40–70 phút | 2.7–4.7 |
| EXP-03 | MLB | 10.5 M | pooled | 3–4.7 giờ | 3–5 | 50–90 phút | 3.3–6 |
| EXP-04 | MFB | 17 M | pooled | 3.7–5.8 giờ | 3.7–6 | 60–110 phút | 4–7.3 |
| EXP-05 | Cross-Attn Bridge | 22 M | patch | 7.5–12 giờ | 7.5–12 | 2–3.3 giờ | 8–13 |
| EXP-06 | Q-Former từ đầu | 190 M | patch | 15–22 giờ | 15–22 | 3.7–5.8 giờ | 14.8–23 |
| EXP-07 | Perceiver Resampler | 58 M | patch | 9–14 giờ | 9–14 | 2.3–3.7 giờ | 9–15 |
| Bước 0 | Trích xuất đặc trưng | — | ảnh | 12–18 phút | 0.3 | 3–6 phút | 0.4 |
| | **TỔNG** | | | **~43–65 giờ** | **~43–66 CU** | **~11–17 giờ** | **~44–72 CU** |

> Tổng CU giữa T4 và A100 gần tương đương với cùng khối lượng. Sự khác biệt chính là thời gian thực tế: A100 nhanh hơn 3–5 lần.
>
> **Khuyến nghị:** Dùng T4 cho EXP-01 đến EXP-04 (chạy qua đêm nếu cần). Chỉ dùng A100 cho EXP-05 đến EXP-07 để tiết kiệm thời gian.

### Chiến lược phân bổ tài nguyên được đề xuất

| Phần cứng | Thí nghiệm | CU ước tính |
|---|---|---|
| T4 | EXP-01, 02, 03, 04 + Bước 0 | ~12–18 CU |
| A100 | EXP-05, 06, 07 | ~32–51 CU |
| **Tổng** | | **~44–69 CU** |

---

## 5. Thứ tự thực hiện

```
Bước 0  → Trích xuất đặc trưng (cache CLS cho EXP-01–04, on-the-fly cho EXP-05–07)

Đợt 1 (T4, model dùng pooled features — nhẹ, chạy tuần tự):
  EXP-01 → EXP-02 → EXP-03 → EXP-04

Đợt 2 (A100, model dùng patch-level — nặng):
  EXP-07 → EXP-05 → EXP-06
  (Perceiver trước — tốn vừa phải; Q-Former cuối — tốn nhất)
```

---

## 6. Yêu cầu lưu trữ

| Tài nguyên | Kích thước |
|---|---|
| Ảnh COCO train2014 | ~13 GB |
| Ảnh COCO val2014 | ~6 GB |
| Annotation VQAv2 (tất cả split) | ~0.5 GB |
| Cache CLS-only | ~0.24 GB |
| Cache đầy đủ patch (tuỳ chọn) | ~63 GB |
| Checkpoint (7 model × 10 epoch × ~200–800 MB) | ~15–60 GB |

> Cache đầy đủ patch không thực tế trên `/tmp` của Colab (tối đa ~78 GB). Nên stream từ Google Drive hoặc tính on-the-fly.

---

## 7. Ngân sách VRAM theo model

| Model | Batch tối thiểu | VRAM (fp16) | Ghi chú |
|---|---|---|---|
| EXP-01 | 32 | ~1 GB | Không vấn đề |
| EXP-02 | 32 | ~1.2 GB | Không vấn đề |
| EXP-03 | 32 | ~1.5 GB | Không vấn đề |
| EXP-04 | 32 | ~2 GB | Không vấn đề |
| EXP-05 | 8 | ~8–10 GB | T4: batch=16 vẫn an toàn |
| EXP-06 | 4 | ~13–15 GB | T4: batch=8, theo dõi VRAM |
| EXP-07 | 8 | ~10–12 GB | T4: batch=12 |
