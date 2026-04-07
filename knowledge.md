# BLIP-2 & VQAv2 — Kiến thức nền cho thí nghiệm Multimodal Fusion

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Vision Encoder — CLIP ViT-L/14](#2-vision-encoder--clip-vit-l14)
3. [Text Encoder — BERT-base](#3-text-encoder--bert-base)
4. [BLIP-2 & Q-Former](#4-blip-2--q-former)
5. [Các phương pháp Fusion](#5-các-phương-pháp-fusion)
   - [EXP-01: Mean Pooling + Linear](#51-exp-01-mean-pooling--linear)
   - [EXP-02: Concat + MLP](#52-exp-02-concat--mlp)
   - [EXP-03: MLB — Multi-modal Low-rank Bilinear](#53-exp-03-mlb--multi-modal-low-rank-bilinear)
   - [EXP-04: MFB — Multi-modal Factorized Bilinear](#54-exp-04-mfb--multi-modal-factorized-bilinear)
   - [EXP-05: Query Cross-Attention Bridge](#55-exp-05-query-cross-attention-bridge)
   - [EXP-06: Q-Former from Scratch](#56-exp-06-q-former-from-scratch)
   - [EXP-07: Perceiver Resampler](#57-exp-07-perceiver-resampler)
6. [So sánh các phương pháp](#6-so-sánh-các-phương-pháp)
7. [Dataset VQAv2](#7-dataset-vqav2)
8. [Evaluation trên VQAv2](#8-evaluation-trên-vqav2)
9. [Tóm tắt nhanh (Cheat Sheet)](#9-tóm-tắt-nhanh-cheat-sheet)

---

## 1. Tổng quan dự án

Dự án này so sánh **7 chiến lược fusion đa phương thức** cho bài toán VQA (Visual Question Answering) trên dataset VQAv2. Tất cả các thí nghiệm dùng chung:

- **Vision features** trích xuất từ CLIP ViT-L/14 (frozen)
- **Text features** từ BERT-base (frozen)
- **Bộ phân loại** trên 3.129 câu trả lời cố định

Cấu trúc chung của pipeline:

```
Ảnh  → [CLIP ViT-L/14 frozen] → visual tokens [B, 257, 1024]
                                        │
                                  [Fusion Module]  ← text [B, 768]
                                        │
                              [Classifier MLP] → logits [B, 3129]
```

Mục tiêu: tìm hiểu xem module fusion nào khai thác tốt nhất thông tin visual và textual, từ cách tiếp cận đơn giản nhất (pooling + linear) đến phức tạp nhất (Q-Former, Perceiver Resampler).

---

## 2. Vision Encoder — CLIP ViT-L/14

### 2.1 Tổng quan

**CLIP** (Contrastive Language-Image Pre-training, Radford et al. 2021) được pre-train trên 400M cặp ảnh-text bằng contrastive learning. Vision encoder của CLIP là **ViT-L/14** (Vision Transformer Large, patch size 14).

### 2.2 Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Kiến trúc | Vision Transformer (ViT) |
| Kích thước patch | 14×14 px |
| Ảnh đầu vào | 224×224 px |
| Số patch token | 256 (= 16×16 patches) |
| CLS token | 1 |
| Tổng token output | **257** (1 CLS + 256 patch) |
| Chiều output | **1024** |
| Số params | ~307M |
| Trạng thái | **Đóng băng hoàn toàn** trong tất cả thí nghiệm |

### 2.3 Cách sử dụng trong dự án

```
Input:  ảnh RGB [B, 3, 224, 224]
Output: [B, 257, 1024]  — toàn bộ patch tokens (dùng cho EXP-05, 06, 07)
        [B, 1024]       — CLS token hoặc mean pool (dùng cho EXP-01 đến 04)
```

**Pre-extraction**: Do ViT-L/14 là frozen, features được trích xuất trước và lưu vào HDF5 cache để tránh chạy lại mỗi epoch:
- Cache CLS-only: ~0.24 GB (EXP-01 đến 04)
- Cache full patch: ~63 GB (EXP-05 đến 07, hoặc chạy on-the-fly)

### 2.4 Lý do chọn ViT-L/14

- Được dùng trong paper BLIP-2 gốc làm một trong hai lựa chọn encoder
- Cân bằng giữa hiệu năng và chi phí tính toán (nhẹ hơn EVA-CLIP ViT-g/14)
- SDK CLIP của HuggingFace đơn giản, ổn định

---

## 3. Text Encoder — BERT-base

### 3.1 Tổng quan

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al. 2019) là mô hình ngôn ngữ bidirectional được pre-train trên BookCorpus + Wikipedia.

### 3.2 Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Kiến trúc | Transformer Encoder |
| Số lớp | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Vocab size | 30,522 |
| Số params | ~110M |
| Độ dài tối đa | 512 token |
| Trạng thái | **Đóng băng** (chỉ lấy CLS token embedding) |

### 3.3 Cách sử dụng

```
Input:  câu hỏi đã tokenize [B, 50]  (max_question_length = 50)
Output: CLS token [B, 768]  — đại diện toàn bộ câu hỏi
```

Tất cả 7 thí nghiệm dùng CLS token của BERT làm text representation.

---

## 4. BLIP-2 & Q-Former

### 4.1 BLIP-2 là gì

**BLIP-2** (Bootstrapping Language-Image Pre-training 2, Li et al. ICML 2023) đề xuất dùng một module nhỏ trainable gọi là **Q-Former** để kết nối frozen Vision Encoder với frozen LLM, thay vì fine-tune toàn bộ.

```
[Frozen ViT] → Image patches
                     ↓
               [Q-Former] ← Text / Instructions
                     ↓
           32 Query token outputs (768-dim)
                     ↓
             [Linear Projection]
                     ↓
              [Frozen LLM (OPT / FlanT5)]
```

### 4.2 Q-Former — Kiến trúc chi tiết

Q-Former dựa trên BERT-base (12 lớp, 768 hidden, 12 heads) với **32 learnable query token** được thêm vào.

```
Learnable queries: [B, 32, 768]
Visual patches:    [B, 257, 1024] → proj → [B, 257, 768]

Mỗi QFormerLayer:
  - Self-attention: queries attend to queries
  - Cross-attention: queries attend to visual patches  ← chỉ ở lớp chẵn (0, 2, 4, ...)
  - Feed-forward network

Output: [B, 32, 768]  — 32 visual query representations
```

**Cross-attention frequency**: Cứ 2 lớp 1 lần (lớp chẵn) mới có cross-attention với visual patches. Lớp lẻ chỉ có self-attention → tiết kiệm tính toán.

### 4.3 Vai trò trong dự án

Q-Former xuất hiện ở **hai dạng** trong bộ thí nghiệm:

| | EXP-06 (Q-Former scratch) | EXP-05 (Cross-Attn Bridge) |
|---|---|---|
| Số lớp | 12 | 3 |
| Cross-attn freq | 2 | 1 (mỗi lớp) |
| Query tokens | 32 | 32 |
| Params | ~190M | ~22M |
| Khởi tạo | Random | Random |

EXP-06 là triển khai đầy đủ Q-Former theo paper BLIP-2. EXP-05 là phiên bản đơn giản hóa.

### 4.4 Kết quả BLIP-2 trên VQAv2 (tham khảo)

| Model | Test-dev (zero-shot) |
|---|---|
| BLIP-2 ViT-L/14 + FlanT5-XL | 62.3% |
| BLIP-2 EVA-CLIP + FlanT5-XXL | 65.0% |
| BLIP-2 FlanT5-XXL (finetuned) | ~82% |

---

## 5. Các phương pháp Fusion

### 5.1 EXP-01: Mean Pooling + Linear

**Ý tưởng**: Đơn giản nhất có thể — gộp visual patch tokens bằng mean, ghép với text CLS, rồi dùng một lớp linear dự đoán câu trả lời.

**Luồng xử lý**:
```
visual: [B, 257, 1024] → mean(dim=1) → [B, 1024]
text:   [B, 768]
concat: [B, 1792] → Linear(1792, 3129) → logits
```

**Điểm mạnh**: Cực kỳ nhanh, dễ debug, là điểm mốc so sánh tuyệt đối.

**Điểm yếu**: Không có phi tuyến → khả năng biểu diễn thấp. Mean pooling mất thông tin không gian.

**Tham số trainable**: ~5.7M

---

### 5.2 EXP-02: Concat + MLP

**Paper tham khảo**: Antol et al., *VQA: Visual Question Answering*, ICCV 2015 (baseline gốc của VQA).

**Ý tưởng**: Thêm phi tuyến (ReLU + Dropout) vào EXP-01. MLP học được biểu diễn tốt hơn linear thuần.

**Luồng xử lý**:
```
visual: [B, 257, 1024] → mean(dim=1) → [B, 1024]
text:   [B, 768]
concat: [B, 1792]
  → Linear(1792, 1024) → ReLU → Dropout(0.1)
  → Linear(1024, 3129) → logits
```

**Điểm mạnh**: Đơn giản, ổn định, là baseline mạnh hơn EXP-01.

**Điểm yếu**: Vẫn mất thông tin không gian (dùng mean pool). Fusion chỉ là additive/concat, không bắt được tương tác nhân giữa hai modaliti.

**Tham số trainable**: ~7.2M

---

### 5.3 EXP-03: MLB — Multi-modal Low-rank Bilinear

**Paper**: Kim et al., *Hadamard Product for Low-rank Bilinear Pooling*, ICLR 2017.

**Ý tưởng cốt lõi**: Bilinear pooling đầy đủ $f = v^T W t$ có $O(d_v \times d_t \times d_{out})$ tham số — quá lớn. MLB xấp xỉ bằng cách chiếu cả hai về cùng chiều rồi dùng **tích Hadamard** (element-wise product):

$$f = \tanh(W_v v) \odot \tanh(W_t t)$$

**Luồng xử lý**:
```
visual: [B, 1024] → Linear(1024, 2048) → tanh → Dropout → [B, 2048]
text:   [B, 768]  → Linear(768,  2048) → tanh → Dropout → [B, 2048]
fused:  v ⊙ t                                            → [B, 2048]
  → Linear(2048, 1024) → ReLU → Dropout
  → Linear(1024, 3129) → logits
```

**Tại sao Hadamard tốt hơn concat?** Tích element-wise mô hình hóa tương tác **nhân** giữa từng chiều của visual và text — bắt được cross-modal correlation mà concat bỏ qua.

**Điểm yếu**: Chỉ có **một bậc** bilinear interaction, có thể chưa đủ phức tạp.

**Tham số trainable**: ~10.5M

---

### 5.4 EXP-04: MFB — Multi-modal Factorized Bilinear

**Paper**: Zhou et al., *MFB and Co-Attention for Visual Question Answering*, ICCV 2017.

**Ý tưởng cốt lõi**: Thay vì dùng 1 Hadamard product (MLB), MFB dùng **k=5 nhân tố song song** rồi **sum-pool** kết quả và **L2-normalize**:

$$f = \text{L2-norm}\left(\sum_{i=1}^{k} \tanh(W_v^{(i)} v) \odot \tanh(W_t^{(i)} t)\right)$$

**Luồng xử lý**:
```
visual: [B, 1024] → Linear(1024, 5120) → [B, 5120]
text:   [B, 768]  → Linear(768,  5120) → [B, 5120]
fused:  v ⊙ t                          → [B, 5120]
reshape: [B, 5, 1024]
sum over k=5: [B, 1024]
L2-norm: [B, 1024]
  → Linear(1024, 1024) → ReLU → Dropout
  → Linear(1024, 3129) → logits
```

**MFB vs MLB**:

| | MLB | MFB |
|---|---|---|
| Số nhân tố | 1 | k=5 |
| Normalization | Không | L2-norm sau sum-pool |
| Biểu diễn | Kém phong phú hơn | Phong phú hơn |
| Params | ~10.5M | ~17M |

**L2-norm sau sum-pool**: Giúp ổn định training và tránh gradient vanishing.

**Tham số trainable**: ~17M

---

### 5.5 EXP-05: Query Cross-Attention Bridge

**Ý tưởng**: Thay vì pool visual patches thành một vector rồi fusion, dùng **learnable query tokens** attend trực tiếp vào toàn bộ patch sequence. Câu hỏi không guide attention một cách tường minh — queries tự học cách trích xuất thông tin visual quan trọng.

**Luồng xử lý**:
```
queries:  [1, 32, 768] → expand → [B, 32, 768]
visual:   [B, 257, 1024] → Linear(1024, 768) → [B, 257, 768]

Lặp 3 lần (3 lớp cross-attention):
  queries = LayerNorm(queries)
  queries = queries + MultiHeadCrossAttn(Q=queries, K=visual, V=visual, heads=8)
  queries = queries + FFN(LayerNorm(queries))

pooled = queries.mean(dim=1) → [B, 768]
fused  = concat(pooled, text_cls) → [B, 1536]
  → MLP(1536, 1024, 3129) → logits
```

**Tại sao chỉ 3 lớp?** Đây là phiên bản rút gọn của Q-Former — đủ để học spatial attention mà không quá tốn tài nguyên.

**Điểm mạnh**: Giữ được thông tin không gian từ patch tokens. Queries có thể focus vào các vùng ảnh khác nhau.

**Điểm yếu**: Text chưa guide visual attention — text chỉ được ghép lại sau khi attend.

**Tham số trainable**: ~22M

---

### 5.6 EXP-06: Q-Former from Scratch

**Paper**: Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, ICML 2023.

**Ý tưởng**: Triển khai đầy đủ Q-Former như paper BLIP-2 mô tả, train từ đầu (không load pretrained weights). Đây là "chuẩn vàng" của bộ thí nghiệm.

**Luồng xử lý** (xem `models/qformer.py`):
```
queries:  [1, 32, 768] → expand → [B, 32, 768]
visual:   [B, 257, 1024] → visual_proj → visual_norm → [B, 257, 768]

12 khối QFormerLayer:
  Lớp chẵn (0, 2, 4, ..., 10):
    queries → self-attn(queries, queries) → cross-attn(queries, visual) → FFN
  Lớp lẻ  (1, 3, 5, ..., 11):
    queries → self-attn(queries, queries) → FFN

output: [B, 32, 768] → mean(dim=1) → [B, 768]
fused:  concat([B, 768], text_cls[B, 768]) → [B, 1536]
  → MLP(1536, 1024, 3129) → logits
```

**Cross-attention frequency = 2**: Lớp chẵn có cross-attention với visual, lớp lẻ không — giúp giảm tính toán mà vẫn đủ visual grounding.

**Tại sao "from scratch"?** Trong project này không load pretrained BLIP-2 weights (quá nặng). Ta train Q-Former từ random init trực tiếp trên VQAv2 — kết quả sẽ thấp hơn pretrained nhưng cho thấy khả năng học của kiến trúc.

**Tham số trainable**: ~190M

---

### 5.7 EXP-07: Perceiver Resampler

**Paper**: Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.

**Ý tưởng**: Thay vì dùng BERT-style transformer như Q-Former, Flamingo dùng **Perceiver Resampler** — kiến trúc đơn giản hơn: latent queries attend vào visual patches qua cross-attention, với visual patches làm **context** (không phải key/value riêng).

**Điểm khác biệt quan trọng so với Q-Former**:

| | Q-Former (EXP-06) | Perceiver Resampler (EXP-07) |
|---|---|---|
| Self-attention giữa queries | ✓ | ✓ |
| Cross-attn với visual | Ở lớp chẵn | Ở **mỗi** lớp |
| Queries làm K/V trong cross-attn | ✗ | ✓ (concat queries + visual làm K/V) |
| Số lớp | 12 | 4 |
| Query count | 32 | 64 |
| Text interaction | Không tường minh | Không tường minh |
| Pre-training objectives | ITC + ITM + ITG (trong BLIP-2) | Chỉ generative (trong Flamingo) |

**Luồng xử lý**:
```
latents: [1, 64, 768] → expand → [B, 64, 768]
visual:  [B, 257, 1024] → Linear(1024, 768) → [B, 257, 768]

Lặp 4 lần (4 lớp Perceiver):
  latents = LayerNorm(latents)
  context = concat(latents, visual) → [B, 64+257, 768]   ← latents + visual làm K/V
  latents = latents + MultiHeadCrossAttn(Q=latents, K=context, V=context)
  latents = latents + FFN(LayerNorm(latents))

pooled = latents.mean(dim=1) → [B, 768]
fused  = concat(pooled, text_cls) → [B, 1536]
  → MLP(1536, 1024, 3129) → logits
```

**Tại sao concat latents + visual làm K/V?** Cho phép latent queries attend cả vào nhau **và** vào visual patches trong một phép cross-attention — tiết kiệm tính toán hơn so với tách thành self-attention riêng + cross-attention riêng.

**Tham số trainable**: ~58M

---

## 6. So sánh các phương pháp

### 6.1 Theo loại visual input

| | Pooled vector [B, 1024] | Patch sequence [B, 257, 1024] |
|---|---|---|
| EXP-01 Mean+Linear | ✓ | |
| EXP-02 Concat+MLP | ✓ | |
| EXP-03 MLB | ✓ | |
| EXP-04 MFB | ✓ | |
| EXP-05 Cross-Attn | | ✓ |
| EXP-06 Q-Former | | ✓ |
| EXP-07 Perceiver | | ✓ |

EXP-01 đến 04 **mất thông tin không gian** vì dùng pooled vector. EXP-05 đến 07 giữ nguyên toàn bộ 257 patch tokens.

### 6.2 Theo loại fusion

| | Additive/Concat | Bilinear (Hadamard) | Cross-attention |
|---|---|---|---|
| EXP-01 | ✓ | | |
| EXP-02 | ✓ | | |
| EXP-03 MLB | | ✓ | |
| EXP-04 MFB | | ✓ (k-factor) | |
| EXP-05 | ✓ (cuối) | | ✓ (visual) |
| EXP-06 Q-Former | ✓ (cuối) | | ✓ (visual) |
| EXP-07 Perceiver | ✓ (cuối) | | ✓ (visual) |

### 6.3 Độ phức tạp tăng dần

```
EXP-01 (5.7M)  →  EXP-02 (7.2M)  →  EXP-03 (10.5M)  →  EXP-04 (17M)
     ↑ pooled features, tăng dần bilinear interaction

EXP-05 (22M)  →  EXP-07 (58M)  →  EXP-06 (190M)
     ↑ patch-level, tăng dần độ sâu transformer
```

### 6.4 Giả thuyết kết quả kỳ vọng

- EXP-01 < EXP-02 : Phi tuyến giúp ích
- EXP-02 < EXP-03 < EXP-04 : Bilinear tốt hơn concat; MFB tốt hơn MLB
- EXP-04 < EXP-05 : Patch-level attention tốt hơn pooled features
- EXP-05 < EXP-07 ≈ EXP-06 : Perceiver tương đương Q-Former với ít params hơn

---

## 7. Dataset VQAv2

### 7.1 Tổng quan

**VQAv2** (Goyal et al., CVPR 2017) khắc phục language bias của VQA v1 bằng **complementary image pairs** — mỗi câu hỏi có 2 ảnh với câu trả lời ngược nhau, buộc mô hình phải thực sự "nhìn" ảnh.

### 7.2 Thống kê

| Thống kê | Giá trị |
|---|---|
| Tổng QA pairs | ~1.1 triệu |
| Ảnh nguồn | MS-COCO 2014 (~123K ảnh) |
| Annotators/câu hỏi | 10 người |
| **Train** | 443.757 QA pairs · 82.783 ảnh |
| **Validation** | 214.354 QA pairs · 40.504 ảnh |
| **Test-dev** | 107.394 QA pairs |
| Bộ từ vựng câu trả lời | 3.129 (top answers) |

### 7.3 Phân loại câu hỏi

| Loại | Tỉ lệ | Ví dụ |
|---|---|---|
| Yes/No | ~38% | "Is there a dog in this image?" |
| Number | ~12% | "How many people are there?" |
| Other | ~50% | "What color is the car?" |

### 7.4 Soft Answer Scoring

Mỗi câu hỏi có **10 câu trả lời** từ 10 annotators. Thay vì one-hot label, dùng **soft score**:

```
score(a) = min(1.0, count(a trong 10 answers) / 3)
```

Ví dụ: 3 người trả lời "yes" → score = 1.0; 1 người → score = 0.33.

Loss function sử dụng **soft-target BCE** trên 3.129 classes với các soft score này.

---

## 8. Evaluation trên VQAv2

### 8.1 VQA Accuracy

Metric chính:

$$\text{Acc}(a, A) = \min\!\left(1,\ \frac{\text{count}(a \in A)}{3}\right)$$

$$\text{Overall} = \frac{1}{N} \sum_{i=1}^{N} \text{Acc}(a_i, A_i)$$

Trong đó $A$ là tập 10 câu trả lời của annotators, $a$ là câu trả lời dự đoán.

### 8.2 Báo cáo kết quả

Thường báo cáo overall + breakdown theo loại:

| Metric | Mô tả |
|---|---|
| `overall` | Trung bình toàn bộ |
| `yes/no` | Chỉ câu hỏi Yes/No (~38%) |
| `number` | Chỉ câu hỏi Number (~12%) |
| `other` | Còn lại (~50%) |

### 8.3 Kết quả tham chiếu

| Model | VQAv2 val (accuracy) |
|---|---|
| Random baseline | ~25% |
| Language-only (no image) | ~50% |
| Simple CNN + LSTM (VQA v1 baseline) | ~58% |
| MLB (Kim et al. 2017) | ~65% |
| MFB (Zhou et al. 2017) | ~66% |
| BLIP-2 ViT-L/14 + FlanT5-XL (zero-shot) | ~62% |
| BLIP-2 EVA-CLIP + FlanT5-XXL (finetuned) | ~82% |

---

## 9. Tóm tắt nhanh (Cheat Sheet)

### Các chiều tensor quan trọng

```
ViT-L/14 output:      [B, 257, 1024]   (257 = 1 CLS + 256 patches)
BERT-base CLS:        [B, 768]
Q-Former query out:   [B, 32, 768]
Perceiver latent out: [B, 64, 768]
Answer logits:        [B, 3129]
```

### Công thức VQA Accuracy

$$\text{Acc}(a, A) = \min\!\left(1,\ \frac{\text{count}(a \in A)}{3}\right)$$

### Loss function (tất cả thí nghiệm)

$$\mathcal{L} = -\sum_{c=1}^{3129} s_c \cdot \log \sigma(\hat{y}_c) + (1 - s_c) \cdot \log(1 - \sigma(\hat{y}_c))$$

Trong đó $s_c$ là soft score của answer $c$, $\hat{y}_c$ là logit tương ứng.

### Tham số trainable theo thí nghiệm

| Thí nghiệm | Params |
|---|---|
| EXP-01 Mean Pool + Linear | ~5.7M |
| EXP-02 Concat + MLP | ~7.2M |
| EXP-03 MLB | ~10.5M |
| EXP-04 MFB | ~17M |
| EXP-05 Cross-Attn Bridge | ~22M |
| EXP-07 Perceiver Resampler | ~58M |
| EXP-06 Q-Former from Scratch | ~190M |

### Tài liệu tham khảo chính

1. **BLIP-2**: Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, ICML 2023.
2. **VQAv2**: Goyal et al., *Making the V in VQA Matter*, CVPR 2017.
3. **MLB**: Kim et al., *Hadamard Product for Low-rank Bilinear Pooling*, ICLR 2017.
4. **MFB**: Zhou et al., *MFB and Co-Attention for Visual Question Answering*, ICCV 2017.
5. **Flamingo / Perceiver**: Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
6. **CLIP**: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
7. **BERT**: Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.

---

*File knowledge.md — cập nhật theo bộ thí nghiệm 7 fusion baselines.*
- Large Language Model (OPT hoặc FlanT5)

Và chỉ **train một module nhỏ** gọi là **Q-Former** (~188M params) làm cầu nối.

### 1.3 Đóng góp chính

| Đóng góp | Mô tả |
|---|---|
| Q-Former | Module trainable kết nối vision ↔ language |
| 2-stage pre-training | Stage 1: vision-language representation; Stage 2: generative language |
| Frozen backbone | Tận dụng pretrained knowledge mà không phá vỡ |
| Zero-shot VQA | Không cần fine-tune vẫn đạt kết quả tốt |

---

## 2. Kiến trúc BLIP-2

```
┌─────────────────────────────────────────────────────────────────┐
│                         BLIP-2 Pipeline                         │
│                                                                 │
│  Image ──► [Frozen Image Encoder] ──► Image Features           │
│                                              │                  │
│                                              ▼                  │
│                                      [Q-Former]                 │
│                                    (Trainable ~188M)            │
│                                              │                  │
│                                    Query Output (32 tokens)     │
│                                              │                  │
│                                     [FC Projection]             │
│                                              │                  │
│  Text ────────────────────────────► [Frozen LLM]               │
│                                    (OPT / FlanT5)               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Image Encoder

- **BLIP-2 w/ ViT-L/14**: CLIP ViT-L/14, output 257 patch tokens (196 + 1 CLS), dim 1024
- **BLIP-2 w/ EVA-CLIP ViT-g/14**: EVA-CLIP, output 1409 patch tokens, dim 1408
- **Frozen hoàn toàn** trong cả hai stage — không update gradient
- Image được chia thành patches, mỗi patch → một visual token

### 2.2 Q-Former

Xem chi tiết ở Section 3.

### 2.3 Language Model

| Variant | LLM | Params | Type |
|---|---|---|---|
| BLIP-2 OPT-2.7B | OPT-2.7B | 2.7B | Decoder-only |
| BLIP-2 OPT-6.7B | OPT-6.7B | 6.7B | Decoder-only |
| BLIP-2 FlanT5-XL | FlanT5-XL | 3B | Encoder-Decoder |
| BLIP-2 FlanT5-XXL | FlanT5-XXL | 11B | Encoder-Decoder |

- **Decoder-only (OPT)**: Query output → prepend vào text sequence → autoregressive generation
- **Encoder-Decoder (FlanT5)**: Query output → input của encoder → decoder tạo ra answer

### 2.4 FC Projection Layer

Một **linear layer** đơn giản chiếu dimension của Q-Former output sang LLM's embedding dimension.

- Q-Former output dim: 768 (BERT-base size)
- LLM embedding dim: tùy LLM (ví dụ OPT-2.7B là 2560)

---

## 3. Q-Former — Trái tim của BLIP-2

### 3.1 Định nghĩa

Q-Former (Querying Transformer) là module **transformer nhỏ** khởi tạo từ **BERT-base** (12 layers, 768 hidden dim, 12 heads) với **trọng số chia sẻ** (shared weights) giữa hai sub-module bên trong.

### 3.2 Cấu trúc bên trong

```
Q-Former gồm 2 transformer sub-modules chia sẻ self-attention layers:

┌─────────────────────────────────────────────────────┐
│                     Q-Former                        │
│                                                     │
│  Learned Query Tokens (32 × 768)                   │
│         │                                           │
│  ┌──────▼──────────────────────────────────────┐   │
│  │  Self-Attention (queries attend to queries) │   │
│  │  Cross-Attention (queries attend to image)  │   │
│  │  Feed-Forward Network                       │   │
│  └──────────────────────────────────────────────┘   │
│         │                    │                      │
│  [Image Sub-module]   [Text Sub-module]             │
│  (interacts w/ image) (interacts w/ text)           │
└─────────────────────────────────────────────────────┘
```

### 3.3 Learned Query Tokens

- **32 learnable query vectors** (embeddings), mỗi vector dim = 768
- Đây là tham số được **learn** trong quá trình pre-training
- Query tokens **attend to** image features qua cross-attention
- Query tokens **attend to each other** qua self-attention
- Output: 32 query vectors chứa thông tin visual đã "distill"

**Tại sao 32?** Đây là hyperparameter. Số queries = số visual tokens đưa vào LLM. Ít hơn → LLM nhận ít thông tin visual hơn nhưng tiết kiệm context. Tác giả chọn 32 sau ablation.

### 3.4 Attention Masking Mechanism

Q-Former sử dụng **khác nhau** attention mask trong từng pre-training objective:

| Objective | Query-Query Attn | Query-Text Attn | Text-Text Attn |
|---|---|---|---|
| ITC | Uni-modal (no cross) | ✗ | Causal mask |
| ITM | Bi-directional | ✓ (full) | Bi-directional |
| ITG | Bi-directional | Causal | Causal |

### 3.5 Shared Weights

Hai sub-module (image sub-module và text sub-module) **chia sẻ self-attention weights** nhưng cross-attention weights chỉ tồn tại ở image sub-module.

---

## 4. Hai giai đoạn huấn luyện

### Stage 1: Vision-Language Representation Learning

**Mục tiêu**: Dạy Q-Former extract visual features liên quan đến ngôn ngữ từ frozen image encoder.

```
[Frozen Image Encoder] → Image Features
                              ↓
                        [Q-Former]  ← Text (caption)
                              ↓
                    3 Pre-training Objectives
                    (ITC, ITM, ITG)
```

- **Image Encoder bị freeze**
- **Q-Former được train** với 3 objectives đồng thời
- Data: image-text pairs (COCO, CC3M, CC12M, SBU, LAION-400M) — ~129M pairs
- Epochs: không nhiều, vài epochs
- Optimizer: AdamW

### Stage 2: Vision-to-Language Generative Learning

**Mục tiêu**: Kết nối Q-Former với frozen LLM để tạo text.

```
[Q-Former] → 32 query outputs (768-dim)
                    ↓
             [Linear Projection]
                    ↓
             [Frozen LLM Input]
             (prepend to text tokens)
                    ↓
              Autoregressive generation
              (Language Modeling Loss)
```

- **Cả Image Encoder và LLM đều freeze**
- Chỉ **Q-Former + Linear Projection** được train
- Loss: **Language Modeling Loss** (cross-entropy trên text tokens)
- Query output tokens được xem là "soft visual prompts" cho LLM

**Insight quan trọng**: Q-Former học cách tóm tắt visual info thành 32 tokens sao cho LLM có thể generate text caption/answer từ đó.

---

## 5. Các mục tiêu học (Pre-training Objectives)

### 5.1 Image-Text Contrastive Learning (ITC)

**Cơ chế**: Giống CLIP — align image và text trong embedding space.

```
Image → Q-Former → 32 query outputs → max pooling → z_q (image rep)
Text  → Q-Former (text sub-module) → [CLS] token → z_t (text rep)

Loss: InfoNCE contrastive loss
      - Positive pairs: (image, its caption)
      - Negative pairs: in-batch negatives
```

**Attention mask**: Query tokens KHÔNG attend to text tokens (tránh information leakage trong ITC).

**Mục tiêu**: Học alignment giữa visual và language representations.

### 5.2 Image-Grounded Text Matching (ITM)

**Cơ chế**: Binary classification — phân biệt matched vs. unmatched image-text pairs.

```
Image + Text → Q-Former (full bi-directional attention)
            → 32 query outputs → linear classifier → match/no-match
```

**Attention mask**: Query tokens CÓ THỂ attend to text tokens (full bidirectional).

**Hard Negative Mining**: Sử dụng similarity scores từ ITC để chọn hard negatives (những cặp gần nhau nhưng không match).

### 5.3 Image-Grounded Text Generation (ITG)

**Cơ chế**: Dạy Q-Former generate text từ image — autoregressive LM.

```
Image → Q-Former → [query tokens attend to image]
Text  → Q-Former → [causal attention on text, text attends to queries]
       → generate next token from previous tokens + visual context
```

**Attention mask**: Causal mask trên text (text token i chỉ attend to j ≤ i), nhưng text tokens có thể attend to query tokens.

**Token đặc biệt**: `[DEC]` token thay thế `[CLS]` để signal task decoding.

---

## 6. Inference & Zero-shot Generalization

### 6.1 VQA Inference

Có hai cách sử dụng BLIP-2 cho VQA:

**Cách 1: Open-ended generation (dùng LLM)**
```
Prompt: "Question: {question} Answer:"
Image → Q-Former → 32 tokens → [prepend] → LLM → generate answer
```

**Cách 2: Few-shot prompting**
```
Prompt: "Question: What is this? Answer: A cat. Question: {new_q} Answer:"
Sử dụng in-context learning của LLM
```

### 6.2 Image Captioning

```
Prompt: "a photo of"
Image → Q-Former → 32 tokens → LLM → generate full caption
```

### 6.3 Visual Conversation / Chatting

Với FlanT5, có thể thực hiện multi-turn QA:
```
Q: "What is in the image?"
A: [generated]
Q: "What color is it?"
A: [generated, conditioned on image + previous context]
```

---

## 7. So sánh BLIP-2 với các mô hình khác

### 7.1 Về kiến trúc

| Mô hình | Vision Encoder | Bridge Module | LLM | Trainable |
|---|---|---|---|---|
| BLIP-2 | Frozen ViT | Q-Former | Frozen LLM | Q-Former only |
| Flamingo | Frozen ViT | Perceiver Resampler + Gated X-Attn | Frozen LM | Resampler + X-Attn layers |
| LLaVA | Frozen ViT | Linear / MLP | Fine-tuned LLM | Projection + LLM |
| InstructBLIP | Frozen ViT | Q-Former (instruction-aware) | Frozen LLM | Q-Former |
| MiniGPT-4 | Frozen ViT | Q-Former | Frozen Vicuna | 1 linear layer |

### 7.2 Q-Former vs. Perceiver Resampler (Flamingo)

| Tiêu chí | Q-Former | Perceiver Resampler |
|---|---|---|
| Base architecture | BERT-based Transformer | Transformer |
| Text interaction | ✓ (có text sub-module) | ✗ (chỉ visual) |
| Pre-training objectives | ITC + ITM + ITG (3 objectives) | Chỉ generative |
| Output tokens | 32 learned queries | 64 latent tokens |
| Cross-attention injection | Tập trung vào Q-Former | Xen kẽ vào nhiều LLM layers |
| Visual grounding | Tốt hơn nhờ ITC/ITM | Yếu hơn |

**Điểm mạnh Q-Former**: Nhờ ITC và ITM, Q-Former học được alignment tốt hơn giữa vision và language.

**Điểm mạnh Perceiver Resampler**: Cross-attention xen kẽ sâu vào LLM có thể giúp LLM "nhìn" hình ảnh trực tiếp hơn.

### 7.3 Kết quả so sánh trên VQAv2

| Mô hình | VQAv2 (test-dev) | Zero-shot |
|---|---|---|
| Flamingo-80B | 56.3 | ✓ |
| BLIP-2 FlanT5-XXL | **65.0** | ✓ |
| BLIP-2 OPT-6.7B | 54.4 | ✓ |
| InstructBLIP FlanT5-XXL | 63.1 (w/ finetuning) | — |

---

## 8. Dataset VQAv2

### 8.1 Tổng quan

**VQAv2** (Visual Question Answering v2) là benchmark chuẩn cho VQA task, do Goyal et al. (2017) tạo ra nhằm khắc phục bias của VQA v1.

- **Paper**: "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering" (CVPR 2017)
- **Website**: https://visualqa.org

### 8.2 Động lực tạo VQAv2

VQA v1 có **language bias nghiêm trọ**:
- Mô hình có thể trả lời đúng chỉ dựa vào câu hỏi, không cần nhìn ảnh
- Ví dụ: "Is there a..." → "yes" luôn đúng 90% trường hợp

**Giải pháp VQAv2**: Tạo **complementary image pairs** — mỗi câu hỏi có 2 ảnh với **câu trả lời ngược nhau**.

```
Original pair:    Image A + Question → "yes"
Complementary:    Image B + Question → "no"
(Image B is visually similar to A but changes the answer)
```

### 8.3 Thống kê dataset

| Thống kê | Số lượng |
|---|---|
| Tổng QA pairs | ~1.1 triệu |
| Số câu hỏi | ~265K |
| Số ảnh (COCO) | ~123K |
| Annotators/câu hỏi | 10 người |
| **Training set** | 443,757 QA pairs (~82K images) |
| **Validation set** | 214,354 QA pairs (~41K images) |
| **Test-dev set** | 107,394 QA pairs (~81K images) |
| **Test-std set** | 107,394 QA pairs (~81K images) |

> **Lưu ý**: Test-dev và test-std dùng cùng ảnh nhưng câu hỏi khác nhau. Annotation của test set không công khai.

### 8.4 Phân loại câu hỏi

VQAv2 có 3 loại câu hỏi chính:

| Loại | % dataset | Ví dụ |
|---|---|---|
| **Yes/No** | ~38% | "Is there a dog?" |
| **Number** | ~12% | "How many cats?" |
| **Other** | ~50% | "What color is the car?" |

### 8.5 Phân bố câu trả lời

- Câu trả lời là **free-form text** nhưng thường ngắn (1-3 từ)
- Top answers: "yes", "no", "2", "1", "white", "3", "red", ...
- Có **long-tail distribution** — nhiều câu trả lời hiếm
- Evaluation dùng **soft accuracy** (xem section 9)

### 8.6 Cách tạo dataset

```
Bước 1: Lấy images từ COCO 2014
Bước 2: Annotators đặt câu hỏi về ảnh (5 câu/ảnh)
Bước 3: 10 annotators khác trả lời mỗi câu hỏi
Bước 4: Tạo complementary pairs (thêm ảnh ngược đáp án)
         → đảm bảo câu hỏi thực sự cần nhìn ảnh
```

### 8.7 Nguồn ảnh

VQAv2 sử dụng **MS-COCO** (Microsoft Common Objects in Context):
- 80 object categories
- Ảnh thực tế, phức tạp, đa dạng
- Resolution thay đổi, thường ~640×480
- Chứa nhiều objects, bối cảnh phong phú

---

## 9. Evaluation trên VQAv2

### 9.1 VQA Accuracy (Soft Accuracy)

Metric chính của VQAv2 là **VQA Accuracy**, không phải hard accuracy.

**Công thức**:

```
VQA_Accuracy(answer a, GT answers A) = min(1, count(a in A) / 3)
```

Trong đó:
- `A` = tập hợp 10 câu trả lời từ 10 annotators
- `count(a in A)` = số lần answer `a` xuất hiện trong A
- **Chia 3** và **min với 1** → nếu ≥3 annotators đồng ý → score = 1.0

**Ví dụ**:
```
GT answers: ["yes", "yes", "yes", "no", "yes", "yes", "no", "yes", "yes", "yes"]
Predict "yes": count = 8 → min(1, 8/3) = 1.0 ✓
Predict "no":  count = 2 → min(1, 2/3) = 0.67
Predict "cat": count = 0 → 0.0
```

**Overall Accuracy** = average over all questions.

### 9.2 Per-type Accuracy

Thường báo cáo accuracy theo từng loại:
- `Acc (Yes/No)` — cao nhất, thường >80%
- `Acc (Number)` — thấp nhất, ~50-60%
- `Acc (Other)` — ~50-65%
- `Acc (Overall)` — weighted average

### 9.3 Submission

- **Test-dev**: Có thể submit nhiều lần, kết quả hiện lên leaderboard
- **Test-std**: Submit hạn chế (1 lần/năm trong competition mode)
- Server: EvalAI (https://eval.ai/web/challenges/challenge-page/830)

### 9.4 Kết quả BLIP-2 trên VQAv2

| Model | Test-dev | Test-std |
|---|---|---|
| BLIP-2 FlanT5-XL (zero-shot) | 62.3 | — |
| BLIP-2 FlanT5-XXL (zero-shot) | **65.0** | — |
| BLIP-2 OPT-2.7B (zero-shot) | 53.5 | — |
| BLIP-2 OPT-6.7B (zero-shot) | 54.4 | — |
| BLIP-2 FlanT5-XXL (finetuned) | ~82.x | — |

> Zero-shot = không fine-tune trên VQAv2 training set.

---

## 10. Ablation Study Q-Former

### 10.1 Số lượng Query Tokens

| # Queries | VQAv2 | COCO CIDEr |
|---|---|---|
| 16 | - 0.8% | - 1.2 |
| **32** | **baseline** | **baseline** |
| 64 | +0.1% | +0.3 |

→ 32 là sweet spot: đủ thông tin, không quá tốn kém.

### 10.2 Pre-training Objectives

| Objectives | VQAv2 |
|---|---|
| Only ITG | baseline - 2.1% |
| ITC + ITG | baseline - 0.8% |
| ITM + ITG | baseline - 0.5% |
| **ITC + ITM + ITG** | **baseline** |

→ Cả 3 objectives đều cần thiết, ITG quan trọng nhất.

### 10.3 Stage 1 vs. Stage 2

| Training | VQAv2 |
|---|---|
| Only Stage 2 (no Stage 1) | -4.5% |
| Stage 1 + Stage 2 | baseline |

→ Stage 1 (representation learning) rất quan trọng.

### 10.4 Image Encoder

| Encoder | VQAv2 |
|---|---|
| ViT-L/14 (CLIP) | 62.3 |
| **EVA-CLIP ViT-g/14** | **65.0** |

→ Encoder mạnh hơn → kết quả tốt hơn đáng kể.

---

## 11. Hạn chế & Hướng phát triển

### 11.1 Hạn chế của BLIP-2

**In-context learning yếu**:
- OPT và FlanT5 không phải instruction-following models
- Few-shot performance kém hơn Flamingo vì LLM không được tune cho in-context VQA

**Knowledge hallucination**:
- LLM có thể generate text không liên quan đến ảnh
- Không có cơ chế hard constraint buộc LLM "nhìn" ảnh

**Computation bottleneck**:
- Mặc dù Q-Former nhỏ, LLM vẫn rất lớn → inference chậm
- 32 query tokens = thêm 32 tokens vào LLM context

**VQAv2 fine-tuning**:
- Zero-shot 65% → vẫn còn khoảng cách lớn so với specialized models (~85%+)
- Fine-tuning cần infrastructure lớn

### 11.2 Hướng phát triển

**InstructBLIP (2023)**:
- Q-Former nhận thêm instruction text → instruction-aware visual features
- Training trên instruction-tuning datasets
- Cải thiện generalization đáng kể

**MiniGPT-4 (2023)**:
- Dùng Q-Former của BLIP-2 + Vicuna (instruction-tuned LLaMA)
- Chỉ train 1 linear layer → rất hiệu quả
- Nổi bật về conversational ability

**LLaVA-1.5 (2023)**:
- Thay Q-Former bằng MLP projection đơn giản
- Fine-tune LLM → kết quả mạnh hơn trên benchmarks

**Hướng nghiên cứu thêm**:
- Perceiver Resampler vs. Q-Former comparison
- Efficient visual tokenization
- Better instruction following without full LLM fine-tuning

---

## 12. Tóm tắt nhanh (Cheat Sheet)

### BLIP-2 Key Numbers

```
Q-Former:
  - Base: BERT-base (12L, 768H, 12A)
  - Learned queries: 32
  - Params: ~188M

Image Encoders:
  - ViT-L/14: 307M params, 1024-dim, 256 tokens
  - EVA-CLIP ViT-g/14: 1B params, 1408-dim, 1408 tokens

LLMs used:
  - OPT-2.7B, OPT-6.7B (decoder-only)
  - FlanT5-XL (3B), FlanT5-XXL (11B) (encoder-decoder)

Pre-training data: ~129M image-text pairs
Pre-training objectives: ITC + ITM + ITG
```

### VQAv2 Key Numbers

```
Images: ~123K (từ COCO 2014)
QA pairs: ~1.1M
Annotators/question: 10
Answer types: Yes/No (38%), Number (12%), Other (50%)
Metric: VQA Accuracy = min(1, count/3)
Train: 443,757 | Val: 214,354 | Test: ~107K×2
```

### Công thức quan trọng

**VQA Accuracy**:
```
Acc(a, A) = min(1.0, count(a ∈ A) / 3)
Overall = (1/N) Σ Acc(aᵢ, Aᵢ)
```

**ITC Loss (InfoNCE)**:
```
L_ITC = -log [exp(sim(q,t)/τ) / Σⱼ exp(sim(q,tⱼ)/τ)]
```

**ITM Loss (Binary CE)**:
```
L_ITM = -[y·log(p) + (1-y)·log(1-p)]
```

**ITG Loss (LM Loss)**:
```
L_ITG = -Σₜ log P(wₜ | w<t, image)
```

---

## Tài liệu tham khảo

1. **BLIP-2 Paper**: Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023.
2. **VQAv2 Paper**: Goyal, Y., et al. "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering." CVPR 2017.
3. **BLIP Paper**: Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation." ICML 2022.
4. **Flamingo Paper**: Alayrac, J-B., et al. "Flamingo: a Visual Language Model for Few-Shot Learning." NeurIPS 2022.
5. **InstructBLIP**: Dai, W., et al. "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning." NeurIPS 2023.
6. **EVA-CLIP**: Sun, Q., et al. "EVA-CLIP: Improved Training Techniques for CLIP at Scale." 2023.
7. **OPT**: Zhang, S., et al. "OPT: Open Pre-trained Transformer Language Models." 2022.
8. **FlanT5**: Chung, H.W., et al. "Scaling Instruction-Finetuned Language Models." 2022.

---

*File được tạo cho project Q-Former (BLIP-2) VQA trên VQAv2 — Deep Learning Course Group Project.*