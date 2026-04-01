# BLIP-2 & VQAv2 — Toàn bộ kiến thức

---

## MỤC LỤC

1. [BLIP-2 Overview](#1-blip-2-overview)
2. [Kiến trúc BLIP-2](#2-kiến-trúc-blip-2)
3. [Q-Former — Trái tim của BLIP-2](#3-q-former--trái-tim-của-blip-2)
4. [Hai giai đoạn huấn luyện](#4-hai-giai-đoạn-huấn-luyện)
5. [Các mục tiêu học (Pre-training Objectives)](#5-các-mục-tiêu-học-pre-training-objectives)
6. [Inference & Zero-shot Generalization](#6-inference--zero-shot-generalization)
7. [So sánh BLIP-2 với các mô hình khác](#7-so-sánh-blip-2-với-các-mô-hình-khác)
8. [Dataset VQAv2](#8-dataset-vqav2)
9. [Evaluation trên VQAv2](#9-evaluation-trên-vqav2)
10. [Ablation Study Q-Former](#10-ablation-study-q-former)
11. [Hạn chế & Hướng phát triển](#11-hạn-chế--hướng-phát-triển)
12. [Tóm tắt nhanh (Cheat Sheet)](#12-tóm-tắt-nhanh-cheat-sheet)

---

## 1. BLIP-2 Overview

### 1.1 Bối cảnh

**BLIP-2** (Bootstrapping Language-Image Pre-training 2) được Salesforce Research công bố năm 2023 (Li et al., ICML 2023). Đây là bước tiến hóa từ BLIP gốc (2022), với mục tiêu chính:

- **Kết nối hiệu quả** frozen Image Encoder (vision) với frozen LLM (language) mà **không cần fine-tune cả hai**.
- Giảm chi phí tính toán trong khi vẫn đạt SOTA trên nhiều vision-language task.
- Giải quyết bài toán **modality gap** — khoảng cách biểu diễn giữa vision và language.

### 1.2 Ý tưởng cốt lõi

> "Instead of end-to-end training, use a lightweight trainable module (Q-Former) to bridge the frozen vision encoder and the frozen LLM."

Thay vì train toàn bộ mô hình (tốn kém), BLIP-2 **đóng băng (freeze)** cả:
- Image Encoder (ViT-L/14 từ CLIP hoặc EVA-CLIP ViT-g/14)
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