# BLIP-2 Fusion Experiment for VQA

A research framework for Visual Question Answering (VQA) that integrates BLIP-2's Q-Former architecture with multiple fusion baselines for systematic comparison.

## Overview

This repository implements:
- **BLIP-2 Q-Former** – query transformer that bridges a frozen vision encoder with a language model
- **Fusion Baselines** – concatenation, bilinear, attention-based, and MLB fusion strategies
- **Training & Evaluation** – full training loop, VQAv2 accuracy metric, and checkpoint management
- **Scripts** – train, evaluate, feature extraction, and interactive demo

## Project Structure

```
blip2-qformer-vqa/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   └── vqa_dataset.py
├── models/
│   ├── __init__.py
│   ├── qformer.py
│   ├── fusion_baselines.py
│   └── blip2_vqa.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── __init__.py
│   └── vqa_eval.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_features.py
│   └── demo.py
├── notebooks/
│   └── analysis.ipynb
└── utils/
    ├── __init__.py
    └── helpers.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split val
```

### Feature Extraction

```bash
python scripts/extract_features.py \
    --config configs/default.yaml \
    --image_dir /path/to/images \
    --output_dir data/features/
```

### Interactive Demo

```bash
python scripts/demo.py --checkpoint checkpoints/best_model.pth
```

## Configuration

Edit `configs/default.yaml` to change model architecture, training hyperparameters, and dataset paths.

## Models

| Model | Description |
|-------|-------------|
| `blip2_vqa` | Full BLIP-2 + Q-Former + LM decoder |
| `concat_fusion` | Concatenation of visual and text features |
| `bilinear_fusion` | Bilinear pooling fusion |
| `attention_fusion` | Attention-weighted fusion |
| `mlb_fusion` | Multi-modal Low-rank Bilinear pooling |

## License

MIT
