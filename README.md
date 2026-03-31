# Deep Learning assignment

## Topic: Exploring BLIP-2 for VQA and enhancing nultimodal fusion with perceiver-based architecture

Description: Team experiments on Visual Question Answering (VQA) tasks that integrates BLIP-2's Q-Former architecture with multiple fusion baselines for systematic comparison.

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

## Knowledge
All knowledge of BLIP-2 model and VQAv2 dataset included in "knowledge.md"