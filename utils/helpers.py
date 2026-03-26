"""Shared utility helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Nested dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``"cuda"``, ``"cpu"``, ``"mps"``, or ``None`` (auto-detect).

    Returns:
        The resolved :class:`torch.device`.
    """
    if device_str is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler factories
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Build an optimizer from a config dictionary.

    The config dict must have a ``training`` sub-dict with at least
    ``learning_rate`` and ``weight_decay`` keys, and an ``optimizer`` sub-dict
    with a ``name`` key.

    Args:
        model: The model whose parameters to optimise.
        cfg: Full config dict as returned by :func:`load_config`.

    Returns:
        A PyTorch optimizer.
    """
    train_cfg = cfg.get("training", {})
    opt_cfg = cfg.get("optimizer", {})

    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    # Separate params with / without weight decay
    decay_params, no_decay_params = _split_decay_params(model)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = opt_cfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer '{name}'")


def _split_decay_params(model: nn.Module):
    """Return (decay_params, no_decay_params) for the model."""
    no_decay_names = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_names):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a learning-rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Full config dict.
        num_training_steps: Total number of gradient steps.

    Returns:
        A PyTorch LR scheduler.
    """
    sched_cfg = cfg.get("scheduler", {})
    train_cfg = cfg.get("training", {})

    name = sched_cfg.get("name", "cosine").lower()
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(num_training_steps - warmup_steps, 1),
            eta_min=min_lr,
        )
    if name == "step":
        step_size = int(sched_cfg.get("step_size", 3))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise ValueError(f"Unknown scheduler '{name}'")


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total (trainable) parameter count of *model*.

    Args:
        model: The model to inspect.
        trainable_only: If ``True``, count only parameters with
            ``requires_grad=True``.

    Returns:
        Integer parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Running average
# ---------------------------------------------------------------------------


class AverageMeter:
    """Tracks running average of a scalar value.

    Example::

        meter = AverageMeter("loss")
        for batch_loss in losses:
            meter.update(batch_loss)
        print(meter.avg)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add *n* samples with value *val*.

        Args:
            val: Observed value (e.g. batch loss).
            n: Number of samples (e.g. batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, count={self.count})"
