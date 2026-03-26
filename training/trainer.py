"""Training loop for BLIP-2 VQA experiments."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import VQALoss

logger = logging.getLogger(__name__)


class VQATrainer:
    """Training / evaluation coordinator for VQA models.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        optimizer: PyTorch optimizer.
        scheduler: Learning-rate scheduler (optional).
        loss_fn: Loss module; defaults to :class:`~training.losses.VQALoss`.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        output_dir: Directory for saving checkpoints and logs.
        gradient_accumulation_steps: Number of steps before an optimizer step.
        gradient_clip: Max gradient norm (0 = disabled).
        mixed_precision: Enable automatic mixed precision (AMP).
        eval_metric_fn: Optional callable ``(predictions, targets) → float``
            used to compute a scalar validation metric.
        log_every: Log training stats every *N* optimizer steps.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cuda",
        output_dir: str = "checkpoints",
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        mixed_precision: bool = True,
        eval_metric_fn: Optional[Callable] = None,
        log_every: int = 100,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or VQALoss(loss_type="bce")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        self.eval_metric_fn = eval_metric_fn
        self.log_every = log_every

        self.model = self.model.to(self.device)

        # AMP scaler
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

        self.global_step = 0
        self.best_val_metric = float("-inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_epochs: int) -> None:
        """Train for *num_epochs* epochs.

        Args:
            num_epochs: Number of training epochs.
        """
        for epoch in range(1, num_epochs + 1):
            logger.info("=== Epoch %d / %d ===", epoch, num_epochs)
            train_loss = self._train_epoch(epoch)
            val_results = self._val_epoch(epoch)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f",
                epoch,
                train_loss,
                val_results["loss"],
            )

            # Scheduler step (epoch-level)
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            metric = val_results.get("metric", -val_results["loss"])
            self._save_checkpoint(epoch, metric)

    def evaluate(self) -> Dict[str, float]:
        """Run a single evaluation pass over the validation set.

        Returns:
            Dictionary with at least ``"loss"`` and optionally ``"metric"``.
        """
        return self._val_epoch(epoch=0)

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        t0 = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=False)
        ):
            loss = self._forward_batch(batch)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Step %d | loss=%.4f | %.1f steps/s",
                        self.global_step,
                        loss.item() * self.gradient_accumulation_steps,
                        self.log_every / elapsed,
                    )
                    t0 = time.time()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / num_batches

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in tqdm(self.val_loader, desc=f"Val Epoch {epoch}", leave=False):
            loss = self._forward_batch(batch)
            total_loss += loss.item()

            if self.eval_metric_fn is not None:
                preds = self._get_predictions(batch)
                all_preds.extend(preds)
                if "answer_label" in batch:
                    all_targets.extend(batch["answer_label"].tolist())

        results: Dict[str, float] = {"loss": total_loss / max(len(self.val_loader), 1)}

        if self.eval_metric_fn is not None and all_targets:
            results["metric"] = self.eval_metric_fn(all_preds, all_targets)

        return results

    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """Move batch to device and compute loss."""
        pixel_values = batch["image"].to(self.device)
        answer_scores = batch.get("answer_scores")
        if answer_scores is not None:
            answer_scores = answer_scores.to(self.device)
        answer_label = batch.get("answer_label")
        if answer_label is not None:
            answer_label = answer_label.to(self.device)

        import contextlib
        ctx = (
            torch.cuda.amp.autocast()
            if self.scaler is not None
            else contextlib.nullcontext()
        )
        with ctx:
            out = self.model(
                pixel_values=pixel_values,
                answer_scores=answer_scores,
            )

        if "loss" in out:
            return out["loss"]

        if "logits" in out:
            targets = answer_scores if answer_scores is not None else answer_label
            if targets is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            return self.loss_fn(out["logits"], targets)

        return torch.tensor(0.0, device=self.device, requires_grad=True)

    @torch.no_grad()
    def _get_predictions(self, batch: Dict):
        pixel_values = batch["image"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        if "logits" in out:
            return out["logits"].argmax(dim=-1).tolist()
        return []

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric": metric,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

        if metric > self.best_val_metric:
            self.best_val_metric = metric
            best_path = self.output_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info("New best model saved (metric=%.4f)", metric)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a checkpoint and restore model / optimizer state.

        Args:
            checkpoint_path: Path to a ``.pth`` checkpoint file.

        Returns:
            The epoch at which the checkpoint was saved.
        """
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_metric = state.get("metric", float("-inf"))
        epoch = state.get("epoch", 0)
        logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, epoch)
        return epoch
