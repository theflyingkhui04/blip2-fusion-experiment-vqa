"""Vòng lặp huấn luyện cho các thí nghiệm BLIP-2 VQA."""

from __future__ import annotations

import contextlib
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

from configs.contracts import (
    KEY_ANSWER_LABEL,
    KEY_ANSWER_SCORES,
    KEY_ANSWER_TYPE,
    KEY_ATTENTION_MASK,
    KEY_BEST_VAL_METRIC,
    KEY_EPOCH,
    KEY_GLOBAL_STEP,
    KEY_IMAGE_FEATURES,
    KEY_INPUT_IDS,
    KEY_LOGITS,
    KEY_LOSS,
    KEY_MODEL_STATE,
    KEY_NUMBER_ACC,
    KEY_OTHER_ACC,
    KEY_OVERALL_ACC,
    KEY_OPTIM_STATE,
    KEY_PIXEL_VALUES,
    KEY_SCHED_STATE,
    KEY_YESNO_ACC,
    LOSS_BCE,
    EvalResult,
)
from training.losses import VQALoss

logger = logging.getLogger(__name__)


class VQATrainer:
    """Bộ điều phối huấn luyện / đánh giá cho các mô hình VQA.

    Args:
        model: Mô hình cần huấn luyện.
        train_loader: DataLoader cho tập huấn luyện.
        val_loader: DataLoader cho tập validation.
        optimizer: PyTorch optimizer.
        scheduler: Learning-rate scheduler (tuỳ chọn).
        loss_fn: Module tính loss; mặc định là :class:`~training.losses.VQALoss`.
        device: Thiết bị Torch (ví dụ. ``"cuda"`` hoặc ``"cpu"``).
        output_dir: Thư mục lưu checkpoint và log.
        gradient_accumulation_steps: Số bước tích luỹ gradient trước mỗi optimizer step.
        gradient_clip: Max-norm cho gradient clipping (0 = tắt).
        mixed_precision: Bật Automatic Mixed Precision (AMP).
        eval_metric_fn: Callable tuỳ chọn ``(predictions, targets) → float``
            dùng để tính metric scalar khi validation.
        log_every: Ghi log mỗi *N* optimizer steps.
        text_encoder: FrozenTextEncoder cho EXP pipeline (None = legacy BLIP2VQA).
        wandb_run: Đối tượng ``wandb.Run`` đang active (None = không log W&B).
            Truyền vào từ ``scripts/train.py`` sau khi gọi ``wandb.init()``.
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
        text_encoder: Optional[nn.Module] = None,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or VQALoss(loss_type=LOSS_BCE)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        self.eval_metric_fn = eval_metric_fn
        self.log_every = log_every
        # text_encoder (FrozenTextEncoder) dùng cho EXP fusion models (Phương án B).
        # Nếu None → dùng legacy BLIP2VQA pipeline (pixel_values + input_ids).
        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.eval()  # BERT luôn ở eval mode

        # wandb_run: đối tượng wandb.Run hoặc None (không bắt buộc cài wandb)
        self.wandb_run = wandb_run

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

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Huấn luyện từ *start_epoch+1* đến *num_epochs*.

        Args:
            num_epochs:   Tổng số epoch cần huấn luyện.
            start_epoch:  Epoch đã hoàn thành trước đó (từ checkpoint).
                          Train sẽ bắt đầu từ ``start_epoch + 1``.
                          Nếu ``start_epoch >= num_epochs`` thì không làm gì.
        """
        if start_epoch >= num_epochs:
            logger.info("start_epoch=%d >= num_epochs=%d — không cần train thêm.",
                        start_epoch, num_epochs)
            return

        logger.info("Bắt đầu từ epoch %d, train đến epoch %d.",
                    start_epoch + 1, num_epochs)

        for epoch in range(start_epoch + 1, num_epochs + 1):
            logger.info("=== Epoch %d / %d ===", epoch, num_epochs)
            train_loss = self._train_epoch(epoch)
            val_results = self._val_epoch(epoch)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f "
                "(yes/no=%.4f  number=%.4f  other=%.4f)",
                epoch,
                train_loss,
                val_results[KEY_LOSS],
                val_results.get(KEY_OVERALL_ACC, 0.0),
                val_results.get(KEY_YESNO_ACC, 0.0),
                val_results.get(KEY_NUMBER_ACC, 0.0),
                val_results.get(KEY_OTHER_ACC, 0.0),
            )

            # Log epoch-level metrics lên W&B
            if self.wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train/loss_epoch":  train_loss,
                    "val/loss":          val_results[KEY_LOSS],
                    "val/acc":           val_results.get(KEY_OVERALL_ACC, 0.0),
                    "val/acc_yesno":     val_results.get(KEY_YESNO_ACC, 0.0),
                    "val/acc_number":    val_results.get(KEY_NUMBER_ACC, 0.0),
                    "val/acc_other":     val_results.get(KEY_OTHER_ACC, 0.0),
                }
                self.wandb_run.log(log_dict, step=self.global_step)

            # Lưu checkpoint sau mỗi epoch
            metric = val_results.get("metric", -val_results[KEY_LOSS])
            self._save_checkpoint(epoch, metric)

    def evaluate(self) -> EvalResult:
        """Chạy một lượt đánh giá trên tập validation.

        Returns:
            Dict có ít nhất khoá ``"loss"`` và tuỳ chọn ``"metric"``;
            xem :class:`configs.contracts.EvalResult`.
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

                # Bước scheduler theo gradient step để cosine decay hoạt động đúng
                # (T_max = num_training_steps — không phải num_epochs)
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.global_step % self.log_every == 0:
                    elapsed = time.time() - t0
                    cur_loss = loss.item() * self.gradient_accumulation_steps
                    cur_lr   = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Step %d | loss=%.4f | lr=%.2e | %.1f steps/s",
                        self.global_step,
                        cur_loss,
                        cur_lr,
                        self.log_every / elapsed,
                    )
                    # Log step-level metrics lên W&B
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "train/loss": cur_loss,
                                "train/lr":   cur_lr,
                                "train/steps_per_sec": self.log_every / elapsed,
                            },
                            step=self.global_step,
                        )
                    t0 = time.time()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / num_batches

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> EvalResult:
        self.model.eval()
        total_loss = 0.0

        # Accumulators for VQA accuracy (soft-target per-type)
        type_scores: Dict[str, list] = {"yes/no": [], "number": [], "other": []}
        all_scores: list = []

        for batch in tqdm(self.val_loader, desc=f"Val Epoch {epoch}", leave=False):
            loss = self._forward_batch(batch)
            total_loss += loss.item()

            # ── Compute soft VQA accuracy in same pass ────────────────────
            answer_scores = batch.get(KEY_ANSWER_SCORES)  # [B, num_answers]
            answer_types  = batch.get(KEY_ANSWER_TYPE)    # List[str] or None

            if answer_scores is not None:
                # Get logits to find predicted class
                if self.text_encoder is not None:
                    visual_features = batch[KEY_IMAGE_FEATURES].to(self.device)
                    input_ids       = batch[KEY_INPUT_IDS].to(self.device)
                    attention_mask  = batch.get(KEY_ATTENTION_MASK)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    text_features = self.text_encoder(input_ids, attention_mask)
                    logits = self.model(visual_features, text_features)
                else:
                    pixel_values = batch[KEY_PIXEL_VALUES].to(self.device)
                    input_ids = batch.get(KEY_INPUT_IDS)
                    if input_ids is not None:
                        input_ids = input_ids.to(self.device)
                    attention_mask = batch.get(KEY_ATTENTION_MASK)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    out = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = out.get(KEY_LOGITS, out) if isinstance(out, dict) else out

                pred_idx = logits.argmax(dim=-1).cpu()  # [B]
                scores_cpu = answer_scores.cpu()        # [B, num_answers]

                # Soft VQA acc per sample = answer_scores[b, pred_idx[b]]
                batch_accs = scores_cpu[
                    torch.arange(len(pred_idx)), pred_idx
                ].tolist()

                all_scores.extend(batch_accs)

                if answer_types is not None:
                    for acc, atype in zip(batch_accs, answer_types):
                        bucket = atype if atype in type_scores else "other"
                        type_scores[bucket].append(acc)

        num_batches = max(len(self.val_loader), 1)
        results: EvalResult = {KEY_LOSS: total_loss / num_batches}  # type: ignore[misc]

        if all_scores:
            overall = sum(all_scores) / len(all_scores)
            results[KEY_OVERALL_ACC] = overall
            results["metric"] = overall  # used by checkpoint logic
            results[KEY_YESNO_ACC] = (
                sum(type_scores["yes/no"]) / len(type_scores["yes/no"])
                if type_scores["yes/no"] else 0.0
            )
            results[KEY_NUMBER_ACC] = (
                sum(type_scores["number"]) / len(type_scores["number"])
                if type_scores["number"] else 0.0
            )
            results[KEY_OTHER_ACC] = (
                sum(type_scores["other"]) / len(type_scores["other"])
                if type_scores["other"] else 0.0
            )

        return results

    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """Chuyển batch lên device và tính loss.

        Hỗ trợ hai pipeline:
          - **EXP pipeline** (khi ``self.text_encoder`` khác None):
            Dùng ``KEY_IMAGE_FEATURES`` (HDF5 cache) + BERT → ``text_features``
            → gọi ``model(visual_features, text_features)`` → tensor logits.
          - **Legacy BLIP2VQA pipeline** (khi ``self.text_encoder`` là None):
            Dùng ``KEY_PIXEL_VALUES`` + ``input_ids`` truyền thẳng vào model.
        """
        answer_scores = batch.get(KEY_ANSWER_SCORES)
        if answer_scores is not None:
            answer_scores = answer_scores.to(self.device)
        answer_label = batch.get(KEY_ANSWER_LABEL)
        if answer_label is not None:
            answer_label = answer_label.to(self.device)

        ctx = (
            torch.cuda.amp.autocast()
            if self.scaler is not None
            else contextlib.nullcontext()
        )

        if self.text_encoder is not None:
            # --- EXP fusion model pipeline ---
            visual_features = batch[KEY_IMAGE_FEATURES].to(self.device)  # [B, 257, 1024]
            input_ids       = batch[KEY_INPUT_IDS].to(self.device)
            attention_mask  = batch.get(KEY_ATTENTION_MASK)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # BERT luôn chạy ngoài AMP (tránh precision issues với BertLayerNorm)
            # và trong no_grad (frozen — không cần lưu activation)
            text_features = self.text_encoder(input_ids, attention_mask)  # [B, 768]

            with ctx:
                logits = self.model(visual_features, text_features)  # [B, num_answers]

            targets = answer_scores if answer_scores is not None else answer_label
            if targets is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            return self.loss_fn(logits, targets)

        else:
            # --- Legacy BLIP2VQA pipeline ---
            pixel_values = batch[KEY_PIXEL_VALUES].to(self.device)
            input_ids = batch.get(KEY_INPUT_IDS)
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            attention_mask = batch.get(KEY_ATTENTION_MASK)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with ctx:
                out = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    answer_scores=answer_scores,
                )

            if KEY_LOSS in out:
                return out[KEY_LOSS]
            if KEY_LOGITS in out:
                targets = answer_scores if answer_scores is not None else answer_label
                if targets is None:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                return self.loss_fn(out[KEY_LOGITS], targets)

            return torch.tensor(0.0, device=self.device, requires_grad=True)

    @torch.no_grad()
    def _get_predictions(self, batch: Dict):
        """Trả về danh sách index câu trả lời dự đoán cho một batch."""
        if self.text_encoder is not None:
            # EXP pipeline
            visual_features = batch[KEY_IMAGE_FEATURES].to(self.device)
            input_ids       = batch[KEY_INPUT_IDS].to(self.device)
            attention_mask  = batch.get(KEY_ATTENTION_MASK)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            text_features = self.text_encoder(input_ids, attention_mask)
            logits = self.model(visual_features, text_features)
            return logits.argmax(dim=-1).tolist()
        else:
            # Legacy BLIP2VQA pipeline
            pixel_values = batch[KEY_PIXEL_VALUES].to(self.device)
            input_ids = batch.get(KEY_INPUT_IDS)
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            attention_mask = batch.get(KEY_ATTENTION_MASK)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            out = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if KEY_LOGITS in out:
                return out[KEY_LOGITS].argmax(dim=-1).tolist()
            return []

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        state = {
            KEY_EPOCH: epoch,
            KEY_GLOBAL_STEP: self.global_step,
            KEY_MODEL_STATE: self.model.state_dict(),
            KEY_OPTIM_STATE: self.optimizer.state_dict(),
            KEY_BEST_VAL_METRIC: metric,
        }
        if self.scheduler is not None:
            state[KEY_SCHED_STATE] = self.scheduler.state_dict()

        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(state, path)
        logger.info("Lưu checkpoint: %s", path)

        # Cập nhật best model nếu metric cải thiện
        if metric > self.best_val_metric:
            self.best_val_metric = metric
            best_path = self.output_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info("Best model mới được lưu (metric=%.4f)", metric)
            # Đánh dấu best checkpoint lên W&B
            if self.wandb_run is not None:
                self.wandb_run.summary["best_val_metric"] = metric
                self.wandb_run.summary["best_epoch"]      = epoch

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Tải checkpoint và khôi phục trạng thái model / optimizer.

        Args:
            checkpoint_path: Đường dẫn tới file checkpoint ``.pth``.

        Returns:
            Epoch tại thời điểm lưu checkpoint.
        """
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state[KEY_MODEL_STATE])
        self.optimizer.load_state_dict(state[KEY_OPTIM_STATE])
        if self.scheduler is not None and KEY_SCHED_STATE in state:
            self.scheduler.load_state_dict(state[KEY_SCHED_STATE])
        self.global_step = state.get(KEY_GLOBAL_STEP, 0)
        self.best_val_metric = state.get(KEY_BEST_VAL_METRIC, float("-inf"))
        epoch = state.get(KEY_EPOCH, 0)
        logger.info("Tải checkpoint từ %s (epoch %d)", checkpoint_path, epoch)
        return epoch
