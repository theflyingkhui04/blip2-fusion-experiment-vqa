"""BLIP-2 VQA model.

This module wraps the HuggingFace ``Blip2ForConditionalGeneration`` model for
Visual Question Answering while also exposing a classification head that can
be used with a fixed answer vocabulary (as in VQAv2 evaluation).

The architecture follows the BLIP-2 pipeline:
    1. Frozen ViT vision encoder → visual patch features.
    2. Q-Former → compact visual query representations.
    3. Linear projection → language-model embedding space.
    4. Frozen (or fine-tuned) language model → answer tokens.

For classification experiments (``mode="classify"``), a linear head is placed
on top of the Q-Former output instead.

Reference:
    Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
    Image Encoders and Large Language Models", ICML 2023.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

try:
    from transformers import (
        Blip2ForConditionalGeneration,
        Blip2Processor,
        AutoTokenizer,
    )
    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False

from models.qformer import QFormer, QFormerConfig


class BLIP2VQA(nn.Module):
    """BLIP-2 model adapted for Visual Question Answering.

    Two operating modes are supported:

    * ``"generate"`` – uses the full BLIP-2 generative pipeline (frozen ViT +
      Q-Former + language model) to produce free-form answer strings.
    * ``"classify"`` – uses only the vision encoder and a lightweight Q-Former
      to produce logits over a fixed answer vocabulary.  This mode does **not**
      require a large language model and is much cheaper to fine-tune.

    Args:
        model_name: HuggingFace model identifier for the BLIP-2 checkpoint.
        num_answers: Number of answer classes (used in ``"classify"`` mode).
        mode: ``"generate"`` or ``"classify"``.
        freeze_vision_encoder: Whether to freeze the vision encoder weights.
        freeze_qformer: Whether to freeze Q-Former weights.
        qformer_config: Optional :class:`~models.qformer.QFormerConfig`; if
            ``None`` a default config is used.  Ignored when loading a
            pretrained HuggingFace checkpoint (the pretrained Q-Former is
            used directly).
        max_answer_length: Maximum number of new tokens to generate (generate
            mode only).
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        num_answers: int = 3129,
        mode: str = "generate",
        freeze_vision_encoder: bool = True,
        freeze_qformer: bool = False,
        qformer_config: Optional[QFormerConfig] = None,
        max_answer_length: int = 10,
    ) -> None:
        super().__init__()

        if mode not in {"generate", "classify"}:
            raise ValueError(f"mode must be 'generate' or 'classify', got '{mode}'")

        self.mode = mode
        self.num_answers = num_answers
        self.max_answer_length = max_answer_length

        if _HF_AVAILABLE:
            self._init_from_pretrained(
                model_name,
                freeze_vision_encoder,
                freeze_qformer,
            )
        else:
            # Fall back to a lightweight custom Q-Former when transformers is
            # not installed (useful for unit tests).
            self._init_custom(qformer_config)

        if mode == "classify":
            hidden = self._hidden_size()
            self.classifier = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden, num_answers),
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_from_pretrained(
        self,
        model_name: str,
        freeze_vision: bool,
        freeze_qformer: bool,
    ) -> None:
        """Load a pretrained BLIP-2 checkpoint from HuggingFace."""
        self._blip2 = Blip2ForConditionalGeneration.from_pretrained(
            model_name, ignore_mismatched_sizes=True
        )
        if freeze_vision:
            for p in self._blip2.vision_model.parameters():
                p.requires_grad_(False)
        if freeze_qformer:
            for p in self._blip2.qformer.parameters():
                p.requires_grad_(False)
        self._use_hf = True

    def _init_custom(self, config: Optional[QFormerConfig]) -> None:
        """Build a lightweight custom Q-Former (no pretrained weights)."""
        if config is None:
            config = QFormerConfig()
        self._qformer = QFormer(config)
        self._use_hf = False

    def _hidden_size(self) -> int:
        if self._use_hf:
            return self._blip2.config.qformer_config.hidden_size
        return self._qformer.hidden_size

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        answer_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            pixel_values: Pre-processed image tensor ``[B, 3, H, W]``.
            input_ids: Tokenised question ids ``[B, L]`` (required for generate
                mode).
            attention_mask: Attention mask for *input_ids* ``[B, L]``.
            labels: Target token ids for teacher-forcing loss ``[B, L]``.
            answer_scores: Soft answer score vector ``[B, num_answers]`` used
                for BCE loss in classify mode.

        Returns:
            Dictionary with subset of:
            ``{"loss", "logits", "visual_features"}``.
        """
        if self._use_hf:
            return self._forward_hf(
                pixel_values, input_ids, attention_mask, labels, answer_scores
            )
        return self._forward_custom(pixel_values, answer_scores)

    def _forward_hf(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        answer_scores: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        if self.mode == "generate":
            hf_out = self._blip2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if labels is not None:
                outputs["loss"] = hf_out.loss
            outputs["logits"] = hf_out.logits

        else:  # classify
            # Extract Q-Former output from the vision side only
            vision_out = self._blip2.vision_model(pixel_values=pixel_values)
            image_embeds = vision_out.last_hidden_state   # [B, N, D]
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            qformer_out = self._blip2.qformer(
                query_embeds=self._blip2.query_tokens.expand(
                    image_embeds.shape[0], -1, -1
                ),
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
            )
            query_output = qformer_out.last_hidden_state[:, : self._blip2.query_tokens.shape[1], :]
            # Mean-pool over query tokens
            pooled = query_output.mean(dim=1)                # [B, hidden]
            logits = self.classifier(pooled)                 # [B, num_answers]
            outputs["logits"] = logits

            if answer_scores is not None:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, answer_scores
                )
                outputs["loss"] = loss

        return outputs

    def _forward_custom(
        self,
        pixel_values: torch.Tensor,
        answer_scores: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Custom forward when HuggingFace transformers is unavailable."""
        B = pixel_values.shape[0]
        # Dummy visual features (replace with a real ViT in production)
        visual_feats = torch.zeros(
            B, 256, self._qformer.config.vision_width, device=pixel_values.device
        )
        query_out = self._qformer(visual_feats)               # [B, Nq, hidden]
        pooled = query_out.mean(dim=1)                        # [B, hidden]

        outputs: Dict[str, torch.Tensor] = {
            "visual_features": pooled,
        }
        if self.mode == "classify":
            logits = self.classifier(pooled)
            outputs["logits"] = logits
            if answer_scores is not None:
                outputs["loss"] = nn.functional.binary_cross_entropy_with_logits(
                    logits, answer_scores
                )
        return outputs

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_answers(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> List[str]:
        """Generate answer strings for a batch of (image, question) pairs.

        Requires ``mode="generate"`` and the HuggingFace backend.

        Args:
            pixel_values: Pre-processed images ``[B, 3, H, W]``.
            input_ids: Tokenised questions ``[B, L]``.
            attention_mask: Padding mask ``[B, L]``.
            **generate_kwargs: Extra kwargs forwarded to
                :meth:`~transformers.Blip2ForConditionalGeneration.generate`.

        Returns:
            List of decoded answer strings.
        """
        if not self._use_hf:
            raise RuntimeError("generate_answers requires the HuggingFace backend.")

        generated = self._blip2.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_answer_length,
            **generate_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self._blip2.config._name_or_path
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def predict_answers(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        idx_to_answer: Optional[List[str]] = None,
    ) -> List[str]:
        """Predict the most likely answer class for each sample.

        In ``"generate"`` mode this delegates to :meth:`generate_answers`.
        In ``"classify"`` mode it returns the highest-scoring answer string.

        Args:
            pixel_values: Pre-processed images ``[B, 3, H, W]``.
            input_ids: Tokenised questions (generate mode).
            attention_mask: Padding mask (generate mode).
            idx_to_answer: List mapping class index → answer string (classify
                mode).

        Returns:
            List of predicted answer strings.
        """
        if self.mode == "generate":
            return self.generate_answers(pixel_values, input_ids, attention_mask)

        # classify mode
        out = self.forward(pixel_values, input_ids, attention_mask)
        pred_ids = out["logits"].argmax(dim=-1).tolist()
        if idx_to_answer:
            return [idx_to_answer[i] for i in pred_ids]
        return [str(i) for i in pred_ids]
