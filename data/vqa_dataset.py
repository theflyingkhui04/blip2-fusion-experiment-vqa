"""VQAv2 Dataset — canonical merged version.

Base: Sprint-2 VQAv2Dataset (Task 11+12), extended with:
  - Answer normalisation helpers (used by evaluator)
  - Soft-target answer score computation (_get_answer_scores)
  - Optional pre-built answer-vocab loading from answer_list.json
  - Graceful blank-image fallback instead of zero-tensor
  - All output keys aligned with configs/contracts.py

COCO 2014 filenames: COCO_{train,val}2014_000000XXXXXX.jpg

Usage (config-object interface, primary):
    from data.vqa_dataset import VQAv2Dataset, build_dataloader
    loader = build_dataloader("train", config, use_cache=True)

Usage (legacy alias):
    from data.vqa_dataset import VQADataset, build_vqa_dataloader
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from configs.contracts import (
    ANSWER_VOCAB_SIZE,
    IMAGE_SIZE,
    KEY_ANSWER_LABEL,
    KEY_ANSWER_SCORES,
    KEY_ANSWER_TEXT,
    KEY_ANSWER_TYPE,
    KEY_ANSWERS,
    KEY_ATTENTION_MASK,
    KEY_IMAGE_FEATURES,
    KEY_IMAGE_IDS,
    KEY_INPUT_IDS,
    KEY_PIXEL_VALUES,
    KEY_QUESTION_IDS,
    KEY_QUESTION_TEXT,
    MAX_QUESTION_LENGTH,
    VQA_SCORE_DENOMINATOR,
)


# ---------------------------------------------------------------------------
# Answer normalisation (mirrors the official VQA evaluation script)
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "thisheres": "this here's",
    "tod": "to'd", "tod've": "to'd've", "to'dve": "to'd've", "tome": "to me",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "were": "we're", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}

_ARTICLES = {"a", "an", "the"}
_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or text[-1] == p:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return out.strip()


def _process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _CONTRACTIONS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)


def normalize_answer(answer: str) -> str:
    """Normalise a VQA answer string to canonical form."""
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = _process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer


# ---------------------------------------------------------------------------
# Answer vocabulary builder
# ---------------------------------------------------------------------------


def build_answer_vocab(annotation_file: str, top_k: int = ANSWER_VOCAB_SIZE) -> Dict[str, int]:
    """Count answer frequencies from a VQAv2 annotations JSON; keep top_k.

    Answers are normalised before counting so the vocab is canonical.

    Args:
        annotation_file: Path to VQAv2 annotations JSON (train split recommended).
        top_k: Vocabulary size (default ANSWER_VOCAB_SIZE = 3129).

    Returns:
        ``{normalised_answer: index}`` ordered by descending frequency.
    """
    with open(annotation_file) as f:
        data = json.load(f)
    counter: Counter = Counter()
    for ann in data["annotations"]:
        for a in ann["answers"]:
            counter[normalize_answer(a["answer"])] += 1
    return {ans: idx for idx, (ans, _) in enumerate(counter.most_common(top_k))}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


class VQAv2Dataset(Dataset):
    """VQAv2 dataset supporting HDF5 pre-extracted feature cache and raw COCO images.

    Primary interface — takes a config object:

        dataset = VQAv2Dataset("train", config, use_cache=True)

    Config object must expose ``config.data.*`` with the keys defined in
    :class:`configs.contracts.DataConfig` (Sprint-2 style).

    Args:
        split:     ``"train"`` or ``"val"``.
        config:    Config object with a ``.data`` attribute (OmegaConf or similar).
        use_cache: Load pre-extracted HDF5 features instead of raw images.
                   Cache must be created first with ``scripts/pre_extract_features.py``.
    """

    #: Mapping from split name to file/directory metadata.
    SPLIT_FILES = {
        "train": {
            "ann":        "v2_mscoco_train2014_annotations.json",
            "ques":       "v2_OpenEnded_mscoco_train2014_questions.json",
            "img_dir":    "train2014",
            "img_prefix": "COCO_train2014_",
            "cache":      "train_features.h5",
        },
        "val": {
            "ann":        "v2_mscoco_val2014_annotations.json",
            "ques":       "v2_OpenEnded_mscoco_val2014_questions.json",
            "img_dir":    "val2014",
            "img_prefix": "COCO_val2014_",
            "cache":      "val_features.h5",
        },
    }

    def __init__(self, split: str, config, use_cache: bool = True) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"
        self.split = split
        self.use_cache = use_cache

        cfg = config.data
        data_root = cfg.data_root
        meta = self.SPLIT_FILES[split]

        ann_path  = os.path.join(data_root, cfg.vqav2_dir, meta["ann"])
        ques_path = os.path.join(data_root, cfg.vqav2_dir, meta["ques"])
        self.image_dir  = os.path.join(data_root, cfg.coco_dir, meta["img_dir"])
        self.img_prefix = meta["img_prefix"]
        self._img_size  = int(getattr(cfg, "image_size", IMAGE_SIZE))

        # ── Load questions + annotations, merge into self.samples ────────────
        with open(ann_path) as f:
            ann_data = json.load(f)
        with open(ques_path) as f:
            ques_data = json.load(f)

        qid2question = {q["question_id"]: q["question"] for q in ques_data["questions"]}
        self.samples: List[Dict] = []
        for ann in ann_data["annotations"]:
            qid = ann["question_id"]
            question = qid2question.get(qid, "")
            if not question:
                continue
            raw_answers = [a["answer"] for a in ann["answers"]]
            self.samples.append({
                "question_id": qid,
                "image_id":    ann["image_id"],
                "question":    question,
                "answers":     raw_answers,
                "answer":      ann.get("multiple_choice_answer",
                                       raw_answers[0] if raw_answers else ""),
                "answer_type": ann.get("answer_type", "other"),
            })

        # ── Stratified subset selection ──────────────────────────────────────
        subset_size = int(cfg.train_size if split == "train" else cfg.val_size)
        seed = int(getattr(cfg, "seed", 42))
        indices = VQAv2Dataset._stratified_indices(self.samples, subset_size, seed)
        self.samples = [self.samples[i] for i in indices]

        # ── Image transform ──────────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.Resize((self._img_size, self._img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        # ── Answer vocabulary ────────────────────────────────────────────────
        # Priority 1: pre-built answer_list file from config (if present)
        # Priority 2: build dynamically from this split's annotation file
        answer_list_path = getattr(cfg, "answer_list", None)
        if answer_list_path and os.path.exists(str(answer_list_path)):
            with open(answer_list_path) as f:
                raw_vocab = json.load(f)
            if isinstance(raw_vocab, dict):
                # {answer: index} — preserve original indices (may be non-contiguous)
                self.answer_to_idx: Dict[str, int] = {
                    normalize_answer(a): int(idx) for a, idx in raw_vocab.items()
                }
            else:
                # list of answers — assign indices 0..N-1
                self.answer_to_idx = {
                    normalize_answer(a): i for i, a in enumerate(raw_vocab)
                }
            self.idx_to_answer: List[str] = [""] * ANSWER_VOCAB_SIZE
            for ans, idx in self.answer_to_idx.items():
                if idx < ANSWER_VOCAB_SIZE:
                    self.idx_to_answer[idx] = ans
        else:
            self.answer_to_idx = build_answer_vocab(ann_path, top_k=ANSWER_VOCAB_SIZE)
            self.idx_to_answer = [""] * ANSWER_VOCAB_SIZE
            for ans, idx in self.answer_to_idx.items():
                self.idx_to_answer[idx] = ans

        # ── HDF5 cache ───────────────────────────────────────────────────────
        self._h5 = None
        if use_cache:
            self._h5_path = os.path.join(data_root, cfg.cache_dir, meta["cache"])
            if not os.path.exists(self._h5_path):
                raise FileNotFoundError(
                    f"Cache not found: {self._h5_path}\n"
                    "Run scripts/pre_extract_features.py first, or set use_cache=False."
                )
            # Filter samples to only those whose image_id exists in the cache.
            # This handles partial caches (e.g. smoke-test extractions) gracefully
            # instead of crashing with KeyError mid-training.
            with h5py.File(self._h5_path, "r") as _h5:
                cached_ids = set(_h5.keys())  # set of str(image_id)
            before = len(self.samples)
            self.samples = [s for s in self.samples if str(s["image_id"]) in cached_ids]
            dropped = before - len(self.samples)
            if dropped:
                print(f"[VQAv2Dataset/{split}] WARNING: {dropped:,} samples dropped "
                      f"(image_id not in cache). Cache coverage: "
                      f"{len(self.samples):,}/{before:,}.")

        print(f"[VQAv2Dataset/{split}] {len(self.samples):,} samples | "
              f"vocab={len(self.answer_to_idx):,} | use_cache={use_cache}")

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        item: Dict = {
            "question_id": s["question_id"],
            "image_id":    s["image_id"],
            "question":    s["question"],
            KEY_ANSWERS:   s["answers"],        # List[str] — all 10 raw answers
            KEY_ANSWER_TEXT: s["answer"],       # str       — most common answer
            KEY_ANSWER_TYPE: s["answer_type"],  # str       — "yes/no"|"number"|"other"
        }

        # ── Image / features ─────────────────────────────────────────────────
        if self.use_cache:
            if self._h5 is None:                        # lazy open (DataLoader fork-safe)
                self._h5 = h5py.File(self._h5_path, "r")
            item[KEY_IMAGE_FEATURES] = torch.from_numpy(
                self._h5[str(s["image_id"])][()].astype("float32")
            )
        else:
            p = os.path.join(self.image_dir,
                             f"{self.img_prefix}{s['image_id']:012d}.jpg")
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
            else:
                img = Image.new("RGB", (self._img_size, self._img_size), color=128)
            item[KEY_PIXEL_VALUES] = self.transform(img)

        # ── Soft-target answer scores + hard label ───────────────────────────
        if self.answer_to_idx:
            item[KEY_ANSWER_SCORES] = self._get_answer_scores(s["answers"])
            label_idx = self.answer_to_idx.get(normalize_answer(s["answer"]), -1)
            item[KEY_ANSWER_LABEL] = torch.tensor(label_idx, dtype=torch.long)

        return item

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_answer_scores(self, raw_answers: List[str]) -> torch.Tensor:
        """Compute soft-target VQA score vector from raw answer strings.

        Score formula: ``min(count / VQA_SCORE_DENOMINATOR, 1.0)`` per answer.

        Args:
            raw_answers: List of up to 10 raw answer strings for one question.

        Returns:
            Float tensor of shape ``[ANSWER_VOCAB_SIZE]`` with values in ``[0, 1]``.
        """
        scores = torch.zeros(ANSWER_VOCAB_SIZE, dtype=torch.float)
        answer_counts: Dict[str, int] = {}
        for ans in raw_answers:
            norm = normalize_answer(ans)
            answer_counts[norm] = answer_counts.get(norm, 0) + 1
        for ans, count in answer_counts.items():
            if ans in self.answer_to_idx:
                scores[self.answer_to_idx[ans]] = min(
                    count / VQA_SCORE_DENOMINATOR, 1.0
                )
        return scores

    @staticmethod
    def _stratified_indices(
        samples: List[Dict], subset_size: int, seed: int = 42
    ) -> List[int]:
        """Sample ``subset_size`` indices with stratification by ``answer_type``.

        Each answer type gets a proportional share of the subset.  Remaining
        slots (rounding differences) are filled randomly.

        Args:
            samples:     Full sample list with ``"answer_type"`` key.
            subset_size: Desired number of samples.
            seed:        Random seed for reproducibility.

        Returns:
            List of integer indices into ``samples``.
        """
        rng = random.Random(seed)
        type2idx: Dict[str, List[int]] = {}
        for i, s in enumerate(samples):
            type2idx.setdefault(s["answer_type"], []).append(i)

        total = len(samples)
        selected: List[int] = []
        for indices in type2idx.values():
            n = int(round(len(indices) / total * subset_size))
            rng.shuffle(indices)
            selected.extend(indices[:n])

        if len(selected) > subset_size:
            rng.shuffle(selected)
            selected = selected[:subset_size]
        elif len(selected) < subset_size:
            remaining = list(set(range(total)) - set(selected))
            rng.shuffle(remaining)
            selected.extend(remaining[: subset_size - len(selected)])

        return selected


# ---------------------------------------------------------------------------
# Tokenizer (module-level singleton, DataLoader worker-safe)
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """Return a cached BERT tokenizer (loaded once per process)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# ---------------------------------------------------------------------------
# collate_fn  — keys aligned with configs/contracts.py
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of :meth:`VQAv2Dataset.__getitem__` dicts into a batch.

    Key mapping (all names from contracts.KEY_*):
        input_ids      — tokenized question ids      [B, MAX_QUESTION_LENGTH]
        attention_mask — padding mask                [B, MAX_QUESTION_LENGTH]
        question_text  — raw question strings        List[str]
        question_ids   — VQAv2 question_id           List[int]
        image_ids      — COCO image_id               List[int]
        answer         — most common answer          List[str]
        answers        — all 10 raw answers          List[List[str]]
        answer_type    — answer type string          List[str]
        pixel_values   — raw image tensor            [B, 3, H, W]   (use_cache=False)
        image_features — pre-extracted features      [B, N, D]      (use_cache=True)
        answer_scores  — soft-target VQA scores      [B, ANSWER_VOCAB_SIZE]
        answer_label   — hard label index            [B]
    """
    tokenizer = get_tokenizer()
    questions = [b["question"] for b in batch]
    enc = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=MAX_QUESTION_LENGTH,
        return_tensors="pt",
    )

    result: Dict = {
        KEY_INPUT_IDS:      enc["input_ids"],
        KEY_ATTENTION_MASK: enc["attention_mask"],
        KEY_QUESTION_TEXT:  questions,
        KEY_QUESTION_IDS:   [b["question_id"]     for b in batch],
        KEY_IMAGE_IDS:      [b["image_id"]        for b in batch],
        KEY_ANSWER_TEXT:    [b[KEY_ANSWER_TEXT]   for b in batch],
        KEY_ANSWERS:        [b[KEY_ANSWERS]       for b in batch],
        KEY_ANSWER_TYPE:    [b[KEY_ANSWER_TYPE]   for b in batch],
    }

    # Exactly one of image_features / pixel_values is present per batch
    if KEY_IMAGE_FEATURES in batch[0]:
        result[KEY_IMAGE_FEATURES] = torch.stack([b[KEY_IMAGE_FEATURES] for b in batch])
    else:
        result[KEY_PIXEL_VALUES] = torch.stack([b[KEY_PIXEL_VALUES] for b in batch])

    # Optional: answer supervision tensors (absent when vocab not loaded)
    if KEY_ANSWER_SCORES in batch[0]:
        result[KEY_ANSWER_SCORES] = torch.stack([b[KEY_ANSWER_SCORES] for b in batch])
    if KEY_ANSWER_LABEL in batch[0]:
        result[KEY_ANSWER_LABEL] = torch.stack([b[KEY_ANSWER_LABEL] for b in batch])

    return result


# ---------------------------------------------------------------------------
# DataLoader factory (primary)
# ---------------------------------------------------------------------------

def build_dataloader(
    split: str,
    config,
    use_cache: bool = True,
) -> DataLoader:
    """Build a stratified DataLoader for a VQAv2 split.

    Args:
        split:     ``"train"`` or ``"val"``.
        config:    Config object with ``config.data.*`` attributes.
        use_cache: Use pre-extracted HDF5 features (requires cache file).

    Returns:
        A :class:`torch.utils.data.DataLoader` with :func:`collate_fn` applied.
    """
    dataset = VQAv2Dataset(split, config, use_cache=use_cache)
    return DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        shuffle=(split == "train"),
        num_workers=int(getattr(config.data, "num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

#: Alias for users/scripts that import the original class name.
VQADataset = VQAv2Dataset

#: Alias kept for scripts that import ``build_vqa_dataloader``.
build_vqa_dataloader = build_dataloader


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "data": {
            "data_root":  "/content/data",
            "vqav2_dir":  "vqav2",
            "coco_dir":   "coco",
            "cache_dir":  "cache",
            "train_size": 512,
            "val_size":   128,
            "image_size": IMAGE_SIZE,
            "seed":       42,
            "batch_size": 8,
            "num_workers": 0,
        },
        "model": {
            "image_encoder": "openai/clip-vit-large-patch14",
        },
    })

    loader = build_dataloader("train", cfg, use_cache=False)
    batch  = next(iter(loader))
    print("input_ids:    ", batch[KEY_INPUT_IDS].shape)
    print("pixel_values: ", batch[KEY_PIXEL_VALUES].shape)
    print("question_ids: ", batch[KEY_QUESTION_IDS][:3])
    print("answer_scores:", batch[KEY_ANSWER_SCORES].shape)
