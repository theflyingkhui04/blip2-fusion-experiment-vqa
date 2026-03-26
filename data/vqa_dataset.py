"""VQA dataset loader for VQAv2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
# Dataset
# ---------------------------------------------------------------------------


class VQADataset(Dataset):
    """Dataset for VQAv2.

    Args:
        question_file: Path to VQAv2 questions JSON.
        annotation_file: Path to VQAv2 annotations JSON (optional for test split).
        image_dir: Directory containing COCO images.
        answer_list_file: Path to JSON file with list of answer vocab strings.
        transform: Image transform; defaults to a standard resize/normalize.
        max_question_length: Truncation length for questions.
        is_train: Whether this is a training split (enables answer sampling).
    """

    def __init__(
        self,
        question_file: str,
        annotation_file: Optional[str],
        image_dir: str,
        answer_list_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_question_length: int = 50,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.max_question_length = max_question_length
        self.is_train = is_train

        # Load questions
        with open(question_file, "r") as f:
            question_data = json.load(f)
        self.questions: List[Dict] = question_data["questions"]

        # Build question-id → question map
        self._qid_to_question: Dict[int, Dict] = {
            q["question_id"]: q for q in self.questions
        }

        # Load annotations (ground-truth answers) if provided
        self._annotations: Dict[int, Dict] = {}
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                ann_data = json.load(f)
            for ann in ann_data["annotations"]:
                self._annotations[ann["question_id"]] = ann

        # Load answer vocabulary
        self.answer_to_idx: Dict[str, int] = {}
        self.idx_to_answer: List[str] = []
        if answer_list_file and os.path.exists(answer_list_file):
            with open(answer_list_file, "r") as f:
                answers = json.load(f)
            self.idx_to_answer = [normalize_answer(a) for a in answers]
            self.answer_to_idx = {a: i for i, a in enumerate(self.idx_to_answer)}

        # Image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.questions)

    def _image_filename(self, image_id: int) -> Path:
        """Return the image file path for a given COCO image_id."""
        filename = f"COCO_{'train' if self.is_train else 'val'}2014_{image_id:012d}.jpg"
        path = self.image_dir / filename
        if not path.exists():
            # Fallback: bare id without split prefix
            path = self.image_dir / f"{image_id:012d}.jpg"
        return path

    def _get_answer_scores(self, question_id: int) -> torch.Tensor:
        """Return a soft-target score vector over the answer vocabulary."""
        scores = torch.zeros(len(self.answer_to_idx), dtype=torch.float)
        if question_id not in self._annotations:
            return scores
        ann = self._annotations[question_id]
        answer_counts: Dict[str, int] = {}
        for a in ann.get("answers", []):
            norm = normalize_answer(a["answer"])
            answer_counts[norm] = answer_counts.get(norm, 0) + 1
        for ans, count in answer_counts.items():
            if ans in self.answer_to_idx:
                # VQA accuracy: min(count / 3, 1)
                scores[self.answer_to_idx[ans]] = min(count / 3.0, 1.0)
        return scores

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict:
        question_info = self.questions[idx]
        question_id = question_info["question_id"]
        image_id = question_info["image_id"]
        question_text = question_info["question"]

        # Truncate question
        words = question_text.split()
        if len(words) > self.max_question_length:
            question_text = " ".join(words[: self.max_question_length])

        # Load image
        image_path = self._image_filename(image_id)
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
        else:
            # Return a blank image when file is not available (offline testing)
            image = Image.new("RGB", (224, 224), color=128)
        image = self.transform(image)

        item = {
            "question_id": question_id,
            "image_id": image_id,
            "image": image,
            "question": question_text,
        }

        # Add answer labels when annotations are available
        if self._annotations:
            ann = self._annotations.get(question_id, {})
            answer_type = ann.get("answer_type", "other")
            most_common = ann.get("multiple_choice_answer", "")
            item["answer_type"] = answer_type
            item["most_common_answer"] = normalize_answer(most_common)
            if self.answer_to_idx:
                item["answer_scores"] = self._get_answer_scores(question_id)
                label = self.answer_to_idx.get(normalize_answer(most_common), -1)
                item["answer_label"] = torch.tensor(label, dtype=torch.long)

        return item


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_vqa_dataloader(
    question_file: str,
    annotation_file: Optional[str],
    image_dir: str,
    answer_list_file: Optional[str] = None,
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    is_train: bool = True,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Convenience function to construct a :class:`VQADataset` DataLoader.

    Args:
        question_file: Path to the VQAv2 questions JSON file.
        annotation_file: Path to the VQAv2 annotations JSON (``None`` for test split).
        image_dir: Directory containing COCO images.
        answer_list_file: Path to the answer-vocabulary JSON.
        transform: Optional image transform.
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        is_train: Whether this split is used for training.
        shuffle: Whether to shuffle the dataset; defaults to ``is_train``.

    Returns:
        A PyTorch :class:`~torch.utils.data.DataLoader`.
    """
    dataset = VQADataset(
        question_file=question_file,
        annotation_file=annotation_file,
        image_dir=image_dir,
        answer_list_file=answer_list_file,
        transform=transform,
        is_train=is_train,
    )
    if shuffle is None:
        shuffle = is_train
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
