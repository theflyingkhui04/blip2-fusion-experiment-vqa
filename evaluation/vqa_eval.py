"""VQA evaluation utilities.

Implements the official VQAv2 accuracy metric:

    accuracy = min(#humans_who_gave_answer / 3, 1.0)

for each question, averaged over the dataset.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from data.vqa_dataset import normalize_answer


class VQAEvaluator:
    """Evaluate model predictions against VQAv2 ground-truth annotations.

    Args:
        annotation_file: Path to the VQAv2 annotations JSON.
        question_file: Path to the VQAv2 questions JSON (optional; used to
            populate per-question-type breakdowns).
    """

    def __init__(
        self,
        annotation_file: str,
        question_file: Optional[str] = None,
    ) -> None:
        with open(annotation_file, "r") as f:
            ann_data = json.load(f)

        # Build question_id → annotation mapping
        self._annotations: Dict[int, Dict] = {
            ann["question_id"]: ann for ann in ann_data["annotations"]
        }

        # Optional question metadata
        self._question_types: Dict[int, str] = {}
        if question_file is not None and Path(question_file).exists():
            with open(question_file, "r") as f:
                q_data = json.load(f)
            for q in q_data["questions"]:
                self._question_types[q["question_id"]] = q.get("question_type", "")

    # ------------------------------------------------------------------
    # Core metric
    # ------------------------------------------------------------------

    def compute_accuracy(
        self,
        predictions: List[Dict[str, Union[int, str]]],
    ) -> Dict[str, float]:
        """Compute VQA accuracy from a list of prediction dicts.

        Args:
            predictions: Each dict must contain:
                * ``"question_id"`` (int)
                * ``"answer"`` (str) – the predicted answer string.

        Returns:
            A dictionary with overall accuracy and per-type breakdowns::

                {
                    "overall": 0.6234,
                    "yes/no": 0.8012,
                    "number": 0.4321,
                    "other": 0.5123,
                }
        """
        if not predictions:
            return {"overall": 0.0}

        type_correct: Dict[str, float] = defaultdict(float)
        type_total: Dict[str, int] = defaultdict(int)
        overall_correct = 0.0

        for pred in predictions:
            qid = int(pred["question_id"])
            predicted_answer = normalize_answer(str(pred["answer"]))

            if qid not in self._annotations:
                continue

            ann = self._annotations[qid]
            answer_type = ann.get("answer_type", "other")

            # Count how many annotators gave the predicted answer
            human_answers = [
                normalize_answer(a["answer"]) for a in ann.get("answers", [])
            ]
            acc = min(human_answers.count(predicted_answer) / 3.0, 1.0)

            overall_correct += acc
            type_correct[answer_type] += acc
            type_total[answer_type] += 1

        overall = overall_correct / len(predictions)
        results: Dict[str, float] = {"overall": round(overall * 100, 2)}

        for ans_type, total in type_total.items():
            if total > 0:
                results[ans_type] = round(type_correct[ans_type] / total * 100, 2)

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def evaluate_from_file(self, result_file: str) -> Dict[str, float]:
        """Load predictions from a JSON file and evaluate.

        The file must be a JSON array of objects with ``"question_id"`` and
        ``"answer"`` keys (standard VQA result format).

        Args:
            result_file: Path to the predictions JSON file.

        Returns:
            Accuracy dict; see :meth:`compute_accuracy`.
        """
        with open(result_file, "r") as f:
            predictions = json.load(f)
        return self.compute_accuracy(predictions)

    def save_predictions(
        self,
        question_ids: List[int],
        answers: List[str],
        output_file: str,
    ) -> None:
        """Serialize predictions to a VQA-format JSON file.

        Args:
            question_ids: List of question ids.
            answers: Corresponding predicted answer strings.
            output_file: Destination path.
        """
        if len(question_ids) != len(answers):
            raise ValueError("question_ids and answers must have the same length.")

        results = [
            {"question_id": int(qid), "answer": str(ans)}
            for qid, ans in zip(question_ids, answers)
        ]
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Per-sample scoring (used during training for soft metrics)
    # ------------------------------------------------------------------

    @staticmethod
    def score_answer(predicted: str, human_answers: List[str]) -> float:
        """Compute VQA accuracy for a single prediction.

        Args:
            predicted: The model's predicted answer string.
            human_answers: List of human annotator answers (up to 10).

        Returns:
            Float in ``[0, 1]``.
        """
        pred_norm = normalize_answer(predicted)
        human_norms = [normalize_answer(a) for a in human_answers]
        return min(human_norms.count(pred_norm) / 3.0, 1.0)
