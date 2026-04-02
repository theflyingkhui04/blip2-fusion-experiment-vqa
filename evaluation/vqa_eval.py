"""Tiện ích đánh giá mô hình VQA.

Triển khai metric accuracy chính thức của VQAv2:

    accuracy = min(số_người_chọn_đáp_án / 3, 1.0)

cho từng câu hỏi, sau đó lấy trung bình trên toàn bộ dataset.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from configs.contracts import (
    KEY_ANSWER,
    KEY_ANSWER_TYPE,
    KEY_ANSWERS,
    KEY_NUMBER_ACC,
    KEY_OTHER_ACC,
    KEY_OVERALL_ACC,
    KEY_QUESTION_ID,
    KEY_YESNO_ACC,
    EvalResult,
    PredictionRecord,
)
from data.vqa_dataset import normalize_answer


class VQAEvaluator:
    """Đánh giá kết quả dự đoán của mô hình so với ground-truth VQAv2.

    Args:
        annotation_file: Đường dẫn tới file JSON annotations của VQAv2.
        question_file: Đường dẫn tới file JSON câu hỏi VQAv2 (tuỳ chọn;
            dùng để phân tích theo từng loại câu hỏi).
    """

    def __init__(
        self,
        annotation_file: str,
        question_file: Optional[str] = None,
    ) -> None:
        with open(annotation_file, "r") as f:
            ann_data = json.load(f)

        # Xây dựng mapping question_id → annotation
        self._annotations: Dict[int, Dict] = {
            ann[KEY_QUESTION_ID]: ann for ann in ann_data["annotations"]
        }

        # Metadata câu hỏi (tuỳ chọn)
        self._question_types: Dict[int, str] = {}
        if question_file is not None and Path(question_file).exists():
            with open(question_file, "r") as f:
                q_data = json.load(f)
            for q in q_data["questions"]:
                self._question_types[q[KEY_QUESTION_ID]] = q.get("question_type", "")

    # ------------------------------------------------------------------
    # Metric chính
    # ------------------------------------------------------------------

    def compute_accuracy(
        self,
        predictions: List[PredictionRecord],
    ) -> EvalResult:
        """Tính VQA accuracy từ danh sách các dự đoán.

        Args:
            predictions: Danh sách dict theo chuẩn :class:`configs.contracts.PredictionRecord`,
                mỗi phần tử bao gồm:
                * ``"question_id"`` (int)
                * ``"answer"`` (str) – chuỗi câu trả lời dự đoán.

        Returns:
            Dict accuracy tổng thể và theo từng loại câu hỏi theo chuẩn
            :class:`configs.contracts.EvalResult`::

                {
                    "overall": 62.34,
                    "yes/no":  80.12,
                    "number":  43.21,
                    "other":   51.23,
                }
        """
        if not predictions:
            return {"overall": 0.0}  # type: ignore[return-value]

        type_correct: Dict[str, float] = defaultdict(float)
        type_total: Dict[str, int] = defaultdict(int)
        overall_correct = 0.0

        for pred in predictions:
            qid = int(pred[KEY_QUESTION_ID])
            predicted_answer = normalize_answer(str(pred[KEY_ANSWER]))

            if qid not in self._annotations:
                continue

            ann = self._annotations[qid]
            answer_type = ann.get(KEY_ANSWER_TYPE, "other")

            # Đếm số annotator đã chọn đúng đáp án dự đoán
            human_answers = [
                normalize_answer(a[KEY_ANSWER]) for a in ann.get(KEY_ANSWERS, [])
            ]
            acc = min(human_answers.count(predicted_answer) / 3.0, 1.0)

            overall_correct += acc
            type_correct[answer_type] += acc
            type_total[answer_type] += 1

        overall = overall_correct / len(predictions)
        results: Dict[str, float] = {KEY_OVERALL_ACC: round(overall * 100, 2)}

        for ans_type, total in type_total.items():
            if total > 0:
                # Dùng key constant tương ứng với từng loại câu hỏi
                if ans_type == "yes/no":
                    results[KEY_YESNO_ACC] = round(type_correct[ans_type] / total * 100, 2)
                elif ans_type == "number":
                    results[KEY_NUMBER_ACC] = round(type_correct[ans_type] / total * 100, 2)
                else:
                    results[KEY_OTHER_ACC] = round(type_correct[ans_type] / total * 100, 2)

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------

    def evaluate_from_file(self, result_file: str) -> EvalResult:
        """Tải dự đoán từ file JSON và thực hiện đánh giá.

        File phải là một mảng JSON gồm các object có khoá ``"question_id"``
        và ``"answer"`` (định dạng chuẩn VQA result).

        Args:
            result_file: Đường dẫn tới file JSON chứa dự đoán.

        Returns:
            Dict accuracy; xem :meth:`compute_accuracy`.
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
        """Lưu dự đoán ra file JSON theo định dạng chuẩn VQA.

        Args:
            question_ids: Danh sách question id.
            answers: Chuỗi câu trả lời tương ứng với mỗi question id.
            output_file: Đường dẫn file đầu ra.
        """
        if len(question_ids) != len(answers):
            raise ValueError("question_ids và answers phải có cùng độ dài.")

        results = [
            {KEY_QUESTION_ID: int(qid), KEY_ANSWER: str(ans)}
            for qid, ans in zip(question_ids, answers)
        ]
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Tính điểm từng mẫu (dùng trong training để tính soft metric)
    # ------------------------------------------------------------------

    @staticmethod
    def score_answer(predicted: str, human_answers: List[str]) -> float:
        """Tính VQA accuracy cho một dự đoán đơn lẻ.

        Args:
            predicted: Câu trả lời dự đoán của mô hình.
            human_answers: Danh sách câu trả lời của annotator (tối đa 10).

        Returns:
            Giá trị float trong khoảng ``[0, 1]``.
        """
        pred_norm = normalize_answer(predicted)
        human_norms = [normalize_answer(a) for a in human_answers]
        return min(human_norms.count(pred_norm) / 3.0, 1.0)
