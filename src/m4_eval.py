from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# RAGAS >= 0.2 đổi tên cột dataset; map về tên "cũ" mà lab đang dùng.
_COLUMN_ALIASES = {
    "question": ("question", "user_input"),
    "answer": ("answer", "response"),
    "contexts": ("contexts", "retrieved_contexts"),
    "ground_truth": ("ground_truth", "reference"),
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "answer_relevancy_score", "response_relevancy"),
    "context_precision": ("context_precision", "llm_context_precision_with_reference"),
    "context_recall": ("context_recall", "llm_context_recall"),
}


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pick(row, key, default=None):
    """Lấy giá trị theo tên cột, chấp nhận cả tên cũ lẫn tên mới của RAGAS.

    Lưu ý: cột `contexts` trong DataFrame của RAGAS là numpy array, nên tuyệt đối
    không dùng `value or default` ở đây — numpy sẽ ném "truth value of an array
    with more than one element is ambiguous".
    """
    for alias in _COLUMN_ALIASES.get(key, (key,)):
        if alias in row:
            value = row[alias]
            if value is None:
                return default
            # NaN xuất hiện khi 1 metric fail riêng lẻ trên 1 câu hỏi.
            if isinstance(value, float) and value != value:
                return default
            return value
    return default


def _to_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if number != number else number  # loại NaN


def _dump_eval_inputs(questions, answers, contexts, ground_truths,
                      path: str = "eval_inputs.json") -> None:
    """Lưu lại (question, answer, contexts, ground_truth) trước khi gọi RAGAS."""
    try:
        os.makedirs("reports", exist_ok=True)
        with open(os.path.join("reports", path), "w", encoding="utf-8") as f:
            json.dump([
                {"question": q, "answer": a, "contexts": c, "ground_truth": g}
                for q, a, c, g in zip(questions, answers, contexts, ground_truths)
            ], f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ⚠️  Không lưu được eval inputs: {e}")


def _empty_result() -> dict:
    return {**{m: 0.0 for m in METRIC_NAMES}, "per_question": []}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    4 metric chia làm 2 nhóm:
      - Generation: faithfulness (câu trả lời có bịa so với context không),
        answer_relevancy (câu trả lời có đúng trọng tâm câu hỏi không).
      - Retrieval: context_precision (context lấy về có bị nhiễu không),
        context_recall (context có đủ thông tin để trả lời không).
    Toàn bộ được bọc trong try/except vì RAGAS cần OPENAI_API_KEY và có thể
    fail vì lý do môi trường — pipeline không được sập chỉ vì eval lỗi.
    """
    # Dump input trước khi eval: retrieval + LLM answer là phần tốn tiền/thời gian
    # nhất, nếu RAGAS lỗi thì vẫn còn dữ liệu để chấm lại mà không phải chạy lại.
    _dump_eval_inputs(questions, answers, contexts, ground_truths)

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)

        payload = {
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        }
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            # RAGAS >= 0.2: EvaluationDataset + tên cột mới.
            from ragas import EvaluationDataset

            dataset = EvaluationDataset.from_list([
                {"user_input": q, "response": a, "retrieved_contexts": c, "reference": g}
                for q, a, c, g in zip(questions, answers, contexts, ground_truths)
            ])
        except ImportError:
            # RAGAS 0.1.x: Dataset của HuggingFace + tên cột cũ.
            from datasets import Dataset
            dataset = Dataset.from_dict(payload)

        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=str(_pick(row, "question", "")),
                answer=str(_pick(row, "answer", "")),
                contexts=[str(c) for c in _pick(row, "contexts", [])],
                ground_truth=str(_pick(row, "ground_truth", "")),
                faithfulness=_to_float(_pick(row, "faithfulness", 0.0)),
                answer_relevancy=_to_float(_pick(row, "answer_relevancy", 0.0)),
                context_precision=_to_float(_pick(row, "context_precision", 0.0)),
                context_recall=_to_float(_pick(row, "context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        if not per_question:
            return _empty_result()

        aggregate = {
            metric: round(sum(getattr(r, metric) for r in per_question) / len(per_question), 4)
            for metric in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return _empty_result()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Với mỗi câu hỏi tệ nhất, ta xác định *metric thấp nhất* rồi tra Diagnostic
    Tree để biết lỗi nằm ở khâu nào của pipeline (generation hay retrieval) và
    fix tương ứng — thay vì đoán mò "model kém".
    """
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời chứa thông tin không có trong context",
                         "Siết prompt ('chỉ dùng context'), hạ temperature, buộc trích dẫn nguồn"),
        "context_recall": ("Missing relevant chunks — retrieval bỏ sót đoạn chứa đáp án",
                           "Cải thiện chunking (hierarchical/structure), thêm BM25, tăng top_k"),
        "context_precision": ("Too many irrelevant chunks — context bị nhiễu, đáp án bị chôn",
                              "Thêm reranking cross-encoder hoặc metadata filter theo version/category"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lạc trọng tâm",
                             "Cải thiện prompt template, yêu cầu trả lời trực tiếp câu hỏi"),
    }

    scored = []
    for result in eval_results:
        metric_scores = {m: getattr(result, m) for m in METRIC_NAMES}
        avg = sum(metric_scores.values()) / len(metric_scores)
        worst_metric = min(metric_scores, key=lambda m: metric_scores[m])
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metric_scores[worst_metric], 4),
            "metrics": {m: round(v, 4) for m, v in metric_scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda item: item["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
