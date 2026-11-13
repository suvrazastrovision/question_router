import json
from collections import Counter

import evaluate
from classifier import Category, Classification, ClassificationError


def test_twenty_eight_case_dataset():
    cases = json.loads(evaluate.DATASET.read_text(encoding="utf-8"))
    assert len(cases) == 28
    assert {case["id"] for case in cases} == set(range(1, 29))
    assert len({case["question"] for case in cases}) == 28
    assert Counter(case["kind"] for case in cases) == {
        "lateral_hypothalamus": 11, "hippocampus": 11,
        "ambiguous": 3, "mixed": 2, "out-of-scope": 1,
    }
    advanced = [case for case in cases if case.get("difficulty") == "advanced"]
    assert Counter(case["kind"] for case in advanced) == {
        "lateral_hypothalamus": 4, "hippocampus": 4,
    }


def test_metrics_count_wrong_predictions_and_errors(monkeypatch):
    """Verify scoring logic, without claiming this measures real Jev accuracy."""
    predictions = iter([
        Classification(Category.HIPPOCAMPUS, 0.9),
        Classification(Category.LATERAL_HYPOTHALAMUS, 0.9),
        Classification(Category.HIPPOCAMPUS, 0.2),
        ClassificationError("Service unavailable"),
    ])

    def classify(question):
        result = next(predictions)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(evaluate, "classify_question", classify)
    cases = [{"id": i, "question": "Test", "category": "HIPPOCAMPUS", "route": "Hippocampus Assistant"}
             for i in range(4)]
    report = evaluate.evaluate(cases)
    assert report["classification_accuracy"] == 0.5
    assert report["routing_accuracy"] == 0.25
    assert report["completed"] == 3
    assert report["errors"] == 1
