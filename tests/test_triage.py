import pytest

from classifier import Category, Classification
from triage import CONFIDENCE_THRESHOLD, Route, triage


@pytest.mark.parametrize("category,expected", [
    (Category.LATERAL_HYPOTHALAMUS, Route.LATERAL_HYPOTHALAMUS),
    (Category.HIPPOCAMPUS, Route.HIPPOCAMPUS),
])
def test_confidence_boundary(category, expected):
    assert triage(Classification(category, CONFIDENCE_THRESHOLD)).route == expected
    assert triage(Classification(category, CONFIDENCE_THRESHOLD - 0.0001)).route == Route.CLARIFY


@pytest.mark.parametrize("category,confidence", [
    (Category.UNKNOWN, 1.0), (Category.UNKNOWN, 0.0),
    (Category.LATERAL_HYPOTHALAMUS, 0.1), (Category.HIPPOCAMPUS, 0.69),
])
def test_uncertainty_always_has_clarification(category, confidence):
    decision = triage(Classification(category, confidence))
    assert decision.route == Route.CLARIFY
    assert decision.clarification


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_and_threshold(value):
    with pytest.raises(ValueError):
        Classification(Category.HIPPOCAMPUS, value)
    with pytest.raises(ValueError):
        triage(Classification(Category.HIPPOCAMPUS, 0.9), threshold=value)
