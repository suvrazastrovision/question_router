"""Pure, deterministic routing. No network calls or model judgments here."""

import math
from dataclasses import dataclass
from enum import StrEnum

from classifier import Category, Classification


CONFIDENCE_THRESHOLD = 0.70


class Route(StrEnum):
    LATERAL_HYPOTHALAMUS = "Lateral Hypothalamus Assistant"
    HIPPOCAMPUS = "Hippocampus Assistant"
    CLARIFY = "Ask for clarification"


@dataclass(frozen=True)
class TriageDecision:
    route: Route
    clarification: str | None = None


def triage(
    classification: Classification, threshold: float = CONFIDENCE_THRESHOLD
) -> TriageDecision:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if classification.category == Category.UNKNOWN:
        return TriageDecision(
            Route.CLARIFY,
            "Would you like help with the lateral hypothalamus or hippocampus? Please add context; "
            "if your question covers both, split it into one question for each domain.",
        )
    if classification.confidence < threshold:
        return TriageDecision(
            Route.CLARIFY,
            "I am unsure which domain you mean. Please rephrase your question and "
            "specify whether you want an explanation about the lateral hypothalamus or hippocampus.",
        )
    routes = {Category.LATERAL_HYPOTHALAMUS: Route.LATERAL_HYPOTHALAMUS, Category.HIPPOCAMPUS: Route.HIPPOCAMPUS}
    return TriageDecision(routes[classification.category])
