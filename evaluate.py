"""Measure real Jev classification and routing on the labeled question dataset."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from classifier import ClassificationError, DEFAULT_JEV_MODEL, classify_question
from triage import CONFIDENCE_THRESHOLD, triage


DATASET = Path(__file__).parent / "tests" / "questions.json"


def evaluate(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        row = dict(case)
        try:
            classification = classify_question(case["question"])
            decision = triage(classification)
            row.update(
                predicted_category=classification.category.value,
                confidence=classification.confidence,
                predicted_route=decision.route.value,
                classification_correct=classification.category.value == case["category"],
                routing_correct=decision.route.value == case["route"],
            )
        except ClassificationError as exc:
            row.update(error=str(exc), classification_correct=False, routing_correct=False)
        rows.append(row)
    total = len(rows)
    if total == 0:
        raise ValueError("The evaluation dataset must not be empty.")
    errors = sum("error" in row for row in rows)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("TYPESAFE_DEFAULT_MODEL", "").strip() or DEFAULT_JEV_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total": total,
        "completed": total - errors,
        "errors": errors,
        "classification_accuracy": sum(row["classification_correct"] for row in rows) / total,
        "routing_accuracy": sum(row["routing_correct"] for row in rows) / total,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("TYPESAFE_API_KEY", "").strip():
        parser.error("Set TYPESAFE_API_KEY in .env. Evaluation makes one real Jev API call per question.")
    report = evaluate(json.loads(DATASET.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        status = "ERROR" if "error" in row else (
            "PASS" if row["classification_correct"] and row["routing_correct"] else "FAIL"
        )
        print(f"{row['id']:02d} {status:5} {row.get('predicted_category', '-'):10} "
              f"{row.get('predicted_route', '-')} | {row['question']}")
    print(f"\nCompleted: {report['completed']}/{report['total']}; errors: {report['errors']}")
    print(f"Classification accuracy: {report['classification_accuracy']:.1%}")
    print(f"Routing accuracy: {report['routing_accuracy']:.1%}")
    print(f"Denominator: all {report['total']} questions; API errors count as failures.")
    print(f"Saved details to {args.output}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
