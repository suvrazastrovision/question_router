import json

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

import classifier
from classifier import Category, ClassificationError, classify_question


@pytest.mark.parametrize("label", list(Category))
@pytest.mark.parametrize("with_history", [False, True])
def test_real_sdk_request_and_response(monkeypatch, label, with_history):
    """Use the installed SDK over a fake HTTP transport, without calling Jev."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-1.14.0")
    history = [{"role": role, "content": f"{role} {turn}"}
               for turn in range(4) for role in ("user", "assistant")]

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        assert payload["model"] == "jev-1.14.0"
        expected_state = {"user_question": "Explain this question."}
        if with_history:
            expected_state["conversation_history"] = history[-6:]
        assert payload["state"] == expected_state
        assert set(payload["questions"]["domain"]["criteria"]) == set(Category)
        return httpx2.Response(200, json={
            "model": "jev-1.14.0",
            "usage": {"input_tokens": 100},
            "answers": {"domain": {
                "type": "choice", "choice": label.value, "confidence": 0.85,
                "probabilities": {c.value: 0.9 if c == label else 0.05 for c in Category},
            }},
        })

    monkeypatch.setattr(classifier, "TypeSafeClient", lambda **kwargs: TypeSafeClient(
        **kwargs, compression=False, transport=httpx2.MockTransport(handler)
    ))
    result = (classify_question("  Explain this question.  ", history=history) if with_history
              else classify_question("  Explain this question.  "))
    assert result.category == label
    assert result.confidence == 0.85  # Preserve Jev's confidence; don't replace with top probability.


def test_service_failure_is_not_unknown(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(classifier, "TypeSafeClient", lambda **kwargs: TypeSafeClient(
        **kwargs, compression=False,
        transport=httpx2.MockTransport(lambda request: httpx2.Response(401, json={"error": "Unauthorized"})),
    ))
    with pytest.raises(ClassificationError, match="classification failed"):
        classify_question("What is the hippocampus?")


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ClassificationError, match="TYPESAFE_API_KEY"):
        classify_question("What is the hippocampus?")


@pytest.mark.parametrize("question", ["", " \n ", "a" * 4001])
def test_invalid_input_never_calls_api(monkeypatch, question):
    monkeypatch.setattr(classifier, "TypeSafeClient", lambda **kwargs: pytest.fail("API called"))
    with pytest.raises(ValueError):
        classify_question(question)
