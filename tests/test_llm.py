import json

import httpx
import pytest
from openai import OpenAI

import llm
from llm import AnswerError, SYSTEM_PROMPTS, generate_answer
from triage import Route


@pytest.mark.parametrize("route", [Route.LATERAL_HYPOTHALAMUS, Route.HIPPOCAMPUS])
@pytest.mark.parametrize("with_history", [False, True])
def test_same_model_with_domain_prompt(monkeypatch, route, with_history):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    history = [{"role": role, "content": f"{role} {turn}"}
               for turn in range(4) for role in ("user", "assistant")]

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert payload["model"] == "gpt-4.1-mini"
        assert payload["instructions"] == SYSTEM_PROMPTS[route]
        expected_input = ([*history[-6:], {"role": "user", "content": "Explain this concept."}]
                          if with_history else "Explain this concept.")
        assert payload["input"] == expected_input
        assert payload["store"] is False
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "gpt-4.1-mini", "status": "completed",
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": "A concise answer.", "annotations": []}]}],
        })

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: OpenAI(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ))
    answer = (generate_answer("Explain this concept.", route, history=history) if with_history
              else generate_answer("Explain this concept.", route))
    assert answer == "A concise answer."


def test_clarification_never_calls_llm(monkeypatch):
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: pytest.fail("LLM called"))
    with pytest.raises(ValueError, match="Clarification"):
        generate_answer("What is it?", Route.CLARIFY)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AnswerError, match="OPENAI_API_KEY"):
        generate_answer("What is the hippocampus?", Route.HIPPOCAMPUS)
