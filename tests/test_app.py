from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import classifier
import llm
from classifier import Category, Classification, ClassificationError
from llm import AnswerError
from triage import Route


APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def services(monkeypatch):
    classify = Mock(return_value=Classification(Category.HIPPOCAMPUS, 0.95))
    answer = Mock(side_effect=lambda question, route, history=None: f"Answer: {question}")
    monkeypatch.setattr(classifier, "classify_question", classify)
    monkeypatch.setattr(llm, "generate_answer", answer)
    return classify, answer


def open_chat():
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    return app


def submit(app, question):
    app.chat_input[0].set_value(question).run()
    assert not app.exception


def test_empty_chat_and_invalid_submission(services):
    classify, answer = services
    app = open_chat()
    assert len(app.get("chat_message")) == 0
    submit(app, " \n ")
    assert app.error[0].value == "Please enter a question."
    assert not app.session_state["messages"]
    classify.assert_not_called()
    answer.assert_not_called()


@pytest.mark.parametrize("category,confidence", [(Category.UNKNOWN, 0.99), (Category.HIPPOCAMPUS, 0.2)])
def test_clarification_skips_generation(services, category, confidence):
    classify, answer = services
    classify.return_value = Classification(category, confidence)
    app = open_chat()
    submit(app, "Explain this.")
    assert len(app.get("chat_message")) == 2
    assert app.metric[0].value == category.value
    assert "Ask for clarification" in str([item.value for item in app.markdown])
    assert app.session_state["messages"][1]["content"]
    answer.assert_not_called()


def test_answer_and_metadata_persist_without_repeating_api_calls(services):
    classify, answer = services
    app = open_chat()
    submit(app, "What is the hippocampus?")
    assert app.metric[0].value == "HIPPOCAMPUS"
    assert app.metric[1].value == "95.0%"
    app.run()
    assert len(app.get("chat_message")) == 2
    assert "Answer: What is the hippocampus?" in [item.value for item in app.markdown]
    classify.assert_called_once_with("What is the hippocampus?", history=[])
    answer.assert_called_once_with("What is the hippocampus?", Route.HIPPOCAMPUS, history=[])
    submit(app, " ")
    assert app.error
    assert app.metric[0].value == "HIPPOCAMPUS"
    assert len(app.session_state["messages"]) == 2


def test_followups_and_domain_switching(services):
    classify, answer = services
    classify.side_effect = [Classification(Category.LATERAL_HYPOTHALAMUS, 0.9),
                            Classification(Category.LATERAL_HYPOTHALAMUS, 0.92),
                            Classification(Category.HIPPOCAMPUS, 0.95)]
    app = open_chat()
    questions = ["What is the lateral hypothalamus?", "Give an example.", "What is the hippocampus?"]
    expected_history = []
    for question in questions:
        submit(app, question)
        assert classify.call_args.kwargs["history"] == expected_history
        assert answer.call_args.kwargs["history"] == expected_history
        expected_history.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": f"Answer: {question}"}])
    assert [call.args[1] for call in answer.call_args_list] == [
        Route.LATERAL_HYPOTHALAMUS, Route.LATERAL_HYPOTHALAMUS, Route.HIPPOCAMPUS
    ]
    assert len(app.get("chat_message")) == 6
    assert [metric.value for metric in app.metric] == [
        "LATERAL_HYPOTHALAMUS", "90.0%", "LATERAL_HYPOTHALAMUS", "92.0%", "HIPPOCAMPUS", "95.0%"
    ]


def test_clarification_reply_keeps_original_question(services):
    classify, answer = services
    classify.side_effect = [Classification(Category.UNKNOWN, 0.99),
                            Classification(Category.HIPPOCAMPUS, 0.95)]
    app = open_chat()
    submit(app, "Explain this circuit.")
    clarification = app.session_state["messages"][1]["content"]
    answer.assert_not_called()
    submit(app, "I mean the hippocampal memory circuit.")
    expected = [{"role": "user", "content": "Explain this circuit."},
                {"role": "assistant", "content": clarification}]
    assert classify.call_args.kwargs["history"] == expected
    assert answer.call_args.kwargs["history"] == expected
    assert len(app.get("chat_message")) == 4


def test_context_is_limited_but_transcript_stays_visible(services):
    classify, answer = services
    app = open_chat()
    for index in range(5):
        submit(app, f"Question {index}")
    expected = [{"role": role, "content": f"Question {index}" if role == "user"
                 else f"Answer: Question {index}"}
                for index in range(1, 4) for role in ("user", "assistant")]
    assert classify.call_args.kwargs["history"] == expected
    assert answer.call_args.kwargs["history"] == expected
    assert len(app.get("chat_message")) == 10


@pytest.mark.parametrize("failure_stage", ["classification", "answer"])
def test_errors_stay_visible_and_failed_exchange_is_excluded(services, failure_stage):
    classify, answer = services
    app = open_chat()
    submit(app, "What is the hippocampus?")
    if failure_stage == "classification":
        classify.side_effect = ClassificationError("Classification service unavailable")
    else:
        answer.side_effect = AnswerError("Answer service unavailable")
    submit(app, "Failed question")
    assert len(app.get("chat_message")) == 4
    assert app.error
    if failure_stage == "classification":
        assert answer.call_count == 1
        assert len(app.metric) == 2  # Only the first turn has a classification.
    else:
        assert len(app.metric) == 4  # Keep the failed answer's classification.
    classify.side_effect = None
    answer.side_effect = lambda question, route, history=None: f"Answer: {question}"
    submit(app, "Explain that further.")
    expected = [{"role": "user", "content": "What is the hippocampus?"},
                {"role": "assistant", "content": "Answer: What is the hippocampus?"}]
    assert classify.call_args.kwargs["history"] == expected
    assert answer.call_args.kwargs["history"] == expected
    assert len(app.get("chat_message")) == 6
    assert app.error  # The earlier failure remains attached to its turn.


def test_new_chat_clears_messages_and_context(services):
    classify, answer = services
    app = open_chat()
    submit(app, "What is the hippocampus?")
    app.button[0].click().run()
    assert not app.exception
    assert not app.session_state["messages"]
    assert not app.get("chat_message")
    assert not app.metric
    submit(app, "What is hippocampal LTP?")
    assert classify.call_args.kwargs["history"] == []
    assert answer.call_args.kwargs["history"] == []
    assert len(app.get("chat_message")) == 2
