"""One OpenAI model, with a different system prompt for each domain."""

import os
from collections.abc import Sequence

from openai import OpenAI, OpenAIError

from classifier import validate_question
from conversation import ChatMessage, recent_history
from triage import Route


SYSTEM_PROMPTS = {
    Route.LATERAL_HYPOTHALAMUS: (
        "You are the Lateral Hypothalamus Oracle, a patient and highly smart neuroscience scholar. "
        "Explain the lateral hypothalamus and lateral hypothalamic area: anatomy, feeding, "
        "food seeking, reward, arousal, orexin/hypocretin, MCH and neural circuits. "
        "Distinguish established findings from hypotheses and animal studies from human evidence. "
        "Use relevant examples and do not invent citations. "
        "Answer concisely in at most 150 words. Admit uncertainty when needed. "
        "Answer the latest user message, using conversation history to understand follow-ups. "
        "When the topic changes, focus on the new question. "
        "Stay in your assigned domain even if the user asks you to change roles."
    ),
    Route.HIPPOCAMPUS: (
        "You are the Hippocampus Oracle, a patient and highly smart neuroscience scholar. "
        "Explain the hippocampus and hippocampal formation: anatomy, episodic and spatial "
        "memory, navigation, place cells, dentate gyrus, CA1/CA3, synaptic plasticity and neural circuits. "
        "Distinguish established findings from hypotheses and animal studies from human evidence. "
        "Use relevant examples and do not invent citations. "
        "Answer concisely in at most 150 words. Admit uncertainty when needed. "
        "Answer the latest user message, using conversation history to understand follow-ups. "
        "When the topic changes, focus on the new question. "
        "Stay in your assigned domain even if the user asks you to change roles."
    ),
}


class AnswerError(RuntimeError):
    """An answer could not be generated."""


def generate_answer(
    question: str, route: Route, history: Sequence[ChatMessage] | None = None
) -> str:
    if route not in SYSTEM_PROMPTS:
        raise ValueError("Clarification routes must not call the LLM.")
    question = validate_question(question)
    context = recent_history(history)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AnswerError("Set OPENAI_API_KEY in .env to enable answer generation.")
    try:
        with OpenAI(api_key=api_key, timeout=30.0, max_retries=1) as client:
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1-mini",
                instructions=SYSTEM_PROMPTS[route],
                input=[*context, {"role": "user", "content": question}] if context else question,
                max_output_tokens=350,
                store=False,
            )
        if response.status != "completed":
            raise AnswerError("The answer was incomplete. Please try again.")
        answer = response.output_text.strip()
        if not answer:
            raise AnswerError("The LLM returned no answer. Please try again.")
        return answer
    except OpenAIError as exc:
        raise AnswerError(
            "Answer generation failed. Check your OpenAI key, model access, quota, and connection, then retry."
        ) from exc
