"""Local chatbot showing classification, routing, and answer generation."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from classifier import ClassificationError, MAX_QUESTION_LENGTH, classify_question, validate_question
from conversation import ChatMessage, recent_history
from llm import AnswerError, generate_answer
from triage import CONFIDENCE_THRESHOLD, Route, triage


def conversation_history(messages: list[dict]) -> list[ChatMessage]:
    """Keep complete exchanges; a failed API turn stays visible only in the UI."""
    history = []
    for user, assistant in zip(messages[::2], messages[1::2]):
        if not assistant.get("error"):
            history.extend([user, assistant])
    return recent_history(history)


def render_detection(message: dict) -> None:
    if "classification" not in message:
        return
    category, confidence = st.columns(2)
    category.metric("Detected category", message["classification"].category.value)
    confidence.metric("Jev confidence", f"{message['classification'].confidence:.1%}")
    st.write(f"**Selected route:** {message['decision'].route.value}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "user":
            render_detection(message)


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    st.set_page_config(page_title="Smart Question Router", page_icon="📚")
    st.title("Smart Question Router")
    st.caption("Your question → Jev classification → Triage → Domain prompt → Answer")
    st.write("Chat about the lateral hypothalamus or hippocampus. You can ask follow-ups or switch topics.")
    with st.expander("How the three stages differ"):
        st.markdown(
            "**1. Jev classifies** each message and returns its confidence.\n\n"
            f"**2. Triage routes** known categories with confidence ≥ {CONFIDENCE_THRESHOLD:.0%}. "
            "UNKNOWN and lower-confidence results ask for clarification.\n\n"
            "**3. One LLM generates** an answer using the chosen domain's system prompt. "
            "Clarification requests skip this API call."
        )
        st.caption("Jev confidence describes certainty; it is not measured classification accuracy.")
    st.caption("Remembers the last 3 successful exchanges in this session, including clarification. "
               "Start a new chat to clear the conversation.")
    if st.button("New chat"):
        st.session_state["messages"] = []
        st.session_state.pop("chat_prompt", None)
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        render_message(message)

    prompt = st.chat_input(
        "Ask about the lateral hypothalamus or hippocampus…", max_chars=MAX_QUESTION_LENGTH, key="chat_prompt"
    )
    if prompt is None:
        return
    try:
        question = validate_question(prompt)
    except ValueError as exc:
        st.error(str(exc))
        return

    # Capture preceding exchanges before adding the new user message.
    history = conversation_history(messages)
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)
        detection = st.container()

    try:
        with st.spinner("Jev is classifying your message…"):
            classification = classify_question(question, history=history)
        decision = triage(classification)
        user_message.update(classification=classification, decision=decision)
        with detection:
            render_detection(user_message)
    except ClassificationError as exc:
        assistant_message = {"role": "assistant", "content": str(exc), "error": True}
    else:
        if decision.route == Route.CLARIFY:
            assistant_message = {"role": "assistant", "content": decision.clarification}
        else:
            try:
                with st.spinner("Generating your answer…"):
                    answer = generate_answer(question, decision.route, history=history)
                assistant_message = {"role": "assistant", "content": answer}
            except AnswerError as exc:
                assistant_message = {"role": "assistant", "content": str(exc), "error": True}
    messages.append(assistant_message)
    render_message(assistant_message)


if __name__ == "__main__":
    main()
