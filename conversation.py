"""Shared format and size limit for the context sent to both APIs."""

from collections.abc import Sequence
from typing import Literal, TypedDict


MAX_HISTORY_EXCHANGES = 3


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def recent_history(history: Sequence[ChatMessage] | None = None) -> list[ChatMessage]:
    """Copy the last three exchanges, keeping display metadata out of API calls."""
    messages = list(history or [])[-2 * MAX_HISTORY_EXCHANGES:]
    return [{"role": message["role"], "content": message["content"]} for message in messages]
