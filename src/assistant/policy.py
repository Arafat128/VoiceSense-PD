"""Session limits for the assistant. Nothing here is written to disk."""
from __future__ import annotations

from dataclasses import dataclass

MAX_INPUT_CHARS = 500
MAX_OUTPUT_TOKENS = 400
MAX_SESSION_MESSAGES = 20
MIN_SECONDS_BETWEEN = 3.0
MAX_HISTORY_TURNS = 6
REQUEST_TIMEOUT_SECONDS = 20.0

NOT_CONFIGURED = (
    "Assistant not configured. Add XAI_API_KEY to a gitignored .env file "
    "(see .env.example) and restart the app."
)
TOO_LONG = "Please keep questions to 500 characters or fewer."
RATE_COUNT = (
    "This visit has used its 20 questions. Refresh the page to start a new chat. "
    "Nothing from this chat is saved."
)
RATE_TIME = "Please wait a few seconds before the next question."
UNAVAILABLE = "The assistant could not answer just now. Nothing was saved."


@dataclass
class Quota:
    sent: int = 0
    last_sent_at: float = 0.0


def trim_history(messages: list[dict[str, str]], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, str]]:
    """Keep the last `max_turns` user turns and the replies that follow them."""
    turns: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    kept = turns[-max_turns:]
    return [item for turn in kept for item in turn]


def limit_notice(text: str, quota: Quota, now: float) -> str | None:
    if len(text) > MAX_INPUT_CHARS:
        return TOO_LONG
    if quota.sent >= MAX_SESSION_MESSAGES:
        return RATE_COUNT
    if quota.last_sent_at and (now - quota.last_sent_at) < MIN_SECONDS_BETWEEN:
        return RATE_TIME
    return None
