"""Turn a user question into a grounded answer. Chat text stays in memory."""
from __future__ import annotations

from dataclasses import dataclass

from . import client
from .client import AssistantNotConfigured, AssistantUnavailable
from .policy import NOT_CONFIGURED, UNAVAILABLE, Quota, limit_notice, trim_history


@dataclass
class TurnResult:
    history: list[dict[str, str]]
    quota: Quota
    notice: str | None
    called_api: bool


def submit_message(
    history: list[dict[str, str]],
    quota: Quota,
    text: str,
    *,
    now: float,
) -> TurnResult:
    question = text.strip()
    if not question:
        return TurnResult(history, quota, None, False)
    notice = limit_notice(question, quota, now)
    if notice:
        return TurnResult(history, quota, notice, False)
    if client.select_provider() is None:
        return TurnResult(history, quota, NOT_CONFIGURED, False)
    messages = trim_history([*trim_history(history), {"role": "user", "content": question}])
    try:
        answer = client.complete(messages)
    except AssistantNotConfigured:
        return TurnResult(history, quota, NOT_CONFIGURED, False)
    except AssistantUnavailable:
        return TurnResult(history, quota, UNAVAILABLE, True)
    updated = trim_history([*messages, {"role": "assistant", "content": answer}])
    return TurnResult(
        updated,
        Quota(sent=quota.sent + 1, last_sent_at=now),
        None,
        True,
    )
