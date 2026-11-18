"""Server-side Grok client. The API key never goes to the browser or the logs."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from .policy import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS
from .prompt import system_prompt

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4"
ROOT = Path(__file__).resolve().parents[2]


class AssistantNotConfigured(Exception):
    """XAI_API_KEY is not available on the server."""


class AssistantUnavailable(Exception):
    """The provider did not return a usable answer."""


def _clean(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _from_env_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw = stripped.split("=", 1)
        if name.strip() == "XAI_API_KEY":
            return _clean(raw)
    return ""


def _from_secrets(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    raw = data.get("XAI_API_KEY") if isinstance(data, dict) else ""
    return _clean(str(raw or ""))


def load_api_key(
    environ: dict[str, str] | None = None,
    env_file: Path | None = None,
    secrets_file: Path | None = None,
) -> str | None:
    source = os.environ if environ is None else environ
    key = _clean(str(source.get("XAI_API_KEY") or ""))
    if key:
        return key
    env_path = ROOT / ".env" if env_file is None else env_file
    key = _from_env_file(env_path)
    if key:
        return key
    secrets_path = ROOT / ".streamlit" / "secrets.toml" if secrets_file is None else secrets_file
    key = _from_secrets(secrets_path)
    return key or None


def model_name(environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    chosen = _clean(str(source.get("VOICESENSE_ASSISTANT_MODEL") or ""))
    return chosen or DEFAULT_MODEL


def complete(
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    http: httpx.Client | None = None,
) -> str:
    key = api_key if api_key is not None else load_api_key()
    if not key:
        raise AssistantNotConfigured()
    payload = {
        "model": model_name(),
        "messages": [{"role": "system", "content": system_prompt()}, *messages],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}"}
    if http is None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = _post(client, headers, payload)
    else:
        response = _post(http, headers, payload)
    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AssistantUnavailable() from exc
    answer = str(text or "").strip()
    if not answer:
        raise AssistantUnavailable()
    return answer


def _post(client: httpx.Client, headers: dict[str, str], payload: dict) -> httpx.Response:
    try:
        response = client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AssistantUnavailable() from exc
    return response
