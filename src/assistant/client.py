"""Server-side chat client. API keys never go to the browser or the logs."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from .policy import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS
from .prompt import system_prompt

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
API_URL = XAI_API_URL
DEFAULT_XAI_MODEL = "grok-4"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
ROOT = Path(__file__).resolve().parents[2]

_PROVIDERS = {
    "xai": {"url": XAI_API_URL, "key_name": "XAI_API_KEY", "model": DEFAULT_XAI_MODEL},
    "openai": {
        "url": OPENAI_API_URL,
        "key_name": "OPENAI_API_KEY",
        "model": DEFAULT_OPENAI_MODEL,
    },
}


class AssistantNotConfigured(Exception):
    """Neither provider has an API key on the server."""


class AssistantUnavailable(Exception):
    """The provider did not return a usable answer."""


def _clean(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _from_env_file(path: Path, key_name: str) -> str:
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
        if name.strip() == key_name:
            return _clean(raw)
    return ""


def _from_secrets(path: Path, key_name: str) -> str:
    if not path.is_file():
        return ""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    raw = data.get(key_name) if isinstance(data, dict) else ""
    return _clean(str(raw or ""))


def _paths(
    env_file: Path | None,
    secrets_file: Path | None,
) -> tuple[Path, Path]:
    env_path = ROOT / ".env" if env_file is None else env_file
    secrets_path = ROOT / ".streamlit" / "secrets.toml" if secrets_file is None else secrets_file
    return env_path, secrets_path


def _setting(
    key_name: str,
    environ: dict[str, str] | None = None,
    env_file: Path | None = None,
    secrets_file: Path | None = None,
) -> str:
    source = os.environ if environ is None else environ
    value = _clean(str(source.get(key_name) or ""))
    if value:
        return value
    env_path, secrets_path = _paths(env_file, secrets_file)
    value = _from_env_file(env_path, key_name)
    if value:
        return value
    return _from_secrets(secrets_path, key_name)


def load_api_key(
    environ: dict[str, str] | None = None,
    env_file: Path | None = None,
    secrets_file: Path | None = None,
) -> str | None:
    return _setting("XAI_API_KEY", environ, env_file, secrets_file) or None


def load_provider_key(
    provider: str,
    environ: dict[str, str] | None = None,
    env_file: Path | None = None,
    secrets_file: Path | None = None,
) -> str | None:
    spec = _PROVIDERS.get(provider)
    if spec is None:
        return None
    return _setting(spec["key_name"], environ, env_file, secrets_file) or None


def select_provider(
    environ: dict[str, str] | None = None,
    env_file: Path | None = None,
    secrets_file: Path | None = None,
) -> str | None:
    """Pick openai or xai. Auto prefers OpenAI when that key is set."""
    requested = _setting(
        "VOICESENSE_ASSISTANT_PROVIDER", environ, env_file, secrets_file
    ).lower()
    if requested in _PROVIDERS:
        if load_provider_key(requested, environ, env_file, secrets_file):
            return requested
        return None
    if requested not in ("", "auto"):
        return None
    if load_provider_key("openai", environ, env_file, secrets_file):
        return "openai"
    if load_provider_key("xai", environ, env_file, secrets_file):
        return "xai"
    return None


def model_name(
    provider: str,
    environ: dict[str, str] | None = None,
    env_file: Path | None = None,
    secrets_file: Path | None = None,
) -> str:
    override = _setting("VOICESENSE_ASSISTANT_MODEL", environ, env_file, secrets_file)
    if override:
        return override
    spec = _PROVIDERS.get(provider)
    return spec["model"] if spec else DEFAULT_XAI_MODEL


def complete(
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    http: httpx.Client | None = None,
    provider: str | None = None,
) -> str:
    chosen = provider or select_provider()
    if chosen not in _PROVIDERS:
        raise AssistantNotConfigured()
    key = api_key if api_key is not None else load_provider_key(chosen)
    if not key:
        raise AssistantNotConfigured()
    payload = {
        "model": model_name(chosen),
        "messages": [{"role": "system", "content": system_prompt()}, *messages],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}"}
    if http is None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = _post(client, _PROVIDERS[chosen]["url"], headers, payload)
    else:
        response = _post(http, _PROVIDERS[chosen]["url"], headers, payload)
    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AssistantUnavailable() from exc
    answer = str(text or "").strip()
    if not answer:
        raise AssistantUnavailable()
    return answer


def _post(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> httpx.Response:
    try:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AssistantUnavailable() from exc
    return response
