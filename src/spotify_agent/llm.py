"""OpenAI client helpers with timeouts, bounded retries, and response checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, NoReturn

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from spotify_agent.paths import CACHE_DIR, REPO_ROOT, ensure_dirs
from spotify_agent.safety import redact_secrets

_CACHE_KEY = re.compile(r"[^A-Za-z0-9._-]")


class LLMError(RuntimeError):
    """A chat completion failed and should not be retried by callers."""


class LLMResponseError(LLMError):
    """The provider returned a body that is not a JSON object."""


def load_env() -> None:
    """Load the repo ``.env`` only. Do not pull secrets from other home configs."""
    load_dotenv(REPO_ROOT / ".env")


def _timeout_seconds() -> float:
    raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "30")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 30.0
    if not math.isfinite(value):
        value = 30.0
    return min(max(value, 1.0), 120.0)


def _validate_api_key(api_key: str) -> str:
    key = api_key.strip()
    if not key:
        raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add a key.")
    if any(char.isspace() or ord(char) < 32 for char in key):
        raise LLMError("OPENAI_API_KEY contains whitespace or control characters.")
    return key


def _validate_base_url(url: str) -> str:
    raw = url.strip()
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host:
        return raw
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
        return raw
    raise LLMError("OPENAI_BASE_URL must use https, or http only for localhost.")


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise LLMError("Model name is empty.")
    if any(ord(char) < 32 for char in model):
        raise LLMError("Model name contains control characters.")
    return model.strip()


def get_client() -> OpenAI:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add a key.")
    kwargs: dict[str, Any] = {
        "api_key": _validate_api_key(api_key),
        "timeout": _timeout_seconds(),
        # Retries are implemented here so non-retryable 4xx errors fail fast.
        "max_retries": 0,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = _validate_base_url(base_url)
    return OpenAI(**kwargs)


def agent_model() -> str:
    load_env()
    return os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")


def judge_model() -> str:
    load_env()
    return os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")


def _cache_path(key: str) -> Path:
    ensure_dirs()
    safe = _CACHE_KEY.sub("_", key)[:120]
    if safe in {"", ".", ".."}:
        safe = "chat"
    root = CACHE_DIR.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise LLMError("Refusing to write a cache file outside the cache directory.")
    return path


def cache_get(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def cache_set(key: str, value: dict[str, Any]) -> None:
    _cache_path(key).write_text(json.dumps(value, indent=2))


def make_cache_key(prefix: str, payload: dict[str, Any]) -> str:
    safe_prefix = _CACHE_KEY.sub("_", prefix)[:40] or "chat"
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    return f"{safe_prefix}_{digest}"


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a model body into one JSON object, including fenced or wrapped JSON."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError("Model response was not valid JSON.") from exc
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Model response was not valid JSON.") from exc
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Model response was not a JSON object.") from exc
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
        else:
            raise LLMResponseError("Model response JSON was not an object.")
    if not isinstance(value, dict):
        raise LLMResponseError("Model response JSON was not an object.")
    return value


def _message_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMResponseError("Model response did not include any choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("Model response content was empty.")
    return content


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return min(max(value, 0.0), 30.0)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMResponseError):
        return True
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, APIStatusError) and isinstance(status, int) and status >= 500


def _backoff_seconds(exc: BaseException, attempt: int, rate_limit_sleep: bool) -> float:
    if rate_limit_sleep:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return retry_after
        if isinstance(exc, RateLimitError):
            return min(2.0 * (2**attempt), 30.0)
    return min(0.4 * (2**attempt), 8.0)


def _fail(last_err: BaseException, attempts: int) -> NoReturn:
    message = (
        f"chat_json failed after {attempts} attempt(s): "
        f"{type(last_err).__name__}: {redact_secrets(str(last_err))}"
    )
    raise LLMError(message) from None


def chat_json(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    use_cache: bool = True,
    cache_prefix: str = "chat",
    max_retries: int = 2,
    rate_limit_sleep: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call chat completions and return one JSON object.

    ``max_retries`` is the number of retries after the first attempt.
    Timeouts, connection failures, rate limits, HTTP 5xx, and malformed JSON
    are retried. Authentication and other 4xx errors fail immediately.
    Empty or non-object bodies are not cached.
    """
    if not isinstance(system, str) or not isinstance(user, str):
        raise LLMError("system and user prompts must be strings.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise LLMError("temperature must be a finite number.")
    if not math.isfinite(float(temperature)):
        raise LLMError("temperature must be a finite number.")
    model_name = _validate_model(model)
    load_env()
    payload = {
        "provider": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": model_name,
        "system": system,
        "user": user,
        "temperature": float(temperature),
    }
    key = make_cache_key(cache_prefix, payload)
    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            return cached

    client = get_client()
    request_timeout = _timeout_seconds() if timeout is None else min(max(float(timeout), 1.0), 120.0)
    attempts = max(1, int(max_retries) + 1)
    last_err: BaseException | None = None
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=float(temperature),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=request_timeout,
            )
            parsed = parse_json_object(_message_text(response))
        except LLMResponseError as exc:
            last_err = exc
            retryable = True
        except OpenAIError as exc:
            last_err = exc
            retryable = _is_retryable(exc)
        else:
            if use_cache:
                cache_set(key, parsed)
            return parsed
        if not retryable or attempt >= attempts - 1:
            break
        time.sleep(_backoff_seconds(last_err, attempt, rate_limit_sleep))
    if last_err is None:
        raise LLMError("chat_json failed without a provider error.")
    _fail(last_err, attempt + 1)
