"""OpenAI client helpers with optional response caching."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from spotify_agent.paths import CACHE_DIR, REPO_ROOT, ensure_dirs


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        fallback = Path.home() / ".openclaw" / ".env"
        if fallback.exists():
            load_dotenv(fallback, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        hermes = Path.home() / ".hermes" / ".env"
        if hermes.exists():
            load_dotenv(hermes, override=False)


def get_client() -> OpenAI:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
        )
    kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def agent_model() -> str:
    load_env()
    return os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")


def judge_model() -> str:
    load_env()
    return os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")


def _cache_path(key: str) -> Path:
    ensure_dirs()
    return CACHE_DIR / f"{key}.json"


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text())
    return None


def cache_set(key: str, value: Dict[str, Any]) -> None:
    _cache_path(key).write_text(json.dumps(value, indent=2))


def make_cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def chat_json(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    use_cache: bool = True,
    cache_prefix: str = "chat",
    max_retries: int = 5,
    rate_limit_sleep: bool = True,
) -> Dict[str, Any]:
    """Call chat completions requesting JSON object output."""
    load_env()
    payload = {
        "provider": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": model,
        "system": system,
        "user": user,
        "temperature": temperature,
    }
    key = make_cache_key(cache_prefix, payload)
    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            return cached

    client = get_client()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content or "{}"
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()
            parsed = json.loads(content)
            if use_cache:
                cache_set(key, parsed)
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc)
            if rate_limit_sleep and (
                "429" in msg or "rate" in msg.lower() or "RESOURCE_EXHAUSTED" in msg
            ):
                time.sleep(min(15 + 5 * attempt, 45))
            else:
                time.sleep(min(1.2 * (attempt + 1), 6))
    raise RuntimeError(f"chat_json failed after {max_retries} retries: {last_err}")
