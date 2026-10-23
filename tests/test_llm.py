"""LLM client behavior with the provider mocked."""

from __future__ import annotations

import os
from pathlib import Path

import httpx2
import pytest
from openai import APITimeoutError, AuthenticationError, RateLimitError

from spotify_agent import llm
from spotify_agent.llm import LLMError, LLMResponseError, parse_json_object


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        self._client.calls += 1
        self._client.last_kwargs = kwargs
        item = self._client.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Response(item)


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.last_kwargs = None
        completions = _Completions(self)
        self.chat = type("Chat", (), {"completions": completions})()


def _request():
    return httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _timeout():
    return APITimeoutError(request=_request())


def _auth():
    response = httpx2.Response(401, request=_request())
    return AuthenticationError(
        "Incorrect API key provided: sk-secretABCDEF123456",
        response=response,
        body=None,
    )


def _rate_limit(retry_after: str):
    response = httpx2.Response(429, request=_request(), headers={"retry-after": retry_after})
    return RateLimitError("rate limited", response=response, body=None)


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "load_env", lambda: None)
    return tmp_path


def test_timeout_is_retried_then_succeeds(monkeypatch, cache_dir):
    client = ScriptedClient([_timeout(), _timeout(), '{"reply": "ok"}'])
    sleeps = []
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    parsed = llm.chat_json(
        model="m",
        system="s",
        user="u",
        use_cache=False,
        max_retries=2,
        rate_limit_sleep=False,
    )
    assert parsed == {"reply": "ok"}
    assert client.calls == 3
    assert sleeps == pytest.approx([0.4, 0.8])
    assert client.last_kwargs["timeout"] >= 1


def test_auth_errors_are_not_retried_and_secrets_are_redacted(monkeypatch, cache_dir):
    client = ScriptedClient([_auth(), _auth()])
    sleeps = []
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secretABCDEF123456")
    with pytest.raises(LLMError) as raised:
        llm.chat_json(model="m", system="s", user="u", use_cache=False, max_retries=4)
    message = str(raised.value)
    assert "sk-secretABCDEF123456" not in message
    assert "[REDACTED]" in message
    assert client.calls == 1
    assert sleeps == []
    assert list(cache_dir.glob("*.json")) == []


def test_bad_json_is_retried_and_not_cached(monkeypatch, cache_dir):
    client = ScriptedClient(["not-json", "still bad", "[1, 2]"])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    with pytest.raises(LLMError, match="attempt"):
        llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=2)
    assert client.calls == 3
    assert list(cache_dir.glob("*.json")) == []


def test_empty_content_is_not_treated_as_an_object(monkeypatch, cache_dir):
    client = ScriptedClient(["   ", '{"a": 1}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    assert llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=1) == {"a": 1}
    assert client.calls == 2


def test_cache_hit_skips_the_provider(monkeypatch, cache_dir):
    client = ScriptedClient(['{"a": 1}', '{"a": 2}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    first = llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=0)
    second = llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=0)
    assert first == second == {"a": 1}
    assert client.calls == 1


def test_corrupt_or_non_object_cache_is_a_miss(monkeypatch, cache_dir):
    client = ScriptedClient(['{"a": 1}', '{"a": 2}', '{"a": 3}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    assert llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=0) == {"a": 1}
    cached = next(cache_dir.glob("*.json"))
    cached.write_text("{not json")
    assert llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=0) == {"a": 2}
    cached.write_text('["nope"]')
    assert llm.chat_json(model="m", system="s", user="u", use_cache=True, max_retries=0) == {"a": 3}
    assert client.calls == 3


def test_cache_key_cannot_escape_the_cache_directory(cache_dir):
    key = llm.make_cache_key("../../etc/passwd", {"user": "x"})
    path = llm._cache_path(key)
    assert path.is_relative_to(cache_dir.resolve())
    assert "/" not in path.name


def test_parse_json_object_accepts_fences_and_single_item_lists():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('note {"a": 2} trailing') == {"a": 2}
    assert parse_json_object('[{"a": 3}]') == {"a": 3}
    with pytest.raises(LLMResponseError):
        parse_json_object('[{"a": 1}, {"b": 2}]')


def test_retry_after_header_is_capped(monkeypatch, cache_dir):
    client = ScriptedClient([_rate_limit("999"), '{"ok": true}'])
    sleeps = []
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    parsed = llm.chat_json(
        model="m",
        system="s",
        user="u",
        use_cache=False,
        max_retries=1,
        rate_limit_sleep=True,
    )
    assert parsed == {"ok": True}
    assert sleeps == [30.0]


def test_base_url_and_api_key_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com/v1")
    with pytest.raises(LLMError, match="https"):
        llm.get_client()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    client = llm.get_client()
    assert str(client.base_url).startswith("https://example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test\nInjected: yes")
    with pytest.raises(LLMError, match="control"):
        llm.get_client()


def test_load_env_does_not_read_home_directory_secrets(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    repo.mkdir()
    (home / ".hermes").mkdir(parents=True)
    (home / ".openclaw").mkdir()
    (home / ".hermes" / ".env").write_text("OPENAI_API_KEY=sk-from-home\nOTHER_SECRET=abc\n")
    (home / ".openclaw" / ".env").write_text("OPENAI_API_KEY=sk-from-openclaw\n")
    monkeypatch.setattr(llm, "REPO_ROOT", repo)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    llm.load_env()
    assert os.getenv("OPENAI_API_KEY") is None
    assert os.getenv("OTHER_SECRET") is None
    (repo / ".env").write_text("OPENAI_API_KEY=sk-from-repo\n")
    llm.load_env()
    try:
        assert os.environ["OPENAI_API_KEY"] == "sk-from-repo"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OTHER_SECRET", None)
