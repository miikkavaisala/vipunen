"""Minimal OpenAI-compatible chat client for a locally hosted model.

Talks to a standard OpenAI /v1/chat/completions endpoint (LM Studio,
llama.cpp server, vLLM, Ollama, ... — or a hosted API that requires an
API key) using only the standard library — keeping with the project's
minimal-dependency rule.

No endpoint or model is built in: the pipeline runs against the user's own
server, so both are configuration. Set LLM_BASE_URL / LLM_MODEL (and
LLM_API_KEY for hosted endpoints that need one) in the environment, pass
the `base_url` / `model` / `api_key` arguments, or use the --llm-url /
--llm-model / --llm-api-key flags on the pipeline CLIs (main.py,
src/filter.py, src/summarize.py, src/digest.py).

Usage:
    from llm import chat

    answer = chat([{"role": "user", "content": "Say OK."}])
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

# No DEFAULT_BASE_URL / DEFAULT_MODEL on purpose: the endpoint and model are
# the user's own local setup, so they come from configuration only (see the
# module docstring) and a missing one raises a clear error at call time.
DEFAULT_TIMEOUT_SECONDS = 120

# Transient server-side failures are retried before giving up: HTTP 5xx,
# connection-level errors (server or inference engine restarting), and the
# LM Studio engine-crash signature — a runaway generation that exhausts the
# context kills the engine mid-request and LM Studio answers HTTP 400
# "Engine protocol predict request failed: fetch failed" ("Channel Error" in
# its log); the next request reloads the model and usually succeeds.
# Genuine client errors (other 4xx) are raised immediately, not retried.
ENGINE_CRASH_MARKERS = ("fetch failed", "channel error", "predict request failed")
TRANSIENT_ATTEMPTS = 3
TRANSIENT_DELAYS_SECONDS = (2, 5)


def _http_json(
    url: str,
    payload: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    api_key: str | None = None,
):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    for attempt in range(1, TRANSIENT_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            engine_crash = any(marker in body.lower() for marker in ENGINE_CRASH_MARKERS)
            if exc.code < 500 and not engine_crash:
                raise RuntimeError(
                    f"LLM endpoint returned HTTP {exc.code} for {url}: {body}"
                ) from exc
            if attempt < TRANSIENT_ATTEMPTS:
                time.sleep(TRANSIENT_DELAYS_SECONDS[attempt - 1])
                continue
            hint = (
                " (the server's inference engine crashed on this request; "
                "reloading the model in the server UI usually clears it)"
                if engine_crash
                else ""
            )
            raise RuntimeError(
                f"LLM endpoint returned HTTP {exc.code} for {url} after "
                f"{attempt} attempts: {body}{hint}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < TRANSIENT_ATTEMPTS:
                time.sleep(TRANSIENT_DELAYS_SECONDS[attempt - 1])
                continue
            raise RuntimeError(
                f"LLM endpoint unreachable at {url} after {attempt} attempts: {exc}"
            ) from exc


def _base_url(base_url: str | None) -> str:
    url = base_url or os.environ.get("LLM_BASE_URL")
    if not url:
        raise RuntimeError(
            "no LLM endpoint configured: set LLM_BASE_URL (e.g. "
            "http://localhost:1234/v1) or pass --llm-url to the pipeline CLI"
        )
    return url.rstrip("/")


def _api_key(api_key: str | None) -> str | None:
    """The API key for authenticated endpoints: the argument, else
    LLM_API_KEY, else None. A missing key is fine for local servers —
    no Authorization header is sent then."""
    return api_key or os.environ.get("LLM_API_KEY")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared --llm-url / --llm-model / --llm-api-key CLI flags."""
    parser.add_argument(
        "--llm-url",
        default=None,
        metavar="URL",
        help="LLM endpoint base URL, e.g. http://localhost:1234/v1 "
        "(no built-in default; env override: LLM_BASE_URL)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help="model name sent with every request (no built-in default; "
        "env override: LLM_MODEL)",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        metavar="KEY",
        help="API key for hosted endpoints that require one, sent as "
        "'Authorization: Bearer KEY' (optional — local servers need no "
        "key; no built-in default; env override: LLM_API_KEY)",
    )


def apply_llm_args(args: argparse.Namespace) -> None:
    """Apply parsed --llm-url / --llm-model / --llm-api-key flags by setting
    the env vars, so every later chat() call in this process uses them.
    A flag beats the environment variable, which beats the built-in default."""
    if args.llm_url:
        os.environ["LLM_BASE_URL"] = args.llm_url
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key


def current_model(model: str | None = None) -> str:
    """The effective model name: the argument, else LLM_MODEL, else a clear
    error. Single resolution point for chat calls and model-keyed caches."""
    resolved = model or os.environ.get("LLM_MODEL")
    if not resolved:
        raise RuntimeError(
            "no LLM model configured: set LLM_MODEL or pass --llm-model "
            "to the pipeline CLI"
        )
    return resolved


def chat(
    messages: list[dict],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    think: bool = True,
) -> str:
    """Send a chat request to the local endpoint; return the assistant text.

    Temperature defaults to 0.0 for reproducible, deterministic pipeline
    calls — adjust per call if a stage needs variety.

    Thinking/reasoning models put their chain of thought in
    `reasoning_content` and the answer in `content`; only the answer is
    returned. Such models spend tokens on reasoning before the answer, so
    small max_tokens values can cut the reply (raises a clear error).

    With think=False the request carries reasoning_effort="none", which this
    endpoint honors: the model answers directly with no reasoning tokens.
    Use it for cheap pipeline calls (filtering, one-liners); keep think=True
    when the stage benefits from deeper reasoning.
    """
    content, _usage = chat_with_metrics(
        messages,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        think=think,
    )
    return content


def chat_with_metrics(
    messages: list[dict],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    think: bool = True,
) -> tuple[str, dict]:
    """Like `chat()`, but also returns the usage block for perf measurement.

    Returns (assistant_text, usage) where usage is the endpoint's token
    accounting, e.g. {"prompt_tokens": 15, "completion_tokens": 191,
    "total_tokens": 206, "completion_tokens_details": {"reasoning_tokens":
    187}}.
    """
    model = current_model(model)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if not think:
        payload["reasoning_effort"] = "none"

    data = _http_json(
        f"{_base_url(base_url)}/chat/completions", payload, timeout, api_key=_api_key(api_key)
    )
    choices = data.get("choices")
    if not choices:
        # some servers answer unknown paths with HTTP 200 plus an error
        # body (e.g. {"error": "..."}) when the /v1 prefix is missing
        # from the base URL — surface that instead of a bare KeyError
        body = json.dumps(data)[:500]
        raise RuntimeError(
            f"LLM endpoint returned no choices for model {model!r} "
            f"(check the base URL — it usually includes the /v1 prefix, "
            f"e.g. http://localhost:1234/v1): {body}"
        )
    choice = choices[0]
    content = choice["message"].get("content") or ""
    if not content.strip():
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "empty reply: max_tokens cut the generation (reasoning models "
                "spend tokens on reasoning_content before any content; "
                "increase max_tokens or omit it)"
            )
        raise RuntimeError(f"empty reply from model: {choice['message']}")
    return content, data.get("usage", {})


if __name__ == "__main__":
    reply = chat([{"role": "user", "content": "Reply with exactly: OK"}])
    print("reply:", reply)
