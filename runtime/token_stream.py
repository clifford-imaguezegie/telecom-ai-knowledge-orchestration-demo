"""Optional final-answer token streaming for the original Gemma model calls.

The existing Streamlit and JSON APIs call the original nonstreaming OpenRouter
method. Only the native SSE three-path experiment binds a per-path observer.
No prompt, routing, evidence, model or retry policy is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Callable, Iterator

PathObserver = Callable[[str, str], None]
_ACTIVE_PATH: ContextVar[tuple[str, PathObserver] | None] = ContextVar(
    "tebunor_gemma_final_answer_observer", default=None
)
MAX_PUBLIC_ANSWER_CHARS = 12000
VALID_PATHS = frozenset(("llm_only", "strict", "augmented"))


@contextmanager
def output_tokens_for(path: str, observer: PathObserver | None) -> Iterator[None]:
    """Bind a path-local callback; ContextVar propagates through asyncio.to_thread."""
    if observer is None:
        yield
        return
    if path not in VALID_PATHS:
        raise ValueError("Unknown public answer path")
    reset = _ACTIVE_PATH.set((path, observer))
    try:
        yield
    finally:
        _ACTIVE_PATH.reset(reset)


def final_answer_completion(client, **kwargs):
    """Preserve the existing completion contract while streaming visible Gemma text.

    Planner/router/judge calls must never use this helper. Without an active
    native SSE observer it calls the original client exactly as before.
    """
    observer = _ACTIVE_PATH.get()
    if observer is None:
        return client.chat.completions.create(**kwargs)

    path, on_token = observer
    request = dict(kwargs)
    request["stream"] = True
    request["stream_options"] = {"include_usage": True}
    stream = client.chat.completions.create(**request)
    parts: list[str] = []
    chars = 0
    usage = None
    finish_reason = None
    model = kwargs.get("model")
    try:
        for chunk in stream:
            if getattr(chunk, "error", None):
                raise RuntimeError("Gemma upstream stream failed")
            model = getattr(chunk, "model", None) or model
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or ()
            for choice in choices:
                reason = getattr(choice, "finish_reason", None)
                if reason is not None:
                    finish_reason = reason
                    if reason == "error":
                        raise RuntimeError("Gemma upstream stream terminated with error")
                delta = getattr(getattr(choice, "delta", None), "content", None)
                if not isinstance(delta, str) or not delta:
                    continue
                chars += len(delta)
                if chars > MAX_PUBLIC_ANSWER_CHARS:
                    raise RuntimeError("Original answer exceeds public streaming length limit")
                parts.append(delta)
                # A failing UI observer must never interrupt the provider or
                # rerun inference; final answer is still returned and scored.
                try:
                    on_token(path, delta)
                except Exception:
                    pass
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    answer = "".join(parts).strip()
    if not answer:
        raise RuntimeError("Gemma returned no visible answer text")
    # Match precisely what the original callers need from a ChatCompletion:
    # choices[0].message.content, finish_reason, usage and model.
    return SimpleNamespace(
        model=model,
        usage=usage,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer),
            finish_reason=finish_reason
        )],
    )
