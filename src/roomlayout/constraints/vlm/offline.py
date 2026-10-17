"""Offline Anthropic clients for tests and eval replay. Never touches the network.

``offline_client`` is a real ``anthropic.Anthropic`` whose HTTP transport is replaced by
an in-process handler serving prepared Messages API JSON. Requests are serialized and
responses deserialized by the SDK exactly as in production, so ``parse_request`` runs
the same code path it does live; only the bytes on the wire are canned.

Fixture provenance: responses built with ``synthetic_response`` are hand-authored, not
recorded from the API. They are marked ``"id": "msg_synthetic_..."`` and files holding
them carry ``"synthetic": true``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

import anthropic
import httpx2  # the SDK's HTTP library (anthropic 1.x); provides the mock transport

Responder = Callable[[dict[str, Any]], dict[str, Any]]


class OfflineExhausted(RuntimeError):
    """More requests were made than prepared responses."""


def offline_client(
    responses: Iterable[dict[str, Any]] | Responder,
    *,
    requests: list[dict[str, Any]] | None = None,
) -> anthropic.Anthropic:
    """An SDK client that answers from ``responses`` (raw Messages API JSON, served in
    order) or from a callable ``body -> response JSON``. If ``requests`` is given, each
    request is appended to it as ``{"path", "headers", "body"}`` for assertions."""
    if callable(responses):
        responder: Responder = responses
    else:
        queue = list(responses)

        def responder(body: dict[str, Any]) -> dict[str, Any]:
            if not queue:
                raise OfflineExhausted("no prepared response left for this request")
            return queue.pop(0)

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content or b"{}")
        if requests is not None:
            requests.append(
                {"path": request.url.path, "headers": dict(request.headers), "body": body}
            )
        return httpx2.Response(200, json=responder(body))

    return anthropic.Anthropic(
        api_key="offline-no-network",
        base_url="https://offline.invalid",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def synthetic_response(
    output: dict[str, Any] | str | None,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: dict[str, int] | None = None,
    stop_details: dict[str, Any] | None = None,
    fallback_from: str | None = None,
    tag: str = "0",
) -> dict[str, Any]:
    """Hand-authored Messages API response JSON (thinking block + JSON text block).

    ``output`` is the structured-output object (or raw text, or None for no text block).
    ``fallback_from`` adds a leading ``fallback`` block as when a server-side fallback
    served the request from ``model``.
    """
    content: list[dict[str, Any]] = []
    if fallback_from:
        content.append({"type": "fallback", "from": {"model": fallback_from}, "to": {"model": model}})
    content.append({"type": "thinking", "thinking": "", "signature": f"synthetic-signature-{tag}"})
    if output is not None:
        text = output if isinstance(output, str) else json.dumps(output)
        content.append({"type": "text", "text": text})
    return {
        "id": f"msg_synthetic_{tag}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": usage
        or {
            "input_tokens": 900,
            "output_tokens": 600,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 5200,
        },
    }
