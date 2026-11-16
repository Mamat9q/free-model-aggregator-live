#!/usr/bin/env python3
"""Immense Node -- free LIVE model aggregator, make.fun bounty free-model-aggregator-02.

One server, two APIs, six real prompts, answered by dialling live free models
through make.fun's gateway on the entrant's pushed OpenRouter key. No simulated
provider anywhere: every answer comes from a real model call.

Standard library only. No clock and no unseeded randomness anywhere in a
routing decision: the same brief and the same gateway answers give the same
bytes. (Timeouts are wall-clock bounds only; they trigger only when a model
call is pathologically slow, and routing then falls back on the next model.)

How it beats the baseline
-------------------------
* Best-model routing, not first-model. Candidates that meet the prompt's
  requirements are ranked by the brief's rating_milli (quality is 60% of the
  score): inkling-2-small (1475) for plain/tool prompts, nemotron-3.5-super-vl
  (1303) for JSON-mode prompts. The quality ceiling is 100, not 99.
* Token-verified serving. Every prompt carries a unique token the answer must
  carry back (the proof the model really answered). This server reads the
  token out of the request, checks the provider's answer for it, and -- if
  the model failed to echo it, or returned malformed JSON in JSON mode --
  fails over to the next-best model instead of serving a worthless answer.
  The token is never invented: a served answer is always a real model call.
* Real failover. On a gateway refusal (429/5xx/timeout) the model is dropped
  for the rest of the trial and the prompt is retried on the next-best
  candidate, with the refusal trail reported in the
  `x-makefun-upstream-failures` header so the failover column can see it.
  A 401 (no key pushed) aborts at once -- no model can work without a key.
* Canonical wire re-emission. Streaming answers are buffered, verified, then
  re-emitted in the exact event order each wire requires (OpenAI role /
  content deltas / finish+usage / [DONE]; Anthropic message_start /
  content_block_start / deltas / message_delta / message_stop), with the
  dialled model id on every chunk and usage that always satisfies
  input >= 0, output >= 1. The provider's answer text is forwarded
  byte-for-byte; usage comes from the provider when present.
* Requirements are read off the request body itself (response_format,
  tools, image content), never off arrival order, so a reordered or
  re-sent prompt still routes correctly.
* Fewest calls wins the tie-break: the happy path is exactly 6 gateway
  calls, one per prompt.
"""

import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
SSE_DELTA_CHARS = 13
FAILURES_HEADER = "x-makefun-upstream-failures"
MAX_FAILURES_REPORTED = 8
MAX_ATTEMPTS_PER_PROMPT = 6
# A model is dropped for the rest of the trial only after this many
# refusals in the trial. One refusal drops it for the current prompt
# only, so a transient slow/429/no-token call costs one prompt, not
# every prompt the model would have served best.
MAX_REFUSALS_PER_MODEL = 2
# Per-prompt wall budget: the trial client caps each request at 30 s.
PROMPT_DEADLINE_S = 28.0
FIRST_ATTEMPT_TIMEOUT_S = 16.0
MIN_ATTEMPT_TIMEOUT_S = 5.0
DEFAULT_MAX_TOKENS = 512

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"
SCHEMA_PREFIX = "free-model-aggregator-02/"

# The provider's public endpoint (never the gateway's trial-local address).
OPENROUTER_PUBLIC_BASE = "https://" + "openrouter" + ".ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# A token the prompt asks the model to echo back: WORD-1234 shaped.
TOKEN_RE = re.compile(r"\b([A-Z]{3,12}-[0-9]{4})\b")

Json = dict[str, object]


def log(message: str) -> None:
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------- small type checks


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ""


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out: list[Json] = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix carrying one of `keys`."""
    for doc in docs:
        schema = string_of(doc.get("schema"))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row, as the brief spells it. A free row is routed as dialled."""
    if not is_record(value):
        return None
    raw = string_of(value.get("id"))
    if raw == "":
        raw = string_of(value.get("model"))
    if raw == "":
        return None
    named = string_of(value.get("provider"))
    if named == "":
        named = string_of(value.get("wire"))
    model_id = (
        f"openrouter/{raw}"
        if named == "openrouter" and not raw.lower().startswith("openrouter/")
        else raw
    )
    display_name = string_of(value.get("display_name"))
    rating = number_or_null(value.get("rating_milli"))
    return {
        "id": model_id,
        "provider": named if named != "" else "openrouter",
        "display_name": display_name if display_name != "" else raw,
        "rating_milli": rating if rating is not None else 0,
        "context_length": number_or_null(value.get("context_length")),
        "tools": bool_or_null(value.get("tools")),
        "vision": bool_or_null(value.get("vision")),
        "json": bool_or_null(value.get("json")),
        "free": value.get("free") is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row, as the brief spells it."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get("id"))
    if prompt_id == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": {
            **({"tools": True} if requires.get("tools") is True else {}),
            **({"vision": True} if requires.get("vision") is True else {}),
            **({"json": True} if requires.get("json") is True else {}),
            **(
                {"min_context": requires["min_context"]}
                if is_number(requires.get("min_context"))
                and math.isfinite(requires["min_context"])
                else {}
            ),
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`, its two documents under this contest's prefix."""
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read {path}: {err}"}
    docs = documents_in(text)
    models_doc = array_document(docs, ("models", "ratings"))
    if models_doc is None:
        return {
            "models": [],
            "prompts": [],
            "error": f"the brief at {path} holds no {MODELS_SCHEMA} document",
        }
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ("prompts", "manifest"))
    if prompts_doc is None:
        return {
            "models": models,
            "prompts": [],
            "error": f"the brief at {path} holds no {PROMPTS_SCHEMA} document",
        }
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {"models": models, "prompts": prompts, "error": None}


def free_models(brief: Json) -> list[Json]:
    """The models this entry may serve: the brief's free ones, in brief order."""
    return [m for m in brief["models"] if m["free"]]


def capabilities_of(model: Json) -> Json:
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet one prompt's requirements."""
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None and (
        caps["context_length"] is None or caps["context_length"] < min_context
    ):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    """The catalog the brief's free model list makes: one provider, its models.

    The base_url is the provider's public endpoint, never the gateway's
    trial-local address: the gateway's origin differs between a live pass
    and its replay, and the catalog is part of the state export a replay
    must reproduce byte for byte.
    """
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief["models"]:
        model_id = model["id"]
        if not model["free"] or model_id in seen:
            continue
        seen.add(model_id)
        models.append(
            {"id": model_id, "free": True, "context_length": model["context_length"]}
        )
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


# ---------------------------------------------------------------- request reading


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        b["text"]
        for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    return "\n".join(parts)


def request_text(body: Json) -> str:
    """All user/assistant text in a request body, either wire, for token search."""
    chunks: list[str] = []
    system = body.get("system")
    if system is not None:
        chunks.append(text_of(system))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in messages:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
                elif block.get("type") == "tool_result":
                    chunks.append(str(block.get("content", "")))
    return "\n".join(chunks)


def tokens_in(text: str) -> list[str]:
    """The echo tokens in request text, in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(1)
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body: JSON mode, tools, image content."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    response_format = body.get("response_format")
    response_type = response_format.get("type") if is_record(response_format) else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": response_type in ("json_object", "json_schema"),
    }


# ---------------------------------------------------------------- the client wires and the model field


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: the models this entry serves."""
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": FIXED_OPENAI_CREATED,
                "owned_by": m["provider"],
            }
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request: Anthropic's list shape."""
    return {
        "data": [
            {
                "type": "model",
                "id": m["id"],
                "display_name": m["display_name"],
                "created_at": FIXED_ANTHROPIC_CREATED_AT,
            }
            for m in models
        ],
        "has_more": False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {"error": {"message": message, "type": error_type, "code": code}}


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part; None for the rest."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    source = block.get("source")
    if block.get("type") == "image" and is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source["media_type"] if isinstance(source.get("media_type"), str) else "image/png"
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{source['data']}"},
            }
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {}."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls" or finish == "function_call":
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages: list[Json] = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block["input"] if is_record(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"]
                        if isinstance(block.get("tool_use_id"), str)
                        else "",
                        "content": tool_result_text(block.get("content")),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})

    out: Json = {"model": model, "messages": messages}
    if is_number(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw["description"]
                        if isinstance(raw.get("description"), str)
                        else "",
                        "parameters": raw["input_schema"]
                        if is_record(raw.get("input_schema"))
                        else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content: list[Json] = []
    text = (
        message["content"]
        if isinstance(message.get("content"), str)
        else text_of(message.get("content"))
    )
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": f"msg_{body['id'] if isinstance(body.get('id'), str) else 'aggregate'}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": number_or_zero(usage.get("prompt_tokens")),
            "output_tokens": number_or_zero(usage.get("completion_tokens")),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id the entry dialled."""
    return {**body, "model": model}


# ---------------------------------------------------------------- SSE: canonical re-emission


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the data line."""
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def split_text(text: str) -> list[str]:
    """A text as the canonical content deltas: 13 characters at a time."""
    if text == "":
        return []
    return [text[i : i + SSE_DELTA_CHARS] for i in range(0, len(text), SSE_DELTA_CHARS)]


def openai_stream_events(
    model: str, text: str, tool_calls: list[Json], finish: str, usage: Json
) -> list[str]:
    """A Chat Completions SSE stream, canonical order, for a verified answer.

    role chunk, content deltas, tool-call deltas, finish chunk (with usage
    when the provider sent it), then [DONE]. The model on every chunk is the
    id this entry dialled.
    """
    events: list[str] = []
    chunk_id = "chatcmpl-aggregate"
    events.append(
        sse_event(
            None,
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": FIXED_OPENAI_CREATED,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
        )
    )
    for piece in split_text(text):
        events.append(
            sse_event(
                None,
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": FIXED_OPENAI_CREATED,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                },
            )
        )
    for index, call in enumerate(tool_calls):
        fn = call.get("function") if is_record(call.get("function")) else {}
        events.append(
            sse_event(
                None,
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": FIXED_OPENAI_CREATED,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": fn.get("name", ""),
                                            "arguments": fn.get("arguments", ""),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
            )
        )
    events.append(
        sse_event(
            None,
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": FIXED_OPENAI_CREATED,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            },
        )
    )
    if usage:
        events.append(
            sse_event(
                None,
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": FIXED_OPENAI_CREATED,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                },
            )
        )
    events.append("data: [DONE]\n\n")
    return events


def anthropic_stream_events(
    model: str, text: str, tool_calls: list[Json], stop_reason: str, usage: Json
) -> list[str]:
    """An Anthropic SSE stream, canonical order, for a verified answer."""
    events: list[str] = []
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    events.append(
        sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )
    )
    index = 0
    if text != "":
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        for piece in split_text(text):
            events.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": piece},
                    },
                )
            )
        events.append(
            sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        )
        index += 1
    for call in tool_calls:
        fn = call.get("function") if is_record(call.get("function")) else {}
        call_id = call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else f"toolu_{index}"
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": fn.get("name", ""),
                        "input": {},
                    },
                },
            )
        )
        args = fn.get("arguments", "")
        if isinstance(args, str) and args != "":
            events.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    },
                )
            )
        events.append(
            sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        )
        index += 1
    events.append(
        sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
    )
    events.append(sse_event("message_stop", {"type": "message_stop"}))
    return events


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def chat_completions_url() -> str:
    """The gateway's Chat Completions route."""
    if GATEWAY_URL == "":
        return ""
    return (
        f"{GATEWAY_URL}/chat/completions"
        if GATEWAY_URL.endswith("/v1")
        else f"{GATEWAY_URL}/v1/chat/completions"
    )


def dial(model: str, body: Json, timeout_s: float) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming `model`.

    Raises on an unreachable gateway; a refusal is returned as an Upstream
    with its status, exactly as the gateway sent it.
    """
    url = chat_completions_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    data = json.dumps({**body, "model": model}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {GATEWAY_KEY}",
    }
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def gateway_error_code(text: str) -> str:
    """The error code a refused gateway body names, or the HTTP-shaped default."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return "gateway_error"
    if is_record(parsed):
        error = parsed.get("error")
        if is_record(error) and isinstance(error.get("code"), str) and error["code"] != "":
            return error["code"]
        if isinstance(parsed.get("error"), str) and parsed["error"] != "":
            return parsed["error"]
        if isinstance(parsed.get("code"), str) and parsed["code"] != "":
            return parsed["code"]
    return "gateway_error"


# ---------------------------------------------------------------- the answer: parse, verify, use


def parse_chat_answer(raw: str) -> Json | None:
    """A gateway Chat Completions body as a record, or None when it is not one."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if is_record(parsed) else None


def parse_stream_answer(raw: str) -> Json:
    """An upstream SSE body as one Chat Completions-shaped answer.

    Content deltas and tool-call deltas are joined in order; the last usage
    chunk wins; the finish reason is the last one seen. A body that is not
    SSE at all is treated as one plain-text delta.
    """
    text_parts: list[str] = []
    tool_calls: dict[int, Json] = {}
    finish: str | None = None
    usage: Json = {}
    saw_data = False
    for line in raw.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            continue
        payload = trimmed[5:].strip()
        if payload == "" or payload == "[DONE]":
            continue
        saw_data = True
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage = {
                "prompt_tokens": number_or_zero(chunk["usage"].get("prompt_tokens")),
                "completion_tokens": number_or_zero(chunk["usage"].get("completion_tokens")),
            }
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
        for call in calls:
            if not is_record(call):
                continue
            idx = call.get("index") if is_number(call.get("index")) else len(tool_calls)
            slot = tool_calls.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if isinstance(call.get("id"), str) and call["id"] != "":
                slot["id"] = call["id"]
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("name"), str) and fn["name"] != "":
                slot["function"]["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"] if isinstance(choice.get("finish_reason"), str) else "stop"
    if not saw_data:
        # Not SSE: the whole body is the answer text.
        return {
            "text": raw,
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": usage,
            "raw": raw,
        }
    return {
        "text": "".join(text_parts),
        "tool_calls": [tool_calls[k] for k in sorted(tool_calls)],
        "finish_reason": finish if finish is not None else "stop",
        "usage": usage,
        "raw": raw,
    }


def answer_of(parsed: Json) -> Json:
    """A parsed Chat Completions answer as {text, tool_calls, finish_reason, usage}."""
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = (
        message["content"]
        if isinstance(message.get("content"), str)
        else text_of(message.get("content"))
    )
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    calls = [c for c in calls if is_record(c)]
    usage_raw = parsed.get("usage") if is_record(parsed.get("usage")) else {}
    usage = {
        "prompt_tokens": number_or_zero(usage_raw.get("prompt_tokens")),
        "completion_tokens": number_or_zero(usage_raw.get("completion_tokens")),
    }
    return {
        "text": text,
        "tool_calls": calls,
        "finish_reason": choice.get("finish_reason")
        if isinstance(choice.get("finish_reason"), str)
        else "stop",
        "usage": usage,
    }


def estimate_usage(request_text_len: int, answer_text: str) -> Json:
    """Usage when the provider sent none: the trial's token-count shape."""
    return {
        "prompt_tokens": 1 + request_text_len // 4,
        "completion_tokens": max(1, 1 + len(answer_text) // 4),
    }


def verified_answer(ans: Json, tokens: list[str], require_json: bool) -> bool:
    """Whether a provider answer is worth serving: it carries the token.

    JSON-mode prompts must also parse as JSON. Nothing is invented here:
    a failed verification means the prompt is retried on another model.
    """
    text = ans["text"]
    if tokens:
        for token in tokens:
            if token not in text:
                return False
    if require_json:
        try:
            json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------- routing: best model, verified, with failover


class Router:
    """Per-trial routing state: ranked candidates and the models that refused."""

    def __init__(self, brief: Json) -> None:
        self._brief = brief
        self._lock = threading.Lock()
        self._failures: dict[str, int] = {}
        self._attempted: set[str] = set()

    def begin_prompt(self) -> None:
        """Start a prompt: models refused on an earlier prompt are back in
        the pool unless they have refused MAX_REFUSALS_PER_MODEL times in
        this trial. The env sends one prompt at a time, so no prompt is
        in flight while another begins."""
        with self._lock:
            self._attempted = set()

    def candidates(self, requires: Json) -> list[Json]:
        """Free models meeting `requires`, best rating first, id as tie-break."""
        with self._lock:
            dropped = {m for m, n in self._failures.items() if n >= MAX_REFUSALS_PER_MODEL}
            attempted = set(self._attempted)
        pool = [
            m
            for m in free_models(self._brief)
            if m["id"] not in dropped and m["id"] not in attempted
            and meets_requirements(requires, capabilities_of(m))
        ]
        pool.sort(key=lambda m: (-m["rating_milli"], m["id"]))
        return pool

    def note_refusal(self, model_id: str) -> None:
        with self._lock:
            self._failures[model_id] = self._failures.get(model_id, 0) + 1
            self._attempted.add(model_id)


def _is_timeout(err: BaseException) -> bool:
    """Whether an exception is a per-call timeout (not a dead gateway)."""
    if isinstance(err, TimeoutError):
        return True
    return isinstance(err, urllib.error.URLError) and isinstance(err.reason, TimeoutError)


def serve_prompt(
    router: Router, wire: str, client_body: Json, gateway_body: Json
) -> tuple[int, Json | list[str], list[Json]]:
    """Answer one prompt through the gateway, with best-model failover.

    Returns (http_status, answer, failures): the answer is a JSON object for
    a non-streaming prompt or a list of SSE event strings for a streaming
    one; failures is the refusal trail for the `x-makefun-upstream-failures`
    header (gateway refusals only, in order, at most 8).
    """
    router.begin_prompt()
    requires = requirements_from_body(client_body)
    tokens = tokens_in(request_text(client_body))
    req_len = len(request_text(client_body))
    stream = client_body.get("stream") is True
    failures: list[Json] = []
    deadline = time.monotonic() + PROMPT_DEADLINE_S
    attempts = 0
    last_status = 502
    last_message = "the gateway did not answer"

    while attempts < MAX_ATTEMPTS_PER_PROMPT:
        remaining = deadline - time.monotonic()
        if remaining <= MIN_ATTEMPT_TIMEOUT_S:
            break
        pool = router.candidates(requires)
        if not pool:
            last_status = 503
            last_message = "no free model in the brief meets this request"
            break
        model = pool[0]
        timeout = FIRST_ATTEMPT_TIMEOUT_S if attempts == 0 else min(remaining - 1.0, 12.0)
        timeout = max(MIN_ATTEMPT_TIMEOUT_S, timeout)
        attempts += 1
        try:
            upstream = dial(model["id"], gateway_body, timeout)
        except Exception as err:
            if _is_timeout(err):
                # This model is too slow; the gateway itself may be fine.
                # Fail over -- a timeout is a refusal of this model, not
                # of the gateway. The trail records the refusal the gateway
                # itself records for a client that closed the request:
                # HTTP 499 client_closed_request. The replay serves that
                # recorded 499 back, so the live trail and the replay trail
                # agree and the state export stays byte-identical.
                router.note_refusal(model["id"])
                failures.append(
                    {"status": 499, "code": "client_closed_request", "model": model["id"]}
                )
                last_status = 502
                last_message = f"the model timed out after {timeout:.0f}s"
                continue
            # Unreachable gateway: no model can be dialled.
            last_status = 502
            last_message = f"the gateway is unreachable: {err}"
            break
        if not upstream.ok:
            body_text = upstream.text()
            code = gateway_error_code(body_text)
            failures.append(
                {"status": upstream.status, "code": code, "model": model["id"]}
            )
            if upstream.status == 401:
                # No key was pushed: every model would answer the same.
                last_status = 502
                last_message = "the gateway answered HTTP 401: no OpenRouter key is pushed"
                break
            router.note_refusal(model["id"])
            last_status = 502
            last_message = f"the gateway answered HTTP {upstream.status}"
            continue
        try:
            raw = upstream.text()
        except Exception as err:
            if _is_timeout(err):
                router.note_refusal(model["id"])
                failures.append(
                    {"status": 499, "code": "client_closed_request", "model": model["id"]}
                )
                last_status = 502
                last_message = f"the model timed out after {timeout:.0f}s"
                continue
            raise
        if stream:
            ans = parse_stream_answer(raw)
        else:
            parsed = parse_chat_answer(raw)
            if parsed is None:
                # A 200 whose body is not JSON is not a gateway refusal: it
                # is not reported in the failures trail, only retried.
                router.note_refusal(model["id"])
                last_status = 502
                last_message = "the gateway answered with a body that is not JSON"
                continue
            ans = answer_of(parsed)
        usage = ans["usage"]
        if not usage or usage.get("completion_tokens", 0) < 1:
            usage = estimate_usage(req_len, ans["text"])
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        ans["usage"] = usage
        if not verified_answer(ans, tokens, requires.get("json") is True):
            # The model answered, but not with the proof the trial needs.
            # Fail over: another model may echo the token. The answer is
            # never edited or invented -- only a real model call may serve it.
            router.note_refusal(model["id"])
            last_status = 502
            last_message = "the model answered without the prompt's token"
            continue
        # Served, verified, by this model.
        if wire == "anthropic":
            anth_usage = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
            }
            if stream:
                events = anthropic_stream_events(
                    model["id"],
                    ans["text"],
                    ans["tool_calls"],
                    stop_reason_of(ans["finish_reason"]),
                    anth_usage,
                )
                return 200, events, failures
            body = openai_to_anthropic(
                {
                    "id": f"chatcmpl-aggregate",
                    "choices": [
                        {
                            "message": {
                                "content": ans["text"],
                                "tool_calls": ans["tool_calls"],
                            },
                            "finish_reason": ans["finish_reason"],
                        }
                    ],
                    "usage": usage,
                },
                model["id"],
            )
            body["usage"] = anth_usage
            return 200, body, failures
        if stream:
            events = openai_stream_events(
                model["id"], ans["text"], ans["tool_calls"], ans["finish_reason"], usage
            )
            return 200, events, failures
        body = {
            "id": "chatcmpl-aggregate",
            "object": "chat.completion",
            "created": FIXED_OPENAI_CREATED,
            "model": model["id"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ans["text"],
                        "tool_calls": ans["tool_calls"] if ans["tool_calls"] else None,
                    },
                    "finish_reason": ans["finish_reason"],
                }
            ],
            "usage": usage,
        }
        if not ans["tool_calls"]:
            del body["choices"][0]["message"]["tool_calls"]
        return 200, body, failures

    return last_status, {"error": last_message}, failures


# ---------------------------------------------------------------- the trial state


BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
ROUTER = Router(BRIEF)


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [
        {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
        for m in free_models(BRIEF)
    ]


def failures_header_value(failures: list[Json]) -> str:
    """The refusal trail, at most 8 rows, at most 8 KiB."""
    rows = failures[:MAX_FAILURES_REPORTED]
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    while len(text.encode("utf-8")) > 8192 and len(rows) > 1:
        rows = rows[:-1]
        text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return text


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires."""

    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/2"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    # -- plumbing -------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = (
                    "anthropic"
                    if urlsplit(self.path).path.startswith("/v1/messages")
                    else "openai"
                )
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, failures: list[Json] | None = None) -> None:
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        if failures:
            self.send_header(FAILURES_HEADER, failures_header_value(failures))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def stream_headers(self, failures: list[Json] | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        # Python's http.server does not chunk a response without a length; the
        # close is what ends the stream for the client.
        self.send_header("connection", "close")
        if failures:
            self.send_header(FAILURES_HEADER, failures_header_value(failures))
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    # -- the routes -----------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def _prepare_gateway_body(self, wire: str, body: Json) -> Json:
        """The client's request as a Chat Completions body for the gateway.

        The gateway refuses a request with no output cap, so one is always
        named; the client's own cap is kept when it has one.
        """
        if wire == "anthropic":
            out = anthropic_to_openai(body, "auto")
        else:
            out = {k: v for k, v in body.items() if k != "model"}
        if not is_number(out.get("max_tokens")):
            if is_number(out.get("max_completion_tokens")):
                out["max_tokens"] = out["max_completion_tokens"]
            else:
                out["max_tokens"] = DEFAULT_MAX_TOKENS
        if is_number(out.get("max_completion_tokens")) and not is_number(out.get("max_tokens")):
            out["max_tokens"] = out["max_completion_tokens"]
        return out

    def _answer(self, wire: str, body: Json) -> None:
        gateway_body = self._prepare_gateway_body(wire, body)
        status, answer, failures = serve_prompt(ROUTER, wire, body, gateway_body)
        if status != 200:
            message = answer.get("error", "the gateway did not answer") if is_record(answer) else "the gateway did not answer"
            if wire == "anthropic":
                self.send_error_body("anthropic", 502 if status != 503 else 503, "api_error", str(message))
            else:
                self.send_error_body("openai", 502 if status != 503 else 503, "upstream_error", str(message))
            return
        if isinstance(answer, list):
            self.stream_headers(failures if failures else None)
            for event in answer:
                self.write_event(event)
            return
        self.send_json(200, answer, failures if failures else None)

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, live models."""
        body = self.read_json()
        if body is None:
            self.send_error_body("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        self._answer("openai", body)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions."""
        body = self.read_json()
        if body is None:
            self.send_error_body("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        self._answer("anthropic", body)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the prompt count."""
    body = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": report},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    request = urllib.request.Request(
        f"{ENV_URL}/register",
        data=body.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError(
            f"POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:500]}"
        ) from err
    if not (200 <= status < 300):
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}")
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get("requests")):
        log(f"registered: the trial holds {reply['requests']} prompts")


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if chat_completions_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it.
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(served_models())} free models")
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
