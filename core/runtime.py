"""Runtime guards of the LangGraph runners: replayable message histories and a wall-clock cap."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def openai_safe_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The history with every AI message rebuilt from its text, normalized tool calls and ``source`` tag.

    Replaying a malformed raw tool-call payload from ``additional_kwargs`` can make
    an OpenAI-compatible server reject the next request.
    """
    return [_safe_ai_message(msg) if isinstance(msg, AIMessage) else msg for msg in messages]


def _safe_ai_message(msg: AIMessage) -> AIMessage:
    source = (msg.additional_kwargs or {}).get("source")
    tool_calls = []
    for index, call in enumerate(msg.tool_calls or []):
        name = _field(call, "name")
        if name:
            call_id = _field(call, "id")
            args = _json_safe_args(_field(call, "args", {}))
            tool_calls.append(
                {"name": str(name), "args": args, "id": str(call_id or f"call_{index}"), "type": "tool_call"}
            )
    content = msg.content or ""
    invalid = []
    for index, call in enumerate(msg.invalid_tool_calls or []):
        call_id = str(call.get("id") or f"invalid_{msg.id or 'message'}_{index}")
        invalid.append(call_id)
        tool_calls.append(
            {"name": call.get("name") or "invalid_tool_call", "args": {}, "id": call_id, "type": "tool_call"}
        )
    extras = {"source": source} if source else {}
    invalid = list(dict.fromkeys([*(msg.additional_kwargs.get("invalid_call_ids") or []), *invalid]))
    if invalid:
        extras["invalid_call_ids"] = invalid
    # Preserve message IDs, usage and finish_reason for telemetry and graph reducers.
    return msg.model_copy(
        update={
            "content": content if isinstance(content, str) else str(content),
            "tool_calls": tool_calls,
            "invalid_tool_calls": [],
            "additional_kwargs": extras,
        }
    )


def closed_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Close unexecuted calls at a completed agent boundary, never before a pending ToolNode.

    The inserted result reports an error to the model. It does not execute a tool
    or fabricate a successful result, and it keeps the original tool_call_id.
    """
    safe = openai_safe_messages(messages)
    result = []
    for index, msg in enumerate(safe):
        result.append(msg)
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        end = index + 1
        while end < len(safe) and isinstance(safe[end], ToolMessage):
            end += 1
        replies = {m.tool_call_id for m in safe[index + 1 : end]}
        # Existing replies must stay directly after their assistant tool request.
        result.extend(safe[index + 1 : end])
        for call in msg.tool_calls:
            if call["id"] not in replies:
                reason = (
                    "tool_arguments_invalid"
                    if call["id"] in msg.additional_kwargs.get("invalid_call_ids", [])
                    else "tool_not_executed"
                )
                result.append(
                    ToolMessage(
                        content=json.dumps({"status": "error", "error": reason, "tool": call["name"]}),
                        tool_call_id=call["id"],
                        id=f"tool_error_{call['id']}",
                        status="error",
                    )
                )
    # Replies already included above are otherwise encountered a second time.
    seen, unique = set(), []
    for msg in result:
        marker = id(msg)
        if marker not in seen:
            unique.append(msg)
            seen.add(marker)
    return unique


def safe_tool_node(tools):
    """ToolNode with explicit malformed-argument errors, preserving valid tool execution."""
    from langgraph.prebuilt import ToolNode

    node = ToolNode(tools, handle_tool_errors=True)

    def invoke(state):
        messages = openai_safe_messages(list(state["messages"]))
        ai = messages[-1]
        invalid = set(ai.additional_kwargs.get("invalid_call_ids", []))
        valid = [call for call in ai.tool_calls if call["id"] not in invalid]
        replies = []
        if valid:
            replies = node.invoke({"messages": [ai.model_copy(update={"tool_calls": valid})]})["messages"]
        replies += [
            ToolMessage(
                content=json.dumps({"status": "error", "error": "tool_arguments_invalid"}),
                tool_call_id=call["id"],
                status="error",
            )
            for call in ai.tool_calls
            if call["id"] in invalid
        ]
        return {"messages": [ai, *replies]}

    return invoke


def generation_diagnostics(messages, *, turn_limit_reached=False) -> dict:
    reasons = [m.response_metadata.get("finish_reason") for m in messages if isinstance(m, AIMessage)]
    retrieval = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or not isinstance(msg.content, str):
            continue
        try:
            payload = json.loads(msg.content)
        except ValueError:
            continue
        if isinstance(payload, dict) and "http_status" in payload:
            retrieval.append(
                {
                    "tool_call_id": msg.tool_call_id,
                    "status": payload.get("status"),
                    "http_status": payload["http_status"],
                }
            )
    return {
        "finish_reasons": reasons,
        "output_truncated": "length" in reasons,
        "turn_limit_reached": bool(turn_limit_reached),
        "retrieval_requests": retrieval,
    }


def _field(call, name: str, default=None):
    return call.get(name, default) if isinstance(call, dict) else getattr(call, name, default)


def _json_safe_args(args) -> dict:
    if not isinstance(args, dict):
        return {}
    try:
        safe = json.loads(json.dumps(args, default=str))
    except Exception:
        return {str(k): str(v) for k, v in args.items()}
    return safe if isinstance(safe, dict) else {}


class _RowTimeout(Exception):
    """Raised by :func:`row_timeout` when the wall-clock cap is reached."""


@contextlib.contextmanager
def row_timeout(seconds: int) -> Iterator[None]:
    """Raise after ``seconds`` of wall-clock time (SIGALRM; a no-op off the main thread)."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(signum, frame):
        raise _RowTimeout(f"row exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
