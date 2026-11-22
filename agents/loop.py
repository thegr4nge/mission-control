"""The tool-calling loop every agent runs: ask the model, run the tools it asks for, repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents.model_adapter import ChatModel, PhantomToolCall


@dataclass
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result_text: str


@dataclass
class AgentRun:
    final_text: str = ""
    tool_calls: list[ToolRecord] = field(default_factory=list)
    turns: int = 0
    error: str | None = None


def echo_message(msg: dict) -> dict:
    """The assistant turn as it goes back to the provider: role, content and tool calls only, plus
    Gemini's extra_content (thought signatures, which Gemini requires back). Reasoning fields some
    providers return (Groq's `reasoning`) are dropped, since they reject them on input."""
    out = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if msg.get("extra_content"):
        out["extra_content"] = msg["extra_content"]
    return out


async def run_agent(model: ChatModel, call_tool: Callable[[str, dict], Awaitable[tuple[str, bool]]],
                    tools: list[dict[str, Any]], messages: list[dict[str, Any]], max_turns: int = 12,
                    on_tool: Callable[[ToolRecord], None] | None = None) -> AgentRun:
    """Run until the model answers without asking for a tool. `messages` is extended in place, so a
    caller can continue the same conversation (for a revision) by appending a user turn and rerunning.

    call_tool(name, args) returns (result text, ok). Tools the agent was not given are refused
    without being run, so a role cannot reach outside its remit.
    """
    run = AgentRun()
    allowed = {t["function"]["name"] for t in tools}
    phantoms = 0
    for turn in range(max_turns):
        run.turns = turn + 1
        try:
            msg = model.chat(messages, tools)
        except PhantomToolCall:
            phantoms += 1
            if phantoms > 2:
                raise
            hint = (f"You can only call these tools: {', '.join(sorted(allowed))}." if allowed
                    else "You have no tools in this step. Reply in plain text only.")
            messages.append({"role": "user", "content": hint})
            continue
        calls = msg.get("tool_calls") or []
        for i, call in enumerate(calls):
            call["id"] = call.get("id") or f"call_{turn}_{i}"
        messages.append(echo_message(msg))
        if not calls:
            run.final_text = msg.get("content") or ""
            return run
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                text, ok, args = f"invalid JSON arguments: {exc}", False, {"_raw": fn.get("arguments")}
            else:
                if name not in allowed:
                    text, ok = f"tool {name!r} is not available to this role; use one of {sorted(allowed)}", False
                else:
                    text, ok = await call_tool(name, args)
            rec = ToolRecord(name, args, ok, text)
            run.tool_calls.append(rec)
            if on_tool:
                on_tool(rec)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
    run.error = f"no final answer after {max_turns} turns"
    return run
