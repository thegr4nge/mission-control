"""Run a task on a model and record the transcript: what it called, what came back, what it said."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from agents.model_adapter import ChatModel, ModelError, mcp_tools_to_openai
from evals.scoring import SYSTEM_PROMPT, ToolCall, Transcript, answer_instructions
from evals.tasks import Task

REPO = Path(__file__).resolve().parents[1]
MANIFEST = str(REPO / "pixi.toml")
SERVER_NAME = "physics"


def pixi_exe() -> str:
    return shutil.which("pixi") or "pixi"


def server_params() -> StdioServerParameters:
    return StdioServerParameters(command=pixi_exe(), args=["run", "--manifest-path", MANIFEST, "mcp-server"])


def _result_text(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


def echo_message(msg: dict) -> dict:
    """The assistant turn as it goes back to the provider: role, content and tool calls only, plus
    Gemini's extra_content (thought signatures, which Gemini requires back). Reasoning fields that
    some providers return (Groq's `reasoning`) are dropped, since they reject them on input."""
    out = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if msg.get("extra_content"):
        out["extra_content"] = msg["extra_content"]
    return out


async def run_openai_tasks(model: ChatModel, tasks: list[Task], max_turns: int = 12, on_done=None) -> list[Transcript]:
    """One physics server for the whole run; a fresh conversation per task."""
    out = []
    async with Client(server_params()) as session:
        tools = mcp_tools_to_openai((await session.list_tools()).tools, model.provider.simple_schemas)
        for task in tasks:
            tr = Transcript(task.id, model.label, "")
            t0 = time.time()
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": answer_instructions(task)}]
            try:
                for turn in range(max_turns):
                    tr.turns = turn + 1
                    msg = model.chat(messages, tools)
                    calls = msg.get("tool_calls") or []
                    for i, call in enumerate(calls):
                        call["id"] = call.get("id") or f"call_{turn}_{i}"
                    messages.append(echo_message(msg))
                    if not calls:
                        tr.final_text = msg.get("content") or ""
                        break
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
                            res = await session.call_tool(name, args)
                            text, ok = _result_text(res), not res.is_error
                        tr.tool_calls.append(ToolCall(name, args, ok, text))
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                else:
                    tr.error = f"no final answer after {max_turns} turns"
            except ModelError as exc:
                tr.error = str(exc)
            except Exception as exc:  # one broken task must not lose the run
                tr.error = f"harness error: {type(exc).__name__}: {exc}"
            tr.seconds = time.time() - t0
            out.append(tr)
            if on_done:
                on_done(tr)
    return out


def run_claude_task(task: Task, model: str | None = None, timeout_s: float = 600.0) -> Transcript:
    """Claude through the Claude Code CLI on the user's plan (no API key, no API spend).

    User settings are not loaded, so hooks and plugins stay out of the eval; no CLAUDE.md
    is loaded; only the physics tools are available; it runs in an empty temporary folder.
    """
    label = f"claude-cli:{model or 'default'}"
    claude = shutil.which("claude")
    if claude is None:
        return Transcript(task.id, label, "", error="claude CLI not found on PATH")
    with tempfile.TemporaryDirectory(prefix="mc-eval-", ignore_cleanup_errors=True) as tmp:
        cfg = Path(tmp) / "mcp.json"
        p = server_params()
        cfg.write_text(json.dumps({"mcpServers": {SERVER_NAME: {"command": p.command, "args": p.args}}}),
                       encoding="utf-8")
        cmd = [claude, "-p", "--setting-sources", "project", "--disable-slash-commands",
               "--mcp-config", str(cfg), "--strict-mcp-config", "--tools", "",
               "--allowedTools", f"mcp__{SERVER_NAME}__*", "--no-session-persistence",
               "--output-format", "stream-json", "--verbose", "--system-prompt", SYSTEM_PROMPT]
        if model:
            cmd += ["--model", model]
        cmd.append(answer_instructions(task))
        t0 = time.time()
        # No CLAUDE.md either: the user's personal instructions would otherwise ride along.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}
        proc = subprocess.Popen(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", shell=False, env=env)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Kill the whole tree: claude, the pixi it launched and the physics server under that.
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, shell=False)
            stdout, stderr = proc.communicate()
            tr = parse_claude_stream(task.id, label, stdout)  # keep what happened before the hang
            last = [ln for ln in stdout.splitlines() if ln.strip()][-1:] or [stderr.strip()]
            tr.error = f"timed out after {timeout_s:.0f} s; last output: {last[0][:300]}"
            tr.seconds = timeout_s
            return tr
    tr = parse_claude_stream(task.id, label, stdout)
    tr.seconds = time.time() - t0
    if not tr.final_text and not tr.error:
        tr.error = f"claude exited {proc.returncode}: {stderr.strip()[:300]}"
    return tr


def parse_claude_stream(task_id: str, label: str, stdout: str) -> Transcript:
    tr = Transcript(task_id, label, "")
    pending: dict[str, tuple[str, dict]] = {}
    prefix = f"mcp__{SERVER_NAME}__"
    for line in stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = d.get("type")
        if kind == "system" and d.get("subtype") == "init":
            if d.get("model"):
                tr.model = f"claude-cli:{d['model']}"
            status = {s.get("name"): s.get("status") for s in d.get("mcp_servers", [])}
            if status.get(SERVER_NAME) != "connected":
                tr.error = f"physics MCP server did not connect (status {status.get(SERVER_NAME)!r}): harness fault"
        elif kind == "assistant":
            for c in d["message"].get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    pending[c["id"]] = (c["name"].removeprefix(prefix), c.get("input") or {})
        elif kind == "user":
            content = d["message"].get("content", [])
            for c in content if isinstance(content, list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    name, args = pending.pop(c.get("tool_use_id"), ("?", {}))
                    body = c.get("content")
                    text = body if isinstance(body, str) else "\n".join(
                        x.get("text", "") for x in (body or []) if isinstance(x, dict))
                    tr.tool_calls.append(ToolCall(name, args, not c.get("is_error", False), text))
        elif kind == "result":
            tr.final_text = d.get("result") or ""
            tr.turns = d.get("num_turns") or 0
            if d.get("is_error") and not tr.error:
                tr.error = f"claude reported an error: {tr.final_text[:300]}"
    if "Not logged in" in tr.final_text:
        tr.error = "claude CLI is not logged in: run `claude auth login`"
    return tr
