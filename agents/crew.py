"""The built-in crew: a flight director, a navigator and a propulsion engineer.

Give it a plain-English goal. The director scopes the problem, the navigator explores
trajectories with the physics tools, propulsion checks them against the spacecraft, and
the director briefs you on what was tried and why the chosen route won. Agents
orchestrate, pykep computes: the briefing is checked so that every number in it traces to
a tool result or to your goal, and is sent back for revision if not.

  pixi run crew "Send a 1500 kg orbiter with 4000 kg of propellant and a 320 s engine to Mars in the next window"
  pixi run crew --model gemini --model-name gemini-3.1-flash-lite "..."
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import anyio
from mcp.client import Client

from agents.loop import AgentRun, ToolRecord, run_agent
from agents.model_adapter import ChatModel, ModelError, mcp_tools_to_openai
from evals.scoring import grounding

REPO = Path(__file__).resolve().parents[1]
MISSIONS = REPO / "missions"

NAVIGATOR_TOOLS = ("next_launch_opportunities", "find_launch_window", "solve_transfer", "propose_flyby_sequence",
                   "propagate", "hohmann_transfer")
PROPULSION_TOOLS = ("check_delta_v_budget", "hohmann_transfer")

RULE = ("Every number you state about a trajectory, date, orbit, speed, energy or delta-v must be copied from a "
        "tool result in this conversation or from the mission goal. Never estimate, convert, subtract or add numbers "
        "yourself; if you need a derived number, call a tool that returns it.")


def director_plan_prompt(today: str) -> str:
    return (
        "You are the flight director of a spacecraft mission design crew. You have no physics tools: the navigator "
        "computes trajectories and the propulsion engineer checks them against the spacecraft. Read the mission goal "
        "and write a short brief (under 150 words) for the navigator: the origin and destination planets, the "
        "dates if the goal fixes them, what to optimise, and at least one alternative worth comparing (the following "
        "opportunity, Type I against Type II, or a gravity assist). Today is " + today + ". Do not name launch "
        "dates or windows yourself unless the goal gives them: tell the navigator to find the next opportunities "
        "after today with next_launch_opportunities. Say which spacecraft facts the goal gives propulsion to "
        "check. Do not state any numbers that are not in the goal: nothing has been computed yet."
    )


NAVIGATOR_PROMPT = (
    "You are the navigator on a spacecraft mission design crew, with physics tools computed by ESA's pykep. "
    "Follow the flight director's brief. If the goal does not fix the dates, call next_launch_opportunities first "
    "and work from the windows it finds; never assume when a window opens. Compare at least two options with the "
    "tools, then report each option's "
    "departure and arrival dates, time of flight, C3, arrival v-infinity, and departure, capture and total delta-v "
    "exactly as the tools returned them, recommend one, and say why in one or two sentences. If a tool warns that "
    "the best point is on the edge of the search box, widen the search and try again. Keep the report under 250 "
    "words. " + RULE
)

PROPULSION_PROMPT = (
    "You are the propulsion engineer on a spacecraft mission design crew. You have check_delta_v_budget. "
    "If the mission goal gives the spacecraft's dry mass, propellant mass and specific impulse, check every option in "
    "the navigator's report: pass the departure and capture burns exactly as the navigator reported them, with a "
    "10% margin unless the goal says otherwise, and report which options are feasible and the surplus or shortfall. "
    "If the goal does not give those spacecraft facts, do not invent them: say which facts are missing and that "
    "the budget cannot be checked yet. Keep the report under 150 words. " + RULE
)

BRIEFING_PROMPT = (
    "You are the flight director, now briefing the mission owner, a physicist. Using only the crew's reports and "
    "the tool results below, write the briefing (under 300 words): the recommended route with its key numbers, "
    "what else the crew tried and the numbers that ruled it out, why the recommendation won, whether the spacecraft "
    "can fly it, and the model's limits (patched conics, impulsive burns). " + RULE
)


@dataclass
class CrewTurn:
    role: str
    text: str
    tool_calls: list[ToolRecord] = field(default_factory=list)


@dataclass
class Mission:
    goal: str
    model: str
    started_utc: str
    turns: list[CrewTurn] = field(default_factory=list)
    briefing: str = ""
    status: str = "running"  # grounded | ungrounded (numbers left unverified) | failed
    ungrounded: dict[str, list[str]] = field(default_factory=dict)  # role -> numbers no tool returned
    revisions: int = 0
    error: str | None = None
    seconds: float = 0.0

    def all_tools(self) -> list[ToolRecord]:
        return [c for t in self.turns for c in t.tool_calls]


# Explanatory fields every result repeats; the model has them in the tool descriptions already.
_VERBOSE_KEYS = {"model", "method", "note", "search"}


def compact(text: str) -> str:
    """A tool result as the model sees it: minified JSON without the boilerplate fields. Free tiers
    count tokens per minute (Groq: 8k), and indented results with repeated model notes cost several
    times more. The full text is kept for the record and for checking grounding."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in _VERBOSE_KEYS}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(data), separators=(",", ":"))


def _tool_log(records: list[ToolRecord]) -> str:
    lines = [f"- {r.name}({json.dumps(r.arguments, separators=(',', ':'))}) returned: {compact(r.result_text)}"
             for r in records if r.ok]
    return "\n".join(lines) or "(no successful tool calls)"


class RoleFailed(RuntimeError):
    pass


def pick_tools(listing, model: ChatModel, names) -> list[dict]:
    tools = mcp_tools_to_openai(listing.tools, model.provider.simple_schemas)
    return [t for t in tools if t["function"]["name"] in names]


async def fly(goal: str, model: ChatModel, session, say: Callable[[str, str], None],
              today: str | None = None, max_revisions: int = 1) -> Mission:
    """Run the crew on one goal. say(role, text) is called as each crew member speaks.

    Every role's report is checked the same way: each number and ISO date must trace to a tool
    result so far, a search bound the crew chose, or the goal. A report that fails is sent back
    once with the offending numbers; if it still fails it is shown with them marked and the
    mission's status is 'ungrounded', not 'grounded'. Never raises: failures go in the transcript.
    """
    today = today or dt.date.today().isoformat()
    m = Mission(goal, model.label, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    t_start = time.time()

    async def model_tool(name: str, args: dict) -> tuple[str, bool]:
        res = await session.call_tool(name, args)
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        ok = not res.is_error
        model_tool.last_full = text
        return (compact(text) if ok else text), ok

    async def role(name: str, messages: list[dict], tools: list[dict], max_turns: int = 12) -> str:
        """Run one crew member, check its numbers, send it back once if needed."""
        turn = CrewTurn(name, "")
        m.turns.append(turn)

        def keep(rec: ToolRecord) -> None:  # record the full text, not the compacted one the model saw
            rec.result_text = getattr(model_tool, "last_full", rec.result_text) if rec.ok else rec.result_text
            turn.tool_calls.append(rec)

        for attempt in range(max_revisions + 1):
            run = await run_agent(model, model_tool, tools, messages, max_turns, on_tool=keep)
            if run.error:
                raise RoleFailed(f"{name}: {run.error}")
            turn.text = run.final_text.strip()
            ok = [r for r in m.all_tools() if r.ok]
            # Sources: tool results, the crew's own search bounds, the goal, and today's date (which the crew is told).
            _, bad = grounding(turn.text, [r.result_text for r in ok], f"{goal} Today is {today}.",
                               [r.arguments for r in ok])
            if not bad:
                m.ungrounded.pop(name, None)
                break
            m.ungrounded[name] = bad
            if attempt == max_revisions:
                break
            m.revisions += 1
            say("system", f"{name} report sent back: these numbers do not come from any tool result or the goal: "
                + ", ".join(bad))
            messages.append({"role": "user", "content": (
                "These numbers in your report do not appear in any tool result or in the goal: " + ", ".join(bad)
                + ". Rewrite the report using only numbers copied from tool results; drop any number you worked out "
                "yourself, or call a tool that returns it.")})
        shown = turn.text + (f"\n\n[unverified numbers: {', '.join(m.ungrounded[name])}]"
                             if name in m.ungrounded else "")
        say(name, shown)
        return turn.text

    try:
        say("system", f"Mission goal: {goal}")
        listing = await session.list_tools()
        brief = await role("flight director", [
            {"role": "system", "content": director_plan_prompt(today)},
            {"role": "user", "content": f"Mission goal: {goal}"}], [])
        nav_report = await role("navigator", [
            {"role": "system", "content": NAVIGATOR_PROMPT},
            {"role": "user", "content": f"Mission goal: {goal}\n\nFlight director's brief:\n{brief}"}],
            pick_tools(listing, model, NAVIGATOR_TOOLS), max_turns=16)
        nav_records = m.turns[-1].tool_calls
        prop_report = await role("propulsion", [
            {"role": "system", "content": PROPULSION_PROMPT},
            {"role": "user", "content": (f"Mission goal: {goal}\n\nNavigator's report:\n{nav_report}\n\n"
                                         f"Navigator's tool results:\n{_tool_log(nav_records)}")}],
            pick_tools(listing, model, PROPULSION_TOOLS))
        m.briefing = await role("briefing", [
            {"role": "system", "content": BRIEFING_PROMPT},
            {"role": "user", "content": (f"Mission goal: {goal}\n\nYour brief to the crew:\n{brief}\n\n"
                                         f"Navigator's report:\n{nav_report}\n\n"
                                         f"Propulsion's report:\n{prop_report}\n\n"
                                         f"All tool results:\n{_tool_log(m.all_tools())}")}], [])
        m.status = "ungrounded" if m.ungrounded else "grounded"
    except Exception as exc:  # a model, MCP or harness failure: keep the transcript, say what happened
        m.error = str(exc) if isinstance(exc, (RoleFailed, ModelError)) else f"{type(exc).__name__}: {exc}"
        m.status = "failed"
        say("system", f"Crew stopped: {m.error}")
    m.seconds = time.time() - t_start
    return m


def save(m: Mission) -> Path:
    MISSIONS.mkdir(exist_ok=True)
    path = MISSIONS / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(asdict(m), indent=2), encoding="utf-8")
    return path


def main(argv=None) -> int:
    from evals.runners import server_params
    from mcp_server.push import flush, publish_chat

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goal")
    ap.add_argument("--model", default="groq", choices=["groq", "gemini", "ollama"])
    ap.add_argument("--model-name")
    ap.add_argument("--bridge", default="ws://127.0.0.1:8765/ws", help="browser bridge, or 'off'")
    a = ap.parse_args(argv)

    import os
    os.environ["MISSION_CONTROL_BRIDGE"] = a.bridge  # this process publishes the crew's chat

    try:
        model = ChatModel(a.model, a.model_name)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def say(role: str, text: str) -> None:
        print(f"\n== {role} ==\n{text}", flush=True)
        publish_chat(role, text)

    async def go():
        async with Client(server_params(bridge=a.bridge)) as session:
            return await fly(a.goal, model, session, say)

    try:
        m = anyio.run(go)
    except Exception as exc:  # the physics server failed to start, or died outside a role
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        flush()
        return 1
    flush()
    print(f"\nsaved {save(m)}: {m.status}, {m.seconds:.0f} s, {len(m.all_tools())} tool calls, "
          f"{m.revisions} revision(s), unverified numbers: {m.ungrounded or 'none'}")
    return {"grounded": 0, "ungrounded": 3}.get(m.status, 1)


if __name__ == "__main__":
    sys.exit(main())
