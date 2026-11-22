"""Step 5: the crew, run end to end against the real physics server with a scripted model."""

import json
import re

import anyio
import httpx
from mcp.client import Client

from agents.crew import compact, fly
from agents.model_adapter import ChatModel
from evals.runners import server_params

GOAL = ("Send a 1500 kg dry orbiter with 4000 kg of propellant and a 320 s engine from Earth to Mars, "
        "leaving 2026-10-31 and arriving 2027-08-19.")


def scripted_crew(briefing="revises", navigator_loops=False):
    """Plays every role.

    briefing: 'revises' slips an invented number into the first briefing and drops it when sent back;
    'stubborn' keeps inventing; 'invents_date' quotes a launch date no tool returned.
    navigator_loops: the navigator never stops calling tools (it must hit max_turns and fail cleanly).
    """
    seen = {"briefings": 0, "requests": []}

    def reply(msg):
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    def call(name, args, i=0):
        return reply({"role": "assistant", "content": None, "tool_calls": [{
            "id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})

    def say(text):
        return reply({"role": "assistant", "content": text})

    solve = {"origin": "earth", "destination": "mars", "departure_date": "2026-10-31", "arrival_date": "2027-08-19"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["requests"].append(body)
        system = body["messages"][0]["content"]
        last = body["messages"][-1]
        tools = {t["function"]["name"] for t in body.get("tools", [])}
        if system.startswith("You are the flight director of"):
            return say("Navigator: Earth to Mars on the dates in the goal; compare against a C3 search.")
        if system.startswith("You are the navigator"):
            if navigator_loops:
                return call("solve_transfer", solve, len(body["messages"]))
            if last["role"] == "user":
                # Try a tool this role does not have first: it must be refused, not run.
                return call("check_delta_v_budget", {"maneuvers_km_s": {"x": 1}, "isp_s": 300,
                                                     "dry_mass_kg": 1, "propellant_mass_kg": 1})
            if "not available to this role" in last["content"]:
                return call("solve_transfer", solve, 1)
            r = json.loads(last["content"])
            seen["nav"] = r
            return say(f"Fixed dates: C3 {r['c3_km2_s2']} km2/s2, departure burn {r['dv_departure_km_s']} km/s, "
                       f"capture {r['dv_capture_km_s']} km/s, total {r['dv_total_km_s']} km/s. Recommend it.")
        if system.startswith("You are the propulsion engineer"):
            assert tools == {"check_delta_v_budget", "hohmann_transfer"}
            if last["role"] == "user" and "Navigator's report" in last["content"]:
                r = seen["nav"]
                return call("check_delta_v_budget", {
                    "maneuvers_km_s": {"departure": r["dv_departure_km_s"], "capture": r["dv_capture_km_s"]},
                    "isp_s": 320, "dry_mass_kg": 1500, "propellant_mass_kg": 4000, "margin_percent": 10})
            b = json.loads(last["content"])
            seen["budget"] = b
            return say(f"Not feasible: needs {b['required_with_margin_km_s']} km/s, has {b['available_km_s']} km/s.")
        if system.startswith("You are the flight director, now briefing"):
            seen["briefings"] += 1
            r, b = seen["nav"], seen["budget"]
            core = (f"Recommended: leave 2026-10-31, C3 {r['c3_km2_s2']} km2/s2, total {r['dv_total_km_s']} km/s. "
                    f"The spacecraft cannot fly it: it needs {b['required_with_margin_km_s']} km/s with margin "
                    f"and has {b['available_km_s']} km/s.")
            if briefing == "stubborn" or (briefing == "revises" and seen["briefings"] == 1):
                return say(core + " A bigger tank of about 7.5 km/s would fix it.")  # invented number
            if briefing == "invents_date":
                return say(core + " A backup launch on 2027-02-14 is also possible.")  # invented date
            return say(core)
        raise AssertionError(f"unexpected request: {system[:60]}")

    return ChatModel("groq", "scripted", transport=httpx.MockTransport(handler)), seen


def run_crew(model):
    spoken = []

    async def go():
        async with Client(server_params()) as session:
            return await fly(GOAL, model, session, lambda role, text: spoken.append((role, text)), today="2026-11-23")

    return anyio.run(go), spoken


def test_crew_flies_a_mission_and_grounds_its_briefing():
    model, seen = scripted_crew()
    m, spoken = run_crew(model)
    assert m.error is None and m.status == "grounded", (m.error, m.ungrounded)
    assert [t.role for t in m.turns] == ["flight director", "navigator", "propulsion", "briefing"]
    nav_calls = m.turns[1].tool_calls
    assert [(c.name, c.ok) for c in nav_calls] == [("check_delta_v_budget", False), ("solve_transfer", True)]
    assert "not available to this role" in nav_calls[0].result_text
    assert [c.name for c in m.turns[2].tool_calls] == ["check_delta_v_budget"]
    assert '"model"' in nav_calls[1].result_text  # the record keeps the full tool text

    # The invented 7.5 was caught, sent back once, and the revised briefing is fully grounded.
    assert m.revisions == 1 and m.ungrounded == {} and "7.5" not in m.briefing
    assert any("7.5" in text for role, text in spoken if role == "system")
    assert seen["budget"]["feasible"] is False and seen["nav"]["c3_km2_s2"] == 9.145
    assert re.search(r"C3 9\.145", m.briefing)
    roles = [r for r, _ in spoken]
    assert roles[0] == "system" and roles[-1] == "briefing"

    # Every crew request carried only that role's tools, and tool results reached the model compacted.
    for body in seen["requests"]:
        names = {t["function"]["name"] for t in body.get("tools", [])}
        system = body["messages"][0]["content"]
        if system.startswith("You are the navigator"):
            assert names == {"next_launch_opportunities", "find_launch_window", "solve_transfer",
                             "propose_flyby_sequence", "propagate", "hohmann_transfer"}
        elif system.startswith("You are the flight director"):
            assert names == set()
        for msg in body["messages"]:
            if msg["role"] == "tool" and msg["content"].startswith("{"):
                assert '"model"' not in msg["content"] and "\n" not in msg["content"]


def test_briefing_still_ungrounded_after_revision_is_flagged():
    model, _ = scripted_crew(briefing="stubborn")
    m, spoken = run_crew(model)
    assert m.status == "ungrounded" and m.ungrounded == {"briefing": ["7.5"]} and m.revisions == 1
    assert "[unverified numbers: 7.5]" in spoken[-1][1]


def test_invented_launch_date_is_caught():
    model, _ = scripted_crew(briefing="invents_date")
    m, _ = run_crew(model)
    assert m.status == "ungrounded" and m.ungrounded["briefing"] == ["2027-02-14"]


def test_role_that_never_finishes_fails_cleanly_and_keeps_its_work():
    model, _ = scripted_crew(navigator_loops=True)
    m, spoken = run_crew(model)
    assert m.status == "failed" and "navigator" in m.error and "16 turns" in m.error
    assert [t.role for t in m.turns] == ["flight director", "navigator"]
    assert len(m.turns[1].tool_calls) == 16 and all(c.ok for c in m.turns[1].tool_calls)
    assert spoken[-1][0] == "system" and "Crew stopped" in spoken[-1][1]


def test_compact_strips_boilerplate_only():
    text = json.dumps({"c3": 9.145, "model": "long note", "nested": [{"note": "x", "v": 1}]}, indent=2)
    assert json.loads(compact(text)) == {"c3": 9.145, "nested": [{"v": 1}]}
    assert compact("not json") == "not json"


def test_mission_log_reads_as_a_record():
    from agents.crew import mission_log
    model, _ = scripted_crew()
    m, _ = run_crew(model)
    log = mission_log(m)
    assert log.startswith("# Mission log") and "status **grounded**" in log
    assert log.index("## Briefing") < log.index("## Navigator") < log.index("## Propulsion")
    assert "`solve_transfer` ok" in log and "`check_delta_v_budget` FAILED" in log  # the refused call
    assert "C3 9.145" in log
