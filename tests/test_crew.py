"""Step 5: the crew, run end to end against the real physics server with a scripted model."""

import json
import re

import anyio
import httpx
from mcp.client import Client

from agents.crew import fly
from agents.model_adapter import ChatModel
from evals.runners import server_params

GOAL = ("Send a 1500 kg dry orbiter with 4000 kg of propellant and a 320 s engine from Earth to Mars, "
        "leaving 2026-10-31 and arriving 2027-08-19.")


def scripted_crew():
    """Plays every role. The first briefing slips in an invented number; the revision removes it."""
    seen = {"briefings": 0, "requests": []}

    def reply(msg):
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    def call(name, args, i=0):
        return reply({"role": "assistant", "content": None, "tool_calls": [{
            "id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})

    def say(text):
        return reply({"role": "assistant", "content": text})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["requests"].append(body)
        system = body["messages"][0]["content"]
        msgs = body["messages"]
        last = msgs[-1]
        tools = {t["function"]["name"] for t in body.get("tools", [])}
        if system.startswith("You are the flight director of"):
            return say("Navigator: Earth to Mars, compare the fixed dates in the goal against a C3 search.")
        if system.startswith("You are the navigator"):
            if last["role"] == "user":
                # Try a tool this role does not have first: it must be refused, not run.
                return call("check_delta_v_budget", {"maneuvers_km_s": {"x": 1}, "isp_s": 300,
                                                     "dry_mass_kg": 1, "propellant_mass_kg": 1})
            if "not available to this role" in last["content"]:
                return call("solve_transfer", {"origin": "earth", "destination": "mars",
                                               "departure_date": "2026-10-31", "arrival_date": "2027-08-19"}, 1)
            r = json.loads(last["content"])
            seen["nav"] = r
            return say(f"Fixed dates: C3 {r['c3_km2_s2']} km2/s2, departure burn {r['dv_departure_km_s']} km/s, "
                       f"capture {r['dv_capture_km_s']} km/s, total {r['dv_total_km_s']} km/s. Recommend it.")
        if system.startswith("You are the propulsion engineer"):
            assert tools == {"check_delta_v_budget", "hohmann_transfer"}
            if last["role"] == "user":
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
            if seen["briefings"] == 1:
                return say(core + " A bigger tank of about 7.5 km/s would fix it.")  # invented number
            return say(core)
        raise AssertionError(f"unexpected request: {system[:60]}")

    return ChatModel("groq", "scripted", transport=httpx.MockTransport(handler)), seen


def test_crew_flies_a_mission_and_grounds_its_briefing():
    model, seen = scripted_crew()
    spoken = []

    async def go():
        async with Client(server_params()) as session:
            return await fly(GOAL, model, session, lambda role, text: spoken.append((role, text)), today="2026-11-23")

    m = anyio.run(go)
    assert m.error is None, m.error
    assert [t.role for t in m.turns] == ["flight director", "navigator", "propulsion"]
    nav_calls = m.turns[1].tool_calls
    assert [(c.name, c.ok) for c in nav_calls] == [("check_delta_v_budget", False), ("solve_transfer", True)]
    assert "not available to this role" in nav_calls[0].result_text
    assert [c.name for c in m.turns[2].tool_calls] == ["check_delta_v_budget"]

    # The invented 7.5 was caught, sent back once, and the revised briefing is fully grounded.
    assert m.revisions == 1 and m.ungrounded == [] and "7.5" not in m.briefing
    assert any("7.5" in text for role, text in spoken if role == "system")
    assert seen["budget"]["feasible"] is False and seen["nav"]["c3_km2_s2"] == 9.145
    assert re.search(r"C3 9\.145", m.briefing)
    roles = [r for r, _ in spoken]
    assert roles[0] == "system" and roles[-1] == "flight director"

    # Every crew request carried only that role's tools.
    for body in seen["requests"]:
        names = {t["function"]["name"] for t in body.get("tools", [])}
        system = body["messages"][0]["content"]
        if system.startswith("You are the navigator"):
            assert names == {"next_launch_opportunities", "find_launch_window", "solve_transfer",
                             "propose_flyby_sequence", "propagate", "hohmann_transfer"}
        elif system.startswith("You are the flight director"):
            assert names == set()
