"""The eval harness: scoring rules, the agent loop, and that every task is solvable with the tools."""

import json

import anyio
import httpx
import pytest

from agents.model_adapter import ChatModel, simplify_schema
from evals.runners import parse_claude_stream, run_openai_tasks
from evals.scoring import Transcript, ToolCall, check_key, grounding, parse_answer, score
from evals.tasks import TASKS, Expect

BY_ID = {t.id: t for t in TASKS}


def test_ten_tasks_with_sources():
    assert len(TASKS) == 10 and len(BY_ID) == 10
    for t in TASKS:
        assert t.required_tools and t.expected
        for k, e in t.expected.items():
            assert e.source, f"{t.id}.{k} has no source"


def test_parse_answer():
    assert parse_answer('blah\nANSWER: {"a": 1, "b": {"c": 2}} trailing') == {"a": 1, "b": {"c": 2}}
    assert parse_answer('ANSWER: {"a": 1}\n... ANSWER: ```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_answer("no marker") is None
    assert parse_answer("ANSWER: {not json}") is None


def test_check_key_kinds():
    assert check_key(Expect("number", 9.144, 0.01), 9.2) and not check_key(Expect("number", 9.144, 0.01), 9.3)
    assert not check_key(Expect("number", 1.0, 0.1), True)
    assert check_key(Expect("date", "2026-10-31", 3), "2026-11-02T12:00") and not check_key(Expect("date", "2026-10-31", 3), "2026-11-04")
    assert check_key(Expect("text", "II"), "Type II") and not check_key(Expect("text", "II"), "I")
    assert check_key(Expect("text", "II"), "type-2") and check_key(Expect("text", "I"), "Type 1")
    assert not check_key(Expect("max", 0.05), -5) and not check_key(Expect("max", 0.05), float("-inf"))
    assert check_key(Expect("bool", False), False) and not check_key(Expect("bool", False), "false")
    assert check_key(Expect("sequence", ["earth", "venus"]), ["Earth", "Venus"])
    assert check_key(Expect("max", 0.05), 0.0) and not check_key(Expect("max", 0.05), 0.2)


def test_grounding_flags_invented_numbers():
    tool = ['{"c3_km2_s2": 9.145, "time_of_flight_hours": 5.257}']
    g, bad = grounding("C3 is 9.145 km2/s2 (about 9.15), flight 315.4 minutes, Type II, 10% margin.", tool,
                       "with a 10% margin")
    assert g == 1.0 and bad == []
    assert grounding("10% margin", tool, "")[1] == ["10"]  # not in the prompt, so not grounded
    g, bad = grounding("C3 is 9.145 and the arrival speed is 2.871 km/s.", tool, "")
    assert bad == ["2.871"] and g == 0.5
    # U+2212 minus, a deficit quoted from a negative surplus, a search bound the model chose, 180 deg,
    # a citation, a month-year from an ISO date.
    g, bad = grounding("x = −3297.8 km, 2.261 km/s short, flights up to 450 days, over 180 deg, 1,234 m, "
                       "as in Curtis Example 4.3, arriving August 2027.",
                       ['{"x": -3297.797, "surplus_km_s": -2.261, "y": 1234, "arrival": "2027-08-19T00:00"}'], "",
                       [{"max_time_of_flight_days": 450}])
    assert bad == [] and g == 1.0
    g, bad = grounding("energy −22.99", ['{"e": -22.68}'], "")
    assert bad == ["-22.99"]


def test_grounding_rejects_loose_roundings_and_small_inventions():
    tool = ['{"c3_km2_s2": 9.144}']
    assert grounding("C3 is 9.14", tool, "")[1] == []
    assert grounding("C3 is 9.1", tool, "")[1] == []       # 9.144 rounds to 9.1
    assert grounding("C3 is 9.18", tool, "")[1] == ["9.18"]  # not a rounding of 9.144
    assert grounding("about 4.1 km/s", tool, "")[1] == ["4.1"]
    assert grounding("", tool, "")[0] == 0.0
    assert grounding("C3 is 2160", tool, "")[1] == ["2160"]  # constants are not unit-scaled
    assert grounding("radius 6 569.48041 km", tool, "radius 6569.48041 km")[1] == []
    assert grounding("searched to 31 Dec 2026", tool, "", [{"latest_departure": "2026-12-31"}])[1] == []
    tiny = ['{"dv_km_s": 1.279e-09}']
    for written in ("1.3×10⁻⁹ km/s", "1.3 x 10^-9 km/s", "1.28e-9 km/s"):
        assert grounding(written, tiny, "")[1] == [], written
    assert grounding("4.2×10⁻⁹ km/s", tiny, "")[1] == ["4.2e-9"]
    assert grounding("flyby Δv 1.279e‑9 km/s", tiny, "")[1] == []  # non-breaking hyphen exponent


def test_grounding_catches_laundered_inputs():
    """Passing an invented delta-v into the budget tool does not make the echoed number grounded."""
    tools = ['{"dv_total_km_s": 5.762}', '{"required_km_s": 3.6, "feasible": true}']
    args = [{"origin": "earth"}, {"maneuvers_km_s": {"TMI": 3.6}, "isp_s": 320, "dry_mass_kg": 1500}]
    g, bad = grounding("It needs 3.6 km/s.", tools, "Isp 320 s, dry mass 1500 kg", args)
    assert "input:3.6" in bad and "3.6" in bad and g == 0.0
    ok_args = [{"origin": "earth"}, {"maneuvers_km_s": {"all": 5.762}, "isp_s": 320, "dry_mass_kg": 1500}]
    g, bad = grounding("It needs 5.762 km/s.", ['{"dv_total_km_s": 5.762}', '{"required_km_s": 5.762}'],
                       "Isp 320 s, dry mass 1500 kg", ok_args)
    assert bad == [] and g == 1.0


def test_score_zero_when_no_tools_called():
    t = BY_ID["T05_hohmann_leo_geo"]
    tr = Transcript(t.id, "m", 'Guessing. ANSWER: {"dv_first_burn_km_s": 2.457, "dv_second_burn_km_s": 1.478, '
                               '"dv_total_km_s": 3.935, "time_of_flight_hours": 5.257}')
    s = score(t, tr)
    assert s.tool_use == 0.0 and s.correctness == 1.0 and s.grounding == 0.0  # right numbers, but not from a tool


# --------------------------------------------------------------------------
# The OpenAI-dialect loop, end to end against the real physics server, with a
# scripted stand-in for the model.
# --------------------------------------------------------------------------

def scripted_model():
    t = BY_ID["T05_hohmann_leo_geo"]
    state = {"n": 0, "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["bodies"].append(body)
        state["n"] += 1
        if state["n"] == 1:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "hohmann_transfer",
                    "arguments": json.dumps({"initial_radius_km": 6569.48041, "final_radius_km": 42159.48487})}}]}
        else:
            result = json.loads(body["messages"][-1]["content"])
            ans = {k: result[k] for k in t.expected}
            msg = {"role": "assistant", "content": (
                "Hohmann transfer: a perigee burn onto the transfer ellipse, then an apogee burn to circularise. "
                f"First burn {ans['dv_first_burn_km_s']} km/s, second {ans['dv_second_burn_km_s']} km/s.\n"
                f"ANSWER: {json.dumps(ans)}")}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    return ChatModel("groq", "scripted", transport=httpx.MockTransport(handler)), state


def test_openai_loop_runs_tools_and_scores_full_marks():
    model, state = scripted_model()
    t = BY_ID["T05_hohmann_leo_geo"]
    [tr] = anyio.run(lambda: run_openai_tasks(model, [t]))
    assert tr.error is None and tr.turns == 2
    assert [c.name for c in tr.tool_calls] == ["hohmann_transfer"] and tr.tool_calls[0].ok
    s = score(t, tr)
    assert (s.tool_use, s.correctness, s.grounding, s.explanation) == (1.0, 1.0, 1.0, 1.0)
    sent_tools = {f["function"]["name"] for f in state["bodies"][0]["tools"]}
    assert "hohmann_transfer" in sent_tools and len(sent_tools) == 6
    assert state["bodies"][1]["messages"][-1]["role"] == "tool"


def test_gemini_schema_has_no_anyof_or_titles():
    from mcp_server.server import server  # noqa: F401  (tools register on import)
    import asyncio
    tools = asyncio.run(server.list_tools())
    for t in tools:
        text = json.dumps(simplify_schema(t.input_schema))
        assert '"anyOf"' not in text and '"title"' not in text, t.name


def test_parse_claude_stream():
    down = parse_claude_stream("T03", "c", json.dumps(
        {"type": "system", "subtype": "init", "mcp_servers": [{"name": "physics", "status": "failed"}]}))
    assert down.error and "did not connect" in down.error
    lines = [
        {"type": "system", "subtype": "init", "model": "claude-x", "mcp_servers": [{"name": "physics", "status": "connected"}]},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "u1", "name": "mcp__physics__solve_transfer", "input": {"origin": "earth"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "u1", "content": [{"type": "text", "text": '{"c3_km2_s2": 9.145}'}]}]}},
        {"type": "result", "result": 'Done. ANSWER: {"c3_km2_s2": 9.145}', "num_turns": 2, "is_error": False},
    ]
    tr = parse_claude_stream("T03", "claude-cli:default", "\n".join(json.dumps(x) for x in lines))
    assert tr.model == "claude-cli:claude-x" and tr.turns == 2 and tr.error is None
    assert tr.tool_calls == [ToolCall("solve_transfer", {"origin": "earth"}, True, '{"c3_km2_s2": 9.145}')]


# --------------------------------------------------------------------------
# Every task is solvable: the tools reproduce each known answer.
# --------------------------------------------------------------------------

def test_tools_reproduce_every_known_answer():
    from physics.bodies import to_iso, to_mjd2000
    from physics.budget import check_budget
    from physics.flyby import optimise_sequence
    from physics.hohmann import hohmann
    from physics.propagation import propagate
    from physics.references import CURTIS_4_3, VALLADO_6_1
    from physics.transfers import find_launch_window, solve_transfer

    def check(task_id, got):
        for k, e in BY_ID[task_id].expected.items():
            assert check_key(e, got[k]), f"{task_id}.{k}: tools give {got[k]!r}, expected {e.value!r}"

    for tid, (a, b) in {"T01_window_2026": ("2026-09-01", "2027-03-01"),
                        "T02_window_2028": ("2028-09-01", "2029-03-01")}.items():
        w = find_launch_window("earth", "mars", to_mjd2000(a), to_mjd2000(b), 100, 450, "c3")
        check(tid, {"departure_date": to_iso(w.best.depart_mjd2000), "c3_km2_s2": w.best.c3_km2_s2})

    for tid in ("T03_transfer_2026", "T04_transfer_2028_type1"):
        import re
        dep, arr = re.findall(r"\d{4}-\d{2}-\d{2}", BY_ID[tid].prompt)
        tr = solve_transfer("earth", "mars", to_mjd2000(dep), to_mjd2000(arr) - to_mjd2000(dep))
        check(tid, {"c3_km2_s2": tr.c3_km2_s2, "vinf_arrival_km_s": tr.vinf_arr_km_s,
                    "trajectory_type": tr.trajectory_type})

    h = hohmann(VALLADO_6_1["r1_km"], VALLADO_6_1["r2_km"])
    check("T05_hohmann_leo_geo", {"dv_first_burn_km_s": h.dv1_km_s, "dv_second_burn_km_s": h.dv2_km_s,
                                  "dv_total_km_s": h.dv_total_km_s, "time_of_flight_hours": h.tof_s / 3600})

    t0, t1 = to_mjd2000("1973-10-01"), to_mjd2000("1974-01-29")
    via = optimise_sequence(["earth", "venus", "mercury"], t0, t1, arrival_mode="flyby")
    direct = optimise_sequence(["earth", "mercury"], t0, t1, arrival_mode="flyby")
    best = via if via.dv_total_km_s < direct.dv_total_km_s else direct
    check("T06_mercury_venus_flyby", {"best_sequence": list(best.bodies), "venus_flyby_dv_km_s": via.encounters[1].dv_km_s,
                                      "venus_flyby_date": to_iso(via.encounters[1].mjd2000)})

    from physics.references import PROPAGATION_CASES
    c = next(c for c in PROPAGATION_CASES if c.name == "Curtis Example 3.7")
    p = propagate(c.r0_km, c.v0_km_s, 3600.0, "earth")
    check("T07_propagate_curtis_3_7", {"final_x_km": p.r_km[0], "final_y_km": p.r_km[1]})

    el = propagate(CURTIS_4_3["r_km"], CURTIS_4_3["v_km_s"], 0.0, "earth").initial
    check("T08_elements_curtis_4_3", {"semi_major_axis_km": el.a_km, "eccentricity": el.e,
                                      "inclination_deg": el.i_deg, "raan_deg": el.raan_deg})

    tr = solve_transfer("earth", "mars", to_mjd2000("2026-10-31"), 292.0)
    b = check_budget({"departure": tr.dv_dep_km_s, "capture": tr.dv_arr_km_s}, 320.0, 1500.0, 4000.0, 10.0)
    check("T09_budget_mars_2026", {"feasible": b.feasible, "required_km_s": b.required_km_s,
                                   "available_km_s": b.available_km_s})

    w = find_launch_window("earth", "mars", to_mjd2000("2026-09-01"), to_mjd2000("2026-10-15"), 100, 450, "c3")
    check("T10_window_constraint_binds", {"departure_date": to_iso(w.best.depart_mjd2000),
                                          "deadline_binds": w.departure_at_edge, "c3_km2_s2": w.best.c3_km2_s2})


def test_flyby_task_solvable_through_the_tool_with_defaults():
    """T06 via the MCP tool itself (default candidates, max_flybys=1), not just the physics layer."""
    from mcp_server.server import propose_flyby_sequence
    r = propose_flyby_sequence("earth", "mercury", "1973-10-01", "1974-01-29", arrival="flyby")
    best = r["ranked"][0]
    venus = next(e for e in best["encounters"] if e["role"] == "flyby")
    check = {"best_sequence": best["sequence"], "venus_flyby_dv_km_s": venus["dv_km_s"],
             "venus_flyby_date": venus["date_utc"]}
    for k, e in BY_ID["T06_mercury_venus_flyby"].expected.items():
        assert check_key(e, check[k]), (k, check[k])


def test_wrong_answers_fail():
    """The tolerances are tight enough to reject a near miss."""
    e = BY_ID["T01_window_2026"].expected
    assert not check_key(e["c3_km2_s2"], 9.144 * 1.02) and not check_key(e["departure_date"], "2026-11-05")
    assert not check_key(BY_ID["T10_window_constraint_binds"].expected["departure_date"], "2026-10-31")
