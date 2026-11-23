"""End to end: launch the MCP server exactly as a client would (pixi, stdio) and call every tool."""

import json
import math
from pathlib import Path

import anyio
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

MANIFEST = str(Path(__file__).resolve().parents[1] / "pixi.toml")
EXPECTED_TOOLS = {"solve_transfer", "find_launch_window", "propagate", "check_delta_v_budget", "propose_flyby_sequence",
                  "hohmann_transfer", "next_launch_opportunities", "design_dsm_trajectory"}


def _reject_constant(name):
    raise AssertionError(f"tool output text contains {name}, which is not valid JSON")


def body(result):
    """Structured content, after checking that the text block an agent reads is strict JSON."""
    assert not result.is_error, [c.text for c in result.content]
    json.loads(result.content[0].text, parse_constant=_reject_constant)
    return result.structured_content


async def _session():
    from mcp.client.stdio import get_default_environment
    params = StdioServerParameters(command="pixi", args=["run", "--manifest-path", MANIFEST, "mcp-server"],
                                   env={**get_default_environment(), "MISSION_CONTROL_BRIDGE": "off"})
    out = {}
    async with Client(params) as c:
        out["tools"] = {t.name for t in (await c.list_tools()).tools}
        out["transfer"] = body(await c.call_tool("solve_transfer", {
            "origin": "earth", "destination": "mars", "departure_date": "2026-10-31", "arrival_date": "2027-08-19"}))
        out["window"] = body(await c.call_tool("find_launch_window", {
            "origin": "earth", "destination": "mars", "earliest_departure": "2028-09-01",
            "latest_departure": "2029-03-01", "min_time_of_flight_days": 100, "max_time_of_flight_days": 450,
            "objective": "c3"}))
        out["propagate"] = body(await c.call_tool("propagate", {
            "position_km": [7000, -12124, 0], "velocity_km_s": [2.6679, 4.6210, 0],
            "duration_days": 1 / 24, "central_body": "earth"}))
        out["budget"] = body(await c.call_tool("check_delta_v_budget", {
            "maneuvers_km_s": {"TMI": 3.636, "MOI": 2.053}, "isp_s": 320, "dry_mass_kg": 1000,
            "propellant_mass_kg": 5000}))
        out["flyby"] = body(await c.call_tool("propose_flyby_sequence", {
            "origin": "earth", "destination": "mercury", "earliest_departure": "1973-10-01",
            "latest_departure": "1974-01-29", "sequences": [["earth", "mercury"], ["earth", "venus", "mercury"]],
            "arrival": "flyby"}))
        out["equatorial"] = body(await c.call_tool("propagate", {
            "position_km": [7000, 0, 0], "velocity_km_s": [0, 7.546, 0], "duration_days": 0.01,
            "central_body": "earth"}))
        out["hohmann"] = body(await c.call_tool("hohmann_transfer", {
            "initial_radius_km": 6569.48041, "final_radius_km": 42159.48487}))
        out["opps"] = body(await c.call_tool("next_launch_opportunities", {
            "origin": "earth", "destination": "mars", "after_date": "2026-11-23", "count": 1}))
        out["dsm"] = body(await c.call_tool("design_dsm_trajectory", {
            "sequence": ["earth", "mars"], "earliest_departure": "2028-06-01", "latest_departure": "2029-03-01",
            "effort": "quick"}))
        late = await c.call_tool("solve_transfer", {
            "origin": "earth", "destination": "mars", "departure_date": "2160-01-01",
            "arrival_date": "2160-10-01", "ephemeris": "de440s"})
        out["de440s_out_of_range"] = (late.is_error, " ".join(getattr(x, "text", "") for x in late.content))
        bad = await c.call_tool("solve_transfer", {
            "origin": "earth", "destination": "mars", "departure_date": "not a date", "arrival_date": "2027-08-19"})
        out["bad_date"] = (bad.is_error, " ".join(getattr(x, "text", "") for x in bad.content))
    return out


@pytest.fixture(scope="module")
def session():
    return anyio.run(_session)


def test_lists_all_tools(session):
    assert session["tools"] == EXPECTED_TOOLS


def test_solve_transfer_matches_nasa_2026_min_c3(session):
    t = session["transfer"]
    assert t["c3_km2_s2"] == 9.145  # NASA/TM-2010-216764 Table 2: 9.144
    assert t["vinf_arrival_km_s"] == 2.729  # Table 2: 2.729
    assert t["trajectory_type"] == "II"


def test_delta_v_matches_hand_calculation(session):
    """Independent of the code's constants: vis-viva with Earth mu 398600.4418 km^3/s^2, radius 6378 km,
    Mars mu 42828 km^3/s^2, radius 3397 km (jpl_lp values), at the tool's own C3 and arrival v_inf."""
    t = session["transfer"]
    mu_e, r_e = 398600.4418, 6378.0 + 200.0
    mu_m, r_m = 42828.0, 3397.0 + 300.0
    dep = math.sqrt(t["c3_km2_s2"] + 2 * mu_e / r_e) - math.sqrt(mu_e / r_e)
    cap = math.sqrt(t["vinf_arrival_km_s"] ** 2 + 2 * mu_m / r_m) - math.sqrt(mu_m / r_m)
    assert t["dv_departure_km_s"] == pytest.approx(dep, rel=1e-3)
    assert t["dv_capture_km_s"] == pytest.approx(cap, rel=1e-3)
    assert t["dv_total_km_s"] == pytest.approx(dep + cap, rel=1e-3)


def test_equatorial_orbit_gives_null_angles_not_nan(session):
    el = session["equatorial"]["initial_elements"]
    assert el["raan_deg"] is None and el["inclination_deg"] == 0.0
    assert el["semi_major_axis_km"] == pytest.approx(7000.0, rel=1e-3)


def test_physics_engine_failure_explains_itself(session):
    is_error, text = session["de440s_out_of_range"]
    assert is_error and "physics engine failed" in text and len(text) > 60


def test_find_launch_window_matches_nasa_2028(session):
    w = session["window"]
    assert abs(w["c3_km2_s2"] - 8.928) / 8.928 < 2e-3  # Table 1
    assert w["departure_utc"].startswith("2028-12-0")
    assert w["departure_at_search_edge"] is False and "warning" not in w


def test_propagate_matches_curtis_3_7(session):
    p = session["propagate"]
    assert p["final_position_km"][0] == pytest.approx(-3296.8, rel=1e-3)
    assert p["final_velocity_km_s"][1] == pytest.approx(-0.96309, rel=2e-3)


def test_budget(session):
    b = session["budget"]
    assert b["feasible"] is False and b["required_km_s"] == 5.689


def test_flyby_ranks_venus_assist_first(session):
    ranked = session["flyby"]["ranked"]
    assert ranked[0]["sequence"] == ["earth", "venus", "mercury"]
    assert ranked[0]["encounters"][1]["dv_km_s"] < 0.01


def test_hohmann_matches_vallado_6_1(session):
    h = session["hohmann"]
    assert (h["dv_first_burn_km_s"], h["dv_second_burn_km_s"], h["dv_total_km_s"], h["time_of_flight_hours"]) == (
        2.457, 1.478, 3.935, 5.257)  # Vallado 4th ed. Example 6-1 (recalled): 2.457038, 1.478187, 3.935224, 5.256713 h


def test_next_launch_opportunities_over_mcp(session):
    o = session["opps"]
    [opp] = o["opportunities"]
    assert opp["best_c3"]["departure_utc"].startswith("2028-12-0")  # NASA/TM-2010-216764 Table 3: 2028-12-02
    assert abs(opp["best_c3"]["c3_km2_s2"] - 8.928) / 8.928 < 2e-3
    assert opp["suggested_earliest_departure"] < opp["best_c3"]["departure_utc"] < opp["suggested_latest_departure"]


def test_dsm_trajectory_over_mcp(session):
    d = session["dsm"]
    assert d["sequence"] == ["earth", "mars"] and d["dv_deep_space_total_km_s"] < 0.01
    assert abs(d["dv_total_km_s"] - 5.884) / 5.884 < 1e-3  # equals the Lambert optimum for this window
    assert [e["role"] for e in d["encounters"]] == ["departure", "arrival"]
    assert len(d["deep_space_manoeuvres"]) == 1


def test_bad_input_is_a_tool_error_not_a_crash(session):
    is_error, text = session["bad_date"]
    assert is_error and "cannot parse date" in text


def test_outputs_are_four_significant_figures(session):
    def check(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in ("final_position_km", "final_velocity_km_s"):
                    check(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for v in obj:
                check(v, path)
        elif isinstance(obj, float) and obj != 0:
            assert float(f"{obj:.4g}") == obj, f"{path} = {obj!r} is not 4 s.f."
    for key in ("transfer", "window", "propagate", "budget", "flyby", "equatorial", "hohmann", "opps", "dsm"):
        check(session[key], key)
    json.dumps(session["flyby"])  # serialisable
