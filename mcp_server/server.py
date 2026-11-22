"""Mission Control physics MCP server (stdio).

Agents orchestrate, pykep computes: every number an agent quotes about a trajectory
must come from one of these tools. Reported quantities are rounded to four
significant figures with units in the key names; state vectors keep full precision
so they can be fed back into propagate.

Run:  pixi run mcp-server
"""

from __future__ import annotations

import functools
import math
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from physics.bodies import PLANETS, to_iso, to_mjd2000
from physics.budget import check_budget
from physics.flyby import candidate_sequences, default_flyby_candidates, optimise_sequence
from physics.hohmann import hohmann
from physics.visual import planet_tracks, transfer_path
from mcp_server.push import publish
from physics.propagation import Elements, propagate as kepler_propagate
from physics.transfers import (
    DEFAULT_ARRIVAL_ALT_KM,
    DEFAULT_DEPARTURE_ALT_KM,
    Transfer,
    find_launch_window as window_search,
    launch_opportunities,
    solve_transfer as lambert_transfer,
)

Planet = Literal["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
Ephemeris = Literal["jpl_lp", "de440s"]

MODEL_NOTE = (
    "Patched conics: Sun-centred single-revolution prograde Lambert arcs, impulsive burns from/into "
    "circular orbits, planets as points except for flyby turning. Dates are UTC as pykep epochs (no leap seconds)."
)

server = MCPServer(
    "mission-control-physics",
    instructions=(
        "Physics tools for interplanetary mission design, computed with ESA's pykep. "
        "Quote only numbers these tools return; never estimate a delta-v, date or orbit yourself. "
        f"Planets: {', '.join(PLANETS)}. Dates are ISO strings such as 2026-10-31 or 2026-10-31T12:00."
    ),
)


def user_errors(fn):
    """Bad input raises ValueError in physics/; send its message to the client as a tool error.

    MCPServer withholds the text of any other exception, which would leave an agent
    with no idea what to fix.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # pykep RuntimeError, SPICE errors (e.g. de440s date out of range)
            raise ToolError(f"physics engine failed: {type(exc).__name__}: {exc}") from exc
    return wrapper


def sig4(x: float | None) -> float | None:
    """Round to four significant figures. Non-finite values become None (JSON null), never NaN."""
    if x is None or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")


def scene(bodies: list[str], epochs: list[float], ephemeris: str = "jpl_lp") -> dict[str, Any]:
    """3D geometry for the browser: Lambert legs and every planet's track over the mission, from the
    same ephemeris the tool used, so the arc drawn is the arc whose numbers were reported."""
    return {"legs": transfer_path(bodies, epochs, ephemeris),
            "tracks": planet_tracks(PLANETS, epochs[0], epochs[-1], ephemeris=ephemeris)}


def porkchop(res, max_axis: int = 120) -> dict[str, Any]:
    """The window search's coarse grid, thinned to at most max_axis points a side for display."""
    deps, tofs, vals = res.grid_departures, res.grid_tofs, res.grid_values
    si = max(1, -(-len(deps) // max_axis))
    sj = max(1, -(-len(tofs) // max_axis))
    return {"objective": res.objective,
            "departures_mjd2000": list(deps[::si]), "tofs_days": list(tofs[::sj]),
            "values": [[sig4(v) for v in row[::sj]] for row in vals[::si]],
            "best": {"departure_mjd2000": res.best.depart_mjd2000, "tof_days": res.best.tof_days}}


def transfer_dict(tr: Transfer) -> dict[str, Any]:
    return {
        "origin": tr.origin,
        "destination": tr.destination,
        "departure_utc": to_iso(tr.depart_mjd2000),
        "arrival_utc": to_iso(tr.arrive_mjd2000),
        "time_of_flight_days": sig4(tr.tof_days),
        "trajectory_type": tr.trajectory_type,
        "transfer_angle_deg": sig4(tr.transfer_angle_deg),
        "c3_km2_s2": sig4(tr.c3_km2_s2),
        "vinf_departure_km_s": sig4(tr.vinf_dep_km_s),
        "vinf_arrival_km_s": sig4(tr.vinf_arr_km_s),
        "dv_departure_km_s": sig4(tr.dv_dep_km_s),
        "dv_capture_km_s": sig4(tr.dv_arr_km_s),
        "dv_total_km_s": sig4(tr.dv_total_km_s),
        "departure_orbit": f"{tr.departure_alt_km:g} km circular at {tr.origin}",
        "capture_orbit": f"{tr.arrival_alt_km:g} km circular at {tr.destination}",
        "ephemeris": tr.ephemeris,
        "model": MODEL_NOTE,
    }


def elements_dict(el: Elements) -> dict[str, Any]:
    return {
        "semi_major_axis_km": sig4(el.a_km),
        "eccentricity": sig4(el.e),
        "inclination_deg": sig4(el.i_deg),
        "raan_deg": sig4(el.raan_deg),
        "argument_of_periapsis_deg": sig4(el.argp_deg),
        "true_anomaly_deg": sig4(el.true_anomaly_deg),
        "period_days": sig4(el.period_days),
        "note": ("null angles are undefined for this orbit: RAAN and argument of periapsis need an inclined "
                 "orbit, argument of periapsis and true anomaly need a non-circular one; period needs e < 1"),
    }


@server.tool(structured_output=True, title="Solve transfer")
@user_errors
def solve_transfer(
    origin: Planet,
    destination: Planet,
    departure_date: str,
    arrival_date: str,
    departure_orbit_altitude_km: float = DEFAULT_DEPARTURE_ALT_KM,
    capture_orbit_altitude_km: float = DEFAULT_ARRIVAL_ALT_KM,
    ephemeris: Ephemeris = "jpl_lp",
) -> dict[str, Any]:
    """Ballistic transfer between two planets on fixed dates.

    Returns departure C3, hyperbolic excess speeds, the burn to leave a circular parking
    orbit, the burn to capture into a circular orbit at the destination, and the total.
    Trajectory type I means a transfer angle under 180 deg, type II 180-360 deg.
    """
    t0, t1 = to_mjd2000(departure_date), to_mjd2000(arrival_date)
    tr = lambert_transfer(origin, destination, t0, t1 - t0, departure_orbit_altitude_km,
                          capture_orbit_altitude_km, ephemeris)
    out = transfer_dict(tr)
    publish("solve_transfer", out, lambda: scene([origin, destination], [t0, t1], ephemeris))
    return out


@server.tool(structured_output=True, title="Find launch window")
@user_errors
def find_launch_window(
    origin: Planet,
    destination: Planet,
    earliest_departure: str,
    latest_departure: str,
    min_time_of_flight_days: float,
    max_time_of_flight_days: float,
    objective: Literal["dv_total", "c3", "vinf_arrival"] = "dv_total",
    departure_orbit_altitude_km: float = DEFAULT_DEPARTURE_ALT_KM,
    capture_orbit_altitude_km: float = DEFAULT_ARRIVAL_ALT_KM,
    ephemeris: Ephemeris = "jpl_lp",
) -> dict[str, Any]:
    """Best ballistic transfer in a departure window, by grid search then local refinement.

    objective: 'dv_total' (departure burn + capture burn), 'c3' (launch energy, what a
    launch vehicle cares about) or 'vinf_arrival' (arrival speed, matters for aerocapture
    or entry). For Earth to Mars use a window of about 6 months around the opportunity and
    times of flight of roughly 100-450 days.
    """
    t0, t1 = to_mjd2000(earliest_departure), to_mjd2000(latest_departure)
    res = window_search(origin, destination, t0, t1, min_time_of_flight_days, max_time_of_flight_days,
                        objective, departure_orbit_altitude_km, capture_orbit_altitude_km, ephemeris)
    out = transfer_dict(res.best)
    out.update({
        "objective": res.objective,
        "search": (f"{res.evaluations} Lambert solves: {res.grid_step_days:.3g} day grid over departure date and "
                   f"time of flight, refined on a {res.refine_step_days:.3g} day grid around the best point."),
        "departure_at_search_edge": res.departure_at_edge,
        "time_of_flight_at_search_edge": res.tof_at_edge,
    })
    if res.departure_at_edge or res.tof_at_edge:
        out["warning"] = ("The best point lies on the edge of the search box, so the true optimum is probably "
                          "outside it. Widen the departure window or time-of-flight range and search again.")
    b = res.best
    publish("find_launch_window", out, lambda: {**scene([origin, destination], [b.depart_mjd2000, b.arrive_mjd2000],
                                                      ephemeris),
                                                "porkchop": porkchop(res)})
    return out


@server.tool(structured_output=True, title="Next launch opportunities")
@user_errors
def next_launch_opportunities(
    origin: Planet,
    destination: Planet,
    after_date: str,
    count: int = 2,
    ephemeris: Ephemeris = "jpl_lp",
) -> dict[str, Any]:
    """When are the next launch windows? Use this first whenever the dates are not given.

    Scans departure dates over about `count` synodic periods after `after_date`, finds each
    window's minimum launch energy (C3) and returns, in date order, the best departure, its
    transfer, and a departure range and time-of-flight range to pass to find_launch_window
    for a closer look (for example to optimise total delta-v instead of C3). Takes about 3 s.
    """
    after = to_mjd2000(after_date)
    scan = launch_opportunities(origin, destination, after, count, ephemeris)
    opps = scan.opportunities
    out = {
        "origin": origin,
        "destination": destination,
        "after_utc": to_iso(after),
        "synodic_period_days": sig4(scan.synodic_days),
        "opportunities": [{
            "best_c3": transfer_dict(o.best),
            "suggested_earliest_departure": to_iso(o.search_start_mjd2000)[:10],
            "suggested_latest_departure": to_iso(o.search_end_mjd2000)[:10],
            "suggested_min_time_of_flight_days": sig4(o.tof_min_days),
            "suggested_max_time_of_flight_days": sig4(o.tof_max_days),
            **({"note": "This window is still open but its best day was before after_date; C3 is rising."}
               if o.minimum_before_start else {}),
        } for o in opps],
        "method": ("Minimum C3 over times of flight from 0.35 to 1.75 Hohmann transfer times, on a departure grid "
                   "of synodic period / 150 (at least 2 days); each minimum refined with find_launch_window."),
    }
    if scan.truncated:
        out["warning"] = (f"The {ephemeris} ephemeris ends in {to_iso(scan.scanned_until_mjd2000)[:4]} or so, so the scan "
                          "or its times of flight were cut short and later windows may be missing"
                          + ("; ephemeris de440s runs to 2150." if ephemeris == "jpl_lp" else "."))
    if len(opps) < count:
        out.setdefault("warning", f"Only {len(opps)} window(s) found in the scanned span.")
    if opps:
        b = opps[0].best
        publish("next_launch_opportunities", out,
                lambda: scene([origin, destination], [b.depart_mjd2000, b.arrive_mjd2000], ephemeris))
    else:
        publish("next_launch_opportunities", out)
    return out


@server.tool(structured_output=True, title="Propagate orbit")
@user_errors
def propagate(
    position_km: list[float],
    velocity_km_s: list[float],
    duration_days: float | None = None,
    central_body: Literal["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"] = "sun",
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Two-body (Keplerian) propagation of a state vector for a duration (negative goes backwards).

    Give the duration as exactly one of duration_days or duration_s (seconds; use it for short
    arcs so no precision is lost converting minutes to days). Inputs and outputs are Cartesian
    in km and km/s, ecliptic J2000 axes for 'sun'. Returns the final state at full precision
    plus initial and final orbital elements.
    """
    if (duration_days is None) == (duration_s is None):
        raise ValueError("give exactly one of duration_days or duration_s")
    seconds = duration_s if duration_s is not None else duration_days * 86400.0
    res = kepler_propagate(position_km, velocity_km_s, seconds, central_body)
    out = {
        "central_body": central_body,
        "duration_days": sig4(seconds / 86400.0),
        "duration_s": sig4(seconds),
        "final_position_km": list(res.r_km),
        "final_velocity_km_s": list(res.v_km_s),
        "specific_energy_km2_s2": sig4(res.specific_energy_km2_s2),
        "initial_elements": elements_dict(res.initial),
        "final_elements": elements_dict(res.final),
        "model": "Keplerian two-body, Lagrange coefficients (pykep.propagate_lagrangian). No perturbations.",
    }
    publish("propagate", out)
    return out


@server.tool(structured_output=True, title="Hohmann transfer")
@user_errors
def hohmann_transfer(
    initial_radius_km: float,
    final_radius_km: float,
    central_body: Literal["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"] = "earth",
) -> dict[str, Any]:
    """Two-burn Hohmann transfer between coplanar circular orbits about one body.

    Radii are measured from the body's centre, not altitudes: add the body's radius
    (Earth 6378 km) to an altitude first. Returns both burns, the total and the transfer time.
    """
    h = hohmann(initial_radius_km, final_radius_km, central_body)
    out = {
        "central_body": central_body,
        "initial_radius_km": sig4(h.r1_km),
        "final_radius_km": sig4(h.r2_km),
        "dv_first_burn_km_s": sig4(h.dv1_km_s),
        "dv_second_burn_km_s": sig4(h.dv2_km_s),
        "dv_total_km_s": sig4(h.dv_total_km_s),
        "time_of_flight_hours": sig4(h.tof_s / 3600.0),
        "transfer_semi_major_axis_km": sig4(h.transfer_a_km),
        "transfer_eccentricity": sig4(h.transfer_e),
        "model": "Two-body, impulsive burns, coplanar circular orbits (vis-viva).",
    }
    publish("hohmann_transfer", out)
    return out


@server.tool(structured_output=True, title="Check delta-v budget")
@user_errors
def check_delta_v_budget(
    maneuvers_km_s: dict[str, float],
    isp_s: float,
    dry_mass_kg: float,
    propellant_mass_kg: float,
    margin_percent: float = 10.0,
) -> dict[str, Any]:
    """Can one stage with this engine and propellant load fly these maneuvers?

    maneuvers_km_s maps a maneuver name to its delta-v, e.g. {"trans-Mars injection": 3.6,
    "Mars orbit insertion": 2.1}; take the numbers from solve_transfer or find_launch_window.
    Uses the Tsiolkovsky rocket equation with g0 = 9.80665 m/s^2 and adds margin_percent to
    the required total.
    """
    b = check_budget(maneuvers_km_s, isp_s, dry_mass_kg, propellant_mass_kg, margin_percent)
    out = {
        "maneuvers_km_s": {k: sig4(v) for k, v in maneuvers_km_s.items()},
        "required_km_s": sig4(b.required_km_s),
        "margin_percent": sig4(margin_percent),
        "required_with_margin_km_s": sig4(b.required_with_margin_km_s),
        "available_km_s": sig4(b.available_km_s),
        "surplus_km_s": sig4(b.surplus_km_s),
        "feasible": b.feasible,
        "propellant_needed_kg": sig4(b.propellant_needed_kg),
        "propellant_available_kg": sig4(b.propellant_available_kg),
        "exhaust_velocity_km_s": sig4(b.exhaust_velocity_km_s),
        "model": "Single stage, impulsive, dv = Isp g0 ln(wet mass / dry mass).",
    }
    publish("check_delta_v_budget", out)
    return out


@server.tool(structured_output=True, title="Propose flyby sequence")
@user_errors
def propose_flyby_sequence(
    origin: Planet,
    destination: Planet,
    earliest_departure: str,
    latest_departure: str,
    flyby_candidates: list[Planet] | None = None,
    max_flybys: int = 1,
    sequences: list[list[Planet]] | None = None,
    arrival: Literal["capture", "flyby", "rendezvous"] = "capture",
    departure_orbit_altitude_km: float = DEFAULT_DEPARTURE_ALT_KM,
    capture_orbit_altitude_km: float = DEFAULT_ARRIVAL_ALT_KM,
) -> dict[str, Any]:
    """Rank gravity-assist sequences (and the direct transfer) by total delta-v.

    Either give explicit `sequences` (each from origin to destination, e.g.
    [["earth", "venus", "mars"]]) or let the tool enumerate every sequence with up to
    `max_flybys` (0-2) intermediate flybys drawn from `flyby_candidates` (default venus,
    earth, mars, jupiter, plus saturn for uranus and saturn and uranus for neptune). Each sequence is optimised over departure date and leg times
    with self-adaptive differential evolution; flybys are powered (pykep fb_dv), so a
    flyby dv near zero means a free gravity assist. Arrival: 'capture' into a circular
    orbit, 'rendezvous' (match the planet's velocity) or 'flyby' (no arrival burn).
    Takes roughly 5 s per sequence.
    """
    t0, t1 = to_mjd2000(earliest_departure), to_mjd2000(latest_departure)
    if sequences is not None:
        if not sequences:
            raise ValueError("sequences is empty; omit it to enumerate candidates")
        seqs = [tuple(s) for s in sequences]
        for s in seqs:
            if s[0] != origin or s[-1] != destination:
                raise ValueError(f"sequence {list(s)} must start at {origin} and end at {destination}")
    else:
        if not 0 <= max_flybys <= 2:
            raise ValueError("max_flybys must be 0, 1 or 2")
        if flyby_candidates is not None and not flyby_candidates:
            raise ValueError("flyby_candidates is empty; omit it to use the defaults, or use max_flybys=0")
        candidates = default_flyby_candidates(destination) if flyby_candidates is None else flyby_candidates
        seqs = candidate_sequences(origin, destination, candidates, max_flybys)
    if origin == destination:
        raise ValueError("origin and destination must differ")

    results = []
    for s in seqs:
        r = optimise_sequence(s, t0, t1, arrival_mode=arrival, departure_alt_km=departure_orbit_altitude_km,
                              arrival_alt_km=capture_orbit_altitude_km)
        results.append(r)
    results.sort(key=lambda r: r.dv_total_km_s)

    ranked = []
    for r in results:
        enc = []
        for i, e in enumerate(r.encounters):
            role = "departure" if i == 0 else ("arrival" if i == len(r.encounters) - 1 else "flyby")
            enc.append({
                "body": e.body,
                "role": role,
                "date_utc": to_iso(e.mjd2000),
                "vinf_in_km_s": sig4(e.vinf_in_km_s),
                "vinf_out_km_s": sig4(e.vinf_out_km_s),
                "dv_km_s": sig4(e.dv_km_s),
            })
        ranked.append({
            "sequence": list(r.bodies),
            "dv_total_km_s": sig4(r.dv_total_km_s),
            "c3_km2_s2": sig4(r.c3_km2_s2),
            "time_of_flight_days": sig4(r.tof_days),
            "encounters": enc,
        })
    out = {
        "arrival": arrival,
        "ranked": ranked,
        "model": MODEL_NOTE + " Flybys: instantaneous rotation of v_inf no closer than pykep's safe radius; "
                              "any mismatch is paid as dv. Optimiser is stochastic; results are best found, not proven optimal.",
    }
    top = results[0]
    publish("propose_flyby_sequence", out,
            lambda: scene(list(top.bodies), [e.mjd2000 for e in top.encounters]))
    return out


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
