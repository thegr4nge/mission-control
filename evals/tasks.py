"""Ten mission tasks with known answers, used to score models on the physics tools.

Every expected value comes from a published reference (physics/references.py) or a
closed-form calculation independent of the tool code, with one labelled exception (T10's
C3, which has no published value). Tolerances allow for the
reference's own rounding and the model of the tools (patched conics, jpl_lp
ephemerides, a 2 day search grid); they are not loose enough to pass a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physics.references import CURTIS_4_3, MARINER_10, NASA_OPTIMA, NASA_TRANSFERS, PROPAGATION_CASES, VALLADO_6_1


@dataclass(frozen=True)
class Expect:
    """One answer key. kind: 'number' (rel tolerance), 'date' (+/- days), 'text' (case-insensitive equal),
    'bool', 'sequence' (list of names, case-insensitive), 'max' (number must not exceed value)."""
    kind: str
    value: Any
    tol: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    expected: dict[str, Expect]
    required_tools: tuple[str, ...]
    # Concepts a sound explanation should mention: each entry is a tuple of alternative phrasings.
    concepts: tuple[tuple[str, ...], ...] = field(default=())


def _nasa(label: str):
    return next(t for t in NASA_TRANSFERS if t.label == label)


_N26 = _nasa("2026 Type II min C3")
_N28I = _nasa("2028 Type I min C3")
_C37 = next(c for c in PROPAGATION_CASES if c.name == "Curtis Example 3.7")
_NASA = "NASA/TM-2010-216764"

TASKS: tuple[Task, ...] = (
    Task(
        id="T01_window_2026",
        prompt=("Find the Earth to Mars departure in the late-2026 launch opportunity that needs the least launch "
                "energy (C3). Report the departure date and the C3."),
        expected={
            "departure_date": Expect("date", NASA_OPTIMA[0][4], 3.0, f"{_NASA} Table 2"),
            "c3_km2_s2": Expect("number", NASA_OPTIMA[0][3], 0.01, f"{_NASA} Table 1"),
        },
        required_tools=("find_launch_window",),
        concepts=(("c3", "launch energy"), ("type ii", "type 2", "long way", "more than 180", "over 180")),
    ),
    Task(
        id="T02_window_2028",
        prompt=("Find the Earth to Mars departure in the 2028-2029 launch opportunity that needs the least launch "
                "energy (C3). Report the departure date and the C3."),
        expected={
            "departure_date": Expect("date", NASA_OPTIMA[1][4], 3.0, f"{_NASA} Table 3"),
            "c3_km2_s2": Expect("number", NASA_OPTIMA[1][3], 0.01, f"{_NASA} Table 1"),
        },
        required_tools=("find_launch_window",),
        concepts=(("c3", "launch energy"), ("type ii", "type 2", "long way", "more than 180", "over 180")),
    ),
    Task(
        id="T03_transfer_2026",
        prompt=(f"A spacecraft leaves Earth for Mars on {_N26.depart} and arrives on {_N26.arrive}. What are the "
                "departure C3, the arrival hyperbolic excess speed, and the trajectory type (I or II)?"),
        expected={
            "c3_km2_s2": Expect("number", _N26.c3_km2_s2, 0.005, f"{_NASA} Table 2"),
            "vinf_arrival_km_s": Expect("number", _N26.vinf_arr_km_s, 0.005, f"{_NASA} Table 2"),
            "trajectory_type": Expect("text", _N26.trajectory_type, source=f"{_NASA} Table 2"),
        },
        required_tools=("solve_transfer",),
        concepts=(("transfer angle", "180"), ("c3",)),
    ),
    Task(
        id="T04_transfer_2028_type1",
        prompt=(f"A spacecraft leaves Earth for Mars on {_N28I.depart} and arrives on {_N28I.arrive}. What are the "
                "departure C3, the arrival hyperbolic excess speed, and the trajectory type (I or II)?"),
        expected={
            "c3_km2_s2": Expect("number", _N28I.c3_km2_s2, 0.005, f"{_NASA} Table 3"),
            "vinf_arrival_km_s": Expect("number", _N28I.vinf_arr_km_s, 0.005, f"{_NASA} Table 3"),
            "trajectory_type": Expect("text", _N28I.trajectory_type, source=f"{_NASA} Table 3"),
        },
        required_tools=("solve_transfer",),
        concepts=(("transfer angle", "180"), ("c3",)),
    ),
    Task(
        id="T05_hohmann_leo_geo",
        prompt=(f"A satellite is in a circular Earth orbit of radius {VALLADO_6_1['r1_km']:.5f} km and must move to a "
                f"circular orbit of radius {VALLADO_6_1['r2_km']:.5f} km with a Hohmann transfer. Give both burns, "
                "the total delta-v and the transfer time in hours."),
        expected={
            "dv_first_burn_km_s": Expect("number", VALLADO_6_1["dv1_km_s"], 0.001, VALLADO_6_1["source"]),
            "dv_second_burn_km_s": Expect("number", VALLADO_6_1["dv2_km_s"], 0.001, VALLADO_6_1["source"]),
            "dv_total_km_s": Expect("number", VALLADO_6_1["dv_total_km_s"], 0.001, VALLADO_6_1["source"]),
            "time_of_flight_hours": Expect("number", VALLADO_6_1["tof_hours"], 0.001, VALLADO_6_1["source"]),
        },
        required_tools=("hohmann_transfer",),
        concepts=(("hohmann",), ("ellipse", "elliptical"), ("apogee", "apoapsis", "perigee", "periapsis")),
    ),
    Task(
        id="T06_mercury_venus_flyby",
        # The arrival mode matters: with orbit capture at Mercury the best Venus flyby moves to May 1974
        # and is far from free, so a model must read "flies past without stopping" and choose a flyby arrival.
        prompt=("Plan a mission that flies past Mercury without stopping (no orbit insertion), launching from "
                "Earth between 1973-10-01 and 1974-01-29. "
                "Compare going direct with using a Venus gravity assist. Which route needs less delta-v, how much "
                "delta-v does the Venus flyby itself need, and on what date does it happen?"),
        expected={
            "best_sequence": Expect("sequence", ["earth", "venus", "mercury"], source="Mariner 10, NASA NSSDCA 1973-085A"),
            "venus_flyby_dv_km_s": Expect("max", 0.05, source="Mariner 10 flew it unpowered (NSSDCA)"),
            "venus_flyby_date": Expect("date", MARINER_10["venus_flyby_utc"][:10], 5.0, "Mariner 10, NSSDCA"),
        },
        required_tools=("propose_flyby_sequence",),
        concepts=(("gravity assist", "flyby", "fly-by"), ("direct",)),
    ),
    Task(
        id="T07_propagate_curtis_3_7",
        prompt=(f"An Earth satellite has position {list(_C37.r0_km)} km and velocity {list(_C37.v0_km_s)} km/s "
                "(geocentric equatorial frame). Where is it 60 minutes later? Give the x and y position in km."),
        expected={
            "final_x_km": Expect("number", _C37.r_ref_km[0], 0.001, _C37.source),
            "final_y_km": Expect("number", _C37.r_ref_km[1], 0.001, _C37.source),
        },
        required_tools=("propagate",),
        concepts=(("two-body", "kepler", "keplerian"),),
    ),
    Task(
        id="T08_elements_curtis_4_3",
        prompt=(f"An Earth satellite has position {list(CURTIS_4_3['r_km'])} km and velocity "
                f"{list(CURTIS_4_3['v_km_s'])} km/s (geocentric equatorial frame). What are its semi-major axis, "
                "eccentricity, inclination and right ascension of the ascending node?"),
        expected={
            "semi_major_axis_km": Expect("number", CURTIS_4_3["a_km"], 0.002, CURTIS_4_3["source"]),
            "eccentricity": Expect("number", CURTIS_4_3["e"], 0.002, CURTIS_4_3["source"]),
            "inclination_deg": Expect("number", CURTIS_4_3["i_deg"], 0.002, CURTIS_4_3["source"]),
            "raan_deg": Expect("number", CURTIS_4_3["raan_deg"], 0.002, CURTIS_4_3["source"]),
        },
        required_tools=("propagate",),
        concepts=(("retrograde", "greater than 90", "over 90", "more than 90"),),
    ),
    Task(
        id="T09_budget_mars_2026",
        prompt=(f"A probe with dry mass 1500 kg and 4000 kg of propellant has one engine with Isp 320 s. It must leave "
                f"a 200 km circular Earth orbit on {_N26.depart} and capture into a 300 km circular Mars orbit on "
                f"{_N26.arrive}. With a 10% delta-v margin, can it do the mission? Report whether it is feasible, the "
                "delta-v it needs without margin, and the delta-v it has available."),
        expected={
            # Departure and capture burns from the NASA Table 2 C3 and v_inf via vis-viva (jpl_lp Earth and Mars
            # mu and radii); 5.762 km/s is also checked by hand in tests/test_mcp_server.py.
            "feasible": Expect("bool", False, source="rocket equation"),
            "required_km_s": Expect("number", 5.762, 0.005, f"{_NASA} Table 2 C3 and v_inf, vis-viva"),
            # 320 s * 9.80665 m/s^2 * ln(5500/1500), exact.
            "available_km_s": Expect("number", 4.077, 0.002, "Tsiolkovsky, g0 = 9.80665 m/s^2"),
        },
        required_tools=("solve_transfer", "check_delta_v_budget"),
        concepts=(("rocket equation", "tsiolkovsky", "mass ratio"), ("margin",)),
    ),
    Task(
        id="T10_window_constraint_binds",
        prompt=("A Mars mission must launch between 2026-09-01 and 2026-10-15 and cannot change those dates. Find the "
                "departure in that range with the least launch energy (C3). Report the departure date, its C3, and "
                "whether the 2026-10-15 deadline is what limits the answer."),
        expected={
            # The unconstrained 2026 optimum is 2026-10-31 (NASA Table 2), after the deadline, and C3 falls
            # monotonically towards it, so the best allowed date is the deadline itself.
            "departure_date": Expect("date", "2026-10-15", 1.0, f"{_NASA} Table 2 optimum lies after the deadline"),
            "deadline_binds": Expect("bool", True, source=f"{_NASA} Table 2"),
            # No published value: this one is our own tool's answer, cross-checked between ephemerides
            # (jpl_lp 10.87, de440s 10.87 km^2/s^2, 2026-10-15, about 310 day flight).
            "c3_km2_s2": Expect("number", 10.87, 0.01, "find_launch_window, jpl_lp and de440s agree; not published"),
        },
        required_tools=("find_launch_window",),
        concepts=(("edge", "boundary", "deadline", "constraint"), ("2026-10-31", "31 october", "october 31", "later")),
    ),
)

ANSWER_KEYS = {t.id: list(t.expected) for t in TASKS}
