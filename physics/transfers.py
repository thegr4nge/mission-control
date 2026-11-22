"""Ballistic planet-to-planet transfers: patched conics on a single-revolution Lambert arc."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pykep as pk

from physics.bodies import MU_SUN, check_planet, physical, planet

DEFAULT_DEPARTURE_ALT_KM = 200.0  # circular parking orbit at the origin planet
DEFAULT_ARRIVAL_ALT_KM = 300.0  # circular capture orbit at the destination planet
OBJECTIVES = ("dv_total", "c3", "vinf_arrival")
MAX_GRID_POINTS = 40_000


def burn_from_circular(v_inf: float, mu: float, r_park: float) -> float:
    """Impulsive burn (m/s) between a circular orbit of radius r_park and a hyperbola of excess v_inf."""
    return float(np.sqrt(v_inf**2 + 2.0 * mu / r_park) - np.sqrt(mu / r_park))


def transfer_angle_deg(r1, r2) -> float:
    """Heliocentric angle swept by a prograde transfer from r1 to r2, 0-360 deg."""
    cos_angle = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return 360.0 - angle if np.cross(r1, r2)[2] < 0.0 else angle


@dataclass(frozen=True)
class Transfer:
    origin: str
    destination: str
    depart_mjd2000: float
    tof_days: float
    transfer_angle_deg: float
    c3_km2_s2: float
    vinf_dep_km_s: float
    vinf_arr_km_s: float
    dv_dep_km_s: float
    dv_arr_km_s: float
    departure_alt_km: float
    arrival_alt_km: float
    ephemeris: str

    @property
    def arrive_mjd2000(self) -> float:
        return self.depart_mjd2000 + self.tof_days

    @property
    def dv_total_km_s(self) -> float:
        return self.dv_dep_km_s + self.dv_arr_km_s

    @property
    def trajectory_type(self) -> str:
        """NASA/TM-2010-216764 convention: Type I < 180 deg transfer angle, Type II 180-360 deg."""
        return "I" if self.transfer_angle_deg < 180.0 else "II"

    def score(self, objective: str) -> float:
        if objective == "dv_total":
            return self.dv_total_km_s
        if objective == "c3":
            return self.c3_km2_s2
        if objective == "vinf_arrival":
            return self.vinf_arr_km_s
        raise ValueError(f"unknown objective {objective!r}; expected one of {', '.join(OBJECTIVES)}")


def solve_transfer(origin: str, destination: str, depart_mjd2000: float, tof_days: float,
                   departure_alt_km: float = DEFAULT_DEPARTURE_ALT_KM,
                   arrival_alt_km: float = DEFAULT_ARRIVAL_ALT_KM,
                   ephemeris: str = "jpl_lp") -> Transfer:
    """Prograde single-revolution transfer leaving a circular orbit and capturing into one."""
    origin, destination = check_planet(origin), check_planet(destination)
    if origin == destination:
        raise ValueError("origin and destination must differ")
    if not all(math.isfinite(v) for v in (depart_mjd2000, tof_days, departure_alt_km, arrival_alt_km)):
        raise ValueError("dates, time of flight and altitudes must be finite numbers")
    if tof_days <= 0:
        raise ValueError("time of flight must be positive")
    if departure_alt_km < 0 or arrival_alt_km < 0:
        raise ValueError("orbit altitudes must be non-negative")

    r1, v_p1 = planet(origin, ephemeris).eph(pk.epoch(depart_mjd2000))
    r2, v_p2 = planet(destination, ephemeris).eph(pk.epoch(depart_mjd2000 + tof_days))
    lp = pk.lambert_problem(r1, r2, tof_days * pk.DAY2SEC, MU_SUN, False, 0)
    vinf_dep = float(np.linalg.norm(np.array(lp.v0[0]) - np.array(v_p1)))  # m/s
    vinf_arr = float(np.linalg.norm(np.array(lp.v1[0]) - np.array(v_p2)))

    mu1, rad1, _ = physical(origin)
    mu2, rad2, _ = physical(destination)
    return Transfer(
        origin=origin,
        destination=destination,
        depart_mjd2000=depart_mjd2000,
        tof_days=tof_days,
        transfer_angle_deg=transfer_angle_deg(r1, r2),
        c3_km2_s2=(vinf_dep / 1e3) ** 2,
        vinf_dep_km_s=vinf_dep / 1e3,
        vinf_arr_km_s=vinf_arr / 1e3,
        dv_dep_km_s=burn_from_circular(vinf_dep, mu1, rad1 + departure_alt_km * 1e3) / 1e3,
        dv_arr_km_s=burn_from_circular(vinf_arr, mu2, rad2 + arrival_alt_km * 1e3) / 1e3,
        departure_alt_km=departure_alt_km,
        arrival_alt_km=arrival_alt_km,
        ephemeris=ephemeris,
    )


@dataclass(frozen=True)
class WindowSearch:
    best: Transfer
    objective: str
    grid_step_days: float
    refine_step_days: float
    evaluations: int
    departure_at_edge: bool  # best departure within one refine step of the window bounds
    tof_at_edge: bool  # best time of flight within one refine step of its bounds
    # The coarse grid, for a porkchop plot: objective value at grid_values[i][j] for departure
    # grid_departures[i] (MJD2000) and time of flight grid_tofs[j] (days).
    grid_departures: tuple[float, ...] = ()
    grid_tofs: tuple[float, ...] = ()
    grid_values: tuple[tuple[float, ...], ...] = ()


def find_launch_window(origin: str, destination: str, dep_start: float, dep_end: float,
                       tof_min_days: float, tof_max_days: float, objective: str = "dv_total",
                       departure_alt_km: float = DEFAULT_DEPARTURE_ALT_KM,
                       arrival_alt_km: float = DEFAULT_ARRIVAL_ALT_KM,
                       ephemeris: str = "jpl_lp") -> WindowSearch:
    """Grid search over departure date and time of flight, then a finer grid around the best point.

    The coarse step is at least 2 days and grows so the coarse grid holds about
    MAX_GRID_POINTS points (slightly more, as both ends of each axis are included).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {', '.join(OBJECTIVES)}")
    if dep_end <= dep_start:
        raise ValueError("latest departure must be after earliest departure")
    if not 0 < tof_min_days < tof_max_days:
        raise ValueError("need 0 < min time of flight < max time of flight")

    area = (dep_end - dep_start) * (tof_max_days - tof_min_days)
    step = max(2.0, float(np.sqrt(area / MAX_GRID_POINTS)))
    fine = step / 20.0
    count = 0

    def search(deps, tofs, record=None):
        nonlocal count
        best = None
        for d in deps:
            row = []
            for t in tofs:
                tr = solve_transfer(origin, destination, float(d), float(t),
                                    departure_alt_km, arrival_alt_km, ephemeris)
                count += 1
                row.append(tr.score(objective))
                if best is None or tr.score(objective) < best.score(objective):
                    best = tr
            if record is not None:
                record.append(tuple(row))
        return best

    grid_deps = np.arange(dep_start, dep_end + 1e-9, step)
    grid_tofs = np.arange(tof_min_days, tof_max_days + 1e-9, step)
    grid: list[tuple[float, ...]] = []
    coarse = search(grid_deps, grid_tofs, grid)
    best = search(
        np.arange(max(dep_start, coarse.depart_mjd2000 - step), min(dep_end, coarse.depart_mjd2000 + step) + 1e-9, fine),
        np.arange(max(tof_min_days, coarse.tof_days - step), min(tof_max_days, coarse.tof_days + step) + 1e-9, fine),
    )
    edge = 1.5 * fine
    return WindowSearch(
        best=best, objective=objective, grid_step_days=step, refine_step_days=fine, evaluations=count,
        departure_at_edge=min(best.depart_mjd2000 - dep_start, dep_end - best.depart_mjd2000) < edge,
        tof_at_edge=min(best.tof_days - tof_min_days, tof_max_days - best.tof_days) < edge,
        grid_departures=tuple(float(d) for d in grid_deps), grid_tofs=tuple(float(t) for t in grid_tofs),
        grid_values=tuple(grid),
    )


# ---------------------------------------------------------------------------
# Where are the launch windows?
# ---------------------------------------------------------------------------


def synodic_period_days(a: str, b: str) -> float:
    """Time between successive alignments of two planets: 1 / |1/T_a - 1/T_b|."""
    from physics.bodies import PERIOD_DAYS
    ta, tb = PERIOD_DAYS[check_planet(a)], PERIOD_DAYS[check_planet(b)]
    if ta == tb:
        raise ValueError("origin and destination must differ")
    return 1.0 / abs(1.0 / ta - 1.0 / tb)


def hohmann_tof_days(a: str, b: str) -> float:
    """Half the period of the ellipse touching both planets' mean orbits (from their periods, Kepler III)."""
    from physics.bodies import PERIOD_DAYS
    ra = PERIOD_DAYS[check_planet(a)] ** (2.0 / 3.0)
    rb = PERIOD_DAYS[check_planet(b)] ** (2.0 / 3.0)  # radius in units where 1 day^(2/3) ~ length
    return 0.5 * ((ra + rb) / 2.0) ** 1.5


@dataclass(frozen=True)
class Opportunity:
    best: Transfer  # minimum-C3 transfer, refined
    search_start_mjd2000: float  # a departure window worth handing to find_launch_window
    search_end_mjd2000: float
    tof_min_days: float
    tof_max_days: float


def launch_opportunities(origin: str, destination: str, after_mjd2000: float, count: int = 2,
                         ephemeris: str = "jpl_lp") -> tuple[list[Opportunity], float]:
    """The next `count` launch opportunities after a date, by minimum launch energy (C3).

    Scans departure dates over count + 0.3 synodic periods; for each, takes the lowest C3
    over times of flight from 0.35 to 1.75 Hohmann transfer times; finds the separate local
    minima of that curve; refines each with find_launch_window. Returns the opportunities in
    date order and the synodic period. A minimum at the very start of the scan counts only if
    it is as good as the others (today may sit inside a window that is still open).
    """
    origin, destination = check_planet(origin), check_planet(destination)
    if not 1 <= count <= 5:
        raise ValueError("count must be 1 to 5")
    if not math.isfinite(after_mjd2000):
        raise ValueError("date must be finite")
    synodic = synodic_period_days(origin, destination)
    hohmann = hohmann_tof_days(origin, destination)
    tof_lo, tof_hi = 0.35 * hohmann, 1.75 * hohmann
    span = (count + 0.3) * synodic
    dep_step = max(2.0, synodic / 150.0)
    tof_step = max(2.0, (tof_hi - tof_lo) / 60.0)
    deps = np.arange(after_mjd2000, after_mjd2000 + span + 1e-9, dep_step)
    tofs = np.arange(tof_lo, tof_hi + 1e-9, tof_step)
    curve = np.array([min(solve_transfer(origin, destination, float(d), float(t), ephemeris=ephemeris).c3_km2_s2
                          for t in tofs) for d in deps])

    half = max(1, int(round(synodic / 3.0 / dep_step)))
    minima = [i for i in range(len(deps))
              if curve[i] == curve[max(0, i - half):i + half + 1].min()]
    interior = [i for i in minima if 0 < i < len(deps) - 1]
    best_interior = min((curve[i] for i in interior), default=math.inf)
    picked = [i for i in minima if 0 < i < len(deps) - 1 or curve[i] <= best_interior]
    picked = sorted(picked)[:count]

    out = []
    for i in picked:
        lo = max(after_mjd2000, float(deps[i]) - 3 * dep_step)
        hi = float(deps[i]) + 3 * dep_step
        refined = find_launch_window(origin, destination, lo, hi, tof_lo, tof_hi, "c3", ephemeris=ephemeris).best
        out.append(Opportunity(refined, max(after_mjd2000, refined.depart_mjd2000 - 0.1 * synodic),
                               refined.depart_mjd2000 + 0.1 * synodic, tof_lo, tof_hi))
    return out, synodic
