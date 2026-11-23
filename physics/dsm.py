"""Multiple gravity assists with one deep-space manoeuvre (DSM) per leg: pykep's MGA-1DSM model.

Each leg leaves a planet on a coast arc, fires one impulsive burn somewhere along it, then
follows a Lambert arc to the next planet. Flybys are unpowered: the planet only rotates
v_inf, at a periapsis no lower than its safe radius. Unlike the powered-flyby model in
physics/flyby.py, the DSMs let a trajectory fix a v_inf mismatch away from the planet, and
allow returns to the same planet (Earth-Earth, Venus-Venus resonant legs).

The model itself is pykep's mga_1dsm. It was checked against ESA's GTOP Cassini2 best solution
(physics/gtop.py, tests/test_dsm.py): 8.385 km/s here against 8.383 km/s published.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pygmo as pg
import pykep as pk

from physics.bodies import MU_SUN, PERIOD_DAYS, check_planet, physical, planet
from physics.flyby import ARRIVAL_MODES, leg_tof_bounds
from physics.transfers import (DEFAULT_ARRIVAL_ALT_KM, DEFAULT_DEPARTURE_ALT_KM, EPHEMERIS_RANGE,
                               burn_from_circular)

# (independent restarts, migration rounds) per effort level; a round is 500 SADE generations per island.
EFFORT = {"quick": (1, 10), "standard": (1, 40), "thorough": (3, 40)}
ISLANDS = max(2, min(8, os.cpu_count() or 2))
ETA_BOUNDS = (0.01, 0.99)
RP_UPPER_RADII = 100.0  # highest flyby periapsis searched, planet radii


class _Ephemeris:
    """pykep udpla: positions from de440s, physical constants from jpl_lp (de440s carries none)."""

    def __init__(self, body: str, ephemeris: str):
        self.body, self.ephemeris = body, ephemeris
        self._planet = planet(body, ephemeris)
        self._mu, self._radius, self._safe = physical(body)

    def __reduce__(self):
        # Rebuilt from scratch in each optimiser worker: constructing pykep's de440s planet is what
        # loads the SPICE kernel, and an unpickled copy would find no kernel in a fresh process.
        return _Ephemeris, (self.body, self.ephemeris)

    def eph(self, ep):
        return self._planet.eph(pk.epoch(float(ep)) if isinstance(ep, (int, float)) else ep)

    def get_mu_central_body(self):
        return MU_SUN

    def get_mu_self(self):
        return self._mu

    def get_radius(self):
        return self._radius

    def get_safe_radius(self):
        return self._safe

    def get_name(self):
        return f"{self.body} ({self.ephemeris})"


def _planets(bodies, ephemeris):
    if ephemeris == "jpl_lp":
        return [planet(b, "jpl_lp") for b in bodies]
    return [pk.planet(_Ephemeris(b, ephemeris)) for b in bodies]


def dsm_leg_tof_bounds(a: str, b: str) -> tuple[float, float]:
    """Search bounds (days) for one leg. A return to the same planet spans 0.4 to 2.5 of its period."""
    if a == b:
        return 0.4 * PERIOD_DAYS[a], 2.5 * PERIOD_DAYS[a]
    return leg_tof_bounds(a, b)


@dataclass(frozen=True)
class DsmEncounter:
    body: str
    role: str  # departure, flyby or arrival
    mjd2000: float
    vinf_km_s: float  # launch v_inf, flyby v_inf (unchanged in size by an unpowered flyby) or arrival v_inf
    dv_km_s: float  # departure burn, 0 for a flyby, arrival burn
    periapsis_altitude_km: float | None = None  # flybys only


@dataclass(frozen=True)
class DeepSpaceManoeuvre:
    leg: int  # 1-based
    mjd2000: float
    dv_km_s: float


@dataclass(frozen=True)
class DsmTrajectory:
    bodies: tuple[str, ...]
    encounters: tuple[DsmEncounter, ...]
    manoeuvres: tuple[DeepSpaceManoeuvre, ...]
    arrival_mode: str
    ephemeris: str
    segments: tuple[tuple[float, tuple, float], ...]  # (start MJD2000, (r m, v m/s), duration days) coasts

    @property
    def dv_total_km_s(self) -> float:
        return sum(e.dv_km_s for e in self.encounters) + sum(m.dv_km_s for m in self.manoeuvres)

    @property
    def dsm_total_km_s(self) -> float:
        return sum(m.dv_km_s for m in self.manoeuvres)

    @property
    def c3_km2_s2(self) -> float:
        return self.encounters[0].vinf_km_s ** 2

    @property
    def tof_days(self) -> float:
        return self.encounters[-1].mjd2000 - self.encounters[0].mjd2000


class DsmProblem:
    """pygmo problem around pykep's mga_1dsm (direct encoding), scoring the whole mission in km/s:
    the burn from a circular parking orbit, every DSM, and the arrival burn for the chosen mode."""

    # Far above any feasible total (a bad Earth-Jupiter leg can cost thousands of km/s), plus the
    # overshoot in days, so the search always prefers in-range points and is steered back inside.
    OUT_OF_RANGE_PENALTY = 1.0e6

    def __init__(self, bodies, dep_start, dep_end, tof_bounds, vinf_bounds_km_s, arrival_mode="capture",
                 departure_alt_km=DEFAULT_DEPARTURE_ALT_KM, arrival_alt_km=DEFAULT_ARRIVAL_ALT_KM,
                 ephemeris="jpl_lp"):
        self.bodies = tuple(bodies)
        self.arrival_mode = arrival_mode
        self.ephemeris = ephemeris
        self.seq = _planets(self.bodies, ephemeris)
        self.consts = [physical(b) for b in self.bodies]
        self.r_dep = self.consts[0][1] + departure_alt_km * 1e3
        self.r_arr = self.consts[-1][1] + arrival_alt_km * 1e3
        self.udp = pk.trajopt.mga_1dsm(
            seq=self.seq, t0=[float(dep_start), float(dep_end)], tof=[list(map(float, b)) for b in tof_bounds],
            vinf=list(map(float, vinf_bounds_km_s)), add_vinf_dep=False, add_vinf_arr=False,
            tof_encoding="direct", multi_objective=False, orbit_insertion=False,
            eta_bounds=list(ETA_BOUNDS), rp_ub=RP_UPPER_RADII)

    def get_bounds(self):
        return self.udp.get_bounds()

    def in_range(self, x) -> bool:
        lo, hi = EPHEMERIS_RANGE[self.ephemeris]
        return bool(x[0] >= lo and x[0] + sum(x[5::4]) <= hi)

    def _arrival_dv(self, vinf: float) -> float:
        if self.arrival_mode == "capture":
            return burn_from_circular(vinf, self.consts[-1][0], self.r_arr)
        if self.arrival_mode == "rendezvous":
            return vinf
        return 0.0

    def _core(self, x):
        dvs, lamberts, tofs, coasts, coast_eps = self.udp._compute_dvs(list(x))
        t_arr = x[0] + sum(tofs)
        v_pla = np.array(self.seq[-1].eph(pk.epoch(float(t_arr)))[1])
        vinf_arr = float(np.linalg.norm(np.array(lamberts[-1].v1[0]) - v_pla))
        dep = burn_from_circular(x[3], self.consts[0][0], self.r_dep)
        return dvs, lamberts, tofs, coasts, coast_eps, vinf_arr, dep

    def fitness(self, x):
        if not self.in_range(x):
            lo, hi = EPHEMERIS_RANGE[self.ephemeris]
            overshoot = max(lo - x[0], x[0] + sum(x[5::4]) - hi, 0.0)
            return [self.OUT_OF_RANGE_PENALTY + overshoot]
        dvs, _, _, _, _, vinf_arr, dep = self._core(x)
        total = (dep + sum(dvs[:-1]) + self._arrival_dv(vinf_arr)) / 1e3
        return [total if np.isfinite(total) else self.OUT_OF_RANGE_PENALTY / 2]

    def evaluate(self, x) -> DsmTrajectory:
        dvs, lamberts, tofs, coasts, coast_eps, vinf_arr, dep = self._core(x)
        epochs = [x[0] + sum(tofs[:i]) for i in range(len(self.bodies))]
        enc = [DsmEncounter(self.bodies[0], "departure", float(epochs[0]), x[3] / 1e3, dep / 1e3)]
        for i in range(1, len(self.bodies) - 1):
            v_pla = np.array(self.seq[i].eph(pk.epoch(float(epochs[i])))[1])
            vinf = float(np.linalg.norm(np.array(lamberts[i - 1].v1[0]) - v_pla))
            rp = x[7 + (i - 1) * 4] * self.consts[i][1]
            enc.append(DsmEncounter(self.bodies[i], "flyby", float(epochs[i]), vinf / 1e3, 0.0,
                                    (rp - self.consts[i][1]) / 1e3))
        enc.append(DsmEncounter(self.bodies[-1], "arrival", float(epochs[-1]), vinf_arr / 1e3,
                                self._arrival_dv(vinf_arr) / 1e3))
        etas = x[4::4]
        dsms = tuple(DeepSpaceManoeuvre(i + 1, float(epochs[i] + etas[i] * tofs[i]), dvs[i] / 1e3)
                     for i in range(len(self.bodies) - 1))
        # Two coasts per leg: planet to DSM, then DSM to the next planet along the Lambert arc.
        segs = []
        for k, (state, t0) in enumerate(zip(coasts, coast_eps)):
            leg = k // 2
            end = epochs[leg] + etas[leg] * tofs[leg] if k % 2 == 0 else epochs[leg + 1]
            segs.append((float(t0), (tuple(map(float, state[0])), tuple(map(float, state[1]))), float(end - t0)))
        return DsmTrajectory(self.bodies, tuple(enc), dsms, self.arrival_mode, self.ephemeris, tuple(segs))


_POOL: ProcessPoolExecutor | None = None


def _pool() -> ProcessPoolExecutor:
    """One process pool per Python process, started on first use. Unlike pygmo's mp_island it can be
    started from any thread, which matters because the MCP server runs tools off the main thread."""
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=ISLANDS)
    return _POOL


def _evolve(pop: pg.population, seed: int) -> pg.population:
    """One island's work for one round: 500 generations of self-adaptive DE."""
    algo = pg.algorithm(pg.sade(gen=500, variant=2, variant_adptv=2, memory=False, ftol=0, xtol=0, seed=seed))
    return algo.evolve(pop)


def optimise_dsm(bodies, dep_start, dep_end, arrival_mode="capture", tof_bounds=None,
                 vinf_bounds_km_s=(0.5, 12.0), departure_alt_km=DEFAULT_DEPARTURE_ALT_KM,
                 arrival_alt_km=DEFAULT_ARRIVAL_ALT_KM, ephemeris="jpl_lp", effort="standard",
                 seed: int = 1) -> DsmTrajectory:
    """Lowest total-dv MGA-1DSM trajectory found for a fixed body sequence.

    Self-adaptive differential evolution on parallel islands, the best champion then polished by
    monotonic basin hopping around compass search. Stochastic: best found, not proven optimal.
    The first call in a process also starts the island worker pool (about 20 s on Windows). Workers
    are spawned processes, so a script calling this must guard its entry point with
    `if __name__ == "__main__":`, or each worker re-runs the script and the call never returns.
    """
    bodies = tuple(check_planet(b) for b in bodies)
    if len(bodies) < 2:
        raise ValueError("a sequence needs at least an origin and a destination")
    if bodies[0] == bodies[-1] and len(bodies) == 2:
        raise ValueError("origin and destination must differ, or add flybys between them")
    if arrival_mode not in ARRIVAL_MODES:
        raise ValueError(f"unknown arrival mode {arrival_mode!r}; expected one of {', '.join(ARRIVAL_MODES)}")
    if effort not in EFFORT:
        raise ValueError(f"unknown effort {effort!r}; expected one of {', '.join(EFFORT)}")
    if dep_end <= dep_start:
        raise ValueError("latest departure must be after earliest departure")
    lo, hi = vinf_bounds_km_s
    if not 0 < lo < hi:
        raise ValueError("launch v_inf bounds must satisfy 0 < min < max")
    if tof_bounds is None:
        tof_bounds = [dsm_leg_tof_bounds(a, b) for a, b in zip(bodies, bodies[1:])]
    elif len(tof_bounds) != len(bodies) - 1:
        raise ValueError(f"need {len(bodies) - 1} leg time-of-flight bounds, got {len(tof_bounds)}")

    udp = DsmProblem(bodies, dep_start, dep_end, tof_bounds, vinf_bounds_km_s, arrival_mode,
                     departure_alt_km, arrival_alt_km, ephemeris)
    prob = pg.problem(udp)
    # Island model: SADE on each island in its own process, champions migrating around a ring, then a
    # basin-hopping polish of the best champion. Earth-Venus-Earth-Jupiter 2030-31 at 40 rounds gave
    # 6.488-7.701 km/s over three seeds (serial multi-start SADE: 8.836), so 'thorough' keeps the best
    # of three independent restarts.
    restarts, rounds = EFFORT[effort]
    best_x, best_f = None, np.inf
    pool = _pool()
    for k in range(restarts):
        s = seed + 1000 * k
        pops = [pg.population(prob, 20, seed=s + i) for i in range(ISLANDS)]
        for r in range(rounds):
            pops = list(pool.map(_evolve, pops, [s * 7919 + r * ISLANDS + i for i in range(ISLANDS)]))
            # Ring migration: each island's champion replaces its neighbour's worst individual.
            champs = [(p.champion_x, p.champion_f) for p in pops]
            for i, p in enumerate(pops):
                cx, cf = champs[i - 1]
                p.set_xf(p.worst_idx(), cx, cf)
        champ = min(pops, key=lambda p: p.champion_f[0])
        pop = pg.population(prob, 0, seed=s)
        pop.push_back(champ.champion_x, champ.champion_f)
        polish = pg.mbh(pg.algorithm(pg.compass_search(max_fevals=3000, start_range=0.02, stop_range=1e-6)),
                        stop=3, perturb=0.01, seed=s)
        pop = pg.algorithm(polish).evolve(pop)
        if pop.champion_f[0] < best_f:
            best_x, best_f = pop.champion_x, pop.champion_f[0]
    if best_x is None:
        raise ValueError(f"the optimiser found no valid {'-'.join(bodies)} trajectory; widen the departure window")
    if best_f >= DsmProblem.OUT_OF_RANGE_PENALTY:
        raise ValueError(f"no {'-'.join(bodies)} trajectory found inside the {ephemeris} ephemeris's date range"
                         + (" (jpl_lp ends in 2050; try ephemeris de440s, which runs to 2150)"
                            if ephemeris == "jpl_lp" else ""))
    if not np.isfinite(best_f) or best_f >= DsmProblem.OUT_OF_RANGE_PENALTY / 2:
        raise ValueError(f"the optimiser found no valid {'-'.join(bodies)} trajectory; widen the departure window")
    return udp.evaluate(best_x)


def coast_points_au(traj: DsmTrajectory, n_per_segment: int = 80) -> list[dict]:
    """3D path for the browser, one entry per leg: Kepler-propagated coasts on either side of the DSM."""
    legs = []
    for leg in range(len(traj.bodies) - 1):
        pts = []
        for start, (r, v), days in traj.segments[2 * leg:2 * leg + 2]:
            for f in np.linspace(0.0, 1.0, n_per_segment):
                rr = r if f == 0.0 else pk.propagate_lagrangian([list(r), list(v)], f * days * pk.DAY2SEC,
                                                                 MU_SUN, False)[0]
                pts.append([float(c) / pk.AU for c in rr])
        legs.append({"from": traj.bodies[leg], "to": traj.bodies[leg + 1],
                     "depart_mjd2000": traj.encounters[leg].mjd2000,
                     "arrive_mjd2000": traj.encounters[leg + 1].mjd2000, "points_au": pts})
    return legs
