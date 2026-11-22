"""Multiple gravity assist (MGA) sequences: Lambert legs patched by powered flybys.

Each flyby is modelled as pykep's fb_dv: the planet rotates the incoming relative
velocity for free (no closer than its safe radius), and any mismatch in magnitude or
excess turning is paid as an impulsive dv. A flyby dv near zero is a free gravity assist.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pygmo as pg
import pykep as pk

from physics.bodies import MU_SUN, check_planet, physical, planet
from physics.transfers import (DEFAULT_ARRIVAL_ALT_KM, DEFAULT_DEPARTURE_ALT_KM, EPHEMERIS_RANGE,
                               burn_from_circular)

ARRIVAL_MODES = ("capture", "flyby", "rendezvous")
DEFAULT_FLYBY_CANDIDATES = ("venus", "earth", "mars", "jupiter")

# Mean orbital radii (AU), used only to set search bounds on each leg's time of flight.
_MEAN_RADIUS_AU = {"mercury": 0.387, "venus": 0.723, "earth": 1.0, "mars": 1.524,
                   "jupiter": 5.203, "saturn": 9.537, "uranus": 19.19, "neptune": 30.07}


def leg_tof_bounds(a: str, b: str) -> tuple[float, float]:
    """Search bounds (days) for one leg: 0.25 to 2.5 times the Hohmann transfer time."""
    a_m = (_MEAN_RADIUS_AU[a] + _MEAN_RADIUS_AU[b]) / 2.0 * pk.AU
    hohmann_days = np.pi * np.sqrt(a_m**3 / MU_SUN) / 86400.0
    return 0.25 * hohmann_days, 2.5 * hohmann_days


@dataclass(frozen=True)
class Encounter:
    body: str
    mjd2000: float
    vinf_in_km_s: float | None
    vinf_out_km_s: float | None
    dv_km_s: float  # departure burn, flyby dv, or arrival burn


@dataclass(frozen=True)
class Sequence:
    bodies: tuple[str, ...]
    encounters: tuple[Encounter, ...]
    arrival_mode: str

    @property
    def dv_total_km_s(self) -> float:
        return sum(e.dv_km_s for e in self.encounters)

    @property
    def c3_km2_s2(self) -> float:
        return self.encounters[0].vinf_out_km_s ** 2

    @property
    def tof_days(self) -> float:
        return self.encounters[-1].mjd2000 - self.encounters[0].mjd2000


class MgaProblem:
    """pygmo user-defined problem. Decision vector: [departure MJD2000, tof leg 1, ..., tof leg N] (days)."""

    def __init__(self, bodies, dep_start, dep_end, tof_bounds, arrival_mode="capture",
                 departure_alt_km=DEFAULT_DEPARTURE_ALT_KM, arrival_alt_km=DEFAULT_ARRIVAL_ALT_KM,
                 ephemeris="jpl_lp"):
        self.bodies = tuple(bodies)
        self.planets = [planet(b, ephemeris) for b in self.bodies]
        self.consts = [physical(b) for b in self.bodies]
        self.lb = [dep_start] + [lo for lo, _ in tof_bounds]
        self.ub = [dep_end] + [hi for _, hi in tof_bounds]
        self.arrival_mode = arrival_mode
        self.ephemeris = ephemeris
        self.r_dep = self.consts[0][1] + departure_alt_km * 1e3
        self.r_arr = self.consts[-1][1] + arrival_alt_km * 1e3

    def get_bounds(self):
        return self.lb, self.ub

    def evaluate(self, x) -> Sequence:
        epochs = np.cumsum(x)
        states = [p.eph(pk.epoch(float(t))) for p, t in zip(self.planets, epochs)]
        legs = []
        for i in range(len(self.bodies) - 1):
            lp = pk.lambert_problem(states[i][0], states[i + 1][0], float(x[i + 1]) * pk.DAY2SEC, MU_SUN, False, 0)
            legs.append((np.array(lp.v0[0]), np.array(lp.v1[0])))

        encounters = []
        v_out = legs[0][0] - states[0][1]
        vinf = float(np.linalg.norm(v_out))
        dv = burn_from_circular(vinf, self.consts[0][0], self.r_dep)
        encounters.append(Encounter(self.bodies[0], float(epochs[0]), None, vinf / 1e3, dv / 1e3))
        for i in range(1, len(self.bodies) - 1):
            v_in = legs[i - 1][1] - states[i][1]
            v_out = legs[i][0] - states[i][1]
            mu, _, safe = self.consts[i]
            dv = pk.fb_dv(list(v_in), list(v_out), mu, safe)
            encounters.append(Encounter(self.bodies[i], float(epochs[i]), float(np.linalg.norm(v_in)) / 1e3,
                                        float(np.linalg.norm(v_out)) / 1e3, dv / 1e3))
        v_in = legs[-1][1] - states[-1][1]
        vinf = float(np.linalg.norm(v_in))
        if self.arrival_mode == "capture":
            dv = burn_from_circular(vinf, self.consts[-1][0], self.r_arr)
        elif self.arrival_mode == "rendezvous":
            dv = vinf
        else:
            dv = 0.0
        encounters.append(Encounter(self.bodies[-1], float(epochs[-1]), vinf / 1e3, None, dv / 1e3))
        return Sequence(self.bodies, tuple(encounters), self.arrival_mode)

    # Returned for trajectories the ephemeris cannot evaluate (an encounter after jpl_lp's 2050 or
    # de440s's 2150 limit), so the optimiser steers away from them instead of crashing.
    OUT_OF_RANGE_PENALTY = 1.0e3

    def in_range(self, x) -> bool:
        lo, hi = EPHEMERIS_RANGE[self.ephemeris]
        epochs = np.cumsum(x)
        return bool(epochs[0] >= lo and epochs[-1] <= hi)

    def fitness(self, x):
        if not self.in_range(x):  # checked up front: SPICE fails with OSError, jpl_lp with ValueError
            return [self.OUT_OF_RANGE_PENALTY + float(np.sum(x[1:])) * 1e-3]
        return [self.evaluate(x).dv_total_km_s]


def optimise_sequence(bodies, dep_start, dep_end, tof_bounds=None, arrival_mode="capture",
                      departure_alt_km=DEFAULT_DEPARTURE_ALT_KM, arrival_alt_km=DEFAULT_ARRIVAL_ALT_KM,
                      ephemeris="jpl_lp", seeds=4, pop_size=40, generations=400) -> Sequence:
    """Best total-dv trajectory for a fixed body sequence: self-adaptive DE from several seeds."""
    bodies = tuple(check_planet(b) for b in bodies)
    if len(bodies) < 2:
        raise ValueError("a sequence needs at least an origin and a destination")
    if any(a == b for a, b in zip(bodies, bodies[1:])):
        raise ValueError("consecutive encounters with the same planet need multi-revolution legs, not supported")
    if arrival_mode not in ARRIVAL_MODES:
        raise ValueError(f"unknown arrival mode {arrival_mode!r}; expected one of {', '.join(ARRIVAL_MODES)}")
    if dep_end <= dep_start:
        raise ValueError("latest departure must be after earliest departure")
    if tof_bounds is None:
        tof_bounds = [leg_tof_bounds(a, b) for a, b in zip(bodies, bodies[1:])]

    udp = MgaProblem(bodies, dep_start, dep_end, tof_bounds, arrival_mode, departure_alt_km, arrival_alt_km, ephemeris)
    prob = pg.problem(udp)
    best_x, best_f = None, np.inf
    for seed in range(seeds):
        pop = pg.population(prob, pop_size, seed=seed)
        pop = pg.algorithm(pg.sade(gen=generations, seed=seed)).evolve(pop)
        if pop.champion_f[0] < best_f:
            best_x, best_f = pop.champion_x, pop.champion_f[0]
    if best_f >= MgaProblem.OUT_OF_RANGE_PENALTY:
        raise ValueError(f"no {'-'.join(bodies)} trajectory found inside the {ephemeris} ephemeris's date range"
                         + (" (jpl_lp ends in 2050; try ephemeris de440s, which runs to 2150)"
                            if ephemeris == "jpl_lp" else ""))
    return udp.evaluate(best_x)


def default_flyby_candidates(destination: str) -> tuple[str, ...]:
    """Venus, Earth, Mars and Jupiter for any target; past Saturn, Saturn too (and Uranus for Neptune),
    since the outer planets are mostly reached by assists from the giant planets inward of them."""
    destination = check_planet(destination)
    extra = {"uranus": ("saturn",), "neptune": ("saturn", "uranus")}.get(destination, ())
    return DEFAULT_FLYBY_CANDIDATES + extra


def candidate_sequences(origin: str, destination: str, candidates=DEFAULT_FLYBY_CANDIDATES,
                        max_flybys: int = 1) -> list[tuple[str, ...]]:
    """Direct transfer plus every sequence with up to max_flybys intermediate flybys."""
    origin, destination = check_planet(origin), check_planet(destination)
    candidates = tuple(check_planet(c) for c in candidates)
    out = []
    for n in range(max_flybys + 1):
        for middle in itertools.product(candidates, repeat=n):
            seq = (origin, *middle, destination)
            if all(a != b for a, b in zip(seq, seq[1:])):
                out.append(seq)
    return out
