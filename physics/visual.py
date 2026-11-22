"""Geometry for the 3D view: trajectories, planet orbits and tracks, in AU, ecliptic J2000.

Everything here is computed by pykep from the same ephemerides and Lambert solutions the
tools use, so the picture is the physics, not an illustration of it: arcs and planet tracks
use the mission's own ephemeris. Only the faint full-orbit outlines always use jpl_lp.
"""

from __future__ import annotations

import numpy as np
import pykep as pk

from physics.bodies import MU_SUN, PERIOD_DAYS, PLANETS, check_planet, planet

AU_M = pk.AU



def _au(r) -> list[float]:
    return [float(x) / AU_M for x in r]


def planet_position_au(name: str, mjd2000: float, ephemeris: str = "jpl_lp") -> list[float]:
    r, _ = planet(check_planet(name), ephemeris).eph(pk.epoch(float(mjd2000)))
    return _au(r)


def lambert_arc(r1_m, r2_m, tof_days: float, n: int = 160) -> tuple[list[list[float]], np.ndarray]:
    """Points along the prograde single-revolution Lambert arc (AU) and its departure velocity (m/s).

    The arc is drawn by Kepler-propagating the Lambert departure state, so its last point
    lands on r2 to solver precision (the tests check this).
    """
    tof_s = tof_days * pk.DAY2SEC
    lp = pk.lambert_problem(list(r1_m), list(r2_m), tof_s, MU_SUN, False, 0)
    v1 = np.array(lp.v0[0])
    pts = []
    for f in np.linspace(0.0, 1.0, n):
        if f == 0.0:
            pts.append(_au(r1_m))
            continue
        r, _ = pk.propagate_lagrangian([list(r1_m), list(v1)], f * tof_s, MU_SUN, False)
        pts.append(_au(r))
    return pts, v1


def transfer_path(bodies: list[str], epochs_mjd2000: list[float], ephemeris: str = "jpl_lp",
                  n_per_leg: int = 160) -> list[dict]:
    """One leg per consecutive pair of encounters: [{from, to, depart, arrive, points}]."""
    if len(bodies) != len(epochs_mjd2000) or len(bodies) < 2:
        raise ValueError("need matching bodies and epochs, at least two of each")
    legs = []
    for i in range(len(bodies) - 1):
        t0, t1 = float(epochs_mjd2000[i]), float(epochs_mjd2000[i + 1])
        if t1 <= t0:
            raise ValueError("encounter epochs must increase")
        r1, _ = planet(bodies[i], ephemeris).eph(pk.epoch(t0))
        r2, _ = planet(bodies[i + 1], ephemeris).eph(pk.epoch(t1))
        pts, _ = lambert_arc(r1, r2, t1 - t0, n_per_leg)
        legs.append({"from": bodies[i], "to": bodies[i + 1], "depart_mjd2000": t0, "arrive_mjd2000": t1,
                     "points_au": pts})
    return legs


# jpl_lp is valid for 1800-2050 only (pykep raises outside it).
JPL_LP_START_MJD2000 = -73048.0  # 1800-01-01
JPL_LP_END_MJD2000 = 18262.0  # 2050-01-01


def planet_orbits(epoch_mjd2000: float = 0.0, n: int = 180, names=PLANETS) -> dict[str, list[list[float]]]:
    """One closed orbit polyline per planet, sampled over a sidereal period centred near epoch.

    Neptune's 165 year period does not fit after 2000 inside jpl_lp's 1800-2050 range, so each
    sampling window is shifted to lie inside it; the orbit drawn is the same ellipse.
    """
    out = {}
    for name in names:
        p = planet(name, "jpl_lp")
        period = PERIOD_DAYS[name]
        start = epoch_mjd2000 - period / 2.0
        start = min(max(start, JPL_LP_START_MJD2000 + 1.0), JPL_LP_END_MJD2000 - 1.0 - period)
        out[name] = [_au(p.eph(pk.epoch(start + period * k / n))[0]) for k in range(n + 1)]
    return out


def planet_tracks(names, t0_mjd2000: float, t1_mjd2000: float, max_samples: int = 400,
                  ephemeris: str = "jpl_lp") -> dict:
    """Planet positions on a regular time grid covering [t0, t1], for animating the time slider."""
    if t1_mjd2000 <= t0_mjd2000:
        raise ValueError("track end must be after its start")
    n = int(min(max_samples, max(2, np.ceil(t1_mjd2000 - t0_mjd2000) + 1)))
    times = np.linspace(t0_mjd2000, t1_mjd2000, n)
    return {"times_mjd2000": [float(t) for t in times],
            "positions_au": {name: [planet_position_au(name, t, ephemeris) for t in times] for name in names}}
