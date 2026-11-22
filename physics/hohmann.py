"""Hohmann transfer between two coplanar circular orbits (closed form, vis-viva)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics.bodies import central_mu, check_planet, physical

SUN_RADIUS_KM = 695700.0  # IAU 2015 nominal solar radius


@dataclass(frozen=True)
class Hohmann:
    r1_km: float
    r2_km: float
    dv1_km_s: float  # first burn, at the initial orbit
    dv2_km_s: float  # second burn, at the final orbit
    dv_total_km_s: float
    tof_s: float  # half the transfer ellipse period
    transfer_a_km: float
    transfer_e: float


def hohmann(r1_km: float, r2_km: float, central_body: str = "earth") -> Hohmann:
    """Two-impulse transfer from a circular orbit of radius r1 to one of radius r2 (either direction)."""
    if not (math.isfinite(r1_km) and math.isfinite(r2_km)) or r1_km <= 0.0 or r2_km <= 0.0:
        raise ValueError("orbit radii must be positive finite numbers in km (radius from the body's centre)")
    if r1_km == r2_km:
        raise ValueError("initial and final radii are equal; no transfer is needed")
    body = central_body.strip().lower()
    body_radius_km = SUN_RADIUS_KM if body == "sun" else physical(check_planet(body))[1] / 1e3
    if min(r1_km, r2_km) <= body_radius_km:
        raise ValueError(f"a radius of {min(r1_km, r2_km):g} km is inside {body} (radius {body_radius_km:g} km). "
                         "Radii are from the body's centre: add the body's radius to an altitude.")
    mu = central_mu(body) / 1e9  # km^3/s^2
    a = (r1_km + r2_km) / 2.0
    v_c1, v_c2 = math.sqrt(mu / r1_km), math.sqrt(mu / r2_km)
    v_t1 = math.sqrt(mu * (2.0 / r1_km - 1.0 / a))
    v_t2 = math.sqrt(mu * (2.0 / r2_km - 1.0 / a))
    dv1, dv2 = abs(v_t1 - v_c1), abs(v_c2 - v_t2)
    tof = math.pi * math.sqrt(a**3 / mu)
    e = abs(r2_km - r1_km) / (r1_km + r2_km)
    return Hohmann(r1_km, r2_km, dv1, dv2, dv1 + dv2, tof, a, e)
