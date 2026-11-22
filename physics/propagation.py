"""Two-body (Keplerian) propagation and orbital elements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pykep as pk

from physics.bodies import central_mu


@dataclass(frozen=True)
class Elements:
    a_km: float  # negative for hyperbolic orbits
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    true_anomaly_deg: float
    period_days: float | None  # None unless elliptic


def elements(r_km, v_km_s, mu_km3_s2: float) -> Elements:
    a, e, i, raan, argp, f = pk.ic2par([list(r_km), list(v_km_s)], mu_km3_s2)
    period = 2.0 * np.pi * np.sqrt(a**3 / mu_km3_s2) / 86400.0 if 0.0 <= e < 1.0 else None
    deg = np.degrees
    return Elements(a, e, deg(i), deg(raan) % 360.0, deg(argp) % 360.0, deg(f) % 360.0, period)


@dataclass(frozen=True)
class PropagationResult:
    r_km: tuple[float, float, float]
    v_km_s: tuple[float, float, float]
    initial: Elements
    final: Elements
    specific_energy_km2_s2: float


def propagate(r_km, v_km_s, duration_s: float, central_body: str = "sun") -> PropagationResult:
    """Keplerian propagation in km, km/s, s about 'sun' or a planet."""
    if len(r_km) != 3 or len(v_km_s) != 3:
        raise ValueError("position and velocity must each have 3 components")
    if not (np.all(np.isfinite(r_km)) and np.all(np.isfinite(v_km_s)) and np.isfinite(duration_s)):
        raise ValueError("position, velocity and duration must be finite numbers")
    if np.linalg.norm(r_km) == 0.0:
        raise ValueError("position must be non-zero")
    mu = central_mu(central_body) / 1e9  # km^3/s^2
    r1, v1 = pk.propagate_lagrangian([list(r_km), list(v_km_s)], duration_s, mu, False)
    energy = float(np.dot(v_km_s, v_km_s) / 2.0 - mu / np.linalg.norm(r_km))
    return PropagationResult(tuple(r1), tuple(v1), elements(r_km, v_km_s, mu), elements(r1, v1, mu), energy)
