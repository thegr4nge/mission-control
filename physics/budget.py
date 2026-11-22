"""Delta-v budget against the Tsiolkovsky rocket equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.bodies import G0


@dataclass(frozen=True)
class Budget:
    required_km_s: float
    required_with_margin_km_s: float
    available_km_s: float
    surplus_km_s: float
    feasible: bool
    propellant_needed_kg: float
    propellant_available_kg: float
    exhaust_velocity_km_s: float


def check_budget(maneuvers_km_s: dict[str, float], isp_s: float, dry_mass_kg: float,
                 propellant_mass_kg: float, margin_percent: float = 0.0) -> Budget:
    """Single-stage budget: dv = Isp g0 ln(m_wet / m_dry)."""
    if not maneuvers_km_s:
        raise ValueError("give at least one maneuver")
    numbers = [*maneuvers_km_s.values(), isp_s, dry_mass_kg, propellant_mass_kg, margin_percent]
    if not all(np.isfinite(v) for v in numbers):
        raise ValueError("all inputs must be finite numbers")
    if any(dv < 0 for dv in maneuvers_km_s.values()):
        raise ValueError("maneuver delta-v must be non-negative")
    if isp_s <= 0 or dry_mass_kg <= 0 or propellant_mass_kg < 0 or margin_percent < 0:
        raise ValueError("need Isp > 0, dry mass > 0, propellant mass >= 0, margin >= 0")

    ve = isp_s * G0 / 1e3  # km/s
    required = float(sum(maneuvers_km_s.values()))
    with_margin = required * (1.0 + margin_percent / 100.0)
    available = ve * float(np.log((dry_mass_kg + propellant_mass_kg) / dry_mass_kg))
    propellant_needed = dry_mass_kg * (float(np.exp(with_margin / ve)) - 1.0)
    return Budget(
        required_km_s=required,
        required_with_margin_km_s=with_margin,
        available_km_s=available,
        surplus_km_s=available - with_margin,
        feasible=available >= with_margin,
        propellant_needed_kg=propellant_needed,
        propellant_available_kg=propellant_mass_kg,
        exhaust_velocity_km_s=ve,
    )
