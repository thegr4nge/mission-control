"""Step 1 of v0: prove the pykep physics stack on this machine.

Part A: solve textbook Lambert problems with published answers and report % error.
Part B: find the best Earth-to-Mars transfers in the next launch windows using
pykep's JPL low-precision ephemerides.

Run:  pixi run python physics/lambert_check.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pykep as pk

# ---------------------------------------------------------------------------
# Part A: textbook reference cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LambertCase:
    name: str
    source: str
    r1_km: tuple[float, float, float]
    r2_km: tuple[float, float, float]
    tof_s: float
    mu_km3_s2: float
    v1_ref_km_s: tuple[float, float, float]
    v2_ref_km_s: tuple[float, float, float]
    verified_note: str


# Reference values are transcribed from the textbooks named in `source`.
# They could not be checked against an online copy of the book, so each case is
# also checked for internal consistency: propagating the published v1 from r1
# for tof must land on r2 (see reference_self_consistency_km below).
CURTIS_5_2 = LambertCase(
    name="Curtis Example 5.2",
    source=(
        "H. D. Curtis, Orbital Mechanics for Engineering Students, "
        "Example 5.2 (numbering as in the 2nd/3rd editions)"
    ),
    r1_km=(5000.0, 10000.0, 2100.0),
    r2_km=(-14600.0, 2500.0, 7000.0),
    tof_s=3600.0,
    mu_km3_s2=398600.0,
    v1_ref_km_s=(-5.9925, 1.9254, 3.2456),
    v2_ref_km_s=(-3.3125, -4.1966, -0.38529),
    verified_note="from memory of the book; not checked against a physical copy",
)

VALLADO_7_5 = LambertCase(
    name="Vallado Example 7-5",
    source=(
        "D. A. Vallado, Fundamentals of Astrodynamics and Applications, "
        "4th ed., Example 7-5 (universal-variable Lambert)"
    ),
    r1_km=(15945.34, 0.0, 0.0),
    r2_km=(12214.83899, 10249.46731, 0.0),
    tof_s=76.0 * 60.0,
    mu_km3_s2=398600.4418,
    v1_ref_km_s=(2.058913, 2.915965, 0.0),
    v2_ref_km_s=(-3.451565, 0.910315, 0.0),
    verified_note="from memory of the book; not checked against a physical copy",
)

TEXTBOOK_CASES = (CURTIS_5_2, VALLADO_7_5)


def solve_case(case: LambertCase) -> tuple[np.ndarray, np.ndarray]:
    """Single-revolution prograde Lambert solution, in km and km/s."""
    lp = pk.lambert_problem(case.r1_km, case.r2_km, case.tof_s, case.mu_km3_s2, False, 0)
    return np.array(lp.v0[0]), np.array(lp.v1[0])


def pct_error(computed: np.ndarray, reference: np.ndarray) -> float:
    """Vector % error: |computed - reference| / |reference| * 100."""
    return float(np.linalg.norm(computed - reference) / np.linalg.norm(reference) * 100.0)


def reference_self_consistency_km(case: LambertCase) -> float:
    """Miss distance (km) when the *published* v1 is propagated from r1 for tof.

    Independent of pykep's Lambert solver: it uses the Lagrangian propagator, so a
    transcription error in a reference value shows up as a large miss.
    """
    r_end, _ = pk.propagate_lagrangian(
        [list(case.r1_km), list(case.v1_ref_km_s)], case.tof_s, case.mu_km3_s2, False
    )
    return float(np.linalg.norm(np.array(r_end) - np.array(case.r2_km)))


# ---------------------------------------------------------------------------
# Part B: Earth to Mars
# ---------------------------------------------------------------------------

LEO_ALT_M = 200e3  # departure parking orbit: 200 km circular
LMO_ALT_M = 300e3  # arrival orbit at Mars: 300 km circular

EARTH = pk.planet(pk.udpla.jpl_lp("earth"))
MARS = pk.planet(pk.udpla.jpl_lp("mars"))


def burn_from_circular(v_inf: float, mu: float, r_park: float) -> float:
    """Impulsive burn between a circular orbit of radius r_park and a hyperbola of excess v_inf."""
    return float(np.sqrt(v_inf**2 + 2.0 * mu / r_park) - np.sqrt(mu / r_park))


@dataclass(frozen=True)
class Transfer:
    depart_mjd2000: float
    tof_days: float
    c3_km2_s2: float
    vinf_dep_km_s: float
    vinf_arr_km_s: float
    dv_dep_km_s: float
    dv_arr_km_s: float

    @property
    def dv_total_km_s(self) -> float:
        return self.dv_dep_km_s + self.dv_arr_km_s


def earth_mars_transfer(depart_mjd2000: float, tof_days: float) -> Transfer:
    r_e, v_e = EARTH.eph(pk.epoch(depart_mjd2000))
    r_m, v_m = MARS.eph(pk.epoch(depart_mjd2000 + tof_days))
    lp = pk.lambert_problem(r_e, r_m, tof_days * pk.DAY2SEC, pk.MU_SUN, False, 0)
    vinf_dep = float(np.linalg.norm(np.array(lp.v0[0]) - np.array(v_e)))  # m/s
    vinf_arr = float(np.linalg.norm(np.array(lp.v1[0]) - np.array(v_m)))
    dv_dep = burn_from_circular(vinf_dep, EARTH.get_mu_self(), EARTH.get_radius() + LEO_ALT_M)
    dv_arr = burn_from_circular(vinf_arr, MARS.get_mu_self(), MARS.get_radius() + LMO_ALT_M)
    return Transfer(
        depart_mjd2000=depart_mjd2000,
        tof_days=tof_days,
        c3_km2_s2=(vinf_dep / 1e3) ** 2,
        vinf_dep_km_s=vinf_dep / 1e3,
        vinf_arr_km_s=vinf_arr / 1e3,
        dv_dep_km_s=dv_dep / 1e3,
        dv_arr_km_s=dv_arr / 1e3,
    )


def best_transfer(dep_start: float, dep_end: float, tof_min: float = 100.0,
                  tof_max: float = 450.0, step: float = 2.0) -> Transfer:
    """Min total-dv transfer on a coarse grid, then refined on a 0.1 day grid."""
    def search(deps, tofs):
        best = None
        for d in deps:
            for t in tofs:
                tr = earth_mars_transfer(float(d), float(t))
                if best is None or tr.dv_total_km_s < best.dv_total_km_s:
                    best = tr
        return best

    coarse = search(np.arange(dep_start, dep_end, step), np.arange(tof_min, tof_max, step))
    fine_deps = np.arange(max(dep_start, coarse.depart_mjd2000 - step),
                          min(dep_end, coarse.depart_mjd2000 + step), 0.1)
    fine_tofs = np.arange(coarse.tof_days - step, coarse.tof_days + step, 0.1)
    return search(fine_deps, fine_tofs)


def fmt(x: float) -> str:
    """Four significant figures."""
    return f"{x:.4g}"


def fmt_vec(v) -> str:
    return "[" + ", ".join(f"{c:.4g}" for c in v) + "]"


def main() -> None:
    print("=" * 78)
    print("PART A: textbook Lambert cases (prograde, single revolution)")
    print("=" * 78)
    for case in TEXTBOOK_CASES:
        v1, v2 = solve_case(case)
        print(f"\n{case.name}\n  source: {case.source}\n  reference status: {case.verified_note}")
        print(f"  v1 computed  {fmt_vec(v1)} km/s")
        print(f"  v1 published {fmt_vec(case.v1_ref_km_s)} km/s   error {fmt(pct_error(v1, np.array(case.v1_ref_km_s)))} %")
        print(f"  v2 computed  {fmt_vec(v2)} km/s")
        print(f"  v2 published {fmt_vec(case.v2_ref_km_s)} km/s   error {fmt(pct_error(v2, np.array(case.v2_ref_km_s)))} %")
        print(f"  self-check: published v1 propagated for tof misses r2 by {fmt(reference_self_consistency_km(case))} km")

    print("\n" + "=" * 78)
    print("PART B: Earth to Mars, JPL low-precision ephemerides, min total dv")
    print(f"  dv_total = burn from {LEO_ALT_M/1e3:.0f} km circular LEO + capture into "
          f"{LMO_ALT_M/1e3:.0f} km circular Mars orbit")
    print("=" * 78)
    today = pk.epoch("2026-11-22T00:00:00").mjd2000
    windows = (
        ("2026 window (search from 1 Sep 2026)", pk.epoch("2026-09-01T00:00:00").mjd2000,
         pk.epoch("2027-03-01T00:00:00").mjd2000),
        ("2026 window, departures from today (22 Nov 2026)", today,
         pk.epoch("2027-03-01T00:00:00").mjd2000),
        ("2028-29 window", pk.epoch("2028-09-01T00:00:00").mjd2000,
         pk.epoch("2029-03-01T00:00:00").mjd2000),
    )
    for label, d0, d1 in windows:
        tr = best_transfer(d0, d1)
        print(f"\n{label}")
        print(f"  departure     {pk.epoch(tr.depart_mjd2000)}")
        print(f"  arrival       {pk.epoch(tr.depart_mjd2000 + tr.tof_days)}")
        print(f"  time of flight {fmt(tr.tof_days)} days")
        print(f"  C3            {fmt(tr.c3_km2_s2)} km^2/s^2")
        print(f"  v_inf arrive  {fmt(tr.vinf_arr_km_s)} km/s")
        print(f"  dv depart     {fmt(tr.dv_dep_km_s)} km/s, dv capture {fmt(tr.dv_arr_km_s)} km/s")
        print(f"  dv total      {fmt(tr.dv_total_km_s)} km/s")


if __name__ == "__main__":
    main()
