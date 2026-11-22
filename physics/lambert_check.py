"""Step 1 of v0: prove the pykep physics stack on this machine.

Part A: solve textbook Lambert problems with published answers and report % error.
Part B: reproduce NASA's published Earth-to-Mars energy minima at their dates.
Part C: search the 2026 and 2028-29 Earth-to-Mars windows ourselves.

Run:  pixi run lambert-check
"""

from __future__ import annotations

import numpy as np
import pykep as pk

from physics.bodies import to_iso, to_mjd2000
from physics.references import LAMBERT_CASES, NASA_SOURCE, NASA_TRANSFERS, LambertCase, NasaTransfer
from physics.transfers import DEFAULT_ARRIVAL_ALT_KM, DEFAULT_DEPARTURE_ALT_KM, Transfer, find_launch_window, solve_transfer

# Kept for existing imports: the textbook cases now live in physics.references.
TEXTBOOK_CASES = LAMBERT_CASES


def solve_case(case: LambertCase) -> tuple[np.ndarray, np.ndarray]:
    """Single-revolution Lambert solution, in km and km/s."""
    lp = pk.lambert_problem(case.r1_km, case.r2_km, case.tof_s, case.mu_km3_s2, case.retrograde, 0)
    return np.array(lp.v0[0]), np.array(lp.v1[0])


def pct_error(computed, reference) -> float:
    """Vector % error: |computed - reference| / |reference| * 100."""
    computed, reference = np.asarray(computed), np.asarray(reference)
    return float(np.linalg.norm(computed - reference) / np.linalg.norm(reference) * 100.0)


def signed_pct(computed: float, reference: float) -> float:
    return (computed - reference) / reference * 100.0


def reference_self_consistency_km(case: LambertCase) -> float:
    """Miss distance (km) when the *published* v1 is propagated from r1 for tof.

    Independent of pykep's Lambert solver: it uses the Lagrangian propagator, so a
    transcription error in a reference value shows up as a large miss.
    """
    r_end, _ = pk.propagate_lagrangian(
        [list(case.r1_km), list(case.v1_ref_km_s)], case.tof_s, case.mu_km3_s2, False
    )
    return float(np.linalg.norm(np.array(r_end) - np.array(case.r2_km)))


def solve_nasa_transfer(ref: NasaTransfer, ephemeris: str) -> Transfer:
    t0, t1 = to_mjd2000(ref.depart), to_mjd2000(ref.arrive)
    return solve_transfer("earth", "mars", t0, t1 - t0, ephemeris=ephemeris)


def fmt(x: float) -> str:
    """Four significant figures."""
    return f"{x:.4g}"


def fmt_vec(v) -> str:
    return "[" + ", ".join(f"{c:.4g}" for c in v) + "]"


def main() -> None:
    print("=" * 78)
    print("PART A: textbook Lambert cases (single revolution)")
    print("=" * 78)
    for case in LAMBERT_CASES:
        v1, v2 = solve_case(case)
        print(f"\n{case.name}\n  source: {case.source}\n  verified: {case.verification}")
        print(f"  v1 computed  {fmt_vec(v1)} km/s")
        print(f"  v1 published {fmt_vec(case.v1_ref_km_s)} km/s   error {fmt(pct_error(v1, case.v1_ref_km_s))} %")
        print(f"  v2 computed  {fmt_vec(v2)} km/s")
        print(f"  v2 published {fmt_vec(case.v2_ref_km_s)} km/s   error {fmt(pct_error(v2, case.v2_ref_km_s))} %")
        print(f"  self-check: published v1 propagated for tof misses r2 by {fmt(reference_self_consistency_km(case))} km")

    print("\n" + "=" * 78)
    print("PART B: NASA handbook energy minima, recomputed at the published dates")
    print(f"  source: {NASA_SOURCE}")
    print("=" * 78)
    for eph in ("jpl_lp", "de440s"):
        print(f"\n  ephemeris {eph}")
        for ref in NASA_TRANSFERS:
            tr = solve_nasa_transfer(ref, eph)
            print(f"  {ref.table} {ref.label:34s} type {tr.trajectory_type:2s} "
                  f"C3 {fmt(tr.c3_km2_s2)} km^2/s^2 (NASA {ref.c3_km2_s2}, {signed_pct(tr.c3_km2_s2, ref.c3_km2_s2):+.2f} %)  "
                  f"v_inf {fmt(tr.vinf_arr_km_s)} km/s (NASA {ref.vinf_arr_km_s}, {signed_pct(tr.vinf_arr_km_s, ref.vinf_arr_km_s):+.2f} %)")

    print("\n" + "=" * 78)
    print("PART C: Earth to Mars window search, jpl_lp ephemerides")
    print(f"  dv_total = burn from {DEFAULT_DEPARTURE_ALT_KM:.0f} km circular LEO + capture into "
          f"{DEFAULT_ARRIVAL_ALT_KM:.0f} km circular Mars orbit")
    print("=" * 78)
    windows = (
        ("2026 window", "2026-09-01", "2027-03-01"),
        ("2026 window, departing from today", "2026-11-22", "2027-03-01"),
        ("2028-29 window", "2028-09-01", "2029-03-01"),
    )
    for label, d0, d1 in windows:
        for objective in ("c3", "dv_total"):
            tr = find_launch_window("earth", "mars", to_mjd2000(d0), to_mjd2000(d1), 100.0, 450.0, objective).best
            print(f"\n{label}, min {objective}")
            print(f"  departure {to_iso(tr.depart_mjd2000)}  arrival {to_iso(tr.arrive_mjd2000)}  "
                  f"TOF {fmt(tr.tof_days)} days  type {tr.trajectory_type}")
            print(f"  C3 {fmt(tr.c3_km2_s2)} km^2/s^2  v_inf arrive {fmt(tr.vinf_arr_km_s)} km/s  "
                  f"dv depart {fmt(tr.dv_dep_km_s)} km/s  dv capture {fmt(tr.dv_arr_km_s)} km/s  "
                  f"dv total {fmt(tr.dv_total_km_s)} km/s")


if __name__ == "__main__":
    main()
