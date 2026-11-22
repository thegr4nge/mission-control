"""Step 1 of v0: prove the pykep physics stack on this machine.

Part A: solve textbook Lambert problems with published answers and report % error.
Part B: reproduce NASA's published Earth-to-Mars energy minima at their dates.
Part C: search the 2026 and 2028-29 Earth-to-Mars windows ourselves.

Run:  pixi run lambert-check
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
    verification: str
    r1_km: tuple[float, float, float]
    r2_km: tuple[float, float, float]
    tof_s: float
    mu_km3_s2: float
    v1_ref_km_s: tuple[float, float, float]
    v2_ref_km_s: tuple[float, float, float]
    retrograde: bool = False  # pykep `cw`: True gives the long way for these planar cases


# Each case is also checked for internal consistency: propagating the published
# v1 from r1 for tof must land on r2 (see reference_self_consistency_km below).
CURTIS_5_2 = LambertCase(
    name="Curtis Example 5.2",
    source="H. D. Curtis, Orbital Mechanics for Engineering Students, 3rd ed. (Elsevier, 2014), Example 5.2",
    verification=(
        "checked 2026-11-22 against an OCR scan of the 2014 Elsevier chapter 5 "
        "(DOI 10.1016/B978-0-08-097747-8.00005-0). The OCR drops leading minus signs; "
        "the i-component signs follow from the example's own f, g and r1 x r2 values"
    ),
    r1_km=(5000.0, 10000.0, 2100.0),
    r2_km=(-14600.0, 2500.0, 7000.0),
    tof_s=3600.0,
    mu_km3_s2=398600.0,
    v1_ref_km_s=(-5.9925, 1.9254, 3.2456),
    v2_ref_km_s=(-3.3125, -4.1966, -0.38529),
)

_VALLADO_SOURCE = (
    "D. A. Vallado, Fundamentals of Astrodynamics and Applications, 4th ed. (2013), Example 7-5"
)
_VALLADO_VERIFICATION = (
    "secondary source: matches the Nyx (nyx-space) Lambert unit tests "
    "test_lambert_vallado_shortway / _lonway digit for digit; Vallado's own lambertu.m "
    "cites 'vallado 2013, 489-493, alg 58, ex 7-5'. Not checked against the printed book"
)

VALLADO_7_5_SHORT = LambertCase(
    name="Vallado Example 7-5 (short way)",
    source=_VALLADO_SOURCE,
    verification=_VALLADO_VERIFICATION,
    r1_km=(15945.34, 0.0, 0.0),
    r2_km=(12214.83899, 10249.46731, 0.0),
    tof_s=76.0 * 60.0,
    mu_km3_s2=398600.4418,
    v1_ref_km_s=(2.058913, 2.915965, 0.0),
    v2_ref_km_s=(-3.451565, 0.910315, 0.0),
)

VALLADO_7_5_LONG = LambertCase(
    name="Vallado Example 7-5 (long way)",
    source=_VALLADO_SOURCE,
    verification=_VALLADO_VERIFICATION,
    r1_km=VALLADO_7_5_SHORT.r1_km,
    r2_km=VALLADO_7_5_SHORT.r2_km,
    tof_s=VALLADO_7_5_SHORT.tof_s,
    mu_km3_s2=VALLADO_7_5_SHORT.mu_km3_s2,
    v1_ref_km_s=(-3.811158, -2.003854, 0.0),
    v2_ref_km_s=(4.207569, 0.914724, 0.0),
    retrograde=True,
)

TEXTBOOK_CASES = (CURTIS_5_2, VALLADO_7_5_SHORT, VALLADO_7_5_LONG)


def solve_case(case: LambertCase) -> tuple[np.ndarray, np.ndarray]:
    """Single-revolution Lambert solution, in km and km/s."""
    lp = pk.lambert_problem(case.r1_km, case.r2_km, case.tof_s, case.mu_km3_s2, case.retrograde, 0)
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
# Ephemerides and Earth-to-Mars transfers
# ---------------------------------------------------------------------------

LEO_ALT_M = 200e3  # departure parking orbit: 200 km circular
LMO_ALT_M = 300e3  # arrival orbit at Mars: 300 km circular


def make_planets(ephemeris: str) -> tuple[pk.planet, pk.planet]:
    """Earth and Mars from 'jpl_lp' (analytic low precision) or 'de440s' (SPICE kernel).

    de440s must use the Sun as observer. Its default observer is the solar system
    barycentre, which is offset from the Sun; with MU_SUN in the Lambert solve that
    inflates C3 by up to 9 % versus the NASA handbook.
    """
    if ephemeris == "jpl_lp":
        return pk.planet(pk.udpla.jpl_lp("earth")), pk.planet(pk.udpla.jpl_lp("mars"))
    if ephemeris == "de440s":
        return (
            pk.planet(pk.udpla.de440s("EARTH BARYCENTER", "ECLIPJ2000", "SUN")),
            pk.planet(pk.udpla.de440s("MARS BARYCENTER", "ECLIPJ2000", "SUN")),
        )
    raise ValueError(f"unknown ephemeris {ephemeris!r}")


EARTH, MARS = make_planets("jpl_lp")
# Physical constants come from jpl_lp: de440s planets carry no mu or radius.
MU_EARTH, R_EARTH = EARTH.get_mu_self(), EARTH.get_radius()
MU_MARS, R_MARS = MARS.get_mu_self(), MARS.get_radius()


def burn_from_circular(v_inf: float, mu: float, r_park: float) -> float:
    """Impulsive burn between a circular orbit of radius r_park and a hyperbola of excess v_inf."""
    return float(np.sqrt(v_inf**2 + 2.0 * mu / r_park) - np.sqrt(mu / r_park))


@dataclass(frozen=True)
class Transfer:
    depart_mjd2000: float
    tof_days: float
    transfer_angle_deg: float
    c3_km2_s2: float
    vinf_dep_km_s: float
    vinf_arr_km_s: float
    dv_dep_km_s: float
    dv_arr_km_s: float

    @property
    def dv_total_km_s(self) -> float:
        return self.dv_dep_km_s + self.dv_arr_km_s

    @property
    def trajectory_type(self) -> str:
        """NASA handbook convention: Type I < 180 deg heliocentric transfer angle, Type II > 180."""
        return "I" if self.transfer_angle_deg < 180.0 else "II"


def earth_mars_transfer(depart_mjd2000: float, tof_days: float, planets=None) -> Transfer:
    earth, mars = planets if planets is not None else (EARTH, MARS)
    r_e, v_e = earth.eph(pk.epoch(depart_mjd2000))
    r_m, v_m = mars.eph(pk.epoch(depart_mjd2000 + tof_days))
    lp = pk.lambert_problem(r_e, r_m, tof_days * pk.DAY2SEC, pk.MU_SUN, False, 0)
    vinf_dep = float(np.linalg.norm(np.array(lp.v0[0]) - np.array(v_e)))  # m/s
    vinf_arr = float(np.linalg.norm(np.array(lp.v1[0]) - np.array(v_m)))

    cos_angle = np.dot(r_e, r_m) / (np.linalg.norm(r_e) * np.linalg.norm(r_m))
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    if np.cross(r_e, r_m)[2] < 0.0:  # prograde transfer sweeping past 180 deg
        angle = 360.0 - angle

    return Transfer(
        depart_mjd2000=depart_mjd2000,
        tof_days=tof_days,
        transfer_angle_deg=angle,
        c3_km2_s2=(vinf_dep / 1e3) ** 2,
        vinf_dep_km_s=vinf_dep / 1e3,
        vinf_arr_km_s=vinf_arr / 1e3,
        dv_dep_km_s=burn_from_circular(vinf_dep, MU_EARTH, R_EARTH + LEO_ALT_M) / 1e3,
        dv_arr_km_s=burn_from_circular(vinf_arr, MU_MARS, R_MARS + LMO_ALT_M) / 1e3,
    )


# ---------------------------------------------------------------------------
# Part B: NASA handbook reference transfers
# ---------------------------------------------------------------------------

NASA_SOURCE = (
    "L. M. Burke, R. D. Falck, M. L. McGuire, Interplanetary Mission Design Handbook: "
    "Earth-to-Mars Mission Opportunities 2026 to 2045, NASA/TM-2010-216764 (2010)"
)


@dataclass(frozen=True)
class NasaTransfer:
    label: str
    table: str
    depart: str  # ISO date, 00:00 assumed (the handbook gives dates only)
    arrive: str
    c3_km2_s2: float
    vinf_arr_km_s: float
    trajectory_type: str


# Rows of Tables 2 (p. 12) and 3 (p. 19), ballistic energy minima. The PDF text layer
# interleaves the row labels, so the row-to-type mapping below was confirmed
# physically: each row's transfer angle puts it on the stated side of 180 deg.
NASA_TRANSFERS = (
    NasaTransfer("2026 Type I min C3 and min v_inf", "Table 2", "2026-11-14", "2027-08-09", 11.11, 2.915, "I"),
    NasaTransfer("2026 Type II min C3", "Table 2", "2026-10-31", "2027-08-19", 9.144, 2.729, "II"),
    NasaTransfer("2026 Type II min v_inf", "Table 2", "2026-11-06", "2027-09-08", 9.646, 2.565, "II"),
    NasaTransfer("2028 Type I min C3", "Table 3", "2028-12-10", "2029-07-20", 9.048, 4.892, "I"),
    NasaTransfer("2028 Type I min v_inf", "Table 3", "2029-01-17", "2029-09-02", 24.12, 3.593, "I"),
    NasaTransfer("2028 Type II min C3", "Table 3", "2028-12-02", "2029-10-16", 8.928, 3.261, "II"),
    NasaTransfer("2028 Type II min v_inf", "Table 3", "2028-11-20", "2029-09-18", 9.315, 2.966, "II"),
)


def solve_nasa_transfer(ref: NasaTransfer, planets) -> Transfer:
    t0 = pk.epoch(ref.depart + "T00:00:00").mjd2000
    t1 = pk.epoch(ref.arrive + "T00:00:00").mjd2000
    return earth_mars_transfer(t0, t1 - t0, planets)


def signed_pct(computed: float, reference: float) -> float:
    return (computed - reference) / reference * 100.0


# ---------------------------------------------------------------------------
# Part C: our own window search
# ---------------------------------------------------------------------------


def best_transfer(dep_start: float, dep_end: float, objective: str = "dv_total",
                  tof_min: float = 100.0, tof_max: float = 450.0, step: float = 2.0) -> Transfer:
    """Minimise `objective` ('dv_total' or 'c3') on a 2 day grid, then refine on a 0.1 day grid."""
    key = (lambda tr: tr.dv_total_km_s) if objective == "dv_total" else (lambda tr: tr.c3_km2_s2)

    def search(deps, tofs):
        return min((earth_mars_transfer(float(d), float(t)) for d in deps for t in tofs), key=key)

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


def date(mjd2000: float) -> str:
    return str(pk.epoch(mjd2000))[:16]


def main() -> None:
    print("=" * 78)
    print("PART A: textbook Lambert cases (single revolution)")
    print("=" * 78)
    for case in TEXTBOOK_CASES:
        v1, v2 = solve_case(case)
        print(f"\n{case.name}\n  source: {case.source}\n  verified: {case.verification}")
        print(f"  v1 computed  {fmt_vec(v1)} km/s")
        print(f"  v1 published {fmt_vec(case.v1_ref_km_s)} km/s   error {fmt(pct_error(v1, np.array(case.v1_ref_km_s)))} %")
        print(f"  v2 computed  {fmt_vec(v2)} km/s")
        print(f"  v2 published {fmt_vec(case.v2_ref_km_s)} km/s   error {fmt(pct_error(v2, np.array(case.v2_ref_km_s)))} %")
        print(f"  self-check: published v1 propagated for tof misses r2 by {fmt(reference_self_consistency_km(case))} km")

    print("\n" + "=" * 78)
    print("PART B: NASA handbook energy minima, recomputed at the published dates")
    print(f"  source: {NASA_SOURCE}")
    print("=" * 78)
    for eph in ("jpl_lp", "de440s"):
        planets = make_planets(eph)
        print(f"\n  ephemeris {eph}")
        for ref in NASA_TRANSFERS:
            tr = solve_nasa_transfer(ref, planets)
            print(f"  {ref.table} {ref.label:34s} type {tr.trajectory_type:2s} "
                  f"C3 {fmt(tr.c3_km2_s2)} km^2/s^2 (NASA {ref.c3_km2_s2}, {signed_pct(tr.c3_km2_s2, ref.c3_km2_s2):+.2f} %)  "
                  f"v_inf {fmt(tr.vinf_arr_km_s)} km/s (NASA {ref.vinf_arr_km_s}, {signed_pct(tr.vinf_arr_km_s, ref.vinf_arr_km_s):+.2f} %)")

    print("\n" + "=" * 78)
    print("PART C: Earth to Mars window search, jpl_lp ephemerides")
    print(f"  dv_total = burn from {LEO_ALT_M/1e3:.0f} km circular LEO + capture into "
          f"{LMO_ALT_M/1e3:.0f} km circular Mars orbit")
    print("=" * 78)
    today = pk.epoch("2026-11-22T00:00:00").mjd2000
    windows = (
        ("2026 window", pk.epoch("2026-09-01T00:00:00").mjd2000, pk.epoch("2027-03-01T00:00:00").mjd2000),
        ("2026 window, departing from today", today, pk.epoch("2027-03-01T00:00:00").mjd2000),
        ("2028-29 window", pk.epoch("2028-09-01T00:00:00").mjd2000, pk.epoch("2029-03-01T00:00:00").mjd2000),
    )
    for label, d0, d1 in windows:
        for objective in ("c3", "dv_total"):
            tr = best_transfer(d0, d1, objective)
            print(f"\n{label}, min {objective}")
            print(f"  departure {date(tr.depart_mjd2000)}  arrival {date(tr.depart_mjd2000 + tr.tof_days)}  "
                  f"TOF {fmt(tr.tof_days)} days  type {tr.trajectory_type}")
            print(f"  C3 {fmt(tr.c3_km2_s2)} km^2/s^2  v_inf arrive {fmt(tr.vinf_arr_km_s)} km/s  "
                  f"dv depart {fmt(tr.dv_dep_km_s)} km/s  dv capture {fmt(tr.dv_arr_km_s)} km/s  "
                  f"dv total {fmt(tr.dv_total_km_s)} km/s")


if __name__ == "__main__":
    main()
