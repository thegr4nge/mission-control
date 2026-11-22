"""Published reference cases the physics is tested against. Each names its source and how it was checked."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Lambert problems
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

_VALLADO_SOURCE = "D. A. Vallado, Fundamentals of Astrodynamics and Applications, 4th ed. (2013), Example 7-5"
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

LAMBERT_CASES = (CURTIS_5_2, VALLADO_7_5_SHORT, VALLADO_7_5_LONG)

# ---------------------------------------------------------------------------
# Kepler propagation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationCase:
    name: str
    source: str
    verification: str
    r0_km: tuple[float, float, float]
    v0_km_s: tuple[float, float, float]
    duration_s: float
    central_body: str
    r_ref_km: tuple[float, float, float]
    v_ref_km_s: tuple[float, float, float]
    max_pct_error: float


PROPAGATION_CASES = (
    PropagationCase(
        name="Curtis Example 3.7",
        source="H. D. Curtis, Orbital Mechanics for Engineering Students, 3rd ed. (Elsevier, 2014), Example 3.7",
        verification=(
            "checked 2026-11-22 against an OCR scan of the 2014 Elsevier chapter 3 "
            "(DOI 10.1016/B978-0-08-097747-8.00003-7); signs recovered from the example's printed f, g, "
            "fdot, gdot. The book carries 5 significant figures through its intermediate steps, so its "
            "answer differs from an exact propagation by about 0.01 %"
        ),
        r0_km=(7000.0, -12124.0, 0.0),
        v0_km_s=(2.6679, 4.6210, 0.0),
        duration_s=3600.0,
        central_body="earth",  # the book uses mu = 398600; pykep's 398600.4418 changes nothing at 0.1 %
        r_ref_km=(-3296.8, 7413.9, 0.0),
        v_ref_km_s=(-8.2977, -0.96309, 0.0),
        max_pct_error=0.1,
    ),
    PropagationCase(
        name="Vallado Example 2-4",
        source="D. A. Vallado, Fundamentals of Astrodynamics and Applications, 4th ed. (2013), Example 2-4",
        verification=(
            "recalled, not checked against a source. pykep reproduces all 8 printed digits of the "
            "recalled answer, which rules out a transcription error but not a misattributed example"
        ),
        r0_km=(1131.340, -2282.343, 6672.423),
        v0_km_s=(-5.64305, 4.30333, 2.42879),
        duration_s=40.0 * 60.0,
        central_body="earth",
        r_ref_km=(-4219.7527, 4363.0292, -3958.7666),
        v_ref_km_s=(3.689866, -1.916735, -6.112511),
        max_pct_error=1e-3,
    ),
)

# Orbital elements from a state vector. Recalled, not checked against a source: pykep
# reproduces every recalled value at 4 significant figures, which rules out a garbled
# memory but not a misattributed example number.
CURTIS_4_3 = {
    "source": "H. D. Curtis, Orbital Mechanics for Engineering Students, Example 4.3 (recalled)",
    "r_km": (-6045.0, -3490.0, 2500.0),
    "v_km_s": (-3.457, 6.618, 2.533),
    "mu_km3_s2": 398600.0,
    "a_km": 8788.0,
    "e": 0.1712,
    "i_deg": 153.2,
    "raan_deg": 255.3,
    "argp_deg": 20.07,
    "true_anomaly_deg": 28.45,
    "period_hours": 2.278,  # the book rounds its intermediate h; exact propagation gives 2.277
}

# Hohmann transfer LEO to GEO. Recalled, not checked against a source: the closed-form
# tool reproduces all 7 recalled digits of every value, which rules out a garbled
# memory but not a misattributed example number. The analytic vis-viva check in the
# tests does not depend on this recollection.
VALLADO_6_1 = {
    "source": "D. A. Vallado, Fundamentals of Astrodynamics and Applications, 4th ed. (2013), Example 6-1 (recalled)",
    "r1_km": 6378.1363 + 191.34411,
    "r2_km": 6378.1363 + 35781.34857,
    "dv1_km_s": 2.457038,
    "dv2_km_s": 1.478187,
    "dv_total_km_s": 3.935224,
    "tof_hours": 5.256713,
}

# ---------------------------------------------------------------------------
# NASA Earth-to-Mars energy minima
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

# Table 1: optimal ballistic C3 per opportunity, both Type II. Searched on a 2 day grid.
NASA_OPTIMA = (
    # (year, search start, search end, C3 km^2/s^2, departure date from Table 2/3)
    (2026, "2026-09-01", "2027-03-01", 9.144, "2026-10-31"),
    (2028, "2028-09-01", "2029-03-01", 8.928, "2028-12-02"),
)

# ---------------------------------------------------------------------------
# Mariner 10: Earth - Venus flyby - Mercury
# ---------------------------------------------------------------------------

MARINER_10_SOURCE = (
    "NASA NSSDCA Master Catalog, Mariner 10 (1973-085A), "
    "https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1973-085A, read 2026-11-22"
)
MARINER_10 = {
    "launch_date": "1973-11-03",  # NSSDCA gives the date only
    "venus_flyby_utc": "1974-02-05T17:01",  # closest range 5768 km
    "mercury_1_utc": "1974-03-29T20:46",  # about 704 km from the surface
}
# The flight was ballistic apart from two small trajectory correction maneuvers
# (about 10 days after launch, and 21 January 1974), magnitudes not given by NSSDCA.
