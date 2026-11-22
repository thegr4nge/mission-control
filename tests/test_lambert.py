"""Physics must reproduce published references.

Textbook Lambert cases: within 0.1 %.
NASA/TM-2010-216764 Earth-to-Mars energy minima: within 0.2 % on both ephemerides.
"""

import numpy as np
import pykep as pk
import pytest

from physics.lambert_check import (
    NASA_TRANSFERS,
    TEXTBOOK_CASES,
    best_transfer,
    make_planets,
    pct_error,
    reference_self_consistency_km,
    signed_pct,
    solve_case,
    solve_nasa_transfer,
)

MAX_PCT_ERROR = 0.1

# The handbook prints 4 significant figures and whole-day dates. Measured worst case
# is 0.06 % (jpl_lp) and 0.10 % (de440s); a real ephemeris or frame mistake shows up
# at the percent level (e.g. 9 % with de440s centred on the barycentre).
NASA_MAX_PCT_ERROR = 0.2

CASES = pytest.mark.parametrize("case", TEXTBOOK_CASES, ids=lambda c: c.name)


@CASES
def test_v1_matches_reference(case):
    v1, _ = solve_case(case)
    assert pct_error(v1, np.array(case.v1_ref_km_s)) < MAX_PCT_ERROR


@CASES
def test_v2_matches_reference(case):
    _, v2 = solve_case(case)
    assert pct_error(v2, np.array(case.v2_ref_km_s)) < MAX_PCT_ERROR


@CASES
def test_reference_values_are_self_consistent(case):
    """Guards against a mistyped reference: published v1 must fly from r1 to r2.

    Published velocities are rounded to 5-7 significant figures, so allow a miss
    of 0.01 % of |r2|.
    """
    tolerance_km = 1e-4 * np.linalg.norm(case.r2_km)
    assert reference_self_consistency_km(case) < tolerance_km


@pytest.fixture(scope="module", params=["jpl_lp", "de440s"])
def planets(request):
    return make_planets(request.param)


@pytest.mark.parametrize("ref", NASA_TRANSFERS, ids=lambda r: r.label)
def test_nasa_handbook_transfer(ref, planets):
    tr = solve_nasa_transfer(ref, planets)
    assert abs(signed_pct(tr.c3_km2_s2, ref.c3_km2_s2)) < NASA_MAX_PCT_ERROR
    assert abs(signed_pct(tr.vinf_arr_km_s, ref.vinf_arr_km_s)) < NASA_MAX_PCT_ERROR
    assert tr.trajectory_type == ref.trajectory_type


# NASA/TM-2010-216764 Table 1: optimal ballistic C3 per opportunity (both Type II).
# The handbook searched on a 2 day grid, so its optimum date can differ by a day.
@pytest.mark.parametrize("year, start, end, c3_ref, depart_ref", [
    (2026, "2026-09-01", "2027-03-01", 9.144, "2026-10-31"),
    (2028, "2028-09-01", "2029-03-01", 8.928, "2028-12-02"),
])
def test_window_search_finds_nasa_min_c3(year, start, end, c3_ref, depart_ref):
    tr = best_transfer(pk.epoch(start + "T00:00:00").mjd2000, pk.epoch(end + "T00:00:00").mjd2000, "c3")
    assert abs(signed_pct(tr.c3_km2_s2, c3_ref)) < NASA_MAX_PCT_ERROR
    assert abs(tr.depart_mjd2000 - pk.epoch(depart_ref + "T00:00:00").mjd2000) < 2.0
    assert tr.trajectory_type == "II"
