"""Physics must reproduce published references.

Textbook Lambert cases: within 0.1 %.
NASA/TM-2010-216764 Earth-to-Mars energy minima: within 0.2 % on both ephemerides.
"""

import numpy as np
import pytest

from physics.bodies import to_mjd2000
from physics.lambert_check import (
    pct_error,
    reference_self_consistency_km,
    signed_pct,
    solve_case,
    solve_nasa_transfer,
)
from physics.references import LAMBERT_CASES, NASA_OPTIMA, NASA_TRANSFERS
from physics.transfers import find_launch_window

MAX_PCT_ERROR = 0.1

# The handbook prints 4 significant figures and whole-day dates. Measured worst case
# is 0.06 % (jpl_lp) and 0.10 % (de440s); a real ephemeris or frame mistake shows up
# at the percent level (e.g. 9 % with de440s centred on the barycentre).
NASA_MAX_PCT_ERROR = 0.2

CASES = pytest.mark.parametrize("case", LAMBERT_CASES, ids=lambda c: c.name)


@CASES
def test_v1_matches_reference(case):
    v1, _ = solve_case(case)
    assert pct_error(v1, case.v1_ref_km_s) < MAX_PCT_ERROR


@CASES
def test_v2_matches_reference(case):
    _, v2 = solve_case(case)
    assert pct_error(v2, case.v2_ref_km_s) < MAX_PCT_ERROR


@CASES
def test_reference_values_are_self_consistent(case):
    """Guards against a mistyped reference: published v1 must fly from r1 to r2.

    Published velocities are rounded to 5-7 significant figures, so allow a miss
    of 0.01 % of |r2|.
    """
    tolerance_km = 1e-4 * np.linalg.norm(case.r2_km)
    assert reference_self_consistency_km(case) < tolerance_km


@pytest.mark.parametrize("ephemeris", ["jpl_lp", "de440s"])
@pytest.mark.parametrize("ref", NASA_TRANSFERS, ids=lambda r: r.label)
def test_nasa_handbook_transfer(ref, ephemeris):
    tr = solve_nasa_transfer(ref, ephemeris)
    assert abs(signed_pct(tr.c3_km2_s2, ref.c3_km2_s2)) < NASA_MAX_PCT_ERROR
    assert abs(signed_pct(tr.vinf_arr_km_s, ref.vinf_arr_km_s)) < NASA_MAX_PCT_ERROR
    assert tr.trajectory_type == ref.trajectory_type


@pytest.mark.parametrize("year, start, end, c3_ref, depart_ref", NASA_OPTIMA, ids=lambda v: str(v))
def test_window_search_finds_nasa_min_c3(year, start, end, c3_ref, depart_ref):
    """The handbook searched on a 2 day grid, so its optimum date can differ by a day."""
    tr = find_launch_window("earth", "mars", to_mjd2000(start), to_mjd2000(end), 100.0, 450.0, "c3").best
    assert abs(signed_pct(tr.c3_km2_s2, c3_ref)) < NASA_MAX_PCT_ERROR
    assert abs(tr.depart_mjd2000 - to_mjd2000(depart_ref)) < 2.0
    assert tr.trajectory_type == "II"
