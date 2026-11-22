"""Lambert solver must reproduce published textbook answers to within 0.1 %."""

import numpy as np
import pytest

from physics.lambert_check import TEXTBOOK_CASES, pct_error, reference_self_consistency_km, solve_case

MAX_PCT_ERROR = 0.1

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
