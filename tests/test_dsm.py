"""MGA-1DSM: the model against ESA GTOP's Cassini2 record, the optimiser against known optima."""

import numpy as np
import pykep as pk
import pytest

from physics import gtop
from physics.bodies import planet, to_mjd2000
from physics.dsm import DsmProblem, coast_points_au, dsm_leg_tof_bounds, optimise_dsm
from physics.flyby import optimise_sequence


def test_gtop_earth_state_is_sane():
    r, v = gtop.state_km("earth", 0.0)
    assert 0.98 < np.linalg.norm(r) / gtop.KM_AU < 1.02
    assert 29.0 < np.linalg.norm(v) < 30.6
    assert abs(r[2]) < 1.0  # GTOP's Earth has zero inclination


def test_gtop_cassini2_record_reproduces():
    """The published 8.383 km/s vector, run through pykep's mga_1dsm on GTOP's own planet model."""
    prob = gtop.cassini2_problem()
    x = gtop.gtop_to_pykep(gtop.CASSINI2_BEST_X, gtop.CASSINI2_SEQUENCE)
    lb, ub = prob.get_bounds()
    assert all(lo <= xi <= hi for lo, xi, hi in zip(lb, x, ub))
    j = prob.fitness(x)[0] / 1e3
    assert abs(j - gtop.CASSINI2_BEST_KM_S) / gtop.CASSINI2_BEST_KM_S < 5e-4  # 8.385 against 8.383


def test_cassini2_vector_costs_more_on_jpl_ephemeris():
    """Same vector on jpl_lp: the gap is the ephemeris, the reason the GTOP model is needed to verify."""
    seq = []
    for body, safe in zip(gtop.CASSINI2_SEQUENCE, (1.1,) + gtop.CASSINI2_MIN_RP + (1.1,)):
        u = pk.udpla.jpl_lp(body)
        u.safe_radius = safe * u.radius
        seq.append(pk.planet(u))
    prob = pk.trajopt.mga_1dsm(seq=seq, t0=[-1000, 0], tof=[list(b) for b in gtop.CASSINI2_TOF_BOUNDS],
                               vinf=[3.0, 5.0], add_vinf_dep=True, add_vinf_arr=True, tof_encoding="direct",
                               orbit_insertion=False, eta_bounds=[0.01, 0.9], rp_ub=291.0)
    x = gtop.gtop_to_pykep(gtop.CASSINI2_BEST_X, gtop.CASSINI2_SEQUENCE)
    assert prob.fitness(x)[0] / 1e3 > 10.0


@pytest.fixture(scope="module")
def mars_2028():
    t0, t1 = to_mjd2000("2028-06-01"), to_mjd2000("2029-03-01")
    return optimise_dsm(["earth", "mars"], t0, t1, effort="quick"), optimise_sequence(["earth", "mars"], t0, t1)


def test_direct_transfer_needs_no_deep_space_burn(mars_2028):
    """With no flyby to fix, the best DSM is none: the optimum is the Lambert optimum."""
    dsm, lam = mars_2028
    assert dsm.dsm_total_km_s < 0.01
    assert abs(dsm.dv_total_km_s - lam.dv_total_km_s) / lam.dv_total_km_s < 1e-3
    assert abs(dsm.encounters[0].mjd2000 - lam.encounters[0].mjd2000) < 3.0


def test_breakdown_adds_up(mars_2028):
    dsm, _ = mars_2028
    parts = sum(e.dv_km_s for e in dsm.encounters) + sum(m.dv_km_s for m in dsm.manoeuvres)
    assert parts == pytest.approx(dsm.dv_total_km_s)
    assert dsm.c3_km2_s2 == pytest.approx(dsm.encounters[0].vinf_km_s ** 2)
    m = dsm.manoeuvres[0]
    assert dsm.encounters[0].mjd2000 < m.mjd2000 < dsm.encounters[1].mjd2000


def test_drawn_path_ends_at_the_planets(mars_2028):
    dsm, _ = mars_2028
    [leg] = coast_points_au(dsm, n_per_segment=20)
    for point, enc in ((leg["points_au"][0], dsm.encounters[0]), (leg["points_au"][-1], dsm.encounters[1])):
        r, _ = planet(enc.body).eph(pk.epoch(enc.mjd2000))
        assert np.linalg.norm(np.array(point) - np.array(r) / pk.AU) < 1e-6


def test_flyby_sequence_with_resonant_return_runs():
    """Earth-Earth legs are allowed here (they are not in the powered-flyby model)."""
    t0, t1 = to_mjd2000("2030-01-01"), to_mjd2000("2031-01-01")
    r = optimise_dsm(["earth", "earth", "mars"], t0, t1, effort="quick")
    assert [e.role for e in r.encounters] == ["departure", "flyby", "arrival"]
    assert r.encounters[1].periapsis_altitude_km >= 0.0
    assert np.isfinite(r.dv_total_km_s)


def test_de440s_runs_past_2050():
    t0, t1 = to_mjd2000("2060-01-01"), to_mjd2000("2062-06-01")
    r = optimise_dsm(["earth", "mars"], t0, t1, effort="quick", ephemeris="de440s")
    assert r.ephemeris == "de440s" and r.dsm_total_km_s < 0.05


def test_out_of_range_is_penalised_not_a_crash():
    p = DsmProblem(["earth", "mars"], 18000, 18200, [(100, 400)], (0.5, 6.0))
    x = [18200, 0.5, 0.5, 3000.0, 0.5, 399.0]
    assert p.fitness(x)[0] >= DsmProblem.OUT_OF_RANGE_PENALTY


@pytest.mark.parametrize("kwargs, message", [
    ({"bodies": ["earth"]}, "at least an origin"),
    ({"bodies": ["earth", "earth"]}, "must differ"),
    ({"arrival_mode": "orbit"}, "unknown arrival mode"),
    ({"effort": "max"}, "unknown effort"),
    ({"vinf_bounds_km_s": (3.0, 1.0)}, "launch v_inf bounds"),
    ({"tof_bounds": [(100, 300), (100, 300)]}, "leg time-of-flight bounds"),
])
def test_bad_input(kwargs, message):
    args = {"bodies": ["earth", "mars"], "dep_start": 10000.0, "dep_end": 10300.0, **kwargs}
    with pytest.raises(ValueError, match=message):
        optimise_dsm(**args)


def test_resonant_leg_bounds_cover_one_and_two_periods():
    lo, hi = dsm_leg_tof_bounds("venus", "venus")
    assert lo < 224.7 and hi > 2 * 224.7
