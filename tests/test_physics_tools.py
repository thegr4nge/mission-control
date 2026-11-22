"""Propagation, delta-v budget and flyby physics against references."""

import numpy as np
import pytest

from physics.bodies import G0, to_iso, to_mjd2000
from physics.budget import check_budget
from physics.flyby import MgaProblem, candidate_sequences, optimise_sequence
from physics.lambert_check import pct_error
from physics.propagation import propagate
from physics.references import MARINER_10, PROPAGATION_CASES
from physics.transfers import solve_transfer


@pytest.mark.parametrize("case", PROPAGATION_CASES, ids=lambda c: c.name)
def test_propagation_matches_reference(case):
    res = propagate(case.r0_km, case.v0_km_s, case.duration_s, case.central_body)
    assert pct_error(res.r_km, case.r_ref_km) < case.max_pct_error
    assert pct_error(res.v_km_s, case.v_ref_km_s) < case.max_pct_error


def test_propagation_conserves_energy_and_round_trips():
    r0, v0 = (1131.340, -2282.343, 6672.423), (-5.64305, 4.30333, 2.42879)
    out = propagate(r0, v0, 5000.0, "earth")
    back = propagate(out.r_km, out.v_km_s, -5000.0, "earth")
    assert np.allclose(back.r_km, r0, atol=1e-6) and np.allclose(back.v_km_s, v0, atol=1e-9)
    assert out.initial.a_km == pytest.approx(out.final.a_km, rel=1e-10)


def test_budget_rocket_equation():
    """Wet/dry mass ratio e gives dv = Isp g0 exactly (Tsiolkovsky, g0 = 9.80665 m/s^2)."""
    dry = 1000.0
    b = check_budget({"burn": 1.0}, isp_s=300.0, dry_mass_kg=dry, propellant_mass_kg=dry * (np.e - 1.0))
    assert b.available_km_s == pytest.approx(300.0 * G0 / 1e3, rel=1e-12)
    assert G0 == 9.80665
    # Propellant needed for exactly the available dv is the propellant carried.
    exact = check_budget({"burn": b.available_km_s}, 300.0, dry, dry * (np.e - 1.0))
    assert exact.propellant_needed_kg == pytest.approx(dry * (np.e - 1.0), rel=1e-12)
    margin = check_budget({"a": 2.0, "b": 1.0}, 300.0, dry, 100.0, margin_percent=10.0)
    assert margin.required_with_margin_km_s == pytest.approx(3.3) and not margin.feasible


def test_budget_rejects_bad_input():
    with pytest.raises(ValueError):
        check_budget({}, 300.0, 1000.0, 100.0)
    with pytest.raises(ValueError):
        check_budget({"x": 1.0}, -1.0, 1000.0, 100.0)


def test_transfer_rejects_bad_input():
    with pytest.raises(ValueError):
        solve_transfer("earth", "earth", 9000.0, 200.0)
    with pytest.raises(ValueError):
        solve_transfer("earth", "pluto", 9000.0, 200.0)
    with pytest.raises(ValueError):
        solve_transfer("earth", "mars", 9000.0, -5.0)


def test_mariner_10_venus_flyby_is_free_at_flown_dates():
    """Mariner 10 flew Earth - Venus - Mercury ballistically (NASA NSSDCA 1973-085A).

    With launch on its recorded day and the first Mercury encounter at its recorded
    time, the Venus flyby time that best patches the two Lambert legs must be the
    recorded 1974-02-05 17:01 UT, and the patch must be essentially free (an
    unpowered gravity assist). Measured: 1.1e-5 km/s at the recorded time.
    """
    t_launch_day = to_mjd2000(MARINER_10["launch_date"])
    t_venus = to_mjd2000(MARINER_10["venus_flyby_utc"])
    t_mercury = to_mjd2000(MARINER_10["mercury_1_utc"])
    prob = MgaProblem(["earth", "venus", "mercury"], 0.0, 1.0, [(0.0, 1.0), (0.0, 1.0)], arrival_mode="flyby")

    best = None
    for t0 in np.arange(t_launch_day, t_launch_day + 1.0 + 1e-9, 0.05):
        for tv in np.arange(t_venus - 5.0, t_venus + 5.0, 0.02):
            dv = prob.evaluate([t0, tv - t0, t_mercury - tv]).encounters[1].dv_km_s
            if best is None or dv < best[0]:
                best = (dv, tv)
    dv, tv = best
    assert dv < 0.01, f"Venus flyby needs {dv:.4g} km/s"
    assert abs(tv - t_venus) < 0.1, f"best Venus flyby at {to_iso(tv)}"


def test_flyby_optimiser_finds_free_venus_assist():
    """The optimiser alone, given a 4 month window, finds the Venus-assisted route to Mercury
    with a free flyby, and it beats the direct transfer."""
    t0 = to_mjd2000("1973-10-01")
    via_venus = optimise_sequence(["earth", "venus", "mercury"], t0, t0 + 120.0, arrival_mode="flyby")
    direct = optimise_sequence(["earth", "mercury"], t0, t0 + 120.0, arrival_mode="flyby")
    assert via_venus.encounters[1].dv_km_s < 0.01
    assert via_venus.dv_total_km_s < direct.dv_total_km_s


def test_mga_with_no_flyby_equals_lambert_transfer():
    """A two-body sequence must give the same numbers as solve_transfer."""
    prob = MgaProblem(["earth", "mars"], 0.0, 1.0, [(0.0, 1.0)], arrival_mode="capture")
    t0, tof = to_mjd2000("2026-10-31"), 292.0
    seq = prob.evaluate([t0, tof])
    tr = solve_transfer("earth", "mars", t0, tof)
    assert seq.dv_total_km_s == pytest.approx(tr.dv_total_km_s, rel=1e-12)
    assert seq.c3_km2_s2 == pytest.approx(tr.c3_km2_s2, rel=1e-12)


def test_candidate_sequences():
    seqs = candidate_sequences("earth", "mars", ("venus", "earth"), max_flybys=1)
    assert seqs == [("earth", "mars"), ("earth", "venus", "mars")]
    with pytest.raises(ValueError):
        optimise_sequence(["earth", "venus", "venus", "mars"], 0.0, 10.0)


def test_orbital_elements_match_curtis_4_3():
    from physics.references import CURTIS_4_3 as ref
    el = propagate(ref["r_km"], ref["v_km_s"], 0.0, "earth").initial
    assert el.a_km == pytest.approx(ref["a_km"], rel=1e-3)
    assert el.e == pytest.approx(ref["e"], rel=1e-3)
    for key in ("i_deg", "raan_deg", "argp_deg", "true_anomaly_deg"):
        assert getattr(el, key) == pytest.approx(ref[key], rel=1e-3), key
    assert el.period_days * 24.0 == pytest.approx(ref["period_hours"], rel=1e-3)


def test_window_search_flags_an_optimum_on_the_box_edge():
    """2026 Earth-Mars optimum is 31 Oct; a window ending 15 Oct must say its answer is on the edge."""
    from physics.transfers import find_launch_window
    clipped = find_launch_window("earth", "mars", to_mjd2000("2026-09-01"), to_mjd2000("2026-10-15"), 100, 450, "c3")
    assert clipped.departure_at_edge
    full = find_launch_window("earth", "mars", to_mjd2000("2026-09-01"), to_mjd2000("2027-03-01"), 100, 450, "c3")
    assert not full.departure_at_edge and not full.tof_at_edge


def test_dates():
    assert to_mjd2000("2026-10-31T12:00:00Z") == to_mjd2000("2026-10-31T12:00") == 9800.5
    assert to_iso(9800.5 - 0.1 / 86400.0) == "2026-10-31T12:00"  # rounds, does not truncate
    with pytest.raises(ValueError):
        to_mjd2000("31/10/2026")


def test_non_finite_inputs_rejected():
    with pytest.raises(ValueError):
        solve_transfer("earth", "mars", 9800.0, 300.0, departure_alt_km=float("nan"))
    with pytest.raises(ValueError):
        propagate((7000.0, 0.0, 0.0), (0.0, float("nan"), 0.0), 60.0, "earth")
    with pytest.raises(ValueError):
        check_budget({"x": float("inf")}, 300.0, 1000.0, 100.0)
