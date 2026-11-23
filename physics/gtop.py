"""ESA GTOP's planetary model, used only to verify our MGA-1DSM against GTOP's published results.

GTOP (Global Trajectory Optimisation Problems, ESA Advanced Concepts Team) scores its
benchmarks with its own analytic ephemeris and constants. A published GTOP decision vector
reproduces its published objective only under that model, so this module rebuilds it:
mean orbital elements as polynomials in Julian centuries from 1900 (the coefficients in
GTOP's Pl_Eph_An.cpp), solved with Kepler's equation in the ecliptic frame, plus GTOP's
gravitational parameters and planet radii (mga_dsm.cpp). It is a fresh implementation of
the same published model, not a copy of that code.

Mission design tools never use this ephemeris. It is several hundred thousand km off
JPL's for Earth, which is why the same decision vector costs 8.385 km/s here and
12.40 km/s on jpl_lp.
"""

from __future__ import annotations

import math

import numpy as np
import pykep as pk

KM_AU = 149597870.66
MU_SUN_KM3_S2 = 1.32712428e11
MU_KM3_S2 = {"venus": 324860.0, "earth": 398601.19, "mars": 42828.3, "jupiter": 126.7e6,
             "saturn": 0.37939519708830e8}
RADIUS_KM = {"venus": 6052.0, "earth": 6378.0, "mars": 3397.0, "jupiter": 71492.0, "saturn": 60330.0}


def _elements(body: str, T: float) -> tuple[float, float, float, float, float, float]:
    """(a AU, e, i deg, RAAN deg, arg. periapsis deg, mean anomaly deg) at T Julian centuries from 1900."""
    if body == "venus":
        return (0.72333160, 0.006820690 - 0.000047740 * T + 0.0000000910 * T**2,
                3.393630555555555560 + 1.00583333333333333e-3 * T - 9.72222222222222222e-7 * T**2,
                7.57796472222222222e1 + 8.9985e-1 * T + 4.1e-4 * T**2,
                5.43841861111111111e1 + 5.08186111111111111e-1 * T - 1.38638888888888889e-3 * T**2,
                2.12603219444444444e2 + (5.8517803875e4 + 1.28605555555555556e-3 * T) * T)
    if body == "earth":
        return (1.000000230, 0.016751040 - 0.000041800 * T - 0.0000001260 * T**2, 0.0, 0.0,
                1.01220833333333333e2 + 1.7191750 * T + 4.52777777777777778e-4 * T**2
                + 3.33333333333333333e-6 * T**3,
                3.58475844444444444e2 + (3.599904975e4 - 1.50277777777777778e-4 * T
                                         - 3.33333333333333333e-6 * T**2) * T)
    if body == "mars":
        return (1.5236883990, 0.093312900 + 0.0000920640 * T - 0.0000000770 * T**2,
                1.850333333333333330 - 6.75e-4 * T + 1.26111111111111111e-5 * T**2,
                4.87864416666666667e1 + 7.70991666666666667e-1 * T - 1.38888888888888889e-6 * T**2
                - 5.33333333333333333e-6 * T**3,
                2.85431761111111111e2 + 1.069766666666666670 * T + 1.3125e-4 * T**2
                + 4.13888888888888889e-6 * T**3,
                3.19529425e2 + (1.91398585e4 + 1.80805555555555556e-4 * T + 1.19444444444444444e-6 * T**2) * T)
    if body == "jupiter":
        return (5.2025610, 0.048334750 + 0.000164180 * T - 0.00000046760 * T**2 - 0.00000000170 * T**3,
                1.308736111111111110 - 5.69611111111111111e-3 * T + 3.88888888888888889e-6 * T**2,
                9.94433861111111111e1 + 1.010530 * T + 3.52222222222222222e-4 * T**2
                - 8.51111111111111111e-6 * T**3,
                2.73277541666666667e2 + 5.99431666666666667e-1 * T + 7.0405e-4 * T**2
                + 5.07777777777777778e-6 * T**3,
                2.25328327777777778e2 + (3.03469202388888889e3 - 7.21588888888888889e-4 * T
                                         + 1.78444444444444444e-6 * T**2) * T)
    if body == "saturn":
        return (9.5547470, 0.055892320 - 0.00034550 * T - 0.0000007280 * T**2 + 0.000000000740 * T**3,
                2.492519444444444440 - 3.91888888888888889e-3 * T - 1.54888888888888889e-5 * T**2
                + 4.44444444444444444e-8 * T**3,
                1.12790388888888889e2 + 8.73195138888888889e-1 * T - 1.52180555555555556e-4 * T**2
                - 5.30555555555555556e-6 * T**3,
                3.38307772222222222e2 + 1.085220694444444440 * T + 9.78541666666666667e-4 * T**2
                + 9.91666666666666667e-6 * T**3,
                1.75466216666666667e2 + (1.22155146777777778e3 - 5.01819444444444444e-4 * T
                                         - 5.19444444444444444e-6 * T**2) * T)
    raise ValueError(f"GTOP model has no {body!r} here")


def state_km(body: str, mjd2000: float) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric ecliptic position (km) and velocity (km/s) in GTOP's analytic model."""
    a_au, e, i, raan, argp, mean = _elements(body, (mjd2000 + 36525.0) / 36525.0)
    a = a_au * KM_AU
    i, raan, argp = math.radians(i), math.radians(raan), math.radians(argp)
    M = math.fmod(math.radians(mean), 2.0 * math.pi)
    E = M
    for _ in range(50):
        step = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= step
        if abs(step) < 1e-15:
            break
    b = a * math.sqrt(1.0 - e * e)
    n = math.sqrt(MU_SUN_KM3_S2 / a**3)
    denom = 1.0 - e * math.cos(E)
    r_pf = np.array([a * (math.cos(E) - e), b * math.sin(E), 0.0])
    v_pf = np.array([-a * n * math.sin(E) / denom, b * n * math.cos(E) / denom, 0.0])
    cO, sO, cw, sw, ci, si = (math.cos(raan), math.sin(raan), math.cos(argp), math.sin(argp),
                              math.cos(i), math.sin(i))
    R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return R @ r_pf, R @ v_pf


class GtopPlanet:
    """pykep user-defined planet (udpla) wrapping GTOP's model, in SI units as pykep expects."""

    def __init__(self, body: str, safe_radius_factor: float = 1.1):
        _elements(body, 1.0)  # validates the name
        self.body = body
        self.safe_radius_factor = safe_radius_factor

    def eph(self, ep):
        t = ep if isinstance(ep, (int, float)) else ep.mjd2000
        r, v = state_km(self.body, float(t))
        return [list(r * 1e3), list(v * 1e3)]

    def get_mu_central_body(self):
        return MU_SUN_KM3_S2 * 1e9

    def get_mu_self(self):
        return MU_KM3_S2[self.body] * 1e9

    def get_radius(self):
        return RADIUS_KM[self.body] * 1e3

    def get_safe_radius(self):
        return self.safe_radius_factor * RADIUS_KM[self.body] * 1e3

    def get_name(self):
        return f"{self.body} (GTOP model)"


# GTOP Cassini2: Earth-Venus-Venus-Earth-Jupiter-Saturn, one deep-space manoeuvre per leg,
# rendezvous with Saturn; objective = launch v_inf + every DSM + arrival v_inf (km/s).
CASSINI2_SEQUENCE = ("earth", "venus", "venus", "earth", "jupiter", "saturn")
CASSINI2_MIN_RP = (1.05, 1.05, 1.15, 1.7)  # flyby periapsis lower bounds, planet radii
CASSINI2_TOF_BOUNDS = ((100, 400), (100, 500), (30, 300), (400, 1600), (800, 2200))
CASSINI2_BEST_KM_S = 8.383  # Schlueter, Fiala & Gerdts (MIDACO), May 2009
# Their decision vector in GTOP's order: [t0, Vinf, u, v, T1..T5, eta1..eta5, rp1..rp4, gamma1..gamma4]
CASSINI2_BEST_X = (
    -779.046753814506, 3.25911446832345, 0.525976214695235, 0.38086496458657, 167.378952534645,
    424.028254165204, 53.2897409769205, 589.766954923325, 2200, 0.769483451363201, 0.513289529822621,
    0.0274175362264024, 0.263985256705873, 0.599984695281461, 1.34877968657176, 1.05, 1.30730278372017,
    69.8090142993495, -1.5937371121191, -1.95952512232447, -1.55498859283059, -1.5134625299674,
)
CASSINI2_SOURCE = ("ESA ACT GTOP database, Cassini2 (archived: https://web.archive.org/web/20200807220234/"
                   "http://www.esa.int/gsp/ACT/projects/gtop/cassini2/)")


def gtop_to_pykep(x, bodies) -> list[float]:
    """GTOP MGA-DSM decision vector to pykep mga_1dsm's direct encoding.

    Two conventions differ. GTOP orders the vector by quantity ([t0, Vinf, u, v, T..., eta...,
    rp..., gamma...]), pykep by leg ([t0, u, v, Vinf, eta1, T1] + [beta, rp, eta, T] per leg).
    GTOP points the launch v_inf in a frame tied to the departure planet's velocity
    (i along v, k along r x v); pykep points it in the heliocentric ecliptic frame.
    pykep also takes Vinf in m/s.
    """
    legs = len(bodies) - 1
    t0, vinf, u, v = x[0:4]
    tof = x[4:4 + legs]
    eta = x[4 + legs:4 + 2 * legs]
    rp = x[4 + 2 * legs:4 + 2 * legs + legs - 1]
    gamma = x[4 + 3 * legs - 1:4 + 4 * legs - 2]
    r0, v0 = state_km(bodies[0], t0)
    k = np.cross(r0, v0)
    k /= np.linalg.norm(k)
    i = v0 / np.linalg.norm(v0)
    j = np.cross(k, i)
    theta = 2.0 * math.pi * u
    phi = math.acos(2.0 * v - 1.0) - math.pi / 2.0
    d = math.cos(theta) * math.cos(phi) * i + math.sin(theta) * math.cos(phi) * j + math.sin(phi) * k
    u_h = (math.atan2(d[1], d[0]) / (2.0 * math.pi)) % 1.0
    v_h = (1.0 - d[2]) / 2.0  # inverse of phi = acos(2v - 1) - pi/2
    out = [t0, u_h, v_h, vinf * 1e3, eta[0], tof[0]]
    for leg in range(1, legs):
        out += [gamma[leg - 1], rp[leg - 1], eta[leg], tof[leg]]
    return out


def cassini2_problem() -> pk.trajopt.mga_1dsm:
    """GTOP Cassini2 as a pykep mga_1dsm on GTOP's planetary model; fitness is in m/s."""
    seq = [pk.planet(GtopPlanet(b, f)) for b, f in zip(CASSINI2_SEQUENCE, (1.1,) + CASSINI2_MIN_RP + (1.1,))]
    return pk.trajopt.mga_1dsm(seq=seq, t0=[-1000, 0], tof=[list(b) for b in CASSINI2_TOF_BOUNDS],
                               vinf=[3.0, 5.0], add_vinf_dep=True, add_vinf_arr=True, tof_encoding="direct",
                               orbit_insertion=False, eta_bounds=[0.01, 0.9], rp_ub=291.0)
