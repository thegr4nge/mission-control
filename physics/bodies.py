"""Solar system bodies, constants and date handling shared by the physics modules."""

from __future__ import annotations

import re
from functools import lru_cache

import pykep as pk

PLANETS = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")
EPHEMERIDES = ("jpl_lp", "de440s")

MU_SUN = pk.MU_SUN  # m^3/s^2
G0 = pk.G0  # standard gravity, m/s^2

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_planet(name: str) -> str:
    key = name.strip().lower()
    if key not in PLANETS:
        raise ValueError(f"unknown planet {name!r}; expected one of {', '.join(PLANETS)}")
    return key


@lru_cache(maxsize=None)
def planet(name: str, ephemeris: str = "jpl_lp") -> pk.planet:
    """A pykep planet. 'jpl_lp' is JPL's analytic low-precision model; 'de440s' is the SPICE kernel.

    de440s is Sun-centred on purpose. Its default observer is the solar system
    barycentre, which sits off the Sun; combined with MU_SUN in a Lambert solve that
    inflates Earth-Mars C3 by up to 9 % versus NASA/TM-2010-216764.

    Each planet is its system barycentre, which for Earth is the Earth-Moon barycentre.
    That is what heliocentric trajectory design wants: the geocentre wobbles about it by
    12-13 m/s, enough to move a C3 by up to 0.8 %.
    """
    key = check_planet(name)
    if ephemeris == "jpl_lp":
        return pk.planet(pk.udpla.jpl_lp(key))
    if ephemeris == "de440s":
        return pk.planet(pk.udpla.de440s(f"{key.upper()} BARYCENTER", "ECLIPJ2000", "SUN"))
    raise ValueError(f"unknown ephemeris {ephemeris!r}; expected one of {', '.join(EPHEMERIDES)}")


@lru_cache(maxsize=None)
def physical(name: str) -> tuple[float, float, float]:
    """(mu m^3/s^2, radius m, safe flyby radius m) from jpl_lp; de440s planets carry no constants."""
    p = planet(name, "jpl_lp")
    return p.get_mu_self(), p.get_radius(), p.get_safe_radius()


def central_mu(body: str) -> float:
    """Gravitational parameter (m^3/s^2) of 'sun' or a planet."""
    if body.strip().lower() == "sun":
        return MU_SUN
    return physical(body)[0]


def to_mjd2000(date: str) -> float:
    """ISO date or datetime (UTC, no leap seconds, as pykep) to MJD2000 days."""
    s = date.strip()
    for utc_suffix in ("Z", "z", "+00:00"):
        if s.endswith(utc_suffix):
            s = s[: -len(utc_suffix)]
    if _DATE_ONLY.match(s):
        s += "T00:00:00"
    try:
        return pk.epoch(s).mjd2000
    except Exception as exc:  # pykep raises a bare RuntimeError on bad strings
        raise ValueError(f"cannot parse date {date!r}; use ISO format like 2026-10-31 or 2026-10-31T12:00:00") from exc


def to_iso(mjd2000: float) -> str:
    """MJD2000 days to an ISO datetime string, rounded to the nearest minute."""
    return str(pk.epoch(round(mjd2000 * 1440.0) / 1440.0))[:16]
