"""Step 4: the geometry the browser draws, the bridge, and the push from the MCP server."""

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pykep as pk
import pytest
from websockets.sync.client import connect

from physics.bodies import to_mjd2000
from physics.transfers import find_launch_window
from physics.visual import lambert_arc, planet_orbits, planet_position_au, planet_tracks, transfer_path


# ------------------------------------------------------------------ geometry

def test_lambert_arc_starts_and_ends_on_the_planets():
    """NASA 2026 Type II minimum C3: the drawn arc leaves Earth and lands on Mars."""
    t0, t1 = to_mjd2000("2026-10-31"), to_mjd2000("2027-08-19")
    [leg] = transfer_path(["earth", "mars"], [t0, t1])
    pts = np.array(leg["points_au"])
    assert np.allclose(pts[0], planet_position_au("earth", t0), atol=1e-12)
    assert np.allclose(pts[-1], planet_position_au("mars", t1), atol=1e-6)  # propagated, so solver precision
    r = np.linalg.norm(pts, axis=1)
    assert 0.98 < r.min() and r.max() < 1.7  # between Earth's and a bit beyond Mars's orbit


def test_lambert_arc_sweeps_more_than_180_degrees_for_type_ii():
    t0, t1 = to_mjd2000("2026-10-31"), to_mjd2000("2027-08-19")
    pts = np.array(transfer_path(["earth", "mars"], [t0, t1])[0]["points_au"])
    angles = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    assert 180 < np.degrees(angles[-1] - angles[0]) < 360


def test_multi_leg_path_is_continuous():
    t = [to_mjd2000("1973-11-05T13:07"), to_mjd2000("1974-02-06T12:37"), to_mjd2000("1974-04-05T18:28")]
    legs = transfer_path(["earth", "venus", "mercury"], t)
    assert [(l["from"], l["to"]) for l in legs] == [("earth", "venus"), ("venus", "mercury")]
    assert np.allclose(legs[0]["points_au"][-1], legs[1]["points_au"][0], atol=1e-6)
    with pytest.raises(ValueError):
        transfer_path(["earth", "mars"], [100.0, 50.0])


def test_planet_orbits_close_and_have_the_right_size():
    orbits = planet_orbits()
    assert set(orbits) == {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}
    for name, (lo, hi) in {"earth": (0.98, 1.02), "mars": (1.38, 1.67), "neptune": (29.7, 30.4)}.items():
        pts = np.array(orbits[name])
        r = np.linalg.norm(pts, axis=1)
        assert lo < r.min() and r.max() < hi, name
        assert np.linalg.norm(pts[0] - pts[-1]) < 0.01 * r.mean(), f"{name} orbit does not close"


def test_planet_tracks_grid():
    tr = planet_tracks(["earth"], 0.0, 365.25, max_samples=400)
    assert len(tr["times_mjd2000"]) == 367 and len(tr["positions_au"]["earth"]) == 367
    assert np.allclose(tr["positions_au"]["earth"][0], tr["positions_au"]["earth"][-1], atol=0.01)  # a year


def test_porkchop_grid_matches_the_search():
    w = find_launch_window("earth", "mars", to_mjd2000("2026-09-01"), to_mjd2000("2027-03-01"), 100, 450, "c3")
    vals = np.array(w.grid_values)
    assert vals.shape == (len(w.grid_departures), len(w.grid_tofs))
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    assert vals.min() >= w.best.c3_km2_s2 - 1e-12  # refinement only improves on the grid
    assert abs(w.grid_departures[i] - w.best.depart_mjd2000) <= w.grid_step_days
    assert abs(vals.min() - 9.144) / 9.144 < 0.01  # NASA/TM-2010-216764 Table 1


# ------------------------------------------------------------------ bridge

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def bridge():
    from bridge.server import run
    port = free_port()
    loop = asyncio.new_event_loop()
    ready = asyncio.Event()
    task = None

    def serve():
        nonlocal task
        asyncio.set_event_loop(loop)
        task = loop.create_task(run(port=port, ready=ready))
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    for _ in range(100):
        if ready.is_set():
            break
        time.sleep(0.05)
    assert ready.is_set(), "bridge did not start"
    yield port
    loop.call_soon_threadsafe(task.cancel)
    th.join(timeout=5)


@contextlib.contextmanager
def browser(port):
    with connect(f"ws://127.0.0.1:{port}/ws", max_size=2**24) as ws:
        ws.send(json.dumps({"type": "hello", "role": "browser"}))
        hist = json.loads(ws.recv(timeout=5))
        assert hist["type"] == "history"
        yield ws, hist["events"]


def test_bridge_serves_the_app_and_nothing_else(bridge):
    with urllib.request.urlopen(f"http://127.0.0.1:{bridge}/") as r:
        assert r.status == 200 and "text/html" in r.headers["Content-Type"] and b"Mission Control" in r.read()
    with urllib.request.urlopen(f"http://127.0.0.1:{bridge}/app.js") as r:
        assert "javascript" in r.headers["Content-Type"]
    for bad in ("/../pixi.toml", "/%2e%2e/pixi.toml", "/nope.js"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{bridge}{bad}")
        assert e.value.code == 404, bad


def test_bridge_relays_and_replays(bridge):
    with connect(f"ws://127.0.0.1:{bridge}/ws") as pub, browser(bridge) as (b1, _):
        pub.send(json.dumps({"type": "clear"}))
        assert json.loads(b1.recv(timeout=5))["type"] == "cleared"
        pub.send(json.dumps({"type": "publish", "event": {"type": "mission", "kind": "test", "n": 1}}))
        assert json.loads(b1.recv(timeout=5))["n"] == 1
        with browser(bridge) as (_, hist2):  # a late browser gets the history
            assert [e.get("n") for e in hist2 if e.get("kind") == "test"] == [1]


def test_bridge_answers_geometry_requests(bridge):
    with browser(bridge) as (ws, _):
        ws.send(json.dumps({"type": "request", "id": 7, "what": "planets"}))
        msg = json.loads(ws.recv(timeout=30))
        assert msg["type"] == "response" and msg["id"] == 7 and len(msg["data"]["earth"]) > 100
        ws.send(json.dumps({"type": "request", "id": 8, "what": "tracks", "t0_mjd2000": 10, "t1_mjd2000": 5}))
        msg = json.loads(ws.recv(timeout=10))
        assert msg["id"] == 8 and "error" in msg


def test_bridge_refuses_other_websites(bridge):
    """A page from another origin cannot open the websocket; our own page and non-browser clients can."""
    from websockets.exceptions import InvalidStatus
    with pytest.raises(InvalidStatus):
        with connect(f"ws://127.0.0.1:{bridge}/ws", origin="https://evil.example"):
            pass
    with connect(f"ws://127.0.0.1:{bridge}/ws", origin=f"http://localhost:{bridge}") as ws:
        ws.send(json.dumps({"type": "hello", "role": "browser"}))
        assert json.loads(ws.recv(timeout=5))["type"] == "history"


def test_bridge_refuses_foreign_host_header(bridge):
    """DNS rebinding: a hostile name resolving to 127.0.0.1 is refused even for static files."""
    req = urllib.request.Request(f"http://127.0.0.1:{bridge}/", headers={"Host": f"evil.example:{bridge}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 421


def test_bridge_survives_malformed_messages(bridge):
    with browser(bridge) as (ws, _):
        for bad in ("[1, 2]", "42", "not json"):
            ws.send(bad)
            assert json.loads(ws.recv(timeout=5))["type"] == "error"
        ws.send(json.dumps({"type": "request", "id": 1, "what": "nonsense"}))
        assert "error" in json.loads(ws.recv(timeout=5))  # connection still alive


def test_replay_keeps_geometry_only_for_recent_missions():
    from bridge.server import FULL_GEOMETRY, Bridge
    b = Bridge()
    for i in range(FULL_GEOMETRY + 3):
        b.history.append({"type": "mission", "kind": "k", "n": i, "legs": [1], "tracks": {}, "result": {"i": i}})
    b.history.append({"type": "chat", "role": "navigator", "text": "hi"})
    out = b.replay()
    assert [e.get("n") for e in out[:-1]] == list(range(FULL_GEOMETRY + 3))  # nothing dropped from the log
    assert sum("legs" in e for e in out) == FULL_GEOMETRY and "legs" in out[-2] and "legs" not in out[0]
    assert all("result" in e for e in out[:-1])


def test_de440s_mission_is_drawn_with_de440s():
    t0, t1 = to_mjd2000("2026-10-31"), to_mjd2000("2027-08-19")
    [leg] = transfer_path(["earth", "mars"], [t0, t1], "de440s")
    assert np.allclose(leg["points_au"][-1], planet_position_au("mars", t1, "de440s"), atol=1e-6)
    late = transfer_path(["earth", "mars"], [to_mjd2000("2100-01-01"), to_mjd2000("2100-10-01")], "de440s")
    assert len(late[0]["points_au"]) == 160  # past jpl_lp's 2050 limit, still drawable


def test_bridge_refuses_non_loopback():
    from bridge.server import run
    with pytest.raises(ValueError):
        asyncio.run(run(host="0.0.0.0", port=free_port()))


# ------------------------------------------------------------------ push from the MCP server

def test_push_delivers_tool_results_with_geometry(bridge, monkeypatch):
    from mcp_server import push
    monkeypatch.setenv("MISSION_CONTROL_BRIDGE", f"ws://127.0.0.1:{bridge}/ws")
    with browser(bridge) as (ws, _):
        push.publish("unit_test", {"x": 1}, lambda: {"legs": [], "extra": 2})
        msg = json.loads(ws.recv(timeout=10))
        assert msg["type"] == "mission" and msg["kind"] == "unit_test" and msg["result"] == {"x": 1}
        assert msg["extra"] == 2
        push.publish("broken_geometry", {"x": 2}, lambda: 1 / 0)
        msg = json.loads(ws.recv(timeout=10))
        assert msg["kind"] == "broken_geometry" and "ZeroDivisionError" in msg["geometry_error"]


def test_push_without_a_bridge_never_blocks(monkeypatch):
    from mcp_server import push
    monkeypatch.setenv("MISSION_CONTROL_BRIDGE", f"ws://127.0.0.1:{free_port()}/ws")  # nothing listening
    t = time.time()
    for i in range(50):
        push.publish("nobody", {"i": i})
    assert time.time() - t < 0.5
    assert push.flush(timeout_s=10)


def test_mcp_tool_call_reaches_the_browser(bridge):
    """End to end: an MCP client calls solve_transfer over stdio; the browser gets the arc."""
    import anyio
    from mcp.client import Client
    from evals.runners import server_params

    async def call():
        async with Client(server_params(bridge=f"ws://127.0.0.1:{bridge}/ws")) as c:
            r = await c.call_tool("solve_transfer", {"origin": "earth", "destination": "mars",
                                                     "departure_date": "2026-10-31", "arrival_date": "2027-08-19"})
            assert not r.is_error
            await anyio.sleep(3)  # the push thread runs after the response

    with browser(bridge) as (ws, _):
        anyio.run(call)
        msg = json.loads(ws.recv(timeout=20))
    assert msg["kind"] == "solve_transfer" and msg["result"]["c3_km2_s2"] == 9.145
    leg = msg["legs"][0]
    assert (leg["from"], leg["to"]) == ("earth", "mars") and len(leg["points_au"]) == 160
    assert set(msg["tracks"]["positions_au"]) >= {"earth", "mars"}
