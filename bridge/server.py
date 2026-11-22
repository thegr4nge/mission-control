"""Local bridge between the physics MCP server and the browser.

One port, loopback only:
  GET /            the web app (web/)
  WS  /ws          publishers (the MCP server, the crew) send {"type": "publish", "event": {...}};
                   browsers receive every event, plus the recent history when they connect, and can
                   ask {"type": "request", "id": n, "what": "planets" | "tracks", ...} for geometry.

Run:  pixi run bridge          then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
from collections import deque
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

log = logging.getLogger("mission_control.bridge")

WEB_ROOT = (Path(__file__).resolve().parents[1] / "web").resolve()
HISTORY = 200  # events replayed to a browser that connects
FULL_GEOMETRY = 5  # of those, only the latest few keep their arcs, tracks and porkchop
MAX_MESSAGE = 2**22  # 4 MiB; a launch-window event with porkchop and tracks is about 215 kB
GEOMETRY_KEYS = ("legs", "tracks", "porkchop")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


@lru_cache(maxsize=1)
def _orbits() -> dict:
    from physics.visual import planet_orbits
    return planet_orbits()


def _tracks(t0: float, t1: float) -> dict:
    from physics.bodies import PLANETS
    from physics.visual import planet_tracks
    return planet_tracks(PLANETS, t0, t1)


class Bridge:
    def __init__(self, history: int = HISTORY):
        self.browsers: set[ServerConnection] = set()
        self.history: deque[dict] = deque(maxlen=history)

    @staticmethod
    def allowed_hosts(connection: ServerConnection) -> set[str]:
        port = connection.local_address[1]
        return {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}

    @classmethod
    def allowed_origins(cls, connection: ServerConnection) -> set[str]:
        return {f"http://{h}" for h in cls.allowed_hosts(connection)}

    def replay(self) -> list[dict]:
        """History for a new browser: every event, but geometry only on the latest FULL_GEOMETRY missions
        (older entries still show in the log; their arcs are dropped to keep the replay small)."""
        out, kept = [], 0
        for ev in reversed(self.history):
            if any(k in ev for k in GEOMETRY_KEYS):
                if kept >= FULL_GEOMETRY:
                    ev = {k: v for k, v in ev.items() if k not in GEOMETRY_KEYS}
                kept += 1
            out.append(ev)
        return out[::-1]

    def static(self, connection: ServerConnection, request: Request) -> Response | None:
        """Serve files from web/ for plain HTTP requests; let /ws through to the websocket handler."""
        # DNS rebinding: a hostile domain resolving to 127.0.0.1 arrives with its own Host header.
        if request.headers.get("Host") not in self.allowed_hosts(connection):
            return _response(HTTPStatus.MISDIRECTED_REQUEST, b"unexpected Host", "text/plain")
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            # Browsers send Origin on websocket upgrades: refuse any page but our own, so a website open
            # in another tab cannot drive or read the bridge. Non-browser clients (the MCP server) send none.
            origin = request.headers.get("Origin")
            if origin is not None and origin not in self.allowed_origins(connection):
                return _response(HTTPStatus.FORBIDDEN, b"origin not allowed", "text/plain")
            return None
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB_ROOT / rel).resolve()
        if WEB_ROOT not in target.parents and target != WEB_ROOT or not target.is_file():
            return _response(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return _response(HTTPStatus.OK, target.read_bytes(), ctype)

    async def handler(self, ws: ServerConnection) -> None:
        role = "browser"
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    msg = None
                if not isinstance(msg, dict):
                    await ws.send(json.dumps({"type": "error", "error": "messages must be JSON objects"}))
                    continue
                kind = msg.get("type")
                if kind == "hello":
                    role = msg.get("role", "browser")
                    if role == "browser":
                        self.browsers.add(ws)
                        await ws.send(json.dumps({"type": "history", "events": self.replay()}))
                elif kind == "publish":
                    event = msg.get("event")
                    if not isinstance(event, dict):
                        continue
                    self.history.append(event)
                    self.broadcast(json.dumps(event))
                elif kind == "request":
                    await ws.send(json.dumps(await self.answer(msg)))
                elif kind == "clear":
                    self.history.clear()
                    self.broadcast(json.dumps({"type": "cleared"}))
        except ConnectionClosed:
            pass
        finally:
            self.browsers.discard(ws)

    def broadcast(self, text: str) -> None:
        """Send without waiting on any browser: a stalled tab must not stall the publisher."""
        broadcast(self.browsers, text)

    async def answer(self, msg: dict) -> dict:
        rid, what = msg.get("id"), msg.get("what")
        try:
            if what == "planets":
                data = await asyncio.to_thread(_orbits)
            elif what == "tracks":
                t0, t1 = float(msg["t0_mjd2000"]), float(msg["t1_mjd2000"])
                if not t1 > t0 or t1 - t0 > 200 * 365.25:
                    raise ValueError("need t0 < t1 and a span under 200 years")
                data = await asyncio.to_thread(_tracks, t0, t1)
            else:
                raise ValueError(f"unknown request {what!r}")
            return {"type": "response", "id": rid, "what": what, "data": data}
        except (KeyError, TypeError, ValueError) as exc:
            return {"type": "response", "id": rid, "what": what, "error": str(exc)}


def _response(status: HTTPStatus, body: bytes, ctype: str) -> Response:
    headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                       ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff")])
    return Response(status.value, status.phrase, headers, body)


async def run(host: str = "127.0.0.1", port: int = 8765, ready: asyncio.Event | None = None) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the bridge serves loopback only")
    bridge = Bridge()
    async with serve(bridge.handler, host, port, process_request=bridge.static, max_size=MAX_MESSAGE) as server:
        log.info("Mission Control bridge on http://%s:%d/", host, port)
        print(f"Mission Control on http://{host}:{port}/", flush=True)
        if ready is not None:
            ready.set()
        await server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mission Control browser bridge (loopback only)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(port=a.port))


if __name__ == "__main__":
    main()
