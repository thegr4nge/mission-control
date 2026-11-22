"""Drive the physics MCP server the way an agent would, so the browser shows a live mission.

  pixi run bridge          (in one terminal, then open http://127.0.0.1:8765/)
  pixi run python -m scripts.demo_calls
"""

from __future__ import annotations

import json
import sys
import time

import anyio
from mcp.client import Client

from evals.runners import server_params

CALLS = [
    ("find_launch_window", {"origin": "earth", "destination": "mars", "earliest_departure": "2026-09-01",
                            "latest_departure": "2027-03-01", "min_time_of_flight_days": 100,
                            "max_time_of_flight_days": 450, "objective": "c3"}),
    ("check_delta_v_budget", {"maneuvers_km_s": {"departure": 3.632, "capture": 2.13}, "isp_s": 320,
                              "dry_mass_kg": 1500, "propellant_mass_kg": 4000}),
    ("propose_flyby_sequence", {"origin": "earth", "destination": "mercury", "earliest_departure": "1973-10-01",
                                "latest_departure": "1974-01-29",
                                "sequences": [["earth", "mercury"], ["earth", "venus", "mercury"]],
                                "arrival": "flyby"}),
]


async def main(names: list[str]) -> None:
    calls = [c for c in CALLS if not names or c[0] in names]
    async with Client(server_params(bridge="ws://127.0.0.1:8765/ws")) as c:
        for name, args in calls:
            t = time.time()
            r = await c.call_tool(name, args)
            text = " ".join(getattr(x, "text", "") for x in r.content)
            print(f"{name}: {'ERROR ' if r.is_error else ''}{time.time() - t:.1f} s, {json.dumps(json.loads(text))[:160] if not r.is_error else text[:200]}")
        await anyio.sleep(3)  # let the push thread deliver before the server exits


if __name__ == "__main__":
    anyio.run(main, sys.argv[1:])
