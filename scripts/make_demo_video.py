"""Record the demo video: the real app, real tool calls, a replay of real crew missions.

  pixi run python -m scripts.make_demo_video            -> docs/media/mission-control-demo.mp4

How it works: starts its own bridge on a spare port (your open browser is not touched), opens the
app in headless Chrome, and drives it over the Chrome DevTools Protocol one frame at a time, so
the animation is smooth and the result repeatable. Arcs, porkchop and numbers come from live MCP
tool calls; crew messages are replayed from saved missions (missions/*.json); every number in a
caption is read from a tool result on the page. Frames are encoded with ffmpeg.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets
from mcp.client import Client

from evals.runners import pixi_exe, server_params

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "media" / "mission-control-demo.mp4"
PORT, CDP_PORT, FPS, W, H = 8799, 9333, 20, 1280, 720
MARS = REPO / "missions" / "20261123-052839.json"
SATURN = REPO / "missions" / "20261123-053115.json"
CHROME = next(p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") if Path(p).exists())

OVERLAY_JS = r"""
(() => {
  if (window.__cap) return;
  const d = document.createElement('div');
  d.id = '__cap';
  d.style.cssText = 'position:fixed;left:16px;right:436px;top:56px;padding:10px 16px;border-radius:8px;' +
    'background:rgba(5,7,13,0.82);color:#fff;font:600 20px/1.35 system-ui,sans-serif;text-align:center;' +
    'z-index:50;transition:opacity .3s;opacity:0;pointer-events:none';
  document.body.appendChild(d);
  const card = document.createElement('div');
  card.id = '__card';
  card.style.cssText = 'position:fixed;inset:0;display:none;align-items:center;justify-content:center;' +
    'flex-direction:column;background:#05070d;color:#d8e1f0;font:16px system-ui;z-index:60;text-align:center';
  document.body.appendChild(card);
  window.__cap = (t) => { d.innerHTML = t; d.style.opacity = t ? 1 : 0; };
  window.__card = (html) => { card.innerHTML = html || ''; card.style.display = html ? 'flex' : 'none'; };
  window.__orbit = (a) => {
    const c = __mc.camera, t = __mc.controls.target;
    const x = c.position.x - t.x, z = c.position.z - t.z;
    c.position.x = t.x + x * Math.cos(a) - z * Math.sin(a);
    c.position.z = t.z + x * Math.sin(a) + z * Math.cos(a);
    c.lookAt(t);
  };
  window.__zoom = (f) => { const c = __mc.camera, t = __mc.controls.target; c.position.sub(t).multiplyScalar(f).add(t); };
  window.__tab = (name) => document.querySelector(`nav button[data-tab="${name}"]`).click();
  window.__select = (kind, nth = 0) => {
    const items = [...document.querySelectorAll('#log > li')].filter(li => li.querySelector('.kind').textContent === kind);
    const li = items[items.length - 1 - nth]; if (li) li.click(); return !!li;
  };
  window.__crewBottom = () => { const t = document.getElementById('tab-crew'); t.scrollTop = t.scrollHeight; };
  window.__sf = (v) => (typeof v === 'number' ? (Math.abs(v) >= 1e4 ? String(v) : v.toPrecision(4)) : v);
})();
"""


class Page:
    """Minimal Chrome DevTools Protocol client."""

    def __init__(self, ws):
        self.ws, self.n, self.frames, self.dir = ws, 0, 0, None

    async def call(self, method, **params):
        self.n += 1
        my = self.n
        await self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error in {expr[:80]}: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    async def frame(self):
        await self.js("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        shot = await self.call("Page.captureScreenshot", format="jpeg", quality=88)
        import base64
        (self.dir / f"{self.frames:05d}.jpg").write_bytes(base64.b64decode(shot["data"]))
        self.frames += 1

    async def hold(self, seconds, each=None):
        n = max(1, round(seconds * FPS))
        for i in range(n):
            if each:
                await self.js(each(i, n))
            await self.frame()

    async def hold_while(self, task, each=None, min_s=1.0):
        """Keep recording (with an optional per-frame script) until the task finishes."""
        i = 0
        start = time.time()
        while not task.done() or time.time() - start < min_s:
            await self.js(each(i, 10**6) if each else "0")
            await self.frame()
            i += 1
        return task.result()


class Publisher:
    def __init__(self, ws):
        self.ws = ws

    async def send(self, event):
        await self.ws.send(json.dumps({"type": "publish", "event": event}))

    async def chat(self, role, text):
        await self.send({"type": "chat", "role": role, "text": text, "time_unix": time.time()})


def turn(mission, role):
    return next(t for t in mission["turns"] if t["role"] == role)


def first_paragraphs(text, n_chars=700):
    out, total = [], 0
    for para in text.split("\n\n"):
        if total + len(para) > n_chars and out:
            break
        out.append(para)
        total += len(para)
    return "\n\n".join(out)


async def record(frames_dir: Path):
    mars = json.loads(MARS.read_text(encoding="utf-8"))
    saturn = json.loads(SATURN.read_text(encoding="utf-8"))

    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as pub_ws, \
            Client(server_params(bridge=f"ws://127.0.0.1:{PORT}/ws")) as mcp:
        pub = Publisher(pub_ws)
        await pub_ws.send(json.dumps({"type": "hello", "role": "publisher"}))
        await pub_ws.send(json.dumps({"type": "clear"}))

        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list").read())
        page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        async with websockets.connect(page_ws, max_size=None) as cdp:
            p = Page(cdp)
            p.dir = frames_dir
            await p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await p.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/")
            await asyncio.sleep(6)  # page, three.js from the CDN, planet orbits from pykep
            await p.js(OVERLAY_JS)
            await p.js("__mc.state.playing = false; __mc.fitCamera(); __zoom(1.3)")

            async def tool(name, args):
                r = await mcp.call_tool(name, args)
                return json.loads("\n".join(getattr(c, "text", "") for c in r.content))

            # 1. Title
            await p.js("__card('<div style=\"font-size:44px;font-weight:700;letter-spacing:.04em\">Mission Control</div>"
                       "<div style=\"margin-top:14px;font-size:20px;color:#7d8aa5\">An AI crew designs space missions "
                       "on real orbital mechanics</div><div style=\"margin-top:28px;font-size:16px;color:#ffb347\">"
                       "Agents orchestrate. pykep computes. Every number comes from a physics tool.</div>')")
            await p.hold(4.0)
            await p.js("__card('')")
            await p.js("__cap('The solar system today, from ESA&#39;s pykep ephemerides')")
            await p.hold(3.5, lambda i, n: "__orbit(0.004)")

            # 2. Mars mission: goal, crew, real tool calls
            await p.js("__tab('crew')")
            await pub.chat("system", f"Mission goal: {mars['goal']}")
            await p.js("__cap('A plain-English goal goes to the crew')")
            await p.hold(3.0, lambda i, n: "__orbit(0.003)")
            await pub.chat("flight director", first_paragraphs(turn(mars, "flight director")["text"], 600))
            await p.hold(3.0, lambda i, n: "__orbit(0.003)")

            nav_calls = turn(mars, "navigator")["tool_calls"]
            await p.js("__tab('log'); __cap('The navigator calls the physics tools. When is the next Mars window?')")
            task = asyncio.create_task(tool(nav_calls[0]["name"], nav_calls[0]["arguments"]))
            opp = await p.hold_while(task, lambda i, n: "__orbit(0.003)")
            await p.hold(2.0, lambda i, n: "__orbit(0.003)")
            best = opp["opportunities"][0]["best_c3"]
            await p.js(f"__cap('Next window found by search, not memory: {best['departure_utc'][:10]}, "
                       f"C3 ' + __sf({best['c3_km2_s2']}) + ' km&sup2;/s&sup2;')")
            await p.hold(3.0, lambda i, n: "__orbit(0.003)")

            task = asyncio.create_task(tool(nav_calls[1]["name"], nav_calls[1]["arguments"]))
            await p.hold_while(task, lambda i, n: "__orbit(0.003)")
            await p.js("__tab('porkchop'); __cap('Porkchop plot of the window search: the circle is the optimum')")
            await p.hold(5.0)
            await p.js("__tab('crew')")
            await pub.chat("navigator", first_paragraphs(turn(mars, "navigator")["text"], 900))
            await p.hold(3.0)

            prop_call = turn(mars, "propulsion")["tool_calls"][0]
            budget = await tool(prop_call["name"], prop_call["arguments"])
            await pub.chat("propulsion", turn(mars, "propulsion")["text"])
            await p.js(f"__crewBottom(); __cap('Propulsion checks the spacecraft: needs ' + __sf({budget['required_with_margin_km_s']}) + "
                       f"' km/s with margin, has ' + __sf({budget['available_km_s']}) + ' km/s. "
                       f"{'Feasible' if budget['feasible'] else 'Not feasible'}.')")
            await p.hold(4.0)

            # 3. Fly it
            await p.js("__tab('log'); __select('find_launch_window', 1); __zoom(0.9)")
            r = await p.js("JSON.stringify(__mc.state.selected.result)")
            r = json.loads(r)
            await p.js(f"__cap('Earth to Mars, Type {r['trajectory_type']}: departs {r['departure_utc'][:10]}, "
                       f"' + __sf({r['time_of_flight_days']}) + ' days, total &Delta;v ' + __sf({r['dv_total_km_s']}) + ' km/s')")
            await p.hold(9.0, lambda i, n: f"(() => {{ const s = __mc.state; __mc.setTime(s.t0 + (s.t1 - s.t0) * {i} / {n - 1}); __orbit(0.002); }})()")

            # 4. Grounding
            await p.js("__tab('crew')")
            await pub.chat("system", "navigator report sent back: these numbers do not come from any tool result or "
                                     "the goal: 12.6, 5.9")
            await p.js("__crewBottom(); __cap('Numbers no tool returned are caught and sent back. "
                       "Here the crew had converted days to years by itself')")
            await p.hold(5.0)

            # 5. Saturn: direct against a Jupiter assist
            await pub.chat("system", f"Mission goal: {saturn['goal']}")
            sat_call = turn(saturn, "navigator")["tool_calls"][1]
            await p.js("__crewBottom(); __cap('Next goal: a Saturn flyby. Direct, or via a Jupiter gravity assist?')")
            task = asyncio.create_task(tool(sat_call["name"], sat_call["arguments"]))
            ranked = (await p.hold_while(task, lambda i, n: "__orbit(0.003)", min_s=3.0))["ranked"]
            await p.js("__tab('log')")
            await p.hold(1.0)
            a, b = ranked[0], ranked[-1]
            await p.js(f"__cap('{' &rarr; '.join(s.title() for s in a['sequence'])}: &Delta;v ' + __sf({a['dv_total_km_s']}) + "
                       f"' km/s over ' + __sf({a['time_of_flight_days']}) + ' days, against ' + __sf({b['dv_total_km_s']}) + "
                       f"' km/s and ' + __sf({b['time_of_flight_days']}) + ' days direct')")
            await p.hold(9.0, lambda i, n: f"(() => {{ const s = __mc.state; __mc.setTime(s.t0 + (s.t1 - s.t0) * {i} / {n - 1}); __orbit(0.0015); }})()")

            # 6. Voyager 2: the real grand tour, checked against the flown dates
            from physics.bodies import PLANETS, to_mjd2000
            from physics.flyby import MgaProblem
            from physics.references import VOYAGER_2
            from physics.visual import planet_tracks, transfer_path
            bodies = list(VOYAGER_2)
            epochs = [to_mjd2000(VOYAGER_2[b] + "T12:00") for b in bodies]
            enc = MgaProblem(bodies, 0, 1, [(0, 1)] * 4, arrival_mode="flyby").evaluate(
                [epochs[0]] + [epochs[i + 1] - epochs[i] for i in range(4)]).encounters
            pct = [100 * e.dv_km_s / e.vinf_in_km_s for e in enc[1:-1]]
            await pub.send({"type": "mission", "kind": "voyager_2_flown_dates", "time_unix": time.time(),
                            "result": {"note": "Voyager 2 on its NSSDCA dates, patched conics"},
                            "legs": transfer_path(bodies, epochs), "tracks": planet_tracks(PLANETS, epochs[0], epochs[-1])})
            await p.hold(1.0)
            await p.js("__zoom(1.2)")
            worst = f"{max(pct):.2g}"  # rounded, never truncated: 1.597 -> 1.6
            await p.js(f"__cap('Voyager 2 on its real 1977-1989 dates: no giant-planet flyby needs more than {worst} % "
                       f"of its arrival speed in burns. The physics reproduces a near-free grand tour')")
            await p.hold(10.0, lambda i, n: f"(() => {{ const s = __mc.state; __mc.setTime(s.t0 + (s.t1 - s.t0) * {i} / {n - 1}); __orbit(0.002); }})()")

            # 7. End card
            await p.js("__cap(''); __card('<div style=\"font-size:34px;font-weight:700\">Mission Control</div>"
                       "<div style=\"margin-top:16px;font-size:18px;color:#7d8aa5\">Verified against NASA mission design "
                       "tables, Curtis and Vallado, Mariner 10 and Voyager 2</div><div style=\"margin-top:10px;font-size:18px;"
                       "color:#7d8aa5\">Crew on free-tier models &middot; Claude through MCP &middot; Apache-2.0</div>"
                       "<div style=\"margin-top:26px;font-size:18px;color:#ffb347\">github.com/thegr4nge/mission-control</div>')")
            await p.hold(4.0)
            return p.frames


def main() -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found", file=sys.stderr)
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mc-video-", ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        frames = tmp / "frames"
        frames.mkdir()
        bridge = subprocess.Popen([pixi_exe(), "run", "--manifest-path", str(REPO / "pixi.toml"), "python", "-m",
                                   "bridge.server", "--port", str(PORT)], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
                                   f"--window-size={W},{H}", "--hide-scrollbars", "--mute-audio",
                                   "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
                                   f"--user-data-dir={tmp / 'profile'}", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):  # wait for both servers
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
                    urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=1)
                    break
                except OSError:
                    time.sleep(0.5)
            t = time.time()
            n = asyncio.run(record(frames))
            print(f"captured {n} frames ({n / FPS:.1f} s of video) in {time.time() - t:.0f} s", flush=True)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.jpg"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart",
                            str(OUT)], check=True)
            shutil.copy(frames / f"{n // 2:05d}.jpg", OUT.with_name("frame-mid.jpg"))
        finally:
            for proc in (chrome, bridge):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
