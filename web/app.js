// Mission Control front end. Draws what the physics server computed; computes no physics itself.
// Positions arrive in AU, ecliptic J2000 (x, y in the ecliptic, z north); three.js is y-up, so
// ecliptic (x, y, z) is drawn at (x, z, -y).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const PLANET_COLORS = {
  mercury: 0xb5b5b5, venus: 0xe8c27a, earth: 0x4fa3ff, mars: 0xff6a3d,
  jupiter: 0xd9a066, saturn: 0xe3d18a, uranus: 0x8fe3e8, neptune: 0x5b7bff,
};
const LEG_COLORS = [0x00e5ff, 0xff6bd6, 0x5ad19a, 0xffffff]; // distinct from every planet colour
const MS_PER_DAY = 86400000;
const J2000_MS = Date.UTC(2000, 0, 1, 0, 0, 0); // pykep MJD2000 day 0

const $ = (id) => document.getElementById(id);
const toVec = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
const mjdToDate = (mjd) => new Date(J2000_MS + mjd * MS_PER_DAY);
const dateToMjd = (d) => (d.getTime() - J2000_MS) / MS_PER_DAY;
const isoDay = (mjd) => mjdToDate(mjd).toISOString().slice(0, 10);
const isoMinute = (mjd) => mjdToDate(mjd).toISOString().slice(0, 16).replace("T", " ");
// The server rounds to 4 significant figures; JSON drops trailing zeros (2.700 -> 2.7), so put them back.
const sf4 = (v) => {
  if (typeof v !== "number" || !Number.isFinite(v)) return v;
  if (v === 0) return "0.000";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return v.toPrecision(4);   // 2 -> "2.000", 320 -> "320.0", 9.15 -> "9.150"
  if (a >= 1e4 && a < 1e6 && Number.isInteger(v)) return String(v); // 42160 stays whole
  return v.toExponential(3);
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- scene
const container = $("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = "absolute";
labels.domElement.style.inset = "0";
labels.domElement.style.pointerEvents = "none";
container.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
camera.position.set(0, 2.2, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const sun = new THREE.Mesh(new THREE.SphereGeometry(0.03, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffd27a }));
scene.add(sun);
addLabel(sun, "Sun");
scene.add(new THREE.GridHelper(80, 80, 0x16203a, 0x0e1526)); // 1 AU squares in the ecliptic

const orbitGroup = new THREE.Group();
const planetGroup = new THREE.Group();
const missionGroup = new THREE.Group();
scene.add(orbitGroup, planetGroup, missionGroup);

const planetMarkers = {};
for (const [name, color] of Object.entries(PLANET_COLORS)) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), new THREE.MeshBasicMaterial({ color }));
  m.visible = false;
  addLabel(m, name[0].toUpperCase() + name.slice(1));
  planetGroup.add(m);
  planetMarkers[name] = m;
}
const craft = new THREE.Mesh(new THREE.OctahedronGeometry(0.02), new THREE.MeshBasicMaterial({ color: 0xffffff }));
craft.visible = false;
addLabel(craft, "spacecraft");
scene.add(craft);

function addLabel(obj, text) {
  const div = document.createElement("div");
  div.className = "label";
  div.textContent = text;
  const l = new CSS2DObject(div);
  l.position.set(0, 0.04, 0);
  obj.add(l);
}

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  labels.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function drawOrbits(orbits) {
  disposeGroup(orbitGroup);
  for (const [name, pts] of Object.entries(orbits)) {
    const g = new THREE.BufferGeometry().setFromPoints(pts.map(toVec));
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: PLANET_COLORS[name], transparent: true, opacity: 0.35 }));
    line.name = name;
    orbitGroup.add(line);
  }
}

// ---------------------------------------------------------------- time and tracks
const state = {
  tracks: null,          // {times_mjd2000, positions_au: {planet: [[x,y,z]]}}
  legs: [],              // [{from, to, depart_mjd2000, arrive_mjd2000, points_au}]
  t0: dateToMjd(new Date()) - 30, t1: dateToMjd(new Date()) + 3 * 365,
  t: dateToMjd(new Date()),
  playing: false,
  events: [],
  selected: null,
};

function interp(times, series, t) {
  if (!times || times.length === 0) return null;
  if (t <= times[0]) return series[0];
  const n = times.length - 1;
  if (t >= times[n]) return series[n];
  let lo = 0, hi = n;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (times[mid] <= t) lo = mid; else hi = mid; }
  const f = (t - times[lo]) / (times[hi] - times[lo]);
  const a = series[lo], b = series[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function craftPosition(t) {
  for (const leg of state.legs) {
    if (t >= leg.depart_mjd2000 && t <= leg.arrive_mjd2000) {
      const pts = leg.points_au; // sampled uniformly in time by the server
      const f = (t - leg.depart_mjd2000) / (leg.arrive_mjd2000 - leg.depart_mjd2000) * (pts.length - 1);
      const i = Math.min(Math.floor(f), pts.length - 2), u = f - i;
      return { pos: pts[i].map((v, k) => v + (pts[i + 1][k] - v) * u), leg };
    }
  }
  return null;
}

function setTime(t) {
  state.t = Math.min(Math.max(t, state.t0), state.t1);
  $("time").value = String(Math.round(1000 * (state.t - state.t0) / (state.t1 - state.t0)));
  $("date").textContent = isoMinute(state.t) + " UTC";
  if (state.tracks) {
    for (const [name, series] of Object.entries(state.tracks.positions_au)) {
      const p = interp(state.tracks.times_mjd2000, series, state.t);
      if (p) { planetMarkers[name].position.copy(toVec(p)); planetMarkers[name].visible = true; }
    }
  }
  const c = craftPosition(state.t);
  craft.visible = !!c;
  if (c) craft.position.copy(toVec(c.pos));
  const first = state.legs[0], last = state.legs[state.legs.length - 1];
  let phase = "";
  if (c) phase = `cruise ${c.leg.from} → ${c.leg.to}, day ${(state.t - first.depart_mjd2000).toFixed(0)}`;
  else if (first && state.t < first.depart_mjd2000) phase = `launch in ${(first.depart_mjd2000 - state.t).toFixed(0)} days`;
  else if (last) phase = `arrived ${last.to}`;
  $("phase").textContent = phase;
}

$("time").addEventListener("input", (e) => {
  setTime(state.t0 + (state.t1 - state.t0) * Number(e.target.value) / 1000);
});
$("play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play").innerHTML = state.playing ? "&#10074;&#10074;" : "&#9654;";
  if (state.playing && state.t >= state.t1) setTime(state.t0);
});

let lastFrame = performance.now();
function frame(now) {
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (state.playing) {
    const span = state.t1 - state.t0;
    setTime(state.t + span * dt / 20); // the whole span in 20 s
    if (state.t >= state.t1) { state.playing = false; $("play").innerHTML = "&#9654;"; }
  }
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------------------------------------------------------------- missions
function disposeGroup(group) {
  for (const o of group.children) { o.geometry?.dispose(); o.material?.dispose(); }
  group.clear();
}

function showMission(ev) {
  disposeGroup(missionGroup);
  state.legs = ev.legs || [];
  state.legs.forEach((leg, i) => {
    const g = new THREE.BufferGeometry().setFromPoints(leg.points_au.map(toVec));
    missionGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: LEG_COLORS[i % LEG_COLORS.length] })));
  });
  if (ev.tracks) {
    state.tracks = ev.tracks;
    state.t0 = ev.tracks.times_mjd2000[0];
    state.t1 = ev.tracks.times_mjd2000[ev.tracks.times_mjd2000.length - 1];
  }
  const bodies = new Set(state.legs.flatMap((l) => [l.from, l.to]));
  for (const line of orbitGroup.children) line.material.opacity = bodies.size === 0 || bodies.has(line.name) ? 0.6 : 0.18;
  fitCamera();
  setTime(state.t0);
  $("mission-title").textContent = title(ev);
  if (ev.porkchop) drawPorkchop(ev.porkchop); else clearPorkchop();
  legend();
}

function fitCamera() {
  let r = 1.2;
  for (const leg of state.legs) for (const p of leg.points_au) r = Math.max(r, Math.hypot(p[0], p[1], p[2]) * 1.15);
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.set(0, 0, 0);
  camera.position.copy(dir.multiplyScalar(r * 2.4));
}

function legend() {
  const legs = state.legs.map((l, i) =>
    `<span style="color:#${LEG_COLORS[i % LEG_COLORS.length].toString(16).padStart(6, "0")}">━</span> ${esc(l.from)} → ${esc(l.to)}`);
  $("legend").innerHTML = (legs.length ? legs.join("<br>") + "<br>" : "") +
    "grid: 1 AU squares, ecliptic J2000. Planets and arcs from pykep.";
}

function title(ev) {
  const r = ev.result || {};
  if (ev.kind === "propose_flyby_sequence" && r.ranked && r.ranked[0]) return r.ranked[0].sequence.join(" → ");
  if (r.origin) return `${r.origin} → ${r.destination}, departing ${String(r.departure_utc).slice(0, 10)}`;
  return ev.kind;
}

function facts(ev) {
  const r = ev.result || {};
  const f = (label, v, unit = "") => (v === null || v === undefined ? "" : `${label} ${esc(sf4(v))}${unit ? " " + unit : ""}`);
  const join = (...xs) => xs.filter(Boolean).join(" · ");
  switch (ev.kind) {
    case "solve_transfer":
    case "find_launch_window":
      return join(f("depart", String(r.departure_utc).slice(0, 10)), f("arrive", String(r.arrival_utc).slice(0, 10)),
        f("TOF", r.time_of_flight_days, "d"), f("Type", r.trajectory_type), f("C3", r.c3_km2_s2, "km²/s²"),
        f("v∞ arr", r.vinf_arrival_km_s, "km/s"), f("Δv total", r.dv_total_km_s, "km/s"));
    case "propose_flyby_sequence":
      return (r.ranked || []).map((s, i) => `${i + 1}. ${esc(s.sequence.join("→"))}: Δv ${esc(sf4(s.dv_total_km_s))} km/s, C3 ${esc(sf4(s.c3_km2_s2))} km²/s²`).join("<br>");
    case "hohmann_transfer":
      return join(f("burn 1", r.dv_first_burn_km_s, "km/s"), f("burn 2", r.dv_second_burn_km_s, "km/s"),
        f("total", r.dv_total_km_s, "km/s"), f("TOF", r.time_of_flight_hours, "h"));
    case "check_delta_v_budget":
      return join(f("needs", r.required_with_margin_km_s, "km/s"), f("has", r.available_km_s, "km/s"),
        r.feasible ? "feasible" : "NOT feasible");
    case "propagate": {
      const e = r.final_elements || {};
      return join(f("a", e.semi_major_axis_km, "km"), f("e", e.eccentricity), f("i", e.inclination_deg, "°"),
        f("about", r.central_body));
    }
    default:
      return "";
  }
}

function addLog(ev, select) {
  state.events.push(ev);
  const li = document.createElement("li");
  const when = ev.time_unix ? new Date(ev.time_unix * 1000).toLocaleTimeString() : "";
  li.innerHTML = `<span class="when">${esc(when)}</span><span class="kind">${esc(ev.kind)}</span>` +
    `<div class="facts">${facts(ev)}</div>` +
    (ev.result && ev.result.warning ? `<div class="warn">${esc(ev.result.warning)}</div>` : "") +
    (ev.geometry_error ? `<div class="warn">geometry: ${esc(ev.geometry_error)}</div>` : "");
  if (ev.legs) {
    li.addEventListener("click", () => select_(li, ev));
    if (select) select_(li, ev);
  }
  $("log").prepend(li);
}

function select_(li, ev) {
  document.querySelectorAll("#log li.selected").forEach((x) => x.classList.remove("selected"));
  li.classList.add("selected");
  state.selected = ev;
  showMission(ev);
}

function addChat(ev) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="who">${esc(ev.role || "crew")}</span><span class="text">${esc(ev.text || "")}</span>`;
  $("crew").append(li);
  li.scrollIntoView({ block: "end" });
}

// ---------------------------------------------------------------- porkchop
function drawPorkchop(pc) {
  const cv = $("porkchop"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, L = 60, B = 40, T = 10, R = 10;
  const deps = pc.departures_mjd2000, tofs = pc.tofs_days, vals = pc.values;
  const flat = vals.flat().filter((v) => v !== null && Number.isFinite(v)).sort((a, b) => a - b);
  if (!flat.length) return;
  const vmin = flat[0], vmax = Math.min(flat[Math.floor(flat.length * 0.6)], vmin * 4 + 1e-9);
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  const cw = (W - L - R) / deps.length, ch = (H - T - B) / tofs.length;
  for (let i = 0; i < deps.length; i++) for (let j = 0; j < tofs.length; j++) {
    const v = vals[i][j];
    if (v === null || v > vmax) continue;
    ctx.fillStyle = viridis((v - vmin) / (vmax - vmin || 1));
    ctx.fillRect(L + i * cw, H - B - (j + 1) * ch, Math.ceil(cw), Math.ceil(ch));
  }
  // Cell i covers [i, i+1) * cw, centred on departure deps[i]: map dates onto cell centres.
  const dStep = deps.length > 1 ? deps[1] - deps[0] : 1, tStep = tofs.length > 1 ? tofs[1] - tofs[0] : 1;
  const x = (mjd) => L + ((mjd - deps[0]) / dStep + 0.5) * cw;
  const y = (tof) => H - B - ((tof - tofs[0]) / tStep + 0.5) * ch;
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x(pc.best.departure_mjd2000), y(pc.best.tof_days), 6, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = "#7d8aa5"; ctx.font = "12px system-ui";
  for (let k = 0; k <= 4; k++) {
    const d = deps[0] + (deps[deps.length - 1] - deps[0]) * k / 4;
    ctx.fillText(isoDay(d), x(d) - 30, H - B + 16);
    const t = tofs[0] + (tofs[tofs.length - 1] - tofs[0]) * k / 4;
    ctx.fillText(t.toFixed(0) + " d", 8, y(t) + 4);
  }
  ctx.fillText("departure date →", W / 2 - 50, H - 6);
  $("porkchop-caption").textContent =
    `${pc.objective} over the coarse search grid (${deps.length} × ${tofs.length} shown); ` +
    `colour from ${sf4(vmin)} (dark) to ${sf4(vmax)}, blank above. Circle: refined best.`;
  cv.onmousemove = (e) => {
    const rect = cv.getBoundingClientRect();
    const px = (e.clientX - rect.left) * W / rect.width, py = (e.clientY - rect.top) * H / rect.height;
    const i = Math.floor((px - L) / cw), j = Math.floor((H - B - py) / ch);
    if (i < 0 || j < 0 || i >= deps.length || j >= tofs.length) { $("porkchop-hover").textContent = ""; return; }
    const v = vals[i][j];
    $("porkchop-hover").textContent = `depart ${isoDay(deps[i])}, TOF ${tofs[j].toFixed(0)} d: ${pc.objective} ${v === null ? "n/a" : sf4(v)}`;
  };
}

function clearPorkchop() {
  const cv = $("porkchop");
  cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  cv.onmousemove = null;
  $("porkchop-hover").textContent = "";
  $("porkchop-caption").textContent = "This mission has no launch-window search. Run find_launch_window to see a porkchop plot.";
}

function viridis(t) {
  const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(t), stops.length - 2), u = t - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---------------------------------------------------------------- tabs
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.id === "tab-" + b.dataset.tab));
}));

// ---------------------------------------------------------------- bridge
let ws = null, nextId = 1;
const pending = new Map();

function request(what, extra = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ type: "request", id, what, ...extra }));
  });
}

function handle(msg) {
  if (msg.type === "history") {
    let lastGeo = null;
    for (const ev of msg.events) {
      try { if (ev.type === "chat") addChat(ev); else { addLog(ev, false); if (ev.legs) lastGeo = ev; } }
      catch (e) { console.error("skipped a bad history event", e, ev); }
    }
    if (lastGeo) showMission(lastGeo);
  } else if (msg.type === "mission") {
    try { addLog(msg, !!msg.legs); } catch (e) { console.error("bad mission event", e, msg); }
  } else if (msg.type === "chat") {
    addChat(msg);
  } else if (msg.type === "response") {
    const p = pending.get(msg.id);
    if (p) { pending.delete(msg.id); msg.error ? p.reject(new Error(msg.error)) : p.resolve(msg.data); }
  } else if (msg.type === "cleared") {
    resetView();
  }
}

function resetView() {
  $("log").innerHTML = ""; $("crew").innerHTML = "";
  state.events = []; state.legs = []; state.selected = null;
  disposeGroup(missionGroup);
  craft.visible = false;
  clearPorkchop();
  $("porkchop-caption").textContent = "Run find_launch_window to see a porkchop plot.";
  $("mission-title").textContent = "No mission yet: ask the crew, or call a physics tool";
  legend();
}

function connect() {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  ws = new WebSocket(url);
  ws.onopen = async () => {
    $("status").textContent = "live"; $("status").className = "status on";
    resetView();
    ws.send(JSON.stringify({ type: "hello", role: "browser" }));
    try {
      drawOrbits(await request("planets"));
      if (!state.legs.length) {
        state.tracks = await request("tracks", { t0_mjd2000: state.t0, t1_mjd2000: state.t1 });
        setTime(state.t);
      }
    } catch (e) { console.error("bridge request failed", e); }
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    $("status").textContent = "offline"; $("status").className = "status off";
    for (const p of pending.values()) p.reject(new Error("bridge closed"));
    pending.clear();
    setTimeout(connect, 2000);
  };
}
legend();
setTime(state.t);
connect();
window.__mc = { state, scene }; // for debugging from the console
