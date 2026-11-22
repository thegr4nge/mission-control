# Mission Control

A browser 3D solar system on real physics where a crew of AI agents designs and flies space missions from a plain English goal, then briefs the user on what they tried and why the final route won. Built for Edan first (physics honours graduate, astrophysics postgrad at ANU), tuned toward professional mission designers later. North star: helping humanity get off planet.

## Hard rules
- Every number an agent states must come from a physics tool call. Agents orchestrate, pykep computes. Never let a model invent a delta v, date or orbit.
- Physics is verified against published reference cases and pykep examples before any demo. Quote physics results to four significant figures with units.
- No paid AI API spend. Edan drives the tools from the Claude desktop app on his Pro plan. The built in crew uses free tiers (Gemini via Google AI Studio by default, Groq free as fallback, Ollama local as last resort) behind one model adapter.
- Repo will be public under Apache-2.0. Never commit API keys; use a .env file listed in .gitignore.

## Architecture
- physics/: Python on pykep v3 (ESA Advanced Concepts Team). Lambert, propagation, flyby sequences, porkchop grids.
- mcp_server/: exposes the physics as MCP tools (find_launch_window, solve_transfer, propagate, check_delta_v_budget, propose_flyby_sequence). Pushes mission state to the browser over a local websocket.
- web/: three.js 3D scene, time slider, porkchop panel, crew chat panel, mission log.
- agents/: crew roles (flight director, navigator, propulsion) and the model adapter.
- evals/: mission tasks with known answers used to score models.

## v0 build order (target Sunday 29 Nov 2026)
1. Install pykep v3 in a venv and reproduce one Earth to Mars Lambert transfer against a known reference. This proves the stack. Do this first.
2. MCP physics server with the core tools, plus tests against reference transfers.
3. Eval: 10 mission tasks with known answers (Earth to Mars windows, Hohmann case, a Venus flyby). Run Claude via the desktop app, Gemini free and Groq free. Score tool call success, answer correctness and explanation quality.
4. 3D front end plus websocket bridge.
5. Crew agents and a 90 second demo video.

## After v0
- v1: gravity assists, outer planets, mission log briefing.
- v2: full crew, scoring against one past GTOC problem.
- v3: professional features (export, saved studies, team use).

## Environment
- Runs in WSL Ubuntu on Edan's laptop (Ryzen 7, 32GB RAM, no discrete GPU). Python 3.12.
- Edan wants momentum: small verified steps, commit after each working step, keep explanations short.