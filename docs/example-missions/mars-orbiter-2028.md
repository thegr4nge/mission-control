# Mission log

**Goal:** Send an orbiter with 1500 kg dry mass, 4000 kg of propellant and a 320 s engine from Earth to Mars in the next launch window. Can it make it, and what is the best route?

Model groq:openai/gpt-oss-120b · started 2026-11-22T14:05:19+00:00 · 96 s · status **grounded** · 1 report(s) sent back for unverified numbers

## Briefing

**Navigator Brief – Earth → Mars Orbiter (Physicist)**  

**Mission baseline** – dry mass = 1500 kg, propellant = 4000 kg, engine Isp = 320 s.  

**Best direct transfer (patched‑conic, impulsive burns)**  
- **Departure:** 2028‑11‑23T16:02 UTC from a 200 km circular Earth parking orbit  
- **Arrival:** 2029‑09‑19T18:29 UTC into a 300 km circular Mars orbit  
- **Time‑of‑flight:** 300.1 days (trajectory type II, transfer angle 212.8°)  
- **C3:** 9.118 km² s⁻²  
- **Δv departure:** 3.631 km s⁻¹ (vinf out 3.02 km s⁻¹)  
- **Δv capture:** 2.253 km s⁻¹ (vinf in 2.97 km s⁻¹)  
- **Total Δv:** 5.884 km s⁻¹  

**Alternative: Venus‑gravity‑assist sequence**  
- **Departure:** 2028‑09‑14T00:00 UTC, Δv departure 7.582 km s⁻¹ (vinf out 10.72 km s⁻¹)  
- **Venus flyby:** 2029‑08‑15T06:17 UTC, Δv ≈ 0 (vinf in 6.721 km s⁻¹, vinf out 6.721 km s⁻¹)  
- **Arrival:** 2030‑06‑13T17:28 UTC, Δv capture 6.813 km s⁻¹ (vinf in 9.011 km s⁻¹)  
- **Time‑of‑flight:** 637.7 days, **C3:** 114.9 km² s⁻², **Total Δv:** 14.39 km s⁻¹  

**Why the direct route wins** – It has the lowest C3 (9.118 km² s⁻²) and the smallest total Δv (5.884 km s⁻¹). The Venus option requires far more energy (C3 114.9 km² s⁻², total Δv 14.39 km s⁻¹) and a much longer flight, offering no Δv advantage.

**Can the spacecraft fly the direct transfer?**  
Δv budget check (10 % margin) gives:  
- **Required Δv with margin:** 6.472 km s⁻¹  
- **Available Δv:** 4.077 km s⁻¹  
- **Surplus:** –2.395 km s⁻¹  
- **Propellant needed:** 10 300 kg vs 4000 kg available  

Result: **Not feasible** with the current mass and Isp. The Venus‑assist check shows an even larger shortfall (surplus –11.76 km s⁻¹, propellant needed 231 600 kg).  

**Model limits** – All trajectories are computed with patched‑conic approximations and assume instantaneous (impulsive) burns. Real‑world gravity‑assist dynamics, finite‑burn arcs, and spacecraft attitude constraints are not captured.  

**Bottom line:** The direct Type II Earth‑to‑Mars transfer is the optimal path, but the vehicle’s Δv capability is insufficient. A lighter spacecraft, higher‑Isp propulsion, or aerocapture at Mars would be required to meet the mission goal.

## Flight director

**Navigator Brief – Earth → Mars Orbiter**

- **Origin / Destination:** Earth departure to Mars orbit insertion.  
- **Launch Timing:** Run `next_launch_opportunities` to locate the next viable window after the current date.  
- **Mission Objective:** Verify that the spacecraft (dry mass = 1500 kg, propellant = 4000 kg, engine Isp = 320 s) can reach Mars orbit and identify the most efficient trajectory.

**Optimization Priorities**  
1. Maximize propellant margin after Mars orbit insertion.  
2. Keep the time‑of‑flight within operational limits.

**Alternatives to Compare**  
- **Type I (Hohmann) transfer** vs. **Type II (longer‑duration) transfer**.  
- **Direct transfer** vs. a **Venus gravity‑assist** trajectory that could lower ΔV at the expense of added flight time.

**Propulsion Check:** Send any candidate ΔV budgets and burn sequences to the propulsion engineer for validation against the 1500 kg dry mass, 4000 kg propellant, and 320 s Isp engine.

## Navigator

**Next launch window (Earth → Mars, direct)**  
- **Departure:** 2028‑11‑23 T16:02 UTC (200 km circular Earth parking orbit)  
- **Arrival:** 2029‑09‑19 T18:29 UTC (300 km circular Mars orbit)  
- **Time‑of‑flight:** 300.1 days  
- **Trajectory type:** II (transfer angle 212.8°)  
- **C3:** 9.118 km² s⁻²  
- **Departure v∞:** 3.02 km s⁻¹ (Δv departure = 3.631 km s⁻¹)  
- **Arrival v∞:** 2.97 km s⁻¹ (Δv capture = 2.253 km s⁻¹)  
- **Total Δv:** 5.884 km s⁻¹  

**Venus‑gravity‑assist alternative**  
- **Sequence:** Earth → Venus → Mars (capture)  
- **Departure:** 2028‑09‑14 T00:00 UTC (Δv departure = 7.582 km s⁻¹)  
- **Venus flyby:** 2029‑08‑15 T06:17 UTC (Δv ≈ 0)  
- **Arrival:** 2030‑06‑13 T17:28 UTC (Δv capture = 6.813 km s⁻¹)  
- **Time‑of‑flight:** 637.7 days  
- **C3:** 114.9 km² s⁻²  
- **Total Δv:** 14.39 km s⁻¹  

**Propulsion check** – With a 1500 kg dry mass, 4000 kg propellant and a 320 s engine, the spacecraft cannot supply the 5.884 km s⁻¹ total Δv required for the direct capture trajectory; the available propellant would be exhausted before Mars orbit insertion. The Venus‑assist option is even less feasible (Δv 14.39 km s⁻¹).

**Recommendation** – The direct Earth‑to‑Mars Type II transfer is the most efficient option (lowest Δv and C3). However, the vehicle’s propellant budget is insufficient for a circular‑orbit insertion at Mars; a mission concept using aerocapture or a lighter spacecraft would be required.

- `next_launch_opportunities` ok: `{"after_date":"2026-11-22","count":2,"destination":"mars","origin":"earth"}`
- `find_launch_window` ok: `{"destination":"mars","earliest_departure":"2028-09-14","latest_departure":"2029-02-17","max_time_of_flight_days":453,"min_time_of_flight_days":90,"objective":"dv_total","origin":"earth"}`
- `propose_flyby_sequence` ok: `{"arrival":"capture","capture_orbit_altitude_km":300,"departure_orbit_altitude_km":200,"destination":"mars","earliest_departure":"2028-09-14","flyby_candidates":["venus"],"latest_departure":"2029-02-17","max_flybys":1,"objective":"dv_total","origin":"earth"}`

## Propulsion

**Propulsion budget check (10 % margin)**  

- **Direct Earth‑to‑Mars (departure Δv 3.631 km s⁻¹, capture Δv 2.253 km s⁻¹)** – required Δv with margin 6.472 km s⁻¹; spacecraft can deliver 4.077 km s⁻¹. Surplus ‑2.395 km s⁻¹. Propellant needed 10 300 kg vs 4 000 kg available. **Not feasible**.  

- **Venus‑gravity‑assist (departure Δv 7.582 km s⁻¹, capture Δv 6.813 km s⁻¹)** – required Δv with margin 15.83 km s⁻¹; spacecraft can deliver 4.077 km s⁻¹. Surplus ‑11.76 km s⁻¹. Propellant needed 231 600 kg vs 4 000 kg available. **Not feasible**.  

Both options exceed the vehicle’s Δv capability; a lighter spacecraft, higher‑Isp engine, or aerocapture at Mars would be required.

- `check_delta_v_budget` ok: `{"dry_mass_kg":1500,"isp_s":320,"maneuvers_km_s":{"capture":2.253,"departure":3.631},"margin_percent":10,"propellant_mass_kg":4000}`
- `check_delta_v_budget` ok: `{"dry_mass_kg":1500,"isp_s":320,"maneuvers_km_s":{"capture":6.813,"departure":7.582},"margin_percent":10,"propellant_mass_kg":4000}`

---
Every number above was checked against the physics tool results (pykep); full tool outputs are in the JSON file beside this one.
