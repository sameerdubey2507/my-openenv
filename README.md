<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/OpenEnv-Phase_1-00D4FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" />
</p>

# 🚑 EMERGI-ENV

## Emergency Medical Intelligence & Resource Governance Environment

> **National AI Hackathon · HuggingFace OpenEnv Framework · v1.4.0**

**EMERGI-ENV** is a production-grade reinforcement learning environment that simulates India's national **108/112 Emergency Medical Services (EMS) dispatch network** — the world's largest ambulance coordination system, processing **50,000+ emergency calls per day** across Indian states.

The core insight: **ambulance dispatch is a Partially-Observable Markov Decision Process (POMDP)**. Every second an AI agent delays a P1 patient dispatch, survival probability decays along a condition-specific curve. Standard rule-based heuristics fail under mass-casualty events, hospital saturation, and communication outages. Learned policies must generalise across all of these simultaneously.

---

## 🌍 What Makes This Real

EMERGI-ENV models a **Pune-inspired 12-zone metropolitan grid** with **8 hospitals**, a heterogeneous ambulance fleet (**BLS / ALS / Mobile ICU**), and a live incident queue driven by **200+ medically-accurate symptom templates**. It is built on the **OpenEnv framework** and passes the full Phase 1 automated validator.

| Metric | Value |
|--------|-------|
| Calls simulated per day | **50,000+** |
| City zones modelled | **12** (Pune-inspired) |
| Hospital facilities | **8** (with real specialties) |
| Ambulance types | **3** (BLS, ALS, MICU) |
| Incident templates | **200+** |
| Graded tasks | **9** across 3 tiers |
| Simulation features | **10** advanced modules |
| Survival curve conditions | **15** |

---

## 📐 Architecture

```
emergi-env/
├── inference.py                  ← LLM baseline agent (hybrid rule+LLM)
├── openenv.yaml                  ← OpenEnv metadata, 9 tasks, full schemas
├── Dockerfile                    ← python:3.11-slim, uvicorn entrypoint
├── requirements.txt              ← FastAPI, Pydantic, numpy, etc.
│
├── server/
│   ├── main.py                   ← FastAPI: /reset /step /grade /health /tasks
│   ├── env.py                    ← EmergiEnv class, episode lifecycle, step loop
│   │
│   ├── models/
│   │   ├── observation.py        ← Pydantic: incident_queue, fleet, hospitals, traffic
│   │   ├── action.py             ← Pydantic: dispatch, reroute, tag, transfer, noop...
│   │   ├── state.py              ← Full internal state (superset of obs + ground truth)
│   │   └── reward.py             ← Step reward, episode score, component breakdown
│   │
│   ├── simulation/
│   │   ├── incidentengine.py     ← 200+ templates, seeded RNG, severity distribution
│   │   ├── fleetsimulator.py     ← Ambulance movement, ETA, crew fatigue state machine
│   │   ├── hospitalnetwork.py    ← Capacity tracking, specialty matching, diversion
│   │   ├── trafficmodel.py       ← Time-varying 12×12 OD matrix, peak-hour penalties
│   │   ├── communication.py      ← Per-unit radio failure, last-known-pos cache
│   │   ├── multiagency.py        ← Police/Fire/EMS coordination, extrication prereqs
│   │   ├── demandforecaster.py   ← 12-hour zone heatmap, ±20% noise injection
│   │   └── mutualaid.py          ← Cross-zone sharing, 12-min delay, over-request penalty
│   │
│   ├── medical/
│   │   ├── triage.py             ← START protocol: RPM → Immediate/Delayed/Minimal/Expectant
│   │   ├── traumascoring.py      ← ISS and RTS computation from symptom descriptions
│   │   ├── survivalcurves.py     ← Per-condition decay (STEMI, polytrauma, stroke, burns...)
│   │   └── protocolchecker.py    ← EMS protocol compliance validation, +0.15 bonus
│   │
│   └── graders/
│       ├── basegrader.py         ← Abstract base: normalize, deterministic seed, validate
│       ├── taskgrader1.py        ← Easy: triage + unit type + hospital → [0.0-1.0]
│       ├── taskgrader2.py        ← Easy: specialty + capacity + travel time
│       ├── taskgrader3.py        ← Easy: exact unit type match
│       ├── taskgrader4.py        ← Medium: weighted survival across patients
│       ├── taskgrader5.py        ← Medium: reroute correctness + time saved
│       ├── taskgrader6.py        ← Medium: avg response time vs random baseline
│       ├── taskgrader7.py        ← Hard: START accuracy × response × hospital spread
│       ├── taskgrader8.py        ← Hard: transfer appropriateness + timing + utilisation
│       └── taskgrader9.py        ← Hard: system survival + cascade avoidance + mutual aid
│
├── data/
│   ├── hospitalprofiles.json     ← 8 hospitals: beds, ICU, specialties, cath lab, helipad
│   ├── templateincident.json     ← 200+ symptom templates with ground-truth severity
│   ├── trafficpattern.json       ← 24-hour × 12-zone OD matrices
│   ├── demandhistory.json        ← Historical incident density by zone × hour
│   ├── survivalparam.json        ← Curve params for 15 conditions
│   └── peopledataset.json        ← Patient and hospital dataset
│
├── static/                       ← React/Vite Command Nucleus dashboard
│   └── dist/                     ← Production build (served at localhost:7860)
│
├── docs/
│   ├── emergi_docs.html          ← Documentation hub with nav to all sub-pages
│   ├── emergi_openapi.html       ← Interactive OpenAPI endpoint explorer
│   ├── emergi_health.html        ← Patient health & clinical outcomes dashboard
│   ├── emergi_automation.html    ← System automation & business scope visualiser
│   └── emergi_login.html         ← Admin panel with telemetry & operator feedback
│
├── tests/
│   ├── test_graders.py           ← Determinism, score range [0,1], all 9 tasks
│   └── test_env.py               ← reset() state, step() schema, done flag logic
│
└── scripts/
    ├── validate-submission.sh    ← Official OpenEnv validator
    ├── run_baseline.sh           ← Runs inference.py against all 9 tasks
    └── generate_scenarios.py     ← Seed-based scenario generator
```

---

## 🎯 Task Suite — 9 Tasks Across 3 Difficulty Tiers

| # | Task | Difficulty | Max Steps | Baseline |
|---|------|-----------|-----------|----------|
| 1 | Single Call Triage & Dispatch | 🟢 Easy | 20 | 0.61 |
| 2 | Hospital Route Selection | 🟢 Easy | 20 | 0.72 |
| 3 | Unit Type Matching | 🟢 Easy | 10 | 0.68 |
| 4 | Multi-Incident Queue Management | 🟡 Medium | 50 | 0.44 |
| 5 | Dynamic Rerouting Under Traffic & Diversion | 🟡 Medium | 40 | 0.38 |
| 6 | Fleet Pre-Positioning | 🟡 Medium | 60 | 0.42 |
| 7 | Mass Casualty Incident — START Triage | 🔴 Hard | 80 | 0.29 |
| 8 | Inter-Hospital Transfer Cascade | 🔴 Hard | 70 | 0.24 |
| 9 | City-Wide Surge — Three Simultaneous MCIs | 🔴 Hard | 120 | 0.17 |

### Easy Tier (Tasks 1–3)

**Task 1 — Single Call Triage & Dispatch**
Agent receives one incident with a natural-language symptom description. Must classify severity (P1/P2/P3), select the correct ambulance type (BLS/ALS/MICU), and route to the appropriate specialty hospital. Grader: triage class (0.40) + unit type (0.30) + hospital match (0.30). Protocol compliance bonus up to +0.15.

**Task 2 — Hospital Route Selection**
Agent picks the optimal hospital given patient symptoms, location, and live hospital network state. Two hospitals are on diversion. Grader: specialty match (0.50) + capacity check (0.30) + travel time (0.20).

**Task 3 — Unit Type Matching**
Pure medical protocol knowledge test. Agent selects BLS/ALS/MICU for 10 scenarios drawn from the condition library. Binary exact-match grader per scenario, averaged.

### Medium Tier (Tasks 4–6)

**Task 4 — Multi-Incident Queue Management**
5–8 simultaneous calls with only 3 ambulances available. Agent must triage and prioritise. P1 patients carry 3× weight. Penalty −0.08 per step a P1 patient goes unaddressed.

**Task 5 — Dynamic Rerouting**
Mid-episode traffic spike and hospital diversion flip force the agent to reroute active units. Score = reroute correctness (0.50) + net time saved (0.50).

**Task 6 — Fleet Pre-Positioning**
No active incidents. Agent repositions fleet using a 12-hour demand forecast heatmap (±20% noise). Reward is entirely delayed — graded when incidents arrive at step 30.

### Hard Tier (Tasks 7–9)

**Task 7 — Mass Casualty Incident (MCI)**
20–40 simultaneous victims. Agent must apply START triage protocol (RPM scoring → Immediate/Delayed/Minimal/Expectant), dispatch correctly, spread across hospitals. Wrong tag on Immediate→Expectant = −0.50 penalty. Communication failures active at p=0.12.

**Task 8 — Inter-Hospital Transfer Cascade**
ICU patients need specialist care with expiring transfer windows. Agent manages transfers with bed utilisation constraints. Communication failures active.

**Task 9 — City-Wide Surge**
Three simultaneous MCIs. All hospitals near saturation. Communications failing. Agent must declare surge, request mutual aid, apply START triage, spread patients, and prevent cascade collapse. **Designed to score < 0.30 even for GPT-4-class models.**

---

## ⚙️ 10 Advanced Simulation Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Golden Hour Survival Curves** | Condition-specific survival probability decay (STEMI: steep cliff at 90min, polytrauma: linear at 60min). Reward integrates over survival probability. |
| 2 | **START Triage Protocol Engine** | Full RPM scoring (Respirations, Pulse, Mental Status) with deterministic ground truth per victim. |
| 3 | **Time-Varying Traffic Matrix** | Updates every 3 steps. Peak hours (8–10am, 5–7pm): 30–60% penalty. Active incidents spawn secondary slowdowns. |
| 4 | **Hospital Diversion Protocol** | ER occupancy ≥ 90% triggers diversion. Routing to diverted hospital: −0.30 penalty + 6-min forced redirect. |
| 5 | **Crew Fatigue State Machine** | 10h duty threshold → degradation (0.75× multiplier). Crew swap deploys in 8 sim-minutes. |
| 6 | **Multi-Agency Coordination** | Trapped victims need Police + Fire + EMS before treatment. Single EMS dispatch: −0.15 penalty. |
| 7 | **Communication Failure** | Tasks 7–9: p=0.12 per step per unit. Shows last-known-pos with staleness timestamp. |
| 8 | **Demand Forecast Heatmap** | 12-zone, 12-hour look-ahead with ±20% noise. Used by Task 6. |
| 9 | **Protocol Compliance Scoring** | Validates EMS rules (MICU for STEMI? Level-1 trauma for polytrauma?). Up to +0.15 bonus. |
| 10 | **Mutual Aid Cross-Zone** | Request ambulances from neighbours (12-min delay). Over-request: −0.10. Under-request in Task 9: cascade failure. |

---

## 🔬 Observation Space

Each `/reset` and `/step` returns a structured observation:

```json
{
  "episode_id": "uuid-string",
  "task_id": "task1_single_triage",
  "step": 0,
  "sim_clock_min": 0.0,
  "queue_length": 1,
  "active_patients": 0,
  "resolved_patients": 0,
  "surge_declared": false,
  "mci_active": false,
  "incident_queue": [...],
  "fleet_status": [...],
  "hospital_network": [...],
  "traffic_snapshot": {...},
  "demand_forecast": {...}
}
```

### Key Sub-Observations

| Component | Content |
|-----------|---------|
| `incident_queue` | Pending incidents with symptom descriptions, zone, severity hint, RPM snapshot (Tasks 7–9) |
| `fleet_status` | Unit ID, type (BLS/ALS/MICU), status, zone, fatigue flag, comm status, ETA |
| `hospital_network` | 8 hospitals: specialties, ER/ICU occupancy, diversion flags, cath lab, helipad |
| `traffic_snapshot` | 12×12 OD matrix, peak-hour flag, active incident slowdowns |
| `demand_forecast` | Zone-level 12-hour heatmap with noise (Task 6) |

---

## 🎮 Action Space

| Action Type | Required Fields | Key Use |
|-------------|----------------|---------|
| `dispatch` | `incident_id`, `unit_id`, `hospital_id` | Core dispatch action |
| `reroute` | `unit_id`, `new_hospital_id` | Redirect en-route units |
| `tag` | `incident_id`, `tag` | START triage (Tasks 7, 9) |
| `transfer` | `patient_id`, `dest_hospital_id` | Inter-hospital transfer (Task 8) |
| `preposition` | `unit_id`, `target_zone` | Fleet repositioning (Task 6) |
| `request_mutual_aid` | `n_units`, `from_zone`, `unit_type` | Cross-zone ambulance request |
| `declare_surge` | — | System-wide MCI declaration |
| `crew_swap` | `unit_id` | Fresh crew for fatigued unit |
| `escalate` | `incident_id`, `agencies` | Multi-agency coordination |
| `noop` | — | No operation this step |

---

## 💰 Reward Function

Reward fires **every step** (dense). No sparse terminal-only rewards.

| Component | Range | Trigger |
|-----------|-------|---------|
| Survival component | +0.0 to +1.0 | Scaled by patient survival probability at dispatch |
| Dispatch efficiency | +0.0 to +0.2 | Faster response → higher reward |
| Protocol compliance | +0.0 to +0.15 | Per-episode bonus for correct protocols |
| Surge bonus | +0.08 | Correct surge declaration |
| Diversion penalty | −0.30 | Routing to diverted hospital |
| P1 noop penalty | −0.08/step | P1 unaddressed while units available |
| Fatigue penalty | variable | Dispatching fatigued crew |
| Comms penalty | −0.10 | Noop on unit with comm loss |
| Mutual aid penalty | −0.10 | Over-requesting mutual aid |
| Cascade penalty | −0.35 | Hospital cascade collapse (Task 9) |

---

## 🗺️ City Zone Map (Pune-Inspired)

| Zone | Type | Real-World Analogue | Risk Profile |
|------|------|---------------------|-------------|
| zone_1 | metro_core | Shivajinagar | Highest incident density |
| zone_2 | metro_core | Deccan | Dense commercial |
| zone_3 | urban | Kothrud | Residential moderate |
| zone_4 | urban | Hadapsar | IT corridor, young workforce |
| zone_5 | urban | Viman Nagar | Airport proximity, trauma risk |
| zone_6 | suburban | Wakad | Growing suburb |
| zone_7 | suburban | Baner | Mixed residential/commercial |
| zone_8 | suburban | Kondhwa | Southern suburbs |
| zone_9 | industrial | Bhosari | Heavy industry, burn/crush |
| zone_10 | highway | Pune-Mumbai Expressway | High trauma rate |
| zone_11 | highway | Pune-Solapur Highway | Accident corridor |
| zone_12 | rural | Uruli Kanchan | Low density, long response |

---

## 🏥 Hospital Profiles

| ID | Name | Key Specialties | L1 Trauma | Cath Lab | Helipad |
|----|------|-----------------|-----------|----------|---------|
| H1 | Sassoon General Hospital | Cardiology, Neurology, Trauma | ✅ | ✅ | ✅ |
| H2 | Ruby Hall Clinic | Cardiology, Orthopaedics | ❌ | ✅ | ❌ |
| H3 | KEM Hospital | Obstetrics, Paediatrics | ❌ | ❌ | ❌ |
| H4 | Jehangir Hospital | Neurology, Neurosurgery | ❌ | ❌ | ❌ |
| H5 | Deenanath Mangeshkar | Cardiology, Oncology | ❌ | ✅ | ❌ |
| H6 | Aditya Birla Memorial | Burns, Trauma, Ortho | ✅ | ❌ | ✅ |
| H7 | Poona Hospital | Paediatrics, Obstetrics | ❌ | ❌ | ❌ |
| H8 | Sanjeevan Hospital | General Surgery, Ortho | ❌ | ❌ | ❌ |

---

## 🚀 Quickstart

### Docker (Recommended)

```bash
git clone https://github.com/sameerdubey2507/emergi-env.git
cd emergi-env

docker build -t emergi-env:latest .
docker run -d --name emergi-env -p 7860:7860 emergi-env:latest

# Verify
curl http://localhost:7860/health
```

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload
```

### Verify Installation

```bash
# Health check
curl http://localhost:7860/health

# Run a single episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_single_triage", "seed": 42}'
```

---

## 🤖 Running Inference

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export HF_TOKEN=your_token_here
export AGENT_MODE=hybrid

python inference.py
```

Expected output:
```
╔══════════════════════════════════════════════════════════════════════════╗
║              EMERGI-ENV — Baseline Inference Results                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Task                              │ Score  │ Baseline │ Delta  │ Status ║
╠══════════════════════════════════════════════════════════════════════════╣
║ task1_single_triage               │ 0.71   │ 0.61     │ +0.10  │  BEAT  ║
║ task2_hospital_route              │ 0.68   │ 0.72     │ -0.04  │  MISS  ║
║ task3_unit_type                   │ 0.74   │ 0.68     │ +0.06  │  BEAT  ║
║ task4_multi_incident              │ 0.41   │ 0.44     │ -0.03  │  MISS  ║
║ task5_dynamic_rerouting           │ 0.35   │ 0.38     │ -0.03  │  MISS  ║
║ task6_prepositioning              │ 0.44   │ 0.42     │ +0.02  │  BEAT  ║
║ task7_mci_start                   │ 0.22   │ 0.29     │ -0.07  │  MISS  ║
║ task8_transfer_cascade            │ 0.19   │ 0.24     │ -0.05  │  MISS  ║
║ task9_surge                       │ 0.13   │ 0.17     │ -0.04  │  MISS  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Aggregate Score                   │ 0.430  │ 0.439    │ -0.009 │        ║
║ Tasks Beating Baseline            │ 3 / 9  │                            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe — status, version, uptime |
| `POST` | `/reset` | Reinitialise all sim state, returns initial observation |
| `POST` | `/step` | Advance sim 1 timestep with agent action |
| `POST` | `/grade` | Run task grader, returns float in [0.0, 1.0] |
| `POST` | `/grade/batch` | Batch grade multiple tasks |
| `GET` | `/tasks` | List all 9 tasks with metadata |
| `GET` | `/state` | Full internal state (includes ground truth) |
| `POST` | `/validate` | OpenEnv Phase 1 validation checklist |
| `GET` | `/metrics` | Operational metrics |
| `GET` | `/docs` | Interactive documentation hub |

---

## 🖥️ Web Interfaces

EMERGI-ENV ships with a full-stack Command Nucleus dashboard:

| Route | Interface | Description |
|-------|-----------|-------------|
| `/` | **Command Nucleus** | React dashboard with live map, fleet tracking, incident matrix |
| `/docs` | **Documentation Hub** | Full API docs with navigation to all sub-pages |
| `/openapi-viewer` | **OpenAPI Explorer** | Interactive endpoint testing |
| `/health-dashboard` | **Clinical Outcomes** | Patient health telemetry & recovery data |
| `/automation` | **System Automation** | Workflow orchestrator, architecture & business scope |
| `/login` | **Admin Panel** | Operator authentication, telemetry, feedback system |

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Server port |
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-8B-Instruct` | LLM model for inference |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `HF_TOKEN` | — | HuggingFace API token |
| `AGENT_MODE` | `hybrid` | `llm` / `rule_based` / `hybrid` |
| `MAX_LLM_TOKENS` | `1000` | Max tokens per LLM call |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LOG_LEVEL` | `info` | Logging level |

---

## 📊 Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Step duration | 2 simulation minutes |
| Zones | 12 (Pune-inspired grid) |
| Hospitals | 8 |
| Fleet (default) | 4 BLS + 4 ALS + 4 MICU |
| Diversion threshold | ER occupancy ≥ 90% |
| Crew max duty | 10 hours |
| Crew swap delay | 8 sim-minutes |
| Mutual aid delay | 12 sim-minutes |
| Comm failure probability | p = 0.12 (Tasks 7–9) |
| Demand forecast horizon | 12 hours |
| Demand forecast noise | ±20% |
| Peak hour multiplier | 1.30× – 1.60× |
| Traffic update interval | Every 3 steps |

---

## 🧠 Technical Design Decisions

**Why POMDP?** Fleet positions are partially observable under communication failures. Hospital future capacity is uncertain. Demand forecasts carry ±20% noise. A purely observable MDP would be unrealistically easy.

**Why dense rewards?** Sparse terminal rewards make RL convergence impractically slow on 9 tasks with highly variable episode lengths. Dense per-step survival-probability integration gives meaningful gradient signal every step.

**Why 12-zone Pune grid?** Pune's mix of metro core, IT corridors, industrial zones, and rural outskirts creates natural diversity in zone types, travel times, and incident profiles — representative of Indian metro EMS without requiring proprietary geodata.

**Why condition-specific survival curves?** Binary "dispatched/not dispatched" rewards fail to capture clinically important distinctions. A 2-minute STEMI delay is far more costly than a 2-minute psychiatric case delay.

---

## 📝 Grader Specifications

All graders inherit from `BaseGrader` and guarantee:
- Output is always `float` in `[0.0, 1.0]` — never outside this range
- Deterministic: same `(task_id, seed, action_sequence)` → same score
- No external state dependencies

| Grader | Components | Max Bonus | Hardest Penalty |
|--------|-----------|-----------|-----------------|
| Task 1 | triage(0.4) + unit(0.3) + hospital(0.3) | +0.15 | −0.30 diversion |
| Task 2 | specialty(0.5) + capacity(0.3) + time(0.2) | +0.05 | −0.30 diversion |
| Task 3 | exact_match per scenario, averaged | — | −0.15 under-triage |
| Task 4 | survival(0.7) + efficiency(0.3) | — | −0.12 under-triage |
| Task 5 | reroute(0.5) + time_saved(0.5) | — | −0.30 diversion |
| Task 6 | response_time(0.7) + coverage(0.3) | — | −0.08 clustering |
| Task 7 | start_accuracy(0.45) + response(0.30) + spread(0.25) | +0.15 | −0.50 Imm→Exp |
| Task 8 | appropriateness(0.40) + timing(0.35) + utilisation(0.25) | +0.15 | −0.30 diversion |
| Task 9 | survival(0.40) + cascade(0.30) + mutual_aid(0.30) | +0.15 | −0.50 Imm→Exp |

---

## 📦 Codebase Statistics

| Module | Files | Total Size | Description |
|--------|-------|-----------|-------------|
| `server/models/` | 5 | ~379 KB | Pydantic data models |
| `server/simulation/` | 9 | ~549 KB | Simulation engine modules |
| `server/medical/` | 5 | ~574 KB | Medical protocol & triage |
| `server/graders/` | 11 | ~644 KB | Task-specific grading logic |
| `server/` (core) | 3 | ~143 KB | FastAPI server, env, init |
| `data/` | 6 | ~1.5 MB | JSON configuration datasets |
| `static/` | — | ~828 KB | React Command Nucleus UI |
| `docs/` | 5 | — | HTML documentation pages |
| **Total** | **~50+** | **~4.6 MB** | Production-grade RL environment |

---

## 📄 Citation

```bibtex
@software{emergienv2026,
  title     = {EMERGI-ENV: Emergency Medical Intelligence & Resource Governance Environment},
  author    = {Sameer Dubey},
  year      = {2026},
  version   = {1.4.0},
  url       = {https://huggingface.co/spaces/emergi-env/emergi-env},
  license   = {Apache-2.0}
}
```

---

## 📜 License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## 👤 Contact

**Author:** Sameer Dubey
**Email:** craftedlegend25@gmail.com
**HuggingFace Space:** [huggingface.co/spaces/emergi-env/emergi-env](https://huggingface.co/spaces/emergi-env/emergi-env)
