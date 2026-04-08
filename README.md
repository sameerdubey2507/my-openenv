---
title: "EMERGI-ENV: Command Nucleus"
emoji: "🚑"
colorFrom: "red"
colorTo: "gray"
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---

# 🚑 EMERGI-ENV: Command Nucleus
### *Emergency Medical Intelligence & Resource Governance Environment*

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/sameerdubey2507/emergi-env)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

**EMERGI-ENV** is a high-fidelity Reinforcement Learning environment (POMDP) that simulates India's 108/112 ambulance dispatch network. It challenges AI agents to manage a fleet of medical units across a Pune-inspired city grid, prioritizing patients under resource scarcity to maximize survival probability.

---

## 🌐 Live Command Center
Access the production deployment and tactical overwatch dashboard here:
👉 **[https://huggingface.co/spaces/sameerdubey2507/emergi-env](https://huggingface.co/spaces/sameerdubey2507/emergi-env)**

---

## 🛰️ System Features

### 1. Tactical Command Nucleus (Landing Page)
The primary overwatch interface for real-time fleet management.
*   **Live Map Visualization**: Track ALS, BLS, and MICU units as they navigate urban traffic.
*   **Incident Queue**: Real-time triage of incoming cardiac, trauma, and respiratory calls.
*   **Rapid Dispatch**: One-click manual override or AI-driven unit assignment.

### 2. Trauma Flow & Medical Intelligence (`/health-dashboard`)
Advanced clinical monitoring to track if the AI is making "medically sound" decisions.
*   **Golden Hour Tracker**: Measures survival probability decay for P1 critical patients.
*   **Protocol Compliance**: Real-time scoring of medical protocols (e.g., Oxygen Admin, Tourniquet usage).
*   **Hospital Matrix**: Live diversion status of 8 major hospitals (Tier 1 Trauma to Cardiology).

### 3. Emergi-Env Automation (`/automation`)
Visual representation of the environment's internal RL logic.
*   **START Triage Engine**: Visualizing the Simple Triage and Rapid Treatment (START) algorithm.
*   **Demand Forecast Heatmaps**: Predicting where incidents will occur in the next 12 hours.
*   **System Architecture**: Deep dive into the YAML-driven simulation nodes.

---

### 🧪 Interactive API Sandbox (`/docs`)
Found under the **"Functions"** button in the dashboard, this section allows for real-time interaction without any coding.
*   **Command Interface**: Select an endpoint, fill the JSON body, and hit "Send Request."
*   **Available Commands**:
    *   `POST /reset`: Start a new medical simulation episode.
    *   `POST /step`: Dispatch units or perform medical actions.
    *   `GET /state`: View ground-truth telemetry and hidden variables.
    *   `GET /tasks`: Explore the 9 graded hackathon tasks.
    *   `POST /grade`: Manually trigger the agent grading pipeline.
    *   `GET /health`: Monitor server performance and uptime.
    *   `WS /ws/{id}`: Connect a live WebSocket for real-time telemetry streaming.

---

## 🎮 Developer Quick-Start (CLI)
You can also control the environment directly from your terminal.

**1. Reset the Environment (Start Episode)**
```powershell
Invoke-RestMethod -Uri "https://sameerdubey2507-emergi-env.hf.space/reset" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"task_id": "task1_single_triage", "seed": 42}'
```

**2. Query hidden Ground-Truth (State)**
```powershell
Invoke-RestMethod -Uri "https://sameerdubey2507-emergi-env.hf.space/state?session_id=default" `
  -Method Get
```

**3. Send a Dispatch Action (Step)**
```powershell
Invoke-RestMethod -Uri "https://sameerdubey2507-emergi-env.hf.space/step" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "action": {
        "action_type": "dispatch",
        "unit_id": "AMB-001",
        "incident_id": "INC-001",
        "hospital_id": "HOSP-001"
    }
}'
```

---

## 🎖️ Evaluation & Grading System
The environment features 9 calibrated tasks designed to test different aspects of medical resource management.

### Task Tiers
| Tier | Tasks | Grader Logic |
| :--- | :--- | :--- |
| **🟢 EASY** | T1 - T3 | Focuses on single-call triage and exact unit-type matching. |
| **🟡 MEDIUM** | T4 - T6 | Focuses on multi-incident queues and fleet pre-positioning. |
| **🔴 HARD** | T7 - T9 | Massive Casualty Incidents (MCI) and system-wide surge management. |

### Sample Grader Output (`POST /grade`)
When an agent completes a task, the grader returns a detailed performance breakdown:
```json
{
  "task_id": "task1_single_triage",
  "episode_score": 0.842,
  "components": {
    "triage_accuracy": 1.0,
    "response_time_penalty": -0.05,
    "protocol_bonus": 0.15,
    "hospital_matching": 0.92
  },
  "summary": "Agent successfully triaged incident with minimal protocol deviation."
}
```

---

---

## 📋 Hackathon Summary
*   **City Grid**: 12 zones, 8 hospitals, 50+ incident templates.
*   **AI Goal**: Learn to triage, dispatch, and reroute under dynamic traffic and hospital saturation.
*   **Grading**: 9 calibrated tasks (Easy → Hard) with dense reward signals.
*   **Stack**: FastAPI (Backend) | React-Vite (Frontend) | OpenEnv (Framework).

---

## 🛠️ Dashboard & Functional Guide

### Tactical Modules
| Module | Use Case |
| :--- | :--- |
| **Command Center** | Tactical map view of the city grid and live fleet positions. |
| **Grid Routing** | Visualizes dynamic pathfinding and traffic impact. |
| **Rapid Dispatch** | Real-time incident triage and unit assignment queue. |
| **Trauma Flow** | Clinical monitoring of "Golden Hour" survival probabilities. |
| **System Analytics** | Live performance telemetry and reward accumulation. |

### 🏅 Reward & Task Logic
The simulation uses **Dense Rewards**, ensuring continuous feedback for every agent decision.

#### Scoring Formula:
The environment calculates rewards based on:
1. **Survival Probability**: $P_{survival}$ decay across the "Golden Hour".
2. **Protocol Compliance**: Adherence to START triage and medical guidelines.
3. **Operational Efficiency**: Minimizing travel time and hospital wait-times.

#### Task Tiers (9 Evaluated Scenarios):
*   **Easy**: Single call triage, Basic unit matching.
*   **Medium**: Queue prioritization, Dynamic hospital rerouting.
*   **Hard**: Mass Casualty Incidents (MCI), System-wide resource surges.

---

> [!TIP]
> Use the **"Functions"** tab in the dashboard header to access technical sub-pages for API testing and clinical validation.

Developed for the **Hugging Face OpenEnv Hackathon**.
