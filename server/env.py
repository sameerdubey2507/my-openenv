from __future__ import annotations
import random
import uuid
import math
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
ENV_VERSION: str = "2.0.0"
MINUTES_PER_STEP: float = 3.0  
TASK_IDS: Set[str] = {
    "task1_single_triage",
    "task2_hospital_route",
    "task3_unit_type",
    "task4_multi_incident",
    "task5_dynamic_rerouting",
    "task6_prepositioning",
    "task7_mci_start",
    "task8_transfer_cascade",
    "task9_surge",
}
DIFFICULTY_MAP: Dict[str, str] = {
    "task1_single_triage":    "easy",
    "task2_hospital_route":   "easy",
    "task3_unit_type":        "easy",
    "task4_multi_incident":   "medium",
    "task5_dynamic_rerouting":"medium",
    "task6_prepositioning":   "medium",
    "task7_mci_start":        "hard",
    "task8_transfer_cascade": "hard",
    "task9_surge":            "hard",
}
MAX_STEPS_BY_TASK: Dict[str, int] = {
    "task1_single_triage":    15,
    "task2_hospital_route":   15,
    "task3_unit_type":        10,
    "task4_multi_incident":   30,
    "task5_dynamic_rerouting":25,
    "task6_prepositioning":   20,
    "task7_mci_start":        50,
    "task8_transfer_cascade": 40,
    "task9_surge":            60,
}
ZONES = [f"Z{i}" for i in range(1, 13)]  
HOSPITALS = [
    {"id": "H1", "name": "Sassoon General",       "zone": "Z3",  "beds": 800, "icu": 60,  "specialties": ["trauma","cardiology","neurology"],   "helipad": True},
    {"id": "H2", "name": "KEM Hospital",           "zone": "Z5",  "beds": 600, "icu": 40,  "specialties": ["trauma","ortho","surgery"],           "helipad": False},
    {"id": "H3", "name": "Ruby Hall Clinic",       "zone": "Z6",  "beds": 400, "icu": 50,  "specialties": ["cardiology","cath_lab","stroke"],     "helipad": False},
    {"id": "H4", "name": "Jehangir Hospital",      "zone": "Z4",  "beds": 350, "icu": 30,  "specialties": ["general","burns","pediatrics"],       "helipad": False},
    {"id": "H5", "name": "Deenanath Mangeshkar",   "zone": "Z7",  "beds": 450, "icu": 45,  "specialties": ["cardiology","neurology","oncology"],  "helipad": True},
    {"id": "H6", "name": "Bharati Hospital",       "zone": "Z9",  "beds": 300, "icu": 25,  "specialties": ["general","trauma","pediatrics"],      "helipad": False},
    {"id": "H7", "name": "Aditya Birla Memorial",  "zone": "Z10", "beds": 500, "icu": 55,  "specialties": ["cardiology","transplant","stroke"],   "helipad": True},
    {"id": "H8", "name": "Columbia Asia",          "zone": "Z11", "beds": 250, "icu": 20,  "specialties": ["general","ortho","burns"],            "helipad": False},
]
HOSPITAL_IDS = [h["id"] for h in HOSPITALS]
ZONE_ADJACENCY: Dict[str, List[str]] = {
    "Z1":  ["Z2","Z4"],
    "Z2":  ["Z1","Z3","Z5"],
    "Z3":  ["Z2","Z4","Z6"],
    "Z4":  ["Z1","Z3","Z7"],
    "Z5":  ["Z2","Z6","Z8"],
    "Z6":  ["Z3","Z5","Z9"],
    "Z7":  ["Z4","Z8","Z10"],
    "Z8":  ["Z5","Z7","Z11"],
    "Z9":  ["Z6","Z10","Z12"],
    "Z10": ["Z7","Z9","Z11"],
    "Z11": ["Z8","Z10","Z12"],
    "Z12": ["Z9","Z11"],
}
def _base_travel_time(z1: str, z2: str) -> float:
    if z1 == z2:
        return 2.0
    visited = {z1}
    queue = [(z1, 0)]
    while queue:
        cur, dist = queue.pop(0)
        for nb in ZONE_ADJACENCY.get(cur, []):
            if nb == z2:
                return (dist + 1) * 5.0
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return 30.0  
INCIDENT_TEMPLATES = [
    {"condition": "STEMI",          "severity": "P1", "unit": "MICU",  "specialty": "cardiology",  "description": "Chest pain, ST elevation on ECG, diaphoresis"},
    {"condition": "Stroke",         "severity": "P1", "unit": "MICU",  "specialty": "neurology",   "description": "Sudden facial droop, arm weakness, slurred speech"},
    {"condition": "Polytrauma",     "severity": "P1", "unit": "ALS",   "specialty": "trauma",      "description": "MVA victim, multiple injuries, BP 80/50"},
    {"condition": "Burns",          "severity": "P1", "unit": "ALS",   "specialty": "burns",       "description": "House fire, 40% BSA burns, airway compromise"},
    {"condition": "Cardiac_Arrest", "severity": "P1", "unit": "MICU",  "specialty": "cardiology",  "description": "Unresponsive, no pulse, bystander CPR in progress"},
    {"condition": "Respiratory",    "severity": "P2", "unit": "ALS",   "specialty": "general",     "description": "Severe asthma attack, SpO2 82%, accessory muscles"},
    {"condition": "Fracture",       "severity": "P2", "unit": "BLS",   "specialty": "ortho",       "description": "Closed femur fracture, moderate pain, stable vitals"},
    {"condition": "Abdominal",      "severity": "P2", "unit": "ALS",   "specialty": "surgery",     "description": "Acute abdomen, guarding, rebound tenderness"},
    {"condition": "Pediatric",      "severity": "P2", "unit": "ALS",   "specialty": "pediatrics",  "description": "Febrile seizure in 3-year-old, postictal"},
    {"condition": "Minor_Laceration","severity":"P3", "unit": "BLS",   "specialty": "general",     "description": "Laceration to hand, controlled bleeding, ambulatory"},
    {"condition": "Fall",           "severity": "P3", "unit": "BLS",   "specialty": "general",     "description": "Elderly fall, bruising, no LOC, vitals stable"},
    {"condition": "Diabetic",       "severity": "P2", "unit": "BLS",   "specialty": "general",     "description": "Hypoglycaemia, confused, BG 2.1 mmol/L"},
    {"condition": "Trapped",        "severity": "P1", "unit": "ALS",   "specialty": "trauma",      "description": "Vehicle extrication required, mechanism of injury severe"},
    {"condition": "Poisoning",      "severity": "P2", "unit": "ALS",   "specialty": "general",     "description": "Suspected drug overdose, altered consciousness"},
    {"condition": "Obstetric",      "severity": "P2", "unit": "ALS",   "specialty": "general",     "description": "30-week pregnancy, heavy bleeding, contractions"},
]
UNIT_TYPES = ["BLS", "ALS", "MICU"]
SEVERITY_LEVELS = ["P1", "P2", "P3"]
TRIAGE_TAGS = ["Immediate", "Delayed", "Minimal", "Expectant"]
SURVIVAL_PARAMS = {
    "STEMI":          {"base": 0.97, "decay_per_min": 0.006, "golden_hour": 90},
    "Stroke":         {"base": 0.95, "decay_per_min": 0.004, "golden_hour": 60},
    "Polytrauma":     {"base": 0.92, "decay_per_min": 0.005, "golden_hour": 60},
    "Cardiac_Arrest": {"base": 0.90, "decay_per_min": 0.012, "golden_hour": 20},
    "Burns":          {"base": 0.88, "decay_per_min": 0.003, "golden_hour": 120},
    "Respiratory":    {"base": 0.95, "decay_per_min": 0.003, "golden_hour": 90},
    "Trapped":        {"base": 0.90, "decay_per_min": 0.004, "golden_hour": 60},
    "default":        {"base": 0.99, "decay_per_min": 0.001, "golden_hour": 180},
}
def _survival_prob(condition: str, elapsed_min: float) -> float:
    p = SURVIVAL_PARAMS.get(condition, SURVIVAL_PARAMS["default"])
    decay = min(elapsed_min * p["decay_per_min"], 0.95)
    golden_penalty = 0.15 if elapsed_min > p["golden_hour"] else 0.0
    return max(0.05, p["base"] - decay - golden_penalty)
def _start_triage(respirations: int, pulse: int, mental_status: str) -> str:
    if respirations == 0:
        return "Expectant"
    if respirations > 30 or respirations < 10:
        return "Immediate"
    if pulse > 120 or pulse < 50:
        return "Immediate"
    if mental_status in ("unresponsive", "posturing"):
        return "Immediate"
    if mental_status == "confused":
        return "Delayed"
    if pulse == 0:
        return "Expectant"
    return "Minimal"
def _make_incident(rng: random.Random, incident_id: str, zone: Optional[str] = None) -> Dict[str, Any]:
    tpl = rng.choice(INCIDENT_TEMPLATES)
    z = zone or rng.choice(ZONES)
    return {
        "incident_id": incident_id,
        "zone": z,
        "condition": tpl["condition"],
        "severity": tpl["severity"],
        "correct_unit": tpl["unit"],
        "required_specialty": tpl["specialty"],
        "description": tpl["description"],
        "elapsed_min": 0.0,
        "dispatched": False,
        "resolved": False,
        "unit_assigned": None,
        "hospital_assigned": None,
        "arrival_time": None,
        "survival_prob": _survival_prob(tpl["condition"], 0.0),
        "respirations": rng.randint(10, 28),
        "pulse": rng.randint(60, 100),
        "mental_status": rng.choice(["alert", "alert", "alert", "confused"]),
        "triage_tag": None,
        "requires_police": tpl["condition"] == "Trapped",
        "requires_fire": tpl["condition"] == "Trapped",
    }
class EmergiEnv:
    def __init__(self) -> None:
        self._rng = random.Random(42)
        self._task_id: str = "task1_single_triage"
        self._seed: int = 42
        self._step_count: int = 0
        self._sim_clock_min: float = 0.0
        self._episode_reward: float = 0.0
        self._done: bool = False
        self._episode_id: str = str(uuid.uuid4())
        self._incident_queue: List[Dict[str, Any]] = []
        self._active_incidents: Dict[str, Dict[str, Any]] = {}
        self._resolved: List[Dict[str, Any]] = []
        self._fleet: List[Dict[str, Any]] = []
        self._hospitals: List[Dict[str, Any]] = deepcopy(HOSPITALS)
        for h in self._hospitals:
            h["available_beds"] = h["beds"]
            h["er_load"] = 0.0
            h["diverted"] = False
        self._traffic_mult: float = 1.0
        self._mutual_aid_pending: int = 0
        self._mutual_aid_over_requests: int = 0
        self._comm_failures: Dict[str, bool] = {}
        self._protocol_violations: int = 0
        self._protocol_compliant: int = 0
    @property
    def episode_id(self) -> str:
        return self._episode_id
    @property
    def task_id(self) -> str:
        return self._task_id
    @property
    def step_count(self) -> int:
        return self._step_count
    @property
    def is_done(self) -> bool:
        return self._done
    @property
    def incident_queue_length(self) -> int:
        return len(self._incident_queue)
    @property
    def active_patient_count(self) -> int:
        return len(self._active_incidents)
    def reset(
        self,
        task_id: str = "task1_single_triage",
        seed: Optional[int] = None,
        scenario_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._task_id = task_id
        self._seed = seed if seed is not None else 42
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._sim_clock_min = 0.0
        self._episode_reward = 0.0
        self._done = False
        self._episode_id = str(uuid.uuid4())
        self._active_incidents = {}
        self._resolved = []
        self._mutual_aid_pending = 0
        self._mutual_aid_over_requests = 0
        self._protocol_violations = 0
        self._protocol_compliant = 0
        self._comm_failures = {}
        self._hospitals = deepcopy(HOSPITALS)
        for h in self._hospitals:
            h["available_beds"] = h["beds"]
            h["er_load"] = 0.0
            h["diverted"] = False
        self._fleet = self._init_fleet(task_id)
        self._incident_queue = self._init_incidents(task_id)
        hour = (self._seed % 24)
        self._traffic_mult = self._traffic_for_hour(hour)
        if scenario_override:
            self._apply_override(scenario_override)
        return self._build_observation()
    def _init_fleet(self, task_id: str) -> List[Dict[str, Any]]:
        difficulty = DIFFICULTY_MAP.get(task_id, "easy")
        if difficulty == "easy":
            counts = {"BLS": 3, "ALS": 3, "MICU": 2}
        elif difficulty == "medium":
            counts = {"BLS": 4, "ALS": 4, "MICU": 2}
        else:
            counts = {"BLS": 6, "ALS": 5, "MICU": 3}
        fleet = []
        uid = 1
        for utype, cnt in counts.items():
            for i in range(cnt):
                zone = self._rng.choice(ZONES)
                fleet.append({
                    "unit_id": f"U{uid:03d}",
                    "unit_type": utype,
                    "zone": zone,
                    "status": "available",  
                    "hours_on_duty": round(self._rng.uniform(0, 6), 1),
                    "incident_id": None,
                    "destination_hospital": None,
                    "eta_min": 0.0,
                    "last_known_zone": zone,
                    "comm_ok": True,
                })
                uid += 1
        return fleet
    def _init_incidents(self, task_id: str) -> List[Dict[str, Any]]:
        incidents = []
        iid = 1
        if task_id == "task1_single_triage":
            incidents.append(_make_incident(self._rng, f"INC{iid:04d}"))
        elif task_id == "task2_hospital_route":
            incidents.append(_make_incident(self._rng, f"INC{iid:04d}"))
        elif task_id == "task3_unit_type":
            incidents.append(_make_incident(self._rng, f"INC{iid:04d}"))
        elif task_id == "task4_multi_incident":
            n = self._rng.randint(5, 8)
            for _ in range(n):
                incidents.append(_make_incident(self._rng, f"INC{iid:04d}"))
                iid += 1
        elif task_id == "task5_dynamic_rerouting":
            n = self._rng.randint(3, 5)
            for _ in range(n):
                incidents.append(_make_incident(self._rng, f"INC{iid:04d}"))
                iid += 1
        elif task_id == "task6_prepositioning":
            pass
        elif task_id == "task7_mci_start":
            n = self._rng.randint(20, 40)
            for _ in range(n):
                inc = _make_incident(self._rng, f"INC{iid:04d}", zone="Z3")
                inc["respirations"] = self._rng.randint(0, 35)
                inc["pulse"] = self._rng.randint(0, 140)
                inc["mental_status"] = self._rng.choice(
                    ["alert","confused","unresponsive","posturing","alert","alert"]
                )
                inc["triage_tag"] = None
                incidents.append(inc)
                iid += 1
        elif task_id == "task8_transfer_cascade":
            n = self._rng.randint(5, 10)
            for _ in range(n):
                inc = _make_incident(self._rng, f"INC{iid:04d}")
                inc["needs_transfer"] = True
                inc["transfer_specialty"] = self._rng.choice(["cardiology","neurology","transplant"])
                incidents.append(inc)
                iid += 1
        elif task_id == "task9_surge":
            mci_zones = ["Z2", "Z6", "Z10"]
            for zone in mci_zones:
                n = self._rng.randint(8, 15)
                for _ in range(n):
                    inc = _make_incident(self._rng, f"INC{iid:04d}", zone=zone)
                    inc["respirations"] = self._rng.randint(0, 35)
                    inc["pulse"] = self._rng.randint(0, 140)
                    inc["mental_status"] = self._rng.choice(
                        ["alert","confused","unresponsive","posturing","alert"]
                    )
                    incidents.append(inc)
                    iid += 1
            for h in self._hospitals[:2]:
                h["er_load"] = 0.92
                h["diverted"] = True
                h["available_beds"] = max(0, h["beds"] // 10)
        return incidents
    def _traffic_for_hour(self, hour: int) -> float:
        if 8 <= hour <= 10 or 17 <= hour <= 19:
            return 1.5
        if 22 <= hour or hour <= 5:
            return 0.8
        return 1.0
    def _apply_override(self, override: Dict[str, Any]) -> None:
        if "incidents" in override:
            self._incident_queue = override["incidents"]
        if "fleet" in override:
            self._fleet = override["fleet"]
        if "traffic_mult" in override:
            self._traffic_mult = float(override["traffic_mult"])
    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        self._step_count += 1
        self._sim_clock_min += MINUTES_PER_STEP
        if self._step_count % 3 == 0:
            hour = int((self._sim_clock_min / 60) % 24)
            self._traffic_mult = self._traffic_for_hour(hour)
        if DIFFICULTY_MAP.get(self._task_id) == "hard":
            for unit in self._fleet:
                if unit["status"] == "dispatched":
                    if self._rng.random() < 0.12:
                        unit["comm_ok"] = False
                    else:
                        unit["comm_ok"] = True
            self._comm_failures = {
                u["unit_id"]: not u["comm_ok"]
                for u in self._fleet
            }
        reward, info = self._process_action(action)
        self._advance_simulation()
        survival_reward = self._compute_survival_reward()
        reward += survival_reward
        self._episode_reward += reward
        max_steps = MAX_STEPS_BY_TASK.get(self._task_id, 50)
        queue_empty = (
            len(self._incident_queue) == 0
            and len(self._active_incidents) == 0
        )
        self._done = queue_empty or (self._step_count >= max_steps)
        info.setdefault("cascade_events", [])
        info["step"] = self._step_count
        info["sim_clock_min"] = round(self._sim_clock_min, 2)
        info["survival_reward"] = round(survival_reward, 4)
        info["episode_reward"] = round(self._episode_reward, 4)
        info["queue_length"] = len(self._incident_queue)
        info["active_patients"] = len(self._active_incidents)
        info["resolved_count"] = len(self._resolved)
        obs = self._build_observation()
        return obs, round(reward, 4), self._done, info
    def _process_action(self, action: Dict[str, Any]) -> Tuple[float, Dict]:
        action_type = action.get("action_type", "noop")
        reward = 0.0
        info: Dict[str, Any] = {"action_type": action_type}
        if action_type == "dispatch":
            reward, info = self._handle_dispatch(action)
        elif action_type == "reroute":
            reward, info = self._handle_reroute(action)
        elif action_type == "tag":
            reward, info = self._handle_tag(action)
        elif action_type == "transfer":
            reward, info = self._handle_transfer(action)
        elif action_type == "request_mutual_aid":
            reward, info = self._handle_mutual_aid(action)
        elif action_type == "escalate":
            reward, info = self._handle_escalate(action)
        elif action_type == "reposition":
            reward, info = self._handle_reposition(action)
        elif action_type == "noop":
            p1_unresolved = sum(
                1 for inc in self._active_incidents.values()
                if inc["severity"] == "P1" and not inc["dispatched"]
            )
            reward = -0.05 * p1_unresolved
            info = {"action_type": "noop", "p1_penalty": p1_unresolved}
        else:
            reward = -0.1  
            info = {"action_type": "unknown", "error": f"Unknown action_type: {action_type}"}
        return reward, info
    def _handle_dispatch(self, action: Dict) -> Tuple[float, Dict]:
        incident_id = action.get("incident_id")
        unit_id = action.get("unit_id")
        hospital_id = action.get("hospital_id")
        reward = 0.0
        info = {"action_type": "dispatch"}
        incident = self._find_incident(incident_id)
        if incident is None:
            return -0.1, {"action_type": "dispatch", "error": "incident not found"}
        unit = self._find_unit(unit_id)
        if unit is None:
            return -0.1, {"action_type": "dispatch", "error": "unit not found"}
        if unit["status"] != "available":
            return -0.05, {"action_type": "dispatch", "error": "unit not available"}
        hospital = self._find_hospital(hospital_id)
        if hospital is None:
            return -0.1, {"action_type": "dispatch", "error": "hospital not found"}
        if hospital.get("diverted"):
            reward -= 0.3
            info["diversion_penalty"] = True
        correct_unit = incident.get("correct_unit", "BLS")
        if unit["unit_type"] == correct_unit:
            reward += 0.3
            info["unit_match"] = True
            self._protocol_compliant += 1
        else:
            reward -= 0.1
            info["unit_match"] = False
            self._protocol_violations += 1
        required_spec = incident.get("required_specialty", "general")
        if required_spec in hospital.get("specialties", []):
            reward += 0.2
            info["specialty_match"] = True
        else:
            reward -= 0.05
            info["specialty_match"] = False
        travel_time = (
            _base_travel_time(unit["zone"], incident["zone"]) * self._traffic_mult
        )
        eta_to_hospital = travel_time + _base_travel_time(
            incident["zone"], hospital.get("zone", "Z1")
        ) * self._traffic_mult
        if incident["severity"] == "P1" and travel_time < 15:
            reward += 0.2
        elif incident["severity"] == "P1" and travel_time > 30:
            reward -= 0.1
        if incident.get("requires_police") or incident.get("requires_fire"):
            if not action.get("coordinate_agencies"):
                reward -= 0.1
                info["agency_penalty"] = True
        if unit["hours_on_duty"] > 10:
            reward -= 0.05
            info["fatigue_warning"] = True
        unit["status"] = "dispatched"
        unit["incident_id"] = incident_id
        unit["destination_hospital"] = hospital_id
        unit["eta_min"] = round(eta_to_hospital, 1)
        incident["dispatched"] = True
        incident["unit_assigned"] = unit_id
        incident["hospital_assigned"] = hospital_id
        incident["arrival_time"] = self._sim_clock_min + eta_to_hospital
        self._incident_queue = [
            i for i in self._incident_queue if i["incident_id"] != incident_id
        ]
        if incident_id not in self._active_incidents:
            self._active_incidents[incident_id] = incident
        hospital["er_load"] = min(1.0, hospital["er_load"] + 0.05)
        if hospital["er_load"] > 0.9:
            hospital["diverted"] = True
        info.update({
            "travel_time_min": round(travel_time, 1),
            "eta_to_hospital_min": round(eta_to_hospital, 1),
            "reward": round(reward, 4),
        })
        return reward, info
    def _handle_reroute(self, action: Dict) -> Tuple[float, Dict]:
        unit_id = action.get("unit_id")
        new_hospital_id = action.get("hospital_id")
        reward = 0.0
        unit = self._find_unit(unit_id)
        if unit is None or unit["status"] != "dispatched":
            return -0.1, {"action_type": "reroute", "error": "unit not dispatched"}
        old_hosp = unit.get("destination_hospital")
        new_hosp = self._find_hospital(new_hospital_id)
        if new_hosp is None:
            return -0.1, {"action_type": "reroute", "error": "hospital not found"}
        old_hosp_obj = self._find_hospital(old_hosp)
        if old_hosp_obj and old_hosp_obj.get("diverted") and not new_hosp.get("diverted"):
            reward += 0.4
        elif new_hosp.get("diverted"):
            reward -= 0.2
        unit["destination_hospital"] = new_hospital_id
        old_eta = unit.get("eta_min", 10.0)
        unit["eta_min"] = max(2.0, old_eta - self._rng.uniform(2, 8))
        return reward, {"action_type": "reroute", "time_saved": round(old_eta - unit["eta_min"], 1)}
    def _handle_tag(self, action: Dict) -> Tuple[float, Dict]:
        incident_id = action.get("incident_id")
        tag = action.get("triage_tag")
        reward = 0.0
        incident = self._find_incident(incident_id)
        if incident is None:
            return -0.1, {"action_type": "tag", "error": "incident not found"}
        if tag not in TRIAGE_TAGS:
            return -0.1, {"action_type": "tag", "error": f"invalid tag: {tag}"}
        correct_tag = _start_triage(
            incident.get("respirations", 16),
            incident.get("pulse", 80),
            incident.get("mental_status", "alert"),
        )
        if tag == correct_tag:
            reward += 0.3
        elif correct_tag == "Immediate" and tag == "Expectant":
            reward -= 0.5  
        else:
            reward -= 0.15
        incident["triage_tag"] = tag
        return reward, {
            "action_type": "tag",
            "correct_tag": correct_tag,
            "given_tag": tag,
            "correct": tag == correct_tag,
        }
    def _handle_transfer(self, action: Dict) -> Tuple[float, Dict]:
        incident_id = action.get("incident_id")
        from_hosp = action.get("from_hospital_id")
        to_hosp = action.get("to_hospital_id")
        reward = 0.0
        incident = self._find_incident(incident_id)
        to_hospital = self._find_hospital(to_hosp)
        if to_hospital is None:
            return -0.1, {"action_type": "transfer", "error": "destination hospital not found"}
        required_spec = incident.get("transfer_specialty", "cardiology") if incident else "cardiology"
        if required_spec in to_hospital.get("specialties", []):
            reward += 0.4
        else:
            reward -= 0.1
        if to_hospital.get("diverted"):
            reward -= 0.3
        if to_hospital.get("available_beds", 0) > 0:
            to_hospital["available_beds"] -= 1
            reward += 0.1
        return reward, {"action_type": "transfer", "specialty_match": required_spec in to_hospital.get("specialties", [])}
    def _handle_mutual_aid(self, action: Dict) -> Tuple[float, Dict]:
        n_units = int(action.get("units_requested", 1))
        self._mutual_aid_pending += n_units
        reward = 0.0
        available_own = sum(1 for u in self._fleet if u["status"] == "available")
        if n_units > 3 and available_own > 2:
            reward -= 0.1 * (n_units - 3)
            self._mutual_aid_over_requests += 1
        if self._task_id == "task9_surge" and len(self._incident_queue) > 20 and n_units >= 2:
            reward += 0.3
        return reward, {"action_type": "request_mutual_aid", "units_requested": n_units, "eta_min": 12}
    def _handle_escalate(self, action: Dict) -> Tuple[float, Dict]:
        return 0.1, {"action_type": "escalate", "escalated": True}
    def _handle_reposition(self, action: Dict) -> Tuple[float, Dict]:
        unit_id = action.get("unit_id")
        target_zone = action.get("zone")
        reward = 0.0
        unit = self._find_unit(unit_id)
        if unit is None or unit["status"] != "available":
            return -0.05, {"action_type": "reposition", "error": "unit not available"}
        if target_zone not in ZONES:
            return -0.1, {"action_type": "reposition", "error": "invalid zone"}
        old_zone = unit["zone"]
        unit["zone"] = target_zone
        unit["last_known_zone"] = target_zone
        high_demand = self._get_demand_heatmap()
        demand_score = high_demand.get(target_zone, 0.0)
        reward = demand_score * 0.3
        return reward, {"action_type": "reposition", "from": old_zone, "to": target_zone, "demand": round(demand_score, 3)}
    def _advance_simulation(self) -> None:
        to_resolve = []
        for inc_id, incident in self._active_incidents.items():
            incident["elapsed_min"] += MINUTES_PER_STEP
            incident["survival_prob"] = _survival_prob(
                incident["condition"], incident["elapsed_min"]
            )
            if (
                incident.get("arrival_time") is not None
                and self._sim_clock_min >= incident["arrival_time"]
                and incident["dispatched"]
            ):
                to_resolve.append(inc_id)
        for inc_id in to_resolve:
            incident = self._active_incidents.pop(inc_id)
            incident["resolved"] = True
            incident["resolved_at_min"] = self._sim_clock_min
            self._resolved.append(incident)
            for unit in self._fleet:
                if unit["incident_id"] == inc_id:
                    unit["status"] = "available"
                    unit["incident_id"] = None
                    unit["destination_hospital"] = None
                    unit["hours_on_duty"] += round(MINUTES_PER_STEP / 60, 2)
                    if unit["hours_on_duty"] > 10:
                        unit["status"] = "fatigue"
        for inc in self._incident_queue:
            inc["elapsed_min"] += MINUTES_PER_STEP
            inc["survival_prob"] = _survival_prob(inc["condition"], inc["elapsed_min"])
        if self._mutual_aid_pending > 0:
            if self._step_count % 4 == 0:
                n_arrive = min(self._mutual_aid_pending, 2)
                self._mutual_aid_pending -= n_arrive
                for _ in range(n_arrive):
                    zone = self._rng.choice(ZONES)
                    self._fleet.append({
                        "unit_id": f"MA{self._step_count:03d}_{self._rng.randint(0,99)}",
                        "unit_type": "ALS",
                        "zone": zone,
                        "status": "available",
                        "hours_on_duty": 0.0,
                        "incident_id": None,
                        "destination_hospital": None,
                        "eta_min": 0.0,
                        "last_known_zone": zone,
                        "comm_ok": True,
                    })
        if self._task_id == "task9_surge" and self._step_count <= 20:
            if self._step_count % 5 == 0:
                inc = _make_incident(
                    self._rng,
                    f"INC_SURGE_{self._step_count}",
                    zone=self._rng.choice(["Z2","Z6","Z10"]),
                )
                self._incident_queue.append(inc)
    def _compute_survival_reward(self) -> float:
        reward = 0.0
        for inc in self._active_incidents.values():
            if inc["severity"] == "P1":
                prob = inc.get("survival_prob", 1.0)
                reward += (prob - 0.5) * 0.02
        return round(reward, 4)
    def _get_demand_heatmap(self) -> Dict[str, float]:
        heatmap = {}
        base = {z: self._rng.uniform(0.1, 0.9) for z in ZONES}
        for inc in self._incident_queue + list(self._active_incidents.values()):
            zone = inc.get("zone", "Z1")
            base[zone] = min(1.0, base.get(zone, 0.0) + 0.1)
        return base
    def _build_observation(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "step": self._step_count,
            "sim_clock_min": round(self._sim_clock_min, 2),
            "incident_queue": [
                self._serialise_incident(i) for i in self._incident_queue
            ],
            "active_incidents": {
                k: self._serialise_incident(v)
                for k, v in self._active_incidents.items()
            },
            "fleet_status": [self._serialise_unit(u) for u in self._fleet],
            "hospital_network": [self._serialise_hospital(h) for h in self._hospitals],
            "traffic_multiplier": round(self._traffic_mult, 2),
            "demand_forecast": self._get_demand_heatmap(),
            "comm_failures": self._comm_failures,
            "mutual_aid_pending": self._mutual_aid_pending,
            "resolved_count": len(self._resolved),
            "episode_reward": round(self._episode_reward, 4),
            "done": self._done,
        }
    def _serialise_incident(self, inc: Dict) -> Dict:
        return {
            "incident_id":     inc.get("incident_id"),
            "zone":            inc.get("zone"),
            "condition":       inc.get("condition"),
            "severity":        inc.get("severity"),
            "description":     inc.get("description"),
            "elapsed_min":     round(inc.get("elapsed_min", 0.0), 2),
            "survival_prob":   round(inc.get("survival_prob", 1.0), 4),
            "dispatched":      inc.get("dispatched", False),
            "unit_assigned":   inc.get("unit_assigned"),
            "hospital_assigned": inc.get("hospital_assigned"),
            "triage_tag":      inc.get("triage_tag"),
            "respirations":    inc.get("respirations"),
            "pulse":           inc.get("pulse"),
            "mental_status":   inc.get("mental_status"),
            "requires_police": inc.get("requires_police", False),
            "requires_fire":   inc.get("requires_fire", False),
        }
    def _serialise_unit(self, u: Dict) -> Dict:
        return {
            "unit_id":      u.get("unit_id"),
            "unit_type":    u.get("unit_type"),
            "zone":         u.get("zone"),
            "status":       u.get("status"),
            "hours_on_duty": round(u.get("hours_on_duty", 0.0), 1),
            "incident_id":  u.get("incident_id"),
            "destination_hospital": u.get("destination_hospital"),
            "eta_min":      round(u.get("eta_min", 0.0), 1),
            "comm_ok":      u.get("comm_ok", True),
        }
    def _serialise_hospital(self, h: Dict) -> Dict:
        return {
            "id":            h.get("id"),
            "name":          h.get("name"),
            "zone":          h.get("zone"),
            "available_beds": h.get("available_beds", 0),
            "icu":           h.get("icu", 0),
            "er_load":       round(h.get("er_load", 0.0), 3),
            "diverted":      h.get("diverted", False),
            "specialties":   h.get("specialties", []),
            "helipad":       h.get("helipad", False),
        }
    def get_state(self) -> Dict[str, Any]:
        return {
            "task_id":           self._task_id,
            "episode_id":        self._episode_id,
            "seed":              self._seed,
            "step_count":        self._step_count,
            "sim_clock_min":     round(self._sim_clock_min, 2),
            "episode_reward":    round(self._episode_reward, 4),
            "done":              self._done,
            "incident_queue":    [self._serialise_incident(i) for i in self._incident_queue],
            "active_incidents":  {k: self._serialise_incident(v) for k, v in self._active_incidents.items()},
            "resolved":          [self._serialise_incident(r) for r in self._resolved],
            "fleet":             [self._serialise_unit(u) for u in self._fleet],
            "hospitals":         [self._serialise_hospital(h) for h in self._hospitals],
            "traffic_mult":      round(self._traffic_mult, 2),
            "demand_heatmap":    self._get_demand_heatmap(),
            "comm_failures":     self._comm_failures,
            "mutual_aid_pending": self._mutual_aid_pending,
            "mutual_aid_over_requests": self._mutual_aid_over_requests,
            "protocol_violations": self._protocol_violations,
            "protocol_compliant": self._protocol_compliant,
        }
    def _find_incident(self, incident_id: Optional[str]) -> Optional[Dict]:
        if incident_id is None:
            return None
        for inc in self._incident_queue:
            if inc["incident_id"] == incident_id:
                return inc
        return self._active_incidents.get(incident_id)
    def _find_unit(self, unit_id: Optional[str]) -> Optional[Dict]:
        if unit_id is None:
            return None
        for u in self._fleet:
            if u["unit_id"] == unit_id:
                return u
        return None
    def _find_hospital(self, hospital_id: Optional[str]) -> Optional[Dict]:
        if hospital_id is None:
            return None
        for h in self._hospitals:
            if h["id"] == hospital_id:
                return h
        return None
def make_env() -> EmergiEnv:
    return EmergiEnv()