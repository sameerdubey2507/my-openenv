from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        pass                                                    
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import httpx

try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("emergi_env.inference")

API_BASE_URL: str   = os.getenv("API_BASE_URL", "http://localhost:7860").rstrip("/")
SERVER_URL: str     = os.getenv("SERVER_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME: str     = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

HF_TOKEN: str       = os.getenv("HF_TOKEN")
API_KEY: str        = HF_TOKEN
MAX_LLM_TOKENS: int = int(os.getenv("MAX_LLM_TOKENS", "1000"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
LLM_TIMEOUT: float  = float(os.getenv("LLM_TIMEOUT", "45.0"))
AGENT_SESSION_PREFIX: str = os.getenv("AGENT_SESSION_PREFIX", "agent")
RESULTS_OUTPUT_PATH: str  = os.getenv("RESULTS_OUTPUT_PATH", "inference_results.json")

TASK_IDS: List[str] = [
    "task1_single_triage",
    "task2_hospital_route",
    "task3_unit_type",
    "task4_multi_incident",
    "task5_dynamic_rerouting",
    "task6_prepositioning",
    "task7_mci_start",
    "task8_transfer_cascade",
    "task9_surge",
]

TASK_SEEDS: Dict[str, int] = {
    "task1_single_triage":    42,
    "task2_hospital_route":   43,
    "task3_unit_type":        44,
    "task4_multi_incident":   45,
    "task5_dynamic_rerouting":46,
    "task6_prepositioning":   47,
    "task7_mci_start":        48,
    "task8_transfer_cascade": 49,
    "task9_surge":            50,
}

TASK_BASELINES: Dict[str, float] = {
    "task1_single_triage":    0.61,
    "task2_hospital_route":   0.72,
    "task3_unit_type":        0.68,
    "task4_multi_incident":   0.44,
    "task5_dynamic_rerouting":0.38,
    "task6_prepositioning":   0.42,
    "task7_mci_start":        0.29,
    "task8_transfer_cascade": 0.24,
    "task9_surge":            0.17,
}

TASK_DIFFICULTIES: Dict[str, str] = {
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

MAX_STEPS: Dict[str, int] = {
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

CONDITION_UNIT_MAP: Dict[str, str] = {
    "stemi_anterior": "MICU", "stemi_inferior": "MICU",
    "stemi_posterior": "MICU", "stemi_with_vf_arrest": "MICU",
    "stemi_cocaine": "MICU", "stemi_post_cabg": "MICU",
    "cardiac_arrest": "MICU", "cardiac_arrest_vf": "MICU",
    "cardiac_arrest_pea": "MICU", "complete_heart_block": "MICU",
    "wpw_svt": "MICU", "hypertensive_emergency": "MICU",
    "ischemic_stroke": "ALS", "ischemic_stroke_wake_up": "ALS",
    "hemorrhagic_stroke_sah": "ALS", "paediatric_stroke": "ALS",
    "severe_tbi": "MICU", "seizure_status_epilepticus": "ALS",
    "polytrauma_blunt": "MICU", "polytrauma_penetrating": "MICU",
    "blast_injury": "MICU", "crush_syndrome": "MICU",
    "chest_trauma": "ALS", "splenic_laceration": "ALS",
    "traumatic_amputation": "MICU",
    "severe_asthma": "ALS", "tension_pneumothorax": "MICU",
    "respiratory_failure": "ALS", "anaphylaxis": "ALS",
    "eclampsia": "MICU", "postpartum_haemorrhage": "MICU",
    "cord_prolapse": "MICU",
    "burns_major": "MICU", "burns_moderate": "ALS", "burns_minor": "BLS",
    "general_illness": "BLS", "trauma_minor": "BLS", "hypoglycaemia": "ALS",
}

CONDITION_SPECIALTY_MAP: Dict[str, str] = {
    "stemi_anterior": "cardiology", "stemi_inferior": "cardiology",
    "stemi_posterior": "cardiology", "stemi_with_vf_arrest": "cardiology",
    "cardiac_arrest": "cardiology", "cardiac_arrest_vf": "cardiology",
    "ischemic_stroke": "neurology", "hemorrhagic_stroke_sah": "neurology",
    "paediatric_stroke": "neurology", "severe_tbi": "neurosurgery",
    "polytrauma_blunt": "trauma", "polytrauma_penetrating": "trauma",
    "blast_injury": "trauma", "burns_major": "burns",
    "eclampsia": "obstetrics", "postpartum_haemorrhage": "obstetrics",
    "cord_prolapse": "obstetrics",
}

CATH_LAB_CONDITIONS: set = {
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
    "cardiac_arrest_vf", "cardiac_arrest_pea",
}

STROKE_UNIT_CONDITIONS: set = {
    "ischemic_stroke", "ischemic_stroke_wake_up",
    "hemorrhagic_stroke_sah", "paediatric_stroke",
}

def rpm_to_start_tag(respirations: str, pulse: str, mental_status: str) -> str:
    r_ok = respirations not in ("absent",)
    p_ok = pulse not in ("absent",)
    m_ok = mental_status in ("alert",)
    if not r_ok:
        return "Expectant"
    if not p_ok:
        return "Immediate"
    if respirations in ("present_rapid", "present_abnormal"):
        return "Immediate"
    if not m_ok:
        return "Delayed"
    return "Minimal"

class AgentMode(str, Enum):
    LLM        = "llm"
    RULE_BASED = "rule_based"
    HYBRID     = "hybrid"

@dataclass
class StepMetrics:
    step: int
    action_type: str
    action_dict: Dict[str, Any]
    reward: float
    done: bool
    llm_used: bool
    llm_latency_ms: float
    tokens_used: int
    rule_fallback: bool
    error: Optional[str]

@dataclass
class EpisodeResult:
    task_id: str
    episode_id: str
    seed: int
    session_id: str
    mode: str
    final_score: float
    baseline: float
    beats_baseline: bool
    delta_vs_baseline: float
    total_steps: int
    total_reward: float
    grader_status: str
    grader_components: Dict[str, float]
    grader_penalties: List[Dict[str, Any]]
    p1_survival_rate: float
    protocol_violations: int
    total_llm_tokens: int
    avg_step_latency_ms: float
    cascade_events: int
    mutual_aid_requests: int
    surge_declared: bool
    step_metrics: List[StepMetrics] = field(default_factory=list)
    error_message: Optional[str] = None
    run_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary_line(self) -> str:
        beat = "✓ BEATS" if self.beats_baseline else "✗ BELOW"
        diff = TASK_DIFFICULTIES.get(self.task_id, "?")
        return (
            f"  {self.task_id:<28}  {diff:<6}  "
            f"score={self.final_score:.4f}  "
            f"baseline={self.baseline:.2f}  "
            f"{beat}  (Δ={self.delta_vs_baseline:+.4f})  "
            f"steps={self.total_steps}  "
            f"tokens={self.total_llm_tokens}"
        )

class RetryPolicy:

    def __init__(
        self,
        max_attempts: int = 15,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
        jitter: float = 0.3,
        retryable_status: Tuple[int, ...] = (429, 500, 502, 503, 504, 0),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_status = retryable_status

    def delays(self) -> Iterator[float]:
        for attempt in range(self.max_attempts):
            if attempt == 0:
                yield 0.0
            else:
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                jitter_amount = delay * self.jitter * random.uniform(-1.0, 1.0)
                yield max(0.0, delay + jitter_amount)

    def should_retry(self, exc: Exception) -> bool:
        if _OPENAI_AVAILABLE:
            if isinstance(exc, RateLimitError):
                return True
            if isinstance(exc, APITimeoutError):
                return True
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ProtocolError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in self.retryable_status
        return False

class ServerClient:

    def __init__(
        self,
        base_url: str = SERVER_URL,
        timeout: float = REQUEST_TIMEOUT,
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"EMERGI-ENV-Inference/{MODEL_NAME}",
                **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
            },
            follow_redirects=True,
        )
        self._retry = retry or RetryPolicy()

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for delay in self._retry.delays():
            if delay > 0:
                logger.debug("Retry after %.1fs for %s %s", delay, method, path)
                time.sleep(delay)
            try:
                resp = self._client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if not self._retry.should_retry(exc):
                    break
                logger.warning("Transient error (%s) for %s %s — will retry", exc, method, path)
        raise RuntimeError(f"Request failed after retries: {last_exc}") from last_exc

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def reset(
        self,
        task_id: str,
        seed: Optional[int] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        return self._request("POST", "/reset", json={
            "task_id": task_id,
            "seed": seed,
            "session_id": session_id,
        })

    def step(
        self,
        action: Dict[str, Any],
        session_id: str = "default",
    ) -> Dict[str, Any]:
        return self._request("POST", "/step", json={
            "action": action,
            "session_id": session_id,
        })

    def grade(
        self,
        session_id: str = "default",
        force_complete: bool = True,
    ) -> Dict[str, Any]:
        return self._request("POST", "/grade", json={
            "session_id": session_id,
            "force_complete": force_complete,
        })

    def grade_batch(self, task_ids: List[str]) -> Dict[str, Any]:
        return self._request("POST", "/grade/batch", json={"task_ids": task_ids})

    def get_state(self, session_id: str = "default") -> Dict[str, Any]:
        return self._request("GET", "/state", params={"session_id": session_id})

    def get_tasks(self) -> Dict[str, Any]:
        return self._request("GET", "/tasks")

    def validate(self) -> Dict[str, Any]:
        return self._request("POST", "/validate")

    def get_metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/metrics")

    def get_episode_ledger(self, session_id: str = "default") -> Dict[str, Any]:
        return self._request("GET", "/episode/ledger", params={"session_id": session_id})

    def close(self) -> None:
        self._client.close()

class ObservationParser:

    @staticmethod
    def parse(obs_raw: Dict[str, Any]) -> "ParsedObservation":
        obs = obs_raw.get("observation", obs_raw)
                                                                                       
        incident_queue = obs.get("incident_queue", [])
        active_incidents = obs.get("active_incidents", {})
        resolved_count = int(obs.get("resolved_count", 0))
        return ParsedObservation(
            task_id=obs.get("task_id", ""),
            episode_id=obs.get("episode_id", ""),
            step=int(obs.get("step", 0)),
            sim_clock_min=float(obs.get("sim_clock_min", 0.0)),
            queue_length=len(incident_queue),
            active_patients=len(active_incidents) if isinstance(active_incidents, dict) else int(obs.get("active_patients", 0)),
            resolved_patients=resolved_count,
            surge_declared=bool(obs.get("surge_declared", False)),
            mci_active=bool(obs.get("mci_active", len(incident_queue) >= 10)),
            incidents=ObservationParser._parse_incidents(incident_queue),
            units=ObservationParser._parse_units(obs.get("fleet_status", [])),
            hospitals=ObservationParser._parse_hospitals(obs.get("hospital_network", [])),
            traffic=ObservationParser._parse_traffic(obs.get("traffic_snapshot", {})),
            demand=ObservationParser._parse_demand(obs.get("demand_forecast")),
        )

    @staticmethod
    def _parse_incidents(raw: List[Dict]) -> List["IncidentSummary"]:
        result = []
        for inc in raw:
                                                                  
            result.append(IncidentSummary(
                incident_id=inc.get("incident_id", ""),
                symptom_description=inc.get("description", inc.get("symptom_description", "")),
                zone_id=inc.get("zone", inc.get("zone_id", "Z1")),
                zone_type=inc.get("zone_type", "metro_core"),
                severity_hint=inc.get("severity", inc.get("severity_hint", "P2")),
                condition_family=inc.get("condition", inc.get("condition_family", "general")),
                requires_extrication=bool(inc.get("requires_fire", inc.get("requires_extrication", False))),
                requires_police=bool(inc.get("requires_police", False)),
                requires_fire=bool(inc.get("requires_fire", False)),
            ))
        sev_order = {"P1": 0, "P2": 1, "P3": 2, "P0": 3}
        result.sort(key=lambda i: sev_order.get(i.severity_hint, 9))
        return result

    @staticmethod
    def _parse_units(raw: List[Dict]) -> List["UnitSummary"]:
        return [
            UnitSummary(
                unit_id=u.get("unit_id", ""),
                unit_type=u.get("unit_type", "BLS"),
                                                                             
                status=u.get("status", "available"),
                zone_id=u.get("zone", u.get("zone_id") or u.get("last_known_zone", "Z1")),
                hours_on_duty=float(u.get("hours_on_duty", 0.0)),
                fatigue_flag=bool(u.get("fatigue_flag", u.get("status") == "fatigue")),
                comm_ok=bool(u.get("comm_ok", True)),
                current_patient_id=u.get("incident_id", u.get("current_patient_id")),
            )
            for u in raw
        ]

    @staticmethod
    def _parse_hospitals(raw: List[Dict]) -> List["HospitalSummary"]:
        return [
            HospitalSummary(
                                                                                     
                hospital_id=h.get("id", h.get("hospital_id", "")),
                name=h.get("name", ""),
                zone_id=h.get("zone", h.get("zone_id", "Z1")),
                specialties=h.get("specialties", []),
                er_occupancy=float(h.get("er_load", h.get("er_occupancy", h.get("er_occupancy_pct", 0.0)))),
                icu_utilisation=float(h.get("icu_utilisation", h.get("icu_utilisation_pct", 0.0))),
                on_diversion=bool(h.get("diverted", h.get("on_diversion", False))),
                has_cath_lab=bool("cath_lab" in h.get("specialties", []) or h.get("has_cath_lab", False)),
                has_helipad=bool(h.get("helipad", h.get("has_helipad", False))),
                is_level1_trauma=bool("trauma" in h.get("specialties", []) and h.get("is_level1_trauma", False)),
                pre_alerted=bool(h.get("pre_alerted", False)),
            )
            for h in raw
        ]

    @staticmethod
    def _parse_traffic(raw: Dict) -> "TrafficSummary":
                                                                          
        return TrafficSummary(
            sim_clock_min=float(raw.get("sim_clock_min", 0.0)),
            is_peak_hour=bool(raw.get("is_peak_hour", False)),
            matrix=raw.get("matrix", {}),
        )

    @staticmethod
    def _parse_demand(raw: Optional[Dict]) -> Optional["DemandSummary"]:
        if not raw:
            return None
                                                                                     
        if isinstance(raw, dict) and "heatmap" not in raw:
            return DemandSummary(heatmap=raw)
        return DemandSummary(heatmap=raw.get("heatmap", raw))

@dataclass
class IncidentSummary:
    incident_id: str
    symptom_description: str
    zone_id: str
    zone_type: str
    severity_hint: str
    condition_family: str
    requires_extrication: bool
    requires_police: bool
    requires_fire: bool

    @property
    def needs_multi_agency(self) -> bool:
        return self.requires_extrication or self.requires_police or self.requires_fire

    @property
    def inferred_unit_type(self) -> str:
        desc = self.symptom_description.lower()
        for cond, unit in CONDITION_UNIT_MAP.items():
            if cond.replace("_", " ") in desc or cond in desc:
                return unit
        family_map = {
            "cardiac": "MICU", "neuro": "ALS", "trauma": "ALS",
            "obstetric": "MICU", "burns": "ALS", "respiratory": "ALS",
            "toxicology": "ALS", "general": "BLS",
        }
        return family_map.get(self.condition_family, "ALS")

@dataclass
class UnitSummary:
    unit_id: str
    unit_type: str
    status: str
    zone_id: str
    hours_on_duty: float
    fatigue_flag: bool
    comm_ok: bool
    current_patient_id: Optional[str]

    @property
    def is_available(self) -> bool:
        return self.status == "available" and not self.fatigue_flag

    @property
    def is_deployable(self) -> bool:
        return self.status in ("available",) and self.comm_ok

@dataclass
class HospitalSummary:
    hospital_id: str
    name: str
    zone_id: str
    specialties: List[str]
    er_occupancy: float
    icu_utilisation: float
    on_diversion: bool
    has_cath_lab: bool
    has_helipad: bool
    is_level1_trauma: bool
    pre_alerted: bool

    @property
    def accepts_patients(self) -> bool:
        return not self.on_diversion and self.er_occupancy < 0.90

    def has_specialty(self, spec: str) -> bool:
        return any(spec.lower() in s.lower() for s in self.specialties)

    @property
    def capacity_score(self) -> float:
        if self.on_diversion:
            return 0.0
        return max(0.0, 1.0 - self.er_occupancy)

@dataclass
class TrafficSummary:
    sim_clock_min: float
    is_peak_hour: bool
    matrix: Dict[str, Dict[str, float]]

    def travel_time(self, from_zone: str, to_zone: str) -> float:
        base = self.matrix.get(from_zone, {}).get(to_zone, 12.0)
        if self.is_peak_hour:
            base *= 1.45
        return base

@dataclass
class DemandSummary:
    heatmap: Dict[str, float]

    def top_zones(self, n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.heatmap.items(), key=lambda x: x[1], reverse=True)[:n]

@dataclass
class ParsedObservation:
    task_id: str
    episode_id: str
    step: int
    sim_clock_min: float
    queue_length: int
    active_patients: int
    resolved_patients: int
    surge_declared: bool
    mci_active: bool
    incidents: List[IncidentSummary]
    units: List[UnitSummary]
    hospitals: List[HospitalSummary]
    traffic: TrafficSummary
    demand: Optional[DemandSummary]

    @property
    def available_units(self) -> List[UnitSummary]:
        return [u for u in self.units if u.is_deployable]

    @property
    def available_by_type(self) -> Dict[str, List[UnitSummary]]:
        result: Dict[str, List[UnitSummary]] = defaultdict(list)
        for u in self.available_units:
            result[u.unit_type].append(u)
        return result

    @property
    def accepting_hospitals(self) -> List[HospitalSummary]:
        return [h for h in self.hospitals if h.accepts_patients]

    def best_unit_for_incident(self, inc: IncidentSummary) -> Optional[UnitSummary]:
        needed_type = inc.inferred_unit_type
        type_rank = {"MICU": 3, "ALS": 2, "BLS": 1}
        needed_rank = type_rank.get(needed_type, 1)
        candidates = [u for u in self.available_units if type_rank.get(u.unit_type, 0) >= needed_rank]
        if not candidates:
            candidates = self.available_units
        if not candidates:
            return None
        same_zone = [u for u in candidates if u.zone_id == inc.zone_id]
        return same_zone[0] if same_zone else candidates[0]

    def best_hospital_for_incident(self, inc: IncidentSummary) -> Optional[HospitalSummary]:
        from_zone = inc.zone_id
        desc = inc.symptom_description.lower()
        needed_spec = None
        for cond, spec in CONDITION_SPECIALTY_MAP.items():
            if cond.replace("_", " ") in desc or cond in desc:
                needed_spec = spec
                break

        candidates = self.accepting_hospitals
        if not candidates:
            candidates = self.hospitals

        def score(h: HospitalSummary) -> float:
            spec_score = 1.0 if (needed_spec and h.has_specialty(needed_spec)) else 0.3
            capacity = h.capacity_score
            travel = self.traffic.travel_time(from_zone, h.zone_id)
            travel_score = max(0.0, 1.0 - travel / 60.0)
            return 0.5 * spec_score + 0.3 * capacity + 0.2 * travel_score

        return max(candidates, key=score) if candidates else None

    def surge_warranted(self) -> bool:
        diverted = sum(1 for h in self.hospitals if h.on_diversion)
        avail = len(self.available_units)
        return diverted >= 3 or (self.queue_length >= 10) or (self.queue_length >= 5 and avail == 0)

class PromptBuilder:

    SYSTEM_BASE: str = """You are an expert emergency medical dispatcher for India's 108/112 EMS system.
You manage ambulance dispatch, hospital routing, and resource allocation across a 12-zone Pune city grid.
You must follow all Indian EMS protocols and maximise patient survival.

KEY PROTOCOLS YOU MUST FOLLOW:
1. STEMI/cardiac arrest → MICU ONLY. Dispatch to cath-lab hospital. Pre-alert cath lab.
2. Stroke → ALS or MICU minimum. Route to neurology hospital. Pre-alert stroke unit.
3. Major trauma → Level-1 trauma centre within 30 minutes. Multi-agency if victim is trapped.
4. NEVER route to a hospital marked on_diversion=true (penalty: -0.30 per violation).
5. START triage in MCI: RPM → Immediate (airway/circulation issue) / Delayed (walking wounded) / Minimal (minor) / Expectant (non-survivable).
   CRITICAL: Never tag Immediate as Expectant (penalty: -0.50).
6. If >3 hospitals diverted OR queue ≥10, declare SURGE before dispatching.
7. Request mutual aid (12-min delay) when queue > available units + 2.
8. After 10h on-duty, crew fatigue degrades performance → request crew_swap.

RESPONSE FORMAT — You MUST respond with valid JSON and nothing else:
{"action_type": "...", ...fields...}

Think step by step BEFORE outputting JSON. Use a <reasoning> block first, then output the JSON action."""

    TASK_SYSTEM_ADDENDA: Dict[str, str] = {
        "task1_single_triage": """
TASK 1 — SINGLE CALL TRIAGE & DISPATCH:
You will receive ONE incident. Your score depends on:
- Correct triage class P1/P2/P3 (0.40 weight)
- Correct unit type BLS/ALS/MICU (0.30 weight)
- Correct hospital specialty match (0.30 weight)
HOSPITAL IDs (use EXACTLY these): H1=Sassoon General (trauma/cardiology/neurology), H2=KEM (trauma/ortho), H3=Ruby Hall (cardiology/cath_lab/stroke), H4=Jehangir (general/burns/pediatrics), H5=Deenanath (cardiology/neurology), H6=Bharati (general/trauma), H7=Aditya Birla (cardiology/transplant/stroke), H8=Columbia Asia (general/ortho/burns)
Required JSON: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "hospital_id": "H1", "unit_type": "MICU"}""",

        "task2_hospital_route": """
TASK 2 — HOSPITAL ROUTE SELECTION:
Score depends on: specialty match (0.50), capacity (0.30), travel time (0.20).
CRITICAL: Use action_type="dispatch" with hospital_id from list below. Check er_load (diverted=true means skip).
HOSPITAL IDs: H1=Sassoon General (trauma/cardiology/neurology, zone Z3), H2=KEM (trauma/ortho, Z5), H3=Ruby Hall (cardiology/cath_lab/stroke, Z6), H4=Jehangir (general/burns/pediatrics, Z4), H5=Deenanath (cardiology/neurology, Z7), H6=Bharati (general/trauma, Z9), H7=Aditya Birla (cardiology/transplant/stroke, Z10), H8=Columbia Asia (general/ortho/burns, Z11)
Required JSON: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "hospital_id": "H3"}""",

        "task3_unit_type": """
TASK 3 — UNIT TYPE MATCHING:
Pure protocol knowledge. Select EXACTLY the correct unit type:
- BLS: minor trauma, non-urgent (falls, lacerations, minor burns)
- ALS: stroke, moderate cardiac, respiratory, burns-moderate, seizures
- MICU: STEMI, cardiac arrest, major trauma, eclampsia, polytrauma
CRITICAL: use action_type="dispatch" (NOT "unit_type_match" or "unit_assignment")
Required JSON: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "unit_type": "MICU", "hospital_id": "H1"}""",

        "task4_multi_incident": """
TASK 4 — MULTI-INCIDENT QUEUE (5-8 calls, 3 ambulances):
Resource scarcity. Prioritise P1 patients — they carry 3x weight in scoring.
If P1 unserved and unit available: dispatch immediately.
Strategy: P1 → MICU/ALS first, P2 → ALS, P3 → BLS. Accept some P3 delays.
Required JSON: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "hospital_id": "H1"}""",

        "task5_dynamic_rerouting": """
TASK 5 — DYNAMIC REROUTING:
Monitor: if hospital diverted=true → REROUTE immediately.
Action: {"action_type": "reroute", "unit_id": "Uxxx", "hospital_id": "H3"}
Or dispatch fresh: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "hospital_id": "H1"}""",

        "task6_prepositioning": """
TASK 6 — PRE-POSITIONING (no active incidents):
Use demand_forecast heatmap to pre-position ambulances to high-demand zones.
Required JSON: {"action_type": "reposition", "unit_id": "Uxxx", "zone": "Z3"}
Valid zones: Z1 through Z12. Pick zones with highest demand score.""",

        "task7_mci_start": """
TASK 7 — MASS CASUALTY INCIDENT (20-40 victims):
Apply START triage FIRST: tag each victim with Immediate/Delayed/Minimal/Expectant.
Action: {"action_type": "tag", "incident_id": "INCxxxx", "triage_tag": "Immediate"}
Then dispatch Immediate victims: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "hospital_id": "H1"}
CRITICAL PENALTY: triage_tag="Expectant" for Immediate (P1) patients = -0.50. NEVER do this.""",

        "task8_transfer_cascade": """
TASK 8 — INTER-HOSPITAL TRANSFER CASCADE:
Patients need specialist care. Initiate transfer to specialist hospital.
Required JSON: {"action_type": "transfer", "incident_id": "INCxxxx", "to_hospital_id": "H3"}
Pick specialist: H3=cardiology/cath_lab, H5=cardiology/neurology, H7=cardiology/transplant, H1=trauma""",

        "task9_surge": """
TASK 9 — CITY-WIDE SURGE (3 simultaneous MCIs):
STEP 1: Use escalate to signal surge: {"action_type": "escalate", "reason": "surge"}
STEP 2: Request mutual aid: {"action_type": "request_mutual_aid", "units_requested": 4}
STEP 3: Tag victims: {"action_type": "tag", "incident_id": "INCxxxx", "triage_tag": "Immediate"}
STEP 4: Dispatch: {"action_type": "dispatch", "incident_id": "INCxxxx", "unit_id": "Uxxx", "hospital_id": "H1"}""",
    }

    @classmethod
    def build_system_prompt(cls, task_id: str) -> str:
        addendum = cls.TASK_SYSTEM_ADDENDA.get(task_id, "")
        return cls.SYSTEM_BASE + "\n" + addendum

    @classmethod
    def build_user_prompt(cls, obs: ParsedObservation, step_history: List[Dict[str, Any]]) -> str:
        lines = [
            f"=== EMERGI-ENV Step {obs.step} | Task: {obs.task_id} | Clock: {obs.sim_clock_min:.0f} min ===",
            "",
        ]

        if obs.incidents:
            lines.append(f"INCIDENT QUEUE ({obs.queue_length} pending):")
            for inc in obs.incidents[:10]:
                ma = " [MULTI-AGENCY REQUIRED]" if inc.needs_multi_agency else ""
                lines.append(
                    f"  [{inc.severity_hint}] {inc.incident_id} | Zone: {inc.zone_id} | "
                    f"Family: {inc.condition_family} | Suggested unit: {inc.inferred_unit_type}{ma}"
                )
                lines.append(f"    Symptoms: {inc.symptom_description[:120]}")
            if obs.queue_length > 10:
                lines.append(f"  ... and {obs.queue_length - 10} more incidents")
        else:
            lines.append("INCIDENT QUEUE: empty")

        lines.append("")

        avail = obs.available_units
        lines.append(f"FLEET STATUS ({len(avail)} available / {len(obs.units)} total):")
        for unit in obs.units[:15]:
            comm_flag = " [COMM LOST]" if not unit.comm_ok else ""
            fatigue_flag = " [FATIGUED]" if unit.fatigue_flag else ""
            patient_flag = f" → patient={unit.current_patient_id}" if unit.current_patient_id else ""
            lines.append(
                f"  {unit.unit_id} ({unit.unit_type}) | status={unit.status} | "
                f"zone={unit.zone_id} | {unit.hours_on_duty:.1f}h on-duty"
                f"{comm_flag}{fatigue_flag}{patient_flag}"
            )

        lines.append("")

        lines.append("HOSPITAL NETWORK:")
        for hosp in obs.hospitals:
            div_flag = " *** ON DIVERSION ***" if hosp.on_diversion else ""
            cath_flag = " [CATH LAB]" if hosp.has_cath_lab else ""
            trauma_flag = " [LEVEL-1 TRAUMA]" if hosp.is_level1_trauma else ""
            lines.append(
                f"  {hosp.hospital_id} | {hosp.name[:30]} | zone={hosp.zone_id} | "
                f"ER={hosp.er_occupancy:.0%} | ICU={hosp.icu_utilisation:.0%}{div_flag}{cath_flag}{trauma_flag}"
            )
            lines.append(f"    Specialties: {', '.join(hosp.specialties[:4])}")

        lines.append("")

        peak = " *** PEAK HOUR ***" if obs.traffic.is_peak_hour else ""
        lines.append(f"TRAFFIC: clock={obs.traffic.sim_clock_min:.0f}min{peak}")

        if obs.demand:
            top = obs.demand.top_zones(5)
            lines.append(f"DEMAND FORECAST (top zones): {', '.join(f'{z}={v:.2f}' for z, v in top)}")

        if obs.surge_declared:
            lines.append("SURGE: *** SURGE ALREADY DECLARED ***")
        elif obs.surge_warranted():
            diverted = sum(1 for h in obs.hospitals if h.on_diversion)
            lines.append(
                f"SURGE WARNING: {diverted} hospitals diverted, queue={obs.queue_length}, "
                f"avail={len(avail)} — CONSIDER declare_surge"
            )

        if step_history:
            lines.append("")
            lines.append(f"RECENT ACTIONS (last {min(3, len(step_history))} steps):")
            for h in step_history[-3:]:
                lines.append(f"  Step {h.get('step',0)}: {h.get('action_type','?')} → reward={h.get('reward', 0.0):.4f}")

        lines.append("")
        lines.append("Your action (JSON only, preceded by <reasoning>...</reasoning>):")
        return "\n".join(lines)

class ActionExtractor:

    @staticmethod
    def extract(text: str) -> Optional[Dict[str, Any]]:
        strategies = [
            ActionExtractor._extract_code_block,
            ActionExtractor._extract_bare_json,
            ActionExtractor._extract_after_reasoning,
            ActionExtractor._extract_first_json_object,
        ]
        for strategy in strategies:
            result = strategy(text)
            if result and isinstance(result, dict) and "action_type" in result:
                return ActionExtractor._normalise(result)
        return None

    @staticmethod
    def _extract_code_block(text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _extract_bare_json(text: str) -> Optional[Dict[str, Any]]:
        matches = list(re.finditer(r"\{[^{}]*\"action_type\"[^{}]*\}", text, re.DOTALL))
        if matches:
            try:
                return json.loads(matches[-1].group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _extract_after_reasoning(text: str) -> Optional[Dict[str, Any]]:
        stripped = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = -1
        return None

    _HOSPITAL_NAME_TO_ID: Dict[str, str] = {
        "sassoon general": "H1", "sassoon": "H1",
        "kem hospital": "H2", "kem": "H2",
        "ruby hall clinic": "H3", "ruby hall": "H3", "ruby": "H3",
        "jehangir hospital": "H4", "jehangir": "H4",
        "deenanath mangeshkar": "H5", "deenanath": "H5", "mangeshkar": "H5",
        "bharati hospital": "H6", "bharati": "H6",
        "aditya birla memorial": "H7", "aditya birla": "H7", "birla": "H7",
        "columbia asia": "H8", "columbia": "H8",
    }

    @staticmethod
    def _resolve_hospital_id(val: str) -> str:
        """Convert hospital names to canonical IDs (H1-H8). Pass through H1-H8 unchanged."""
        if not val:
            return val
        stripped = val.strip()
                            
        if stripped.upper() in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"):
            return stripped.upper()
                                            
        lookup = stripped.lower()
        return ActionExtractor._HOSPITAL_NAME_TO_ID.get(lookup, stripped)

    @staticmethod
    def _normalise(action: Dict[str, Any]) -> Dict[str, Any]:
        if "unit_type" in action:
            action["unit_type"] = str(action["unit_type"]).upper()
        if "assigned_priority" in action:
            p = str(action["assigned_priority"]).upper()
            if p not in ("P1", "P2", "P3"):
                p = "P2"
            action["assigned_priority"] = p
                                                                                  
        for tag_key in ("triage_tag", "tag"):
            if tag_key in action:
                tag = str(action[tag_key]).strip().title()
                if tag not in ("Immediate", "Delayed", "Minimal", "Expectant"):
                    tag = "Delayed"
                action["triage_tag"] = tag                               
                if "tag" in action:
                    del action["tag"]
                break
                                                                       
        for hkey in ("hospital_id", "to_hospital_id", "from_hospital_id",
                     "new_hospital_id", "destination_hospital"):
            if hkey in action and action[hkey]:
                action[hkey] = ActionExtractor._resolve_hospital_id(str(action[hkey]))
                                                       
        at = action.get("action_type", "noop")
        ACTION_TYPE_MAP = {
            "declare_surge": "escalate",
            "surge_declaration": "escalate",
            "crew_swap": "reposition",
            "preposition": "reposition",
            "hospital_route_selected": "dispatch",
            "hospital_select": "dispatch",
            "unit_assignment": "dispatch",
            "unit_type_match": "dispatch",
            "triage_and_dispatch": "dispatch",
        }
        if at in ACTION_TYPE_MAP:
            action["action_type"] = ACTION_TYPE_MAP[at]
                                                                  
        if action.get("action_type") == "reposition" and "target_zone" in action:
            action["zone"] = action.pop("target_zone")
                                                                      
        if action.get("action_type") == "reroute" and "new_hospital_id" in action:
            action["hospital_id"] = action.pop("new_hospital_id")
                                                                                                    
        if action.get("action_type") == "transfer":
            if "patient_id" in action and "incident_id" not in action:
                action["incident_id"] = action.pop("patient_id")
            if "dest_hospital_id" in action and "to_hospital_id" not in action:
                action["to_hospital_id"] = action.pop("dest_hospital_id")
                                                                     
        if action.get("action_type") == "request_mutual_aid" and "n_units" in action:
            action["units_requested"] = action.pop("n_units")
                                                                                      
        known_keys = {
            "action_type", "incident_id", "unit_id", "unit_type", "hospital_id",
            "assigned_priority", "triage_tag", "patient_id",
            "to_hospital_id", "from_hospital_id", "new_hospital_id",
            "dest_hospital_id", "units_requested", "n_units", "zone",
            "target_zone", "reason", "coordinate_agencies",
            "cath_lab_activated", "stroke_unit_notified",
            "travel_time_min", "congestion_hotspots", "route_path",
        }
        return {k: v for k, v in action.items() if k in known_keys}

class RuleBasedFallback:

    def decide(self, obs: ParsedObservation, step_history: List[Dict]) -> Dict[str, Any]:
        task = obs.task_id

        if task == "task6_prepositioning":
            return self._decide_preposition(obs)

        if task == "task8_transfer_cascade":
            return self._decide_transfer(obs)

        if not obs.surge_declared and obs.surge_warranted():
            if task in ("task9_surge", "task7_mci_start"):
                                                                               
                return {"action_type": "escalate", "reason": "surge_conditions_met"}

        fatigued = [u for u in obs.units if u.fatigue_flag and u.status != "available"]
        if fatigued and task in ("task4_multi_incident", "task7_mci_start", "task9_surge"):
                                                                                  
            pass                            

        if task in ("task7_mci_start", "task9_surge"):
            avail = len(obs.available_units)
            if obs.queue_length > avail + 2:
                return {
                    "action_type": "request_mutual_aid",
                    "units_requested": min(4, obs.queue_length - avail),              
                }

        if task == "task5_dynamic_rerouting":
            reroute = self._check_reroute_needed(obs)
            if reroute:
                return reroute

        if task in ("task7_mci_start", "task9_surge"):
            for inc in obs.incidents:
                                                              
                if inc.severity_hint in ("P1", "P2", "P3"):
                    tag_result = self._decide_mci_tag(obs, inc)
                    if tag_result:
                        return tag_result

        for inc in obs.incidents:
            unit = obs.best_unit_for_incident(inc)
            if not unit:
                continue
            hospital = obs.best_hospital_for_incident(inc)
            if not hospital:
                continue

            action: Dict[str, Any] = {
                "action_type": "dispatch",
                "incident_id": inc.incident_id,
                "unit_id": unit.unit_id,
                "unit_type": unit.unit_type,
                "hospital_id": hospital.hospital_id,
                "assigned_priority": inc.severity_hint,
            }
            return action

        return {
            "action_type": "noop",
            "reason": "no_available_units_or_incidents",
            "p1_incidents_unaddressed": sum(
                1 for i in obs.incidents if i.severity_hint == "P1"
            ),
        }

    def _decide_preposition(self, obs: ParsedObservation) -> Dict[str, Any]:
        if not obs.demand:
                                                                     
            for unit in obs.available_units:
                return {"action_type": "reposition", "unit_id": unit.unit_id, "zone": "Z3"}
            return {"action_type": "noop", "reason": "no_demand_forecast"}
        top_zones = [z for z, _ in obs.demand.top_zones(6)]
        for unit in obs.available_units:
            if unit.zone_id not in top_zones:
                target = top_zones[0] if top_zones else "Z1"
                return {
                                                                                                
                    "action_type": "reposition",
                    "unit_id": unit.unit_id,
                    "zone": target,
                }
        return {"action_type": "noop", "reason": "all_units_positioned"}

    def _decide_transfer(self, obs: ParsedObservation) -> Dict[str, Any]:
        if not obs.incidents:
            return {"action_type": "noop", "reason": "no_pending_transfers"}
        inc = obs.incidents[0]
        hospital = obs.best_hospital_for_incident(inc)
        if hospital:
            return {
                "action_type": "transfer",
                                                                                                  
                "incident_id": inc.incident_id,
                "to_hospital_id": hospital.hospital_id,
            }
        return {"action_type": "noop", "reason": "no_specialist_hospital_available"}

    def _check_reroute_needed(self, obs: ParsedObservation) -> Optional[Dict[str, Any]]:
        for unit in obs.units:
            if unit.status in ("dispatched", "en_route", "transporting") and unit.comm_ok:
                alt = obs.best_hospital_for_incident(
                    IncidentSummary(
                        incident_id="reroute_check",
                        symptom_description="",
                        zone_id=unit.zone_id,
                        zone_type="metro_core",
                        severity_hint="P1",
                        condition_family="general",
                        requires_extrication=False,
                        requires_police=False,
                        requires_fire=False,
                    )
                )
                if alt:
                    return {
                        "action_type": "reroute",
                        "unit_id": unit.unit_id,
                                                                                         
                        "hospital_id": alt.hospital_id,
                    }
        return None

    def _decide_mci_tag(self, obs: ParsedObservation, inc: IncidentSummary) -> Optional[Dict[str, Any]]:
        tag_map = {"P1": "Immediate", "P2": "Delayed", "P3": "Minimal", "P0": "Expectant"}
        tag = tag_map.get(inc.severity_hint, "Delayed")
                                                              
        return {
            "action_type": "tag",
            "incident_id": inc.incident_id,
            "triage_tag": tag,
        }

class LLMAgent:

    def __init__(
        self,
        model: str = MODEL_NAME,
        api_base: str = API_BASE_URL,
        api_key: str = API_KEY,
        max_tokens: int = MAX_LLM_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        timeout: float = LLM_TIMEOUT,
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._retry = retry or RetryPolicy(max_attempts=3, base_delay=2.0)
        self._fallback = RuleBasedFallback()
        self._extractor = ActionExtractor()

        if _OPENAI_AVAILABLE and api_key:
            self._client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
            )
            self._enabled = True
        else:
            self._client = None
            self._enabled = False
            if not api_key:
                logger.warning(
                    "API_KEY (HF_TOKEN or GROQ_API_KEY) not set — LLM agent disabled, using rule-based fallback."
                )
            elif not _OPENAI_AVAILABLE:
                logger.warning(
                    "openai package not installed — using rule-based fallback."
                )

    def decide(
        self,
        obs: ParsedObservation,
        system_prompt: str,
        user_prompt: str,
        step_history: List[Dict],
    ) -> Tuple[Dict[str, Any], bool, int, float]:
        if not self._enabled:
            t0 = time.monotonic()
            action = self._fallback.decide(obs, step_history)
            return action, False, 0, (time.monotonic() - t0) * 1000.0

        last_exc: Optional[Exception] = None
        for delay in self._retry.delays():
            if delay > 0:
                time.sleep(delay)
            try:
                t0 = time.monotonic()
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                raw_text = response.choices[0].message.content or ""
                tokens_used = getattr(response.usage, "total_tokens", 0) if response.usage else 0

                logger.debug("LLM response (%d tokens, %.0fms):\n%s", tokens_used, latency_ms, raw_text[:400])

                action = self._extractor.extract(raw_text)
                if action:
                    return action, True, tokens_used, latency_ms

                logger.warning("LLM output unparseable — falling back to rule-based")
                break

            except Exception as exc:
                last_exc = exc
                if not self._retry.should_retry(exc):
                    logger.error("LLM call failed (non-retryable): %s", exc)
                    break
                logger.warning("LLM transient error: %s — retrying", exc)

        t0 = time.monotonic()
        action = self._fallback.decide(obs, step_history)
        return action, False, 0, (time.monotonic() - t0) * 1000.0

class EpisodeRunner:

    def __init__(
        self,
        server: ServerClient,
        agent: LLMAgent,
        fallback: RuleBasedFallback,
        mode: AgentMode = AgentMode.HYBRID,
        session_id: Optional[str] = None,
    ) -> None:
        self._server = server
        self._agent = agent
        self._fallback = fallback
        self._mode = mode
        self._session_id = session_id or f"{AGENT_SESSION_PREFIX}-{uuid.uuid4().hex[:8]}"

    def run(
        self,
        task_id: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> EpisodeResult:
        effective_seed = seed if seed is not None else TASK_SEEDS.get(task_id, 42)
        step_metrics: List[StepMetrics] = []
        step_history: List[Dict[str, Any]] = []
        total_tokens = 0
        episode_id = "unknown"

        t0 = time.monotonic()
        try:
            reset_resp = self._server.reset(task_id, seed=effective_seed, session_id=self._session_id)
            obs_raw = reset_resp
            episode_id = reset_resp.get("info", {}).get("episode_id", reset_resp.get("episode_id", "unknown"))
            print(f"START: {task_id}", flush=True)
            print(json.dumps({"event": "[START]", "task_id": task_id,
            "episode_id": episode_id, "seed": effective_seed,
            "session_id": self._session_id}), flush=True)
        except Exception as exc:
            logger.error("Reset failed for %s: %s", task_id, exc)
            return self._failed_result(task_id, effective_seed, episode_id, str(exc))

        system_prompt = PromptBuilder.build_system_prompt(task_id)
        max_s = MAX_STEPS.get(task_id, 50)

        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < max_s:
            step += 1
            obs = ObservationParser.parse(obs_raw)
            user_prompt = PromptBuilder.build_user_prompt(obs, step_history)

            if self._mode == AgentMode.RULE_BASED:
                t_act = time.monotonic()
                action = self._fallback.decide(obs, step_history)
                action_latency_ms = (time.monotonic() - t_act) * 1000.0
                llm_used = False
                tokens = 0
                rule_fallback = True
            elif self._mode == AgentMode.LLM:
                action, llm_used, tokens, action_latency_ms = self._agent.decide(
                    obs, system_prompt, user_prompt, step_history
                )
                if not llm_used:
                    action = self._fallback.decide(obs, step_history)
                rule_fallback = not llm_used
            else:
                action, llm_used, tokens, action_latency_ms = self._agent.decide(
                    obs, system_prompt, user_prompt, step_history
                )
                rule_fallback = not llm_used

            total_tokens += tokens

            error_msg = None
            try:
                step_resp = self._server.step(action, session_id=self._session_id)
                reward = float(step_resp.get("reward", 0.0))
                done = bool(step_resp.get("done", False))
                obs_raw = step_resp
                episode_reward += reward
                print(f"STEP: {step}", flush=True)
                print(json.dumps({"event": "[STEP]", "task_id": task_id,
                "episode_id": episode_id, "step": step,
                "action_type": action.get("action_type"), "reward": round(reward, 6),
                "done": done}), flush=True)

            except Exception as exc:
                logger.warning("Step %d failed: %s", step, exc)
                reward = 0.0
                error_msg = str(exc)
                done = True

            sm = StepMetrics(
                step=step,
                action_type=action.get("action_type", "unknown"),
                action_dict=action,
                reward=round(reward, 6),
                done=done,
                llm_used=llm_used,
                llm_latency_ms=round(action_latency_ms, 2),
                tokens_used=tokens,
                rule_fallback=rule_fallback,
                error=error_msg,
            )
            step_metrics.append(sm)
            step_history.append({
                "step": step,
                "action_type": action.get("action_type"),
                "reward": reward,
            })

            if verbose:
                print(f"STEP: {step}", flush=True)
                llm_flag = "[LLM]" if llm_used else "[RULE]"
                print(
                    f"    Step {step:3d} | {llm_flag} {action.get('action_type','?'):<22} | "
                    f"reward={reward:+.4f}  cumulative={episode_reward:+.4f}"
                )

        try:
            grade_resp = self._server.grade(session_id=self._session_id, force_complete=True)
        except Exception as exc:
            logger.error("Grading failed for %s: %s", task_id, exc)
            return self._failed_result(task_id, effective_seed, episode_id, f"grading: {exc}")

        final_score = float(grade_resp.get("score", grade_resp.get("final_score", 0.0)))
        baseline = float(grade_resp.get("baseline", TASK_BASELINES.get(task_id, 0.0)))
        beats = bool(grade_resp.get("beats_baseline", False))
        delta = round(final_score - baseline, 4)

        components = grade_resp.get("component_breakdown", grade_resp.get("components", {}))
        if isinstance(components, list):
            components = {c["name"]: c.get("weighted", 0.0) for c in components}

        penalties = grade_resp.get("penalties", [])
        p1_survival = float(grade_resp.get("p1_survival_rate", 0.0))
        prot_violations = int(grade_resp.get("protocol_violation_count", grade_resp.get("protocol_violations", 0)))

        print(f"END: {task_id} | score: {final_score:.4f}", flush=True)
        print(json.dumps({"event": "[END]", "task_id": task_id,
        "episode_id": episode_id, "final_score": round(final_score, 4),
        "baseline": round(baseline, 4), "beats_baseline": beats,
        "total_steps": step, "total_llm_tokens": total_tokens,
        "p1_survival_rate": round(p1_survival, 4),
        "protocol_violations": prot_violations}), flush=True)
        grader_status = str(grade_resp.get("status", "unknown"))

        cascade_events = 0
        mutual_aid_req = 0
        surge_declared = False
        try:
            ledger_resp = self._server.get_episode_ledger(session_id=self._session_id)
            ledger = ledger_resp.get("ledger", {})
            cascade_events = len(ledger.get("cascade_events", []))
            mutual_aid_req = int(ledger.get("mutual_aid_requests", 0))
            surge_declared = bool(ledger.get("surge_declared", False))
        except Exception:
            pass

        total_latency_ms = (time.monotonic() - t0) * 1000.0
        avg_step_latency = total_latency_ms / max(step, 1)

        return EpisodeResult(
            task_id=task_id,
            episode_id=episode_id,
            seed=effective_seed,
            session_id=self._session_id,
            mode=self._mode.value,
            final_score=final_score,
            baseline=baseline,
            beats_baseline=beats,
            delta_vs_baseline=delta,
            total_steps=step,
            total_reward=round(episode_reward, 4),
            grader_status=grader_status,
            grader_components=components,
            grader_penalties=penalties,
            p1_survival_rate=p1_survival,
            protocol_violations=prot_violations,
            total_llm_tokens=total_tokens,
            avg_step_latency_ms=round(avg_step_latency, 2),
            cascade_events=cascade_events,
            mutual_aid_requests=mutual_aid_req,
            surge_declared=surge_declared,
            step_metrics=step_metrics,
        )

    def _failed_result(
        self, task_id: str, seed: int, episode_id: str, error: str
    ) -> EpisodeResult:
        return EpisodeResult(
            task_id=task_id,
            episode_id=episode_id,
            seed=seed,
            session_id=self._session_id,
            mode=self._mode.value,
            final_score=0.0,
            baseline=TASK_BASELINES.get(task_id, 0.0),
            beats_baseline=False,
            delta_vs_baseline=-TASK_BASELINES.get(task_id, 0.0),
            total_steps=0,
            total_reward=0.0,
            grader_status="failed",
            grader_components={},
            grader_penalties=[],
            p1_survival_rate=0.0,
            protocol_violations=0,
            total_llm_tokens=0,
            avg_step_latency_ms=0.0,
            cascade_events=0,
            mutual_aid_requests=0,
            surge_declared=False,
            step_metrics=[],
            error_message=error,
        )

class ScoreDashboard:

    WIDTH = 120

    @staticmethod
    def header(mode: str, model: str) -> None:
        print()
        print("═" * ScoreDashboard.WIDTH)
        print("  EMERGI-ENV  ·  National AI Hackathon  ·  India 108/112 EMS Simulation")
        print(f"  Agent Mode: {mode.upper()}  |  Model: {model}  |  Server: {API_BASE_URL}")
        print(f"  Run started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("═" * ScoreDashboard.WIDTH)

    @staticmethod
    def task_start(task_id: str, difficulty: str, seed: int) -> None:
        icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
        print(f"\n{icon}  [{difficulty.upper():6}]  {task_id}  (seed={seed})")
        print(f"  {'─' * (ScoreDashboard.WIDTH - 4)}")

    @staticmethod
    def task_done(result: EpisodeResult) -> None:
        beat_icon = "✅" if result.beats_baseline else "❌"
        bar = ScoreDashboard._score_bar(result.final_score)
        print(f"  {beat_icon}  Score: {result.final_score:.4f}  {bar}  baseline={result.baseline:.2f}  Δ={result.delta_vs_baseline:+.4f}")
        print(f"     Steps: {result.total_steps}  |  P1 survival: {result.p1_survival_rate:.0%}  |  "
              f"Protocol violations: {result.protocol_violations}  |  Tokens: {result.total_llm_tokens}")
        if result.error_message:
            print(f"     ⚠️  Error: {result.error_message}")
        if result.grader_components:
            comp_str = "  ".join(f"{k}={v:.3f}" for k, v in list(result.grader_components.items())[:5])
            print(f"     Components: {comp_str}")

    @staticmethod
    def summary(results: List[EpisodeResult], elapsed: float) -> None:
        print()
        print("═" * ScoreDashboard.WIDTH)
        print("  FINAL RESULTS")
        print("═" * ScoreDashboard.WIDTH)
        print(
            f"  {'Task':<28}  {'Diff':<6}  {'Score':>6}  {'Base':>5}  "
            f"{'Δ':>7}  {'Beat?':<7}  {'Steps':>5}  {'Tokens':>7}  {'Status':<10}"
        )
        print(f"  {'─' * 28}  {'─' * 6}  {'─' * 6}  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 5}  {'─' * 7}  {'─' * 10}")

        total_score = 0.0
        beats_count = 0
        total_tokens = 0
        easy_scores, medium_scores, hard_scores = [], [], []

        for r in results:
            diff = TASK_DIFFICULTIES.get(r.task_id, "?")
            beat_str = "✓ YES" if r.beats_baseline else "✗ NO "
            err_flag = " ⚠" if r.error_message else ""
            print(
                f"  {r.task_id:<28}  {diff:<6}  {r.final_score:>6.4f}  {r.baseline:>5.2f}  "
                f"{r.delta_vs_baseline:>+7.4f}  {beat_str:<7}  {r.total_steps:>5d}  "
                f"{r.total_llm_tokens:>7d}  {r.grader_status:<10}{err_flag}"
            )
            total_score += r.final_score
            total_tokens += r.total_llm_tokens
            if r.beats_baseline:
                beats_count += 1
            if diff == "easy":
                easy_scores.append(r.final_score)
            elif diff == "medium":
                medium_scores.append(r.final_score)
            else:
                hard_scores.append(r.final_score)

        n = len(results)
        avg = total_score / n if n else 0.0
        print(f"  {'─' * 28}  {'─' * 6}  {'─' * 6}  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 5}  {'─' * 7}  {'─' * 10}")
        print(
            f"  {'AGGREGATE':<28}  {'─' * 6}  {avg:>6.4f}  {'─' * 5}  {'─' * 7}  "
            f"{beats_count}/{n} beat  {'─' * 5}  {total_tokens:>7d}"
        )
        print()
        if easy_scores:
            print(f"  Easy   avg: {sum(easy_scores)/len(easy_scores):.4f}  ({len(easy_scores)} tasks)")
        if medium_scores:
            print(f"  Medium avg: {sum(medium_scores)/len(medium_scores):.4f}  ({len(medium_scores)} tasks)")
        if hard_scores:
            print(f"  Hard   avg: {sum(hard_scores)/len(hard_scores):.4f}  ({len(hard_scores)} tasks)")
        print()
        print(f"  Total time: {elapsed:.1f}s  |  Tasks run: {n}  |  Total tokens: {total_tokens}")
        print("═" * ScoreDashboard.WIDTH)
        print()

    @staticmethod
    def _score_bar(score: float, width: int = 30) -> str:
        filled = int(round(score * width))
        bar = "█" * filled + "░" * (width - filled)
        return f"|{bar}|"

    @staticmethod
    def validation_report(resp: Dict[str, Any]) -> None:
        print()
        print("═" * ScoreDashboard.WIDTH)
        print("  OPENENV CONTRACT VALIDATION REPORT")
        print("═" * ScoreDashboard.WIDTH)
        passed = resp.get("passed", False)
        status_icon = "✅ ALL CHECKS PASSED" if passed else "❌ VALIDATION FAILED"
        print(f"  {status_icon}")
        print()
        for check, ok in resp.get("checks", {}).items():
            icon = "✓" if ok else "✗"
            print(f"  {icon}  {check}")
        failures = resp.get("failures", [])
        if failures:
            print()
            print("  FAILURES:")
            for f in failures:
                print(f"    ✗ {f}")
        warnings = resp.get("warnings", [])
        if warnings:
            print()
            print("  WARNINGS:")
            for w in warnings:
                print(f"    ⚠ {w}")
        print("═" * ScoreDashboard.WIDTH)

class BenchmarkRunner:

    def __init__(
        self,
        server: ServerClient,
        mode: AgentMode = AgentMode.HYBRID,
        verbose: bool = False,
    ) -> None:
        self._server = server
        self._mode = mode
        self._verbose = verbose
        self._agent = LLMAgent()
        self._fallback = RuleBasedFallback()

    def run(
        self,
        task_ids: Optional[List[str]] = None,
        seeds: Optional[Dict[str, int]] = None,
    ) -> List[EpisodeResult]:
        tasks = task_ids or TASK_IDS
        seeds = seeds or {}
        results: List[EpisodeResult] = []

        ScoreDashboard.header(self._mode.value, MODEL_NAME)

        for task_id in tasks:
            diff = TASK_DIFFICULTIES.get(task_id, "?")
            seed = seeds.get(task_id, TASK_SEEDS.get(task_id, 42))
            ScoreDashboard.task_start(task_id, diff, seed)

            session_id = f"{AGENT_SESSION_PREFIX}-{task_id}-{uuid.uuid4().hex[:6]}"
            runner = EpisodeRunner(
                server=self._server,
                agent=self._agent,
                fallback=self._fallback,
                mode=self._mode,
                session_id=session_id,
            )

            try:
                result = runner.run(task_id, seed=seed, verbose=self._verbose)
            except Exception as exc:
                logger.exception("EpisodeRunner crashed for %s: %s", task_id, exc)
                result = runner._failed_result(task_id, seed, "crash", str(exc))

            ScoreDashboard.task_done(result)
            results.append(result)

        return results

    def run_batch_grade(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        tasks = task_ids or TASK_IDS
        resp = self._server.grade_batch(tasks)
        return resp

def _results_to_json(results: List[EpisodeResult], elapsed: float) -> Dict[str, Any]:
    per_task = {}
    for r in results:
        per_task[r.task_id] = {
            "task_id": r.task_id,
            "difficulty": TASK_DIFFICULTIES.get(r.task_id, "?"),
            "episode_id": r.episode_id,
            "seed": r.seed,
            "session_id": r.session_id,
            "mode": r.mode,
            "final_score": round(r.final_score, 4),
            "baseline": round(r.baseline, 4),
            "beats_baseline": r.beats_baseline,
            "delta_vs_baseline": round(r.delta_vs_baseline, 4),
            "total_steps": r.total_steps,
            "total_reward": round(r.total_reward, 4),
            "grader_status": r.grader_status,
            "grader_components": {k: round(v, 4) for k, v in r.grader_components.items()},
            "grader_penalties": r.grader_penalties,
            "p1_survival_rate": round(r.p1_survival_rate, 4),
            "protocol_violations": r.protocol_violations,
            "total_llm_tokens": r.total_llm_tokens,
            "avg_step_latency_ms": round(r.avg_step_latency_ms, 2),
            "cascade_events": r.cascade_events,
            "mutual_aid_requests": r.mutual_aid_requests,
            "surge_declared": r.surge_declared,
            "error_message": r.error_message,
            "run_timestamp": r.run_timestamp,
        }

    scores = [r.final_score for r in results]
    beats = [r for r in results if r.beats_baseline]
    agg = sum(scores) / len(scores) if scores else 0.0

    return {
        "metadata": {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "agent_mode": results[0].mode if results else "unknown",
            "elapsed_seconds": round(elapsed, 2),
            "tasks_evaluated": len(results),
        },
        "summary": {
            "aggregate_score": round(agg, 4),
            "tasks_beat_baseline": len(beats),
            "tasks_total": len(results),
            "total_llm_tokens": sum(r.total_llm_tokens for r in results),
            "all_scores_in_range": all(0.0 <= r.final_score <= 1.0 for r in results),
            "easy_avg": round(
                sum(r.final_score for r in results if TASK_DIFFICULTIES.get(r.task_id) == "easy") /
                max(sum(1 for r in results if TASK_DIFFICULTIES.get(r.task_id) == "easy"), 1),
                4,
            ),
            "medium_avg": round(
                sum(r.final_score for r in results if TASK_DIFFICULTIES.get(r.task_id) == "medium") /
                max(sum(1 for r in results if TASK_DIFFICULTIES.get(r.task_id) == "medium"), 1),
                4,
            ),
            "hard_avg": round(
                sum(r.final_score for r in results if TASK_DIFFICULTIES.get(r.task_id) == "hard") /
                max(sum(1 for r in results if TASK_DIFFICULTIES.get(r.task_id) == "hard"), 1),
                4,
            ),
        },
        "per_task": per_task,
    }

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EMERGI-ENV inference agent — runs LLM/rule-based agent across all 9 tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py                                   # all 9 tasks, hybrid mode
  python inference.py --task task1_single_triage        # single task
  python inference.py --tasks task4 task7 task9         # subset
  python inference.py --mode rule_based                 # no LLM
  python inference.py --mode llm                        # LLM only
  python inference.py --seed 12345                      # override all seeds
  python inference.py --validate                        # run /validate then exit
  python inference.py --batch-grade                     # use /grade/batch endpoint
  python inference.py --verbose --output my_results.json
        """,
    )
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--mode", choices=["llm", "rule_based", "hybrid"], default="hybrid")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--output", type=str, default=RESULTS_OUTPUT_PATH)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--batch-grade", action="store_true")
    p.add_argument("--health-check", action="store_true")
    p.add_argument("--server", type=str, default=SERVER_URL)
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--no-output", action="store_true")
    return p

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    global SERVER_URL, MODEL_NAME
    if args.server:
        SERVER_URL = args.server.rstrip("/")
    if args.model:
        MODEL_NAME = args.model

    server = ServerClient(base_url=SERVER_URL)
    
    # Crucial: Wait for server to be healthy before starting benchmark
    print(f"📡 Connecting to EMERGI-ENV server at {SERVER_URL} ...")
    wait_start = time.monotonic()
    healthy = False
    while time.monotonic() - wait_start < 120:
        try:
            h = server.health()
            if h.get("status") in ("ok", "healthy") or h.get("version"):
                healthy = True
                print(f"✅ Server is LIVE (responded in {time.monotonic()-wait_start:.1f}s)")
                break
        except Exception:
            pass
        print(f"⏳ Waiting for server... ({time.monotonic()-wait_start:.0f}s)", end="\r")
        time.sleep(2.0)
    
    if not healthy:
        print(f"\n❌ Server at {SERVER_URL} is UNREACHABLE after 120s. Aborting.", file=sys.stderr)
        return 1

    if args.health_check:
        try:
            health = server.health()
            print(json.dumps(health, indent=2))
            return 0
        except Exception as exc:
            print(f"Health check failed: {exc}", file=sys.stderr)
            return 1

    if args.validate:
        try:
            resp = server.validate()
            ScoreDashboard.validation_report(resp)
            return 0 if resp.get("passed", False) else 1
        except Exception as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            return 1

    if args.batch_grade:
        task_ids = _resolve_task_ids(args)
        print(f"Running /grade/batch for {len(task_ids)} tasks...")
        try:
            resp = server.grade_batch(task_ids)
            print(json.dumps(resp, indent=2, default=str))
            all_ok = resp.get("all_scores_in_range", False)
            agg = resp.get("aggregate_score", 0.0)
            print(f"\nAggregate score: {agg:.4f}  |  All scores in [0,1]: {all_ok}")
            return 0 if all_ok else 1
        except Exception as exc:
            print(f"Batch grade failed: {exc}", file=sys.stderr)
            return 1

    task_ids = _resolve_task_ids(args)
    seeds = {tid: args.seed for tid in task_ids} if args.seed else {}
    mode = AgentMode(args.mode)

    runner = BenchmarkRunner(server=server, mode=mode, verbose=args.verbose)

    t0 = time.monotonic()
    try:
        results = runner.run(task_ids=task_ids, seeds=seeds)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        server.close()
        return 130
    except Exception as exc:
        print(f"BenchmarkRunner crashed: {exc}", file=sys.stderr)
        traceback.print_exc()
        server.close()
        return 1

    elapsed = time.monotonic() - t0
    ScoreDashboard.summary(results, elapsed)

    if not args.no_output:
        output = _results_to_json(results, elapsed)
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"  Results saved → {args.output}")
        except OSError as exc:
            print(f"  ⚠️  Could not write results: {exc}", file=sys.stderr)

    server.close()

    if any(r.final_score == 0.0 and r.error_message for r in results):
        return 1
    return 0

def _resolve_task_ids(args: argparse.Namespace) -> List[str]:
    if args.task:
        full_map = {t.split("_")[0]: t for t in TASK_IDS}
        full_map.update({t: t for t in TASK_IDS})
        tid = full_map.get(args.task, args.task)
        if tid not in TASK_IDS:
            print(f"Unknown task '{args.task}'. Valid: {TASK_IDS}", file=sys.stderr)
            sys.exit(1)
        return [tid]
    if args.tasks:
        full_map = {t.split("_")[0]: t for t in TASK_IDS}
        full_map.update({t: t for t in TASK_IDS})
        resolved = []
        for t in args.tasks:
            tid = full_map.get(t, t)
            if tid not in TASK_IDS:
                print(f"Unknown task '{t}'. Skipping.", file=sys.stderr)
            else:
                resolved.append(tid)
        return resolved or TASK_IDS
    return TASK_IDS

if __name__ == "__main__":
    sys.exit(main())