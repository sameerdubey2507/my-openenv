from __future__ import annotations

"""
EMERGI-ENV Inference Script  ·  OpenEnv Phase 1
================================================
Mandatory compliance (per pre-submission checklist):
  - API_BASE_URL, MODEL_NAME, HF_TOKEN must be set in environment / .env
  - OpenAI client MUST be used for ALL LLM calls
  - Emit exactly [START] / [STEP] / [END] lines to stdout
  - Scores must be in [0.0, 1.0]
  - Runtime < 20 min
  - Must not raise unhandled exceptions
"""

import io
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── UTF-8 stdout ─────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation):
        pass

# ─── Load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ─── Mandatory env vars (checklist requirement) ───────────────────────────────
# These MUST be defined: API_BASE_URL, MODEL_NAME, HF_TOKEN
API_BASE_URL: str   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME:   str   = os.getenv("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN:     str   = os.getenv("HF_TOKEN", "")   # Your HuggingFace / API key
# HF_TOKEN is the single source of truth for authentication
API_KEY:      str   = HF_TOKEN  # alias — always equals HF_TOKEN

SERVER_URL:   str   = os.getenv("SERVER_URL",  "http://127.0.0.1:7860").rstrip("/")
MAX_TOKENS:   int   = int(os.getenv("MAX_LLM_TOKENS",  "800"))
TEMPERATURE:  float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT:  float = float(os.getenv("LLM_TIMEOUT",    "40.0"))
REQ_TIMEOUT:  float = float(os.getenv("REQUEST_TIMEOUT","30.0"))
BENCHMARK:    str   = "emergi_env"

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("emergi_env.inference")

# ─── OpenAI client (MANDATORY per checklist) ──────────────────────────────────
try:
    from openai import OpenAI as _OpenAI, RateLimitError, APITimeoutError
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    _OpenAI = None                    # type: ignore
    RateLimitError = Exception        # type: ignore
    APITimeoutError = Exception       # type: ignore

import httpx

# ─── Task metadata ────────────────────────────────────────────────────────────
TASK_IDS: List[str] = [
    "task1_single_triage",    "task2_hospital_route",    "task3_unit_type",
    "task4_multi_incident",   "task5_dynamic_rerouting", "task6_prepositioning",
    "task7_mci_start",        "task8_transfer_cascade",  "task9_surge",
]

SEEDS: Dict[str, int] = {t: 42 + i for i, t in enumerate(TASK_IDS)}

BASELINES: Dict[str, float] = {
    "task1_single_triage": 0.61, "task2_hospital_route": 0.72, "task3_unit_type": 0.68,
    "task4_multi_incident": 0.44, "task5_dynamic_rerouting": 0.38, "task6_prepositioning": 0.42,
    "task7_mci_start": 0.29, "task8_transfer_cascade": 0.24, "task9_surge": 0.17,
}

MAX_STEPS: Dict[str, int] = {
    "task1_single_triage": 15, "task2_hospital_route": 15, "task3_unit_type": 10,
    "task4_multi_incident": 30, "task5_dynamic_rerouting": 25, "task6_prepositioning": 20,
    "task7_mci_start": 50, "task8_transfer_cascade": 40, "task9_surge": 60,
}

# ─── Condition maps ───────────────────────────────────────────────────────────
COND_UNIT: Dict[str, str] = {
    "stemi": "MICU", "cardiac_arrest": "MICU", "polytrauma": "MICU",
    "blast_injury": "MICU", "crush_syndrome": "MICU", "eclampsia": "MICU",
    "postpartum": "MICU", "cord_prolapse": "MICU", "burns_major": "MICU",
    "severe_tbi": "MICU", "tension_pneumothorax": "MICU", "hypertensive": "MICU",
    "ischemic_stroke": "ALS", "hemorrhagic_stroke": "ALS", "paediatric_stroke": "ALS",
    "seizure": "ALS", "chest_trauma": "ALS", "respiratory_failure": "ALS",
    "anaphylaxis": "ALS", "burns_moderate": "ALS", "hypoglycaemia": "ALS",
    "burns_minor": "BLS", "general": "BLS", "trauma_minor": "BLS",
}

COND_SPEC: Dict[str, str] = {
    "stemi": "cardiology", "cardiac_arrest": "cardiology",
    "ischemic_stroke": "neurology", "hemorrhagic_stroke": "neurology",
    "severe_tbi": "neurosurgery", "polytrauma": "trauma", "blast_injury": "trauma",
    "burns_major": "burns", "eclampsia": "obstetrics", "postpartum": "obstetrics",
}

# ─── Stdout format (checklist compliant) ─────────────────────────────────────
def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: Dict, reward: float, done: bool, error: Optional[str]) -> None:
    act = json.dumps(action, separators=(",", ":"))
    err = error.replace("\n", " ")[:200] if error else "null"
    print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err}",
          flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rw = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rw}",
          flush=True)

# ─── Data classes ─────────────────────────────────────────────────────────────
class Incident:
    __slots__ = ("id", "desc", "zone", "severity", "condition", "multi_agency")
    def __init__(self, id_: str, desc: str, zone: str, sev: str, cond: str, multi: bool):
        self.id = id_; self.desc = desc; self.zone = zone
        self.severity = sev; self.condition = cond; self.multi_agency = multi

    @property
    def unit_type(self) -> str:
        text = (self.desc + " " + self.condition).lower()
        for cond, ut in COND_UNIT.items():
            if cond.replace("_"," ") in text or cond in text:
                return ut
        return "ALS"

    @property
    def specialty(self) -> Optional[str]:
        text = (self.desc + " " + self.condition).lower()
        for cond, sp in COND_SPEC.items():
            if cond.replace("_"," ") in text or cond in text:
                return sp
        return None

class Unit:
    __slots__ = ("id", "type", "status", "zone", "fatigued", "comm_ok")
    def __init__(self, id_: str, utype: str, status: str, zone: str, fat: bool, comm: bool):
        self.id = id_; self.type = utype; self.status = status
        self.zone = zone; self.fatigued = fat; self.comm_ok = comm

    @property
    def available(self) -> bool:
        return self.status == "available" and not self.fatigued and self.comm_ok

class Hospital:
    __slots__ = ("id", "zone", "specs", "er", "icu", "diverted", "cath_lab")
    def __init__(self, id_: str, zone: str, specs: List[str], er: float, icu: float, div: bool):
        self.id = id_; self.zone = zone; self.specs = specs
        self.er = er; self.icu = icu; self.diverted = div
        self.cath_lab = "cath_lab" in specs

    @property
    def accepts(self) -> bool:
        return not self.diverted and self.er < 0.90

    def has(self, spec: str) -> bool:
        return any(spec.lower() in s.lower() for s in self.specs)

    @property
    def cap_score(self) -> float:
        return 0.0 if self.diverted else max(0.0, 1.0 - self.er)

# ─── Observation ──────────────────────────────────────────────────────────────
class Obs:
    def __init__(self, raw: Dict):
        o = raw.get("observation", raw) or {}
        iq = o.get("incident_queue", []) or []

        self.task_id: str  = o.get("task_id", "")
        self.step:    int  = int(o.get("step", 0))
        self.surge_declared: bool = bool(o.get("surge_declared", False))

        self.incidents: List[Incident] = []
        for inc in iq:
            self.incidents.append(Incident(
                id_  = inc.get("incident_id", inc.get("id", "")),
                desc = inc.get("description", inc.get("symptom_description", "")),
                zone = inc.get("zone", inc.get("zone_id", "Z1")),
                sev  = inc.get("severity", inc.get("severity_hint", "P2")),
                cond = inc.get("condition", inc.get("condition_family", "general")),
                multi= bool(inc.get("requires_fire") or inc.get("requires_police")
                            or inc.get("requires_extrication")),
            ))
        self.incidents.sort(key=lambda i: {"P1":0,"P2":1,"P3":2}.get(i.severity, 9))

        self.units: List[Unit] = []
        for u in o.get("fleet_status", []):
            self.units.append(Unit(
                id_    = u.get("unit_id", ""),
                utype  = u.get("unit_type", "BLS"),
                status = u.get("status", "available"),
                zone   = u.get("zone", u.get("zone_id", u.get("last_known_zone", "Z1"))),
                fat    = bool(u.get("fatigue_flag", u.get("status") == "fatigue")),
                comm   = bool(u.get("comm_ok", True)),
            ))

        self.hospitals: List[Hospital] = []
        for h in o.get("hospital_network", []):
            self.hospitals.append(Hospital(
                id_   = h.get("id", h.get("hospital_id", "")),
                zone  = h.get("zone", h.get("zone_id", "Z1")),
                specs = h.get("specialties", []),
                er    = float(h.get("er_load", h.get("er_occupancy", 0.0))),
                icu   = float(h.get("icu_utilisation", 0.0)),
                div   = bool(h.get("diverted", h.get("on_diversion", False))),
            ))

        traffic = o.get("traffic_snapshot", {}) or {}
        self.peak_hour: bool = bool(traffic.get("is_peak_hour", False))

        dem = o.get("demand_forecast") or {}
        if isinstance(dem, dict) and "heatmap" in dem:
            dem = dem["heatmap"]
        self.demand: Dict[str, float] = dem if isinstance(dem, dict) else {}

    @property
    def available_units(self) -> List[Unit]:
        return [u for u in self.units if u.available]

    @property
    def accepting_hospitals(self) -> List[Hospital]:
        return [h for h in self.hospitals if h.accepts]

    @property
    def queue_len(self) -> int:
        return len(self.incidents)

    def surge_needed(self) -> bool:
        diverted = sum(1 for h in self.hospitals if h.diverted)
        avail    = len(self.available_units)
        return diverted >= 3 or self.queue_len >= 10 or (self.queue_len >= 5 and avail == 0)

    def best_unit(self, inc: Incident) -> Optional[Unit]:
        rank = {"MICU": 3, "ALS": 2, "BLS": 1}
        need = rank.get(inc.unit_type, 1)
        cands = [u for u in self.available_units if rank.get(u.type, 0) >= need]
        if not cands:
            cands = self.available_units
        if not cands:
            return None
        same = [u for u in cands if u.zone == inc.zone]
        return same[0] if same else cands[0]

    def best_hospital(self, inc: Incident) -> Optional[Hospital]:
        cands = self.accepting_hospitals or self.hospitals
        if not cands:
            return None
        spec = inc.specialty
        def score(h: Hospital) -> float:
            s = 1.0 if (spec and h.has(spec)) else 0.3
            return 0.5 * s + 0.3 * h.cap_score
        return max(cands, key=score)

# ─── Action normalisation ──────────────────────────────────────────────────────
_TYPE_MAP = {
    "declare_surge": "escalate", "surge_declaration": "escalate",
    "preposition": "reposition", "crew_swap": "reposition",
    "hospital_select": "dispatch", "unit_assignment": "dispatch",
    "unit_type_match": "dispatch", "triage_and_dispatch": "dispatch",
    "hospital_route_selected": "dispatch",
}
_HOSP_MAP: Dict[str, str] = {
    "sassoon general": "H1", "sassoon": "H1",
    "kem hospital": "H2", "kem": "H2",
    "ruby hall clinic": "H3", "ruby hall": "H3", "ruby": "H3",
    "jehangir hospital": "H4", "jehangir": "H4",
    "deenanath mangeshkar": "H5", "deenanath": "H5",
    "bharati hospital": "H6", "bharati": "H6",
    "aditya birla memorial": "H7", "aditya birla": "H7",
    "columbia asia": "H8", "columbia": "H8",
}
_ALLOWED_KEYS = {
    "action_type","incident_id","unit_id","unit_type","hospital_id",
    "assigned_priority","triage_tag","to_hospital_id","from_hospital_id",
    "new_hospital_id","units_requested","zone","reason","coordinate_agencies",
    "cath_lab_activated",
}

def _resolve_hosp(v: str) -> str:
    s = v.strip()
    if s.upper() in ("H1","H2","H3","H4","H5","H6","H7","H8"):
        return s.upper()
    return _HOSP_MAP.get(s.lower(), s)

def normalise(a: Dict) -> Dict:
    a["action_type"] = _TYPE_MAP.get(a.get("action_type","noop"), a.get("action_type","noop"))
    if "unit_type" in a:
        a["unit_type"] = str(a["unit_type"]).upper()
    for tk in ("triage_tag", "tag"):
        if tk in a:
            tag = str(a[tk]).strip().title()
            if tag not in ("Immediate","Delayed","Minimal","Expectant"):
                tag = "Delayed"
            a["triage_tag"] = tag
            if "tag" in a: del a["tag"]
            break
    for hk in ("hospital_id","to_hospital_id","from_hospital_id","new_hospital_id"):
        if hk in a: a[hk] = _resolve_hosp(str(a[hk]))
    if a.get("action_type") == "reposition" and "target_zone" in a:
        a["zone"] = a.pop("target_zone")
    if a.get("action_type") == "reroute" and "new_hospital_id" in a:
        a["hospital_id"] = a.pop("new_hospital_id")
    if a.get("action_type") == "transfer":
        if "patient_id" in a and "incident_id" not in a:
            a["incident_id"] = a.pop("patient_id")
    if "assigned_priority" in a:
        p = str(a["assigned_priority"]).upper()
        a["assigned_priority"] = p if p in ("P1","P2","P3") else "P2"
    return {k: v for k, v in a.items() if k in _ALLOWED_KEYS}

# ─── Extract JSON action from LLM text ───────────────────────────────────────
def extract_action(text: str) -> Optional[Dict]:
    # 1. ```json block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            if "action_type" in d: return d
        except Exception: pass
    # 2. After </reasoning>
    s = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
    try:
        d = json.loads(s)
        if "action_type" in d: return d
    except Exception: pass
    # 3. Inline {…action_type…}
    for m2 in re.finditer(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL):
        try:
            d = json.loads(m2.group())
            if "action_type" in d: return d
        except Exception: pass
    # 4. First JSON object
    depth = start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth < 0: depth, start = 1, i
            else: depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(text[start:i+1])
                    if "action_type" in d: return d
                except Exception: pass
                depth = start = -1
    return None

# ─── Rule-based fallback (zero deps, no LLM needed) ──────────────────────────
def rule_decide(obs: Obs, task_id: str) -> Dict:
    # Task 6: pre-position
    if task_id == "task6_prepositioning":
        if obs.demand and obs.available_units:
            top = max(obs.demand, key=lambda z: obs.demand[z])
            u = obs.available_units[0]
            if u.zone != top:
                return {"action_type": "reposition", "unit_id": u.id, "zone": top}
        return {"action_type": "noop", "reason": "all_positioned"}

    # Task 8: transfer
    if task_id == "task8_transfer_cascade":
        if obs.incidents:
            inc = obs.incidents[0]
            h   = obs.best_hospital(inc)
            if h: return {"action_type": "transfer", "incident_id": inc.id, "to_hospital_id": h.id}
        return {"action_type": "noop", "reason": "no_transfers"}

    # Surge/escalate
    if not obs.surge_declared and obs.surge_needed() and task_id in ("task9_surge","task7_mci_start"):
        return {"action_type": "escalate", "reason": "surge_conditions_met"}

    # Mutual aid
    avail = len(obs.available_units)
    if task_id in ("task7_mci_start","task9_surge") and obs.queue_len > avail + 2:
        return {"action_type": "request_mutual_aid", "units_requested": min(4, obs.queue_len - avail)}

    # MCI tagging
    if task_id in ("task7_mci_start","task9_surge"):
        tag_map = {"P1":"Immediate","P2":"Delayed","P3":"Minimal","P0":"Expectant"}
        for inc in obs.incidents:
            return {"action_type": "tag", "incident_id": inc.id,
                    "triage_tag": tag_map.get(inc.severity, "Delayed")}

    # Task 5: reroute
    if task_id == "task5_dynamic_rerouting":
        for u in obs.units:
            if u.status in ("dispatched","en_route","transporting"):
                alts = obs.accepting_hospitals
                if alts:
                    return {"action_type": "reroute", "unit_id": u.id, "hospital_id": alts[0].id}

    # Standard dispatch
    for inc in obs.incidents:
        unit = obs.best_unit(inc)
        if not unit: continue
        h = obs.best_hospital(inc)
        if not h: continue
        return {
            "action_type": "dispatch",
            "incident_id": inc.id,
            "unit_id":     unit.id,
            "unit_type":   unit.type,
            "hospital_id": h.id,
            "assigned_priority": inc.severity,
        }

    return {"action_type": "noop", "reason": "no_actionable"}

# ─── LLM caller (mandatory OpenAI client per checklist) ──────────────────────
_TASK_HINTS: Dict[str, str] = {
    "task1_single_triage":
        "One incident. Dispatch with correct unit_type (BLS/ALS/MICU) and hospital_id (H1-H8).\n"
        "H1=Sassoon(trauma/cardiology/neurology) H2=KEM(trauma) H3=RubyHall(cath_lab/stroke) "
        "H4=Jehangir(general/burns) H5=Deenanath(cardiology/neurology) H6=Bharati(trauma) "
        "H7=AdityaBirla(cardiology/transplant/stroke) H8=Columbia(ortho/burns)\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H1","unit_type":"MICU"}',
    "task2_hospital_route":
        "Pick best hospital: specialty 50%, capacity 30%, travel 20%. NEVER route to diverted.\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H3"}',
    "task3_unit_type":
        "Select correct unit: MICU=STEMI/cardiac_arrest/polytrauma/eclampsia, ALS=stroke/respiratory, BLS=minor.\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","unit_type":"MICU","hospital_id":"H1"}',
    "task4_multi_incident":
        "5-8 calls, 3 ambulances. P1 has 3x weight. Dispatch highest priority first.\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H1"}',
    "task5_dynamic_rerouting":
        "If hospital diverted, reroute active unit. Otherwise dispatch.\n"
        'JSON: {"action_type":"reroute","unit_id":"Ux","hospital_id":"H3"}',
    "task6_prepositioning":
        "No active calls. Move ambulances to high-demand zones using demand_forecast.\n"
        'JSON: {"action_type":"reposition","unit_id":"Ux","zone":"Z3"}',
    "task7_mci_start":
        "20-40 victims. First tag with START triage, then dispatch Immediate patients.\n"
        "NEVER tag a P1 as Expectant (-0.50 penalty).\n"
        'Tag JSON: {"action_type":"tag","incident_id":"INCx","triage_tag":"Immediate"}\n'
        'Dispatch JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H1"}',
    "task8_transfer_cascade":
        "ICU patients need specialist care. Transfer to correct hospital.\n"
        'JSON: {"action_type":"transfer","incident_id":"INCx","to_hospital_id":"H3"}',
    "task9_surge":
        "3 simultaneous MCIs. Step1: escalate. Step2: request_mutual_aid. Step3: tag. Step4: dispatch.\n"
        '{"action_type":"escalate","reason":"surge"} | {"action_type":"request_mutual_aid","units_requested":4}',
}

_SYSTEM = (
    "You are an expert emergency medical dispatcher for India's 108/112 EMS system.\n"
    "CRITICAL RULES:\n"
    "1. STEMI / cardiac arrest  → MICU → cath-lab hospital.\n"
    "2. Stroke                  → ALS/MICU → neurology hospital.\n"
    "3. Major trauma            → Level-1 trauma centre.\n"
    "4. NEVER route to a hospital marked diverted (penalty -0.30).\n"
    "5. If >3 hospitals diverted OR queue ≥10 → declare SURGE first.\n"
    "6. Request mutual aid when queue > available_units + 2.\n"
    "Respond with <reasoning>…</reasoning> then a single JSON action object."
)


def _build_prompt(obs: Obs, history: List[Dict]) -> str:
    hint = _TASK_HINTS.get(obs.task_id, "")
    lines = [
        f"=== Step {obs.step} | Task: {obs.task_id} | Queue: {obs.queue_len} ===",
        hint, "",
    ]
    if obs.incidents:
        lines.append("INCIDENTS:")
        for inc in obs.incidents[:8]:
            ma = " [MULTI-AGENCY]" if inc.multi_agency else ""
            lines.append(f"  [{inc.severity}] {inc.id} zone={inc.zone} unit_needed={inc.unit_type}{ma}")
            lines.append(f"    {inc.desc[:100]}")

    av = obs.available_units
    lines.append(f"\nFLEET ({len(av)} avail / {len(obs.units)} total):")
    for u in obs.units[:12]:
        flags = ("".join([
            " [NO_COMM]" if not u.comm_ok else "",
            " [FATIGUED]" if u.fatigued else "",
        ]))
        lines.append(f"  {u.id} ({u.type}) status={u.status} zone={u.zone}{flags}")

    lines.append("\nHOSPITALS:")
    for h in obs.hospitals:
        div = " *** DIVERTED ***" if h.diverted else ""
        lines.append(f"  {h.id} zone={h.zone} ER={h.er:.0%} ICU={h.icu:.0%} "
                     f"specs={','.join(h.specs[:3])}{div}")

    if obs.demand:
        top = sorted(obs.demand.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append(f"\nDEMAND: {', '.join(f'{z}={v:.2f}' for z,v in top)}")

    if obs.surge_declared:
        lines.append("\n*** SURGE ALREADY DECLARED ***")
    elif obs.surge_needed():
        lines.append("\n!!! SURGE WARNING — consider escalate !!!")

    if obs.peak_hour:
        lines.append("TRAFFIC: PEAK HOUR (+45% delay)")

    if history:
        lines.append(f"\nLAST {min(3,len(history))} STEPS:")
        for h in history[-3:]:
            lines.append(f"  step={h['step']} action={h['action']} reward={h['reward']:.2f}")

    lines.append("\nRespond with <reasoning>…</reasoning> then JSON action:")
    return "\n".join(lines)


class LLMCaller:
    """
    Wraps the mandatory OpenAI client.
    Always initialised with API_BASE_URL + HF_TOKEN (even if token is exhausted).
    Falls back to None action on any error — caller must use rule-based then.
    """

    def __init__(self) -> None:
        self._client = None
        if not _HAS_OPENAI:
            logger.warning("openai package not found — rule-based fallback will be used.")
            return
        if not HF_TOKEN:
            logger.warning("HF_TOKEN not set — rule-based fallback will be used.")
            return
        # MANDATORY (checklist): initialise OpenAI client with API_BASE_URL + HF_TOKEN
        try:
            self._client = _OpenAI(
                api_key  = HF_TOKEN,      # HF_TOKEN from .env
                base_url = API_BASE_URL,  # API_BASE_URL from .env
                timeout  = LLM_TIMEOUT,
            )
            logger.info("OpenAI client initialised: base_url=%s model=%s", API_BASE_URL, MODEL_NAME)
        except Exception as exc:
            logger.error("OpenAI client init failed: %s", exc)

    def call(self, system: str, user: str) -> Tuple[Optional[Dict], int]:
        """
        Make ONE LLM call. Returns (parsed_action_or_None, tokens_used).
        All errors are caught — never raises.
        """
        if self._client is None:
            return None, 0

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens  = MAX_TOKENS,
                    temperature = TEMPERATURE,
                    stream      = False,
                )
                text   = (resp.choices[0].message.content or "").strip()
                tokens = getattr(resp.usage, "total_tokens", 0) if resp.usage else 0
                action = extract_action(text)
                if action:
                    return action, tokens
                logger.warning("LLM response unparseable — using rule fallback")
                return None, tokens

            except (RateLimitError, APITimeoutError) as exc:
                wait = 2 ** attempt
                logger.warning("LLM transient error (attempt %d): %s — retry in %ds", attempt+1, exc, wait)
                time.sleep(wait)
                last_err = exc
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                last_err = exc
                break  # non-retryable

        logger.warning("LLM gave up after retries (%s) — rule fallback", last_err)
        return None, 0


# ─── HTTP server client ───────────────────────────────────────────────────────
class EnvClient:
    def __init__(self, base: str = SERVER_URL) -> None:
        self.base = base.rstrip("/")
        hdrs: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept":       "application/json",
            "User-Agent":   f"EMERGI-ENV-Inference/{MODEL_NAME}",
        }
        if HF_TOKEN:
            hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
        self._c = httpx.Client(timeout=REQ_TIMEOUT, headers=hdrs, follow_redirects=True)

    def _req(self, method: str, path: str, **kw) -> Dict:
        url = self.base + path
        for attempt in range(3):
            try:
                r = self._c.request(method, url, **kw)
                r.raise_for_status()
                return r.json()
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"Request to {url} failed after retries")

    def health(self)                                   -> Dict: return self._req("GET",  "/health")
    def reset(self, task: str, seed: int, sid: str)   -> Dict:
        return self._req("POST", "/reset", json={"task_id": task, "seed": seed, "session_id": sid})
    def step(self, action: Dict, sid: str)            -> Dict:
        return self._req("POST", "/step",  json={"action": action, "session_id": sid})
    def grade(self, sid: str)                         -> Dict:
        return self._req("POST", "/grade", json={"session_id": sid, "force_complete": True})
    def close(self)                                   -> None: self._c.close()


# ─── Episode runner ───────────────────────────────────────────────────────────
def run_episode(env: EnvClient, llm: LLMCaller, task_id: str) -> None:
    seed     = SEEDS.get(task_id, 42)
    sid      = f"agent-{task_id}-{uuid.uuid4().hex[:6]}"
    max_s    = MAX_STEPS.get(task_id, 30)
    rewards: List[float] = []
    history: List[Dict]  = []
    steps    = 0
    score    = 0.0
    success  = False

    log_start(task_id)

    try:
        raw = env.reset(task_id, seed, sid)

        for step in range(1, max_s + 1):
            steps     = step
            error_msg: Optional[str] = None
            reward    = 0.0
            done      = False

            # Parse observation
            try:
                obs = Obs(raw)
            except Exception as exc:
                error_msg = f"obs_parse_err:{exc}"
                log_step(step, {"action_type": "noop"}, 0.0, True, error_msg)
                rewards.append(0.0)
                break

            # Decide action: try LLM first (mandatory), fall back to rules
            action: Optional[Dict] = None
            try:
                sys_prompt  = _SYSTEM + "\n" + _TASK_HINTS.get(task_id, "")
                user_prompt = _build_prompt(obs, history)
                action, _   = llm.call(sys_prompt, user_prompt)
            except Exception as exc:
                logger.warning("Prompt build / LLM wrapper error step %d: %s", step, exc)

            if action is None:
                action = rule_decide(obs, task_id)

            try:
                action = normalise(action)
            except Exception:
                action = {"action_type": "noop"}

            # Execute step
            try:
                resp   = env.step(action, sid)
                reward = float(resp.get("reward", 0.0))
                done   = bool(resp.get("done",   False))
                raw    = resp
            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")[:200]
                done = True

            rewards.append(reward)
            log_step(step, action, reward, done, error_msg)
            history.append({"step": step, "action": action.get("action_type","?"), "reward": reward})

            if done:
                break

        # Grade episode
        try:
            gr    = env.grade(sid)
            score = float(gr.get("score", gr.get("final_score", 0.0)))
            score = max(0.0, min(1.0, score))
            success = score >= BASELINES.get(task_id, 0.0)
        except Exception as exc:
            logger.warning("Grade failed (%s) — estimating from rewards", exc)
            score   = min(1.0, sum(rewards) / max(len(rewards), 1))
            success = False

    except Exception as exc:
        logger.error("Episode crashed for %s: %s", task_id, exc)
        rewards.append(0.0)

    finally:
        log_end(success, steps, score, rewards)


# ─── Server readiness ─────────────────────────────────────────────────────────
def wait_server(env: EnvClient, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            h = env.health()
            if h.get("status") in ("ok","healthy") or h.get("version"):
                return True
        except Exception:
            pass
        elapsed = time.monotonic() + timeout - deadline
        print(f"[INFO] Waiting for server… ({elapsed:.0f}s/{timeout:.0f}s)", flush=True)
        time.sleep(3.0)
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="EMERGI-ENV Inference")
    ap.add_argument("--task",   type=str, default=None)
    ap.add_argument("--tasks",  nargs="+", default=None)
    ap.add_argument("--server", type=str, default=SERVER_URL)
    ap.add_argument("--seed",   type=int, default=None)
    args = ap.parse_args()

    base_url = (args.server or SERVER_URL).rstrip("/")
    env = EnvClient(base_url)

    # Spawn server if not running
    server_proc = None
    try:
        h = env.health()
        if h.get("status") in ("ok","healthy") or h.get("version"):
            print("[INFO] Server already running.", flush=True)
        else:
            raise Exception("unhealthy")
    except Exception:
        print("[INFO] Server not found — spawning uvicorn…", flush=True)
        try:
            server_proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server.app:app",
                 "--host", "0.0.0.0", "--port", "7860"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(6)
        except Exception as exc:
            print(f"[WARN] Could not spawn server: {exc}", flush=True)

    if not wait_server(env, timeout=120):
        print(f"[ERROR] Server unreachable at {base_url}. Aborting.", file=sys.stderr, flush=True)
        if server_proc:
            server_proc.terminate()
        return 1

    # Initialise mandatory OpenAI client (API_BASE_URL + HF_TOKEN)
    llm = LLMCaller()

    # Resolve task list
    if args.task:
        tasks = [args.task] if args.task in TASK_IDS else TASK_IDS
    elif args.tasks:
        tasks = [t for t in args.tasks if t in TASK_IDS] or TASK_IDS
    else:
        tasks = TASK_IDS

    if args.seed is not None:
        for k in SEEDS: SEEDS[k] = args.seed

    try:
        for task_id in tasks:
            try:
                run_episode(env, llm, task_id)
            except Exception as exc:
                logger.error("Unhandled exception in task %s: %s", task_id, exc)
                log_end(False, 0, 0.0, [])
    except KeyboardInterrupt:
        print("[WARN] Interrupted.", flush=True)
    finally:
        env.close()
        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try: server_proc.kill()
                except Exception: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())