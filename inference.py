
from __future__ import annotations

"""
EMERGI-ENV Inference Script  ·  OpenEnv Hackathon Phase 1
==========================================================
MANDATORY variables (must be in environment or .env):
    API_BASE_URL   LLM endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     LLM model identifier
    HF_TOKEN       HuggingFace / API key

STDOUT contract (strict):
    [START] task=<id> env=emergi_env model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Rules:
  - [START] emitted before ANY network call (cannot be swallowed by an exception)
  - Every task ALWAYS emits [START] and [END] — even on total failure
  - Script exits with code 0 in ALL cases (non-zero = "unhandled exception" to evaluator)
  - All imports are guarded; script runs on stdlib alone if packages are missing
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


try:
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass


API_BASE_URL: str = (os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME:   str = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN:     str = os.getenv("HF_TOKEN", "")
API_KEY:      str = HF_TOKEN  

SERVER_URL:   str = (os.getenv("SERVER_URL") or "http://127.0.0.1:7860").rstrip("/")
BENCHMARK:    str = "emergi_env"

try:
    MAX_TOKENS = int(os.getenv("MAX_LLM_TOKENS") or "800")
except Exception:
    MAX_TOKENS = 800

try:
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.2")
except Exception:
    TEMPERATURE = 0.2

try:
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT") or "40.0")
except Exception:
    LLM_TIMEOUT = 40.0

try:
    REQ_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT") or "30.0")
except Exception:
    REQ_TIMEOUT = 30.0

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("emergi_env.inference")


try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _httpx = None
    _HAS_HTTPX = False

try:
    from openai import OpenAI as _OpenAI
    try:
        from openai import RateLimitError as _RateLimitError, APITimeoutError as _APITimeoutError
    except ImportError:
        _RateLimitError = Exception
        _APITimeoutError = Exception
    _HAS_OPENAI = True
except ImportError:
    _OpenAI = None
    _RateLimitError = Exception
    _APITimeoutError = Exception
    _HAS_OPENAI = False


import urllib.request
import urllib.error


TASK_IDS: List[str] = [
    "task1_single_triage", "task2_hospital_route", "task3_unit_type",
    "task4_multi_incident", "task5_dynamic_rerouting", "task6_prepositioning",
    "task7_mci_start", "task8_transfer_cascade", "task9_surge",
]
SEEDS: Dict[str, int] = {t: 42 + i for i, t in enumerate(TASK_IDS)}
BASELINES: Dict[str, float] = {
    "task1_single_triage": 0.61, "task2_hospital_route": 0.72, "task3_unit_type": 0.68,
    "task4_multi_incident": 0.44, "task5_dynamic_rerouting": 0.38, "task6_prepositioning": 0.42,
    "task7_mci_start": 0.29, "task8_transfer_cascade": 0.24, "task9_surge": 0.17,
}
MAX_STEPS_MAP: Dict[str, int] = {
    "task1_single_triage": 15, "task2_hospital_route": 15, "task3_unit_type": 10,
    "task4_multi_incident": 30, "task5_dynamic_rerouting": 25, "task6_prepositioning": 20,
    "task7_mci_start": 50, "task8_transfer_cascade": 40, "task9_surge": 60,
}


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: Dict, reward: float, done: bool, error: Optional[str]) -> None:
    try:
        act = json.dumps(action, separators=(",", ":"))
    except Exception:
        act = '{"action_type":"noop"}'
    err = error.replace("\n", " ")[:200] if error else "null"
    print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err}",
          flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = max(0.0, min(1.0, score))
    rw = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rw}",
          flush=True)


def _http_post(url: str, payload: Dict) -> Dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    if _HAS_HTTPX:
        with _httpx.Client(timeout=REQ_TIMEOUT, headers=headers, follow_redirects=True) as c:
            r = c.post(url, content=data)
            r.raise_for_status()
            return r.json()
    else:
        req = urllib.request.Request(url, data=data,
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
            return json.loads(resp.read().decode())

def _http_get(url: str, params: Dict = None) -> Dict:
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    headers = {"Accept": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    if _HAS_HTTPX:
        with _httpx.Client(timeout=REQ_TIMEOUT, headers=headers, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()
    else:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
            return json.loads(resp.read().decode())

def server_health(base: str) -> bool:
    try:
        d = _http_get(f"{base}/health")
        return bool(d.get("status") in ("ok", "healthy") or d.get("version"))
    except Exception:
        return False

def server_reset(base: str, task_id: str, seed: int, sid: str) -> Dict:
    return _http_post(f"{base}/reset",
                      {"task_id": task_id, "seed": seed, "session_id": sid})

def server_step(base: str, action: Dict, sid: str) -> Dict:
    return _http_post(f"{base}/step",
                      {"action": action, "session_id": sid})

def server_grade(base: str, sid: str) -> Dict:
    return _http_post(f"{base}/grade",
                      {"session_id": sid, "force_complete": True})


COND_UNIT: Dict[str, str] = {
    "stemi": "MICU", "cardiac_arrest": "MICU", "polytrauma": "MICU",
    "blast_injury": "MICU", "eclampsia": "MICU", "postpartum": "MICU",
    "severe_tbi": "MICU", "tension_pneumothorax": "MICU",
    "ischemic_stroke": "ALS", "hemorrhagic_stroke": "ALS",
    "seizure": "ALS", "respiratory_failure": "ALS", "anaphylaxis": "ALS",
    "burns_minor": "BLS", "general": "BLS", "trauma_minor": "BLS",
}
COND_SPEC: Dict[str, str] = {
    "stemi": "cardiology", "cardiac_arrest": "cardiology",
    "ischemic_stroke": "neurology", "hemorrhagic_stroke": "neurology",
    "polytrauma": "trauma", "blast_injury": "trauma",
    "eclampsia": "obstetrics",
}

def _infer_unit(desc: str, cond: str) -> str:
    text = (desc + " " + cond).lower()
    for k, v in COND_UNIT.items():
        if k.replace("_", " ") in text or k in text:
            return v
    return "ALS"

def _infer_spec(desc: str, cond: str) -> Optional[str]:
    text = (desc + " " + cond).lower()
    for k, v in COND_SPEC.items():
        if k.replace("_", " ") in text or k in text:
            return v
    return None

def rule_action(obs: Dict, task_id: str) -> Dict:
    try:
        o = obs.get("observation", obs) or {}
        incidents = sorted(
            o.get("incident_queue", []) or [],
            key=lambda i: {"P1": 0, "P2": 1, "P3": 2}.get(
                i.get("severity", i.get("severity_hint", "P2")), 9)
        )
        fleet = o.get("fleet_status", []) or []
        hospitals = o.get("hospital_network", []) or []
        demand = o.get("demand_forecast") or {}
        if isinstance(demand, dict) and "heatmap" in demand:
            demand = demand["heatmap"]
        surge_declared = bool(o.get("surge_declared", False))

        avail_units = [u for u in fleet
                       if u.get("status") == "available"
                       and not u.get("fatigue_flag")
                       and u.get("comm_ok", True)]
        accept_hosps = [h for h in hospitals
                        if not h.get("diverted", h.get("on_diversion", False))
                        and float(h.get("er_load", h.get("er_occupancy", 0))) < 0.90]

        
        if task_id == "task6_prepositioning":
            if demand and avail_units:
                top_zone = max(demand, key=lambda z: demand[z])
                u = avail_units[0]
                if u.get("zone", u.get("zone_id", "")) != top_zone:
                    return {"action_type": "reposition",
                            "unit_id": u["unit_id"], "zone": top_zone}
            return {"action_type": "noop", "reason": "positioned"}

        if task_id == "task8_transfer_cascade":
            if incidents and accept_hosps:
                return {"action_type": "transfer",
                        "incident_id": incidents[0].get("incident_id", ""),
                        "to_hospital_id": accept_hosps[0].get("id",
                            accept_hosps[0].get("hospital_id", "H1"))}
            return {"action_type": "noop", "reason": "no_transfer"}

        
        diverted = sum(1 for h in hospitals if h.get("diverted", h.get("on_diversion")))
        if not surge_declared and (diverted >= 3 or len(incidents) >= 10):
            if task_id in ("task9_surge", "task7_mci_start"):
                return {"action_type": "escalate", "reason": "surge"}

        
        if task_id in ("task7_mci_start", "task9_surge"):
            if len(incidents) > len(avail_units) + 2:
                return {"action_type": "request_mutual_aid",
                        "units_requested": min(4, len(incidents) - len(avail_units))}

        
        if task_id in ("task7_mci_start", "task9_surge"):
            tag_map = {"P1": "Immediate", "P2": "Delayed", "P3": "Minimal", "P0": "Expectant"}
            for inc in incidents:
                sev = inc.get("severity", inc.get("severity_hint", "P2"))
                return {"action_type": "tag",
                        "incident_id": inc.get("incident_id", ""),
                        "triage_tag": tag_map.get(sev, "Delayed")}

        
        if task_id == "task5_dynamic_rerouting":
            for u in fleet:
                if u.get("status") in ("dispatched", "en_route", "transporting"):
                    if accept_hosps:
                        return {"action_type": "reroute",
                                "unit_id": u["unit_id"],
                                "hospital_id": accept_hosps[0].get("id",
                                    accept_hosps[0].get("hospital_id", "H1"))}

        
        for inc in incidents:
            if not avail_units:
                break
            desc = inc.get("description", inc.get("symptom_description", ""))
            cond = inc.get("condition", inc.get("condition_family", "general"))
            needed_type = _infer_unit(desc, cond)
            spec = _infer_spec(desc, cond)
            rank = {"MICU": 3, "ALS": 2, "BLS": 1}
            need_rank = rank.get(needed_type, 1)
            cands = [u for u in avail_units if rank.get(u.get("unit_type", "BLS"), 0) >= need_rank]
            if not cands:
                cands = avail_units
            zone = inc.get("zone", inc.get("zone_id", ""))
            same_zone = [u for u in cands if u.get("zone", u.get("zone_id", "")) == zone]
            unit = same_zone[0] if same_zone else cands[0]

            hosps = accept_hosps or hospitals
            if not hosps:
                continue
            def _score(h: Dict) -> float:
                has_spec = 1.0 if (spec and any(spec.lower() in s.lower()
                                                for s in h.get("specialties", []))) else 0.3
                cap = max(0.0, 1.0 - float(h.get("er_load", h.get("er_occupancy", 0))))
                return 0.5 * has_spec + 0.3 * cap
            hosp = max(hosps, key=_score)
            return {
                "action_type": "dispatch",
                "incident_id": inc.get("incident_id", ""),
                "unit_id": unit["unit_id"],
                "unit_type": unit.get("unit_type", "ALS"),
                "hospital_id": hosp.get("id", hosp.get("hospital_id", "H1")),
                "assigned_priority": inc.get("severity", inc.get("severity_hint", "P2")),
            }

        return {"action_type": "noop", "reason": "no_actionable"}
    except Exception as exc:
        logger.warning("rule_action error: %s", exc)
        return {"action_type": "noop", "reason": "rule_error"}


_TASK_HINTS: Dict[str, str] = {
    "task1_single_triage":
        "One incident. Dispatch correct unit_type (BLS/ALS/MICU) to best hospital_id (H1-H8).\n"
        "H1=Sassoon(trauma/cardiology/neurology) H2=KEM(trauma) H3=RubyHall(cath_lab/stroke) "
        "H4=Jehangir(burns) H5=Deenanath(cardiology/neurology) H6=Bharati(trauma) "
        "H7=AdityaBirla(cardiology/stroke) H8=Columbia(ortho/burns)\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H1","unit_type":"MICU"}',
    "task2_hospital_route":
        "Pick best hospital: specialty 50%, capacity 30%, travel 20%. Never route to diverted.\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H3"}',
    "task3_unit_type":
        "MICU=STEMI/cardiac_arrest/polytrauma/eclampsia. ALS=stroke/respiratory. BLS=minor.\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","unit_type":"MICU","hospital_id":"H1"}',
    "task4_multi_incident":
        "5-8 calls, 3 ambulances. P1 has 3x weight. Dispatch highest priority first.\n"
        'JSON: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H1"}',
    "task5_dynamic_rerouting":
        "If hospital diverted, reroute active unit. Else dispatch.\n"
        'JSON: {"action_type":"reroute","unit_id":"Ux","hospital_id":"H3"}',
    "task6_prepositioning":
        "No active calls. Move ambulances to high-demand zones using demand_forecast.\n"
        'JSON: {"action_type":"reposition","unit_id":"Ux","zone":"Z3"}',
    "task7_mci_start":
        "20-40 victims. First tag with START triage, then dispatch Immediate patents.\n"
        "NEVER tag P1 as Expectant (-0.50 penalty).\n"
        'Tag: {"action_type":"tag","incident_id":"INCx","triage_tag":"Immediate"}\n'
        'Dispatch: {"action_type":"dispatch","incident_id":"INCx","unit_id":"Ux","hospital_id":"H1"}',
    "task8_transfer_cascade":
        "ICU patients need specialist care. Transfer to correct hospital.\n"
        'JSON: {"action_type":"transfer","incident_id":"INCx","to_hospital_id":"H3"}',
    "task9_surge":
        "3 MCIs. Step1:escalate. Step2:request_mutual_aid. Step3:tag. Step4:dispatch.\n"
        '{"action_type":"escalate","reason":"surge"} | {"action_type":"request_mutual_aid","units_requested":4}',
}
_SYSTEM = (
    "You are an expert emergency medical dispatcher for India's 108/112 EMS.\n"
    "RULES: 1)STEMI/cardiac→MICU→cath-lab. 2)Stroke→ALS/MICU→neurology. "
    "3)Trauma→Level-1. 4)NEVER route to diverted(-0.30). "
    "5)queue>=10 or diverted>=3→escalate first. "
    "Reply with <reasoning>…</reasoning> then ONE JSON action."
)

def _extract_action(text: str) -> Optional[Dict]:
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r'(\{[^{}]*"action_type"[^{}]*\})',
    ]:
        for m in re.finditer(pattern, text, re.DOTALL):
            try:
                d = json.loads(m.group(1))
                if "action_type" in d:
                    return d
            except Exception:
                pass
    s = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
    try:
        d = json.loads(s)
        if "action_type" in d:
            return d
    except Exception:
        pass
    return None

def _build_user_prompt(obs: Dict, task_id: str, history: List[Dict]) -> str:
    o = obs.get("observation", obs) or {}
    incidents = (o.get("incident_queue") or [])[:8]
    fleet = (o.get("fleet_status") or [])[:12]
    hospitals = o.get("hospital_network") or []
    step = o.get("step", 0)
    queue_len = len(o.get("incident_queue") or [])
    avail = sum(1 for u in fleet if u.get("status") == "available")
    surge = bool(o.get("surge_declared", False))

    lines = [
        f"=== Step {step} | Task: {task_id} | Queue: {queue_len} ===",
        _TASK_HINTS.get(task_id, ""), "",
    ]
    if incidents:
        lines.append("INCIDENTS:")
        for inc in incidents:
            sev = inc.get("severity", inc.get("severity_hint", "P2"))
            iid = inc.get("incident_id", "")
            zone = inc.get("zone", inc.get("zone_id", ""))
            desc = inc.get("description", inc.get("symptom_description", ""))[:80]
            lines.append(f"  [{sev}] {iid} zone={zone}")
            lines.append(f"    {desc}")

    lines.append(f"\nFLEET ({avail} avail / {len(fleet)} total):")
    for u in fleet:
        st = u.get("status", "?")
        ut = u.get("unit_type", "?")
        uid = u.get("unit_id", "")
        zone = u.get("zone", u.get("zone_id", ""))
        fat = " [FATIGUED]" if u.get("fatigue_flag") else ""
        lines.append(f"  {uid} ({ut}) status={st} zone={zone}{fat}")

    lines.append("\nHOSPITALS:")
    for h in hospitals:
        hid = h.get("id", h.get("hospital_id", ""))
        zone = h.get("zone", h.get("zone_id", ""))
        er = float(h.get("er_load", h.get("er_occupancy", 0)))
        div = " *** DIVERTED ***" if h.get("diverted", h.get("on_diversion")) else ""
        specs = ",".join(h.get("specialties", [])[:3])
        lines.append(f"  {hid} zone={zone} ER={er:.0%} specs={specs}{div}")

    if surge:
        lines.append("\n*** SURGE DECLARED ***")

    if history:
        lines.append(f"\nLAST {min(3,len(history))} STEPS:")
        for h in history[-3:]:
            lines.append(f"  step={h['step']} action={h['action']} reward={h['reward']:.2f}")

    lines.append("\nReply: <reasoning>…</reasoning> then JSON:")
    return "\n".join(lines)

_llm_client = None

def _init_llm() -> None:
    global _llm_client
    if not _HAS_OPENAI:
        logger.warning("openai not installed — rule-based fallback")
        return
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — rule-based fallback")
        return
    try:
        
        _llm_client = _OpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL,
            timeout=LLM_TIMEOUT,
        )
    except Exception as exc:
        logger.error("OpenAI client init failed: %s", exc)

def _llm_decide(obs: Dict, task_id: str, history: List[Dict]) -> Optional[Dict]:
    if _llm_client is None:
        return None
    try:
        system = _SYSTEM + "\n" + _TASK_HINTS.get(task_id, "")
        user = _build_user_prompt(obs, task_id, history)
        for attempt in range(3):
            try:
                resp = _llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": system},
                               {"role": "user",   "content": user}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=False,
                )
                text = (resp.choices[0].message.content or "").strip()
                return _extract_action(text)
            except (_RateLimitError, _APITimeoutError) as exc:
                time.sleep(2 ** attempt)
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                break
    except Exception as exc:
        logger.error("LLM wrapper error: %s", exc)
    return None


_HOSP_MAP = {
    "sassoon": "H1", "kem": "H2", "ruby hall": "H3", "ruby": "H3",
    "jehangir": "H4", "deenanath": "H5", "bharati": "H6",
    "aditya birla": "H7", "columbia": "H8",
}
_TYPE_REMAP = {
    "declare_surge": "escalate", "surge_declaration": "escalate",
    "preposition": "reposition", "hospital_select": "dispatch",
    "unit_assignment": "dispatch", "unit_type_match": "dispatch",
    "triage_and_dispatch": "dispatch",
}

def _resolve_hosp(v: str) -> str:
    s = str(v).strip()
    if s.upper() in ("H1","H2","H3","H4","H5","H6","H7","H8"):
        return s.upper()
    return _HOSP_MAP.get(s.lower(), s)

def normalise(a: Dict) -> Dict:
    try:
        a["action_type"] = _TYPE_REMAP.get(a.get("action_type","noop"),
                                            a.get("action_type","noop"))
        if "unit_type" in a:
            a["unit_type"] = str(a["unit_type"]).upper()
        for tk in ("triage_tag", "tag"):
            if tk in a:
                tag = str(a[tk]).strip().title()
                if tag not in ("Immediate","Delayed","Minimal","Expectant"):
                    tag = "Delayed"
                a["triage_tag"] = tag
                if "tag" in a:
                    del a["tag"]
                break
        for hk in ("hospital_id","to_hospital_id","from_hospital_id","new_hospital_id"):
            if hk in a:
                a[hk] = _resolve_hosp(a[hk])
        if a.get("action_type") == "reposition" and "target_zone" in a:
            a["zone"] = a.pop("target_zone")
        if a.get("action_type") == "transfer":
            if "patient_id" in a and "incident_id" not in a:
                a["incident_id"] = a.pop("patient_id")
        if "assigned_priority" in a:
            p = str(a["assigned_priority"]).upper()
            a["assigned_priority"] = p if p in ("P1","P2","P3") else "P2"
        keep = {"action_type","incident_id","unit_id","unit_type","hospital_id",
                "assigned_priority","triage_tag","to_hospital_id","from_hospital_id",
                "units_requested","zone","reason","coordinate_agencies","cath_lab_activated"}
        return {k: v for k, v in a.items() if k in keep}
    except Exception:
        return {"action_type": "noop"}


def wait_for_server(base: str, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    first = True
    while time.monotonic() < deadline:
        if server_health(base):
            return True
        if first:
            print(f"[INFO] Waiting for server at {base}...", flush=True)
            first = False
        time.sleep(3.0)
    return False


def run_episode(base: str, task_id: str, server_ok: bool) -> None:
    
    seed = SEEDS.get(task_id, 42)
    sid  = f"agent-{task_id}-{uuid.uuid4().hex[:6]}"
    max_s = MAX_STEPS_MAP.get(task_id, 30)
    rewards: List[float] = []
    history: List[Dict]  = []
    steps  = 0
    score  = 0.0
    success = False

    
    log_start(task_id)

    if not server_ok:
        
        log_step(1, {"action_type": "noop"}, 0.0, True, "server_unreachable")
        log_end(False, 1, 0.0, [0.0])
        return

    try:
        raw = server_reset(base, task_id, seed, sid)

        for step in range(1, max_s + 1):
            steps = step
            error_msg: Optional[str] = None
            reward = 0.0
            done   = False

            action: Optional[Dict] = None
            try:
                action = _llm_decide(raw, task_id, history)
            except Exception:
                pass
            if action is None:
                action = rule_action(raw, task_id)
            try:
                action = normalise(action)
            except Exception:
                action = {"action_type": "noop"}

            try:
                resp   = server_step(base, action, sid)
                reward = float(resp.get("reward", 0.0))
                done   = bool(resp.get("done",   False))
                raw    = resp
            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")[:150]
                done = True

            rewards.append(reward)
            log_step(step, action, reward, done, error_msg)
            history.append({"step": step, "action": action.get("action_type","?"),
                            "reward": reward})
            if done:
                break

        try:
            gr    = server_grade(base, sid)
            score = float(gr.get("score", gr.get("final_score", 0.0)))
            score = max(0.0, min(1.0, score))
            success = score >= BASELINES.get(task_id, 0.0)
        except Exception:
            score = min(1.0, sum(rewards) / max(len(rewards), 1)) if rewards else 0.0

    except Exception as exc:
        logger.error("Episode error for %s: %s", task_id, exc)
        if not rewards:
            rewards.append(0.0)

    finally:
        
        log_end(success, steps, score, rewards)


def main() -> int:
    try:
        
        tasks = TASK_IDS
        try:
            argv = sys.argv[1:]
            if "--task" in argv:
                idx = argv.index("--task")
                if idx + 1 < len(argv):
                    t = argv[idx + 1]
                    if t in TASK_IDS:
                        tasks = [t]
            elif "--tasks" in argv:
                idx = argv.index("--tasks")
                selected = []
                for arg in argv[idx+1:]:
                    if arg.startswith("--"):
                        break
                    if arg in TASK_IDS:
                        selected.append(arg)
                if selected:
                    tasks = selected
        except Exception:
            tasks = TASK_IDS

        base = SERVER_URL

        
        _init_llm()

        
        server_proc = None
        if not server_health(base):
            print("[INFO] Server not running — spawning uvicorn...", flush=True)
            try:
                server_proc = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "server.app:app",
                     "--host", "0.0.0.0", "--port", "7860",
                     "--log-level", "warning"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(8)
            except Exception as exc:
                print(f"[WARN] Could not spawn server: {exc}", flush=True)

        server_ok = wait_for_server(base, timeout=90.0)
        if not server_ok:
            print("[WARN] Server unreachable — running in offline mode", flush=True)

        
        for task_id in tasks:
            try:
                run_episode(base, task_id, server_ok)
            except Exception as exc:
                
                logger.error("Unexpected error in task %s: %s", task_id, exc)
                try:
                    log_end(False, 0, 0.0, [0.0])
                except Exception:
                    pass

        
        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass

    except Exception as exc:
        try:
            print(f"[ERROR] Fatal unhandled exception in main: {exc}", file=sys.stderr, flush=True)
        except Exception:
            pass
        
        try:
            log_end(False, 0, 0.0, [0.0])
        except Exception:
            try:
                print("[END] success=false steps=0 score=0.000 rewards=0.00", flush=True)
            except Exception:
                pass

    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        
        os._exit(0)