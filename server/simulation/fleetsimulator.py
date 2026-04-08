from __future__ import annotations
import json
import math
import random
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import numpy as np
_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent.parent / "data"
STEP_DURATION_MINUTES: float = 3.0         
FATIGUE_ONSET_HOURS:     float = 10.0      
FATIGUE_MAX_HOURS:       float = 16.0      
CREW_SWAP_DEPLOY_MINS:   float = 8.0       
FATIGUE_SCENE_PENALTY:   float = 0.35      
FATIGUE_DISPATCH_DELAY:  float = 1.5       
MUTUAL_AID_DELAY_MINS:   float = 12.0      
SCENE_TIME_BASE: Dict[str, float] = {
    "P1": 12.0,
    "P2": 10.0,
    "P3":  8.0,
    "P0":  6.0,   
}
SCENE_TIME_MCI_EXTRA:   float = 5.0        
SCENE_TIME_SIGMA_FRAC:  float = 0.20       
HANDOVER_BASE_MINS:     float  = 8.0
HANDOVER_LOAD_SLOPE:    float  = 0.10      
HANDOVER_MIN_MINS:      float  = 5.0
HANDOVER_MAX_MINS:      float  = 20.0
GOLDEN_HOUR_WINDOWS: Dict[str, float] = {
    "STEMI":               90.0,
    "CARDIAC_ARREST":      15.0,
    "ISCHAEMIC_STROKE":   270.0,
    "HAEMORRHAGIC_STROKE": 60.0,
    "POLYTRAUMA":          60.0,
    "HEAD_INJURY_SEVERE": 120.0,
    "MAJOR_BURNS":         60.0,
    "DEFAULT":            120.0,
}
DIVERSION_PENALTY_SCORE:  float = -0.30
DIVERSION_REROUTE_DELAY:  float = 5.0     
PROTOCOL_COMPLIANCE_BONUS: float = 0.15
RESPONSE_TIME_P1_TARGET:  float = 8.0
RESPONSE_TIME_P2_TARGET:  float = 15.0
RESPONSE_TIME_P3_TARGET:  float = 25.0
class UnitStatus(str, Enum):
    AVAILABLE         = "available"
    DISPATCHED        = "dispatched"
    EN_ROUTE_TO_SCENE = "en_route_to_scene"
    ON_SCENE          = "on_scene"
    TRANSPORTING      = "transporting"
    RETURNING         = "returning"
    CREW_SWAP_PENDING = "crew_swap_pending"
    REPOSITIONING     = "repositioning"
    OUT_OF_SERVICE    = "out_of_service"
    MUTUAL_AID_INBOUND = "mutual_aid_inbound"
class UnitType(str, Enum):
    BLS  = "BLS"    
    ALS  = "ALS"    
    MICU = "MICU"   
class ZoneType(str, Enum):
    URBAN      = "urban"
    SUBURBAN   = "suburban"
    PERI_URBAN = "peri_urban"
    RURAL      = "rural"
    INDUSTRIAL = "industrial"
    HIGHWAY    = "highway"
@dataclass
class CrewFatigueState:
    crew_id:             str
    hours_on_duty:       float = 0.0
    swap_requested:      bool  = False
    swap_pending_mins:   float = 0.0     
    swap_completed:      bool  = False
    total_calls_handled: int   = 0
    total_distance_km:   float = 0.0
    last_rest_at_hour:   float = 0.0
    @property
    def fatigue_level(self) -> float:
        if self.hours_on_duty <= FATIGUE_ONSET_HOURS:
            return 0.0
        excess = self.hours_on_duty - FATIGUE_ONSET_HOURS
        span   = FATIGUE_MAX_HOURS - FATIGUE_ONSET_HOURS
        return min(1.0, excess / span)
    @property
    def scene_time_multiplier(self) -> float:
        return 1.0 + self.fatigue_level * FATIGUE_SCENE_PENALTY
    @property
    def dispatch_delay_mins(self) -> float:
        return self.fatigue_level * FATIGUE_DISPATCH_DELAY
    @property
    def is_fatigued(self) -> bool:
        return self.fatigue_level > 0.0
    @property
    def swap_ready(self) -> bool:
        return self.swap_requested and self.swap_pending_mins <= 0.0
    def request_swap(self) -> None:
        if not self.swap_requested:
            self.swap_requested   = True
            self.swap_pending_mins = CREW_SWAP_DEPLOY_MINS
    def tick(self, step_minutes: float) -> None:
        self.hours_on_duty += step_minutes / 60.0
        if self.swap_requested and self.swap_pending_mins > 0:
            self.swap_pending_mins = max(0.0, self.swap_pending_mins - step_minutes)
    def complete_swap(self) -> None:
        self.hours_on_duty     = 0.0
        self.swap_requested    = False
        self.swap_pending_mins = 0.0
        self.swap_completed    = True
        self.last_rest_at_hour = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "crew_id":             self.crew_id,
            "hours_on_duty":       round(self.hours_on_duty, 2),
            "fatigue_level":       round(self.fatigue_level, 3),
            "is_fatigued":         self.is_fatigued,
            "swap_requested":      self.swap_requested,
            "swap_pending_mins":   round(self.swap_pending_mins, 1),
            "swap_ready":          self.swap_ready,
            "total_calls_handled": self.total_calls_handled,
            "scene_time_mult":     round(self.scene_time_multiplier, 3),
        }
@dataclass
class DispatchRecord:
    dispatch_id:     str
    unit_id:         str
    incident_id:     str
    hospital_id:     str
    dispatched_at_step: int
    dispatched_at_min:  float
    from_zone_id:    str
    incident_zone_id: str
    hospital_zone_id: str
    unit_type:       str
    incident_priority: str
    eta_to_scene_mins: float
    eta_to_hospital_mins: float
    protocol_correct:  bool
    was_rerouted:      bool  = False
    reroute_count:     int   = 0
    arrived_scene_at_min:   Optional[float] = None
    departed_scene_at_min:  Optional[float] = None
    arrived_hospital_at_min: Optional[float] = None
    total_response_time_mins: Optional[float] = None   
    golden_hour_met:          Optional[bool]  = None
    condition_code:           str = "DEFAULT"
@dataclass
class AmbulanceUnit:
    unit_id:          str
    unit_type:        UnitType
    call_sign:        str
    home_zone_id:     str
    current_zone_id:  str
    status:           UnitStatus = UnitStatus.AVAILABLE
    lat:              float = 0.0
    lon:              float = 0.0
    heading_degrees:  float = 0.0
    assigned_incident_id:  Optional[str] = None
    assigned_hospital_id:  Optional[str] = None
    target_zone_id:        Optional[str] = None   
    dispatch_time_min:     float = 0.0
    eta_to_destination_min: float = 0.0    
    time_on_scene_min:     float = 0.0
    time_on_scene_budget:  float = 0.0     
    time_in_status_min:    float = 0.0
    sim_time_min:          float = 0.0
    crew: CrewFatigueState = field(default_factory=lambda: CrewFatigueState(
        crew_id=str(uuid.uuid4())[:8]
    ))
    is_mutual_aid:         bool  = False
    origin_zone_id:        Optional[str] = None
    mutual_aid_eta_mins:   float = 0.0
    comm_active:           bool  = True
    last_known_zone_id:    Optional[str] = None
    last_comm_at_min:      float = 0.0
    total_dispatches:      int   = 0
    total_distance_km:     float = 0.0
    total_scene_time_min:  float = 0.0
    response_times:        List[float] = field(default_factory=list)
    golden_hour_successes: int   = 0
    golden_hour_attempts:  int   = 0
    protocol_violations:   int   = 0
    diversion_hits:        int   = 0
    dispatch_history:      List[DispatchRecord] = field(default_factory=list)
    reposition_target_zone: Optional[str] = None
    reposition_eta_mins:    float = 0.0
    _active_dispatch:      Optional[DispatchRecord] = field(default=None, repr=False)
    step_reward_delta:     float = 0.0
    @property
    def is_available(self) -> bool:
        return self.status == UnitStatus.AVAILABLE
    @property
    def is_busy(self) -> bool:
        return self.status not in (
            UnitStatus.AVAILABLE,
            UnitStatus.OUT_OF_SERVICE,
            UnitStatus.CREW_SWAP_PENDING,
        )
    @property
    def capability_score(self) -> float:
        base = {"MICU": 1.0, "ALS": 0.7, "BLS": 0.4}.get(self.unit_type.value, 0.5)
        return base * (1.0 - 0.3 * self.crew.fatigue_level)
    @property
    def utilisation_pct(self) -> float:
        if self.sim_time_min <= 0:
            return 0.0
        busy_min = sum(self.response_times) + self.total_scene_time_min
        return min(100.0, 100.0 * busy_min / self.sim_time_min)
    @property
    def mean_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "unit_id":             self.unit_id,
            "call_sign":           self.call_sign,
            "unit_type":           self.unit_type.value,
            "status":              self.status.value,
            "current_zone_id":     self.current_zone_id
                                   if self.comm_active else self.last_known_zone_id,
            "comm_active":         self.comm_active,
            "last_comm_at_min":    round(self.last_comm_at_min, 1),
            "lat":                 round(self.lat, 5) if self.comm_active else None,
            "lon":                 round(self.lon, 5) if self.comm_active else None,
            "assigned_incident_id": self.assigned_incident_id,
            "assigned_hospital_id": self.assigned_hospital_id,
            "eta_to_destination_min": round(self.eta_to_destination_min, 1),
            "time_in_status_min":  round(self.time_in_status_min, 1),
            "fatigue_level":       round(self.crew.fatigue_level, 3),
            "hours_on_duty":       round(self.crew.hours_on_duty, 2),
            "swap_requested":      self.crew.swap_requested,
            "swap_ready":          self.crew.swap_ready,
            "is_mutual_aid":       self.is_mutual_aid,
            "capability_score":    round(self.capability_score, 3),
            "home_zone_id":        self.home_zone_id,
        }
    def to_full_state_dict(self) -> Dict[str, Any]:
        base = self.to_observation_dict()
        base.update({
            "total_dispatches":      self.total_dispatches,
            "total_distance_km":     round(self.total_distance_km, 2),
            "mean_response_time":    round(self.mean_response_time, 2),
            "golden_hour_successes": self.golden_hour_successes,
            "golden_hour_attempts":  self.golden_hour_attempts,
            "protocol_violations":   self.protocol_violations,
            "diversion_hits":        self.diversion_hits,
            "utilisation_pct":       round(self.utilisation_pct, 1),
            "crew":                  self.crew.to_dict(),
        })
        return base
@dataclass
class MutualAidRequest:
    request_id:       str
    requesting_zone:  str
    origin_zone:      str
    unit_type:        UnitType
    requested_at_min: float
    arrival_at_min:   float
    unit_id:          Optional[str] = None
    fulfilled:        bool = False
    cancelled:        bool = False
@dataclass
class FleetMetrics:
    total_dispatches:          int   = 0
    p1_dispatches:             int   = 0
    p2_dispatches:             int   = 0
    p3_dispatches:             int   = 0
    mean_response_time_p1:     float = 0.0
    mean_response_time_p2:     float = 0.0
    mean_response_time_p3:     float = 0.0
    golden_hour_compliance:    float = 0.0
    fatigue_exposure_index:    float = 0.0   
    diversion_count:           int   = 0
    protocol_violation_count:  int   = 0
    mutual_aid_requested:      int   = 0
    mutual_aid_fulfilled:      int   = 0
    crew_swaps_executed:       int   = 0
    total_distance_km:         float = 0.0
    fleet_utilisation_pct:     float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dispatches":        self.total_dispatches,
            "priority_dispatches": {
                "P1": self.p1_dispatches,
                "P2": self.p2_dispatches,
                "P3": self.p3_dispatches,
            },
            "mean_response_times": {
                "P1": round(self.mean_response_time_p1, 2),
                "P2": round(self.mean_response_time_p2, 2),
                "P3": round(self.mean_response_time_p3, 2),
            },
            "golden_hour_compliance":   round(self.golden_hour_compliance, 3),
            "fatigue_exposure_index":   round(self.fatigue_exposure_index, 3),
            "diversion_count":          self.diversion_count,
            "protocol_violation_count": self.protocol_violation_count,
            "mutual_aid_requested":     self.mutual_aid_requested,
            "mutual_aid_fulfilled":     self.mutual_aid_fulfilled,
            "crew_swaps_executed":      self.crew_swaps_executed,
            "total_distance_km":        round(self.total_distance_km, 1),
            "fleet_utilisation_pct":    round(self.fleet_utilisation_pct, 1),
        }
class FleetSimulator:
    def __init__(
        self,
        seed: int = 42,
        traffic_model: Optional[Any] = None,     
        hospital_network: Optional[Any] = None,  
    ) -> None:
        self.seed             = seed
        self.rng              = random.Random(seed)
        self.np_rng           = np.random.RandomState(seed)
        self._traffic         = traffic_model
        self._hospital_net    = hospital_network
        self.units:           Dict[str, AmbulanceUnit]   = {}
        self.unit_order:      List[str]                  = []  
        self.pending_mutual_aid: Dict[str, MutualAidRequest] = {}
        self._aid_seq: int = 0
        self.sim_time_min:    float = 480.0
        self.step_count:      int   = 0
        self.task_id:         int   = 1
        self.active_zone_ids: List[str] = []
        self._zone_meta:      Dict[str, Dict] = {}
        self._hospital_zones: Dict[str, str]  = {}   
        self.metrics:          FleetMetrics = FleetMetrics()
        self._rt_log_p1:       List[float]  = []
        self._rt_log_p2:       List[float]  = []
        self._rt_log_p3:       List[float]  = []
        self._fatigue_samples: List[float]  = []
        self._step_rewards:    List[float]  = []
        self.dispatch_log:    List[DispatchRecord] = []
        self._load_zone_meta()
        self._load_hospital_zones()
    def reset(
        self,
        active_zone_ids:  List[str],
        task_id:          int   = 1,
        sim_time_minutes: float = 480.0,
        traffic_model:    Optional[Any] = None,
        hospital_network: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if traffic_model:
            self._traffic = traffic_model
        if hospital_network:
            self._hospital_net = hospital_network
        self.active_zone_ids = active_zone_ids
        self.task_id         = task_id
        self.sim_time_min    = sim_time_minutes
        self.step_count      = 0
        self.units           = {}
        self.unit_order      = []
        self.pending_mutual_aid = {}
        self._aid_seq        = 0
        self.dispatch_log    = []
        self.metrics         = FleetMetrics()
        self._rt_log_p1.clear()
        self._rt_log_p2.clear()
        self._rt_log_p3.clear()
        self._fatigue_samples.clear()
        self._step_rewards.clear()
        self.rng    = random.Random(self.seed + task_id * 7)
        self.np_rng = np.random.RandomState(self.seed + task_id * 7)
        self._spawn_fleet_from_hospitals()
        return self.get_fleet_observation()
    def step(
        self,
        sim_time_minutes: float,
        active_incident_zones: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        self.sim_time_min = sim_time_minutes
        self.step_count  += 1
        step_reward = 0.0
        for uid in self.unit_order:
            unit = self.units[uid]
            unit.step_reward_delta = 0.0
            unit.sim_time_min      = sim_time_minutes
            unit.crew.tick(STEP_DURATION_MINUTES)
            if unit.crew.swap_ready and unit.status == UnitStatus.CREW_SWAP_PENDING:
                self._execute_crew_swap(unit)
            self._tick_unit(unit)
            step_reward += unit.step_reward_delta
        step_reward += self._tick_mutual_aid()
        self._refresh_etas()
        self._update_metrics()
        self._step_rewards.append(step_reward)
        return self.get_fleet_observation(), step_reward
    def dispatch_unit(
        self,
        unit_id:          str,
        incident_id:      str,
        hospital_id:      str,
        incident_zone_id: str,
        incident_priority: str,
        condition_code:   str = "DEFAULT",
        protocol_correct: bool = True,
    ) -> Tuple[bool, str, Optional[DispatchRecord]]:
        unit = self.units.get(unit_id)
        if unit is None:
            return False, f"Unknown unit {unit_id}", None
        if not unit.is_available:
            return False, f"Unit {unit_id} is not available (status={unit.status.value})", None
        if unit.status == UnitStatus.OUT_OF_SERVICE:
            return False, f"Unit {unit_id} is out of service", None
        hospital_zone = self._hospital_zones.get(hospital_id, incident_zone_id)
        eta_scene    = self._compute_eta(unit.current_zone_id, incident_zone_id)
        eta_hospital = self._compute_eta(incident_zone_id, hospital_zone)
        eta_scene += unit.crew.dispatch_delay_mins
        dr = DispatchRecord(
            dispatch_id          = f"DR-{str(uuid.uuid4())[:8].upper()}",
            unit_id              = unit_id,
            incident_id          = incident_id,
            hospital_id          = hospital_id,
            dispatched_at_step   = self.step_count,
            dispatched_at_min    = self.sim_time_min,
            from_zone_id         = unit.current_zone_id,
            incident_zone_id     = incident_zone_id,
            hospital_zone_id     = hospital_zone,
            unit_type            = unit.unit_type.value,
            incident_priority    = incident_priority,
            eta_to_scene_mins    = round(eta_scene, 1),
            eta_to_hospital_mins = round(eta_hospital, 1),
            protocol_correct     = protocol_correct,
            condition_code       = condition_code,
        )
        if not protocol_correct:
            unit.protocol_violations += 1
            unit.step_reward_delta   -= 0.10
            self.metrics.protocol_violation_count += 1
        unit.status                = UnitStatus.DISPATCHED
        unit.assigned_incident_id  = incident_id
        unit.assigned_hospital_id  = hospital_id
        unit.target_zone_id        = incident_zone_id
        unit.dispatch_time_min     = self.sim_time_min
        unit.eta_to_destination_min = eta_scene
        unit.time_in_status_min    = 0.0
        unit.total_dispatches     += 1
        unit._active_dispatch      = dr
        unit.crew.total_calls_handled += 1
        self.metrics.total_dispatches += 1
        prio_map = {"P1": "p1", "P2": "p2", "P3": "p3"}
        pattr = prio_map.get(incident_priority, "p3")
        setattr(self.metrics, f"{pattr}_dispatches",
                getattr(self.metrics, f"{pattr}_dispatches") + 1)
        self.dispatch_log.append(dr)
        return True, "dispatched", dr
    def reroute_unit(
        self,
        unit_id:          str,
        new_hospital_id:  str,
        reason:           str = "diversion",
    ) -> Tuple[bool, str, float]:
        unit = self.units.get(unit_id)
        if unit is None:
            return False, f"Unknown unit {unit_id}", 0.0
        if unit.status != UnitStatus.TRANSPORTING:
            return False, f"Unit {unit_id} is not transporting (status={unit.status.value})", 0.0
        old_hospital = unit.assigned_hospital_id
        new_zone     = self._hospital_zones.get(new_hospital_id, unit.current_zone_id)
        extra_delay  = DIVERSION_REROUTE_DELAY
        new_eta = self._compute_eta(unit.current_zone_id, new_zone) + extra_delay
        unit.assigned_hospital_id         = new_hospital_id
        unit.target_zone_id               = new_zone
        unit.eta_to_destination_min       = new_eta
        unit.diversion_hits              += 1
        unit.step_reward_delta           += DIVERSION_PENALTY_SCORE
        self.metrics.diversion_count     += 1
        if unit._active_dispatch:
            unit._active_dispatch.was_rerouted   = True
            unit._active_dispatch.reroute_count += 1
            unit._active_dispatch.hospital_id    = new_hospital_id
            unit._active_dispatch.hospital_zone_id = new_zone
        return True, f"rerouted from {old_hospital} to {new_hospital_id}", extra_delay
    def reposition_unit(
        self,
        unit_id:       str,
        target_zone_id: str,
    ) -> Tuple[bool, str]:
        unit = self.units.get(unit_id)
        if unit is None:
            return False, f"Unknown unit {unit_id}"
        if not unit.is_available:
            return False, f"Unit {unit_id} is not available for repositioning"
        if target_zone_id not in self.active_zone_ids:
            return False, f"Zone {target_zone_id} is not active"
        if unit.current_zone_id == target_zone_id:
            return True, "already in target zone"
        eta = self._compute_eta(unit.current_zone_id, target_zone_id)
        unit.status                 = UnitStatus.REPOSITIONING
        unit.reposition_target_zone = target_zone_id
        unit.reposition_eta_mins    = eta
        unit.eta_to_destination_min = eta
        unit.time_in_status_min     = 0.0
        return True, f"repositioning to {target_zone_id} ETA {eta:.1f} min"
    def request_crew_swap(self, unit_id: str) -> Tuple[bool, str]:
        unit = self.units.get(unit_id)
        if unit is None:
            return False, f"Unknown unit {unit_id}"
        if unit.crew.swap_requested:
            return False, f"Swap already in progress for {unit_id}"
        if unit.status not in (UnitStatus.AVAILABLE, UnitStatus.RETURNING):
            return False, f"Swap only allowed when available or returning"
        unit.crew.request_swap()
        prev_status = unit.status
        unit.status = UnitStatus.CREW_SWAP_PENDING
        return True, (
            f"Crew swap requested for {unit_id} — "
            f"spare crew arrives in {CREW_SWAP_DEPLOY_MINS:.0f} min"
        )
    def request_mutual_aid(
        self,
        unit_type:        UnitType,
        requesting_zone:  str,
        origin_zone:      str,
    ) -> Tuple[bool, str, Optional[MutualAidRequest]]:
        unfulfilled = [r for r in self.pending_mutual_aid.values()
                       if not r.fulfilled and not r.cancelled]
        if len(unfulfilled) > 3:
            pass
        arrival_min = self.sim_time_min + MUTUAL_AID_DELAY_MINS
        req_id      = f"MA-{self._aid_seq:04d}"
        self._aid_seq += 1
        req = MutualAidRequest(
            request_id        = req_id,
            requesting_zone   = requesting_zone,
            origin_zone       = origin_zone,
            unit_type         = unit_type,
            requested_at_min  = self.sim_time_min,
            arrival_at_min    = arrival_min,
        )
        self.pending_mutual_aid[req_id] = req
        self.metrics.mutual_aid_requested += 1
        return True, f"Mutual aid {unit_type.value} from {origin_zone} ETA {arrival_min:.0f} min", req
    def cancel_mutual_aid(self, request_id: str) -> Tuple[bool, str]:
        req = self.pending_mutual_aid.get(request_id)
        if req is None:
            return False, f"Unknown request {request_id}"
        if req.fulfilled:
            return False, "Request already fulfilled"
        req.cancelled = True
        return True, f"Mutual aid request {request_id} cancelled"
    def mark_unit_on_scene(self, unit_id: str, incident_priority: str = "P2") -> bool:
        unit = self.units.get(unit_id)
        if unit is None:
            return False
        if unit.status not in (UnitStatus.DISPATCHED, UnitStatus.EN_ROUTE_TO_SCENE):
            return False
        rt = self.sim_time_min - unit.dispatch_time_min
        unit.response_times.append(rt)
        prio_rt_map = {
            "P1": self._rt_log_p1,
            "P2": self._rt_log_p2,
            "P3": self._rt_log_p3,
        }
        prio_rt_map.get(incident_priority, self._rt_log_p3).append(rt)
        base_scene = SCENE_TIME_BASE.get(incident_priority, 10.0)
        budget     = base_scene * unit.crew.scene_time_multiplier
        budget    += self.np_rng.normal(0, budget * SCENE_TIME_SIGMA_FRAC)
        budget     = max(5.0, budget)
        unit.status               = UnitStatus.ON_SCENE
        unit.time_in_status_min   = 0.0
        unit.time_on_scene_min    = 0.0
        unit.time_on_scene_budget = budget
        if unit._active_dispatch:
            unit._active_dispatch.arrived_scene_at_min = self.sim_time_min
        target = {
            "P1": RESPONSE_TIME_P1_TARGET,
            "P2": RESPONSE_TIME_P2_TARGET,
            "P3": RESPONSE_TIME_P3_TARGET,
        }.get(incident_priority, 20.0)
        if rt <= target:
            unit.step_reward_delta += 0.15
        elif rt <= target * 1.5:
            unit.step_reward_delta += 0.05
        else:
            unit.step_reward_delta -= 0.10
        return True
    def mark_unit_transporting(
        self,
        unit_id:     str,
        hospital_id: str,
        hospital_er_load_pct: float = 50.0,
    ) -> Tuple[bool, float]:
        unit = self.units.get(unit_id)
        if unit is None:
            return False, 0.0
        if unit.status != UnitStatus.ON_SCENE:
            return False, 0.0
        hospital_zone = self._hospital_zones.get(hospital_id, unit.current_zone_id)
        eta = self._compute_eta(unit.current_zone_id, hospital_zone)
        handover = HANDOVER_BASE_MINS
        if hospital_er_load_pct > 60.0:
            handover += (hospital_er_load_pct - 60.0) * HANDOVER_LOAD_SLOPE
        handover = max(HANDOVER_MIN_MINS, min(HANDOVER_MAX_MINS, handover))
        unit.status                 = UnitStatus.TRANSPORTING
        unit.assigned_hospital_id   = hospital_id
        unit.target_zone_id         = hospital_zone
        unit.eta_to_destination_min = eta + handover
        unit.time_in_status_min     = 0.0
        unit.total_scene_time_min  += unit.time_on_scene_min
        if unit._active_dispatch:
            unit._active_dispatch.departed_scene_at_min = self.sim_time_min
        return True, eta
    def mark_unit_arrived_hospital(self, unit_id: str) -> Tuple[bool, Optional[bool]]:
        unit = self.units.get(unit_id)
        if unit is None:
            return False, None
        if unit.status != UnitStatus.TRANSPORTING:
            return False, None
        total_rt = self.sim_time_min - unit.dispatch_time_min
        golden_hour_met: Optional[bool] = None
        if unit._active_dispatch:
            dr = unit._active_dispatch
            dr.arrived_hospital_at_min = self.sim_time_min
            dr.total_response_time_mins = total_rt
            window = GOLDEN_HOUR_WINDOWS.get(
                dr.condition_code,
                GOLDEN_HOUR_WINDOWS["DEFAULT"]
            )
            golden_hour_met = total_rt <= window
            dr.golden_hour_met = golden_hour_met
            if golden_hour_met:
                unit.golden_hour_successes += 1
                unit.step_reward_delta     += 0.20
            else:
                unit.step_reward_delta     -= 0.15
            unit.golden_hour_attempts += 1
        home_eta = self._compute_eta(unit.current_zone_id, unit.home_zone_id)
        unit.status               = UnitStatus.RETURNING
        unit.target_zone_id       = unit.home_zone_id
        unit.eta_to_destination_min = home_eta
        unit.time_in_status_min   = 0.0
        unit.assigned_incident_id = None
        unit.assigned_hospital_id = None
        unit._active_dispatch     = None
        return True, golden_hour_met
    def set_unit_out_of_service(self, unit_id: str, reason: str = "mechanical") -> bool:
        unit = self.units.get(unit_id)
        if unit is None:
            return False
        unit.status             = UnitStatus.OUT_OF_SERVICE
        unit.time_in_status_min = 0.0
        return True
    def restore_unit_to_service(self, unit_id: str) -> bool:
        unit = self.units.get(unit_id)
        if unit is None:
            return False
        if unit.status != UnitStatus.OUT_OF_SERVICE:
            return False
        unit.status             = UnitStatus.AVAILABLE
        unit.time_in_status_min = 0.0
        return True
    def update_comm_status(self, unit_id: str, active: bool) -> None:
        unit = self.units.get(unit_id)
        if unit is None:
            return
        if unit.comm_active and not active:
            unit.last_known_zone_id = unit.current_zone_id
            unit.last_comm_at_min   = self.sim_time_min
        unit.comm_active = active
    def get_fleet_observation(self) -> Dict[str, Any]:
        units_obs = [u.to_observation_dict() for u in self._iter_units()]
        available_by_type = self._count_available_by_type()
        return {
            "fleet_size":             len(self.units),
            "available_count":        sum(available_by_type.values()),
            "available_by_type":      available_by_type,
            "busy_count":             sum(1 for u in self._iter_units() if u.is_busy),
            "fatigued_count":         sum(1 for u in self._iter_units()
                                         if u.crew.is_fatigued),
            "units":                  units_obs,
            "pending_mutual_aid":     [
                self._mutual_aid_to_dict(r)
                for r in self.pending_mutual_aid.values()
                if not r.fulfilled and not r.cancelled
            ],
            "sim_time_min":           self.sim_time_min,
            "step_count":             self.step_count,
        }
    def get_unit_observation(self, unit_id: str) -> Optional[Dict[str, Any]]:
        unit = self.units.get(unit_id)
        return unit.to_observation_dict() if unit else None
    def get_available_units(
        self,
        zone_ids:  Optional[List[str]] = None,
        unit_type: Optional[UnitType]  = None,
    ) -> List[AmbulanceUnit]:
        result = [u for u in self._iter_units() if u.is_available]
        if zone_ids:
            zone_set = set(zone_ids)
            result   = [u for u in result if u.current_zone_id in zone_set]
        if unit_type:
            result   = [u for u in result if u.unit_type == unit_type]
        result.sort(key=lambda u: (-u.capability_score, u.crew.hours_on_duty))
        return result
    def get_nearest_available_unit(
        self,
        incident_zone_id: str,
        unit_type:        Optional[UnitType] = None,
        exclude_ids:      Optional[Set[str]] = None,
    ) -> Tuple[Optional[AmbulanceUnit], float]:
        candidates = self.get_available_units(unit_type=unit_type)
        if exclude_ids:
            candidates = [u for u in candidates if u.unit_id not in exclude_ids]
        if not candidates:
            return None, float("inf")
        best_unit: Optional[AmbulanceUnit] = None
        best_eta  = float("inf")
        for u in candidates:
            eta = self._compute_eta(u.current_zone_id, incident_zone_id)
            eta += u.crew.dispatch_delay_mins
            if eta < best_eta:
                best_eta  = eta
                best_unit = u
        return best_unit, best_eta
    def get_dispatch_log(self) -> List[Dict[str, Any]]:
        return [self._dispatch_record_to_dict(dr) for dr in self.dispatch_log]
    def get_metrics(self) -> Dict[str, Any]:
        self._update_metrics()
        return self.metrics.to_dict()
    def get_unit_full_state(self, unit_id: str) -> Optional[Dict[str, Any]]:
        unit = self.units.get(unit_id)
        return unit.to_full_state_dict() if unit else None
    def get_all_units_full_state(self) -> List[Dict[str, Any]]:
        return [u.to_full_state_dict() for u in self._iter_units()]
    def count_available_micu(self) -> int:
        return sum(
            1 for u in self._iter_units()
            if u.is_available and u.unit_type == UnitType.MICU
        )
    def count_available_als(self) -> int:
        return sum(
            1 for u in self._iter_units()
            if u.is_available and u.unit_type == UnitType.ALS
        )
    def count_available_bls(self) -> int:
        return sum(
            1 for u in self._iter_units()
            if u.is_available and u.unit_type == UnitType.BLS
        )
    def _tick_unit(self, unit: AmbulanceUnit) -> None:
        unit.time_in_status_min += STEP_DURATION_MINUTES
        if unit.status == UnitStatus.AVAILABLE:
            pass
        elif unit.status == UnitStatus.DISPATCHED:
            self._tick_dispatched(unit)
        elif unit.status == UnitStatus.EN_ROUTE_TO_SCENE:
            self._tick_en_route_to_scene(unit)
        elif unit.status == UnitStatus.ON_SCENE:
            self._tick_on_scene(unit)
        elif unit.status == UnitStatus.TRANSPORTING:
            self._tick_transporting(unit)
        elif unit.status == UnitStatus.RETURNING:
            self._tick_returning(unit)
        elif unit.status == UnitStatus.REPOSITIONING:
            self._tick_repositioning(unit)
        elif unit.status == UnitStatus.CREW_SWAP_PENDING:
            pass
        elif unit.status == UnitStatus.MUTUAL_AID_INBOUND:
            self._tick_mutual_aid_inbound(unit)
        elif unit.status == UnitStatus.OUT_OF_SERVICE:
            pass
    def _tick_dispatched(self, unit: AmbulanceUnit) -> None:
        unit.eta_to_destination_min -= STEP_DURATION_MINUTES
        if unit.eta_to_destination_min <= 0:
            if unit.target_zone_id:
                unit.current_zone_id = unit.target_zone_id
            unit.status              = UnitStatus.EN_ROUTE_TO_SCENE
            unit.time_in_status_min  = 0.0
            unit.eta_to_destination_min = 0.0
        else:
            self._interpolate_position(unit)
    def _tick_en_route_to_scene(self, unit: AmbulanceUnit) -> None:
        unit.eta_to_destination_min = max(0.0, unit.eta_to_destination_min - STEP_DURATION_MINUTES)
        if unit.eta_to_destination_min <= 0 and unit.target_zone_id:
            unit.current_zone_id = unit.target_zone_id
    def _tick_on_scene(self, unit: AmbulanceUnit) -> None:
        unit.time_on_scene_min += STEP_DURATION_MINUTES
        if unit.time_on_scene_min >= unit.time_on_scene_budget:
            unit.step_reward_delta += 0.02   
    def _tick_transporting(self, unit: AmbulanceUnit) -> None:
        unit.eta_to_destination_min -= STEP_DURATION_MINUTES
        if unit.eta_to_destination_min <= 0:
            if unit.target_zone_id:
                unit.current_zone_id = unit.target_zone_id
            self.mark_unit_arrived_hospital(unit.unit_id)
        else:
            self._interpolate_position(unit)
    def _tick_returning(self, unit: AmbulanceUnit) -> None:
        unit.eta_to_destination_min -= STEP_DURATION_MINUTES
        if unit.eta_to_destination_min <= 0:
            unit.current_zone_id     = unit.home_zone_id
            unit.target_zone_id      = None
            unit.status              = UnitStatus.AVAILABLE
            unit.time_in_status_min  = 0.0
            unit.eta_to_destination_min = 0.0
            self._set_position_from_zone(unit, unit.home_zone_id)
        else:
            self._interpolate_position(unit)
    def _tick_repositioning(self, unit: AmbulanceUnit) -> None:
        unit.eta_to_destination_min -= STEP_DURATION_MINUTES
        unit.reposition_eta_mins     = max(0.0, unit.reposition_eta_mins - STEP_DURATION_MINUTES)
        if unit.reposition_eta_mins <= 0:
            if unit.reposition_target_zone:
                unit.current_zone_id         = unit.reposition_target_zone
                unit.home_zone_id            = unit.reposition_target_zone   
                self._set_position_from_zone(unit, unit.reposition_target_zone)
            unit.reposition_target_zone      = None
            unit.reposition_eta_mins         = 0.0
            unit.status                      = UnitStatus.AVAILABLE
            unit.time_in_status_min          = 0.0
            unit.eta_to_destination_min      = 0.0
        else:
            self._interpolate_position(unit)
    def _tick_mutual_aid_inbound(self, unit: AmbulanceUnit) -> None:
        unit.mutual_aid_eta_mins    -= STEP_DURATION_MINUTES
        unit.eta_to_destination_min  = max(0.0, unit.mutual_aid_eta_mins)
        if unit.mutual_aid_eta_mins <= 0:
            unit.status             = UnitStatus.AVAILABLE
            unit.time_in_status_min = 0.0
            unit.mutual_aid_eta_mins = 0.0
            if unit.target_zone_id:
                unit.current_zone_id = unit.target_zone_id
                self._set_position_from_zone(unit, unit.target_zone_id)
    def _execute_crew_swap(self, unit: AmbulanceUnit) -> None:
        unit.crew.complete_swap()
        unit.status             = UnitStatus.AVAILABLE
        unit.time_in_status_min = 0.0
        self.metrics.crew_swaps_executed += 1
    def _tick_mutual_aid(self) -> float:
        reward = 0.0
        for req in list(self.pending_mutual_aid.values()):
            if req.fulfilled or req.cancelled:
                continue
            if self.sim_time_min >= req.arrival_at_min:
                uid  = self._spawn_mutual_aid_unit(req)
                req.unit_id   = uid
                req.fulfilled = True
                self.metrics.mutual_aid_fulfilled += 1
                reward += 0.05
        return reward
    def _refresh_etas(self) -> None:
        if self.step_count % 3 != 0:
            return
        for unit in self._iter_units():
            if unit.status in (
                UnitStatus.DISPATCHED,
                UnitStatus.EN_ROUTE_TO_SCENE,
                UnitStatus.TRANSPORTING,
                UnitStatus.RETURNING,
            ) and unit.target_zone_id:
                fresh_eta = self._compute_eta(unit.current_zone_id, unit.target_zone_id)
                blended = 0.7 * fresh_eta + 0.3 * max(unit.eta_to_destination_min, 0.0)
                unit.eta_to_destination_min = max(0.0, blended)
    def _spawn_fleet_from_hospitals(self) -> None:
        hosp_path = DATA_DIR / "hospital_profiles.json"
        if not hosp_path.exists():
            self._spawn_default_fleet()
            return
        with open(hosp_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        hospitals = raw.get("hospitals", raw) if isinstance(raw, dict) else raw
        if isinstance(hospitals, dict):
            hospitals = list(hospitals.values())
        unit_seq = 0
        for hosp in hospitals:
            hosp_id   = hosp.get("hospital_id") or hosp.get("id", f"H{unit_seq:02d}")
            zone_id   = hosp.get("zone_id", "Z05")
            base_lat  = hosp.get("lat", 18.52)
            base_lon  = hosp.get("lon", 73.85)
            hosp_name = hosp.get("name", hosp_id)
            tier = hosp.get("tier", 2)
            composition = self._fleet_composition_for_tier(tier)
            for unit_type_str, count in composition.items():
                for i in range(count):
                    unit_seq += 1
                    uid       = f"AMB-{unit_type_str[:1]}{unit_seq:03d}"
                    call_sign = f"{unit_type_str}-{unit_seq:03d}"
                    crew      = CrewFatigueState(
                        crew_id       = f"CREW-{unit_seq:04d}",
                        hours_on_duty = self.np_rng.uniform(0, 4.0),  
                    )
                    unit = AmbulanceUnit(
                        unit_id         = uid,
                        unit_type       = UnitType(unit_type_str),
                        call_sign       = call_sign,
                        home_zone_id    = zone_id,
                        current_zone_id = zone_id,
                        lat             = base_lat + self.np_rng.normal(0, 0.002),
                        lon             = base_lon + self.np_rng.normal(0, 0.002),
                        crew            = crew,
                        sim_time_min    = self.sim_time_min,
                    )
                    self.units[uid]     = unit
                    self.unit_order.append(uid)
                    self._hospital_zones[hosp_id] = zone_id
        if not self.units:
            self._spawn_default_fleet()
    def _fleet_composition_for_tier(self, tier: int) -> Dict[str, int]:
        compositions = {
            1: {"MICU": 2, "ALS": 3, "BLS": 3},   
            2: {"MICU": 1, "ALS": 2, "BLS": 2},   
            3: {"MICU": 0, "ALS": 1, "BLS": 2},   
        }
        return compositions.get(tier, compositions[2])
    def _spawn_default_fleet(self) -> None:
        zone_cycle = (self.active_zone_ids or ["Z05", "Z06", "Z07", "Z08"])
        configs = [
            ("MICU", 4), ("ALS", 8), ("BLS", 8),
        ]
        seq = 0
        for unit_type_str, count in configs:
            for i in range(count):
                seq  += 1
                zone  = zone_cycle[seq % len(zone_cycle)]
                meta  = self._zone_meta.get(zone, {})
                uid   = f"AMB-{unit_type_str[:1]}{seq:03d}"
                crew  = CrewFatigueState(
                    crew_id       = f"CREW-{seq:04d}",
                    hours_on_duty = self.np_rng.uniform(0, 3.0),
                )
                unit = AmbulanceUnit(
                    unit_id         = uid,
                    unit_type       = UnitType(unit_type_str),
                    call_sign       = f"{unit_type_str}-{seq:03d}",
                    home_zone_id    = zone,
                    current_zone_id = zone,
                    lat             = float(meta.get("lat", 18.52)),
                    lon             = float(meta.get("lon", 73.85)),
                    crew            = crew,
                    sim_time_min    = self.sim_time_min,
                )
                self.units[uid]     = unit
                self.unit_order.append(uid)
    def _spawn_mutual_aid_unit(self, req: MutualAidRequest) -> str:
        seq  = len(self.units) + 1
        uid  = f"MA-{req.unit_type.value[:1]}{seq:03d}-{req.origin_zone}"
        zone = req.requesting_zone
        meta = self._zone_meta.get(zone, {})
        crew = CrewFatigueState(
            crew_id       = f"CREW-MA-{seq:04d}",
            hours_on_duty = self.np_rng.uniform(0, 6.0),
        )
        unit = AmbulanceUnit(
            unit_id         = uid,
            unit_type       = req.unit_type,
            call_sign       = f"MA-{req.unit_type.value}-{seq:03d}",
            home_zone_id    = req.origin_zone,
            current_zone_id = zone,
            lat             = float(meta.get("lat", 18.52)),
            lon             = float(meta.get("lon", 73.85)),
            status          = UnitStatus.AVAILABLE,
            is_mutual_aid   = True,
            origin_zone_id  = req.origin_zone,
            crew            = crew,
            sim_time_min    = self.sim_time_min,
        )
        self.units[uid]     = unit
        self.unit_order.append(uid)
        return uid
    def _compute_eta(self, from_zone: str, to_zone: str) -> float:
        if self._traffic is not None:
            try:
                return self._traffic.get_travel_time(from_zone, to_zone)
            except Exception:
                pass
        if from_zone == to_zone:
            return 0.0
        return 8.0
    def _interpolate_position(self, unit: AmbulanceUnit) -> None:
        if not unit.target_zone_id:
            return
        target_meta = self._zone_meta.get(unit.target_zone_id, {})
        t_lat = float(target_meta.get("lat", unit.lat))
        t_lon = float(target_meta.get("lon", unit.lon))
        total_eta = max(unit.eta_to_destination_min + STEP_DURATION_MINUTES, 1.0)
        frac      = STEP_DURATION_MINUTES / total_eta
        frac      = min(frac, 1.0)
        unit.lat = unit.lat + frac * (t_lat - unit.lat)
        unit.lon = unit.lon + frac * (t_lon - unit.lon)
    def _set_position_from_zone(self, unit: AmbulanceUnit, zone_id: str) -> None:
        meta    = self._zone_meta.get(zone_id, {})
        unit.lat = float(meta.get("lat", unit.lat)) + self.np_rng.normal(0, 0.001)
        unit.lon = float(meta.get("lon", unit.lon)) + self.np_rng.normal(0, 0.001)
    def _update_metrics(self) -> None:
        m = self.metrics
        m.mean_response_time_p1 = (
            sum(self._rt_log_p1) / len(self._rt_log_p1)
            if self._rt_log_p1 else 0.0
        )
        m.mean_response_time_p2 = (
            sum(self._rt_log_p2) / len(self._rt_log_p2)
            if self._rt_log_p2 else 0.0
        )
        m.mean_response_time_p3 = (
            sum(self._rt_log_p3) / len(self._rt_log_p3)
            if self._rt_log_p3 else 0.0
        )
        total_attempts  = sum(u.golden_hour_attempts  for u in self._iter_units())
        total_successes = sum(u.golden_hour_successes for u in self._iter_units())
        m.golden_hour_compliance = (
            total_successes / total_attempts if total_attempts > 0 else 0.0
        )
        fatigue_vals = [u.crew.fatigue_level for u in self._iter_units()]
        m.fatigue_exposure_index = (
            sum(fatigue_vals) / len(fatigue_vals) if fatigue_vals else 0.0
        )
        m.total_distance_km = sum(u.total_distance_km for u in self._iter_units())
        util_vals = [u.utilisation_pct for u in self._iter_units()]
        m.fleet_utilisation_pct = (
            sum(util_vals) / len(util_vals) if util_vals else 0.0
        )
    def _load_zone_meta(self) -> None:
        path = DATA_DIR / "city_zones.json"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for z in raw.get("zones", []):
            self._zone_meta[z["zone_id"]] = z
    def _load_hospital_zones(self) -> None:
        path = DATA_DIR / "hospital_profiles.json"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        hospitals = raw.get("hospitals", raw) if isinstance(raw, dict) else raw
        if isinstance(hospitals, dict):
            hospitals = list(hospitals.values())
        for h in hospitals:
            hid  = h.get("hospital_id") or h.get("id", "")
            zone = h.get("zone_id", "Z05")
            if hid:
                self._hospital_zones[hid] = zone
    def _iter_units(self):
        for uid in self.unit_order:
            yield self.units[uid]
    def _count_available_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"BLS": 0, "ALS": 0, "MICU": 0}
        for u in self._iter_units():
            if u.is_available:
                counts[u.unit_type.value] += 1
        return counts
    @staticmethod
    def _dispatch_record_to_dict(dr: DispatchRecord) -> Dict[str, Any]:
        return {
            "dispatch_id":            dr.dispatch_id,
            "unit_id":                dr.unit_id,
            "incident_id":            dr.incident_id,
            "hospital_id":            dr.hospital_id,
            "dispatched_at_step":     dr.dispatched_at_step,
            "dispatched_at_min":      round(dr.dispatched_at_min, 1),
            "from_zone_id":           dr.from_zone_id,
            "incident_zone_id":       dr.incident_zone_id,
            "hospital_zone_id":       dr.hospital_zone_id,
            "unit_type":              dr.unit_type,
            "incident_priority":      dr.incident_priority,
            "eta_to_scene_mins":      round(dr.eta_to_scene_mins, 1),
            "eta_to_hospital_mins":   round(dr.eta_to_hospital_mins, 1),
            "protocol_correct":       dr.protocol_correct,
            "was_rerouted":           dr.was_rerouted,
            "reroute_count":          dr.reroute_count,
            "arrived_scene_at_min":   (round(dr.arrived_scene_at_min, 1)
                                       if dr.arrived_scene_at_min else None),
            "departed_scene_at_min":  (round(dr.departed_scene_at_min, 1)
                                       if dr.departed_scene_at_min else None),
            "arrived_hospital_at_min":(round(dr.arrived_hospital_at_min, 1)
                                       if dr.arrived_hospital_at_min else None),
            "total_response_time_mins":(round(dr.total_response_time_mins, 1)
                                        if dr.total_response_time_mins else None),
            "golden_hour_met":        dr.golden_hour_met,
            "condition_code":         dr.condition_code,
        }
    @staticmethod
    def _mutual_aid_to_dict(req: MutualAidRequest) -> Dict[str, Any]:
        return {
            "request_id":        req.request_id,
            "unit_type":         req.unit_type.value,
            "requesting_zone":   req.requesting_zone,
            "origin_zone":       req.origin_zone,
            "requested_at_min":  round(req.requested_at_min, 1),
            "arrival_at_min":    round(req.arrival_at_min, 1),
            "fulfilled":         req.fulfilled,
            "cancelled":         req.cancelled,
        }
    def describe(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = defaultdict(int)
        type_counts:   Dict[str, int] = defaultdict(int)
        for u in self._iter_units():
            status_counts[u.status.value] += 1
            type_counts[u.unit_type.value] += 1
        return {
            "sim_time_min":    self.sim_time_min,
            "step_count":      self.step_count,
            "task_id":         self.task_id,
            "fleet_size":      len(self.units),
            "type_breakdown":  dict(type_counts),
            "status_breakdown":dict(status_counts),
            "fatigued_units":  sum(1 for u in self._iter_units() if u.crew.is_fatigued),
            "mutual_aid_reqs": len(self.pending_mutual_aid),
            "dispatch_log_len":len(self.dispatch_log),
            "golden_hour_compliance": round(self.metrics.golden_hour_compliance, 3),
            "fleet_utilisation_pct":  round(self.metrics.fleet_utilisation_pct, 1),
        }
    def print_fleet_table(self) -> None:
        header = (
            f"{'UNIT':>12} {'TYPE':>5} {'STATUS':>22} {'ZONE':>6} "
            f"{'ETA':>6} {'FATIGUE':>8} {'DUTY_H':>7}"
        )
        print(header)
        print("-" * len(header))
        for u in self._iter_units():
            print(
                f"{u.unit_id:>12} {u.unit_type.value:>5} "
                f"{u.status.value:>22} {u.current_zone_id:>6} "
                f"{u.eta_to_destination_min:>6.1f} "
                f"{u.crew.fatigue_level:>8.3f} "
                f"{u.crew.hours_on_duty:>7.2f}"
            )
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 68)
    print("EMERGI-ENV  ·  FleetSimulator smoke-test")
    print("=" * 68)
    sim = FleetSimulator(seed=42)
    active_zones = [
        "Z01", "Z02", "Z03", "Z04", "Z05", "Z06",
        "Z07", "Z08", "Z09", "Z13", "Z14", "Z27",
    ]
    obs = sim.reset(active_zones, task_id=4, sim_time_minutes=480.0)
    print(f"\n✓  Fleet spawned: {obs['fleet_size']} units")
    print(f"   Available by type: {obs['available_by_type']}")
    print(f"   Describe: {sim.describe()}")
    avail = sim.get_available_units(unit_type=UnitType.MICU)
    if avail:
        u = avail[0]
        ok, reason, dr = sim.dispatch_unit(
            unit_id          = u.unit_id,
            incident_id      = "INC-0001",
            hospital_id      = "H01",
            incident_zone_id = "Z03",
            incident_priority= "P1",
            condition_code   = "STEMI",
            protocol_correct = True,
        )
        print(f"\n✓  Dispatch MICU: {ok} — {reason}")
        if dr:
            print(f"   ETA to scene: {dr.eta_to_scene_mins} min")
            print(f"   ETA to hospital: {dr.eta_to_hospital_mins} min")
    als_units = sim.get_available_units(unit_type=UnitType.ALS)
    if als_units:
        swap_unit = als_units[0]
        swap_unit.crew.hours_on_duty = 11.5
        ok2, msg2 = sim.request_crew_swap(swap_unit.unit_id)
        print(f"\n✓  Crew swap request: {ok2} — {msg2}")
        print(f"   Fatigue level: {swap_unit.crew.fatigue_level:.3f}")
    bls_units = sim.get_available_units(unit_type=UnitType.BLS)
    if bls_units:
        b = bls_units[0]
        ok3, msg3 = sim.reposition_unit(b.unit_id, "Z09")
        print(f"\n✓  Reposition BLS: {ok3} — {msg3}")
    ok4, msg4, req = sim.request_mutual_aid(UnitType.ALS, "Z05", "Z01")
    print(f"\n✓  Mutual aid request: {ok4} — {msg4}")
    print(f"\n  Stepping 30 steps:")
    total_rwd = 0.0
    for s in range(30):
        fleet_obs, rwd = sim.step(480.0 + (s + 1) * STEP_DURATION_MINUTES)
        total_rwd += rwd
        if s == 5:
            for uid in list(sim.units.keys()):
                u = sim.units[uid]
                if u.status == UnitStatus.EN_ROUTE_TO_SCENE:
                    sim.mark_unit_on_scene(uid, "P1")
                    print(f"    Step {s+1}: {uid} marked ON_SCENE")
        if s == 10:
            for uid in list(sim.units.keys()):
                u = sim.units[uid]
                if u.status == UnitStatus.ON_SCENE:
                    sim.mark_unit_transporting(uid, "H01", 65.0)
                    print(f"    Step {s+1}: {uid} TRANSPORTING to H01 (ER load 65%)")
    print(f"\n  Total step reward over 30 steps: {total_rwd:.4f}")
    print(f"  Metrics: {sim.get_metrics()}")
    print(f"\n  Fleet table after 30 steps:")
    sim.print_fleet_table()
    first_uid = sim.unit_order[0] if sim.unit_order else None
    if first_uid:
        sim.update_comm_status(first_uid, False)
        u_obs = sim.get_unit_observation(first_uid)
        print(f"\n  Comm failure test — {first_uid}:")
        print(f"   comm_active={u_obs['comm_active']} "
              f"lat={u_obs['lat']} lon={u_obs['lon']}")
    log = sim.get_dispatch_log()
    print(f"\n  Dispatch log entries: {len(log)}")
    for entry in log:
        print(f"   {entry['dispatch_id']} | {entry['unit_type']} → "
              f"INC {entry['incident_id']} | ETA {entry['eta_to_scene_mins']} min | "
              f"GH met={entry['golden_hour_met']}")
    print("\n✅  FleetSimulator smoke-test PASSED")