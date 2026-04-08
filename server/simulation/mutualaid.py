from __future__ import annotations
import heapq
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
TIER_DELAY_MINUTES: Dict[int, float] = {
    0: 12.0,   
    1: 20.0,   
    2: 35.0,   
    3: 60.0,   
}
TIER_RECALL_DELAY_MINUTES: Dict[int, float] = {
    0: 18.0,
    1: 28.0,
    2: 50.0,
    3: 90.0,
}
MAX_AID_FRACTION:           float = 0.40   
MIN_ZONE_AUTONOMY_UNITS:    int   = 2      
OVER_REQUEST_THRESHOLD:     int   = 3      
OVER_REQUEST_PENALTY:       float = -0.10
UNDER_REQUEST_TASK9_PENALTY:float = -0.25
CASCADE_FAILURE_DIVERTED_N: int   = 5      
CASCADE_FAILURE_P1_QUEUE:   int   = 8      
CONVOY_MIN_UNITS:           int   = 3      
PRE_POSITION_RESERVE_FRACTION: float = 0.25  
PRE_POSITION_DEPLOY_MINUTES:   float = 6.0   
SCORE_WEIGHT_RESPONSE_TIME:   float = 0.30
SCORE_WEIGHT_APPROPRIATENESS: float = 0.25
SCORE_WEIGHT_AUTONOMY_GUARD:  float = 0.20
SCORE_WEIGHT_PROTOCOL:        float = 0.15
SCORE_WEIGHT_CASCADE_AVOID:   float = 0.10
REWARD_AID_DISPATCHED:      float = +0.06
REWARD_AID_ARRIVED:         float = +0.12
REWARD_AID_RETURNED:        float = +0.04
REWARD_CONVOY_COORDINATED:  float = +0.08
REWARD_PRE_POSITION_HIT:    float = +0.10   
REWARD_TREATY_HONOURED:     float = +0.05
REWARD_CASCADE_AVERTED:     float = +0.20
PENALTY_TREATY_BROKEN:      float = -0.12
PENALTY_UNIT_STRANDED:      float = -0.08   
PENALTY_WRONG_TIER:         float = -0.06   
PENALTY_OVERSTAY:           float = -0.04   
ZONE_TYPE_CAPACITY_MULT: Dict[str, float] = {
    "urban":      1.30,
    "suburban":   1.00,
    "peri_urban": 0.80,
    "rural":      0.60,
    "industrial": 1.10,
    "highway":    0.70,
}
class AidTier(int, Enum):
    ADJACENT  = 0
    DISTRICT  = 1
    STATE     = 2
    NATIONAL  = 3
class AidStatus(str, Enum):
    PENDING     = "pending"
    ACCEPTED    = "accepted"
    DISPATCHED  = "dispatched"
    EN_ROUTE    = "en_route"
    ARRIVED     = "arrived"
    ACTIVE      = "active"        
    RETURNING   = "returning"
    COMPLETED   = "completed"
    REJECTED    = "rejected"
    CANCELLED   = "cancelled"
    RECALLED    = "recalled"      
class AidUnitType(str, Enum):
    BLS  = "BLS"
    ALS  = "ALS"
    MICU = "MICU"
class TreatyType(str, Enum):
    BILATERAL     = "bilateral"      
    MULTILATERAL  = "multilateral"   
class CascadeStage(str, Enum):
    NONE      = "none"
    RISK      = "risk"       
    IMMINENT  = "imminent"   
    ACTIVE    = "active"     
    CONTAINED = "contained"  
@dataclass
class ZoneCapacity:
    zone_id:            str
    zone_type:          str
    total_units:        int
    available_units:    int
    units_on_aid_out:   int   = 0   
    units_received_aid: int   = 0   
    max_giveable:       int   = 0   
    can_give_aid:       bool  = True
    def refresh(self) -> None:
        self.max_giveable = max(
            0,
            min(
                int(self.total_units * MAX_AID_FRACTION),
                self.available_units - MIN_ZONE_AUTONOMY_UNITS,
            ),
        )
        self.can_give_aid = self.max_giveable > 0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id":          self.zone_id,
            "zone_type":        self.zone_type,
            "total_units":      self.total_units,
            "available_units":  self.available_units,
            "units_on_aid_out": self.units_on_aid_out,
            "units_received":   self.units_received_aid,
            "max_giveable":     self.max_giveable,
            "can_give_aid":     self.can_give_aid,
        }
@dataclass
class AidUnit:
    aid_unit_id:      str
    unit_type:        AidUnitType
    origin_zone_id:   str
    destination_zone_id: str
    tier:             AidTier
    status:           AidStatus = AidStatus.PENDING
    request_id:       str       = ""
    dispatched_at_min: float    = 0.0
    arrived_at_min:    Optional[float] = None
    released_at_min:   Optional[float] = None
    returned_at_min:   Optional[float] = None
    eta_minutes:       float    = 0.0
    recall_eta_minutes: float   = 0.0
    was_pre_staged:    bool     = False
    incidents_handled: int      = 0
    distance_covered_km: float  = 0.0
    overstay_steps:    int      = 0
    expected_release_min: float = 0.0
    underlying_unit_id: Optional[str] = None  
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aid_unit_id":        self.aid_unit_id,
            "unit_type":          self.unit_type.value,
            "origin_zone":        self.origin_zone_id,
            "destination_zone":   self.destination_zone_id,
            "tier":               self.tier.value,
            "status":             self.status.value,
            "request_id":         self.request_id,
            "dispatched_at_min":  round(self.dispatched_at_min, 1),
            "arrived_at_min":     round(self.arrived_at_min, 1) if self.arrived_at_min else None,
            "eta_minutes":        round(self.eta_minutes, 1),
            "was_pre_staged":     self.was_pre_staged,
            "incidents_handled":  self.incidents_handled,
        }
@dataclass
class MutualAidRequest:
    request_id:       str
    requesting_zone:  str
    unit_type:        AidUnitType
    units_needed:     int
    tier_requested:   AidTier
    incident_id:      Optional[str]
    incident_priority: str
    reason:           str
    requested_at_min: float
    requested_at_step: int
    expected_duration_min: float   = 60.0
    status:           AidStatus    = AidStatus.PENDING
    accepted_at_min:  Optional[float] = None
    fulfilled_at_min: Optional[float] = None
    rejected_reason:  Optional[str] = None
    aid_units:        List[AidUnit] = field(default_factory=list)
    source_zones:     List[str]    = field(default_factory=list)
    total_units_received: int      = 0
    was_over_request: bool         = False
    was_under_request: bool        = False
    protocol_correct: bool         = True
    grader_score:     Optional[float] = None
    @property
    def is_active(self) -> bool:
        return self.status in (
            AidStatus.ACCEPTED, AidStatus.DISPATCHED,
            AidStatus.EN_ROUTE, AidStatus.ARRIVED, AidStatus.ACTIVE,
        )
    @property
    def response_time_minutes(self) -> Optional[float]:
        if self.fulfilled_at_min is None:
            return None
        return self.fulfilled_at_min - self.requested_at_min
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id":         self.request_id,
            "requesting_zone":    self.requesting_zone,
            "unit_type":          self.unit_type.value,
            "units_needed":       self.units_needed,
            "tier_requested":     self.tier_requested.value,
            "incident_id":        self.incident_id,
            "incident_priority":  self.incident_priority,
            "reason":             self.reason,
            "status":             self.status.value,
            "requested_at_min":   round(self.requested_at_min, 1),
            "accepted_at_min":    round(self.accepted_at_min, 1) if self.accepted_at_min else None,
            "fulfilled_at_min":   round(self.fulfilled_at_min, 1) if self.fulfilled_at_min else None,
            "total_units_received": self.total_units_received,
            "source_zones":       self.source_zones,
            "was_over_request":   self.was_over_request,
            "was_under_request":  self.was_under_request,
            "protocol_correct":   self.protocol_correct,
            "response_time_min":  (round(self.response_time_minutes, 1)
                                   if self.response_time_minutes is not None else None),
            "aid_units":          [u.to_dict() for u in self.aid_units],
            "grader_score":       round(self.grader_score, 4) if self.grader_score else None,
        }
@dataclass
class AidTreaty:
    treaty_id:        str
    treaty_type:      TreatyType
    zone_ids:         List[str]    
    unit_guarantee:   Dict[str, int]  
    max_duration_min: float = 240.0   
    activation_threshold_p1: int = 3  
    is_active:        bool  = True
    activations:      int   = 0
    last_activated_min: Optional[float] = None
    times_honoured:   int   = 0
    times_broken:     int   = 0
    created_at_min:   float = 0.0
    def involves_zone(self, zone_id: str) -> bool:
        return zone_id in self.zone_ids
    def guarantees_for(self, requesting_zone: str, donor_zone: str) -> int:
        if requesting_zone not in self.zone_ids:
            return 0
        if donor_zone not in self.zone_ids:
            return 0
        return self.unit_guarantee.get(donor_zone, 0)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "treaty_id":      self.treaty_id,
            "type":           self.treaty_type.value,
            "zones":          self.zone_ids,
            "guarantees":     self.unit_guarantee,
            "is_active":      self.is_active,
            "activations":    self.activations,
            "times_honoured": self.times_honoured,
            "times_broken":   self.times_broken,
        }
@dataclass
class ConvoyDispatch:
    convoy_id:         str
    request_id:        str
    destination_zone:  str
    unit_ids:          List[str]        
    lead_unit_id:      str
    dispatched_at_min: float
    arrival_eta_min:   float
    actual_arrival_min: Optional[float] = None
    status:            str = "en_route"  
    escort_police:     bool = False       
    air_escort:        bool = False       
    @property
    def escort_time_reduction(self) -> float:
        mult = 1.0
        if self.escort_police:
            mult *= 0.85
        if self.air_escort:
            mult *= 0.90
        return mult
    def to_dict(self) -> Dict[str, Any]:
        return {
            "convoy_id":        self.convoy_id,
            "request_id":       self.request_id,
            "destination_zone": self.destination_zone,
            "unit_count":       len(self.unit_ids),
            "lead_unit":        self.lead_unit_id,
            "dispatched_at_min":round(self.dispatched_at_min, 1),
            "arrival_eta_min":  round(self.arrival_eta_min, 1),
            "actual_arrival_min":(round(self.actual_arrival_min, 1)
                                  if self.actual_arrival_min else None),
            "status":           self.status,
            "escort_police":    self.escort_police,
        }
@dataclass
class CascadeEvent:
    event_id:         str
    detected_at_min:  float
    detected_at_step: int
    stage:            CascadeStage
    hospitals_diverted: int
    p1_unhandled:     int
    zones_affected:   List[str]
    aid_requested:    bool
    aid_received:     bool
    contained_at_min: Optional[float] = None
    contained_at_step: Optional[int]  = None
    avoidable:        bool = True
    cascade_penalty:  float = 0.0
    cascade_bonus:    float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":           self.event_id,
            "detected_at_min":    round(self.detected_at_min, 1),
            "stage":              self.stage.value,
            "hospitals_diverted": self.hospitals_diverted,
            "p1_unhandled":       self.p1_unhandled,
            "zones_affected":     self.zones_affected,
            "aid_requested":      self.aid_requested,
            "aid_received":       self.aid_received,
            "contained_at_min":   (round(self.contained_at_min, 1)
                                   if self.contained_at_min else None),
            "avoidable":          self.avoidable,
            "cascade_penalty":    round(self.cascade_penalty, 4),
            "cascade_bonus":      round(self.cascade_bonus, 4),
        }
@dataclass
class AidSurgeDeclaration:
    declaration_id:    str
    zone_id:           str
    declared_at_min:   float
    declared_at_step:  int
    tier_escalated_to: AidTier
    trigger_reason:    str
    p1_queue_size:     int
    available_units:   int
    is_active:         bool    = True
    resolved_at_min:   Optional[float] = None
    auto_aid_requests: List[str] = field(default_factory=list)  
    def to_dict(self) -> Dict[str, Any]:
        return {
            "declaration_id":   self.declaration_id,
            "zone_id":          self.zone_id,
            "declared_at_min":  round(self.declared_at_min, 1),
            "tier_escalated_to":self.tier_escalated_to.value,
            "trigger_reason":   self.trigger_reason,
            "p1_queue_size":    self.p1_queue_size,
            "available_units":  self.available_units,
            "is_active":        self.is_active,
            "auto_aid_requests":self.auto_aid_requests,
        }
@dataclass
class PreStagedPosition:
    position_id:    str
    zone_id:        str         
    unit_type:      AidUnitType
    donor_zone_id:  str
    treaty_id:      Optional[str]
    staged_at_min:  float
    expires_at_min: float       
    is_deployed:    bool = False
    deployed_to_request_id: Optional[str] = None
    deployed_at_min: Optional[float] = None
    @property
    def is_expired(self) -> bool:
        return False  
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id":  self.position_id,
            "zone_id":      self.zone_id,
            "unit_type":    self.unit_type.value,
            "donor_zone":   self.donor_zone_id,
            "treaty_id":    self.treaty_id,
            "staged_at_min":round(self.staged_at_min, 1),
            "expires_at_min":round(self.expires_at_min, 1),
            "is_deployed":  self.is_deployed,
        }
@dataclass
class MutualAidLedger:
    episode_id:           str
    task_id:              int
    total_requests:       int   = 0
    requests_fulfilled:   int   = 0
    requests_rejected:    int   = 0
    requests_cancelled:   int   = 0
    over_requests:        int   = 0
    under_requests:       int   = 0
    tier0_requests:       int   = 0
    tier1_requests:       int   = 0
    tier2_requests:       int   = 0
    tier3_requests:       int   = 0
    convoy_dispatches:    int   = 0
    treaties_activated:   int   = 0
    treaties_honoured:    int   = 0
    treaties_broken:      int   = 0
    cascade_events:       int   = 0
    cascade_averted:      int   = 0
    surge_declarations:   int   = 0
    units_dispatched:     int   = 0
    units_returned:       int   = 0
    pre_staged_hits:      int   = 0
    total_over_request_penalty:  float = 0.0
    total_under_request_penalty: float = 0.0
    total_treaty_penalty:        float = 0.0
    total_cascade_penalty:       float = 0.0
    total_cascade_bonus:         float = 0.0
    mean_response_time_min:      float = 0.0
    p1_mean_response_time_min:   float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id":           self.episode_id,
            "task_id":              self.task_id,
            "total_requests":       self.total_requests,
            "requests_fulfilled":   self.requests_fulfilled,
            "requests_rejected":    self.requests_rejected,
            "requests_cancelled":   self.requests_cancelled,
            "over_requests":        self.over_requests,
            "under_requests":       self.under_requests,
            "tier_breakdown": {
                "tier0": self.tier0_requests,
                "tier1": self.tier1_requests,
                "tier2": self.tier2_requests,
                "tier3": self.tier3_requests,
            },
            "convoy_dispatches":    self.convoy_dispatches,
            "treaties_activated":   self.treaties_activated,
            "treaties_honoured":    self.treaties_honoured,
            "treaties_broken":      self.treaties_broken,
            "cascade_events":       self.cascade_events,
            "cascade_averted":      self.cascade_averted,
            "surge_declarations":   self.surge_declarations,
            "units_dispatched":     self.units_dispatched,
            "units_returned":       self.units_returned,
            "pre_staged_hits":      self.pre_staged_hits,
            "penalties": {
                "over_request":   round(self.total_over_request_penalty,  4),
                "under_request":  round(self.total_under_request_penalty, 4),
                "treaty":         round(self.total_treaty_penalty,        4),
                "cascade":        round(self.total_cascade_penalty,       4),
            },
            "bonuses": {
                "cascade_averted": round(self.total_cascade_bonus, 4),
            },
            "mean_response_time_min":    round(self.mean_response_time_min,   2),
            "p1_mean_response_time_min": round(self.p1_mean_response_time_min, 2),
        }
class MutualAidCoordinator:
    def __init__(
        self,
        seed:              int = 42,
        task_id:           int = 1,
        zone_data:         Optional[Dict[str, Any]] = None,
        traffic_model:     Optional[Any] = None,
        fleet_simulator:   Optional[Any] = None,
        hospital_network:  Optional[Any] = None,
        incident_engine:   Optional[Any] = None,
        demand_forecaster: Optional[Any] = None,
        comms_manager:     Optional[Any] = None,
    ) -> None:
        self.seed              = seed
        self.task_id           = task_id
        self.rng               = random.Random(seed)
        self.np_rng            = np.random.RandomState(seed)
        self._traffic          = traffic_model
        self._fleet            = fleet_simulator
        self._hospital_net     = hospital_network
        self._incident_engine  = incident_engine
        self._demand           = demand_forecaster
        self._comms            = comms_manager
        self._zone_meta:  Dict[str, Any]   = zone_data or {}
        self._adjacency:  Dict[str, List[str]] = {}
        self.active_zone_ids: List[str]    = []
        self.requests:    Dict[str, MutualAidRequest]  = {}
        self.aid_units:   Dict[str, AidUnit]           = {}
        self.convoys:     Dict[str, ConvoyDispatch]    = {}
        self.treaties:    Dict[str, AidTreaty]         = {}
        self.surge_decls: Dict[str, AidSurgeDeclaration] = {}
        self.pre_staged:  Dict[str, PreStagedPosition] = {}
        self.cascade_events: List[CascadeEvent]        = []
        self.zone_capacities: Dict[str, ZoneCapacity]  = {}
        self._req_seq   = 0
        self._unit_seq  = 0
        self._convoy_seq = 0
        self._treaty_seq = 0
        self._surge_seq  = 0
        self._stage_seq  = 0
        self._cascade_seq = 0
        self.sim_time_min: float = 480.0
        self.step_count:   int   = 0
        self.ledger: MutualAidLedger = MutualAidLedger(
            episode_id=str(uuid.uuid4())[:8], task_id=task_id
        )
        self._step_rewards:     List[float] = []
        self._total_reward:     float = 0.0
        self._rt_log_all:       List[float] = []
        self._rt_log_p1:        List[float] = []
        self._cascade_stage:    CascadeStage = CascadeStage.NONE
        self._cascade_steps:    int = 0
        self._last_cascade_check_step: int = -1
        self._req_window:       Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self._load_zone_meta()
    def reset(
        self,
        active_zone_ids:  List[str],
        task_id:          int   = 1,
        sim_time_minutes: float = 480.0,
        traffic_model:    Optional[Any] = None,
        fleet_simulator:  Optional[Any] = None,
        hospital_network: Optional[Any] = None,
        incident_engine:  Optional[Any] = None,
        demand_forecaster:Optional[Any] = None,
        comms_manager:    Optional[Any] = None,
    ) -> Dict[str, Any]:
        if traffic_model:    self._traffic       = traffic_model
        if fleet_simulator:  self._fleet         = fleet_simulator
        if hospital_network: self._hospital_net  = hospital_network
        if incident_engine:  self._incident_engine = incident_engine
        if demand_forecaster:self._demand        = demand_forecaster
        if comms_manager:    self._comms         = comms_manager
        self.task_id        = task_id
        self.sim_time_min   = sim_time_minutes
        self.step_count     = 0
        self.rng    = random.Random(self.seed + task_id * 41)
        self.np_rng = np.random.RandomState(self.seed + task_id * 41)
        self.active_zone_ids = list(active_zone_ids)
        self.requests       = {}
        self.aid_units      = {}
        self.convoys        = {}
        self.treaties       = {}
        self.surge_decls    = {}
        self.pre_staged     = {}
        self.cascade_events = []
        self._req_seq = self._unit_seq = self._convoy_seq = 0
        self._treaty_seq = self._surge_seq = self._stage_seq = self._cascade_seq = 0
        self.ledger = MutualAidLedger(
            episode_id=str(uuid.uuid4())[:8], task_id=task_id
        )
        self._step_rewards  = []
        self._total_reward  = 0.0
        self._rt_log_all    = []
        self._rt_log_p1     = []
        self._req_window    = defaultdict(lambda: deque(maxlen=20))
        self._cascade_stage = CascadeStage.NONE
        self._cascade_steps = 0
        self._last_cascade_check_step = -1
        active_set = set(active_zone_ids)
        for zid in active_zone_ids:
            z   = self._zone_meta.get(zid, {})
            adj = z.get("adjacent_zones", z.get("adjacent_zone_ids", []))
            self._adjacency[zid] = [n for n in adj if n in active_set]
        self._refresh_zone_capacities()
        self._initialise_default_treaties()
        if task_id in (6, 9):
            self._pre_stage_fleet()
        return self.get_observation()
    def step(
        self,
        sim_time_minutes: float,
        fleet_available:  Optional[Dict[str, int]] = None,
        hospital_diverted_count: int = 0,
        p1_queue_count: int = 0,
    ) -> Tuple[Dict[str, Any], float]:
        self.sim_time_min = sim_time_minutes
        self.step_count  += 1
        step_reward        = 0.0
        self._refresh_zone_capacities(fleet_available)
        step_reward += self._tick_aid_units()
        step_reward += self._tick_convoys()
        step_reward += self._tick_pre_staged()
        step_reward += self._tick_surge_declarations()
        if self.task_id >= 7:
            step_reward += self._check_cascade(
                hospital_diverted_count, p1_queue_count
            )
        step_reward += self._audit_request_balance()
        step_reward += self._audit_treaties()
        self._step_rewards.append(step_reward)
        self._total_reward += step_reward
        return self.get_observation(), step_reward
    def request_aid(
        self,
        requesting_zone:       str,
        unit_type:             AidUnitType,
        units_needed:          int = 1,
        tier:                  AidTier = AidTier.ADJACENT,
        incident_id:           Optional[str] = None,
        incident_priority:     str = "P2",
        reason:                str = "capacity_shortage",
        expected_duration_min: float = 60.0,
        force_convoy:          bool  = False,
        use_treaty:            bool  = True,
    ) -> Tuple[bool, str, Optional[MutualAidRequest]]:
        if requesting_zone not in set(self.active_zone_ids):
            return False, f"Zone {requesting_zone} is not active", None
        active_reqs_for_zone = [
            r for r in self.requests.values()
            if r.requesting_zone == requesting_zone and r.is_active
        ]
        is_over = len(active_reqs_for_zone) >= OVER_REQUEST_THRESHOLD
        if is_over:
            self._step_rewards[-1:] = self._step_rewards[-1:] or [0.0]
            self._total_reward += OVER_REQUEST_PENALTY
            self.ledger.over_requests += 1
            self.ledger.total_over_request_penalty += abs(OVER_REQUEST_PENALTY)
        protocol_correct = self._check_tier_appropriateness(
            requesting_zone, tier, units_needed, incident_priority
        )
        self._req_seq += 1
        req_id = f"MAR-{self._req_seq:05d}"
        req = MutualAidRequest(
            request_id             = req_id,
            requesting_zone        = requesting_zone,
            unit_type              = unit_type,
            units_needed           = units_needed,
            tier_requested         = tier,
            incident_id            = incident_id,
            incident_priority      = incident_priority,
            reason                 = reason,
            requested_at_min       = self.sim_time_min,
            requested_at_step      = self.step_count,
            expected_duration_min  = expected_duration_min,
            was_over_request       = is_over,
            protocol_correct       = protocol_correct,
        )
        self.requests[req_id] = req
        self.ledger.total_requests += 1
        self.ledger.tier0_requests += (tier == AidTier.ADJACENT)
        self.ledger.tier1_requests += (tier == AidTier.DISTRICT)
        self.ledger.tier2_requests += (tier == AidTier.STATE)
        self.ledger.tier3_requests += (tier == AidTier.NATIONAL)
        dispatched, reason_str = self._find_and_dispatch(
            req, use_treaty=use_treaty, force_convoy=force_convoy
        )
        if dispatched:
            req.status         = AidStatus.ACCEPTED
            req.accepted_at_min = self.sim_time_min
            self._total_reward += REWARD_AID_DISPATCHED
            if is_over:
                return True, f"accepted_but_over_request:{reason_str}", req
            return True, reason_str, req
        else:
            req.status          = AidStatus.REJECTED
            req.rejected_reason = reason_str
            self.ledger.requests_rejected += 1
            return False, reason_str, req
    def cancel_request(
        self,
        request_id: str,
        reason:     str = "situation_resolved",
    ) -> Tuple[bool, str]:
        req = self.requests.get(request_id)
        if req is None:
            return False, f"Unknown request {request_id}"
        if req.status in (AidStatus.COMPLETED, AidStatus.CANCELLED):
            return False, f"Request already {req.status.value}"
        recall_reward = 0.0
        for au in req.aid_units:
            if au.status in (AidStatus.EN_ROUTE, AidStatus.DISPATCHED):
                au.status            = AidStatus.RECALLED
                au.recall_eta_minutes = TIER_RECALL_DELAY_MINUTES[req.tier_requested.value]
                recall_reward        += PENALTY_UNIT_STRANDED
        req.status = AidStatus.CANCELLED
        self._total_reward    += recall_reward
        self.ledger.requests_cancelled += 1
        return True, f"cancelled:{reason} recall_penalty={recall_reward:.3f}"
    def recall_aid_unit(
        self,
        aid_unit_id: str,
        reason:      str = "zone_capacity_restored",
    ) -> Tuple[bool, float]:
        au = self.aid_units.get(aid_unit_id)
        if au is None:
            return False, 0.0
        penalty = 0.0
        if au.status == AidStatus.EN_ROUTE:
            au.status = AidStatus.RECALLED
            au.recall_eta_minutes = TIER_RECALL_DELAY_MINUTES[au.tier.value] * 0.5
            penalty += PENALTY_UNIT_STRANDED
            self._total_reward += penalty
        elif au.status == AidStatus.ARRIVED:
            au.status = AidStatus.RETURNING
            au.recall_eta_minutes = self._compute_eta(
                au.destination_zone_id, au.origin_zone_id
            )
        else:
            return False, 0.0  
        return True, penalty
    def release_aid_unit(
        self,
        aid_unit_id: str,
    ) -> Tuple[bool, str, float]:
        au = self.aid_units.get(aid_unit_id)
        if au is None:
            return False, "unknown_unit", 0.0
        if au.status not in (AidStatus.ARRIVED, AidStatus.ACTIVE):
            return False, f"unit_not_releasable:{au.status.value}", 0.0
        return_eta = self._compute_eta(au.destination_zone_id, au.origin_zone_id)
        au.status            = AidStatus.RETURNING
        au.released_at_min   = self.sim_time_min
        au.recall_eta_minutes = return_eta
        self._total_reward   += REWARD_AID_RETURNED
        self.ledger.units_returned += 1
        return True, "returning", REWARD_AID_RETURNED
    def declare_surge(
        self,
        zone_id:         str,
        p1_queue_size:   int,
        available_units: int,
        tier:            AidTier = AidTier.ADJACENT,
        trigger_reason:  str     = "capacity_critical",
    ) -> Tuple[bool, str, Optional[AidSurgeDeclaration]]:
        if zone_id not in set(self.active_zone_ids):
            return False, "unknown_zone", None
        existing = [d for d in self.surge_decls.values()
                    if d.zone_id == zone_id and d.is_active]
        if existing:
            return False, "surge_already_active", existing[0]
        self._surge_seq += 1
        decl = AidSurgeDeclaration(
            declaration_id  = f"SURGE-{self._surge_seq:04d}",
            zone_id         = zone_id,
            declared_at_min = self.sim_time_min,
            declared_at_step= self.step_count,
            tier_escalated_to = tier,
            trigger_reason  = trigger_reason,
            p1_queue_size   = p1_queue_size,
            available_units = available_units,
        )
        self.surge_decls[decl.declaration_id] = decl
        self.ledger.surge_declarations += 1
        units_needed = max(1, p1_queue_size - available_units)
        ok, msg, req = self.request_aid(
            requesting_zone       = zone_id,
            unit_type             = AidUnitType.ALS,
            units_needed          = units_needed,
            tier                  = tier,
            incident_priority     = "P1",
            reason                = f"surge_declaration:{trigger_reason}",
            expected_duration_min = 120.0,
        )
        if ok and req:
            decl.auto_aid_requests.append(req.request_id)
        return True, f"surge_declared_tier{tier.value}", decl
    def resolve_surge(self, declaration_id: str) -> bool:
        decl = self.surge_decls.get(declaration_id)
        if decl is None:
            return False
        decl.is_active         = False
        decl.resolved_at_min   = self.sim_time_min
        return True
    def create_treaty(
        self,
        zone_ids:        List[str],
        unit_guarantee:  Dict[str, int],
        treaty_type:     TreatyType = TreatyType.BILATERAL,
        max_duration_min: float = 240.0,
        activation_threshold_p1: int = 3,
    ) -> AidTreaty:
        self._treaty_seq += 1
        treaty = AidTreaty(
            treaty_id        = f"TREATY-{self._treaty_seq:04d}",
            treaty_type      = treaty_type,
            zone_ids         = list(zone_ids),
            unit_guarantee   = dict(unit_guarantee),
            max_duration_min = max_duration_min,
            activation_threshold_p1 = activation_threshold_p1,
            created_at_min   = self.sim_time_min,
        )
        self.treaties[treaty.treaty_id] = treaty
        return treaty
    def activate_treaty(
        self,
        treaty_id:       str,
        requesting_zone: str,
        p1_count:        int,
    ) -> Tuple[bool, str, float]:
        treaty = self.treaties.get(treaty_id)
        if treaty is None:
            return False, "unknown_treaty", 0.0
        if not treaty.is_active:
            return False, "treaty_inactive", 0.0
        if requesting_zone not in treaty.zone_ids:
            return False, "zone_not_party", 0.0
        reward = 0.0
        for donor_zone in treaty.zone_ids:
            if donor_zone == requesting_zone:
                continue
            guaranteed = treaty.guarantees_for(requesting_zone, donor_zone)
            if guaranteed <= 0:
                continue
            cap = self.zone_capacities.get(donor_zone)
            if cap and cap.max_giveable >= guaranteed:
                ok, msg, req = self.request_aid(
                    requesting_zone   = requesting_zone,
                    unit_type         = AidUnitType.ALS,
                    units_needed      = guaranteed,
                    tier              = AidTier.ADJACENT,
                    incident_priority = "P1",
                    reason            = f"treaty_activation:{treaty_id}",
                    use_treaty        = False,
                )
                if ok:
                    treaty.times_honoured += 1
                    reward += REWARD_TREATY_HONOURED
                else:
                    treaty.times_broken += 1
                    reward += PENALTY_TREATY_BROKEN
                    self.ledger.treaties_broken += 1
                    self.ledger.total_treaty_penalty += abs(PENALTY_TREATY_BROKEN)
            else:
                treaty.times_broken += 1
                reward += PENALTY_TREATY_BROKEN
                self.ledger.treaties_broken += 1
                self.ledger.total_treaty_penalty += abs(PENALTY_TREATY_BROKEN)
        treaty.activations         += 1
        treaty.last_activated_min   = self.sim_time_min
        self.ledger.treaties_activated += 1
        self.ledger.treaties_honoured  += treaty.times_honoured
        self._total_reward             += reward
        return True, f"treaty_activated:{treaty.times_honoured}_honoured", reward
    def pre_stage_unit(
        self,
        unit_type:    AidUnitType,
        staging_zone: str,
        donor_zone:   str,
        treaty_id:    Optional[str] = None,
        duration_min: float = 240.0,
    ) -> Tuple[bool, str, Optional[PreStagedPosition]]:
        cap = self.zone_capacities.get(donor_zone)
        if cap is None or not cap.can_give_aid:
            return False, f"donor_zone_{donor_zone}_cannot_give_aid", None
        self._stage_seq += 1
        pos = PreStagedPosition(
            position_id   = f"PSP-{self._stage_seq:05d}",
            zone_id       = staging_zone,
            unit_type     = unit_type,
            donor_zone_id = donor_zone,
            treaty_id     = treaty_id,
            staged_at_min = self.sim_time_min,
            expires_at_min= self.sim_time_min + duration_min,
        )
        self.pre_staged[pos.position_id] = pos
        cap.units_on_aid_out += 1
        cap.refresh()
        return True, f"pre_staged:{pos.position_id}", pos
    def use_pre_staged(
        self,
        request_id:   str,
        staging_zone: str,
        unit_type:    AidUnitType,
    ) -> Optional[PreStagedPosition]:
        for pos in self.pre_staged.values():
            if (not pos.is_deployed
                    and pos.zone_id == staging_zone
                    and pos.unit_type == unit_type
                    and self.sim_time_min < pos.expires_at_min):
                pos.is_deployed               = True
                pos.deployed_to_request_id    = request_id
                pos.deployed_at_min           = self.sim_time_min
                self.ledger.pre_staged_hits   += 1
                self._total_reward            += REWARD_PRE_POSITION_HIT
                return pos
        return None
    def form_convoy(
        self,
        request_id:       str,
        aid_unit_ids:     List[str],
        escort_police:    bool = False,
        escort_air:       bool = False,
    ) -> Tuple[bool, str, Optional[ConvoyDispatch]]:
        if len(aid_unit_ids) < CONVOY_MIN_UNITS:
            return False, f"convoy_needs_min_{CONVOY_MIN_UNITS}_units", None
        req = self.requests.get(request_id)
        if req is None:
            return False, "unknown_request", None
        units = [self.aid_units[uid] for uid in aid_unit_ids
                 if uid in self.aid_units]
        if len(units) < CONVOY_MIN_UNITS:
            return False, "insufficient_valid_units", None
        etas = [u.eta_minutes for u in units if u.eta_minutes > 0]
        base_eta = max(etas) if etas else TIER_DELAY_MINUTES[0]
        lead_unit = units[0]
        self._convoy_seq += 1
        convoy_id = f"CONVOY-{self._convoy_seq:04d}"
        convoy_eta = base_eta * 0.90   
        c = ConvoyDispatch(
            convoy_id         = convoy_id,
            request_id        = request_id,
            destination_zone  = req.requesting_zone,
            unit_ids          = aid_unit_ids,
            lead_unit_id      = lead_unit.aid_unit_id,
            dispatched_at_min = self.sim_time_min,
            arrival_eta_min   = self.sim_time_min + convoy_eta,
            escort_police     = escort_police,
            air_escort        = escort_air,
        )
        if escort_police or escort_air:
            reduction = c.escort_time_reduction
            c.arrival_eta_min = (
                self.sim_time_min + convoy_eta * reduction
            )
        self.convoys[convoy_id] = c
        self.ledger.convoy_dispatches += 1
        self._total_reward += REWARD_CONVOY_COORDINATED
        return True, f"convoy_formed:{convoy_id}", c
    def get_observation(self) -> Dict[str, Any]:
        active_requests = [
            r.to_dict() for r in self.requests.values()
            if r.is_active
        ]
        active_units = [
            u.to_dict() for u in self.aid_units.values()
            if u.status not in (AidStatus.COMPLETED, AidStatus.RECALLED)
        ]
        active_convoys = [
            c.to_dict() for c in self.convoys.values()
            if c.status != "dispersed"
        ]
        active_surges = [
            d.to_dict() for d in self.surge_decls.values()
            if d.is_active
        ]
        pre_staged_obs = [
            p.to_dict() for p in self.pre_staged.values()
            if not p.is_deployed and self.sim_time_min < p.expires_at_min
        ]
        zone_cap_obs = {
            zid: cap.to_dict()
            for zid, cap in self.zone_capacities.items()
        }
        active_treaties = [
            t.to_dict() for t in self.treaties.values()
            if t.is_active
        ]
        return {
            "sim_time_min":       round(self.sim_time_min, 1),
            "step_count":         self.step_count,
            "task_id":            self.task_id,
            "cascade_stage":      self._cascade_stage.value,
            "cascade_steps":      self._cascade_steps,
            "active_requests":    len(active_requests),
            "active_units":       len(active_units),
            "active_convoys":     len(active_convoys),
            "active_surges":      len(active_surges),
            "pre_staged_count":   len(pre_staged_obs),
            "requests":           active_requests,
            "aid_units":          active_units,
            "convoys":            active_convoys,
            "surge_declarations": active_surges,
            "pre_staged":         pre_staged_obs,
            "zone_capacities":    zone_cap_obs,
            "treaties":           active_treaties,
            "cascade_events":     [e.to_dict() for e in self.cascade_events[-5:]],
            "ledger_snapshot": {
                "total_requests":    self.ledger.total_requests,
                "fulfilled":         self.ledger.requests_fulfilled,
                "over_requests":     self.ledger.over_requests,
                "under_requests":    self.ledger.under_requests,
                "cascade_events":    self.ledger.cascade_events,
                "cascade_averted":   self.ledger.cascade_averted,
                "total_reward":      round(self._total_reward, 4),
            },
        }
    def get_zone_aid_status(self, zone_id: str) -> Dict[str, Any]:
        outgoing = [
            u.to_dict() for u in self.aid_units.values()
            if u.origin_zone_id == zone_id and
               u.status not in (AidStatus.COMPLETED, AidStatus.RETURNED)
        ]
        incoming = [
            u.to_dict() for u in self.aid_units.values()
            if u.destination_zone_id == zone_id and
               u.status not in (AidStatus.COMPLETED, AidStatus.RETURNED)
        ]
        cap = self.zone_capacities.get(zone_id, {})
        active_reqs = [
            r.to_dict() for r in self.requests.values()
            if r.requesting_zone == zone_id and r.is_active
        ]
        surge = [
            d.to_dict() for d in self.surge_decls.values()
            if d.zone_id == zone_id and d.is_active
        ]
        return {
            "zone_id":         zone_id,
            "capacity":        cap.to_dict() if hasattr(cap, "to_dict") else cap,
            "outgoing_units":  outgoing,
            "incoming_units":  incoming,
            "active_requests": active_reqs,
            "surge_active":    bool(surge),
            "surge_details":   surge,
        }
    def get_optimal_aid_source(
        self,
        requesting_zone: str,
        unit_type:       AidUnitType,
        units_needed:    int = 1,
        exclude_zones:   Optional[List[str]] = None,
    ) -> List[Tuple[str, int, float]]:
        excluded = set(exclude_zones or [])
        excluded.add(requesting_zone)
        candidates: List[Tuple[str, int, float]] = []
        for zid in self.active_zone_ids:
            if zid in excluded:
                continue
            cap = self.zone_capacities.get(zid)
            if cap is None or not cap.can_give_aid:
                continue
            can_give = min(cap.max_giveable, units_needed)
            if can_give <= 0:
                continue
            eta = self._compute_eta(zid, requesting_zone)
            candidates.append((zid, can_give, eta))
        candidates.sort(key=lambda x: (x[2], -x[1]))
        return candidates
    def get_episode_analytics(self) -> Dict[str, Any]:
        if self._rt_log_all:
            self.ledger.mean_response_time_min = (
                sum(self._rt_log_all) / len(self._rt_log_all)
            )
        if self._rt_log_p1:
            self.ledger.p1_mean_response_time_min = (
                sum(self._rt_log_p1) / len(self._rt_log_p1)
            )
        fulfillment_rate = (
            self.ledger.requests_fulfilled
            / max(self.ledger.total_requests, 1)
        )
        over_req_rate = (
            self.ledger.over_requests
            / max(self.ledger.total_requests, 1)
        )
        cascade_avoidance_score = self._compute_cascade_avoidance_score()
        aid_efficiency_score    = self._compute_aid_efficiency_score()
        return {
            "ledger":                 self.ledger.to_dict(),
            "fulfillment_rate":       round(fulfillment_rate, 4),
            "over_request_rate":      round(over_req_rate, 4),
            "cascade_avoidance_score":round(cascade_avoidance_score, 4),
            "aid_efficiency_score":   round(aid_efficiency_score, 4),
            "cascade_history":        [e.to_dict() for e in self.cascade_events],
            "surge_history":          [
                d.to_dict() for d in self.surge_decls.values()
            ],
            "zone_aid_balance": {
                zid: {
                    "gave":     sum(
                        1 for u in self.aid_units.values()
                        if u.origin_zone_id == zid
                    ),
                    "received": sum(
                        1 for u in self.aid_units.values()
                        if u.destination_zone_id == zid
                    ),
                }
                for zid in self.active_zone_ids
            },
            "total_step_reward":     round(self._total_reward, 4),
            "mean_step_reward":      round(
                self._total_reward / max(self.step_count, 1), 4
            ),
        }
    def get_cascade_assessment(self) -> Dict[str, Any]:
        diverted = 0
        if self._hospital_net is not None:
            try:
                diverted = self._hospital_net.count_diverted()
            except Exception:
                pass
        p1_queue = 0
        if self._incident_engine is not None:
            try:
                p1_queue = self._incident_engine.p1_count
            except Exception:
                pass
        active_aid_units = sum(
            1 for u in self.aid_units.values()
            if u.status in (AidStatus.EN_ROUTE, AidStatus.ARRIVED, AidStatus.ACTIVE)
        )
        aid_requested = self.ledger.total_requests > 0
        aid_received  = self.ledger.requests_fulfilled > 0
        hosp_score = min(1.0, diverted / max(CASCADE_FAILURE_DIVERTED_N, 1))
        p1_score   = min(1.0, p1_queue  / max(CASCADE_FAILURE_P1_QUEUE,  1))
        aid_score  = max(0.0, 1.0 - active_aid_units / max(10.0, 1.0))
        risk = hosp_score * 0.40 + p1_score * 0.40 + aid_score * 0.20
        return {
            "cascade_stage":        self._cascade_stage.value,
            "risk_score":           round(risk, 4),
            "hospitals_diverted":   diverted,
            "p1_unhandled":         p1_queue,
            "active_aid_units":     active_aid_units,
            "aid_requested":        aid_requested,
            "aid_received":         aid_received,
            "cascade_events_total": len(self.cascade_events),
            "cascade_averted":      self.ledger.cascade_averted,
        }
    def compute_grader_score(self) -> float:
        rt_score   = self._score_response_times()
        app_score  = self._score_appropriateness()
        auto_score = self._score_autonomy_guard()
        proto_score= self._score_protocol_compliance()
        casc_score = self._compute_cascade_avoidance_score()
        raw = (
            SCORE_WEIGHT_RESPONSE_TIME   * rt_score
            + SCORE_WEIGHT_APPROPRIATENESS * app_score
            + SCORE_WEIGHT_AUTONOMY_GUARD  * auto_score
            + SCORE_WEIGHT_PROTOCOL        * proto_score
            + SCORE_WEIGHT_CASCADE_AVOID   * casc_score
        )
        over_penalty  = min(0.30, self.ledger.total_over_request_penalty  / 10.0)
        under_penalty = min(0.30, self.ledger.total_under_request_penalty / 10.0)
        treaty_pen    = min(0.15, self.ledger.total_treaty_penalty        / 10.0)
        cascade_pen   = min(0.30, self.ledger.total_cascade_penalty       / 10.0)
        score = raw - over_penalty - under_penalty - treaty_pen - cascade_pen
        return max(0.0, min(1.0, score))
    def describe(self) -> Dict[str, Any]:
        return {
            "step":            self.step_count,
            "sim_time_min":    round(self.sim_time_min, 1),
            "task_id":         self.task_id,
            "active_requests": sum(1 for r in self.requests.values() if r.is_active),
            "active_units":    sum(
                1 for u in self.aid_units.values()
                if u.status not in (AidStatus.COMPLETED, AidStatus.RECALLED)
            ),
            "cascade_stage":   self._cascade_stage.value,
            "over_requests":   self.ledger.over_requests,
            "under_requests":  self.ledger.under_requests,
            "total_reward":    round(self._total_reward, 4),
        }
    def _tick_aid_units(self) -> float:
        reward = 0.0
        for au in self.aid_units.values():
            if au.status == AidStatus.DISPATCHED:
                au.eta_minutes = max(0.0, au.eta_minutes - STEP_DURATION_MINUTES)
                if au.eta_minutes <= 0:
                    au.status = AidStatus.EN_ROUTE
            elif au.status == AidStatus.EN_ROUTE:
                au.eta_minutes = max(0.0, au.eta_minutes - STEP_DURATION_MINUTES)
                if au.eta_minutes <= 0:
                    au.status          = AidStatus.ARRIVED
                    au.arrived_at_min  = self.sim_time_min
                    reward            += REWARD_AID_ARRIVED
                    self._total_reward += REWARD_AID_ARRIVED
                    self.ledger.units_dispatched += 1
                    req = self.requests.get(au.request_id)
                    if req:
                        req.total_units_received += 1
                        if req.fulfilled_at_min is None:
                            req.fulfilled_at_min = self.sim_time_min
                            req.status = AidStatus.ARRIVED
                            self.ledger.requests_fulfilled += 1
                            rt = self.sim_time_min - req.requested_at_min
                            self._rt_log_all.append(rt)
                            if req.incident_priority == "P1":
                                self._rt_log_p1.append(rt)
            elif au.status == AidStatus.ARRIVED:
                if au.released_at_min is None:
                    if (au.arrived_at_min and
                            self.sim_time_min > au.arrived_at_min + au.expected_release_min):
                        au.overstay_steps += 1
                        if au.overstay_steps > 4:   
                            reward += PENALTY_OVERSTAY
                            self._total_reward += PENALTY_OVERSTAY
            elif au.status == AidStatus.RETURNING:
                au.recall_eta_minutes = max(
                    0.0, au.recall_eta_minutes - STEP_DURATION_MINUTES
                )
                if au.recall_eta_minutes <= 0:
                    au.status         = AidStatus.COMPLETED
                    au.returned_at_min = self.sim_time_min
                    cap = self.zone_capacities.get(au.origin_zone_id)
                    if cap:
                        cap.units_on_aid_out = max(0, cap.units_on_aid_out - 1)
                        cap.refresh()
                    cap_dest = self.zone_capacities.get(au.destination_zone_id)
                    if cap_dest:
                        cap_dest.units_received_aid = max(
                            0, cap_dest.units_received_aid - 1
                        )
            elif au.status == AidStatus.RECALLED:
                au.recall_eta_minutes = max(
                    0.0, au.recall_eta_minutes - STEP_DURATION_MINUTES
                )
                if au.recall_eta_minutes <= 0:
                    au.status = AidStatus.COMPLETED
                    cap = self.zone_capacities.get(au.origin_zone_id)
                    if cap:
                        cap.units_on_aid_out = max(0, cap.units_on_aid_out - 1)
                        cap.refresh()
        return reward
    def _tick_convoys(self) -> float:
        reward = 0.0
        for convoy in self.convoys.values():
            if convoy.status != "en_route":
                continue
            if self.sim_time_min >= convoy.arrival_eta_min:
                convoy.status             = "arrived"
                convoy.actual_arrival_min = self.sim_time_min
                for uid in convoy.unit_ids:
                    au = self.aid_units.get(uid)
                    if au and au.status == AidStatus.EN_ROUTE:
                        au.status         = AidStatus.ARRIVED
                        au.arrived_at_min = self.sim_time_min
                        reward           += REWARD_AID_ARRIVED * 0.5
        return reward
    def _tick_pre_staged(self) -> float:
        expired = [
            pid for pid, p in self.pre_staged.items()
            if not p.is_deployed and self.sim_time_min >= p.expires_at_min
        ]
        for pid in expired:
            pos = self.pre_staged[pid]
            cap = self.zone_capacities.get(pos.donor_zone_id)
            if cap:
                cap.units_on_aid_out = max(0, cap.units_on_aid_out - 1)
                cap.refresh()
            del self.pre_staged[pid]
        return 0.0
    def _tick_surge_declarations(self) -> float:
        reward = 0.0
        for decl in self.surge_decls.values():
            if not decl.is_active:
                continue
        return reward
    def _check_cascade(
        self,
        hospitals_diverted: int,
        p1_queue:           int,
    ) -> float:
        reward = 0.0
        prev_stage = self._cascade_stage
        hosp_risk = hospitals_diverted >= CASCADE_FAILURE_DIVERTED_N
        p1_risk   = p1_queue >= CASCADE_FAILURE_P1_QUEUE
        aid_arriving = any(
            u.status in (AidStatus.EN_ROUTE, AidStatus.ARRIVED, AidStatus.ACTIVE)
            for u in self.aid_units.values()
        )
        if hosp_risk and p1_risk:
            if not aid_arriving:
                self._cascade_stage  = CascadeStage.ACTIVE
                self._cascade_steps += 1
                if self.ledger.total_requests == 0:
                    reward += UNDER_REQUEST_TASK9_PENALTY
                    self._total_reward  += UNDER_REQUEST_TASK9_PENALTY
                    self.ledger.under_requests      += 1
                    self.ledger.total_under_request_penalty += abs(UNDER_REQUEST_TASK9_PENALTY)
            else:
                self._cascade_stage = CascadeStage.IMMINENT
        elif hosp_risk or p1_risk:
            self._cascade_stage = CascadeStage.RISK
        else:
            if prev_stage == CascadeStage.ACTIVE:
                self._cascade_stage = CascadeStage.CONTAINED
                reward             += REWARD_CASCADE_AVERTED
                self._total_reward += REWARD_CASCADE_AVERTED
                self.ledger.cascade_averted += 1
                self.ledger.total_cascade_bonus += REWARD_CASCADE_AVERTED
            else:
                self._cascade_stage = CascadeStage.NONE
        if self._cascade_stage != prev_stage:
            self._record_cascade_event(
                hospitals_diverted, p1_queue,
                aid_requested = self.ledger.total_requests > 0,
                aid_received  = self.ledger.requests_fulfilled > 0,
            )
        return reward
    def _audit_request_balance(self) -> float:
        reward = 0.0
        zone_active_counts: Dict[str, int] = defaultdict(int)
        for req in self.requests.values():
            if req.is_active:
                zone_active_counts[req.requesting_zone] += 1
        for zone, count in zone_active_counts.items():
            if count > OVER_REQUEST_THRESHOLD:
                excess  = count - OVER_REQUEST_THRESHOLD
                penalty = OVER_REQUEST_PENALTY * excess * 0.1  
                reward += penalty
                self._total_reward += penalty
                self.ledger.total_over_request_penalty += abs(penalty)
        return reward
    def _audit_treaties(self) -> float:
        reward = 0.0
        for treaty in self.treaties.values():
            if not treaty.is_active:
                continue
            if (treaty.last_activated_min and
                    self.sim_time_min > treaty.last_activated_min + treaty.max_duration_min):
                treaty.is_active = False
        return reward
    def _find_and_dispatch(
        self,
        req:          MutualAidRequest,
        use_treaty:   bool = True,
        force_convoy: bool = False,
    ) -> Tuple[bool, str]:
        units_remaining = req.units_needed
        sources: List[Tuple[str, int, float]] = []
        for pos in list(self.pre_staged.values()):
            if units_remaining <= 0:
                break
            if (not pos.is_deployed
                    and pos.zone_id == req.requesting_zone
                    and pos.unit_type == req.unit_type
                    and self.sim_time_min < pos.expires_at_min):
                pos.is_deployed               = True
                pos.deployed_to_request_id    = req.request_id
                pos.deployed_at_min           = self.sim_time_min
                self.ledger.pre_staged_hits   += 1
                self._total_reward            += REWARD_PRE_POSITION_HIT
                au = self._spawn_aid_unit(
                    req           = req,
                    origin_zone   = pos.donor_zone_id,
                    eta_override  = PRE_POSITION_DEPLOY_MINUTES,
                    was_pre_staged= True,
                )
                req.aid_units.append(au)
                if pos.donor_zone_id not in req.source_zones:
                    req.source_zones.append(pos.donor_zone_id)
                sources.append((pos.donor_zone_id, 1, PRE_POSITION_DEPLOY_MINUTES))
                units_remaining -= 1
        if use_treaty and units_remaining > 0:
            for treaty in self.treaties.values():
                if not treaty.is_active:
                    continue
                if not treaty.involves_zone(req.requesting_zone):
                    continue
                for donor in treaty.zone_ids:
                    if units_remaining <= 0:
                        break
                    if donor == req.requesting_zone:
                        continue
                    guaranteed = treaty.guarantees_for(req.requesting_zone, donor)
                    if guaranteed <= 0:
                        continue
                    cap = self.zone_capacities.get(donor)
                    if cap is None or cap.max_giveable < 1:
                        continue
                    n_give = min(guaranteed, cap.max_giveable, units_remaining)
                    eta = self._eta_for_tier(donor, req.requesting_zone, req.tier_requested)
                    for _ in range(n_give):
                        au = self._spawn_aid_unit(req, donor, eta)
                        req.aid_units.append(au)
                        if donor not in req.source_zones:
                            req.source_zones.append(donor)
                        cap.units_on_aid_out += 1
                        cap.refresh()
                    self._update_dest_cap(req.requesting_zone, n_give)
                    sources.append((donor, n_give, eta))
                    units_remaining -= n_give
                    treaty.times_honoured += 1
                    self.ledger.treaties_honoured += 1
                    self._total_reward += REWARD_TREATY_HONOURED
                if units_remaining <= 0:
                    break
        if units_remaining > 0:
            candidates = self.get_optimal_aid_source(
                requesting_zone=req.requesting_zone,
                unit_type=req.unit_type,
                units_needed=units_remaining,
                exclude_zones=req.source_zones,
            )
            for donor, can_give, eta in candidates:
                if units_remaining <= 0:
                    break
                n_give   = min(can_give, units_remaining)
                tier_eta = self._eta_for_tier(donor, req.requesting_zone, req.tier_requested)
                cap      = self.zone_capacities[donor]
                for _ in range(n_give):
                    au = self._spawn_aid_unit(req, donor, tier_eta)
                    req.aid_units.append(au)
                    if donor not in req.source_zones:
                        req.source_zones.append(donor)
                    cap.units_on_aid_out += 1
                    cap.refresh()
                self._update_dest_cap(req.requesting_zone, n_give)
                sources.append((donor, n_give, tier_eta))
                units_remaining -= n_give
        if units_remaining > 0 and req.tier_requested.value >= AidTier.DISTRICT.value:
            eta_reserve = TIER_DELAY_MINUTES[req.tier_requested.value]
            for _ in range(units_remaining):
                au = self._spawn_aid_unit(
                    req         = req,
                    origin_zone = self._pick_reserve_zone(req.requesting_zone),
                    eta_override= eta_reserve,
                )
                req.aid_units.append(au)
                req.source_zones.append(au.origin_zone_id)
            units_remaining = 0
        if not req.aid_units:
            return False, "no_capacity_in_any_donor_zone"
        if force_convoy or len(req.aid_units) >= CONVOY_MIN_UNITS:
            self.form_convoy(
                request_id=req.request_id,
                aid_unit_ids=[u.aid_unit_id for u in req.aid_units],
            )
        self.ledger.units_dispatched += len(req.aid_units)
        return True, f"dispatched_{len(req.aid_units)}_units_from_{len(sources)}_zones"
    def _spawn_aid_unit(
        self,
        req:           MutualAidRequest,
        origin_zone:   str,
        eta_override:  Optional[float] = None,
        was_pre_staged: bool = False,
    ) -> AidUnit:
        self._unit_seq += 1
        aid_unit_id = f"AU-{self._unit_seq:06d}"
        eta = eta_override if eta_override is not None else (
            self._eta_for_tier(origin_zone, req.requesting_zone, req.tier_requested)
        )
        au = AidUnit(
            aid_unit_id          = aid_unit_id,
            unit_type            = req.unit_type,
            origin_zone_id       = origin_zone,
            destination_zone_id  = req.requesting_zone,
            tier                 = req.tier_requested,
            status               = AidStatus.DISPATCHED,
            request_id           = req.request_id,
            dispatched_at_min    = self.sim_time_min,
            eta_minutes          = eta,
            was_pre_staged       = was_pre_staged,
            expected_release_min = req.expected_duration_min,
        )
        self.aid_units[aid_unit_id] = au
        return au
    def _refresh_zone_capacities(
        self,
        fleet_available: Optional[Dict[str, int]] = None,
    ) -> None:
        for zid in self.active_zone_ids:
            z        = self._zone_meta.get(zid, {})
            zone_type = z.get("zone_type", "urban")
            avail    = 2   
            total    = 5
            if fleet_available and zid in fleet_available:
                avail = fleet_available[zid]
            elif self._fleet is not None:
                try:
                    avail_units = self._fleet.get_available_units(zone_ids=[zid])
                    avail = len(avail_units)
                    all_units = [
                        u for u in getattr(self._fleet, "units", {}).values()
                        if getattr(u, "home_zone_id", None) == zid
                    ]
                    total = len(all_units) if all_units else max(avail + 2, 5)
                except Exception:
                    pass
            if zid in self.zone_capacities:
                cap = self.zone_capacities[zid]
                cap.available_units = avail
                cap.total_units     = total
                cap.refresh()
            else:
                cap = ZoneCapacity(
                    zone_id         = zid,
                    zone_type       = zone_type,
                    total_units     = total,
                    available_units = avail,
                )
                cap.refresh()
                self.zone_capacities[zid] = cap
    def _update_dest_cap(self, zone_id: str, n_received: int) -> None:
        cap = self.zone_capacities.get(zone_id)
        if cap:
            cap.units_received_aid += n_received
    def _initialise_default_treaties(self) -> None:
        seen: Set[frozenset] = set()
        for zid in self.active_zone_ids:
            for adj in self._adjacency.get(zid, []):
                pair = frozenset({zid, adj})
                if pair in seen:
                    continue
                seen.add(pair)
                self.create_treaty(
                    zone_ids        = [zid, adj],
                    unit_guarantee  = {zid: 1, adj: 1},
                    treaty_type     = TreatyType.BILATERAL,
                    max_duration_min= 9999.0,   
                    activation_threshold_p1 = 3,
                )
        urban_zones = [
            z for z in self.active_zone_ids[:6]
            if self._zone_meta.get(z, {}).get("zone_type", "urban") == "urban"
        ]
        if len(urban_zones) >= 3:
            self.create_treaty(
                zone_ids        = urban_zones,
                unit_guarantee  = {z: 2 for z in urban_zones},
                treaty_type     = TreatyType.MULTILATERAL,
                max_duration_min= 9999.0,
                activation_threshold_p1 = 5,
            )
    def _pre_stage_fleet(self) -> None:
        hotspots: List[str] = []
        if self._demand is not None:
            try:
                hs = self._demand.get_hotspot_zones(k=4)
                hotspots = [h["zone_id"] for h in hs]
            except Exception:
                pass
        if not hotspots:
            hotspots = self.active_zone_ids[:4]
        for staging_zone in hotspots:
            donors = self._adjacency.get(staging_zone, [])
            if not donors:
                continue
            donor = donors[0]
            cap   = self.zone_capacities.get(donor)
            if cap and cap.can_give_aid:
                self.pre_stage_unit(
                    unit_type    = AidUnitType.ALS,
                    staging_zone = staging_zone,
                    donor_zone   = donor,
                    duration_min = 240.0,
                )
    def _record_cascade_event(
        self,
        hospitals_diverted: int,
        p1_unhandled:       int,
        aid_requested:      bool,
        aid_received:       bool,
    ) -> None:
        self._cascade_seq += 1
        ev = CascadeEvent(
            event_id          = f"CASC-{self._cascade_seq:04d}",
            detected_at_min   = self.sim_time_min,
            detected_at_step  = self.step_count,
            stage             = self._cascade_stage,
            hospitals_diverted= hospitals_diverted,
            p1_unhandled      = p1_unhandled,
            zones_affected    = list(self.active_zone_ids[:5]),  
            aid_requested     = aid_requested,
            aid_received      = aid_received,
            avoidable         = not aid_requested,
        )
        if self._cascade_stage == CascadeStage.ACTIVE:
            ev.cascade_penalty = UNDER_REQUEST_TASK9_PENALTY
            self.ledger.total_cascade_penalty += abs(UNDER_REQUEST_TASK9_PENALTY)
            self.ledger.cascade_events        += 1
        elif self._cascade_stage == CascadeStage.CONTAINED:
            ev.cascade_bonus = REWARD_CASCADE_AVERTED
            self.ledger.total_cascade_bonus  += REWARD_CASCADE_AVERTED
        self.cascade_events.append(ev)
    def _score_response_times(self) -> float:
        if not self._rt_log_all:
            return 1.0   
        p1_target   = TIER_DELAY_MINUTES[0]   
        all_target  = TIER_DELAY_MINUTES[0] * 1.5
        scores = []
        for rt in self._rt_log_all:
            if rt <= p1_target:
                scores.append(1.0)
            elif rt <= all_target:
                scores.append(0.6)
            elif rt <= all_target * 2.0:
                scores.append(0.3)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores)
    def _score_appropriateness(self) -> float:
        if not self.requests:
            return 1.0
        scores = []
        for req in self.requests.values():
            p = req.incident_priority
            t = req.tier_requested.value
            if p == "P1":
                scores.append(1.0 if t <= 1 else 0.4)
            elif p == "P2":
                scores.append(1.0 if t <= 2 else 0.6)
            else:
                scores.append(1.0)
        return sum(scores) / len(scores)
    def _score_autonomy_guard(self) -> float:
        violations = 0
        for cap in self.zone_capacities.values():
            effective = cap.available_units - cap.units_on_aid_out
            if effective < MIN_ZONE_AUTONOMY_UNITS:
                violations += 1
        total = max(len(self.zone_capacities), 1)
        return max(0.0, 1.0 - violations / total)
    def _score_protocol_compliance(self) -> float:
        if not self.requests:
            return 1.0
        compliant = sum(1 for r in self.requests.values() if r.protocol_correct)
        return compliant / len(self.requests)
    def _compute_cascade_avoidance_score(self) -> float:
        total_cascade = len(self.cascade_events)
        averted       = self.ledger.cascade_averted
        if total_cascade == 0:
            return 1.0
        active_cascade = sum(
            1 for e in self.cascade_events
            if e.stage == CascadeStage.ACTIVE and not e.aid_received
        )
        return max(0.0, 1.0 - active_cascade / (total_cascade + 1))
    def _compute_aid_efficiency_score(self) -> float:
        total   = len(self.aid_units)
        arrived = sum(
            1 for u in self.aid_units.values()
            if u.status in (AidStatus.ARRIVED, AidStatus.ACTIVE,
                            AidStatus.COMPLETED, AidStatus.RETURNING)
        )
        return arrived / max(total, 1)
    def _compute_eta(self, from_zone: str, to_zone: str) -> float:
        if from_zone == to_zone:
            return 0.0
        if self._traffic is not None:
            try:
                return self._traffic.get_travel_time(from_zone, to_zone)
            except Exception:
                pass
        fz = self._zone_meta.get(from_zone, {})
        tz = self._zone_meta.get(to_zone,   {})
        if fz and tz:
            return self._haversine_min(
                fz.get("lat", 18.52), fz.get("lon", 73.85),
                tz.get("lat", 18.52), tz.get("lon", 73.85),
            )
        return 10.0   
    def _eta_for_tier(
        self,
        from_zone:  str,
        to_zone:    str,
        tier:       AidTier,
    ) -> float:
        travel  = self._compute_eta(from_zone, to_zone)
        floor   = TIER_DELAY_MINUTES[tier.value]
        return max(travel, floor)
    def _check_tier_appropriateness(
        self,
        requesting_zone: str,
        tier:            AidTier,
        units_needed:    int,
        priority:        str,
    ) -> bool:
        cap = self.zone_capacities.get(requesting_zone)
        if cap is None:
            return True   
        if cap.available_units >= MIN_ZONE_AUTONOMY_UNITS + 1:
            if tier.value >= AidTier.STATE.value and priority not in ("P1",):
                return False   
        if tier == AidTier.NATIONAL and priority not in ("P1",):
            return False
        return True
    def _pick_reserve_zone(self, requesting_zone: str) -> str:
        non_adj = [
            z for z in self.active_zone_ids
            if z != requesting_zone and
               z not in self._adjacency.get(requesting_zone, [])
        ]
        if non_adj:
            return self.rng.choice(non_adj)
        return self.rng.choice(self.active_zone_ids) if self.active_zone_ids else "Z99"
    def _load_zone_meta(self) -> None:
        if self._zone_meta:
            return
        path = DATA_DIR / "city_zones.json"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for z in raw.get("zones", []):
            self._zone_meta[z["zone_id"]] = z
        adj_raw = raw.get("zone_adjacency_matrix", {})
        for zid, neighbours in adj_raw.items():
            self._adjacency[zid] = [
                n for n in neighbours if n in self._zone_meta
            ]
    @staticmethod
    def _haversine_min(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        avg_kmh: float = 40.0,
    ) -> float:
        R  = 6371.0
        φ1 = math.radians(lat1); φ2 = math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        km = R * 2 * math.asin(math.sqrt(a))
        return (km / avg_kmh) * 60.0
    def print_request_table(self) -> None:
        header = (
            f"{'REQ_ID':>10} {'ZONE':>6} {'TYPE':>5} "
            f"{'TIER':>5} {'STATUS':>12} {'UNITS':>6} "
            f"{'OVER':>5} {'PRIO':>4}"
        )
        print(header)
        print("-" * len(header))
        for req in self.requests.values():
            print(
                f"{req.request_id:>10} {req.requesting_zone:>6} "
                f"{req.unit_type.value:>5} "
                f"{req.tier_requested.value:>5} "
                f"{req.status.value:>12} "
                f"{req.total_units_received:>3}/{req.units_needed:<2} "
                f"{'YES' if req.was_over_request else 'no':>5} "
                f"{req.incident_priority:>4}"
            )
    def print_aid_unit_table(self, max_rows: int = 20) -> None:
        header = (
            f"{'AID_UNIT':>12} {'TYPE':>5} {'ORIGIN':>6} "
            f"{'DEST':>6} {'STATUS':>12} {'ETA':>6} "
            f"{'TIER':>5} {'PRE':>4}"
        )
        print(header)
        print("-" * len(header))
        count = 0
        for au in self.aid_units.values():
            if count >= max_rows:
                break
            print(
                f"{au.aid_unit_id:>12} {au.unit_type.value:>5} "
                f"{au.origin_zone_id:>6} {au.destination_zone_id:>6} "
                f"{au.status.value:>12} {au.eta_minutes:>6.1f} "
                f"{au.tier.value:>5} "
                f"{'YES' if au.was_pre_staged else 'no':>4}"
            )
            count += 1
    def print_zone_capacity_table(self) -> None:
        header = (
            f"{'ZONE':>6} {'TYPE':>10} {'TOTAL':>6} "
            f"{'AVAIL':>6} {'GIVEN':>6} {'RECV':>5} "
            f"{'MAX_GIVE':>9} {'CAN_GIVE':>9}"
        )
        print(header)
        print("-" * len(header))
        for zid, cap in self.zone_capacities.items():
            print(
                f"{zid:>6} {cap.zone_type:>10} "
                f"{cap.total_units:>6} "
                f"{cap.available_units:>6} "
                f"{cap.units_on_aid_out:>6} "
                f"{cap.units_received_aid:>5} "
                f"{cap.max_giveable:>9} "
                f"{'YES' if cap.can_give_aid else 'no':>9}"
            )
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 72)
    print("EMERGI-ENV  ·  MutualAidCoordinator smoke-test")
    print("=" * 72)
    coord = MutualAidCoordinator(seed=42, task_id=7)
    active_zones = [
        "Z01", "Z02", "Z03", "Z04", "Z05",
        "Z06", "Z07", "Z08", "Z09",
    ]
    obs = coord.reset(
        active_zone_ids  = active_zones,
        task_id          = 7,
        sim_time_minutes = 540.0,
    )
    print(f"\n✓  Reset OK — {len(coord.active_zone_ids)} active zones")
    print(f"   Treaties created: {len(coord.treaties)}")
    print(f"   Zone capacities:  {len(coord.zone_capacities)}")
    print(f"   Pre-staged units: {len(coord.pre_staged)}")
    print("\n── Test 1: Basic adjacent-zone aid request (Task 7 MCI) ──")
    ok, msg, req = coord.request_aid(
        requesting_zone       = "Z05",
        unit_type             = AidUnitType.MICU,
        units_needed          = 2,
        tier                  = AidTier.ADJACENT,
        incident_id           = "INC-0042",
        incident_priority     = "P1",
        reason                = "mci_mass_casualty_20_victims",
        expected_duration_min = 90.0,
    )
    print(f"✓  Request: ok={ok} msg='{msg}'")
    if req:
        print(f"   req_id={req.request_id} units={len(req.aid_units)} "
              f"sources={req.source_zones} over={req.was_over_request}")
    print("\n── Test 2: Zone surge declaration ──")
    ok2, msg2, decl = coord.declare_surge(
        zone_id         = "Z02",
        p1_queue_size   = 9,
        available_units = 1,
        tier            = AidTier.DISTRICT,
        trigger_reason  = "fleet_exhausted",
    )
    print(f"✓  Surge: ok={ok2} msg='{msg2}'")
    if decl:
        print(f"   decl_id={decl.declaration_id} "
              f"auto_requests={decl.auto_aid_requests}")
    print("\n── Test 3: Over-requesting (should trigger penalty) ──")
    for i in range(5):
        ok3, msg3, _ = coord.request_aid(
            requesting_zone   = "Z05",
            unit_type         = AidUnitType.ALS,
            units_needed      = 1,
            tier              = AidTier.ADJACENT,
            incident_priority = "P2",
            reason            = f"test_over_request_{i}",
        )
    print(f"✓  Over-request audit: over_count={coord.ledger.over_requests} "
          f"penalty={coord.ledger.total_over_request_penalty:.3f}")
    print("\n── Test 4: Treaty activation ──")
    if coord.treaties:
        tid = list(coord.treaties.keys())[0]
        treaty = coord.treaties[tid]
        ok4, msg4, rwd4 = coord.activate_treaty(
            treaty_id        = tid,
            requesting_zone  = treaty.zone_ids[0],
            p1_count         = 6,
        )
        print(f"✓  Treaty {tid}: ok={ok4} rwd={rwd4:.3f} "
              f"honoured={treaty.times_honoured} broken={treaty.times_broken}")
    print("\n── Test 5: Convoy formation ──")
    if req and len(req.aid_units) >= CONVOY_MIN_UNITS:
        ok5, msg5, convoy = coord.form_convoy(
            request_id    = req.request_id,
            aid_unit_ids  = [u.aid_unit_id for u in req.aid_units],
            escort_police = True,
        )
        print(f"✓  Convoy: ok={ok5} msg='{msg5}'")
        if convoy:
            print(f"   convoy_id={convoy.convoy_id} "
                  f"eta={convoy.arrival_eta_min:.1f} min "
                  f"police_escort={convoy.escort_police}")
    else:
        print("   (skipped — insufficient aid units for convoy)")
    print(f"\n── Test 6: Stepping 30 steps ──")
    total_rwd = 0.0
    for s in range(30):
        obs2, rwd = coord.step(
            sim_time_minutes        = 540.0 + (s + 1) * STEP_DURATION_MINUTES,
            fleet_available         = {z: max(1, 4 - s // 10) for z in active_zones},
            hospital_diverted_count = min(s // 5, 7),
            p1_queue_count          = min(s, 12),
        )
        total_rwd += rwd
        if s % 6 == 0:
            print(f"  Step {s+1:02d}: active_reqs={obs2['active_requests']} "
                  f"active_units={obs2['active_units']} "
                  f"cascade={obs2['cascade_stage']} "
                  f"rwd={rwd:+.4f}")
    print(f"\n  Total reward over 30 steps: {total_rwd:+.4f}")
    print("\n── Test 7: Pre-staging (Task 9) ──")
    coord9 = MutualAidCoordinator(seed=42, task_id=9)
    coord9.reset(active_zone_ids=active_zones, task_id=9, sim_time_minutes=600.0)
    print(f"✓  Task 9 reset: pre_staged={len(coord9.pre_staged)} "
          f"treaties={len(coord9.treaties)}")
    ok6, msg6, pos = coord9.pre_stage_unit(
        unit_type    = AidUnitType.MICU,
        staging_zone = "Z05",
        donor_zone   = "Z06",
        duration_min = 180.0,
    )
    print(f"✓  Pre-stage: ok={ok6} pos_id={pos.position_id if pos else None}")
    ok7, msg7, req7 = coord9.request_aid(
        requesting_zone   = "Z05",
        unit_type         = AidUnitType.MICU,
        units_needed      = 1,
        tier              = AidTier.ADJACENT,
        incident_priority = "P1",
        reason            = "task9_city_wide_surge",
    )
    print(f"✓  Aid request (uses pre-staged): ok={ok7} "
          f"pre_staged_hits={coord9.ledger.pre_staged_hits}")
    print("\n── Test 8: Cascade failure detection (Task 9) ──")
    for s in range(10):
        _, rwd = coord9.step(
            sim_time_minutes        = 600.0 + (s + 1) * STEP_DURATION_MINUTES,
            hospital_diverted_count = 6,   
            p1_queue_count          = 10,  
        )
    print(f"✓  Cascade stage after 10 steps: {coord9._cascade_stage.value}")
    print(f"   cascade_events: {len(coord9.cascade_events)}")
    print(f"   cascade_penalty: {coord9.ledger.total_cascade_penalty:.3f}")
    print(f"   under_requests:  {coord9.ledger.under_requests}")
    print("\n── Test 9: Recall and release ──")
    if coord.aid_units:
        first_unit = list(coord.aid_units.values())[0]
        if first_unit.status in (AidStatus.EN_ROUTE, AidStatus.ARRIVED,
                                 AidStatus.ACTIVE, AidStatus.DISPATCHED):
            ok8, pen8 = coord.recall_aid_unit(first_unit.aid_unit_id)
            print(f"✓  Recall {first_unit.aid_unit_id}: ok={ok8} penalty={pen8:.3f}")
    if req and req.aid_units:
        last_unit = req.aid_units[-1]
        last_unit.status = AidStatus.ARRIVED  
        ok9, msg9, rwd9 = coord.release_aid_unit(last_unit.aid_unit_id)
        print(f"✓  Release {last_unit.aid_unit_id}: ok={ok9} rwd={rwd9:.3f}")
    print("\n── Zone capacity table ──")
    coord.print_zone_capacity_table()
    print("\n── Request table ──")
    coord.print_request_table()
    print("\n── Aid unit table ──")
    coord.print_aid_unit_table(max_rows=12)
    grader_score = coord.compute_grader_score()
    print(f"\n✓  Grader score: {grader_score:.4f}")
    cascade_assess = coord9.get_cascade_assessment()
    print(f"✓  Task 9 cascade assessment: risk={cascade_assess['risk_score']:.4f} "
          f"stage={cascade_assess['cascade_stage']}")
    analytics = coord.get_episode_analytics()
    print("\n  Episode analytics (Task 7 episode):")
    for k, v in analytics.items():
        if not isinstance(v, dict) and not isinstance(v, list):
            print(f"   {k}: {v}")
    optimal = coord.get_optimal_aid_source(
        requesting_zone = "Z05",
        unit_type       = AidUnitType.ALS,
        units_needed    = 2,
    )
    print(f"\n✓  Optimal aid sources for Z05: {optimal[:3]}")
    print(f"\n  describe(): {coord.describe()}")
    print("\n✅  MutualAidCoordinator smoke-test PASSED")