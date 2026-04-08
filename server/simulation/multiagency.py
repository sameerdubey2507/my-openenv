from __future__ import annotations
import json
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent.parent / "data"
STEP_DURATION_MINUTES: float = 3.0
AGENCY_RESPONSE_TIMES: Dict[str, Tuple[float, float]] = {
    "police_urban":     ( 6.0,  2.0),
    "police_suburban":  ( 9.0,  3.0),
    "police_rural":     (14.0,  5.0),
    "fire_urban":       ( 8.0,  2.5),
    "fire_suburban":    (12.0,  3.5),
    "fire_rural":       (18.0,  6.0),
    "hazmat_urban":     (20.0,  6.0),
    "hazmat_suburban":  (28.0,  8.0),
    "hazmat_rural":     (40.0, 12.0),
    "coast_guard":      (25.0, 10.0),
}
SCENE_SECURITY_DURATION:    Tuple[float, float] = ( 5.0,  2.0)   
EXTRICATION_DURATION:       Tuple[float, float] = (15.0,  6.0)   
MCI_EXTRICATION_EXTRA_MIN:  float               =  8.0           
STRUCTURE_CLEARANCE_MINS:   Tuple[float, float] = (12.0,  4.0)   
HAZMAT_DECON_DURATION:      Tuple[float, float] = (22.0,  8.0)   
HAZMAT_PATIENT_DECON_MINS:  float               =  4.0           
WATER_RESCUE_SETUP_MINS:    Tuple[float, float] = (18.0,  6.0)   
COORDINATION_EXCELLENT:  float = 0.85
COORDINATION_GOOD:       float = 0.65
COORDINATION_POOR:       float = 0.40
PENALTY_SOLO_EMS_TRAPPED:     float = -0.25
PENALTY_PREMATURE_EMS:        float = -0.18
PENALTY_PREMATURE_TRANSPORT:  float = -0.18
PENALTY_HAZMAT_VIOLATION:     float = -0.30
PENALTY_NO_COMMAND_POST_MCI:  float = -0.10
PENALTY_AGENCY_WAITING:       float = -0.05   
PENALTY_WRONG_AGENCY_ORDER:   float = -0.12
PENALTY_MISSING_AGENCY:       float = -0.15   
BONUS_CORRECT_DISPATCH:       float = +0.12
BONUS_PREREQ_MET_ON_TIME:     float = +0.08
BONUS_COMMAND_POST_ACTIVATED: float = +0.06
BONUS_PRE_NOTIFICATION:       float = +0.05
BONUS_EXCELLENT_COORDINATION: float = +0.10
BONUS_RAPID_HANDOFF:          float = +0.04   
WAITING_GRACE_STEPS:     int = 2
COMMAND_POST_MCI_THRESHOLD:   int = 5     
COMMAND_POST_AGENCY_THRESHOLD:int = 2     
MAX_POLICE_PER_SCENE:   int = 6
MAX_FIRE_UNITS_PER_SCENE: int = 4
MAX_HAZMAT_UNITS:       int = 2
class AgencyType(str, Enum):
    EMS         = "ems"
    POLICE      = "police"
    FIRE        = "fire"
    HAZMAT      = "hazmat"
    COAST_GUARD = "coast_guard"
    NDRF        = "ndrf"          
class SceneStage(str, Enum):
    REPORTED         = "reported"
    AGENCIES_ALERTED = "agencies_alerted"
    POLICE_EN_ROUTE  = "police_en_route"
    POLICE_ON_SCENE  = "police_on_scene"
    SCENE_SECURED    = "scene_secured"
    FIRE_EN_ROUTE    = "fire_en_route"
    FIRE_ON_SCENE    = "fire_on_scene"
    EXTRICATION_IN_PROGRESS = "extrication_in_progress"
    DECON_IN_PROGRESS       = "decon_in_progress"
    CLEARANCE_GRANTED       = "clearance_granted"     
    EMS_TREATING            = "ems_treating"
    EMS_TRANSPORTING        = "ems_transporting"
    SCENE_CLOSED            = "scene_closed"
    ABORTED                 = "aborted"
class CoordinationViolationType(str, Enum):
    SOLO_EMS_TRAPPED    = "solo_ems_trapped"
    PREMATURE_EMS       = "premature_ems_entry"
    PREMATURE_TRANSPORT = "premature_transport"
    HAZMAT_VIOLATION    = "hazmat_violation"
    NO_COMMAND_POST     = "no_command_post"
    WRONG_AGENCY_ORDER  = "wrong_agency_order"
    MISSING_AGENCY      = "missing_agency"
    OVER_REQUEST        = "over_request"
    SCENE_ABANDONED     = "scene_abandoned"
class AgencyAvailabilityStatus(str, Enum):
    AVAILABLE   = "available"
    COMMITTED   = "committed"
    EN_ROUTE    = "en_route"
    ON_SCENE    = "on_scene"
    RETURNING   = "returning"
    UNAVAILABLE = "unavailable"
@dataclass
class AgencyUnit:
    unit_id:          str
    agency_type:      AgencyType
    call_sign:        str
    home_zone_id:     str
    current_zone_id:  str
    status:           AgencyAvailabilityStatus = AgencyAvailabilityStatus.AVAILABLE
    assigned_scene_id: Optional[str] = None
    eta_minutes:       float = 0.0
    time_on_scene_min: float = 0.0
    personnel_count:   int   = 4     
    has_extrication_gear:  bool = False
    has_decon_capability:  bool = False
    has_water_rescue:      bool = False
    has_aerial_platform:   bool = False
    total_deployments:     int   = 0
    total_scene_time_min:  float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id":        self.unit_id,
            "agency_type":    self.agency_type.value,
            "call_sign":      self.call_sign,
            "status":         self.status.value,
            "current_zone":   self.current_zone_id,
            "assigned_scene": self.assigned_scene_id,
            "eta_minutes":    round(self.eta_minutes, 1),
            "personnel":      self.personnel_count,
        }
@dataclass
class CoordinationViolation:
    violation_id:  str
    scene_id:      str
    incident_id:   str
    violation_type: CoordinationViolationType
    detected_at_step:  int
    detected_at_min:   float
    penalty_applied:   float
    description:       str
    ems_unit_id:       Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id":   self.violation_id,
            "scene_id":       self.scene_id,
            "incident_id":    self.incident_id,
            "type":           self.violation_type.value,
            "detected_at_min":round(self.detected_at_min, 1),
            "penalty":        round(self.penalty_applied, 4),
            "description":    self.description,
        }
@dataclass
class CommandPost:
    cp_id:            str
    scene_id:         str
    zone_id:          str
    activated_at_min: float
    activated_at_step: int
    incident_commander: str   
    agencies_represented: List[str] = field(default_factory=list)
    can_request_ndrf:     bool = False
    can_request_army:     bool = False
    staging_area_active:  bool = True
    media_exclusion_zone: bool = True
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cp_id":                self.cp_id,
            "scene_id":             self.scene_id,
            "zone_id":              self.zone_id,
            "activated_at_min":     round(self.activated_at_min, 1),
            "incident_commander":   self.incident_commander,
            "agencies_represented": self.agencies_represented,
            "staging_area_active":  self.staging_area_active,
        }
@dataclass
class AgencyArrivalRecord:
    agency_type:      AgencyType
    unit_id:          str
    dispatched_at_min: float
    arrived_at_min:    float
    cleared_at_min:    Optional[float] = None   
    personnel_count:   int = 4
    equipment_deployed: List[str] = field(default_factory=list)
    @property
    def response_time_min(self) -> float:
        return self.arrived_at_min - self.dispatched_at_min
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agency":            self.agency_type.value,
            "unit_id":           self.unit_id,
            "response_time_min": round(self.response_time_min, 1),
            "arrived_at_min":    round(self.arrived_at_min,     1),
            "cleared_at_min":    (round(self.cleared_at_min,   1)
                                  if self.cleared_at_min else None),
            "equipment":         self.equipment_deployed,
        }
@dataclass
class MultiAgencyScene:
    scene_id:          str
    incident_id:       str
    zone_id:           str
    zone_type:         str                        
    required_agency:   str                        
    agency_flags:      List[str]                  
    victim_count:      int
    is_mci:            bool
    mci_type:          Optional[str]
    created_at_min:    float
    created_at_step:   int
    stage:             SceneStage = SceneStage.REPORTED
    stage_history:     List[Tuple[str, float]] = field(default_factory=list)
    current_subtask_eta: float = 0.0
    current_subtask_label: str = ""
    agencies_dispatched:  Set[AgencyType]         = field(default_factory=set)
    agencies_on_scene:    Set[AgencyType]          = field(default_factory=set)
    arrival_records:      List[AgencyArrivalRecord]= field(default_factory=list)
    agency_units_assigned: Dict[str, List[str]]    = field(default_factory=dict)  
    scene_secured_at_min:    Optional[float] = None
    extrication_cleared_min: Optional[float] = None
    decon_cleared_min:       Optional[float] = None
    structure_cleared_min:   Optional[float] = None
    ems_clearance_granted_at_min: Optional[float] = None
    ems_arrived_at_min:      Optional[float] = None
    ems_treating_since_min:  Optional[float] = None
    transport_cleared_at_min:Optional[float] = None
    resolved_at_min:         Optional[float] = None
    command_post:            Optional[CommandPost] = None
    command_post_required:   bool = False
    violations:              List[CoordinationViolation] = field(default_factory=list)
    violation_seq:           int = 0
    ems_waiting_steps:       int = 0     
    agency_waiting_steps:    Dict[str, int] = field(default_factory=dict)
    pre_notification_given:  bool = False
    coordination_quality_score: Optional[float] = None
    closed_correctly:           bool = False
    step_reward_accumulator:    float = 0.0
    def advance_stage(self, new_stage: SceneStage, sim_time_min: float) -> None:
        self.stage = new_stage
        self.stage_history.append((new_stage.value, round(sim_time_min, 1)))
    @property
    def total_elapsed_minutes(self) -> float:
        if self.resolved_at_min:
            return self.resolved_at_min - self.created_at_min
        return 0.0
    @property
    def has_police(self) -> bool:
        return "police" in self.agency_flags
    @property
    def has_fire(self) -> bool:
        return "fire" in self.agency_flags
    @property
    def has_hazmat(self) -> bool:
        return "hazmat" in self.agency_flags
    @property
    def has_coast_guard(self) -> bool:
        return "coast_guard" in self.agency_flags
    @property
    def ems_cleared_to_treat(self) -> bool:
        return self.ems_clearance_granted_at_min is not None
    @property
    def ems_cleared_to_transport(self) -> bool:
        return self.transport_cleared_at_min is not None
    @property
    def total_violations_penalty(self) -> float:
        return sum(v.penalty_applied for v in self.violations)
    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "scene_id":              self.scene_id,
            "incident_id":           self.incident_id,
            "zone_id":               self.zone_id,
            "required_agency":       self.required_agency,
            "agency_flags":          self.agency_flags,
            "victim_count":          self.victim_count,
            "is_mci":                self.is_mci,
            "stage":                 self.stage.value,
            "agencies_dispatched":   [a.value for a in self.agencies_dispatched],
            "agencies_on_scene":     [a.value for a in self.agencies_on_scene],
            "scene_secured":         self.scene_secured_at_min is not None,
            "extrication_cleared":   self.extrication_cleared_min is not None,
            "decon_cleared":         self.decon_cleared_min is not None,
            "ems_cleared_to_treat":  self.ems_cleared_to_treat,
            "ems_cleared_transport": self.ems_cleared_to_transport,
            "current_subtask":       self.current_subtask_label,
            "subtask_eta_min":       round(self.current_subtask_eta, 1),
            "command_post_active":   self.command_post is not None,
            "violations_count":      len(self.violations),
            "pre_notification_given":self.pre_notification_given,
            "created_at_min":        round(self.created_at_min, 1),
        }
    def to_full_state_dict(self) -> Dict[str, Any]:
        base = self.to_observation_dict()
        base.update({
            "violations":           [v.to_dict() for v in self.violations],
            "arrival_records":      [r.to_dict() for r in self.arrival_records],
            "stage_history":        self.stage_history,
            "command_post":         self.command_post.to_dict() if self.command_post else None,
            "coordination_score":   (round(self.coordination_quality_score, 4)
                                     if self.coordination_quality_score is not None else None),
            "closed_correctly":     self.closed_correctly,
            "total_violation_penalty": round(self.total_violations_penalty, 4),
            "ems_waiting_steps":    self.ems_waiting_steps,
            "resolved_at_min":      (round(self.resolved_at_min, 1)
                                     if self.resolved_at_min else None),
        })
        return base
@dataclass
class AgencyResourcePool:
    zone_id:      str
    police_units: List[AgencyUnit] = field(default_factory=list)
    fire_units:   List[AgencyUnit] = field(default_factory=list)
    hazmat_units: List[AgencyUnit] = field(default_factory=list)
    coast_guard_units: List[AgencyUnit] = field(default_factory=list)
    ndrf_units:   List[AgencyUnit] = field(default_factory=list)
    def get_available(self, agency_type: AgencyType) -> List[AgencyUnit]:
        pool_map = {
            AgencyType.POLICE:      self.police_units,
            AgencyType.FIRE:        self.fire_units,
            AgencyType.HAZMAT:      self.hazmat_units,
            AgencyType.COAST_GUARD: self.coast_guard_units,
            AgencyType.NDRF:        self.ndrf_units,
        }
        pool = pool_map.get(agency_type, [])
        return [u for u in pool if u.status == AgencyAvailabilityStatus.AVAILABLE]
    def total_available(self, agency_type: AgencyType) -> int:
        return len(self.get_available(agency_type))
    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "zone_id":       self.zone_id,
            "police_avail":  self.total_available(AgencyType.POLICE),
            "fire_avail":    self.total_available(AgencyType.FIRE),
            "hazmat_avail":  self.total_available(AgencyType.HAZMAT),
            "cg_avail":      self.total_available(AgencyType.COAST_GUARD),
        }
class MultiAgencyCoordinator:
    def __init__(
        self,
        seed:          int  = 42,
        task_id:       int  = 1,
        zone_data:     Optional[Dict[str, Any]] = None,
    ) -> None:
        self.seed    = seed
        self.task_id = task_id
        self.rng     = random.Random(seed)
        self.np_rng  = np.random.RandomState(seed)
        self._zone_meta:  Dict[str, Any]  = zone_data or {}
        self._active_zones: List[str]     = []
        self.active_scenes:   Dict[str, MultiAgencyScene]  = {}
        self.closed_scenes:   List[MultiAgencyScene]       = []
        self._scene_seq:      int = 0
        self._violation_seq:  int = 0
        self.agency_pools:    Dict[str, AgencyResourcePool] = {}
        self.sim_time_min: float = 480.0
        self.step_count:   int   = 0
        self._total_bonus:   float = 0.0
        self._total_penalty: float = 0.0
        self._step_rewards:  List[float] = []
        self.all_violations: List[CoordinationViolation] = []
        self._load_zone_meta()
    def reset(
        self,
        active_zone_ids:  List[str],
        task_id:          int   = 1,
        sim_time_minutes: float = 480.0,
    ) -> Dict[str, Any]:
        self.task_id       = task_id
        self.sim_time_min  = sim_time_minutes
        self.step_count    = 0
        self._active_zones = list(active_zone_ids)
        self.rng    = random.Random(self.seed + task_id * 17)
        self.np_rng = np.random.RandomState(self.seed + task_id * 17)
        self.active_scenes  = {}
        self.closed_scenes  = []
        self._scene_seq     = 0
        self._violation_seq = 0
        self._total_bonus   = 0.0
        self._total_penalty = 0.0
        self._step_rewards  = []
        self.all_violations = []
        self._spawn_agency_pools(active_zone_ids)
        return self.get_coordinator_observation()
    def step(self, sim_time_minutes: float) -> Tuple[Dict[str, Any], float]:
        self.sim_time_min = sim_time_minutes
        self.step_count  += 1
        step_reward       = 0.0
        for scene in list(self.active_scenes.values()):
            scene_reward = self._tick_scene(scene)
            step_reward += scene_reward
        step_reward += self._tick_agency_units()
        self._step_rewards.append(step_reward)
        return self.get_coordinator_observation(), step_reward
    def register_incident(
        self,
        incident_id:     str,
        zone_id:         str,
        required_agency: str,   
        agency_flags:    List[str],
        victim_count:    int    = 1,
        is_mci:          bool   = False,
        mci_type:        Optional[str] = None,
    ) -> Optional[MultiAgencyScene]:
        if not agency_flags and required_agency == "ems_only":
            return None
        zone_meta  = self._zone_meta.get(zone_id, {})
        zone_type  = zone_meta.get("zone_type", "urban")
        needs_cp   = (
            victim_count >= COMMAND_POST_MCI_THRESHOLD
            or len(agency_flags) >= COMMAND_POST_AGENCY_THRESHOLD
            or is_mci
        )
        self._scene_seq += 1
        scene_id = f"MAS-{self._scene_seq:04d}"
        scene = MultiAgencyScene(
            scene_id         = scene_id,
            incident_id      = incident_id,
            zone_id          = zone_id,
            zone_type        = zone_type,
            required_agency  = required_agency,
            agency_flags     = list(agency_flags),
            victim_count     = victim_count,
            is_mci           = is_mci,
            mci_type         = mci_type,
            created_at_min   = self.sim_time_min,
            created_at_step  = self.step_count,
            command_post_required = needs_cp,
        )
        scene.advance_stage(SceneStage.AGENCIES_ALERTED, self.sim_time_min)
        self.active_scenes[scene_id] = scene
        for flag in agency_flags:
            scene.agency_waiting_steps[flag] = 0
        return scene
    def dispatch_agency(
        self,
        scene_id:          str,
        agency_type:       AgencyType,
        unit_id:           str,
        from_zone_id:      str,
        personnel_count:   int  = 4,
        equipment:         Optional[List[str]] = None,
    ) -> Tuple[bool, str, float]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            return False, f"Unknown scene {scene_id}", 0.0
        if agency_type == AgencyType.EMS:
            return False, "Use FleetSimulator.dispatch_unit() for EMS units", 0.0
        over = self._check_over_request(scene, agency_type)
        if over:
            penalty_val = -0.05
            self._record_violation(
                scene, CoordinationViolationType.OVER_REQUEST,
                f"Over-dispatch: {agency_type.value} units exceed safe limit",
                penalty_val
            )
        eta = self._compute_agency_eta(from_zone_id, scene.zone_id,
                                       agency_type, scene.zone_type)
        pool = self.agency_pools.get(from_zone_id)
        unit = None
        if pool:
            avail = pool.get_available(agency_type)
            unit = next((u for u in avail if u.unit_id == unit_id), None)
            if unit is None and avail:
                unit = avail[0]
        if unit:
            unit.status           = AgencyAvailabilityStatus.EN_ROUTE
            unit.assigned_scene_id = scene_id
            unit.eta_minutes       = eta
        scene.agencies_dispatched.add(agency_type)
        if agency_type.value not in scene.agency_units_assigned:
            scene.agency_units_assigned[agency_type.value] = []
        scene.agency_units_assigned[agency_type.value].append(unit_id)
        if agency_type == AgencyType.POLICE and scene.stage == SceneStage.AGENCIES_ALERTED:
            scene.advance_stage(SceneStage.POLICE_EN_ROUTE, self.sim_time_min)
        elif agency_type == AgencyType.FIRE and SceneStage.POLICE_ON_SCENE in [
            SceneStage(s) for s, _ in scene.stage_history
        ]:
            scene.advance_stage(SceneStage.FIRE_EN_ROUTE, self.sim_time_min)
        scene.current_subtask_eta   = eta
        scene.current_subtask_label = f"{agency_type.value}_en_route"
        reward = 0.0
        if self._dispatch_is_correct_sequence(scene, agency_type):
            reward += BONUS_CORRECT_DISPATCH
            self._total_bonus += BONUS_CORRECT_DISPATCH
        return True, f"{agency_type.value} dispatched ETA {eta:.1f} min", eta
    def pre_notify_receiving_agency(
        self,
        scene_id:    str,
        agency_type: AgencyType,
    ) -> Tuple[bool, float]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            return False, 0.0
        if scene.pre_notification_given:
            return True, 0.0  
        if agency_type.value in scene.agency_flags:
            scene.pre_notification_given = True
            scene.step_reward_accumulator += BONUS_PRE_NOTIFICATION
            self._total_bonus += BONUS_PRE_NOTIFICATION
            return True, BONUS_PRE_NOTIFICATION
        return False, 0.0
    def activate_command_post(
        self,
        scene_id:           str,
        incident_commander: str = "fire",
        agencies_present:   Optional[List[str]] = None,
    ) -> Tuple[bool, str, Optional[CommandPost]]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            return False, "Unknown scene", None
        if scene.command_post is not None:
            return False, "Command post already active", scene.command_post
        cp = CommandPost(
            cp_id              = f"CP-{scene_id}",
            scene_id           = scene_id,
            zone_id            = scene.zone_id,
            activated_at_min   = self.sim_time_min,
            activated_at_step  = self.step_count,
            incident_commander = incident_commander,
            agencies_represented = agencies_present or list(scene.agency_flags),
        )
        if scene.victim_count >= 20:
            cp.can_request_ndrf  = True
        if scene.victim_count >= 50:
            cp.can_request_army  = True
        scene.command_post = cp
        reward = BONUS_COMMAND_POST_ACTIVATED
        scene.step_reward_accumulator += reward
        self._total_bonus += reward
        return True, "Command post activated", cp
    def notify_ems_arrived(
        self,
        scene_id:   str,
        ems_unit_id: str,
    ) -> Tuple[bool, str, float]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            return False, "Unknown scene", 0.0
        scene.ems_arrived_at_min = self.sim_time_min
        penalty = 0.0
        if scene.ems_clearance_granted_at_min is not None:
            wait = self.sim_time_min - scene.ems_clearance_granted_at_min
            if wait <= 3.0:
                bonus = BONUS_RAPID_HANDOFF
                scene.step_reward_accumulator += bonus
                self._total_bonus += bonus
            return True, "scene_already_cleared", 0.0
        needs_clearance = scene.has_police or scene.has_fire or scene.has_hazmat
        if not needs_clearance:
            self._grant_ems_clearance(scene)
            return True, "no_clearance_required", 0.0
        if scene.has_police and scene.scene_secured_at_min is None:
            if self._is_trapped_or_hazardous(scene):
                penalty = PENALTY_SOLO_EMS_TRAPPED
                self._record_violation(
                    scene, CoordinationViolationType.SOLO_EMS_TRAPPED,
                    f"EMS unit {ems_unit_id} arrived at unsecured scene requiring "
                    f"police clearance — MAJOR PROTOCOL VIOLATION",
                    penalty
                )
            else:
                penalty = PENALTY_PREMATURE_EMS
                self._record_violation(
                    scene, CoordinationViolationType.PREMATURE_EMS,
                    f"EMS {ems_unit_id} arrived before police scene clearance",
                    penalty
                )
            self._total_penalty += abs(penalty)
            return False, "awaiting_police_clearance", penalty
        if scene.has_hazmat and scene.decon_cleared_min is None:
            penalty = PENALTY_HAZMAT_VIOLATION
            self._record_violation(
                scene, CoordinationViolationType.HAZMAT_VIOLATION,
                f"EMS {ems_unit_id} entered scene before HAZMAT decon corridor",
                penalty
            )
            self._total_penalty += abs(penalty)
            return False, "awaiting_hazmat_decon", penalty
        if scene.has_fire:
            if self._requires_extrication(scene) and scene.extrication_cleared_min is None:
                return False, "awaiting_fire_extrication", 0.0
            if self._requires_structure_clearance(scene) and scene.structure_cleared_min is None:
                return False, "awaiting_structure_clearance", 0.0
        self._grant_ems_clearance(scene)
        return True, "clearance_granted", 0.0
    def notify_ems_transporting(
        self,
        scene_id:    str,
        ems_unit_id: str,
    ) -> Tuple[bool, str, float]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            return False, "Unknown scene", 0.0
        penalty = 0.0
        if scene.has_fire and self._requires_extrication(scene):
            if scene.extrication_cleared_min is None:
                penalty = PENALTY_PREMATURE_TRANSPORT
                self._record_violation(
                    scene, CoordinationViolationType.PREMATURE_TRANSPORT,
                    f"EMS {ems_unit_id} attempted transport before fire extrication "
                    f"clearance — patient spinal injury risk",
                    penalty
                )
                self._total_penalty += abs(penalty)
                return False, "extrication_not_complete", penalty
        scene.transport_cleared_at_min = self.sim_time_min
        scene.advance_stage(SceneStage.EMS_TRANSPORTING, self.sim_time_min)
        return True, "transport_cleared", 0.0
    def get_scene_for_incident(self, incident_id: str) -> Optional[MultiAgencyScene]:
        for sc in self.active_scenes.values():
            if sc.incident_id == incident_id:
                return sc
        for sc in self.closed_scenes:
            if sc.incident_id == incident_id:
                return sc
        return None
    def get_scene_state(self, scene_id: str) -> Optional[Dict[str, Any]]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            scene = next((s for s in self.closed_scenes if s.scene_id == scene_id), None)
        if scene is None:
            return None
        return scene.to_full_state_dict()
    def close_scene(
        self,
        scene_id:  str,
        resolved:  bool = True,
    ) -> Tuple[bool, float, Optional[float]]:
        scene = self.active_scenes.get(scene_id)
        if scene is None:
            return False, 0.0, None
        scene.resolved_at_min = self.sim_time_min
        scene.advance_stage(
            SceneStage.SCENE_CLOSED if resolved else SceneStage.ABORTED,
            self.sim_time_min
        )
        reward = self._check_missing_agencies_on_close(scene)
        if scene.command_post_required and scene.command_post is None:
            penalty = PENALTY_NO_COMMAND_POST_MCI
            self._record_violation(
                scene, CoordinationViolationType.NO_COMMAND_POST,
                "MCI scene closed without Unified Command post activation",
                penalty
            )
            reward += penalty
            self._total_penalty += abs(penalty)
        score = self._compute_coordination_score(scene)
        scene.coordination_quality_score = score
        scene.closed_correctly = (
            resolved
            and len(scene.violations) == 0
            and score >= COORDINATION_GOOD
        )
        if score >= COORDINATION_EXCELLENT:
            reward += BONUS_EXCELLENT_COORDINATION
            self._total_bonus += BONUS_EXCELLENT_COORDINATION
        for ag_type_str, unit_ids in scene.agency_units_assigned.items():
            zone = scene.zone_id
            pool = self.agency_pools.get(zone)
            if pool:
                ag_type = AgencyType(ag_type_str)
                for uid in unit_ids:
                    for u in (pool.get_available(ag_type) +
                              [u for u in self._get_all_pool_units(pool, ag_type)]):
                        if u.unit_id == uid:
                            u.status           = AgencyAvailabilityStatus.RETURNING
                            u.assigned_scene_id = None
        del self.active_scenes[scene_id]
        self.closed_scenes.append(scene)
        return True, reward, score
    def evaluate_dispatch_action(
        self,
        incident_id:          str,
        dispatched_agencies:  List[str],   
        ems_unit_id:          str,
        dispatched_ems_alone: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        scene = self.get_scene_for_incident(incident_id)
        if scene is None:
            return 1.0, {"reason": "ems_only_no_agency_required"}
        required_flags = set(scene.agency_flags)
        dispatched_set = set(dispatched_agencies)
        missing        = required_flags - dispatched_set
        extra          = dispatched_set - required_flags - {"ems"}
        breakdown: Dict[str, Any] = {
            "required":   list(required_flags),
            "dispatched": list(dispatched_set),
            "missing":    list(missing),
            "extra":      list(extra),
        }
        if dispatched_ems_alone and required_flags:
            solo_penalty = PENALTY_SOLO_EMS_TRAPPED if self._is_trapped_or_hazardous(scene) else PENALTY_PREMATURE_EMS
            breakdown["solo_ems_penalty"] = solo_penalty
            return max(0.0, 0.30 + solo_penalty), breakdown  
        if not required_flags:
            agency_score = 1.0
        else:
            agency_score = (len(required_flags) - len(missing)) / len(required_flags)
        sequence_ok = self._dispatch_sequence_correct(scene, dispatched_agencies)
        seq_bonus   = 0.10 if sequence_ok else 0.0
        over_penalty = 0.0
        for ag in extra:
            over_penalty -= 0.05
        total = min(1.0, max(0.0, agency_score + seq_bonus + over_penalty))
        breakdown["agency_score"]   = round(agency_score, 3)
        breakdown["sequence_bonus"] = seq_bonus
        breakdown["over_penalty"]   = over_penalty
        breakdown["final_score"]    = round(total, 3)
        return total, breakdown
    def check_solo_ems_violation(self, incident_id: str) -> Tuple[bool, float]:
        scene = self.get_scene_for_incident(incident_id)
        if scene is None or not scene.agency_flags:
            return False, 0.0
        if not scene.agencies_dispatched - {AgencyType.EMS}:
            if self._is_trapped_or_hazardous(scene):
                return True, PENALTY_SOLO_EMS_TRAPPED
            return True, PENALTY_PREMATURE_EMS
        return False, 0.0
    def get_coordinator_observation(self) -> Dict[str, Any]:
        scenes_obs = [sc.to_observation_dict() for sc in self.active_scenes.values()]
        agency_avail = {
            zid: pool.to_observation_dict()
            for zid, pool in self.agency_pools.items()
        }
        return {
            "active_scenes":          len(self.active_scenes),
            "closed_scenes":          len(self.closed_scenes),
            "violations_total":       len(self.all_violations),
            "sim_time_min":           round(self.sim_time_min, 1),
            "step_count":             self.step_count,
            "scenes":                 scenes_obs,
            "agency_availability":    agency_avail,
        }
    def get_episode_analytics(self) -> Dict[str, Any]:
        all_scenes = list(self.active_scenes.values()) + self.closed_scenes
        total_scenes      = len(all_scenes)
        closed_correctly  = sum(1 for s in self.closed_scenes if s.closed_correctly)
        violation_counts: Dict[str, int] = defaultdict(int)
        for v in self.all_violations:
            violation_counts[v.violation_type.value] += 1
        scores = [
            s.coordination_quality_score
            for s in self.closed_scenes
            if s.coordination_quality_score is not None
        ]
        mean_score = sum(scores) / max(len(scores), 1) if scores else 0.0
        cp_activated = sum(1 for s in all_scenes if s.command_post is not None)
        cp_required  = sum(1 for s in all_scenes if s.command_post_required)
        mean_rt_police = self._mean_response_time(AgencyType.POLICE)
        mean_rt_fire   = self._mean_response_time(AgencyType.FIRE)
        return {
            "total_scenes":              total_scenes,
            "scenes_closed_correctly":   closed_correctly,
            "coordination_compliance_pct": round(
                100 * closed_correctly / max(total_scenes, 1), 1
            ),
            "mean_coordination_score":   round(mean_score, 4),
            "violations_by_type":        dict(violation_counts),
            "total_violations":          len(self.all_violations),
            "total_bonus_earned":        round(self._total_bonus,   4),
            "total_penalty_incurred":    round(self._total_penalty, 4),
            "net_coord_reward":          round(self._total_bonus - self._total_penalty, 4),
            "command_post_rate":         round(cp_activated / max(cp_required, 1), 3),
            "mean_police_response_min":  round(mean_rt_police, 1),
            "mean_fire_response_min":    round(mean_rt_fire,   1),
        }
    def describe(self) -> Dict[str, Any]:
        return {
            "step":            self.step_count,
            "sim_time_min":    round(self.sim_time_min, 1),
            "task_id":         self.task_id,
            "active_scenes":   len(self.active_scenes),
            "closed_scenes":   len(self.closed_scenes),
            "violations":      len(self.all_violations),
            "total_bonus":     round(self._total_bonus,   4),
            "total_penalty":   round(self._total_penalty, 4),
        }
    def _tick_scene(self, scene: MultiAgencyScene) -> float:
        reward = scene.step_reward_accumulator
        scene.step_reward_accumulator = 0.0
        if scene.stage in (SceneStage.SCENE_CLOSED, SceneStage.ABORTED):
            return reward
        if scene.current_subtask_eta > 0:
            scene.current_subtask_eta = max(
                0.0, scene.current_subtask_eta - STEP_DURATION_MINUTES
            )
            if scene.current_subtask_eta <= 0:
                reward += self._fire_subtask_completion(scene)
        if (scene.ems_arrived_at_min is not None
                and scene.ems_clearance_granted_at_min is None):
            scene.ems_waiting_steps += 1
            if scene.ems_waiting_steps > WAITING_GRACE_STEPS:
                reward += PENALTY_AGENCY_WAITING
                self._total_penalty += abs(PENALTY_AGENCY_WAITING)
        return reward
    def _fire_subtask_completion(self, scene: MultiAgencyScene) -> float:
        reward  = 0.0
        stage   = scene.stage
        sim_t   = self.sim_time_min
        if stage == SceneStage.POLICE_EN_ROUTE:
            scene.advance_stage(SceneStage.POLICE_ON_SCENE, sim_t)
            scene.agencies_on_scene.add(AgencyType.POLICE)
            self._record_arrival(scene, AgencyType.POLICE)
            duration = self._sample(*SCENE_SECURITY_DURATION)
            scene.current_subtask_eta   = duration
            scene.current_subtask_label = "police_securing_scene"
            scene.advance_stage(SceneStage.SCENE_SECURED, sim_t)
        elif stage == SceneStage.SCENE_SECURED:
            scene.scene_secured_at_min = sim_t
            scene.current_subtask_label = "awaiting_fire_or_ems"
            if AgencyType.FIRE in scene.agencies_dispatched:
                fire_eta = self._compute_agency_eta(
                    scene.zone_id, scene.zone_id, AgencyType.FIRE, scene.zone_type
                )
                scene.current_subtask_eta   = fire_eta
                scene.current_subtask_label = "fire_en_route"
                scene.advance_stage(SceneStage.FIRE_EN_ROUTE, sim_t)
            else:
                if not scene.has_fire and not scene.has_hazmat:
                    self._grant_ems_clearance(scene)
            reward += BONUS_PREREQ_MET_ON_TIME
            self._total_bonus += BONUS_PREREQ_MET_ON_TIME
        elif stage == SceneStage.FIRE_EN_ROUTE:
            scene.advance_stage(SceneStage.FIRE_ON_SCENE, sim_t)
            scene.agencies_on_scene.add(AgencyType.FIRE)
            self._record_arrival(scene, AgencyType.FIRE)
            if self._requires_extrication(scene):
                base, sigma = EXTRICATION_DURATION
                extra = max(0, scene.victim_count - 1) * MCI_EXTRICATION_EXTRA_MIN
                duration = self._sample(base + extra, sigma)
                scene.current_subtask_eta   = duration
                scene.current_subtask_label = "fire_extricating"
                scene.advance_stage(SceneStage.EXTRICATION_IN_PROGRESS, sim_t)
            elif scene.has_hazmat:
                duration = self._sample(*HAZMAT_DECON_DURATION)
                scene.current_subtask_eta   = duration
                scene.current_subtask_label = "hazmat_decon"
                scene.advance_stage(SceneStage.DECON_IN_PROGRESS, sim_t)
            else:
                duration = self._sample(*STRUCTURE_CLEARANCE_MINS)
                scene.current_subtask_eta   = duration
                scene.current_subtask_label = "structure_clearance"
        elif stage == SceneStage.EXTRICATION_IN_PROGRESS:
            scene.extrication_cleared_min = sim_t
            reward += BONUS_PREREQ_MET_ON_TIME
            self._total_bonus += BONUS_PREREQ_MET_ON_TIME
            if scene.has_hazmat:
                decon_duration = self._sample(*HAZMAT_DECON_DURATION)
                extra_per_victim = scene.victim_count * HAZMAT_PATIENT_DECON_MINS
                scene.current_subtask_eta   = decon_duration + extra_per_victim
                scene.current_subtask_label = "hazmat_decon"
                scene.advance_stage(SceneStage.DECON_IN_PROGRESS, sim_t)
            else:
                self._grant_ems_clearance(scene)
                reward += BONUS_PREREQ_MET_ON_TIME
                self._total_bonus += BONUS_PREREQ_MET_ON_TIME
        elif stage == SceneStage.DECON_IN_PROGRESS:
            scene.decon_cleared_min = sim_t
            scene.structure_cleared_min = sim_t
            self._grant_ems_clearance(scene)
            reward += BONUS_PREREQ_MET_ON_TIME
            self._total_bonus += BONUS_PREREQ_MET_ON_TIME
        elif stage == SceneStage.CLEARANCE_GRANTED:
            scene.advance_stage(SceneStage.EMS_TREATING, sim_t)
        return reward
    def _grant_ems_clearance(self, scene: MultiAgencyScene) -> None:
        scene.ems_clearance_granted_at_min = self.sim_time_min
        scene.current_subtask_label = "ems_cleared_to_treat"
        scene.advance_stage(SceneStage.CLEARANCE_GRANTED, self.sim_time_min)
    def _tick_agency_units(self) -> float:
        reward = 0.0
        for pool in self.agency_pools.values():
            for unit in self._get_all_pool_units_all_types(pool):
                if unit.status == AgencyAvailabilityStatus.EN_ROUTE:
                    unit.eta_minutes -= STEP_DURATION_MINUTES
                    if unit.eta_minutes <= 0:
                        unit.status = AgencyAvailabilityStatus.ON_SCENE
                        unit.eta_minutes = 0.0
                elif unit.status == AgencyAvailabilityStatus.ON_SCENE:
                    unit.time_on_scene_min += STEP_DURATION_MINUTES
                    unit.total_scene_time_min += STEP_DURATION_MINUTES
                elif unit.status == AgencyAvailabilityStatus.RETURNING:
                    unit.time_on_scene_min += STEP_DURATION_MINUTES
                    if unit.time_on_scene_min > 6.0:
                        unit.status            = AgencyAvailabilityStatus.AVAILABLE
                        unit.time_on_scene_min = 0.0
        return reward
    def _record_violation(
        self,
        scene:          MultiAgencyScene,
        violation_type: CoordinationViolationType,
        description:    str,
        penalty:        float,
        ems_unit_id:    Optional[str] = None,
    ) -> CoordinationViolation:
        self._violation_seq += 1
        v = CoordinationViolation(
            violation_id    = f"VIO-{self._violation_seq:04d}",
            scene_id        = scene.scene_id,
            incident_id     = scene.incident_id,
            violation_type  = violation_type,
            detected_at_step= self.step_count,
            detected_at_min = self.sim_time_min,
            penalty_applied = penalty,
            description     = description,
            ems_unit_id     = ems_unit_id,
        )
        scene.violations.append(v)
        scene.step_reward_accumulator += penalty
        self.all_violations.append(v)
        return v
    def _record_arrival(
        self,
        scene:       MultiAgencyScene,
        agency_type: AgencyType,
    ) -> None:
        units = scene.agency_units_assigned.get(agency_type.value, ["UNKNOWN"])
        uid   = units[0] if units else "UNKNOWN"
        dispatch_min = scene.created_at_min
        for stage_name, t in scene.stage_history:
            if f"{agency_type.value}_en_route" in stage_name:
                dispatch_min = t
                break
        record = AgencyArrivalRecord(
            agency_type       = agency_type,
            unit_id           = uid,
            dispatched_at_min = dispatch_min,
            arrived_at_min    = self.sim_time_min,
            personnel_count   = 4,
            equipment_deployed= self._default_equipment(agency_type),
        )
        scene.arrival_records.append(record)
    def _compute_coordination_score(self, scene: MultiAgencyScene) -> float:
        required = set(scene.agency_flags)
        on_scene  = {a.value for a in scene.agencies_on_scene}
        if required:
            completeness = len(required & on_scene) / len(required)
        else:
            completeness = 1.0
        seq_score = self._score_sequence_compliance(scene)
        prereq_score = 1.0
        if scene.has_police and scene.scene_secured_at_min is None:
            prereq_score -= 0.40
        if scene.has_fire and self._requires_extrication(scene) \
                and scene.extrication_cleared_min is None:
            prereq_score -= 0.30
        if scene.has_hazmat and scene.decon_cleared_min is None:
            prereq_score -= 0.50
        wait_penalty = min(0.30, scene.ems_waiting_steps * 0.03)
        prereq_score = max(0.0, prereq_score - wait_penalty)
        cp_score = 1.0
        if scene.command_post_required:
            cp_score = 1.0 if scene.command_post is not None else 0.0
        rt_score = self._score_response_times(scene)
        vio_deduction = min(0.50, sum(
            abs(v.penalty_applied) * 0.5 for v in scene.violations
        ))
        raw = (
            completeness * 0.30
            + seq_score  * 0.25
            + prereq_score * 0.20
            + cp_score   * 0.10
            + rt_score   * 0.15
        ) - vio_deduction
        return max(0.0, min(1.0, raw))
    def _score_sequence_compliance(self, scene: MultiAgencyScene) -> float:
        stage_names = [s for s, _ in scene.stage_history]
        if scene.required_agency == "ems_only":
            return 1.0
        if scene.required_agency == "police_ems":
            p_idx = self._stage_index(stage_names, "scene_secured")
            e_idx = self._stage_index(stage_names, "ems_treating")
            if p_idx >= 0 and e_idx >= 0:
                return 1.0 if p_idx < e_idx else 0.3
            return 0.5  
        if scene.required_agency in ("fire_ems",):
            f_idx = self._stage_index(stage_names, "clearance_granted")
            e_idx = self._stage_index(stage_names, "ems_treating")
            if f_idx >= 0 and e_idx >= 0:
                return 1.0 if f_idx < e_idx else 0.2
            return 0.5
        if scene.required_agency == "all_three":
            p_idx = self._stage_index(stage_names, "scene_secured")
            f_idx = self._stage_index(stage_names, "clearance_granted")
            e_idx = self._stage_index(stage_names, "ems_treating")
            if p_idx >= 0 and f_idx >= 0 and e_idx >= 0:
                return 1.0 if p_idx < f_idx < e_idx else 0.1
            completed = sum([p_idx >= 0, f_idx >= 0, e_idx >= 0])
            return 0.3 * completed / 3
        if scene.required_agency == "hazmat":
            d_idx = self._stage_index(stage_names, "clearance_granted")
            e_idx = self._stage_index(stage_names, "ems_treating")
            if d_idx >= 0 and e_idx >= 0:
                return 1.0 if d_idx < e_idx else 0.0
            return 0.5
        return 0.7   
    def _score_response_times(self, scene: MultiAgencyScene) -> float:
        if not scene.arrival_records:
            return 0.5
        scores: List[float] = []
        for rec in scene.arrival_records:
            zone_type = scene.zone_type
            key = f"{rec.agency_type.value}_{zone_type}"
            target_mean, _ = AGENCY_RESPONSE_TIMES.get(key, (15.0, 5.0))
            rt = rec.response_time_min
            if rt <= target_mean:
                scores.append(1.0)
            elif rt <= target_mean * 1.5:
                scores.append(0.6)
            elif rt <= target_mean * 2.0:
                scores.append(0.3)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores)
    def _check_missing_agencies_on_close(self, scene: MultiAgencyScene) -> float:
        reward = 0.0
        for flag in scene.agency_flags:
            try:
                ag_type = AgencyType(flag)
            except ValueError:
                continue
            if ag_type not in scene.agencies_on_scene:
                penalty = PENALTY_MISSING_AGENCY
                self._record_violation(
                    scene, CoordinationViolationType.MISSING_AGENCY,
                    f"Scene closed without required {flag} agency on scene",
                    penalty
                )
                reward += penalty
                self._total_penalty += abs(penalty)
        return reward
    def _is_trapped_or_hazardous(self, scene: MultiAgencyScene) -> bool:
        hazardous_required = {"all_three", "hazmat", "coast_guard"}
        entrap_types = {
            "trapped_victim_rta", "building_collapse", "industrial_explosion",
            "flood_rescue", "chemical_spill",
        }
        if scene.required_agency in hazardous_required:
            return True
        if scene.mci_type and scene.mci_type.lower() in entrap_types:
            return True
        return scene.has_hazmat
    def _requires_extrication(self, scene: MultiAgencyScene) -> bool:
        extrication_types = {
            "all_three", "fire_ems",
        }
        mci_entrap = {
            "building_collapse", "industrial_explosion",
            "train_accident", "road_traffic_accident",
        }
        if scene.required_agency in extrication_types:
            return True
        if scene.mci_type and scene.mci_type.lower() in mci_entrap:
            return True
        return False
    def _requires_structure_clearance(self, scene: MultiAgencyScene) -> bool:
        clearance_types = {
            "building_collapse", "industrial_explosion", "fire_multi_victim",
        }
        if scene.mci_type and scene.mci_type.lower() in clearance_types:
            return True
        return False
    def _dispatch_is_correct_sequence(
        self,
        scene:       MultiAgencyScene,
        agency_type: AgencyType,
    ) -> bool:
        if scene.required_agency == "all_three":
            if agency_type == AgencyType.POLICE:
                return True
            if agency_type == AgencyType.FIRE:
                return AgencyType.POLICE in scene.agencies_dispatched
        if scene.required_agency == "police_ems":
            return agency_type == AgencyType.POLICE
        if scene.required_agency in ("fire_ems", "hazmat"):
            return agency_type in (AgencyType.FIRE, AgencyType.HAZMAT)
        return True
    def _dispatch_sequence_correct(
        self,
        scene:               MultiAgencyScene,
        dispatched_agencies: List[str],
    ) -> bool:
        req = scene.required_agency
        d_set = set(dispatched_agencies)
        if req == "all_three":
            return "police" in d_set and "fire" in d_set
        if req == "police_ems":
            return "police" in d_set
        if req == "fire_ems":
            return "fire" in d_set
        if req == "hazmat":
            return "fire" in d_set or "hazmat" in d_set
        return True
    def _check_over_request(
        self, scene: MultiAgencyScene, agency_type: AgencyType
    ) -> bool:
        current_count = len(scene.agency_units_assigned.get(agency_type.value, []))
        limits = {
            AgencyType.POLICE:  MAX_POLICE_PER_SCENE,
            AgencyType.FIRE:    MAX_FIRE_UNITS_PER_SCENE,
            AgencyType.HAZMAT:  MAX_HAZMAT_UNITS,
        }
        limit = limits.get(agency_type, 99)
        return current_count >= limit
    def _spawn_agency_pools(self, active_zone_ids: List[str]) -> None:
        self.agency_pools = {}
        for zid in active_zone_ids:
            z = self._zone_meta.get(zid, {})
            zone_type   = z.get("zone_type", "urban")
            pop_density = float(z.get("population_density_per_sqkm", 200))
            pool = AgencyResourcePool(zone_id=zid)
            n_police = self._units_for_zone("police", zone_type, pop_density)
            n_fire   = self._units_for_zone("fire",   zone_type, pop_density)
            n_hazmat = self._units_for_zone("hazmat", zone_type, pop_density)
            has_coast = z.get("geography", {}).get("coastal", False)
            seq = self._scene_seq * 100
            for i in range(n_police):
                seq += 1
                pool.police_units.append(AgencyUnit(
                    unit_id      = f"POL-{zid}-{seq:03d}",
                    agency_type  = AgencyType.POLICE,
                    call_sign    = f"PCR-{seq:03d}",
                    home_zone_id = zid,
                    current_zone_id = zid,
                    personnel_count = self.rng.randint(2, 5),
                ))
            for i in range(n_fire):
                seq += 1
                has_extric = zone_type in ("urban", "industrial") or i == 0
                pool.fire_units.append(AgencyUnit(
                    unit_id      = f"FIRE-{zid}-{seq:03d}",
                    agency_type  = AgencyType.FIRE,
                    call_sign    = f"FT-{seq:03d}",
                    home_zone_id = zid,
                    current_zone_id = zid,
                    personnel_count = self.rng.randint(4, 8),
                    has_extrication_gear = has_extric,
                ))
            for i in range(n_hazmat):
                seq += 1
                pool.hazmat_units.append(AgencyUnit(
                    unit_id      = f"HZM-{zid}-{seq:03d}",
                    agency_type  = AgencyType.HAZMAT,
                    call_sign    = f"HZM-{seq:03d}",
                    home_zone_id = zid,
                    current_zone_id = zid,
                    personnel_count = self.rng.randint(4, 6),
                    has_decon_capability = True,
                ))
            if has_coast:
                seq += 1
                pool.coast_guard_units.append(AgencyUnit(
                    unit_id      = f"CG-{zid}-{seq:03d}",
                    agency_type  = AgencyType.COAST_GUARD,
                    call_sign    = f"CG-{seq:03d}",
                    home_zone_id = zid,
                    current_zone_id = zid,
                    personnel_count = 6,
                    has_water_rescue = True,
                ))
            self.agency_pools[zid] = pool
    @staticmethod
    def _units_for_zone(agency: str, zone_type: str, pop_density: float) -> int:
        base = {
            "police": {"urban": 4, "suburban": 3, "peri_urban": 2,
                       "rural": 1, "industrial": 3, "highway": 2},
            "fire":   {"urban": 3, "suburban": 2, "peri_urban": 2,
                       "rural": 1, "industrial": 3, "highway": 1},
            "hazmat": {"urban": 1, "suburban": 1, "peri_urban": 0,
                       "rural": 0, "industrial": 2, "highway": 0},
        }
        n = base.get(agency, {}).get(zone_type, 1)
        if pop_density > 10000:
            n += 1
        return max(0, n)
    def _compute_agency_eta(
        self,
        from_zone_id: str,
        to_zone_id:   str,
        agency_type:  AgencyType,
        zone_type:    str,
    ) -> float:
        key = f"{agency_type.value}_{zone_type}"
        mean, sigma = AGENCY_RESPONSE_TIMES.get(key, (12.0, 4.0))
        if from_zone_id != to_zone_id:
            mean += 3.0   
        return max(2.0, float(self.np_rng.normal(mean, sigma)))
    def _sample(self, mean: float, sigma: float) -> float:
        return max(mean * 0.3, float(self.np_rng.normal(mean, sigma)))
    def _mean_response_time(self, agency_type: AgencyType) -> float:
        times: List[float] = []
        all_scenes = list(self.active_scenes.values()) + self.closed_scenes
        for sc in all_scenes:
            for rec in sc.arrival_records:
                if rec.agency_type == agency_type:
                    times.append(rec.response_time_min)
        return sum(times) / max(len(times), 1) if times else 0.0
    @staticmethod
    def _stage_index(stage_names: List[str], target: str) -> int:
        for i, name in enumerate(stage_names):
            if target in name:
                return i
        return -1
    @staticmethod
    def _default_equipment(agency_type: AgencyType) -> List[str]:
        equip_map = {
            AgencyType.POLICE:      ["barricade", "cones", "radio"],
            AgencyType.FIRE:        ["hose", "ladder", "jaws_of_life", "breathing_apparatus"],
            AgencyType.HAZMAT:      ["decon_tent", "chem_suits", "air_monitor", "antidote_kit"],
            AgencyType.COAST_GUARD: ["zodiac_boat", "life_rings", "diving_gear"],
            AgencyType.NDRF:        ["search_dogs", "sonar", "collapse_kit"],
        }
        return equip_map.get(agency_type, [])
    def _get_all_pool_units(
        self, pool: AgencyResourcePool, agency_type: AgencyType
    ) -> List[AgencyUnit]:
        pool_map = {
            AgencyType.POLICE:      pool.police_units,
            AgencyType.FIRE:        pool.fire_units,
            AgencyType.HAZMAT:      pool.hazmat_units,
            AgencyType.COAST_GUARD: pool.coast_guard_units,
            AgencyType.NDRF:        pool.ndrf_units,
        }
        return pool_map.get(agency_type, [])
    def _get_all_pool_units_all_types(
        self, pool: AgencyResourcePool
    ) -> List[AgencyUnit]:
        return (pool.police_units + pool.fire_units + pool.hazmat_units
                + pool.coast_guard_units + pool.ndrf_units)
    def _load_zone_meta(self) -> None:
        if self._zone_meta:
            return
        path = DATA_DIR / "city_zones.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for z in raw.get("zones", []):
                self._zone_meta[z["zone_id"]] = z
    def print_scene_table(self) -> None:
        header = (
            f"{'SCENE_ID':>10} {'INC_ID':>12} {'ZONE':>5} {'STAGE':>28} "
            f"{'AGENCIES':>20} {'VIOLS':>6} {'EMS_CLR':>8}"
        )
        print(header)
        print("-" * len(header))
        for sc in self.active_scenes.values():
            disp = " | ".join(a.value for a in sc.agencies_dispatched) or "—"
            clr  = "YES" if sc.ems_cleared_to_treat else "no"
            print(
                f"{sc.scene_id:>10} {sc.incident_id:>12} {sc.zone_id:>5} "
                f"{sc.stage.value:>28} {disp:>20} "
                f"{len(sc.violations):>6} {clr:>8}"
            )
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 72)
    print("EMERGI-ENV  ·  MultiAgencyCoordinator smoke-test")
    print("=" * 72)
    coord = MultiAgencyCoordinator(seed=42, task_id=7)
    active_zones = ["Z01","Z02","Z03","Z04","Z05","Z06","Z07","Z08","Z09"]
    obs = coord.reset(
        active_zone_ids  = active_zones,
        task_id          = 7,
        sim_time_minutes = 540.0,
    )
    print(f"\n✓  Coordinator reset — {len(coord.agency_pools)} zone pools")
    for zid, pool in list(coord.agency_pools.items())[:3]:
        p = pool.to_observation_dict()
        print(f"   {zid}: police={p['police_avail']} fire={p['fire_avail']} "
              f"hazmat={p['hazmat_avail']}")
    print("\n── Scenario A: Trapped victim RTA (all_three) ──")
    sc_a = coord.register_incident(
        incident_id     = "INC-0001",
        zone_id         = "Z05",
        required_agency = "all_three",
        agency_flags    = ["police", "fire"],
        victim_count    = 2,
        is_mci          = False,
        mci_type        = "road_traffic_accident",
    )
    print(f"✓  Scene registered: {sc_a.scene_id} | stage={sc_a.stage.value}")
    ok, bonus = coord.pre_notify_receiving_agency(sc_a.scene_id, AgencyType.FIRE)
    print(f"✓  Pre-notification Fire: {ok} bonus={bonus}")
    cleared, reason, penalty = coord.notify_ems_arrived(sc_a.scene_id, "AMB-M001")
    print(f"✓  EMS arrived (premature): cleared={cleared} reason={reason} "
          f"penalty={penalty}")
    ok_p, msg_p, eta_p = coord.dispatch_agency(
        scene_id       = sc_a.scene_id,
        agency_type    = AgencyType.POLICE,
        unit_id        = "POL-Z05-001",
        from_zone_id   = "Z05",
    )
    print(f"✓  Police dispatched: {ok_p} — {msg_p}")
    ok_f, msg_f, eta_f = coord.dispatch_agency(
        scene_id       = sc_a.scene_id,
        agency_type    = AgencyType.FIRE,
        unit_id        = "FIRE-Z05-001",
        from_zone_id   = "Z05",
    )
    print(f"✓  Fire dispatched: {ok_f} — {msg_f}")
    ok_cp, msg_cp, cp = coord.activate_command_post(
        sc_a.scene_id, incident_commander="fire"
    )
    print(f"✓  Command post: {ok_cp} — {msg_cp}")
    print(f"\n  Stepping 20 steps:")
    total_rwd = 0.0
    for s in range(20):
        obs2, rwd = coord.step(540.0 + (s + 1) * STEP_DURATION_MINUTES)
        total_rwd += rwd
        sc_now = coord.active_scenes.get(sc_a.scene_id)
        if sc_now and s % 4 == 0:
            print(f"   Step {s+1:02d}: stage={sc_now.stage.value:35s} "
                  f"ems_clr={sc_now.ems_cleared_to_treat} rwd={rwd:+.4f}")
    sc_check = coord.active_scenes.get(sc_a.scene_id)
    if sc_check:
        t_ok, t_reason, t_penalty = coord.notify_ems_transporting(
            sc_a.scene_id, "AMB-M001"
        )
        print(f"\n✓  EMS transport: {t_ok} — {t_reason} penalty={t_penalty}")
    closed_ok, final_rwd, final_score = coord.close_scene(sc_a.scene_id)
    print(f"✓  Scene closed: {closed_ok} | final_rwd={final_rwd:+.4f} | "
          f"coord_score={final_score:.4f}")
    print("\n── Scenario B: Chemical spill (hazmat) ──")
    sc_b = coord.register_incident(
        incident_id     = "INC-0005",
        zone_id         = "Z02",
        required_agency = "hazmat",
        agency_flags    = ["fire", "hazmat"],
        victim_count    = 8,
        is_mci          = True,
        mci_type        = "chemical_spill",
    )
    print(f"✓  HAZMAT scene: {sc_b.scene_id}")
    score, breakdown = coord.evaluate_dispatch_action(
        incident_id         = "INC-0005",
        dispatched_agencies = ["fire", "hazmat"],
        ems_unit_id         = "AMB-A003",
        dispatched_ems_alone= False,
    )
    print(f"✓  Dispatch eval (correct): score={score:.3f} | {breakdown}")
    score_bad, breakdown_bad = coord.evaluate_dispatch_action(
        incident_id         = "INC-0005",
        dispatched_agencies = [],
        ems_unit_id         = "AMB-A004",
        dispatched_ems_alone= True,
    )
    print(f"✓  Dispatch eval (solo EMS): score={score_bad:.3f} | {breakdown_bad}")
    print("\n── Scenario C: Festival stampede (police_ems) ──")
    sc_c = coord.register_incident(
        incident_id     = "INC-0010",
        zone_id         = "Z07",
        required_agency = "police_ems",
        agency_flags    = ["police"],
        victim_count    = 25,
        is_mci          = True,
        mci_type        = "stampede",
    )
    ok_cp2, _, cp2 = coord.activate_command_post(sc_c.scene_id,
                                                  incident_commander="police")
    print(f"✓  CP for stampede: {ok_cp2}")
    solo_vio, solo_pen = coord.check_solo_ems_violation("INC-0010")
    print(f"✓  Solo EMS check: violation={solo_vio} penalty={solo_pen}")
    dispatch_police = coord.dispatch_agency(
        sc_c.scene_id, AgencyType.POLICE, "POL-Z07-001", "Z07"
    )
    print(f"✓  Police dispatched to stampede: {dispatch_police[0]} ETA={dispatch_police[2]:.1f}")
    print(f"\n  Final scene table:")
    coord.print_scene_table()
    analytics = coord.get_episode_analytics()
    print(f"\n  Episode analytics:")
    for k, v in analytics.items():
        print(f"   {k}: {v}")
    print(f"\n  Describe: {coord.describe()}")
    print("\n✅  MultiAgencyCoordinator smoke-test PASSED")