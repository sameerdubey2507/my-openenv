from __future__ import annotations
import copy
import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum, unique
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from collections import defaultdict, deque
from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from server.models import (
    EmergiBaseModel,
    ImmutableEmergiModel,
    ModelRegistry,
    ActionType,
    AgencyType,
    DecayModel,
    HospitalTier,
    HospitalType,
    IncidentType,
    ProtocolRule,
    Season,
    SeverityLevel,
    TaskDifficulty,
    TaskID,
    TriageTag,
    UnitStatus,
    UnitType,
    ZoneType,
    HospitalID,
    IncidentID,
    Minutes,
    Probability,
    Score,
    StepNumber,
    TemplateID,
    UnitID,
    ZoneID,
    CapacitySnapshot,
    CommunicationsState,
    CrewFatigueState,
    GeoCoordinate,
    MutualAidRequest,
    PatientVitals,
    ACTIVE_ZONES_PER_EPISODE,
    COMMS_FAILURE_PROBABILITY_PER_STEP,
    CREW_FATIGUE_THRESHOLD_HOURS,
    CREW_SWAP_DEPLOY_DELAY_MIN,
    DEMAND_FORECAST_HORIZON_HOURS,
    DEMAND_FORECAST_NOISE_PCT,
    DIVERSION_FLAG_RESET_STEPS,
    DIVERSION_PENALTY,
    DIVERSION_REDIRECT_DELAY_MIN,
    DIVERSION_THRESHOLD_PCT,
    FLEET_SIZE_DEFAULT,
    INCIDENT_QUEUE_MAX,
    INCIDENT_QUEUE_MIN,
    MAX_STEPS_PER_EPISODE,
    MUTUAL_AID_DELAY_MIN,
    MUTUAL_AID_OVER_REQUEST_PENALTY,
    NUM_HOSPITALS,
    NUM_ZONES,
    PEAK_EVENING_HOURS,
    PEAK_HOUR_MULTIPLIER_MAX,
    PEAK_MORNING_HOURS,
    PROTOCOL_COMPLIANCE_MAX_BONUS,
    SCORE_CEILING,
    SCORE_FLOOR,
    SECONDARY_SLOWDOWN_DURATION_STEPS,
    SECONDARY_SLOWDOWN_MULTIPLIER,
    TIMESTEP_MINUTES,
    TRAFFIC_UPDATE_EVERY_N_STEPS,
    WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY,
    HospitalSpecialty,
    CONDITION_REQUIRED_UNIT,
    CONDITION_REQUIRED_SPECIALTY,
    RPMScore,
    SurvivalCurve,
    clamp_score,
    new_episode_id,
    utc_now_iso,
    validate_hospital_id,
    validate_unit_id,
    validate_zone_id,
    __schema_version__,
)
logger = logging.getLogger("emergi_env.models.state")
__state_version__: int = 7          
__module__: str        = "server.models.state"
EVENT_LOG_MAX_SIZE: int = 5_000
MAX_CHECKPOINTS_PER_EPISODE: int = 10
SCENE_ARRIVAL_PROCESSING_STEPS: int = 1
HOSPITAL_HANDOFF_PROCESSING_STEPS: int = 1
RETURN_BASE_PROCESSING_STEPS: int = 1
SURVIVAL_RECALC_EVERY_N_STEPS: int = 1
HOSPITAL_OCCUPANCY_SIGMA: float = 0.08       
HOSPITAL_BASE_ER_OCCUPANCY_MEAN: float = 0.62
HOSPITAL_BASE_ICU_OCCUPANCY_MEAN: float = 0.58
ZONE_PEAK_DEMAND_MULTIPLIER: float = 2.20
ZONE_OFF_PEAK_MULTIPLIER: float = 0.55
TRANSFER_STABILISATION_MINUTES: float = 15.0
CASCADE_LEVEL_1_THRESHOLD: float = 0.55     
CASCADE_LEVEL_2_THRESHOLD: float = 0.70     
CASCADE_LEVEL_3_THRESHOLD: float = 0.85     
COMMS_RESTORE_PROBABILITY_PER_STEP: float = 0.25
MUTUAL_AID_OVER_REQUEST_WINDOW_STEPS: int = 8
@unique
class IncidentPhase(str, Enum):
    PENDING          = "pending"           
    DISPATCHED       = "dispatched"        
    UNIT_ON_SCENE    = "unit_on_scene"     
    TREATMENT        = "treatment"         
    TRANSPORT        = "transport"         
    HOSPITAL_HANDOFF = "hospital_handoff"  
    RESOLVED         = "resolved"          
    FAILED           = "failed"            
    CANCELLED        = "cancelled"         
    @property
    def is_terminal(self) -> bool:
        return self in (IncidentPhase.RESOLVED, IncidentPhase.FAILED, IncidentPhase.CANCELLED)
    @property
    def is_active(self) -> bool:
        return not self.is_terminal and self != IncidentPhase.PENDING
@unique
class UnitMissionPhase(str, Enum):
    EN_ROUTE_TO_SCENE   = "en_route_to_scene"
    ON_SCENE            = "on_scene"
    LOADING_PATIENT     = "loading_patient"
    EN_ROUTE_TO_HOSP    = "en_route_to_hospital"
    AT_HOSPITAL         = "at_hospital"
    RETURNING_TO_BASE   = "returning_to_base"
    IDLE                = "idle"
    CREW_SWAP           = "crew_swap"
    PREPOSITION_MOVE    = "preposition_move"
    MUTUAL_AID_DEPLOY   = "mutual_aid_deploy"
    COMMS_BLACKOUT      = "comms_blackout"
@unique
class HospitalDiversionState(str, Enum):
    ACCEPTING              = "accepting"
    NEAR_CAPACITY_WARNING  = "near_capacity_warning"
    ON_DIVERSION           = "on_diversion"
    SURGE_OVERFLOW         = "surge_overflow"
    RECOVERING             = "recovering"
@unique
class ZoneAlertLevel(str, Enum):
    GREEN  = "green"    
    AMBER  = "amber"    
    RED    = "red"      
    BLACK  = "black"    
@unique
class SystemEventType(str, Enum):
    INCIDENT_CREATED          = "incident_created"
    INCIDENT_DISPATCHED       = "incident_dispatched"
    INCIDENT_UNIT_ON_SCENE    = "incident_unit_on_scene"
    INCIDENT_TREATMENT_STARTED= "incident_treatment_started"
    INCIDENT_TRANSPORT_STARTED= "incident_transport_started"
    INCIDENT_HOSPITAL_HANDOFF = "incident_hospital_handoff"
    INCIDENT_RESOLVED         = "incident_resolved"
    INCIDENT_FAILED           = "incident_failed"
    INCIDENT_CANCELLED        = "incident_cancelled"
    INCIDENT_ESCALATED        = "incident_escalated"
    INCIDENT_MCI_DECLARED     = "incident_mci_declared"
    UNIT_DISPATCHED           = "unit_dispatched"
    UNIT_ARRIVED_SCENE        = "unit_arrived_scene"
    UNIT_TRANSPORT_STARTED    = "unit_transport_started"
    UNIT_ARRIVED_HOSPITAL     = "unit_arrived_hospital"
    UNIT_RETURNED_BASE        = "unit_returned_base"
    UNIT_FATIGUED             = "unit_fatigued"
    UNIT_CREW_SWAP_REQUESTED  = "unit_crew_swap_requested"
    UNIT_CREW_SWAP_COMPLETED  = "unit_crew_swap_completed"
    UNIT_COMMS_LOST           = "unit_comms_lost"
    UNIT_COMMS_RESTORED       = "unit_comms_restored"
    UNIT_PREPOSITIONED        = "unit_prepositioned"
    UNIT_MUTUAL_AID_DEPLOYED  = "unit_mutual_aid_deployed"
    HOSPITAL_NEAR_CAPACITY    = "hospital_near_capacity"
    HOSPITAL_DIVERSION_ON     = "hospital_diversion_on"
    HOSPITAL_DIVERSION_OFF    = "hospital_diversion_off"
    HOSPITAL_SURGE_OVERFLOW   = "hospital_surge_overflow"
    HOSPITAL_BED_ADMITTED     = "hospital_bed_admitted"
    HOSPITAL_BED_DISCHARGED   = "hospital_bed_discharged"
    TRIAGE_TAG_APPLIED        = "triage_tag_applied"
    TRIAGE_MISMATCH_PENALISED = "triage_mismatch_penalised"
    TRAFFIC_SLOWDOWN_STARTED  = "traffic_slowdown_started"
    TRAFFIC_SLOWDOWN_CLEARED  = "traffic_slowdown_cleared"
    TRAFFIC_MATRIX_UPDATED    = "traffic_matrix_updated"
    GREEN_CORRIDOR_ACTIVATED  = "green_corridor_activated"
    SURGE_DECLARED            = "surge_declared"
    CASCADE_FAILURE_WARNING   = "cascade_failure_warning"
    CASCADE_FAILURE_OCCURRED  = "cascade_failure_occurred"
    MUTUAL_AID_REQUESTED      = "mutual_aid_requested"
    MUTUAL_AID_FULFILLED      = "mutual_aid_fulfilled"
    MUTUAL_AID_OVER_REQUESTED = "mutual_aid_over_requested"
    PROTOCOL_VIOLATION        = "protocol_violation"
    PROTOCOL_COMPLIANCE_BONUS = "protocol_compliance_bonus"
    REWARD_COMPUTED           = "reward_computed"
    EPISODE_STARTED           = "episode_started"
    EPISODE_ENDED             = "episode_ended"
    CHECKPOINT_SAVED          = "checkpoint_saved"
    CHECKPOINT_RESTORED       = "checkpoint_restored"
    STATE_TRANSITION_INVALID  = "state_transition_invalid"
@unique
class PatientOutcome(str, Enum):
    SURVIVED_FULL_RECOVERY   = "survived_full_recovery"
    SURVIVED_WITH_DEFICIT    = "survived_with_deficit"
    SURVIVED_CRITICAL_CARE   = "survived_critical_care"
    TRANSFERRED_TERTIARY     = "transferred_tertiary"
    DIED_AFTER_ARRIVAL       = "died_after_arrival"
    DIED_PRE_HOSPITAL        = "died_pre_hospital"
    DIED_ON_SCENE            = "died_on_scene"
    REFUSED_TRANSPORT        = "refused_transport"
    TREAT_AND_RELEASE        = "treat_and_release"
    EXPECTANT_PALLIATED      = "expectant_palliated"
    UNKNOWN                  = "unknown"
@unique
class RewardChannel(str, Enum):
    SURVIVAL                  = "survival"
    RESPONSE_TIME             = "response_time"
    TRIAGE_ACCURACY           = "triage_accuracy"
    PROTOCOL_COMPLIANCE       = "protocol_compliance"
    HOSPITAL_ROUTING          = "hospital_routing"
    DIVERSION_AVOIDANCE       = "diversion_avoidance"
    MULTI_AGENCY_COORDINATION = "multi_agency_coordination"
    CREW_FATIGUE_MANAGEMENT   = "crew_fatigue_management"
    PREPOSITION_EFFICIENCY    = "preposition_efficiency"
    TRANSFER_TIMELINESS       = "transfer_timeliness"
    SURGE_MANAGEMENT          = "surge_management"
    MUTUAL_AID_EFFICIENCY     = "mutual_aid_efficiency"
    CASCADE_AVOIDANCE         = "cascade_avoidance"
    COMMS_RESILIENCE          = "comms_resilience"
    NOOP_PENALTY              = "noop_penalty"
class PatientState(EmergiBaseModel):
    patient_id: str                 = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: IncidentID
    victim_index: int               = Field(..., ge=0)
    vitals: PatientVitals
    triage_tag: Optional[TriageTag] = Field(default=None)
    triage_tag_step: Optional[int]  = Field(default=None, ge=0)
    triage_correct: Optional[bool]  = Field(default=None)
    decay_model: DecayModel         = Field(default=DecayModel.EXPONENTIAL)
    decay_params: Dict[str, float]  = Field(default_factory=dict)
    survival_at_incident_start: float = Field(default=0.90, ge=0.0, le=1.0)
    current_survival_probability: float = Field(default=0.90, ge=0.0, le=1.0)
    survival_last_recalc_step: int  = Field(default=0, ge=0)
    survival_deltas: List[float]    = Field(default_factory=list)   
    first_unit_contact_step: Optional[int]   = Field(default=None, ge=0)
    treatment_started_step: Optional[int]    = Field(default=None, ge=0)
    hospital_arrival_step: Optional[int]     = Field(default=None, ge=0)
    final_hospital_id: Optional[HospitalID] = Field(default=None)
    outcome: PatientOutcome                  = Field(default=PatientOutcome.UNKNOWN)
    outcome_step: Optional[int]              = Field(default=None, ge=0)
    response_time_minutes: Optional[float]   = Field(default=None, ge=0.0)
    time_to_definitive_care_minutes: Optional[float] = Field(default=None, ge=0.0)
    golden_hour_met: Optional[bool]          = Field(default=None)
    dnr_present: bool       = Field(default=False)
    paediatric: bool        = Field(default=False)
    obstetric: bool         = Field(default=False)
    requires_micu: bool     = Field(default=False)
    @computed_field
    @property
    def is_alive(self) -> bool:
        return self.outcome not in (
            PatientOutcome.DIED_AFTER_ARRIVAL,
            PatientOutcome.DIED_PRE_HOSPITAL,
            PatientOutcome.DIED_ON_SCENE,
        )
    @computed_field
    @property
    def outcome_determined(self) -> bool:
        return self.outcome != PatientOutcome.UNKNOWN
    @computed_field
    @property
    def total_survival_delta(self) -> float:
        return sum(self.survival_deltas)
    @computed_field
    @property
    def triage_was_applied(self) -> bool:
        return self.triage_tag is not None
    def compute_survival(self, current_step: int) -> float:
        if not self.decay_params:
            return self.current_survival_probability
        elapsed_minutes = (current_step - (self.survival_last_recalc_step or 0)) * TIMESTEP_MINUTES
        if elapsed_minutes <= 0.0:
            return self.current_survival_probability
        t_total = current_step * TIMESTEP_MINUTES
        new_p = SurvivalCurve.compute(self.decay_model, t_total, self.decay_params)
        new_p = clamp_score(new_p)
        delta = new_p - self.current_survival_probability
        self.survival_deltas.append(delta)
        self.current_survival_probability = new_p
        self.survival_last_recalc_step = current_step
        return new_p
    def apply_triage(
        self,
        tag: TriageTag,
        step: int,
        correct: bool,
    ) -> float:
        self.triage_tag = tag
        self.triage_tag_step = step
        self.triage_correct = correct
        if not correct and tag == TriageTag.EXPECTANT:
            return WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY
        return 0.0
    def finalise_outcome(
        self,
        outcome: PatientOutcome,
        step: int,
    ) -> None:
        self.outcome = outcome
        self.outcome_step = step
class IncidentDispatchRecord(EmergiBaseModel):
    dispatch_id: str          = Field(default_factory=lambda: str(uuid.uuid4()))
    unit_id: UnitID
    unit_type: UnitType
    dispatched_step: StepNumber
    scene_arrival_step: Optional[int] = Field(default=None)
    scene_departure_step: Optional[int] = Field(default=None)
    hospital_arrival_step: Optional[int] = Field(default=None)
    returned_step: Optional[int] = Field(default=None)
    destination_hospital_id: Optional[HospitalID] = Field(default=None)
    routing_strategy: str = Field(default="standard")
    @computed_field
    @property
    def response_time_minutes(self) -> Optional[float]:
        if self.scene_arrival_step is None:
            return None
        return (self.scene_arrival_step - self.dispatched_step) * float(TIMESTEP_MINUTES)
    @computed_field
    @property
    def on_scene_duration_minutes(self) -> Optional[float]:
        if self.scene_arrival_step is None or self.scene_departure_step is None:
            return None
        return (self.scene_departure_step - self.scene_arrival_step) * float(TIMESTEP_MINUTES)
    @computed_field
    @property
    def transport_duration_minutes(self) -> Optional[float]:
        if self.scene_departure_step is None or self.hospital_arrival_step is None:
            return None
        return (self.hospital_arrival_step - self.scene_departure_step) * float(TIMESTEP_MINUTES)
class IncidentState(EmergiBaseModel):
    incident_id: IncidentID
    template_id: TemplateID
    call_number: str
    condition: str
    incident_type: IncidentType
    severity: SeverityLevel
    symptom_description: str
    protocol_notes: str
    zone_id: ZoneID
    location: GeoCoordinate
    location_description: str = Field(default="")
    phase: IncidentPhase              = Field(default=IncidentPhase.PENDING)
    phase_history: List[Tuple[IncidentPhase, int]] = Field(default_factory=list)
    step_created: StepNumber
    step_dispatched: Optional[int]    = Field(default=None)
    step_first_unit_on_scene: Optional[int] = Field(default=None)
    step_resolved: Optional[int]      = Field(default=None)
    golden_hour_minutes: int          = Field(..., ge=1)
    minutes_elapsed: float            = Field(default=0.0, ge=0.0)
    target_response_minutes: int      = Field(..., ge=1)
    actual_response_minutes: Optional[float] = Field(default=None)
    patients: List[PatientState]      = Field(default_factory=list)
    victim_count: int                 = Field(default=1, ge=1)
    mci_declared: bool                = Field(default=False)
    mci_declaration_step: Optional[int] = Field(default=None)
    triage_completed: bool            = Field(default=False)
    triage_step: Optional[int]        = Field(default=None)
    start_protocol_applied: bool      = Field(default=False)
    multi_agency_required: bool       = Field(default=False)
    agencies_needed: List[str]        = Field(default_factory=list)
    agencies_on_scene: List[str]      = Field(default_factory=list)
    police_scene_cleared: bool        = Field(default=False)
    fire_extrication_complete: bool   = Field(default=False)
    hazmat_present: bool              = Field(default=False)
    hazmat_class: Optional[str]       = Field(default=None)
    trapped_victim: bool              = Field(default=False)
    requires_micu: bool       = Field(default=False)
    requires_als_minimum: bool= Field(default=False)
    paediatric: bool          = Field(default=False)
    obstetric: bool           = Field(default=False)
    psychiatric_risk: bool    = Field(default=False)
    dnr_present: bool         = Field(default=False)
    dispatches: List[IncidentDispatchRecord] = Field(default_factory=list)
    units_assigned: List[UnitID]      = Field(default_factory=list)
    units_on_scene: List[UnitID]      = Field(default_factory=list)
    nearest_available_eta_minutes: float = Field(default=math.inf, ge=0.0)
    outcome_score_contribution: float = Field(default=0.0)
    golden_hour_outcome: Optional[bool] = Field(default=None)  
    protocol_violations_this_incident: int = Field(default=0, ge=0)
    peak_survival_probability: float  = Field(default=0.90, ge=0.0, le=1.0)
    final_survival_probability: float = Field(default=0.90, ge=0.0, le=1.0)
    @computed_field
    @property
    def is_terminal(self) -> bool:
        return self.phase.is_terminal
    @computed_field
    @property
    def is_dispatched(self) -> bool:
        return self.phase not in (IncidentPhase.PENDING,) and not self.is_terminal
    @computed_field
    @property
    def golden_hour_exceeded(self) -> bool:
        return self.minutes_elapsed > self.golden_hour_minutes
    @computed_field
    @property
    def golden_hour_fraction(self) -> float:
        return min(1.0, self.minutes_elapsed / max(1.0, self.golden_hour_minutes))
    @computed_field
    @property
    def victims_tagged(self) -> int:
        return sum(1 for p in self.patients if p.triage_was_applied)
    @computed_field
    @property
    def victims_immediate(self) -> int:
        return sum(1 for p in self.patients if p.triage_tag == TriageTag.IMMEDIATE)
    @computed_field
    @property
    def victims_delayed(self) -> int:
        return sum(1 for p in self.patients if p.triage_tag == TriageTag.DELAYED)
    @computed_field
    @property
    def victims_minimal(self) -> int:
        return sum(1 for p in self.patients if p.triage_tag == TriageTag.MINIMAL)
    @computed_field
    @property
    def victims_expectant(self) -> int:
        return sum(1 for p in self.patients if p.triage_tag == TriageTag.EXPECTANT)
    @computed_field
    @property
    def all_patients_finalised(self) -> bool:
        return all(p.outcome_determined for p in self.patients)
    @computed_field
    @property
    def ems_blocked(self) -> bool:
        if self.multi_agency_required:
            if "Police" in self.agencies_needed and not self.police_scene_cleared:
                return True
            if self.trapped_victim and not self.fire_extrication_complete:
                return True
        return False
    @computed_field
    @property
    def average_patient_survival(self) -> float:
        if not self.patients:
            return 0.0
        return sum(p.current_survival_probability for p in self.patients) / len(self.patients)
    def transition_phase(self, new_phase: IncidentPhase, step: int) -> None:
        legal_transitions: Dict[IncidentPhase, Set[IncidentPhase]] = {
            IncidentPhase.PENDING:          {IncidentPhase.DISPATCHED, IncidentPhase.CANCELLED},
            IncidentPhase.DISPATCHED:       {IncidentPhase.UNIT_ON_SCENE, IncidentPhase.CANCELLED},
            IncidentPhase.UNIT_ON_SCENE:    {IncidentPhase.TREATMENT, IncidentPhase.FAILED},
            IncidentPhase.TREATMENT:        {IncidentPhase.TRANSPORT, IncidentPhase.RESOLVED, IncidentPhase.FAILED},
            IncidentPhase.TRANSPORT:        {IncidentPhase.HOSPITAL_HANDOFF, IncidentPhase.FAILED},
            IncidentPhase.HOSPITAL_HANDOFF: {IncidentPhase.RESOLVED, IncidentPhase.FAILED},
            IncidentPhase.RESOLVED:         set(),
            IncidentPhase.FAILED:           set(),
            IncidentPhase.CANCELLED:        set(),
        }
        if new_phase not in legal_transitions.get(self.phase, set()):
            raise ValueError(
                f"Illegal incident phase transition: {self.phase} → {new_phase} "
                f"for incident {self.incident_id}"
            )
        self.phase_history.append((self.phase, step))
        self.phase = new_phase
        if new_phase in (IncidentPhase.RESOLVED, IncidentPhase.FAILED, IncidentPhase.CANCELLED):
            self.step_resolved = step
    def advance_time(self, steps: int = 1) -> None:
        self.minutes_elapsed += steps * TIMESTEP_MINUTES
        current_step_approx = int(self.minutes_elapsed / TIMESTEP_MINUTES)
        for p in self.patients:
            p.compute_survival(current_step_approx)
    def add_dispatch(
        self,
        unit_id: UnitID,
        unit_type: UnitType,
        step: int,
        hospital_id: Optional[HospitalID] = None,
        routing_strategy: str = "standard",
    ) -> IncidentDispatchRecord:
        record = IncidentDispatchRecord(
            unit_id=unit_id,
            unit_type=unit_type,
            dispatched_step=step,
            destination_hospital_id=hospital_id,
            routing_strategy=routing_strategy,
        )
        self.dispatches.append(record)
        if unit_id not in self.units_assigned:
            self.units_assigned.append(unit_id)
        return record
class UnitMissionState(EmergiBaseModel):
    mission_id: str                           = Field(default_factory=lambda: str(uuid.uuid4()))
    mission_phase: UnitMissionPhase           = Field(default=UnitMissionPhase.IDLE)
    incident_id: Optional[IncidentID]         = Field(default=None)
    destination_hospital_id: Optional[HospitalID] = Field(default=None)
    destination_zone_id: Optional[ZoneID]     = Field(default=None)
    steps_remaining_to_scene: Optional[int]   = Field(default=None, ge=0)
    steps_remaining_to_hospital: Optional[int]= Field(default=None, ge=0)
    steps_remaining_to_base: Optional[int]    = Field(default=None, ge=0)
    total_mission_steps: int                  = Field(default=0, ge=0)
    patient_ids_on_board: List[str]           = Field(default_factory=list)
    patient_severity: Optional[str]           = Field(default=None)
    patient_condition_key: Optional[str]      = Field(default=None)
    routing_strategy: str                     = Field(default="standard")
    via_green_corridor: bool                  = Field(default=False)
    bypass_bottleneck_ids: List[str]          = Field(default_factory=list)
    can_be_rerouted: bool                     = Field(default=True)
    reroute_count: int                        = Field(default=0, ge=0)
    last_reroute_step: Optional[int]          = Field(default=None)
    cumulative_reroute_delay_minutes: float   = Field(default=0.0, ge=0.0)
    @computed_field
    @property
    def has_patient(self) -> bool:
        return len(self.patient_ids_on_board) > 0
    @computed_field
    @property
    def is_transportable(self) -> bool:
        return self.can_be_rerouted and not self.has_patient
    @computed_field
    @property
    def eta_minutes(self) -> float:
        steps = (
            self.steps_remaining_to_hospital
            or self.steps_remaining_to_scene
            or self.steps_remaining_to_base
            or 0
        )
        return steps * float(TIMESTEP_MINUTES)
    @computed_field
    @property
    def current_destination_label(self) -> str:
        if self.mission_phase == UnitMissionPhase.EN_ROUTE_TO_SCENE:
            return f"Scene ({self.incident_id})"
        if self.mission_phase in (UnitMissionPhase.EN_ROUTE_TO_HOSP, UnitMissionPhase.AT_HOSPITAL):
            return f"Hospital ({self.destination_hospital_id})"
        if self.mission_phase == UnitMissionPhase.RETURNING_TO_BASE:
            return "Base"
        return "Idle"
class UnitState(EmergiBaseModel):
    unit_id: UnitID
    unit_type: UnitType
    registration: str
    current_zone: ZoneID
    home_zone: ZoneID
    current_lat: float              = Field(..., ge=-90.0, le=90.0)
    current_lon: float              = Field(..., ge=-180.0, le=180.0)
    staging_post_id: Optional[str] = Field(default=None)
    status: UnitStatus              = Field(default=UnitStatus.AVAILABLE)
    mission: Optional[UnitMissionState] = Field(default=None)
    crew_fatigue: CrewFatigueState
    comms: CommunicationsState
    radio_channel: str              = Field(default="CH1")
    missions_completed: int         = Field(default=0, ge=0)
    missions_failed: int            = Field(default=0, ge=0)
    total_distance_km: float        = Field(default=0.0, ge=0.0)
    total_patients_transported: int = Field(default=0, ge=0)
    cumulative_response_time_min: float = Field(default=0.0, ge=0.0)
    response_time_count: int        = Field(default=0, ge=0)
    is_mutual_aid_unit: bool        = Field(default=False)
    mutual_aid_source_zone: Optional[ZoneID] = Field(default=None)
    mutual_aid_eta_steps_remaining: Optional[int] = Field(default=None, ge=0)
    crew_swap_requested: bool       = Field(default=False)
    crew_swap_eta_steps: Optional[int] = Field(default=None, ge=0)
    crew_swap_count_this_episode: int = Field(default=0, ge=0)
    status_history: List[Tuple[str, str, int]] = Field(default_factory=list)
    @computed_field
    @property
    def is_available(self) -> bool:
        return (
            self.status == UnitStatus.AVAILABLE
            and not self.crew_swap_requested
            and not (self.mutual_aid_eta_steps_remaining and self.mutual_aid_eta_steps_remaining > 0)
        )
    @computed_field
    @property
    def is_deployable(self) -> bool:
        return self.is_available
    @computed_field
    @property
    def is_fatigued(self) -> bool:
        return self.crew_fatigue.fatigued
    @computed_field
    @property
    def comms_active(self) -> bool:
        return self.comms.comms_active
    @computed_field
    @property
    def avg_response_time_minutes(self) -> Optional[float]:
        if self.response_time_count == 0:
            return None
        return self.cumulative_response_time_min / self.response_time_count
    @computed_field
    @property
    def effective_speed_multiplier(self) -> float:
        return max(0.70, 1.0 - self.crew_fatigue.fatigue_penalty)
    @computed_field
    @property
    def utilisation_rate_this_episode(self) -> float:
        total = self.missions_completed + self.missions_failed
        if total == 0:
            return 0.0
        return self.missions_completed / total
    @computed_field
    @property
    def operational_summary(self) -> str:
        fatigue = f"FAT({self.crew_fatigue.hours_on_duty:.1f}h)" if self.is_fatigued else "fresh"
        comms = "COMMS_OK" if self.comms_active else "COMMS_LOST"
        return (
            f"[{self.unit_id}] {self.unit_type} | {self.status} | "
            f"Zone {self.current_zone} | {fatigue} | {comms} | "
            f"missions={self.missions_completed}"
        )
    def transition_status(
        self,
        new_status: UnitStatus,
        step: int,
    ) -> None:
        legal: Dict[UnitStatus, Set[UnitStatus]] = {
            UnitStatus.AVAILABLE: {
                UnitStatus.DISPATCHED,
                UnitStatus.PREPOSITION_MOVE if hasattr(UnitStatus, "PREPOSITION_MOVE") else UnitStatus.DISPATCHED,
                UnitStatus.CREW_SWAP_PENDING,
                UnitStatus.MUTUAL_AID_DEPLOY,
                UnitStatus.OUT_OF_SERVICE,
                UnitStatus.COMMS_LOST,
            },
            UnitStatus.DISPATCHED: {
                UnitStatus.ON_SCENE,
                UnitStatus.AVAILABLE,     
                UnitStatus.COMMS_LOST,
            },
            UnitStatus.ON_SCENE: {
                UnitStatus.TRANSPORTING,
                UnitStatus.RETURNING,     
                UnitStatus.COMMS_LOST,
            },
            UnitStatus.TRANSPORTING: {
                UnitStatus.AT_HOSPITAL,
                UnitStatus.COMMS_LOST,
            },
            UnitStatus.AT_HOSPITAL: {
                UnitStatus.RETURNING,
            },
            UnitStatus.RETURNING: {
                UnitStatus.AVAILABLE,
                UnitStatus.COMMS_LOST,
            },
            UnitStatus.MUTUAL_AID_DEPLOY: {
                UnitStatus.AVAILABLE,
                UnitStatus.DISPATCHED,
            },
            UnitStatus.CREW_SWAP_PENDING: {
                UnitStatus.AVAILABLE,
            },
            UnitStatus.COMMS_LOST: {
                UnitStatus.AVAILABLE,
                UnitStatus.DISPATCHED,
                UnitStatus.ON_SCENE,
                UnitStatus.TRANSPORTING,
                UnitStatus.AT_HOSPITAL,
                UnitStatus.RETURNING,
            },
            UnitStatus.OUT_OF_SERVICE: {
                UnitStatus.AVAILABLE,
            },
        }
        allowed = legal.get(self.status, set())
        if new_status not in allowed:
            logger.warning(
                "Unit %s: illegal status transition %s → %s at step %d",
                self.unit_id, self.status, new_status, step,
            )
        self.status_history.append((self.status.value, new_status.value, step))
        self.status = new_status
    def complete_mission(self, response_time_min: Optional[float] = None) -> None:
        self.missions_completed += 1
        self.crew_fatigue.complete_mission()
        if response_time_min is not None:
            self.cumulative_response_time_min += response_time_min
            self.response_time_count += 1
    def advance_fatigue(self, steps: int = 1) -> None:
        self.crew_fatigue.advance_time(steps)
    def tick_comms(self, rng_value: float) -> bool:
        changed = False
        if self.comms.comms_active:
            if rng_value < COMMS_FAILURE_PROBABILITY_PER_STEP:
                self.comms.comms_active = False
                self.comms.failure_count_this_episode += 1
                changed = True
        else:
            if rng_value < COMMS_RESTORE_PROBABILITY_PER_STEP:
                self.comms.comms_active = True
                changed = True
        return changed
class HospitalBedTracking(EmergiBaseModel):
    er_beds_total: int          = Field(..., ge=0)
    er_beds_occupied: int       = Field(default=0, ge=0)
    er_beds_reserved_surge: int = Field(default=0, ge=0)
    icu_beds_total: int         = Field(..., ge=0)
    icu_beds_occupied: int      = Field(default=0, ge=0)
    ccu_beds_total: int         = Field(..., ge=0)
    ccu_beds_occupied: int      = Field(default=0, ge=0)
    trauma_bays_total: int      = Field(..., ge=0)
    trauma_bays_occupied: int   = Field(default=0, ge=0)
    ventilators_total: int      = Field(..., ge=0)
    ventilators_occupied: int   = Field(default=0, ge=0)
    total_admissions_episode: int   = Field(default=0, ge=0)
    total_discharges_episode: int   = Field(default=0, ge=0)
    total_diversions_episode: int   = Field(default=0, ge=0)
    @computed_field
    @property
    def er_occupancy_pct(self) -> float:
        if self.er_beds_total == 0:
            return 0.0
        return self.er_beds_occupied / self.er_beds_total
    @computed_field
    @property
    def icu_occupancy_pct(self) -> float:
        if self.icu_beds_total == 0:
            return 0.0
        return self.icu_beds_occupied / self.icu_beds_total
    @computed_field
    @property
    def total_occupancy_pct(self) -> float:
        total = self.er_beds_total + self.icu_beds_total + self.ccu_beds_total
        if total == 0:
            return 0.0
        occupied = self.er_beds_occupied + self.icu_beds_occupied + self.ccu_beds_occupied
        return occupied / total
    @computed_field
    @property
    def er_available(self) -> int:
        return max(0, self.er_beds_total - self.er_beds_occupied - self.er_beds_reserved_surge)
    @computed_field
    @property
    def trauma_bays_available(self) -> int:
        return max(0, self.trauma_bays_total - self.trauma_bays_occupied)
    @computed_field
    @property
    def ventilators_available(self) -> int:
        return max(0, self.ventilators_total - self.ventilators_occupied)
    def admit_patient(self, requires_icu: bool = False, requires_trauma_bay: bool = False) -> bool:
        if self.er_available <= 0:
            return False
        self.er_beds_occupied += 1
        self.total_admissions_episode += 1
        if requires_icu and self.icu_beds_occupied < self.icu_beds_total:
            self.icu_beds_occupied += 1
        if requires_trauma_bay and self.trauma_bays_occupied < self.trauma_bays_total:
            self.trauma_bays_occupied += 1
        return True
    def discharge_patient(self, from_icu: bool = False) -> None:
        self.er_beds_occupied = max(0, self.er_beds_occupied - 1)
        self.total_discharges_episode += 1
        if from_icu:
            self.icu_beds_occupied = max(0, self.icu_beds_occupied - 1)
    def to_capacity_snapshot(self) -> CapacitySnapshot:
        return CapacitySnapshot(
            total_beds=self.er_beds_total + self.icu_beds_total + self.ccu_beds_total,
            er_beds=self.er_beds_total,
            icu_beds=self.icu_beds_total,
            ccu_beds=self.ccu_beds_total,
            trauma_bays=self.trauma_bays_total,
            ventilators=self.ventilators_total,
            er_occupancy_pct=round(self.er_occupancy_pct, 4),
            icu_occupancy_pct=round(self.icu_occupancy_pct, 4),
            total_occupancy_pct=round(self.total_occupancy_pct, 4),
            on_diversion=self.er_occupancy_pct >= DIVERSION_THRESHOLD_PCT,
        )
class HospitalState(EmergiBaseModel):
    hospital_id: HospitalID
    name: str
    short_name: str
    hospital_type: HospitalType
    tier: HospitalTier
    zone_id: ZoneID
    district: str
    location: GeoCoordinate
    specialties: Set[str]       = Field(default_factory=set)
    infrastructure: Set[str]    = Field(default_factory=set)
    beds: HospitalBedTracking
    door_to_doctor_min: float   = Field(default=8.0, ge=0.0)
    door_to_balloon_min: float  = Field(default=90.0, ge=0.0)
    door_to_needle_min: float   = Field(default=60.0, ge=0.0)
    door_to_ct_min: float       = Field(default=30.0, ge=0.0)
    trauma_activation_time_min: float = Field(default=5.0, ge=0.0)
    emergency_physicians_on_duty: int = Field(default=2, ge=0)
    trauma_surgeons_on_call: int      = Field(default=1, ge=0)
    diversion_state: HospitalDiversionState = Field(default=HospitalDiversionState.ACCEPTING)
    diversion_activated_step: Optional[int]  = Field(default=None)
    diversion_reset_step: Optional[int]      = Field(default=None)
    diversion_reason: Optional[str]          = Field(default=None)
    consecutive_diversion_steps: int         = Field(default=0, ge=0)
    diversion_history: List[Tuple[int, int, str]] = Field(default_factory=list)  
    routing_priorities: Dict[str, int]   = Field(default_factory=dict)
    base_score_weight: float             = Field(default=1.0, ge=0.0, le=1.0)
    surge_capacity_additional_beds: int  = Field(default=0, ge=0)
    surge_beds_activated: int            = Field(default=0, ge=0)
    surge_activated_step: Optional[int]  = Field(default=None)
    active_incoming_transports: List[UnitID] = Field(default_factory=list)
    @computed_field
    @property
    def is_on_diversion(self) -> bool:
        return self.diversion_state in (
            HospitalDiversionState.ON_DIVERSION,
            HospitalDiversionState.SURGE_OVERFLOW,
        )
    @computed_field
    @property
    def accepts_patients(self) -> bool:
        return (
            not self.is_on_diversion
            and self.beds.er_occupancy_pct < DIVERSION_THRESHOLD_PCT
        )
    @computed_field
    @property
    def er_occupancy_pct(self) -> float:
        return self.beds.er_occupancy_pct
    @computed_field
    @property
    def effective_er_occupancy(self) -> float:
        transport_load = len(self.active_incoming_transports) * 0.02
        return min(1.0, self.er_occupancy_pct + transport_load)
    @computed_field
    @property
    def capacity_pressure(self) -> float:
        return clamp_score(self.effective_er_occupancy / DIVERSION_THRESHOLD_PCT)
    @computed_field
    @property
    def surge_capacity_remaining(self) -> int:
        if self.beds.total_occupancy_pct >= 0.95:
            return 0
        return max(0, self.surge_capacity_additional_beds - self.surge_beds_activated)
    def activate_diversion(self, step: int, reason: str) -> None:
        if self.is_on_diversion:
            return
        self.diversion_state = HospitalDiversionState.ON_DIVERSION
        self.diversion_activated_step = step
        self.diversion_reset_step = step + DIVERSION_FLAG_RESET_STEPS
        self.diversion_reason = reason
        self.consecutive_diversion_steps = 0
        self.beds.total_diversions_episode += 1
    def deactivate_diversion(self, step: int) -> None:
        if not self.is_on_diversion:
            return
        self.diversion_history.append((
            self.diversion_activated_step or step,
            step,
            self.diversion_reason or "unknown",
        ))
        self.diversion_state = HospitalDiversionState.RECOVERING
        self.diversion_activated_step = None
        self.diversion_reset_step = None
        self.diversion_reason = None
    def tick_diversion(self, step: int) -> None:
        if self.diversion_state == HospitalDiversionState.RECOVERING:
            if self.beds.er_occupancy_pct < 0.75:
                self.diversion_state = HospitalDiversionState.ACCEPTING
            return
        if self.diversion_state == HospitalDiversionState.ACCEPTING:
            if self.beds.er_occupancy_pct >= DIVERSION_THRESHOLD_PCT:
                self.diversion_state = HospitalDiversionState.NEAR_CAPACITY_WARNING
            return
        if self.diversion_state == HospitalDiversionState.NEAR_CAPACITY_WARNING:
            if self.beds.er_occupancy_pct >= 0.95:
                self.activate_diversion(step, "auto_capacity")
            elif self.beds.er_occupancy_pct < DIVERSION_THRESHOLD_PCT:
                self.diversion_state = HospitalDiversionState.ACCEPTING
            return
        if self.is_on_diversion:
            self.consecutive_diversion_steps += 1
            if self.diversion_reset_step and step >= self.diversion_reset_step:
                if self.beds.er_occupancy_pct < DIVERSION_THRESHOLD_PCT:
                    self.deactivate_diversion(step)
    def add_incoming_transport(self, unit_id: UnitID) -> None:
        if unit_id not in self.active_incoming_transports:
            self.active_incoming_transports.append(unit_id)
    def complete_incoming_transport(self, unit_id: UnitID) -> bool:
        if unit_id in self.active_incoming_transports:
            self.active_incoming_transports.remove(unit_id)
            return True
        return False
    def has_specialty(self, specialty: str) -> bool:
        return specialty in self.specialties
    def routing_score(self, required_specialty: Optional[str] = None) -> float:
        if not self.accepts_patients:
            return 0.0
        if required_specialty and not self.has_specialty(required_specialty):
            return 0.0
        spec_count = len(self.specialties)
        spec_score = clamp_score(spec_count / len(HospitalSpecialty.ALL))
        cap_score  = 1.0 - self.capacity_pressure
        return clamp_score(
            self.base_score_weight * (0.6 * spec_score + 0.4 * cap_score)
        )
class ZoneStagingPost(EmergiBaseModel):
    staging_post_id: str
    zone_id: ZoneID
    name: str
    location: GeoCoordinate
    unit_ids_stationed: List[UnitID] = Field(default_factory=list)
    capacity: int = Field(default=4, ge=1)
    @computed_field
    @property
    def utilisation(self) -> float:
        return len(self.unit_ids_stationed) / self.capacity
    @computed_field
    @property
    def has_capacity(self) -> bool:
        return len(self.unit_ids_stationed) < self.capacity
class ZoneState(EmergiBaseModel):
    zone_id: ZoneID
    zone_type: ZoneType
    name: str
    district: str
    location: GeoCoordinate
    alert_level: ZoneAlertLevel = Field(default=ZoneAlertLevel.GREEN)
    alert_level_step: int       = Field(default=0, ge=0)
    base_call_volume_per_day: float     = Field(default=5.0, ge=0.0)
    active_incident_ids: List[IncidentID] = Field(default_factory=list)
    resolved_incident_ids: List[IncidentID] = Field(default_factory=list)
    failed_incident_ids: List[IncidentID]   = Field(default_factory=list)
    mci_active: bool                    = Field(default=False)
    simultaneous_mci_count: int         = Field(default=0, ge=0)
    stationed_unit_ids: List[UnitID]    = Field(default_factory=list)
    staging_posts: List[ZoneStagingPost]= Field(default_factory=list)
    mutual_aid_units_available: int     = Field(default=0, ge=0)
    cardiac_arrest_risk: float  = Field(default=0.0, ge=0.0)
    trauma_risk: float          = Field(default=0.0, ge=0.0)
    mci_risk: float             = Field(default=0.0, ge=0.0)
    flood_risk: str             = Field(default="low")
    industrial_hazard_risk: str = Field(default="low")
    local_traffic_multiplier: float = Field(default=1.0, ge=0.3, le=5.0)
    route_closures: List[Tuple[ZoneID, ZoneID]] = Field(default_factory=list)
    avg_response_time_this_episode: float = Field(default=0.0, ge=0.0)
    response_time_samples: int            = Field(default=0, ge=0)
    target_response_minutes: int          = Field(default=10, ge=1)
    @computed_field
    @property
    def active_incident_count(self) -> int:
        return len(self.active_incident_ids)
    @computed_field
    @property
    def coverage_unit_count(self) -> int:
        return len(self.stationed_unit_ids)
    @computed_field
    @property
    def is_covered(self) -> bool:
        return self.coverage_unit_count > 0
    @computed_field
    @property
    def demand_index(self) -> float:
        base = self.base_call_volume_per_day / 24.0  
        active_factor = 1.0 + self.active_incident_count * 0.25
        risk_factor = 1.0 + self.cardiac_arrest_risk + self.mci_risk * 2.0
        return clamp_score(base * active_factor * risk_factor / 5.0)
    @computed_field
    @property
    def response_gap(self) -> float:
        return max(0.0, self.avg_response_time_this_episode - self.target_response_minutes)
    @computed_field
    @property
    def coverage_priority_score(self) -> float:
        demand = self.demand_index
        gap = clamp_score(self.response_gap / 30.0)
        risk = clamp_score(self.cardiac_arrest_risk + self.mci_risk)
        covered = 0.0 if self.is_covered else 0.3
        return clamp_score(0.35 * demand + 0.30 * gap + 0.20 * risk + 0.15 * covered)
    def update_alert_level(self, step: int) -> None:
        if self.mci_active or self.active_incident_count >= 5:
            new_level = ZoneAlertLevel.RED
        elif self.active_incident_count >= 2 or self.mci_risk > 0.6:
            new_level = ZoneAlertLevel.AMBER
        else:
            new_level = ZoneAlertLevel.GREEN
        if new_level != self.alert_level:
            self.alert_level = new_level
            self.alert_level_step = step
    def record_response_time(self, minutes: float) -> None:
        n = self.response_time_samples
        self.avg_response_time_this_episode = (
            (self.avg_response_time_this_episode * n + minutes) / (n + 1)
        )
        self.response_time_samples += 1
class TrafficSlowdownState(EmergiBaseModel):
    slowdown_id: str
    affected_zones: List[ZoneID]
    multiplier: float           = Field(..., ge=1.0, le=5.0)
    steps_remaining: int        = Field(..., ge=0)
    cause: str
    requires_police_clearance: bool = Field(default=False)
    requires_fire_clearance: bool   = Field(default=False)
    ems_exempt: bool                = Field(default=False)
    created_step: int               = Field(..., ge=0)
    secondary: bool                 = Field(default=False)  
    @computed_field
    @property
    def is_active(self) -> bool:
        return self.steps_remaining > 0
    @computed_field
    @property
    def severity_label(self) -> str:
        if self.multiplier >= 2.5:
            return "SEVERE"
        if self.multiplier >= 1.7:
            return "MODERATE"
        return "MILD"
    def tick(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1
class TrafficEngineState(EmergiBaseModel):
    od_matrix_base: Dict[ZoneID, Dict[ZoneID, float]] = Field(default_factory=dict)
    od_matrix_current: Dict[ZoneID, Dict[ZoneID, float]] = Field(default_factory=dict)
    last_update_step: int       = Field(default=0, ge=0)
    next_update_step: int       = Field(default=TRAFFIC_UPDATE_EVERY_N_STEPS, ge=0)
    current_global_multiplier: float = Field(default=1.0, ge=0.3, le=5.0)
    active_slowdowns: List[TrafficSlowdownState] = Field(default_factory=list)
    slowdown_history: List[Dict[str, Any]]       = Field(default_factory=list)
    closed_routes: List[Tuple[ZoneID, ZoneID]]  = Field(default_factory=list)
    ghat_closures: List[str]                     = Field(default_factory=list)
    green_corridor_active: bool                  = Field(default=False)
    green_corridor_route: Optional[List[ZoneID]] = Field(default=None)
    green_corridor_expires_step: Optional[int]   = Field(default=None)
    active_ems_corridors: List[str]              = Field(default_factory=list)
    special_event_active: Optional[str]          = Field(default=None)
    event_traffic_multiplier: float              = Field(default=1.0, ge=0.5, le=4.0)
    secondary_slowdown_counter: int              = Field(default=0, ge=0)
    @computed_field
    @property
    def num_active_slowdowns(self) -> int:
        return len([s for s in self.active_slowdowns if s.is_active])
    @computed_field
    @property
    def worst_slowdown_multiplier(self) -> float:
        active = [s.multiplier for s in self.active_slowdowns if s.is_active and not s.ems_exempt]
        return max(active, default=1.0)
    @computed_field
    @property
    def traffic_congestion_index(self) -> float:
        mult_f = clamp_score((self.current_global_multiplier - 0.4) / 1.6)
        sd_f   = clamp_score(self.num_active_slowdowns / 10.0)
        cl_f   = clamp_score(len(self.closed_routes) / 5.0)
        return clamp_score(0.5 * mult_f + 0.3 * sd_f + 0.2 * cl_f)
    def get_travel_time(
        self,
        origin: ZoneID,
        destination: ZoneID,
        unit_type_str: Optional[str] = None,
    ) -> float:
        if (origin, destination) in self.closed_routes:
            return math.inf
        row = self.od_matrix_current.get(origin)
        if row is None:
            return math.inf
        base = row.get(destination, math.inf)
        if not math.isfinite(base):
            return math.inf
        speed_mod = 1.0
        if unit_type_str:
            speed_mod = {
                UnitType.BLS.value: 1.08,
                UnitType.ALS.value: 1.04,
                UnitType.MICU.value: 1.00,
            }.get(unit_type_str, 1.0)
            base *= speed_mod
        sd_factor = 1.0
        for sd in self.active_slowdowns:
            if not sd.is_active or sd.ems_exempt:
                continue
            if origin in sd.affected_zones or destination in sd.affected_zones:
                sd_factor = max(sd_factor, sd.multiplier)
        if self.green_corridor_active and self.green_corridor_route:
            if origin in self.green_corridor_route and destination in self.green_corridor_route:
                return base * 0.55  
        return base * sd_factor
    def add_slowdown(
        self,
        slowdown_id: str,
        zones: List[ZoneID],
        multiplier: float,
        duration_steps: int,
        cause: str,
        step: int,
        ems_exempt: bool = False,
    ) -> TrafficSlowdownState:
        sd = TrafficSlowdownState(
            slowdown_id=slowdown_id,
            affected_zones=zones,
            multiplier=multiplier,
            steps_remaining=duration_steps,
            cause=cause,
            created_step=step,
            ems_exempt=ems_exempt,
        )
        self.active_slowdowns.append(sd)
        return sd
    def tick(self, step: int, hour: int, rng: float) -> None:
        for sd in self.active_slowdowns:
            sd.tick()
        self.active_slowdowns = [s for s in self.active_slowdowns if s.is_active]
        hour_table = {
            0: 0.55, 1: 0.48, 2: 0.44, 3: 0.42, 4: 0.45, 5: 0.58,
            6: 0.82, 7: 1.38, 8: 1.72, 9: 1.85, 10: 1.55, 11: 1.28,
            12: 1.32, 13: 1.25, 14: 1.18, 15: 1.22, 16: 1.48, 17: 1.88,
            18: 1.95, 19: 1.78, 20: 1.42, 21: 1.15, 22: 0.88, 23: 0.68,
        }
        self.current_global_multiplier = hour_table.get(hour, 1.0)
        if self.special_event_active:
            self.current_global_multiplier = min(
                5.0, self.current_global_multiplier * self.event_traffic_multiplier
            )
        if rng < 0.05 and self.num_active_slowdowns > 0 and self.secondary_slowdown_counter == 0:
            self.secondary_slowdown_counter = SECONDARY_SLOWDOWN_DURATION_STEPS
        if self.secondary_slowdown_counter > 0:
            self.secondary_slowdown_counter -= 1
        if self.green_corridor_active and self.green_corridor_expires_step:
            if step >= self.green_corridor_expires_step:
                self.green_corridor_active = False
                self.green_corridor_route = None
                self.green_corridor_expires_step = None
        self._rebuild_od_matrix()
        self.last_update_step = step
        self.next_update_step = step + TRAFFIC_UPDATE_EVERY_N_STEPS
    def _rebuild_od_matrix(self) -> None:
        self.od_matrix_current = {
            origin: {
                dest: travel_min * self.current_global_multiplier
                for dest, travel_min in dest_row.items()
            }
            for origin, dest_row in self.od_matrix_base.items()
        }
class RewardEntry(ImmutableEmergiModel):
    entry_id: str                   = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber
    channel: RewardChannel
    value: float
    incident_id: Optional[IncidentID] = Field(default=None)
    unit_id: Optional[UnitID]         = Field(default=None)
    hospital_id: Optional[HospitalID] = Field(default=None)
    description: str                  = Field(default="")
    @computed_field
    @property
    def is_penalty(self) -> bool:
        return self.value < 0.0
    @computed_field
    @property
    def is_bonus(self) -> bool:
        return self.value > 0.0
class StepRewardBreakdown(EmergiBaseModel):
    step: StepNumber
    entries: List[RewardEntry]   = Field(default_factory=list)
    @computed_field
    @property
    def total(self) -> float:
        return sum(e.value for e in self.entries)
    @computed_field
    @property
    def by_channel(self) -> Dict[str, float]:
        result: Dict[str, float] = defaultdict(float)
        for e in self.entries:
            result[e.channel.value] += e.value
        return dict(result)
    @computed_field
    @property
    def penalty_total(self) -> float:
        return sum(e.value for e in self.entries if e.is_penalty)
    @computed_field
    @property
    def bonus_total(self) -> float:
        return sum(e.value for e in self.entries if e.is_bonus)
    def add(
        self,
        channel: RewardChannel,
        value: float,
        step: int,
        description: str = "",
        incident_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
    ) -> None:
        self.entries.append(RewardEntry(
            step=step,
            channel=channel,
            value=value,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
            description=description,
        ))
class EpisodeRewardAccumulator(EmergiBaseModel):
    survival: float               = Field(default=0.0)
    response_time: float          = Field(default=0.0)
    triage_accuracy: float        = Field(default=0.0)
    protocol_compliance: float    = Field(default=0.0)
    hospital_routing: float       = Field(default=0.0)
    diversion_avoidance: float    = Field(default=0.0)
    multi_agency_coordination: float = Field(default=0.0)
    crew_fatigue_management: float= Field(default=0.0)
    preposition_efficiency: float = Field(default=0.0)
    transfer_timeliness: float    = Field(default=0.0)
    surge_management: float       = Field(default=0.0)
    mutual_aid_efficiency: float  = Field(default=0.0)
    cascade_avoidance: float      = Field(default=0.0)
    comms_resilience: float       = Field(default=0.0)
    noop_penalty: float           = Field(default=0.0)
    CHANNEL_WEIGHTS: ClassVar[Dict[str, float]] = {
        "survival":                  0.30,
        "response_time":             0.20,
        "triage_accuracy":           0.10,
        "protocol_compliance":       0.10,
        "hospital_routing":          0.08,
        "diversion_avoidance":       0.05,
        "multi_agency_coordination": 0.04,
        "crew_fatigue_management":   0.03,
        "preposition_efficiency":    0.03,
        "transfer_timeliness":       0.02,
        "surge_management":          0.02,
        "mutual_aid_efficiency":     0.01,
        "cascade_avoidance":         0.01,
        "comms_resilience":          0.005,
        "noop_penalty":             -0.005,  
    }
    step_breakdowns: List[StepRewardBreakdown] = Field(default_factory=list)
    total_entries: int              = Field(default=0, ge=0)
    @computed_field
    @property
    def weighted_score(self) -> Score:
        raw = 0.0
        for ch, w in self.CHANNEL_WEIGHTS.items():
            raw += w * getattr(self, ch, 0.0)
        return clamp_score(raw)
    @computed_field
    @property
    def total_reward(self) -> float:
        return (
            self.survival + self.response_time + self.triage_accuracy
            + self.protocol_compliance + self.hospital_routing
            + self.diversion_avoidance + self.multi_agency_coordination
            + self.crew_fatigue_management + self.preposition_efficiency
            + self.transfer_timeliness + self.surge_management
            + self.mutual_aid_efficiency + self.cascade_avoidance
            + self.comms_resilience + self.noop_penalty
        )
    @computed_field
    @property
    def cumulative_diversion_penalty(self) -> float:
        return self.diversion_avoidance  
    @computed_field
    @property
    def cumulative_protocol_reward(self) -> float:
        return self.protocol_compliance
    @computed_field
    @property
    def cumulative_survival_reward(self) -> float:
        return self.survival
    def apply(self, breakdown: StepRewardBreakdown) -> None:
        for entry in breakdown.entries:
            ch = entry.channel.value if hasattr(entry.channel, "value") else entry.channel
            if hasattr(self, ch):
                current = getattr(self, ch)
                setattr(self, ch, current + entry.value)
        self.step_breakdowns.append(breakdown)
        self.total_entries += len(breakdown.entries)
    def add_direct(
        self,
        channel: RewardChannel,
        value: float,
        step: int,
        description: str = "",
        incident_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
    ) -> None:
        breakdown = StepRewardBreakdown(step=step)
        breakdown.add(
            channel=channel,
            value=value,
            step=step,
            description=description,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
        )
        self.apply(breakdown)
    def channel_breakdown_summary(self) -> str:
        lines = ["EpisodeRewardAccumulator:"]
        for ch in self.CHANNEL_WEIGHTS:
            v = getattr(self, ch, 0.0)
            lines.append(f"  {ch:35s}: {v:+.4f}")
        lines.append(f"  {'weighted_score':35s}: {self.weighted_score:.4f}")
        return "\n".join(lines)
class ProtocolEvent(ImmutableEmergiModel):
    event_id: str               = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber
    rule: ProtocolRule
    compliant: bool
    value: float                
    incident_id: Optional[IncidentID] = Field(default=None)
    unit_id: Optional[UnitID]         = Field(default=None)
    hospital_id: Optional[HospitalID] = Field(default=None)
    description: str            = Field(default="")
class ProtocolComplianceState(EmergiBaseModel):
    events: List[ProtocolEvent]  = Field(default_factory=list)
    rule_correct: Dict[str, int]    = Field(default_factory=dict)
    rule_violated: Dict[str, int]   = Field(default_factory=dict)
    bonus_earned: float    = Field(default=0.0, ge=0.0)
    penalty_total: float   = Field(default=0.0, le=0.0)
    micu_stemi_correct: int   = Field(default=0, ge=0)
    micu_stemi_total: int     = Field(default=0, ge=0)
    start_triage_correct: int = Field(default=0, ge=0)
    start_triage_total: int   = Field(default=0, ge=0)
    diversion_violations: int = Field(default=0, ge=0)
    multi_agency_violations: int = Field(default=0, ge=0)
    @computed_field
    @property
    def total_correct(self) -> int:
        return sum(self.rule_correct.values())
    @computed_field
    @property
    def total_violations(self) -> int:
        return sum(self.rule_violated.values())
    @computed_field
    @property
    def compliance_rate(self) -> float:
        total = self.total_correct + self.total_violations
        if total == 0:
            return 1.0
        return clamp_score(self.total_correct / total)
    @computed_field
    @property
    def net_compliance_score(self) -> float:
        return clamp_score(self.bonus_earned + self.penalty_total)
    @computed_field
    @property
    def micu_stemi_correct_rate(self) -> float:
        if self.micu_stemi_total == 0:
            return 1.0
        return clamp_score(self.micu_stemi_correct / self.micu_stemi_total)
    @computed_field
    @property
    def start_triage_correct_rate(self) -> float:
        if self.start_triage_total == 0:
            return 1.0
        return clamp_score(self.start_triage_correct / self.start_triage_total)
    def record_event(
        self,
        rule: ProtocolRule,
        compliant: bool,
        step: int,
        incident_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
        description: str = "",
    ) -> float:
        rule_key = rule.value
        if compliant:
            delta = min(rule.bonus_per_correct, PROTOCOL_COMPLIANCE_MAX_BONUS - self.bonus_earned)
            self.bonus_earned = clamp_score(self.bonus_earned + delta)
            self.rule_correct[rule_key] = self.rule_correct.get(rule_key, 0) + 1
        else:
            delta = rule.penalty_per_violation
            self.penalty_total += delta
            self.rule_violated[rule_key] = self.rule_violated.get(rule_key, 0) + 1
            if rule == ProtocolRule.NO_ROUTING_TO_DIVERTED:
                self.diversion_violations += 1
            if rule == ProtocolRule.MULTI_AGENCY_TRAPPED:
                self.multi_agency_violations += 1
        ev = ProtocolEvent(
            step=step,
            rule=rule,
            compliant=compliant,
            value=delta,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
            description=description,
        )
        self.events.append(ev)
        if rule == ProtocolRule.MICU_FOR_STEMI:
            self.micu_stemi_total += 1
            if compliant:
                self.micu_stemi_correct += 1
        if rule == ProtocolRule.START_TRIAGE_IN_MCI:
            self.start_triage_total += 1
            if compliant:
                self.start_triage_correct += 1
        return delta
class MutualAidZonePairing(EmergiBaseModel):
    requesting_zone: ZoneID
    providing_zone: ZoneID
    requests_this_episode: int   = Field(default=0, ge=0)
    units_sent_this_episode: int = Field(default=0, ge=0)
    units_returned: int          = Field(default=0, ge=0)
    over_requests: int           = Field(default=0, ge=0)
    total_penalty_incurred: float= Field(default=0.0, le=0.0)
    @computed_field
    @property
    def net_units_active(self) -> int:
        return max(0, self.units_sent_this_episode - self.units_returned)
    @computed_field
    @property
    def pairing_key(self) -> str:
        return f"{self.requesting_zone}:{self.providing_zone}"
class MutualAidCoordinationState(EmergiBaseModel):
    active_requests: List[MutualAidRequest]   = Field(default_factory=list)
    fulfilled_requests: List[MutualAidRequest]= Field(default_factory=list)
    cancelled_requests: List[MutualAidRequest]= Field(default_factory=list)
    zone_pairings: Dict[str, MutualAidZonePairing] = Field(default_factory=dict)
    zone_available_units: Dict[ZoneID, int]    = Field(default_factory=dict)
    over_request_penalty_total: float         = Field(default=0.0, le=0.0)
    total_units_received: int                 = Field(default=0, ge=0)
    total_units_returned: int                 = Field(default=0, ge=0)
    @computed_field
    @property
    def active_request_count(self) -> int:
        return len([r for r in self.active_requests if not r.cancelled and not r.fulfilled])
    @computed_field
    @property
    def units_en_route(self) -> int:
        return len([r for r in self.active_requests if not r.fulfilled and not r.cancelled])
    def submit_request(self, request: MutualAidRequest) -> None:
        self.active_requests.append(request)
        key = f"{request.requesting_zone}:{request.providing_zone}"
        if key not in self.zone_pairings:
            self.zone_pairings[key] = MutualAidZonePairing(
                requesting_zone=request.requesting_zone,
                providing_zone=request.providing_zone,
            )
        self.zone_pairings[key].requests_this_episode += 1
    def fulfil_request(self, request_id: str, step: int) -> Optional[MutualAidRequest]:
        for r in self.active_requests:
            if r.request_id == request_id:
                r.fulfilled = True
                self.fulfilled_requests.append(r)
                self.active_requests.remove(r)
                self.total_units_received += r.units_requested
                key = f"{r.requesting_zone}:{r.providing_zone}"
                if key in self.zone_pairings:
                    self.zone_pairings[key].units_sent_this_episode += r.units_requested
                return r
        return None
    def penalise_over_request(self, units_over: int) -> float:
        penalty = MUTUAL_AID_OVER_REQUEST_PENALTY * units_over
        self.over_request_penalty_total += penalty
        return penalty
    def tick(self, step: int) -> List[MutualAidRequest]:
        fulfilled_this_step: List[MutualAidRequest] = []
        for r in self.active_requests:
            if r.cancelled or r.fulfilled:
                continue
            if hasattr(r, "eta_steps") and r.eta_steps > 0:
                pass
        return fulfilled_this_step
class CascadeFailureTracker(EmergiBaseModel):
    risk_score: float           = Field(default=0.0, ge=0.0, le=1.0)
    level: int                  = Field(default=0, ge=0, le=3)  
    occurred: bool              = Field(default=False)
    occurrence_step: Optional[int] = Field(default=None)
    steps_until_cascade: Optional[int] = Field(default=None, ge=0)
    warning_issued_step: Optional[int] = Field(default=None)
    hospital_diversion_count: int = Field(default=0, ge=0)
    fleet_unavailable_pct: float  = Field(default=0.0, ge=0.0, le=1.0)
    er_occupancy_avg: float       = Field(default=0.0, ge=0.0, le=1.0)
    simultaneous_mci_count: int   = Field(default=0, ge=0)
    mutual_aid_saturated: bool    = Field(default=False)
    @computed_field
    @property
    def is_critical(self) -> bool:
        return self.risk_score >= CASCADE_LEVEL_2_THRESHOLD
    @computed_field
    @property
    def is_imminent(self) -> bool:
        return self.risk_score >= CASCADE_LEVEL_3_THRESHOLD
    def recompute(
        self,
        hospital_diversion_count: int,
        fleet_unavailable_pct: float,
        er_occupancy_avg: float,
        simultaneous_mci_count: int,
        mutual_aid_saturated: bool,
        step: int,
    ) -> None:
        self.hospital_diversion_count = hospital_diversion_count
        self.fleet_unavailable_pct    = fleet_unavailable_pct
        self.er_occupancy_avg         = er_occupancy_avg
        self.simultaneous_mci_count   = simultaneous_mci_count
        self.mutual_aid_saturated     = mutual_aid_saturated
        hosp_factor    = clamp_score(hospital_diversion_count / 8.0)
        fleet_factor   = clamp_score(fleet_unavailable_pct)
        er_factor      = clamp_score((er_occupancy_avg - 0.7) / 0.3)
        mci_factor     = clamp_score(simultaneous_mci_count / 5.0)
        aid_factor     = 0.20 if mutual_aid_saturated else 0.0
        self.risk_score = clamp_score(
            0.30 * hosp_factor
            + 0.25 * fleet_factor
            + 0.25 * er_factor
            + 0.15 * mci_factor
            + 0.05 * aid_factor
        )
        old_level = self.level
        if self.risk_score >= CASCADE_LEVEL_3_THRESHOLD:
            self.level = 3
        elif self.risk_score >= CASCADE_LEVEL_2_THRESHOLD:
            self.level = 2
        elif self.risk_score >= CASCADE_LEVEL_1_THRESHOLD:
            self.level = 1
        else:
            self.level = 0
        if self.level >= 1 and old_level == 0:
            self.warning_issued_step = step
        if not self.occurred and self.is_imminent:
            estimated_steps = max(1, int((1.0 - self.risk_score) / 0.05))
            self.steps_until_cascade = estimated_steps
class SurgeCoordinationState(EmergiBaseModel):
    surge_declared: bool              = Field(default=False)
    surge_level: int                  = Field(default=0, ge=0, le=4)
    surge_declared_step: Optional[int]= Field(default=None)
    surge_scope_zones: List[ZoneID]   = Field(default_factory=list)
    surge_hospitals_notified: List[HospitalID] = Field(default_factory=list)
    off_duty_staff_recalled: bool     = Field(default=False)
    ndma_notified: bool               = Field(default=False)
    cascade: CascadeFailureTracker    = Field(default_factory=CascadeFailureTracker)
    mutual_aid: MutualAidCoordinationState = Field(default_factory=MutualAidCoordinationState)
    hospitals_on_diversion: List[HospitalID]  = Field(default_factory=list)
    hospitals_at_capacity: List[HospitalID]   = Field(default_factory=list)
    system_er_occupancy_avg: float            = Field(default=0.0, ge=0.0, le=1.0)
    system_icu_occupancy_avg: float           = Field(default=0.0, ge=0.0, le=1.0)
    simultaneous_mci_count: int               = Field(default=0, ge=0)
    comms_failure_count: int                  = Field(default=0, ge=0)
    @computed_field
    @property
    def system_under_stress(self) -> bool:
        return (
            self.system_er_occupancy_avg >= 0.80
            or len(self.hospitals_on_diversion) >= 3
            or self.simultaneous_mci_count >= 2
        )
    @computed_field
    @property
    def surge_required(self) -> bool:
        return (
            self.simultaneous_mci_count >= 3
            or self.system_er_occupancy_avg >= 0.85
            or self.cascade.risk_score >= 0.50
            or len(self.hospitals_on_diversion) >= 4
        )
    @computed_field
    @property
    def diversion_penalty_exposure(self) -> float:
        return len(self.hospitals_on_diversion) * abs(DIVERSION_PENALTY)
    def declare_surge(
        self,
        level: int,
        zones: List[ZoneID],
        hospitals: List[HospitalID],
        step: int,
        recall_staff: bool = False,
        notify_ndma: bool = False,
    ) -> None:
        if self.surge_declared and level <= self.surge_level:
            return
        self.surge_declared = True
        self.surge_level = level
        self.surge_declared_step = step
        self.surge_scope_zones = zones
        self.surge_hospitals_notified = hospitals
        self.off_duty_staff_recalled = recall_staff
        self.ndma_notified = notify_ndma
    def update_system_metrics(
        self,
        hospital_states: List[HospitalState],
        fleet_states: List["UnitState"],
        active_incidents: List[IncidentState],
    ) -> None:
        on_div = [h.hospital_id for h in hospital_states if h.is_on_diversion]
        at_cap = [
            h.hospital_id for h in hospital_states
            if h.beds.er_occupancy_pct >= 0.95
        ]
        self.hospitals_on_diversion = on_div
        self.hospitals_at_capacity  = at_cap
        if hospital_states:
            self.system_er_occupancy_avg = sum(
                h.beds.er_occupancy_pct for h in hospital_states
            ) / len(hospital_states)
            self.system_icu_occupancy_avg = sum(
                h.beds.icu_occupancy_pct for h in hospital_states
            ) / len(hospital_states)
        self.simultaneous_mci_count = sum(1 for i in active_incidents if i.mci_declared)
        self.comms_failure_count    = sum(1 for u in fleet_states if not u.comms_active)
        comms_lost_total  = self.comms_failure_count
        fleet_unavailable = sum(1 for u in fleet_states if not u.is_available)
        fleet_unavail_pct = fleet_unavailable / max(1, len(fleet_states))
        self.cascade.recompute(
            hospital_diversion_count=len(on_div),
            fleet_unavailable_pct=fleet_unavail_pct,
            er_occupancy_avg=self.system_er_occupancy_avg,
            simultaneous_mci_count=self.simultaneous_mci_count,
            mutual_aid_saturated=(self.mutual_aid.active_request_count >= 4),
            step=0,  
        )
class SystemEvent(ImmutableEmergiModel):
    event_id: str               = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SystemEventType
    step: StepNumber
    episode_id: str
    incident_id: Optional[IncidentID]   = Field(default=None)
    unit_id: Optional[UnitID]           = Field(default=None)
    hospital_id: Optional[HospitalID]   = Field(default=None)
    zone_id: Optional[ZoneID]           = Field(default=None)
    payload: Dict[str, Any]             = Field(default_factory=dict)
    reward_delta: float                 = Field(default=0.0)
    reward_channel: Optional[str]       = Field(default=None)
    timestamp_iso: str                  = Field(default_factory=utc_now_iso)
    @computed_field
    @property
    def has_reward(self) -> bool:
        return self.reward_delta != 0.0
    @computed_field
    @property
    def summary(self) -> str:
        parts = [f"[{self.event_type.value}] step={self.step}"]
        if self.incident_id:
            parts.append(f"incident={self.incident_id}")
        if self.unit_id:
            parts.append(f"unit={self.unit_id}")
        if self.hospital_id:
            parts.append(f"hospital={self.hospital_id}")
        if self.has_reward:
            parts.append(f"Δreward={self.reward_delta:+.3f}")
        return " ".join(parts)
class EventLog(EmergiBaseModel):
    episode_id: str
    events: List[SystemEvent]   = Field(default_factory=list)
    archived_event_count: int   = Field(default=0, ge=0)
    max_size: int               = Field(default=EVENT_LOG_MAX_SIZE)
    @computed_field
    @property
    def total_events(self) -> int:
        return len(self.events) + self.archived_event_count
    @computed_field
    @property
    def last_event(self) -> Optional[SystemEvent]:
        return self.events[-1] if self.events else None
    def append(self, event: SystemEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_size:
            flush_count = self.max_size // 4
            self.archived_event_count += flush_count
            self.events = self.events[flush_count:]
    def events_for_incident(self, incident_id: IncidentID) -> List[SystemEvent]:
        return [e for e in self.events if e.incident_id == incident_id]
    def events_for_unit(self, unit_id: UnitID) -> List[SystemEvent]:
        return [e for e in self.events if e.unit_id == unit_id]
    def events_of_type(self, event_type: SystemEventType) -> List[SystemEvent]:
        return [e for e in self.events if e.event_type == event_type]
    def reward_events(self) -> List[SystemEvent]:
        return [e for e in self.events if e.has_reward]
    def step_events(self, step: int) -> List[SystemEvent]:
        return [e for e in self.events if e.step == step]
class IncidentOutcome(ImmutableEmergiModel):
    incident_id: IncidentID
    template_id: TemplateID
    condition: str
    incident_type: IncidentType
    severity: SeverityLevel
    zone_id: ZoneID
    phase_final: IncidentPhase
    step_created: StepNumber
    step_resolved: Optional[int]
    victim_count: int
    patients_survived: int
    patients_died: int
    patients_unknown: int
    response_time_minutes: Optional[float]
    golden_hour_met: bool
    golden_hour_fraction: float
    triage_accuracy: float
    protocol_violations: int
    units_used: List[UnitID]
    primary_hospital_id: Optional[HospitalID]
    dispatch_count: int
    survival_delta_total: float
    peak_survival_probability: float
    final_survival_probability: float
    score_contribution: float
    @computed_field
    @property
    def case_fatality_rate(self) -> float:
        if self.victim_count == 0:
            return 0.0
        return self.patients_died / self.victim_count
    @computed_field
    @property
    def was_successful(self) -> bool:
        return self.phase_final == IncidentPhase.RESOLVED
    @computed_field
    @property
    def outcome_summary(self) -> str:
        status = "✓ RESOLVED" if self.was_successful else "✗ FAILED"
        return (
            f"{status} | {self.incident_id} | {self.condition} | "
            f"Victims: {self.victim_count} "
            f"(survived={self.patients_survived}, died={self.patients_died}) | "
            f"RT={self.response_time_minutes:.1f}min | "
            f"GH={'met' if self.golden_hour_met else 'exceeded'} | "
            f"score={self.score_contribution:+.4f}"
        )
class FleetStatistics(EmergiBaseModel):
    total_missions_completed: int  = Field(default=0, ge=0)
    total_missions_failed: int     = Field(default=0, ge=0)
    total_patients_transported: int= Field(default=0, ge=0)
    total_distance_km: float       = Field(default=0.0, ge=0.0)
    avg_response_time_minutes: float = Field(default=0.0, ge=0.0)
    avg_on_scene_duration_minutes: float = Field(default=0.0, ge=0.0)
    peak_simultaneous_deployed: int= Field(default=0, ge=0)
    fatigue_events: int            = Field(default=0, ge=0)
    crew_swaps_completed: int      = Field(default=0, ge=0)
    comms_failures_total: int      = Field(default=0, ge=0)
    mutual_aid_units_received: int = Field(default=0, ge=0)
    micu_missions: int = Field(default=0, ge=0)
    als_missions: int  = Field(default=0, ge=0)
    bls_missions: int  = Field(default=0, ge=0)
    @computed_field
    @property
    def mission_success_rate(self) -> float:
        total = self.total_missions_completed + self.total_missions_failed
        if total == 0:
            return 1.0
        return self.total_missions_completed / total
    @computed_field
    @property
    def fleet_utilisation_pct(self) -> float:
        return clamp_score(self.peak_simultaneous_deployed / max(1, FLEET_SIZE_DEFAULT))
class HospitalNetworkStatistics(EmergiBaseModel):
    total_admissions: int          = Field(default=0, ge=0)
    total_diversions_activated: int= Field(default=0, ge=0)
    total_bypasses_executed: int   = Field(default=0, ge=0)
    total_transfers: int           = Field(default=0, ge=0)
    peak_diversion_count: int      = Field(default=0, ge=0)
    avg_er_occupancy: float        = Field(default=0.0, ge=0.0, le=1.0)
    avg_icu_occupancy: float       = Field(default=0.0, ge=0.0, le=1.0)
    cascade_failure_occurred: bool = Field(default=False)
    surge_declared: bool           = Field(default=False)
class EpisodeStatistics(EmergiBaseModel):
    episode_id: str
    task_id: TaskID
    task_difficulty: TaskDifficulty
    seed: int
    steps_executed: int             = Field(default=0, ge=0)
    episode_duration_minutes: float = Field(default=0.0, ge=0.0)
    terminated_early: bool          = Field(default=False)
    termination_reason: Optional[str] = Field(default=None)
    total_incidents_created: int    = Field(default=0, ge=0)
    incidents_resolved: int         = Field(default=0, ge=0)
    incidents_failed: int           = Field(default=0, ge=0)
    incidents_cancelled: int        = Field(default=0, ge=0)
    p1_incidents_total: int         = Field(default=0, ge=0)
    p1_incidents_resolved: int      = Field(default=0, ge=0)
    p2_incidents_total: int         = Field(default=0, ge=0)
    p3_incidents_total: int         = Field(default=0, ge=0)
    mci_events: int                 = Field(default=0, ge=0)
    total_patients: int             = Field(default=0, ge=0)
    patients_survived: int          = Field(default=0, ge=0)
    patients_died: int              = Field(default=0, ge=0)
    triage_accuracy_avg: float      = Field(default=0.0, ge=0.0, le=1.0)
    fleet: FleetStatistics          = Field(default_factory=FleetStatistics)
    hospital_network: HospitalNetworkStatistics = Field(default_factory=HospitalNetworkStatistics)
    final_episode_score: float      = Field(default=0.0, ge=0.0, le=1.0)
    baseline_score: float           = Field(default=0.0, ge=0.0, le=1.0)
    score_vs_baseline: float        = Field(default=0.0)
    incident_outcomes: List[IncidentOutcome] = Field(default_factory=list)
    @computed_field
    @property
    def p1_resolution_rate(self) -> float:
        if self.p1_incidents_total == 0:
            return 1.0
        return clamp_score(self.p1_incidents_resolved / self.p1_incidents_total)
    @computed_field
    @property
    def overall_resolution_rate(self) -> float:
        if self.total_incidents_created == 0:
            return 1.0
        return clamp_score(self.incidents_resolved / self.total_incidents_created)
    @computed_field
    @property
    def case_fatality_rate(self) -> float:
        if self.total_patients == 0:
            return 0.0
        return self.patients_died / self.total_patients
    @computed_field
    @property
    def beating_baseline(self) -> bool:
        return self.final_episode_score > self.baseline_score
    def record_outcome(self, outcome: IncidentOutcome) -> None:
        self.incident_outcomes.append(outcome)
        if outcome.was_successful:
            self.incidents_resolved += 1
        else:
            self.incidents_failed += 1
        self.patients_survived += outcome.patients_survived
        self.patients_died     += outcome.patients_died
        self.total_patients    += outcome.victim_count
class StateCheckpoint(EmergiBaseModel):
    checkpoint_id: str          = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber
    episode_id: str
    task_id: TaskID
    created_at: str             = Field(default_factory=utc_now_iso)
    incident_states_json: str   = Field(default="{}")
    unit_states_json: str       = Field(default="{}")
    hospital_states_json: str   = Field(default="{}")
    zone_states_json: str       = Field(default="{}")
    traffic_state_json: str     = Field(default="{}")
    reward_accumulator_json: str= Field(default="{}")
    protocol_state_json: str    = Field(default="{}")
    surge_state_json: str       = Field(default="{}")
    state_hash: str             = Field(default="")
    @computed_field
    @property
    def size_bytes_estimate(self) -> int:
        return (
            len(self.incident_states_json)
            + len(self.unit_states_json)
            + len(self.hospital_states_json)
            + len(self.zone_states_json)
            + len(self.traffic_state_json)
            + len(self.reward_accumulator_json)
            + len(self.protocol_state_json)
            + len(self.surge_state_json)
        )
    def compute_hash(self) -> str:
        blob = (
            self.incident_states_json
            + self.unit_states_json
            + self.hospital_states_json
        )
        self.state_hash = hashlib.md5(blob.encode()).hexdigest()
        return self.state_hash
    def verify_integrity(self) -> bool:
        expected = hashlib.md5(
            (
                self.incident_states_json
                + self.unit_states_json
                + self.hospital_states_json
            ).encode()
        ).hexdigest()
        return expected == self.state_hash
class StateDiff(ImmutableEmergiModel):
    from_step: StepNumber
    to_step: StepNumber
    episode_id: str
    incidents_created: List[IncidentID]     = Field(default_factory=list)
    incidents_resolved: List[IncidentID]    = Field(default_factory=list)
    incidents_failed: List[IncidentID]      = Field(default_factory=list)
    units_status_changed: Dict[UnitID, Tuple[str, str]] = Field(default_factory=dict)
    hospitals_diverted: List[HospitalID]    = Field(default_factory=list)
    hospitals_cleared: List[HospitalID]     = Field(default_factory=list)
    reward_delta: float                     = Field(default=0.0)
    score_delta: float                      = Field(default=0.0)
    events_count: int                       = Field(default=0, ge=0)
    @computed_field
    @property
    def steps_covered(self) -> int:
        return self.to_step - self.from_step
    @computed_field
    @property
    def has_significant_changes(self) -> bool:
        return bool(
            self.incidents_created
            or self.incidents_resolved
            or self.incidents_failed
            or self.units_status_changed
            or self.hospitals_diverted
            or self.hospitals_cleared
            or abs(self.reward_delta) > 0.01
        )
    @computed_field
    @property
    def summary(self) -> str:
        return (
            f"Diff steps {self.from_step}→{self.to_step} | "
            f"+{len(self.incidents_created)} incidents | "
            f"{len(self.incidents_resolved)} resolved | "
            f"{len(self.incidents_failed)} failed | "
            f"{len(self.units_status_changed)} unit changes | "
            f"Δscore={self.score_delta:+.4f}"
        )
class DemandGeneratorState(EmergiBaseModel):
    task_id: TaskID
    seed: int
    rng_calls_consumed: int         = Field(default=0, ge=0)
    zone_baseline_rates: Dict[ZoneID, float] = Field(default_factory=dict)
    forecast_generated_step: int    = Field(default=0, ge=0)
    forecast_horizon_hours: int     = Field(default=DEMAND_FORECAST_HORIZON_HOURS)
    noise_pct: float                = Field(default=DEMAND_FORECAST_NOISE_PCT)
    incidents_generated_by_zone: Dict[ZoneID, int] = Field(default_factory=dict)
    incidents_suppressed: int       = Field(default=0, ge=0)   
    template_ids_used: List[TemplateID] = Field(default_factory=list)
    template_cooldown: Dict[TemplateID, int] = Field(default_factory=dict)  
    @computed_field
    @property
    def total_incidents_generated(self) -> int:
        return sum(self.incidents_generated_by_zone.values())
    @computed_field
    @property
    def highest_demand_zone(self) -> Optional[ZoneID]:
        if not self.incidents_generated_by_zone:
            return None
        return max(self.incidents_generated_by_zone, key=self.incidents_generated_by_zone.get)
    def register_generation(self, zone_id: ZoneID, template_id: TemplateID, step: int) -> None:
        self.incidents_generated_by_zone[zone_id] = (
            self.incidents_generated_by_zone.get(zone_id, 0) + 1
        )
        self.template_ids_used.append(template_id)
        self.rng_calls_consumed += 1
    def is_template_on_cooldown(self, template_id: TemplateID, step: int) -> bool:
        cooldown_until = self.template_cooldown.get(template_id, 0)
        return step < cooldown_until
    def set_template_cooldown(self, template_id: TemplateID, step: int, cooldown_steps: int = 5) -> None:
        self.template_cooldown[template_id] = step + cooldown_steps
class ClockState(EmergiBaseModel):
    episode_id: str
    task_id: TaskID
    task_seed: int
    step: int           = Field(default=0, ge=0, le=MAX_STEPS_PER_EPISODE)
    episode_minute: float = Field(default=0.0, ge=0.0)
    wall_clock_hour: int  = Field(default=7, ge=0, le=23)    
    wall_clock_minute: int= Field(default=0, ge=0, le=59)
    day_of_week: str      = Field(default="monday")
    season: Season        = Field(default=Season.SUMMER)
    is_peak_morning: bool = Field(default=False)
    is_peak_evening: bool = Field(default=False)
    next_traffic_update_step: int = Field(default=TRAFFIC_UPDATE_EVERY_N_STEPS, ge=0)
    episode_start_iso: str = Field(default_factory=utc_now_iso)
    @computed_field
    @property
    def steps_remaining(self) -> int:
        return MAX_STEPS_PER_EPISODE - self.step
    @computed_field
    @property
    def elapsed_hours(self) -> float:
        return self.episode_minute / 60.0
    @computed_field
    @property
    def traffic_update_due(self) -> bool:
        return self.step >= self.next_traffic_update_step
    def advance(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.step += 1
            self.episode_minute += TIMESTEP_MINUTES
            total_minutes = int(self.episode_minute)
            self.wall_clock_minute = (self.wall_clock_hour * 60 + total_minutes) % 60
            new_hour = ((self.wall_clock_hour * 60 + total_minutes) // 60) % 24
            self.wall_clock_hour = new_hour
            self.is_peak_morning = new_hour in PEAK_MORNING_HOURS
            self.is_peak_evening = new_hour in PEAK_EVENING_HOURS
    def schedule_next_traffic_update(self) -> None:
        self.next_traffic_update_step = self.step + TRAFFIC_UPDATE_EVERY_N_STEPS
class SimulationState(EmergiBaseModel):
    episode_id: str             = Field(default_factory=new_episode_id)
    schema_version: int         = Field(default=__state_version__)
    task_id: TaskID
    task_seed: int              = Field(..., ge=42, le=50)
    created_at: str             = Field(default_factory=utc_now_iso)
    clock: ClockState
    incidents: Dict[IncidentID, IncidentState]   = Field(default_factory=dict)
    units: Dict[UnitID, UnitState]                = Field(default_factory=dict)
    hospitals: Dict[HospitalID, HospitalState]    = Field(default_factory=dict)
    zones: Dict[ZoneID, ZoneState]                = Field(default_factory=dict)
    incident_queue: List[IncidentID]              = Field(default_factory=list)
    traffic: TrafficEngineState                   = Field(default_factory=TrafficEngineState)
    surge: SurgeCoordinationState                 = Field(default_factory=SurgeCoordinationState)
    protocol: ProtocolComplianceState             = Field(default_factory=ProtocolComplianceState)
    demand_gen: DemandGeneratorState
    rewards: EpisodeRewardAccumulator             = Field(default_factory=EpisodeRewardAccumulator)
    event_log: EventLog
    stats: EpisodeStatistics
    checkpoints: List[StateCheckpoint]            = Field(default_factory=list)
    checkpoint_step_index: Dict[int, str]         = Field(default_factory=dict)  
    episode_done: bool          = Field(default=False)
    termination_reason: Optional[str] = Field(default=None)
    legal_actions: List[str]    = Field(default_factory=list)
    action_mask: Dict[str, bool]= Field(default_factory=dict)
    step_rng_values: List[float]= Field(default_factory=list)  
    def get_incident(self, incident_id: IncidentID) -> Optional[IncidentState]:
        return self.incidents.get(incident_id)
    def get_unit(self, unit_id: UnitID) -> Optional[UnitState]:
        return self.units.get(unit_id)
    def get_hospital(self, hospital_id: HospitalID) -> Optional[HospitalState]:
        return self.hospitals.get(hospital_id)
    def get_zone(self, zone_id: ZoneID) -> Optional[ZoneState]:
        return self.zones.get(zone_id)
    @computed_field
    @property
    def current_step(self) -> int:
        return self.clock.step
    @computed_field
    @property
    def active_incidents(self) -> List[IncidentState]:
        return [
            inc for inc in self.incidents.values()
            if not inc.is_terminal
        ]
    @computed_field
    @property
    def queued_incidents(self) -> List[IncidentState]:
        return [
            self.incidents[iid]
            for iid in self.incident_queue
            if iid in self.incidents and not self.incidents[iid].is_terminal
        ]
    @computed_field
    @property
    def available_units(self) -> List[UnitState]:
        return [u for u in self.units.values() if u.is_available]
    @computed_field
    @property
    def deployable_units(self) -> List[UnitState]:
        return [u for u in self.units.values() if u.is_deployable]
    @computed_field
    @property
    def accepting_hospitals(self) -> List[HospitalState]:
        return [h for h in self.hospitals.values() if h.accepts_patients]
    @computed_field
    @property
    def diverted_hospital_ids(self) -> List[HospitalID]:
        return [h.hospital_id for h in self.hospitals.values() if h.is_on_diversion]
    @computed_field
    @property
    def fleet_available_count(self) -> int:
        return len(self.available_units)
    @computed_field
    @property
    def fleet_deployable_count(self) -> int:
        return len(self.deployable_units)
    @computed_field
    @property
    def p1_incident_ids(self) -> List[IncidentID]:
        return [
            iid for iid in self.incident_queue
            if iid in self.incidents
            and self.incidents[iid].severity == SeverityLevel.P1
            and not self.incidents[iid].is_terminal
        ]
    @computed_field
    @property
    def unassigned_p1_count(self) -> int:
        return sum(
            1 for iid in self.p1_incident_ids
            if self.incidents[iid].phase == IncidentPhase.PENDING
        )
    @computed_field
    @property
    def resource_deficit(self) -> int:
        return self.unassigned_p1_count - self.fleet_deployable_count
    @computed_field
    @property
    def system_stress_level(self) -> str:
        if self.resource_deficit >= 5 or self.surge.cascade.risk_score >= 0.7:
            return "critical"
        if self.resource_deficit >= 3 or self.surge.system_under_stress:
            return "high"
        if self.resource_deficit >= 1:
            return "elevated"
        return "normal"
    @computed_field
    @property
    def current_score(self) -> float:
        return clamp_score(self.rewards.weighted_score)
    @computed_field
    @property
    def mci_active(self) -> bool:
        return any(i.mci_declared for i in self.active_incidents)
    @computed_field
    @property
    def comms_failure_active(self) -> bool:
        return any(not u.comms_active for u in self.units.values())
    @computed_field
    @property
    def units_fatigued_count(self) -> int:
        return sum(1 for u in self.units.values() if u.is_fatigued)
    @computed_field
    @property
    def units_comms_lost_count(self) -> int:
        return sum(1 for u in self.units.values() if not u.comms_active)
    def register_incident(self, incident: IncidentState) -> None:
        if len(self.incident_queue) >= INCIDENT_QUEUE_MAX:
            logger.warning(
                "Incident queue full (%d). Incident %s suppressed.",
                INCIDENT_QUEUE_MAX, incident.incident_id,
            )
            self.demand_gen.incidents_suppressed += 1
            return
        self.incidents[incident.incident_id] = incident
        self.incident_queue.append(incident.incident_id)
        self.stats.total_incidents_created += 1
        zone = self.get_zone(incident.zone_id)
        if zone:
            zone.active_incident_ids.append(incident.incident_id)
            zone.update_alert_level(self.current_step)
        self._emit(SystemEventType.INCIDENT_CREATED, incident_id=incident.incident_id)
    def remove_incident_from_queue(self, incident_id: IncidentID) -> None:
        if incident_id in self.incident_queue:
            self.incident_queue.remove(incident_id)
        zone_id = self.incidents.get(incident_id, None)
        if zone_id:
            inc = self.incidents[incident_id]
            zone = self.get_zone(inc.zone_id)
            if zone and incident_id in zone.active_incident_ids:
                zone.active_incident_ids.remove(incident_id)
                if inc.is_terminal:
                    if inc.phase == IncidentPhase.RESOLVED:
                        zone.resolved_incident_ids.append(incident_id)
                    else:
                        zone.failed_incident_ids.append(incident_id)
                    zone.update_alert_level(self.current_step)
    def register_unit(self, unit: UnitState) -> None:
        self.units[unit.unit_id] = unit
        zone = self.get_zone(unit.home_zone)
        if zone and unit.unit_id not in zone.stationed_unit_ids:
            zone.stationed_unit_ids.append(unit.unit_id)
    def register_hospital(self, hospital: HospitalState) -> None:
        self.hospitals[hospital.hospital_id] = hospital
    def register_zone(self, zone: ZoneState) -> None:
        self.zones[zone.zone_id] = zone
    def available_units_of_type(self, unit_type: UnitType) -> List[UnitState]:
        return [
            u for u in self.units.values()
            if UnitType(u.unit_type) == unit_type and u.is_deployable
        ]
    def hospitals_with_specialty(self, specialty: str) -> List[HospitalState]:
        return [
            h for h in self.hospitals.values()
            if h.has_specialty(specialty) and h.accepts_patients
        ]
    def nearest_available_unit(
        self,
        incident_zone: ZoneID,
        unit_type: Optional[UnitType] = None,
    ) -> Optional[UnitState]:
        candidates = self.deployable_units
        if unit_type:
            candidates = [u for u in candidates if UnitType(u.unit_type) == unit_type]
        if not candidates:
            return None
        def eta(u: UnitState) -> float:
            return self.traffic.get_travel_time(
                u.current_zone, incident_zone, u.unit_type.value if hasattr(u.unit_type, "value") else str(u.unit_type)
            )
        return min(candidates, key=eta)
    def best_hospital_for_condition(
        self,
        condition_key: str,
        origin_zone: ZoneID,
    ) -> Optional[HospitalState]:
        required_specialty = CONDITION_REQUIRED_SPECIALTY.get(condition_key)
        if required_specialty:
            candidates = self.hospitals_with_specialty(required_specialty)
        else:
            candidates = self.accepting_hospitals
        if not candidates:
            return None
        def score(h: HospitalState) -> float:
            pri_score   = clamp_score(1.0 - (h.get_routing_priority(condition_key) - 1) / 8.0)
            cap_score   = 1.0 - h.capacity_pressure
            travel_min  = self.traffic.get_travel_time(origin_zone, h.zone_id)
            travel_score= clamp_score(1.0 - travel_min / 120.0)
            return 0.4 * pri_score + 0.3 * cap_score + 0.3 * travel_score
        return max(candidates, key=score)
    def compute_incident_eta(
        self,
        unit_id: UnitID,
        incident_zone: ZoneID,
    ) -> float:
        unit = self.get_unit(unit_id)
        if unit is None:
            return math.inf
        return self.traffic.get_travel_time(
            unit.current_zone,
            incident_zone,
            str(unit.unit_type) if not hasattr(unit.unit_type, "value") else unit.unit_type.value,
        )
    def _emit(
        self,
        event_type: SystemEventType,
        incident_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        reward_delta: float = 0.0,
        reward_channel: Optional[str] = None,
    ) -> SystemEvent:
        ev = SystemEvent(
            event_type=event_type,
            step=self.clock.step,
            episode_id=self.episode_id,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
            zone_id=zone_id,
            payload=payload or {},
            reward_delta=reward_delta,
            reward_channel=reward_channel,
        )
        self.event_log.append(ev)
        return ev
    def emit_reward(
        self,
        channel: RewardChannel,
        value: float,
        description: str = "",
        incident_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
    ) -> float:
        clamped = value  
        self.rewards.add_direct(
            channel=channel,
            value=clamped,
            step=self.clock.step,
            description=description,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
        )
        self._emit(
            SystemEventType.REWARD_COMPUTED,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
            payload={"channel": channel.value, "value": clamped, "description": description},
            reward_delta=clamped,
            reward_channel=channel.value,
        )
        return clamped
    def transition_incident(
        self,
        incident_id: IncidentID,
        new_phase: IncidentPhase,
    ) -> None:
        inc = self.get_incident(incident_id)
        if inc is None:
            logger.error("transition_incident: incident %s not found", incident_id)
            return
        inc.transition_phase(new_phase, self.clock.step)
        event_map = {
            IncidentPhase.DISPATCHED:       SystemEventType.INCIDENT_DISPATCHED,
            IncidentPhase.UNIT_ON_SCENE:    SystemEventType.INCIDENT_UNIT_ON_SCENE,
            IncidentPhase.TREATMENT:        SystemEventType.INCIDENT_TREATMENT_STARTED,
            IncidentPhase.TRANSPORT:        SystemEventType.INCIDENT_TRANSPORT_STARTED,
            IncidentPhase.HOSPITAL_HANDOFF: SystemEventType.INCIDENT_HOSPITAL_HANDOFF,
            IncidentPhase.RESOLVED:         SystemEventType.INCIDENT_RESOLVED,
            IncidentPhase.FAILED:           SystemEventType.INCIDENT_FAILED,
            IncidentPhase.CANCELLED:        SystemEventType.INCIDENT_CANCELLED,
        }
        event_type = event_map.get(new_phase)
        if event_type:
            self._emit(event_type, incident_id=incident_id)
        if new_phase in (IncidentPhase.RESOLVED, IncidentPhase.FAILED, IncidentPhase.CANCELLED):
            self.remove_incident_from_queue(incident_id)
    def dispatch_unit(
        self,
        unit_id: UnitID,
        incident_id: IncidentID,
        destination_hospital_id: Optional[HospitalID],
        destination_zone_id: Optional[ZoneID],
        routing_strategy: str = "standard",
        eta_to_scene_steps: int = 2,
    ) -> bool:
        unit = self.get_unit(unit_id)
        if unit is None or not unit.is_deployable:
            logger.warning("dispatch_unit: unit %s not deployable", unit_id)
            return False
        inc = self.get_incident(incident_id)
        if inc is None:
            logger.warning("dispatch_unit: incident %s not found", incident_id)
            return False
        unit.transition_status(UnitStatus.DISPATCHED, self.clock.step)
        unit.mission = UnitMissionState(
            mission_phase=UnitMissionPhase.EN_ROUTE_TO_SCENE,
            incident_id=incident_id,
            destination_hospital_id=destination_hospital_id,
            destination_zone_id=destination_zone_id,
            steps_remaining_to_scene=eta_to_scene_steps,
            routing_strategy=routing_strategy,
        )
        inc.add_dispatch(
            unit_id=unit_id,
            unit_type=UnitType(unit.unit_type),
            step=self.clock.step,
            hospital_id=destination_hospital_id,
            routing_strategy=routing_strategy,
        )
        if destination_hospital_id:
            hosp = self.get_hospital(destination_hospital_id)
            if hosp:
                hosp.add_incoming_transport(unit_id)
        self._emit(
            SystemEventType.UNIT_DISPATCHED,
            unit_id=unit_id,
            incident_id=incident_id,
            payload={"hospital": destination_hospital_id, "routing": routing_strategy},
        )
        return True
    def return_unit_to_base(self, unit_id: UnitID, steps: int = 2) -> None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return
        if unit.mission and unit.mission.destination_hospital_id:
            hosp = self.get_hospital(unit.mission.destination_hospital_id)
            if hosp:
                hosp.complete_incoming_transport(unit_id)
        unit.transition_status(UnitStatus.RETURNING, self.clock.step)
        if unit.mission:
            unit.mission.mission_phase = UnitMissionPhase.RETURNING_TO_BASE
            unit.mission.steps_remaining_to_base = steps
        self._emit(SystemEventType.UNIT_RETURNED_BASE, unit_id=unit_id)
    def complete_unit_return(self, unit_id: UnitID) -> None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return
        rt = None
        if unit.mission and unit.mission.incident_id:
            inc = self.get_incident(unit.mission.incident_id)
            if inc and inc.step_dispatched is not None and inc.step_first_unit_on_scene is not None:
                rt = (inc.step_first_unit_on_scene - inc.step_dispatched) * float(TIMESTEP_MINUTES)
        unit.complete_mission(response_time_min=rt)
        unit.mission = None
        unit.transition_status(UnitStatus.AVAILABLE, self.clock.step)
    def tick(self, rng_float: float = 0.5) -> None:
        step = self.clock.step
        self.clock.advance()
        if self.clock.traffic_update_due:
            self.traffic.tick(
                step=self.clock.step,
                hour=self.clock.wall_clock_hour,
                rng=rng_float,
            )
            self.clock.schedule_next_traffic_update()
            self._emit(SystemEventType.TRAFFIC_MATRIX_UPDATED)
        for unit in self.units.values():
            unit.advance_fatigue()
            if not unit.is_fatigued and unit.crew_fatigue.fatigued:
                self._emit(SystemEventType.UNIT_FATIGUED, unit_id=unit.unit_id)
            changed = unit.tick_comms(rng_float)
            if changed:
                ev_type = (
                    SystemEventType.UNIT_COMMS_LOST
                    if not unit.comms_active
                    else SystemEventType.UNIT_COMMS_RESTORED
                )
                self._emit(ev_type, unit_id=unit.unit_id)
            if unit.crew_swap_requested and unit.crew_swap_eta_steps is not None:
                unit.crew_swap_eta_steps -= 1
                if unit.crew_swap_eta_steps <= 0:
                    unit.crew_swap_requested = False
                    unit.crew_swap_eta_steps = None
                    unit.crew_swap_count_this_episode += 1
                    unit.transition_status(UnitStatus.AVAILABLE, self.clock.step)
                    self._emit(SystemEventType.UNIT_CREW_SWAP_COMPLETED, unit_id=unit.unit_id)
            if unit.mutual_aid_eta_steps_remaining is not None and unit.mutual_aid_eta_steps_remaining > 0:
                unit.mutual_aid_eta_steps_remaining -= 1
                if unit.mutual_aid_eta_steps_remaining == 0:
                    unit.mutual_aid_eta_steps_remaining = None
                    self._emit(SystemEventType.UNIT_MUTUAL_AID_DEPLOYED, unit_id=unit.unit_id)
            if unit.mission:
                m = unit.mission
                if m.mission_phase == UnitMissionPhase.EN_ROUTE_TO_SCENE:
                    if m.steps_remaining_to_scene is not None:
                        m.steps_remaining_to_scene = max(0, m.steps_remaining_to_scene - 1)
                        if m.steps_remaining_to_scene == 0:
                            unit.transition_status(UnitStatus.ON_SCENE, self.clock.step)
                            m.mission_phase = UnitMissionPhase.ON_SCENE
                            if m.incident_id:
                                inc = self.get_incident(m.incident_id)
                                if inc:
                                    inc.step_first_unit_on_scene = self.clock.step
                                    if not inc.units_on_scene:
                                        try:
                                            self.transition_incident(m.incident_id, IncidentPhase.UNIT_ON_SCENE)
                                        except ValueError:
                                            pass
                                    if unit.unit_id not in inc.units_on_scene:
                                        inc.units_on_scene.append(unit.unit_id)
                            self._emit(SystemEventType.UNIT_ARRIVED_SCENE, unit_id=unit.unit_id)
                elif m.mission_phase == UnitMissionPhase.EN_ROUTE_TO_HOSP:
                    if m.steps_remaining_to_hospital is not None:
                        m.steps_remaining_to_hospital = max(0, m.steps_remaining_to_hospital - 1)
                        if m.steps_remaining_to_hospital == 0:
                            unit.transition_status(UnitStatus.AT_HOSPITAL, self.clock.step)
                            m.mission_phase = UnitMissionPhase.AT_HOSPITAL
                            if m.destination_hospital_id:
                                hosp = self.get_hospital(m.destination_hospital_id)
                                if hosp:
                                    hosp.beds.admit_patient()
                            self._emit(SystemEventType.UNIT_ARRIVED_HOSPITAL, unit_id=unit.unit_id)
                elif m.mission_phase == UnitMissionPhase.RETURNING_TO_BASE:
                    if m.steps_remaining_to_base is not None:
                        m.steps_remaining_to_base = max(0, m.steps_remaining_to_base - 1)
                        if m.steps_remaining_to_base == 0:
                            self.complete_unit_return(unit.unit_id)
        for inc in list(self.incidents.values()):
            if not inc.is_terminal:
                inc.advance_time(steps=1)
        for hosp in self.hospitals.values():
            hosp.tick_diversion(self.clock.step)
        self.surge.update_system_metrics(
            hospital_states=list(self.hospitals.values()),
            fleet_states=list(self.units.values()),
            active_incidents=self.active_incidents,
        )
        if self.surge.cascade.is_critical:
            self._emit(
                SystemEventType.CASCADE_FAILURE_WARNING,
                payload={"risk": self.surge.cascade.risk_score},
            )
        if not self.surge.cascade.occurred and self.surge.cascade.is_imminent:
            self.surge.cascade.occurred = True
            self.surge.cascade.occurrence_step = self.clock.step
            self.stats.hospital_network.cascade_failure_occurred = True
            self._emit(SystemEventType.CASCADE_FAILURE_OCCURRED)
        for zone in self.zones.values():
            zone.update_alert_level(self.clock.step)
        if self.clock.step >= MAX_STEPS_PER_EPISODE:
            self.episode_done = True
            self.termination_reason = "max_steps_reached"
            self._emit(SystemEventType.EPISODE_ENDED)
        self.stats.steps_executed = self.clock.step
        self.stats.episode_duration_minutes = self.clock.episode_minute
        self.stats.final_episode_score = self.current_score
    def snapshot(self) -> StateCheckpoint:
        cp = StateCheckpoint(
            step=self.clock.step,
            episode_id=self.episode_id,
            task_id=self.task_id,
            incident_states_json=json.dumps(
                {iid: inc.model_dump(mode="json") for iid, inc in self.incidents.items()}
            ),
            unit_states_json=json.dumps(
                {uid: u.model_dump(mode="json") for uid, u in self.units.items()}
            ),
            hospital_states_json=json.dumps(
                {hid: h.model_dump(mode="json") for hid, h in self.hospitals.items()}
            ),
            zone_states_json=json.dumps(
                {zid: z.model_dump(mode="json") for zid, z in self.zones.items()}
            ),
            traffic_state_json=self.traffic.model_dump_json(),
            reward_accumulator_json=self.rewards.model_dump_json(),
            protocol_state_json=self.protocol.model_dump_json(),
            surge_state_json=self.surge.model_dump_json(),
        )
        cp.compute_hash()
        while len(self.checkpoints) >= MAX_CHECKPOINTS_PER_EPISODE:
            evicted = self.checkpoints.pop(0)
            del self.checkpoint_step_index[evicted.step]
        self.checkpoints.append(cp)
        self.checkpoint_step_index[cp.step] = cp.checkpoint_id
        self._emit(
            SystemEventType.CHECKPOINT_SAVED,
            payload={"checkpoint_id": cp.checkpoint_id, "step": cp.step},
        )
        logger.debug(
            "SimulationState: checkpoint saved at step %d (id=%s, ~%d bytes)",
            cp.step, cp.checkpoint_id, cp.size_bytes_estimate,
        )
        return cp
    def restore(self, checkpoint: StateCheckpoint) -> None:
        if not checkpoint.verify_integrity():
            raise ValueError(
                f"Checkpoint {checkpoint.checkpoint_id} failed integrity check."
            )
        raw_incidents = json.loads(checkpoint.incident_states_json)
        raw_units     = json.loads(checkpoint.unit_states_json)
        raw_hospitals = json.loads(checkpoint.hospital_states_json)
        raw_zones     = json.loads(checkpoint.zone_states_json)
        self.incidents  = {iid: IncidentState.model_validate(v)  for iid, v in raw_incidents.items()}
        self.units      = {uid: UnitState.model_validate(v)       for uid, v in raw_units.items()}
        self.hospitals  = {hid: HospitalState.model_validate(v)   for hid, v in raw_hospitals.items()}
        self.zones      = {zid: ZoneState.model_validate(v)       for zid, v in raw_zones.items()}
        self.traffic    = TrafficEngineState.model_validate_json(checkpoint.traffic_state_json)
        self.rewards    = EpisodeRewardAccumulator.model_validate_json(checkpoint.reward_accumulator_json)
        self.protocol   = ProtocolComplianceState.model_validate_json(checkpoint.protocol_state_json)
        self.surge      = SurgeCoordinationState.model_validate_json(checkpoint.surge_state_json)
        self.incident_queue = [
            iid for iid, inc in self.incidents.items()
            if not inc.is_terminal
        ]
        self._emit(
            SystemEventType.CHECKPOINT_RESTORED,
            payload={"checkpoint_id": checkpoint.checkpoint_id, "restored_to_step": checkpoint.step},
        )
        logger.info(
            "SimulationState: restored from checkpoint %s (step %d)",
            checkpoint.checkpoint_id, checkpoint.step,
        )
    def diff(self, other_checkpoint: StateCheckpoint) -> StateDiff:
        raw_other_incidents = json.loads(other_checkpoint.incident_states_json)
        raw_other_units     = json.loads(other_checkpoint.unit_states_json)
        raw_other_hospitals = json.loads(other_checkpoint.hospital_states_json)
        old_incidents = set(raw_other_incidents.keys())
        curr_incidents= set(self.incidents.keys())
        new_created   = list(curr_incidents - old_incidents)
        new_resolved  = [
            iid for iid in old_incidents
            if iid in self.incidents and self.incidents[iid].phase == IncidentPhase.RESOLVED
            and json.loads(raw_other_incidents.get(iid, "{}")).get("phase") != IncidentPhase.RESOLVED.value
        ]
        new_failed = [
            iid for iid in old_incidents
            if iid in self.incidents and self.incidents[iid].phase == IncidentPhase.FAILED
            and json.loads(raw_other_incidents.get(iid, "{}")).get("phase") != IncidentPhase.FAILED.value
        ]
        unit_changes: Dict[UnitID, Tuple[str, str]] = {}
        for uid, udata in raw_other_units.items():
            if uid in self.units:
                old_status = udata.get("status", "")
                new_status = self.units[uid].status.value if hasattr(self.units[uid].status, "value") else str(self.units[uid].status)
                if old_status != new_status:
                    unit_changes[uid] = (old_status, new_status)
        old_div = {hid for hid, hdata in raw_other_hospitals.items()
                   if hdata.get("diversion_state") in ("on_diversion", "surge_overflow")}
        curr_div = set(self.diverted_hospital_ids)
        new_diverted = list(curr_div - old_div)
        new_cleared  = list(old_div - curr_div)
        other_rewards = EpisodeRewardAccumulator.model_validate_json(checkpoint.reward_accumulator_json)
        return StateDiff(
            from_step=other_checkpoint.step,
            to_step=self.clock.step,
            episode_id=self.episode_id,
            incidents_created=new_created,
            incidents_resolved=new_resolved,
            incidents_failed=new_failed,
            units_status_changed=unit_changes,
            hospitals_diverted=new_diverted,
            hospitals_cleared=new_cleared,
            reward_delta=self.rewards.total_reward - other_rewards.total_reward,
            score_delta=self.current_score - other_rewards.weighted_score,
            events_count=self.event_log.total_events,
        )
    def finalise_episode(self) -> EpisodeStatistics:
        for inc in self.active_incidents:
            if not inc.is_terminal:
                try:
                    inc.transition_phase(IncidentPhase.FAILED, self.clock.step)
                    self.stats.incidents_failed += 1
                except ValueError:
                    pass
        self.stats.final_episode_score = self.current_score
        self.stats.baseline_score      = TaskID(self.task_id).baseline_score
        self.stats.score_vs_baseline   = self.stats.final_episode_score - self.stats.baseline_score
        self.stats.terminated_early    = self.termination_reason != "max_steps_reached"
        self.stats.fleet.total_missions_completed = sum(
            u.missions_completed for u in self.units.values()
        )
        self.stats.fleet.comms_failures_total = sum(
            u.comms.failure_count_this_episode for u in self.units.values()
        )
        self.stats.hospital_network.total_admissions = sum(
            h.beds.total_admissions_episode for h in self.hospitals.values()
        )
        self.stats.hospital_network.total_diversions_activated = sum(
            h.beds.total_diversions_episode for h in self.hospitals.values()
        )
        self.stats.hospital_network.surge_declared = self.surge.surge_declared
        self._emit(
            SystemEventType.EPISODE_ENDED,
            payload={
                "score": self.stats.final_episode_score,
                "baseline": self.stats.baseline_score,
                "steps": self.clock.step,
            },
        )
        logger.info(
            "Episode %s finalised | task=%s | score=%.4f (baseline=%.2f) | steps=%d",
            self.episode_id, self.task_id,
            self.stats.final_episode_score, self.stats.baseline_score,
            self.clock.step,
        )
        return self.stats
    def summary(self) -> str:
        return (
            f"[Step {self.clock.step}/{MAX_STEPS_PER_EPISODE}] "
            f"Task={self.task_id} | "
            f"Queue={len(self.incident_queue)} ({self.unassigned_p1_count} P1 unassigned) | "
            f"Fleet={self.fleet_available_count}/{len(self.units)} avail | "
            f"Stress={self.system_stress_level} | "
            f"Score={self.current_score:.4f} | "
            f"Cascade={self.surge.cascade.risk_score:.0%}"
        )
    def incident_summary_table(self) -> str:
        rows = ["incident_id          | phase              | sev | victims | elapsed_min"]
        rows.append("-" * 70)
        for inc in sorted(self.incidents.values(), key=lambda i: i.step_created):
            rows.append(
                f"{inc.incident_id[:20]:20s} | "
                f"{inc.phase.value:18s} | "
                f"{inc.severity:3s} | "
                f"{inc.victim_count:7d} | "
                f"{inc.minutes_elapsed:.1f}"
            )
        return "\n".join(rows)
    def fleet_summary_table(self) -> str:
        rows = ["unit_id       | type | status               | zone | fatigue | comms"]
        rows.append("-" * 70)
        for u in sorted(self.units.values(), key=lambda u: u.unit_id):
            fatigue = f"{u.crew_fatigue.hours_on_duty:.1f}h"
            comms = "OK" if u.comms_active else "LOST"
            rows.append(
                f"{u.unit_id:13s} | "
                f"{str(u.unit_type):4s} | "
                f"{str(u.status):20s} | "
                f"{u.current_zone:4s} | "
                f"{fatigue:7s} | "
                f"{comms}"
            )
        return "\n".join(rows)
    @model_validator(mode="after")
    def validate_state_consistency(self) -> "SimulationState":
        if self.schema_version != __state_version__:
            raise ValueError(
                f"SimulationState schema_version {self.schema_version} "
                f"!= current {__state_version__}."
            )
        if TaskID(self.task_id).seed != self.task_seed:
            raise ValueError(
                f"task_seed {self.task_seed} does not match "
                f"expected seed for {self.task_id}"
            )
        return self
def make_clock_state(task_id: TaskID, start_hour: int = 7, season: Season = Season.SUMMER) -> ClockState:
    task = TaskID(task_id)
    episode_id = new_episode_id()
    return ClockState(
        episode_id=episode_id,
        task_id=task_id,
        task_seed=task.seed,
        step=0,
        episode_minute=0.0,
        wall_clock_hour=start_hour,
        wall_clock_minute=0,
        day_of_week="monday",
        season=season,
        is_peak_morning=start_hour in PEAK_MORNING_HOURS,
        is_peak_evening=start_hour in PEAK_EVENING_HOURS,
    )
def make_demand_generator_state(task_id: TaskID) -> DemandGeneratorState:
    task = TaskID(task_id)
    return DemandGeneratorState(
        task_id=task_id,
        seed=task.seed,
    )
def make_simulation_state(
    task_id: TaskID,
    start_hour: int = 7,
    season: Season = Season.SUMMER,
) -> SimulationState:
    task = TaskID(task_id)
    episode_id = new_episode_id()
    clock = make_clock_state(task_id, start_hour, season)
    clock.episode_id = episode_id
    return SimulationState(
        episode_id=episode_id,
        task_id=task_id,
        task_seed=task.seed,
        clock=clock,
        demand_gen=make_demand_generator_state(task_id),
        event_log=EventLog(episode_id=episode_id),
        stats=EpisodeStatistics(
            episode_id=episode_id,
            task_id=task_id,
            task_difficulty=task.difficulty,
            seed=task.seed,
            baseline_score=task.baseline_score,
        ),
        rewards=EpisodeRewardAccumulator(),
        protocol=ProtocolComplianceState(),
        surge=SurgeCoordinationState(),
        traffic=TrafficEngineState(),
    )
ModelRegistry.register("PatientState",                PatientState)
ModelRegistry.register("IncidentDispatchRecord",      IncidentDispatchRecord)
ModelRegistry.register("IncidentState",               IncidentState)
ModelRegistry.register("UnitMissionState",            UnitMissionState)
ModelRegistry.register("UnitState",                   UnitState)
ModelRegistry.register("HospitalBedTracking",         HospitalBedTracking)
ModelRegistry.register("HospitalState",               HospitalState)
ModelRegistry.register("ZoneStagingPost",             ZoneStagingPost)
ModelRegistry.register("ZoneState",                   ZoneState)
ModelRegistry.register("TrafficSlowdownState",        TrafficSlowdownState)
ModelRegistry.register("TrafficEngineState",          TrafficEngineState)
ModelRegistry.register("RewardEntry",                 RewardEntry)
ModelRegistry.register("StepRewardBreakdown",         StepRewardBreakdown)
ModelRegistry.register("EpisodeRewardAccumulator",    EpisodeRewardAccumulator)
ModelRegistry.register("ProtocolEvent",               ProtocolEvent)
ModelRegistry.register("ProtocolComplianceState",     ProtocolComplianceState)
ModelRegistry.register("MutualAidZonePairing",        MutualAidZonePairing)
ModelRegistry.register("MutualAidCoordinationState",  MutualAidCoordinationState)
ModelRegistry.register("CascadeFailureTracker",       CascadeFailureTracker)
ModelRegistry.register("SurgeCoordinationState",      SurgeCoordinationState)
ModelRegistry.register("SystemEvent",                 SystemEvent)
ModelRegistry.register("EventLog",                    EventLog)
ModelRegistry.register("IncidentOutcome",             IncidentOutcome)
ModelRegistry.register("FleetStatistics",             FleetStatistics)
ModelRegistry.register("HospitalNetworkStatistics",   HospitalNetworkStatistics)
ModelRegistry.register("EpisodeStatistics",           EpisodeStatistics)
ModelRegistry.register("StateCheckpoint",             StateCheckpoint)
ModelRegistry.register("StateDiff",                   StateDiff)
ModelRegistry.register("DemandGeneratorState",        DemandGeneratorState)
ModelRegistry.register("ClockState",                  ClockState)
ModelRegistry.register("SimulationState",             SimulationState)
EpisodeState  = SimulationState
PatientRecord = PatientState
UnitRecord    = UnitState
logger.info(
    "state.py: registered %d state models  (state_version=%d, schema_version=%d)",
    33, __state_version__, __schema_version__,
)
__all__ = [
    "__state_version__",
    "EVENT_LOG_MAX_SIZE",
    "MAX_CHECKPOINTS_PER_EPISODE",
    "SCENE_ARRIVAL_PROCESSING_STEPS",
    "HOSPITAL_HANDOFF_PROCESSING_STEPS",
    "RETURN_BASE_PROCESSING_STEPS",
    "SURVIVAL_RECALC_EVERY_N_STEPS",
    "CASCADE_LEVEL_1_THRESHOLD",
    "CASCADE_LEVEL_2_THRESHOLD",
    "CASCADE_LEVEL_3_THRESHOLD",
    "COMMS_RESTORE_PROBABILITY_PER_STEP",
    "IncidentPhase",
    "UnitMissionPhase",
    "HospitalDiversionState",
    "ZoneAlertLevel",
    "SystemEventType",
    "PatientOutcome",
    "RewardChannel",
    "PatientState",
    "IncidentDispatchRecord",
    "EpisodeState", 
    "PatientRecord",
    "UnitRecord",
    "IncidentState",
    "UnitMissionState",
    "UnitState",
    "HospitalBedTracking",
    "HospitalState",
    "ZoneStagingPost",
    "ZoneState",
    "TrafficSlowdownState",
    "TrafficEngineState",
    "RewardEntry",
    "StepRewardBreakdown",
    "EpisodeRewardAccumulator",
    "ProtocolEvent",
    "ProtocolComplianceState",
    "MutualAidZonePairing",
    "MutualAidCoordinationState",
    "CascadeFailureTracker",
    "SurgeCoordinationState",
    "SystemEvent",
    "EventLog",
    "IncidentOutcome",
    "FleetStatistics",
    "HospitalNetworkStatistics",
    "EpisodeStatistics",
    "StateCheckpoint",
    "StateDiff",
    "DemandGeneratorState",
    "ClockState",
    "SimulationState",
    "make_clock_state",
    "make_demand_generator_state",
    "make_simulation_state",
]
def _self_test() -> None:
    assert IncidentPhase.RESOLVED.is_terminal
    assert not IncidentPhase.DISPATCHED.is_terminal
    assert IncidentPhase.DISPATCHED.is_active
    accumulator_fields = {
        f for f in EpisodeRewardAccumulator.model_fields
        if not f.startswith("_") and f not in (
            "step_breakdowns", "total_entries", "CHANNEL_WEIGHTS",
        )
    }
    for ch in RewardChannel:
        assert ch.value in EpisodeRewardAccumulator.CHANNEL_WEIGHTS, (
            f"RewardChannel.{ch.name} missing from CHANNEL_WEIGHTS"
        )
    p = SurvivalCurve.compute(
        DecayModel.EXPONENTIAL,
        30.0,
        {"survival_at_zero_min": 0.97, "survival_floor": 0.20, "decay_rate": 0.04},
    )
    assert 0.0 <= p <= 1.0, f"SurvivalCurve out of bounds: {p}"
    beds = HospitalBedTracking(
        er_beds_total=10, er_beds_occupied=5,
        icu_beds_total=4, icu_beds_occupied=2,
        ccu_beds_total=2, ccu_beds_occupied=0,
        trauma_bays_total=2, trauma_bays_occupied=0,
        ventilators_total=6, ventilators_occupied=1,
    )
    assert beds.admit_patient()
    assert beds.er_beds_occupied == 6
    beds.discharge_patient()
    assert beds.er_beds_occupied == 5
    snap = beds.to_capacity_snapshot()
    assert 0.0 <= snap.er_occupancy_pct <= 1.0
    clock = make_clock_state(TaskID.T1, start_hour=8)
    assert clock.is_peak_morning
    clock.advance(steps=10)
    assert clock.step == 10
    assert clock.episode_minute == 30.0
    acc = EpisodeRewardAccumulator()
    acc.add_direct(RewardChannel.SURVIVAL, 0.25, step=1, description="test")
    assert abs(acc.survival - 0.25) < 1e-6
    assert acc.total_entries == 1
    pcs = ProtocolComplianceState()
    delta = pcs.record_event(ProtocolRule.MICU_FOR_STEMI, True, step=5)
    assert delta > 0.0
    assert pcs.total_correct == 1
    assert pcs.micu_stemi_correct_rate == 1.0
    cft = CascadeFailureTracker()
    cft.recompute(
        hospital_diversion_count=6,
        fleet_unavailable_pct=0.8,
        er_occupancy_avg=0.92,
        simultaneous_mci_count=4,
        mutual_aid_saturated=True,
        step=10,
    )
    assert cft.risk_score > 0.5
    assert cft.is_critical
    inc = IncidentState(
        incident_id="INC-0001",
        template_id="T-STEMI-01",
        call_number="108-001",
        condition="stemi_anterior",
        incident_type=IncidentType.STEMI,
        severity=SeverityLevel.P1,
        symptom_description="A" * 25,
        protocol_notes="B" * 15,
        zone_id="Z1",
        location=GeoCoordinate(lat=19.0, lon=72.8),
        step_created=0,
        golden_hour_minutes=60,
        target_response_minutes=8,
    )
    inc.transition_phase(IncidentPhase.DISPATCHED, step=2)
    assert inc.phase == IncidentPhase.DISPATCHED
    try:
        inc.transition_phase(IncidentPhase.RESOLVED, step=3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    state = make_simulation_state(TaskID.T4)
    assert state.clock.step == 0
    assert state.episode_done is False
    assert len(state.incidents) == 0
    log = EventLog(episode_id="ep-test", max_size=5)
    for i in range(8):
        log.append(SystemEvent(
            event_type=SystemEventType.INCIDENT_CREATED,
            step=i,
            episode_id="ep-test",
        ))
    assert len(log.events) <= log.max_size
    logger.debug("state.py self-test passed.")
_self_test()
logger.info(
    "EMERGI-ENV server.models.state v%d loaded — %d models registered.",
    __state_version__,
    33,
)