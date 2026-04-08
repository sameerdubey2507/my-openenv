from __future__ import annotations

import logging
import math
import uuid
from typing import (
    Any,
    ClassVar,
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
    GeoCoordinate,
    MutualAidRequest,
    COMMS_FAILURE_PROBABILITY_PER_STEP,
    CREW_FATIGUE_THRESHOLD_HOURS,
    CREW_SWAP_DEPLOY_DELAY_MIN,
    DIVERSION_PENALTY,
    DIVERSION_REDIRECT_DELAY_MIN,
    DIVERSION_THRESHOLD_PCT,
    FLEET_SIZE_DEFAULT,
    MAX_STEPS_PER_EPISODE,
    MUTUAL_AID_DELAY_MIN,
    MUTUAL_AID_OVER_REQUEST_PENALTY,
    NUM_HOSPITALS,
    NUM_ZONES,
    PROTOCOL_COMPLIANCE_MAX_BONUS,
    SCORE_CEILING,
    SCORE_FLOOR,
    TIMESTEP_MINUTES,
    WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY,
    HospitalSpecialty,
    CONDITION_REQUIRED_UNIT,
    CONDITION_REQUIRED_SPECIALTY,
    RPMScore,
    clamp_score,
    utc_now_iso,
    validate_hospital_id,
    validate_unit_id,
    validate_zone_id,
    __schema_version__,
)

logger = logging.getLogger("emergi_env.models.action")

MAX_UNITS_PER_DISPATCH: int = 6
MAX_ACTIONS_PER_BATCH: int = FLEET_SIZE_DEFAULT
MAX_MUTUAL_AID_UNITS_PER_REQUEST: int = 8
MAX_TRANSFER_PATIENTS: int = 4
MAX_PREPOSITION_ZONES: int = FLEET_SIZE_DEFAULT
MAX_TAG_VICTIMS_PER_ACTION: int = 50

NOOP_ON_COMMS_LOST_PENALTY: float = -0.05
MULTI_AGENCY_OMISSION_PENALTY: float = -0.025
WRONG_UNIT_DISPATCH_PENALTY: float = -0.20
WRONG_HOSPITAL_DISPATCH_PENALTY: float = -0.25

DISPATCH_COST_BLS: int = 1
DISPATCH_COST_ALS: int = 2
DISPATCH_COST_MICU: int = 4

REROUTE_IMPROVEMENT_BONUS: float = 0.05
REROUTE_NO_IMPROVEMENT_PENALTY: float = -0.02

ESCALATION_VALID_TRANSITIONS: FrozenSet[Tuple[str, str]] = frozenset({
    (SeverityLevel.P3.value, SeverityLevel.P2.value),
    (SeverityLevel.P3.value, SeverityLevel.P1.value),
    (SeverityLevel.P2.value, SeverityLevel.P1.value),
})

TRANSFER_GOLDEN_WINDOW_MINUTES: float = 30.0
TRANSFER_LATE_PENALTY: float = -0.10

PREPOSITION_DEMAND_MATCH_BONUS: float = 0.04
PREPOSITION_WASTEFUL_PENALTY: float = -0.02

SURGE_DECLARATION_BONUS: float = 0.08
SURGE_LATE_DECLARATION_PENALTY: float = -0.12
SURGE_CASCADE_AVOIDANCE_BONUS: float = 0.20

HOSPITAL_BYPASS_CORRECT_BONUS: float = 0.03
HOSPITAL_BYPASS_UNNECESSARY_PENALTY: float = -0.04

CREW_SWAP_CORRECT_BONUS: float = 0.02
CREW_SWAP_PREMATURE_PENALTY: float = -0.02

class BaseAction(EmergiBaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType = Field(...)
    step: StepNumber = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    task_id: TaskID = Field(...)
    reasoning: Optional[str] = Field(default=None, max_length=500)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    submitted_at: str = Field(default_factory=utc_now_iso)

    @computed_field
    @property
    def action_cost(self) -> int:
        return 0

    @computed_field
    @property
    def is_intervention(self) -> bool:
        return self.action_type != ActionType.NOOP

    @computed_field
    @property
    def difficulty_tier(self) -> TaskDifficulty:
        return TaskID(self.task_id).difficulty

    @computed_field
    @property
    def action_label(self) -> str:
        return f"[{self.action_type.upper()}] step={self.step} task={self.task_id}"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseAction":
        return cls.model_validate(data)

    @classmethod
    def schema_version(cls) -> int:
        return __schema_version__

    def validate_step_consistency(self, current_step: int) -> bool:
        return self.step == current_step

class DispatchRouting(ImmutableEmergiModel):
    record_type: Literal["dispatch_routing"] = "dispatch_routing"
    destination_hospital_id: HospitalID = Field(...)
    destination_zone_id: ZoneID = Field(...)
    estimated_travel_time_minutes: float = Field(..., ge=0.0)
    via_expressway: bool = Field(default=False)
    via_green_corridor: bool = Field(default=False)
    ems_corridor_id: Optional[str] = Field(default=None)
    bypass_bottleneck_ids: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def routing_strategy(self) -> str:
        if self.via_green_corridor: return "green_corridor"
        if self.ems_corridor_id: return "ems_corridor"
        if self.via_expressway: return "expressway"
        return "standard"

    @computed_field
    @property
    def expected_effective_time_minutes(self) -> float:
        factors = {"green_corridor": 0.55, "ems_corridor": 0.70, "expressway": 0.80, "standard": 1.00}
        return self.estimated_travel_time_minutes * factors[self.routing_strategy]

class DispatchAction(BaseAction):
    action_type: Literal[ActionType.DISPATCH] = ActionType.DISPATCH
    incident_id: IncidentID = Field(...)
    incident_zone_id: ZoneID = Field(...)
    declared_severity: SeverityLevel = Field(...)
    declared_start_tag: TriageTag = Field(...)
    unit_ids: List[UnitID] = Field(..., min_length=1, max_length=MAX_UNITS_PER_DISPATCH)
    primary_unit_type: UnitType = Field(...)
    routing: DispatchRouting = Field(...)
    multi_agency_coordinated: bool = Field(default=False)
    agencies_requested: List[str] = Field(default_factory=list)
    cath_lab_activated: bool = Field(default=False)
    stroke_unit_notified: bool = Field(default=False)
    trauma_activation_sent: bool = Field(default=False)
    treat_in_place: bool = Field(default=False)
    condition_key: Optional[str] = Field(default=None)
    victim_index: Optional[int] = Field(default=None, ge=0)
    estimated_response_time_minutes: float = Field(default=0.0, ge=0.0)

    @field_validator("unit_ids")
    @classmethod
    def validate_unit_ids(cls, v: List[UnitID]) -> List[UnitID]:
        for uid in v:
            if not validate_unit_id(uid): raise ValueError(f"Invalid unit_id format: '{uid}'.")
        if len(v) != len(set(v)): raise ValueError("unit_ids must not contain duplicates.")
        return v

    @field_validator("incident_zone_id")
    @classmethod
    def validate_incident_zone(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError(f"Invalid incident_zone_id: '{v}'.")
        return v

    @model_validator(mode="after")
    def validate_dispatch_consistency(self) -> "DispatchAction":
        expected_tag = SeverityLevel(self.declared_severity).start_tag
        if self.declared_start_tag != expected_tag.value and self.declared_start_tag != expected_tag:
            logger.warning("DispatchAction %s: inconsistent triage tag.", self.action_id)
        if self.treat_in_place and self.declared_severity not in (SeverityLevel.P0.value, SeverityLevel.P0):
            raise ValueError("treat_in_place=True only for P0.")
        if self.primary_unit_type == UnitType.MICU and len(self.unit_ids) > 3:
            raise ValueError("Max 3 MICUs per incident.")
        if not validate_hospital_id(self.routing.destination_hospital_id):
            raise ValueError(f"Invalid destination_hospital_id: '{self.routing.destination_hospital_id}'.")
        if not validate_zone_id(self.routing.destination_zone_id):
            raise ValueError(f"Invalid destination_zone_id: '{self.routing.destination_zone_id}'.")
        return self

    @computed_field
    @property
    def action_cost(self) -> int:
        cost_map = {UnitType.BLS.value: DISPATCH_COST_BLS, UnitType.ALS.value: DISPATCH_COST_ALS, UnitType.MICU.value: DISPATCH_COST_MICU}
        return sum(cost_map.get(uid.split("-")[0], 1) for uid in self.unit_ids)

    @computed_field
    @property
    def primary_unit_id(self) -> UnitID: return self.unit_ids[0]

    @computed_field
    @property
    def is_micu_dispatch(self) -> bool: return self.primary_unit_type == UnitType.MICU

    @computed_field
    @property
    def is_mci_dispatch(self) -> bool: return len(self.unit_ids) > 1

    @computed_field
    @property
    def protocol_hint_correct_unit(self) -> Optional[bool]:
        if self.condition_key is None: return None
        required = CONDITION_REQUIRED_UNIT.get(self.condition_key)
        return UnitType(self.primary_unit_type) == required if required else None

    @computed_field
    @property
    def protocol_hint_correct_hospital_specialty(self) -> Optional[str]:
        return CONDITION_REQUIRED_SPECIALTY.get(self.condition_key) if self.condition_key else None

    @computed_field
    @property
    def pre_notification_bonus_eligible(self) -> bool:
        return self.cath_lab_activated or self.stroke_unit_notified or self.trauma_activation_sent

    @computed_field
    @property
    def estimated_golden_hour_compliance(self) -> bool:
        target = SeverityLevel(self.declared_severity).target_response_minutes
        return self.estimated_response_time_minutes <= target

    @computed_field
    @property
    def action_label(self) -> str:
        units_str = ", ".join(self.unit_ids)
        return f"[DISPATCH] {self.incident_id} → units=[{units_str}] type={self.primary_unit_type} hospital={self.routing.destination_hospital_id} severity={self.declared_severity} tag={self.declared_start_tag}"

    @computed_field
    @property
    def dispatch_summary(self) -> str:
        p_str = ""
        if self.cath_lab_activated: p_str += " [CATH_LAB_ACTIVATED]"
        if self.stroke_unit_notified: p_str += " [STROKE_NOTIFIED]"
        if self.trauma_activation_sent: p_str += " [TRAUMA_ACTIVATION]"
        if self.multi_agency_coordinated: p_str += " [MULTI_AGENCY]"
        return f"DISPATCH incident={self.incident_id} zone={self.incident_zone_id} primary={self.primary_unit_id} → H{self.routing.destination_hospital_id} via {self.routing.routing_strategy} ETA≈{self.estimated_response_time_minutes:.1f}min severity={self.declared_severity}{p_str}"

class RerouteReason(str):
    TRAFFIC_UPDATE = "traffic_update"
    HOSPITAL_DIVERSION = "hospital_diversion"
    ROUTE_CLOSURE = "route_closure"
    BETTER_SPECIALTY = "better_specialty"
    CAPACITY_SHORTAGE = "capacity_shortage"
    SCENE_CONDITION = "scene_condition"
    COMMS_RESTORATION = "comms_restoration"
    MCI_REDISTRIBUTION = "mci_redistribution"

class RerouteAction(BaseAction):
    action_type: Literal[ActionType.REROUTE] = ActionType.REROUTE
    unit_id: UnitID = Field(...)
    new_hospital_id: HospitalID = Field(...)
    new_hospital_zone_id: ZoneID = Field(...)
    new_estimated_travel_time_minutes: float = Field(..., ge=0.0)
    old_hospital_id: HospitalID = Field(...)
    old_estimated_travel_time_minutes: float = Field(..., ge=0.0)
    reroute_reason: str = Field(...)
    triggered_by_traffic_step: Optional[int] = Field(default=None, ge=0)
    triggered_by_diversion_hospital_id: Optional[HospitalID] = Field(default=None)
    via_green_corridor: bool = Field(default=False)
    via_ems_corridor_id: Optional[str] = Field(default=None)
    bypass_bottleneck_ids: List[str] = Field(default_factory=list)
    patient_on_board: bool = Field(default=False)
    patient_severity: Optional[str] = Field(default=None)
    patient_condition_key: Optional[str] = Field(default=None)

    @field_validator("unit_id")
    @classmethod
    def validate_unit(cls, v: UnitID) -> UnitID:
        if not validate_unit_id(v): raise ValueError(f"Invalid unit_id: '{v}'.")
        return v

    @field_validator("new_hospital_id", "old_hospital_id")
    @classmethod
    def validate_hospital(cls, v: HospitalID) -> HospitalID:
        if not validate_hospital_id(v): raise ValueError(f"Invalid hospital_id: '{v}'.")
        return v

    @field_validator("new_hospital_zone_id")
    @classmethod
    def validate_zone(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError(f"Invalid zone_id: '{v}'.")
        return v

    @model_validator(mode="after")
    def validate_reroute_logic(self) -> "RerouteAction":
        if self.new_hospital_id == self.old_hospital_id: raise ValueError("Destination must change.")
        return self

    @computed_field
    @property
    def time_saved_estimate_minutes(self) -> float: return self.old_estimated_travel_time_minutes - self.new_estimated_travel_time_minutes

    @computed_field
    @property
    def is_improvement(self) -> bool: return self.time_saved_estimate_minutes > 0.0

    @computed_field
    @property
    def is_diversion_triggered(self) -> bool: return self.reroute_reason == RerouteReason.HOSPITAL_DIVERSION

    @computed_field
    @property
    def action_cost(self) -> int: return 1

    @computed_field
    @property
    def action_label(self) -> str:
        d = "▲" if self.is_improvement else "▼"
        return f"[REROUTE] unit={self.unit_id} {self.old_hospital_id}→{self.new_hospital_id} Δt={self.time_saved_estimate_minutes:+.1f}min {d} reason={self.reroute_reason}"

class EscalateAction(BaseAction):                      
    action_type: Literal[ActionType.ESCALATE] = ActionType.ESCALATE
    incident_id: IncidentID = Field(...)
    current_severity: SeverityLevel = Field(...)
    new_severity: SeverityLevel = Field(...)
    escalation_reason: str = Field(..., min_length=5, max_length=300)
    new_victim_count: Optional[int] = Field(default=None, ge=1)
    additional_units_needed: Optional[List[UnitID]] = Field(default=None)
    new_agency_requirements: Optional[List[str]] = Field(default=None)
    declare_mci: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_escalation_direction(self) -> "EscalateAction":
        order = {SeverityLevel.P0: 0, SeverityLevel.P3: 1, SeverityLevel.P2: 2, SeverityLevel.P1: 3}
        c = order.get(SeverityLevel(self.current_severity), 0)
        n = order.get(SeverityLevel(self.new_severity), 0)
        if n <= c: raise ValueError("Escalation must be upward.")
        if self.declare_mci and (self.new_victim_count is None or self.new_victim_count < 2):
            raise ValueError("declare_mci requires victims >= 2.")
        return self

    @field_validator("incident_id")
    @classmethod
    def validate_incident(cls, v: IncidentID) -> IncidentID:
        if not v or len(v) < 4: raise ValueError("Invalid incident_id.")
        return v

    @computed_field
    @property
    def severity_jump_magnitude(self) -> int:
        order = {SeverityLevel.P3.value: 1, SeverityLevel.P2.value: 2, SeverityLevel.P1.value: 3}
        return order.get(SeverityLevel(self.new_severity).value, 0) - order.get(SeverityLevel(self.current_severity).value, 0)

    @computed_field
    @property
    def is_double_jump(self) -> bool: return self.severity_jump_magnitude == 2

    @computed_field
    @property
    def is_mci_upgrade(self) -> bool: return self.declare_mci or (self.new_victim_count is not None and self.new_victim_count > 1)

    @computed_field
    @property
    def action_cost(self) -> int: return self.severity_jump_magnitude

    @computed_field
    @property
    def action_label(self) -> str:
        m = " [MCI DECLARED]" if self.declare_mci else ""
        return f"[ESCALATE] {self.incident_id} {self.current_severity}→{self.new_severity}{m}"

class RequestMutualAidAction(BaseAction):
    action_type: Literal[ActionType.REQUEST_MUTUAL_AID] = ActionType.REQUEST_MUTUAL_AID
    target_zone_id: str = Field(...)
    num_units: int = Field(..., ge=1, le=5)
    unit_type_priority: str = Field(default="BLS")

class InterHospitalTransferAction(BaseAction):
    action_type: Literal[ActionType.TRANSFER] = ActionType.TRANSFER
    incident_id: str = Field(...)
    from_hospital_id: str = Field(...)
    to_hospital_id: str = Field(...)
    reason: str = Field(...)

class StartTagAction(BaseAction):
    action_type: Literal[ActionType.TAG] = ActionType.TAG
    incident_id: str = Field(...)
    patient_index: int = Field(..., ge=0)
    tag: Literal["Immediate", "Delayed", "Minimal", "Expectant"] = Field(...)

class VictimTriageAssignment(EmergiBaseModel):
    victim_index: int = Field(..., ge=0, le=MAX_TAG_VICTIMS_PER_ACTION - 1)
    assigned_tag: TriageTag = Field(...)
    rpm_respirations: str = Field(...)
    rpm_pulse: str = Field(...)
    rpm_mental_status: str = Field(...)
    agent_reasoning: Optional[str] = Field(default=None, max_length=200)
    unit_requested: Optional[UnitType] = Field(default=None)

    @computed_field
    @property
    def rpm_derived_tag(self) -> TriageTag: return TriageTag.from_rpm(self.rpm_respirations, self.rpm_pulse, self.rpm_mental_status)

    @computed_field
    @property
    def tag_is_correct(self) -> bool: return TriageTag(self.assigned_tag) == self.rpm_derived_tag

    @computed_field
    @property
    def is_critical_mismatch(self) -> bool: return self.rpm_derived_tag == TriageTag.IMMEDIATE and TriageTag(self.assigned_tag) == TriageTag.EXPECTANT

    @computed_field
    @property
    def survival_modifier(self) -> float: return RPMScore.compute_modifier(self.rpm_respirations, self.rpm_pulse, self.rpm_mental_status)

class TriageTagAction(BaseAction):
    action_type: Literal[ActionType.TAG] = ActionType.TAG
    incident_id: IncidentID = Field(...)
    incident_zone_id: ZoneID = Field(...)
    triage_assignments: List[VictimTriageAssignment] = Field(..., min_length=1, max_length=MAX_TAG_VICTIMS_PER_ACTION)
    start_protocol_applied: bool = Field(default=True)
    rpm_data_source: str = Field(default="observation")
    command_post_established: bool = Field(default=False)

    @field_validator("incident_zone_id")
    @classmethod
    def validate_zone(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError("Invalid zone_id.")
        return v

    @model_validator(mode="after")
    def validate_no_duplicate_victims(self) -> "TriageTagAction":
        idx = [a.victim_index for a in self.triage_assignments]
        if len(idx) != len(set(idx)): raise ValueError("Duplicate victim_index.")
        return self

    @computed_field
    @property
    def victim_count_tagged(self) -> int: return len(self.triage_assignments)

    @computed_field
    @property
    def immediate_count(self) -> int: return sum(1 for a in self.triage_assignments if TriageTag(a.assigned_tag) == TriageTag.IMMEDIATE)

    @computed_field
    @property
    def delayed_count(self) -> int: return sum(1 for a in self.triage_assignments if TriageTag(a.assigned_tag) == TriageTag.DELAYED)

    @computed_field
    @property
    def minimal_count(self) -> int: return sum(1 for a in self.triage_assignments if TriageTag(a.assigned_tag) == TriageTag.MINIMAL)

    @computed_field
    @property
    def expectant_count(self) -> int: return sum(1 for a in self.triage_assignments if TriageTag(a.assigned_tag) == TriageTag.EXPECTANT)

    @computed_field
    @property
    def critical_mismatches(self) -> int: return sum(1 for a in self.triage_assignments if a.is_critical_mismatch)

    @computed_field
    @property
    def critical_mismatch_penalty(self) -> float: return self.critical_mismatches * WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY

    @computed_field
    @property
    def correct_tag_count(self) -> int: return sum(1 for a in self.triage_assignments if a.tag_is_correct)

    @computed_field
    @property
    def tag_accuracy(self) -> float: return clamp_score(self.correct_tag_count / len(self.triage_assignments)) if self.triage_assignments else 0.0

    @computed_field
    @property
    def immediate_victim_indices(self) -> List[int]: return [a.victim_index for a in self.triage_assignments if TriageTag(a.assigned_tag) == TriageTag.IMMEDIATE]

    @computed_field
    @property
    def expectant_victim_indices(self) -> List[int]: return [a.victim_index for a in self.triage_assignments if TriageTag(a.assigned_tag) == TriageTag.EXPECTANT]

    @computed_field
    @property
    def action_cost(self) -> int: return self.victim_count_tagged

    @computed_field
    @property
    def action_label(self) -> str:
        return f"[TAG] incident={self.incident_id} victims={self.victim_count_tagged} [I={self.immediate_count} D={self.delayed_count} M={self.minimal_count} E={self.expectant_count}] accuracy≈{self.tag_accuracy:.0%}"

class TransferIndication(EmergiBaseModel):
    indication_code: str = Field(...)
    clinical_urgency: SeverityLevel = Field(...)
    current_hospital_limitation: str = Field(..., max_length=200)
    destination_hospital_capability_required: str = Field(...)

    @computed_field
    @property
    def is_urgent(self) -> bool: return SeverityLevel(self.clinical_urgency) == SeverityLevel.P1

    @computed_field
    @property
    def within_golden_window(self) -> bool: return self.is_urgent

class TransferAction(BaseAction):
    action_type: Literal[ActionType.TRANSFER] = ActionType.TRANSFER
    patient_ids: List[str] = Field(..., min_length=1, max_length=MAX_TRANSFER_PATIENTS)
    incident_id: IncidentID = Field(...)
    sending_hospital_id: HospitalID = Field(...)
    receiving_hospital_id: HospitalID = Field(...)
    receiving_hospital_zone_id: ZoneID = Field(...)
    indication: TransferIndication = Field(...)
    transport_unit_id: UnitID = Field(...)
    transport_unit_type: UnitType = Field(...)
    estimated_transfer_time_minutes: float = Field(..., ge=0.0)
    critical_care_nurse_assigned: bool = Field(default=False)
    consultant_accompanies: bool = Field(default=False)
    receiving_team_notified: bool = Field(default=False)
    bed_confirmed_at_receiving: bool = Field(default=False)
    sending_icu_occupancy_after: float = Field(default=0.0, ge=0.0, le=1.0)
    receiving_icu_occupancy_after: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("sending_hospital_id", "receiving_hospital_id")
    @classmethod
    def validate_hospitals(cls, v: HospitalID) -> HospitalID:
        if not validate_hospital_id(v): raise ValueError("Invalid hospital_id.")
        return v

    @field_validator("transport_unit_id")
    @classmethod
    def validate_transport_unit(cls, v: UnitID) -> UnitID:
        if not validate_unit_id(v): raise ValueError("Invalid transport_unit_id.")
        return v

    @field_validator("receiving_hospital_zone_id")
    @classmethod
    def validate_zone(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError("Invalid zone_id.")
        return v

    @model_validator(mode="after")
    def validate_transfer_logic(self) -> "TransferAction":
        if self.sending_hospital_id == self.receiving_hospital_id: raise ValueError("Hospitals must differ.")
        if len(self.patient_ids) != len(set(self.patient_ids)): raise ValueError("Duplicate patient_ids.")
        return self

    @computed_field
    @property
    def is_within_timing_window(self) -> bool:
        t = TRANSFER_GOLDEN_WINDOW_MINUTES if self.indication.is_urgent else 90.0
        return self.estimated_transfer_time_minutes <= t

    @computed_field
    @property
    def timing_score_estimate(self) -> float:
        t = TRANSFER_GOLDEN_WINDOW_MINUTES if self.indication.is_urgent else 90.0
        return clamp_score(1.0 - self.estimated_transfer_time_minutes / t)

    @computed_field
    @property
    def pre_notification_complete(self) -> bool: return self.receiving_team_notified and self.bed_confirmed_at_receiving

    @computed_field
    @property
    def action_cost(self) -> int: return len(self.patient_ids) * UnitType(self.transport_unit_type).dispatch_cost

    @computed_field
    @property
    def patient_count(self) -> int: return len(self.patient_ids)

    @computed_field
    @property
    def action_label(self) -> str:
        u = "URGENT" if self.indication.is_urgent else "SEMI-URGENT"
        return f"[TRANSFER] {self.sending_hospital_id}→{self.receiving_hospital_id} patients={self.patient_count} ({u}) unit={self.transport_unit_id}"

class MutualAidRequestAction(BaseAction):
    action_type: Literal[ActionType.REQUEST_MUTUAL_AID] = ActionType.REQUEST_MUTUAL_AID
    requesting_zone: ZoneID = Field(...)
    providing_zone: ZoneID = Field(...)
    unit_type: UnitType = Field(...)
    units_requested: int = Field(..., ge=1, le=MAX_MUTUAL_AID_UNITS_PER_REQUEST)
    justification: str = Field(..., min_length=10, max_length=300)
    expected_deployment_steps: int = Field(default=4, ge=1, le=20)
    priority: str = Field(default="normal")
    own_fleet_available_count: int = Field(..., ge=0)
    incident_ids_affected: List[IncidentID] = Field(default_factory=list)

    @field_validator("requesting_zone", "providing_zone")
    @classmethod
    def validate_zones(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError("Invalid zone_id.")
        return v

    @model_validator(mode="after")
    def validate_mutual_aid_logic(self) -> "MutualAidRequestAction":
        if self.requesting_zone == self.providing_zone: raise ValueError("Zones must differ.")
        if self.priority not in ("normal", "urgent", "surge"): raise ValueError("Invalid priority.")
        if self.expected_deployment_steps < 4: raise ValueError("Min deployment 4 steps.")
        return self

    @computed_field
    @property
    def deployment_delay_minutes(self) -> float: return self.expected_deployment_steps * TIMESTEP_MINUTES

    @computed_field
    @property
    def is_likely_over_request(self) -> bool: return self.own_fleet_available_count > 0

    @computed_field
    @property
    def estimated_over_request_penalty(self) -> float: return MUTUAL_AID_OVER_REQUEST_PENALTY if self.is_likely_over_request else 0.0

    @computed_field
    @property
    def total_unit_cost(self) -> int: return self.units_requested * UnitType(self.unit_type).dispatch_cost

    @computed_field
    @property
    def action_cost(self) -> int: return self.total_unit_cost

    @computed_field
    @property
    def action_label(self) -> str: return f"[MUTUAL_AID] {self.requesting_zone}←{self.providing_zone} {self.units_requested}×{self.unit_type} delay={self.deployment_delay_minutes:.0f}min"

    def to_mutual_aid_request(self) -> MutualAidRequest:
        return MutualAidRequest(requesting_zone=self.requesting_zone, providing_zone=self.providing_zone, unit_type=self.unit_type, units_requested=self.units_requested, eta_steps=self.expected_deployment_steps, step_requested=self.step)

class PrepositionAssignment(ImmutableEmergiModel):
    unit_id: UnitID = Field(...)
    target_zone_id: ZoneID = Field(...)
    target_staging_post_id: Optional[str] = Field(default=None)
    demand_forecast_basis: Optional[str] = Field(default=None)
    expected_demand_index: float = Field(default=0.0, ge=0.0)

    @field_validator("unit_id")
    @classmethod
    def validate_uid(cls, v: UnitID) -> UnitID:
        if not validate_unit_id(v): raise ValueError("Invalid unit_id.")
        return v

    @field_validator("target_zone_id")
    @classmethod
    def validate_zone(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError("Invalid zone_id.")
        return v

    @computed_field
    @property
    def forecast_informed(self) -> bool: return self.demand_forecast_basis is not None and self.expected_demand_index > 0.0

class PrepositionAction(BaseAction):
    action_type: Literal[ActionType.PREPOSITION] = ActionType.PREPOSITION
    assignments: List[PrepositionAssignment] = Field(..., min_length=1, max_length=MAX_PREPOSITION_ZONES)
    forecast_step_used: int = Field(..., ge=0)
    forecast_horizon: str = Field(default="06_09h")
    coverage_strategy: str = Field(default="demand_weighted")
    zones_targeted: List[ZoneID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_preposition_assignments(self) -> "PrepositionAction":
        u = [a.unit_id for a in self.assignments]
        if len(u) != len(set(u)): raise ValueError("Duplicate unit_ids.")
        if self.forecast_horizon not in ("00_03h", "03_06h", "06_09h", "09_12h"): raise ValueError("Invalid horizon.")
        return self

    @computed_field
    @property
    def units_repositioned(self) -> int: return len(self.assignments)

    @computed_field
    @property
    def zones_covered(self) -> int: return len({a.target_zone_id for a in self.assignments})

    @computed_field
    @property
    def forecast_informed_assignments(self) -> int: return sum(1 for a in self.assignments if a.forecast_informed)

    @computed_field
    @property
    def forecast_informed_ratio(self) -> float: return self.forecast_informed_assignments / len(self.assignments) if self.assignments else 0.0

    @computed_field
    @property
    def micu_repositioned(self) -> int: return sum(1 for a in self.assignments if a.unit_id.startswith(UnitType.MICU.value))

    @computed_field
    @property
    def als_repositioned(self) -> int: return sum(1 for a in self.assignments if a.unit_id.startswith(UnitType.ALS.value))

    @computed_field
    @property
    def bls_repositioned(self) -> int: return sum(1 for a in self.assignments if a.unit_id.startswith(UnitType.BLS.value))

    @computed_field
    @property
    def action_cost(self) -> int: return self.units_repositioned

    @computed_field
    @property
    def action_label(self) -> str: return f"[PREPOSITION] {self.units_repositioned} units → {self.zones_covered} zones horizon={self.forecast_horizon}"

class CrewSwapAction(BaseAction):
    action_type: Literal[ActionType.CREW_SWAP] = ActionType.CREW_SWAP
    unit_id: UnitID = Field(...)
    current_hours_on_duty: float = Field(..., ge=0.0)
    current_missions_completed: int = Field(..., ge=0)
    swap_location_zone_id: Optional[ZoneID] = Field(default=None)
    return_to_staging_post: Optional[str] = Field(default=None)
    urgent: bool = Field(default=False)

    @field_validator("unit_id")
    @classmethod
    def validate_unit(cls, v: UnitID) -> UnitID:
        if not validate_unit_id(v): raise ValueError("Invalid unit_id.")
        return v

    @field_validator("swap_location_zone_id")
    @classmethod
    def validate_zone(cls, v: Optional[ZoneID]) -> Optional[ZoneID]:
        if v and not validate_zone_id(v): raise ValueError("Invalid zone_id.")
        return v

    @model_validator(mode="after")
    def validate_swap_logic(self) -> "CrewSwapAction":
        return self

    @computed_field
    @property
    def crew_is_fatigued(self) -> bool: return self.current_hours_on_duty >= CREW_FATIGUE_THRESHOLD_HOURS

    @computed_field
    @property
    def swap_is_justified(self) -> bool: return self.current_hours_on_duty >= 8.0

    @computed_field
    @property
    def hours_over_threshold(self) -> float: return max(0.0, self.current_hours_on_duty - CREW_FATIGUE_THRESHOLD_HOURS)

    @computed_field
    @property
    def estimated_swap_delay_steps(self) -> int: return math.ceil(CREW_SWAP_DEPLOY_DELAY_MIN / TIMESTEP_MINUTES)

    @computed_field
    @property
    def estimated_swap_delay_minutes(self) -> float: return float(CREW_SWAP_DEPLOY_DELAY_MIN)

    @computed_field
    @property
    def protocol_compliance_estimate(self) -> float: return CREW_SWAP_CORRECT_BONUS if self.swap_is_justified else CREW_SWAP_PREMATURE_PENALTY

    @computed_field
    @property
    def action_cost(self) -> int: return 1

    @computed_field
    @property
    def action_label(self) -> str:
        s = "FATIGUED" if self.crew_is_fatigued else "FRESH"
        return f"[CREW_SWAP] unit={self.unit_id} hours={self.current_hours_on_duty:.1f}h ({s}) missions={self.current_missions_completed}"

class SurgeDeclarationScope(ImmutableEmergiModel):
    surge_level: int = Field(..., ge=1, le=4)
    geographic_scope: str = Field(...)
    affected_zones: List[ZoneID] = Field(..., min_length=1)
    hospitals_notified: List[HospitalID] = Field(default_factory=list)
    off_duty_staff_recalled: bool = Field(default=False)
    ndma_notified: bool = Field(default=False)
    media_blackout: bool = Field(default=False)

class DeclareSurgeAction(BaseAction):
    action_type: Literal[ActionType.DECLARE_SURGE] = ActionType.DECLARE_SURGE
    scope: SurgeDeclarationScope = Field(...)
    justification: str = Field(..., min_length=20, max_length=500)
    simultaneous_mci_count_observed: int = Field(..., ge=0)
    hospitals_on_diversion_count_observed: int = Field(..., ge=0)
    system_er_occupancy_observed: float = Field(..., ge=0.0, le=1.0)
    mutual_aid_already_active: bool = Field(default=False)
    cascade_risk_observed: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_surge_logic(self) -> "DeclareSurgeAction":
        if self.scope.surge_level == 4:
            if not self.scope.ndma_notified: raise ValueError("Level 4 requires NDMA.")
            if not self.scope.off_duty_staff_recalled: raise ValueError("Level 4 requires staff recall.")
        return self

    @computed_field
    @property
    def declaration_appears_justified(self) -> bool:
        return self.simultaneous_mci_count_observed >= 2 or self.hospitals_on_diversion_count_observed >= 3 or self.system_er_occupancy_observed >= 0.85 or self.cascade_risk_observed >= 0.50

    @computed_field
    @property
    def appears_premature(self) -> bool:
        return self.simultaneous_mci_count_observed == 0 and self.hospitals_on_diversion_count_observed == 0 and self.system_er_occupancy_observed < 0.70

    @computed_field
    @property
    def estimated_declaration_bonus(self) -> float:
        if self.appears_premature: return -0.05
        return SURGE_DECLARATION_BONUS if self.declaration_appears_justified else 0.0

    @computed_field
    @property
    def action_cost(self) -> int: return self.scope.surge_level * 2

    @computed_field
    @property
    def action_label(self) -> str:
        return f"[DECLARE_SURGE] level={self.scope.surge_level} scope={self.scope.geographic_scope} zones={len(self.scope.affected_zones)}"

class HospitalBypassAction(BaseAction):
    action_type: Literal[ActionType.HOSPITAL_BYPASS] = ActionType.HOSPITAL_BYPASS
    unit_id: UnitID = Field(...)
    bypassed_hospital_id: HospitalID = Field(...)
    bypass_reason: str = Field(...)
    new_hospital_id: HospitalID = Field(...)
    new_hospital_zone_id: ZoneID = Field(...)
    time_penalty_accepted_minutes: float = Field(default=0.0, ge=0.0)
    confirmed_diversion_flag: bool = Field(default=False)

    @field_validator("unit_id")
    @classmethod
    def validate_unit(cls, v: UnitID) -> UnitID:
        if not validate_unit_id(v): raise ValueError("Invalid unit_id.")
        return v

    @field_validator("bypassed_hospital_id", "new_hospital_id")
    @classmethod
    def validate_hospitals(cls, v: HospitalID) -> HospitalID:
        if not validate_hospital_id(v): raise ValueError("Invalid hospital_id.")
        return v

    @field_validator("new_hospital_zone_id")
    @classmethod
    def validate_zone(cls, v: ZoneID) -> ZoneID:
        if not validate_zone_id(v): raise ValueError("Invalid zone_id.")
        return v

    @model_validator(mode="after")
    def validate_bypass_logic(self) -> "HospitalBypassAction":
        if self.bypassed_hospital_id == self.new_hospital_id: raise ValueError("Hospitals must differ.")
        return self

    @computed_field
    @property
    def diversion_was_observed(self) -> bool: return self.bypass_reason in ("on_diversion", "er_at_capacity")

    @computed_field
    @property
    def time_penalty_vs_diversion_delay(self) -> float: return float(DIVERSION_REDIRECT_DELAY_MIN) - self.time_penalty_accepted_minutes

    @computed_field
    @property
    def bypass_appears_correct(self) -> bool: return self.confirmed_diversion_flag or self.diversion_was_observed

    @computed_field
    @property
    def action_cost(self) -> int: return 1

    @computed_field
    @property
    def action_label(self) -> str:
        c = "✓" if self.bypass_appears_correct else "?"
        return f"[BYPASS] unit={self.unit_id} skipping={self.bypassed_hospital_id} → {self.new_hospital_id} {c}"

class NoopReason(str):
    NO_INCIDENTS = "no_incidents"
    WAITING_MUTUAL_AID = "waiting_mutual_aid"
    WAITING_CREW_SWAP = "waiting_crew_swap"
    WAITING_TRAFFIC_UPDATE = "waiting_traffic_update"
    ALL_UNITS_DEPLOYED = "all_units_deployed"
    MONITORING = "monitoring"
    COMMS_LOST_UNIT = "comms_lost_unit"
    INTENTIONAL_HOLD = "intentional_hold"

class NoopAction(BaseAction):
    action_type: Literal[ActionType.NOOP] = ActionType.NOOP
    reason: str = Field(default=NoopReason.MONITORING)
    units_with_comms_lost: List[UnitID] = Field(default_factory=list)
    pending_mutual_aid_requests: int = Field(default=0, ge=0)
    p1_incidents_unaddressed: int = Field(default=0, ge=0)

    @field_validator("units_with_comms_lost")
    @classmethod
    def validate_comms_units(cls, v: List[UnitID]) -> List[UnitID]:
        for uid in v:
            if not validate_unit_id(uid): raise ValueError("Invalid unit_id.")
        return v

    @model_validator(mode="after")
    def validate_noop_context(self) -> "NoopAction":
        return self

    @computed_field
    @property
    def comms_penalty_applies(self) -> bool: return len(self.units_with_comms_lost) > 0 and self.reason != NoopReason.COMMS_LOST_UNIT

    @computed_field
    @property
    def estimated_noop_penalty(self) -> float: return NOOP_ON_COMMS_LOST_PENALTY * len(self.units_with_comms_lost) if self.comms_penalty_applies else 0.0

    @computed_field
    @property
    def is_justified(self) -> bool: return self.reason in (NoopReason.NO_INCIDENTS, NoopReason.ALL_UNITS_DEPLOYED, NoopReason.WAITING_MUTUAL_AID, NoopReason.COMMS_LOST_UNIT) and self.p1_incidents_unaddressed == 0

    @computed_field
    @property
    def action_cost(self) -> int: return 0

    @computed_field
    @property
    def action_label(self) -> str: return f"[NOOP] reason={self.reason} unaddressed_P1={self.p1_incidents_unaddressed}"

AnyAction = Union[DispatchAction, RerouteAction, EscalateAction, TriageTagAction, TransferAction, MutualAidRequestAction, PrepositionAction, CrewSwapAction, DeclareSurgeAction, HospitalBypassAction, NoopAction]

ACTION_TYPE_MAP: Dict[ActionType, Type[BaseAction]] = {ActionType.DISPATCH: DispatchAction, ActionType.REROUTE: RerouteAction, ActionType.ESCALATE: EscalateAction, ActionType.TAG: TriageTagAction, ActionType.TRANSFER: TransferAction, ActionType.REQUEST_MUTUAL_AID: MutualAidRequestAction, ActionType.PREPOSITION: PrepositionAction, ActionType.CREW_SWAP: CrewSwapAction, ActionType.DECLARE_SURGE: DeclareSurgeAction, ActionType.HOSPITAL_BYPASS: HospitalBypassAction, ActionType.NOOP: NoopAction}

TASK_VALID_ACTIONS: Dict[TaskID, FrozenSet[ActionType]] = {
    TaskID.T1: frozenset({ActionType.DISPATCH, ActionType.NOOP}),
    TaskID.T2: frozenset({ActionType.DISPATCH, ActionType.NOOP}),
    TaskID.T3: frozenset({ActionType.DISPATCH, ActionType.NOOP}),
    TaskID.T4: frozenset({ActionType.DISPATCH, ActionType.ESCALATE, ActionType.CREW_SWAP, ActionType.NOOP}),
    TaskID.T5: frozenset({ActionType.DISPATCH, ActionType.REROUTE, ActionType.HOSPITAL_BYPASS, ActionType.CREW_SWAP, ActionType.NOOP}),
    TaskID.T6: frozenset({ActionType.PREPOSITION, ActionType.NOOP}),
    TaskID.T7: frozenset({ActionType.DISPATCH, ActionType.TAG, ActionType.ESCALATE, ActionType.CREW_SWAP, ActionType.REQUEST_MUTUAL_AID, ActionType.NOOP}),
    TaskID.T8: frozenset({ActionType.TRANSFER, ActionType.REROUTE, ActionType.HOSPITAL_BYPASS, ActionType.CREW_SWAP, ActionType.NOOP}),
    TaskID.T9: frozenset({ActionType.DISPATCH, ActionType.REROUTE, ActionType.ESCALATE, ActionType.TAG, ActionType.TRANSFER, ActionType.REQUEST_MUTUAL_AID, ActionType.PREPOSITION, ActionType.CREW_SWAP, ActionType.DECLARE_SURGE, ActionType.HOSPITAL_BYPASS, ActionType.NOOP}),
}

def parse_action(data: Dict[str, Any]) -> AnyAction:
    t_str = data.get("action_type")
    if t_str is None: raise ValueError("Missing 'action_type'.")
    t = ActionType(t_str)
    cls = ACTION_TYPE_MAP.get(t)
    if cls is None: raise ValueError(f"No class for {t_str}.")
    return cls.model_validate(data)

class ActionBatch(EmergiBaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    task_id: TaskID = Field(...)
    actions: List[AnyAction] = Field(..., min_length=1, max_length=MAX_ACTIONS_PER_BATCH)

    @model_validator(mode="after")
    def validate_batch_consistency(self) -> "ActionBatch":
        for a in self.actions:
            if a.step != self.step: raise ValueError("Step mismatch.")
            if a.task_id != self.task_id: raise ValueError("Task mismatch.")
            if ActionType(a.action_type) not in TASK_VALID_ACTIONS.get(TaskID(self.task_id), frozenset()):
                raise ValueError(f"Action {a.action_type} invalid for {self.task_id}.")
        return self

    @computed_field
    @property
    def action_count(self) -> int: return len(self.actions)

    @computed_field
    @property
    def total_cost(self) -> int: return sum(a.action_cost for a in self.actions)

class MCIActionBatch(EmergiBaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    task_id: TaskID = Field(default=TaskID.T7)
    incident_id: IncidentID = Field(...)
    incident_zone_id: ZoneID = Field(...)
    triage_action: TriageTagAction = Field(...)
    dispatch_actions: List[DispatchAction] = Field(..., min_length=1, max_length=MAX_ACTIONS_PER_BATCH)
    mutual_aid_action: Optional[MutualAidRequestAction] = Field(default=None)
    incident_command_post_zone: Optional[ZoneID] = Field(default=None)

    @model_validator(mode="after")
    def validate_mci_bundle(self) -> "MCIActionBatch":
        if self.triage_action.incident_id != self.incident_id: raise ValueError("Incident mismatch.")
        if any(da.incident_id != self.incident_id for da in self.dispatch_actions): raise ValueError("Incident mismatch.")
        if sum(1 for da in self.dispatch_actions if UnitType(da.primary_unit_type) == UnitType.MICU) > 3: raise ValueError("Max 3 MICUs.")
        return self

class SurgeResponseBundle(EmergiBaseModel):
    bundle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    surge_declaration: DeclareSurgeAction = Field(...)
    mutual_aid_requests: List[MutualAidRequestAction] = Field(default_factory=list, max_length=6)
    dispatch_actions: List[DispatchAction] = Field(default_factory=list, max_length=MAX_ACTIONS_PER_BATCH)
    preposition_action: Optional[PrepositionAction] = Field(default=None)
    hospital_bypass_actions: List[HospitalBypassAction] = Field(default_factory=list, max_length=10)

class ActionLegalityChecker:
    @staticmethod
    def is_legal(action: BaseAction) -> bool: return ActionType(action.action_type) in TASK_VALID_ACTIONS.get(TaskID(action.task_id), frozenset())

class ActionCostEstimator:
    @staticmethod
    def estimate_dispatch_reward(action: DispatchAction) -> float:
        s = 0.0
        if action.protocol_hint_correct_unit is True: s += 0.30
        elif action.protocol_hint_correct_unit is False: s += WRONG_UNIT_DISPATCH_PENALTY * 0.30
        if action.pre_notification_bonus_eligible: s += 0.05
        if action.estimated_golden_hour_compliance: s += 0.25
        return clamp_score(s)

class ProtocolAdvisor:
    @staticmethod
    def recommend_unit_type(k: str) -> UnitType: return CONDITION_REQUIRED_UNIT.get(k, UnitType.ALS)
    @staticmethod
    def recommend_hospital_specialty(k: str) -> Optional[str]: return CONDITION_REQUIRED_SPECIALTY.get(k)
    @staticmethod
    def should_activate_cath_lab(k: str) -> bool: return k in {"stemi_anterior", "stemi_inferior", "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg", "cardiac_arrest_vf", "complete_heart_block", "wpw_svt"}
    @staticmethod
    def should_notify_stroke_unit(k: str) -> bool: return k in {"ischemic_stroke", "ischemic_stroke_wake_up", "hemorrhagic_stroke_sah", "meningitis_cryptococcal", "paediatric_stroke"}
    @staticmethod
    def should_send_trauma_activation(k: str) -> bool: return k in {"polytrauma_blunt", "polytrauma_penetrating", "severe_tbi", "mci_rta", "blast_injury", "crush_syndrome", "chest_trauma", "splenic_laceration", "gsw_shoulder", "degloving_crush", "traumatic_amputation", "skull_base_fracture"}
    @staticmethod
    def recommended_agencies(k: str) -> List[str]:
        m = {"polytrauma_blunt": ["Police", "Fire"], "mci_rta": ["Police", "Fire"], "mci_natural_disaster": ["Police", "Fire", "NDRF"], "carbon_monoxide": ["Fire"]}
        return m.get(k, [])

ModelRegistry.register("BaseAction", BaseAction)
ModelRegistry.register("DispatchRouting", DispatchRouting)
ModelRegistry.register("DispatchAction", DispatchAction)
ModelRegistry.register("RerouteAction", RerouteAction)
ModelRegistry.register("EscalateAction", EscalateAction)
ModelRegistry.register("VictimTriageAssignment", VictimTriageAssignment)
ModelRegistry.register("TriageTagAction", TriageTagAction)
ModelRegistry.register("TransferIndication", TransferIndication)
ModelRegistry.register("TransferAction", TransferAction)
ModelRegistry.register("MutualAidRequestAction", MutualAidRequestAction)
ModelRegistry.register("PrepositionAssignment", PrepositionAssignment)
ModelRegistry.register("PrepositionAction", PrepositionAction)
ModelRegistry.register("CrewSwapAction", CrewSwapAction)
ModelRegistry.register("SurgeDeclarationScope", SurgeDeclarationScope)
ModelRegistry.register("DeclareSurgeAction", DeclareSurgeAction)
ModelRegistry.register("HospitalBypassAction", HospitalBypassAction)
ModelRegistry.register("NoopAction", NoopAction)
ModelRegistry.register("ActionBatch", ActionBatch)
ModelRegistry.register("MCIActionBatch", MCIActionBatch)
ModelRegistry.register("SurgeResponseBundle", SurgeResponseBundle)
ActionModel    = ActionBatch
validate_action = parse_action

__all__ = ["MAX_UNITS_PER_DISPATCH", "MAX_ACTIONS_PER_BATCH", "MAX_MUTUAL_AID_UNITS_PER_REQUEST", "MAX_TRANSFER_PATIENTS", "MAX_PREPOSITION_ZONES", "MAX_TAG_VICTIMS_PER_ACTION", "NOOP_ON_COMMS_LOST_PENALTY", "WRONG_UNIT_DISPATCH_PENALTY", "WRONG_HOSPITAL_DISPATCH_PENALTY", "REROUTE_IMPROVEMENT_BONUS", "REROUTE_NO_IMPROVEMENT_PENALTY", "ESCALATION_VALID_TRANSITIONS", "TRANSFER_GOLDEN_WINDOW_MINUTES", "PREPOSITION_DEMAND_MATCH_BONUS", "SURGE_DECLARATION_BONUS", "SURGE_LATE_DECLARATION_PENALTY", "SURGE_CASCADE_AVOIDANCE_BONUS",
"ActionModel",
"validate_action","HOSPITAL_BYPASS_CORRECT_BONUS", "HOSPITAL_BYPASS_UNNECESSARY_PENALTY", "CREW_SWAP_CORRECT_BONUS", "CREW_SWAP_PREMATURE_PENALTY", "RerouteReason", "NoopReason", "BaseAction", "DispatchRouting", "DispatchAction", "RerouteAction", "EscalateAction", "VictimTriageAssignment", "TriageTagAction", "TransferIndication", "TransferAction", "MutualAidRequestAction", "PrepositionAssignment", "PrepositionAction", "CrewSwapAction", "SurgeDeclarationScope", "DeclareSurgeAction", "HospitalBypassAction", "NoopAction", "AnyAction", "ACTION_TYPE_MAP", "TASK_VALID_ACTIONS", "parse_action", "ActionBatch", "MCIActionBatch", "SurgeResponseBundle", "ActionLegalityChecker", "ActionCostEstimator", "ProtocolAdvisor"]