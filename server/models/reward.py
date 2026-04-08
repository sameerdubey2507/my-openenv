from __future__ import annotations
import logging
import math
import uuid
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Union
from pydantic import Field, computed_field, field_validator, model_validator
from server.models import (
    EmergiBaseModel,
    ImmutableEmergiModel,
    ModelRegistry,
    DecayModel,
    HospitalTier,
    IncidentType,
    ProtocolRule,
    SeverityLevel,
    TaskDifficulty,
    TaskID,
    TriageTag,
    UnitType,
    HospitalID,
    IncidentID,
    Probability,
    Score,
    StepNumber,
    UnitID,
    ZoneID,
    ACTIVE_ZONES_PER_EPISODE,
    COMMS_FAILURE_PROBABILITY_PER_STEP,
    DIVERSION_PENALTY,
    FLEET_SIZE_DEFAULT,
    MAX_STEPS_PER_EPISODE,
    MUTUAL_AID_OVER_REQUEST_PENALTY,
    PROTOCOL_COMPLIANCE_MAX_BONUS,
    SCORE_CEILING,
    SCORE_FLOOR,
    TIMESTEP_MINUTES,
    WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY,
    HospitalSpecialty,
    CONDITION_REQUIRED_UNIT,
    CONDITION_REQUIRED_SPECIALTY,
    SurvivalCurve,
    clamp_score,
    normalise_response_time,
    survival_probability_delta,
    weighted_sum_score,
    utc_now_iso,
    __schema_version__,
)
logger = logging.getLogger("emergi_env.models.reward")
__reward_version__: int = 7          
SURVIVAL_REWARD_SCALE: float = 1.0           
SURVIVAL_P1_WEIGHT: float    = 1.00          
SURVIVAL_P2_WEIGHT: float    = 0.55          
SURVIVAL_P3_WEIGHT: float    = 0.20          
SURVIVAL_P0_WEIGHT: float    = 0.00          
RESPONSE_TIME_PERFECT_P1_MIN: float  = 8.0   
RESPONSE_TIME_PERFECT_P2_MIN: float  = 30.0  
RESPONSE_TIME_PERFECT_P3_MIN: float  = 120.0 
RESPONSE_TIME_WORST_CASE_MIN: float  = 90.0  
TRIAGE_CORRECT_TAG_BONUS: float          = 0.08
TRIAGE_IMMEDIATE_AS_EXPECTANT_PENALTY: float = WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY
TRIAGE_MINOR_MISMATCH_PENALTY: float     = -0.04
TRIAGE_START_PROTOCOL_BONUS: float       = 0.05
HOSPITAL_SPECIALTY_MATCH_BONUS: float    = 0.20
HOSPITAL_SPECIALTY_MISMATCH_PENALTY: float = -0.25
HOSPITAL_CAPACITY_PENALTY_PER_PCT: float = -0.15   
HOSPITAL_DIVERSION_PENALTY: float        = DIVERSION_PENALTY
HOSPITAL_CATH_LAB_PRENOTIFY_BONUS: float = 0.18
HOSPITAL_STROKE_PRENOTIFY_BONUS: float   = 0.15
HOSPITAL_TRAUMA_ACTIVATION_BONUS: float  = 0.12
UNIT_TYPE_EXACT_MATCH_BONUS: float       = 0.30
UNIT_TYPE_UPGRADE_BONUS: float           = 0.10    
UNIT_TYPE_DOWNGRADE_PENALTY: float       = -0.20   
UNIT_MICU_FOR_STEMI_BONUS: float         = 0.15
MULTI_AGENCY_CORRECT_DISPATCH_BONUS: float  = 0.15
MULTI_AGENCY_OMISSION_PENALTY: float        = -0.025  
SINGLE_EMS_TRAPPED_PENALTY: float           = -0.25
REROUTE_TIME_SAVED_PER_MIN: float        = 0.002   
REROUTE_NO_IMPROVEMENT_PENALTY: float    = -0.02
REROUTE_DIVERSION_BYPASS_BONUS: float    = 0.05
PREPOSITION_RESPONSE_TIME_IMPROVEMENT_BONUS: float = 0.04
PREPOSITION_DEMAND_MATCH_BONUS: float    = 0.03
PREPOSITION_WASTEFUL_PENALTY: float      = -0.02
TRANSFER_TIMELY_BONUS: float             = 0.08
TRANSFER_LATE_PENALTY: float             = -0.10
TRANSFER_BED_CONFIRMED_BONUS: float      = 0.04
TRANSFER_WRONG_DESTINATION_PENALTY: float= -0.12
SURGE_CORRECT_DECLARATION_BONUS: float   = 0.08
SURGE_LATE_DECLARATION_PENALTY: float    = -0.12
SURGE_CASCADE_AVOIDANCE_BONUS: float     = 0.20
SURGE_CASCADE_FAILURE_PENALTY: float     = -0.35
MUTUAL_AID_CORRECT_REQUEST_BONUS: float  = 0.05
MUTUAL_AID_OVER_REQUEST_PENALTY: float   = MUTUAL_AID_OVER_REQUEST_PENALTY
MUTUAL_AID_UNDER_REQUEST_SURGE_PENALTY: float = -0.20
CREW_SWAP_JUSTIFIED_BONUS: float         = 0.02
CREW_SWAP_PREMATURE_PENALTY: float       = -0.02
COMMS_NOOP_ON_LOST_UNIT_PENALTY: float   = -0.05
NOOP_WITH_UNADDRESSED_P1_PENALTY: float  = -0.03
GOLDEN_HOUR_MET_BONUS: float             = 0.12
GOLDEN_HOUR_EXCEEDED_PENALTY: float      = -0.08
NORMALISATION_EPSILON: float             = 1e-8
TASK_REWARD_WEIGHTS: Dict[TaskID, Dict[str, float]] = {
    TaskID.T1: {
        "triage_accuracy":           0.40,
        "hospital_routing":          0.30,
        "response_time":             0.15,
        "protocol_compliance":       0.10,
        "survival":                  0.05,
        "diversion_avoidance":       0.00,
        "multi_agency_coordination": 0.00,
        "crew_fatigue_management":   0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "comms_resilience":          0.00,
        "noop_penalty":              0.00,
    },
    TaskID.T2: {
        "hospital_routing":          0.50,
        "triage_accuracy":           0.20,
        "response_time":             0.15,
        "diversion_avoidance":       0.10,
        "protocol_compliance":       0.05,
        "survival":                  0.00,
        "multi_agency_coordination": 0.00,
        "crew_fatigue_management":   0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "comms_resilience":          0.00,
        "noop_penalty":              0.00,
    },
    TaskID.T3: {
        "triage_accuracy":           0.60,
        "protocol_compliance":       0.25,
        "response_time":             0.10,
        "survival":                  0.05,
        "hospital_routing":          0.00,
        "diversion_avoidance":       0.00,
        "multi_agency_coordination": 0.00,
        "crew_fatigue_management":   0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "comms_resilience":          0.00,
        "noop_penalty":              0.00,
    },
    TaskID.T4: {
        "survival":                  0.35,
        "response_time":             0.25,
        "triage_accuracy":           0.15,
        "protocol_compliance":       0.10,
        "hospital_routing":          0.08,
        "crew_fatigue_management":   0.04,
        "noop_penalty":              0.03,
        "diversion_avoidance":       0.00,
        "multi_agency_coordination": 0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "comms_resilience":          0.00,
    },
    TaskID.T5: {
        "response_time":             0.30,
        "hospital_routing":          0.25,
        "diversion_avoidance":       0.20,
        "survival":                  0.12,
        "protocol_compliance":       0.08,
        "crew_fatigue_management":   0.05,
        "noop_penalty":              0.00,
        "triage_accuracy":           0.00,
        "multi_agency_coordination": 0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "comms_resilience":          0.00,
    },
    TaskID.T6: {
        "preposition_efficiency":    0.60,
        "response_time":             0.25,
        "protocol_compliance":       0.10,
        "survival":                  0.05,
        "hospital_routing":          0.00,
        "triage_accuracy":           0.00,
        "diversion_avoidance":       0.00,
        "multi_agency_coordination": 0.00,
        "crew_fatigue_management":   0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "comms_resilience":          0.00,
        "noop_penalty":              0.00,
    },
    TaskID.T7: {
        "triage_accuracy":           0.40,
        "survival":                  0.25,
        "response_time":             0.15,
        "multi_agency_coordination": 0.08,
        "protocol_compliance":       0.05,
        "hospital_routing":          0.04,
        "comms_resilience":          0.03,
        "diversion_avoidance":       0.00,
        "crew_fatigue_management":   0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "noop_penalty":              0.00,
    },
    TaskID.T8: {
        "transfer_timeliness":       0.35,
        "hospital_routing":          0.25,
        "survival":                  0.18,
        "protocol_compliance":       0.10,
        "comms_resilience":          0.07,
        "crew_fatigue_management":   0.05,
        "triage_accuracy":           0.00,
        "diversion_avoidance":       0.00,
        "multi_agency_coordination": 0.00,
        "preposition_efficiency":    0.00,
        "surge_management":          0.00,
        "mutual_aid_efficiency":     0.00,
        "cascade_avoidance":         0.00,
        "noop_penalty":              0.00,
        "response_time":             0.00,
    },
    TaskID.T9: {
        "survival":                  0.25,
        "cascade_avoidance":         0.22,
        "surge_management":          0.15,
        "mutual_aid_efficiency":     0.12,
        "hospital_routing":          0.08,
        "diversion_avoidance":       0.06,
        "triage_accuracy":           0.04,
        "response_time":             0.03,
        "comms_resilience":          0.02,
        "protocol_compliance":       0.02,
        "multi_agency_coordination": 0.01,
        "crew_fatigue_management":   0.00,
        "preposition_efficiency":    0.00,
        "transfer_timeliness":       0.00,
        "noop_penalty":              0.00,
    },
}
for _task, _wt in TASK_REWARD_WEIGHTS.items():
    _total = sum(_wt.values())
    assert abs(_total - 1.0) < 1e-6, (
        f"TASK_REWARD_WEIGHTS[{_task}] sums to {_total:.6f}, not 1.0"
    )
class SurvivalDelta(ImmutableEmergiModel):
    record_type: Literal["survival_delta"] = "survival_delta"
    patient_id: str
    incident_id: IncidentID
    victim_index: int               = Field(..., ge=0)
    severity: SeverityLevel
    condition_key: str
    p_before: Probability           = Field(..., ge=0.0, le=1.0)
    p_after: Probability            = Field(..., ge=0.0, le=1.0)
    delta: float                    
    severity_weight: float          = Field(..., ge=0.0, le=1.0)
    reward_contribution: float      
    response_received: bool         = Field(default=False)
    golden_hour_elapsed_pct: float  = Field(default=0.0, ge=0.0)
    decay_model: DecayModel
    step: StepNumber
    @computed_field
    @property
    def deteriorating(self) -> bool:
        return self.delta < 0.0
    @computed_field
    @property
    def delta_magnitude(self) -> float:
        return abs(self.delta)
    @computed_field
    @property
    def severity_label(self) -> str:
        return SeverityLevel(self.severity).colour
    @computed_field
    @property
    def expected_remaining_survival(self) -> float:
        return self.p_after
    @classmethod
    def compute(
        cls,
        patient_id: str,
        incident_id: IncidentID,
        victim_index: int,
        severity: SeverityLevel,
        condition_key: str,
        p_before: float,
        p_after: float,
        decay_model: DecayModel,
        step: int,
        response_received: bool = False,
        golden_hour_elapsed_pct: float = 0.0,
    ) -> "SurvivalDelta":
        weight_map = {
            SeverityLevel.P1: SURVIVAL_P1_WEIGHT,
            SeverityLevel.P2: SURVIVAL_P2_WEIGHT,
            SeverityLevel.P3: SURVIVAL_P3_WEIGHT,
            SeverityLevel.P0: SURVIVAL_P0_WEIGHT,
        }
        sev_enum = SeverityLevel(severity) if isinstance(severity, str) else severity
        w = weight_map.get(sev_enum, 0.0)
        delta = clamp_score(p_after) - clamp_score(p_before)
        contrib = delta * w * SURVIVAL_REWARD_SCALE
        return cls(
            patient_id=patient_id,
            incident_id=incident_id,
            victim_index=victim_index,
            severity=severity,
            condition_key=condition_key,
            p_before=clamp_score(p_before),
            p_after=clamp_score(p_after),
            delta=delta,
            severity_weight=w,
            reward_contribution=contrib,
            response_received=response_received,
            golden_hour_elapsed_pct=min(1.0, golden_hour_elapsed_pct),
            decay_model=decay_model,
            step=step,
        )
class SurvivalSummary(EmergiBaseModel):
    total_patients: int             = Field(default=0, ge=0)
    p1_patients: int                = Field(default=0, ge=0)
    p2_patients: int                = Field(default=0, ge=0)
    p3_patients: int                = Field(default=0, ge=0)
    p0_patients: int                = Field(default=0, ge=0)
    patients_survived: int          = Field(default=0, ge=0)
    patients_died: int              = Field(default=0, ge=0)
    patients_unknown: int           = Field(default=0, ge=0)
    p1_survived: int                = Field(default=0, ge=0)
    p1_died: int                    = Field(default=0, ge=0)
    avg_survival_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    total_survival_delta: float     = Field(default=0.0)
    golden_hour_met_count: int      = Field(default=0, ge=0)
    golden_hour_total: int          = Field(default=0, ge=0)
    avg_response_time_minutes: Optional[float] = Field(default=None, ge=0.0)
    cumulative_survival_reward: float = Field(default=0.0)
    condition_survival_rates: Dict[str, float] = Field(default_factory=dict)
    @computed_field
    @property
    def case_fatality_rate(self) -> float:
        if self.total_patients == 0:
            return 0.0
        return clamp_score(self.patients_died / self.total_patients)
    @computed_field
    @property
    def p1_survival_rate(self) -> float:
        total = self.p1_survived + self.p1_died
        if total == 0:
            return 1.0
        return clamp_score(self.p1_survived / total)
    @computed_field
    @property
    def golden_hour_compliance_rate(self) -> float:
        if self.golden_hour_total == 0:
            return 1.0
        return clamp_score(self.golden_hour_met_count / self.golden_hour_total)
    @computed_field
    @property
    def overall_survival_score(self) -> Score:
        cfr_score = 1.0 - self.case_fatality_rate
        p1_score = self.p1_survival_rate
        gh_score = self.golden_hour_compliance_rate
        avg_p_score = self.avg_survival_probability
        return clamp_score(
            0.30 * cfr_score
            + 0.30 * p1_score
            + 0.20 * gh_score
            + 0.20 * avg_p_score
        )
class ProtocolRewardEvent(ImmutableEmergiModel):
    record_type: Literal["protocol_event"] = "protocol_event"
    event_id: str               = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber
    rule: ProtocolRule
    compliant: bool
    reward_delta: float
    incident_id: Optional[IncidentID]   = Field(default=None)
    unit_id: Optional[UnitID]           = Field(default=None)
    hospital_id: Optional[HospitalID]   = Field(default=None)
    description: str                    = Field(default="")
    accumulated_bonus_after: float      = Field(default=0.0, ge=0.0, le=PROTOCOL_COMPLIANCE_MAX_BONUS)
    @computed_field
    @property
    def is_violation(self) -> bool:
        return not self.compliant
    @computed_field
    @property
    def severity_label(self) -> str:
        if self.reward_delta <= -0.04:
            return "SEVERE"
        if self.reward_delta < 0:
            return "WARNING"
        return "BONUS"
    @computed_field
    @property
    def summary(self) -> str:
        status = "✓ COMPLIANT" if self.compliant else "✗ VIOLATION"
        return (
            f"{status} | rule={self.rule.value} | Δ={self.reward_delta:+.3f} | "
            f"step={self.step}"
            + (f" | incident={self.incident_id}" if self.incident_id else "")
        )
class PenaltyEvent(ImmutableEmergiModel):
    record_type: Literal["penalty_event"] = "penalty_event"
    event_id: str               = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber
    penalty_type: str           = Field(
        ...,
        description=(
            "diversion_routing | comms_noop | mutual_aid_over_request | "
            "single_ems_trapped | noop_unaddressed_p1 | cascade_failure | "
            "triage_critical_mismatch | wrong_unit_type | wrong_hospital"
        ),
    )
    penalty_value: float        = Field(..., le=0.0)
    incident_id: Optional[IncidentID]   = Field(default=None)
    unit_id: Optional[UnitID]           = Field(default=None)
    hospital_id: Optional[HospitalID]   = Field(default=None)
    description: str            = Field(default="")
    @computed_field
    @property
    def is_severe(self) -> bool:
        return self.penalty_value <= -0.20
    @computed_field
    @property
    def summary(self) -> str:
        return (
            f"PENALTY | type={self.penalty_type} | value={self.penalty_value:+.3f} | "
            f"step={self.step}"
            + (f" | incident={self.incident_id}" if self.incident_id else "")
        )
class DispatchGraderHint(ImmutableEmergiModel):
    record_type: Literal["dispatch_hint"] = "dispatch_hint"
    incident_id: IncidentID
    condition_key: str
    dispatched_unit_type: str
    correct_unit_type: str
    unit_type_correct: bool
    dispatched_hospital_id: HospitalID
    required_specialty: Optional[str]
    hospital_specialty_correct: bool
    hospital_on_diversion: bool
    cath_lab_activated: bool
    cath_lab_required: bool
    golden_hour_compliance: bool
    estimated_response_minutes: float
    pre_notification_sent: bool
    multi_agency_correct: bool
    score_contribution: float       = Field(ge=0.0, le=1.0)
    @computed_field
    @property
    def grade(self) -> str:
        if self.score_contribution >= 0.85:
            return "EXCELLENT"
        if self.score_contribution >= 0.65:
            return "GOOD"
        if self.score_contribution >= 0.45:
            return "FAIR"
        return "POOR"
class TriageGraderHint(ImmutableEmergiModel):
    record_type: Literal["triage_hint"] = "triage_hint"
    incident_id: IncidentID
    victim_count_tagged: int
    correct_tags: int
    wrong_tags: int
    critical_mismatches: int
    tag_accuracy: float             = Field(ge=0.0, le=1.0)
    triage_score: float             = Field(ge=0.0, le=1.0)
    start_protocol_applied: bool
    @computed_field
    @property
    def grade(self) -> str:
        if self.triage_score >= 0.9 and self.critical_mismatches == 0:
            return "PERFECT"
        if self.triage_score >= 0.7 and self.critical_mismatches == 0:
            return "GOOD"
        if self.critical_mismatches > 0:
            return "CRITICAL_ERROR"
        return "POOR"
class RerouteGraderHint(ImmutableEmergiModel):
    record_type: Literal["reroute_hint"] = "reroute_hint"
    unit_id: UnitID
    time_saved_minutes: float
    diversion_triggered: bool
    is_improvement: bool
    new_hospital_id: HospitalID
    new_hospital_specialty_correct: bool
    reroute_score: float            = Field(ge=0.0, le=1.0)
class TransferGraderHint(ImmutableEmergiModel):
    record_type: Literal["transfer_hint"] = "transfer_hint"
    patient_count: int
    receiving_hospital_correct: bool
    timing_within_window: bool
    bed_confirmed: bool
    transfer_score: float           = Field(ge=0.0, le=1.0)
class SurgeGraderHint(ImmutableEmergiModel):
    record_type: Literal["surge_hint"] = "surge_hint"
    surge_justified: bool
    surge_timely: bool
    cascade_prevented: bool
    mutual_aid_adequate: bool
    surge_score: float              = Field(ge=0.0, le=1.0)
class GraderHints(EmergiBaseModel):
    record_type: Literal["grader_hints"] = "grader_hints"
    step: StepNumber
    dispatch_hints: List[DispatchGraderHint]    = Field(default_factory=list)
    triage_hints: List[TriageGraderHint]        = Field(default_factory=list)
    reroute_hints: List[RerouteGraderHint]      = Field(default_factory=list)
    transfer_hints: List[TransferGraderHint]    = Field(default_factory=list)
    surge_hints: List[SurgeGraderHint]          = Field(default_factory=list)
    general_warnings: List[str]                 = Field(default_factory=list)
    general_commendations: List[str]            = Field(default_factory=list)
    @computed_field
    @property
    def has_critical_errors(self) -> bool:
        return any(h.grade == "CRITICAL_ERROR" for h in self.triage_hints)
    @computed_field
    @property
    def total_hints(self) -> int:
        return (
            len(self.dispatch_hints)
            + len(self.triage_hints)
            + len(self.reroute_hints)
            + len(self.transfer_hints)
            + len(self.surge_hints)
        )
    @computed_field
    @property
    def avg_dispatch_score(self) -> float:
        if not self.dispatch_hints:
            return 1.0
        return sum(h.score_contribution for h in self.dispatch_hints) / len(self.dispatch_hints)
class StepRewardSignal(EmergiBaseModel):
    record_type: Literal["step_reward"] = "step_reward"
    schema_version: int             = Field(default=__schema_version__)
    step: StepNumber                = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    episode_id: str
    task_id: TaskID
    channels: Dict[str, float]      = Field(
        ...,
        description="RewardChannel name → reward delta for THIS step only",
    )
    total_step_reward: float        = Field(
        ...,
        description=(
            "Weighted sum of channel deltas for this step. "
            "This is the scalar 'r' in the RL update rule. "
            "May be negative (penalty step)."
        ),
    )
    cumulative_score: Score         = Field(
        ...,
        ge=SCORE_FLOOR,
        le=SCORE_CEILING,
        description="Running episode score so far, clamped to [0.0, 1.0]",
    )
    score_delta: float              = Field(
        ...,
        description="Change in cumulative score from previous step. (+) = improvement.",
    )
    survival_deltas: List[SurvivalDelta]           = Field(default_factory=list)
    protocol_events: List[ProtocolRewardEvent]     = Field(default_factory=list)
    penalty_events: List[PenaltyEvent]             = Field(default_factory=list)
    grader_hints: GraderHints
    num_patients_deteriorating: int = Field(default=0, ge=0)
    num_protocol_violations: int    = Field(default=0, ge=0)
    num_protocol_bonuses: int       = Field(default=0, ge=0)
    total_penalty_this_step: float  = Field(default=0.0)
    total_bonus_this_step: float    = Field(default=0.0)
    episode_done: bool              = Field(default=False)
    termination_reason: Optional[str] = Field(default=None)
    computed_at: str                = Field(default_factory=utc_now_iso)
    @computed_field
    @property
    def net_reward_this_step(self) -> float:
        return self.total_bonus_this_step + self.total_penalty_this_step
    @computed_field
    @property
    def is_penalty_step(self) -> bool:
        return self.total_step_reward < 0.0
    @computed_field
    @property
    def reward_breakdown_str(self) -> str:
        lines = [f"Step {self.step} | r={self.total_step_reward:+.4f} | score={self.cumulative_score:.4f}"]
        for ch, val in sorted(self.channels.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(val) > 0.001:
                lines.append(f"  {ch:35s}: {val:+.4f}")
        return "\n".join(lines)
    @computed_field
    @property
    def dominant_channel(self) -> str:
        if not self.channels:
            return "none"
        return max(self.channels, key=lambda k: abs(self.channels[k]))
    @computed_field
    @property
    def survival_reward_this_step(self) -> float:
        return self.channels.get("survival", 0.0)
    @computed_field
    @property
    def protocol_reward_this_step(self) -> float:
        return self.channels.get("protocol_compliance", 0.0)
    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, val in v.items():
            if not math.isfinite(val):
                v[k] = 0.0
        return v
    @model_validator(mode="after")
    def validate_step_reward_consistency(self) -> "StepRewardSignal":
        if not math.isfinite(self.total_step_reward):
            raise ValueError("total_step_reward must be finite.")
        if not math.isfinite(self.cumulative_score):
            raise ValueError("cumulative_score must be finite.")
        if self.schema_version != __schema_version__:
            raise ValueError(
                f"StepRewardSignal schema_version {self.schema_version} "
                f"!= {__schema_version__}"
            )
        return self
class Task1ComponentScore(ImmutableEmergiModel):
    triage_class_score: Score           = Field(..., ge=0.0, le=1.0)
    unit_type_score: Score              = Field(..., ge=0.0, le=1.0)
    hospital_match_score: Score         = Field(..., ge=0.0, le=1.0)
    declared_severity: str
    correct_severity: str
    declared_unit_type: str
    correct_unit_type: str
    declared_hospital_id: HospitalID
    required_specialty: Optional[str]
    specialty_matched: bool
    cath_lab_activated_when_needed: bool = Field(default=False)
    @computed_field
    @property
    def composite_score(self) -> Score:
        return clamp_score(
            0.40 * self.triage_class_score
            + 0.30 * self.unit_type_score
            + 0.30 * self.hospital_match_score
        )
class Task2ComponentScore(ImmutableEmergiModel):
    specialty_match_score: Score        = Field(..., ge=0.0, le=1.0)
    capacity_check_score: Score         = Field(..., ge=0.0, le=1.0)
    travel_time_score: Score            = Field(..., ge=0.0, le=1.0)
    required_specialty: str
    hospital_has_specialty: bool
    hospital_er_occupancy: float
    hospital_on_diversion: bool
    estimated_travel_minutes: float
    golden_hour_minutes: int
    @computed_field
    @property
    def composite_score(self) -> Score:
        return clamp_score(
            0.50 * self.specialty_match_score
            + 0.30 * self.capacity_check_score
            + 0.20 * self.travel_time_score
        )
class Task3ComponentScore(ImmutableEmergiModel):
    exact_match: bool
    declared_unit_type: str
    correct_unit_type: str
    condition_key: str
    condition_severity: str
    @computed_field
    @property
    def composite_score(self) -> Score:
        return 1.0 if self.exact_match else 0.0
class Task4ComponentScore(ImmutableEmergiModel):
    weighted_survival_score: Score      = Field(..., ge=0.0, le=1.0)
    delay_penalty_total: float          = Field(default=0.0, le=0.0)
    incidents_dispatched: int           = Field(default=0, ge=0)
    incidents_total: int                = Field(default=0, ge=0)
    p1_response_rate: float             = Field(default=1.0, ge=0.0, le=1.0)
    avg_weighted_survival: float        = Field(default=0.0, ge=0.0, le=1.0)
    resource_allocation_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    per_patient_scores: Dict[str, float] = Field(default_factory=dict)
    @computed_field
    @property
    def composite_score(self) -> Score:
        raw = self.weighted_survival_score + self.delay_penalty_total
        return clamp_score(raw)
class Task5ComponentScore(ImmutableEmergiModel):
    reroute_correctness_score: Score    = Field(..., ge=0.0, le=1.0)
    time_saved_score: Score             = Field(..., ge=0.0, le=1.0)
    reroutes_executed: int              = Field(default=0, ge=0)
    reroutes_correct: int               = Field(default=0, ge=0)
    total_time_saved_minutes: float     = Field(default=0.0)
    avg_time_saved_per_reroute: float   = Field(default=0.0)
    diversions_avoided: int             = Field(default=0, ge=0)
    @computed_field
    @property
    def composite_score(self) -> Score:
        return clamp_score(
            0.50 * self.reroute_correctness_score
            + 0.50 * self.time_saved_score
        )
class Task6ComponentScore(ImmutableEmergiModel):
    avg_response_time_minutes: float    = Field(..., ge=0.0)
    random_baseline_response_minutes: float = Field(..., ge=0.0)
    improvement_pct: float              = Field(default=0.0)
    zones_covered: int                  = Field(default=0, ge=0)
    total_active_zones: int             = Field(default=ACTIVE_ZONES_PER_EPISODE, ge=1)
    forecast_informed_ratio: float      = Field(default=0.0, ge=0.0, le=1.0)
    delayed_response_events: int        = Field(default=0, ge=0)
    @computed_field
    @property
    def coverage_rate(self) -> float:
        return clamp_score(self.zones_covered / self.total_active_zones)
    @computed_field
    @property
    def composite_score(self) -> Score:
        if self.random_baseline_response_minutes <= 0.0:
            return 0.0
        improvement_score = clamp_score(
            self.improvement_pct / 50.0   
        )
        coverage_score = self.coverage_rate
        forecast_score = self.forecast_informed_ratio
        return clamp_score(
            0.50 * improvement_score
            + 0.30 * coverage_score
            + 0.20 * forecast_score
        )
class Task7ComponentScore(ImmutableEmergiModel):
    start_accuracy_score: Score         = Field(..., ge=0.0, le=1.0)
    response_score: Score               = Field(..., ge=0.0, le=1.0)
    hospital_spread_score: Score        = Field(..., ge=0.0, le=1.0)
    victims_total: int                  = Field(..., ge=1)
    victims_correctly_tagged: int       = Field(default=0, ge=0)
    critical_mismatches: int            = Field(default=0, ge=0)
    hospitals_used: int                 = Field(default=1, ge=1)
    hospitals_available: int            = Field(default=1, ge=1)
    critical_mismatch_penalty_total: float = Field(default=0.0, le=0.0)
    avg_time_to_immediate_dispatch_minutes: float = Field(default=0.0, ge=0.0)
    @computed_field
    @property
    def tag_accuracy_rate(self) -> float:
        return clamp_score(self.victims_correctly_tagged / max(1, self.victims_total))
    @computed_field
    @property
    def hospital_spread_rate(self) -> float:
        return clamp_score(self.hospitals_used / max(1, min(4, self.hospitals_available)))
    @computed_field
    @property
    def composite_score(self) -> Score:
        raw = (
            0.45 * self.start_accuracy_score
            + 0.30 * self.response_score
            + 0.25 * self.hospital_spread_score
            + self.critical_mismatch_penalty_total  
        )
        return clamp_score(raw)
class Task8ComponentScore(ImmutableEmergiModel):
    transfer_appropriateness_score: Score = Field(..., ge=0.0, le=1.0)
    timing_score: Score                 = Field(..., ge=0.0, le=1.0)
    utilisation_score: Score            = Field(..., ge=0.0, le=1.0)
    transfers_attempted: int            = Field(default=0, ge=0)
    transfers_appropriate: int          = Field(default=0, ge=0)
    transfers_timely: int               = Field(default=0, ge=0)
    avg_transfer_time_minutes: float    = Field(default=0.0, ge=0.0)
    icu_utilisation_after_transfer: float = Field(default=0.0, ge=0.0, le=1.0)
    bed_confirmation_rate: float        = Field(default=0.0, ge=0.0, le=1.0)
    @computed_field
    @property
    def composite_score(self) -> Score:
        return clamp_score(
            0.45 * self.transfer_appropriateness_score
            + 0.30 * self.timing_score
            + 0.25 * self.utilisation_score
        )
class Task9ComponentScore(ImmutableEmergiModel):
    system_survival_score: Score        = Field(..., ge=0.0, le=1.0)
    cascade_avoidance_score: Score      = Field(..., ge=0.0, le=1.0)
    mutual_aid_score: Score             = Field(..., ge=0.0, le=1.0)
    surge_declared_timely: bool
    cascade_occurred: bool
    hospitals_survived_surge: int       = Field(default=0, ge=0)
    hospitals_total: int                = Field(default=1, ge=1)
    mutual_aid_requests_correct: int    = Field(default=0, ge=0)
    mutual_aid_requests_total: int      = Field(default=0, ge=0)
    over_requests: int                  = Field(default=0, ge=0)
    under_requests: int                 = Field(default=0, ge=0)
    system_er_occupancy_at_end: float   = Field(default=0.0, ge=0.0, le=1.0)
    simultaneous_mci_count_peak: int    = Field(default=0, ge=0)
    over_request_penalty_total: float   = Field(default=0.0, le=0.0)
    cascade_failure_penalty: float      = Field(default=0.0, le=0.0)
    @computed_field
    @property
    def hospital_survival_rate(self) -> float:
        return clamp_score(self.hospitals_survived_surge / max(1, self.hospitals_total))
    @computed_field
    @property
    def mutual_aid_accuracy(self) -> float:
        if self.mutual_aid_requests_total == 0:
            return 1.0
        return clamp_score(self.mutual_aid_requests_correct / self.mutual_aid_requests_total)
    @computed_field
    @property
    def composite_score(self) -> Score:
        raw = (
            0.45 * self.system_survival_score
            + 0.35 * self.cascade_avoidance_score
            + 0.20 * self.mutual_aid_score
            + self.over_request_penalty_total
            + self.cascade_failure_penalty
        )
        return clamp_score(raw)
TaskComponentScore = Union[
    Task1ComponentScore,
    Task2ComponentScore,
    Task3ComponentScore,
    Task4ComponentScore,
    Task5ComponentScore,
    Task6ComponentScore,
    Task7ComponentScore,
    Task8ComponentScore,
    Task9ComponentScore,
]
class EpisodePenaltySummary(EmergiBaseModel):
    diversion_routing_penalty: float    = Field(default=0.0, le=0.0)
    wrong_unit_type_penalty: float      = Field(default=0.0, le=0.0)
    wrong_hospital_penalty: float       = Field(default=0.0, le=0.0)
    triage_critical_mismatch_penalty: float = Field(default=0.0, le=0.0)
    single_ems_trapped_penalty: float   = Field(default=0.0, le=0.0)
    noop_unaddressed_p1_penalty: float  = Field(default=0.0, le=0.0)
    comms_noop_penalty: float           = Field(default=0.0, le=0.0)
    mutual_aid_over_request_penalty: float = Field(default=0.0, le=0.0)
    mutual_aid_under_request_penalty: float = Field(default=0.0, le=0.0)
    cascade_failure_penalty: float      = Field(default=0.0, le=0.0)
    surge_late_declaration_penalty: float = Field(default=0.0, le=0.0)
    transfer_late_penalty: float        = Field(default=0.0, le=0.0)
    crew_swap_premature_penalty: float  = Field(default=0.0, le=0.0)
    protocol_violation_total: float     = Field(default=0.0, le=0.0)
    @computed_field
    @property
    def total_penalty(self) -> float:
        return (
            self.diversion_routing_penalty
            + self.wrong_unit_type_penalty
            + self.wrong_hospital_penalty
            + self.triage_critical_mismatch_penalty
            + self.single_ems_trapped_penalty
            + self.noop_unaddressed_p1_penalty
            + self.comms_noop_penalty
            + self.mutual_aid_over_request_penalty
            + self.mutual_aid_under_request_penalty
            + self.cascade_failure_penalty
            + self.surge_late_declaration_penalty
            + self.transfer_late_penalty
            + self.crew_swap_premature_penalty
            + self.protocol_violation_total
        )
    @computed_field
    @property
    def penalty_breakdown_str(self) -> str:
        fields = [
            ("diversion_routing", self.diversion_routing_penalty),
            ("wrong_unit_type", self.wrong_unit_type_penalty),
            ("wrong_hospital", self.wrong_hospital_penalty),
            ("triage_critical_mismatch", self.triage_critical_mismatch_penalty),
            ("single_ems_trapped", self.single_ems_trapped_penalty),
            ("noop_unaddressed_p1", self.noop_unaddressed_p1_penalty),
            ("comms_noop", self.comms_noop_penalty),
            ("mutual_aid_over_request", self.mutual_aid_over_request_penalty),
            ("cascade_failure", self.cascade_failure_penalty),
            ("surge_late_declaration", self.surge_late_declaration_penalty),
            ("protocol_violations", self.protocol_violation_total),
        ]
        lines = [f"  {'TOTAL':35s}: {self.total_penalty:+.4f}"]
        for name, val in fields:
            if abs(val) > 1e-6:
                lines.append(f"  {name:35s}: {val:+.4f}")
        return "\n".join(lines)
class GraderTraceEntry(ImmutableEmergiModel):
    trace_id: str               = Field(default_factory=lambda: str(uuid.uuid4()))
    step: StepNumber
    action_type: str
    action_id: str              = Field(default="")
    score_component: str        = Field(
        ...,
        description="Which component of the task grader this applies to",
    )
    raw_value: float
    normalised_value: Score     = Field(..., ge=0.0, le=1.0)
    weight: float               = Field(..., ge=0.0, le=1.0)
    weighted_contribution: float
    explanation: str
    incident_id: Optional[IncidentID]   = Field(default=None)
    unit_id: Optional[UnitID]           = Field(default=None)
    hospital_id: Optional[HospitalID]   = Field(default=None)
    @computed_field
    @property
    def is_positive(self) -> bool:
        return self.weighted_contribution >= 0.0
    @computed_field
    @property
    def trace_summary(self) -> str:
        sign = "+" if self.is_positive else ""
        return (
            f"[{self.action_type:20s}] {self.score_component:35s} | "
            f"raw={self.raw_value:.3f} → norm={self.normalised_value:.3f} | "
            f"w={self.weight:.2f} → {sign}{self.weighted_contribution:.4f} | "
            f"{self.explanation}"
        )
class BaselineComparison(ImmutableEmergiModel):
    task_id: TaskID
    task_difficulty: TaskDifficulty
    episode_score: Score            = Field(..., ge=0.0, le=1.0)
    baseline_score: Score           = Field(..., ge=0.0, le=1.0)
    delta: float                    
    percentile_estimate: float      = Field(default=0.5, ge=0.0, le=1.0)
    beats_baseline: bool
    improvement_pct: float
    @computed_field
    @property
    def performance_tier(self) -> str:
        if self.delta >= 0.25:
            return "EXCEPTIONAL"
        if self.delta >= 0.10:
            return "STRONG"
        if self.delta >= 0.0:
            return "ABOVE_BASELINE"
        if self.delta >= -0.10:
            return "NEAR_BASELINE"
        return "BELOW_BASELINE"
    @computed_field
    @property
    def rank_hint(self) -> str:
        tier = self.performance_tier
        tier_ranks = {
            "EXCEPTIONAL": "Top 5% estimated",
            "STRONG": "Top 20% estimated",
            "ABOVE_BASELINE": "Above average",
            "NEAR_BASELINE": "Near baseline",
            "BELOW_BASELINE": "Below baseline",
        }
        return tier_ranks.get(tier, "Unknown")
    @classmethod
    def build(
        cls,
        task_id: TaskID,
        episode_score: float,
    ) -> "BaselineComparison":
        t = TaskID(task_id)
        baseline = t.baseline_score
        delta = episode_score - baseline
        improvement_pct = (delta / max(baseline, NORMALISATION_EPSILON)) * 100.0
        if delta >= 0.25:
            pct = 0.95
        elif delta >= 0.10:
            pct = 0.80
        elif delta >= 0.0:
            pct = 0.60
        elif delta >= -0.10:
            pct = 0.40
        else:
            pct = 0.20
        return cls(
            task_id=task_id,
            task_difficulty=t.difficulty,
            episode_score=clamp_score(episode_score),
            baseline_score=baseline,
            delta=delta,
            percentile_estimate=pct,
            beats_baseline=episode_score > baseline,
            improvement_pct=improvement_pct,
        )
class EpisodeScoreResult(EmergiBaseModel):
    record_type: Literal["episode_score"] = "episode_score"
    schema_version: int             = Field(default=__schema_version__)
    episode_id: str
    task_id: TaskID
    task_difficulty: TaskDifficulty
    task_seed: int                  = Field(..., ge=42, le=50)
    steps_executed: int             = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    terminated_early: bool          = Field(default=False)
    termination_reason: Optional[str] = Field(default=None)
    final_score: Score              = Field(
        ...,
        ge=SCORE_FLOOR,
        le=SCORE_CEILING,
        description=(
            "Final normalised episode score in [0.0, 1.0]. "
            "This is the single metric used by the OpenEnv leaderboard."
        ),
    )
    final_score_before_protocol_bonus: Score = Field(
        ...,
        ge=SCORE_FLOOR,
        le=SCORE_CEILING,
    )
    component_scores: Dict[str, float]  = Field(
        ...,
        description="Grader dimension name → score [0.0, 1.0]",
    )
    channel_scores: Dict[str, float]    = Field(
        ...,
        description="RewardChannel name → accumulated value across episode",
    )
    task_specific: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task-specific component score object (serialised)",
    )
    protocol_bonus: float           = Field(
        ...,
        ge=0.0,
        le=PROTOCOL_COMPLIANCE_MAX_BONUS,
        description=f"Protocol compliance bonus, max {PROTOCOL_COMPLIANCE_MAX_BONUS}",
    )
    penalties: EpisodePenaltySummary
    survival_summary: SurvivalSummary
    grader_trace: List[GraderTraceEntry]    = Field(default_factory=list)
    protocol_events: List[ProtocolRewardEvent] = Field(default_factory=list)
    baseline_comparison: BaselineComparison
    scored_at: str                  = Field(default_factory=utc_now_iso)
    @field_validator("final_score", "final_score_before_protocol_bonus")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        return clamp_score(v)
    @field_validator("component_scores", "channel_scores")
    @classmethod
    def validate_score_dicts(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {k: clamp_score(val) if "score" in k else val for k, val in v.items()}
    @model_validator(mode="after")
    def validate_episode_score_consistency(self) -> "EpisodeScoreResult":
        if self.schema_version != __schema_version__:
            raise ValueError(
                f"EpisodeScoreResult schema_version {self.schema_version} "
                f"!= {__schema_version__}"
            )
        expected_seed = TaskID(self.task_id).seed
        if self.task_seed != expected_seed:
            raise ValueError(
                f"task_seed {self.task_seed} != expected {expected_seed} for {self.task_id}"
            )
        if not (SCORE_FLOOR <= self.final_score <= SCORE_CEILING):
            raise ValueError(
                f"final_score {self.final_score} is outside [0.0, 1.0]"
            )
        if self.protocol_bonus > PROTOCOL_COMPLIANCE_MAX_BONUS + 1e-6:
            raise ValueError(
                f"protocol_bonus {self.protocol_bonus} exceeds max {PROTOCOL_COMPLIANCE_MAX_BONUS}"
            )
        return self
    @computed_field
    @property
    def improvement_over_baseline(self) -> float:
        return self.final_score - self.baseline_comparison.baseline_score
    @computed_field
    @property
    def beats_baseline(self) -> bool:
        return self.final_score > self.baseline_comparison.baseline_score
    @computed_field
    @property
    def score_summary_str(self) -> str:
        b = "✓ BEATS BASELINE" if self.beats_baseline else "✗ BELOW BASELINE"
        lines = [
            f"╔═══════════════════════════════════════════════════╗",
            f"  EMERGI-ENV Episode Score Report",
            f"  Task: {self.task_id} ({self.task_difficulty})",
            f"  Episode: {self.episode_id[:8]}…",
            f"  Steps: {self.steps_executed}/{MAX_STEPS_PER_EPISODE}",
            f"",
            f"  FINAL SCORE : {self.final_score:.4f}",
            f"  Baseline    : {self.baseline_comparison.baseline_score:.4f}",
            f"  Delta       : {self.improvement_over_baseline:+.4f}  {b}",
            f"  Protocol +  : {self.protocol_bonus:+.4f}",
            f"  Penalties   : {self.penalties.total_penalty:+.4f}",
            f"",
            f"  Performance : {self.baseline_comparison.performance_tier}",
            f"  Survival    : {self.survival_summary.overall_survival_score:.4f}",
            f"  P1 Survive  : {self.survival_summary.p1_survival_rate:.0%}",
            f"  GH Compliance: {self.survival_summary.golden_hour_compliance_rate:.0%}",
            f"╚═══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
    def top_contributing_channels(self, n: int = 5) -> List[Tuple[str, float]]:
        sorted_channels = sorted(
            self.channel_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_channels[:n]
class RewardComputationEngine:
    @staticmethod
    def compute_survival_reward(
        survival_deltas: List[SurvivalDelta],
        task_id: TaskID,
    ) -> float:
        if not survival_deltas:
            return 0.0
        total = sum(d.reward_contribution for d in survival_deltas)
        n = max(1, len(survival_deltas))
        return total / n
    @staticmethod
    def compute_patient_survival_delta(
        p_before: float,
        p_after: float,
        severity: str,
    ) -> float:
        weight_map = {
            "P1": SURVIVAL_P1_WEIGHT,
            "P2": SURVIVAL_P2_WEIGHT,
            "P3": SURVIVAL_P3_WEIGHT,
            "P0": SURVIVAL_P0_WEIGHT,
        }
        w = weight_map.get(severity, 0.0)
        delta = clamp_score(p_after) - clamp_score(p_before)
        return delta * w
    @staticmethod
    def compute_response_time_reward(
        actual_minutes: float,
        severity: str,
    ) -> float:
        targets = {
            "P1": RESPONSE_TIME_PERFECT_P1_MIN,
            "P2": RESPONSE_TIME_PERFECT_P2_MIN,
            "P3": RESPONSE_TIME_PERFECT_P3_MIN,
            "P0": 0.0,
        }
        target = targets.get(severity, RESPONSE_TIME_PERFECT_P1_MIN)
        return normalise_response_time(
            actual_minutes,
            target,
            worst_case_minutes=RESPONSE_TIME_WORST_CASE_MIN,
        )
    @staticmethod
    def compute_triage_accuracy_reward(
        assigned_tag: str,
        correct_tag: str,
        severity: str,
    ) -> float:
        a = TriageTag(assigned_tag) if isinstance(assigned_tag, str) else assigned_tag
        c = TriageTag(correct_tag) if isinstance(correct_tag, str) else correct_tag
        if a == c:
            return TRIAGE_CORRECT_TAG_BONUS
        if c == TriageTag.IMMEDIATE and a == TriageTag.EXPECTANT:
            return TRIAGE_IMMEDIATE_AS_EXPECTANT_PENALTY
        return TRIAGE_MINOR_MISMATCH_PENALTY
    @staticmethod
    def compute_start_triage_score(
        tagged_correctly: int,
        tagged_total: int,
        critical_mismatches: int,
    ) -> float:
        if tagged_total == 0:
            return 0.0
        accuracy = clamp_score(tagged_correctly / tagged_total)
        mismatch_penalty = critical_mismatches * abs(TRIAGE_IMMEDIATE_AS_EXPECTANT_PENALTY)
        raw = accuracy - mismatch_penalty
        return clamp_score(raw)
    @staticmethod
    def compute_hospital_routing_reward(
        hospital_has_specialty: bool,
        hospital_on_diversion: bool,
        hospital_er_occupancy: float,
        travel_time_minutes: float,
        golden_hour_minutes: int,
        cath_lab_activated: bool = False,
        stroke_notified: bool = False,
        trauma_activated: bool = False,
    ) -> float:
        if hospital_on_diversion:
            return HOSPITAL_DIVERSION_PENALTY
        specialty_score = HOSPITAL_SPECIALTY_MATCH_BONUS if hospital_has_specialty \
                          else HOSPITAL_SPECIALTY_MISMATCH_PENALTY
        capacity_score = 0.0
        if hospital_er_occupancy >= 0.90:
            capacity_score = HOSPITAL_CAPACITY_PENALTY_PER_PCT
        travel_fraction = min(1.0, travel_time_minutes / max(1.0, golden_hour_minutes))
        travel_score = clamp_score(1.0 - travel_fraction) * 0.15
        prenotify_score = 0.0
        if cath_lab_activated:
            prenotify_score += HOSPITAL_CATH_LAB_PRENOTIFY_BONUS
        if stroke_notified:
            prenotify_score += HOSPITAL_STROKE_PRENOTIFY_BONUS
        if trauma_activated:
            prenotify_score += HOSPITAL_TRAUMA_ACTIVATION_BONUS
        raw = specialty_score + capacity_score + travel_score + prenotify_score
        return clamp_score(raw)
    @staticmethod
    def compute_unit_type_reward(
        dispatched_type: str,
        correct_type: str,
    ) -> float:
        d = UnitType(dispatched_type) if isinstance(dispatched_type, str) else dispatched_type
        c = UnitType(correct_type) if isinstance(correct_type, str) else correct_type
        if d == c:
            return UNIT_TYPE_EXACT_MATCH_BONUS
        if d.dispatch_cost > c.dispatch_cost:
            return UNIT_TYPE_UPGRADE_BONUS
        return UNIT_TYPE_DOWNGRADE_PENALTY
    @staticmethod
    def compute_task3_unit_type_score(
        dispatched_type: str,
        correct_type: str,
    ) -> float:
        d = UnitType(dispatched_type) if isinstance(dispatched_type, str) else dispatched_type
        c = UnitType(correct_type) if isinstance(correct_type, str) else correct_type
        return 1.0 if d == c else 0.0
    @staticmethod
    def compute_preposition_reward(
        actual_response_minutes: float,
        baseline_response_minutes: float,
        forecast_informed_ratio: float,
    ) -> float:
        if baseline_response_minutes <= 0.0:
            return 0.0
        improvement = max(0.0, baseline_response_minutes - actual_response_minutes)
        improvement_pct = improvement / baseline_response_minutes
        improvement_score = clamp_score(improvement_pct * 2.0)  
        forecast_bonus = forecast_informed_ratio * PREPOSITION_DEMAND_MATCH_BONUS
        return clamp_score(improvement_score + forecast_bonus)
    @staticmethod
    def compute_reroute_reward(
        time_saved_minutes: float,
        is_improvement: bool,
        diversion_triggered: bool,
        new_hospital_specialty_correct: bool,
    ) -> float:
        if not is_improvement:
            return REROUTE_NO_IMPROVEMENT_PENALTY
        time_score = min(0.30, time_saved_minutes * REROUTE_TIME_SAVED_PER_MIN)
        diversion_bonus = REROUTE_DIVERSION_BYPASS_BONUS if diversion_triggered else 0.0
        specialty_bonus = 0.10 if new_hospital_specialty_correct else 0.0
        return clamp_score(time_score + diversion_bonus + specialty_bonus)
    @staticmethod
    def compute_task5_score(
        reroutes_correct: int,
        reroutes_total: int,
        total_time_saved_minutes: float,
        baseline_time_minutes: float,
    ) -> float:
        correctness = clamp_score(reroutes_correct / max(1, reroutes_total))
        time_saved_score = clamp_score(
            total_time_saved_minutes / max(1.0, baseline_time_minutes)
        )
        return clamp_score(0.50 * correctness + 0.50 * time_saved_score)
    @staticmethod
    def compute_transfer_reward(
        is_appropriate: bool,
        is_timely: bool,
        bed_confirmed: bool,
        transfer_time_minutes: float,
        golden_window_minutes: float = 30.0,
    ) -> float:
        if not is_appropriate:
            return TRANSFER_LATE_PENALTY  
        timely_score = TRANSFER_TIMELY_BONUS if is_timely else TRANSFER_LATE_PENALTY
        bed_score = TRANSFER_BED_CONFIRMED_BONUS if bed_confirmed else 0.0
        time_score = normalise_response_time(
            transfer_time_minutes,
            golden_window_minutes,
            worst_case_minutes=120.0,
        ) * 0.05
        return clamp_score(timely_score + bed_score + time_score)
    @staticmethod
    def compute_surge_reward(
        surge_declared: bool,
        surge_justified: bool,
        surge_timely: bool,
        cascade_occurred: bool,
        mutual_aid_adequate: bool,
    ) -> float:
        if cascade_occurred:
            return SURGE_CASCADE_FAILURE_PENALTY
        if not surge_declared and surge_justified:
            return SURGE_LATE_DECLARATION_PENALTY
        reward = 0.0
        if surge_declared and surge_justified and surge_timely:
            reward += SURGE_CORRECT_DECLARATION_BONUS
        if not cascade_occurred:
            reward += SURGE_CASCADE_AVOIDANCE_BONUS
        if mutual_aid_adequate:
            reward += MUTUAL_AID_CORRECT_REQUEST_BONUS
        return clamp_score(reward)
    @staticmethod
    def compute_multi_agency_reward(
        agencies_dispatched: List[str],
        agencies_required: List[str],
        victim_trapped: bool,
    ) -> float:
        if not agencies_required:
            return 0.0
        correct_set = set(agencies_required)
        dispatched_set = set(agencies_dispatched)
        if dispatched_set >= correct_set:
            return MULTI_AGENCY_CORRECT_DISPATCH_BONUS
        if victim_trapped and "Fire" not in dispatched_set:
            return SINGLE_EMS_TRAPPED_PENALTY
        missing = correct_set - dispatched_set
        return MULTI_AGENCY_OMISSION_PENALTY * len(missing)
    @staticmethod
    def aggregate_episode_score(
        channel_scores: Dict[str, float],
        task_id: TaskID,
        protocol_bonus: float,
    ) -> float:
        weights = TASK_REWARD_WEIGHTS.get(TaskID(task_id), {})
        if not weights:
            return clamp_score(sum(channel_scores.values()) / max(1, len(channel_scores)))
        raw = 0.0
        for channel, weight in weights.items():
            raw += weight * channel_scores.get(channel, 0.0)
        capped_bonus = min(PROTOCOL_COMPLIANCE_MAX_BONUS, max(0.0, protocol_bonus))
        raw += capped_bonus
        return clamp_score(raw)
    @staticmethod
    def compute_weighted_survival_across_patients(
        patient_survival_probs: Dict[str, float],
        patient_severities: Dict[str, str],
    ) -> float:
        if not patient_survival_probs:
            return 0.0
        weight_map = {
            "P1": SURVIVAL_P1_WEIGHT,
            "P2": SURVIVAL_P2_WEIGHT,
            "P3": SURVIVAL_P3_WEIGHT,
            "P0": SURVIVAL_P0_WEIGHT,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for pid, prob in patient_survival_probs.items():
            sev = patient_severities.get(pid, "P2")
            w = weight_map.get(sev, 0.5)
            weighted_sum += clamp_score(prob) * w
            total_weight += w
        if total_weight < NORMALISATION_EPSILON:
            return 0.0
        return clamp_score(weighted_sum / total_weight)
    @staticmethod
    def compute_hospital_spread_score(
        hospitals_used: int,
        total_hospitals_available: int,
    ) -> float:
        if hospitals_used <= 1:
            return 0.0
        target_spread = min(4, total_hospitals_available)
        return clamp_score(hospitals_used / target_spread)
    @staticmethod
    def normalise_task_score(
        raw_score: float,
        task_id: TaskID,
    ) -> Score:
        return clamp_score(raw_score)
class StepRewardBuilder:
    def __init__(
        self,
        episode_id: str,
        task_id: TaskID,
        step: int,
    ) -> None:
        self._episode_id = episode_id
        self._task_id = task_id
        self._step = step
        self._survival_deltas: List[SurvivalDelta] = []
        self._protocol_events: List[ProtocolRewardEvent] = []
        self._penalty_events: List[PenaltyEvent] = []
        self._channels: Dict[str, float] = {ch: 0.0 for ch in [
            "survival", "response_time", "triage_accuracy",
            "protocol_compliance", "hospital_routing", "diversion_avoidance",
            "multi_agency_coordination", "crew_fatigue_management",
            "preposition_efficiency", "transfer_timeliness",
            "surge_management", "mutual_aid_efficiency",
            "cascade_avoidance", "comms_resilience", "noop_penalty",
        ]}
        self._dispatch_hints: List[DispatchGraderHint] = []
        self._triage_hints: List[TriageGraderHint] = []
        self._reroute_hints: List[RerouteGraderHint] = []
        self._transfer_hints: List[TransferGraderHint] = []
        self._surge_hints: List[SurgeGraderHint] = []
        self._warnings: List[str] = []
        self._commendations: List[str] = []
    def add_channel(self, channel: str, value: float) -> "StepRewardBuilder":
        if channel in self._channels:
            self._channels[channel] += value
        else:
            self._channels[channel] = value
        return self
    def add_survival_delta(self, delta: SurvivalDelta) -> "StepRewardBuilder":
        self._survival_deltas.append(delta)
        self._channels["survival"] += delta.reward_contribution
        return self
    def add_protocol_event(self, event: ProtocolRewardEvent) -> "StepRewardBuilder":
        self._protocol_events.append(event)
        self._channels["protocol_compliance"] += event.reward_delta
        return self
    def add_penalty(self, penalty: PenaltyEvent) -> "StepRewardBuilder":
        self._penalty_events.append(penalty)
        channel_map = {
            "diversion_routing": "diversion_avoidance",
            "comms_noop": "comms_resilience",
            "mutual_aid_over_request": "mutual_aid_efficiency",
            "mutual_aid_under_request": "mutual_aid_efficiency",
            "single_ems_trapped": "multi_agency_coordination",
            "noop_unaddressed_p1": "noop_penalty",
            "cascade_failure": "cascade_avoidance",
            "wrong_unit_type": "triage_accuracy",
            "wrong_hospital": "hospital_routing",
            "triage_critical_mismatch": "triage_accuracy",
        }
        ch = channel_map.get(penalty.penalty_type, "noop_penalty")
        self._channels[ch] += penalty.penalty_value
        return self
    def add_dispatch_hint(self, hint: DispatchGraderHint) -> "StepRewardBuilder":
        self._dispatch_hints.append(hint)
        return self
    def add_triage_hint(self, hint: TriageGraderHint) -> "StepRewardBuilder":
        self._triage_hints.append(hint)
        return self
    def add_reroute_hint(self, hint: RerouteGraderHint) -> "StepRewardBuilder":
        self._reroute_hints.append(hint)
        return self
    def add_transfer_hint(self, hint: TransferGraderHint) -> "StepRewardBuilder":
        self._transfer_hints.append(hint)
        return self
    def add_surge_hint(self, hint: SurgeGraderHint) -> "StepRewardBuilder":
        self._surge_hints.append(hint)
        return self
    def warn(self, message: str) -> "StepRewardBuilder":
        self._warnings.append(message)
        return self
    def commend(self, message: str) -> "StepRewardBuilder":
        self._commendations.append(message)
        return self
    def build(
        self,
        cumulative_score: float,
        prev_cumulative_score: float,
        episode_done: bool = False,
        termination_reason: Optional[str] = None,
    ) -> StepRewardSignal:
        weights = TASK_REWARD_WEIGHTS.get(self._task_id, {})
        total_step_reward = 0.0
        for ch, weight in weights.items():
            total_step_reward += weight * self._channels.get(ch, 0.0)
        hints = GraderHints(
            step=self._step,
            dispatch_hints=self._dispatch_hints,
            triage_hints=self._triage_hints,
            reroute_hints=self._reroute_hints,
            transfer_hints=self._transfer_hints,
            surge_hints=self._surge_hints,
            general_warnings=self._warnings,
            general_commendations=self._commendations,
        )
        total_penalty = sum(p.penalty_value for p in self._penalty_events)
        total_bonus = sum(
            e.reward_delta for e in self._protocol_events if e.reward_delta > 0
        )
        return StepRewardSignal(
            step=self._step,
            episode_id=self._episode_id,
            task_id=self._task_id,
            channels=dict(self._channels),
            total_step_reward=total_step_reward,
            cumulative_score=clamp_score(cumulative_score),
            score_delta=cumulative_score - prev_cumulative_score,
            survival_deltas=self._survival_deltas,
            protocol_events=self._protocol_events,
            penalty_events=self._penalty_events,
            grader_hints=hints,
            num_patients_deteriorating=sum(
                1 for d in self._survival_deltas if d.deteriorating
            ),
            num_protocol_violations=sum(
                1 for e in self._protocol_events if e.is_violation
            ),
            num_protocol_bonuses=sum(
                1 for e in self._protocol_events if not e.is_violation
            ),
            total_penalty_this_step=total_penalty,
            total_bonus_this_step=total_bonus,
            episode_done=episode_done,
            termination_reason=termination_reason,
        )
class EpisodeScoreBuilder:
    @staticmethod
    def build(
        episode_id: str,
        task_id: TaskID,
        steps_executed: int,
        channel_scores: Dict[str, float],
        protocol_bonus: float,
        penalties: EpisodePenaltySummary,
        survival_summary: SurvivalSummary,
        grader_trace: List[GraderTraceEntry],
        protocol_events: List[ProtocolRewardEvent],
        task_specific: Optional[Dict[str, Any]] = None,
        terminated_early: bool = False,
        termination_reason: Optional[str] = None,
    ) -> EpisodeScoreResult:
        t = TaskID(task_id)
        component_scores = {
            ch: clamp_score(val)
            for ch, val in channel_scores.items()
        }
        final_score_before_bonus = RewardComputationEngine.aggregate_episode_score(
            channel_scores=channel_scores,
            task_id=task_id,
            protocol_bonus=0.0,  
        )
        capped_bonus = min(PROTOCOL_COMPLIANCE_MAX_BONUS, max(0.0, protocol_bonus))
        final_score = clamp_score(final_score_before_bonus + capped_bonus)
        baseline = BaselineComparison.build(task_id, final_score)
        return EpisodeScoreResult(
            episode_id=episode_id,
            task_id=task_id,
            task_difficulty=t.difficulty,
            task_seed=t.seed,
            steps_executed=steps_executed,
            terminated_early=terminated_early,
            termination_reason=termination_reason,
            final_score=final_score,
            final_score_before_protocol_bonus=clamp_score(final_score_before_bonus),
            component_scores=component_scores,
            channel_scores={k: float(v) for k, v in channel_scores.items()},
            task_specific=task_specific,
            protocol_bonus=capped_bonus,
            penalties=penalties,
            survival_summary=survival_summary,
            grader_trace=grader_trace,
            protocol_events=protocol_events,
            baseline_comparison=baseline,
        )
    @staticmethod
    def minimal(
        episode_id: str,
        task_id: TaskID,
        steps_executed: int,
        raw_score: float,
    ) -> EpisodeScoreResult:
        task_weights = TASK_REWARD_WEIGHTS.get(TaskID(task_id), {})
        ch_scores = {ch: raw_score for ch in task_weights.keys()}
        return EpisodeScoreBuilder.build(
            episode_id=episode_id,
            task_id=task_id,
            steps_executed=steps_executed,
            channel_scores=ch_scores,
            protocol_bonus=0.0,
            penalties=EpisodePenaltySummary(),
            survival_summary=SurvivalSummary(),
            grader_trace=[],
            protocol_events=[],
            task_specific=None,
        )
class GraderResult(ImmutableEmergiModel):
    record_type: Literal["grader_result"] = "grader_result"
    grader_id: str                  = Field(
        ...,
        description="e.g. 'task1_grader', 'task7_grader'",
    )
    task_id: TaskID
    episode_id: str
    score: Score                    = Field(
        ...,
        ge=SCORE_FLOOR,
        le=SCORE_CEILING,
        description=(
            "Final grader score ALWAYS in [0.0, 1.0]. "
            "OpenEnv Phase 1 validator rejects any value outside this range."
        ),
    )
    full_result: Optional[EpisodeScoreResult] = Field(
        default=None,
        description="Full score breakdown (may be None for lightweight graders)",
    )
    deterministic_check: str        = Field(
        default="",
        description=(
            "MD5 hash of (episode_id + task_id + str(score)) for "
            "determinism verification in tests/test_graders.py"
        ),
    )
    graded_at: str                  = Field(default_factory=utc_now_iso)
    @field_validator("score")
    @classmethod
    def enforce_score_range(cls, v: float) -> float:
        clamped = clamp_score(v)
        if abs(clamped - v) > 1e-6:
            logger.warning(
                "GraderResult: score %f clamped to %f. Grader must clamp internally.",
                v, clamped,
            )
        return clamped
    @model_validator(mode="after")
    def validate_grader_result(self) -> "GraderResult":
        if not (SCORE_FLOOR <= self.score <= SCORE_CEILING):
            raise ValueError(
                f"GraderResult.score {self.score} is outside [0.0, 1.0]. "
                "All graders MUST return scores in this range."
            )
        return self
    @computed_field
    @property
    def passes_openenv_validator(self) -> bool:
        return SCORE_FLOOR <= self.score <= SCORE_CEILING
    @computed_field
    @property
    def baseline_score(self) -> float:
        return TaskID(self.task_id).baseline_score
    @computed_field
    @property
    def beats_baseline(self) -> bool:
        return self.score > self.baseline_score
    @computed_field
    @property
    def grade_label(self) -> str:
        delta = self.score - self.baseline_score
        if delta >= 0.25:
            return "EXCEPTIONAL"
        if delta >= 0.10:
            return "STRONG"
        if delta >= 0.0:
            return "ABOVE_BASELINE"
        if delta >= -0.10:
            return "NEAR_BASELINE"
        return "BELOW_BASELINE"
    @classmethod
    def from_score(
        cls,
        grader_id: str,
        task_id: TaskID,
        episode_id: str,
        score: float,
        full_result: Optional[EpisodeScoreResult] = None,
    ) -> "GraderResult":
        import hashlib
        check = hashlib.md5(
            f"{episode_id}{task_id}{score:.6f}".encode()
        ).hexdigest()
        return cls(
            grader_id=grader_id,
            task_id=task_id,
            episode_id=episode_id,
            score=clamp_score(score),
            full_result=full_result,
            deterministic_check=check,
        )
GraderFunctionType = Callable[..., GraderResult]
class GraderRegistry:
    _registry: ClassVar[Dict[TaskID, GraderFunctionType]] = {}
    @classmethod
    def register(cls, task_id: TaskID, fn: GraderFunctionType) -> None:
        cls._registry[TaskID(task_id)] = fn
        logger.debug("GraderRegistry: registered grader for %s", task_id)
    @classmethod
    def get(cls, task_id: TaskID) -> GraderFunctionType:
        t = TaskID(task_id)
        if t not in cls._registry:
            raise KeyError(
                f"GraderRegistry: no grader registered for {task_id}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[t]
    @classmethod
    def all_task_ids(cls) -> List[TaskID]:
        return sorted(cls._registry.keys(), key=lambda t: t.value)
    @classmethod
    def is_complete(cls) -> bool:
        return set(cls._registry.keys()) == set(TaskID)
    @classmethod
    def run_all(cls, **kwargs: Any) -> Dict[TaskID, GraderResult]:
        results: Dict[TaskID, GraderResult] = {}
        for t, fn in cls._registry.items():
            try:
                results[t] = fn(**kwargs)
            except Exception as exc:
                logger.error("GraderRegistry: grader for %s failed: %s", t, exc)
                results[t] = GraderResult.from_score(
                    grader_id=f"{t.value}_grader",
                    task_id=t,
                    episode_id=kwargs.get("episode_id", "unknown"),
                    score=0.0,
                )
        return results
class RewardStatistics(EmergiBaseModel):
    task_id: TaskID
    steps_counted: int              = Field(default=0, ge=0)
    channel_totals: Dict[str, float]= Field(default_factory=dict)
    channel_minimums: Dict[str, float] = Field(default_factory=dict)
    channel_maximums: Dict[str, float] = Field(default_factory=dict)
    total_reward_sum: float         = Field(default=0.0)
    total_penalty_sum: float        = Field(default=0.0)
    total_bonus_sum: float          = Field(default=0.0)
    score_trajectory: List[float]   = Field(default_factory=list)
    step_rewards: List[float]       = Field(default_factory=list, max_length=MAX_STEPS_PER_EPISODE)
    penalty_events_total: int       = Field(default=0, ge=0)
    bonus_events_total: int         = Field(default=0, ge=0)
    worst_step: Optional[int]       = Field(default=None)
    best_step: Optional[int]        = Field(default=None)
    def ingest(self, signal: StepRewardSignal) -> None:
        self.steps_counted += 1
        self.total_reward_sum += signal.total_step_reward
        self.total_penalty_sum += signal.total_penalty_this_step
        self.total_bonus_sum += signal.total_bonus_this_step
        self.penalty_events_total += signal.num_protocol_violations
        self.bonus_events_total += signal.num_protocol_bonuses
        self.score_trajectory.append(signal.cumulative_score)
        self.step_rewards.append(signal.total_step_reward)
        for ch, val in signal.channels.items():
            if ch not in self.channel_totals:
                self.channel_totals[ch] = 0.0
                self.channel_minimums[ch] = float("inf")
                self.channel_maximums[ch] = float("-inf")
            self.channel_totals[ch] += val
            self.channel_minimums[ch] = min(self.channel_minimums[ch], val)
            self.channel_maximums[ch] = max(self.channel_maximums[ch], val)
        if self.step_rewards:
            self.worst_step = int(self.step_rewards.index(min(self.step_rewards)))
            self.best_step = int(self.step_rewards.index(max(self.step_rewards)))
    @computed_field
    @property
    def avg_step_reward(self) -> float:
        if self.steps_counted == 0:
            return 0.0
        return self.total_reward_sum / self.steps_counted
    @computed_field
    @property
    def avg_step_penalty(self) -> float:
        if self.steps_counted == 0:
            return 0.0
        return self.total_penalty_sum / self.steps_counted
    @computed_field
    @property
    def current_score(self) -> float:
        return self.score_trajectory[-1] if self.score_trajectory else 0.0
    @computed_field
    @property
    def score_trend(self) -> str:
        if len(self.score_trajectory) < 5:
            return "insufficient_data"
        recent = self.score_trajectory[-5:]
        first, last = recent[0], recent[-1]
        delta = last - first
        if delta > 0.02:
            return "improving"
        if delta < -0.02:
            return "declining"
        return "stable"
    @computed_field
    @property
    def dominant_penalty_channel(self) -> Optional[str]:
        negatives = {
            ch: val for ch, val in self.channel_totals.items() if val < 0
        }
        if not negatives:
            return None
        return min(negatives, key=negatives.get)
    @computed_field
    @property
    def dominant_reward_channel(self) -> Optional[str]:
        positives = {
            ch: val for ch, val in self.channel_totals.items() if val > 0
        }
        if not positives:
            return None
        return max(positives, key=positives.get)
    def channel_summary(self) -> str:
        lines = ["RewardStatistics:"]
        for ch, total in sorted(self.channel_totals.items(), key=lambda x: abs(x[1]), reverse=True):
            avg = total / max(1, self.steps_counted)
            mn = self.channel_minimums.get(ch, 0.0)
            mx = self.channel_maximums.get(ch, 0.0)
            lines.append(
                f"  {ch:35s}: total={total:+.4f}  avg={avg:+.4f}  "
                f"min={mn:+.4f}  max={mx:+.4f}"
            )
        lines.append(f"  {'CURRENT_SCORE':35s}: {self.current_score:.4f}")
        lines.append(f"  {'TREND':35s}: {self.score_trend}")
        return "\n".join(lines)
def make_noop_step_reward(
    episode_id: str,
    task_id: TaskID,
    step: int,
    cumulative_score: float,
    prev_cumulative_score: float,
    unaddressed_p1_count: int = 0,
    units_with_comms_lost: Optional[List[str]] = None,
    episode_done: bool = False,
    termination_reason: Optional[str] = None,
) -> StepRewardSignal:
    builder = StepRewardBuilder(episode_id, TaskID(task_id), step)
    if unaddressed_p1_count > 0:
        penalty = PenaltyEvent(
            step=step,
            penalty_type="noop_unaddressed_p1",
            penalty_value=NOOP_WITH_UNADDRESSED_P1_PENALTY * unaddressed_p1_count,
            description=f"{unaddressed_p1_count} unaddressed P1 incidents during NOOP",
        )
        builder.add_penalty(penalty)
        builder.warn(f"NOOP with {unaddressed_p1_count} unaddressed P1 incidents — penalty applied.")
    if units_with_comms_lost:
        for uid in units_with_comms_lost:
            penalty = PenaltyEvent(
                step=step,
                penalty_type="comms_noop",
                penalty_value=COMMS_NOOP_ON_LOST_UNIT_PENALTY,
                unit_id=uid,
                description=f"NOOP on comms-lost unit {uid} — action required",
            )
            builder.add_penalty(penalty)
        builder.warn(
            f"NOOP with {len(units_with_comms_lost)} comms-lost units — penalty per unit."
        )
    return builder.build(
        cumulative_score=cumulative_score,
        prev_cumulative_score=prev_cumulative_score,
        episode_done=episode_done,
        termination_reason=termination_reason,
    )
ModelRegistry.register("SurvivalDelta",           SurvivalDelta)
ModelRegistry.register("SurvivalSummary",         SurvivalSummary)
ModelRegistry.register("ProtocolRewardEvent",     ProtocolRewardEvent)
ModelRegistry.register("PenaltyEvent",            PenaltyEvent)
ModelRegistry.register("DispatchGraderHint",      DispatchGraderHint)
ModelRegistry.register("TriageGraderHint",        TriageGraderHint)
ModelRegistry.register("RerouteGraderHint",       RerouteGraderHint)
ModelRegistry.register("TransferGraderHint",      TransferGraderHint)
ModelRegistry.register("SurgeGraderHint",         SurgeGraderHint)
ModelRegistry.register("GraderHints",             GraderHints)
ModelRegistry.register("StepRewardSignal",        StepRewardSignal)
ModelRegistry.register("Task1ComponentScore",     Task1ComponentScore)
ModelRegistry.register("Task2ComponentScore",     Task2ComponentScore)
ModelRegistry.register("Task3ComponentScore",     Task3ComponentScore)
ModelRegistry.register("Task4ComponentScore",     Task4ComponentScore)
ModelRegistry.register("Task5ComponentScore",     Task5ComponentScore)
ModelRegistry.register("Task6ComponentScore",     Task6ComponentScore)
ModelRegistry.register("Task7ComponentScore",     Task7ComponentScore)
ModelRegistry.register("Task8ComponentScore",     Task8ComponentScore)
ModelRegistry.register("Task9ComponentScore",     Task9ComponentScore)
ModelRegistry.register("EpisodePenaltySummary",   EpisodePenaltySummary)
ModelRegistry.register("GraderTraceEntry",        GraderTraceEntry)
ModelRegistry.register("BaselineComparison",      BaselineComparison)
ModelRegistry.register("EpisodeScoreResult",      EpisodeScoreResult)
ModelRegistry.register("GraderResult",            GraderResult)
ModelRegistry.register("RewardStatistics",        RewardStatistics)
logger.info(
    "reward.py: registered %d reward models (reward_version=%d).",
    25, __reward_version__,
)
def _self_test() -> None:
    for t in TaskID:
        weights = TASK_REWARD_WEIGHTS.get(t, {})
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights for {t} sum to {total}"
    sd = SurvivalDelta.compute(
        patient_id="P001",
        incident_id="INC-001",
        victim_index=0,
        severity=SeverityLevel.P1,
        condition_key="stemi_anterior",
        p_before=0.90,
        p_after=0.85,
        decay_model=DecayModel.SIGMOID,
        step=5,
    )
    assert sd.deteriorating
    assert abs(sd.delta - (-0.05)) < 1e-6
    assert sd.severity_weight == SURVIVAL_P1_WEIGHT
    gr = GraderResult.from_score(
        grader_id="task1_grader",
        task_id=TaskID.T1,
        episode_id="ep-test",
        score=0.75,
    )
    assert gr.passes_openenv_validator
    assert 0.0 <= gr.score <= 1.0
    gr2 = GraderResult.from_score(
        grader_id="task1_grader",
        task_id=TaskID.T1,
        episode_id="ep-test",
        score=1.5,  
    )
    assert gr2.score == 1.0, f"Expected 1.0, got {gr2.score}"
    bc = BaselineComparison.build(TaskID.T9, 0.25)
    assert bc.beats_baseline  
    assert bc.performance_tier in ("EXCEPTIONAL", "STRONG", "ABOVE_BASELINE")
    survival_r = RewardComputationEngine.compute_response_time_reward(5.0, "P1")
    assert survival_r == 1.0  
    slow_r = RewardComputationEngine.compute_response_time_reward(200.0, "P1")
    assert slow_r == 0.0  
    unit_r = RewardComputationEngine.compute_unit_type_reward("MICU", "MICU")
    assert unit_r == UNIT_TYPE_EXACT_MATCH_BONUS
    down_r = RewardComputationEngine.compute_unit_type_reward("BLS", "MICU")
    assert down_r == UNIT_TYPE_DOWNGRADE_PENALTY
    hosp_r = RewardComputationEngine.compute_hospital_routing_reward(
        hospital_has_specialty=True,
        hospital_on_diversion=False,
        hospital_er_occupancy=0.70,
        travel_time_minutes=10.0,
        golden_hour_minutes=90,
        cath_lab_activated=True,
    )
    assert hosp_r > 0.0
    triage_r = RewardComputationEngine.compute_triage_accuracy_reward(
        "Immediate", "Immediate", "P1"
    )
    assert triage_r == TRIAGE_CORRECT_TAG_BONUS
    mismatch_r = RewardComputationEngine.compute_triage_accuracy_reward(
        "Expectant", "Immediate", "P1"
    )
    assert mismatch_r == TRIAGE_IMMEDIATE_AS_EXPECTANT_PENALTY
    builder = StepRewardBuilder("ep-001", TaskID.T1, step=3)
    builder.add_channel("hospital_routing", 0.20)
    builder.add_channel("triage_accuracy", 0.30)
    signal = builder.build(
        cumulative_score=0.55,
        prev_cumulative_score=0.50,
    )
    assert 0.0 <= signal.cumulative_score <= 1.0
    assert math.isfinite(signal.total_step_reward)
    eps = EpisodePenaltySummary(
        diversion_routing_penalty=-0.30,
        wrong_unit_type_penalty=-0.20,
    )
    assert abs(eps.total_penalty - (-0.50)) < 1e-6
    result = EpisodeScoreBuilder.minimal(
        episode_id="ep-002",
        task_id=TaskID.T3,
        steps_executed=1,
        raw_score=0.68,
    )
    assert 0.0 <= result.final_score <= 1.0
    stats = RewardStatistics(task_id=TaskID.T4)
    assert stats.avg_step_reward == 0.0
    assert stats.score_trend == "insufficient_data"
    logger.debug("reward.py self-test passed.")
_self_test()
logger.info(
    "EMERGI-ENV server.models.reward v%d loaded — %d models, %d grader channels.",
    __reward_version__,
    25,
    len(TASK_REWARD_WEIGHTS[TaskID.T1]),
)
__all__ = [
    "__reward_version__",
    "TASK_REWARD_WEIGHTS",
    "SURVIVAL_P1_WEIGHT",
    "SURVIVAL_P2_WEIGHT",
    "SURVIVAL_P3_WEIGHT",
    "SURVIVAL_P0_WEIGHT",
    "TRIAGE_CORRECT_TAG_BONUS",
    "TRIAGE_IMMEDIATE_AS_EXPECTANT_PENALTY",
    "HOSPITAL_SPECIALTY_MATCH_BONUS",
    "HOSPITAL_SPECIALTY_MISMATCH_PENALTY",
    "HOSPITAL_DIVERSION_PENALTY",
    "UNIT_TYPE_EXACT_MATCH_BONUS",
    "UNIT_TYPE_DOWNGRADE_PENALTY",
    "MULTI_AGENCY_CORRECT_DISPATCH_BONUS",
    "SINGLE_EMS_TRAPPED_PENALTY",
    "SURGE_CORRECT_DECLARATION_BONUS",
    "SURGE_CASCADE_FAILURE_PENALTY",
    "GOLDEN_HOUR_MET_BONUS",
    "NOOP_WITH_UNADDRESSED_P1_PENALTY",
    "PROTOCOL_COMPLIANCE_MAX_BONUS",
    "SurvivalDelta",
    "SurvivalSummary",
    "ProtocolRewardEvent",
    "PenaltyEvent",
    "DispatchGraderHint",
    "TriageGraderHint",
    "RerouteGraderHint",
    "TransferGraderHint",
    "SurgeGraderHint",
    "GraderHints",
    "StepRewardSignal",
    "StepRewardBuilder",
    "Task1ComponentScore",
    "Task2ComponentScore",
    "Task3ComponentScore",
    "Task4ComponentScore",
    "Task5ComponentScore",
    "Task6ComponentScore",
    "Task7ComponentScore",
    "Task8ComponentScore",
    "Task9ComponentScore",
    "TaskComponentScore",
    "EpisodePenaltySummary",
    "GraderTraceEntry",
    "BaselineComparison",
    "EpisodeScoreResult",
    "EpisodeScoreBuilder",
    "GraderResult",
    "GraderRegistry",
    "GraderFunctionType",
    "RewardStatistics",
    "RewardComputationEngine",
    "make_noop_step_reward",
    "NORMALISATION_EPSILON",
]
class MultiPatientRewardAggregator:
    def __init__(
        self,
        task_id: TaskID,
        episode_id: str,
        step: int,
    ) -> None:
        self._task_id = TaskID(task_id)
        self._episode_id = episode_id
        self._step = step
        self._patient_records: List[Dict[str, Any]] = []
        self._tag_records: List[Dict[str, Any]] = []
        self._hospital_assignments: Dict[str, str] = {}  
    def record_patient(
        self,
        patient_id: str,
        incident_id: str,
        victim_index: int,
        severity: str,
        condition_key: str,
        p_before: float,
        p_after: float,
        decay_model: str,
        response_received: bool = False,
        golden_hour_elapsed_pct: float = 0.0,
    ) -> None:
        self._patient_records.append(dict(
            patient_id=patient_id,
            incident_id=incident_id,
            victim_index=victim_index,
            severity=severity,
            condition_key=condition_key,
            p_before=p_before,
            p_after=p_after,
            decay_model=decay_model,
            response_received=response_received,
            golden_hour_elapsed_pct=golden_hour_elapsed_pct,
        ))
    def record_triage_tag(
        self,
        patient_id: str,
        incident_id: str,
        assigned_tag: str,
        correct_tag: str,
        severity: str,
    ) -> None:
        self._tag_records.append(dict(
            patient_id=patient_id,
            incident_id=incident_id,
            assigned_tag=assigned_tag,
            correct_tag=correct_tag,
            severity=severity,
        ))
    def record_hospital_assignment(self, patient_id: str, hospital_id: str) -> None:
        self._hospital_assignments[patient_id] = hospital_id
    def flush(self, builder: StepRewardBuilder) -> Tuple[int, int, int]:
        for rec in self._patient_records:
            delta = SurvivalDelta.compute(
                patient_id=rec["patient_id"],
                incident_id=rec["incident_id"],
                victim_index=rec["victim_index"],
                severity=rec["severity"],
                condition_key=rec["condition_key"],
                p_before=rec["p_before"],
                p_after=rec["p_after"],
                decay_model=rec["decay_model"],
                step=self._step,
                response_received=rec["response_received"],
                golden_hour_elapsed_pct=rec["golden_hour_elapsed_pct"],
            )
            builder.add_survival_delta(delta)
        correct = wrong = critical = 0
        for tag in self._tag_records:
            reward_val = RewardComputationEngine.compute_triage_accuracy_reward(
                assigned_tag=tag["assigned_tag"],
                correct_tag=tag["correct_tag"],
                severity=tag["severity"],
            )
            if tag["assigned_tag"] == tag["correct_tag"]:
                correct += 1
            elif (
                tag["correct_tag"] == TriageTag.IMMEDIATE.value
                and tag["assigned_tag"] == TriageTag.EXPECTANT.value
            ):
                critical += 1
                wrong += 1
                builder.add_penalty(PenaltyEvent(
                    step=self._step,
                    penalty_type="triage_critical_mismatch",
                    penalty_value=TRIAGE_IMMEDIATE_AS_EXPECTANT_PENALTY,
                    incident_id=tag["incident_id"],
                    description=(
                        f"CRITICAL: patient {tag['patient_id']} tagged "
                        f"EXPECTANT but ground truth is IMMEDIATE"
                    ),
                ))
            else:
                wrong += 1
            builder.add_channel("triage_accuracy", reward_val)
        unique_hospitals = set(self._hospital_assignments.values())
        if len(unique_hospitals) >= 2:
            spread_bonus = min(
                0.10,
                len(unique_hospitals) * 0.025,
            )
            builder.add_channel("hospital_routing", spread_bonus)
            if spread_bonus > 0.05:
                builder.commend(
                    f"Good hospital spread — {len(unique_hospitals)} hospitals used "
                    f"for {len(self._hospital_assignments)} patients."
                )
        self._patient_records.clear()
        self._tag_records.clear()
        self._hospital_assignments.clear()
        return correct, wrong, critical
    @property
    def patient_count(self) -> int:
        return len(self._patient_records)
    @property
    def tag_count(self) -> int:
        return len(self._tag_records)
class RewardShaper:
    def __init__(
        self,
        task_id: TaskID,
        episode_id: str,
        rng_seed: int,
    ) -> None:
        self._task_id = TaskID(task_id)
        self._episode_id = episode_id
        self._seed = rng_seed
        self._channel_accumulators: Dict[str, float] = {
            ch: 0.0 for ch in TASK_REWARD_WEIGHTS.get(self._task_id, {}).keys()
        }
        self._protocol_bonus_accumulated: float = 0.0
        self._protocol_events_log: List[ProtocolRewardEvent] = []
        self._penalty_log: List[PenaltyEvent] = []
        self._stats = RewardStatistics(task_id=task_id)
        self._cumulative_score: float = 0.0
        self._prev_score: float = 0.0
        self._step_count: int = 0
        self._crew_hours: Dict[str, float] = {}
        self._comms_lost_units: Dict[str, int] = {}
        self._cascade_occurred: bool = False
        self._hospitals_on_diversion: set = set()
        logger.info(
            "RewardShaper: task=%s episode=%s seed=%d",
            task_id, episode_id, rng_seed,
        )
    def record_protocol_event(
        self,
        rule: str,
        compliant: bool,
        reward_delta: float,
        step: int,
        incident_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
        description: str = "",
    ) -> None:
        if compliant:
            new_total = self._protocol_bonus_accumulated + reward_delta
            capped = min(PROTOCOL_COMPLIANCE_MAX_BONUS, new_total)
            actual_delta = capped - self._protocol_bonus_accumulated
            self._protocol_bonus_accumulated = capped
        else:
            actual_delta = reward_delta
        event = ProtocolRewardEvent(
            step=step,
            rule=ProtocolRule(rule) if isinstance(rule, str) else rule,
            compliant=compliant,
            reward_delta=actual_delta,
            incident_id=incident_id,
            unit_id=unit_id,
            hospital_id=hospital_id,
            description=description,
            accumulated_bonus_after=self._protocol_bonus_accumulated,
        )
        self._protocol_events_log.append(event)
    def update_crew_hours(self, unit_id: str, delta_hours: float) -> bool:
        current = self._crew_hours.get(unit_id, 0.0)
        self._crew_hours[unit_id] = current + delta_hours
        return self._crew_hours[unit_id] >= 10.0
    def reward_crew_swap(
        self,
        unit_id: str,
        step: int,
        justified: bool,
    ) -> float:
        if justified:
            self._crew_hours[unit_id] = 0.0   
            return CREW_SWAP_JUSTIFIED_BONUS
        return CREW_SWAP_PREMATURE_PENALTY
    def mark_comms_lost(self, unit_id: str, step: int) -> None:
        self._comms_lost_units[unit_id] = step
    def mark_comms_restored(self, unit_id: str) -> None:
        self._comms_lost_units.pop(unit_id, None)
    @property
    def comms_lost_units(self) -> List[str]:
        return list(self._comms_lost_units.keys())
    def record_hospital_diversion(self, hospital_id: str) -> None:
        self._hospitals_on_diversion.add(hospital_id)
    def record_cascade_failure(self) -> None:
        self._cascade_occurred = True
    @property
    def cascade_occurred(self) -> bool:
        return self._cascade_occurred
    @property
    def diversion_count(self) -> int:
        return len(self._hospitals_on_diversion)
    def shape_step(
        self,
        step: int,
        builder: StepRewardBuilder,
        unaddressed_p1_count: int = 0,
    ) -> StepRewardSignal:
        self._step_count = step
        self._prev_score = self._cumulative_score
        if unaddressed_p1_count > 0:
            penalty = PenaltyEvent(
                step=step,
                penalty_type="noop_unaddressed_p1",
                penalty_value=NOOP_WITH_UNADDRESSED_P1_PENALTY * unaddressed_p1_count,
                description=f"{unaddressed_p1_count} P1 incidents unaddressed this step",
            )
            builder.add_penalty(penalty)
            self._penalty_log.append(penalty)
        for uid in self.comms_lost_units:
            penalty = PenaltyEvent(
                step=step,
                penalty_type="comms_noop",
                penalty_value=COMMS_NOOP_ON_LOST_UNIT_PENALTY,
                unit_id=uid,
                description=f"No action taken on comms-lost unit {uid}",
            )
            builder.add_penalty(penalty)
            self._penalty_log.append(penalty)
        weights = TASK_REWARD_WEIGHTS.get(self._task_id, {})
        step_total = 0.0
        for ch, w in weights.items():
            step_val = builder._channels.get(ch, 0.0)
            self._channel_accumulators[ch] = (
                self._channel_accumulators.get(ch, 0.0) + step_val
            )
            step_total += w * step_val
        new_score = clamp_score(self._cumulative_score + step_total)
        self._cumulative_score = new_score
        signal = builder.build(
            cumulative_score=self._cumulative_score,
            prev_cumulative_score=self._prev_score,
        )
        self._stats.ingest(signal)
        return signal
    def finalise(self) -> Tuple[Dict[str, float], float, EpisodePenaltySummary]:
        ps = EpisodePenaltySummary()
        for p in self._penalty_log:
            ptype = p.penalty_type
            if ptype == "diversion_routing":
                ps.diversion_routing_penalty += p.penalty_value
            elif ptype == "wrong_unit_type":
                ps.wrong_unit_type_penalty += p.penalty_value
            elif ptype == "wrong_hospital":
                ps.wrong_hospital_penalty += p.penalty_value
            elif ptype == "triage_critical_mismatch":
                ps.triage_critical_mismatch_penalty += p.penalty_value
            elif ptype == "single_ems_trapped":
                ps.single_ems_trapped_penalty += p.penalty_value
            elif ptype == "noop_unaddressed_p1":
                ps.noop_unaddressed_p1_penalty += p.penalty_value
            elif ptype == "comms_noop":
                ps.comms_noop_penalty += p.penalty_value
            elif ptype == "mutual_aid_over_request":
                ps.mutual_aid_over_request_penalty += p.penalty_value
            elif ptype == "cascade_failure":
                ps.cascade_failure_penalty += p.penalty_value
            elif ptype == "surge_late_declaration":
                ps.surge_late_declaration_penalty += p.penalty_value
            elif ptype == "transfer_late":
                ps.transfer_late_penalty += p.penalty_value
            elif ptype == "crew_swap_premature":
                ps.crew_swap_premature_penalty += p.penalty_value
        ps.protocol_violation_total = sum(
            e.reward_delta for e in self._protocol_events_log
            if e.reward_delta < 0
        )
        logger.info(
            "RewardShaper.finalise: episode=%s task=%s steps=%d "
            "cumulative=%.4f protocol_bonus=%.4f total_penalty=%.4f",
            self._episode_id, self._task_id, self._step_count,
            self._cumulative_score, self._protocol_bonus_accumulated,
            ps.total_penalty,
        )
        return (
            dict(self._channel_accumulators),
            self._protocol_bonus_accumulated,
            ps,
        )
    @property
    def statistics(self) -> RewardStatistics:
        return self._stats
    @property
    def cumulative_score(self) -> float:
        return self._cumulative_score
    @property
    def protocol_bonus_so_far(self) -> float:
        return self._protocol_bonus_accumulated
    def reset(self) -> None:
        for ch in self._channel_accumulators:
            self._channel_accumulators[ch] = 0.0
        self._protocol_bonus_accumulated = 0.0
        self._protocol_events_log.clear()
        self._penalty_log.clear()
        self._cumulative_score = 0.0
        self._prev_score = 0.0
        self._step_count = 0
        self._crew_hours.clear()
        self._comms_lost_units.clear()
        self._cascade_occurred = False
        self._hospitals_on_diversion.clear()
        self._stats = RewardStatistics(task_id=self._task_id)
        logger.debug("RewardShaper.reset: episode=%s", self._episode_id)
class CascadeFailureDetector:
    CASCADE_DIVERSION_THRESHOLD: int = 3
    CASCADE_SURGE_WINDOW_STEPS: int = 5
    def __init__(self) -> None:
        self._cascade_triggered_step: Optional[int] = None
        self._cascade_confirmed_step: Optional[int] = None
        self._surge_declared_step: Optional[int] = None
        self._diversion_history: List[Tuple[int, int]] = []  
        self._p1_unserved_history: List[Tuple[int, int]] = []
    def update(
        self,
        step: int,
        hospitals_on_diversion: int,
        p1_unserved: int,
        surge_declared: bool,
    ) -> Optional[float]:
        self._diversion_history.append((step, hospitals_on_diversion))
        self._p1_unserved_history.append((step, p1_unserved))
        if surge_declared and self._surge_declared_step is None:
            self._surge_declared_step = step
        if (
            self._cascade_triggered_step is None
            and hospitals_on_diversion >= self.CASCADE_DIVERSION_THRESHOLD
            and p1_unserved >= 1
        ):
            self._cascade_triggered_step = step
            logger.warning(
                "CascadeFailureDetector: cascade TRIGGERED at step %d "
                "(diversions=%d, p1_unserved=%d)",
                step, hospitals_on_diversion, p1_unserved,
            )
        if (
            self._cascade_triggered_step is not None
            and self._cascade_confirmed_step is None
            and step - self._cascade_triggered_step >= self.CASCADE_SURGE_WINDOW_STEPS
        ):
            if (
                self._surge_declared_step is None
                or self._surge_declared_step > self._cascade_triggered_step + self.CASCADE_SURGE_WINDOW_STEPS
            ):
                self._cascade_confirmed_step = step
                logger.error(
                    "CascadeFailureDetector: cascade CONFIRMED at step %d. "
                    "Surge declared at step %s. Penalty: %.2f",
                    step,
                    str(self._surge_declared_step),
                    SURGE_CASCADE_FAILURE_PENALTY,
                )
                return SURGE_CASCADE_FAILURE_PENALTY
        return None
    @property
    def cascade_confirmed(self) -> bool:
        return self._cascade_confirmed_step is not None
    @property
    def cascade_triggered(self) -> bool:
        return self._cascade_triggered_step is not None
    @property
    def surge_was_timely(self) -> bool:
        if self._cascade_triggered_step is None:
            return True  
        if self._surge_declared_step is None:
            return False
        return (
            self._surge_declared_step
            <= self._cascade_triggered_step + self.CASCADE_SURGE_WINDOW_STEPS
        )
    def score_cascade_avoidance(self) -> float:
        if self.cascade_confirmed:
            return 0.0
        if not self.cascade_triggered:
            return 1.0
        if self.surge_was_timely:
            return 0.75
        return 0.40
class SurvivalProbabilityIntegrator:
    def __init__(
        self,
        patient_id: str,
        condition_key: str,
        severity: str,
        initial_survival_prob: float,
        golden_hour_minutes: float,
    ) -> None:
        self.patient_id = patient_id
        self.condition_key = condition_key
        self.severity = severity
        self._p_current = clamp_score(initial_survival_prob)
        self._p_at_treatment: Optional[float] = None
        self._golden_hour_minutes = max(1.0, golden_hour_minutes)
        self._time_elapsed_minutes: float = 0.0
        self._treatment_received: bool = False
        self._integral: float = 0.0     
        self._step_count: int = 0
        self._golden_hour_met: bool = False
    def step(
        self,
        dt_minutes: float,
        treatment_received: bool = False,
    ) -> SurvivalDelta:
        p_before = self._p_current
        if treatment_received and not self._treatment_received:
            self._treatment_received = True
            self._p_at_treatment = self._p_current
            self._golden_hour_met = (
                self._time_elapsed_minutes <= self._golden_hour_minutes
            )
        if not self._treatment_received:
            p_new = survival_probability_delta(
                condition_key=self.condition_key,
                time_elapsed_minutes=self._time_elapsed_minutes,
                dt_minutes=dt_minutes,
                p_current=self._p_current,
            )
            self._p_current = clamp_score(p_new)
            self._time_elapsed_minutes += dt_minutes
        self._integral += self._p_current * dt_minutes
        self._step_count += 1
        elapsed_pct = min(1.0, self._time_elapsed_minutes / self._golden_hour_minutes)
        return SurvivalDelta.compute(
            patient_id=self.patient_id,
            incident_id=self.patient_id,
            victim_index=0,
            severity=self.severity,
            condition_key=self.condition_key,
            p_before=p_before,
            p_after=self._p_current,
            decay_model=DecayModel.SIGMOID,
            step=self._step_count,
            response_received=treatment_received,
            golden_hour_elapsed_pct=elapsed_pct,
        )
    @property
    def current_survival_prob(self) -> float:
        return self._p_current
    @property
    def integral(self) -> float:
        return self._integral
    @property
    def normalised_integral(self) -> float:
        perfect = self._p_current * self._step_count * TIMESTEP_MINUTES
        if perfect < NORMALISATION_EPSILON:
            return 0.0
        return clamp_score(self._integral / perfect)
    @property
    def golden_hour_met(self) -> bool:
        return self._golden_hour_met
    @property
    def time_elapsed_minutes(self) -> float:
        return self._time_elapsed_minutes
    @property
    def treatment_received(self) -> bool:
        return self._treatment_received
def compute_fleet_utilisation_bonus(
    active_units: int,
    total_units: int,
    p1_pending: int,
) -> float:
    if total_units == 0:
        return 0.0
    utilisation = active_units / total_units
    if p1_pending > 0 and utilisation < 0.50:
        return max(-0.05, -0.01 * p1_pending)
    if p1_pending == 0:
        return 0.0
    return 0.0  
def compute_response_time_benchmark_score(
    actual_response_minutes: float,
    target_minutes: float,
    severity: str,
) -> Score:
    if actual_response_minutes <= target_minutes:
        return 1.0
    overshoot = actual_response_minutes - target_minutes
    decay_range = target_minutes  
    score = 1.0 - (overshoot / decay_range)
    return clamp_score(score)
def compute_preposition_coverage_score(
    zones_with_unit: List[str],
    all_active_zones: List[str],
    demand_weights: Dict[str, float],
) -> Score:
    if not all_active_zones:
        return 0.0
    total_demand = sum(demand_weights.get(z, 1.0) for z in all_active_zones)
    if total_demand < NORMALISATION_EPSILON:
        return clamp_score(len(zones_with_unit) / len(all_active_zones))
    covered_demand = sum(
        demand_weights.get(z, 1.0)
        for z in zones_with_unit
        if z in all_active_zones
    )
    return clamp_score(covered_demand / total_demand)
def compute_mutual_aid_efficiency_score(
    requests_sent: int,
    requests_justified: int,
    requests_timely: int,
    over_requests: int,
) -> Score:
    if requests_sent == 0:
        return 0.5   
    accuracy = clamp_score(requests_justified / requests_sent)
    timeliness = clamp_score(requests_timely / max(1, requests_justified))
    over_penalty = min(0.30, over_requests * abs(MUTUAL_AID_OVER_REQUEST_PENALTY))
    raw = 0.60 * accuracy + 0.40 * timeliness - over_penalty
    return clamp_score(raw)
class RewardCheckpoint(ImmutableEmergiModel):
    record_type: Literal["reward_checkpoint"] = "reward_checkpoint"
    episode_id: str
    task_id: TaskID
    step: StepNumber
    channel_accumulators: Dict[str, float]
    protocol_bonus_so_far: float    = Field(ge=0.0, le=PROTOCOL_COMPLIANCE_MAX_BONUS)
    cumulative_score: Score         = Field(ge=SCORE_FLOOR, le=SCORE_CEILING)
    diversion_count: int            = Field(default=0, ge=0)
    cascade_triggered: bool         = Field(default=False)
    comms_lost_count: int           = Field(default=0, ge=0)
    crew_fatigue_units: List[str]   = Field(default_factory=list)
    checkpointed_at: str            = Field(default_factory=utc_now_iso)
    @classmethod
    def capture(cls, shaper: RewardShaper, step: int) -> "RewardCheckpoint":
        return cls(
            episode_id=shaper._episode_id,
            task_id=shaper._task_id,
            step=step,
            channel_accumulators=dict(shaper._channel_accumulators),
            protocol_bonus_so_far=shaper.protocol_bonus_so_far,
            cumulative_score=shaper.cumulative_score,
            diversion_count=shaper.diversion_count,
            cascade_triggered=shaper.cascade_occurred,
            comms_lost_count=len(shaper.comms_lost_units),
            crew_fatigue_units=[
                uid for uid, hrs in shaper._crew_hours.items() if hrs >= 10.0
            ],
        )
ModelRegistry.register("RewardCheckpoint",               RewardCheckpoint)
ModelRegistry.register("MultiPatientRewardAggregator",   MultiPatientRewardAggregator)
ModelRegistry.register("SurvivalProbabilityIntegrator",  SurvivalProbabilityIntegrator)
ModelRegistry.register("CascadeFailureDetector",         CascadeFailureDetector)
ModelRegistry.register("RewardShaper",                   RewardShaper)
RewardBreakdown = StepRewardSignal
StepReward      = StepRewardSignal
logger.info(
    "reward.py: 5 additional runtime classes registered "
    "(RewardShaper, MultiPatientRewardAggregator, SurvivalProbabilityIntegrator, "
    "CascadeFailureDetector, RewardCheckpoint)."
)

RewardModel = StepRewardSignal
EpisodeReward = EpisodeScoreResult
__all__ += [
    "RewardShaper",
    "MultiPatientRewardAggregator",
    "SurvivalProbabilityIntegrator",
    "RewardBreakdown",
    "StepReward",
    "CascadeFailureDetector",
    "RewardCheckpoint",
    "compute_fleet_utilisation_bonus",
    "compute_response_time_benchmark_score",
    "compute_preposition_coverage_score",
    "compute_mutual_aid_efficiency_score",
]