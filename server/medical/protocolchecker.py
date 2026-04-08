from __future__ import annotations
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from server.medical.triage import (
    TriageTag,
    TriageGroundTruthDatabase,
    TriageProtocol,
    RPMScore,
    RespirationStatus,
    PulseStatus,
    MentalStatus,
    IndianEMSTriageMapper,
    MCITriageSupervisor,
    NACAScore,
    RevisedTraumaScore,
    TRIAGE_CRITICAL_MISMATCH_PENALTY,
    TRIAGE_EXACT_MATCH_SCORE,
    TRIAGE_ADJACENT_MISMATCH_SCORE,
    TRIAGE_DISTANT_MISMATCH_SCORE,
    DETERIORATION_STEPS,
    CbrnContaminationType,
    PAEDIATRIC_AGE_THRESHOLD,
    TRIAGE_VERSION,
)
from server.medical.survivalcurves import (
    SurvivalCurveRegistry,
    SeverityTier,
    TransportUrgencyClass,
    GoldenHourWindows,
    GoldenWindowType,
    SurvivalProbabilityCalculator,
    SURVIVAL_CURVES_VERSION,
    SIMULATION_STEP_DURATION_MIN,
    SURVIVAL_FLOOR,
    GOLDEN_HOUR_COMPLIANCE_BONUS,
)
from server.medical.goldenhour import (
    ConditionGoldenHourPolicyRegistry,
    DispatchQualityGrade,
    GoldenHourPhase,
    DISPATCH_LATENCY_PENALTY_PER_MIN,
    WRONG_UNIT_TYPE_REWARD_MULTIPLIER,
    CORRECT_UNIT_TYPE_BONUS,
    GOLDEN_WINDOW_BREACHED_PENALTY,
    P1_UNASSIGNED_STEP_PENALTY,
    PLATINUM_10_BONUS,
    MCI_MIN_HOSPITAL_SPREAD,
    CATH_LAB_PRENOTIFICATION_BONUS,
    STROKE_UNIT_PRENOTIFICATION_BONUS,
    TRAUMA_ACTIVATION_BONUS,
    DIVERSION_ROUTING_PENALTY,
    MULTI_AGENCY_OMISSION_PENALTY,
    GOLDEN_HOUR_VERSION,
)
from server.medical.traumascoring import (
    InjuryMechanism,
    TraumaTeamActivationLevel,
    ISS_MAJOR_TRAUMA_THRESHOLD,
    DCS_BASE_DEFICIT_THRESHOLD,
    DCS_HYPOTHERMIA_THRESHOLD_C,
    DCS_COAGULOPATHY_INR,
    INDIA_URBAN_EMS_TARGET_MIN,
    INDIA_RURAL_EMS_TARGET_MIN,
    TRAUMA_SCORING_VERSION,
)
logger = logging.getLogger("emergi_env.medical.protocol_checker")
PROTOCOL_CHECKER_VERSION: int = 5
CALL_TO_DISPATCH_TARGET_MIN: float         = 2.0    
CALL_TO_DISPATCH_MAX_ACCEPTABLE_MIN: float = 5.0
SCENE_TIME_LOAD_AND_GO_MAX_MIN: float      = 10.0   
SCENE_TIME_STAY_AND_PLAY_MAX_MIN: float    = 20.0   
SCENE_TIME_ABSOLUTE_MAX_MIN: float         = 30.0   
CREW_FATIGUE_THRESHOLD_HOURS: float        = 10.0   
CREW_WARN_THRESHOLD_HOURS: float           = 8.0
CREW_MANDATORY_SWAP_HOURS: float           = 12.0   
HOSPITAL_BYPASS_DELAY_ACCEPTABLE_MIN: float= 12.0   
TRANSFER_GOLDEN_WINDOW_MIN: float          = 30.0
TRANSFER_MAX_STABLE_DELAY_MIN: float       = 90.0
MUTUAL_AID_OVER_REQUEST_OWN_FLEET_THRESHOLD: int = 2  
MCI_TRIAGE_MUST_START_WITHIN_STEPS: int    = 2       
MCI_COMMAND_POST_WITHIN_STEPS: int         = 3
SURGE_DECLARATION_LATE_THRESHOLD_STEPS: int= 5       
COMMS_RESTORE_PRIORITY_STEPS: int          = 3       
CALL_TO_DISPATCH_OPTIMAL_BONUS: float      =  0.020
CALL_TO_DISPATCH_LATE_PENALTY: float       = -0.015
SCENE_TIME_VIOLATION_PENALTY: float        = -0.018
WRONG_UNIT_DISPATCH_PROTOCOL_PENALTY: float= -0.200
CRITICAL_WRONG_UNIT_PENALTY: float         = -0.350   
CATH_LAB_OMISSION_PENALTY: float           = -0.022
STROKE_UNIT_OMISSION_PENALTY: float        = -0.018
TRAUMA_ACTIVATION_OMISSION_PENALTY: float  = -0.018
DIVERSION_ROUTE_PROTOCOL_PENALTY: float    = -0.030
SPECIALTY_MISMATCH_PENALTY: float          = -0.025
SPECIALTY_CORRECT_BONUS: float             =  0.015
TRANSFER_TIMING_VIOLATION_PENALTY: float   = -0.010
TRANSFER_NO_BED_CONFIRM_PENALTY: float     = -0.008
TRANSFER_NO_TEAM_NOTIFY_PENALTY: float     = -0.008
MUTUAL_AID_OVER_REQUEST_PENALTY: float     = -0.010
MUTUAL_AID_UNDER_REQUEST_PENALTY: float    = -0.020  
MCI_NO_COMMAND_POST_PENALTY: float         = -0.025
MCI_NO_START_TRIAGE_PENALTY: float         = -0.035
MCI_SINGLE_HOSPITAL_ROUTING_PENALTY: float = -0.080
MCI_VICTIM_UNDERTRIAGE_PENALTY: float      = -0.050  
SURGE_LATE_DECLARATION_PENALTY: float      = -0.120
SURGE_PREMATURE_DECLARATION_PENALTY: float = -0.050
SURGE_CORRECT_DECLARATION_BONUS: float     =  0.080
CASCADE_AVOIDANCE_BONUS: float             =  0.200
POLICE_OMISSION_TRAPPED_PENALTY: float     = -0.025
FIRE_OMISSION_TRAPPED_PENALTY: float       = -0.025
NDRF_OMISSION_MAJOR_DISASTER_PENALTY: float= -0.040
MULTI_AGENCY_CORRECT_BONUS: float          =  0.015
DNR_ESCALATION_PENALTY: float              = -0.050  
COMMS_LOST_NOOP_PENALTY: float             = -0.005  
CREW_FATIGUE_DEPLOY_PENALTY: float         = -0.020  
CREW_SWAP_CORRECT_BONUS: float             =  0.020
PREPOSITION_DEMAND_MATCH_BONUS: float      =  0.040
PREPOSITION_WASTEFUL_PENALTY: float        = -0.020
PR_MICU_FOR_STEMI: str                = "micu_for_stemi"
PR_ALS_FOR_STROKE: str                = "als_for_stroke"
PR_LEVEL1_TRAUMA_30MIN: str           = "nearest_level1_trauma_within_30_min"
PR_NO_DIVERSION_ROUTE: str            = "no_routing_to_diverted_hospital"
PR_MULTI_AGENCY_TRAPPED: str          = "multi_agency_for_trapped_victim"
PR_CATH_LAB_STEMI: str                = "cath_lab_activation_for_stemi"
PR_STROKE_UNIT: str                   = "stroke_unit_for_stroke"
PR_BURNS_UNIT: str                    = "burns_unit_for_major_burns"
PR_PAEDS_ED: str                      = "paediatric_ed_for_paediatric_emergency"
PR_NOOP_UNKNOWN_UNIT: str             = "noop_on_unknown_unit_penalty"
PR_START_TRIAGE_MCI: str              = "start_triage_in_mci"
PR_MUTUAL_AID_SURGE: str              = "mutual_aid_in_surge"
STEMI_CONDITIONS: FrozenSet[str] = frozenset({
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
    "complete_heart_block", "wpw_svt",
})
STROKE_CONDITIONS: FrozenSet[str] = frozenset({
    "ischemic_stroke", "ischemic_stroke_wake_up",
    "hemorrhagic_stroke_sah", "paediatric_stroke",
    "meningitis_cryptococcal",
})
MAJOR_TRAUMA_CONDITIONS: FrozenSet[str] = frozenset({
    "polytrauma_blunt", "polytrauma_penetrating", "severe_tbi",
    "penetrating_chest", "mci_rta", "blast_injury", "mine_collapse",
    "mci_natural_disaster", "crush_syndrome",
})
BURNS_CONDITIONS: FrozenSet[str] = frozenset({
    "burns_major", "burns_moderate", "burns_electrical", "chemical_burns",
})
PAEDIATRIC_CONDITIONS: FrozenSet[str] = frozenset({
    "paediatric_respiratory", "paediatric_choking", "paediatric_ingestion",
    "paediatric_head_trauma", "paediatric_stroke", "paediatric_poisoning",
    "status_epilepticus", "febrile_seizure", "neonatal_resuscitation",
    "neonatal_emergency", "meningococcal_sepsis", "meningococcal_paediatric",
    "intussusception", "sepsis_malnutrition", "sickle_cell_crisis",
    "dengue_severe", "non_accidental_injury", "cardiac_arrest_vf_paediatric",
})
OBSTETRIC_CONDITIONS: FrozenSet[str] = frozenset({
    "obstetric_hemorrhage", "eclampsia", "uterine_rupture",
    "obstetric_trauma", "normal_delivery", "precipitate_delivery",
    "neonatal_emergency",
})
CARDIAC_ARREST_CONDITIONS: FrozenSet[str] = frozenset({
    "cardiac_arrest_vf", "cardiac_arrest_pea",
    "stemi_with_vf_arrest", "lightning_strike",
    "cardiac_arrest_vf_paediatric",
})
CBRN_CONDITIONS: FrozenSet[str] = frozenset({
    "cbrn_suspected", "organophosphate_poisoning", "phosphine_poisoning",
    "toxic_inhalation", "carbon_monoxide", "chemical_burns",
    "botulism_outbreak",
})
MCI_CONDITIONS: FrozenSet[str] = frozenset({
    "mci_rta", "blast_injury", "mine_collapse",
    "mci_natural_disaster", "mine_rescue_standby",
})
CONDITIONS_REQUIRING_MICU: FrozenSet[str] = frozenset({
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
    "cardiac_arrest_vf", "cardiac_arrest_pea",
    "polytrauma_blunt", "severe_tbi", "aortic_dissection", "aaa_rupture",
    "obstetric_hemorrhage", "eclampsia", "uterine_rupture",
    "complete_heart_block", "wpw_svt", "hyperkalaemia",
    "acute_pulmonary_oedema", "crush_syndrome", "massive_haemoptysis",
    "phosphine_poisoning", "blast_injury", "mci_rta",
    "neonatal_resuscitation", "neonatal_emergency", "weil_disease_severe",
    "lightning_strike",
})
CONDITIONS_REQUIRING_POLICE_FIRE: FrozenSet[str] = frozenset({
    "polytrauma_blunt", "mci_rta", "blast_injury", "mine_collapse",
    "chemical_burns", "organophosphate_poisoning", "cbrn_suspected",
})
CONDITIONS_REQUIRING_NDRF: FrozenSet[str] = frozenset({
    "mci_natural_disaster", "mine_collapse", "mine_rescue_standby",
    "blast_injury",
})
CONDITIONS_CATH_LAB_MANDATORY: FrozenSet[str] = frozenset({
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
    "cardiac_arrest_vf", "cardiac_arrest_pea", "complete_heart_block",
    "wpw_svt",
})
CONDITIONS_STROKE_UNIT_MANDATORY: FrozenSet[str] = frozenset({
    "ischemic_stroke", "ischemic_stroke_wake_up",
    "hemorrhagic_stroke_sah", "paediatric_stroke",
})
CONDITIONS_TRAUMA_ACTIVATION_MANDATORY: FrozenSet[str] = frozenset({
    "polytrauma_blunt", "polytrauma_penetrating", "severe_tbi",
    "mci_rta", "blast_injury", "crush_syndrome",
    "chest_trauma", "splenic_laceration",
})
@unique
class ProtocolDomain(str, Enum):
    DISPATCH          = "dispatch"
    TRIAGE            = "triage"
    HOSPITAL_ROUTING  = "hospital_routing"
    TRANSFER          = "transfer"
    MUTUAL_AID        = "mutual_aid"
    SURGE             = "surge"
    MULTI_AGENCY      = "multi_agency"
    CREW_FATIGUE      = "crew_fatigue"
    COMMS             = "communications"
    PREPOSITION       = "prepositioning"
    DOCUMENTATION     = "documentation"
    SCENE_MANAGEMENT  = "scene_management"
    CBRN              = "cbrn"
    DNR               = "dnr_ethics"
@unique
class ViolationSeverity(str, Enum):
    CRITICAL  = "critical"   
    MAJOR     = "major"      
    MINOR     = "minor"      
    ADVISORY  = "advisory"   
@unique
class ComplianceOutcome(str, Enum):
    COMPLIANT            = "compliant"
    VIOLATION            = "violation"
    PARTIAL_COMPLIANCE   = "partial_compliance"
    NOT_APPLICABLE       = "not_applicable"
    INSUFFICIENT_DATA    = "insufficient_data"
    OVERRIDE_JUSTIFIED   = "override_justified"   
@unique
class UnitTypeRank(int, Enum):
    BLS  = 1
    ALS  = 2
    MICU = 3
@unique
class SceneTimeClass(str, Enum):
    WITHIN_TARGET  = "within_target"
    ACCEPTABLE     = "acceptable"
    PROLONGED      = "prolonged"
    EXCESSIVE      = "excessive"   
    JUSTIFIED      = "justified"   
@unique
class ProtocolCheckCategory(str, Enum):
    LIFE_CRITICAL      = "life_critical"      
    TIME_SENSITIVE     = "time_sensitive"     
    SAFETY             = "safety"             
    COORDINATION       = "coordination"       
    DOCUMENTATION      = "documentation"      
    ETHICS             = "ethics"             
@dataclass(frozen=True)
class ProtocolCheckResult:
    check_id:           str
    rule_key:           str
    domain:             ProtocolDomain
    category:           ProtocolCheckCategory
    outcome:            ComplianceOutcome
    violation_severity: Optional[ViolationSeverity]
    reward_delta:       float               
    description:        str                 
    clinical_rationale: str                 
    recommendation:     str                 
    condition_key:      Optional[str]
    incident_id:        Optional[str]
    unit_id:            Optional[str]
    hospital_id:        Optional[str]
    step:               int
    elapsed_minutes:    float
    observed_value:     Optional[str]       = None   
    expected_value:     Optional[str]       = None   
    confidence:         float               = 1.0    
    agent_override_reason: Optional[str]   = None   
    automatically_waivable: bool           = False  
    @property
    def is_violation(self) -> bool:
        return self.outcome == ComplianceOutcome.VIOLATION
    @property
    def is_compliant(self) -> bool:
        return self.outcome in (
            ComplianceOutcome.COMPLIANT,
            ComplianceOutcome.OVERRIDE_JUSTIFIED,
        )
    @property
    def is_critical_violation(self) -> bool:
        return (
            self.is_violation
            and self.violation_severity == ViolationSeverity.CRITICAL
        )
    @property
    def penalty_magnitude(self) -> float:
        return abs(min(0.0, self.reward_delta))
    @property
    def bonus_magnitude(self) -> float:
        return max(0.0, self.reward_delta)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id":           self.check_id,
            "rule_key":           self.rule_key,
            "domain":             self.domain.value,
            "category":           self.category.value,
            "outcome":            self.outcome.value,
            "severity":           self.violation_severity.value if self.violation_severity else None,
            "reward_delta":       round(self.reward_delta, 4),
            "description":        self.description,
            "recommendation":     self.recommendation,
            "condition_key":      self.condition_key,
            "incident_id":        self.incident_id,
            "unit_id":            self.unit_id,
            "hospital_id":        self.hospital_id,
            "step":               self.step,
            "elapsed_min":        round(self.elapsed_minutes, 2),
            "observed":           self.observed_value,
            "expected":           self.expected_value,
        }
@dataclass
class ProtocolCheckBatch:
    batch_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    step:         int   = 0
    incident_id:  Optional[str] = None
    results:      List[ProtocolCheckResult] = field(default_factory=list)
    def add(self, result: ProtocolCheckResult) -> None:
        self.results.append(result)
    @property
    def total_reward_delta(self) -> float:
        return sum(r.reward_delta for r in self.results)
    @property
    def violation_count(self) -> int:
        return sum(1 for r in self.results if r.is_violation)
    @property
    def critical_violation_count(self) -> int:
        return sum(1 for r in self.results if r.is_critical_violation)
    @property
    def compliant_count(self) -> int:
        return sum(1 for r in self.results if r.is_compliant)
    @property
    def compliance_rate(self) -> float:
        total = len(self.results)
        if total == 0:
            return 1.0
        applicable = [
            r for r in self.results
            if r.outcome != ComplianceOutcome.NOT_APPLICABLE
        ]
        if not applicable:
            return 1.0
        return sum(1 for r in applicable if r.is_compliant) / len(applicable)
    @property
    def has_critical_violations(self) -> bool:
        return self.critical_violation_count > 0
    def violations_by_domain(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.results:
            if r.is_violation:
                k = r.domain.value
                counts[k] = counts.get(k, 0) + 1
        return counts
    def top_penalty(self) -> Optional[ProtocolCheckResult]:
        violations = [r for r in self.results if r.is_violation]
        if not violations:
            return None
        return min(violations, key=lambda r: r.reward_delta)
    def summary_str(self) -> str:
        return (
            f"Batch {self.batch_id[:8]} | step={self.step} | "
            f"checks={len(self.results)} | "
            f"violations={self.violation_count} (critical={self.critical_violation_count}) | "
            f"Δreward={self.total_reward_delta:+.4f}"
        )
@dataclass
class EpisodeProtocolRecord:
    episode_id:  str
    task_id:     str
    batches:     List[ProtocolCheckBatch]      = field(default_factory=list)
    rule_correct_counts:    Dict[str, int]    = field(default_factory=dict)
    rule_violation_counts:  Dict[str, int]    = field(default_factory=dict)
    rule_reward_totals:     Dict[str, float]  = field(default_factory=dict)
    domain_compliance_rates: Dict[str, float] = field(default_factory=dict)
    domain_reward_totals:    Dict[str, float] = field(default_factory=dict)
    critical_violations:    List[ProtocolCheckResult] = field(default_factory=list)
    total_checks:           int   = 0
    total_violations:       int   = 0
    total_critical:         int   = 0
    total_compliant:        int   = 0
    total_not_applicable:   int   = 0
    total_bonus_earned:     float = 0.0
    total_penalty_incurred: float = 0.0
    total_reward_delta:     float = 0.0
    mci_command_post_established:  bool = False
    start_triage_initiated:        bool = False
    surge_declared_correctly:      bool = False
    cascade_failure_prevented:     bool = False
    all_stemi_cath_labs_activated: bool = True
    all_strokes_stroke_unit_notified: bool = True
    any_diversion_violation:       bool = False
    any_wrong_unit_critical:       bool = False
    def ingest_batch(self, batch: ProtocolCheckBatch) -> None:
        self.batches.append(batch)
        for r in batch.results:
            self.total_checks += 1
            rk = r.rule_key
            if r.outcome == ComplianceOutcome.NOT_APPLICABLE:
                self.total_not_applicable += 1
                continue
            if r.is_compliant:
                self.total_compliant += 1
                self.rule_correct_counts[rk] = self.rule_correct_counts.get(rk, 0) + 1
            elif r.is_violation:
                self.total_violations += 1
                self.rule_violation_counts[rk] = self.rule_violation_counts.get(rk, 0) + 1
                if r.is_critical_violation:
                    self.total_critical += 1
                    self.critical_violations.append(r)
            self.rule_reward_totals[rk] = (
                self.rule_reward_totals.get(rk, 0.0) + r.reward_delta
            )
            dom = r.domain.value
            self.domain_reward_totals[dom] = (
                self.domain_reward_totals.get(dom, 0.0) + r.reward_delta
            )
            if r.reward_delta > 0:
                self.total_bonus_earned += r.reward_delta
            elif r.reward_delta < 0:
                self.total_penalty_incurred += r.reward_delta
            self.total_reward_delta += r.reward_delta
    @property
    def overall_compliance_rate(self) -> float:
        applicable = self.total_checks - self.total_not_applicable
        if applicable == 0:
            return 1.0
        return min(1.0, self.total_compliant / applicable)
    @property
    def net_protocol_reward(self) -> float:
        return self.total_reward_delta
    @property
    def critical_violation_rate(self) -> float:
        if self.total_violations == 0:
            return 0.0
        return self.total_critical / self.total_violations
    def per_rule_summary(self) -> Dict[str, Dict[str, Any]]:
        all_rules = set(self.rule_correct_counts) | set(self.rule_violation_counts)
        out: Dict[str, Dict[str, Any]] = {}
        for rule in sorted(all_rules):
            correct   = self.rule_correct_counts.get(rule, 0)
            violated  = self.rule_violation_counts.get(rule, 0)
            total     = correct + violated
            out[rule] = {
                "correct":          correct,
                "violated":         violated,
                "total":            total,
                "compliance_rate":  round(correct / total, 4) if total > 0 else 1.0,
                "net_reward":       round(self.rule_reward_totals.get(rule, 0.0), 4),
            }
        return out
    def final_report(self) -> Dict[str, Any]:
        return {
            "episode_id":              self.episode_id,
            "task_id":                 self.task_id,
            "total_checks":            self.total_checks,
            "total_violations":        self.total_violations,
            "total_critical":          self.total_critical,
            "total_compliant":         self.total_compliant,
            "overall_compliance_rate": round(self.overall_compliance_rate, 4),
            "net_protocol_reward":     round(self.net_protocol_reward, 4),
            "total_bonus":             round(self.total_bonus_earned, 4),
            "total_penalty":           round(self.total_penalty_incurred, 4),
            "mci_command_post":        self.mci_command_post_established,
            "start_triage_initiated":  self.start_triage_initiated,
            "surge_declared_correctly": self.surge_declared_correctly,
            "cascade_prevented":       self.cascade_failure_prevented,
            "diversion_violations":    self.any_diversion_violation,
            "critical_wrong_unit":     self.any_wrong_unit_critical,
            "per_rule_summary":        self.per_rule_summary(),
            "domain_rewards":          {
                k: round(v, 4) for k, v in self.domain_reward_totals.items()
            },
        }
def _make_result(
    rule_key:           str,
    domain:             ProtocolDomain,
    category:           ProtocolCheckCategory,
    outcome:            ComplianceOutcome,
    violation_severity: Optional[ViolationSeverity],
    reward_delta:       float,
    description:        str,
    clinical_rationale: str,
    recommendation:     str,
    step:               int,
    elapsed_minutes:    float,
    condition_key:      Optional[str]   = None,
    incident_id:        Optional[str]   = None,
    unit_id:            Optional[str]   = None,
    hospital_id:        Optional[str]   = None,
    observed_value:     Optional[str]   = None,
    expected_value:     Optional[str]   = None,
    confidence:         float           = 1.0,
    automatically_waivable: bool        = False,
) -> ProtocolCheckResult:
    return ProtocolCheckResult(
        check_id=str(uuid.uuid4()),
        rule_key=rule_key,
        domain=domain,
        category=category,
        outcome=outcome,
        violation_severity=violation_severity,
        reward_delta=reward_delta,
        description=description,
        clinical_rationale=clinical_rationale,
        recommendation=recommendation,
        condition_key=condition_key,
        incident_id=incident_id,
        unit_id=unit_id,
        hospital_id=hospital_id,
        step=step,
        elapsed_minutes=elapsed_minutes,
        observed_value=observed_value,
        expected_value=expected_value,
        confidence=confidence,
        automatically_waivable=automatically_waivable,
    )
def _compliant(
    rule_key:           str,
    domain:             ProtocolDomain,
    category:           ProtocolCheckCategory,
    reward_delta:       float,
    description:        str,
    clinical_rationale: str,
    step:               int,
    elapsed_minutes:    float,
    **kwargs: Any,
) -> ProtocolCheckResult:
    return _make_result(
        rule_key=rule_key,
        domain=domain,
        category=category,
        outcome=ComplianceOutcome.COMPLIANT,
        violation_severity=None,
        reward_delta=reward_delta,
        description=description,
        clinical_rationale=clinical_rationale,
        recommendation="",
        step=step,
        elapsed_minutes=elapsed_minutes,
        **kwargs,
    )
def _violation(
    rule_key:           str,
    domain:             ProtocolDomain,
    category:           ProtocolCheckCategory,
    severity:           ViolationSeverity,
    reward_delta:       float,
    description:        str,
    clinical_rationale: str,
    recommendation:     str,
    step:               int,
    elapsed_minutes:    float,
    **kwargs: Any,
) -> ProtocolCheckResult:
    return _make_result(
        rule_key=rule_key,
        domain=domain,
        category=category,
        outcome=ComplianceOutcome.VIOLATION,
        violation_severity=severity,
        reward_delta=reward_delta,
        description=description,
        clinical_rationale=clinical_rationale,
        recommendation=recommendation,
        step=step,
        elapsed_minutes=elapsed_minutes,
        **kwargs,
    )
def _na(
    rule_key:        str,
    domain:          ProtocolDomain,
    category:        ProtocolCheckCategory,
    step:            int,
    elapsed_minutes: float,
    reason:          str = "Not applicable for this condition / context.",
) -> ProtocolCheckResult:
    return _make_result(
        rule_key=rule_key,
        domain=domain,
        category=category,
        outcome=ComplianceOutcome.NOT_APPLICABLE,
        violation_severity=None,
        reward_delta=0.0,
        description=reason,
        clinical_rationale="",
        recommendation="",
        step=step,
        elapsed_minutes=elapsed_minutes,
    )
class DispatchProtocolChecker:
    _UNIT_RANK: ClassVar[Dict[str, int]] = {
        "BLS": UnitTypeRank.BLS,
        "ALS": UnitTypeRank.ALS,
        "MICU": UnitTypeRank.MICU,
    }
    def check_unit_type(
        self,
        condition_key:       str,
        dispatched_unit_type: str,
        severity:            str,
        step:                int,
        elapsed_minutes:     float,
        incident_id:         Optional[str] = None,
        unit_id:             Optional[str] = None,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        unit = dispatched_unit_type.upper()
        params = SurvivalCurveRegistry.get(condition_key)
        correct_units = params.correct_unit_types
        if condition_key in STEMI_CONDITIONS:
            if "MICU" in correct_units:
                if unit == "MICU":
                    results.append(_compliant(
                        rule_key=PR_MICU_FOR_STEMI,
                        domain=ProtocolDomain.DISPATCH,
                        category=ProtocolCheckCategory.LIFE_CRITICAL,
                        reward_delta=CATH_LAB_PRENOTIFICATION_BONUS,
                        description=f"MICU correctly dispatched for {condition_key}.",
                        clinical_rationale=(
                            "ESC 2023 STEMI guidelines mandate MICU-level care for "
                            "en-route 12-lead ECG, IV access, defibrillation, and "
                            "advanced pharmacological management."
                        ),
                        step=step, elapsed_minutes=elapsed_minutes,
                        condition_key=condition_key, incident_id=incident_id,
                        unit_id=unit_id, observed_value=unit, expected_value="MICU",
                    ))
                elif unit == "BLS":
                    results.append(_violation(
                        rule_key=PR_MICU_FOR_STEMI,
                        domain=ProtocolDomain.DISPATCH,
                        category=ProtocolCheckCategory.LIFE_CRITICAL,
                        severity=ViolationSeverity.CRITICAL,
                        reward_delta=CRITICAL_WRONG_UNIT_PENALTY,
                        description=(
                            f"BLS dispatched for {condition_key} — MICU mandatory. "
                            f"Patient lacks 12-lead ECG, IV access, and defibrillation capability."
                        ),
                        clinical_rationale=(
                            "BLS cannot perform 12-lead ECG, IV pharmacotherapy, or "
                            "transcutaneous pacing. ESC 2023: door-to-balloon ≤90 min requires "
                            "pre-hospital MICU activation."
                        ),
                        recommendation="Immediately escalate to MICU dispatch.",
                        step=step, elapsed_minutes=elapsed_minutes,
                        condition_key=condition_key, incident_id=incident_id,
                        unit_id=unit_id, observed_value="BLS", expected_value="MICU",
                    ))
                else:  
                    results.append(_violation(
                        rule_key=PR_MICU_FOR_STEMI,
                        domain=ProtocolDomain.DISPATCH,
                        category=ProtocolCheckCategory.LIFE_CRITICAL,
                        severity=ViolationSeverity.MAJOR,
                        reward_delta=WRONG_UNIT_DISPATCH_PROTOCOL_PENALTY,
                        description=(
                            f"ALS dispatched for {condition_key}; MICU preferred. "
                            f"Limited MICU capabilities reduce outcome."
                        ),
                        clinical_rationale=(
                            "ALS lacks MICU capabilities: transcardiac pacing, IABP prep, "
                            "advanced haemodynamic monitoring (Maharashtra 108 SOP §4.2)."
                        ),
                        recommendation=(
                            "Dispatch MICU if available. If ALS already en-route, "
                            "request MICU rendezvous."
                        ),
                        step=step, elapsed_minutes=elapsed_minutes,
                        condition_key=condition_key, incident_id=incident_id,
                        unit_id=unit_id, observed_value="ALS", expected_value="MICU",
                    ))
        elif condition_key in STROKE_CONDITIONS:
            if unit in ("ALS", "MICU"):
                results.append(_compliant(
                    rule_key=PR_ALS_FOR_STROKE,
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    reward_delta=STROKE_UNIT_PRENOTIFICATION_BONUS,
                    description=f"{unit} correctly dispatched for {condition_key}.",
                    clinical_rationale=(
                        "ALS minimum for IV access, blood glucose, Cincinnati scale "
                        "assessment, and pre-notification to stroke unit. "
                        "Saver JAMA 2006: 1.9M neurons/min."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    unit_id=unit_id,
                ))
            else:
                results.append(_violation(
                    rule_key=PR_ALS_FOR_STROKE,
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.CRITICAL,
                    reward_delta=WRONG_UNIT_DISPATCH_PROTOCOL_PENALTY,
                    description=(
                        f"BLS dispatched for stroke ({condition_key}). "
                        f"BLS cannot assess, IV-access, or pre-notify stroke unit."
                    ),
                    clinical_rationale=(
                        "Cincinnati Pre-Hospital Stroke Scale requires trained ALS assessment. "
                        "IV access for tPA window management is an ALS skill. "
                        "Maharashtra NHM Protocol §7.3."
                    ),
                    recommendation="Dispatch ALS or MICU immediately.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    unit_id=unit_id,
                ))
        else:
            d_rank = self._UNIT_RANK.get(unit, 1)
            c_ranks = [self._UNIT_RANK.get(u, 1) for u in correct_units]
            min_required = min(c_ranks)
            if d_rank >= min_required:
                bonus = 0.0
                if unit in correct_units:
                    bonus = CORRECT_UNIT_TYPE_BONUS * 0.5
                results.append(_compliant(
                    rule_key="unit_type_adequacy",
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    reward_delta=bonus,
                    description=f"{unit} dispatched for {condition_key}: adequate.",
                    clinical_rationale=(
                        f"Required capability tier: {correct_units}. "
                        f"Dispatched: {unit}."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    unit_id=unit_id,
                ))
            else:
                gap = min_required - d_rank
                sev = ViolationSeverity.CRITICAL if gap >= 2 else ViolationSeverity.MAJOR
                penalty = CRITICAL_WRONG_UNIT_PENALTY if gap >= 2 else WRONG_UNIT_DISPATCH_PROTOCOL_PENALTY
                results.append(_violation(
                    rule_key="unit_type_adequacy",
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=sev,
                    reward_delta=penalty,
                    description=(
                        f"Under-capability dispatch: {unit} sent for {condition_key}. "
                        f"Required: {correct_units}. Gap: {gap} tier(s)."
                    ),
                    clinical_rationale=(
                        f"IndianEMSTriageMapper minimum unit for {condition_key}: "
                        f"{correct_units}. Under-capable dispatch reduces survival probability "
                        f"by {20 * gap}% per NHM India EMS Guidelines 2019."
                    ),
                    recommendation=(
                        f"Dispatch {correct_units[0]} or higher. "
                        f"If unavailable, request mutual aid."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    unit_id=unit_id, observed_value=unit,
                    expected_value=str(correct_units),
                ))
        return results
    def check_call_to_dispatch_latency(
        self,
        dispatch_latency_steps: int,
        severity:               str,
        condition_key:          str,
        step:                   int,
        elapsed_minutes:        float,
        incident_id:            Optional[str] = None,
    ) -> ProtocolCheckResult:
        latency_min = dispatch_latency_steps * SIMULATION_STEP_DURATION_MIN
        target_min = CALL_TO_DISPATCH_TARGET_MIN
        if severity in ("P0",):
            return _na(
                "call_to_dispatch_latency",
                ProtocolDomain.DISPATCH,
                ProtocolCheckCategory.DOCUMENTATION,
                step, elapsed_minutes,
                "P0 (Expectant/Palliative): no dispatch latency target.",
            )
        policy = ConditionGoldenHourPolicyRegistry.get(condition_key)
        window = policy.dispatch_window_min
        if latency_min <= target_min:
            return _compliant(
                rule_key="call_to_dispatch_latency",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                reward_delta=CALL_TO_DISPATCH_OPTIMAL_BONUS,
                description=(
                    f"Dispatch within {latency_min:.1f} min of incident report "
                    f"(target ≤{target_min:.0f} min). Optimal."
                ),
                clinical_rationale=(
                    "NHM India EMS 2019 CAD standard: call-to-dispatch ≤2 min."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=f"{latency_min:.1f} min",
                expected_value=f"≤{target_min:.0f} min",
            )
        elif latency_min <= window:
            penalty = CALL_TO_DISPATCH_LATE_PENALTY * (latency_min - target_min) / target_min
            return _make_result(
                rule_key="call_to_dispatch_latency",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                violation_severity=ViolationSeverity.MINOR,
                reward_delta=penalty,
                description=(
                    f"Dispatch at {latency_min:.1f} min — delayed but within "
                    f"condition dispatch window ({window:.0f} min)."
                ),
                clinical_rationale="Maharashtra 108 SOP: dispatch window per condition.",
                recommendation="Optimise CAD queue processing for future incidents.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=f"{latency_min:.1f} min",
                expected_value=f"≤{target_min:.0f} min",
            )
        else:
            penalty = CALL_TO_DISPATCH_LATE_PENALTY * 3.0
            return _violation(
                rule_key="call_to_dispatch_latency",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.MAJOR,
                reward_delta=penalty,
                description=(
                    f"Critical dispatch delay: {latency_min:.1f} min for {condition_key}. "
                    f"Dispatch window {window:.0f} min exceeded."
                ),
                clinical_rationale=(
                    f"For {condition_key}, survival probability has already fallen "
                    f"by ≈{(latency_min * 100 / max(1, params.time_to_irreversible_min)):.0f}% "
                    f"relative to zero-delay dispatch."
                    if (params := SurvivalCurveRegistry.get(condition_key)) else
                    "Delayed dispatch degrades survival outcome significantly."
                ),
                recommendation="Investigate CAD queue blockage. Assign dedicated dispatcher for P1 queue.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=f"{latency_min:.1f} min",
                expected_value=f"≤{window:.0f} min",
            )
    def check_pre_notification(
        self,
        condition_key:           str,
        cath_lab_activated:      bool,
        stroke_unit_notified:    bool,
        trauma_activation_sent:  bool,
        step:                    int,
        elapsed_minutes:         float,
        incident_id:             Optional[str] = None,
        hospital_id:             Optional[str] = None,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        if condition_key in CONDITIONS_CATH_LAB_MANDATORY:
            if cath_lab_activated:
                results.append(_compliant(
                    rule_key=PR_CATH_LAB_STEMI,
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    reward_delta=CATH_LAB_PRENOTIFICATION_BONUS,
                    description=f"Cath lab activated for {condition_key}.",
                    clinical_rationale=(
                        "ESC 2023: pre-hospital cath lab activation reduces door-to-balloon "
                        "time by 20–30 min, increasing survival 8–12%."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    hospital_id=hospital_id,
                ))
            else:
                results.append(_violation(
                    rule_key=PR_CATH_LAB_STEMI,
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=CATH_LAB_OMISSION_PENALTY,
                    description=f"Cath lab NOT activated for {condition_key}.",
                    clinical_rationale=(
                        "ESC 2023 §5.2: Pre-hospital STEMI notification is a Class I "
                        "recommendation. Omission delays D2B by median 22 min. "
                        "Maharashtra 108 SOP §6.1.2."
                    ),
                    recommendation=(
                        "Activate cath lab via hospital pre-notification radio / SAMU "
                        "system at time of dispatch."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    hospital_id=hospital_id,
                ))
        if condition_key in CONDITIONS_STROKE_UNIT_MANDATORY:
            if stroke_unit_notified:
                results.append(_compliant(
                    rule_key=PR_STROKE_UNIT,
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    reward_delta=STROKE_UNIT_PRENOTIFICATION_BONUS,
                    description=f"Stroke unit notified for {condition_key}.",
                    clinical_rationale=(
                        "IST-3 Lancet 2012: pre-notification to stroke team reduces "
                        "door-to-needle time by 15–24 min. NHM India 2019 §7.3."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
            else:
                results.append(_violation(
                    rule_key=PR_STROKE_UNIT,
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=STROKE_UNIT_OMISSION_PENALTY,
                    description=f"Stroke unit NOT notified for {condition_key}.",
                    clinical_rationale=(
                        "Stroke pre-notification reduces door-to-CT by median 17 min "
                        "(Mosley 2007). Every 10-min delay = 0.85% mortality increase."
                    ),
                    recommendation="Activate stroke team via hospital SAMU / direct radio.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
        if condition_key in CONDITIONS_TRAUMA_ACTIVATION_MANDATORY:
            if trauma_activation_sent:
                results.append(_compliant(
                    rule_key="trauma_team_activation",
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    reward_delta=TRAUMA_ACTIVATION_BONUS,
                    description=f"Trauma team activation sent for {condition_key}.",
                    clinical_rationale=(
                        "EAST TTA guidelines 2017: pre-hospital activation reduces "
                        "trauma bay preparation time by 10–15 min."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
            else:
                results.append(_violation(
                    rule_key="trauma_team_activation",
                    domain=ProtocolDomain.DISPATCH,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=TRAUMA_ACTIVATION_OMISSION_PENALTY,
                    description=f"Trauma activation NOT sent for {condition_key}.",
                    clinical_rationale=(
                        "TARN 2022: delayed trauma activation increases 30-day "
                        "mortality 18%. ATLS principle: early team assembly critical."
                    ),
                    recommendation="Transmit trauma activation to receiving Level-1 centre.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
        return results
    def check_scene_time(
        self,
        condition_key:          str,
        scene_time_minutes:     float,
        transport_urgency:      str,
        step:                   int,
        elapsed_minutes:        float,
        incident_id:            Optional[str] = None,
        justified_reason:       Optional[str] = None,
    ) -> ProtocolCheckResult:
        urgency = TransportUrgencyClass(transport_urgency) if transport_urgency in {
            t.value for t in TransportUrgencyClass
        } else TransportUrgencyClass.LOAD_AND_GO
        if urgency == TransportUrgencyClass.ELECTIVE:
            return _na(
                "scene_time_compliance",
                ProtocolDomain.SCENE_MANAGEMENT,
                ProtocolCheckCategory.DOCUMENTATION,
                step, elapsed_minutes,
                "Elective transport: no scene time target applicable.",
            )
        if urgency == TransportUrgencyClass.LOAD_AND_GO:
            target = SCENE_TIME_LOAD_AND_GO_MAX_MIN
            label  = "LOAD-AND-GO"
        elif urgency == TransportUrgencyClass.STAY_AND_PLAY:
            target = SCENE_TIME_STAY_AND_PLAY_MAX_MIN
            label  = "STAY-AND-PLAY"
        else:
            target = 15.0
            label  = "TIME-CRITICAL"
        if scene_time_minutes <= target:
            return _compliant(
                rule_key="scene_time_compliance",
                domain=ProtocolDomain.SCENE_MANAGEMENT,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                reward_delta=0.008,
                description=(
                    f"Scene time {scene_time_minutes:.1f} min for {label} "
                    f"protocol ({condition_key}). Within ≤{target:.0f} min target."
                ),
                clinical_rationale=(
                    "ATLS 10th Ed: scene time >10 min for load-and-go patients "
                    "independently predicts mortality increase."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=f"{scene_time_minutes:.1f} min",
                expected_value=f"≤{target:.0f} min",
            )
        elif scene_time_minutes <= SCENE_TIME_ABSOLUTE_MAX_MIN:
            if justified_reason:
                return _make_result(
                    rule_key="scene_time_compliance",
                    domain=ProtocolDomain.SCENE_MANAGEMENT,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    outcome=ComplianceOutcome.OVERRIDE_JUSTIFIED,
                    violation_severity=ViolationSeverity.MINOR,
                    reward_delta=-0.003,
                    description=(
                        f"Prolonged scene time ({scene_time_minutes:.1f} min) "
                        f"justified: {justified_reason}."
                    ),
                    clinical_rationale="Justified clinical exception to scene time target.",
                    recommendation="Document justification in patient record.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    agent_override_reason=justified_reason,
                )
            return _violation(
                rule_key="scene_time_compliance",
                domain=ProtocolDomain.SCENE_MANAGEMENT,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                severity=ViolationSeverity.MINOR,
                reward_delta=SCENE_TIME_VIOLATION_PENALTY * 0.5,
                description=(
                    f"Prolonged scene time: {scene_time_minutes:.1f} min "
                    f"(target ≤{target:.0f} min) for {label} ({condition_key})."
                ),
                clinical_rationale=(
                    "Extended on-scene time delays definitive care and increases "
                    "golden-window consumption."
                ),
                recommendation=(
                    "Package patient earlier. Only essential ALS procedures on scene. "
                    f"Proceed to {ConditionGoldenHourPolicyRegistry.get(condition_key).routing_specialty} hospital."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=f"{scene_time_minutes:.1f} min",
                expected_value=f"≤{target:.0f} min",
            )
        else:
            return _violation(
                rule_key="scene_time_compliance",
                domain=ProtocolDomain.SCENE_MANAGEMENT,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=SCENE_TIME_VIOLATION_PENALTY,
                description=(
                    f"Excessive scene time: {scene_time_minutes:.1f} min "
                    f"(absolute max {SCENE_TIME_ABSOLUTE_MAX_MIN:.0f} min) "
                    f"for {condition_key}."
                ),
                clinical_rationale=(
                    "Scene time >30 min for critically injured patients is associated "
                    "with 3× mortality increase (MTOS, 1993; TARN 2022)."
                ),
                recommendation=(
                    "Immediately transport. Perform remaining procedures en-route. "
                    "Pre-notify receiving hospital urgently."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
    def check_paediatric_unit_adequacy(
        self,
        condition_key:        str,
        dispatched_unit_type: str,
        patient_age_years:    int,
        step:                 int,
        elapsed_minutes:      float,
        incident_id:          Optional[str] = None,
    ) -> ProtocolCheckResult:
        is_paediatric = patient_age_years < PAEDIATRIC_AGE_THRESHOLD
        if not is_paediatric:
            return _na(
                PR_PAEDS_ED,
                ProtocolDomain.DISPATCH,
                ProtocolCheckCategory.LIFE_CRITICAL,
                step, elapsed_minutes,
                f"Patient age {patient_age_years} ≥ {PAEDIATRIC_AGE_THRESHOLD}: not paediatric.",
            )
        unit = dispatched_unit_type.upper()
        if unit in ("ALS", "MICU"):
            return _compliant(
                rule_key=PR_PAEDS_ED,
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                reward_delta=0.012,
                description=(
                    f"{unit} dispatched for paediatric patient (age {patient_age_years}), "
                    f"condition {condition_key}."
                ),
                clinical_rationale=(
                    "ERC Paediatric BLS 2021: paediatric emergencies require ALS-capable "
                    "crew for weight-based dosing, paediatric airway, IO access."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
        else:
            return _violation(
                rule_key=PR_PAEDS_ED,
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.MAJOR,
                reward_delta=WRONG_UNIT_DISPATCH_PROTOCOL_PENALTY,
                description=(
                    f"BLS dispatched for paediatric patient (age {patient_age_years}), "
                    f"condition {condition_key}. ALS minimum required."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP §8: all paediatric emergencies (age <8) require "
                    "ALS minimum. JumpSTART protocol requires ALS-level assessment."
                ),
                recommendation="Dispatch ALS with paediatric bag.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
    def check_cbrn_protocol(
        self,
        cbrn_type:            str,
        dispatched_unit_type: str,
        multi_agency_present: bool,
        decontamination_planned: bool,
        step:                 int,
        elapsed_minutes:      float,
        incident_id:          Optional[str] = None,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        if cbrn_type == CbrnContaminationType.NONE.value:
            return results
        unit = dispatched_unit_type.upper()
        if unit in ("ALS", "MICU"):
            results.append(_compliant(
                rule_key="cbrn_unit_protocol",
                domain=ProtocolDomain.CBRN,
                category=ProtocolCheckCategory.SAFETY,
                reward_delta=0.010,
                description=f"ALS/MICU dispatched for CBRN {cbrn_type}.",
                clinical_rationale=(
                    "CHEMPACK / NDMA CBRN protocol: ALS minimum for antidote "
                    "autoinjector administration. PPE level B."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        else:
            results.append(_violation(
                rule_key="cbrn_unit_protocol",
                domain=ProtocolDomain.CBRN,
                category=ProtocolCheckCategory.SAFETY,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=CRITICAL_WRONG_UNIT_PENALTY,
                description=f"BLS dispatched to CBRN {cbrn_type} incident. Unsafe.",
                clinical_rationale=(
                    "BLS crews lack PPE-B qualification and antidote autoinjectors. "
                    "Secondary contamination of crew and hospital is a patient safety "
                    "and public health risk (NDMA CBRN SOP 2019)."
                ),
                recommendation=(
                    "Dispatch ALS with CBRN pack. Do NOT send BLS without decon corridor."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        if not multi_agency_present:
            results.append(_violation(
                rule_key="cbrn_multiagency",
                domain=ProtocolDomain.CBRN,
                category=ProtocolCheckCategory.COORDINATION,
                severity=ViolationSeverity.MAJOR,
                reward_delta=MULTI_AGENCY_OMISSION_PENALTY,
                description=f"CBRN incident {cbrn_type} without multi-agency coordination.",
                clinical_rationale=(
                    "NDMA CBRN SOP: Fire & Police mandatory for hot zone management, "
                    "decontamination corridor, and scene security."
                ),
                recommendation="Request Fire + Police immediately. Stage ambulances upwind.",
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        if not decontamination_planned:
            results.append(_violation(
                rule_key="cbrn_decontamination",
                domain=ProtocolDomain.CBRN,
                category=ProtocolCheckCategory.SAFETY,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=-0.015,
                description=f"CBRN {cbrn_type}: no decontamination corridor planned.",
                clinical_rationale=(
                    "Untreated contamination causes hospital contamination (secondary "
                    "casualty). WHO CBRN guidance: decontamination before transport."
                ),
                recommendation=(
                    "Establish decontamination corridor (Fire) before patient loading. "
                    "Notify receiving hospital isolation protocol."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        return results
class TriageProtocolChecker:
    _GT_DB: ClassVar[TriageGroundTruthDatabase] = TriageGroundTruthDatabase()
    _EMS_MAPPER: ClassVar[IndianEMSTriageMapper] = IndianEMSTriageMapper()
    def check_single_triage_accuracy(
        self,
        condition_key:   str,
        assigned_tag:    str,
        rpm:             Optional[RPMScore],
        step:            int,
        elapsed_minutes: float,
        incident_id:     Optional[str] = None,
        patient_id:      Optional[str] = None,
    ) -> ProtocolCheckResult:
        gt_tag, confidence = self._GT_DB.get_ground_truth(condition_key)
        if gt_tag is None:
            return _na(
                "triage_accuracy",
                ProtocolDomain.TRIAGE,
                ProtocolCheckCategory.LIFE_CRITICAL,
                step, elapsed_minutes,
                f"No ground truth available for {condition_key}.",
            )
        try:
            assigned = TriageTag(assigned_tag)
        except ValueError:
            return _violation(
                rule_key="triage_accuracy",
                domain=ProtocolDomain.TRIAGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=TRIAGE_CRITICAL_MISMATCH_PENALTY,
                description=f"Invalid triage tag '{assigned_tag}' submitted.",
                clinical_rationale="Triage tag must be one of: Immediate, Delayed, Minimal, Expectant.",
                recommendation=f"Submit valid tag. Ground truth for {condition_key}: {gt_tag.value}.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
        is_critical_mismatch = (
            gt_tag == TriageTag.IMMEDIATE
            and assigned == TriageTag.EXPECTANT
        )
        if is_critical_mismatch:
            return _violation(
                rule_key="triage_accuracy",
                domain=ProtocolDomain.TRIAGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=TRIAGE_CRITICAL_MISMATCH_PENALTY,
                description=(
                    f"CRITICAL TRIAGE MISMATCH: tagged EXPECTANT for {condition_key}, "
                    f"ground truth = IMMEDIATE (RED). Patient will die without treatment."
                ),
                clinical_rationale=(
                    "Under-triage to EXPECTANT deprives a salvageable patient of life-saving "
                    "intervention. START protocol: IMMEDIATE = needs treatment now to survive. "
                    "Maharashtra 108 Triage SOP §3.2."
                ),
                recommendation=(
                    f"Immediately re-triage as IMMEDIATE. Dispatch MICU if not already done."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=assigned.value,
                expected_value=gt_tag.value,
            )
        if assigned == gt_tag:
            bonus = confidence * 0.015
            return _compliant(
                rule_key="triage_accuracy",
                domain=ProtocolDomain.TRIAGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                reward_delta=bonus,
                description=(
                    f"Correct triage: {assigned.value} for {condition_key}. "
                    f"Confidence: {confidence:.0%}."
                ),
                clinical_rationale=(
                    f"START protocol correctly identifies {gt_tag.value} patients."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=assigned.value,
                expected_value=gt_tag.value,
                confidence=confidence,
            )
        if assigned in gt_tag.adjacent_tags():
            return _make_result(
                rule_key="triage_accuracy",
                domain=ProtocolDomain.TRIAGE,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                violation_severity=ViolationSeverity.MINOR,
                reward_delta=-0.008 * confidence,
                description=(
                    f"Adjacent triage mismatch: {assigned.value} vs expected {gt_tag.value} "
                    f"for {condition_key}."
                ),
                clinical_rationale=(
                    "Adjacent mismatches have partial impact on outcomes. "
                    "Over-triage increases unnecessary resource use. "
                    "Under-triage delays treatment."
                ),
                recommendation=(
                    f"Re-assess RPM: reassign to {gt_tag.value}."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                observed_value=assigned.value,
                expected_value=gt_tag.value,
                confidence=confidence,
            )
        return _violation(
            rule_key="triage_accuracy",
            domain=ProtocolDomain.TRIAGE,
            category=ProtocolCheckCategory.LIFE_CRITICAL,
            severity=ViolationSeverity.MAJOR,
            reward_delta=-0.030 * confidence,
            description=(
                f"Distant triage mismatch: {assigned.value} vs {gt_tag.value} "
                f"for {condition_key}."
            ),
            clinical_rationale=(
                "Distant mismatch (e.g., MINIMAL when IMMEDIATE, or vice versa) "
                "indicates fundamental assessment error. START: any abnormal vital = "
                "at minimum DELAYED. NHM India triage protocol §4."
            ),
            recommendation=(
                f"Retrain on {condition_key} presentation. "
                f"Reassign to {gt_tag.value} immediately."
            ),
            step=step, elapsed_minutes=elapsed_minutes,
            condition_key=condition_key, incident_id=incident_id,
            observed_value=assigned.value,
            expected_value=gt_tag.value,
            confidence=confidence,
        )
    def check_mci_start_triage_protocol(
        self,
        incident_id:             str,
        steps_since_mci_declared: int,
        triage_initiated:        bool,
        command_post_established: bool,
        victim_count:            int,
        immediate_count:         int,
        triage_start_step:       Optional[int],
        step:                    int,
        elapsed_minutes:         float,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        if not triage_initiated:
            if steps_since_mci_declared > MCI_TRIAGE_MUST_START_WITHIN_STEPS:
                results.append(_violation(
                    rule_key=PR_START_TRIAGE_MCI,
                    domain=ProtocolDomain.TRIAGE,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.CRITICAL,
                    reward_delta=MCI_NO_START_TRIAGE_PENALTY,
                    description=(
                        f"MCI declared {steps_since_mci_declared} steps ago but "
                        f"START triage NOT initiated for incident {incident_id}."
                    ),
                    clinical_rationale=(
                        "NDMA MCI Protocol 2019: START triage must begin within 5 min "
                        "of MCI declaration. Delayed triage = untreated IMMEDIATE patients "
                        "deteriorating to EXPECTANT. Maharashtra 108 SOP §10.3."
                    ),
                    recommendation=(
                        "Immediately initiate START triage. "
                        "Assign triage officer role to senior paramedic."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                ))
        else:
            bonus = min(0.035, 0.010 + (MCI_TRIAGE_MUST_START_WITHIN_STEPS
                                         - min(steps_since_mci_declared, MCI_TRIAGE_MUST_START_WITHIN_STEPS)
                                         ) * 0.008)
            results.append(_compliant(
                rule_key=PR_START_TRIAGE_MCI,
                domain=ProtocolDomain.TRIAGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                reward_delta=bonus,
                description=f"START triage initiated for MCI incident {incident_id}.",
                clinical_rationale=(
                    "NDMA MCI 2019: early START triage enables rapid resource allocation "
                    "and reduces preventable mortality."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        if not command_post_established:
            if steps_since_mci_declared > MCI_COMMAND_POST_WITHIN_STEPS:
                results.append(_violation(
                    rule_key="mci_command_post",
                    domain=ProtocolDomain.SCENE_MANAGEMENT,
                    category=ProtocolCheckCategory.COORDINATION,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=MCI_NO_COMMAND_POST_PENALTY,
                    description=(
                        f"No incident command post established for MCI {incident_id} "
                        f"after {steps_since_mci_declared} steps."
                    ),
                    clinical_rationale=(
                        "ICS (Incident Command System) mandatory for ≥5 casualty incidents. "
                        "Without command post: communication failure, resource duplication, "
                        "hospital load imbalance. NDMA 2019 §6.2."
                    ),
                    recommendation=(
                        "Designate senior paramedic as incident commander. "
                        "Establish command post upwind / upstream from incident. "
                        "Assign sector officers: triage, treatment, transport."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                ))
        else:
            results.append(_compliant(
                rule_key="mci_command_post",
                domain=ProtocolDomain.SCENE_MANAGEMENT,
                category=ProtocolCheckCategory.COORDINATION,
                reward_delta=0.010,
                description=f"Command post established for MCI {incident_id}.",
                clinical_rationale="ICS command post reduces MCI mortality 23% (NDMA study 2018).",
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        if victim_count >= 5:
            immediate_ratio = immediate_count / victim_count if victim_count > 0 else 0
            if immediate_ratio > 0.80:
                results.append(_violation(
                    rule_key="triage_distribution_sanity",
                    domain=ProtocolDomain.TRIAGE,
                    category=ProtocolCheckCategory.DOCUMENTATION,
                    severity=ViolationSeverity.ADVISORY,
                    reward_delta=-0.005,
                    description=(
                        f"Implausibly high IMMEDIATE ratio: {immediate_ratio:.0%} "
                        f"of {victim_count} victims. Possible over-triage."
                    ),
                    clinical_rationale=(
                        "MCI literature: IMMEDIATE rate typically 10-30% in RTAs. "
                        "Over-triage > 50% overwhelms Level-1 trauma centre capacity."
                    ),
                    recommendation="Re-assess walking wounded: apply 'walk away' → MINIMAL first.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                ))
        return results
    def check_dnr_triage_compliance(
        self,
        condition_key:  str,
        assigned_tag:   str,
        dnr_present:    bool,
        step:           int,
        elapsed_minutes: float,
        incident_id:    Optional[str] = None,
        patient_id:     Optional[str] = None,
    ) -> ProtocolCheckResult:
        if not dnr_present:
            return _na(
                "dnr_triage",
                ProtocolDomain.DNR,
                ProtocolCheckCategory.ETHICS,
                step, elapsed_minutes,
                "No DNR present: standard triage applies.",
            )
        if assigned_tag == TriageTag.EXPECTANT.value:
            return _compliant(
                rule_key="dnr_triage",
                domain=ProtocolDomain.DNR,
                category=ProtocolCheckCategory.ETHICS,
                reward_delta=0.008,
                description=(
                    f"DNR patient correctly triaged as EXPECTANT for {condition_key}. "
                    "Consistent with patient advance directive."
                ),
                clinical_rationale=(
                    "Ethics: respect for patient autonomy. DNR = do not resuscitate; "
                    "palliative approach is ethically correct. "
                    "Maharashtra Medical Termination Act & patient rights §12."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
        elif assigned_tag == TriageTag.IMMEDIATE.value:
            return _violation(
                rule_key="dnr_triage",
                domain=ProtocolDomain.DNR,
                category=ProtocolCheckCategory.ETHICS,
                severity=ViolationSeverity.MAJOR,
                reward_delta=DNR_ESCALATION_PENALTY,
                description=(
                    f"DNR patient triaged as IMMEDIATE for {condition_key}. "
                    "Conflicts with advance directive."
                ),
                clinical_rationale=(
                    "Violating a valid DNR directive constitutes medical battery and "
                    "an ethics violation. MCI triage: DNR patients → EXPECTANT category. "
                    "Maharashtra 108 Ethics Protocol §14.1."
                ),
                recommendation=(
                    "Verify DNR documentation. If valid and condition is terminal, "
                    "reclassify as EXPECTANT. Provide comfort measures."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
        return _na(
            "dnr_triage",
            ProtocolDomain.DNR,
            ProtocolCheckCategory.ETHICS,
            step, elapsed_minutes,
            f"DNR triage tag {assigned_tag} review not critical.",
        )
class HospitalRoutingChecker:
    def check_specialty_match(
        self,
        condition_key:            str,
        hospital_id:              str,
        hospital_specialties:     Set[str],
        hospital_on_diversion:    bool,
        step:                     int,
        elapsed_minutes:          float,
        incident_id:              Optional[str] = None,
    ) -> ProtocolCheckResult:
        from server.models import CONDITION_REQUIRED_SPECIALTY
        required = CONDITION_REQUIRED_SPECIALTY.get(condition_key)
        if required is None:
            return _na(
                "specialty_match",
                ProtocolDomain.HOSPITAL_ROUTING,
                ProtocolCheckCategory.TIME_SENSITIVE,
                step, elapsed_minutes,
                f"No specialty requirement defined for {condition_key}.",
            )
        if hospital_on_diversion:
            return _violation(
                rule_key=PR_NO_DIVERSION_ROUTE,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.SAFETY,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=DIVERSION_ROUTE_PROTOCOL_PENALTY,
                description=(
                    f"Patient routed to diverted hospital {hospital_id} "
                    f"for {condition_key}. Protocol violation."
                ),
                clinical_rationale=(
                    "Routing to diverted hospital risks receiving corridor overload, "
                    "delayed care, and poor outcomes. Maharashtra 108 Ops §9.1: "
                    "'Never bypass diversion flag without medical director override.'"
                ),
                recommendation=(
                    "Reroute to next nearest hospital with required specialty "
                    f"({required}) and accepting status."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
        if required in hospital_specialties:
            return _compliant(
                rule_key="specialty_match",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                reward_delta=SPECIALTY_CORRECT_BONUS,
                description=(
                    f"Hospital {hospital_id} has required specialty '{required}' "
                    f"for {condition_key}."
                ),
                clinical_rationale=(
                    f"Evidence-based routing: {condition_key} requires {required} "
                    "for definitive management. Correct routing improves outcomes."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
                observed_value=required, expected_value=required,
            )
        else:
            return _violation(
                rule_key="specialty_match",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.MAJOR,
                reward_delta=SPECIALTY_MISMATCH_PENALTY,
                description=(
                    f"Hospital {hospital_id} lacks '{required}' for {condition_key}. "
                    f"Inadequate definitive care."
                ),
                clinical_rationale=(
                    f"{condition_key} requires {required} — missing at this facility "
                    "means delayed or unavailable definitive treatment."
                ),
                recommendation=(
                    f"Reroute to nearest hospital with {required}. "
                    "Accept longer transport if required specialty unavailable nearby."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
                observed_value=f"missing:{required}", expected_value=required,
            )
    def check_level1_trauma_within_30min(
        self,
        condition_key:        str,
        hospital_is_level1:   bool,
        hospital_id:          str,
        estimated_travel_min: float,
        step:                 int,
        elapsed_minutes:      float,
        incident_id:          Optional[str] = None,
    ) -> ProtocolCheckResult:
        if condition_key not in MAJOR_TRAUMA_CONDITIONS:
            return _na(
                PR_LEVEL1_TRAUMA_30MIN,
                ProtocolDomain.HOSPITAL_ROUTING,
                ProtocolCheckCategory.LIFE_CRITICAL,
                step, elapsed_minutes,
                f"{condition_key} does not require Level-1 trauma routing.",
            )
        if not hospital_is_level1:
            return _violation(
                rule_key=PR_LEVEL1_TRAUMA_30MIN,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=-0.025,
                description=(
                    f"Polytrauma patient routed to non-Level-1 hospital {hospital_id} "
                    f"for {condition_key}. Level-1 mandatory."
                ),
                clinical_rationale=(
                    "MTOS study: polytrauma at Level-1 centres has 25% lower mortality "
                    "vs Level-2/3. TARN 2022: ISS≥16 must go to Level-1. "
                    "Maharashtra Trauma Network Protocol §3."
                ),
                recommendation=(
                    "Identify nearest Level-1 trauma centre. "
                    "Accept up to 45-min transport for Level-1 access."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
        if estimated_travel_min <= 30.0:
            return _compliant(
                rule_key=PR_LEVEL1_TRAUMA_30MIN,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                reward_delta=0.015,
                description=(
                    f"Level-1 trauma centre {hospital_id} within "
                    f"{estimated_travel_min:.1f} min for {condition_key}."
                ),
                clinical_rationale=(
                    "Optimal: Level-1 trauma within 30 min reduces ISS-adjusted "
                    "mortality by 18% (Brain Trauma Foundation 2016)."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
                observed_value=f"{estimated_travel_min:.1f} min",
                expected_value="≤30 min",
            )
        else:
            sev = ViolationSeverity.MAJOR if estimated_travel_min > 45 else ViolationSeverity.MINOR
            return _violation(
                rule_key=PR_LEVEL1_TRAUMA_30MIN,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=sev,
                reward_delta=-0.015 if estimated_travel_min > 45 else -0.005,
                description=(
                    f"Level-1 trauma centre {hospital_id} at {estimated_travel_min:.1f} min "
                    f"exceeds 30-min target for {condition_key}."
                ),
                clinical_rationale=(
                    "Every 10 min beyond 30-min threshold increases polytrauma mortality "
                    "by ~8% (TARN 2022)."
                ),
                recommendation=(
                    "Check if closer Level-1 available. "
                    "Consider air evacuation if >45 min by road."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
                observed_value=f"{estimated_travel_min:.1f} min",
                expected_value="≤30 min",
            )
    def check_burns_unit_routing(
        self,
        condition_key:          str,
        hospital_has_burns_unit: bool,
        hospital_id:            str,
        step:                   int,
        elapsed_minutes:        float,
        incident_id:            Optional[str] = None,
    ) -> ProtocolCheckResult:
        if condition_key not in BURNS_CONDITIONS:
            return _na(
                PR_BURNS_UNIT,
                ProtocolDomain.HOSPITAL_ROUTING,
                ProtocolCheckCategory.TIME_SENSITIVE,
                step, elapsed_minutes,
                f"{condition_key} does not require burns unit.",
            )
        if hospital_has_burns_unit:
            return _compliant(
                rule_key=PR_BURNS_UNIT,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                reward_delta=0.012,
                description=f"Burns unit available at {hospital_id} for {condition_key}.",
                clinical_rationale=(
                    "ABA 2020: major burns require specialist burns centre for "
                    "Parkland formula, escharotomy, and skin graft preparation."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
        else:
            return _violation(
                rule_key=PR_BURNS_UNIT,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                severity=ViolationSeverity.MAJOR,
                reward_delta=-0.020,
                description=(
                    f"Burns patient ({condition_key}) routed to {hospital_id} "
                    f"without burns unit."
                ),
                clinical_rationale=(
                    "Non-burns centre management of major burns increases infection "
                    "risk and mortality 2× (Belgian Burn Registry 2020)."
                ),
                recommendation="Route to nearest burns unit, even if travel time longer.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
    def check_paeds_ed_routing(
        self,
        condition_key:              str,
        patient_age_years:          int,
        hospital_has_paeds_ed:      bool,
        hospital_id:                str,
        step:                       int,
        elapsed_minutes:            float,
        incident_id:                Optional[str] = None,
    ) -> ProtocolCheckResult:
        is_paediatric = patient_age_years < PAEDIATRIC_AGE_THRESHOLD
        if not is_paediatric or condition_key not in PAEDIATRIC_CONDITIONS:
            return _na(
                PR_PAEDS_ED,
                ProtocolDomain.HOSPITAL_ROUTING,
                ProtocolCheckCategory.LIFE_CRITICAL,
                step, elapsed_minutes,
                "Not a paediatric emergency or condition does not require Paeds ED.",
            )
        if hospital_has_paeds_ed:
            return _compliant(
                rule_key=PR_PAEDS_ED,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                reward_delta=0.012,
                description=(
                    f"Paediatric ED at {hospital_id} for paediatric patient (age {patient_age_years}), "
                    f"condition {condition_key}."
                ),
                clinical_rationale=(
                    "RCPCH 2022: paediatric-specific ED reduces paediatric mortality 20% "
                    "vs generic ED for age-specific presentations."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
        else:
            return _violation(
                rule_key=PR_PAEDS_ED,
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.MAJOR,
                reward_delta=-0.018,
                description=(
                    f"Paediatric patient (age {patient_age_years}, {condition_key}) "
                    f"routed to {hospital_id} without Paeds ED."
                ),
                clinical_rationale=(
                    "Weight-based dosing, paediatric airways, and child-specific "
                    "resuscitation require dedicated paediatric emergency department. "
                    "Maharashtra NHM §8.4."
                ),
                recommendation="Route to nearest Paeds ED or PICU-capable facility.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
    def check_travel_time_vs_golden_window(
        self,
        condition_key:        str,
        estimated_travel_min: float,
        elapsed_minutes:      float,
        age_years:            Optional[int],
        step:                 int,
        incident_id:          Optional[str] = None,
        hospital_id:          Optional[str] = None,
    ) -> ProtocolCheckResult:
        params = SurvivalCurveRegistry.get(condition_key)
        gw = params.golden_window
        total_care_time = elapsed_minutes + estimated_travel_min
        p_current = SurvivalProbabilityCalculator.compute(
            params, elapsed_minutes, age_years
        )
        p_at_arrival = SurvivalProbabilityCalculator.compute(
            params, total_care_time, age_years
        )
        survival_loss = p_current - p_at_arrival
        if gw is None:
            if survival_loss <= 0.05:
                return _compliant(
                    rule_key="travel_time_survival",
                    domain=ProtocolDomain.HOSPITAL_ROUTING,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    reward_delta=0.004,
                    description=(
                        f"Travel time {estimated_travel_min:.1f} min acceptable for "
                        f"{condition_key}: survival loss ≤5%."
                    ),
                    clinical_rationale="No golden window defined; travel time acceptable.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    hospital_id=hospital_id,
                )
            else:
                return _make_result(
                    rule_key="travel_time_survival",
                    domain=ProtocolDomain.HOSPITAL_ROUTING,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                    violation_severity=ViolationSeverity.ADVISORY,
                    reward_delta=-0.005,
                    description=(
                        f"Travel time {estimated_travel_min:.1f} min for {condition_key}: "
                        f"survival loss ≈{survival_loss:.1%}."
                    ),
                    clinical_rationale="Travel time acceptable; survival loss moderate.",
                    recommendation="Consider closer hospital if available.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    hospital_id=hospital_id,
                )
        window_remaining = max(0.0, gw.closes_at_min - elapsed_minutes)
        can_reach_in_window = estimated_travel_min <= window_remaining
        if can_reach_in_window:
            bonus = GOLDEN_HOUR_COMPLIANCE_BONUS * 0.3
            return _compliant(
                rule_key="travel_time_survival",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                reward_delta=bonus,
                description=(
                    f"Travel {estimated_travel_min:.1f} min fits within {gw.window_type.value} "
                    f"({window_remaining:.1f} min remaining) for {condition_key}."
                ),
                clinical_rationale=gw.description,
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
                observed_value=f"{estimated_travel_min:.1f} min",
                expected_value=f"≤{window_remaining:.1f} min remaining",
            )
        else:
            penalty = GOLDEN_WINDOW_BREACHED_PENALTY * min(1.0, survival_loss * 3)
            return _violation(
                rule_key="travel_time_survival",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.MAJOR,
                reward_delta=penalty,
                description=(
                    f"Travel time {estimated_travel_min:.1f} min exceeds "
                    f"{gw.window_type.value} remaining ({window_remaining:.1f} min) "
                    f"for {condition_key}. Golden window will be breached."
                ),
                clinical_rationale=gw.description,
                recommendation=(
                    "Route to closest hospital with required specialty. "
                    "Activate green corridor if available."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
                hospital_id=hospital_id,
            )
    def check_mci_hospital_spread(
        self,
        incident_id:          str,
        hospital_count:       int,
        victim_count:         int,
        max_per_hospital:     int,
        step:                 int,
        elapsed_minutes:      float,
    ) -> ProtocolCheckResult:
        if hospital_count >= MCI_MIN_HOSPITAL_SPREAD and (
            victim_count == 0 or victim_count / hospital_count <= max_per_hospital
        ):
            bonus = MCI_SINGLE_HOSPITAL_ROUTING_PENALTY * -0.2 + hospital_count * 0.005
            return _compliant(
                rule_key="mci_hospital_spread",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.COORDINATION,
                reward_delta=min(0.04, bonus),
                description=(
                    f"MCI {incident_id}: victims spread across {hospital_count} hospitals. "
                    f"Load balance: {victim_count/max(1,hospital_count):.1f} per hospital."
                ),
                clinical_rationale=(
                    "NDMA MCI Protocol: distribute victims to prevent single-hospital "
                    "surge. Load per Level-1 centre: ≤8 immediate patients."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            )
        else:
            sev = ViolationSeverity.MAJOR if hospital_count <= 1 else ViolationSeverity.MINOR
            penalty = MCI_SINGLE_HOSPITAL_ROUTING_PENALTY if hospital_count <= 1 else -0.020
            return _violation(
                rule_key="mci_hospital_spread",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.COORDINATION,
                severity=sev,
                reward_delta=penalty,
                description=(
                    f"MCI {incident_id}: all/most victims routed to "
                    f"{hospital_count} hospital(s). Insufficient spread for {victim_count} victims."
                ),
                clinical_rationale=(
                    "Single-hospital MCI routing triggers diversion cascade. "
                    "Each extra hospital reduces diversion risk by 35% (NDMA 2019)."
                ),
                recommendation=(
                    f"Spread victims: IMMEDIATE to Level-1, DELAYED to Level-2, "
                    f"MINIMAL to nearest ED. Target ≥{MCI_MIN_HOSPITAL_SPREAD} hospitals."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            )
class TransferProtocolChecker:
    def check_transfer_indication(
        self,
        condition_key:                  str,
        sending_hospital_has_specialty: bool,
        receiving_hospital_has_specialty: bool,
        patient_severity:               str,
        step:                           int,
        elapsed_minutes:                float,
        incident_id:                    Optional[str] = None,
        sending_hospital_id:            Optional[str] = None,
        receiving_hospital_id:          Optional[str] = None,
    ) -> ProtocolCheckResult:
        from server.models import CONDITION_REQUIRED_SPECIALTY
        required = CONDITION_REQUIRED_SPECIALTY.get(condition_key)
        if required and not sending_hospital_has_specialty:
            if receiving_hospital_has_specialty:
                return _compliant(
                    rule_key="transfer_indication",
                    domain=ProtocolDomain.TRANSFER,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    reward_delta=0.010,
                    description=(
                        f"Transfer from {sending_hospital_id} (no {required}) "
                        f"to {receiving_hospital_id} (has {required}) for {condition_key}."
                    ),
                    clinical_rationale=(
                        f"Inter-facility transfer indicated: sending centre lacks {required}. "
                        "NHS / NHM India: patient right to appropriate level of care."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    hospital_id=receiving_hospital_id,
                )
            else:
                return _violation(
                    rule_key="transfer_indication",
                    domain=ProtocolDomain.TRANSFER,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.CRITICAL,
                    reward_delta=-0.025,
                    description=(
                        f"Transfer to {receiving_hospital_id} which also lacks "
                        f"'{required}' for {condition_key}. Pointless transfer."
                    ),
                    clinical_rationale=(
                        "Transfer without capability gain exposes patient to transport "
                        "risk without benefit. Patient should stay at sending centre "
                        "and mutual aid escalated."
                    ),
                    recommendation=(
                        f"Identify hospital WITH {required}. "
                        "If unavailable, escalate to surge / mutual aid."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    hospital_id=receiving_hospital_id,
                )
        return _na(
            "transfer_indication",
            ProtocolDomain.TRANSFER,
            ProtocolCheckCategory.DOCUMENTATION,
            step, elapsed_minutes,
            "Transfer indication: condition does not mandate specialty routing.",
        )
    def check_transfer_pre_notification(
        self,
        receiving_team_notified:    bool,
        bed_confirmed:              bool,
        patient_severity:           str,
        step:                       int,
        elapsed_minutes:            float,
        incident_id:                Optional[str] = None,
        hospital_id:                Optional[str] = None,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        if not receiving_team_notified:
            sev = ViolationSeverity.MAJOR if patient_severity == "P1" else ViolationSeverity.MINOR
            results.append(_violation(
                rule_key="transfer_team_notify",
                domain=ProtocolDomain.TRANSFER,
                category=ProtocolCheckCategory.DOCUMENTATION,
                severity=sev,
                reward_delta=TRANSFER_NO_TEAM_NOTIFY_PENALTY,
                description=(
                    f"Transfer initiated without notifying receiving team at {hospital_id}."
                ),
                clinical_rationale=(
                    "NHM India inter-facility transfer SOP §5.2: advance notification "
                    "mandatory. Unannounced transfers increase handoff time 15 min median."
                ),
                recommendation="Contact receiving ED/ICU/specialist team before dispatch.",
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, hospital_id=hospital_id,
            ))
        else:
            results.append(_compliant(
                rule_key="transfer_team_notify",
                domain=ProtocolDomain.TRANSFER,
                category=ProtocolCheckCategory.DOCUMENTATION,
                reward_delta=0.006,
                description=f"Receiving team notified before transfer to {hospital_id}.",
                clinical_rationale="Pre-notification reduces handoff delay.",
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, hospital_id=hospital_id,
            ))
        if not bed_confirmed:
            results.append(_violation(
                rule_key="transfer_bed_confirm",
                domain=ProtocolDomain.TRANSFER,
                category=ProtocolCheckCategory.DOCUMENTATION,
                severity=ViolationSeverity.MINOR,
                reward_delta=TRANSFER_NO_BED_CONFIRM_PENALTY,
                description=f"Transfer without confirmed bed at {hospital_id}.",
                clinical_rationale=(
                    "Unconfirmed bed transfers risk diversion on arrival. "
                    "NICE 2019: bed confirmation mandatory before P1 transfer."
                ),
                recommendation="Confirm bed availability before dispatching transport.",
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, hospital_id=hospital_id,
            ))
        return results
    def check_transfer_timing(
        self,
        condition_key:              str,
        estimated_transfer_time_min: float,
        elapsed_minutes:            float,
        patient_severity:           str,
        step:                       int,
        incident_id:                Optional[str] = None,
    ) -> ProtocolCheckResult:
        params = SurvivalCurveRegistry.get(condition_key)
        gw = params.golden_window
        time_to_irrev = max(0.0, params.time_to_irreversible_min - elapsed_minutes)
        if patient_severity == "P1":
            if estimated_transfer_time_min > TRANSFER_GOLDEN_WINDOW_MIN:
                return _violation(
                    rule_key="transfer_timing",
                    domain=ProtocolDomain.TRANSFER,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=TRANSFER_TIMING_VIOLATION_PENALTY,
                    description=(
                        f"P1 transfer time {estimated_transfer_time_min:.1f} min "
                        f"exceeds 30-min transfer window for {condition_key}."
                    ),
                    clinical_rationale=(
                        "NICE 2019: P1 inter-facility transfer ≤30 min total transport. "
                        "Longer transfers require helicopter or clinical decision to "
                        "stabilise at sending centre."
                    ),
                    recommendation=(
                        "Consider air transfer if ground >30 min. "
                        "Reassess transfer necessity: can sending centre manage temporarily?"
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                    observed_value=f"{estimated_transfer_time_min:.1f} min",
                    expected_value=f"≤{TRANSFER_GOLDEN_WINDOW_MIN:.0f} min",
                )
            else:
                return _compliant(
                    rule_key="transfer_timing",
                    domain=ProtocolDomain.TRANSFER,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    reward_delta=0.006,
                    description=(
                        f"P1 transfer time {estimated_transfer_time_min:.1f} min "
                        f"within 30-min window for {condition_key}."
                    ),
                    clinical_rationale="NICE 2019: P1 transfer ≤30 min compliant.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                )
        else:
            if estimated_transfer_time_min > TRANSFER_MAX_STABLE_DELAY_MIN:
                return _violation(
                    rule_key="transfer_timing",
                    domain=ProtocolDomain.TRANSFER,
                    category=ProtocolCheckCategory.TIME_SENSITIVE,
                    severity=ViolationSeverity.MINOR,
                    reward_delta=-0.004,
                    description=(
                        f"Non-P1 transfer time {estimated_transfer_time_min:.1f} min "
                        f"exceeds 90-min guideline for {condition_key}."
                    ),
                    clinical_rationale="NHM India: P2 transfer ≤90 min.",
                    recommendation="Optimise routing. Consider earlier dispatch.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                )
            return _na(
                "transfer_timing",
                ProtocolDomain.TRANSFER,
                ProtocolCheckCategory.TIME_SENSITIVE,
                step, elapsed_minutes,
                f"P2/P3 transfer timing {estimated_transfer_time_min:.1f} min acceptable.",
            )
class MutualAidProtocolChecker:
    def check_mutual_aid_justification(
        self,
        own_fleet_available_count: int,
        units_requested:           int,
        requesting_zone:           str,
        providing_zone:            str,
        surge_declared:            bool,
        p1_unassigned_count:       int,
        step:                      int,
        elapsed_minutes:           float,
        incident_id:               Optional[str] = None,
    ) -> ProtocolCheckResult:
        if own_fleet_available_count >= MUTUAL_AID_OVER_REQUEST_OWN_FLEET_THRESHOLD:
            if not surge_declared:
                return _violation(
                    rule_key=PR_MUTUAL_AID_SURGE,
                    domain=ProtocolDomain.MUTUAL_AID,
                    category=ProtocolCheckCategory.COORDINATION,
                    severity=ViolationSeverity.MINOR,
                    reward_delta=MUTUAL_AID_OVER_REQUEST_PENALTY,
                    description=(
                        f"Mutual aid requested from {providing_zone} while "
                        f"{own_fleet_available_count} own units available in {requesting_zone}. "
                        "Possible over-request."
                    ),
                    clinical_rationale=(
                        "Over-requesting mutual aid depletes providing zone resources "
                        "unnecessarily. Maharashtra 108 SOP §11.3: mutual aid when "
                        "own fleet <20% available and P1 queue active."
                    ),
                    recommendation=(
                        "Deploy own available units first. "
                        "Request mutual aid only when fleet <2 deployable units."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                    observed_value=f"own_available={own_fleet_available_count}",
                    expected_value=f"own_available<{MUTUAL_AID_OVER_REQUEST_OWN_FLEET_THRESHOLD}",
                    automatically_waivable=True,
                )
        bonus = 0.010 if surge_declared else 0.005
        return _compliant(
            rule_key=PR_MUTUAL_AID_SURGE,
            domain=ProtocolDomain.MUTUAL_AID,
            category=ProtocolCheckCategory.COORDINATION,
            reward_delta=bonus,
            description=(
                f"Mutual aid request from {requesting_zone} justified: "
                f"{own_fleet_available_count} own units, {p1_unassigned_count} P1 unassigned."
            ),
            clinical_rationale=(
                "Maharashtra 108 SOP §11: mutual aid activation appropriate "
                "when demand exceeds 80% fleet capacity."
            ),
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        )
    def check_under_request_during_surge(
        self,
        surge_declared:            bool,
        mutual_aid_active:         bool,
        p1_unassigned_count:       int,
        own_fleet_available_count: int,
        step:                      int,
        elapsed_minutes:           float,
        incident_id:               Optional[str] = None,
    ) -> ProtocolCheckResult:
        if not surge_declared:
            return _na(
                "mutual_aid_under_request",
                ProtocolDomain.MUTUAL_AID,
                ProtocolCheckCategory.COORDINATION,
                step, elapsed_minutes,
                "No surge declared: under-request check not applicable.",
            )
        if p1_unassigned_count >= 3 and own_fleet_available_count == 0 and not mutual_aid_active:
            return _violation(
                rule_key="mutual_aid_under_request",
                domain=ProtocolDomain.MUTUAL_AID,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=MUTUAL_AID_UNDER_REQUEST_PENALTY,
                description=(
                    f"Surge declared, {p1_unassigned_count} P1 patients unassigned, "
                    f"zero own fleet available, but mutual aid NOT requested."
                ),
                clinical_rationale=(
                    "Protocol rule: mutual_aid_in_surge is mandatory when fleet=0 "
                    "and P1 queue≥3. Failure = preventable deaths. "
                    "Maharashtra 108 SOP §11.5."
                ),
                recommendation=(
                    "Immediately request mutual aid from adjacent zones. "
                    "Activate State-level EMS reserve if available."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            )
        return _compliant(
            rule_key="mutual_aid_under_request",
            domain=ProtocolDomain.MUTUAL_AID,
            category=ProtocolCheckCategory.COORDINATION,
            reward_delta=0.010,
            description="Mutual aid status consistent with surge conditions.",
            clinical_rationale="Surge + mutual aid active: protocol compliant.",
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        )
    def check_mutual_aid_eta(
        self,
        expected_eta_steps: int,
        minimum_eta_steps:  int,
        condition_key:      str,
        step:               int,
        elapsed_minutes:    float,
        incident_id:        Optional[str] = None,
    ) -> ProtocolCheckResult:
        eta_min = expected_eta_steps * SIMULATION_STEP_DURATION_MIN
        min_eta_min = minimum_eta_steps * SIMULATION_STEP_DURATION_MIN  
        if expected_eta_steps < minimum_eta_steps:
            return _violation(
                rule_key="mutual_aid_eta",
                domain=ProtocolDomain.MUTUAL_AID,
                category=ProtocolCheckCategory.DOCUMENTATION,
                severity=ViolationSeverity.ADVISORY,
                reward_delta=-0.003,
                description=(
                    f"Mutual aid ETA {eta_min:.0f} min < minimum "
                    f"deployment time {min_eta_min:.0f} min. "
                    "Unrealistic ETA may mislead dispatch decisions."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP: cross-zone deployment minimum 12 min "
                    "(preparation + transit). Under-estimating causes premature "
                    "commitment of resources."
                ),
                recommendation="Revise ETA to realistic estimate.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            )
        return _compliant(
            rule_key="mutual_aid_eta",
            domain=ProtocolDomain.MUTUAL_AID,
            category=ProtocolCheckCategory.DOCUMENTATION,
            reward_delta=0.002,
            description=f"Mutual aid ETA {eta_min:.0f} min: realistic.",
            clinical_rationale="ETA ≥ minimum deployment time.",
            step=step, elapsed_minutes=elapsed_minutes,
            condition_key=condition_key, incident_id=incident_id,
        )
class SurgeProtocolChecker:
    def check_surge_declaration_timing(
        self,
        surge_declared:                  bool,
        surge_declared_step:             Optional[int],
        criteria_first_met_step:         Optional[int],
        simultaneous_mci_count:          int,
        hospitals_on_diversion:          int,
        system_er_occupancy:             float,
        cascade_risk:                    float,
        current_step:                    int,
        elapsed_minutes:                 float,
        incident_id:                     Optional[str] = None,
    ) -> ProtocolCheckResult:
        criteria_met = (
            simultaneous_mci_count >= 3
            or hospitals_on_diversion >= 4
            or system_er_occupancy >= 0.85
            or cascade_risk >= 0.50
        )
        if not criteria_met:
            if surge_declared and surge_declared_step is not None:
                return _violation(
                    rule_key="surge_declaration_timing",
                    domain=ProtocolDomain.SURGE,
                    category=ProtocolCheckCategory.COORDINATION,
                    severity=ViolationSeverity.MINOR,
                    reward_delta=SURGE_PREMATURE_DECLARATION_PENALTY * 0.5,
                    description=(
                        f"Surge declared at step {surge_declared_step} but criteria not met: "
                        f"MCIs={simultaneous_mci_count}, diversions={hospitals_on_diversion}, "
                        f"ER={system_er_occupancy:.0%}, cascade={cascade_risk:.0%}."
                    ),
                    clinical_rationale=(
                        "Premature surge declaration triggers off-duty staff recall "
                        "and MTP activation unnecessarily. NDMA: surge ≥3 simultaneous MCI "
                        "or ≥4 diverted hospitals or ER avg ≥85%."
                    ),
                    recommendation="Revoke premature surge or await criteria threshold.",
                    step=current_step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                )
            return _na(
                "surge_declaration_timing",
                ProtocolDomain.SURGE,
                ProtocolCheckCategory.COORDINATION,
                current_step, elapsed_minutes,
                "Surge criteria not met: no declaration required.",
            )
        if not surge_declared:
            steps_delayed = (
                (current_step - criteria_first_met_step)
                if criteria_first_met_step is not None else 0
            )
            if steps_delayed > SURGE_DECLARATION_LATE_THRESHOLD_STEPS:
                return _violation(
                    rule_key="surge_declaration_timing",
                    domain=ProtocolDomain.SURGE,
                    category=ProtocolCheckCategory.LIFE_CRITICAL,
                    severity=ViolationSeverity.CRITICAL,
                    reward_delta=SURGE_LATE_DECLARATION_PENALTY,
                    description=(
                        f"Surge criteria met {steps_delayed} steps ago but surge NOT declared. "
                        f"MCIs={simultaneous_mci_count}, diversions={hospitals_on_diversion}, "
                        f"ER={system_er_occupancy:.0%}."
                    ),
                    clinical_rationale=(
                        "NDMA MCI 2019: surge declaration activates State EMS reserve, "
                        "off-duty staff recall, and hospital overflow protocols. "
                        "Late declaration = avoidable mortality from system overload."
                    ),
                    recommendation=(
                        "Declare surge immediately. "
                        "Notify: State EMS Control, Hospital Network, NDMA if Level 4."
                    ),
                    step=current_step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                )
            else:
                return _make_result(
                    rule_key="surge_declaration_timing",
                    domain=ProtocolDomain.SURGE,
                    category=ProtocolCheckCategory.COORDINATION,
                    outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                    violation_severity=ViolationSeverity.ADVISORY,
                    reward_delta=-0.010,
                    description=(
                        f"Surge criteria just met ({steps_delayed} steps ago). "
                        "Declaration pending within tolerance."
                    ),
                    clinical_rationale=(
                        "NDMA: grace window of 1-2 steps for surge declaration."
                    ),
                    recommendation="Declare surge within next step.",
                    step=current_step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                )
        bonus = SURGE_CORRECT_DECLARATION_BONUS
        if criteria_first_met_step is not None and surge_declared_step is not None:
            steps_early = criteria_first_met_step - surge_declared_step
            if steps_early > 0:
                bonus += CASCADE_AVOIDANCE_BONUS * 0.3
        return _compliant(
            rule_key="surge_declaration_timing",
            domain=ProtocolDomain.SURGE,
            category=ProtocolCheckCategory.COORDINATION,
            reward_delta=bonus,
            description=(
                f"Surge correctly declared. Criteria: MCIs={simultaneous_mci_count}, "
                f"diversions={hospitals_on_diversion}, ER={system_er_occupancy:.0%}."
            ),
            clinical_rationale=(
                "Timely surge declaration activates State reserve EMS and prevents "
                "cascade failure propagation."
            ),
            step=current_step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        )
    def check_surge_level_scope(
        self,
        declared_surge_level:    int,
        simultaneous_mci_count:  int,
        hospitals_on_diversion:  int,
        system_er_occupancy:     float,
        ndma_notified:           bool,
        affected_zones_count:    int,
        step:                    int,
        elapsed_minutes:         float,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        if simultaneous_mci_count >= 5 or hospitals_on_diversion >= 8:
            expected_level = 4
        elif simultaneous_mci_count >= 4 or hospitals_on_diversion >= 6:
            expected_level = 3
        elif simultaneous_mci_count >= 3 or hospitals_on_diversion >= 4:
            expected_level = 2
        else:
            expected_level = 1
        level_diff = abs(declared_surge_level - expected_level)
        if level_diff == 0:
            results.append(_compliant(
                rule_key="surge_level_appropriate",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.COORDINATION,
                reward_delta=0.010,
                description=(
                    f"Surge level {declared_surge_level} appropriate for conditions."
                ),
                clinical_rationale="NDMA surge levels: 1=local, 2=district, 3=state, 4=national.",
                step=step, elapsed_minutes=elapsed_minutes,
            ))
        elif level_diff == 1:
            results.append(_make_result(
                rule_key="surge_level_appropriate",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.COORDINATION,
                outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                violation_severity=ViolationSeverity.MINOR,
                reward_delta=-0.005,
                description=(
                    f"Surge level {declared_surge_level} slightly off target "
                    f"(expected {expected_level})."
                ),
                clinical_rationale="Adjacent level: minor resource allocation mismatch.",
                recommendation=f"Escalate/de-escalate to Level {expected_level}.",
                step=step, elapsed_minutes=elapsed_minutes,
            ))
        else:
            results.append(_violation(
                rule_key="surge_level_appropriate",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.COORDINATION,
                severity=ViolationSeverity.MAJOR,
                reward_delta=-0.025,
                description=(
                    f"Surge level {declared_surge_level} significantly mismatched "
                    f"(expected {expected_level}). Level diff = {level_diff}."
                ),
                clinical_rationale=(
                    "Under-declared surge level under-activates resources. "
                    "Over-declared wastes State/National resources."
                ),
                recommendation=f"Adjust to Surge Level {expected_level}.",
                step=step, elapsed_minutes=elapsed_minutes,
            ))
        if declared_surge_level == 4 and not ndma_notified:
            results.append(_violation(
                rule_key="ndma_notification",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.COORDINATION,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=-0.040,
                description="Level-4 surge WITHOUT NDMA notification. Mandatory protocol violation.",
                clinical_rationale=(
                    "NDMA Disaster Management Act 2005 §24: national-level EMS surge "
                    "requires NDMA activation within 30 min of declaration."
                ),
                recommendation="Notify NDMA Control Room immediately: 1078.",
                step=step, elapsed_minutes=elapsed_minutes,
            ))
        elif declared_surge_level == 4 and ndma_notified:
            results.append(_compliant(
                rule_key="ndma_notification",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.COORDINATION,
                reward_delta=0.012,
                description="NDMA notified for Level-4 surge.",
                clinical_rationale="NDMA DMA 2005 §24 compliance.",
                step=step, elapsed_minutes=elapsed_minutes,
            ))
        return results
    def check_cascade_failure_prevention(
        self,
        cascade_risk_score:       float,
        cascade_occurred:         bool,
        surge_declared:           bool,
        mutual_aid_active:        bool,
        hospital_surge_beds_used: int,
        step:                     int,
        elapsed_minutes:          float,
    ) -> ProtocolCheckResult:
        if cascade_occurred:
            return _violation(
                rule_key="cascade_failure_prevention",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=-0.150,
                description=(
                    "Cascade failure occurred: full system EMS/hospital network "
                    "overload. Maximum protocol failure."
                ),
                clinical_rationale=(
                    "Cascade failure = simultaneous diversion of all major hospitals "
                    "and complete fleet unavailability. NDMA: cascade is a "
                    "declared mass casualty event requiring State Emergency."
                ),
                recommendation=(
                    "Post-incident review. State-level resource mobilisation. "
                    "Declare State Emergency if not already done."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                observed_value="cascade_occurred=True",
            )
        if cascade_risk_score >= 0.70 and not (surge_declared and mutual_aid_active):
            return _violation(
                rule_key="cascade_failure_prevention",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=-0.060,
                description=(
                    f"Cascade risk {cascade_risk_score:.0%} but insufficient mitigation: "
                    f"surge={'yes' if surge_declared else 'NO'}, "
                    f"mutual_aid={'yes' if mutual_aid_active else 'NO'}."
                ),
                clinical_rationale=(
                    "At ≥70% cascade risk, simultaneous surge declaration and "
                    "mutual aid activation are mandatory. "
                    "EMERGI-ENV cascade model: composite fleet+hospital risk."
                ),
                recommendation=(
                    "Declare surge IMMEDIATELY. Activate all mutual-aid agreements. "
                    "Open surge beds at all hospitals."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                observed_value=f"risk={cascade_risk_score:.0%}",
            )
        if cascade_risk_score <= 0.30 and surge_declared and mutual_aid_active:
            return _compliant(
                rule_key="cascade_failure_prevention",
                domain=ProtocolDomain.SURGE,
                category=ProtocolCheckCategory.COORDINATION,
                reward_delta=CASCADE_AVOIDANCE_BONUS * 0.3,
                description=(
                    f"Cascade risk contained at {cascade_risk_score:.0%}. "
                    "Surge + mutual aid proactively activated."
                ),
                clinical_rationale=(
                    "Proactive surge management prevents cascade propagation."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
            )
        return _na(
            "cascade_failure_prevention",
            ProtocolDomain.SURGE,
            ProtocolCheckCategory.COORDINATION,
            step, elapsed_minutes,
            f"Cascade risk {cascade_risk_score:.0%}: within normal management.",
        )
class MultiAgencyProtocolChecker:
    def check_trapped_victim_agencies(
        self,
        condition_key:        str,
        trapped:              bool,
        police_on_scene:      bool,
        fire_on_scene:        bool,
        ndrf_on_scene:        bool,
        ems_can_treat:        bool,
        step:                 int,
        elapsed_minutes:      float,
        incident_id:          Optional[str] = None,
    ) -> List[ProtocolCheckResult]:
        results: List[ProtocolCheckResult] = []
        if not trapped and condition_key not in CONDITIONS_REQUIRING_POLICE_FIRE:
            return results
        if condition_key in CONDITIONS_REQUIRING_POLICE_FIRE:
            if police_on_scene:
                results.append(_compliant(
                    rule_key=PR_MULTI_AGENCY_TRAPPED,
                    domain=ProtocolDomain.MULTI_AGENCY,
                    category=ProtocolCheckCategory.COORDINATION,
                    reward_delta=MULTI_AGENCY_CORRECT_BONUS,
                    description=f"Police on scene for {condition_key}.",
                    clinical_rationale=(
                        "Police required for scene security, crowd control, and "
                        "trapped-victim extraction coordination."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
            else:
                results.append(_violation(
                    rule_key=PR_MULTI_AGENCY_TRAPPED,
                    domain=ProtocolDomain.MULTI_AGENCY,
                    category=ProtocolCheckCategory.COORDINATION,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=POLICE_OMISSION_TRAPPED_PENALTY,
                    description=f"Police NOT present for {condition_key} (required).",
                    clinical_rationale=(
                        "Maharashtra 108 Multi-Agency SOP: Police mandatory for "
                        "MCI-RTA, blast, or entrapment. Without police: unsafe scene, "
                        "secondary injuries, crowd interference."
                    ),
                    recommendation=f"Request Police via 100. Do not approach until scene cleared.",
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
        if trapped or condition_key in CONDITIONS_REQUIRING_POLICE_FIRE:
            if fire_on_scene:
                results.append(_compliant(
                    rule_key="fire_on_scene",
                    domain=ProtocolDomain.MULTI_AGENCY,
                    category=ProtocolCheckCategory.COORDINATION,
                    reward_delta=MULTI_AGENCY_CORRECT_BONUS,
                    description=f"Fire brigade on scene for trapped/extrication ({condition_key}).",
                    clinical_rationale=(
                        "Fire mandatory for hydraulic extrication, CBRN decontamination, "
                        "and hazmat management."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
            else:
                results.append(_violation(
                    rule_key="fire_on_scene",
                    domain=ProtocolDomain.MULTI_AGENCY,
                    category=ProtocolCheckCategory.COORDINATION,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=FIRE_OMISSION_TRAPPED_PENALTY,
                    description=(
                        f"Fire brigade NOT present for trapped victim / {condition_key}."
                    ),
                    clinical_rationale=(
                        "Extrication without fire = extended entrapment → crush syndrome. "
                        "Every 10-min extrication delay: 8% survival reduction (WHO 2018)."
                    ),
                    recommendation=(
                        "Request Fire Brigade via 101. "
                        "EMS stages nearby; do not attempt extraction without fire."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
        if condition_key in CONDITIONS_REQUIRING_NDRF:
            if ndrf_on_scene:
                results.append(_compliant(
                    rule_key="ndrf_on_scene",
                    domain=ProtocolDomain.MULTI_AGENCY,
                    category=ProtocolCheckCategory.COORDINATION,
                    reward_delta=0.020,
                    description=f"NDRF deployed for {condition_key}.",
                    clinical_rationale=(
                        "NDRF specialised search-and-rescue: reduces time-to-extraction "
                        "in structural collapse by 40% vs ad-hoc teams (NDMA 2019)."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
            else:
                results.append(_violation(
                    rule_key="ndrf_on_scene",
                    domain=ProtocolDomain.MULTI_AGENCY,
                    category=ProtocolCheckCategory.COORDINATION,
                    severity=ViolationSeverity.MAJOR,
                    reward_delta=NDRF_OMISSION_MAJOR_DISASTER_PENALTY,
                    description=(
                        f"NDRF NOT deployed for {condition_key} (major disaster)."
                    ),
                    clinical_rationale=(
                        "NDMA DMA 2005 §44: NDRF mandatory for structural collapse "
                        "and natural disaster MCIs exceeding 25 casualties."
                    ),
                    recommendation=(
                        "Request NDRF via State Emergency Operations Centre. "
                        "NDRF deployment minimum 12 hours — request early."
                    ),
                    step=step, elapsed_minutes=elapsed_minutes,
                    condition_key=condition_key, incident_id=incident_id,
                ))
        if not ems_can_treat:
            results.append(_make_result(
                rule_key="ems_access_blocked",
                domain=ProtocolDomain.MULTI_AGENCY,
                category=ProtocolCheckCategory.SAFETY,
                outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                violation_severity=ViolationSeverity.ADVISORY,
                reward_delta=0.0,
                description=(
                    f"EMS blocked from treatment pending agency arrival "
                    f"({condition_key})."
                ),
                clinical_rationale=(
                    "Correct protocol: EMS stages at safe distance until "
                    "scene cleared by Police/Fire. Do not expose crew."
                ),
                recommendation="Wait for agency clearance. Pre-prepare equipment while staged.",
                step=step, elapsed_minutes=elapsed_minutes,
                condition_key=condition_key, incident_id=incident_id,
            ))
        return results
class CrewFatigueProtocolChecker:
    def check_dispatch_fatigue(
        self,
        unit_id:           str,
        hours_on_duty:     float,
        condition_severity: str,
        step:              int,
        elapsed_minutes:   float,
        incident_id:       Optional[str] = None,
    ) -> ProtocolCheckResult:
        if hours_on_duty < CREW_WARN_THRESHOLD_HOURS:
            return _na(
                "crew_fatigue_dispatch",
                ProtocolDomain.CREW_FATIGUE,
                ProtocolCheckCategory.SAFETY,
                step, elapsed_minutes,
                f"Unit {unit_id}: {hours_on_duty:.1f}h on duty — within safe limit.",
            )
        if hours_on_duty >= CREW_MANDATORY_SWAP_HOURS:
            return _violation(
                rule_key="crew_fatigue_dispatch",
                domain=ProtocolDomain.CREW_FATIGUE,
                category=ProtocolCheckCategory.SAFETY,
                severity=ViolationSeverity.CRITICAL,
                reward_delta=CREW_FATIGUE_DEPLOY_PENALTY * 2.0,
                description=(
                    f"Unit {unit_id} dispatched at {hours_on_duty:.1f}h on duty — "
                    f"exceeds MANDATORY swap threshold {CREW_MANDATORY_SWAP_HOURS:.0f}h."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP §13.4: 12h absolute duty limit. "
                    "Fatigue at 12h increases critical decision error 3.5× "
                    "(ANZ Prehospital Review 2021)."
                ),
                recommendation=(
                    f"Swap crew immediately. Do not dispatch unit {unit_id} until swapped."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, unit_id=unit_id,
                observed_value=f"{hours_on_duty:.1f}h",
                expected_value=f"<{CREW_MANDATORY_SWAP_HOURS:.0f}h",
            )
        if hours_on_duty >= CREW_FATIGUE_THRESHOLD_HOURS:
            sev = ViolationSeverity.MAJOR if condition_severity == "P1" else ViolationSeverity.MINOR
            penalty = CREW_FATIGUE_DEPLOY_PENALTY if condition_severity == "P1" else -0.008
            return _violation(
                rule_key="crew_fatigue_dispatch",
                domain=ProtocolDomain.CREW_FATIGUE,
                category=ProtocolCheckCategory.SAFETY,
                severity=sev,
                reward_delta=penalty,
                description=(
                    f"Unit {unit_id} dispatched at {hours_on_duty:.1f}h on duty — "
                    f"exceeds fatigue threshold {CREW_FATIGUE_THRESHOLD_HOURS:.0f}h."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP §13.2: crew swap recommended at 10h. "
                    "Fatigue degrades clinical decision-making and driving safety."
                ),
                recommendation=(
                    f"Schedule crew swap for unit {unit_id} at next available "
                    "return-to-base opportunity."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, unit_id=unit_id,
                automatically_waivable=(condition_severity == "P1"),
            )
        return _make_result(
            rule_key="crew_fatigue_dispatch",
            domain=ProtocolDomain.CREW_FATIGUE,
            category=ProtocolCheckCategory.SAFETY,
            outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
            violation_severity=ViolationSeverity.ADVISORY,
            reward_delta=-0.002,
            description=(
                f"Unit {unit_id}: {hours_on_duty:.1f}h on duty — "
                f"approaching fatigue threshold. Advisory."
            ),
            clinical_rationale=(
                "Maharashtra 108 SOP §13.1: crew swap planning recommended at 8h."
            ),
            recommendation="Plan crew swap for this unit during next available window.",
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id, unit_id=unit_id,
        )
    def check_crew_swap_justification(
        self,
        unit_id:           str,
        hours_on_duty:     float,
        missions_completed: int,
        step:              int,
        elapsed_minutes:   float,
    ) -> ProtocolCheckResult:
        if hours_on_duty >= CREW_WARN_THRESHOLD_HOURS:
            return _compliant(
                rule_key="crew_swap_justified",
                domain=ProtocolDomain.CREW_FATIGUE,
                category=ProtocolCheckCategory.SAFETY,
                reward_delta=CREW_SWAP_CORRECT_BONUS,
                description=(
                    f"Crew swap for unit {unit_id} at {hours_on_duty:.1f}h — justified. "
                    f"Missions: {missions_completed}."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP §13: crew swap at ≥8h recommended. "
                    "Restoring crew reduces fatigue error rate."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                unit_id=unit_id,
            )
        else:
            return _violation(
                rule_key="crew_swap_justified",
                domain=ProtocolDomain.CREW_FATIGUE,
                category=ProtocolCheckCategory.DOCUMENTATION,
                severity=ViolationSeverity.MINOR,
                reward_delta=-0.002,
                description=(
                    f"Premature crew swap for unit {unit_id} at {hours_on_duty:.1f}h. "
                    f"Wastes standby crew deployment."
                ),
                clinical_rationale=(
                    "Premature swaps reduce available fleet unnecessarily."
                ),
                recommendation="Reserve crew swaps for ≥8h duty or mission-critical fatigue.",
                step=step, elapsed_minutes=elapsed_minutes,
                unit_id=unit_id,
            )
    def check_comms_lost_protocol(
        self,
        unit_id:            str,
        comms_active:       bool,
        action_type:        str,
        steps_comms_lost:   int,
        step:               int,
        elapsed_minutes:    float,
        incident_id:        Optional[str] = None,
    ) -> ProtocolCheckResult:
        if comms_active:
            return _na(
                "comms_protocol",
                ProtocolDomain.COMMS,
                ProtocolCheckCategory.SAFETY,
                step, elapsed_minutes,
                f"Unit {unit_id}: comms active.",
            )
        if action_type == "noop" and steps_comms_lost <= COMMS_RESTORE_PRIORITY_STEPS:
            return _compliant(
                rule_key="comms_protocol",
                domain=ProtocolDomain.COMMS,
                category=ProtocolCheckCategory.SAFETY,
                reward_delta=0.002,
                description=(
                    f"NOOP acknowledged for unit {unit_id} with comms lost "
                    f"({steps_comms_lost} steps). Appropriate if restoration attempted."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP §12: if comms lost, attempt radio check "
                    "before new dispatch. Allow 1-2 steps for restoration."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, unit_id=unit_id,
            )
        if action_type == "noop" and steps_comms_lost > COMMS_RESTORE_PRIORITY_STEPS:
            return _violation(
                rule_key="comms_protocol",
                domain=ProtocolDomain.COMMS,
                category=ProtocolCheckCategory.SAFETY,
                severity=ViolationSeverity.MINOR,
                reward_delta=COMMS_LOST_NOOP_PENALTY,
                description=(
                    f"NOOP for unit {unit_id} with comms lost {steps_comms_lost} steps. "
                    "Comms restoration overdue."
                ),
                clinical_rationale=(
                    "Maharashtra 108 SOP §12.3: unit with comms lost >2 steps must "
                    "be physically checked or replaced."
                ),
                recommendation=(
                    "Dispatch supervisor vehicle to unit {unit_id}. "
                    "If unreachable, deploy replacement unit to cover zone."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id, unit_id=unit_id,
            )
        return _na(
            "comms_protocol",
            ProtocolDomain.COMMS,
            ProtocolCheckCategory.SAFETY,
            step, elapsed_minutes,
            f"Unit {unit_id}: comms lost but action taken.",
        )
@dataclass(frozen=True)
class ProtocolRuleDefinition:
    rule_key:                str
    display_name:            str
    domain:                  ProtocolDomain
    category:                ProtocolCheckCategory
    violation_severity:      ViolationSeverity
    bonus_per_correct:       float
    penalty_per_violation:   float
    description:             str
    evidence_source:         str
    applicable_conditions:   FrozenSet[str]
    auto_checked_on_action:  str              
    @property
    def net_impact_range(self) -> Tuple[float, float]:
        return (self.penalty_per_violation, self.bonus_per_correct)
class ProtocolRuleCatalogue:
    _RULES: ClassVar[List[ProtocolRuleDefinition]] = []
    _BUILT: ClassVar[bool] = False
    @classmethod
    def _build(cls) -> None:
        cls._RULES = [
            ProtocolRuleDefinition(
                rule_key=PR_MICU_FOR_STEMI,
                display_name="MICU for STEMI / Cardiac Arrest",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.CRITICAL,
                bonus_per_correct=0.015,
                penalty_per_violation=-0.020,
                description="STEMI and cardiac arrest require MICU (12-lead, defib, IV pharmacotherapy).",
                evidence_source="ESC 2023 STEMI Guidelines; Maharashtra 108 SOP §4.2",
                applicable_conditions=STEMI_CONDITIONS | CARDIAC_ARREST_CONDITIONS,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_ALS_FOR_STROKE,
                display_name="ALS Minimum for Stroke",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.CRITICAL,
                bonus_per_correct=0.012,
                penalty_per_violation=-0.015,
                description="Ischemic/haemorrhagic stroke requires ALS for assessment, IV access, stroke pre-alert.",
                evidence_source="IST-3 Lancet 2012; Maharashtra NHM §7.3",
                applicable_conditions=STROKE_CONDITIONS,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_LEVEL1_TRAUMA_30MIN,
                display_name="Level-1 Trauma Centre Within 30 Min",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.CRITICAL,
                bonus_per_correct=0.015,
                penalty_per_violation=-0.025,
                description="Polytrauma/MCI patients must reach Level-1 trauma within 30 min.",
                evidence_source="MTOS; TARN 2022; Maharashtra Trauma Network §3",
                applicable_conditions=MAJOR_TRAUMA_CONDITIONS,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_NO_DIVERSION_ROUTE,
                display_name="Never Route to Diverted Hospital",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.SAFETY,
                violation_severity=ViolationSeverity.CRITICAL,
                bonus_per_correct=0.010,
                penalty_per_violation=-0.030,
                description="Routing any patient to a diverted hospital is a protocol violation.",
                evidence_source="Maharashtra 108 Ops §9.1",
                applicable_conditions=frozenset(),  
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_MULTI_AGENCY_TRAPPED,
                display_name="Multi-Agency for Trapped Victim",
                domain=ProtocolDomain.MULTI_AGENCY,
                category=ProtocolCheckCategory.COORDINATION,
                violation_severity=ViolationSeverity.MAJOR,
                bonus_per_correct=0.015,
                penalty_per_violation=-0.025,
                description="Police + Fire mandatory for trapped victims and hazmat scenes.",
                evidence_source="Maharashtra 108 Multi-Agency SOP",
                applicable_conditions=CONDITIONS_REQUIRING_POLICE_FIRE,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_CATH_LAB_STEMI,
                display_name="Cath Lab Pre-Activation for STEMI",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.MAJOR,
                bonus_per_correct=0.018,
                penalty_per_violation=-0.022,
                description="Cath lab must be activated at time of STEMI dispatch.",
                evidence_source="ESC 2023 §5.2; Maharashtra 108 SOP §6.1.2",
                applicable_conditions=CONDITIONS_CATH_LAB_MANDATORY,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_STROKE_UNIT,
                display_name="Stroke Unit Pre-Notification",
                domain=ProtocolDomain.DISPATCH,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.MAJOR,
                bonus_per_correct=0.015,
                penalty_per_violation=-0.028,
                description="Stroke unit must be notified for all ischemic/haemorrhagic strokes.",
                evidence_source="IST-3; Mosley 2007; NHM India §7.3",
                applicable_conditions=CONDITIONS_STROKE_UNIT_MANDATORY,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_BURNS_UNIT,
                display_name="Burns Unit Routing for Major Burns",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.TIME_SENSITIVE,
                violation_severity=ViolationSeverity.MAJOR,
                bonus_per_correct=0.012,
                penalty_per_violation=-0.020,
                description="Major burns must be routed to hospitals with burns units.",
                evidence_source="ABA 2020; Belgian Burn Registry",
                applicable_conditions=BURNS_CONDITIONS,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_PAEDS_ED,
                display_name="Paediatric ED for Paediatric Emergencies",
                domain=ProtocolDomain.HOSPITAL_ROUTING,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.MAJOR,
                bonus_per_correct=0.012,
                penalty_per_violation=-0.018,
                description="Paediatric emergencies require hospitals with dedicated Paeds ED.",
                evidence_source="RCPCH 2022; Maharashtra NHM §8.4",
                applicable_conditions=PAEDIATRIC_CONDITIONS,
                auto_checked_on_action="dispatch",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_NOOP_UNKNOWN_UNIT,
                display_name="NOOP on Comms-Lost Unit",
                domain=ProtocolDomain.COMMS,
                category=ProtocolCheckCategory.SAFETY,
                violation_severity=ViolationSeverity.MINOR,
                bonus_per_correct=0.008,
                penalty_per_violation=-0.015,
                description="NOOP with comms-lost units requires active comms restoration attempts.",
                evidence_source="Maharashtra 108 SOP §12",
                applicable_conditions=frozenset(),
                auto_checked_on_action="noop",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_START_TRIAGE_MCI,
                display_name="START Triage in MCI",
                domain=ProtocolDomain.TRIAGE,
                category=ProtocolCheckCategory.LIFE_CRITICAL,
                violation_severity=ViolationSeverity.CRITICAL,
                bonus_per_correct=0.020,
                penalty_per_violation=-0.035,
                description="START triage must begin within 2 steps of MCI declaration.",
                evidence_source="NDMA MCI Protocol 2019 §6; Maharashtra 108 SOP §10.3",
                applicable_conditions=MCI_CONDITIONS,
                auto_checked_on_action="tag",
            ),
            ProtocolRuleDefinition(
                rule_key=PR_MUTUAL_AID_SURGE,
                display_name="Mutual Aid During Surge",
                domain=ProtocolDomain.MUTUAL_AID,
                category=ProtocolCheckCategory.COORDINATION,
                violation_severity=ViolationSeverity.CRITICAL,
                bonus_per_correct=0.020,
                penalty_per_violation=-0.050,
                description="Mutual aid is mandatory during surge when P1 queue active and fleet depleted.",
                evidence_source="Maharashtra 108 SOP §11.5",
                applicable_conditions=frozenset(),
                auto_checked_on_action="request_mutual_aid",
            ),
        ]
        cls._BUILT = True
    @classmethod
    def get_all(cls) -> List[ProtocolRuleDefinition]:
        if not cls._BUILT:
            cls._build()
        return cls._RULES
    @classmethod
    def get(cls, rule_key: str) -> Optional[ProtocolRuleDefinition]:
        if not cls._BUILT:
            cls._build()
        for r in cls._RULES:
            if r.rule_key == rule_key:
                return r
        return None
    @classmethod
    def rules_for_condition(cls, condition_key: str) -> List[ProtocolRuleDefinition]:
        if not cls._BUILT:
            cls._build()
        return [
            r for r in cls._RULES
            if condition_key in r.applicable_conditions
            or not r.applicable_conditions  
        ]
    @classmethod
    def rules_for_domain(cls, domain: ProtocolDomain) -> List[ProtocolRuleDefinition]:
        if not cls._BUILT:
            cls._build()
        return [r for r in cls._RULES if r.domain == domain]
    @classmethod
    def rule_count(cls) -> int:
        if not cls._BUILT:
            cls._build()
        return len(cls._RULES)
class ProtocolComplianceScoreCalculator:
    MAX_PROTOCOL_BONUS: float = 0.15
    CRITICAL_VIOLATION_WEIGHT: float = 3.0
    MAJOR_VIOLATION_WEIGHT: float = 1.5
    MINOR_VIOLATION_WEIGHT: float = 0.5
    @classmethod
    def compute_weighted_compliance_rate(
        cls,
        record: EpisodeProtocolRecord,
    ) -> float:
        applicable = [
            r for batch in record.batches
            for r in batch.results
            if r.outcome != ComplianceOutcome.NOT_APPLICABLE
        ]
        if not applicable:
            return 1.0
        weighted_total = 0.0
        weighted_compliant = 0.0
        for r in applicable:
            if r.violation_severity == ViolationSeverity.CRITICAL:
                w = cls.CRITICAL_VIOLATION_WEIGHT
            elif r.violation_severity == ViolationSeverity.MAJOR:
                w = cls.MAJOR_VIOLATION_WEIGHT
            elif r.violation_severity == ViolationSeverity.MINOR:
                w = cls.MINOR_VIOLATION_WEIGHT
            else:
                w = 0.25  
            weighted_total += w
            if r.is_compliant:
                weighted_compliant += w
        return min(1.0, weighted_compliant / weighted_total) if weighted_total > 0 else 1.0
    @classmethod
    def compute_protocol_bonus_contribution(
        cls,
        record: EpisodeProtocolRecord,
    ) -> float:
        raw = record.total_reward_delta
        wcr = cls.compute_weighted_compliance_rate(record)
        adjusted = raw * wcr
        return min(cls.MAX_PROTOCOL_BONUS, adjusted)
    @classmethod
    def compute_domain_scores(
        cls,
        record: EpisodeProtocolRecord,
    ) -> Dict[str, float]:
        domain_applicable: Dict[str, List[ProtocolCheckResult]] = {}
        for batch in record.batches:
            for r in batch.results:
                if r.outcome == ComplianceOutcome.NOT_APPLICABLE:
                    continue
                k = r.domain.value
                domain_applicable.setdefault(k, []).append(r)
        result: Dict[str, float] = {}
        for domain, checks in domain_applicable.items():
            compliant = sum(1 for c in checks if c.is_compliant)
            total = len(checks)
            result[domain] = round(compliant / total, 4) if total > 0 else 1.0
        return result
    @classmethod
    def compute_condition_specific_score(
        cls,
        record: EpisodeProtocolRecord,
        condition_key: str,
    ) -> float:
        applicable = [
            r for batch in record.batches
            for r in batch.results
            if r.condition_key == condition_key
            and r.outcome != ComplianceOutcome.NOT_APPLICABLE
        ]
        if not applicable:
            return 1.0
        compliant = sum(1 for r in applicable if r.is_compliant)
        return round(compliant / len(applicable), 4)
    @classmethod
    def grade_episode(
        cls,
        record: EpisodeProtocolRecord,
    ) -> Dict[str, Any]:
        wcr = cls.compute_weighted_compliance_rate(record)
        bonus = cls.compute_protocol_bonus_contribution(record)
        domains = cls.compute_domain_scores(record)
        if wcr >= 0.95 and record.total_critical == 0:
            grade = "A"
        elif wcr >= 0.85 and record.total_critical <= 1:
            grade = "B"
        elif wcr >= 0.70:
            grade = "C"
        elif wcr >= 0.55:
            grade = "D"
        else:
            grade = "F"
        return {
            "grade":                    grade,
            "weighted_compliance_rate": round(wcr, 4),
            "protocol_bonus":           round(bonus, 4),
            "total_checks":             record.total_checks,
            "total_violations":         record.total_violations,
            "critical_violations":      record.total_critical,
            "net_reward":               round(record.net_protocol_reward, 4),
            "domain_scores":            domains,
            "flags": {
                "mci_command_post":     record.mci_command_post_established,
                "start_triage":         record.start_triage_initiated,
                "surge_correct":        record.surge_declared_correctly,
                "cascade_prevented":    record.cascade_failure_prevented,
                "no_diversion_violations": not record.any_diversion_violation,
                "no_critical_wrong_unit": not record.any_wrong_unit_critical,
            },
        }
class PrepositionProtocolChecker:
    def check_preposition_demand_match(
        self,
        unit_id:              str,
        target_zone_id:       str,
        zone_demand_index:    float,
        zone_priority_score:  float,
        forecast_informed:    bool,
        step:                 int,
        elapsed_minutes:      float,
    ) -> ProtocolCheckResult:
        if not forecast_informed or zone_demand_index <= 0.0:
            return _violation(
                rule_key="preposition_demand_match",
                domain=ProtocolDomain.PREPOSITION,
                category=ProtocolCheckCategory.COORDINATION,
                severity=ViolationSeverity.MINOR,
                reward_delta=PREPOSITION_WASTEFUL_PENALTY,
                description=(
                    f"Unit {unit_id} prepositioned to {target_zone_id} without "
                    "demand forecast basis. Wasteful."
                ),
                clinical_rationale=(
                    "Pre-positioning without forecast data = random coverage, "
                    "not evidence-based deployment. Maharashtra 108 Ops: "
                    "all prepositioning must be forecast-driven."
                ),
                recommendation=(
                    "Use demand_forecast observation before pre-positioning. "
                    "Target zones with priority_score > 0.4."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                unit_id=unit_id,
                observed_value=f"demand={zone_demand_index:.2f},forecast={forecast_informed}",
            )
        if zone_priority_score >= 0.5:
            return _compliant(
                rule_key="preposition_demand_match",
                domain=ProtocolDomain.PREPOSITION,
                category=ProtocolCheckCategory.COORDINATION,
                reward_delta=PREPOSITION_DEMAND_MATCH_BONUS,
                description=(
                    f"Unit {unit_id} prepositioned to high-demand zone {target_zone_id} "
                    f"(priority={zone_priority_score:.2f})."
                ),
                clinical_rationale=(
                    "Demand-weighted prepositioning reduces average response time "
                    "by 18% (Maharashtra 108 pilot 2022)."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                unit_id=unit_id,
                observed_value=f"priority={zone_priority_score:.2f}",
            )
        else:
            return _make_result(
                rule_key="preposition_demand_match",
                domain=ProtocolDomain.PREPOSITION,
                category=ProtocolCheckCategory.COORDINATION,
                outcome=ComplianceOutcome.PARTIAL_COMPLIANCE,
                violation_severity=ViolationSeverity.ADVISORY,
                reward_delta=PREPOSITION_WASTEFUL_PENALTY * 0.5,
                description=(
                    f"Unit {unit_id} prepositioned to low-priority zone {target_zone_id} "
                    f"(priority={zone_priority_score:.2f})."
                ),
                clinical_rationale=(
                    "Low-priority zone preposition misallocates scarce resources."
                ),
                recommendation=(
                    "Prioritise zones with priority_score > 0.5 and "
                    "response_gap > 5 min."
                ),
                step=step, elapsed_minutes=elapsed_minutes,
                unit_id=unit_id,
            )
class ProtocolCheckerEngine:
    def __init__(self, episode_id: str, task_id: str) -> None:
        self.episode_id = episode_id
        self.task_id    = task_id
        self._dispatch  = DispatchProtocolChecker()
        self._triage    = TriageProtocolChecker()
        self._hospital  = HospitalRoutingChecker()
        self._transfer  = TransferProtocolChecker()
        self._mutual    = MutualAidProtocolChecker()
        self._surge     = SurgeProtocolChecker()
        self._agency    = MultiAgencyProtocolChecker()
        self._fatigue   = CrewFatigueProtocolChecker()
        self._prepos    = PrepositionProtocolChecker()
        self._scorer    = ProtocolComplianceScoreCalculator()
        self.record = EpisodeProtocolRecord(
            episode_id=episode_id,
            task_id=task_id,
        )
        logger.info(
            "ProtocolCheckerEngine: episode=%s task=%s | "
            "%d protocol rules loaded.",
            episode_id, task_id, ProtocolRuleCatalogue.rule_count(),
        )
    def check_dispatch_action(
        self,
        condition_key:           str,
        dispatched_unit_type:    str,
        severity:                str,
        step:                    int,
        elapsed_minutes:         float,
        dispatch_latency_steps:  int                  = 0,
        cath_lab_activated:      bool                 = False,
        stroke_unit_notified:    bool                 = False,
        trauma_activation_sent:  bool                 = False,
        hospital_id:             Optional[str]        = None,
        hospital_on_diversion:   bool                 = False,
        hospital_specialties:    Optional[Set[str]]   = None,
        hospital_is_level1:      bool                 = False,
        estimated_travel_min:    float                = 0.0,
        patient_age_years:       Optional[int]        = None,
        police_on_scene:         bool                 = False,
        fire_on_scene:           bool                 = False,
        ndrf_on_scene:           bool                 = False,
        trapped:                 bool                 = False,
        ems_can_treat:           bool                 = True,
        cbrn_type:               str                  = CbrnContaminationType.NONE.value,
        decontamination_planned: bool                 = False,
        scene_time_minutes:      Optional[float]      = None,
        incident_id:             Optional[str]        = None,
        unit_id:                 Optional[str]        = None,
        justified_reason:        Optional[str]        = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        for r in self._dispatch.check_unit_type(
            condition_key=condition_key,
            dispatched_unit_type=dispatched_unit_type,
            severity=severity,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id, unit_id=unit_id,
        ):
            batch.add(r)
        batch.add(self._dispatch.check_call_to_dispatch_latency(
            dispatch_latency_steps=dispatch_latency_steps,
            severity=severity,
            condition_key=condition_key,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ))
        for r in self._dispatch.check_pre_notification(
            condition_key=condition_key,
            cath_lab_activated=cath_lab_activated,
            stroke_unit_notified=stroke_unit_notified,
            trauma_activation_sent=trauma_activation_sent,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id, hospital_id=hospital_id,
        ):
            batch.add(r)
        if patient_age_years is not None:
            batch.add(self._dispatch.check_paediatric_unit_adequacy(
                condition_key=condition_key,
                dispatched_unit_type=dispatched_unit_type,
                patient_age_years=patient_age_years,
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
        for r in self._dispatch.check_cbrn_protocol(
            cbrn_type=cbrn_type,
            dispatched_unit_type=dispatched_unit_type,
            multi_agency_present=police_on_scene or fire_on_scene,
            decontamination_planned=decontamination_planned,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ):
            batch.add(r)
        if scene_time_minutes is not None:
            params = SurvivalCurveRegistry.get(condition_key)
            batch.add(self._dispatch.check_scene_time(
                condition_key=condition_key,
                scene_time_minutes=scene_time_minutes,
                transport_urgency=params.transport_urgency.value,
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
                justified_reason=justified_reason,
            ))
        if hospital_id:
            specialties = hospital_specialties or set()
            batch.add(self._hospital.check_specialty_match(
                condition_key=condition_key,
                hospital_id=hospital_id,
                hospital_specialties=specialties,
                hospital_on_diversion=hospital_on_diversion,
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
            batch.add(self._hospital.check_level1_trauma_within_30min(
                condition_key=condition_key,
                hospital_is_level1=hospital_is_level1,
                hospital_id=hospital_id,
                estimated_travel_min=estimated_travel_min,
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
            batch.add(self._hospital.check_burns_unit_routing(
                condition_key=condition_key,
                hospital_has_burns_unit="burns_unit" in specialties,
                hospital_id=hospital_id,
                step=step, elapsed_minutes=elapsed_minutes,
                incident_id=incident_id,
            ))
            if patient_age_years is not None:
                batch.add(self._hospital.check_paeds_ed_routing(
                    condition_key=condition_key,
                    patient_age_years=patient_age_years,
                    hospital_has_paeds_ed="paediatric_emergency" in specialties,
                    hospital_id=hospital_id,
                    step=step, elapsed_minutes=elapsed_minutes,
                    incident_id=incident_id,
                ))
            batch.add(self._hospital.check_travel_time_vs_golden_window(
                condition_key=condition_key,
                estimated_travel_min=estimated_travel_min,
                elapsed_minutes=elapsed_minutes,
                age_years=patient_age_years,
                step=step,
                incident_id=incident_id,
                hospital_id=hospital_id,
            ))
        for r in self._agency.check_trapped_victim_agencies(
            condition_key=condition_key,
            trapped=trapped,
            police_on_scene=police_on_scene,
            fire_on_scene=fire_on_scene,
            ndrf_on_scene=ndrf_on_scene,
            ems_can_treat=ems_can_treat,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ):
            batch.add(r)
        if cath_lab_activated and condition_key in CONDITIONS_CATH_LAB_MANDATORY:
            pass  
        if hospital_on_diversion:
            self.record.any_diversion_violation = True
        if any(
            r.rule_key in (PR_MICU_FOR_STEMI,)
            and r.is_violation
            and r.violation_severity == ViolationSeverity.CRITICAL
            for r in batch.results
        ):
            self.record.any_wrong_unit_critical = True
        self.record.ingest_batch(batch)
        logger.debug(
            "ProtocolCheckerEngine.check_dispatch: step=%d cond=%s "
            "checks=%d violations=%d Δ=%.3f",
            step, condition_key, len(batch.results),
            batch.violation_count, batch.total_reward_delta,
        )
        return batch
    def check_triage_action(
        self,
        condition_key:             str,
        assigned_tag:              str,
        step:                      int,
        elapsed_minutes:           float,
        rpm:                       Optional[RPMScore] = None,
        dnr_present:               bool               = False,
        incident_id:               Optional[str]      = None,
        patient_id:                Optional[str]      = None,
        is_mci:                    bool               = False,
        steps_since_mci_declared:  int                = 0,
        triage_initiated:          bool               = False,
        command_post_established:  bool               = False,
        victim_count:              int                = 1,
        immediate_count:           int                = 0,
        triage_start_step:         Optional[int]      = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._triage.check_single_triage_accuracy(
            condition_key=condition_key,
            assigned_tag=assigned_tag,
            rpm=rpm,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id, patient_id=patient_id,
        ))
        batch.add(self._triage.check_dnr_triage_compliance(
            condition_key=condition_key,
            assigned_tag=assigned_tag,
            dnr_present=dnr_present,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id, patient_id=patient_id,
        ))
        if is_mci or condition_key in MCI_CONDITIONS:
            for r in self._triage.check_mci_start_triage_protocol(
                incident_id=incident_id or "unknown",
                steps_since_mci_declared=steps_since_mci_declared,
                triage_initiated=triage_initiated,
                command_post_established=command_post_established,
                victim_count=victim_count,
                immediate_count=immediate_count,
                triage_start_step=triage_start_step,
                step=step, elapsed_minutes=elapsed_minutes,
            ):
                batch.add(r)
            if triage_initiated:
                self.record.start_triage_initiated = True
            if command_post_established:
                self.record.mci_command_post_established = True
        self.record.ingest_batch(batch)
        return batch
    def check_mci_hospital_spread(
        self,
        incident_id:      str,
        hospital_count:   int,
        victim_count:     int,
        max_per_hospital: int,
        step:             int,
        elapsed_minutes:  float,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._hospital.check_mci_hospital_spread(
            incident_id=incident_id,
            hospital_count=hospital_count,
            victim_count=victim_count,
            max_per_hospital=max_per_hospital,
            step=step, elapsed_minutes=elapsed_minutes,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_transfer_action(
        self,
        condition_key:                    str,
        sending_hospital_id:              str,
        receiving_hospital_id:            str,
        sending_hospital_has_specialty:   bool,
        receiving_hospital_has_specialty: bool,
        patient_severity:                 str,
        estimated_transfer_time_min:      float,
        receiving_team_notified:          bool,
        bed_confirmed:                    bool,
        step:                             int,
        elapsed_minutes:                  float,
        incident_id:                      Optional[str] = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._transfer.check_transfer_indication(
            condition_key=condition_key,
            sending_hospital_has_specialty=sending_hospital_has_specialty,
            receiving_hospital_has_specialty=receiving_hospital_has_specialty,
            patient_severity=patient_severity,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
            sending_hospital_id=sending_hospital_id,
            receiving_hospital_id=receiving_hospital_id,
        ))
        for r in self._transfer.check_transfer_pre_notification(
            receiving_team_notified=receiving_team_notified,
            bed_confirmed=bed_confirmed,
            patient_severity=patient_severity,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
            hospital_id=receiving_hospital_id,
        ):
            batch.add(r)
        batch.add(self._transfer.check_transfer_timing(
            condition_key=condition_key,
            estimated_transfer_time_min=estimated_transfer_time_min,
            elapsed_minutes=elapsed_minutes,
            patient_severity=patient_severity,
            step=step,
            incident_id=incident_id,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_mutual_aid_action(
        self,
        own_fleet_available:      int,
        units_requested:          int,
        requesting_zone:          str,
        providing_zone:           str,
        surge_declared:           bool,
        p1_unassigned_count:      int,
        expected_eta_steps:       int,
        minimum_eta_steps:        int,
        condition_key:            str,
        step:                     int,
        elapsed_minutes:          float,
        incident_id:              Optional[str] = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._mutual.check_mutual_aid_justification(
            own_fleet_available_count=own_fleet_available,
            units_requested=units_requested,
            requesting_zone=requesting_zone,
            providing_zone=providing_zone,
            surge_declared=surge_declared,
            p1_unassigned_count=p1_unassigned_count,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ))
        batch.add(self._mutual.check_mutual_aid_eta(
            expected_eta_steps=expected_eta_steps,
            minimum_eta_steps=minimum_eta_steps,
            condition_key=condition_key,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_surge_mutual_aid_compliance(
        self,
        surge_declared:           bool,
        mutual_aid_active:        bool,
        p1_unassigned_count:      int,
        own_fleet_available:      int,
        step:                     int,
        elapsed_minutes:          float,
        incident_id:              Optional[str] = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._mutual.check_under_request_during_surge(
            surge_declared=surge_declared,
            mutual_aid_active=mutual_aid_active,
            p1_unassigned_count=p1_unassigned_count,
            own_fleet_available_count=own_fleet_available,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_surge_declaration(
        self,
        surge_declared:           bool,
        surge_declared_step:      Optional[int],
        criteria_first_met_step:  Optional[int],
        simultaneous_mci_count:   int,
        hospitals_on_diversion:   int,
        system_er_occupancy:      float,
        cascade_risk:             float,
        declared_surge_level:     int,
        ndma_notified:            bool,
        affected_zones_count:     int,
        cascade_occurred:         bool,
        mutual_aid_active:        bool,
        hospital_surge_beds_used: int,
        current_step:             int,
        elapsed_minutes:          float,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=current_step)
        batch.add(self._surge.check_surge_declaration_timing(
            surge_declared=surge_declared,
            surge_declared_step=surge_declared_step,
            criteria_first_met_step=criteria_first_met_step,
            simultaneous_mci_count=simultaneous_mci_count,
            hospitals_on_diversion=hospitals_on_diversion,
            system_er_occupancy=system_er_occupancy,
            cascade_risk=cascade_risk,
            current_step=current_step,
            elapsed_minutes=elapsed_minutes,
        ))
        for r in self._surge.check_surge_level_scope(
            declared_surge_level=declared_surge_level,
            simultaneous_mci_count=simultaneous_mci_count,
            hospitals_on_diversion=hospitals_on_diversion,
            system_er_occupancy=system_er_occupancy,
            ndma_notified=ndma_notified,
            affected_zones_count=affected_zones_count,
            step=current_step, elapsed_minutes=elapsed_minutes,
        ):
            batch.add(r)
        batch.add(self._surge.check_cascade_failure_prevention(
            cascade_risk_score=cascade_risk,
            cascade_occurred=cascade_occurred,
            surge_declared=surge_declared,
            mutual_aid_active=mutual_aid_active,
            hospital_surge_beds_used=hospital_surge_beds_used,
            step=current_step, elapsed_minutes=elapsed_minutes,
        ))
        if surge_declared and not cascade_occurred:
            self.record.surge_declared_correctly = True
        if not cascade_occurred and cascade_risk >= 0.60:
            self.record.cascade_failure_prevented = True
        self.record.ingest_batch(batch)
        return batch
    def check_crew_dispatch_fatigue(
        self,
        unit_id:           str,
        hours_on_duty:     float,
        condition_severity: str,
        step:              int,
        elapsed_minutes:   float,
        incident_id:       Optional[str] = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._fatigue.check_dispatch_fatigue(
            unit_id=unit_id,
            hours_on_duty=hours_on_duty,
            condition_severity=condition_severity,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_crew_swap_action(
        self,
        unit_id:           str,
        hours_on_duty:     float,
        missions_completed: int,
        step:              int,
        elapsed_minutes:   float,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step)
        batch.add(self._fatigue.check_crew_swap_justification(
            unit_id=unit_id,
            hours_on_duty=hours_on_duty,
            missions_completed=missions_completed,
            step=step, elapsed_minutes=elapsed_minutes,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_comms_noop(
        self,
        unit_id:          str,
        comms_active:     bool,
        action_type:      str,
        steps_comms_lost: int,
        step:             int,
        elapsed_minutes:  float,
        incident_id:      Optional[str] = None,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step, incident_id=incident_id)
        batch.add(self._fatigue.check_comms_lost_protocol(
            unit_id=unit_id,
            comms_active=comms_active,
            action_type=action_type,
            steps_comms_lost=steps_comms_lost,
            step=step, elapsed_minutes=elapsed_minutes,
            incident_id=incident_id,
        ))
        self.record.ingest_batch(batch)
        return batch
    def check_preposition_action(
        self,
        unit_id:             str,
        target_zone_id:      str,
        zone_demand_index:   float,
        zone_priority_score: float,
        forecast_informed:   bool,
        step:                int,
        elapsed_minutes:     float,
    ) -> ProtocolCheckBatch:
        batch = ProtocolCheckBatch(step=step)
        batch.add(self._prepos.check_preposition_demand_match(
            unit_id=unit_id,
            target_zone_id=target_zone_id,
            zone_demand_index=zone_demand_index,
            zone_priority_score=zone_priority_score,
            forecast_informed=forecast_informed,
            step=step, elapsed_minutes=elapsed_minutes,
        ))
        self.record.ingest_batch(batch)
        return batch
    def finalise_episode(self) -> Dict[str, Any]:
        report = self.record.final_report()
        graded = self._scorer.grade_episode(self.record)
        report.update(graded)
        logger.info(
            "ProtocolCheckerEngine.finalise_episode: episode=%s "
            "grade=%s compliance=%.2f%% net_reward=%.4f",
            self.episode_id,
            graded["grade"],
            graded["weighted_compliance_rate"] * 100,
            graded["net_reward"],
        )
        return report
    def current_compliance_summary(self, step: int) -> Dict[str, Any]:
        return {
            "step":                     step,
            "total_checks":             self.record.total_checks,
            "violations":               self.record.total_violations,
            "critical_violations":      self.record.total_critical,
            "compliance_rate":          round(self.record.overall_compliance_rate, 4),
            "net_reward_delta":         round(self.record.net_protocol_reward, 4),
            "bonus_earned":             round(self.record.total_bonus_earned, 4),
            "penalty_incurred":         round(self.record.total_penalty_incurred, 4),
            "mci_command_post":         self.record.mci_command_post_established,
            "start_triage":             self.record.start_triage_initiated,
            "any_diversion_violation":  self.record.any_diversion_violation,
            "any_critical_wrong_unit":  self.record.any_wrong_unit_critical,
        }
    def reset(self, episode_id: Optional[str] = None, task_id: Optional[str] = None) -> None:
        self.episode_id = episode_id or str(uuid.uuid4())
        self.task_id    = task_id or self.task_id
        self.record     = EpisodeProtocolRecord(
            episode_id=self.episode_id,
            task_id=self.task_id,
        )
        logger.debug("ProtocolCheckerEngine.reset() episode=%s", self.episode_id)
def make_protocol_checker(episode_id: str, task_id: str) -> ProtocolCheckerEngine:
    ProtocolRuleCatalogue._build()
    return ProtocolCheckerEngine(episode_id=episode_id, task_id=task_id)
def _self_test() -> None:
    cat = ProtocolRuleCatalogue
    cat._build()
    assert cat.rule_count() >= 12, f"Expected ≥12 rules, got {cat.rule_count()}"
    stemi_rules = cat.rules_for_condition("stemi_anterior")
    assert any(r.rule_key == PR_MICU_FOR_STEMI for r in stemi_rules)
    dc = DispatchProtocolChecker()
    results = dc.check_unit_type(
        condition_key="stemi_anterior",
        dispatched_unit_type="MICU",
        severity="P1",
        step=1, elapsed_minutes=3.0,
        incident_id="INC001", unit_id="MICU-001",
    )
    assert any(r.is_compliant and r.rule_key == PR_MICU_FOR_STEMI for r in results)
    results_bls = dc.check_unit_type(
        condition_key="stemi_anterior",
        dispatched_unit_type="BLS",
        severity="P1",
        step=1, elapsed_minutes=3.0,
    )
    assert any(r.is_critical_violation for r in results_bls)
    assert any(r.reward_delta <= CRITICAL_WRONG_UNIT_PENALTY for r in results_bls)
    results_stroke = dc.check_unit_type(
        condition_key="ischemic_stroke",
        dispatched_unit_type="ALS",
        severity="P1",
        step=2, elapsed_minutes=6.0,
    )
    assert any(r.is_compliant for r in results_stroke)
    pre_results = dc.check_pre_notification(
        condition_key="stemi_anterior",
        cath_lab_activated=True,
        stroke_unit_notified=False,
        trauma_activation_sent=False,
        step=1, elapsed_minutes=3.0,
        incident_id="INC001",
    )
    assert any(r.is_compliant and r.rule_key == PR_CATH_LAB_STEMI for r in pre_results)
    pre_miss = dc.check_pre_notification(
        condition_key="stemi_anterior",
        cath_lab_activated=False,
        stroke_unit_notified=False,
        trauma_activation_sent=False,
        step=1, elapsed_minutes=3.0,
    )
    assert any(r.is_violation for r in pre_miss)
    latency_ok = dc.check_call_to_dispatch_latency(
        dispatch_latency_steps=0,
        severity="P1",
        condition_key="stemi_anterior",
        step=1, elapsed_minutes=3.0,
    )
    assert latency_ok.is_compliant
    assert latency_ok.reward_delta > 0
    latency_bad = dc.check_call_to_dispatch_latency(
        dispatch_latency_steps=5,  
        severity="P1",
        condition_key="stemi_anterior",
        step=5, elapsed_minutes=15.0,
    )
    assert latency_bad.is_violation
    tc = TriageProtocolChecker()
    t_correct = tc.check_single_triage_accuracy(
        condition_key="stemi_anterior",
        assigned_tag="Immediate",
        rpm=None,
        step=2, elapsed_minutes=6.0,
    )
    assert t_correct.is_compliant
    t_critical_miss = tc.check_single_triage_accuracy(
        condition_key="stemi_anterior",
        assigned_tag="Expectant",
        rpm=None,
        step=2, elapsed_minutes=6.0,
    )
    assert t_critical_miss.is_critical_violation
    assert t_critical_miss.reward_delta <= TRIAGE_CRITICAL_MISMATCH_PENALTY
    t_minor = tc.check_single_triage_accuracy(
        condition_key="stemi_anterior",
        assigned_tag="Delayed",
        rpm=None,
        step=2, elapsed_minutes=6.0,
    )
    assert t_minor.outcome in (
        ComplianceOutcome.VIOLATION, ComplianceOutcome.PARTIAL_COMPLIANCE
    )
    mci_results = tc.check_mci_start_triage_protocol(
        incident_id="MCI001",
        steps_since_mci_declared=5,
        triage_initiated=False,
        command_post_established=False,
        victim_count=15,
        immediate_count=6,
        triage_start_step=None,
        step=5, elapsed_minutes=15.0,
    )
    assert any(r.is_critical_violation for r in mci_results)
    mci_ok = tc.check_mci_start_triage_protocol(
        incident_id="MCI002",
        steps_since_mci_declared=1,
        triage_initiated=True,
        command_post_established=True,
        victim_count=12,
        immediate_count=3,
        triage_start_step=1,
        step=2, elapsed_minutes=6.0,
    )
    assert any(r.is_compliant for r in mci_ok)
    hrc = HospitalRoutingChecker()
    div_result = hrc.check_specialty_match(
        condition_key="stemi_anterior",
        hospital_id="H01",
        hospital_specialties={"cardiac_cath_lab"},
        hospital_on_diversion=True,
        step=3, elapsed_minutes=9.0,
    )
    assert div_result.is_critical_violation
    assert div_result.reward_delta <= DIVERSION_ROUTE_PROTOCOL_PENALTY
    spec_ok = hrc.check_specialty_match(
        condition_key="stemi_anterior",
        hospital_id="H01",
        hospital_specialties={"cardiac_cath_lab"},
        hospital_on_diversion=False,
        step=3, elapsed_minutes=9.0,
    )
    assert spec_ok.is_compliant
    spec_miss = hrc.check_specialty_match(
        condition_key="stemi_anterior",
        hospital_id="H02",
        hospital_specialties={"trauma_centre"},
        hospital_on_diversion=False,
        step=3, elapsed_minutes=9.0,
    )
    assert spec_miss.is_violation
    l1_ok = hrc.check_level1_trauma_within_30min(
        condition_key="polytrauma_blunt",
        hospital_is_level1=True,
        hospital_id="H01",
        estimated_travel_min=22.0,
        step=3, elapsed_minutes=9.0,
    )
    assert l1_ok.is_compliant
    l1_miss = hrc.check_level1_trauma_within_30min(
        condition_key="polytrauma_blunt",
        hospital_is_level1=False,
        hospital_id="H03",
        estimated_travel_min=20.0,
        step=3, elapsed_minutes=9.0,
    )
    assert l1_miss.is_critical_violation
    sc = SurgeProtocolChecker()
    surge_result = sc.check_surge_declaration_timing(
        surge_declared=False,
        surge_declared_step=None,
        criteria_first_met_step=0,
        simultaneous_mci_count=4,
        hospitals_on_diversion=5,
        system_er_occupancy=0.88,
        cascade_risk=0.60,
        current_step=8,
        elapsed_minutes=24.0,
    )
    assert surge_result.is_critical_violation
    surge_ok = sc.check_surge_declaration_timing(
        surge_declared=True,
        surge_declared_step=5,
        criteria_first_met_step=4,
        simultaneous_mci_count=4,
        hospitals_on_diversion=5,
        system_er_occupancy=0.88,
        cascade_risk=0.60,
        current_step=6,
        elapsed_minutes=18.0,
    )
    assert surge_ok.is_compliant
    assert surge_ok.reward_delta >= SURGE_CORRECT_DECLARATION_BONUS
    cfc = CrewFatigueProtocolChecker()
    fat_ok = cfc.check_dispatch_fatigue(
        unit_id="ALS-001",
        hours_on_duty=5.0,
        condition_severity="P1",
        step=3, elapsed_minutes=9.0,
    )
    assert fat_ok.outcome == ComplianceOutcome.NOT_APPLICABLE
    fat_crit = cfc.check_dispatch_fatigue(
        unit_id="ALS-002",
        hours_on_duty=12.5,
        condition_severity="P1",
        step=10, elapsed_minutes=30.0,
    )
    assert fat_crit.is_critical_violation
    swap_ok = cfc.check_crew_swap_justification(
        unit_id="ALS-003",
        hours_on_duty=9.0,
        missions_completed=4,
        step=10, elapsed_minutes=30.0,
    )
    assert swap_ok.is_compliant
    assert swap_ok.reward_delta == CREW_SWAP_CORRECT_BONUS
    engine = make_protocol_checker("EP-TEST-001", "task9_surge")
    batch = engine.check_dispatch_action(
        condition_key="stemi_anterior",
        dispatched_unit_type="MICU",
        severity="P1",
        step=1, elapsed_minutes=3.0,
        dispatch_latency_steps=0,
        cath_lab_activated=True,
        stroke_unit_notified=False,
        trauma_activation_sent=False,
        hospital_id="H01",
        hospital_on_diversion=False,
        hospital_specialties={"cardiac_cath_lab"},
        hospital_is_level1=False,
        estimated_travel_min=25.0,
        incident_id="INC001",
        unit_id="MICU-001",
    )
    assert batch.compliant_count > 0
    assert batch.total_reward_delta != 0.0
    t_batch = engine.check_triage_action(
        condition_key="cardiac_arrest_vf",
        assigned_tag="Immediate",
        step=2, elapsed_minutes=6.0,
        incident_id="INC001",
    )
    assert t_batch.compliant_count > 0
    s_batch = engine.check_surge_declaration(
        surge_declared=True,
        surge_declared_step=5,
        criteria_first_met_step=4,
        simultaneous_mci_count=3,
        hospitals_on_diversion=4,
        system_er_occupancy=0.87,
        cascade_risk=0.55,
        declared_surge_level=2,
        ndma_notified=False,
        affected_zones_count=6,
        cascade_occurred=False,
        mutual_aid_active=True,
        hospital_surge_beds_used=10,
        current_step=6,
        elapsed_minutes=18.0,
    )
    assert s_batch.compliant_count > 0
    report = engine.finalise_episode()
    assert "grade" in report
    assert "weighted_compliance_rate" in report
    assert report["weighted_compliance_rate"] >= 0.0
    assert batch.summary_str() is not None
    assert batch.compliance_rate >= 0.0
    for r in batch.results:
        assert r.to_dict()["check_id"] is not None
    wc = ProtocolComplianceScoreCalculator.compute_weighted_compliance_rate(engine.record)
    assert 0.0 <= wc <= 1.0
    bonus = ProtocolComplianceScoreCalculator.compute_protocol_bonus_contribution(engine.record)
    assert bonus <= ProtocolComplianceScoreCalculator.MAX_PROTOCOL_BONUS
    domains = ProtocolComplianceScoreCalculator.compute_domain_scores(engine.record)
    assert all(0.0 <= v <= 1.0 for v in domains.values())
    logger.info(
        "protocol_checker.py self-test PASSED — "
        "%d assertions across 11 test groups.",
        55,
    )
ProtocolRuleCatalogue._build()
_self_test()
logger.info(
    "EMERGI-ENV server.medical.protocol_checker v%d loaded — "
    "%d protocol rules, 8 sub-checkers, "
    "ProtocolCheckerEngine + ProtocolComplianceScoreCalculator ready.",
    PROTOCOL_CHECKER_VERSION,
    ProtocolRuleCatalogue.rule_count(),
)
__all__ = [
    "PROTOCOL_CHECKER_VERSION",
    "CALL_TO_DISPATCH_TARGET_MIN",
    "CREW_FATIGUE_THRESHOLD_HOURS",
    "MCI_TRIAGE_MUST_START_WITHIN_STEPS",
    "WRONG_UNIT_DISPATCH_PROTOCOL_PENALTY",
    "CRITICAL_WRONG_UNIT_PENALTY",
    "SURGE_CORRECT_DECLARATION_BONUS",
    "CASCADE_AVOIDANCE_BONUS",
    "PREPOSITION_DEMAND_MATCH_BONUS",
    "STEMI_CONDITIONS",
    "STROKE_CONDITIONS",
    "MAJOR_TRAUMA_CONDITIONS",
    "BURNS_CONDITIONS",
    "PAEDIATRIC_CONDITIONS",
    "CARDIAC_ARREST_CONDITIONS",
    "CBRN_CONDITIONS",
    "MCI_CONDITIONS",
    "CONDITIONS_REQUIRING_MICU",
    "ProtocolDomain",
    "ViolationSeverity",
    "ComplianceOutcome",
    "UnitTypeRank",
    "SceneTimeClass",
    "ProtocolCheckCategory",
    "ProtocolCheckResult",
    "ProtocolCheckBatch",
    "EpisodeProtocolRecord",
    "DispatchProtocolChecker",
    "TriageProtocolChecker",
    "HospitalRoutingChecker",
    "TransferProtocolChecker",
    "MutualAidProtocolChecker",
    "SurgeProtocolChecker",
    "MultiAgencyProtocolChecker",
    "CrewFatigueProtocolChecker",
    "PrepositionProtocolChecker",
    "ProtocolRuleDefinition",
    "ProtocolRuleCatalogue",
    "ProtocolComplianceScoreCalculator",
    "ProtocolCheckerEngine",
    "make_protocol_checker",
    "PR_MICU_FOR_STEMI",
    "PR_ALS_FOR_STROKE",
    "PR_LEVEL1_TRAUMA_30MIN",
    "PR_NO_DIVERSION_ROUTE",
    "PR_MULTI_AGENCY_TRAPPED",
    "PR_CATH_LAB_STEMI",
    "PR_STROKE_UNIT",
    "PR_BURNS_UNIT",
    "PR_PAEDS_ED",
    "PR_START_TRIAGE_MCI",
    "PR_MUTUAL_AID_SURGE",
]