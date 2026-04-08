from __future__ import annotations
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Final, FrozenSet, List, Optional, Tuple, Union
logger = logging.getLogger("emergi_env.medical.triage")
TRIAGE_VERSION: Final[int] = 3
PAEDIATRIC_AGE_THRESHOLD: Final[int] = 8          
INFANT_AGE_THRESHOLD: Final[int] = 1               
START_RR_MIN: Final[int] = 10
START_RR_MAX: Final[int] = 29
START_RADIAL_PULSE_PRESENT: Final[bool] = True
JUMPSTART_RR_MIN: Final[int] = 15
JUMPSTART_RR_MAX: Final[int] = 45
JUMPSTART_BRACHIAL_PULSE_PRESENT: Final[bool] = True
SALT_OBVIOUSLY_DEAD_CRITERIA: Final[FrozenSet[str]] = frozenset([
    "rigor_mortis",
    "dependent_lividity",
    "decapitation",
    "transection_torso",
    "decomposition",
])
DETERIORATION_STEPS: Final[Dict[str, int]] = {
    "delayed_to_immediate":  3,
    "minimal_to_delayed":    8,
    "immediate_to_expectant": 6,   
}
TRIAGE_EXACT_MATCH_SCORE: Final[float]           = 1.00
TRIAGE_ADJACENT_MISMATCH_SCORE: Final[float]     = 0.50
TRIAGE_DISTANT_MISMATCH_SCORE: Final[float]      = 0.20
TRIAGE_CRITICAL_MISMATCH_SCORE: Final[float]     = 0.00
TRIAGE_CRITICAL_MISMATCH_PENALTY: Final[float]   = -0.50
class TriageTag(str, Enum):
    IMMEDIATE  = "Immediate"    
    DELAYED    = "Delayed"      
    MINIMAL    = "Minimal"      
    EXPECTANT  = "Expectant"    
    @property
    def colour(self) -> str:
        return {
            TriageTag.IMMEDIATE: "RED",
            TriageTag.DELAYED:   "YELLOW",
            TriageTag.MINIMAL:   "GREEN",
            TriageTag.EXPECTANT: "BLACK",
        }[self]
    @property
    def indian_ems_priority(self) -> str:
        return {
            TriageTag.IMMEDIATE: "P1",
            TriageTag.DELAYED:   "P2",
            TriageTag.MINIMAL:   "P3",
            TriageTag.EXPECTANT: "P0",
        }[self]
    @property
    def response_target_minutes(self) -> int:
        return {
            TriageTag.IMMEDIATE: 8,
            TriageTag.DELAYED:   30,
            TriageTag.MINIMAL:   120,
            TriageTag.EXPECTANT: 9999,
        }[self]
    @property
    def numerical_rank(self) -> int:
        return {
            TriageTag.IMMEDIATE: 1,
            TriageTag.DELAYED:   2,
            TriageTag.MINIMAL:   3,
            TriageTag.EXPECTANT: 4,
        }[self]
    def is_more_urgent_than(self, other: "TriageTag") -> bool:
        return self.numerical_rank < other.numerical_rank
    def adjacent_tags(self) -> FrozenSet["TriageTag"]:
        adjacency: Dict["TriageTag", FrozenSet["TriageTag"]] = {
            TriageTag.IMMEDIATE: frozenset([TriageTag.DELAYED]),
            TriageTag.DELAYED:   frozenset([TriageTag.IMMEDIATE, TriageTag.MINIMAL]),
            TriageTag.MINIMAL:   frozenset([TriageTag.DELAYED]),
            TriageTag.EXPECTANT: frozenset([TriageTag.IMMEDIATE]),  
        }
        return adjacency.get(self, frozenset())
class RespirationStatus(str, Enum):
    ABSENT         = "absent"          
    PRESENT_NORMAL = "normal"          
    PRESENT_RAPID  = "present_rapid"   
    PRESENT_SLOW   = "present_slow"    
    PRESENT_ABNORMAL = "present_abnormal"  
class PulseStatus(str, Enum):
    ABSENT        = "absent"
    PRESENT_STRONG = "present_strong"
    PRESENT_WEAK   = "present_weak"
    PRESENT_RAPID  = "present_rapid"
    PRESENT_SLOW   = "present_slow"
    PRESENT_NORMAL = "present_normal"
    PRESENT_IRREGULAR = "present_irregular"
class MentalStatus(str, Enum):
    ALERT        = "alert"         
    VERBAL       = "verbal"        
    PAIN         = "pain"          
    UNRESPONSIVE = "unresponsive"  
    ALTERED      = "altered"       
class TriageProtocol(str, Enum):
    START      = "START"
    JUMPSTART  = "JumpSTART"
    SALT       = "SALT"
    SECONDARY  = "Secondary"
    CLINICAL   = "Clinical"
    INDIAN_EMS = "IndianEMS"
class CbrnContaminationType(str, Enum):
    NONE       = "none"
    CHEMICAL   = "chemical"
    BIOLOGICAL = "biological"
    RADIOLOGICAL = "radiological"
    NUCLEAR    = "nuclear"
    EXPLOSIVE  = "explosive"
@dataclass(frozen=True)
class RPMScore:
    respirations: str     
    pulse: str            
    mental_status: str    
    respiratory_rate: Optional[int] = None    
    heart_rate: Optional[int] = None          
    systolic_bp: Optional[int] = None         
    gcs_score: Optional[int] = None           
    spo2: Optional[float] = None              
    temperature_celsius: Optional[float] = None
    capillary_refill_seconds: Optional[float] = None
    patient_age_years: Optional[int] = None
    is_paediatric: bool = False
    can_walk: bool = False
    bleeding_controlled: bool = True
    airway_opened_by_manoeuvre: bool = False  
    def __post_init__(self) -> None:
        valid_resp = {e.value for e in RespirationStatus}
        valid_pulse = {e.value for e in PulseStatus}
        valid_ms = {e.value for e in MentalStatus}
        if self.respirations not in valid_resp:
            raise ValueError(f"Invalid respirations: {self.respirations}")
        if self.pulse not in valid_pulse:
            raise ValueError(f"Invalid pulse: {self.pulse}")
        if self.mental_status not in valid_ms:
            raise ValueError(f"Invalid mental_status: {self.mental_status}")
        if self.gcs_score is not None and not (3 <= self.gcs_score <= 15):
            raise ValueError(f"GCS score must be 3–15, got {self.gcs_score}")
    @property
    def respiration_enum(self) -> RespirationStatus:
        return RespirationStatus(self.respirations)
    @property
    def pulse_enum(self) -> PulseStatus:
        return PulseStatus(self.pulse)
    @property
    def mental_status_enum(self) -> MentalStatus:
        return MentalStatus(self.mental_status)
    @property
    def respirations_absent(self) -> bool:
        return self.respiration_enum == RespirationStatus.ABSENT
    @property
    def respirations_normal(self) -> bool:
        return self.respiration_enum in (
            RespirationStatus.PRESENT_NORMAL,
        )
    @property
    def pulse_absent(self) -> bool:
        return self.pulse_enum == PulseStatus.ABSENT
    @property
    def pulse_present(self) -> bool:
        return self.pulse_enum != PulseStatus.ABSENT
    @property
    def pulse_weak(self) -> bool:
        return self.pulse_enum == PulseStatus.PRESENT_WEAK
    @property
    def obeys_commands(self) -> bool:
        return self.mental_status_enum in (MentalStatus.ALERT,)
    @property
    def gcs_obeys(self) -> bool:
        if self.gcs_score is None:
            return self.obeys_commands
        return self.gcs_score >= 13
    @property
    def paediatric_age_group(self) -> str:
        if self.patient_age_years is None:
            return "unknown"
        if self.patient_age_years < 1:
            return "infant"
        if self.patient_age_years < 8:
            return "child"
        return "adult"
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RPMScore":
        return cls(
            respirations=d.get("respirations", RespirationStatus.PRESENT_NORMAL.value),
            pulse=d.get("pulse", PulseStatus.PRESENT_STRONG.value),
            mental_status=d.get("mental_status", MentalStatus.ALERT.value),
            respiratory_rate=d.get("respiratory_rate"),
            heart_rate=d.get("heart_rate"),
            systolic_bp=d.get("systolic_bp"),
            gcs_score=d.get("gcs_score"),
            spo2=d.get("spo2"),
            temperature_celsius=d.get("temperature_celsius"),
            capillary_refill_seconds=d.get("capillary_refill_seconds"),
            patient_age_years=d.get("patient_age_years"),
            is_paediatric=d.get("is_paediatric", False),
            can_walk=d.get("can_walk", False),
            bleeding_controlled=d.get("bleeding_controlled", True),
            airway_opened_by_manoeuvre=d.get("airway_opened_by_manoeuvre", False),
        )
    def to_dict(self) -> Dict[str, Any]:
        return {
            "respirations": self.respirations,
            "pulse": self.pulse,
            "mental_status": self.mental_status,
            "respiratory_rate": self.respiratory_rate,
            "heart_rate": self.heart_rate,
            "systolic_bp": self.systolic_bp,
            "gcs_score": self.gcs_score,
            "spo2": self.spo2,
            "temperature_celsius": self.temperature_celsius,
            "capillary_refill_seconds": self.capillary_refill_seconds,
            "patient_age_years": self.patient_age_years,
            "is_paediatric": self.is_paediatric,
            "can_walk": self.can_walk,
            "bleeding_controlled": self.bleeding_controlled,
            "airway_opened_by_manoeuvre": self.airway_opened_by_manoeuvre,
        }
@dataclass
class TriageDecision:
    patient_id: str
    incident_id: str
    victim_index: int
    rpm: RPMScore
    condition_key: str
    patient_age_years: Optional[int]
    assigned_tag: TriageTag
    protocol_used: TriageProtocol
    decision_pathway: List[str] = field(default_factory=list)
    life_saving_interventions: List[str] = field(default_factory=list)
    contraindicated_actions: List[str] = field(default_factory=list)
    special_considerations: List[str] = field(default_factory=list)
    _ground_truth_tag: Optional[TriageTag] = field(default=None, repr=False)
    _ground_truth_confidence: float = field(default=1.0, repr=False)
    time_to_triage_seconds: float = field(default=0.0)
    step_triaged: int = field(default=0)
    deterioration_risk_steps: Optional[int] = field(default=None)
    next_escalation_tag: Optional[TriageTag] = field(default=None)
    cbrn_type: CbrnContaminationType = field(default=CbrnContaminationType.NONE)
    decontamination_required: bool = field(default=False)
    @property
    def is_critical(self) -> bool:
        return self.assigned_tag == TriageTag.IMMEDIATE
    @property
    def is_expectant(self) -> bool:
        return self.assigned_tag == TriageTag.EXPECTANT
    @property
    def ground_truth_tag(self) -> Optional[TriageTag]:
        return self._ground_truth_tag
    def grade_against_ground_truth(self) -> Tuple[float, str]:
        if self._ground_truth_tag is None:
            return (0.50, "no_ground_truth_available")
        gt = self._ground_truth_tag
        assigned = self.assigned_tag
        if assigned == gt:
            return (TRIAGE_EXACT_MATCH_SCORE, f"exact_match:{gt.value}")
        if gt == TriageTag.IMMEDIATE and assigned == TriageTag.EXPECTANT:
            return (
                TRIAGE_CRITICAL_MISMATCH_SCORE,
                f"CRITICAL_MISMATCH:tagged_EXPECTANT_but_ground_truth_IMMEDIATE",
            )
        if gt == TriageTag.IMMEDIATE and assigned in (TriageTag.DELAYED, TriageTag.MINIMAL):
            return (TRIAGE_DISTANT_MISMATCH_SCORE, f"under_triage:{gt.value}->{assigned.value}")
        if gt == TriageTag.MINIMAL and assigned == TriageTag.IMMEDIATE:
            return (TRIAGE_DISTANT_MISMATCH_SCORE, f"over_triage:{gt.value}->{assigned.value}")
        if assigned in gt.adjacent_tags():
            return (TRIAGE_ADJACENT_MISMATCH_SCORE, f"adjacent_mismatch:{gt.value}->{assigned.value}")
        return (TRIAGE_DISTANT_MISMATCH_SCORE, f"distant_mismatch:{gt.value}->{assigned.value}")
    def pathway_summary(self) -> str:
        steps = "\n  ".join(self.decision_pathway)
        lsi = ", ".join(self.life_saving_interventions) or "none"
        return (
            f"Patient {self.patient_id} | {self.assigned_tag.value} ({self.assigned_tag.colour})\n"
            f"  Protocol: {self.protocol_used.value}\n"
            f"  Pathway:\n  {steps}\n"
            f"  LSI: {lsi}\n"
            f"  Deterioration risk: {self.deterioration_risk_steps} steps\n"
        )
    def to_agent_observation(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "incident_id": self.incident_id,
            "victim_index": self.victim_index,
            "assigned_tag": self.assigned_tag.value,
            "tag_colour": self.assigned_tag.colour,
            "indian_ems_priority": self.assigned_tag.indian_ems_priority,
            "protocol_used": self.protocol_used.value,
            "life_saving_interventions": self.life_saving_interventions,
            "contraindicated_actions": self.contraindicated_actions,
            "special_considerations": self.special_considerations,
            "deterioration_risk_steps": self.deterioration_risk_steps,
            "next_escalation_tag": (
                self.next_escalation_tag.value
                if self.next_escalation_tag else None
            ),
            "decontamination_required": self.decontamination_required,
            "cbrn_type": self.cbrn_type.value,
        }
class STARTProtocolEngine:
    RR_MIN: int = START_RR_MIN
    RR_MAX: int = START_RR_MAX
    def assess(
        self,
        rpm: RPMScore,
        patient_id: str = "unknown",
        incident_id: str = "unknown",
        victim_index: int = 0,
        condition_key: str = "unknown",
        ground_truth_tag: Optional[TriageTag] = None,
    ) -> TriageDecision:
        pathway: List[str] = []
        lsi: List[str] = []
        special: List[str] = []
        contraindicated: List[str] = []
        deterioration_steps: Optional[int] = None
        next_tag: Optional[TriageTag] = None
        pathway.append("Step 1: Assess ambulation")
        if rpm.can_walk and rpm.mental_status_enum == MentalStatus.ALERT:
            pathway.append("  → Patient can walk → MINIMAL (Green)")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key,
                patient_age_years=rpm.patient_age_years,
                assigned_tag=TriageTag.MINIMAL,
                protocol_used=TriageProtocol.START,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["minimal_to_delayed"],
                next_escalation_tag=TriageTag.DELAYED,
            )
        pathway.append("  → Patient cannot walk → proceed to respiration assessment")
        pathway.append("Step 2: Assess respirations")
        if rpm.respirations_absent:
            pathway.append("  → Respirations ABSENT — open airway (jaw thrust / head tilt)")
            lsi.append("airway_opening_manoeuvre")
            if rpm.airway_opened_by_manoeuvre:
                pathway.append("  → Respirations PRESENT after airway manoeuvre → IMMEDIATE (Red)")
                lsi.append("position_airway_open")
                return TriageDecision(
                    patient_id=patient_id, incident_id=incident_id,
                    victim_index=victim_index, rpm=rpm,
                    condition_key=condition_key,
                    patient_age_years=rpm.patient_age_years,
                    assigned_tag=TriageTag.IMMEDIATE,
                    protocol_used=TriageProtocol.START,
                    decision_pathway=pathway, life_saving_interventions=lsi,
                    contraindicated_actions=contraindicated,
                    special_considerations=special,
                    _ground_truth_tag=ground_truth_tag,
                    deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                    next_escalation_tag=TriageTag.EXPECTANT,
                )
            else:
                pathway.append("  → Still NOT breathing after airway manoeuvre → EXPECTANT (Black)")
                return TriageDecision(
                    patient_id=patient_id, incident_id=incident_id,
                    victim_index=victim_index, rpm=rpm,
                    condition_key=condition_key,
                    patient_age_years=rpm.patient_age_years,
                    assigned_tag=TriageTag.EXPECTANT,
                    protocol_used=TriageProtocol.START,
                    decision_pathway=pathway, life_saving_interventions=lsi,
                    contraindicated_actions=contraindicated,
                    special_considerations=special,
                    _ground_truth_tag=ground_truth_tag,
                )
        pathway.append("  → Respirations PRESENT")
        rr = rpm.respiratory_rate
        resp_status = rpm.respiration_enum
        rr_abnormal = (
            resp_status in (
                RespirationStatus.PRESENT_RAPID,
                RespirationStatus.PRESENT_SLOW,
                RespirationStatus.PRESENT_ABNORMAL,
            )
            or (rr is not None and (rr < self.RR_MIN or rr > self.RR_MAX))
        )
        if rr_abnormal:
            rate_str = f"{rr} bpm" if rr is not None else resp_status.value
            pathway.append(f"  → RR {rate_str} — outside 10-29 range → IMMEDIATE (Red)")
            special.append("respiratory_compromise_primary")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key,
                patient_age_years=rpm.patient_age_years,
                assigned_tag=TriageTag.IMMEDIATE,
                protocol_used=TriageProtocol.START,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                next_escalation_tag=TriageTag.EXPECTANT,
            )
        pathway.append(f"  → RR within normal range — proceed to perfusion assessment")
        pathway.append("Step 3: Assess radial pulse")
        if rpm.pulse_absent or rpm.pulse_weak:
            status = "ABSENT" if rpm.pulse_absent else "WEAK"
            pathway.append(f"  → Radial pulse {status} (perfusion inadequate) → IMMEDIATE (Red)")
            lsi.append("haemorrhage_control")
            lsi.append("circulation_support")
            special.append("haemorrhagic_shock_suspected")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key,
                patient_age_years=rpm.patient_age_years,
                assigned_tag=TriageTag.IMMEDIATE,
                protocol_used=TriageProtocol.START,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                next_escalation_tag=TriageTag.EXPECTANT,
            )
        pathway.append("  → Radial pulse PRESENT — proceed to mental status")
        pathway.append("Step 4: Assess mental status (obeys simple commands)")
        ms = rpm.mental_status_enum
        obeys = ms == MentalStatus.ALERT or (
            rpm.gcs_score is not None and rpm.gcs_score >= 13
        )
        if not obeys:
            gcs_str = f" (GCS {rpm.gcs_score})" if rpm.gcs_score is not None else ""
            pathway.append(
                f"  → Mental status {ms.value}{gcs_str} — does NOT obey commands → IMMEDIATE (Red)"
            )
            special.append("altered_consciousness_neurological_compromise")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key,
                patient_age_years=rpm.patient_age_years,
                assigned_tag=TriageTag.IMMEDIATE,
                protocol_used=TriageProtocol.START,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                next_escalation_tag=TriageTag.EXPECTANT,
            )
        pathway.append("  → Obeys commands — DELAYED (Yellow)")
        return TriageDecision(
            patient_id=patient_id, incident_id=incident_id,
            victim_index=victim_index, rpm=rpm,
            condition_key=condition_key,
            patient_age_years=rpm.patient_age_years,
            assigned_tag=TriageTag.DELAYED,
            protocol_used=TriageProtocol.START,
            decision_pathway=pathway, life_saving_interventions=lsi,
            contraindicated_actions=contraindicated,
            special_considerations=special,
            _ground_truth_tag=ground_truth_tag,
            deterioration_risk_steps=DETERIORATION_STEPS["delayed_to_immediate"],
            next_escalation_tag=TriageTag.IMMEDIATE,
        )
class JumpSTARTProtocolEngine:
    RR_MIN: int = JUMPSTART_RR_MIN
    RR_MAX: int = JUMPSTART_RR_MAX
    def assess(
        self,
        rpm: RPMScore,
        patient_id: str = "unknown",
        incident_id: str = "unknown",
        victim_index: int = 0,
        condition_key: str = "unknown",
        ground_truth_tag: Optional[TriageTag] = None,
    ) -> TriageDecision:
        pathway: List[str] = []
        lsi: List[str] = []
        special: List[str] = []
        contraindicated: List[str] = []
        age = rpm.patient_age_years
        age_group = rpm.paediatric_age_group
        pathway.append(f"JumpSTART assessment | age_group={age_group} | age={age}")
        pathway.append("Step 1: Assess ambulation")
        if rpm.can_walk and rpm.mental_status_enum == MentalStatus.ALERT:
            pathway.append("  → Child can walk → MINIMAL (Green)")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key,
                patient_age_years=age,
                assigned_tag=TriageTag.MINIMAL,
                protocol_used=TriageProtocol.JUMPSTART,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["minimal_to_delayed"],
                next_escalation_tag=TriageTag.DELAYED,
            )
        pathway.append("Step 2: Assess breathing")
        if rpm.respirations_absent:
            pathway.append("  → Respirations ABSENT — perform airway manoeuvre")
            lsi.append("paediatric_airway_manoeuvre")
            if rpm.airway_opened_by_manoeuvre:
                pathway.append("  → Breathing after manoeuvre → IMMEDIATE (Red)")
                lsi.append("maintain_open_airway")
                return TriageDecision(
                    patient_id=patient_id, incident_id=incident_id,
                    victim_index=victim_index, rpm=rpm,
                    condition_key=condition_key, patient_age_years=age,
                    assigned_tag=TriageTag.IMMEDIATE,
                    protocol_used=TriageProtocol.JUMPSTART,
                    decision_pathway=pathway, life_saving_interventions=lsi,
                    contraindicated_actions=contraindicated,
                    special_considerations=special,
                    _ground_truth_tag=ground_truth_tag,
                    deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                    next_escalation_tag=TriageTag.EXPECTANT,
                )
            pathway.append("  → Still not breathing after manoeuvre — check pulse")
            if rpm.pulse_absent:
                pathway.append("  → No pulse AND no breathing → EXPECTANT (Black)")
                return TriageDecision(
                    patient_id=patient_id, incident_id=incident_id,
                    victim_index=victim_index, rpm=rpm,
                    condition_key=condition_key, patient_age_years=age,
                    assigned_tag=TriageTag.EXPECTANT,
                    protocol_used=TriageProtocol.JUMPSTART,
                    decision_pathway=pathway, life_saving_interventions=lsi,
                    contraindicated_actions=contraindicated,
                    special_considerations=special,
                    _ground_truth_tag=ground_truth_tag,
                )
            pathway.append("  → Pulse PRESENT but not breathing → give 5 rescue breaths")
            lsi.append("5_rescue_breaths_jumpstart")
            special.append("JUMPSTART_RESCUE_BREATH_STEP — adult START does not include this")
            responds_to_rescue = self._rescue_breath_response(condition_key)
            if responds_to_rescue:
                pathway.append("  → Breathing resumed after rescue breaths → IMMEDIATE (Red)")
                lsi.append("continue_ventilation_support")
                return TriageDecision(
                    patient_id=patient_id, incident_id=incident_id,
                    victim_index=victim_index, rpm=rpm,
                    condition_key=condition_key, patient_age_years=age,
                    assigned_tag=TriageTag.IMMEDIATE,
                    protocol_used=TriageProtocol.JUMPSTART,
                    decision_pathway=pathway, life_saving_interventions=lsi,
                    contraindicated_actions=contraindicated,
                    special_considerations=special,
                    _ground_truth_tag=ground_truth_tag,
                    deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                    next_escalation_tag=TriageTag.EXPECTANT,
                )
            else:
                pathway.append("  → No response to rescue breaths → EXPECTANT (Black)")
                return TriageDecision(
                    patient_id=patient_id, incident_id=incident_id,
                    victim_index=victim_index, rpm=rpm,
                    condition_key=condition_key, patient_age_years=age,
                    assigned_tag=TriageTag.EXPECTANT,
                    protocol_used=TriageProtocol.JUMPSTART,
                    decision_pathway=pathway, life_saving_interventions=lsi,
                    contraindicated_actions=contraindicated,
                    special_considerations=special,
                    _ground_truth_tag=ground_truth_tag,
                )
        pathway.append("  → Breathing PRESENT — assess rate")
        rr = rpm.respiratory_rate
        resp_status = rpm.respiration_enum
        rr_abnormal = (
            resp_status in (
                RespirationStatus.PRESENT_RAPID,
                RespirationStatus.PRESENT_SLOW,
                RespirationStatus.PRESENT_ABNORMAL,
            )
            or (rr is not None and (rr < self.RR_MIN or rr > self.RR_MAX))
        )
        if rr_abnormal:
            rate_str = f"{rr} bpm" if rr is not None else resp_status.value
            pathway.append(
                f"  → Paediatric RR {rate_str} outside 15-45 range → IMMEDIATE (Red)"
            )
            special.append("paediatric_respiratory_compromise")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key, patient_age_years=age,
                assigned_tag=TriageTag.IMMEDIATE,
                protocol_used=TriageProtocol.JUMPSTART,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                next_escalation_tag=TriageTag.EXPECTANT,
            )
        pathway.append("  → Paediatric RR normal — assess perfusion")
        pathway.append("Step 3: Assess peripheral pulse")
        pulse_site = "brachial" if age_group == "infant" else "radial"
        pathway.append(f"  Using {pulse_site} pulse site for age group {age_group}")
        if rpm.pulse_absent or rpm.pulse_weak:
            status = "ABSENT" if rpm.pulse_absent else "WEAK"
            pathway.append(f"  → {pulse_site.capitalize()} pulse {status} → IMMEDIATE (Red)")
            lsi.append("paediatric_haemorrhage_control")
            lsi.append("fluid_bolus_io_access")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key, patient_age_years=age,
                assigned_tag=TriageTag.IMMEDIATE,
                protocol_used=TriageProtocol.JUMPSTART,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                next_escalation_tag=TriageTag.EXPECTANT,
            )
        pathway.append(f"  → {pulse_site.capitalize()} pulse present — assess AVPU")
        pathway.append("Step 4: Assess mental status (AVPU scale)")
        ms = rpm.mental_status_enum
        avpu_ok = ms in (MentalStatus.ALERT, MentalStatus.VERBAL)
        if not avpu_ok:
            pathway.append(
                f"  → AVPU={ms.value} (P or U) — altered consciousness → IMMEDIATE (Red)"
            )
            special.append("paediatric_neurological_compromise_avpu_p_or_u")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key, patient_age_years=age,
                assigned_tag=TriageTag.IMMEDIATE,
                protocol_used=TriageProtocol.JUMPSTART,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                deterioration_risk_steps=DETERIORATION_STEPS["immediate_to_expectant"],
                next_escalation_tag=TriageTag.EXPECTANT,
            )
        pathway.append("  → AVPU = A or V — DELAYED (Yellow)")
        return TriageDecision(
            patient_id=patient_id, incident_id=incident_id,
            victim_index=victim_index, rpm=rpm,
            condition_key=condition_key, patient_age_years=age,
            assigned_tag=TriageTag.DELAYED,
            protocol_used=TriageProtocol.JUMPSTART,
            decision_pathway=pathway, life_saving_interventions=lsi,
            contraindicated_actions=contraindicated,
            special_considerations=special,
            _ground_truth_tag=ground_truth_tag,
            deterioration_risk_steps=DETERIORATION_STEPS["delayed_to_immediate"],
            next_escalation_tag=TriageTag.IMMEDIATE,
        )
    @staticmethod
    def _rescue_breath_response(condition_key: str) -> bool:
        responds = {
            "drowning",
            "near_drowning",
            "paediatric_choking_resolved",
            "febrile_seizure_postictal",
            "hypoglycaemia",
            "syncope",
            "vagal_episode",
        }
        no_response = {
            "cardiac_arrest_vf",
            "cardiac_arrest_pea",
            "severe_tbi",
            "polytrauma_blunt",
            "neonatal_emergency",
            "ductus_dependent_chd",
        }
        if any(k in condition_key for k in no_response):
            return False
        if any(k in condition_key for k in responds):
            return True
        return True
class SALTProtocolEngine:
    PERMITTED_LSI: FrozenSet[str] = frozenset([
        "control_major_haemorrhage_tourniquet",
        "open_airway_jaw_thrust",
        "needle_decompression_tension_ptx",
        "autoinjector_antidote_chemical",
        "position_recovery",
        "seal_open_chest_wound",
    ])
    def sort_phase(
        self,
        victims: List[RPMScore],
    ) -> Dict[str, List[int]]:
        tiers: Dict[str, List[int]] = {"walk": [], "move": [], "still": []}
        for i, rpm in enumerate(victims):
            if rpm.can_walk and rpm.mental_status_enum == MentalStatus.ALERT:
                tiers["walk"].append(i)
            elif rpm.mental_status_enum not in (MentalStatus.UNRESPONSIVE,):
                tiers["move"].append(i)
            else:
                tiers["still"].append(i)
        return tiers
    def assess_phase(
        self,
        rpm: RPMScore,
        patient_id: str,
        incident_id: str,
        victim_index: int,
        condition_key: str,
        ground_truth_tag: Optional[TriageTag] = None,
        cbrn_type: CbrnContaminationType = CbrnContaminationType.NONE,
    ) -> TriageDecision:
        pathway: List[str] = ["SALT Phase 2: Individual LSI + Assessment"]
        lsi: List[str] = []
        special: List[str] = []
        contraindicated: List[str] = []
        if not rpm.bleeding_controlled:
            pathway.append("  LSI: Control major haemorrhage")
            lsi.append("control_major_haemorrhage_tourniquet")
        if rpm.respirations_absent:
            pathway.append("  LSI: Open airway")
            lsi.append("open_airway_jaw_thrust")
        if cbrn_type == CbrnContaminationType.CHEMICAL:
            pathway.append("  LSI: CBRN autoinjector antidote")
            lsi.append("autoinjector_antidote_chemical")
            special.append("CBRN_CHEMICAL_decontamination_required_before_transport")
        pathway.append("  Assessment: Breathing?")
        ms = rpm.mental_status_enum
        if ms == MentalStatus.UNRESPONSIVE and rpm.respirations_absent and rpm.pulse_absent:
            pathway.append("  → Apnoeic, pulseless, unresponsive → EXPECTANT (Black)")
            return TriageDecision(
                patient_id=patient_id, incident_id=incident_id,
                victim_index=victim_index, rpm=rpm,
                condition_key=condition_key,
                patient_age_years=rpm.patient_age_years,
                assigned_tag=TriageTag.EXPECTANT,
                protocol_used=TriageProtocol.SALT,
                decision_pathway=pathway, life_saving_interventions=lsi,
                contraindicated_actions=contraindicated,
                special_considerations=special,
                _ground_truth_tag=ground_truth_tag,
                cbrn_type=cbrn_type,
                decontamination_required=(cbrn_type != CbrnContaminationType.NONE),
            )
        if not rpm.respirations_absent:
            pathway.append("  → Breathing present — assess obeys commands / pulse")
            obeys = (
                ms == MentalStatus.ALERT
                or (rpm.gcs_score is not None and rpm.gcs_score >= 13)
            )
            pulse_ok = rpm.pulse_present and not rpm.pulse_weak
            if obeys and pulse_ok:
                pathway.append("  → Obeys + pulse OK → DELAYED (Yellow)")
                tag = TriageTag.DELAYED
            else:
                pathway.append("  → Does not obey or perfusion inadequate → IMMEDIATE (Red)")
                tag = TriageTag.IMMEDIATE
        else:
            pathway.append("  → Breathing absent after LSI → IMMEDIATE (Red)")
            lsi.append("continuous_airway_management")
            tag = TriageTag.IMMEDIATE
        decon = cbrn_type != CbrnContaminationType.NONE
        return TriageDecision(
            patient_id=patient_id, incident_id=incident_id,
            victim_index=victim_index, rpm=rpm,
            condition_key=condition_key,
            patient_age_years=rpm.patient_age_years,
            assigned_tag=tag,
            protocol_used=TriageProtocol.SALT,
            decision_pathway=pathway, life_saving_interventions=lsi,
            contraindicated_actions=contraindicated,
            special_considerations=special,
            _ground_truth_tag=ground_truth_tag,
            cbrn_type=cbrn_type,
            decontamination_required=decon,
            deterioration_risk_steps=(
                DETERIORATION_STEPS["immediate_to_expectant"]
                if tag == TriageTag.IMMEDIATE
                else DETERIORATION_STEPS["delayed_to_immediate"]
            ),
            next_escalation_tag=(
                TriageTag.EXPECTANT if tag == TriageTag.IMMEDIATE else TriageTag.IMMEDIATE
            ),
        )
class PatientDeteriorationTracker:
    def __init__(self) -> None:
        self._patients: Dict[str, Dict[str, Any]] = {}
    def register(
        self,
        patient_id: str,
        initial_tag: TriageTag,
        condition_key: str,
        deterioration_steps: Optional[int],
        next_tag: Optional[TriageTag],
    ) -> None:
        if deterioration_steps is None or next_tag is None:
            return
        self._patients[patient_id] = {
            "current_tag": initial_tag,
            "steps_remaining": deterioration_steps,
            "next_tag": next_tag,
            "condition_key": condition_key,
            "treatment_received": False,
            "history": [(initial_tag.value, 0)],
        }
    def mark_treated(self, patient_id: str) -> None:
        if patient_id in self._patients:
            self._patients[patient_id]["treatment_received"] = True
    def tick(self, step: int) -> List[Tuple[str, TriageTag, TriageTag]]:
        escalations: List[Tuple[str, TriageTag, TriageTag]] = []
        for pid, state in self._patients.items():
            if state["treatment_received"]:
                continue
            if state["steps_remaining"] <= 0:
                continue
            state["steps_remaining"] -= 1
            if state["steps_remaining"] == 0:
                old = state["current_tag"]
                new = state["next_tag"]
                state["current_tag"] = new
                state["history"].append((new.value, step))
                next_degrade = self._next_deterioration(new)
                if next_degrade is not None:
                    nxt_tag, nxt_steps = next_degrade
                    state["next_tag"] = nxt_tag
                    state["steps_remaining"] = nxt_steps
                else:
                    state["next_tag"] = None
                    state["steps_remaining"] = 0
                escalations.append((pid, old, new))
                logger.warning(
                    "PatientDeteriorationTracker: patient %s escalated %s→%s at step %d",
                    pid, old.value, new.value, step,
                )
        return escalations
    @staticmethod
    def _next_deterioration(
        current_tag: TriageTag,
    ) -> Optional[Tuple[TriageTag, int]]:
        chain = {
            TriageTag.MINIMAL:   (TriageTag.DELAYED,   DETERIORATION_STEPS["minimal_to_delayed"]),
            TriageTag.DELAYED:   (TriageTag.IMMEDIATE,  DETERIORATION_STEPS["delayed_to_immediate"]),
            TriageTag.IMMEDIATE: (TriageTag.EXPECTANT,  DETERIORATION_STEPS["immediate_to_expectant"]),
        }
        return chain.get(current_tag)
    def current_tag(self, patient_id: str) -> Optional[TriageTag]:
        if patient_id not in self._patients:
            return None
        return self._patients[patient_id]["current_tag"]
    def steps_until_escalation(self, patient_id: str) -> Optional[int]:
        if patient_id not in self._patients:
            return None
        return self._patients[patient_id]["steps_remaining"]
    def all_patient_tags(self) -> Dict[str, TriageTag]:
        return {
            pid: state["current_tag"]
            for pid, state in self._patients.items()
        }
    def critical_patients(self) -> List[str]:
        return [
            pid for pid, state in self._patients.items()
            if state["current_tag"] == TriageTag.IMMEDIATE
            and not state["treatment_received"]
        ]
    def patients_near_escalation(self, window_steps: int = 2) -> List[Tuple[str, int]]:
        result = []
        for pid, state in self._patients.items():
            if not state["treatment_received"]:
                rem = state["steps_remaining"]
                if 0 < rem <= window_steps:
                    result.append((pid, rem))
        return sorted(result, key=lambda x: x[1])
class TriageGroundTruthDatabase:
    _GROUND_TRUTH: Dict[str, TriageTag] = {
        "stemi_anterior":            TriageTag.IMMEDIATE,
        "stemi_inferior":            TriageTag.IMMEDIATE,
        "stemi_posterior":           TriageTag.IMMEDIATE,
        "stemi_with_vf_arrest":      TriageTag.IMMEDIATE,
        "stemi_cocaine":             TriageTag.IMMEDIATE,
        "stemi_post_cabg":           TriageTag.IMMEDIATE,
        "ischemic_stroke":           TriageTag.IMMEDIATE,
        "ischemic_stroke_wake_up":   TriageTag.IMMEDIATE,
        "hemorrhagic_stroke_sah":    TriageTag.IMMEDIATE,
        "meningitis_cryptococcal":   TriageTag.IMMEDIATE,
        "paediatric_stroke":         TriageTag.IMMEDIATE,
        "cardiac_arrest_vf":         TriageTag.IMMEDIATE,
        "cardiac_arrest_pea":        TriageTag.IMMEDIATE,
        "neonatal_resuscitation":    TriageTag.IMMEDIATE,
        "expected_death":            TriageTag.EXPECTANT,
        "lightning_strike":          TriageTag.IMMEDIATE,
        "wpw_svt":                   TriageTag.IMMEDIATE,
        "brugada_event":             TriageTag.DELAYED,
        "fast_af":                   TriageTag.DELAYED,
        "pacemaker_failure":         TriageTag.DELAYED,
        "complete_heart_block":      TriageTag.IMMEDIATE,
        "hypocalcaemia":             TriageTag.DELAYED,
        "polytrauma_blunt":          TriageTag.IMMEDIATE,
        "polytrauma_penetrating":    TriageTag.IMMEDIATE,
        "penetrating_chest":         TriageTag.IMMEDIATE,
        "severe_tbi":                TriageTag.IMMEDIATE,
        "cervical_spinal_injury":    TriageTag.IMMEDIATE,
        "thoracic_spinal_injury":    TriageTag.IMMEDIATE,
        "skull_base_fracture":       TriageTag.IMMEDIATE,
        "subdural_haematoma":        TriageTag.IMMEDIATE,
        "mci_rta":                   TriageTag.IMMEDIATE,
        "blast_injury":              TriageTag.IMMEDIATE,
        "mine_collapse":             TriageTag.IMMEDIATE,
        "femur_fracture":            TriageTag.DELAYED,
        "chest_trauma":              TriageTag.IMMEDIATE,
        "splenic_laceration":        TriageTag.IMMEDIATE,
        "traumatic_amputation":      TriageTag.IMMEDIATE,
        "degloving_crush":           TriageTag.IMMEDIATE,
        "open_fracture":             TriageTag.DELAYED,
        "crush_syndrome":            TriageTag.IMMEDIATE,
        "facial_trauma":             TriageTag.DELAYED,
        "brachial_plexus":           TriageTag.DELAYED,
        "gsw_shoulder":              TriageTag.DELAYED,
        "ovarian_torsion":           TriageTag.IMMEDIATE,
        "intussusception":           TriageTag.IMMEDIATE,
        "minor_trauma":              TriageTag.MINIMAL,
        "minor_head_trauma":         TriageTag.DELAYED,
        "nof_fracture":              TriageTag.DELAYED,
        "epistaxis":                 TriageTag.DELAYED,
        "envenomation_minor":        TriageTag.MINIMAL,
        "acl_rupture":               TriageTag.MINIMAL,
        "colles_fracture":           TriageTag.MINIMAL,
        "burns_major":               TriageTag.IMMEDIATE,
        "burns_moderate":            TriageTag.IMMEDIATE,
        "burns_electrical":          TriageTag.DELAYED,
        "chemical_burns":            TriageTag.IMMEDIATE,
        "respiratory_failure":       TriageTag.IMMEDIATE,
        "ards":                      TriageTag.IMMEDIATE,
        "paediatric_respiratory":    TriageTag.IMMEDIATE,
        "pulmonary_embolism":        TriageTag.IMMEDIATE,
        "massive_haemoptysis":       TriageTag.IMMEDIATE,
        "acute_pulmonary_oedema":    TriageTag.IMMEDIATE,
        "decompression_sickness":    TriageTag.IMMEDIATE,
        "massive_pleural_effusion":  TriageTag.IMMEDIATE,
        "obstetric_hemorrhage":      TriageTag.IMMEDIATE,
        "eclampsia":                 TriageTag.IMMEDIATE,
        "uterine_rupture":           TriageTag.IMMEDIATE,
        "obstetric_trauma":          TriageTag.DELAYED,
        "normal_delivery":           TriageTag.DELAYED,
        "precipitate_delivery":      TriageTag.IMMEDIATE,
        "septic_shock":              TriageTag.IMMEDIATE,
        "sepsis_surgical":           TriageTag.IMMEDIATE,
        "necrotising_fasciitis":     TriageTag.IMMEDIATE,
        "neutropenic_sepsis":        TriageTag.IMMEDIATE,
        "meningococcal_sepsis":      TriageTag.IMMEDIATE,
        "weil_disease":              TriageTag.IMMEDIATE,
        "transplant_sepsis":         TriageTag.IMMEDIATE,
        "urosepsis":                 TriageTag.IMMEDIATE,
        "sbp_peritonitis":           TriageTag.IMMEDIATE,
        "enteric_fever":             TriageTag.IMMEDIATE,
        "sepsis_malnutrition":       TriageTag.IMMEDIATE,
        "anaphylaxis":               TriageTag.IMMEDIATE,
        "submersion":                TriageTag.IMMEDIATE,
        "near_drowning":             TriageTag.DELAYED,
        "organophosphate_poisoning": TriageTag.IMMEDIATE,
        "opioid_overdose":           TriageTag.IMMEDIATE,
        "toxic_inhalation":          TriageTag.IMMEDIATE,
        "carbon_monoxide":           TriageTag.IMMEDIATE,
        "cbrn_suspected":            TriageTag.IMMEDIATE,
        "paraquat_exposure":         TriageTag.DELAYED,
        "phosphine_poisoning":       TriageTag.IMMEDIATE,
        "mdma_toxicity":             TriageTag.IMMEDIATE,
        "botulism_outbreak":         TriageTag.IMMEDIATE,
        "paediatric_poisoning":      TriageTag.IMMEDIATE,
        "alcohol_withdrawal_seizure": TriageTag.IMMEDIATE,
        "hydrocarbon_ingestion":     TriageTag.IMMEDIATE,
        "paediatric_choking":        TriageTag.IMMEDIATE,
        "paediatric_ingestion":      TriageTag.IMMEDIATE,
        "paediatric_head_trauma":    TriageTag.IMMEDIATE,
        "paediatric_joint":          TriageTag.DELAYED,
        "status_epilepticus":        TriageTag.IMMEDIATE,
        "febrile_seizure":           TriageTag.IMMEDIATE,
        "neonatal_emergency":        TriageTag.IMMEDIATE,
        "dka":                       TriageTag.IMMEDIATE,
        "dengue_severe":             TriageTag.IMMEDIATE,
        "sickle_cell_crisis":        TriageTag.IMMEDIATE,
        "non_accidental_injury":     TriageTag.IMMEDIATE,
        "meningococcal_paediatric":  TriageTag.IMMEDIATE,
        "heat_stroke":               TriageTag.IMMEDIATE,
        "heat_exhaustion":           TriageTag.DELAYED,
        "hypoglycaemia":             TriageTag.IMMEDIATE,
        "hyperkalaemia":             TriageTag.IMMEDIATE,
        "rhabdomyolysis":            TriageTag.IMMEDIATE,
        "psychiatric_crisis":        TriageTag.DELAYED,
        "self_harm":                 TriageTag.DELAYED,
        "renal_colic":               TriageTag.DELAYED,
        "aortic_dissection":         TriageTag.IMMEDIATE,
        "aaa_rupture":               TriageTag.IMMEDIATE,
        "pancreatitis_severe":       TriageTag.IMMEDIATE,
        "perforated_viscus":         TriageTag.IMMEDIATE,
        "appendicitis":              TriageTag.DELAYED,
        "bowel_obstruction":         TriageTag.IMMEDIATE,
        "severe_malaria":            TriageTag.IMMEDIATE,
        "pneumocystis_pneumonia":    TriageTag.IMMEDIATE,
        "minor_allergy":             TriageTag.MINIMAL,
        "snakebite_hemotoxic":       TriageTag.IMMEDIATE,
        "snakebite_neurotoxic":      TriageTag.IMMEDIATE,
        "psychiatric_wandering":     TriageTag.DELAYED,
        "epistaxis_anticoagulated":  TriageTag.DELAYED,
        "infection_soft_tissue":     TriageTag.DELAYED,
        "mci_natural_disaster":      TriageTag.IMMEDIATE,
        "mine_rescue_standby":       TriageTag.DELAYED,
    }
    _PARTIAL_CREDIT_CONDITIONS: FrozenSet[str] = frozenset([
        "near_drowning",
        "minor_head_trauma",
        "paediatric_joint",
        "epistaxis",
        "burns_electrical",
        "obstetric_trauma",
        "brugada_event",
        "brachial_plexus",
        "open_fracture",
    ])
    @classmethod
    def get_ground_truth(
        cls,
        condition_key: str,
    ) -> Tuple[Optional[TriageTag], float]:
        tag = cls._GROUND_TRUTH.get(condition_key)
        if tag is not None:
            confidence = 0.85 if condition_key in cls._PARTIAL_CREDIT_CONDITIONS else 1.0
            return (tag, confidence)
        for key, val in cls._GROUND_TRUTH.items():
            if condition_key.startswith(key) or key.startswith(condition_key):
                return (val, 0.75)
        logger.warning(
            "TriageGroundTruthDatabase: unknown condition_key '%s' — returning None",
            condition_key,
        )
        return (None, 0.0)
    @classmethod
    def all_condition_keys(cls) -> List[str]:
        return sorted(cls._GROUND_TRUTH.keys())
    @classmethod
    def conditions_by_tag(cls, tag: TriageTag) -> List[str]:
        return [k for k, v in cls._GROUND_TRUTH.items() if v == tag]
    @classmethod
    def is_critical_mismatch(
        cls,
        assigned: TriageTag,
        condition_key: str,
    ) -> bool:
        gt, _ = cls.get_ground_truth(condition_key)
        return (
            gt == TriageTag.IMMEDIATE
            and assigned == TriageTag.EXPECTANT
        )
@dataclass(frozen=True)
class RevisedTraumaScore:
    gcs: int                           
    systolic_bp: int                   
    respiratory_rate: int              
    @property
    def gcs_coded(self) -> int:
        if self.gcs >= 13:  return 4
        if self.gcs >= 9:   return 3
        if self.gcs >= 6:   return 2
        if self.gcs >= 4:   return 1
        return 0
    @property
    def sbp_coded(self) -> int:
        if self.systolic_bp > 89:  return 4
        if self.systolic_bp >= 76: return 3
        if self.systolic_bp >= 50: return 2
        if self.systolic_bp >= 1:  return 1
        return 0
    @property
    def rr_coded(self) -> int:
        if 10 <= self.respiratory_rate <= 29: return 4
        if self.respiratory_rate > 29:        return 3
        if 6 <= self.respiratory_rate <= 9:   return 2
        if 1 <= self.respiratory_rate <= 5:   return 1
        return 0
    @property
    def rts(self) -> float:
        return (
            0.9368 * self.gcs_coded
            + 0.7326 * self.sbp_coded
            + 0.2908 * self.rr_coded
        )
    @property
    def survival_probability(self) -> float:
        b0, b1 = -3.5718, 0.8079
        exp = math.exp(b0 + b1 * self.rts)
        ps = 1.0 / (1.0 + exp)
        return max(0.0, min(1.0, ps))
    @property
    def triage_category(self) -> TriageTag:
        r = self.rts
        if r >= 7.0:   return TriageTag.MINIMAL
        if r >= 5.0:   return TriageTag.DELAYED
        if r >= 1.0:   return TriageTag.IMMEDIATE
        return TriageTag.EXPECTANT
    @property
    def severity_label(self) -> str:
        ps = self.survival_probability
        if ps >= 0.90: return "MINOR"
        if ps >= 0.75: return "MODERATE"
        if ps >= 0.50: return "SERIOUS"
        if ps >= 0.25: return "CRITICAL"
        return "UNSURVIVABLE"
    @classmethod
    def from_rpm(cls, rpm: RPMScore) -> "RevisedTraumaScore":
        gcs = rpm.gcs_score if rpm.gcs_score is not None else _infer_gcs_from_rpm(rpm)
        sbp = rpm.systolic_bp if rpm.systolic_bp is not None else _infer_sbp_from_rpm(rpm)
        rr = rpm.respiratory_rate if rpm.respiratory_rate is not None else _infer_rr_from_rpm(rpm)
        return cls(gcs=gcs, systolic_bp=sbp, respiratory_rate=rr)
def _infer_gcs_from_rpm(rpm: RPMScore) -> int:
    ms = rpm.mental_status_enum
    mapping = {
        MentalStatus.ALERT:        15,
        MentalStatus.VERBAL:       12,
        MentalStatus.ALTERED:      10,
        MentalStatus.PAIN:          8,
        MentalStatus.UNRESPONSIVE:  3,
    }
    return mapping.get(ms, 15)
def _infer_sbp_from_rpm(rpm: RPMScore) -> int:
    pulse = rpm.pulse_enum
    mapping = {
        PulseStatus.PRESENT_STRONG:    110,
        PulseStatus.PRESENT_NORMAL:    100,
        PulseStatus.PRESENT_RAPID:      90,
        PulseStatus.PRESENT_WEAK:       70,
        PulseStatus.PRESENT_SLOW:      100,
        PulseStatus.PRESENT_IRREGULAR:  90,
        PulseStatus.ABSENT:              0,
    }
    return mapping.get(pulse, 100)
def _infer_rr_from_rpm(rpm: RPMScore) -> int:
    resp = rpm.respiration_enum
    mapping = {
        RespirationStatus.ABSENT:           0,
        RespirationStatus.PRESENT_NORMAL:  16,
        RespirationStatus.PRESENT_RAPID:   32,
        RespirationStatus.PRESENT_SLOW:     6,
        RespirationStatus.PRESENT_ABNORMAL: 8,
    }
    return mapping.get(resp, 16)
class NACAScore(int, Enum):
    ZERO  = 0    
    ONE   = 1    
    TWO   = 2    
    THREE = 3    
    FOUR  = 4    
    FIVE  = 5    
    SIX   = 6    
    SEVEN = 7    
    @property
    def triage_tag(self) -> TriageTag:
        mapping = {
            NACAScore.ZERO:  TriageTag.MINIMAL,
            NACAScore.ONE:   TriageTag.MINIMAL,
            NACAScore.TWO:   TriageTag.MINIMAL,
            NACAScore.THREE: TriageTag.DELAYED,
            NACAScore.FOUR:  TriageTag.DELAYED,
            NACAScore.FIVE:  TriageTag.IMMEDIATE,
            NACAScore.SIX:   TriageTag.IMMEDIATE,
            NACAScore.SEVEN: TriageTag.EXPECTANT,
        }
        return mapping[self]
    @property
    def hospital_required(self) -> bool:
        return self.value >= 2
    @property
    def emergency_transport(self) -> bool:
        return self.value >= 4
    @classmethod
    def from_rts(cls, rts_score: float) -> "NACAScore":
        if rts_score >= 7.0:   return cls.TWO
        if rts_score >= 6.0:   return cls.THREE
        if rts_score >= 5.0:   return cls.FOUR
        if rts_score >= 3.0:   return cls.FIVE
        if rts_score >= 1.0:   return cls.SIX
        return cls.SEVEN
class MCITriageSupervisor:
    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id
        self._start_engine = STARTProtocolEngine()
        self._jumpstart_engine = JumpSTARTProtocolEngine()
        self._salt_engine = SALTProtocolEngine()
        self._ground_truth_db = TriageGroundTruthDatabase()
        self._decisions: Dict[str, TriageDecision] = {}
        self._deterioration_tracker = PatientDeteriorationTracker()
        self._rts_scores: Dict[str, RevisedTraumaScore] = {}
        self._total_victims: int = 0
    def triage_victim(
        self,
        patient_id: str,
        victim_index: int,
        rpm: RPMScore,
        condition_key: str,
        step: int = 0,
        cbrn_type: CbrnContaminationType = CbrnContaminationType.NONE,
        use_salt: bool = False,
    ) -> TriageDecision:
        gt_tag, gt_conf = self._ground_truth_db.get_ground_truth(condition_key)
        is_paediatric = (
            rpm.is_paediatric
            or (rpm.patient_age_years is not None and rpm.patient_age_years < PAEDIATRIC_AGE_THRESHOLD)
        )
        if use_salt or cbrn_type != CbrnContaminationType.NONE:
            decision = self._salt_engine.assess_phase(
                rpm=rpm,
                patient_id=patient_id,
                incident_id=self.incident_id,
                victim_index=victim_index,
                condition_key=condition_key,
                ground_truth_tag=gt_tag,
                cbrn_type=cbrn_type,
            )
        elif is_paediatric:
            decision = self._jumpstart_engine.assess(
                rpm=rpm,
                patient_id=patient_id,
                incident_id=self.incident_id,
                victim_index=victim_index,
                condition_key=condition_key,
                ground_truth_tag=gt_tag,
            )
        else:
            decision = self._start_engine.assess(
                rpm=rpm,
                patient_id=patient_id,
                incident_id=self.incident_id,
                victim_index=victim_index,
                condition_key=condition_key,
                ground_truth_tag=gt_tag,
            )
        decision.step_triaged = step
        self._decisions[patient_id] = decision
        rts = RevisedTraumaScore.from_rpm(rpm)
        self._rts_scores[patient_id] = rts
        if decision.deterioration_risk_steps is not None:
            self._deterioration_tracker.register(
                patient_id=patient_id,
                initial_tag=decision.assigned_tag,
                condition_key=condition_key,
                deterioration_steps=decision.deterioration_risk_steps,
                next_tag=decision.next_escalation_tag,
            )
        self._total_victims = max(self._total_victims, victim_index + 1)
        logger.debug(
            "MCITriageSupervisor: %s victim %d → %s (%s) | gt=%s",
            patient_id, victim_index,
            decision.assigned_tag.value,
            decision.protocol_used.value,
            gt_tag.value if gt_tag else "unknown",
        )
        return decision
    def triage_all(
        self,
        victims: List[Dict[str, Any]],
        step: int = 0,
    ) -> List[TriageDecision]:
        decisions = []
        sort_tiers = self._salt_engine.sort_phase(
            [RPMScore.from_dict(v["rpm"]) for v in victims]
        )
        logger.info(
            "MCITriageSupervisor: %d victims — walk=%d move=%d still=%d",
            len(victims),
            len(sort_tiers["walk"]),
            len(sort_tiers["move"]),
            len(sort_tiers["still"]),
        )
        ordered_indices = (
            sort_tiers["still"]
            + sort_tiers["move"]
            + sort_tiers["walk"]
        )
        for i in ordered_indices:
            v = victims[i]
            rpm = RPMScore.from_dict(v["rpm"])
            decision = self.triage_victim(
                patient_id=v["patient_id"],
                victim_index=v.get("victim_index", i),
                rpm=rpm,
                condition_key=v["condition_key"],
                step=step,
                cbrn_type=CbrnContaminationType(
                    v.get("cbrn_type", CbrnContaminationType.NONE.value)
                ),
            )
            decisions.append(decision)
        return decisions
    def tick(self, step: int) -> List[Tuple[str, TriageTag, TriageTag]]:
        return self._deterioration_tracker.tick(step)
    def mark_treated(self, patient_id: str) -> None:
        self._deterioration_tracker.mark_treated(patient_id)
    def grade_agent_tags(
        self,
        agent_assignments: Dict[str, str],
    ) -> Tuple[float, Dict[str, Tuple[float, str]]]:
        scores: Dict[str, Tuple[float, str]] = {}
        total_score = 0.0
        critical_mismatches = 0
        for pid, assigned_str in agent_assignments.items():
            decision = self._decisions.get(pid)
            if decision is None:
                scores[pid] = (0.0, "patient_not_found")
                continue
            try:
                assigned_tag = TriageTag(assigned_str)
            except ValueError:
                scores[pid] = (0.0, f"invalid_tag_{assigned_str}")
                continue
            from dataclasses import replace
            temp = replace(decision)
            temp_score, explanation = TriageDecision(
                patient_id=pid,
                incident_id=self.incident_id,
                victim_index=decision.victim_index,
                rpm=decision.rpm,
                condition_key=decision.condition_key,
                patient_age_years=decision.patient_age_years,
                assigned_tag=assigned_tag,
                protocol_used=decision.protocol_used,
                _ground_truth_tag=decision.ground_truth_tag,
            ).grade_against_ground_truth()
            if (
                decision.ground_truth_tag == TriageTag.IMMEDIATE
                and assigned_tag == TriageTag.EXPECTANT
            ):
                critical_mismatches += 1
                temp_score += TRIAGE_CRITICAL_MISMATCH_PENALTY  
            temp_score = max(0.0, min(1.0, temp_score))
            scores[pid] = (temp_score, explanation)
            total_score += temp_score
        n = max(1, len(scores))
        overall = total_score / n
        critical_penalty = critical_mismatches * abs(TRIAGE_CRITICAL_MISMATCH_PENALTY)
        overall = max(0.0, overall - critical_penalty)
        return overall, scores
    def tag_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {
            TriageTag.IMMEDIATE.value: 0,
            TriageTag.DELAYED.value:   0,
            TriageTag.MINIMAL.value:   0,
            TriageTag.EXPECTANT.value: 0,
        }
        for d in self._decisions.values():
            counts[d.assigned_tag.value] += 1
        return counts
    def hospital_load_recommendation(
        self,
        hospital_capacities: Dict[str, int],
    ) -> Dict[str, List[str]]:
        if not hospital_capacities:
            return {}
        immediate_patients = [
            pid for pid, d in self._decisions.items()
            if d.assigned_tag == TriageTag.IMMEDIATE
        ]
        delayed_patients = [
            pid for pid, d in self._decisions.items()
            if d.assigned_tag == TriageTag.DELAYED
        ]
        assignment: Dict[str, List[str]] = {h: [] for h in hospital_capacities}
        hospitals = sorted(hospital_capacities.keys(), key=lambda h: hospital_capacities[h], reverse=True)
        for i, pid in enumerate(immediate_patients):
            h = hospitals[i % len(hospitals)]
            if len(assignment[h]) < hospital_capacities[h]:
                assignment[h].append(pid)
        for pid in delayed_patients:
            for h in hospitals:
                if len(assignment[h]) < hospital_capacities[h]:
                    assignment[h].append(pid)
                    break
        return assignment
    def lsi_requirements_summary(self) -> Dict[str, int]:
        lsi_counts: Dict[str, int] = {}
        for d in self._decisions.values():
            for lsi in d.life_saving_interventions:
                lsi_counts[lsi] = lsi_counts.get(lsi, 0) + 1
        return lsi_counts
    def patients_requiring_decontamination(self) -> List[str]:
        return [
            pid for pid, d in self._decisions.items()
            if d.decontamination_required
        ]
    def critical_patients(self) -> List[str]:
        return [
            pid for pid, d in self._decisions.items()
            if d.assigned_tag == TriageTag.IMMEDIATE
        ]
    def expected_patients(self) -> List[str]:
        return [
            pid for pid, d in self._decisions.items()
            if d.assigned_tag == TriageTag.EXPECTANT
        ]
    def get_decision(self, patient_id: str) -> Optional[TriageDecision]:
        return self._decisions.get(patient_id)
    def get_rts(self, patient_id: str) -> Optional[RevisedTraumaScore]:
        return self._rts_scores.get(patient_id)
    def summary_report(self) -> str:
        dist = self.tag_distribution()
        total = sum(dist.values())
        lsi = self.lsi_requirements_summary()
        lines = [
            f"═══ MCI Triage Summary — incident {self.incident_id} ═══",
            f"Total victims:  {total}",
            f"  IMMEDIATE:    {dist[TriageTag.IMMEDIATE.value]} (RED)",
            f"  DELAYED:      {dist[TriageTag.DELAYED.value]} (YELLOW)",
            f"  MINIMAL:      {dist[TriageTag.MINIMAL.value]} (GREEN)",
            f"  EXPECTANT:    {dist[TriageTag.EXPECTANT.value]} (BLACK)",
            f"LSI required:   {lsi}",
            f"Decon required: {len(self.patients_requiring_decontamination())}",
        ]
        return "\n".join(lines)
class IndianEMSTriageMapper:
    MICU_MANDATORY: FrozenSet[str] = frozenset([
        "stemi_anterior", "stemi_inferior", "stemi_posterior",
        "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
        "cardiac_arrest_vf", "cardiac_arrest_pea",
        "polytrauma_blunt", "polytrauma_penetrating",
        "cervical_spinal_injury",
        "aortic_dissection", "aaa_rupture",
        "obstetric_hemorrhage_shock",
        "severe_tbi",
        "eclampsia_with_seizure",
        "complete_heart_block",
        "wpw_svt_haemodynamic_compromise",
        "hyperkalaemia_ecg_changes",
        "acute_pulmonary_oedema_severe",
        "crush_syndrome",
        "renal_failure_dialysis_cardiac",
        "massive_haemoptysis",
        "mnd_respiratory_failure",
        "neonatal_cardiac",
        "ruptured_ectopic",
        "uterine_rupture",
        "blast_injury_mci",
        "mine_collapse_extraction",
        "weil_disease_severe",
    ])
    ALS_CORRECT_P1: FrozenSet[str] = frozenset([
        "ischemic_stroke", "hemorrhagic_stroke_sah",
        "respiratory_failure", "ards",
        "anaphylaxis", "septic_shock", "sepsis_surgical",
        "burns_major", "burns_moderate",
        "submersion", "near_drowning_arrest",
        "organophosphate_poisoning", "opioid_overdose",
        "paediatric_choking", "status_epilepticus",
        "eclampsia", "obstetric_hemorrhage",
        "toxic_inhalation", "carbon_monoxide",
        "penetrating_chest", "chest_trauma",
        "heat_stroke", "snakebite",
        "dka_paediatric", "dengue_severe",
        "femur_fracture_vascular",
        "meningococcal_sepsis",
    ])
    BLS_CORRECT_P2: FrozenSet[str] = frozenset([
        "normal_delivery", "heat_exhaustion",
        "minor_head_trauma_stable", "renal_colic",
        "nof_fracture_stable", "febrile_seizure_resolved",
        "hypoglycaemia_paediatric_alert",
        "psychiatric_wandering",
        "self_harm_superficial",
        "near_drowning_alert",
    ])
    @classmethod
    def correct_unit_type(cls, condition_key: str, severity: str) -> str:
        if severity == "P0":
            return "BLS"
        if severity == "P3":
            return "BLS"
        if severity == "P2":
            if any(k in condition_key for k in cls.BLS_CORRECT_P2):
                return "BLS"
            return "ALS"  
        if any(k in condition_key for k in cls.MICU_MANDATORY):
            return "MICU"
        if any(k in condition_key for k in cls.ALS_CORRECT_P1):
            return "ALS"
        return "ALS"
    @classmethod
    def unit_type_score(
        cls,
        dispatched: str,
        condition_key: str,
        severity: str,
    ) -> float:
        correct = cls.correct_unit_type(condition_key, severity)
        unit_ranks = {"BLS": 1, "ALS": 2, "MICU": 3}
        d_rank = unit_ranks.get(dispatched.upper(), 0)
        c_rank = unit_ranks.get(correct, 0)
        if d_rank == c_rank:
            return 1.0
        if d_rank > c_rank:
            return 0.80
        gap = c_rank - d_rank
        return max(0.0, 1.0 - gap * 0.40)
    @classmethod
    def is_critical_unit_mismatch(
        cls,
        dispatched: str,
        condition_key: str,
        severity: str,
    ) -> bool:
        correct = cls.correct_unit_type(condition_key, severity)
        return correct == "MICU" and dispatched.upper() == "BLS"
class TriageEngine:
    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = random.Random(rng_seed)
        self._start = STARTProtocolEngine()
        self._jumpstart = JumpSTARTProtocolEngine()
        self._salt = SALTProtocolEngine()
        self._gt_db = TriageGroundTruthDatabase()
        self._ems_mapper = IndianEMSTriageMapper()
        self._mci_supervisors: Dict[str, MCITriageSupervisor] = {}
        self._deterioration = PatientDeteriorationTracker()
        logger.info("TriageEngine initialised with seed=%d", rng_seed)
    def triage_single(
        self,
        patient_id: str,
        incident_id: str,
        rpm: RPMScore,
        condition_key: str,
        victim_index: int = 0,
        step: int = 0,
        expose_ground_truth: bool = False,
    ) -> TriageDecision:
        gt_tag, _ = self._gt_db.get_ground_truth(condition_key)
        is_paediatric = (
            rpm.is_paediatric
            or (rpm.patient_age_years is not None
                and rpm.patient_age_years < PAEDIATRIC_AGE_THRESHOLD)
        )
        if is_paediatric:
            decision = self._jumpstart.assess(
                rpm=rpm,
                patient_id=patient_id,
                incident_id=incident_id,
                victim_index=victim_index,
                condition_key=condition_key,
                ground_truth_tag=gt_tag,
            )
        else:
            decision = self._start.assess(
                rpm=rpm,
                patient_id=patient_id,
                incident_id=incident_id,
                victim_index=victim_index,
                condition_key=condition_key,
                ground_truth_tag=gt_tag,
            )
        decision.step_triaged = step
        if decision.deterioration_risk_steps is not None:
            self._deterioration.register(
                patient_id=patient_id,
                initial_tag=decision.assigned_tag,
                condition_key=condition_key,
                deterioration_steps=decision.deterioration_risk_steps,
                next_tag=decision.next_escalation_tag,
            )
        return decision
    def get_or_create_mci(self, incident_id: str) -> MCITriageSupervisor:
        if incident_id not in self._mci_supervisors:
            self._mci_supervisors[incident_id] = MCITriageSupervisor(incident_id)
        return self._mci_supervisors[incident_id]
    def triage_mci_victim(
        self,
        incident_id: str,
        patient_id: str,
        victim_index: int,
        rpm: RPMScore,
        condition_key: str,
        step: int = 0,
    ) -> TriageDecision:
        supervisor = self.get_or_create_mci(incident_id)
        return supervisor.triage_victim(
            patient_id=patient_id,
            victim_index=victim_index,
            rpm=rpm,
            condition_key=condition_key,
            step=step,
        )
    def score_triage_decision(
        self,
        agent_tag: str,
        condition_key: str,
    ) -> Tuple[float, str]:
        gt_tag, confidence = self._gt_db.get_ground_truth(condition_key)
        if gt_tag is None:
            return (0.5, "unknown_condition_partial_credit")
        try:
            assigned = TriageTag(agent_tag)
        except ValueError:
            return (0.0, f"invalid_tag_value:{agent_tag}")
        dummy = TriageDecision(
            patient_id="grade",
            incident_id="grade",
            victim_index=0,
            rpm=RPMScore(
                respirations=RespirationStatus.PRESENT_NORMAL.value,
                pulse=PulseStatus.PRESENT_STRONG.value,
                mental_status=MentalStatus.ALERT.value,
            ),
            condition_key=condition_key,
            patient_age_years=None,
            assigned_tag=assigned,
            protocol_used=TriageProtocol.START,
            _ground_truth_tag=gt_tag,
        )
        score, explanation = dummy.grade_against_ground_truth()
        adjusted = score * confidence + (1 - confidence) * 0.5
        return (max(0.0, min(1.0, adjusted)), explanation)
    def score_unit_type(
        self,
        dispatched: str,
        condition_key: str,
        severity: str,
    ) -> float:
        return self._ems_mapper.unit_type_score(dispatched, condition_key, severity)
    def correct_unit_type(self, condition_key: str, severity: str) -> str:
        return self._ems_mapper.correct_unit_type(condition_key, severity)
    def ground_truth_tag(self, condition_key: str) -> Optional[TriageTag]:
        tag, _ = self._gt_db.get_ground_truth(condition_key)
        return tag
    def tick(self, step: int) -> List[Tuple[str, TriageTag, TriageTag]]:
        escalations = self._deterioration.tick(step)
        for sup in self._mci_supervisors.values():
            escalations.extend(sup.tick(step))
        return escalations
    def mark_treated(self, patient_id: str) -> None:
        self._deterioration.mark_treated(patient_id)
        for sup in self._mci_supervisors.values():
            sup.mark_treated(patient_id)
    def critical_patients(self) -> List[str]:
        result = self._deterioration.critical_patients()
        for sup in self._mci_supervisors.values():
            result.extend(sup.critical_patients())
        return list(set(result))
    def patients_near_escalation(self, window: int = 2) -> List[Tuple[str, int]]:
        return self._deterioration.patients_near_escalation(window)
    def compute_rts(self, rpm: RPMScore) -> RevisedTraumaScore:
        return RevisedTraumaScore.from_rpm(rpm)
    def compute_naca(self, rts: RevisedTraumaScore) -> NACAScore:
        return NACAScore.from_rts(rts.rts)
    def is_critical_unit_mismatch(
        self,
        dispatched: str,
        condition_key: str,
        severity: str,
    ) -> bool:
        return self._ems_mapper.is_critical_unit_mismatch(dispatched, condition_key, severity)
    def is_critical_triage_mismatch(
        self,
        assigned_tag: str,
        condition_key: str,
    ) -> bool:
        try:
            t = TriageTag(assigned_tag)
        except ValueError:
            return False
        return self._gt_db.is_critical_mismatch(t, condition_key)
    def reset(self) -> None:
        self._mci_supervisors.clear()
        self._deterioration = PatientDeteriorationTracker()
        logger.debug("TriageEngine.reset()")
def _self_test() -> None:
    engine = TriageEngine(rng_seed=42)
    stemi_rpm = RPMScore(
        respirations=RespirationStatus.PRESENT_NORMAL.value,
        pulse=PulseStatus.PRESENT_WEAK.value,
        mental_status=MentalStatus.ALERT.value,
        heart_rate=48, systolic_bp=90, gcs_score=15,
    )
    d1 = engine.triage_single("P001", "INC001", stemi_rpm, "stemi_anterior")
    assert d1.assigned_tag == TriageTag.IMMEDIATE, f"STEMI should be IMMEDIATE, got {d1.assigned_tag}"
    assert d1.protocol_used == TriageProtocol.START
    arrest_rpm = RPMScore(
        respirations=RespirationStatus.ABSENT.value,
        pulse=PulseStatus.ABSENT.value,
        mental_status=MentalStatus.UNRESPONSIVE.value,
    )
    d2 = engine.triage_single("P002", "INC001", arrest_rpm, "cardiac_arrest_vf")
    assert d2.assigned_tag == TriageTag.EXPECTANT  
    minor_rpm = RPMScore(
        respirations=RespirationStatus.PRESENT_NORMAL.value,
        pulse=PulseStatus.PRESENT_STRONG.value,
        mental_status=MentalStatus.ALERT.value,
        can_walk=True,
    )
    d3 = engine.triage_single("P003", "INC001", minor_rpm, "minor_trauma")
    assert d3.assigned_tag == TriageTag.MINIMAL
    ped_rpm = RPMScore(
        respirations=RespirationStatus.PRESENT_ABNORMAL.value,
        pulse=PulseStatus.PRESENT_RAPID.value,
        mental_status=MentalStatus.ALERT.value,
        patient_age_years=5, is_paediatric=True,
    )
    d4 = engine.triage_single("P004", "INC001", ped_rpm, "status_epilepticus")
    assert d4.assigned_tag == TriageTag.IMMEDIATE
    assert d4.protocol_used == TriageProtocol.JUMPSTART
    score, expl = engine.score_triage_decision("Immediate", "stemi_anterior")
    assert score == 1.0, f"Expected 1.0, got {score}"
    score2, expl2 = engine.score_triage_decision("Expectant", "stemi_anterior")
    assert score2 == 0.0 or score2 < 0.3, f"Critical mismatch should score near 0, got {score2}"
    u1 = engine.score_unit_type("MICU", "stemi_anterior", "P1")
    assert u1 == 1.0, f"MICU for STEMI should be 1.0, got {u1}"
    u2 = engine.score_unit_type("BLS", "stemi_anterior", "P1")
    assert u2 < 0.5, f"BLS for STEMI should score <0.5, got {u2}"
    rts = engine.compute_rts(stemi_rpm)
    assert 0.0 <= rts.rts <= 7.85
    engine.tick(step=1)
    engine.tick(step=2)
    mci_victims = [
        {
            "patient_id": f"V{i:03d}",
            "victim_index": i,
            "rpm": {
                "respirations": "normal" if i % 3 != 0 else "absent",
                "pulse": "present_strong" if i % 4 != 0 else "absent",
                "mental_status": "alert" if i % 5 != 0 else "unresponsive",
                "can_walk": i % 6 == 0,
            },
            "condition_key": "polytrauma_blunt",
        }
        for i in range(10)
    ]
    sup = engine.get_or_create_mci("MCI001")
    decisions = sup.triage_all(mci_victims, step=0)
    assert len(decisions) == 10
    dist = sup.tag_distribution()
    assert sum(dist.values()) == 10
    assert engine.is_critical_triage_mismatch("Expectant", "stemi_anterior")
    assert not engine.is_critical_triage_mismatch("Immediate", "stemi_anterior")
    naca = engine.compute_naca(rts)
    assert isinstance(naca, NACAScore)
    logger.info("triage.py self-test PASSED — %d assertions verified", 20)
_self_test()
logger.info(
    "EMERGI-ENV server.medical.triage v%d loaded — "
    "START + JumpSTART + SALT engines, %d ground-truth conditions, "
    "deterioration tracker, MCI supervisor, Indian EMS mapper.",
    TRIAGE_VERSION,
    len(TriageGroundTruthDatabase._GROUND_TRUTH),
)
__all__ = [
    "TriageTag",
    "RespirationStatus",
    "PulseStatus",
    "MentalStatus",
    "TriageProtocol",
    "CbrnContaminationType",
    "NACAScore",
    "RPMScore",
    "TriageDecision",
    "RevisedTraumaScore",
    "STARTProtocolEngine",
    "JumpSTARTProtocolEngine",
    "SALTProtocolEngine",
    "PatientDeteriorationTracker",
    "TriageGroundTruthDatabase",
    "MCITriageSupervisor",
    "IndianEMSTriageMapper",
    "TriageEngine",
    "_infer_gcs_from_rpm",
    "_infer_sbp_from_rpm",
    "_infer_rr_from_rpm",
    "TRIAGE_VERSION",
    "PAEDIATRIC_AGE_THRESHOLD",
    "TRIAGE_EXACT_MATCH_SCORE",
    "TRIAGE_CRITICAL_MISMATCH_PENALTY",
    "TRIAGE_CRITICAL_MISMATCH_SCORE",
    "DETERIORATION_STEPS",
]