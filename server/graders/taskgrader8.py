from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from server.graders.basegrader import (
    ActionLogEntry,
    BaseGrader,
    GraderInput,
    GraderResult,
    GraderRegistry,
    GraderStatus,
    PenaltyRecord,
    ProtocolRuleChecker,
    ScoreComponent,
    ScoringUtils,
    TASK_BASELINES,
    TASK_SEEDS,
)
logger = logging.getLogger("emergi_env.graders.task8")
TASK_ID       = "task8_transfer_cascade"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]
W_TRANSFER_APPROP   = 0.35
W_TIMING_QUALITY    = 0.25
W_BED_UTILISATION   = 0.20
W_CASCADE_AVOIDANCE = 0.15
W_PROTOCOL          = 0.05
assert abs(
    W_TRANSFER_APPROP + W_TIMING_QUALITY + W_BED_UTILISATION +
    W_CASCADE_AVOIDANCE + W_PROTOCOL - 1.0
) < 1e-9, "Task8 component weights must sum to 1.0"
HP_WRONG_SPECIALTY_DEST   = 0.20
HP_CRITICAL_IN_BLS        = 0.25
HP_TRANSFER_TO_DIVERTED   = 0.30
HP_CASCADE_FAILURE        = 0.40   
HP_ICU_SATURATION_EVENT   = 0.10   
HP_DEADLOCK_PATIENT       = 0.15   
HP_NO_PRE_ALERT           = 0.05   
HP_OVER_TRANSFER          = 0.08   
HP_WRONG_UNIT_STABLE      = 0.05   
HP_NO_RETURN_CREW         = 0.04   
PB_CATH_LAB_PRE_ACTIVATION  = 0.030
PB_NEUROSURG_PRE_ALERT      = 0.025
PB_ALL_CRITICAL_UNDER_60    = 0.035
PB_ZERO_CASCADE             = 0.040
PB_PERFECT_BED_UTIL         = 0.030
PB_MULTI_SPECIALIST_COORD   = 0.020
PROTOCOL_BONUS_CAP          = 0.15
DEADLOCK_THRESHOLD_MIN      = 90.0    
CRITICAL_TRANSFER_TARGET_MIN= 60.0    
STABLE_TRANSFER_TARGET_MIN  = 120.0
ICU_SATURATION_THRESHOLD    = 0.95    
ICU_IDEAL_LOW               = 0.60    
ICU_IDEAL_HIGH              = 0.85    
DIVERSION_THRESHOLD         = 0.90    
CONDITION_SPECIALTY_MAP: Dict[str, str] = {
    "stemi":              "cath_lab",
    "nstemi":             "cath_lab",
    "aortic_dissection":  "cardiothoracic_surgery",
    "stroke_ischaemic":   "stroke_unit",
    "stroke_haemorrhagic":"neurosurgery",
    "subdural_haematoma": "neurosurgery",
    "epidural_haematoma": "neurosurgery",
    "tbi_severe":         "neurosurgery",
    "burns_major":        "burns_unit",
    "burns_inhalation":   "burns_unit",
    "paediatric_trauma":  "paediatric_icu",
    "paediatric_sepsis":  "paediatric_icu",
    "obstetric_emergency":"obstetric_icu",
    "polytrauma":         "trauma_surgery",
    "abdominal_trauma":   "trauma_surgery",
    "renal_failure":      "nephrology",
    "liver_failure":      "hepatology",
    "septic_shock":       "icu_general",
    "respiratory_failure":"icu_general",
    "post_cardiac_arrest":"icu_general",
}
CRITICAL_TRANSFER_CONDITIONS: FrozenSet[str] = frozenset({
    "stemi", "nstemi", "aortic_dissection",
    "stroke_ischaemic", "stroke_haemorrhagic",
    "subdural_haematoma", "epidural_haematoma",
    "tbi_severe", "burns_inhalation",
    "paediatric_trauma", "obstetric_emergency",
    "polytrauma", "septic_shock", "post_cardiac_arrest",
})
STABLE_TRANSFER_CONDITIONS: FrozenSet[str] = frozenset({
    "renal_failure", "liver_failure",
    "burns_major",  
    "respiratory_failure",  
})
SPECIALTY_HOSPITAL_MAP: Dict[str, List[str]] = {
    "cath_lab":               ["H01", "H03", "H07"],
    "cardiothoracic_surgery": ["H01", "H07"],
    "stroke_unit":            ["H01", "H02", "H05"],
    "neurosurgery":           ["H01", "H03"],
    "burns_unit":             ["H04"],
    "paediatric_icu":         ["H02", "H06"],
    "obstetric_icu":          ["H02", "H06"],
    "trauma_surgery":         ["H01", "H03", "H07"],
    "nephrology":             ["H01", "H02", "H05"],
    "hepatology":             ["H01", "H05"],
    "icu_general":            ["H01","H02","H03","H04","H05","H06","H07","H08"],
}
HOSPITAL_ICU_CAPACITY: Dict[str, int] = {
    "H01": 40, "H02": 30, "H03": 25, "H04": 20,
    "H05": 20, "H06": 15, "H07": 35, "H08": 12,
}
@unique
class TransferStatus(str, Enum):
    COMPLETED  = "completed"
    IN_TRANSIT = "in_transit"
    CANCELLED  = "cancelled"
    STUCK      = "stuck"       
@unique
class PatientAcuity(str, Enum):
    CRITICAL = "critical"
    STABLE   = "stable"
    DECEASED = "deceased"
@dataclass(frozen=True)
class TransferRecord:
    transfer_id:      str
    step:             int
    patient_id:       str
    condition_code:   str
    source_hospital:  str
    dest_hospital:    str
    unit_type:        str           
    pre_alert_given:  bool
    status:           TransferStatus
    delay_min:        float         
    transit_time_min: float         
    cath_lab_activated_dest: bool
    neurosurg_pre_alerted:   bool
    specialist_notified:     bool
    acuity:           PatientAcuity = PatientAcuity.CRITICAL
@dataclass
class HospitalBedState:
    hospital_id:   str
    icu_capacity:  int
    icu_occupied:  int             = 0
    saturation_events: int         = 0
    on_diversion:  bool            = False
    received_transfers: int        = 0
    sent_transfers:     int        = 0
    @property
    def occupancy_rate(self) -> float:
        if self.icu_capacity == 0:
            return 1.0
        return self.icu_occupied / self.icu_capacity
    @property
    def available_beds(self) -> int:
        return max(0, self.icu_capacity - self.icu_occupied)
    def add_patient(self) -> None:
        self.icu_occupied = min(self.icu_capacity, self.icu_occupied + 1)
        if self.occupancy_rate >= ICU_SATURATION_THRESHOLD:
            self.saturation_events += 1
        if self.occupancy_rate >= DIVERSION_THRESHOLD:
            self.on_diversion = True
    def remove_patient(self) -> None:
        self.icu_occupied = max(0, self.icu_occupied - 1)
        if self.occupancy_rate < DIVERSION_THRESHOLD:
            self.on_diversion = False
@dataclass
class PatientTransferState:
    patient_id:      str
    condition_code:  str
    severity:        str           
    source_hospital: str
    icu_need_step:   int           
    acuity:          PatientAcuity = PatientAcuity.CRITICAL
    transfer:        Optional[TransferRecord] = None
    final_outcome:   str = "untransferred"  
    weighted_reward: float = 0.0
class TransferAppropriateness:
    @classmethod
    def required_specialty(cls, condition_code: str) -> str:
        for pattern, specialty in CONDITION_SPECIALTY_MAP.items():
            if pattern in condition_code.lower():
                return specialty
        return "icu_general"
    @classmethod
    def acceptable_hospitals(cls, condition_code: str) -> List[str]:
        specialty = cls.required_specialty(condition_code)
        return SPECIALTY_HOSPITAL_MAP.get(specialty, list(HOSPITAL_ICU_CAPACITY.keys()))
    @classmethod
    def is_critical(cls, condition_code: str) -> bool:
        return any(c in condition_code.lower() for c in CRITICAL_TRANSFER_CONDITIONS)
    @classmethod
    def is_stable(cls, condition_code: str) -> bool:
        return any(c in condition_code.lower() for c in STABLE_TRANSFER_CONDITIONS)
    @classmethod
    def required_unit(cls, condition_code: str) -> str:
        if cls.is_critical(condition_code):
            return "MICU"
        return "ALS"
    @classmethod
    def score_single_transfer(
        cls,
        transfer: TransferRecord,
        hospital_states: Dict[str, HospitalBedState],
    ) -> Tuple[float, List[str]]:
        violations: List[str] = []
        score = 1.0
        acceptable = cls.acceptable_hospitals(transfer.condition_code)
        specialty  = cls.required_specialty(transfer.condition_code)
        if transfer.dest_hospital not in acceptable:
            violations.append(
                f"wrong_specialty: {transfer.dest_hospital} lacks {specialty} "
                f"for {transfer.condition_code}"
            )
            score -= 0.40
        dest_state = hospital_states.get(transfer.dest_hospital)
        if dest_state and dest_state.on_diversion:
            violations.append(
                f"transfer_to_diverted: {transfer.dest_hospital} on diversion"
            )
            score -= 0.30
        if dest_state and dest_state.available_beds == 0:
            violations.append(
                f"no_beds: {transfer.dest_hospital} has 0 available ICU beds"
            )
            score -= 0.20
        req_unit = cls.required_unit(transfer.condition_code)
        if cls.is_critical(transfer.condition_code) and transfer.unit_type == "BLS":
            violations.append(
                f"critical_in_bls: {transfer.patient_id} needs MICU "
                f"(condition={transfer.condition_code})"
            )
            score -= 0.35
        elif req_unit == "MICU" and transfer.unit_type == "ALS":
            score -= 0.10
        if not transfer.pre_alert_given:
            violations.append(f"no_pre_alert: {transfer.patient_id} → {transfer.dest_hospital}")
            score -= 0.05
        return ScoringUtils.clamp(score), violations
class TransferTimingEngine:
    CONDITION_TARGETS: Dict[str, float] = {
        "stemi":              45.0,   
        "nstemi":             90.0,
        "aortic_dissection":  30.0,   
        "stroke_ischaemic":   60.0,   
        "stroke_haemorrhagic":45.0,
        "subdural_haematoma": 45.0,
        "epidural_haematoma": 30.0,
        "tbi_severe":         60.0,
        "burns_major":       180.0,   
        "burns_inhalation":   90.0,
        "paediatric_trauma":  60.0,
        "paediatric_sepsis":  90.0,
        "obstetric_emergency":45.0,
        "polytrauma":         60.0,
        "abdominal_trauma":   90.0,
        "renal_failure":     240.0,   
        "liver_failure":     180.0,
        "septic_shock":       90.0,
        "respiratory_failure":120.0,
        "post_cardiac_arrest":60.0,
    }
    DEFAULT_TARGET_CRITICAL = 60.0
    DEFAULT_TARGET_STABLE   = 180.0
    @classmethod
    def target_minutes(cls, condition_code: str) -> float:
        for pattern, target in cls.CONDITION_TARGETS.items():
            if pattern in condition_code.lower():
                return target
        if TransferAppropriateness.is_critical(condition_code):
            return cls.DEFAULT_TARGET_CRITICAL
        return cls.DEFAULT_TARGET_STABLE
    @classmethod
    def score_transfer_timing(cls, transfer: TransferRecord) -> float:
        total_elapsed = transfer.delay_min + transfer.transit_time_min
        target        = cls.target_minutes(transfer.condition_code)
        worst_case    = target * 4.0
        if transfer.acuity == PatientAcuity.CRITICAL:
            return ScoringUtils.exponential_decay_score(
                elapsed_min=max(0.0, total_elapsed - target),
                lambda_=0.025,
                floor=0.0,
            )
        else:
            return ScoringUtils.response_time_score(
                actual_min=total_elapsed,
                target_min=target,
                worst_case_min=worst_case,
                shape="linear",
            )
    @classmethod
    def all_critical_under_target(cls, transfers: List[TransferRecord]) -> bool:
        critical = [t for t in transfers if t.acuity == PatientAcuity.CRITICAL
                    and t.status == TransferStatus.COMPLETED]
        if not critical:
            return False
        return all(
            (t.delay_min + t.transit_time_min) <= cls.target_minutes(t.condition_code)
            for t in critical
        )
class BedUtilisationEngine:
    @classmethod
    def score_network_utilisation(
        cls,
        hospital_states: Dict[str, HospitalBedState],
    ) -> float:
        if not hospital_states:
            return 0.0
        scores = []
        for state in hospital_states.values():
            occ = state.occupancy_rate
            if occ < 0.40:
                s = ScoringUtils.linear_decay_score(occ, best=ICU_IDEAL_LOW,
                                                    worst=0.0)
                scores.append(s * 0.7)
            elif ICU_IDEAL_LOW <= occ <= ICU_IDEAL_HIGH:
                scores.append(1.0)
            elif occ <= ICU_SATURATION_THRESHOLD:
                s = ScoringUtils.linear_decay_score(
                    occ, best=ICU_IDEAL_HIGH, worst=ICU_SATURATION_THRESHOLD
                )
                scores.append(max(0.4, s))
            else:
                scores.append(0.1)
        return ScoringUtils.clamp(sum(scores) / len(scores))
    @classmethod
    def is_perfect_utilisation(
        cls,
        hospital_states: Dict[str, HospitalBedState],
    ) -> bool:
        if not hospital_states:
            return False
        return all(
            ICU_IDEAL_LOW <= s.occupancy_rate <= ICU_IDEAL_HIGH
            for s in hospital_states.values()
            if s.icu_capacity > 0
        )
    @classmethod
    def count_saturation_events(
        cls,
        hospital_states: Dict[str, HospitalBedState],
    ) -> int:
        return sum(s.saturation_events for s in hospital_states.values())
class CascadeAvoidanceEngine:
    @classmethod
    def score_cascade_avoidance(
        cls,
        cascade_failure_occurred: bool,
        hospital_states: Dict[str, HospitalBedState],
        transfers: List[TransferRecord],
    ) -> float:
        if cascade_failure_occurred:
            return 0.0
        saturated = sum(
            1 for s in hospital_states.values()
            if s.saturation_events > 0
        )
        total_hospitals = max(1, len(hospital_states))
        if saturated == 0:
            return 1.0
        saturation_ratio = saturated / total_hospitals
        base = 1.0 - saturation_ratio
        diverted_transfers = [
            t for t in transfers
            if hospital_states.get(t.dest_hospital, HospitalBedState("X", 1, 1)).on_diversion
        ]
        if diverted_transfers:
            base -= 0.15 * min(1.0, len(diverted_transfers) / max(1, len(transfers)))
        return ScoringUtils.clamp(base)
    @classmethod
    def detect_deadlocked_patients(
        cls,
        patient_states: List[PatientTransferState],
    ) -> List[str]:
        return [
            p.patient_id for p in patient_states
            if p.final_outcome == "deadlocked"
            or (p.transfer and p.transfer.status == TransferStatus.STUCK)
        ]
class Task8InputParser:
    @classmethod
    def parse(
        cls,
        grader_input: GraderInput,
    ) -> Tuple[
        List[PatientTransferState],
        List[TransferRecord],
        Dict[str, HospitalBedState],
        bool,   
        bool,   
        bool,   
        int,    
        int,    
    ]:
        patient_states  = cls._parse_patients(grader_input)
        transfer_records= cls._parse_transfers(grader_input, patient_states)
        hospital_states = cls._build_hospital_states(grader_input, transfer_records)
        cascade_fail    = grader_input.cascade_failure_occurred
        ma_requested    = cls._detect_mutual_aid(grader_input)
        surge_declared  = bool(grader_input.episode_ledger.get("surge_declared", False))
        comm_failures   = int(grader_input.episode_ledger.get("comm_failures", 0))
        noop_on_unknown = int(grader_input.episode_ledger.get("noop_on_unknown_units", 0))
        cls._link_transfers_to_patients(patient_states, transfer_records)
        return (
            patient_states, transfer_records, hospital_states,
            cascade_fail, ma_requested, surge_declared,
            comm_failures, noop_on_unknown,
        )
    @classmethod
    def _parse_patients(cls, gi: GraderInput) -> List[PatientTransferState]:
        summaries = gi.all_patient_summaries()
        states: List[PatientTransferState] = []
        for s in summaries:
            pid  = s.get("patient_id", "UNK")
            cond = s.get("condition_code", "icu_general")
            sev  = s.get("severity", "P1")
            src  = s.get("source_hospital", s.get("hospital_id", "H01"))
            need_step = int(s.get("icu_need_step", s.get("step", 0)))
            outcome   = s.get("phase_at_episode_end", "untransferred")
            reward    = float(s.get("weighted_reward", 0.0))
            if TransferAppropriateness.is_critical(cond):
                acuity = PatientAcuity.CRITICAL
            elif TransferAppropriateness.is_stable(cond):
                acuity = PatientAcuity.STABLE
            else:
                acuity = PatientAcuity.CRITICAL if sev == "P1" else PatientAcuity.STABLE
            ps = PatientTransferState(
                patient_id=pid,
                condition_code=cond,
                severity=sev,
                source_hospital=src,
                icu_need_step=need_step,
                acuity=acuity,
                final_outcome=outcome,
                weighted_reward=reward,
            )
            states.append(ps)
        return states
    @classmethod
    def _parse_transfers(
        cls,
        gi: GraderInput,
        patient_states: List[PatientTransferState],
    ) -> List[TransferRecord]:
        transfer_actions = gi.get_actions_by_type("transfer")
        patient_map = {p.patient_id: p for p in patient_states}
        records: List[TransferRecord] = []
        for i, entry in enumerate(transfer_actions):
            d    = entry.action_data
            pid  = d.get("patient_id", f"PT{i:03d}")
            cond = d.get("condition_code", "")
            if not cond and pid in patient_map:
                cond = patient_map[pid].condition_code
            if not cond:
                cond = "icu_general"
            src   = d.get("source_hospital", "H01")
            dest  = d.get("dest_hospital", d.get("destination_hospital", "H01"))
            unit  = d.get("unit_type", "MICU").upper()
            if unit not in ("MICU", "ALS", "BLS"):
                unit = "MICU"
            status_raw = d.get("status", "completed").lower()
            status_map = {
                "completed":  TransferStatus.COMPLETED,
                "in_transit": TransferStatus.IN_TRANSIT,
                "cancelled":  TransferStatus.CANCELLED,
                "stuck":      TransferStatus.STUCK,
            }
            status = status_map.get(status_raw, TransferStatus.COMPLETED)
            delay_min   = float(d.get("delay_min", d.get("delay_minutes", 30.0)))
            transit_min = float(d.get("transit_time_min", d.get("transit_minutes", 20.0)))
            pre_alert   = bool(d.get("pre_alert_given", d.get("pre_alert", False)))
            cath_act    = bool(d.get("cath_lab_activated_dest", False))
            neuro_alert = bool(d.get("neurosurg_pre_alerted", False))
            spec_notif  = bool(d.get("specialist_notified", False))
            if TransferAppropriateness.is_critical(cond):
                acuity = PatientAcuity.CRITICAL
            else:
                acuity = PatientAcuity.STABLE
            rec = TransferRecord(
                transfer_id=f"TXF-{TASK_ID[:3].upper()}-{i:04d}",
                step=entry.step,
                patient_id=pid,
                condition_code=cond,
                source_hospital=src,
                dest_hospital=dest,
                unit_type=unit,
                pre_alert_given=pre_alert,
                status=status,
                delay_min=delay_min,
                transit_time_min=transit_min,
                cath_lab_activated_dest=cath_act,
                neurosurg_pre_alerted=neuro_alert,
                specialist_notified=spec_notif,
                acuity=acuity,
            )
            records.append(rec)
        return records
    @classmethod
    def _build_hospital_states(
        cls,
        gi: GraderInput,
        transfers: List[TransferRecord],
    ) -> Dict[str, HospitalBedState]:
        states: Dict[str, HospitalBedState] = {}
        fhs = gi.final_hospital_state or {}
        for hid, cap in HOSPITAL_ICU_CAPACITY.items():
            raw     = fhs.get(hid, {})
            occupied= int(raw.get("icu_occupied", raw.get("icu_beds_used", 0)))
            sat_ev  = int(raw.get("saturation_events", 0))
            on_div  = bool(raw.get("on_diversion", False))
            states[hid] = HospitalBedState(
                hospital_id=hid,
                icu_capacity=cap,
                icu_occupied=occupied,
                saturation_events=sat_ev,
                on_diversion=on_div,
            )
        for t in sorted(transfers, key=lambda x: x.step):
            if t.status == TransferStatus.COMPLETED:
                src_state = states.get(t.source_hospital)
                if src_state:
                    src_state.remove_patient()
                    src_state.sent_transfers += 1
                dst_state = states.get(t.dest_hospital)
                if dst_state:
                    dst_state.add_patient()
                    dst_state.received_transfers += 1
        return states
    @classmethod
    def _detect_mutual_aid(cls, gi: GraderInput) -> bool:
        ma_actions = gi.get_actions_by_type("request_mutual_aid")
        if ma_actions:
            return True
        return bool(gi.episode_ledger.get("mutual_aid_requested", False))
    @classmethod
    def _link_transfers_to_patients(
        cls,
        patient_states: List[PatientTransferState],
        transfers: List[TransferRecord],
    ) -> None:
        transfer_map: Dict[str, TransferRecord] = {}
        for t in transfers:
            transfer_map[t.patient_id] = t
        for ps in patient_states:
            if ps.patient_id in transfer_map:
                ps.transfer = transfer_map[ps.patient_id]
class Task8Grader(BaseGrader):
    TASK_ID          = TASK_ID
    TASK_SEED        = TASK_SEED
    TASK_BASELINE    = TASK_BASELINE
    TASK_DIFFICULTY  = "hard"
    COMPONENT_WEIGHTS = {
        "transfer_appropriateness": W_TRANSFER_APPROP,
        "timing_quality":           W_TIMING_QUALITY,
        "bed_utilisation":          W_BED_UTILISATION,
        "cascade_avoidance":        W_CASCADE_AVOIDANCE,
        "protocol_compliance":      W_PROTOCOL,
    }
    def _grade_impl(
        self,
        grader_input: GraderInput,
        result: GraderResult,
    ) -> None:
        patient_summaries = grader_input.all_patient_summaries()
        if not patient_summaries:
            result.status = GraderStatus.INVALID_INPUT
            result.error_message = (
                "Task8: episode_ledger.patient_summaries is empty — "
                "cannot grade transfer cascade without patient data."
            )
            result.add_note("No patient summaries provided; score = 0.0")
            result.final_score = 0.0
            return
        transfer_actions = grader_input.get_actions_by_type("transfer")
        if not transfer_actions:
            result.status = GraderStatus.PARTIAL
            result.add_note("No transfer actions found in action_log — agent took no transfers.")
        (
            patient_states,
            transfers,
            hospital_states,
            cascade_failure,
            mutual_aid_requested,
            surge_declared,
            comm_failures,
            noop_on_unknown,
        ) = Task8InputParser.parse(grader_input)
        result.total_patients = len(patient_states)
        result.p1_patients    = sum(1 for p in patient_states if p.severity == "P1")
        approp_score = self._score_transfer_appropriateness(
            transfers, hospital_states, result
        )
        self._add_component(
            result, "transfer_appropriateness", approp_score, W_TRANSFER_APPROP,
            notes=f"{len(transfers)} transfers evaluated"
        )
        timing_score = self._score_timing_quality(transfers, patient_states, result)
        self._add_component(
            result, "timing_quality", timing_score, W_TIMING_QUALITY,
            notes=f"avg delay scored across {len(transfers)} transfers"
        )
        util_score = BedUtilisationEngine.score_network_utilisation(hospital_states)
        self._add_component(
            result, "bed_utilisation", util_score, W_BED_UTILISATION,
            notes=(
                f"saturation_events={BedUtilisationEngine.count_saturation_events(hospital_states)} "
                f"| {len(hospital_states)} hospitals"
            )
        )
        cascade_score = CascadeAvoidanceEngine.score_cascade_avoidance(
            cascade_failure, hospital_states, transfers
        )
        self._add_component(
            result, "cascade_avoidance", cascade_score, W_CASCADE_AVOIDANCE,
            notes=f"cascade_failure={cascade_failure}"
        )
        proto_raw  = self._score_protocol_compliance(transfers, result)
        proto_score = ScoringUtils.clamp(proto_raw)
        self._add_component(
            result, "protocol_compliance", proto_score, W_PROTOCOL,
            notes="pre-alert + cath-lab + neurosurg pre-notification"
        )
        self._apply_episode_penalties(
            cascade_failure, hospital_states, patient_states,
            transfers, comm_failures, noop_on_unknown, result
        )
        bonus = self._compute_protocol_bonus(
            transfers, hospital_states, cascade_failure,
            mutual_aid_requested, surge_declared
        )
        if bonus > 0:
            result.add_note(f"Protocol bonus applied: +{bonus:.4f}")
            result.extra["protocol_bonus"] = round(bonus, 4)
            result.add_component(
                ScoringUtils.build_score_component(
                    "protocol_bonus", min(bonus, PROTOCOL_BONUS_CAP), 1.0,
                    notes=f"bonus capped at {PROTOCOL_BONUS_CAP}"
                )
            )
        self._attach_extra_metadata(
            result, patient_states, transfers, hospital_states,
            cascade_failure, mutual_aid_requested, surge_declared,
            comm_failures
        )
        result.p1_survival_rate = self._p1_survival_rate(patient_summaries)
    def _score_transfer_appropriateness(
        self,
        transfers: List[TransferRecord],
        hospital_states: Dict[str, HospitalBedState],
        result: GraderResult,
    ) -> float:
        if not transfers:
            result.add_note("No transfers executed — appropriateness score = 0.0")
            return 0.0
        per_transfer_scores: List[float] = []
        for t in transfers:
            score, violations = TransferAppropriateness.score_single_transfer(
                t, hospital_states
            )
            per_transfer_scores.append(score)
            for v in violations:
                result.add_note(f"APPROP VIOLATION [{t.patient_id}]: {v}")
            specialty = TransferAppropriateness.required_specialty(t.condition_code)
            acceptable = TransferAppropriateness.acceptable_hospitals(t.condition_code)
            if t.dest_hospital not in acceptable:
                self._add_penalty(
                    result,
                    name=f"wrong_specialty_{t.patient_id[:8]}",
                    amount=HP_WRONG_SPECIALTY_DEST,
                    reason=f"{t.patient_id}: {t.dest_hospital} lacks {specialty}",
                    rule_ref="EMERGI-ENV Transfer Protocol §4.2",
                )
                result.protocol_violations += 1
            if t.acuity == PatientAcuity.CRITICAL and t.unit_type == "BLS":
                self._add_penalty(
                    result,
                    name=f"critical_in_bls_{t.patient_id[:8]}",
                    amount=HP_CRITICAL_IN_BLS,
                    reason=f"{t.patient_id} (critical) transferred in BLS unit",
                    rule_ref="EMERGI-ENV Transfer Protocol §3.1",
                )
                result.critical_mismatches += 1
            dst_state = hospital_states.get(t.dest_hospital)
            if dst_state and dst_state.on_diversion:
                self._add_penalty(
                    result,
                    name=f"diverted_dest_{t.patient_id[:8]}",
                    amount=HP_TRANSFER_TO_DIVERTED,
                    reason=f"{t.patient_id} sent to diverted hospital {t.dest_hospital}",
                    rule_ref="EMERGI-ENV Transfer Protocol §5.1",
                )
                result.protocol_violations += 1
            if not t.pre_alert_given:
                self._add_penalty(
                    result,
                    name=f"no_pre_alert_{t.patient_id[:8]}",
                    amount=HP_NO_PRE_ALERT,
                    reason=f"{t.patient_id}: no pre-alert to {t.dest_hospital}",
                    rule_ref="EMERGI-ENV Transfer Protocol §6.1",
                )
        return ScoringUtils.clamp(
            sum(per_transfer_scores) / len(per_transfer_scores)
        )
    def _score_timing_quality(
        self,
        transfers: List[TransferRecord],
        patient_states: List[PatientTransferState],
        result: GraderResult,
    ) -> float:
        if not transfers:
            return 0.0
        completed = [t for t in transfers if t.status == TransferStatus.COMPLETED]
        if not completed:
            result.add_note("No completed transfers — timing score = 0.0")
            return 0.0
        weighted_score = 0.0
        total_weight   = 0.0
        for t in completed:
            w     = 2.0 if t.acuity == PatientAcuity.CRITICAL else 1.0
            score = TransferTimingEngine.score_transfer_timing(t)
            weighted_score += w * score
            total_weight   += w
            total_elapsed   = t.delay_min + t.transit_time_min
            target          = TransferTimingEngine.target_minutes(t.condition_code)
            result.add_note(
                f"Timing [{t.patient_id}]: "
                f"elapsed={total_elapsed:.0f}min target={target:.0f}min "
                f"score={score:.3f}"
            )
        stuck_patients = [
            ps for ps in patient_states
            if ps.final_outcome == "deadlocked"
        ]
        for ps in stuck_patients:
            self._add_penalty(
                result,
                name=f"deadlock_{ps.patient_id[:8]}",
                amount=HP_DEADLOCK_PATIENT,
                reason=f"{ps.patient_id} stuck in transfer deadlock >90 min",
                rule_ref="EMERGI-ENV Transfer Protocol §7.3",
            )
        return ScoringUtils.clamp(weighted_score / max(1.0, total_weight))
    def _score_protocol_compliance(
        self,
        transfers: List[TransferRecord],
        result: GraderResult,
    ) -> float:
        if not transfers:
            return 0.0
        scores: List[float] = []
        for t in transfers:
            s = 1.0
            if not t.pre_alert_given:
                s -= 0.30
            cond = t.condition_code.lower()
            if ("stemi" in cond or "nstemi" in cond) and not t.cath_lab_activated_dest:
                s -= 0.25
                result.add_note(
                    f"PROTOCOL [{t.patient_id}]: cath-lab NOT activated at {t.dest_hospital}"
                )
            if ("stroke_haemorrhagic" in cond or "subdural" in cond
                    or "epidural" in cond or "tbi" in cond):
                if not t.neurosurg_pre_alerted:
                    s -= 0.20
                    result.add_note(
                        f"PROTOCOL [{t.patient_id}]: neurosurgery NOT pre-alerted"
                    )
            scores.append(ScoringUtils.clamp(s))
        return ScoringUtils.clamp(sum(scores) / len(scores))
    def _apply_episode_penalties(
        self,
        cascade_failure: bool,
        hospital_states: Dict[str, HospitalBedState],
        patient_states: List[PatientTransferState],
        transfers: List[TransferRecord],
        comm_failures: int,
        noop_on_unknown: int,
        result: GraderResult,
    ) -> None:
        if cascade_failure:
            self._add_penalty(
                result,
                name="cascade_failure_episode",
                amount=HP_CASCADE_FAILURE,
                reason="Full cascade failure: multiple hospital saturation chain",
                rule_ref="EMERGI-ENV §8.1 Cascade Protocol",
            )
            result.add_note("CRITICAL: Cascade failure occurred this episode.")
        for hid, state in hospital_states.items():
            if state.saturation_events > 0:
                penalty = HP_ICU_SATURATION_EVENT * state.saturation_events
                self._add_penalty(
                    result,
                    name=f"icu_saturation_{hid}",
                    amount=penalty,
                    reason=f"{hid}: {state.saturation_events} saturation event(s) (>95% ICU)",
                    rule_ref="EMERGI-ENV §5.3 ICU Management",
                )
        stable_transfers = [
            t for t in transfers
            if t.acuity == PatientAcuity.STABLE
            and t.status == TransferStatus.COMPLETED
        ]
        for t in stable_transfers:
            src_state = hospital_states.get(t.source_hospital)
            if src_state and src_state.occupancy_rate < ICU_IDEAL_LOW:
                self._add_penalty(
                    result,
                    name=f"over_transfer_{t.patient_id[:8]}",
                    amount=HP_OVER_TRANSFER,
                    reason=(
                        f"{t.patient_id}: stable patient transferred away from "
                        f"{t.source_hospital} (occ={src_state.occupancy_rate:.0%})"
                    ),
                    rule_ref="EMERGI-ENV §4.5 Resource Stewardship",
                )
        micu_stable = [
            t for t in transfers
            if t.unit_type == "MICU"
            and TransferAppropriateness.is_stable(t.condition_code)
            and t.status == TransferStatus.COMPLETED
        ]
        for t in micu_stable:
            self._add_penalty(
                result,
                name=f"micu_waste_{t.patient_id[:8]}",
                amount=HP_WRONG_UNIT_STABLE,
                reason=f"MICU used for stable {t.condition_code} — MICU wasted",
                rule_ref="EMERGI-ENV §3.2 Unit Allocation",
            )
        if comm_failures > 0 and noop_on_unknown > 0:
            penalty_count = min(noop_on_unknown, comm_failures)
            self._add_penalty(
                result,
                name="comm_noop_transfer_episode",
                amount=0.05 * penalty_count,
                reason=(
                    f"Agent issued noop() {noop_on_unknown}× while "
                    f"{comm_failures} unit positions unknown"
                ),
                rule_ref="EMERGI-ENV §9.2 Communication Protocol",
            )
    def _compute_protocol_bonus(
        self,
        transfers: List[TransferRecord],
        hospital_states: Dict[str, HospitalBedState],
        cascade_failure: bool,
        mutual_aid_requested: bool,
        surge_declared: bool,
    ) -> float:
        bonus = 0.0
        stemi_transfers = [
            t for t in transfers
            if ("stemi" in t.condition_code.lower())
            and t.cath_lab_activated_dest
        ]
        if stemi_transfers:
            bonus += PB_CATH_LAB_PRE_ACTIVATION
            logger.debug("Bonus: cath-lab pre-activation (%.3f)", PB_CATH_LAB_PRE_ACTIVATION)
        neuro_transfers = [
            t for t in transfers
            if ("stroke_haemorrhagic" in t.condition_code.lower()
                or "subdural" in t.condition_code.lower()
                or "tbi" in t.condition_code.lower())
            and t.neurosurg_pre_alerted
        ]
        if neuro_transfers:
            bonus += PB_NEUROSURG_PRE_ALERT
            logger.debug("Bonus: neurosurg pre-alert (%.3f)", PB_NEUROSURG_PRE_ALERT)
        if TransferTimingEngine.all_critical_under_target(transfers):
            bonus += PB_ALL_CRITICAL_UNDER_60
            logger.debug("Bonus: all critical transfers under target (%.3f)",
                         PB_ALL_CRITICAL_UNDER_60)
        if not cascade_failure:
            sat_events = BedUtilisationEngine.count_saturation_events(hospital_states)
            if sat_events == 0:
                bonus += PB_ZERO_CASCADE
                logger.debug("Bonus: zero cascade events (%.3f)", PB_ZERO_CASCADE)
        if BedUtilisationEngine.is_perfect_utilisation(hospital_states):
            bonus += PB_PERFECT_BED_UTIL
            logger.debug("Bonus: perfect bed utilisation (%.3f)", PB_PERFECT_BED_UTIL)
        specialist_notified = [
            t for t in transfers if t.specialist_notified
        ]
        if len(specialist_notified) >= 2:
            bonus += PB_MULTI_SPECIALIST_COORD
            logger.debug("Bonus: multi-specialist coordination (%.3f)",
                         PB_MULTI_SPECIALIST_COORD)
        return min(bonus, PROTOCOL_BONUS_CAP)
    def _attach_extra_metadata(
        self,
        result: GraderResult,
        patient_states: List[PatientTransferState],
        transfers: List[TransferRecord],
        hospital_states: Dict[str, HospitalBedState],
        cascade_failure: bool,
        mutual_aid_requested: bool,
        surge_declared: bool,
        comm_failures: int,
    ) -> None:
        n_critical   = sum(1 for p in patient_states if p.acuity == PatientAcuity.CRITICAL)
        n_stable     = sum(1 for p in patient_states if p.acuity == PatientAcuity.STABLE)
        n_completed  = sum(1 for t in transfers if t.status == TransferStatus.COMPLETED)
        n_stuck      = sum(1 for t in transfers if t.status == TransferStatus.STUCK)
        n_diverted   = sum(
            1 for t in transfers
            if hospital_states.get(t.dest_hospital,
                                   HospitalBedState("X", 1, 0)).on_diversion
        )
        n_pre_alerted= sum(1 for t in transfers if t.pre_alert_given)
        n_cath_act   = sum(1 for t in transfers if t.cath_lab_activated_dest)
        n_neuro_alert= sum(1 for t in transfers if t.neurosurg_pre_alerted)
        hosp_occ = {
            hid: round(state.occupancy_rate, 3)
            for hid, state in sorted(hospital_states.items())
        }
        cond_counts: Dict[str, int] = defaultdict(int)
        for p in patient_states:
            cond_counts[p.condition_code] += 1
        deadlocked = CascadeAvoidanceEngine.detect_deadlocked_patients(patient_states)
        result.extra.update({
            "n_patients":          len(patient_states),
            "n_critical_patients": n_critical,
            "n_stable_patients":   n_stable,
            "n_transfers":         len(transfers),
            "n_completed":         n_completed,
            "n_stuck":             n_stuck,
            "n_diverted_dest":     n_diverted,
            "n_pre_alerted":       n_pre_alerted,
            "n_cath_lab_activated":n_cath_act,
            "n_neurosurg_alerted": n_neuro_alert,
            "cascade_failure":     cascade_failure,
            "mutual_aid_requested":mutual_aid_requested,
            "surge_declared":      surge_declared,
            "comm_failures":       comm_failures,
            "deadlocked_patients": deadlocked,
            "hospital_occupancy":  hosp_occ,
            "condition_distribution": dict(cond_counts),
            "saturation_events_total": BedUtilisationEngine.count_saturation_events(
                hospital_states
            ),
            "task_version": "1.0.0",
        })
GraderRegistry.register(TASK_ID, Task8Grader)
logger.info(
    "Task8Grader registered — task_id=%s  baseline=%.2f  seed=%d",
    TASK_ID, TASK_BASELINE, TASK_SEED,
)
def _make_patient_summary(
    patient_id: str,
    condition_code: str,
    severity: str = "P1",
    source_hospital: str = "H01",
    icu_need_step: int = 1,
    phase: str = "transferred",
    weighted_reward: float = 0.70,
) -> Dict[str, Any]:
    return {
        "patient_id":      patient_id,
        "condition_code":  condition_code,
        "severity":        severity,
        "source_hospital": source_hospital,
        "hospital_id":     source_hospital,
        "icu_need_step":   icu_need_step,
        "phase_at_episode_end": phase,
        "weighted_reward": weighted_reward,
    }
def _make_transfer_action(
    step: int,
    patient_id: str,
    condition_code: str,
    source: str,
    dest: str,
    unit_type: str = "MICU",
    delay_min: float = 25.0,
    transit_min: float = 20.0,
    pre_alert: bool = True,
    status: str = "completed",
    cath_lab: bool = False,
    neurosurg: bool = False,
    specialist: bool = True,
) -> ActionLogEntry:
    return ActionLogEntry(
        step=step,
        action_type="transfer",
        action_data={
            "patient_id":            patient_id,
            "condition_code":        condition_code,
            "source_hospital":       source,
            "dest_hospital":         dest,
            "unit_type":             unit_type,
            "delay_min":             delay_min,
            "transit_time_min":      transit_min,
            "pre_alert_given":       pre_alert,
            "status":                status,
            "cath_lab_activated_dest": cath_lab,
            "neurosurg_pre_alerted": neurosurg,
            "specialist_notified":   specialist,
        },
    )
def _build_transfer_gi(
    patient_summaries: List[Dict[str, Any]],
    transfer_actions:  List[ActionLogEntry],
    hospital_states:   Optional[Dict[str, Any]] = None,
    cascade_failure:   bool = False,
    episode_id:        str  = "ep-t8-test",
    episode_steps:     int  = 50,
    mutual_aid:        bool = False,
    surge_declared:    bool = False,
    comm_failures:     int  = 0,
    noop_on_unknown:   int  = 0,
) -> GraderInput:
    ledger: Dict[str, Any] = {
        "patient_summaries":   patient_summaries,
        "surge_declared":      surge_declared,
        "mutual_aid_requested":mutual_aid,
        "comm_failures":       comm_failures,
        "noop_on_unknown_units":noop_on_unknown,
    }
    fhs: Dict[str, Any] = {}
    if hospital_states:
        fhs = hospital_states
    else:
        for hid, cap in HOSPITAL_ICU_CAPACITY.items():
            fhs[hid] = {
                "icu_occupied":      int(cap * 0.60),
                "saturation_events": 0,
                "on_diversion":      False,
            }
    return GraderInput(
        task_id=TASK_ID,
        episode_id=episode_id,
        seed=TASK_SEED,
        action_log=transfer_actions,
        episode_ledger=ledger,
        observation_log=[],
        episode_steps=episode_steps,
        total_patients=len(patient_summaries),
        p1_patients=sum(1 for s in patient_summaries if s.get("severity") == "P1"),
        final_hospital_state=fhs,
        cascade_failure_occurred=cascade_failure,
    )
def _run_self_tests() -> None:
    failures: List[str] = []
    def chk(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg or 'assertion false'}")
    grader = Task8Grader()
    patients_perfect = [
        _make_patient_summary("P001", "stemi", "P1", "H02", 1, "transferred", 0.90),
        _make_patient_summary("P002", "stroke_ischaemic", "P1", "H04", 2, "transferred", 0.85),
        _make_patient_summary("P003", "tbi_severe", "P1", "H06", 3, "transferred", 0.80),
    ]
    actions_perfect = [
        _make_transfer_action(2, "P001", "stemi",           "H02", "H01",
                              "MICU", 20, 15, True,  "completed", True, False, True),
        _make_transfer_action(3, "P002", "stroke_ischaemic","H04", "H01",
                              "MICU", 25, 20, True,  "completed", False, False, True),
        _make_transfer_action(4, "P003", "tbi_severe",      "H06", "H03",
                              "MICU", 30, 25, True,  "completed", False, True, True),
    ]
    gi1  = _build_transfer_gi(patients_perfect, actions_perfect, episode_id="ep-t8-001")
    r1   = grader.grade(gi1)
    chk("T1_range",    0.0 <= r1.final_score <= 1.0)
    chk("T1_above_baseline", r1.final_score >= TASK_BASELINE,
        f"Perfect scenario {r1.final_score:.4f} should be ≥ baseline {TASK_BASELINE}")
    chk("T1_status",   r1.status in (GraderStatus.SUCCESS,), f"status={r1.status}")
    actions_wrong = [
        _make_transfer_action(2, "P001", "stemi",           "H02", "H06",
                              "MICU", 20, 15, True, "completed", False, False, False),
        _make_transfer_action(3, "P002", "stroke_ischaemic","H04", "H08",
                              "ALS",  25, 20, True, "completed", False, False, False),
    ]
    gi2 = _build_transfer_gi(patients_perfect[:2], actions_wrong, episode_id="ep-t8-002")
    r2  = grader.grade(gi2)
    chk("T2_range",    0.0 <= r2.final_score <= 1.0)
    chk("T2_lower_than_perfect", r2.final_score < r1.final_score,
        f"Wrong specialty {r2.final_score:.4f} should be < perfect {r1.final_score:.4f}")
    chk("T2_penalty_specialty",
        any("wrong_specialty" in p.name for p in r2.penalties),
        "Missing wrong_specialty penalty")
    actions_bls = [
        _make_transfer_action(2, "P001", "stemi", "H02", "H01",
                              "BLS", 20, 15, True, "completed", False, False, False),
    ]
    gi3 = _build_transfer_gi(patients_perfect[:1], actions_bls, episode_id="ep-t8-003")
    r3  = grader.grade(gi3)
    chk("T3_range",   0.0 <= r3.final_score <= 1.0)
    chk("T3_penalty_bls",
        any("critical_in_bls" in p.name for p in r3.penalties),
        "Missing critical_in_bls penalty")
    chk("T3_lower_than_perfect", r3.final_score < r1.final_score,
        f"BLS-for-critical {r3.final_score:.4f} < perfect {r1.final_score:.4f}")
    gi4 = _build_transfer_gi(
        patients_perfect, actions_perfect,
        cascade_failure=True,
        hospital_states={
            "H01": {"icu_occupied": 40, "saturation_events": 3, "on_diversion": True},
            "H03": {"icu_occupied": 24, "saturation_events": 1, "on_diversion": True},
            **{hid: {"icu_occupied": 0, "saturation_events": 0, "on_diversion": False}
               for hid in HOSPITAL_ICU_CAPACITY if hid not in ("H01", "H03")},
        },
        episode_id="ep-t8-004",
    )
    r4 = grader.grade(gi4)
    chk("T4_range",   0.0 <= r4.final_score <= 1.0)
    chk("T4_cascade_penalty",
        any("cascade_failure" in p.name for p in r4.penalties),
        "Missing cascade_failure penalty")
    chk("T4_lower_cascade", r4.final_score < r1.final_score,
        f"Cascade {r4.final_score:.4f} should be < perfect {r1.final_score:.4f}")
    div_hospital_state = {
        hid: {"icu_occupied": int(cap * 0.60), "saturation_events": 0, "on_diversion": False}
        for hid, cap in HOSPITAL_ICU_CAPACITY.items()
    }
    div_hospital_state["H01"] = {
        "icu_occupied": 40, "saturation_events": 2, "on_diversion": True
    }
    actions_diverted = [
        _make_transfer_action(2, "P001", "stemi", "H02", "H01",   
                              "MICU", 20, 15, True, "completed", True, False, True),
    ]
    gi5 = _build_transfer_gi(
        patients_perfect[:1], actions_diverted,
        hospital_states=div_hospital_state,
        episode_id="ep-t8-005",
    )
    r5 = grader.grade(gi5)
    chk("T5_range",   0.0 <= r5.final_score <= 1.0)
    chk("T5_diverted_penalty",
        any("diverted_dest" in p.name for p in r5.penalties),
        "Missing diverted_dest penalty")
    actions_no_alert = [
        _make_transfer_action(2, "P001", "stemi", "H02", "H01",
                              "MICU", 20, 15, False, "completed", True, False, True),
    ]
    gi6 = _build_transfer_gi(patients_perfect[:1], actions_no_alert, episode_id="ep-t8-006")
    r6  = grader.grade(gi6)
    chk("T6_range",       0.0 <= r6.final_score <= 1.0)
    chk("T6_pre_alert_penalty",
        any("no_pre_alert" in p.name for p in r6.penalties),
        "Missing no_pre_alert penalty")
    actions_bonus = [
        _make_transfer_action(2, "P001", "stemi", "H02", "H01",
                              "MICU", 20, 15, True, "completed",
                              cath_lab=True, neurosurg=False, specialist=True),
    ]
    gi7 = _build_transfer_gi(patients_perfect[:1], actions_bonus, episode_id="ep-t8-007")
    r7  = grader.grade(gi7)
    chk("T7_range",         0.0 <= r7.final_score <= 1.0)
    chk("T7_protocol_bonus",
        r7.extra.get("protocol_bonus", 0.0) > 0,
        f"Expected protocol_bonus > 0, got {r7.extra.get('protocol_bonus')}")
    gi8 = _build_transfer_gi(patients_perfect, [], episode_id="ep-t8-008")
    r8  = grader.grade(gi8)
    chk("T8_range",        0.0 <= r8.final_score <= 1.0)
    chk("T8_partial_status", r8.status == GraderStatus.PARTIAL,
        f"Expected PARTIAL when no transfers, got {r8.status}")
    chk("T8_low_score",    r8.final_score < 0.5,
        f"No-transfer score {r8.final_score:.4f} should be < 0.5")
    gi9 = GraderInput(
        task_id=TASK_ID, episode_id="ep-t8-009", seed=TASK_SEED,
        action_log=[], episode_ledger={"patient_summaries": []},
        observation_log=[], episode_steps=0,
        total_patients=0, p1_patients=0,
    )
    r9 = grader.grade(gi9)
    chk("T9_invalid_input", r9.status == GraderStatus.INVALID_INPUT,
        f"Expected INVALID_INPUT for empty summaries, got {r9.status}")
    big_patients = [
        _make_patient_summary(f"P{i:03d}",
                              ["stemi","stroke_ischaemic","tbi_severe",
                               "polytrauma","septic_shock"][i % 5],
                              "P1" if i % 3 != 0 else "P2",
                              ["H01","H02","H03","H04"][i % 4],
                              i, "transferred", 0.65)
        for i in range(1, 13)
    ]
    big_actions = [
        _make_transfer_action(
            i, f"P{i:03d}",
            ["stemi","stroke_ischaemic","tbi_severe","polytrauma","septic_shock"][i % 5],
            ["H01","H02","H03","H04"][i % 4],
            ["H01","H03","H07","H01","H05"][i % 5],
            "MICU" if i % 3 != 0 else "ALS",
            15 + (i * 3) % 40, 20, True, "completed",
            cath_lab=(i % 5 == 0), neurosurg=(i % 5 == 2), specialist=True
        )
        for i in range(1, 13)
    ]
    gi10 = _build_transfer_gi(big_patients, big_actions, episode_id="ep-t8-010")
    r10  = grader.grade(gi10)
    chk("T10_range", 0.0 <= r10.final_score <= 1.0)
    chk("T10_n_patients", r10.extra.get("n_patients", 0) == 12,
        f"Expected 12 patients, got {r10.extra.get('n_patients')}")
    sat_hospital_state = {
        hid: {"icu_occupied": int(cap * 0.60), "saturation_events": 0, "on_diversion": False}
        for hid, cap in HOSPITAL_ICU_CAPACITY.items()
    }
    sat_hospital_state["H01"] = {
        "icu_occupied": 39, "saturation_events": 3, "on_diversion": False
    }
    gi11 = _build_transfer_gi(
        patients_perfect, actions_perfect,
        hospital_states=sat_hospital_state,
        episode_id="ep-t8-011",
    )
    r11 = grader.grade(gi11)
    chk("T11_range",   0.0 <= r11.final_score <= 1.0)
    chk("T11_sat_penalty",
        any("icu_saturation" in p.name for p in r11.penalties),
        "Missing icu_saturation penalty for H01")
    gi12 = _build_transfer_gi(
        patients_perfect, actions_perfect,
        episode_id="ep-t8-012",
        comm_failures=4, noop_on_unknown=3,
    )
    r12 = grader.grade(gi12)
    chk("T12_range",   0.0 <= r12.final_score <= 1.0)
    chk("T12_comm_noop",
        any("comm_noop" in p.name for p in r12.penalties),
        "Missing comm_noop penalty for noop on unknown units")
    gi13a = _build_transfer_gi(patients_perfect, actions_perfect, episode_id="ep-det-a")
    gi13b = _build_transfer_gi(patients_perfect, actions_perfect, episode_id="ep-det-b")
    r13a  = grader.grade(gi13a)
    r13b  = grader.grade(gi13b)
    chk("T13_determinism",
        abs(r13a.final_score - r13b.final_score) < 1e-9,
        f"Non-deterministic: {r13a.final_score:.6f} vs {r13b.final_score:.6f}")
    d = r1.as_dict()
    chk("T14_dict_keys",
        all(k in d for k in ("final_score","components","penalties","notes","extra")))
    chk("T14_json_valid", len(r1.as_json()) > 300)
    chk("T14_summary_line", TASK_ID in r1.summary_line())
    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task8Grader self-test: {len(failures)} failure(s):\n" +
            "\n".join(failures)
        )
    logger.info(
        "Task8Grader self-test PASSED (%d test cases, "
        "speciality/timing/cascade/utilisation/protocol scoring verified).",
        14,
    )
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task8Grader self-test FAILED at import: %s", _e)
    raise
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 76)
    print("EMERGI-ENV  ·  Task8Grader  ·  Inter-Hospital Transfer Cascade Demo")
    print("=" * 76)
    grader = Task8Grader()
    def run(name: str, patients: List[Dict[str, Any]],
            actions: List[ActionLogEntry], **kw: Any) -> GraderResult:
        gi  = _build_transfer_gi(patients, actions, **kw)
        res = grader.grade(gi)
        beat = "✓" if res.beats_baseline else "✗"
        print(f"\n  [{beat}] {name}")
        print(f"       Score={res.final_score:.4f}  base={TASK_BASELINE:.2f}  "
              f"Δ={res.score_delta_vs_baseline:+.4f}  status={res.status.value}")
        print(f"       Patients={res.total_patients} | "
              f"Transfers={res.extra.get('n_transfers',0)} | "
              f"Completed={res.extra.get('n_completed',0)} | "
              f"Cascade={'YES' if res.extra.get('cascade_failure') else 'no'}")
        print(f"       ProtoBonus={res.extra.get('protocol_bonus',0):.3f}  "
              f"SatEvents={res.extra.get('saturation_events_total',0)}")
        for c in res.components:
            if c.name == "protocol_bonus":
                continue
            bar = "█" * int(c.raw_score * 20)
            print(f"         {c.name:<30} {c.raw_score:.4f} × {c.weight:.2f} "
                  f"= {c.weighted:.4f}  {bar}")
        if res.penalties:
            print(f"       Penalties ({len(res.penalties)}):")
            for p in res.penalties[:5]:
                print(f"         {p.name:<40} {p.amount:+.4f}  {p.reason[:45]}")
        return res
    PATIENTS_S1 = [
        _make_patient_summary("P001", "stemi",            "P1", "H02", 1, "transferred", 0.92),
        _make_patient_summary("P002", "stroke_ischaemic", "P1", "H04", 2, "transferred", 0.86),
        _make_patient_summary("P003", "tbi_severe",       "P1", "H06", 3, "transferred", 0.78),
        _make_patient_summary("P004", "polytrauma",       "P1", "H08", 4, "transferred", 0.82),
        _make_patient_summary("P005", "renal_failure",    "P2", "H03", 5, "transferred", 0.65),
    ]
    ACTIONS_PERFECT = [
        _make_transfer_action(2,  "P001","stemi",            "H02","H01","MICU",20,15,True,"completed",True, False,True),
        _make_transfer_action(3,  "P002","stroke_ischaemic", "H04","H01","MICU",25,20,True,"completed",False,False,True),
        _make_transfer_action(4,  "P003","tbi_severe",       "H06","H03","MICU",30,25,True,"completed",False,True, True),
        _make_transfer_action(5,  "P004","polytrauma",       "H08","H07","MICU",35,30,True,"completed",False,False,True),
        _make_transfer_action(6,  "P005","renal_failure",    "H03","H01","ALS", 60,25,True,"completed",False,False,True),
    ]
    ACTIONS_POOR = [
        _make_transfer_action(2, "P001","stemi",            "H02","H06","BLS", 80,15,False,"completed",False,False,False),
        _make_transfer_action(3, "P002","stroke_ischaemic", "H04","H08","BLS", 90,20,False,"completed",False,False,False),
        _make_transfer_action(4, "P003","tbi_severe",       "H06","H08","ALS",120,25,False,"completed",False,False,False),
    ]
    DEAD_CASCADE_HSP = {
        "H01": {"icu_occupied":40,"saturation_events":4,"on_diversion":True},
        "H03": {"icu_occupied":25,"saturation_events":2,"on_diversion":True},
        "H07": {"icu_occupied":35,"saturation_events":1,"on_diversion":True},
        **{hid: {"icu_occupied":0,"saturation_events":0,"on_diversion":False}
           for hid in HOSPITAL_ICU_CAPACITY if hid not in ("H01","H03","H07")},
    }
    scenarios = [
        (
            "Perfect: correct specialty, MICU, pre-alerts, cath-lab activated",
            PATIENTS_S1, ACTIONS_PERFECT,
            {"surge_declared": False, "mutual_aid": False},
        ),
        (
            "Poor: wrong hospitals, BLS for critical, no pre-alerts",
            PATIENTS_S1[:3], ACTIONS_POOR,
            {},
        ),
        (
            "Cascade failure: 3 hospitals saturated/diverted",
            PATIENTS_S1, ACTIONS_PERFECT,
            {"cascade_failure": True, "hospital_states": DEAD_CASCADE_HSP},
        ),
        (
            "Partial: only 2 of 5 patients transferred",
            PATIENTS_S1, ACTIONS_PERFECT[:2],
            {},
        ),
        (
            "Comm failures + noop penalty",
            PATIENTS_S1, ACTIONS_PERFECT,
            {"comm_failures": 5, "noop_on_unknown": 4},
        ),
    ]
    results_all = []
    for name, pts, acts, kw in scenarios:
        results_all.append(run(name, pts, acts, **kw))
    print("\n" + "=" * 76)
    beats = sum(1 for r in results_all if r.beats_baseline)
    print(f"  {beats}/{len(results_all)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 76)
    print("\n✅  Task8Grader demo complete.")