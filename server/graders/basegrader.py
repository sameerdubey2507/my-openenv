from __future__ import annotations
import abc
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)
logger = logging.getLogger("emergi_env.graders.base")
BASE_GRADER_VERSION: int = 2
SCORE_FLOOR:   float = 0.0
SCORE_CEILING: float = 1.0
TASK_BASELINES: Dict[str, float] = {
    "task1_single_triage":    0.61,
    "task2_hospital_route":   0.72,
    "task3_unit_type":        0.68,
    "task4_multi_incident":   0.44,
    "task5_dynamic_rerouting":0.38,
    "task6_prepositioning":   0.42,
    "task7_mci_start":        0.29,
    "task8_transfer_cascade": 0.24,
    "task9_surge":            0.17,
}
TASK_SEEDS: Dict[str, int] = {
    "task1_single_triage":    42,
    "task2_hospital_route":   43,
    "task3_unit_type":        44,
    "task4_multi_incident":   45,
    "task5_dynamic_rerouting":46,
    "task6_prepositioning":   47,
    "task7_mci_start":        48,
    "task8_transfer_cascade": 49,
    "task9_surge":            50,
}
MAX_PENALTY_FRACTION: float = 0.50  
MIN_COMPONENT_SCORE:  float = 0.0
MAX_COMPONENT_SCORE:  float = 1.0
GRADER_TIMEOUT_SECONDS: float = 5.0  
@unique
class GraderStatus(str, Enum):
    SUCCESS       = "success"
    PARTIAL       = "partial"       
    FAILED        = "failed"        
    TIMEOUT       = "timeout"       
    INVALID_INPUT = "invalid_input" 
@dataclass(frozen=True)
class ScoreComponent:
    name:        str
    raw_score:   float            
    weight:      float            
    weighted:    float            
    max_possible: float = 1.0    
    notes:       str   = ""       
    def __post_init__(self) -> None:
        object.__setattr__(self, "weighted", self.raw_score * self.weight)
    def as_dict(self) -> Dict[str, Any]:
        return {
            "name":         self.name,
            "raw_score":    round(self.raw_score, 4),
            "weight":       round(self.weight, 4),
            "weighted":     round(self.weighted, 4),
            "max_possible": round(self.max_possible, 4),
            "notes":        self.notes,
        }
@dataclass(frozen=True)
class PenaltyRecord:
    name:        str
    amount:      float     
    reason:      str
    rule_ref:    str = ""  
    def as_dict(self) -> Dict[str, Any]:
        return {
            "name":     self.name,
            "amount":   round(self.amount, 4),
            "reason":   self.reason,
            "rule_ref": self.rule_ref,
        }
@dataclass
class GraderResult:
    grader_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id:      str = ""
    episode_id:   str = ""
    seed:         int = 0
    graded_at_ts: float = field(default_factory=time.time)
    final_score:  float = 0.0     
    raw_score:    float = 0.0     
    baseline:     float = 0.0     
    beats_baseline: bool = False
    components:   List[ScoreComponent] = field(default_factory=list)
    penalties:    List[PenaltyRecord]  = field(default_factory=list)
    total_penalty: float = 0.0
    status:       GraderStatus = GraderStatus.SUCCESS
    error_message: Optional[str] = None
    grading_time_ms: float = 0.0
    task_difficulty:    str  = ""
    episode_steps:      int  = 0
    total_patients:     int  = 0
    p1_patients:        int  = 0
    p1_survival_rate:   float = 0.0
    protocol_violations: int = 0
    critical_mismatches: int = 0
    notes:              List[str] = field(default_factory=list)
    extra:              Dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""   
    def add_component(self, component: ScoreComponent) -> None:
        self.components.append(component)
    def add_penalty(self, penalty: PenaltyRecord) -> None:
        self.penalties.append(penalty)
        self.total_penalty += penalty.amount
    def add_note(self, note: str) -> None:
        self.notes.append(note)
    def finalise(self) -> "GraderResult":
        component_total = sum(c.weighted for c in self.components)
        self.raw_score  = component_total + self.total_penalty
        self.final_score = float(max(SCORE_FLOOR, min(SCORE_CEILING, self.raw_score)))
        self.beats_baseline = self.final_score > self.baseline
        return self
    @property
    def component_breakdown(self) -> Dict[str, float]:
        return {c.name: round(c.weighted, 4) for c in self.components}
    @property
    def score_delta_vs_baseline(self) -> float:
        return round(self.final_score - self.baseline, 4)
    def as_dict(self) -> Dict[str, Any]:
        return {
            "grader_id":          self.grader_id,
            "task_id":            self.task_id,
            "episode_id":         self.episode_id,
            "seed":               self.seed,
            "final_score":        round(self.final_score, 4),
            "raw_score":          round(self.raw_score, 4),
            "baseline":           round(self.baseline, 4),
            "beats_baseline":     self.beats_baseline,
            "delta_vs_baseline":  self.score_delta_vs_baseline,
            "status":             self.status.value,
            "error_message":      self.error_message,
            "grading_time_ms":    round(self.grading_time_ms, 2),
            "task_difficulty":    self.task_difficulty,
            "episode_steps":      self.episode_steps,
            "total_patients":     self.total_patients,
            "p1_patients":        self.p1_patients,
            "p1_survival_rate":   round(self.p1_survival_rate, 4),
            "protocol_violations":self.protocol_violations,
            "critical_mismatches":self.critical_mismatches,
            "total_penalty":      round(self.total_penalty, 4),
            "components":         [c.as_dict() for c in self.components],
            "penalties":          [p.as_dict() for p in self.penalties],
            "notes":              self.notes,
            "input_hash":         self.input_hash,
            "extra":              self.extra,
        }
    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)
    def summary_line(self) -> str:
        baseline_str = "✓ BEATS" if self.beats_baseline else "✗ BELOW"
        return (
            f"[{self.task_id}] score={self.final_score:.4f} "
            f"baseline={self.baseline:.2f} {baseline_str} "
            f"(Δ={self.score_delta_vs_baseline:+.4f}) "
            f"status={self.status.value} "
            f"t={self.grading_time_ms:.1f}ms"
        )
class ScoringUtils:
    @staticmethod
    def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(max(lo, min(hi, value)))
    @staticmethod
    def weighted_sum(
        scores:  Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(
                "ScoringUtils.weighted_sum: weights sum to %.4f (expected 1.0). "
                "Normalising.", weight_sum
            )
            weights = {k: v / weight_sum for k, v in weights.items()}
        total = 0.0
        for k, w in weights.items():
            total += w * scores.get(k, 0.0)
        return ScoringUtils.clamp(total)
    @staticmethod
    def response_time_score(
        actual_min:     float,
        target_min:     float,
        worst_case_min: float = 120.0,
        shape:          str   = "linear",
    ) -> float:
        if actual_min <= 0:
            return 1.0
        if actual_min <= target_min:
            return 1.0
        if actual_min >= worst_case_min:
            return 0.0
        excess = actual_min - target_min
        span   = worst_case_min - target_min
        if shape == "exponential":
            return float(math.exp(-3.0 * excess / span))
        if shape == "step":
            return 0.0 if excess > span * 0.5 else 0.5
        return ScoringUtils.clamp(1.0 - excess / span)
    @staticmethod
    def unit_type_score(
        dispatched:     str,
        required_types: Tuple[str, ...],
        severity:       str = "P1",
    ) -> float:
        unit_rank = {"BLS": 1, "ALS": 2, "MICU": 3}
        d_rank = unit_rank.get(dispatched.upper(), 0)
        if d_rank == 0:
            return 0.0
        if dispatched.upper() in [u.upper() for u in required_types]:
            return 1.0
        if not required_types:
            return 1.0
        correct_ranks = [unit_rank.get(u.upper(), 1) for u in required_types]
        min_correct = min(correct_ranks)
        max_correct = max(correct_ranks)
        if d_rank > max_correct:
            return 0.85   
        gap = min_correct - d_rank
        if gap == 1:
            return 0.55
        if gap == 2:
            return 0.20   
        return 0.0
    @staticmethod
    def triage_accuracy_score(
        assigned_tag:    str,
        ground_truth_tag: str,
    ) -> float:
        tag_order = {"Immediate": 3, "Delayed": 2, "Minimal": 1, "Expectant": 0}
        a = tag_order.get(assigned_tag, -1)
        g = tag_order.get(ground_truth_tag, -1)
        if a == g:
            return 1.0
        if ground_truth_tag == "Immediate" and assigned_tag == "Expectant":
            return 0.0
        diff = abs(a - g)
        if diff == 1:
            return 0.50
        if diff == 2:
            return 0.20
        return 0.10
    @staticmethod
    def hospital_specialty_score(
        has_specialty:    bool,
        on_diversion:     bool,
        er_occupancy_pct: float,
    ) -> float:
        if on_diversion:
            return 0.0
        if not has_specialty:
            return 0.20
        if er_occupancy_pct >= 0.90:
            return 0.35
        if er_occupancy_pct >= 0.80:
            return 0.70
        return 1.0
    @staticmethod
    def survival_probability_score(
        p_actual:    float,
        p_optimal:   float,
        p_worst:     float = 0.05,
    ) -> float:
        if p_optimal <= p_worst:
            return 1.0
        span = p_optimal - p_worst
        achieved = p_actual - p_worst
        return ScoringUtils.clamp(achieved / span)
    @staticmethod
    def compute_input_hash(data: Any) -> str:
        try:
            serialised = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(serialised.encode()).hexdigest()[:16]
        except Exception:
            return "hash_error"
    @staticmethod
    def validate_weights(weights: Dict[str, float], tolerance: float = 1e-4) -> bool:
        total = sum(weights.values())
        return abs(total - 1.0) < tolerance
    @staticmethod
    def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}
    @staticmethod
    def linear_decay_score(
        value:    float,
        best:     float,
        worst:    float,
    ) -> float:
        if best == worst:
            return 1.0
        if best < worst:
            if value <= best:
                return 1.0
            if value >= worst:
                return 0.0
            return ScoringUtils.clamp(1.0 - (value - best) / (worst - best))
        if value >= best:
            return 1.0
        if value <= worst:
            return 0.0
        return ScoringUtils.clamp((value - worst) / (best - worst))
    @staticmethod
    def exponential_decay_score(
        elapsed_min: float,
        lambda_:     float = 0.02,
        floor:       float = 0.0,
    ) -> float:
        p = math.exp(-lambda_ * elapsed_min)
        return ScoringUtils.clamp(max(floor, p))
    @staticmethod
    def boolean_score(
        condition: bool,
        if_true:   float = 1.0,
        if_false:  float = 0.0,
    ) -> float:
        return if_true if condition else if_false
    @staticmethod
    def count_score(
        actual:   int,
        target:   int,
        penalty_per_miss: float = 0.10,
    ) -> float:
        if target == 0:
            return 1.0
        miss = max(0, target - actual)
        return ScoringUtils.clamp(1.0 - miss * penalty_per_miss)
    @staticmethod
    def build_score_component(
        name:      str,
        score:     float,
        weight:    float,
        notes:     str = "",
    ) -> ScoreComponent:
        return ScoreComponent(
            name=name,
            raw_score=ScoringUtils.clamp(score),
            weight=weight,
            weighted=ScoringUtils.clamp(score) * weight,
            notes=notes,
        )
    @staticmethod
    def build_penalty(
        name:     str,
        amount:   float,
        reason:   str,
        rule_ref: str = "",
    ) -> PenaltyRecord:
        return PenaltyRecord(
            name=name,
            amount=-abs(amount),
            reason=reason,
            rule_ref=rule_ref,
        )
@dataclass
class ActionLogEntry:
    step:         int
    action_type:  str
    action_data:  Dict[str, Any]
    observation_snapshot: Optional[Dict[str, Any]] = None
    def get(self, key: str, default: Any = None) -> Any:
        return self.action_data.get(key, default)
@dataclass
class GraderInput:
    task_id:        str
    episode_id:     str
    seed:           int
    action_log:     List[ActionLogEntry]
    episode_ledger: Dict[str, Any]         
    observation_log: List[Dict[str, Any]]  
    episode_steps:      int
    total_patients:     int
    p1_patients:        int
    final_fleet_state:  Optional[Dict[str, Any]] = None
    final_hospital_state: Optional[Dict[str, Any]] = None
    mci_coordinator_dumps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cascade_failure_occurred: bool = False
    demand_forecast_snapshot: Optional[Dict[str, Any]] = None
    def get_actions_by_type(self, action_type: str) -> List[ActionLogEntry]:
        return [a for a in self.action_log if a.action_type == action_type]
    def get_patient_summary(self, patient_id: str) -> Optional[Dict[str, Any]]:
        summaries = self.episode_ledger.get("patient_summaries", [])
        for s in summaries:
            if s.get("patient_id") == patient_id:
                return s
        return None
    def all_patient_summaries(self) -> List[Dict[str, Any]]:
        return self.episode_ledger.get("patient_summaries", [])
    def p1_summaries(self) -> List[Dict[str, Any]]:
        return [
            s for s in self.all_patient_summaries()
            if s.get("severity") == "P1"
        ]
class ProtocolRuleChecker:
    @staticmethod
    def check_micu_for_stemi(
        dispatched_unit: str,
        condition_key:   str,
    ) -> Tuple[bool, float, str]:
        stemi_conditions = {
            "stemi_anterior", "stemi_inferior", "stemi_posterior",
            "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
        }
        if condition_key not in stemi_conditions:
            return True, 0.0, "n/a"
        compliant = dispatched_unit.upper() == "MICU"
        if compliant:
            return True, 0.0, "MICU dispatched for STEMI — correct"
        return False, -0.020, f"STEMI requires MICU; dispatched {dispatched_unit}"
    @staticmethod
    def check_als_for_stroke(
        dispatched_unit: str,
        condition_key:   str,
    ) -> Tuple[bool, float, str]:
        stroke_conditions = {
            "ischemic_stroke", "ischemic_stroke_wake_up",
            "hemorrhagic_stroke_sah", "paediatric_stroke",
        }
        if condition_key not in stroke_conditions:
            return True, 0.0, "n/a"
        compliant = dispatched_unit.upper() in ("ALS", "MICU")
        if compliant:
            return True, 0.0, "ALS/MICU for stroke — correct"
        return False, -0.015, f"Stroke requires ALS minimum; dispatched {dispatched_unit}"
    @staticmethod
    def check_no_routing_to_diverted(
        hospital_on_diversion: bool,
        hospital_id:           str,
    ) -> Tuple[bool, float, str]:
        if not hospital_on_diversion:
            return True, 0.0, "hospital accepting"
        return (
            False, -0.030,
            f"Hospital {hospital_id} is on diversion — routing violation"
        )
    @staticmethod
    def check_cath_lab_activation(
        cath_lab_activated: bool,
        condition_key:      str,
    ) -> Tuple[bool, float, str]:
        stemi_conditions = {
            "stemi_anterior", "stemi_inferior", "stemi_posterior",
            "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
            "cardiac_arrest_vf", "cardiac_arrest_pea",
        }
        if condition_key not in stemi_conditions:
            return True, 0.0, "n/a"
        if cath_lab_activated:
            return True, 0.018, "cath lab pre-activated — protocol bonus"
        return False, -0.022, "STEMI cath lab pre-activation missing"
    @staticmethod
    def check_stroke_unit_prenotification(
        stroke_unit_notified: bool,
        condition_key:        str,
    ) -> Tuple[bool, float, str]:
        stroke_conditions = {
            "ischemic_stroke", "ischemic_stroke_wake_up",
            "hemorrhagic_stroke_sah", "paediatric_stroke",
        }
        if condition_key not in stroke_conditions:
            return True, 0.0, "n/a"
        if stroke_unit_notified:
            return True, 0.015, "stroke unit pre-notified — protocol bonus"
        return False, -0.028, "Stroke unit pre-notification missing"
    @staticmethod
    def check_multi_agency_for_trapped(
        multi_agency_coordinated: bool,
        trapped_victim:           bool,
        condition_key:            str,
    ) -> Tuple[bool, float, str]:
        if not trapped_victim:
            return True, 0.0, "n/a"
        requires_multi = {
            "polytrauma_blunt", "mci_rta", "blast_injury",
            "mine_collapse", "crush_syndrome", "mci_natural_disaster",
        }
        if condition_key not in requires_multi:
            return True, 0.0, "n/a"
        if multi_agency_coordinated:
            return True, 0.015, "multi-agency coordinated — correct"
        return False, -0.025, "Trapped victim requires Police/Fire coordination"
    @staticmethod
    def check_level1_trauma_routing(
        hospital_is_level1:   bool,
        response_time_min:    float,
        condition_key:        str,
    ) -> Tuple[bool, float, str]:
        level1_conditions = {
            "polytrauma_blunt", "polytrauma_penetrating",
            "severe_tbi", "mci_rta", "blast_injury",
            "crush_syndrome", "mci_natural_disaster",
        }
        if condition_key not in level1_conditions:
            return True, 0.0, "n/a"
        if not hospital_is_level1:
            return False, -0.025, "Major trauma requires Level-1 trauma centre"
        if response_time_min <= 30.0:
            return True, 0.015, "Level-1 within 30 min — protocol met"
        return True, -0.010, f"Level-1 but response time {response_time_min:.1f} min > 30 min"
    @staticmethod
    def check_start_triage_in_mci(
        start_protocol_applied: bool,
        is_mci:                 bool,
    ) -> Tuple[bool, float, str]:
        if not is_mci:
            return True, 0.0, "n/a"
        if start_protocol_applied:
            return True, 0.020, "START triage protocol applied in MCI — bonus"
        return False, -0.035, "MCI requires START triage — protocol violation"
    @staticmethod
    def check_mutual_aid_in_surge(
        mutual_aid_requested: bool,
        surge_declared:       bool,
    ) -> Tuple[bool, float, str]:
        if not surge_declared:
            return True, 0.0, "n/a"
        if mutual_aid_requested:
            return True, 0.020, "Mutual aid requested in surge — correct"
        return False, -0.050, "Surge declared but no mutual aid requested"
    @classmethod
    def run_all_checks(
        cls,
        condition_key:            str,
        dispatched_unit:          str,
        hospital_on_diversion:    bool,
        hospital_id:              str,
        hospital_is_level1:       bool,
        response_time_min:        float,
        cath_lab_activated:       bool,
        stroke_unit_notified:     bool,
        multi_agency_coordinated: bool,
        trapped_victim:           bool,
        is_mci:                   bool,
        start_protocol_applied:   bool,
        mutual_aid_requested:     bool,
        surge_declared:           bool,
    ) -> Tuple[float, List[PenaltyRecord], List[str]]:
        checks = [
            cls.check_micu_for_stemi(dispatched_unit, condition_key),
            cls.check_als_for_stroke(dispatched_unit, condition_key),
            cls.check_no_routing_to_diverted(hospital_on_diversion, hospital_id),
            cls.check_cath_lab_activation(cath_lab_activated, condition_key),
            cls.check_stroke_unit_prenotification(stroke_unit_notified, condition_key),
            cls.check_multi_agency_for_trapped(
                multi_agency_coordinated, trapped_victim, condition_key
            ),
            cls.check_level1_trauma_routing(
                hospital_is_level1, response_time_min, condition_key
            ),
            cls.check_start_triage_in_mci(start_protocol_applied, is_mci),
            cls.check_mutual_aid_in_surge(mutual_aid_requested, surge_declared),
        ]
        penalties:  List[PenaltyRecord] = []
        notes:      List[str]           = []
        net_score:  float               = 0.0
        for compliant, delta, reason in checks:
            if reason == "n/a":
                continue
            net_score += delta
            notes.append(reason)
            if delta < 0:
                penalties.append(PenaltyRecord(
                    name=f"protocol_{condition_key[:20]}",
                    amount=delta,
                    reason=reason,
                    rule_ref="EMERGI-ENV Protocol v2",
                ))
        return net_score, penalties, notes
class BaseGrader(abc.ABC):
    TASK_ID:        ClassVar[str]   = ""
    TASK_SEED:      ClassVar[int]   = 0
    TASK_BASELINE:  ClassVar[float] = 0.0
    TASK_DIFFICULTY: ClassVar[str]  = "easy"
    COMPONENT_WEIGHTS: ClassVar[Dict[str, float]] = {}
    def __init__(self) -> None:
        self._utils   = ScoringUtils()
        self._checker = ProtocolRuleChecker()
        self._logger  = logging.getLogger(
            f"emergi_env.graders.{self.__class__.__name__}"
        )
        self._validate_class_vars()
    def _validate_class_vars(self) -> None:
        assert self.TASK_ID, f"{self.__class__.__name__}: TASK_ID not set"
        assert self.TASK_SEED > 0, f"{self.__class__.__name__}: TASK_SEED not set"
        assert 0.0 <= self.TASK_BASELINE <= 1.0, f"Invalid TASK_BASELINE"
        if self.COMPONENT_WEIGHTS:
            assert ScoringUtils.validate_weights(self.COMPONENT_WEIGHTS), (
                f"{self.__class__.__name__}: COMPONENT_WEIGHTS do not sum to 1.0 "
                f"(got {sum(self.COMPONENT_WEIGHTS.values()):.4f})"
            )
    def grade(self, grader_input: GraderInput) -> GraderResult:
        start_ts = time.perf_counter()
        result   = self._make_result(grader_input)
        result.input_hash = ScoringUtils.compute_input_hash({
            "task_id":    grader_input.task_id,
            "episode_id": grader_input.episode_id,
            "seed":       grader_input.seed,
            "n_actions":  len(grader_input.action_log),
        })
        try:
            self._grade_impl(grader_input, result)
            result.finalise()
            if result.status == GraderStatus.SUCCESS:
                self._logger.info(result.summary_line())
        except Exception as exc:
            result.status        = GraderStatus.FAILED
            result.error_message = str(exc)
            result.final_score   = 0.0
            result.raw_score     = 0.0
            self._logger.exception(
                "Grader %s FAILED for episode %s: %s",
                self.TASK_ID, grader_input.episode_id, exc,
            )
        finally:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            result.grading_time_ms = round(elapsed_ms, 2)
            if elapsed_ms > GRADER_TIMEOUT_SECONDS * 1000:
                result.status = GraderStatus.TIMEOUT
                result.add_note(
                    f"Grader exceeded timeout ({elapsed_ms:.0f}ms > "
                    f"{GRADER_TIMEOUT_SECONDS*1000:.0f}ms)"
                )
        return result
    @abc.abstractmethod
    def _grade_impl(
        self,
        grader_input: GraderInput,
        result:       GraderResult,
    ) -> None:
        ...
    def _make_result(self, grader_input: GraderInput) -> GraderResult:
        return GraderResult(
            task_id=self.TASK_ID,
            episode_id=grader_input.episode_id,
            seed=self.TASK_SEED,
            baseline=self.TASK_BASELINE,
            task_difficulty=self.TASK_DIFFICULTY,
            episode_steps=grader_input.episode_steps,
            total_patients=grader_input.total_patients,
            p1_patients=grader_input.p1_patients,
        )
    def _add_component(
        self,
        result:  GraderResult,
        name:    str,
        score:   float,
        weight:  float,
        notes:   str = "",
    ) -> None:
        result.add_component(
            ScoringUtils.build_score_component(name, score, weight, notes)
        )
    def _add_penalty(
        self,
        result:   GraderResult,
        name:     str,
        amount:   float,
        reason:   str,
        rule_ref: str = "",
    ) -> None:
        result.add_penalty(
            ScoringUtils.build_penalty(name, amount, reason, rule_ref)
        )
    def _run_protocol_checks(
        self,
        result:   GraderResult,
        **kwargs: Any,
    ) -> float:
        defaults = {
            "condition_key": "unknown",
            "dispatched_unit": "BLS",
            "hospital_on_diversion": False,
            "hospital_id": "H01",
            "hospital_is_level1": False,
            "response_time_min": 999.0,
            "cath_lab_activated": False,
            "stroke_unit_notified": False,
            "multi_agency_coordinated": False,
            "trapped_victim": False,
            "is_mci": False,
            "start_protocol_applied": False,
            "mutual_aid_requested": False,
            "surge_declared": False,
        }
        defaults.update(kwargs)
        net_score, penalties, notes = ProtocolRuleChecker.run_all_checks(**defaults)
        for p in penalties:
            result.add_penalty(p)
        for n in notes:
            result.add_note(n)
        result.protocol_violations += len(penalties)
        return net_score
    def _get_first_dispatch(
        self,
        action_log: List[ActionLogEntry],
        incident_id: Optional[str] = None,
    ) -> Optional[ActionLogEntry]:
        for entry in action_log:
            if entry.action_type != "dispatch":
                continue
            if incident_id is None:
                return entry
            if entry.get("incident_id") == incident_id:
                return entry
        return None
    def _compute_weighted_patient_score(
        self,
        patient_summaries: List[Dict[str, Any]],
    ) -> float:
        if not patient_summaries:
            return 0.0
        total_weight  = 0.0
        weighted_sum  = 0.0
        for s in patient_summaries:
            w = float(s.get("weight", 1.0))
            r = float(s.get("weighted_reward", 0.0))
            weighted_sum  += w * r
            total_weight  += w
        if total_weight == 0:
            return 0.0
        return ScoringUtils.clamp(weighted_sum / total_weight)
    def _p1_survival_rate(self, patient_summaries: List[Dict[str, Any]]) -> float:
        p1 = [s for s in patient_summaries if s.get("severity") == "P1"]
        if not p1:
            return 1.0
        treated = [
            s for s in p1
            if s.get("phase_at_episode_end") == "treated"
        ]
        return len(treated) / len(p1)
    def _avg_dispatch_latency(
        self,
        patient_summaries: List[Dict[str, Any]],
    ) -> float:
        latencies = [
            float(s["dispatch_latency_min"])
            for s in patient_summaries
            if s.get("dispatch_latency_min") is not None
        ]
        if not latencies:
            return 999.0
        return sum(latencies) / len(latencies)
class GraderRegistry:
    _registry: ClassVar[Dict[str, Type[BaseGrader]]] = {}
    @classmethod
    def register(cls, task_id: str, grader_class: Type[BaseGrader]) -> None:
        cls._registry[task_id] = grader_class
        logger.debug("GraderRegistry: registered %s → %s", task_id, grader_class.__name__)
    @classmethod
    def get(cls, task_id: str) -> Type[BaseGrader]:
        if task_id not in cls._registry:
            raise KeyError(
                f"GraderRegistry: no grader for task_id '{task_id}'. "
                f"Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[task_id]
    @classmethod
    def get_instance(cls, task_id: str) -> BaseGrader:
        return cls.get(task_id)()
    @classmethod
    def all_task_ids(cls) -> List[str]:
        return sorted(cls._registry.keys())
    @classmethod
    def is_registered(cls, task_id: str) -> bool:
        return task_id in cls._registry
class GraderPipeline:
    def __init__(self, task_ids: Optional[List[str]] = None) -> None:
        self._task_ids = task_ids or list(TASK_BASELINES.keys())
    def run(
        self,
        inputs: Dict[str, GraderInput],
    ) -> Dict[str, GraderResult]:
        results: Dict[str, GraderResult] = {}
        for task_id in self._task_ids:
            if task_id not in inputs:
                logger.warning("GraderPipeline: no input for %s — skipping", task_id)
                continue
            if not GraderRegistry.is_registered(task_id):
                logger.warning("GraderPipeline: no grader for %s — skipping", task_id)
                continue
            grader = GraderRegistry.get_instance(task_id)
            result = grader.grade(inputs[task_id])
            results[task_id] = result
        return results
    def aggregate_score(
        self,
        results: Dict[str, GraderResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        if not results:
            return 0.0
        if weights is None:
            n = len(results)
            weights = {tid: 1.0 / n for tid in results}
        total = 0.0
        for tid, res in results.items():
            total += weights.get(tid, 0.0) * res.final_score
        return ScoringUtils.clamp(total)
    def summary_table(self, results: Dict[str, GraderResult]) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║         EMERGI-ENV GRADER RESULTS SUMMARY                        ║",
            "╠═══════════════════════╦════════╦══════════╦═══════════╦══════════╣",
            "║ Task                  ║  Score ║ Baseline ║    Δ      ║  Status  ║",
            "╠═══════════════════════╬════════╬══════════╬═══════════╬══════════╣",
        ]
        for tid in sorted(TASK_BASELINES.keys()):
            res = results.get(tid)
            if res is None:
                lines.append(
                    f"║ {tid:<21} ║  n/a   ║  {TASK_BASELINES[tid]:.2f}    ║  n/a      ║  SKIP    ║"
                )
                continue
            beat = "✓" if res.beats_baseline else "✗"
            lines.append(
                f"║ {tid:<21} ║ {res.final_score:.4f} ║  {res.baseline:.2f}    "
                f"║ {res.score_delta_vs_baseline:+.4f}   ║ {beat} {res.status.value:<6} ║"
            )
        lines.append(
            "╚═══════════════════════╩════════╩══════════╩═══════════╩══════════╝"
        )
        return "\n".join(lines)
def _self_test() -> None:
    assert ScoringUtils.clamp(1.5) == 1.0
    assert ScoringUtils.clamp(-0.5) == 0.0
    assert ScoringUtils.clamp(0.72) == 0.72
    assert ScoringUtils.response_time_score(5.0, 8.0) == 1.0
    assert ScoringUtils.response_time_score(8.0, 8.0) == 1.0
    assert ScoringUtils.response_time_score(120.0, 8.0, 120.0) == 0.0
    mid = ScoringUtils.response_time_score(64.0, 8.0, 120.0)
    assert 0.0 < mid < 1.0
    assert ScoringUtils.unit_type_score("MICU", ("MICU",)) == 1.0
    assert ScoringUtils.unit_type_score("ALS", ("MICU",)) < 0.6
    assert ScoringUtils.unit_type_score("BLS", ("MICU",)) == 0.20
    assert ScoringUtils.unit_type_score("MICU", ("BLS",)) == 0.85
    assert ScoringUtils.triage_accuracy_score("Immediate", "Immediate") == 1.0
    assert ScoringUtils.triage_accuracy_score("Expectant", "Immediate") == 0.0
    assert ScoringUtils.triage_accuracy_score("Delayed", "Immediate") < 1.0
    assert ScoringUtils.hospital_specialty_score(True, False, 0.60) == 1.0
    assert ScoringUtils.hospital_specialty_score(True, True, 0.60)  == 0.0
    assert ScoringUtils.hospital_specialty_score(False, False, 0.60) == 0.20
    weights = {"a": 0.4, "b": 0.3, "c": 0.3}
    assert ScoringUtils.validate_weights(weights)
    ws = ScoringUtils.weighted_sum({"a": 1.0, "b": 0.5, "c": 0.8}, weights)
    assert 0.0 <= ws <= 1.0
    r = GraderResult(
        task_id="task1_single_triage", episode_id="ep-001",
        seed=42, baseline=0.61,
    )
    r.add_component(ScoringUtils.build_score_component("triage", 0.8, 0.4, "test"))
    r.add_component(ScoringUtils.build_score_component("unit_type", 1.0, 0.3, "test"))
    r.add_component(ScoringUtils.build_score_component("hospital", 0.9, 0.3, "test"))
    r.add_penalty(ScoringUtils.build_penalty("test_penalty", 0.05, "test", "rule1"))
    r.finalise()
    assert 0.0 <= r.final_score <= 1.0
    assert r.final_score > r.baseline   
    assert len(r.components) == 3
    assert len(r.penalties) == 1
    assert r.total_penalty < 0.0
    summary = r.summary_line()
    assert "task1_single_triage" in summary
    d = r.as_dict()
    assert d["final_score"] == round(r.final_score, 4)
    ok, delta, reason = ProtocolRuleChecker.check_micu_for_stemi("MICU", "stemi_anterior")
    assert ok and delta == 0.0
    fail, delta2, reason2 = ProtocolRuleChecker.check_micu_for_stemi("BLS", "stemi_anterior")
    assert not fail and delta2 < 0
    ok3, _, _ = ProtocolRuleChecker.check_no_routing_to_diverted(True, "H01")
    assert not ok3
    ok4, bonus, _ = ProtocolRuleChecker.check_cath_lab_activation(True, "stemi_anterior")
    assert ok4 and bonus > 0
    assert isinstance(GraderRegistry._registry, dict)
    gi = GraderInput(
        task_id="task1_single_triage",
        episode_id="ep-001",
        seed=42,
        action_log=[
            ActionLogEntry(step=1, action_type="dispatch", action_data={
                "incident_id": "INC001",
                "unit_type": "MICU",
            })
        ],
        episode_ledger={
            "patient_summaries": [
                {"patient_id": "P001", "severity": "P1", "weighted_reward": 0.75,
                 "phase_at_episode_end": "treated", "dispatch_latency_min": 6.0}
            ]
        },
        observation_log=[],
        episode_steps=25,
        total_patients=1,
        p1_patients=1,
    )
    assert len(gi.get_actions_by_type("dispatch")) == 1
    assert len(gi.p1_summaries()) == 1
    ps = gi.get_patient_summary("P001")
    assert ps is not None
    logger.info(
        "base_grader.py self-test PASSED — "
        "ScoringUtils, GraderResult, ProtocolRuleChecker, "
        "GraderRegistry, GraderInput all verified."
    )
_self_test()
logger.info(
    "EMERGI-ENV server.graders.base_grader v%d loaded — "
    "%d task baselines, ScoringUtils + ProtocolRuleChecker + GraderRegistry ready.",
    BASE_GRADER_VERSION,
    len(TASK_BASELINES),
)
__all__ = [
    "BASE_GRADER_VERSION",
    "SCORE_FLOOR",
    "SCORE_CEILING",
    "TASK_BASELINES",
    "TASK_SEEDS",
    "GraderStatus",
    "ScoreComponent",
    "PenaltyRecord",
    "GraderResult",
    "ActionLogEntry",
    "GraderInput",
    "ScoringUtils",
    "ProtocolRuleChecker",
    "BaseGrader",
    "GraderRegistry",
    "GraderPipeline",
]