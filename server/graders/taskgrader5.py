from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
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
logger = logging.getLogger("emergi_env.graders.task5")
TASK_ID       = "task5_dynamic_rerouting"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]
W_CORRECTNESS = 0.50
W_TIME_SAVED  = 0.50
WC_TRIGGER_DETECTION = 0.30
WC_DESTINATION_QUAL  = 0.30
WC_TIMING_WINDOW     = 0.20
WC_FALSE_POSITIVE    = 0.10
WC_UNIT_SELECTION    = 0.10
assert abs(WC_TRIGGER_DETECTION + WC_DESTINATION_QUAL + WC_TIMING_WINDOW +
           WC_FALSE_POSITIVE + WC_UNIT_SELECTION - 1.0) < 1e-9, \
    "Correctness sub-weights must sum to 1.0"
WT_ETA_DELTA   = 0.50
WT_SURVIVAL    = 0.30
WT_CASCADE     = 0.20
assert abs(WT_ETA_DELTA + WT_SURVIVAL + WT_CASCADE - 1.0) < 1e-9, \
    "Time-saved sub-weights must sum to 1.0"
HP_DIVERTED_DESTINATION    = 0.35
HP_MISSED_REROUTE_P1       = 0.25
HP_NO_SPECIALTY_REROUTE    = 0.20
HP_UNIT_REROUTED_ON_SCENE  = 0.20
HP_NEAR_FULL_HOSPITAL      = 0.15
HP_LATE_HANDOVER_REROUTE   = 0.10
PB_PRE_ALERT               = 0.025
PB_TRAUMA_CENTRE_30MIN     = 0.020
PB_SPECIALTY_MATCH_FAST    = 0.015
PB_CASCADE_PREEMPTION      = 0.030
PB_FIRST_TRY_CORRECT       = 0.020
PB_MUTUAL_AID_CORRECT      = 0.010
PROTOCOL_BONUS_CAP         = 0.15
REROUTE_OPTIMAL_STEPS    = 2     
REROUTE_GOOD_STEPS       = 4     
REROUTE_LATE_STEPS       = 7     
DIVERSION_PENALTY_MINUTES   = 12.0   
CONGESTION_EXTRA_FACTOR_MAX = 0.60   
ER_OVERCAPACITY_HANDOVER_EXTRA = 15.0  
DEST_QUALITY_BANDS: List[Tuple[float, float, float]] = [
    (0.85, 1.01, 1.00),
    (0.70, 0.85, 0.80),
    (0.55, 0.70, 0.60),
    (0.40, 0.55, 0.40),
    (0.00, 0.40, 0.15),
]
TRAUMA_L1_CONDITIONS: Set[str] = {
    "polytrauma_blunt", "polytrauma_penetrating", "severe_tbi",
    "blast_injury", "mci_rta", "crush_syndrome",
}
ER_DIVERSION_THRESHOLD = 0.90
ER_NEAR_SATURATION     = 0.85   
class RerouteTriggerType(str, Enum):
    HOSPITAL_DIVERSION      = "hospital_diversion"
    TRAFFIC_CONGESTION_SPIKE = "traffic_congestion_spike"
    SPECIALTY_CLOSURE       = "specialty_closure"
    ER_NEAR_SATURATION      = "er_near_saturation"
    CLOSER_HOSPITAL_FOUND   = "closer_hospital_found"
    PATIENT_CONDITION_UPGRADE = "patient_condition_upgrade"
    MUTUAL_AID_TRIGGERED    = "mutual_aid_triggered"
class RerouteOutcome(str, Enum):
    CORRECT_ON_TIME          = "correct_on_time"
    CORRECT_LATE             = "correct_late"
    WRONG_DESTINATION        = "wrong_destination"
    MISSED                   = "missed"
    FALSE_POSITIVE           = "false_positive"
    UNIT_ALREADY_ON_SCENE    = "unit_already_on_scene"
    AFTER_HANDOVER           = "after_handover"
@dataclass
class RerouteTrigger:
    trigger_id:         str
    trigger_type:       RerouteTriggerType
    step:               int                  
    affected_unit_ids:  List[str]            
    original_hospital_id: str
    recommended_hospital_id: str            
    recommended_hospital_eta_min: float
    original_hospital_eta_min:   float      
    condition_codes:    List[str]           
    severity_levels:    List[str]           
    required_specialties: List[str]
    diversion_declared: bool = False        
    er_occupancy_pct:   float = 0.90
    congestion_factor:  float = 0.0         
    cascade_risk:       bool = False        
    is_mandatory:       bool = True         
    @property
    def units_affected_count(self) -> int:
        return len(self.affected_unit_ids)
    @property
    def max_severity_rank(self) -> int:
        rank_map = {"P1": 3, "P2": 2, "P3": 1, "P0": 0}
        return max((rank_map.get(s, 0) for s in self.severity_levels), default=0)
    @property
    def has_p1_unit(self) -> bool:
        return "P1" in self.severity_levels
    @property
    def counterfactual_extra_minutes(self) -> float:
        if self.diversion_declared:
            return DIVERSION_PENALTY_MINUTES + self.original_hospital_eta_min
        congestion_penalty = self.original_hospital_eta_min * self.congestion_factor
        er_penalty = ER_OVERCAPACITY_HANDOVER_EXTRA if self.er_occupancy_pct >= ER_DIVERSION_THRESHOLD else 0.0
        return congestion_penalty + er_penalty
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id":               self.trigger_id,
            "trigger_type":             self.trigger_type.value,
            "step":                     self.step,
            "affected_unit_ids":        self.affected_unit_ids,
            "original_hospital_id":     self.original_hospital_id,
            "recommended_hospital_id":  self.recommended_hospital_id,
            "is_mandatory":             self.is_mandatory,
            "diversion_declared":       self.diversion_declared,
            "er_occupancy_pct":         round(self.er_occupancy_pct, 3),
            "congestion_factor":        round(self.congestion_factor, 3),
            "cascade_risk":             self.cascade_risk,
            "has_p1_unit":              self.has_p1_unit,
            "counterfactual_extra_min": round(self.counterfactual_extra_minutes, 2),
        }
@dataclass
class AgentRerouteAction:
    action_id:          str
    unit_id:            str
    step:               int
    from_hospital_id:   str
    to_hospital_id:     str
    reason:             str                  
    travel_time_to_new: float               
    pre_alert_sent:     bool = False
    unit_status_at_reroute: str = "transporting"   
    patient_severity:   str = "P2"
    condition_code:     str = "unknown"
    congestion_hotspots_avoided: List[str] = field(default_factory=list)
    @property
    def is_legal(self) -> bool:
        return self.unit_status_at_reroute not in ("on_scene", "available", "returned")
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id":           self.action_id,
            "unit_id":             self.unit_id,
            "step":                self.step,
            "from_hospital_id":    self.from_hospital_id,
            "to_hospital_id":      self.to_hospital_id,
            "travel_time_to_new":  round(self.travel_time_to_new, 2),
            "pre_alert_sent":      self.pre_alert_sent,
            "is_legal":            self.is_legal,
            "patient_severity":    self.patient_severity,
        }
@dataclass
class RerouteEvaluation:
    trigger:             RerouteTrigger
    matched_actions:     List[AgentRerouteAction]
    outcome:             RerouteOutcome
    timing_latency_steps: Optional[int]      
    destination_quality_score: float         
    units_correctly_rerouted:  int
    units_total:               int
    eta_delta_minutes:         float         
    survival_delta:            float         
    notes:                     List[str] = field(default_factory=list)
    @property
    def fraction_handled(self) -> float:
        if self.units_total == 0:
            return 1.0
        return self.units_correctly_rerouted / self.units_total
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id":              self.trigger.trigger_id,
            "trigger_type":            self.trigger.trigger_type.value,
            "outcome":                 self.outcome.value,
            "timing_latency_steps":    self.timing_latency_steps,
            "destination_quality":     round(self.destination_quality_score, 4),
            "units_handled":           f"{self.units_correctly_rerouted}/{self.units_total}",
            "fraction_handled":        round(self.fraction_handled, 4),
            "eta_delta_minutes":       round(self.eta_delta_minutes, 2),
            "survival_delta":          round(self.survival_delta, 4),
            "notes":                   self.notes,
        }
@dataclass
class HospitalNetworkState:
    hospital_id:       str
    step:              int
    on_diversion:      bool
    er_occupancy_pct:  float
    icu_beds_available: int
    specialties:       List[str]
    has_cath_lab:      bool
    is_level1_trauma:  bool
    is_stroke_centre:  bool
    estimated_eta_min: float       
    specialty_proficiency: float   
    @property
    def is_acceptable(self) -> bool:
        return (not self.on_diversion and
                self.er_occupancy_pct < ER_DIVERSION_THRESHOLD)
    def composite_quality(
        self,
        required_specialties: List[str],
        eta_target: float = 20.0,
    ) -> float:
        if self.on_diversion:
            return 0.05
        spec_score = self._specialty_score(required_specialties)
        cap_score  = self._capacity_score()
        eta_score  = ScoringUtils.response_time_score(
            self.estimated_eta_min, eta_target, worst_case_min=60.0
        )
        return ScoringUtils.clamp(
            0.45 * spec_score + 0.30 * cap_score + 0.25 * eta_score
        )
    def _specialty_score(self, required: List[str]) -> float:
        if not required:
            return 0.70
        matched = sum(
            1 for req in required
            if any(req.lower() in s.lower() or s.lower() in req.lower()
                   for s in self.specialties)
        )
        ratio = matched / len(required)
        return ScoringUtils.clamp(ratio * self.specialty_proficiency)
    def _capacity_score(self) -> float:
        occ = self.er_occupancy_pct
        if occ < 0.70:
            return 1.00
        if occ < 0.80:
            return 0.85
        if occ < 0.90:
            return 0.60
        if occ < 0.95:
            return 0.25
        return 0.05
class TriggerDetectionEngine:
    @staticmethod
    def parse_triggers(gi: GraderInput) -> List[RerouteTrigger]:
        raw_triggers = gi.episode_ledger.get("reroute_triggers", [])
        triggers: List[RerouteTrigger] = []
        for rt in raw_triggers:
            try:
                trig_type = RerouteTriggerType(
                    rt.get("trigger_type", "hospital_diversion")
                )
            except ValueError:
                trig_type = RerouteTriggerType.HOSPITAL_DIVERSION
            trig = RerouteTrigger(
                trigger_id              = rt.get("trigger_id", f"TRG-{len(triggers)+1:03d}"),
                trigger_type            = trig_type,
                step                    = int(rt.get("step", 0)),
                affected_unit_ids       = list(rt.get("affected_unit_ids", [])),
                original_hospital_id    = rt.get("original_hospital_id", "H01"),
                recommended_hospital_id = rt.get("recommended_hospital_id", "H02"),
                recommended_hospital_eta_min = float(rt.get("recommended_eta_min", 12.0)),
                original_hospital_eta_min    = float(rt.get("original_eta_min", 15.0)),
                condition_codes         = list(rt.get("condition_codes", ["unknown"])),
                severity_levels         = list(rt.get("severity_levels", ["P2"])),
                required_specialties    = list(rt.get("required_specialties", [])),
                diversion_declared      = bool(rt.get("diversion_declared", False)),
                er_occupancy_pct        = float(rt.get("er_occupancy_pct", 0.90)),
                congestion_factor       = float(rt.get("congestion_factor", 0.0)),
                cascade_risk            = bool(rt.get("cascade_risk", False)),
                is_mandatory            = bool(rt.get("is_mandatory", True)),
            )
            triggers.append(trig)
        return triggers
    @staticmethod
    def infer_triggers_from_observations(
        gi: GraderInput,
        hospital_states: Dict[str, List[HospitalNetworkState]],
    ) -> List[RerouteTrigger]:
        triggers: List[RerouteTrigger] = []
        trigger_seq = 0
        prev_diversion: Dict[str, bool] = {}
        prev_er_occ:    Dict[str, float] = {}
        for obs in gi.observation_log:
            step   = int(obs.get("step", 0))
            hospitals = obs.get("hospital_network", {}) or {}
            for hid, h_data in hospitals.items():
                on_div  = bool(h_data.get("on_diversion", False))
                er_occ  = float(h_data.get("er_occupancy_pct",
                                            h_data.get("er_load_pct", 70) / 100.0))
                was_div = prev_diversion.get(hid, False)
                if on_div and not was_div:
                    affected = TriggerDetectionEngine._find_units_en_route(
                        gi, hid, step
                    )
                    if affected:
                        trigger_seq += 1
                        triggers.append(RerouteTrigger(
                            trigger_id              = f"INF-TRG-{trigger_seq:03d}",
                            trigger_type            = RerouteTriggerType.HOSPITAL_DIVERSION,
                            step                    = step,
                            affected_unit_ids       = affected,
                            original_hospital_id    = hid,
                            recommended_hospital_id = "BEST_AVAILABLE",
                            recommended_hospital_eta_min = 15.0,
                            original_hospital_eta_min    = 10.0,
                            condition_codes         = ["unknown"],
                            severity_levels         = ["P2"],
                            required_specialties    = list(
                                h_data.get("specialties", [])
                            ),
                            diversion_declared      = True,
                            er_occupancy_pct        = er_occ,
                            is_mandatory            = True,
                        ))
                prev_occ = prev_er_occ.get(hid, 0.0)
                if (not on_div and er_occ >= ER_NEAR_SATURATION and
                        prev_occ < ER_NEAR_SATURATION):
                    affected = TriggerDetectionEngine._find_units_en_route(
                        gi, hid, step
                    )
                    if affected:
                        trigger_seq += 1
                        triggers.append(RerouteTrigger(
                            trigger_id              = f"INF-TRG-{trigger_seq:03d}",
                            trigger_type            = RerouteTriggerType.ER_NEAR_SATURATION,
                            step                    = step,
                            affected_unit_ids       = affected,
                            original_hospital_id    = hid,
                            recommended_hospital_id = "BEST_AVAILABLE",
                            recommended_hospital_eta_min = 15.0,
                            original_hospital_eta_min    = 10.0,
                            condition_codes         = ["unknown"],
                            severity_levels         = ["P2"],
                            required_specialties    = [],
                            diversion_declared      = False,
                            er_occupancy_pct        = er_occ,
                            is_mandatory            = False,
                        ))
                prev_diversion[hid] = on_div
                prev_er_occ[hid]    = er_occ
        return triggers
    @staticmethod
    def _find_units_en_route(
        gi: GraderInput, hospital_id: str, step: int
    ) -> List[str]:
        units: List[str] = []
        for action in gi.action_log:
            if action.action_type != "dispatch":
                continue
            if action.step >= step:
                continue
            dest = (action.get("hospital_id") or action.get("destination_hospital"))
            if dest == hospital_id:
                uid = action.get("unit_id")
                if uid:
                    units.append(uid)
        return list(set(units))
    @staticmethod
    def score_detection(
        triggers:        List[RerouteTrigger],
        agent_reroutes:  List[AgentRerouteAction],
        result:          GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not triggers:
            false_positives = len(agent_reroutes)
            if false_positives == 0:
                return 1.0, {"reason": "no_triggers_no_reroutes"}
            score = ScoringUtils.clamp(1.0 - false_positives * 0.10)
            result.add_note(
                f"No reroute triggers but agent issued {false_positives} reroute(s) — "
                f"false positives: score {score:.3f}"
            )
            return score, {"reason": "no_triggers_with_reroutes",
                           "false_positives": false_positives}
        mandatory   = [t for t in triggers if t.is_mandatory]
        optional    = [t for t in triggers if not t.is_mandatory]
        reroute_by_unit: Dict[str, List[int]] = defaultdict(list)
        for ra in agent_reroutes:
            reroute_by_unit[ra.unit_id].append(ra.step)
        mandatory_scores: List[float] = []
        event_details:    List[Dict]  = []
        for trig in mandatory:
            detected_any = False
            for uid in trig.affected_unit_ids:
                if any(s >= trig.step for s in reroute_by_unit.get(uid, [])):
                    detected_any = True
                    break
            if detected_any:
                mandatory_scores.append(1.0)
                event_details.append({
                    "trigger_id": trig.trigger_id,
                    "type":       trig.trigger_type.value,
                    "detected":   True,
                    "step":       trig.step,
                })
            else:
                penalty_mult = 1.5 if trig.has_p1_unit else 1.0
                mandatory_scores.append(0.0)
                event_details.append({
                    "trigger_id":  trig.trigger_id,
                    "type":        trig.trigger_type.value,
                    "detected":    False,
                    "step":        trig.step,
                    "penalty_mult":penalty_mult,
                })
                result.add_note(
                    f"Missed mandatory trigger {trig.trigger_id} "
                    f"({trig.trigger_type.value}) at step {trig.step}"
                )
        optional_scores: List[float] = []
        for trig in optional:
            detected = any(
                any(s >= trig.step for s in reroute_by_unit.get(uid, []))
                for uid in trig.affected_unit_ids
            )
            optional_scores.append(0.80 if detected else 0.50)
        mand_avg = sum(mandatory_scores) / max(len(mandatory_scores), 1)
        opt_avg  = sum(optional_scores)  / max(len(optional_scores),  1)
        weight_m = 0.80 if mandatory else 0.0
        weight_o = 0.20 if optional  else 0.0
        total_w  = weight_m + weight_o or 1.0
        combined = (weight_m * mand_avg + weight_o * opt_avg) / total_w
        return ScoringUtils.clamp(combined), {
            "mandatory_trigger_count":  len(mandatory),
            "optional_trigger_count":   len(optional),
            "mandatory_detection_rate": round(mand_avg, 4),
            "optional_detection_rate":  round(opt_avg,  4),
            "event_details":            event_details,
        }
class DestinationQualityScorer:
    @staticmethod
    def score(
        evaluations:     List[RerouteEvaluation],
        hospital_states: Dict[str, HospitalNetworkState],
        result:          GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not evaluations:
            return 0.80, {"reason": "no_evaluations_needed"}
        acted = [e for e in evaluations if e.matched_actions]
        if not acted:
            result.add_note("No reroute actions matched to triggers — destination score 0")
            return 0.0, {"reason": "no_matched_actions", "count": len(evaluations)}
        scores:  List[float] = []
        details: List[Dict]  = []
        for ev in acted:
            trig = ev.trigger
            best_quality = 0.0
            for action in ev.matched_actions:
                h_state = hospital_states.get(action.to_hospital_id)
                if h_state is None:
                    h_state = DestinationQualityScorer._build_synthetic_state(action)
                q = h_state.composite_quality(
                    trig.required_specialties,
                    eta_target=trig.recommended_hospital_eta_min,
                )
                if action.to_hospital_id == trig.recommended_hospital_id:
                    q = min(1.0, q + 0.10)
                    result.add_note(
                        f"Agent chose optimal hospital {action.to_hospital_id} "
                        f"for trigger {trig.trigger_id} — bonus"
                    )
                best_quality = max(best_quality, q)
            severity_mult = {"P1": 1.5, "P2": 1.0, "P3": 0.6}.get(
                max(trig.severity_levels,
                    key=lambda s: {"P1": 3, "P2": 2, "P3": 1, "P0": 0}.get(s, 0),
                    default="P2"
                ), 1.0
            )
            scores.append(best_quality)
            details.append({
                "trigger_id":      trig.trigger_id,
                "quality_score":   round(best_quality, 4),
                "severity_mult":   severity_mult,
                "actions_matched": len(ev.matched_actions),
            })
        weighted_avg = sum(scores) / len(scores) if scores else 0.0
        return ScoringUtils.clamp(weighted_avg), {
            "evaluated_triggers": len(acted),
            "mean_destination_quality": round(weighted_avg, 4),
            "destination_details": details,
        }
    @staticmethod
    def _build_synthetic_state(action: AgentRerouteAction) -> HospitalNetworkState:
        return HospitalNetworkState(
            hospital_id        = action.to_hospital_id,
            step               = action.step,
            on_diversion       = False,
            er_occupancy_pct   = 0.65,  
            icu_beds_available = 4,
            specialties        = [],
            has_cath_lab       = False,
            is_level1_trauma   = False,
            is_stroke_centre   = False,
            estimated_eta_min  = action.travel_time_to_new,
            specialty_proficiency = 0.70,
        )
class TimingWindowScorer:
    @staticmethod
    def score(
        evaluations: List[RerouteEvaluation],
        result:      GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not evaluations:
            return 1.0, {"reason": "no_triggers"}
        mandatory = [e for e in evaluations if e.trigger.is_mandatory]
        if not mandatory:
            return 0.85, {"reason": "only_optional_triggers"}
        timing_scores:  List[Tuple[float, float]] = []  
        timing_details: List[Dict] = []
        for ev in mandatory:
            sev_weight = {"P1": 2.0, "P2": 1.2, "P3": 0.6}.get(
                "P1" if ev.trigger.has_p1_unit else
                ("P2" if "P2" in ev.trigger.severity_levels else "P3"),
                1.0
            )
            if not ev.matched_actions:
                timing_scores.append((0.0, sev_weight))
                timing_details.append({
                    "trigger_id":     ev.trigger.trigger_id,
                    "latency_steps":  None,
                    "score":          0.0,
                    "status":         "missed",
                })
                result.add_note(
                    f"Trigger {ev.trigger.trigger_id} missed — timing score 0"
                )
                continue
            trigger_step  = ev.trigger.step
            reroute_steps = [
                a.step for a in ev.matched_actions if a.step >= trigger_step
            ]
            if not reroute_steps:
                pre_emptive_steps = [
                    a.step for a in ev.matched_actions if a.step < trigger_step
                ]
                if pre_emptive_steps:
                    timing_scores.append((1.0, sev_weight))
                    timing_details.append({
                        "trigger_id":    ev.trigger.trigger_id,
                        "latency_steps": trigger_step - min(pre_emptive_steps),
                        "score":         1.0,
                        "status":        "pre_emptive",
                    })
                    result.add_note(
                        f"Pre-emptive reroute for {ev.trigger.trigger_id} — full timing credit"
                    )
                else:
                    timing_scores.append((0.0, sev_weight))
                    timing_details.append({
                        "trigger_id":    ev.trigger.trigger_id,
                        "latency_steps": None,
                        "score":         0.0,
                        "status":        "missed",
                    })
                continue
            latency = min(reroute_steps) - trigger_step
            if latency <= REROUTE_OPTIMAL_STEPS:
                t_score = 1.00
                status  = "on_time"
            elif latency <= REROUTE_GOOD_STEPS:
                t_score = 0.75 - 0.05 * (latency - REROUTE_OPTIMAL_STEPS)
                status  = "slightly_late"
            elif latency <= REROUTE_LATE_STEPS:
                t_score = 0.40 - 0.05 * (latency - REROUTE_GOOD_STEPS)
                status  = "late"
            else:
                t_score = max(0.05, 0.20 - 0.02 * (latency - REROUTE_LATE_STEPS))
                status  = "very_late"
                result.add_note(
                    f"Very late reroute: {ev.trigger.trigger_id} latency={latency} steps"
                )
            if ev.trigger.has_p1_unit and t_score < 0.80:
                result.add_note(
                    f"P1 unit reroute latency={latency} steps for trigger "
                    f"{ev.trigger.trigger_id} — golden hour impact"
                )
            timing_scores.append((t_score, sev_weight))
            timing_details.append({
                "trigger_id":    ev.trigger.trigger_id,
                "latency_steps": latency,
                "score":         round(t_score, 4),
                "status":        status,
                "has_p1_unit":   ev.trigger.has_p1_unit,
            })
        if not timing_scores:
            return 0.0, {"reason": "no_mandatory_scored"}
        total_w   = sum(w for _, w in timing_scores)
        weighted  = sum(s * w for s, w in timing_scores) / max(total_w, 1.0)
        return ScoringUtils.clamp(weighted), {
            "timing_details":   timing_details,
            "weighted_score":   round(weighted, 4),
        }
class FalsePositiveScorer:
    @staticmethod
    def score(
        triggers:       List[RerouteTrigger],
        agent_reroutes: List[AgentRerouteAction],
        result:         GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not agent_reroutes:
            return 1.0, {"false_positives": 0, "total_reroutes": 0}
        if not triggers:
            n_fp = len(agent_reroutes)
            result.add_note(f"All {n_fp} reroute(s) are false positives (no triggers)")
            return ScoringUtils.clamp(1.0 - n_fp * 0.15), {
                "false_positives":     n_fp,
                "total_reroutes":      n_fp,
                "false_positive_rate": 1.0,
            }
        justified_windows: Set[Tuple[str, int, int]] = set()
        for trig in triggers:
            for uid in trig.affected_unit_ids:
                for window_step in range(trig.step, trig.step + REROUTE_LATE_STEPS + 2):
                    justified_windows.add((uid, window_step, trig.step))
        false_positives = 0
        justified_count  = 0
        fp_details:      List[Dict] = []
        for ra in agent_reroutes:
            is_justified = any(
                uid == ra.unit_id and step == ra.step
                for uid, step, _ in justified_windows
            )
            if is_justified:
                justified_count += 1
            else:
                false_positives += 1
                fp_details.append({
                    "unit_id":      ra.unit_id,
                    "step":         ra.step,
                    "to_hospital":  ra.to_hospital_id,
                    "reason":       ra.reason[:50] if ra.reason else "N/A",
                })
                result.add_note(
                    f"Possible false positive reroute: unit {ra.unit_id} "
                    f"at step {ra.step} → {ra.to_hospital_id} (no active trigger)"
                )
        total = len(agent_reroutes)
        fp_rate = false_positives / total if total > 0 else 0.0
        score = ScoringUtils.clamp(1.0 - fp_rate * 0.60 - false_positives * 0.05)
        return score, {
            "false_positives":     false_positives,
            "justified_reroutes":  justified_count,
            "total_reroutes":      total,
            "false_positive_rate": round(fp_rate, 4),
            "fp_details":          fp_details[:5],  
        }
class UnitSelectionScorer:
    @staticmethod
    def score(
        triggers:       List[RerouteTrigger],
        agent_reroutes: List[AgentRerouteAction],
        result:         GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not triggers or not agent_reroutes:
            return 1.0 if not triggers else 0.60, {
                "reason": "no_triggers" if not triggers else "no_reroutes"
            }
        required_units: Set[str] = set()
        for trig in triggers:
            required_units.update(trig.affected_unit_ids)
        rerouted_units = {ra.unit_id for ra in agent_reroutes}
        illegal_actions: List[Dict] = []
        for ra in agent_reroutes:
            if not ra.is_legal:
                illegal_actions.append({
                    "unit_id": ra.unit_id,
                    "status":  ra.unit_status_at_reroute,
                    "step":    ra.step,
                })
                result.add_note(
                    f"ILLEGAL reroute: unit {ra.unit_id} is "
                    f"'{ra.unit_status_at_reroute}' — cannot reroute"
                )
        missed_units = required_units - rerouted_units
        for uid in missed_units:
            for trig in triggers:
                if uid in trig.affected_unit_ids and trig.has_p1_unit:
                    result.add_note(
                        f"P1 unit {uid} required reroute but none issued "
                        f"(trigger {trig.trigger_id})"
                    )
                    break
        total_required = len(required_units)
        if total_required == 0:
            coverage = 1.0
        else:
            correctly_rerouted = len(
                required_units & rerouted_units
            ) - len(illegal_actions)
            correctly_rerouted = max(0, correctly_rerouted)
            coverage = correctly_rerouted / total_required
        illegal_penalty = len(illegal_actions) * 0.15
        score = ScoringUtils.clamp(coverage - illegal_penalty)
        return score, {
            "required_units":   sorted(required_units),
            "rerouted_units":   sorted(rerouted_units),
            "missed_units":     sorted(missed_units),
            "illegal_reroutes": len(illegal_actions),
            "coverage_score":   round(coverage, 4),
            "illegal_penalty":  round(illegal_penalty, 4),
        }
class TimeSavedScorer:
    @staticmethod
    def score(
        evaluations: List[RerouteEvaluation],
        triggers:    List[RerouteTrigger],
        result:      GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not triggers:
            return 1.0, {"reason": "no_triggers"}
        eta_scores:  List[Tuple[float, float]] = []
        surv_scores: List[Tuple[float, float]] = []
        for ev in evaluations:
            trig = ev.trigger
            sev_weight = 2.0 if trig.has_p1_unit else 1.0
            if ev.matched_actions:
                actual_eta = min(
                    a.travel_time_to_new for a in ev.matched_actions
                )
                counterfactual_eta = (
                    trig.original_hospital_eta_min +
                    trig.counterfactual_extra_minutes
                )
                eta_delta = counterfactual_eta - actual_eta  
                ev.eta_delta_minutes = eta_delta
                max_save = trig.counterfactual_extra_minutes
                if max_save > 0:
                    eta_score = ScoringUtils.clamp(eta_delta / max_save)
                else:
                    eta_score = 0.80 if eta_delta >= 0 else 0.40
            else:
                ev.eta_delta_minutes = -trig.counterfactual_extra_minutes
                eta_score = 0.0
                result.add_note(
                    f"Missed reroute for trigger {trig.trigger_id}: "
                    f"patient delayed ≈{trig.counterfactual_extra_minutes:.0f} min"
                )
            eta_scores.append((eta_score, sev_weight))
        for ev in evaluations:
            trig = ev.trigger
            sev_weight = 2.0 if trig.has_p1_unit else 1.0
            if ev.matched_actions:
                surv_delta = ev.survival_delta  
                if surv_delta >= 0:
                    s_score = min(1.0, 0.5 + surv_delta * 5.0)
                else:
                    s_score = max(0.0, 0.5 + surv_delta * 3.0)
            else:
                s_score = 0.10  
            surv_scores.append((s_score, sev_weight))
        cascade_triggers = [t for t in triggers if t.cascade_risk]
        if not cascade_triggers:
            cascade_score = 0.90  
        else:
            cascade_avoided = 0
            for trig in cascade_triggers:
                for ev in evaluations:
                    if ev.trigger.trigger_id == trig.trigger_id and ev.matched_actions:
                        cascade_avoided += 1
                        break
            cascade_score = ScoringUtils.clamp(
                cascade_avoided / len(cascade_triggers)
            )
            if cascade_avoided < len(cascade_triggers):
                result.add_note(
                    f"Cascade risk unmitigated: {len(cascade_triggers) - cascade_avoided} "
                    f"cascade-risk trigger(s) not rerouted"
                )
        def weighted_avg(scores: List[Tuple[float, float]]) -> float:
            total_w = sum(w for _, w in scores)
            if total_w == 0:
                return 0.5
            return sum(s * w for s, w in scores) / total_w
        t1 = weighted_avg(eta_scores)
        t2 = weighted_avg(surv_scores)
        t3 = cascade_score
        time_saved_total = ScoringUtils.clamp(
            WT_ETA_DELTA * t1 + WT_SURVIVAL * t2 + WT_CASCADE * t3
        )
        total_eta_saved = sum(
            ev.eta_delta_minutes for ev in evaluations if ev.eta_delta_minutes > 0
        )
        total_eta_lost  = abs(sum(
            ev.eta_delta_minutes for ev in evaluations if ev.eta_delta_minutes < 0
        ))
        return time_saved_total, {
            "eta_score":       round(t1, 4),
            "survival_score":  round(t2, 4),
            "cascade_score":   round(t3, 4),
            "total_eta_saved_min": round(total_eta_saved, 2),
            "total_eta_lost_min":  round(total_eta_lost,  2),
            "cascade_risk_triggers": len(cascade_triggers),
        }
class EvaluationLinker:
    @staticmethod
    def link(
        triggers:       List[RerouteTrigger],
        agent_reroutes: List[AgentRerouteAction],
    ) -> List[RerouteEvaluation]:
        evaluations: List[RerouteEvaluation] = []
        for trig in triggers:
            matched: List[AgentRerouteAction] = []
            for ra in agent_reroutes:
                if ra.unit_id not in trig.affected_unit_ids:
                    continue
                if ra.step < trig.step - 3:
                    continue
                matched.append(ra)
            if not matched:
                outcome = RerouteOutcome.MISSED
                dest_quality = 0.0
                latency = None
                units_correct = 0
                eta_delta  = -trig.counterfactual_extra_minutes
                surv_delta = -0.05 * (1 if trig.has_p1_unit else 0.3)
            else:
                earliest = min(matched, key=lambda a: a.step)
                latency  = max(0, earliest.step - trig.step)
                if not earliest.is_legal:
                    outcome = RerouteOutcome.UNIT_ALREADY_ON_SCENE
                    dest_quality = 0.0
                    units_correct = 0
                elif earliest.to_hospital_id == trig.recommended_hospital_id:
                    outcome = (RerouteOutcome.CORRECT_ON_TIME
                               if latency <= REROUTE_GOOD_STEPS
                               else RerouteOutcome.CORRECT_LATE)
                    dest_quality = 1.0
                    units_correct = sum(
                        1 for a in matched if a.unit_id in trig.affected_unit_ids
                    )
                else:
                    outcome = (RerouteOutcome.CORRECT_ON_TIME
                               if latency <= REROUTE_GOOD_STEPS
                               else RerouteOutcome.WRONG_DESTINATION)
                    dest_quality = 0.60  
                    units_correct = 0
                actual_eta      = min(a.travel_time_to_new for a in matched)
                cf_eta          = trig.original_hospital_eta_min + trig.counterfactual_extra_minutes
                eta_delta       = cf_eta - actual_eta
                time_at_reroute = trig.recommended_hospital_eta_min
                time_no_reroute = cf_eta + 5.0   
                lambda_          = 0.008 if trig.has_p1_unit else 0.003
                surv_delta = (
                    math.exp(-lambda_ * time_at_reroute) -
                    math.exp(-lambda_ * time_no_reroute)
                )
            notes: List[str] = []
            if outcome == RerouteOutcome.MISSED and trig.has_p1_unit:
                notes.append(f"P1 unit missed reroute — severe survival impact")
            if outcome == RerouteOutcome.UNIT_ALREADY_ON_SCENE:
                notes.append("Illegal reroute: unit already on scene")
            evaluations.append(RerouteEvaluation(
                trigger                    = trig,
                matched_actions            = matched,
                outcome                    = outcome,
                timing_latency_steps       = latency,
                destination_quality_score  = dest_quality,
                units_correctly_rerouted   = units_correct if matched else 0,
                units_total                = trig.units_affected_count,
                eta_delta_minutes          = eta_delta if matched else -trig.counterfactual_extra_minutes,
                survival_delta             = surv_delta if matched else -0.05,
                notes                      = notes,
            ))
        return evaluations
class HardPenaltyEngine:
    @staticmethod
    def apply(
        triggers:       List[RerouteTrigger],
        agent_reroutes: List[AgentRerouteAction],
        evaluations:    List[RerouteEvaluation],
        hospital_states: Dict[str, HospitalNetworkState],
        result:         GraderResult,
    ) -> float:
        total_penalty = 0.0
        for ra in agent_reroutes:
            h_state = hospital_states.get(ra.to_hospital_id)
            if h_state and h_state.on_diversion:
                result.add_penalty(PenaltyRecord(
                    name     = f"reroute_to_diverted_{ra.unit_id[:12]}",
                    amount   = -HP_DIVERTED_DESTINATION,
                    reason   = (
                        f"Unit {ra.unit_id} rerouted to {ra.to_hospital_id} "
                        f"which is on diversion at step {ra.step}"
                    ),
                    rule_ref = "EMERGI-ENV Rule T5-H1",
                ))
                result.critical_mismatches += 1
                total_penalty += HP_DIVERTED_DESTINATION
                result.add_note(
                    f"⚠ CRITICAL: Rerouted {ra.unit_id} to diverted hospital "
                    f"{ra.to_hospital_id} — patient will be refused"
                )
        for ev in evaluations:
            if ev.outcome == RerouteOutcome.MISSED and ev.trigger.has_p1_unit:
                n_p1 = sum(1 for s in ev.trigger.severity_levels if s == "P1")
                penalty = HP_MISSED_REROUTE_P1 * n_p1
                result.add_penalty(PenaltyRecord(
                    name     = f"missed_p1_reroute_{ev.trigger.trigger_id}",
                    amount   = -penalty,
                    reason   = (
                        f"Trigger {ev.trigger.trigger_id}: {n_p1} P1 unit(s) "
                        f"not rerouted after {ev.trigger.trigger_type.value}"
                    ),
                    rule_ref = "EMERGI-ENV Rule T5-H2",
                ))
                result.critical_mismatches += n_p1
                total_penalty += penalty
                result.add_note(
                    f"⚠ CRITICAL: {n_p1} P1 patient(s) not rerouted after "
                    f"trigger at step {ev.trigger.step}"
                )
        for ev in evaluations:
            if not ev.matched_actions:
                continue
            for ra in ev.matched_actions:
                if not ev.trigger.required_specialties:
                    continue
                h_state = hospital_states.get(ra.to_hospital_id)
                if h_state is None:
                    continue
                has_any = any(
                    any(req.lower() in s.lower() for s in h_state.specialties)
                    for req in ev.trigger.required_specialties
                )
                if not has_any:
                    result.add_penalty(PenaltyRecord(
                        name     = f"no_specialty_reroute_{ra.unit_id[:12]}",
                        amount   = -HP_NO_SPECIALTY_REROUTE,
                        reason   = (
                            f"Unit {ra.unit_id} rerouted to {ra.to_hospital_id} "
                            f"lacking required specialties "
                            f"{ev.trigger.required_specialties}"
                        ),
                        rule_ref = "EMERGI-ENV Rule T5-H3",
                    ))
                    total_penalty += HP_NO_SPECIALTY_REROUTE
        for ra in agent_reroutes:
            h_state = hospital_states.get(ra.to_hospital_id)
            if h_state and h_state.er_occupancy_pct >= ER_DIVERSION_THRESHOLD:
                result.add_penalty(PenaltyRecord(
                    name     = f"reroute_near_full_{ra.unit_id[:12]}",
                    amount   = -HP_NEAR_FULL_HOSPITAL,
                    reason   = (
                        f"Rerouted {ra.unit_id} to {ra.to_hospital_id} "
                        f"(ER {h_state.er_occupancy_pct:.0%} occupied)"
                    ),
                    rule_ref = "EMERGI-ENV Rule T5-H4",
                ))
                total_penalty += HP_NEAR_FULL_HOSPITAL
        for ra in agent_reroutes:
            if ra.unit_status_at_reroute == "on_scene":
                result.add_penalty(PenaltyRecord(
                    name     = f"reroute_on_scene_{ra.unit_id[:12]}",
                    amount   = -HP_UNIT_REROUTED_ON_SCENE,
                    reason   = (
                        f"Unit {ra.unit_id} rerouted while on_scene — "
                        f"patient cannot be moved mid-treatment"
                    ),
                    rule_ref = "EMERGI-ENV Rule T5-H5",
                ))
                total_penalty += HP_UNIT_REROUTED_ON_SCENE
                result.critical_mismatches += 1
        for ra in agent_reroutes:
            if ra.unit_status_at_reroute in ("returned", "available"):
                result.add_penalty(PenaltyRecord(
                    name     = f"reroute_post_handover_{ra.unit_id[:12]}",
                    amount   = -HP_LATE_HANDOVER_REROUTE,
                    reason   = (
                        f"Unit {ra.unit_id} rerouted after handover "
                        f"(status: {ra.unit_status_at_reroute})"
                    ),
                    rule_ref = "EMERGI-ENV Rule T5-H6",
                ))
                total_penalty += HP_LATE_HANDOVER_REROUTE
        return total_penalty
class ProtocolBonusEngine:
    @staticmethod
    def apply(
        triggers:       List[RerouteTrigger],
        agent_reroutes: List[AgentRerouteAction],
        evaluations:    List[RerouteEvaluation],
        hospital_states: Dict[str, HospitalNetworkState],
        gi:             GraderInput,
        result:         GraderResult,
    ) -> float:
        net_bonus = 0.0
        pre_alerted = sum(1 for ra in agent_reroutes if ra.pre_alert_sent)
        if pre_alerted > 0:
            bonus = PB_PRE_ALERT * min(pre_alerted, 4)
            net_bonus += bonus
            result.add_note(
                f"Protocol bonus: pre-alert sent to {pre_alerted} hospital(s) — "
                f"+{bonus:.3f}"
            )
        for ev in evaluations:
            if not ev.matched_actions:
                continue
            for ra in ev.matched_actions:
                if ra.condition_code not in TRAUMA_L1_CONDITIONS:
                    continue
                h_state = hospital_states.get(ra.to_hospital_id)
                if h_state and h_state.is_level1_trauma and ra.travel_time_to_new <= 30.0:
                    net_bonus += PB_TRAUMA_CENTRE_30MIN
                    result.add_note(
                        f"Trauma bonus: Level-1 centre {ra.to_hospital_id} "
                        f"reached in {ra.travel_time_to_new:.0f} min — "
                        f"+{PB_TRAUMA_CENTRE_30MIN:.3f}"
                    )
                    break  
        for ev in evaluations:
            if not ev.matched_actions or not ev.trigger.required_specialties:
                continue
            earliest = min(ev.matched_actions, key=lambda a: a.step)
            latency  = max(0, earliest.step - ev.trigger.step)
            if latency <= 1:
                h_state = hospital_states.get(earliest.to_hospital_id)
                if h_state:
                    has_spec = any(
                        any(req.lower() in s.lower() for s in h_state.specialties)
                        for req in ev.trigger.required_specialties
                    )
                    if has_spec:
                        net_bonus += PB_SPECIALTY_MATCH_FAST
                        result.add_note(
                            f"Fast specialty match: {earliest.to_hospital_id} "
                            f"within {latency} step(s) — +{PB_SPECIALTY_MATCH_FAST:.3f}"
                        )
        for ev in evaluations:
            trig = ev.trigger
            if not trig.cascade_risk or not ev.matched_actions:
                continue
            pre_emptive = [a for a in ev.matched_actions if a.step < trig.step]
            if pre_emptive:
                net_bonus += PB_CASCADE_PREEMPTION
                result.add_note(
                    f"Cascade pre-emption: rerouted {pre_emptive[0].unit_id} "
                    f"before trigger {trig.trigger_id} fired — "
                    f"+{PB_CASCADE_PREEMPTION:.3f}"
                )
        unit_reroute_counts: Dict[str, int] = defaultdict(int)
        for ra in agent_reroutes:
            unit_reroute_counts[ra.unit_id] += 1
        single_reroute_correct = sum(
            1 for uid, count in unit_reroute_counts.items()
            if count == 1 and any(
                ev.outcome in (RerouteOutcome.CORRECT_ON_TIME, RerouteOutcome.CORRECT_LATE)
                for ev in evaluations
                if any(a.unit_id == uid for a in ev.matched_actions)
            )
        )
        if single_reroute_correct > 0:
            bonus = PB_FIRST_TRY_CORRECT * min(single_reroute_correct, 3)
            net_bonus += bonus
            result.add_note(
                f"First-try correct: {single_reroute_correct} unit(s) correctly "
                f"rerouted on first attempt — +{bonus:.3f}"
            )
        mutual_aid_actions = gi.get_actions_by_type("request_mutual_aid")
        if mutual_aid_actions:
            for ma_action in mutual_aid_actions:
                fleet_avail = ma_action.get("own_fleet_available_count", 99)
                if fleet_avail <= 2:
                    net_bonus += PB_MUTUAL_AID_CORRECT
                    result.add_note(
                        f"Mutual aid correctly requested (fleet={fleet_avail} available) — "
                        f"+{PB_MUTUAL_AID_CORRECT:.3f}"
                    )
                    break  
        capped = ScoringUtils.clamp(net_bonus, 0.0, PROTOCOL_BONUS_CAP)
        if abs(capped - net_bonus) > 1e-6:
            result.add_note(f"Protocol bonus {net_bonus:.3f} capped to {capped:.3f}")
        return capped
class Task5InputParser:
    @staticmethod
    def parse_agent_reroutes(gi: GraderInput) -> List[AgentRerouteAction]:
        reroutes: List[AgentRerouteAction] = []
        for idx, action in enumerate(gi.action_log):
            if action.action_type not in ("reroute", "reroute_unit", "dynamic_reroute"):
                continue
            ra = AgentRerouteAction(
                action_id          = f"RA-{idx:04d}",
                unit_id            = action.get("unit_id") or action.get("ambulance_id") or "UNK",
                step               = action.step,
                from_hospital_id   = (action.get("from_hospital_id") or
                                      action.get("original_hospital") or "UNKNOWN"),
                to_hospital_id     = (action.get("to_hospital_id") or
                                      action.get("new_hospital_id") or
                                      action.get("hospital_id") or "UNKNOWN"),
                reason             = str(action.get("reason") or action.get("rationale") or ""),
                travel_time_to_new = float(action.get("travel_time_min") or
                                           action.get("eta_minutes") or 15.0),
                pre_alert_sent     = bool(action.get("pre_alert_sent", False)),
                unit_status_at_reroute = str(
                    action.get("unit_status") or action.get("current_status") or "transporting"
                ),
                patient_severity   = str(action.get("patient_severity") or
                                         action.get("severity") or "P2"),
                condition_code     = str(action.get("condition_code") or "unknown"),
                congestion_hotspots_avoided = list(
                    action.get("congestion_hotspots_avoided") or []
                ),
            )
            reroutes.append(ra)
        return reroutes
    @staticmethod
    def parse_hospital_states(gi: GraderInput) -> Dict[str, HospitalNetworkState]:
        states: Dict[str, HospitalNetworkState] = {}
        source = gi.final_hospital_state or {}
        if not source and gi.observation_log:
            last_obs = gi.observation_log[-1]
            source   = last_obs.get("hospital_network", {}) or {}
        step = gi.episode_steps
        for hid, h_data in source.items():
            if isinstance(h_data, dict):
                specialties = h_data.get("specialties", []) or []
                er_occ_raw  = h_data.get("er_occupancy_pct",
                              h_data.get("er_load_pct", 0) / 100.0)
                states[hid] = HospitalNetworkState(
                    hospital_id           = hid,
                    step                  = step,
                    on_diversion          = bool(h_data.get("on_diversion",
                                           h_data.get("diversion_status") == "diverted")),
                    er_occupancy_pct      = float(er_occ_raw),
                    icu_beds_available    = int(h_data.get("icu_beds_available",
                                           h_data.get("available_icu", 4))),
                    specialties           = list(specialties),
                    has_cath_lab          = bool(h_data.get("has_cath_lab", False)),
                    is_level1_trauma      = bool(h_data.get("is_level1_trauma",
                                           h_data.get("trauma_level", 3) <= 1)),
                    is_stroke_centre      = bool(h_data.get("is_stroke_centre", False)),
                    estimated_eta_min     = float(h_data.get("estimated_travel_min", 12.0)),
                    specialty_proficiency = float(h_data.get("specialty_proficiency", 0.75)),
                )
        return states
    @staticmethod
    def parse_episode_summary(gi: GraderInput) -> Dict[str, Any]:
        return {
            "fleet_size":          gi.episode_ledger.get("fleet_size", 24),
            "n_reroute_events":    gi.episode_ledger.get("n_reroute_events", 0),
            "traffic_updates":     gi.episode_ledger.get("traffic_updates", 0),
            "diversion_events":    gi.episode_ledger.get("diversion_events", 0),
            "congestion_spikes":   gi.episode_ledger.get("congestion_spikes", 0),
            "cascade_prevented":   gi.episode_ledger.get("cascade_prevented", False),
            "surge_declared":      gi.episode_ledger.get("surge_declared", False),
        }
class Task5Grader(BaseGrader):
    TASK_ID          = TASK_ID
    TASK_SEED        = TASK_SEED
    TASK_BASELINE    = TASK_BASELINE
    TASK_DIFFICULTY  = "medium"
    COMPONENT_WEIGHTS: Dict[str, float] = {
        "reroute_correctness": W_CORRECTNESS,
        "net_time_saved":      W_TIME_SAVED,
    }
    def __init__(self) -> None:
        super().__init__()
        self._parser            = Task5InputParser()
        self._trigger_engine    = TriggerDetectionEngine()
        self._dest_scorer       = DestinationQualityScorer()
        self._timing_scorer     = TimingWindowScorer()
        self._fp_scorer         = FalsePositiveScorer()
        self._unit_scorer       = UnitSelectionScorer()
        self._time_scorer       = TimeSavedScorer()
        self._eval_linker       = EvaluationLinker()
        self._hard_penalty_eng  = HardPenaltyEngine()
        self._protocol_eng      = ProtocolBonusEngine()
    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(
                f"WARNING: episode seed {gi.seed} ≠ task seed {TASK_SEED}. "
                "Determinism may be affected."
            )
        agent_reroutes  = self._parser.parse_agent_reroutes(gi)
        hospital_states = self._parser.parse_hospital_states(gi)
        ep_summary      = self._parser.parse_episode_summary(gi)
        triggers = TriggerDetectionEngine.parse_triggers(gi)
        if not triggers:
            triggers = TriggerDetectionEngine.infer_triggers_from_observations(
                gi, {}  
            )
        result.extra["n_triggers"]          = len(triggers)
        result.extra["n_mandatory_triggers"] = sum(1 for t in triggers if t.is_mandatory)
        result.extra["n_agent_reroutes"]    = len(agent_reroutes)
        result.extra["n_hospitals_in_state"]= len(hospital_states)
        result.extra.update(ep_summary)
        evaluations = self._eval_linker.link(triggers, agent_reroutes)
        correctness_score = self._compute_correctness(
            triggers, agent_reroutes, evaluations, hospital_states, result
        )
        self._add_component(
            result, "reroute_correctness", correctness_score, W_CORRECTNESS,
            f"triggers={len(triggers)} reroutes={len(agent_reroutes)} "
            f"mandatory={sum(1 for t in triggers if t.is_mandatory)}"
        )
        time_saved_score, ts_detail = self._time_scorer.score(
            evaluations, triggers, result
        )
        self._add_component(
            result, "net_time_saved", time_saved_score, W_TIME_SAVED,
            f"eta_saved={ts_detail.get('total_eta_saved_min', 0):.1f}min | "
            f"cascade_score={ts_detail.get('cascade_score', 0):.3f}"
        )
        result.extra["time_saved_detail"] = ts_detail
        self._hard_penalty_eng.apply(
            triggers, agent_reroutes, evaluations, hospital_states, result
        )
        proto_bonus = self._protocol_eng.apply(
            triggers, agent_reroutes, evaluations, hospital_states, gi, result
        )
        if proto_bonus > 0:
            result.add_component(ScoringUtils.build_score_component(
                "protocol_compliance", proto_bonus, 1.0,
                f"Protocol compliance bonus: +{proto_bonus:.4f}"
            ))
        result.extra["protocol_bonus"] = round(proto_bonus, 4)
        outcome_counts: Dict[str, int] = defaultdict(int)
        for ev in evaluations:
            outcome_counts[ev.outcome.value] += 1
        result.extra["outcome_counts"]  = dict(outcome_counts)
        result.extra["evaluations"]     = [ev.to_dict() for ev in evaluations]
        result.extra["triggers_summary"]= [t.to_dict() for t in triggers[:10]]
        result.total_patients   = gi.total_patients
        result.p1_patients      = gi.p1_patients
        result.p1_survival_rate = self._compute_p1_survival(gi)
        result.protocol_violations += sum(
            1 for ev in evaluations
            if ev.outcome in (RerouteOutcome.MISSED, RerouteOutcome.WRONG_DESTINATION)
            and ev.trigger.has_p1_unit
        )
    def _compute_correctness(
        self,
        triggers:        List[RerouteTrigger],
        agent_reroutes:  List[AgentRerouteAction],
        evaluations:     List[RerouteEvaluation],
        hospital_states: Dict[str, HospitalNetworkState],
        result:          GraderResult,
    ) -> float:
        c1_score, c1_detail = TriggerDetectionEngine.score_detection(
            triggers, agent_reroutes, result
        )
        result.extra["c1_trigger_detection"] = {
            "score": round(c1_score, 4), **c1_detail
        }
        result.add_note(f"C1 Trigger detection: {c1_score:.4f}")
        c2_score, c2_detail = DestinationQualityScorer.score(
            evaluations, hospital_states, result
        )
        result.extra["c2_destination_quality"] = {
            "score": round(c2_score, 4), **c2_detail
        }
        result.add_note(f"C2 Destination quality: {c2_score:.4f}")
        c3_score, c3_detail = TimingWindowScorer.score(evaluations, result)
        result.extra["c3_timing_window"] = {
            "score": round(c3_score, 4), **c3_detail
        }
        result.add_note(f"C3 Timing window: {c3_score:.4f}")
        c4_score, c4_detail = FalsePositiveScorer.score(
            triggers, agent_reroutes, result
        )
        result.extra["c4_false_positive"] = {
            "score": round(c4_score, 4), **c4_detail
        }
        result.add_note(f"C4 False positive: {c4_score:.4f}")
        c5_score, c5_detail = UnitSelectionScorer.score(
            triggers, agent_reroutes, result
        )
        result.extra["c5_unit_selection"] = {
            "score": round(c5_score, 4), **c5_detail
        }
        result.add_note(f"C5 Unit selection: {c5_score:.4f}")
        correctness = ScoringUtils.clamp(
            WC_TRIGGER_DETECTION * c1_score +
            WC_DESTINATION_QUAL  * c2_score +
            WC_TIMING_WINDOW     * c3_score +
            WC_FALSE_POSITIVE    * c4_score +
            WC_UNIT_SELECTION    * c5_score
        )
        result.add_note(
            f"Correctness = {WC_TRIGGER_DETECTION}×{c1_score:.3f} + "
            f"{WC_DESTINATION_QUAL}×{c2_score:.3f} + "
            f"{WC_TIMING_WINDOW}×{c3_score:.3f} + "
            f"{WC_FALSE_POSITIVE}×{c4_score:.3f} + "
            f"{WC_UNIT_SELECTION}×{c5_score:.3f} = {correctness:.4f}"
        )
        return correctness
    def _compute_p1_survival(self, gi: GraderInput) -> float:
        p1_summaries = gi.p1_summaries()
        if not p1_summaries:
            return 1.0
        treated = sum(
            1 for s in p1_summaries
            if s.get("phase_at_episode_end") == "treated"
        )
        return treated / len(p1_summaries)
GraderRegistry.register(TASK_ID, Task5Grader)
logger.info(
    "Task5Grader registered: task_id=%s baseline=%.2f seed=%d",
    TASK_ID, TASK_BASELINE, TASK_SEED,
)
def _mk_trigger(
    trigger_id:        str   = "TRG-001",
    trigger_type:      str   = "hospital_diversion",
    step:              int   = 5,
    units:             Optional[List[str]] = None,
    orig_hospital:     str   = "H03",
    rec_hospital:      str   = "H01",
    rec_eta:           float = 12.0,
    orig_eta:          float = 8.0,
    conditions:        Optional[List[str]] = None,
    severities:        Optional[List[str]] = None,
    specialties:       Optional[List[str]] = None,
    diversion:         bool  = True,
    er_occ:            float = 0.95,
    congestion:        float = 0.0,
    cascade:           bool  = False,
    mandatory:         bool  = True,
) -> Dict[str, Any]:
    return {
        "trigger_id":             trigger_id,
        "trigger_type":           trigger_type,
        "step":                   step,
        "affected_unit_ids":      units or ["AMB-M001"],
        "original_hospital_id":   orig_hospital,
        "recommended_hospital_id":rec_hospital,
        "recommended_eta_min":    rec_eta,
        "original_eta_min":       orig_eta,
        "condition_codes":        conditions or ["stemi_anterior"],
        "severity_levels":        severities or ["P1"],
        "required_specialties":   specialties or ["cardiology", "cath_lab"],
        "diversion_declared":     diversion,
        "er_occupancy_pct":       er_occ,
        "congestion_factor":      congestion,
        "cascade_risk":           cascade,
        "is_mandatory":           mandatory,
    }
def _mk_hospital(
    hid:         str   = "H01",
    on_div:      bool  = False,
    er_occ:      float = 0.65,
    icu_avail:   int   = 8,
    specialties: Optional[List[str]] = None,
    cath_lab:    bool  = True,
    is_l1:       bool  = False,
    stroke_ctr:  bool  = False,
    eta_min:     float = 12.0,
    proficiency: float = 0.85,
) -> Dict[str, Any]:
    return {
        "on_diversion":       on_div,
        "er_occupancy_pct":   er_occ,
        "icu_beds_available": icu_avail,
        "specialties":        specialties or ["cardiology", "cath_lab", "emergency"],
        "has_cath_lab":       cath_lab,
        "is_level1_trauma":   is_l1,
        "is_stroke_centre":   stroke_ctr,
        "estimated_travel_min": eta_min,
        "specialty_proficiency": proficiency,
    }
def _mk_gi(
    triggers:        Optional[List[Dict]] = None,
    reroute_actions: Optional[List[Dict]] = None,
    hospital_states: Optional[Dict[str, Dict]] = None,
    episode_id:      str = "ep-t5-001",
    n_patients:      int = 3,
    p1_patients:     int = 2,
    episode_steps:   int = 20,
    observation_log: Optional[List[Dict]] = None,
    extra_ledger:    Optional[Dict] = None,
) -> GraderInput:
    action_log: List[ActionLogEntry] = []
    for idx, ra in enumerate(reroute_actions or []):
        action_log.append(ActionLogEntry(
            step        = ra.get("step", 5),
            action_type = ra.get("action_type", "reroute"),
            action_data = {
                "unit_id":             ra.get("unit_id", "AMB-M001"),
                "from_hospital_id":    ra.get("from_hospital_id", "H03"),
                "to_hospital_id":      ra.get("to_hospital_id", "H01"),
                "reason":              ra.get("reason", "hospital_diverted"),
                "travel_time_min":     ra.get("travel_time_min", 12.0),
                "pre_alert_sent":      ra.get("pre_alert_sent", False),
                "unit_status":         ra.get("unit_status", "transporting"),
                "patient_severity":    ra.get("patient_severity", "P1"),
                "condition_code":      ra.get("condition_code", "stemi_anterior"),
                "congestion_hotspots_avoided": ra.get("hotspots_avoided", []),
                "own_fleet_available_count":   ra.get("fleet_available", 5),
            },
        ))
    ledger: Dict[str, Any] = {
        "reroute_triggers":   triggers or [],
        "fleet_size":         16,
        "n_reroute_events":   len(triggers or []),
        "diversion_events":   sum(1 for t in (triggers or [])
                                  if t.get("diversion_declared", False)),
        "congestion_spikes":  sum(1 for t in (triggers or [])
                                  if t.get("trigger_type") == "traffic_congestion_spike"),
        "patient_summaries":  [
            {
                "patient_id":        f"P-{i+1:03d}",
                "severity":          "P1" if i < p1_patients else "P2",
                "condition_code":    "stemi_anterior",
                "phase_at_episode_end": "treated",
                "weight":            1.0,
                "weighted_reward":   0.80,
            }
            for i in range(n_patients)
        ],
    }
    if extra_ledger:
        ledger.update(extra_ledger)
    return GraderInput(
        task_id           = TASK_ID,
        episode_id        = episode_id,
        seed              = TASK_SEED,
        action_log        = action_log,
        episode_ledger    = ledger,
        observation_log   = observation_log or [],
        episode_steps     = episode_steps,
        total_patients    = n_patients,
        p1_patients       = p1_patients,
        final_hospital_state = hospital_states or {
            "H01": _mk_hospital("H01"),
            "H02": _mk_hospital("H02", er_occ=0.72, cath_lab=False, eta_min=18.0),
            "H03": _mk_hospital("H03", on_div=True, er_occ=0.97, eta_min=8.0),
            "H04": _mk_hospital("H04", er_occ=0.88, eta_min=14.0),
        },
    )
def _run_self_tests() -> None:
    grader   = Task5Grader()
    failures: List[str] = []
    def chk(name: str, cond: bool, msg: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]: {msg}")
    gi1 = _mk_gi(
        triggers = [_mk_trigger()],
        reroute_actions = [{
            "step": 6, "unit_id": "AMB-M001",
            "from_hospital_id": "H03", "to_hospital_id": "H01",
            "reason": "hospital_diverted", "travel_time_min": 12.0,
            "pre_alert_sent": True, "unit_status": "transporting",
            "patient_severity": "P1", "condition_code": "stemi_anterior",
        }],
        episode_id = "ep-t5-001",
    )
    r1 = grader.grade(gi1)
    chk("T1_range",    0.0 <= r1.final_score <= 1.0, f"score={r1.final_score}")
    chk("T1_baseline", r1.final_score >= TASK_BASELINE,
        f"score={r1.final_score:.4f} < baseline={TASK_BASELINE}")
    chk("T1_components", len(r1.components) >= 2)
    gi2 = _mk_gi(
        triggers        = [_mk_trigger()],
        reroute_actions = [{
            "step": 6, "unit_id": "AMB-M001",
            "from_hospital_id": "H03", "to_hospital_id": "H01",
            "reason": "hospital_diverted", "travel_time_min": 12.0,
            "pre_alert_sent": True, "unit_status": "transporting",
            "patient_severity": "P1", "condition_code": "stemi_anterior",
        }],
        episode_id = "ep-t5-002",
    )
    r2 = grader.grade(gi2)
    chk("T2_determinism",
        abs(r1.final_score - r2.final_score) < 1e-9,
        f"{r1.final_score} ≠ {r2.final_score}")
    gi3 = _mk_gi(
        triggers        = [_mk_trigger()],
        reroute_actions = [],
        episode_id      = "ep-t5-003",
    )
    r3 = grader.grade(gi3)
    chk("T3_missed_lower", r3.final_score < r1.final_score,
        f"Missed {r3.final_score:.4f} should be < perfect {r1.final_score:.4f}")
    chk("T3_missed_penalty",
        any("missed_p1_reroute" in p.name for p in r3.penalties),
        "Missing P1 reroute penalty")
    chk("T3_range", 0.0 <= r3.final_score <= 1.0)
    gi4 = _mk_gi(
        triggers        = [_mk_trigger()],
        reroute_actions = [{"step":6,"unit_id":"AMB-M001","from_hospital_id":"H01",
                            "to_hospital_id":"H03",   
                            "travel_time_min":8.0,
                            "unit_status":"transporting","patient_severity":"P1"}],
        episode_id = "ep-t5-004",
    )
    r4 = grader.grade(gi4)
    chk("T4_diverted_penalty",
        any("reroute_to_diverted" in p.name for p in r4.penalties),
        "Missing diverted destination penalty")
    chk("T4_diverted_lower", r4.final_score < r1.final_score)
    chk("T4_range", 0.0 <= r4.final_score <= 1.0)
    gi5 = _mk_gi(
        triggers        = [_mk_trigger()],
        reroute_actions = [{"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H01","travel_time_min":12.0,
                            "unit_status":"on_scene",  
                            "patient_severity":"P1"}],
        episode_id = "ep-t5-005",
    )
    r5 = grader.grade(gi5)
    chk("T5_on_scene_penalty",
        any("reroute_on_scene" in p.name for p in r5.penalties),
        "Missing on_scene reroute penalty")
    chk("T5_range", 0.0 <= r5.final_score <= 1.0)
    gi6 = _mk_gi(
        triggers        = [],   
        reroute_actions = [{"step":5,"unit_id":"AMB-M001","from_hospital_id":"H01",
                            "to_hospital_id":"H02","travel_time_min":18.0,
                            "unit_status":"transporting"}],
        episode_id = "ep-t5-006",
    )
    r6 = grader.grade(gi6)
    chk("T6_fp_lower", r6.final_score < r1.final_score,
        f"FP {r6.final_score:.4f} should be < perfect {r1.final_score:.4f}")
    chk("T6_range", 0.0 <= r6.final_score <= 1.0)
    gi7a = _mk_gi(
        triggers        = [_mk_trigger(step=5)],
        reroute_actions = [{"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H01","travel_time_min":12.0,
                            "unit_status":"transporting"}],
        episode_id = "ep-t5-007a",
    )
    gi7b = _mk_gi(
        triggers        = [_mk_trigger(step=5)],
        reroute_actions = [{"step":14,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H01","travel_time_min":12.0,
                            "unit_status":"transporting"}],
        episode_id = "ep-t5-007b",
    )
    r7a = grader.grade(gi7a)
    r7b = grader.grade(gi7b)
    chk("T7_late_lower", r7a.final_score >= r7b.final_score,
        f"On-time {r7a.final_score:.4f} should ≥ late {r7b.final_score:.4f}")
    chk("T7a_range", 0.0 <= r7a.final_score <= 1.0)
    chk("T7b_range", 0.0 <= r7b.final_score <= 1.0)
    multi_triggers = [
        _mk_trigger("TRG-001", units=["AMB-M001"], step=5),
        _mk_trigger("TRG-002", trigger_type="traffic_congestion_spike",
                    units=["AMB-A003"], step=8,
                    orig_hospital="H04", rec_hospital="H02",
                    conditions=["copd_exacerbation"], severities=["P2"],
                    specialties=["pulmonology"], diversion=False,
                    er_occ=0.72, congestion=0.40),
    ]
    gi8 = _mk_gi(
        triggers = multi_triggers,
        reroute_actions = [
            {"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03",
             "to_hospital_id":"H01","travel_time_min":12.0,
             "unit_status":"transporting","patient_severity":"P1"},
        ],
        episode_id = "ep-t5-008",
    )
    r8 = grader.grade(gi8)
    chk("T8_range",    0.0 <= r8.final_score <= 1.0)
    chk("T8_status",   r8.status in (GraderStatus.SUCCESS, GraderStatus.PARTIAL))
    chk("T8_n_triggers", r8.extra.get("n_triggers", 0) == 2)
    gi9 = _mk_gi(
        triggers        = [_mk_trigger()],
        reroute_actions = [{"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H01","travel_time_min":12.0,
                            "pre_alert_sent":True,"unit_status":"transporting",
                            "patient_severity":"P1","condition_code":"stemi_anterior"}],
        episode_id = "ep-t5-009",
    )
    r9 = grader.grade(gi9)
    chk("T9_protocol_bonus", r9.extra.get("protocol_bonus", 0) > 0,
        f"Expected protocol bonus, got {r9.extra.get('protocol_bonus')}")
    chk("T9_range", 0.0 <= r9.final_score <= 1.0)
    gi10 = _mk_gi(
        triggers        = [_mk_trigger(cascade=True, step=5)],
        reroute_actions = [{"step":4,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H01","travel_time_min":12.0,
                            "unit_status":"transporting","patient_severity":"P1"}],
        episode_id = "ep-t5-010",
    )
    r10 = grader.grade(gi10)
    chk("T10_cascade_range", 0.0 <= r10.final_score <= 1.0)
    ts_detail = r10.extra.get("time_saved_detail", {})
    chk("T10_cascade_score", ts_detail.get("cascade_score", 0) > 0.5,
        f"Cascade score should be >0.5 for pre-emptive reroute, "
        f"got {ts_detail.get('cascade_score', 0):.3f}")
    gi11 = _mk_gi(
        triggers = [_mk_trigger(
            "TRG-CNG", trigger_type="traffic_congestion_spike",
            diversion=False, er_occ=0.70, congestion=0.50
        )],
        reroute_actions = [{"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H01","travel_time_min":12.0,
                            "unit_status":"transporting",
                            "hotspots_avoided":["Z05","Z06"]}],
        episode_id = "ep-t5-011",
    )
    r11 = grader.grade(gi11)
    chk("T11_range", 0.0 <= r11.final_score <= 1.0)
    d = r1.as_dict()
    chk("T12_dict", all(k in d for k in ("final_score","components","penalties","notes")))
    chk("T12_json", len(r1.as_json()) > 200)
    chk("T12_summary", TASK_ID in r1.summary_line())
    gi13 = _mk_gi(triggers=[], reroute_actions=[], episode_id="ep-t5-013")
    r13  = grader.grade(gi13)
    chk("T13_clean_range", 0.0 <= r13.final_score <= 1.0)
    chk("T13_no_penalties", len(r13.penalties) == 0,
        f"Clean episode should have no penalties, got {len(r13.penalties)}")
    gi14 = _mk_gi(
        triggers = [_mk_trigger(
            "TRG-SAT", trigger_type="er_near_saturation",
            diversion=False, er_occ=0.87, mandatory=False
        )],
        reroute_actions = [{"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03",
                            "to_hospital_id":"H02","travel_time_min":18.0,
                            "unit_status":"transporting"}],
        episode_id = "ep-t5-014",
    )
    r14 = grader.grade(gi14)
    chk("T14_range", 0.0 <= r14.final_score <= 1.0)
    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task5Grader self-test: {len(failures)} failure(s):\n" +
            "\n".join(failures)
        )
    logger.info("Task5Grader self-test PASSED (14 test cases).")
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task5Grader self-test FAILED at import: %s", _e)
    raise
if __name__ == "__main__":
    import json as _json
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 72)
    print("EMERGI-ENV  ·  Task5Grader  ·  Dynamic Rerouting Demo")
    print("=" * 72)
    grader = Task5Grader()
    HOSPITALS = {
        "H01_KEM": _mk_hospital("H01_KEM", er_occ=0.62, cath_lab=True, is_l1=False,
                                 specialties=["cardiology","cath_lab","neurology","trauma"],
                                 eta_min=12.0, proficiency=0.92),
        "H02_RUBY": _mk_hospital("H02_RUBY", er_occ=0.78, cath_lab=True,
                                  specialties=["cardiology","cath_lab","orthopaedics"],
                                  eta_min=8.0, proficiency=0.80),
        "H03_SASOON": _mk_hospital("H03_SASOON", on_div=True, er_occ=0.97,
                                    specialties=["trauma","general_surgery"],
                                    eta_min=6.0, proficiency=0.70),
        "H04_DIST": _mk_hospital("H04_DIST", er_occ=0.88,
                                   specialties=["general_medicine","emergency"],
                                   cath_lab=False, eta_min=4.0, proficiency=0.50),
        "H05_TRAUMA": _mk_hospital("H05_TRAUMA", er_occ=0.60, is_l1=True,
                                    specialties=["trauma","emergency_surgery","orthopaedics"],
                                    cath_lab=False, eta_min=22.0, proficiency=0.90),
    }
    TRIGGERS = {
        "diversion": _mk_trigger(
            "TRG-DIV", "hospital_diversion",
            step=5, units=["AMB-M001", "AMB-M002"],
            orig_hospital="H03_SASOON", rec_hospital="H01_KEM",
            rec_eta=12.0, orig_eta=6.0,
            conditions=["stemi_anterior","polytrauma_blunt"],
            severities=["P1","P1"],
            specialties=["cardiology","cath_lab"],
            diversion=True, er_occ=0.97,
        ),
        "congestion": _mk_trigger(
            "TRG-CNG", "traffic_congestion_spike",
            step=8, units=["AMB-A005"],
            orig_hospital="H04_DIST", rec_hospital="H01_KEM",
            rec_eta=14.0, orig_eta=10.0,
            conditions=["ischaemic_stroke"],
            severities=["P1"],
            specialties=["neurology","stroke_unit"],
            diversion=False, er_occ=0.75, congestion=0.55,
        ),
        "cascade": _mk_trigger(
            "TRG-CAS", "hospital_diversion",
            step=10, units=["AMB-B003"],
            orig_hospital="H02_RUBY", rec_hospital="H01_KEM",
            rec_eta=12.0, orig_eta=8.0,
            conditions=["copd_exacerbation"],
            severities=["P2"],
            specialties=["pulmonology"],
            diversion=False, er_occ=0.88,
            cascade=True, mandatory=False,
        ),
    }
    scenarios = [
        ("Perfect: both P1 units rerouted on time, pre-alert",
         [TRIGGERS["diversion"], TRIGGERS["congestion"]],
         [
             {"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H01_KEM","travel_time_min":12.0,
              "pre_alert_sent":True,"unit_status":"transporting",
              "patient_severity":"P1","condition_code":"stemi_anterior"},
             {"step":6,"unit_id":"AMB-M002","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H01_KEM","travel_time_min":12.0,
              "pre_alert_sent":True,"unit_status":"transporting",
              "patient_severity":"P1","condition_code":"polytrauma_blunt"},
             {"step":9,"unit_id":"AMB-A005","from_hospital_id":"H04_DIST",
              "to_hospital_id":"H01_KEM","travel_time_min":14.0,
              "unit_status":"transporting","patient_severity":"P1",
              "condition_code":"ischaemic_stroke",
              "hotspots_avoided":["Z05","Z07"]},
         ]),
        ("Good: rerouted but to second-best hospital",
         [TRIGGERS["diversion"]],
         [
             {"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H02_RUBY","travel_time_min":8.0,  
              "unit_status":"transporting","patient_severity":"P1"},
             {"step":6,"unit_id":"AMB-M002","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H02_RUBY","travel_time_min":8.0,
              "unit_status":"transporting","patient_severity":"P1"},
         ]),
        ("Disaster: rerouted back to diverted hospital",
         [TRIGGERS["diversion"]],
         [
             {"step":6,"unit_id":"AMB-M001","from_hospital_id":"H01_KEM",
              "to_hospital_id":"H03_SASOON","travel_time_min":6.0,  
              "unit_status":"transporting","patient_severity":"P1"},
         ]),
        ("Missed: no reroute issued for any P1 trigger",
         [TRIGGERS["diversion"], TRIGGERS["congestion"]],
         []),
        ("Late response: rerouted at step 15 (trigger step 5)",
         [TRIGGERS["diversion"]],
         [
             {"step":15,"unit_id":"AMB-M001","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H01_KEM","travel_time_min":12.0,
              "unit_status":"transporting","patient_severity":"P1"},
         ]),
        ("On-scene illegal reroute",
         [TRIGGERS["diversion"]],
         [
             {"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H01_KEM","travel_time_min":12.0,
              "unit_status":"on_scene",  
              "patient_severity":"P1"},
         ]),
        ("Pre-emptive cascade reroute (before trigger)",
         [TRIGGERS["cascade"]],
         [
             {"step":8,"unit_id":"AMB-B003","from_hospital_id":"H02_RUBY",
              "to_hospital_id":"H01_KEM","travel_time_min":12.0,  
              "unit_status":"transporting","patient_severity":"P2"},
         ]),
        ("False positive: reroute with no trigger",
         [],
         [
             {"step":5,"unit_id":"AMB-M001","from_hospital_id":"H01_KEM",
              "to_hospital_id":"H04_DIST","travel_time_min":4.0,
              "unit_status":"transporting"},
         ]),
        ("Clean episode: no triggers, no reroutes",
         [], []),
        ("Multi-trigger partial: first handled, second missed",
         [TRIGGERS["diversion"], TRIGGERS["congestion"]],
         [
             {"step":6,"unit_id":"AMB-M001","from_hospital_id":"H03_SASOON",
              "to_hospital_id":"H01_KEM","travel_time_min":12.0,
              "pre_alert_sent":True,"unit_status":"transporting","patient_severity":"P1"},
         ]),
    ]
    results_all = []
    for name, scenario_triggers, scenario_reroutes in scenarios:
        gi = _mk_gi(
            triggers        = scenario_triggers,
            reroute_actions = scenario_reroutes,
            hospital_states = HOSPITALS,
            episode_id      = f"demo_{name[:20].replace(' ','_')}",
        )
        res = grader.grade(gi)
        results_all.append(res)
        beat = "✓" if res.beats_baseline else "✗"
        print(f"\n  [{beat}] {name}")
        print(f"       Score={res.final_score:.4f}  base={TASK_BASELINE:.2f}  "
              f"Δ={res.score_delta_vs_baseline:+.4f}  status={res.status.value}")
        print(f"       Triggers={res.extra.get('n_triggers',0)}  "
              f"Reroutes={res.extra.get('n_agent_reroutes',0)}  "
              f"Proto_bonus={res.extra.get('protocol_bonus',0):.3f}")
        for c in res.components:
            if c.name == "protocol_compliance":
                continue
            bar = "█" * int(c.raw_score * 20)
            print(f"         {c.name:<24} {c.raw_score:.4f} × {c.weight:.2f} "
                  f"= {c.weighted:.4f}  {bar}")
        if res.penalties:
            print(f"       Penalties ({len(res.penalties)}):")
            for p in res.penalties[:3]:
                print(f"         {p.name:<36} {p.amount:+.4f}  {p.reason[:45]}")
        outcomes = res.extra.get("outcome_counts", {})
        if outcomes:
            print(f"       Outcomes: {outcomes}")
        if res.notes:
            for n in res.notes[:2]:
                print(f"         ℹ  {n[:70]}")
    print("\n" + "=" * 72)
    beats = sum(1 for r in results_all if r.beats_baseline)
    print(f"  {beats}/{len(results_all)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 72)
    print("\n✅  Task5Grader demo complete.")