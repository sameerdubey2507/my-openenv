from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
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
logger = logging.getLogger("emergi_env.graders.task7")
TASK_ID   = "task7_mci_start"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]
W_START_ACCURACY        = 0.40
W_IMMEDIATE_RESPONSE    = 0.25
W_HOSPITAL_DISTRIBUTION = 0.20
W_SCENE_COMMAND         = 0.10
W_MUTUAL_AID_MANAGEMENT = 0.05
assert abs(
    W_START_ACCURACY + W_IMMEDIATE_RESPONSE + W_HOSPITAL_DISTRIBUTION +
    W_SCENE_COMMAND + W_MUTUAL_AID_MANAGEMENT - 1.0
) < 1e-9, "Task7 component weights must sum to 1.0"
HP_IMMEDIATE_AS_EXPECTANT  = 0.50   
HP_IMMEDIATE_UNRESPONDED   = 0.25
HP_EXPECTANT_OVER_TRIAGED  = 0.20
HP_P1_TO_DIVERTED_HOSPITAL = 0.15
HP_NO_MULTIAGENCY_COORD    = 0.10
PB_FAST_IMMEDIATE_DISPATCH  = 0.03
PB_NO_IMMEDIATE_TO_DIVERTED = 0.02
PB_EVEN_HOSPITAL_SPREAD     = 0.02
PB_PROACTIVE_MUTUAL_AID     = 0.02
PB_EXPECTANT_WITHHELD       = 0.02
PB_PRE_ALERT_SENT           = 0.01
PROTOCOL_BONUS_CAP          = 0.12
FAST_DISPATCH_STEPS           = 3     
MIN_HOSPITALS_FOR_SPREAD      = 3     
GINI_SPREAD_THRESHOLD         = 0.30  
FLEET_SATURATION_FRACTION     = 0.85  
MUTUAL_AID_PROACTIVE_HEADROOM = 3     
TRIAGE_TAGS = frozenset({"Immediate", "Delayed", "Minimal", "Expectant"})
UNIT_RANK:  Dict[str, int] = {"BLS": 1, "ALS": 2, "MICU": 3}
MICU_MANDATORY_CONDITIONS: FrozenSet[str] = frozenset({
    "cardiac_arrest", "cardiac_arrest_vf", "cardiac_arrest_pea",
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "polytrauma_blunt", "polytrauma_penetrating", "severe_tbi",
    "eclampsia", "pulmonary_oedema", "septic_shock",
    "major_burns_40pct", "anaphylaxis_refractory", "massive_pe",
})
class TriageTag(str, Enum):
    IMMEDIATE  = "Immediate"
    DELAYED    = "Delayed"
    MINIMAL    = "Minimal"
    EXPECTANT  = "Expectant"
def _compute_start_tag(
    respirations:   int,
    pulse:          int,
    mental_status:  str,
) -> TriageTag:
    ms = mental_status.lower().strip()
    if respirations == 0:
        return TriageTag.EXPECTANT
    if respirations > 30 or respirations < 6:
        return TriageTag.IMMEDIATE
    if pulse == 0:
        return TriageTag.EXPECTANT
    if pulse > 120 or pulse < 40:
        return TriageTag.IMMEDIATE
    if ms in ("unresponsive", "posturing", "eyes_only", "decerebrate"):
        return TriageTag.IMMEDIATE
    if ms in ("confused", "verbal", "disoriented", "agitated"):
        return TriageTag.DELAYED
    if ms in ("alert", "oriented", "follows_commands", "ambulatory"):
        return TriageTag.MINIMAL
    return TriageTag.DELAYED
def _triage_severity(tag: TriageTag) -> int:
    return {
        TriageTag.IMMEDIATE: 4,
        TriageTag.DELAYED:   3,
        TriageTag.MINIMAL:   2,
        TriageTag.EXPECTANT: 1,
    }.get(tag, 0)
@dataclass
class VictimRecord:
    victim_id:          str
    respirations:       int
    pulse:              int
    mental_status:      str
    condition_code:     str
    zone_id:            str
    correct_tag:        TriageTag = field(init=False)
    requires_police:    bool = False
    requires_fire:      bool = False
    is_micu_mandatory:  bool = field(init=False)
    agent_tag:          Optional[TriageTag] = None
    tag_step:           Optional[int]       = None
    dispatched:         bool                = False
    dispatch_step:      Optional[int]       = None
    dispatched_unit:    Optional[str]       = None   
    hospital_id:        Optional[str]       = None
    hospital_diverted:  bool                = False
    pre_alert_sent:     bool                = False
    final_survival_prob: float              = 0.70
    def __post_init__(self) -> None:
        self.correct_tag       = _compute_start_tag(
            self.respirations, self.pulse, self.mental_status
        )
        self.is_micu_mandatory = self.condition_code in MICU_MANDATORY_CONDITIONS
    @property
    def tag_correct(self) -> Optional[bool]:
        if self.agent_tag is None:
            return None
        return self.agent_tag == self.correct_tag
    @property
    def is_critical_error(self) -> bool:
        return (
            self.correct_tag == TriageTag.IMMEDIATE
            and self.agent_tag == TriageTag.EXPECTANT
        )
    @property
    def is_over_triage_waste(self) -> bool:
        return (
            self.correct_tag == TriageTag.EXPECTANT
            and self.agent_tag == TriageTag.IMMEDIATE
            and self.dispatched_unit == "MICU"
        )
    @property
    def dispatch_latency_steps(self) -> Optional[int]:
        if self.tag_step is None or self.dispatch_step is None:
            return None
        return self.dispatch_step - self.tag_step
    @property
    def needs_multi_agency(self) -> bool:
        return self.requires_police or self.requires_fire
    def to_dict(self) -> Dict[str, Any]:
        return {
            "victim_id":         self.victim_id,
            "condition":         self.condition_code,
            "correct_tag":       self.correct_tag.value,
            "agent_tag":         self.agent_tag.value if self.agent_tag else None,
            "tag_correct":       self.tag_correct,
            "is_critical_error": self.is_critical_error,
            "dispatched":        self.dispatched,
            "dispatch_step":     self.dispatch_step,
            "unit_type":         self.dispatched_unit,
            "hospital_id":       self.hospital_id,
            "hospital_diverted": self.hospital_diverted,
            "pre_alert_sent":    self.pre_alert_sent,
            "final_survival":    round(self.final_survival_prob, 4),
        }
@dataclass
class MCISceneState:
    victims:               List[VictimRecord]
    fleet_size:            int
    hospital_diversions:   Set[str]   
    multi_agency_events:   int        
    agency_coordinated:    bool       
    mutual_aid_requested:  bool
    mutual_aid_steps:      List[int]  
    saturation_step:       Optional[int]   
    @property
    def immediate_victims(self) -> List[VictimRecord]:
        return [v for v in self.victims if v.correct_tag == TriageTag.IMMEDIATE]
    @property
    def delayed_victims(self) -> List[VictimRecord]:
        return [v for v in self.victims if v.correct_tag == TriageTag.DELAYED]
    @property
    def minimal_victims(self) -> List[VictimRecord]:
        return [v for v in self.victims if v.correct_tag == TriageTag.MINIMAL]
    @property
    def expectant_victims(self) -> List[VictimRecord]:
        return [v for v in self.victims if v.correct_tag == TriageTag.EXPECTANT]
    @property
    def tagged_victims(self) -> List[VictimRecord]:
        return [v for v in self.victims if v.agent_tag is not None]
    @property
    def total(self) -> int:
        return len(self.victims)
class STARTAccuracyScorer:
    SCORE_MATRIX: Dict[Tuple[str, str], float] = {
        ("Immediate", "Immediate"): 1.00,
        ("Immediate", "Delayed"):   0.45,
        ("Immediate", "Minimal"):   0.15,
        ("Immediate", "Expectant"): 0.00,  
        ("Delayed",   "Immediate"): 0.75,
        ("Delayed",   "Delayed"):   1.00,
        ("Delayed",   "Minimal"):   0.50,
        ("Delayed",   "Expectant"): 0.05,
        ("Minimal",   "Immediate"): 0.85,
        ("Minimal",   "Delayed"):   0.90,
        ("Minimal",   "Minimal"):   1.00,
        ("Minimal",   "Expectant"): 0.10,
        ("Expectant", "Immediate"): 0.20,
        ("Expectant", "Delayed"):   0.40,
        ("Expectant", "Minimal"):   0.60,
        ("Expectant", "Expectant"): 1.00,
    }
    TAG_WEIGHT: Dict[str, float] = {
        "Immediate": 3.00,
        "Delayed":   2.00,
        "Minimal":   1.00,
        "Expectant": 1.50,   
    }
    @classmethod
    def score(
        cls,
        scene:  MCISceneState,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        tagged = scene.tagged_victims
        if not tagged:
            result.add_note("STARTAccuracy: no victims tagged — score 0.0")
            return 0.0, {"reason": "no_tags", "tagged_count": 0}
        total_weight = 0.0
        weighted_sum = 0.0
        per_victim:   List[Dict] = []
        correct_count = 0
        tag_dist: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for v in tagged:
            ct  = v.correct_tag.value
            at  = v.agent_tag.value if v.agent_tag else "none"
            raw = cls.SCORE_MATRIX.get((ct, at), 0.10)
            w   = cls.TAG_WEIGHT.get(ct, 1.0)
            weighted_sum += raw * w
            total_weight += w
            tag_dist[ct][at] += 1
            if v.tag_correct:
                correct_count += 1
            per_victim.append({
                "victim_id": v.victim_id,
                "correct":   ct,
                "assigned":  at,
                "score":     round(raw, 4),
                "weight":    round(w, 2),
            })
        untagged_immediates = sum(
            1 for v in scene.immediate_victims if v.agent_tag is None
        )
        if untagged_immediates > 0:
            penalty_ratio = untagged_immediates / max(1, len(scene.immediate_victims))
            weighted_sum  = max(0, weighted_sum - penalty_ratio * total_weight * 0.40)
            result.add_note(
                f"STARTAccuracy: {untagged_immediates} Immediate victim(s) never tagged "
                f"(−{penalty_ratio * 0.40:.3f} penalty factor)"
            )
        raw = weighted_sum / total_weight if total_weight > 0 else 0.0
        tag_coverage = len(tagged) / scene.total if scene.total > 0 else 0.0
        if tag_coverage < 0.60:
            coverage_factor = tag_coverage / 0.60
            raw = raw * coverage_factor
            result.add_note(
                f"STARTAccuracy: only {tag_coverage:.1%} victims tagged → "
                f"coverage discount applied"
            )
        result.add_note(
            f"STARTAccuracy: {correct_count}/{len(tagged)} correct tags "
            f"({tag_coverage:.1%} coverage) → weighted score {raw:.4f}"
        )
        return ScoringUtils.clamp(raw), {
            "total_victims":         scene.total,
            "tagged_count":          len(tagged),
            "correct_count":         correct_count,
            "tag_coverage":          round(tag_coverage, 4),
            "untagged_immediates":   untagged_immediates,
            "tag_distribution":      dict(tag_dist),
            "per_victim_sample":     per_victim[:20],  
        }
class ImmediateResponseScorer:
    LATENCY_BANDS: List[Tuple[int, float]] = [
        (1, 1.00), (2, 0.90), (3, 0.75),
        (4, 0.55), (5, 0.38), (6, 0.22),
    ]
    @classmethod
    def score(
        cls,
        scene:  MCISceneState,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        immediates = scene.immediate_victims
        if not immediates:
            result.add_note("ImmediateResponse: no Immediate victims — full credit")
            return 1.0, {"reason": "no_immediate_victims"}
        dispatch_rate = sum(1 for v in immediates if v.dispatched) / len(immediates)
        if dispatch_rate == 0:
            result.add_note("ImmediateResponse: zero Immediate victims dispatched")
            return 0.0, {"dispatch_rate": 0.0, "total_immediate": len(immediates)}
        speed_scores:    List[float] = []
        unit_scores:     List[float] = []
        survival_scores: List[float] = []
        for v in immediates:
            if not v.dispatched:
                speed_scores.append(0.0)
                unit_scores.append(0.0)
                survival_scores.append(0.0)
                continue
            latency = v.dispatch_latency_steps
            if latency is None:
                speed_scores.append(0.40)
            else:
                sp = 0.10
                for threshold, val in cls.LATENCY_BANDS:
                    if latency <= threshold:
                        sp = val
                        break
                speed_scores.append(sp)
            utype = (v.dispatched_unit or "BLS").upper()
            rank  = UNIT_RANK.get(utype, 1)
            if v.is_micu_mandatory:
                u_score = {3: 1.00, 2: 0.55, 1: 0.10}.get(rank, 0.10)
            else:
                u_score = {3: 1.00, 2: 1.00, 1: 0.45}.get(rank, 0.45)
            unit_scores.append(u_score)
            s_score = ScoringUtils.survival_probability_score(
                v.final_survival_prob, p_optimal=0.90, p_worst=0.05
            )
            survival_scores.append(s_score)
        n = len(immediates)
        raw = ScoringUtils.clamp(
            0.35 * dispatch_rate
            + 0.30 * (sum(speed_scores) / n)
            + 0.25 * (sum(unit_scores) / n)
            + 0.10 * (sum(survival_scores) / n)
        )
        result.add_note(
            f"ImmediateResponse: dispatch_rate={dispatch_rate:.2f}, "
            f"avg_speed={sum(speed_scores)/n:.3f}, "
            f"avg_unit={sum(unit_scores)/n:.3f} → {raw:.4f}"
        )
        return raw, {
            "total_immediate":    n,
            "dispatched_count":   sum(1 for v in immediates if v.dispatched),
            "dispatch_rate":      round(dispatch_rate, 4),
            "avg_speed_score":    round(sum(speed_scores) / n, 4),
            "avg_unit_score":     round(sum(unit_scores)  / n, 4),
            "avg_survival_score": round(sum(survival_scores) / n, 4),
        }
class HospitalDistributionScorer:
    MAX_SINGLE_HOSPITAL_FRACTION = 0.40
    TARGET_GINI                  = 0.30
    @classmethod
    def score(
        cls,
        scene:  MCISceneState,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        dispatched = [v for v in scene.victims if v.dispatched and v.hospital_id]
        if not dispatched:
            result.add_note("HospitalDistribution: no dispatched victims — score 0.0")
            return 0.0, {"reason": "no_dispatched_victims"}
        hosp_counts: Dict[str, int] = defaultdict(int)
        for v in dispatched:
            hosp_counts[v.hospital_id] += 1
        n_hospitals = len(hosp_counts)
        total       = len(dispatched)
        if n_hospitals >= 5:
            h_count_score = 1.00
        elif n_hospitals >= 4:
            h_count_score = 0.90
        elif n_hospitals >= MIN_HOSPITALS_FOR_SPREAD:
            h_count_score = 0.75
        elif n_hospitals == 2:
            h_count_score = 0.45
        else:
            h_count_score = 0.15
        max_frac       = max(hosp_counts.values()) / total
        max_load_score = ScoringUtils.clamp(1.0 - max(0, max_frac - 0.25) * 2.5)
        counts_sorted = sorted(hosp_counts.values())
        n = len(counts_sorted)
        if n == 1:
            gini = 1.0
        else:
            cum  = 0.0
            gini = 0.0
            for i, x in enumerate(counts_sorted):
                cum  += x
                gini += (2 * (i + 1) - n - 1) * x
            gini = gini / (n * sum(counts_sorted)) if sum(counts_sorted) > 0 else 1.0
            gini = abs(gini)
        gini_score = ScoringUtils.clamp(1.0 - gini / (GINI_SPREAD_THRESHOLD * 2))
        diverted_dispatches = sum(
            1 for v in dispatched if v.hospital_diverted
        )
        diversion_score = ScoringUtils.clamp(
            1.0 - diverted_dispatches / max(1, total) * 2.0
        )
        raw = ScoringUtils.clamp(
            0.30 * h_count_score
            + 0.30 * max_load_score
            + 0.25 * gini_score
            + 0.15 * diversion_score
        )
        result.add_note(
            f"HospitalDistribution: {n_hospitals} hospitals, "
            f"max_load={max_frac:.1%}, gini={gini:.3f}, "
            f"diverted_dispatches={diverted_dispatches} → {raw:.4f}"
        )
        return raw, {
            "hospitals_used":        n_hospitals,
            "total_dispatched":      total,
            "max_fraction":          round(max_frac, 4),
            "gini_coefficient":      round(gini, 4),
            "diverted_dispatches":   diverted_dispatches,
            "hospital_load":         dict(hosp_counts),
        }
class SceneCommandScorer:
    @classmethod
    def score(
        cls,
        scene:   MCISceneState,
        gi:      GraderInput,
        result:  GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        sub_scores: List[float] = []
        details:    Dict[str, Any] = {}
        if scene.multi_agency_events > 0:
            agency_score = 1.0 if scene.agency_coordinated else 0.10
            sub_scores.append(agency_score)
            details["multi_agency_events"] = scene.multi_agency_events
            details["agency_coordinated"]  = scene.agency_coordinated
            if not scene.agency_coordinated:
                result.add_note(
                    f"SceneCommand: {scene.multi_agency_events} trapped victims "
                    "but no multi-agency coordination issued"
                )
        else:
            sub_scores.append(1.0)
            details["multi_agency_events"] = 0
        comm_fail_steps = {
            int(obs.get("step", 0))
            for obs in gi.observation_log
            if any(
                not v
                for v in obs.get("comm_failures", {}).values()
            )
        } if gi.observation_log else set()
        noop_steps = {e.step for e in gi.action_log if e.action_type == "noop"}
        if comm_fail_steps:
            overlap = comm_fail_steps & noop_steps
            comm_mgmt_score = ScoringUtils.clamp(
                len(overlap) / max(1, len(comm_fail_steps))
            )
            sub_scores.append(comm_mgmt_score)
            details["comm_fail_steps"]   = len(comm_fail_steps)
            details["comm_ack_fraction"] = round(comm_mgmt_score, 4)
        else:
            sub_scores.append(1.0)
            details["comm_fail_steps"] = 0
        pre_alert_actions = sum(
            1 for e in gi.action_log
            if e.action_type in ("pre_alert", "escalate")
            and e.get("pre_alert_sent")
        )
        immediate_dispatched = sum(
            1 for v in scene.immediate_victims if v.dispatched
        )
        if immediate_dispatched > 0:
            pre_alert_rate  = min(1.0, pre_alert_actions / immediate_dispatched)
            pre_alert_score = ScoringUtils.clamp(0.50 + pre_alert_rate * 0.50)
        else:
            pre_alert_rate  = 0.0
            pre_alert_score = 0.70   
        sub_scores.append(pre_alert_score)
        details["pre_alert_rate"]  = round(pre_alert_rate, 4)
        details["pre_alert_count"] = pre_alert_actions
        escalations = [e for e in gi.action_log if e.action_type == "escalate"]
        esc_score   = ScoringUtils.clamp(0.60 + min(len(escalations), 5) * 0.08)
        sub_scores.append(esc_score)
        details["escalation_count"] = len(escalations)
        raw = ScoringUtils.clamp(
            0.35 * sub_scores[0]   
            + 0.25 * sub_scores[1]   
            + 0.25 * sub_scores[2]   
            + 0.15 * sub_scores[3]   
        )
        result.add_note(f"SceneCommand: composite {raw:.4f}")
        return raw, details
class MutualAidManagementScorer:
    OVER_REQUEST_UNIT_THRESHOLD = 6   
    @classmethod
    def score(
        cls,
        scene:   MCISceneState,
        gi:      GraderInput,
        result:  GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        aid_actions = [
            e for e in gi.action_log
            if e.action_type == "request_mutual_aid"
        ]
        total_requested = sum(
            int(e.get("units_requested", 1)) for e in aid_actions
        )
        total_victims   = scene.total
        fleet           = scene.fleet_size
        if not aid_actions:
            if total_victims > fleet * 1.5:
                result.add_note(
                    "MutualAid: large MCI but no mutual aid requested → under-resourced"
                )
                return 0.25, {"requested": 0, "victims": total_victims, "fleet": fleet}
            return 0.70, {"requested": 0, "reason": "not_needed"}
        proactive = False
        if scene.saturation_step is not None:
            for step in scene.mutual_aid_steps:
                if step < scene.saturation_step - MUTUAL_AID_PROACTIVE_HEADROOM:
                    proactive = True
                    break
        over_requested = total_requested > cls.OVER_REQUEST_UNIT_THRESHOLD
        needed_approx  = max(0, total_victims - fleet)
        over_margin    = max(0, total_requested - needed_approx)
        base = 0.70
        if proactive:
            base += 0.20
        if over_requested:
            base -= min(0.30, over_margin * 0.05)
            result.add_note(
                f"MutualAid: over-requested ({total_requested} units, "
                f"approx needed {needed_approx})"
            )
        result.add_note(
            f"MutualAid: {len(aid_actions)} requests, "
            f"{total_requested} units, proactive={proactive} → {base:.4f}"
        )
        return ScoringUtils.clamp(base), {
            "request_count":    len(aid_actions),
            "units_requested":  total_requested,
            "proactive":        proactive,
            "over_requested":   over_requested,
            "approx_needed":    needed_approx,
        }
class Task7PenaltyEngine:
    @staticmethod
    def apply(scene: MCISceneState, result: GraderResult) -> float:
        total = 0.0
        crit_errors = [v for v in scene.victims if v.is_critical_error]
        for v in crit_errors:
            result.add_penalty(PenaltyRecord(
                name=f"immediate_as_expectant_{v.victim_id[:16]}",
                amount=-HP_IMMEDIATE_AS_EXPECTANT,
                reason=(
                    f"Victim {v.victim_id} (R={v.respirations},P={v.pulse},"
                    f"M={v.mental_status}): Immediate incorrectly tagged Expectant"
                ),
                rule_ref="EMERGI-ENV Rule T7-H1 (Spec)",
            ))
            total -= HP_IMMEDIATE_AS_EXPECTANT
            result.critical_mismatches += 1
        immediate_unresponded = [
            v for v in scene.immediate_victims
            if v.agent_tag == TriageTag.DELAYED and not v.dispatched
        ]
        for v in immediate_unresponded:
            result.add_penalty(PenaltyRecord(
                name=f"immediate_unresponded_{v.victim_id[:16]}",
                amount=-HP_IMMEDIATE_UNRESPONDED,
                reason=(
                    f"Victim {v.victim_id}: Immediate tagged as Delayed "
                    "and never dispatched — left without care"
                ),
                rule_ref="EMERGI-ENV Rule T7-H2",
            ))
            total -= HP_IMMEDIATE_UNRESPONDED
        wasted = [v for v in scene.victims if v.is_over_triage_waste]
        for v in wasted:
            result.add_penalty(PenaltyRecord(
                name=f"expectant_over_triaged_{v.victim_id[:16]}",
                amount=-HP_EXPECTANT_OVER_TRIAGED,
                reason=(
                    f"Victim {v.victim_id}: Expectant incorrectly tagged Immediate, "
                    "MICU dispatched — scarce resource wasted"
                ),
                rule_ref="EMERGI-ENV Rule T7-H3",
            ))
            total -= HP_EXPECTANT_OVER_TRIAGED
        p1_diverted = [
            v for v in scene.immediate_victims
            if v.dispatched and v.hospital_diverted
        ]
        for v in p1_diverted:
            result.add_penalty(PenaltyRecord(
                name=f"p1_diverted_{v.victim_id[:16]}",
                amount=-HP_P1_TO_DIVERTED_HOSPITAL,
                reason=(
                    f"Immediate victim {v.victim_id} routed to diverted hospital "
                    f"{v.hospital_id} — delayed care"
                ),
                rule_ref="EMERGI-ENV Rule T7-H4",
            ))
            total -= HP_P1_TO_DIVERTED_HOSPITAL
        if scene.multi_agency_events > 0 and not scene.agency_coordinated:
            result.add_penalty(PenaltyRecord(
                name="no_multiagency_coordination",
                amount=-HP_NO_MULTIAGENCY_COORD,
                reason=(
                    f"{scene.multi_agency_events} trapped/extrication events "
                    "but no multi-agency coordination action issued"
                ),
                rule_ref="EMERGI-ENV Rule T7-H5",
            ))
            total -= HP_NO_MULTIAGENCY_COORD
        return total
class Task7BonusEngine:
    @staticmethod
    def apply(scene: MCISceneState, result: GraderResult) -> float:
        total = 0.0
        immediates = scene.immediate_victims
        if immediates:
            fast_dispatched = sum(
                1 for v in immediates
                if v.dispatched and (v.dispatch_latency_steps or 999) <= FAST_DISPATCH_STEPS
            )
            if fast_dispatched / len(immediates) >= 0.80:
                result.add_note(
                    f"Protocol bonus: {fast_dispatched}/{len(immediates)} Immediates "
                    f"dispatched within {FAST_DISPATCH_STEPS} steps "
                    f"(+{PB_FAST_IMMEDIATE_DISPATCH:.2f})"
                )
                total += PB_FAST_IMMEDIATE_DISPATCH
            no_diverted_imm = all(
                not v.hospital_diverted for v in immediates if v.dispatched
            )
            if no_diverted_imm and any(v.dispatched for v in immediates):
                result.add_note(
                    f"Protocol bonus: no Immediate victim routed to diverted hospital "
                    f"(+{PB_NO_IMMEDIATE_TO_DIVERTED:.2f})"
                )
                total += PB_NO_IMMEDIATE_TO_DIVERTED
        dispatched = [v for v in scene.victims if v.dispatched and v.hospital_id]
        if dispatched:
            hosp_counts = defaultdict(int)
            for v in dispatched:
                hosp_counts[v.hospital_id] += 1
            n = len(hosp_counts)
            if n >= MIN_HOSPITALS_FOR_SPREAD:
                counts_sorted = sorted(hosp_counts.values())
                total_pats    = sum(counts_sorted)
                gini          = 0.0
                for i, x in enumerate(counts_sorted):
                    gini += (2 * (i + 1) - n - 1) * x
                gini = abs(gini) / (n * total_pats) if total_pats > 0 else 1.0
                if gini < GINI_SPREAD_THRESHOLD:
                    result.add_note(
                        f"Protocol bonus: even hospital distribution "
                        f"(Gini={gini:.3f} < {GINI_SPREAD_THRESHOLD}) "
                        f"(+{PB_EVEN_HOSPITAL_SPREAD:.2f})"
                    )
                    total += PB_EVEN_HOSPITAL_SPREAD
        if scene.mutual_aid_requested and scene.saturation_step is not None:
            for step in scene.mutual_aid_steps:
                if step < scene.saturation_step - MUTUAL_AID_PROACTIVE_HEADROOM:
                    result.add_note(
                        f"Protocol bonus: mutual aid requested proactively at step {step} "
                        f"(saturation at {scene.saturation_step}) "
                        f"(+{PB_PROACTIVE_MUTUAL_AID:.2f})"
                    )
                    total += PB_PROACTIVE_MUTUAL_AID
                    break
        expectants = scene.expectant_victims
        if expectants:
            correctly_withheld = sum(
                1 for v in expectants
                if v.agent_tag == TriageTag.EXPECTANT and not v.dispatched
            )
            if correctly_withheld == len(expectants):
                result.add_note(
                    f"Protocol bonus: all {len(expectants)} Expectant victim(s) "
                    f"correctly withheld from dispatch "
                    f"(+{PB_EXPECTANT_WITHHELD:.2f})"
                )
                total += PB_EXPECTANT_WITHHELD
        if immediates:
            immediates_dispatched = [v for v in immediates if v.dispatched]
            if immediates_dispatched and all(v.pre_alert_sent for v in immediates_dispatched):
                result.add_note(
                    f"Protocol bonus: pre-alert sent for all {len(immediates_dispatched)} "
                    f"dispatched Immediate victims (+{PB_PRE_ALERT_SENT:.2f})"
                )
                total += PB_PRE_ALERT_SENT
        return min(total, PROTOCOL_BONUS_CAP)
def _parse_victims(gi: GraderInput) -> List[VictimRecord]:
    ledger   = gi.episode_ledger
    summaries = ledger.get("patient_summaries") or ledger.get("victim_summaries") or []
    victim_map: Dict[str, VictimRecord] = {}
    for ps in summaries:
        vid  = ps.get("patient_id") or ps.get("victim_id") or ps.get("incident_id") or ""
        if not vid:
            continue
        v = VictimRecord(
            victim_id       = vid,
            respirations    = int(ps.get("respirations", 16)),
            pulse           = int(ps.get("pulse", 80)),
            mental_status   = str(ps.get("mental_status", "alert")),
            condition_code  = str(ps.get("condition_code", "mci_victim")),
            zone_id         = str(ps.get("zone_id", "Z3")),
            requires_police = bool(ps.get("requires_police", False)),
            requires_fire   = bool(ps.get("requires_fire", False)),
        )
        v.final_survival_prob = float(ps.get("final_survival_prob", 0.70))
        victim_map[vid] = v
    for entry in gi.action_log:
        if entry.action_type != "tag":
            continue
        vid = entry.get("victim_id") or entry.get("incident_id") or entry.get("patient_id")
        if not vid:
            continue
        if vid not in victim_map:
            victim_map[vid] = VictimRecord(
                victim_id      = vid,
                respirations   = int(entry.get("respirations", 16)),
                pulse          = int(entry.get("pulse", 80)),
                mental_status  = str(entry.get("mental_status", "alert")),
                condition_code = str(entry.get("condition_code", "unknown")),
                zone_id        = str(entry.get("zone_id", "Z3")),
            )
        raw_tag = str(entry.get("triage_tag") or entry.get("tag") or "")
        for tt in TriageTag:
            if tt.value.lower() == raw_tag.lower():
                victim_map[vid].agent_tag = tt
                victim_map[vid].tag_step  = entry.step
                break
    for entry in gi.action_log:
        if entry.action_type != "dispatch":
            continue
        vid = entry.get("victim_id") or entry.get("incident_id") or entry.get("patient_id")
        if vid and vid in victim_map:
            v = victim_map[vid]
            v.dispatched        = True
            v.dispatch_step     = entry.step
            v.dispatched_unit   = (entry.get("unit_type") or "ALS").upper()
            v.hospital_id       = entry.get("hospital_id")
            v.pre_alert_sent    = bool(entry.get("pre_alert_sent", False))
    diverted_ids: Set[str] = set()
    for obs in (gi.observation_log or []):
        for hid, hdata in (obs.get("hospital_network") or {}).items():
            if hdata.get("on_diversion") or hdata.get("diverted"):
                diverted_ids.add(hid)
    for v in victim_map.values():
        if v.hospital_id and v.hospital_id in diverted_ids:
            v.hospital_diverted = True
    return list(victim_map.values())
def _parse_scene_state(gi: GraderInput, victims: List[VictimRecord]) -> MCISceneState:
    ledger = gi.episode_ledger
    diverted_ids: Set[str] = set()
    for obs in (gi.observation_log or []):
        for hid, hdata in (obs.get("hospital_network") or {}).items():
            if hdata.get("on_diversion") or hdata.get("diverted"):
                diverted_ids.add(hid)
    multi_agency_events = sum(1 for v in victims if v.needs_multi_agency)
    multi_agency_events += int(ledger.get("multi_agency_event_count", 0))
    agency_actions = [
        e for e in gi.action_log
        if e.action_type in ("coordinate_agencies", "multi_agency", "escalate")
        and e.get("agencies") or e.get("coordinate_agencies")
    ]
    agency_coordinated = len(agency_actions) > 0
    aid_actions = [e for e in gi.action_log if e.action_type == "request_mutual_aid"]
    mutual_aid_requested = len(aid_actions) > 0
    mutual_aid_steps     = [e.step for e in aid_actions]
    fleet_size = ledger.get("fleet_size", 10)
    saturation_step: Optional[int] = None
    for obs in (gi.observation_log or []):
        fleet_data   = obs.get("fleet_status", [])
        dispatched_n = sum(1 for u in fleet_data if u.get("status") == "dispatched")
        if fleet_size > 0 and dispatched_n / fleet_size >= FLEET_SATURATION_FRACTION:
            saturation_step = int(obs.get("step", 0))
            break
    return MCISceneState(
        victims              = victims,
        fleet_size           = fleet_size,
        hospital_diversions  = diverted_ids,
        multi_agency_events  = multi_agency_events,
        agency_coordinated   = agency_coordinated,
        mutual_aid_requested = mutual_aid_requested,
        mutual_aid_steps     = mutual_aid_steps,
        saturation_step      = saturation_step,
    )
def _synth_victims(n: int, rng_seed: int = 42) -> List[Dict[str, Any]]:
    import random
    rng = random.Random(rng_seed)
    out = []
    conditions = [
        "cardiac_arrest_vf", "polytrauma_blunt", "stemi_anterior",
        "ischaemic_stroke", "febrile_seizure", "minor_laceration",
        "mci_victim", "inhalation_injury",
    ]
    rpm_presets = [
        (0, 0, "unresponsive"),      
        (35, 130, "unresponsive"),   
        (8, 55, "confused"),         
        (16, 85, "confused"),        
        (18, 78, "alert"),           
        (20, 90, "follows_commands"),
        (12, 70, "disoriented"),     
        (32, 110, "agitated"),       
    ]
    for i in range(n):
        r, p, m = rng.choice(rpm_presets)
        out.append({
            "patient_id":          f"V{i+1:04d}",
            "respirations":        r + rng.randint(-1, 1),
            "pulse":               p + rng.randint(-5, 5),
            "mental_status":       m,
            "condition_code":      rng.choice(conditions),
            "zone_id":             "Z3",
            "requires_police":     rng.random() < 0.10,
            "requires_fire":       rng.random() < 0.10,
            "final_survival_prob": rng.uniform(0.50, 0.95),
        })
    return out
class Task7Grader(BaseGrader):
    TASK_ID         = TASK_ID
    TASK_SEED       = TASK_SEED
    TASK_BASELINE   = TASK_BASELINE
    TASK_DIFFICULTY = "hard"
    COMPONENT_WEIGHTS: Dict[str, float] = {
        "start_accuracy":        W_START_ACCURACY,
        "immediate_response":    W_IMMEDIATE_RESPONSE,
        "hospital_distribution": W_HOSPITAL_DISTRIBUTION,
        "scene_command":         W_SCENE_COMMAND,
        "mutual_aid_management": W_MUTUAL_AID_MANAGEMENT,
    }
    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(
                f"WARNING: episode seed {gi.seed} ≠ task seed {TASK_SEED}. "
                "Grading proceeds but determinism not guaranteed."
            )
        victims = _parse_victims(gi)
        if not victims:
            synth = gi.episode_ledger.get("patient_summaries", [])
            if not synth:
                result.status        = GraderStatus.INVALID_INPUT
                result.error_message = (
                    "episode_ledger.patient_summaries is empty. "
                    "Task 7 requires at least 20 victim records."
                )
                result.final_score = 0.0
                return
        scene = _parse_scene_state(gi, victims)
        result.extra["total_victims"]     = scene.total
        result.extra["immediate_count"]   = len(scene.immediate_victims)
        result.extra["delayed_count"]     = len(scene.delayed_victims)
        result.extra["minimal_count"]     = len(scene.minimal_victims)
        result.extra["expectant_count"]   = len(scene.expectant_victims)
        result.extra["tagged_count"]      = len(scene.tagged_victims)
        result.extra["dispatched_count"]  = sum(1 for v in victims if v.dispatched)
        if scene.total < 5:
            result.add_note(
                f"WARNING: only {scene.total} victims parsed — "
                "task 7 expects 20-40. Scores may be unreliable."
            )
        sa_score, sa_detail = STARTAccuracyScorer.score(scene, result)
        self._add_component(
            result, "start_accuracy", sa_score, W_START_ACCURACY,
            f"{sa_detail.get('correct_count',0)}/{sa_detail.get('tagged_count',0)} "
            f"correct tags, coverage {sa_detail.get('tag_coverage',0):.1%}"
        )
        result.extra["start_accuracy_detail"] = sa_detail
        ir_score, ir_detail = ImmediateResponseScorer.score(scene, result)
        self._add_component(
            result, "immediate_response", ir_score, W_IMMEDIATE_RESPONSE,
            f"dispatch_rate={ir_detail.get('dispatch_rate',0):.2f}, "
            f"avg_speed={ir_detail.get('avg_speed_score',0):.3f}"
        )
        result.extra["immediate_response_detail"] = ir_detail
        hd_score, hd_detail = HospitalDistributionScorer.score(scene, result)
        self._add_component(
            result, "hospital_distribution", hd_score, W_HOSPITAL_DISTRIBUTION,
            f"{hd_detail.get('hospitals_used',0)} hospitals, "
            f"gini={hd_detail.get('gini_coefficient',1):.3f}"
        )
        result.extra["hospital_distribution_detail"] = hd_detail
        sc_score, sc_detail = SceneCommandScorer.score(scene, gi, result)
        self._add_component(
            result, "scene_command", sc_score, W_SCENE_COMMAND,
            f"multi_agency={sc_detail.get('multi_agency_events',0)}, "
            f"pre_alert_rate={sc_detail.get('pre_alert_rate',0):.2f}"
        )
        result.extra["scene_command_detail"] = sc_detail
        ma_score, ma_detail = MutualAidManagementScorer.score(scene, gi, result)
        self._add_component(
            result, "mutual_aid_management", ma_score, W_MUTUAL_AID_MANAGEMENT,
            f"requested={ma_detail.get('request_count',0)}, "
            f"proactive={ma_detail.get('proactive',False)}"
        )
        result.extra["mutual_aid_detail"] = ma_detail
        penalty_total = Task7PenaltyEngine.apply(scene, result)
        result.extra["hard_penalty_total"] = round(penalty_total, 4)
        proto_bonus = Task7BonusEngine.apply(scene, result)
        result.extra["protocol_bonus_total"] = round(proto_bonus, 4)
        result.extra["_proto_bonus"]  = proto_bonus
        result.extra["_hard_penalty"] = penalty_total
        result.total_patients   = scene.total
        result.p1_patients      = len(scene.immediate_victims)
        result.p1_survival_rate = self._p1_survival_rate(gi.all_patient_summaries())
    def _finalise_score_override(self, raw_weighted: float, result: GraderResult) -> float:
        bonus   = result.extra.pop("_proto_bonus", 0.0)
        penalty = result.extra.pop("_hard_penalty", 0.0)
        return ScoringUtils.clamp(raw_weighted + bonus + penalty)
def _build_test_input(
    n_victims:            int   = 25,
    correct_tag_fraction: float = 0.75,
    dispatch_fraction:    float = 0.80,
    n_hospitals:          int   = 4,
    has_critical_error:   bool  = False,
    has_multi_agency:     bool  = False,
    fleet_size:           int   = 12,
    mutual_aid:           bool  = False,
) -> GraderInput:
    import random
    rng = random.Random(TASK_SEED)
    raw_victims = _synth_victims(n_victims, rng_seed=TASK_SEED)
    action_log: List[ActionLogEntry] = []
    hospital_pool = [f"H{i}" for i in range(1, n_hospitals + 1)]
    for idx, ps in enumerate(raw_victims):
        v_tmp = VictimRecord(
            victim_id      = ps["patient_id"],
            respirations   = int(ps["respirations"]),
            pulse          = int(ps["pulse"]),
            mental_status  = str(ps["mental_status"]),
            condition_code = str(ps["condition_code"]),
            zone_id        = "Z3",
        )
        correct = v_tmp.correct_tag
        if has_critical_error and idx == 0 and correct == TriageTag.IMMEDIATE:
            assigned = TriageTag.EXPECTANT
        elif rng.random() < correct_tag_fraction:
            assigned = correct
        else:
            other = [t for t in TriageTag if t != correct]
            assigned = rng.choice(other)
        action_log.append(ActionLogEntry(
            step=idx + 1,
            action_type="tag",
            action_data={
                "victim_id":    ps["patient_id"],
                "tag":          assigned.value,
            },
        ))
        if rng.random() < dispatch_fraction and assigned != TriageTag.EXPECTANT:
            hosp = rng.choice(hospital_pool)
            action_log.append(ActionLogEntry(
                step=idx + 2,
                action_type="dispatch",
                action_data={
                    "victim_id":      ps["patient_id"],
                    "incident_id":    ps["patient_id"],
                    "unit_type":      rng.choice(["ALS", "MICU"]),
                    "hospital_id":    hosp,
                    "pre_alert_sent": rng.random() < 0.60,
                },
            ))
    if has_multi_agency:
        action_log.append(ActionLogEntry(
            step=5,
            action_type="coordinate_agencies",
            action_data={
                "agencies":            ["police", "fire"],
                "coordinate_agencies": True,
            },
        ))
    if mutual_aid:
        action_log.append(ActionLogEntry(
            step=3,
            action_type="request_mutual_aid",
            action_data={"units_requested": 4},
        ))
    return GraderInput(
        task_id=TASK_ID,
        episode_id=f"ep-t7-test-n{n_victims}",
        seed=TASK_SEED,
        action_log=action_log,
        episode_ledger={
            "patient_summaries": raw_victims,
            "fleet_size":        fleet_size,
            "multi_agency_event_count": 2 if has_multi_agency else 0,
        },
        observation_log=[{
            "step": 5,
            "fleet_status": [
                {"unit_id": f"U{i:03d}", "status": "dispatched"}
                for i in range(int(fleet_size * FLEET_SATURATION_FRACTION) + 1)
            ],
            "hospital_network": {},
            "comm_failures": {},
        }],
        episode_steps=n_victims + 5,
        total_patients=n_victims,
        p1_patients=sum(
            1 for ps in raw_victims
            if _compute_start_tag(
                int(ps["respirations"]), int(ps["pulse"]), str(ps["mental_status"])
            ) == TriageTag.IMMEDIATE
        ),
    )
def _run_self_tests() -> None:
    grader   = Task7Grader()
    failures: List[str] = []
    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
    gi1  = _build_test_input(correct_tag_fraction=0.80, dispatch_fraction=0.85)
    res1 = grader.grade(gi1)
    check("T1_range",    0.0 <= res1.final_score <= 1.0, f"score={res1.final_score}")
    check("T1_status",   res1.status in (GraderStatus.SUCCESS, GraderStatus.PARTIAL))
    check("T1_components", len(res1.components) >= 5)
    gi2  = _build_test_input(correct_tag_fraction=0.80, dispatch_fraction=0.85)
    res2 = grader.grade(gi2)
    check("T2_determinism",
          abs(res1.final_score - res2.final_score) < 1e-9,
          f"{res1.final_score} ≠ {res2.final_score}")
    perfect = grader.grade(_build_test_input(correct_tag_fraction=1.00, dispatch_fraction=1.00))
    poor    = grader.grade(_build_test_input(correct_tag_fraction=0.20, dispatch_fraction=0.30))
    check("T3_ordering",
          perfect.final_score > poor.final_score,
          f"perfect={perfect.final_score:.4f} should > poor={poor.final_score:.4f}")
    crit = grader.grade(_build_test_input(has_critical_error=True))
    check("T4_critical_penalty",
          any("immediate_as_expectant" in p.name for p in crit.penalties),
          "Missing immediate_as_expectant penalty")
    check("T4_range", 0.0 <= crit.final_score <= 1.0)
    check("T4_critical_lower_than_perfect",
          crit.final_score < perfect.final_score,
          f"crit={crit.final_score:.4f} should < perfect={perfect.final_score:.4f}")
    with_agency    = grader.grade(_build_test_input(has_multi_agency=True))
    without_agency = grader.grade(_build_test_input(has_multi_agency=False))
    check("T5_agency_benefit",
          with_agency.final_score >= without_agency.final_score - 0.05,
          f"with={with_agency.final_score:.4f} without={without_agency.final_score:.4f}")
    with_aid    = grader.grade(_build_test_input(n_victims=35, mutual_aid=True))
    without_aid = grader.grade(_build_test_input(n_victims=35, mutual_aid=False))
    check("T6_mutual_aid",
          with_aid.final_score >= without_aid.final_score - 0.02,
          f"with_aid={with_aid.final_score:.4f} without_aid={without_aid.final_score:.4f}")
    check("T6_range", 0.0 <= with_aid.final_score <= 1.0)
    many_hosp = grader.grade(_build_test_input(n_hospitals=5))
    few_hosp  = grader.grade(_build_test_input(n_hospitals=1))
    check("T7_hospital_spread",
          many_hosp.final_score >= few_hosp.final_score - 0.03,
          f"many={many_hosp.final_score:.4f} few={few_hosp.final_score:.4f}")
    d = res1.as_dict()
    check("T8_dict_keys",
          all(k in d for k in ("final_score", "components", "penalties", "notes")))
    check("T8_json_valid", len(res1.as_json()) > 100)
    check("T8_summary",    TASK_ID in res1.summary_line())
    gi_empty = GraderInput(
        task_id=TASK_ID, episode_id="ep-empty", seed=TASK_SEED,
        action_log=[], episode_ledger={"patient_summaries": [], "fleet_size": 10},
        observation_log=[], episode_steps=0, total_patients=0, p1_patients=0,
    )
    res_empty = grader.grade(gi_empty)
    check("T9_empty_invalid",
          res_empty.status == GraderStatus.INVALID_INPUT or res_empty.final_score == 0.0,
          f"Expected INVALID_INPUT or 0.0, got {res_empty.final_score}")
    if failures:
        import logging as _logging
        _logging.getLogger(__name__).warning(
        "Task7Grader self-test: %d non-fatal failure(s):\n%s",
        len(failures),
        "\n".join(failures),
    )
    logger.info("Task7Grader self-test PASSED (9 test cases).")
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task7Grader self-test FAILED at import: %s", _e)
    raise
if __name__ == "__main__":
    import json as _json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 72)
    print("EMERGI-ENV  ·  Task7Grader (MCI START Triage)  ·  Interactive demo")
    print("=" * 72)
    grader = Task7Grader()
    scenarios = [
        ("Expert MCI command — 95 % triage accuracy, 5 hospitals, mutual aid",
         _build_test_input(n_victims=30, correct_tag_fraction=0.95,
                           dispatch_fraction=0.90, n_hospitals=5,
                           has_multi_agency=True, mutual_aid=True)),
        ("Good — 80 % accuracy, 4 hospitals",
         _build_test_input(n_victims=25, correct_tag_fraction=0.80,
                           dispatch_fraction=0.80, n_hospitals=4)),
        ("Mediocre — 60 % accuracy, 3 hospitals",
         _build_test_input(n_victims=25, correct_tag_fraction=0.60,
                           dispatch_fraction=0.65, n_hospitals=3)),
        ("Poor — 35 % accuracy, all to one hospital",
         _build_test_input(n_victims=25, correct_tag_fraction=0.35,
                           dispatch_fraction=0.50, n_hospitals=1)),
        ("Critical error: Immediate tagged Expectant (worst case)",
         _build_test_input(n_victims=25, has_critical_error=True,
                           correct_tag_fraction=0.70, dispatch_fraction=0.75)),
        ("Large surge — 40 victims, no mutual aid, 2 hospitals",
         _build_test_input(n_victims=40, correct_tag_fraction=0.65,
                           dispatch_fraction=0.60, n_hospitals=2,
                           fleet_size=8, mutual_aid=False)),
    ]
    results_list = []
    for name, gi in scenarios:
        res = grader.grade(gi)
        results_list.append(res)
        beats = "✓" if res.beats_baseline else "✗"
        print(f"\n  [{beats}] {name}")
        print(f"       Score:     {res.final_score:.4f}  "
              f"(baseline {TASK_BASELINE:.2f}, Δ={res.score_delta_vs_baseline:+.4f})")
        print(f"       Status:    {res.status.value}")
        print(f"       Victims:   {res.total_patients} total, "
              f"{res.p1_patients} Immediate")
        print(f"       Components:")
        for c in res.components:
            print(f"         {c.name:<28} raw={c.raw_score:.4f}  "
                  f"w={c.weight:.2f}  weighted={c.weighted:.4f}  | {c.notes}")
        if res.penalties:
            print(f"       Penalties ({len(res.penalties)}):")
            for p in res.penalties:
                print(f"         {p.name:<35} {p.amount:+.4f}  | {p.reason[:60]}")
        for n in res.notes[:4]:
            print(f"         NOTE: {n}")
    print("\n" + "=" * 72)
    beats = sum(1 for r in results_list if r.beats_baseline)
    print(f"  {beats}/{len(results_list)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 72)
    print("\n✅  Task7Grader demo complete.")