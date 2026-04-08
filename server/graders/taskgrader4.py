from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
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
logger = logging.getLogger("emergi_env.graders.task4")
TASK_ID       = "task4_multi_incident"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]
W_QUEUE_PRIORITISATION = 0.30
W_DISPATCH_EFFICIENCY  = 0.25
W_RESOURCE_ALLOCATION  = 0.20
W_ESCALATION_MGMT      = 0.15
W_CREW_MANAGEMENT      = 0.10
assert abs(
    W_QUEUE_PRIORITISATION + W_DISPATCH_EFFICIENCY + W_RESOURCE_ALLOCATION +
    W_ESCALATION_MGMT + W_CREW_MANAGEMENT - 1.0
) < 1e-9
HP_TRIAGE_INVERSION_P1_AFTER_P3  = 0.25
HP_TRIAGE_INVERSION_P1_AFTER_P2  = 0.12
HP_TRIAGE_INVERSION_P2_AFTER_P3  = 0.05
HP_P1_UNATTENDED_PER_STEP         = 0.015
HP_DIVERTED_HOSPITAL              = 0.030
HP_WRONG_UNIT_CRITICAL            = 0.020
HP_ESCALATION_MISSED              = 0.020
HP_OVER_DEPLETION                 = 0.050
HP_FATIGUE_UNDERMANAGED_PER_UNIT_STEP = 0.015
HP_DIVERSION_PENALTY_ROUTING      = 0.030
FLEET_UTILISATION_OPTIMAL_LO  = 0.60
FLEET_UTILISATION_OPTIMAL_HI  = 0.85
MIN_RESERVE_UNITS             = 2
ESCALATION_WINDOW_FULL_CREDIT = 2   
ESCALATION_WINDOW_PARTIAL_70  = 3
ESCALATION_WINDOW_PARTIAL_40  = 4
CREW_FATIGUE_HOURS_THRESHOLD  = 10.0
CREW_SWAP_JUSTIFIED_HOURS     = 8.0
CREW_SWAP_PREMATURE_HOURS     = 6.0
CREW_SWAP_JUSTIFIED_BONUS     = 0.02
CREW_SWAP_PREMATURE_PENALTY   = 0.02
MAX_INCIDENTS_TASK4           = 8
SEVERITY_RANK: Dict[str, int] = {"P1": 3, "P2": 2, "P3": 1, "P0": 0}
UNIT_RANK:     Dict[str, int] = {"BLS": 1, "ALS": 2, "MICU": 3}
INVERSION_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("P1", "P2"): 0.50,  
    ("P1", "P3"): 1.00,  
    ("P2", "P3"): 0.25,  
}
RESPONSE_TARGETS: Dict[str, float] = {
    "P1": 8.0,
    "P2": 30.0,
    "P3": 120.0,
    "P0": 999.0,
}
MICU_MANDATORY: FrozenSet[str] = frozenset({
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
    "cardiac_arrest_vf", "cardiac_arrest_pea", "cardiac_arrest_asystole",
    "polytrauma_blunt", "severe_tbi", "eclampsia",
    "pulmonary_oedema", "cardiogenic_shock", "aortic_dissection",
    "aaa_rupture", "organophosphate_poisoning",
})
@dataclass
class IncidentRecord:
    incident_id:      str
    severity:         str          
    condition_code:   str
    required_unit:    str          
    zone_id:          str
    step_reported:    int          
    step_dispatched:  Optional[int] = None
    dispatched_unit:  Optional[str] = None
    hospital_id:      Optional[str] = None
    hospital_on_diversion: bool    = False
    hospital_is_level1:    bool    = False
    response_time_min:     float   = 999.0
    travel_time_min:       float   = 999.0
    cath_lab_activated:    bool    = False
    stroke_unit_notified:  bool    = False
    multi_agency_needed:   bool    = False
    multi_agency_coordinated: bool = False
    trapped_victim:        bool    = False
    final_survival_prob:   float   = 0.50
    optimal_survival_prob: float   = 0.85
    phase_at_end:          str     = "untreated"
    dispatch_quality:      str     = "unknown"
    escalated:             bool    = False
    deterioration_step:    Optional[int] = None
    escalation_step:       Optional[int] = None
    notes:                 List[str] = field(default_factory=list)
    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)
    @property
    def was_dispatched(self) -> bool:
        return self.step_dispatched is not None
    @property
    def dispatch_latency_steps(self) -> Optional[int]:
        if self.step_dispatched is None:
            return None
        return self.step_dispatched - self.step_reported
    @property
    def is_critical(self) -> bool:
        return self.severity == "P1"
    @property
    def unit_type_correct(self) -> Optional[bool]:
        if self.dispatched_unit is None:
            return None
        req = UNIT_RANK.get(self.required_unit, 2)
        dis = UNIT_RANK.get(self.dispatched_unit.upper(), 0)
        return dis >= req or dis == req
    @property
    def is_micu_mandatory(self) -> bool:
        return self.condition_code in MICU_MANDATORY
@dataclass
class DispatchOrder:
    incident_id: str
    severity:    str
    step:        int
    action_idx:  int
@dataclass
class CrewSwapRecord:
    unit_id:       str
    hours_on_duty: float
    step:          int
    is_justified:  bool
    is_premature:  bool
@dataclass
class EscalationRecord:
    incident_id:    str
    old_severity:   str
    new_severity:   str
    step:           int
    deterioration_step: Optional[int] = None
    @property
    def response_latency(self) -> Optional[int]:
        if self.deterioration_step is None:
            return None
        return self.step - self.deterioration_step
class QueuePrioritisationScorer:
    @staticmethod
    def score(
        incidents:      List[IncidentRecord],
        dispatch_order: List[DispatchOrder],
        episode_steps:  int,
        result:         GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not incidents:
            return 0.50, {"reason": "no_incidents"}
        dispatched_ordered = sorted(dispatch_order, key=lambda d: (d.step, d.action_idx))
        inversion_score, inversion_detail = QueuePrioritisationScorer._compute_inversions(
            dispatched_ordered, incidents
        )
        p1_urgency_score = QueuePrioritisationScorer._compute_p1_urgency(
            incidents, dispatch_order, result
        )
        p1_unattended_penalty = QueuePrioritisationScorer._compute_p1_unattended_penalty(
            incidents, episode_steps, result
        )
        priority_dominance = QueuePrioritisationScorer._check_priority_dominance(
            incidents, dispatch_order, result
        )
        combined = (
            0.50 * inversion_score
            + 0.30 * p1_urgency_score
            + 0.20 * priority_dominance
        )
        combined += p1_unattended_penalty  
        details = {
            "inversion_score":       round(inversion_score, 4),
            "p1_urgency_score":      round(p1_urgency_score, 4),
            "priority_dominance":    round(priority_dominance, 4),
            "p1_unattended_penalty": round(p1_unattended_penalty, 4),
            **inversion_detail,
        }
        return ScoringUtils.clamp(combined), details
    @staticmethod
    def _compute_inversions(
        dispatch_order: List[DispatchOrder],
        incidents:      List[IncidentRecord],
    ) -> Tuple[float, Dict[str, Any]]:
        inc_severity: Dict[str, str] = {i.incident_id: i.severity for i in incidents}
        n = len(dispatch_order)
        if n < 2:
            return 1.0, {"inversions": 0, "total_pairs": 0}
        total_inversions = 0.0
        total_max        = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                sev_i = inc_severity.get(dispatch_order[i].incident_id, "P3")
                sev_j = inc_severity.get(dispatch_order[j].incident_id, "P3")
                r_i   = SEVERITY_RANK.get(sev_i, 1)
                r_j   = SEVERITY_RANK.get(sev_j, 1)
                pair_weight = INVERSION_WEIGHTS.get((sev_j, sev_i), 0.0)
                total_max  += max(INVERSION_WEIGHTS.values()) if pair_weight > 0 else 0.0
                if r_i < r_j:   
                    w = INVERSION_WEIGHTS.get((sev_j, sev_i), 0.0)
                    total_inversions += w
        if total_max == 0:
            return 1.0, {"inversions": 0, "total_pairs": n * (n - 1) // 2}
        score = 1.0 - (total_inversions / total_max)
        return ScoringUtils.clamp(score), {
            "weighted_inversions": round(total_inversions, 4),
            "max_possible_inversions": round(total_max, 4),
            "inversion_ratio":    round(total_inversions / total_max, 4),
        }
    @staticmethod
    def _compute_p1_urgency(
        incidents:      List[IncidentRecord],
        dispatch_order: List[DispatchOrder],
        result:         GraderResult,
    ) -> float:
        p1s = [i for i in incidents if i.severity == "P1"]
        if not p1s:
            return 1.0
        scores: List[float] = []
        dispatch_steps: Dict[str, int] = {
            d.incident_id: d.step for d in dispatch_order
        }
        for inc in p1s:
            d_step = dispatch_steps.get(inc.incident_id)
            if d_step is None:
                scores.append(0.0)
                result.add_note(
                    f"P1 incident {inc.incident_id} ({inc.condition_code}) "
                    f"never dispatched — 0 urgency score"
                )
                continue
            latency_steps = d_step - inc.step_reported
            if latency_steps <= 1:
                scores.append(1.0)
            elif latency_steps <= 2:
                scores.append(0.85)
            elif latency_steps <= 3:
                scores.append(0.65)
            elif latency_steps <= 5:
                scores.append(0.40)
            else:
                scores.append(max(0.0, 0.30 - (latency_steps - 5) * 0.05))
        return sum(scores) / len(scores) if scores else 1.0
    @staticmethod
    def _compute_p1_unattended_penalty(
        incidents:     List[IncidentRecord],
        episode_steps: int,
        result:        GraderResult,
    ) -> float:
        total_penalty = 0.0
        for inc in incidents:
            if inc.severity != "P1":
                continue
            if inc.step_dispatched is None:
                unattended_steps = max(0, episode_steps - inc.step_reported)
            else:
                unattended_steps = max(0, inc.step_dispatched - inc.step_reported)
            if unattended_steps > 0:
                penalty = HP_P1_UNATTENDED_PER_STEP * unattended_steps
                total_penalty -= penalty
                if penalty >= 0.05:
                    result.add_note(
                        f"P1 {inc.incident_id} unattended {unattended_steps} steps "
                        f"(−{penalty:.3f} penalty)"
                    )
        return total_penalty
    @staticmethod
    def _check_priority_dominance(
        incidents:      List[IncidentRecord],
        dispatch_order: List[DispatchOrder],
        result:         GraderResult,
    ) -> float:
        p1_ids = {i.incident_id for i in incidents if i.severity == "P1"}
        p3_ids = {i.incident_id for i in incidents if i.severity == "P3"}
        if not p1_ids or not p3_ids:
            return 1.0
        order_map: Dict[str, int] = {
            d.incident_id: idx for idx, d in enumerate(dispatch_order)
        }
        violations = 0
        pairs_checked = 0
        for p1_id in p1_ids:
            for p3_id in p3_ids:
                p1_idx = order_map.get(p1_id)
                p3_idx = order_map.get(p3_id)
                if p1_idx is None or p3_idx is None:
                    continue
                pairs_checked += 1
                if p3_idx < p1_idx:  
                    violations += 1
                    result.add_note(
                        f"Priority inversion: P3 {p3_id} dispatched before "
                        f"P1 {p1_id} — severe queue error"
                    )
        if pairs_checked == 0:
            return 1.0
        return ScoringUtils.clamp(1.0 - violations / pairs_checked)
class DispatchEfficiencyScorer:
    @staticmethod
    def score(
        incidents: List[IncidentRecord],
        result:    GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not incidents:
            return 0.50, {"reason": "no_incidents"}
        dispatched = [i for i in incidents if i.was_dispatched]
        undispatched_p1 = [i for i in incidents if not i.was_dispatched and i.severity == "P1"]
        per_incident_scores: List[Dict[str, Any]] = []
        weights: List[float] = []
        for inc in incidents:
            w = DispatchEfficiencyScorer._incident_weight(inc)
            weights.append(w)
            if not inc.was_dispatched:
                per_incident_scores.append({
                    "incident_id": inc.incident_id,
                    "score": 0.0,
                    "reason": "not_dispatched",
                })
                continue
            u_score = DispatchEfficiencyScorer._unit_type_score(inc, result)
            t_score = DispatchEfficiencyScorer._time_score(inc)
            s_score = DispatchEfficiencyScorer._survival_score(inc)
            p_score = DispatchEfficiencyScorer._protocol_score(inc, result)
            incident_score = ScoringUtils.clamp(
                0.40 * u_score + 0.30 * t_score + 0.20 * s_score + 0.10 * p_score
            )
            per_incident_scores.append({
                "incident_id":   inc.incident_id,
                "severity":      inc.severity,
                "unit_score":    round(u_score, 4),
                "time_score":    round(t_score, 4),
                "survival_score":round(s_score, 4),
                "protocol_score":round(p_score, 4),
                "score":         round(incident_score, 4),
                "weight":        round(w, 3),
            })
        if not weights or sum(weights) == 0:
            return 0.0, {"reason": "zero_weight"}
        total = sum(
            w * item["score"]
            for w, item in zip(weights, per_incident_scores)
        )
        aggregate = total / sum(weights)
        for inc in undispatched_p1:
            result.add_note(
                f"⚠ P1 incident {inc.incident_id} ({inc.condition_code}) "
                f"NEVER dispatched — critical failure"
            )
        return ScoringUtils.clamp(aggregate), {
            "dispatched_count":    len(dispatched),
            "total_incidents":     len(incidents),
            "undispatched_p1":     len(undispatched_p1),
            "per_incident":        per_incident_scores,
        }
    @staticmethod
    def _incident_weight(inc: IncidentRecord) -> float:
        severity_weight = {"P1": 2.0, "P2": 1.2, "P3": 0.6, "P0": 0.2}
        micu_bonus = 1.3 if inc.is_micu_mandatory else 1.0
        return severity_weight.get(inc.severity, 1.0) * micu_bonus
    @staticmethod
    def _unit_type_score(inc: IncidentRecord, result: GraderResult) -> float:
        if inc.dispatched_unit is None:
            return 0.0
        d = inc.dispatched_unit.upper()
        r = inc.required_unit.upper()
        score = ScoringUtils.unit_type_score(d, (r,), severity=inc.severity)
        if inc.is_micu_mandatory and d == "BLS":
            result.add_penalty(PenaltyRecord(
                name=f"wrong_unit_{inc.incident_id[:16]}",
                amount=-HP_WRONG_UNIT_CRITICAL,
                reason=f"BLS dispatched to {inc.condition_code} (MICU mandatory)",
                rule_ref="EMERGI-ENV Rule T4-H1",
            ))
            result.critical_mismatches += 1
        return score
    @staticmethod
    def _time_score(inc: IncidentRecord) -> float:
        target = RESPONSE_TARGETS.get(inc.severity, 30.0)
        return ScoringUtils.response_time_score(
            inc.response_time_min, target,
            worst_case_min=min(target * 6, 180.0),
        )
    @staticmethod
    def _survival_score(inc: IncidentRecord) -> float:
        return ScoringUtils.survival_probability_score(
            inc.final_survival_prob,
            inc.optimal_survival_prob,
            p_worst=0.05,
        )
    @staticmethod
    def _protocol_score(inc: IncidentRecord, result: GraderResult) -> float:
        net = 0.0
        if inc.cath_lab_activated and "stemi" in inc.condition_code:
            net += 0.018
        if inc.stroke_unit_notified and "stroke" in inc.condition_code:
            net += 0.015
        if inc.hospital_on_diversion:
            net -= HP_DIVERSION_PENALTY_ROUTING
            result.add_note(
                f"Incident {inc.incident_id}: routed to diverted hospital {inc.hospital_id}"
            )
        if inc.multi_agency_needed and not inc.multi_agency_coordinated:
            net -= 0.025
        return ScoringUtils.clamp(net, -0.20, 0.10)
class ResourceAllocationScorer:
    UTILISATION_BANDS: List[Tuple[float, float, float]] = [
        (0.00, 0.40, 0.60),   
        (0.40, 0.60, 0.85),   
        (0.60, 0.85, 1.00),   
        (0.85, 0.95, 0.75),   
        (0.95, 1.01, 0.30),   
    ]
    @staticmethod
    def score(
        gi:         GraderInput,
        incidents:  List[IncidentRecord],
        result:     GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        util_series = ResourceAllocationScorer._extract_utilisation_series(gi)
        avg_util = sum(util_series) / len(util_series) if util_series else 0.70
        min_util = min(util_series) if util_series else 0.70
        max_util = max(util_series) if util_series else 0.70
        util_score = 0.70  
        for lo, hi, score_val in ResourceAllocationScorer.UTILISATION_BANDS:
            if lo <= avg_util < hi:
                util_score = score_val
                break
        fleet_size   = gi.episode_ledger.get("fleet_size", 24)
        reserve_ok   = ResourceAllocationScorer._check_reserve(
            gi, fleet_size, result
        )
        mutual_aid_score = ResourceAllocationScorer._score_mutual_aid(gi, result)
        coverage_score = ResourceAllocationScorer._score_coverage(incidents)
        combined = (
            0.40 * util_score
            + 0.25 * reserve_ok
            + 0.20 * mutual_aid_score
            + 0.15 * coverage_score
        )
        return ScoringUtils.clamp(combined), {
            "avg_utilisation":   round(avg_util, 4),
            "min_utilisation":   round(min_util, 4),
            "max_utilisation":   round(max_util, 4),
            "utilisation_score": round(util_score, 4),
            "reserve_score":     round(reserve_ok, 4),
            "mutual_aid_score":  round(mutual_aid_score, 4),
            "coverage_score":    round(coverage_score, 4),
        }
    @staticmethod
    def _extract_utilisation_series(gi: GraderInput) -> List[float]:
        series: List[float] = []
        fleet_size = gi.episode_ledger.get("fleet_size", 24)
        if fleet_size <= 0:
            fleet_size = 24
        for obs in gi.observation_log:
            available = obs.get("fleet_available_count") or obs.get("available_units")
            if available is not None:
                deployed = fleet_size - int(available)
                series.append(ScoringUtils.clamp(deployed / fleet_size))
        if not series:
            n_dispatched = len([
                a for a in gi.action_log if a.action_type == "dispatch"
            ])
            est_util = min(0.80, n_dispatched / max(fleet_size, 1))
            series = [est_util]
        return series
    @staticmethod
    def _check_reserve(
        gi:         GraderInput,
        fleet_size: int,
        result:     GraderResult,
    ) -> float:
        depletion_events = 0
        for obs in gi.observation_log:
            available = obs.get("fleet_available_count") or obs.get("available_units")
            if available is not None and int(available) < MIN_RESERVE_UNITS:
                depletion_events += 1
        if depletion_events == 0:
            return 1.0
        elif depletion_events <= 2:
            result.add_note(
                f"Fleet reserve fell below {MIN_RESERVE_UNITS} units "
                f"{depletion_events} time(s) — manageable"
            )
            return 0.70
        else:
            penalty_steps = max(0, depletion_events - 2)
            result.add_penalty(PenaltyRecord(
                name="over_depletion",
                amount=-(HP_OVER_DEPLETION * min(3, penalty_steps / 3)),
                reason=f"Fleet depleted below {MIN_RESERVE_UNITS} units "
                       f"for {depletion_events} steps",
                rule_ref="EMERGI-ENV Rule T4-R1",
            ))
            result.add_note(
                f"⚠ Over-depletion: <{MIN_RESERVE_UNITS} units available "
                f"for {depletion_events} steps"
            )
            return ScoringUtils.clamp(0.50 - (depletion_events - 2) * 0.05)
    @staticmethod
    def _score_mutual_aid(
        gi:     GraderInput,
        result: GraderResult,
    ) -> float:
        aid_actions = gi.get_actions_by_type("request_mutual_aid")
        if not aid_actions:
            surge_occurred = gi.episode_ledger.get("surge_status", {}).get(
                "simultaneous_mci_count", 0
            ) > 0
            if surge_occurred:
                result.add_note(
                    "Mutual aid not requested despite surge conditions — missed resource opportunity"
                )
                return 0.50
            return 0.85  
        over_requests = 0
        justified_requests = 0
        for action in aid_actions:
            own_available = action.get("own_fleet_available_count", 1)
            if own_available > 2:
                over_requests += 1
            else:
                justified_requests += 1
        if over_requests == 0:
            return min(1.0, 0.80 + justified_requests * 0.05)
        else:
            result.add_note(
                f"Mutual aid over-requested {over_requests} time(s) "
                f"(own fleet not depleted)"
            )
            return ScoringUtils.clamp(0.80 - over_requests * 0.10)
    @staticmethod
    def _score_coverage(incidents: List[IncidentRecord]) -> float:
        if not incidents:
            return 0.80
        zones_with_incidents = {i.zone_id for i in incidents}
        zones_served = {
            i.zone_id for i in incidents if i.was_dispatched
        }
        if not zones_with_incidents:
            return 0.80
        coverage_ratio = len(zones_served) / len(zones_with_incidents)
        p1_zones = {i.zone_id for i in incidents if i.severity == "P1"}
        p1_zones_served = {
            i.zone_id for i in incidents
            if i.severity == "P1" and i.was_dispatched
        }
        p1_coverage = (
            len(p1_zones_served) / len(p1_zones) if p1_zones else 1.0
        )
        return ScoringUtils.clamp(0.40 * coverage_ratio + 0.60 * p1_coverage)
class EscalationManagementScorer:
    @staticmethod
    def score(
        gi:          GraderInput,
        escalations: List[EscalationRecord],
        result:      GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        gt_events = gi.episode_ledger.get("deterioration_events", [])
        if not gt_events and not escalations:
            return 1.0, {"reason": "no_deterioration_events"}
        if not gt_events:
            if len(escalations) <= 1:
                return 0.80, {"reason": "no_gt_events_one_escalation"}
            result.add_note(
                f"Agent issued {len(escalations)} escalations with no ground truth events — "
                "possible false escalations"
            )
            return 0.65, {"reason": "no_gt_events_multiple_escalations"}
        escalation_by_incident: Dict[str, List[int]] = defaultdict(list)
        for esc in escalations:
            escalation_by_incident[esc.incident_id].append(esc.step)
        event_scores: List[float] = []
        event_details: List[Dict[str, Any]] = []
        for event in gt_events:
            inc_id    = event.get("incident_id", "")
            det_step  = int(event.get("step", 0))
            old_sev   = event.get("old_severity", "P2")
            new_sev   = event.get("new_severity", "P1")
            urgency   = SEVERITY_RANK.get(new_sev, 1) - SEVERITY_RANK.get(old_sev, 0)
            agent_steps = escalation_by_incident.get(inc_id, [])
            if not agent_steps:
                event_scores.append(0.0)
                result.add_penalty(PenaltyRecord(
                    name=f"escalation_missed_{inc_id[:16]}",
                    amount=-HP_ESCALATION_MISSED * max(1, urgency),
                    reason=f"Missed escalation: {inc_id} deteriorated {old_sev}→{new_sev} "
                           f"at step {det_step}",
                    rule_ref="EMERGI-ENV Rule T4-E1",
                ))
                result.add_note(
                    f"Missed deterioration: {inc_id} {old_sev}→{new_sev} at step {det_step}"
                )
                event_details.append({
                    "incident_id": inc_id,
                    "det_step": det_step,
                    "response_latency": None,
                    "score": 0.0,
                    "status": "missed",
                })
                continue
            relevant = [s for s in agent_steps if s >= det_step]
            if not relevant:
                latency = None
                event_score = 0.10
                status = "late_escalation"
            else:
                latency = min(relevant) - det_step
                if latency <= ESCALATION_WINDOW_FULL_CREDIT:
                    event_score = 1.00
                    status = "on_time"
                elif latency <= ESCALATION_WINDOW_PARTIAL_70:
                    event_score = 0.70
                    status = "slightly_late"
                elif latency <= ESCALATION_WINDOW_PARTIAL_40:
                    event_score = 0.40
                    status = "late"
                else:
                    event_score = 0.15
                    status = "very_late"
                    result.add_note(
                        f"Very late escalation: {inc_id} latency={latency} steps "
                        f"({old_sev}→{new_sev})"
                    )
            event_scores.append(event_score)
            event_details.append({
                "incident_id":      inc_id,
                "det_step":         det_step,
                "response_latency": latency,
                "score":            round(event_score, 4),
                "status":           status,
                "old_severity":     old_sev,
                "new_severity":     new_sev,
            })
        overall = sum(event_scores) / len(event_scores) if event_scores else 1.0
        proactive_count = sum(
            1 for esc in escalations
            if any(
                esc.incident_id == ev.get("incident_id")
                and esc.step < int(ev.get("step", 999))
                for ev in gt_events
            )
        )
        if proactive_count > 0:
            overall = min(1.0, overall + proactive_count * 0.02)
            result.add_note(
                f"Proactive escalation bonus: {proactive_count} pre-emptive escalation(s)"
            )
        return ScoringUtils.clamp(overall), {
            "gt_events":         len(gt_events),
            "escalations_issued":len(escalations),
            "event_details":     event_details,
            "proactive_count":   proactive_count,
        }
class CrewManagementScorer:
    @staticmethod
    def score(
        gi:           GraderInput,
        crew_swaps:   List[CrewSwapRecord],
        episode_steps: int,
        result:       GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        fatigue_events = CrewManagementScorer._extract_fatigue_events(gi)
        if not crew_swaps and not fatigue_events:
            return 0.85, {"reason": "no_fatigue_management_needed"}
        base = 0.80
        swap_bonuses   = 0.0
        swap_penalties = 0.0
        for swap in crew_swaps:
            if swap.is_justified:
                swap_bonuses += CREW_SWAP_JUSTIFIED_BONUS
                result.add_note(
                    f"Justified crew swap: unit {swap.unit_id} "
                    f"({swap.hours_on_duty:.1f}h on duty) → +{CREW_SWAP_JUSTIFIED_BONUS:.3f}"
                )
            elif swap.is_premature:
                swap_penalties += CREW_SWAP_PREMATURE_PENALTY
                result.add_note(
                    f"Premature crew swap: unit {swap.unit_id} "
                    f"({swap.hours_on_duty:.1f}h on duty < {CREW_SWAP_PREMATURE_HOURS}h)"
                )
        unmanaged_fatigue_total = 0.0
        swapped_units = {s.unit_id for s in crew_swaps}
        for unit_id, fatigue_step, hours in fatigue_events:
            if unit_id not in swapped_units:
                steps_unmanaged = max(0, episode_steps - fatigue_step)
                penalty = HP_FATIGUE_UNDERMANAGED_PER_UNIT_STEP * steps_unmanaged
                unmanaged_fatigue_total += penalty
                if steps_unmanaged > 2:
                    result.add_note(
                        f"Unmanaged fatigue: unit {unit_id} ({hours:.1f}h) not swapped "
                        f"for {steps_unmanaged} steps after fatigue threshold"
                    )
        if unmanaged_fatigue_total > 0:
            result.add_penalty(PenaltyRecord(
                name="fatigue_undermanaged",
                amount=-min(0.20, unmanaged_fatigue_total),
                reason=f"Fatigued units not swapped: {len(fatigue_events)} events",
                rule_ref="EMERGI-ENV Rule T4-C1",
            ))
        score = ScoringUtils.clamp(
            base + swap_bonuses - swap_penalties - unmanaged_fatigue_total
        )
        return score, {
            "crew_swaps":              len(crew_swaps),
            "justified_swaps":         sum(1 for s in crew_swaps if s.is_justified),
            "premature_swaps":         sum(1 for s in crew_swaps if s.is_premature),
            "fatigue_events":          len(fatigue_events),
            "unmanaged_penalty":       round(unmanaged_fatigue_total, 4),
            "swap_bonus_total":        round(swap_bonuses, 4),
            "swap_penalty_total":      round(swap_penalties, 4),
        }
    @staticmethod
    def _extract_fatigue_events(gi: GraderInput) -> List[Tuple[str, int, float]]:
        events: List[Tuple[str, int, float]] = []
        for obs in gi.observation_log:
            step = obs.get("step", 0)
            fleet = obs.get("fleet_status", [])
            for unit in fleet:
                hours = float(unit.get("hours_on_duty", 0.0))
                if hours >= CREW_FATIGUE_HOURS_THRESHOLD:
                    uid = unit.get("unit_id", "unknown")
                    events.append((uid, step, hours))
        return events
class Task4InputParser:
    @staticmethod
    def parse_incidents(gi: GraderInput) -> List[IncidentRecord]:
        incidents: List[IncidentRecord] = []
        for s in gi.all_patient_summaries():
            inc = IncidentRecord(
                incident_id    = s.get("incident_id") or s.get("patient_id", "INC_UNK"),
                severity       = s.get("severity", "P2"),
                condition_code = s.get("condition_code", "unknown"),
                required_unit  = s.get("required_unit_type", "ALS"),
                zone_id        = s.get("zone_id", "Z01"),
                step_reported  = int(s.get("step_reported", 0)),
                step_dispatched = s.get("step_dispatched"),
                dispatched_unit = s.get("dispatched_unit") or s.get("dispatched_unit_type"),
                hospital_id    = s.get("hospital_id"),
                hospital_on_diversion = bool(s.get("hospital_on_diversion", False)),
                hospital_is_level1    = bool(s.get("hospital_is_level1", False)),
                response_time_min     = float(s.get("response_time_min", 999.0)),
                travel_time_min       = float(s.get("travel_time_min") or
                                              s.get("optimal_travel_time_min", 999.0)),
                cath_lab_activated    = bool(s.get("cath_lab_activated", False)),
                stroke_unit_notified  = bool(s.get("stroke_unit_notified", False)),
                multi_agency_needed   = bool(s.get("multi_agency_needed", False)),
                multi_agency_coordinated = bool(s.get("multi_agency_coordinated", False)),
                trapped_victim        = bool(s.get("requires_extrication", False)),
                final_survival_prob   = float(s.get("final_survival_prob", 0.50)),
                optimal_survival_prob = float(s.get("optimal_survival_prob", 0.85)),
                phase_at_end          = s.get("phase_at_episode_end", "untreated"),
                deterioration_step    = s.get("deterioration_step"),
            )
            if inc.dispatched_unit is None:
                for action in gi.action_log:
                    if action.action_type == "dispatch":
                        if action.get("incident_id") == inc.incident_id:
                            inc.dispatched_unit = action.get("unit_type")
                            inc.step_dispatched = action.step
                            break
            incidents.append(inc)
        return incidents[:MAX_INCIDENTS_TASK4]
    @staticmethod
    def parse_dispatch_order(gi: GraderInput, incidents: List[IncidentRecord]) -> List[DispatchOrder]:
        incident_ids = {i.incident_id for i in incidents}
        order: List[DispatchOrder] = []
        for idx, action in enumerate(gi.action_log):
            if action.action_type != "dispatch":
                continue
            inc_id   = action.get("incident_id")
            severity = action.get("assigned_priority") or action.get("severity", "P2")
            if inc_id and inc_id in incident_ids:
                order.append(DispatchOrder(
                    incident_id=inc_id,
                    severity=severity,
                    step=action.step,
                    action_idx=idx,
                ))
        return order
    @staticmethod
    def parse_escalations(gi: GraderInput) -> List[EscalationRecord]:
        escalations: List[EscalationRecord] = []
        for action in gi.action_log:
            if action.action_type not in ("escalate", "escalation"):
                continue
            escalations.append(EscalationRecord(
                incident_id  = action.get("incident_id", "unknown"),
                old_severity = action.get("current_severity", "P2"),
                new_severity = action.get("new_severity", "P1"),
                step         = action.step,
            ))
        return escalations
    @staticmethod
    def parse_crew_swaps(gi: GraderInput) -> List[CrewSwapRecord]:
        swaps: List[CrewSwapRecord] = []
        for action in gi.action_log:
            if action.action_type not in ("crew_swap", "crewswap"):
                continue
            hours = float(action.get("current_hours_on_duty", 0.0))
            swaps.append(CrewSwapRecord(
                unit_id      = action.get("unit_id", "unknown"),
                hours_on_duty= hours,
                step         = action.step,
                is_justified = hours >= CREW_SWAP_JUSTIFIED_HOURS,
                is_premature = hours < CREW_SWAP_PREMATURE_HOURS,
            ))
        return swaps
class Task4Grader(BaseGrader):
    TASK_ID          = TASK_ID
    TASK_SEED        = TASK_SEED
    TASK_BASELINE    = TASK_BASELINE
    TASK_DIFFICULTY  = "medium"
    COMPONENT_WEIGHTS: Dict[str, float] = {
        "queue_prioritisation": W_QUEUE_PRIORITISATION,
        "dispatch_efficiency":  W_DISPATCH_EFFICIENCY,
        "resource_allocation":  W_RESOURCE_ALLOCATION,
        "escalation_management":W_ESCALATION_MGMT,
        "crew_management":      W_CREW_MANAGEMENT,
    }
    def __init__(self) -> None:
        super().__init__()
        self._parser          = Task4InputParser()
        self._queue_scorer    = QueuePrioritisationScorer()
        self._dispatch_scorer = DispatchEfficiencyScorer()
        self._resource_scorer = ResourceAllocationScorer()
        self._escalation_scorer = EscalationManagementScorer()
        self._crew_scorer     = CrewManagementScorer()
    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(
                f"WARNING: seed {gi.seed} ≠ task seed {TASK_SEED}. "
                "Determinism not guaranteed."
            )
        incidents      = self._parser.parse_incidents(gi)
        dispatch_order = self._parser.parse_dispatch_order(gi, incidents)
        escalations    = self._parser.parse_escalations(gi)
        crew_swaps     = self._parser.parse_crew_swaps(gi)
        if not incidents:
            result.status        = GraderStatus.INVALID_INPUT
            result.error_message = "No patient_summaries in episode_ledger — cannot grade."
            return
        result.extra["n_incidents"]      = len(incidents)
        result.extra["n_p1_incidents"]   = sum(1 for i in incidents if i.severity == "P1")
        result.extra["n_dispatched"]     = sum(1 for i in incidents if i.was_dispatched)
        result.extra["n_escalations"]    = len(escalations)
        result.extra["n_crew_swaps"]     = len(crew_swaps)
        q_score, q_detail = QueuePrioritisationScorer.score(
            incidents, dispatch_order, gi.episode_steps, result
        )
        self._add_component(
            result, "queue_prioritisation", q_score, W_QUEUE_PRIORITISATION,
            f"inversions={q_detail.get('weighted_inversions', 0):.3f} | "
            f"p1_urgency={q_detail.get('p1_urgency_score', 0):.3f}"
        )
        result.extra["queue_detail"] = q_detail
        d_score, d_detail = DispatchEfficiencyScorer.score(incidents, result)
        self._add_component(
            result, "dispatch_efficiency", d_score, W_DISPATCH_EFFICIENCY,
            f"dispatched={d_detail.get('dispatched_count', 0)}/{d_detail.get('total_incidents', 0)} | "
            f"undispatched_p1={d_detail.get('undispatched_p1', 0)}"
        )
        result.extra["dispatch_detail"] = {
            k: v for k, v in d_detail.items() if k != "per_incident"
        }
        r_score, r_detail = ResourceAllocationScorer.score(gi, incidents, result)
        self._add_component(
            result, "resource_allocation", r_score, W_RESOURCE_ALLOCATION,
            f"avg_util={r_detail.get('avg_utilisation', 0):.1%} | "
            f"reserve={r_detail.get('reserve_score', 0):.3f}"
        )
        result.extra["resource_detail"] = r_detail
        e_score, e_detail = EscalationManagementScorer.score(gi, escalations, result)
        self._add_component(
            result, "escalation_management", e_score, W_ESCALATION_MGMT,
            f"gt_events={e_detail.get('gt_events', 0)} | "
            f"issued={e_detail.get('escalations_issued', 0)}"
        )
        result.extra["escalation_detail"] = e_detail
        c_score, c_detail = CrewManagementScorer.score(
            gi, crew_swaps, gi.episode_steps, result
        )
        self._add_component(
            result, "crew_management", c_score, W_CREW_MANAGEMENT,
            f"swaps={c_detail.get('crew_swaps', 0)} | "
            f"justified={c_detail.get('justified_swaps', 0)} | "
            f"fatigue_events={c_detail.get('fatigue_events', 0)}"
        )
        result.extra["crew_detail"] = c_detail
        self._apply_hard_triage_inversions(incidents, dispatch_order, result)
        self._apply_diverted_hospital_penalties(incidents, result)
        self._compute_aggregate_survival(incidents, result)
        result.total_patients   = len(incidents)
        result.p1_patients      = sum(1 for i in incidents if i.severity == "P1")
        result.p1_survival_rate = self._compute_p1_survival_rate(incidents)
        n_violations = sum(1 for i in incidents if i.hospital_on_diversion)
        result.protocol_violations += n_violations
        n_critical = sum(1 for i in incidents if i.is_micu_mandatory and
                         i.dispatched_unit and i.dispatched_unit.upper() == "BLS")
        result.critical_mismatches += n_critical
    def _apply_hard_triage_inversions(
        self,
        incidents:      List[IncidentRecord],
        dispatch_order: List[DispatchOrder],
        result:         GraderResult,
    ) -> None:
        inc_sev: Dict[str, str] = {i.incident_id: i.severity for i in incidents}
        ordered  = sorted(dispatch_order, key=lambda d: (d.step, d.action_idx))
        for i_idx in range(len(ordered)):
            for j_idx in range(i_idx + 1, len(ordered)):
                sev_i = inc_sev.get(ordered[i_idx].incident_id, "P3")
                sev_j = inc_sev.get(ordered[j_idx].incident_id, "P3")
                r_i   = SEVERITY_RANK.get(sev_i, 1)
                r_j   = SEVERITY_RANK.get(sev_j, 1)
                if r_i < r_j:
                    if sev_j == "P1" and sev_i == "P3":
                        amount = HP_TRIAGE_INVERSION_P1_AFTER_P3
                        label  = "P1_after_P3_inversion"
                        sev_label = "P1 dispatched AFTER P3 (worst)"
                    elif sev_j == "P1" and sev_i == "P2":
                        amount = HP_TRIAGE_INVERSION_P1_AFTER_P2
                        label  = "P1_after_P2_inversion"
                        sev_label = "P1 dispatched AFTER P2"
                    elif sev_j == "P2" and sev_i == "P3":
                        amount = HP_TRIAGE_INVERSION_P2_AFTER_P3
                        label  = "P2_after_P3_inversion"
                        sev_label = "P2 dispatched AFTER P3"
                    else:
                        continue
                    result.add_penalty(PenaltyRecord(
                        name=label,
                        amount=-amount,
                        reason=(
                            f"Queue inversion [{sev_label}]: "
                            f"{ordered[i_idx].incident_id}({sev_i}) before "
                            f"{ordered[j_idx].incident_id}({sev_j})"
                        ),
                        rule_ref="EMERGI-ENV Rule T4-I1",
                    ))
                    result.add_note(
                        f"⚠ Queue inversion: {sev_label} — "
                        f"{ordered[i_idx].incident_id} dispatched before "
                        f"{ordered[j_idx].incident_id}"
                    )
    def _apply_diverted_hospital_penalties(
        self,
        incidents: List[IncidentRecord],
        result:    GraderResult,
    ) -> None:
        for inc in incidents:
            if inc.hospital_on_diversion and inc.was_dispatched:
                result.add_penalty(PenaltyRecord(
                    name=f"diverted_routing_{inc.incident_id[:12]}",
                    amount=-HP_DIVERTED_HOSPITAL,
                    reason=(
                        f"Incident {inc.incident_id} ({inc.severity}) routed to "
                        f"diverted hospital {inc.hospital_id}"
                    ),
                    rule_ref="EMERGI-ENV Rule T4-D1",
                ))
    def _compute_aggregate_survival(
        self,
        incidents: List[IncidentRecord],
        result:    GraderResult,
    ) -> None:
        if not incidents:
            return
        weighted_sum = 0.0
        weight_total = 0.0
        for inc in incidents:
            w = DispatchEfficiencyScorer._incident_weight(inc)
            weighted_sum  += w * inc.final_survival_prob
            weight_total  += w
        avg_survival = weighted_sum / weight_total if weight_total > 0 else 0.50
        result.extra["weighted_avg_survival"] = round(avg_survival, 4)
        if avg_survival < 0.40:
            result.add_note(
                f"Low aggregate survival: {avg_survival:.1%}. "
                "Faster dispatch and better unit matching required."
            )
        elif avg_survival > 0.75:
            result.add_note(
                f"Good aggregate survival: {avg_survival:.1%}."
            )
    def _compute_p1_survival_rate(self, incidents: List[IncidentRecord]) -> float:
        p1 = [i for i in incidents if i.severity == "P1"]
        if not p1:
            return 1.0
        treated = [i for i in p1 if i.phase_at_end == "treated"]
        return len(treated) / len(p1)
GraderRegistry.register(TASK_ID, Task4Grader)
logger.info(
    "Task4Grader registered: task_id=%s baseline=%.2f seed=%d",
    TASK_ID, TASK_BASELINE, TASK_SEED,
)
def _build_multi_incident_ledger(
    incidents_config: List[Dict[str, Any]],
    deterioration_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    summaries = []
    for cfg in incidents_config:
        summaries.append({
            "incident_id":          cfg.get("incident_id", f"INC{len(summaries)+1:03d}"),
            "patient_id":           cfg.get("incident_id", f"P{len(summaries)+1:03d}"),
            "severity":             cfg.get("severity", "P2"),
            "condition_code":       cfg.get("condition_code", "general_medical"),
            "required_unit_type":   cfg.get("required_unit", "ALS"),
            "zone_id":              cfg.get("zone_id", "Z05"),
            "step_reported":        cfg.get("step_reported", 0),
            "step_dispatched":      cfg.get("step_dispatched"),
            "dispatched_unit":      cfg.get("dispatched_unit"),
            "hospital_id":          cfg.get("hospital_id", "H01"),
            "hospital_on_diversion":cfg.get("on_diversion", False),
            "hospital_is_level1":   cfg.get("is_level1", False),
            "response_time_min":    cfg.get("response_time_min", 12.0),
            "travel_time_min":      cfg.get("travel_time_min", 10.0),
            "cath_lab_activated":   cfg.get("cath_lab_activated", False),
            "stroke_unit_notified": cfg.get("stroke_unit_notified", False),
            "multi_agency_needed":  cfg.get("multi_agency_needed", False),
            "multi_agency_coordinated": cfg.get("multi_agency_coordinated", False),
            "requires_extrication": cfg.get("requires_extrication", False),
            "final_survival_prob":  cfg.get("final_survival_prob", 0.80),
            "optimal_survival_prob":cfg.get("optimal_survival_prob", 0.90),
            "phase_at_episode_end": cfg.get("phase", "treated"),
            "weight":               cfg.get("weight", 1.0),
            "weighted_reward":      cfg.get("final_survival_prob", 0.80),
        })
    return {
        "patient_summaries":  summaries,
        "fleet_size":         24,
        "deterioration_events": deterioration_events or [],
        "surge_status": {"simultaneous_mci_count": 0},
    }
def _build_action_log(dispatch_sequence: List[Dict[str, Any]]) -> List[ActionLogEntry]:
    entries: List[ActionLogEntry] = []
    for d in dispatch_sequence:
        entries.append(ActionLogEntry(
            step=d.get("step", 1),
            action_type=d.get("action_type", "dispatch"),
            action_data={
                "incident_id":             d.get("incident_id"),
                "unit_type":               d.get("unit_type", "ALS"),
                "assigned_priority":       d.get("severity", "P2"),
                "hospital_id":             d.get("hospital_id", "H01"),
                "travel_time_min":         d.get("travel_time_min", 10.0),
                "cath_lab_activated":      d.get("cath_lab_activated", False),
                "stroke_unit_notified":    d.get("stroke_unit_notified", False),
                "multi_agency_coordinated":d.get("multi_agency_coordinated", False),
                "current_hours_on_duty":   d.get("hours_on_duty", 0.0),
                "unit_id":                 d.get("unit_id", "ALS-001"),
                "current_severity":        d.get("old_severity", "P2"),
                "new_severity":            d.get("new_severity", "P1"),
                "incident_id":             d.get("incident_id", "INC001"),
                "own_fleet_available_count": d.get("own_fleet_available", 5),
            },
        ))
    return entries
def _build_gi(
    incidents_config:     List[Dict[str, Any]],
    dispatch_sequence:    List[Dict[str, Any]],
    deterioration_events: Optional[List[Dict[str, Any]]] = None,
    observation_log:      Optional[List[Dict[str, Any]]] = None,
    episode_id:           str = "ep-t4-001",
) -> GraderInput:
    return GraderInput(
        task_id          = TASK_ID,
        episode_id       = episode_id,
        seed             = TASK_SEED,
        action_log       = _build_action_log(dispatch_sequence),
        episode_ledger   = _build_multi_incident_ledger(
            incidents_config, deterioration_events
        ),
        observation_log  = observation_log or [],
        episode_steps    = 25,
        total_patients   = len(incidents_config),
        p1_patients      = sum(1 for c in incidents_config if c.get("severity") == "P1"),
    )
def _run_self_tests() -> None:
    grader   = Task4Grader()
    failures: List[str] = []
    def chk(name: str, cond: bool, msg: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]: {msg}")
    perfect_incidents = [
        {"incident_id":"INC001","severity":"P1","condition_code":"stemi_anterior",
         "required_unit":"MICU","step_reported":0,"step_dispatched":1,
         "dispatched_unit":"MICU","final_survival_prob":0.90,"optimal_survival_prob":0.95,
         "cath_lab_activated":True,"phase":"treated"},
        {"incident_id":"INC002","severity":"P2","condition_code":"copd_exacerbation",
         "required_unit":"ALS","step_reported":1,"step_dispatched":3,
         "dispatched_unit":"ALS","final_survival_prob":0.85,"phase":"treated"},
        {"incident_id":"INC003","severity":"P3","condition_code":"minor_trauma",
         "required_unit":"BLS","step_reported":2,"step_dispatched":8,
         "dispatched_unit":"BLS","final_survival_prob":0.98,"phase":"treated"},
    ]
    perfect_dispatch = [
        {"step":1,"incident_id":"INC001","unit_type":"MICU","severity":"P1",
         "cath_lab_activated":True},
        {"step":3,"incident_id":"INC002","unit_type":"ALS","severity":"P2"},
        {"step":8,"incident_id":"INC003","unit_type":"BLS","severity":"P3"},
    ]
    gi1  = _build_gi(perfect_incidents, perfect_dispatch, episode_id="ep-t4-001")
    r1   = grader.grade(gi1)
    chk("T1_range",    0.0 <= r1.final_score <= 1.0)
    chk("T1_baseline", r1.final_score >= TASK_BASELINE,
        f"score={r1.final_score:.4f} < baseline={TASK_BASELINE}")
    chk("T1_components", len(r1.components) >= 5)
    chk("T1_status",   r1.status in (GraderStatus.SUCCESS, GraderStatus.PARTIAL))
    gi2  = _build_gi(perfect_incidents, perfect_dispatch, episode_id="ep-t4-002")
    r2   = grader.grade(gi2)
    chk("T2_determinism", abs(r1.final_score - r2.final_score) < 1e-9,
        f"{r1.final_score} ≠ {r2.final_score}")
    inverted_dispatch = [
        {"step":1,"incident_id":"INC003","unit_type":"BLS","severity":"P3"},  
        {"step":2,"incident_id":"INC002","unit_type":"ALS","severity":"P2"},
        {"step":5,"incident_id":"INC001","unit_type":"MICU","severity":"P1"},  
    ]
    gi3  = _build_gi(perfect_incidents, inverted_dispatch, episode_id="ep-t4-003")
    r3   = grader.grade(gi3)
    chk("T3_inversion_lower", r3.final_score < r1.final_score,
        f"Inverted {r3.final_score:.4f} should be < correct {r1.final_score:.4f}")
    chk("T3_inversion_penalty",
        any("inversion" in p.name for p in r3.penalties),
        "No inversion penalty found")
    chk("T3_range", 0.0 <= r3.final_score <= 1.0)
    wrong_unit_incidents = [
        {"incident_id":"INC001","severity":"P1","condition_code":"stemi_anterior",
         "required_unit":"MICU","step_reported":0,"step_dispatched":1,
         "dispatched_unit":"BLS","final_survival_prob":0.55,"optimal_survival_prob":0.92,
         "phase":"treated"},
    ]
    wrong_unit_dispatch = [
        {"step":1,"incident_id":"INC001","unit_type":"BLS","severity":"P1"},
    ]
    gi4  = _build_gi(wrong_unit_incidents, wrong_unit_dispatch, episode_id="ep-t4-004")
    r4   = grader.grade(gi4)
    chk("T4_wrong_unit_lower", r4.final_score < r1.final_score)
    chk("T4_wrong_unit_penalty",
        any("wrong_unit" in p.name for p in r4.penalties),
        "No wrong_unit penalty found")
    chk("T4_range", 0.0 <= r4.final_score <= 1.0)
    diverted_incidents = [
        {"incident_id":"INC001","severity":"P1","condition_code":"stemi_anterior",
         "required_unit":"MICU","step_reported":0,"step_dispatched":1,
         "dispatched_unit":"MICU","on_diversion":True,"hospital_id":"H03",
         "final_survival_prob":0.70,"optimal_survival_prob":0.92,"phase":"treated"},
    ]
    gi5  = _build_gi(diverted_incidents, [
        {"step":1,"incident_id":"INC001","unit_type":"MICU","severity":"P1"}
    ], episode_id="ep-t4-005")
    r5   = grader.grade(gi5)
    chk("T5_diversion_penalty",
        any("diverted" in p.name for p in r5.penalties),
        "No diverted_routing penalty")
    chk("T5_range", 0.0 <= r5.final_score <= 1.0)
    det_events = [
        {"incident_id":"INC001","step":4,"old_severity":"P2","new_severity":"P1"},
    ]
    escalation_dispatch = [
        {"step":1,"incident_id":"INC001","unit_type":"ALS","severity":"P2"},
        {"step":5,"incident_id":"INC001","action_type":"escalate",
         "old_severity":"P2","new_severity":"P1"},
    ]
    gi6_on  = _build_gi(
        [{"incident_id":"INC001","severity":"P2","condition_code":"polytrauma_blunt",
          "required_unit":"ALS","step_reported":0,"step_dispatched":1,
          "dispatched_unit":"ALS","final_survival_prob":0.78,"phase":"treated"}],
        escalation_dispatch, deterioration_events=det_events,
        episode_id="ep-t4-006a",
    )
    r6_on = grader.grade(gi6_on)
    chk("T6_on_time_range", 0.0 <= r6_on.final_score <= 1.0)
    gi6_miss = _build_gi(
        [{"incident_id":"INC001","severity":"P2","condition_code":"polytrauma_blunt",
          "required_unit":"ALS","step_reported":0,"step_dispatched":1,
          "dispatched_unit":"ALS","final_survival_prob":0.78,"phase":"treated"}],
        [{"step":1,"incident_id":"INC001","unit_type":"ALS","severity":"P2"}],
        deterioration_events=det_events,
        episode_id="ep-t4-006b",
    )
    r6_miss = grader.grade(gi6_miss)
    chk("T6_missed_escalation_penalty",
        any("escalation_missed" in p.name for p in r6_miss.penalties),
        "No escalation_missed penalty")
    chk("T6_missed_lower_than_on_time",
        r6_miss.final_score <= r6_on.final_score,
        f"Missed={r6_miss.final_score:.4f} should ≤ on-time={r6_on.final_score:.4f}")
    justified_swap_dispatch = [
        {"step":1,"incident_id":"INC001","unit_type":"MICU","severity":"P1"},
        {"step":8,"action_type":"crew_swap","unit_id":"MICU-001","hours_on_duty":9.0},
    ]
    gi7 = _build_gi(perfect_incidents[:1], justified_swap_dispatch, episode_id="ep-t4-007")
    r7  = grader.grade(gi7)
    chk("T7_range", 0.0 <= r7.final_score <= 1.0)
    gi8 = _build_gi(perfect_incidents, [], episode_id="ep-t4-008")
    r8  = grader.grade(gi8)
    chk("T8_no_dispatch_low", r8.final_score < r1.final_score)
    chk("T8_range", 0.0 <= r8.final_score <= 1.0)
    eight_incidents = [
        {"incident_id": f"INC{i+1:03d}",
         "severity": ["P1","P2","P1","P3","P2","P1","P3","P2"][i],
         "condition_code": ["stemi_anterior","copd_exacerbation","cardiac_arrest_vf",
                            "minor_trauma","septic_shock","polytrauma_blunt",
                            "general_weakness_stable_elderly","status_epilepticus"][i],
         "required_unit": ["MICU","ALS","MICU","BLS","MICU","MICU","BLS","ALS"][i],
         "step_reported": i, "step_dispatched": i+1,
         "dispatched_unit": ["MICU","ALS","MICU","BLS","MICU","MICU","BLS","ALS"][i],
         "final_survival_prob": [0.88,0.85,0.72,0.99,0.76,0.78,0.99,0.88][i],
         "optimal_survival_prob": [0.94,0.92,0.85,0.99,0.85,0.88,0.99,0.92][i],
         "phase": "treated", "zone_id": f"Z{(i%5)+1:02d}",
        }
        for i in range(8)
    ]
    eight_dispatch = [
        {"step":i+1,"incident_id":inc["incident_id"],
         "unit_type":inc["dispatched_unit"],"severity":inc["severity"]}
        for i, inc in sorted(enumerate(eight_incidents),
                             key=lambda x: -SEVERITY_RANK.get(x[1]["severity"], 0))
    ]
    for i, d in enumerate(eight_dispatch):
        d["step"] = i + 1
    gi9 = _build_gi(eight_incidents, eight_dispatch, episode_id="ep-t4-009")
    r9  = grader.grade(gi9)
    chk("T9_range",    0.0 <= r9.final_score <= 1.0)
    chk("T9_baseline", r9.final_score >= TASK_BASELINE,
        f"8-incident score {r9.final_score:.4f} < baseline {TASK_BASELINE}")
    chk("T9_n_incidents", r9.extra.get("n_incidents", 0) == 8)
    d  = r1.as_dict()
    chk("T10_dict", all(k in d for k in ("final_score","components","penalties","notes")))
    chk("T10_json", len(r1.as_json()) > 200)
    chk("T10_summary", TASK_ID in r1.summary_line())
    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task4Grader self-test: {len(failures)} failure(s):\n" +
            "\n".join(failures)
        )
    logger.info("Task4Grader self-test PASSED (10 test cases).")
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task4Grader self-test FAILED at import: %s", _e)
    raise
if __name__ == "__main__":
    import json as _json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 72)
    print("EMERGI-ENV  ·  Task4Grader  ·  Multi-Incident Queue Management Demo")
    print("=" * 72)
    grader = Task4Grader()
    BASE_INCIDENTS = [
        {"incident_id":"INC001","severity":"P1","condition_code":"stemi_anterior",
         "required_unit":"MICU","step_reported":0,"zone_id":"Z05"},
        {"incident_id":"INC002","severity":"P2","condition_code":"copd_exacerbation",
         "required_unit":"ALS","step_reported":1,"zone_id":"Z05"},
        {"incident_id":"INC003","severity":"P1","condition_code":"polytrauma_blunt",
         "required_unit":"MICU","step_reported":2,"zone_id":"Z07"},
        {"incident_id":"INC004","severity":"P3","condition_code":"minor_trauma",
         "required_unit":"BLS","step_reported":2,"zone_id":"Z08"},
        {"incident_id":"INC005","severity":"P2","condition_code":"septic_shock",
         "required_unit":"MICU","step_reported":3,"zone_id":"Z05"},
    ]
    def fill_dispatched(incs: List[Dict], dispatch_seq: List[Dict]) -> List[Dict]:
        step_map = {d["incident_id"]: d for d in dispatch_seq}
        for inc in incs:
            d = step_map.get(inc["incident_id"], {})
            inc["step_dispatched"]     = d.get("step")
            inc["dispatched_unit"]     = d.get("unit_type")
            inc["final_survival_prob"] = 0.88 if d.get("unit_type") in ("MICU","ALS") else 0.55
            inc["optimal_survival_prob"] = 0.93
            inc["cath_lab_activated"]  = d.get("cath_lab_activated", False)
            inc["phase"] = "treated" if d.get("step") else "untreated"
        return incs
    scenarios = [
        ("Perfect priority: P1 first, MICU, fast",
         [{"step":1,"incident_id":"INC001","unit_type":"MICU","severity":"P1",
           "cath_lab_activated":True},
          {"step":2,"incident_id":"INC003","unit_type":"MICU","severity":"P1"},
          {"step":4,"incident_id":"INC002","unit_type":"ALS","severity":"P2"},
          {"step":5,"incident_id":"INC005","unit_type":"MICU","severity":"P2"},
          {"step":9,"incident_id":"INC004","unit_type":"BLS","severity":"P3"}]),
        ("Inverted: P3 served before P1",
         [{"step":1,"incident_id":"INC004","unit_type":"BLS","severity":"P3"},
          {"step":2,"incident_id":"INC002","unit_type":"ALS","severity":"P2"},
          {"step":4,"incident_id":"INC001","unit_type":"MICU","severity":"P1"},
          {"step":5,"incident_id":"INC003","unit_type":"ALS","severity":"P1"},
          {"step":8,"incident_id":"INC005","unit_type":"ALS","severity":"P2"}]),
        ("Wrong units: BLS for all P1",
         [{"step":1,"incident_id":"INC001","unit_type":"BLS","severity":"P1"},
          {"step":2,"incident_id":"INC003","unit_type":"BLS","severity":"P1"},
          {"step":3,"incident_id":"INC002","unit_type":"BLS","severity":"P2"},
          {"step":4,"incident_id":"INC005","unit_type":"BLS","severity":"P2"},
          {"step":6,"incident_id":"INC004","unit_type":"BLS","severity":"P3"}]),
        ("P1 left unattended (dispatched step 10)",
         [{"step":10,"incident_id":"INC001","unit_type":"MICU","severity":"P1"},
          {"step":2,"incident_id":"INC002","unit_type":"ALS","severity":"P2"},
          {"step":3,"incident_id":"INC004","unit_type":"BLS","severity":"P3"},
          {"step":4,"incident_id":"INC005","unit_type":"ALS","severity":"P2"},
          {"step":11,"incident_id":"INC003","unit_type":"MICU","severity":"P1"}]),
        ("No actions at all",
         []),
    ]
    results_all = []
    for name, dispatch_seq in scenarios:
        incidents = fill_dispatched([dict(i) for i in BASE_INCIDENTS], dispatch_seq)
        gi = _build_gi(incidents, dispatch_seq, episode_id=f"demo_{name[:15]}")
        res = grader.grade(gi)
        results_all.append(res)
        beat = "✓" if res.beats_baseline else "✗"
        print(f"\n  [{beat}] {name}")
        print(f"       Score={res.final_score:.4f}  base={TASK_BASELINE:.2f}  "
              f"Δ={res.score_delta_vs_baseline:+.4f}  status={res.status.value}")
        for c in res.components:
            bar = "█" * int(c.raw_score * 20)
            print(f"         {c.name:<24} {c.raw_score:.4f} × {c.weight:.2f} "
                  f"= {c.weighted:.4f}  {bar}")
        if res.penalties:
            print(f"       Penalties ({len(res.penalties)}):")
            for p in res.penalties[:3]:
                print(f"         {p.name:<32} {p.amount:+.4f}  {p.reason[:55]}")
        if res.notes:
            for n in res.notes[:2]:
                print(f"         ⚙  {n[:70]}")
    print("\n" + "=" * 72)
    beats = sum(1 for r in results_all if r.beats_baseline)
    print(f"  {beats}/{len(results_all)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 72)
    print("\n✅  Task4Grader demo complete.")