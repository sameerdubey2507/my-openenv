from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
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
logger = logging.getLogger("emergi_env.graders.task1")
TASK_ID        = "task1_single_triage"
TASK_SEED      = TASK_SEEDS[TASK_ID]
TASK_BASELINE  = TASK_BASELINES[TASK_ID]
W_TRIAGE   = 0.40
W_UNIT     = 0.30
W_HOSPITAL = 0.30
WH_SPECIALTY = 0.50
WH_CAPACITY  = 0.30
WH_TRAVEL    = 0.20
PROTOCOL_BONUS_CAP = 0.15
HARD_PENALTY_P1_AS_P3        = -0.50   
HARD_PENALTY_BLS_FOR_CARDIAC = -0.30   
HARD_PENALTY_DIVERTED_P1     = -0.30   
TRIAGE_SCORE_MATRIX: Dict[Tuple[str, str], float] = {
    ("P1", "P1"): 1.00,
    ("P2", "P1"): 0.50,
    ("P3", "P1"): 0.00,   
    ("P0", "P1"): 0.00,
    ("P1", "P2"): 0.70,   
    ("P2", "P2"): 1.00,
    ("P3", "P2"): 0.50,
    ("P0", "P2"): 0.10,
    ("P1", "P3"): 0.60,   
    ("P2", "P3"): 0.85,
    ("P3", "P3"): 1.00,
    ("P0", "P3"): 0.20,
    ("P1", "P0"): 0.50,   
    ("P2", "P0"): 0.70,
    ("P3", "P0"): 0.80,
    ("P0", "P0"): 1.00,
}
UNIT_RANK: Dict[str, int] = {"BLS": 1, "ALS": 2, "MICU": 3}
RESPONSE_TIME_TARGETS: Dict[str, float] = {
    "metro_core":        8.0,
    "metro_suburban":   10.0,
    "metro_satellite":  12.0,
    "metro_tier2":      14.0,
    "tier2_city":       15.0,
    "tier3_city":       20.0,
    "semi_urban":       18.0,
    "semi_rural":       20.0,
    "rural":            26.0,
    "tribal_rural":     32.0,
    "tribal_forest":    45.0,
    "coastal_rural":    30.0,
    "industrial_town":  18.0,
}
DEFAULT_RESPONSE_TARGET = 20.0
CARDIAC_ARREST_CONDITIONS = {
    "cardiac_arrest", "cardiac_arrest_vf", "cardiac_arrest_pea",
    "cardiac_arrest_asystole", "vf_arrest", "pulseless_vt",
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_post_cabg",
}
MICU_MANDATORY_CONDITIONS = {
    "cardiac_arrest", "cardiac_arrest_vf", "cardiac_arrest_pea",
    "cardiac_arrest_asystole", "vf_arrest", "pulseless_vt",
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_post_cabg",
    "polytrauma_blunt", "polytrauma_penetrating",
    "haemorrhagic_stroke_sah", "eclampsia",
    "pulmonary_oedema", "septic_shock",
    "major_burns_40pct", "near_drowning",
    "anaphylaxis_refractory", "massive_pe",
}
class Task1Grader(BaseGrader):
    TASK_ID         = TASK_ID
    TASK_SEED       = TASK_SEED
    TASK_BASELINE   = TASK_BASELINE
    TASK_DIFFICULTY = "easy"
    COMPONENT_WEIGHTS: Dict[str, float] = {
        "triage_class":   W_TRIAGE,
        "unit_type":      W_UNIT,
        "hospital_match": W_HOSPITAL,
    }
    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(
                f"WARNING: episode seed {gi.seed} ≠ task seed {TASK_SEED}. "
                "Grading proceeds but determinism not guaranteed."
            )
        gt = self._extract_ground_truth(gi)
        if gt is None:
            result.status        = GraderStatus.INVALID_INPUT
            result.error_message = (
                "episode_ledger.patient_summaries is empty or malformed. "
                "Task 1 requires exactly one patient summary."
            )
            result.final_score = 0.0
            return
        action = self._extract_dispatch_action(gi)
        if action is None:
            result.add_note("No dispatch action found in action_log. Score = 0.")
            self._apply_no_action_scores(result)
            result.extra["no_action"] = True
            result.status = GraderStatus.PARTIAL
            return
        result.extra["time_to_first_action_steps"] = action.step
        result.extra["condition_code"] = gt.get("condition_code", "unknown")
        t_score, t_notes = self._score_triage(action, gt, result)
        self._add_component(result, "triage_class", t_score, W_TRIAGE, t_notes)
        u_score, u_notes = self._score_unit_type(action, gt, result)
        self._add_component(result, "unit_type", u_score, W_UNIT, u_notes)
        h_score, h_notes = self._score_hospital_match(action, gt, gi, result)
        self._add_component(result, "hospital_match", h_score, W_HOSPITAL, h_notes)
        proto_net = self._apply_protocol_checks(action, gt, gi, result)
        result.extra["protocol_net_score"] = round(proto_net, 4)
        self._apply_hard_penalties(action, gt, result)
        self._integrate_survival_reward(gt, result)
        result.total_patients   = 1
        result.p1_patients      = 1 if gt.get("severity") == "P1" else 0
        result.p1_survival_rate = self._p1_survival_rate(gi.all_patient_summaries())
    def _extract_ground_truth(
        self, gi: GraderInput
    ) -> Optional[Dict[str, Any]]:
        summaries = gi.all_patient_summaries()
        if not summaries:
            return None
        gt = dict(summaries[0])  
        gt.setdefault("severity", "P1")
        gt.setdefault("condition_code", "unknown")
        gt.setdefault("required_unit_type", "ALS")
        gt.setdefault("required_hospital_specialties", [])
        gt.setdefault("zone_type", "metro_core")
        gt.setdefault("hospital_on_diversion", False)
        gt.setdefault("hospital_er_occupancy_pct", 0.70)
        gt.setdefault("hospital_has_specialty", True)
        gt.setdefault("hospital_is_level1", False)
        gt.setdefault("response_time_target_min",
                      RESPONSE_TIME_TARGETS.get(gt.get("zone_type", ""), DEFAULT_RESPONSE_TARGET))
        gt.setdefault("optimal_travel_time_min", 12.0)
        gt.setdefault("final_survival_prob", 0.80)
        gt.setdefault("optimal_survival_prob", 0.95)
        gt.setdefault("dispatch_latency_min", None)
        return gt
    def _extract_dispatch_action(
        self, gi: GraderInput
    ) -> Optional[ActionLogEntry]:
        dispatch = self._get_first_dispatch(gi.action_log)
        if dispatch:
            return dispatch
        for entry in gi.action_log:
            if entry.action_type in ("triage", "triage_and_dispatch"):
                return entry
        return None
    def _score_triage(
        self,
        action: ActionLogEntry,
        gt:     Dict[str, Any],
        result: GraderResult,
    ) -> Tuple[float, str]:
        truth: str = gt["severity"]   
        assigned: Optional[str] = (
            action.get("assigned_priority")
            or action.get("priority")
            or action.get("triage_class")
            or action.get("severity")
        )
        if assigned is None:
            unit = (action.get("unit_type") or action.get("ambulance_type") or "").upper()
            assigned = self._infer_priority_from_unit(unit, gt)
            result.add_note(f"Priority inferred from unit_type='{unit}' → '{assigned}'")
        else:
            assigned = str(assigned).upper().strip()
            assigned = self._normalise_priority_label(assigned)
        raw = TRIAGE_SCORE_MATRIX.get((assigned, truth), 0.10)
        tier_map = {"P1": 3, "P2": 2, "P3": 1, "P0": 0}
        delta = tier_map.get(truth, 2) - tier_map.get(assigned, 2)
        result.extra["triage_assigned"]    = assigned
        result.extra["triage_truth"]       = truth
        result.extra["triage_tier_delta"]  = delta
        note = (
            f"Assigned={assigned} | Truth={truth} | "
            f"RawScore={raw:.2f} | Δtier={delta:+d}"
        )
        logger.debug("Task1 triage: %s", note)
        return raw, note
    def _infer_priority_from_unit(
        self, unit: str, gt: Dict[str, Any]
    ) -> str:
        if unit == "MICU":
            return "P1"
        if unit == "ALS":
            return gt.get("severity", "P2")   
        return "P3"
    @staticmethod
    def _normalise_priority_label(label: str) -> str:
        aliases = {
            "IMMEDIATE": "P1", "CRITICAL": "P1", "LIFE_THREATENING": "P1",
            "RED": "P1",       "1": "P1",
            "URGENT": "P2",    "SERIOUS": "P2", "YELLOW": "P2", "2": "P2",
            "DELAYED": "P2",
            "NON_URGENT": "P3","MINOR": "P3",  "GREEN": "P3", "3": "P3",
            "MINIMAL": "P3",
            "EXPECTANT": "P0", "BLACK": "P0",  "0": "P0",
        }
        return aliases.get(label, label if label in ("P1", "P2", "P3", "P0") else "P3")
    def _score_unit_type(
        self,
        action: ActionLogEntry,
        gt:     Dict[str, Any],
        result: GraderResult,
    ) -> Tuple[float, str]:
        dispatched: str = (
            action.get("unit_type")
            or action.get("ambulance_type")
            or action.get("unit")
            or "BLS"
        ).upper().strip()
        required: str = gt["required_unit_type"].upper()
        raw = ScoringUtils.unit_type_score(
            dispatched,
            (required,),
            severity=gt["severity"],
        )
        latency = gt.get("dispatch_latency_min")
        if dispatched == required and latency is not None and float(latency) < 5.0:
            bonus = min(0.05, 1.0 - raw)
            raw   = min(1.0, raw + bonus)
            result.add_note(f"Unit correct + fast dispatch ({latency:.1f} min) → +{bonus:.2f} bonus")
        cond = gt.get("condition_code", "")
        result.extra["dispatched_unit"]    = dispatched
        result.extra["required_unit"]      = required
        result.extra["unit_match"]         = dispatched == required
        result.extra["is_cardiac_arrest"]  = cond in CARDIAC_ARREST_CONDITIONS
        note = (
            f"Dispatched={dispatched} | Required={required} | "
            f"RawScore={raw:.2f}"
        )
        logger.debug("Task1 unit_type: %s", note)
        return raw, note
    def _score_hospital_match(
        self,
        action: ActionLogEntry,
        gt:     Dict[str, Any],
        gi:     GraderInput,
        result: GraderResult,
    ) -> Tuple[float, str]:
        agent_hospital_id: Optional[str] = (
            action.get("hospital_id")
            or action.get("destination_hospital")
            or action.get("hospital")
        )
        gt_hospital_id: Optional[str] = gt.get("hospital_id")
        h_state = self._resolve_hospital_state(
            agent_hospital_id, gi, gt
        )
        spec_score = self._score_specialty_match(h_state, gt, result)
        cap_score = self._score_capacity(h_state, gt, result)
        travel_score = self._score_travel_time(action, gt, gi, h_state, result)
        hospital_raw = (
            WH_SPECIALTY * spec_score +
            WH_CAPACITY  * cap_score  +
            WH_TRAVEL    * travel_score
        )
        hospital_raw = ScoringUtils.clamp(hospital_raw)
        if (agent_hospital_id and gt_hospital_id and
                agent_hospital_id == gt_hospital_id):
            hospital_raw = min(1.0, hospital_raw + 0.05)
            result.add_note("Agent chose optimal hospital — bonus +0.05")
        result.extra["hospital_id"]      = agent_hospital_id or "not_specified"
        result.extra["gt_hospital_id"]   = gt_hospital_id or "unknown"
        result.extra["spec_score"]        = round(spec_score, 4)
        result.extra["cap_score"]         = round(cap_score, 4)
        result.extra["travel_score"]      = round(travel_score, 4)
        note = (
            f"Hospital={agent_hospital_id or 'N/A'} | "
            f"Specialty={spec_score:.2f} | Capacity={cap_score:.2f} | "
            f"Travel={travel_score:.2f} | Weighted={hospital_raw:.2f}"
        )
        logger.debug("Task1 hospital: %s", note)
        return hospital_raw, note
    def _resolve_hospital_state(
        self,
        agent_hospital_id: Optional[str],
        gi:  GraderInput,
        gt:  Dict[str, Any],
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if gi.final_hospital_state and agent_hospital_id:
            live = gi.final_hospital_state.get(agent_hospital_id, {})
            if live:
                state = dict(live)
        if not state:
            state = {
                "hospital_id":           agent_hospital_id or gt.get("hospital_id"),
                "on_diversion":          gt.get("hospital_on_diversion", False),
                "er_occupancy_pct":      gt.get("hospital_er_occupancy_pct", 0.70),
                "has_required_specialty":gt.get("hospital_has_specialty", True),
                "is_level1_trauma":      gt.get("hospital_is_level1", False),
                "specialties":           gt.get("required_hospital_specialties", []),
                "icu_beds_available":    gt.get("icu_beds_available", 4),
                "travel_time_min":       gt.get("optimal_travel_time_min", 15.0),
            }
        state.setdefault("on_diversion", False)
        state.setdefault("er_occupancy_pct", 0.70)
        state.setdefault("has_required_specialty", True)
        state.setdefault("is_level1_trauma", False)
        state.setdefault("travel_time_min", 15.0)
        return state
    def _score_specialty_match(
        self,
        h_state: Dict[str, Any],
        gt:      Dict[str, Any],
        result:  GraderResult,
    ) -> float:
        on_div:    bool  = bool(h_state.get("on_diversion", False))
        occ:       float = float(h_state.get("er_occupancy_pct", 0.70))
        required_specs: List[str] = gt.get("required_hospital_specialties", [])
        hospital_specs: List[str] = h_state.get("specialties", [])
        has_spec_flag:  bool      = bool(h_state.get("has_required_specialty", True))
        if required_specs and hospital_specs:
            matched = sum(
                1 for s in required_specs
                if any(s.lower() in hs.lower() or hs.lower() in s.lower()
                       for hs in hospital_specs)
            )
            has_spec = matched == len(required_specs)
            partial  = 0 < matched < len(required_specs)
        else:
            has_spec = has_spec_flag
            partial  = False
        score = ScoringUtils.hospital_specialty_score(has_spec, on_div, occ)
        if partial and not on_div:
            n_req = len(required_specs)
            n_hit = matched  
            score = 0.20 + (score - 0.20) * (n_hit / n_req)
        result.extra["on_diversion"]       = on_div
        result.extra["er_occupancy"]       = round(occ, 3)
        result.extra["specialty_matched"]  = has_spec
        result.extra["required_specs"]     = required_specs
        return ScoringUtils.clamp(score)
    def _score_capacity(
        self,
        h_state: Dict[str, Any],
        gt:      Dict[str, Any],
        result:  GraderResult,
    ) -> float:
        on_div: bool  = bool(h_state.get("on_diversion", False))
        occ:    float = float(h_state.get("er_occupancy_pct", 0.70))
        if on_div:
            return 0.00
        if occ < 0.70:
            score = 1.00
        elif occ < 0.80:
            score = 0.80
        elif occ < 0.90:
            score = 0.55
        elif occ < 0.95:
            score = 0.25
        else:
            score = 0.05
        severity = gt.get("severity", "P2")
        unit     = gt.get("required_unit_type", "ALS")
        if severity == "P1" and unit == "MICU":
            icu_avail = int(h_state.get("icu_beds_available", 4))
            if icu_avail == 0:
                score = max(0.0, score - 0.15)
                result.add_note(
                    "ICU beds = 0 at chosen hospital for P1/MICU patient — capacity penalty"
                )
        return ScoringUtils.clamp(score)
    def _score_travel_time(
        self,
        action:  ActionLogEntry,
        gt:      Dict[str, Any],
        gi:      GraderInput,
        h_state: Dict[str, Any],
        result:  GraderResult,
    ) -> float:
        target_min: float = float(
            gt.get("response_time_target_min",
                   RESPONSE_TIME_TARGETS.get(gt.get("zone_type", ""), DEFAULT_RESPONSE_TARGET))
        )
        travel_min: float = float(
            action.get("travel_time_min")
            or action.get("eta_minutes")
            or h_state.get("travel_time_min")
            or gt.get("optimal_travel_time_min", 15.0)
        )
        result.extra["travel_time_min"]      = round(travel_min, 2)
        result.extra["response_target_min"]  = round(target_min, 2)
        if travel_min <= target_min:
            score = 1.00
        elif travel_min <= target_min * 1.5:
            excess = travel_min - target_min
            span   = target_min * 0.5
            score  = 1.00 - 0.40 * (excess / span)
        elif travel_min <= 60.0:
            excess = travel_min - target_min * 1.5
            span   = 60.0 - target_min * 1.5
            score  = 0.60 - 0.40 * (excess / max(span, 1.0))
        else:
            score = 0.00   
        severity = gt.get("severity", "P2")
        if severity == "P1" and travel_min > 60.0:
            result.add_note(
                f"P1 patient — golden hour breached ({travel_min:.0f} min > 60 min)"
            )
        return ScoringUtils.clamp(score)
    def _apply_protocol_checks(
        self,
        action: ActionLogEntry,
        gt:     Dict[str, Any],
        gi:     GraderInput,
        result: GraderResult,
    ) -> float:
        cond              = gt.get("condition_code", "unknown")
        dispatched        = result.extra.get("dispatched_unit", "BLS")
        on_div            = result.extra.get("on_diversion", False)
        h_id              = result.extra.get("hospital_id", "unknown")
        is_l1             = bool(gt.get("hospital_is_level1", False))
        travel_min        = result.extra.get("travel_time_min", 999.0)
        cath_activated    = bool(action.get("cath_lab_activated", False))
        stroke_notified   = bool(action.get("stroke_unit_notified", False))
        multi_agency      = bool(action.get("multi_agency_coordinated", False))
        trapped           = bool(gt.get("requires_extrication", False))
        net_score, penalties, notes = ProtocolRuleChecker.run_all_checks(
            condition_key             = cond,
            dispatched_unit           = dispatched,
            hospital_on_diversion     = on_div,
            hospital_id               = h_id,
            hospital_is_level1        = is_l1,
            response_time_min         = travel_min,
            cath_lab_activated        = cath_activated,
            stroke_unit_notified      = stroke_notified,
            multi_agency_coordinated  = multi_agency,
            trapped_victim            = trapped,
            is_mci                    = False,
            start_protocol_applied    = False,
            mutual_aid_requested      = False,
            surge_declared            = False,
        )
        for p in penalties:
            result.add_penalty(p)
        for n in notes:
            result.add_note(n)
        result.protocol_violations += len(penalties)
        capped = ScoringUtils.clamp(net_score, -PROTOCOL_BONUS_CAP, PROTOCOL_BONUS_CAP)
        if abs(capped - net_score) > 1e-6:
            result.add_note(
                f"Protocol net_score {net_score:.3f} capped to {capped:.3f}"
            )
        if capped > 0.0:
            result.add_component(
                ScoringUtils.build_score_component(
                    "protocol_bonus", capped, 1.0,
                    f"Protocol compliance net bonus ({len(notes)} checks)"
                )
            )
        elif capped < 0.0:
            result.add_penalty(
                ScoringUtils.build_penalty(
                    "protocol_net_penalty", abs(capped),
                    f"Protocol net penalty ({len(penalties)} violations)"
                )
            )
        return capped
    def _apply_hard_penalties(
        self,
        action: ActionLogEntry,
        gt:     Dict[str, Any],
        result: GraderResult,
    ) -> None:
        assigned   = result.extra.get("triage_assigned", "P3")
        truth      = result.extra.get("triage_truth", "P1")
        dispatched = result.extra.get("dispatched_unit", "BLS")
        on_div     = result.extra.get("on_diversion", False)
        cond       = gt.get("condition_code", "")
        severity   = gt.get("severity", "P1")
        hard_count = 0
        if truth == "P1" and assigned == "P3":
            self._add_penalty(
                result,
                "critical_under_triage",
                abs(HARD_PENALTY_P1_AS_P3),
                "P1 patient classified as P3 — life-threatening under-triage",
                rule_ref="EMERGI-ENV Rule H1",
            )
            result.critical_mismatches += 1
            hard_count += 1
            result.add_note(
                "⚠ CRITICAL: P1→P3 triage error. "
                "In real EMS this would result in preventable patient death."
            )
        if (cond in CARDIAC_ARREST_CONDITIONS and
                dispatched == "BLS" and severity == "P1"):
            self._add_penalty(
                result,
                "bls_for_cardiac_arrest",
                abs(HARD_PENALTY_BLS_FOR_CARDIAC),
                f"BLS dispatched for {cond} — MICU mandatory for cardiac events",
                rule_ref="EMERGI-ENV Rule H2",
            )
            result.critical_mismatches += 1
            hard_count += 1
            result.add_note(
                f"⚠ CRITICAL: BLS sent to cardiac arrest ({cond}). "
                "MICU with defib and ALS drugs is mandatory."
            )
        if on_div and severity == "P1":
            self._add_penalty(
                result,
                "p1_to_diverted_hospital",
                abs(HARD_PENALTY_DIVERTED_P1),
                "P1 patient routed to hospital on diversion",
                rule_ref="EMERGI-ENV Rule H3",
            )
            result.critical_mismatches += 1
            hard_count += 1
            result.add_note(
                "⚠ CRITICAL: P1 patient sent to diverted hospital. "
                "ED will refuse and force redirect — dangerous delay."
            )
        if hard_count > 0:
            result.extra["hard_penalty_count"] = hard_count
            result.status = GraderStatus.PARTIAL
    def _integrate_survival_reward(
        self,
        gt:     Dict[str, Any],
        result: GraderResult,
    ) -> None:
        p_actual  = float(gt.get("final_survival_prob", 0.80))
        p_optimal = float(gt.get("optimal_survival_prob", 0.95))
        p_worst   = 0.05
        surv_score = ScoringUtils.survival_probability_score(
            p_actual, p_optimal, p_worst
        )
        result.p1_survival_rate = p_actual
        result.extra["final_survival_prob"]  = round(p_actual, 4)
        result.extra["optimal_survival_prob"]= round(p_optimal, 4)
        result.extra["survival_score"]       = round(surv_score, 4)
        if p_actual < 0.50:
            result.add_note(
                f"Low patient survival probability: {p_actual:.1%}. "
                "Faster dispatch or better hospital matching required."
            )
    def _apply_no_action_scores(self, result: GraderResult) -> None:
        self._add_component(
            result, "triage_class", 0.0, W_TRIAGE, "No action — zero score"
        )
        self._add_component(
            result, "unit_type", 0.0, W_UNIT, "No action — zero score"
        )
        self._add_component(
            result, "hospital_match", 0.0, W_HOSPITAL, "No action — zero score"
        )
        self._add_penalty(
            result, "no_action_penalty", 0.20,
            "Agent issued no dispatch action for single-incident task",
            rule_ref="EMERGI-ENV Rule N1",
        )
GraderRegistry.register(TASK_ID, Task1Grader)
logger.info(
    "Task1Grader registered: task_id=%s baseline=%.2f seed=%d",
    TASK_ID, TASK_BASELINE, TASK_SEED,
)
def _build_test_input(
    dispatched_unit:    str  = "MICU",
    assigned_priority:  str  = "P1",
    hospital_id:        str  = "H_PUNE_KEM",
    cath_activated:     bool = True,
    severity:           str  = "P1",
    condition_code:     str  = "stemi_anterior",
    required_unit:      str  = "MICU",
    on_diversion:       bool = False,
    er_occupancy:       float = 0.65,
    travel_time:        float = 11.0,
    dispatch_latency:   float = 4.5,
    episode_id:         str   = "ep-test-001",
) -> GraderInput:
    return GraderInput(
        task_id      = TASK_ID,
        episode_id   = episode_id,
        seed         = TASK_SEED,
        action_log   = [
            ActionLogEntry(
                step=1,
                action_type="dispatch",
                action_data={
                    "incident_id":            "INC-0001",
                    "unit_type":              dispatched_unit,
                    "assigned_priority":      assigned_priority,
                    "hospital_id":            hospital_id,
                    "travel_time_min":        travel_time,
                    "cath_lab_activated":     cath_activated,
                    "stroke_unit_notified":   False,
                    "multi_agency_coordinated": False,
                },
            )
        ],
        episode_ledger={
            "patient_summaries": [
                {
                    "patient_id":                    "P-001",
                    "severity":                      severity,
                    "condition_code":                condition_code,
                    "required_unit_type":            required_unit,
                    "required_hospital_specialties": ["cardiology", "cath_lab"],
                    "zone_id":                       "Z05",
                    "zone_type":                     "metro_core",
                    "hospital_id":                   hospital_id,
                    "hospital_on_diversion":         on_diversion,
                    "hospital_er_occupancy_pct":     er_occupancy,
                    "hospital_has_specialty":        True,
                    "hospital_is_level1":            False,
                    "response_time_target_min":      8.0,
                    "optimal_travel_time_min":       10.0,
                    "dispatch_latency_min":          dispatch_latency,
                    "phase_at_episode_end":          "treated",
                    "final_survival_prob":           0.91,
                    "optimal_survival_prob":         0.96,
                    "weight":                        1.0,
                    "weighted_reward":               0.91,
                    "requires_extrication":          False,
                }
            ]
        },
        observation_log  = [],
        episode_steps    = 5,
        total_patients   = 1,
        p1_patients      = 1,
    )
def _run_self_tests() -> None:
    import traceback
    grader   = Task1Grader()
    failures: List[str] = []
    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
    gi   = _build_test_input()
    res  = grader.grade(gi)
    check("T1_range",   0.0 <= res.final_score <= 1.0, f"score={res.final_score}")
    check("T1_baseline",res.final_score >= TASK_BASELINE,
          f"score {res.final_score:.4f} < baseline {TASK_BASELINE}")
    check("T1_status",  res.status in (GraderStatus.SUCCESS, GraderStatus.PARTIAL))
    check("T1_components", len(res.components) >= 3, f"got {len(res.components)} components")
    gi2  = _build_test_input()
    res2 = grader.grade(gi2)
    check("T2_determinism",
          abs(res.final_score - res2.final_score) < 1e-9,
          f"{res.final_score} ≠ {res2.final_score}")
    gi3  = _build_test_input(dispatched_unit="BLS", assigned_priority="P1")
    res3 = grader.grade(gi3)
    check("T3_bls_stemi_score_lower",
          res3.final_score < res.final_score,
          f"BLS score {res3.final_score:.4f} should be < MICU {res.final_score:.4f}")
    check("T3_has_hard_penalty",
          any("bls_for_cardiac_arrest" in p.name for p in res3.penalties),
          "Missing BLS-cardiac-arrest hard penalty")
    check("T3_range", 0.0 <= res3.final_score <= 1.0)
    gi4  = _build_test_input(assigned_priority="P3", dispatched_unit="BLS",
                              condition_code="stemi_anterior")
    res4 = grader.grade(gi4)
    check("T4_critical_undertriage",
          any("critical_under_triage" in p.name for p in res4.penalties),
          "Missing P1→P3 under-triage hard penalty")
    check("T4_range", 0.0 <= res4.final_score <= 1.0)
    gi5  = _build_test_input(on_diversion=True)
    res5 = grader.grade(gi5)
    check("T5_diversion_penalty",
          any("diverted" in p.name for p in res5.penalties),
          "Missing diversion hard penalty")
    check("T5_range", 0.0 <= res5.final_score <= 1.0)
    gi6  = GraderInput(
        task_id=TASK_ID, episode_id="ep-noaction", seed=TASK_SEED,
        action_log=[], episode_ledger={"patient_summaries": [
            {"patient_id": "P-x", "severity": "P1",
             "required_unit_type": "MICU",
             "condition_code": "stemi_anterior",
             "required_hospital_specialties": ["cardiology"],
             "phase_at_episode_end": "untreated",
             "weight": 1.0, "weighted_reward": 0.0}
        ]},
        observation_log=[], episode_steps=0,
        total_patients=1, p1_patients=1,
    )
    res6 = grader.grade(gi6)
    check("T6_no_action_zero", res6.final_score == 0.0,
          f"No-action score should be 0.0, got {res6.final_score}")
    check("T6_no_action_status",
          res6.status == GraderStatus.PARTIAL,
          f"Expected PARTIAL, got {res6.status}")
    gi7  = _build_test_input(
        severity="P3", required_unit="BLS",
        dispatched_unit="MICU", assigned_priority="P3",
        condition_code="general_weakness",
    )
    res7 = grader.grade(gi7)
    check("T7_range", 0.0 <= res7.final_score <= 1.0)
    unit_comp = next((c for c in res7.components if c.name == "unit_type"), None)
    if unit_comp:
        check("T7_over_equipped_score",
              unit_comp.raw_score >= 0.80,
              f"Over-equipped raw={unit_comp.raw_score:.4f}, expected ≥ 0.80")
    gi8  = _build_test_input(cath_activated=True)
    res8 = grader.grade(gi8)
    check("T8_protocol_bonus",
          res8.final_score >= TASK_BASELINE,
          f"Protocol bonus expected to push above baseline; got {res8.final_score:.4f}")
    gi9a = _build_test_input(travel_time=8.0)    
    gi9b = _build_test_input(travel_time=75.0)   
    r9a  = grader.grade(gi9a)
    r9b  = grader.grade(gi9b)
    check("T9_travel_penalty",
          r9a.final_score > r9b.final_score,
          f"Fast ETA {r9a.final_score:.4f} should beat slow {r9b.final_score:.4f}")
    d    = res.as_dict()
    check("T10_dict_keys",
          all(k in d for k in ("final_score", "components", "penalties", "notes")))
    check("T10_json_valid",
          len(res.as_json()) > 100)
    check("T10_summary_line",
          TASK_ID in res.summary_line())
    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task1Grader self-test: {len(failures)} failure(s):\n" +
            "\n".join(failures)
        )
    logger.info(
        "Task1Grader self-test PASSED (10 test cases, %d component checks).",
        len(res.components),
    )
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task1Grader self-test FAILED at import: %s", _e)
    raise
if __name__ == "__main__":
    import json as _json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 70)
    print("EMERGI-ENV  ·  Task1Grader  ·  Interactive demo")
    print("=" * 70)
    grader = Task1Grader()
    scenarios = [
        ("Perfect STEMI dispatch",
         _build_test_input(dispatched_unit="MICU", assigned_priority="P1",
                           cath_activated=True, travel_time=9.0)),
        ("Under-equipped: ALS for STEMI",
         _build_test_input(dispatched_unit="ALS", assigned_priority="P1",
                           cath_activated=False)),
        ("Critical under-triage: P1→P3",
         _build_test_input(dispatched_unit="BLS", assigned_priority="P3")),
        ("Diverted hospital, P1 patient",
         _build_test_input(on_diversion=True)),
        ("Perfect P3 routine (BLS correct)",
         _build_test_input(severity="P3", required_unit="BLS",
                           dispatched_unit="BLS", assigned_priority="P3",
                           condition_code="general_weakness",
                           travel_time=6.0, er_occupancy=0.55)),
        ("High ER occupancy (90%)",
         _build_test_input(er_occupancy=0.92, travel_time=14.0)),
        ("Golden-hour breach (80 min travel)",
         _build_test_input(travel_time=80.0)),
    ]
    results_list = []
    for name, gi in scenarios:
        res = grader.grade(gi)
        results_list.append(res)
        beats = "✓" if res.beats_baseline else "✗"
        print(f"\n  [{beats}] {name}")
        print(f"       Score:     {res.final_score:.4f}  (baseline {TASK_BASELINE:.2f}, "
              f"Δ={res.score_delta_vs_baseline:+.4f})")
        print(f"       Status:    {res.status.value}")
        print(f"       Components:")
        for c in res.components:
            if c.name == "protocol_bonus":
                continue
            print(f"         {c.name:<18} raw={c.raw_score:.4f}  "
                  f"w={c.weight:.2f}  weighted={c.weighted:.4f}  | {c.notes}")
        if res.penalties:
            print(f"       Penalties ({len(res.penalties)}):")
            for p in res.penalties:
                print(f"         {p.name:<28} {p.amount:+.4f}  | {p.reason}")
        if res.notes:
            for n in res.notes[:3]:
                print(f"         NOTE: {n}")
    print("\n" + "=" * 70)
    print("Baseline comparison:")
    beats = sum(1 for r in results_list if r.beats_baseline)
    print(f"  {beats}/{len(results_list)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 70)
    print("\n✅  Task1Grader demo complete.")