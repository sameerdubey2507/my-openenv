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

logger = logging.getLogger("emergi_env.graders.task2")

TASK_ID       = "task2_hospital_route"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]

W_SPECIALTY = 0.50
W_CAPACITY  = 0.30
W_TRAVEL    = 0.20

WS_SPECIALTY_MATCH = 1.00   

WC_ER       = 0.45
WC_ICU      = 0.25
WC_THEATRE  = 0.20
WC_WARD     = 0.10

WT_RESPONSE = 0.50
WT_GOLDEN   = 0.30
WT_ROUTE    = 0.20

PROTOCOL_BONUS_CAP   = 0.15
PROTOCOL_PENALTY_CAP = 0.20
COMPARATIVE_WORST_THRESHOLD = 0.20
COMPARATIVE_PENALTY         = -0.10

OCC_BANDS: List[Tuple[float, float, float]] = [
    (0.00, 0.80, 1.00),
    (0.80, 0.90, 0.80),
    (0.90, 0.95, 0.45),
    (0.95, 1.01, 0.10),   
]

MCI_BANDS: List[Tuple[float, float, float]] = [
    (0.00, 0.50, 1.00),
    (0.50, 0.65, 0.85),
    (0.65, 0.75, 0.65),
    (0.75, 0.85, 0.40),
    (0.85, 0.90, 0.20),
    (0.90, 1.01, 0.05),
]

GOLDEN_HOUR_THRESHOLDS: Dict[str, float] = {
    "stemi_anterior":           90.0,
    "stemi_inferior":           90.0,
    "stemi_posterior":          90.0,
    "stemi_with_vf_arrest":     60.0,
    "cardiac_arrest":           10.0,
    "cardiac_arrest_vf":        10.0,
    "cardiac_arrest_pea":       10.0,
    "ischaemic_stroke":        270.0,   
    "ischemic_stroke":         270.0,
    "haemorrhagic_stroke_sah":  60.0,
    "haemorrhagic_stroke":      60.0,
    "polytrauma_blunt":         60.0,
    "polytrauma_penetrating":   45.0,
    "severe_tbi":               60.0,
    "blast_injury":             45.0,
    "penetrating_trauma":       45.0,
    "eclampsia":                30.0,
    "postpartum_haemorrhage":   30.0,
    "major_burns_40pct":        60.0,
    "inhalation_injury":        30.0,
    "febrile_seizure":          20.0,
    "choking_infant":            5.0,
    "near_drowning":            30.0,
    "septic_shock":             60.0,
    "anaphylaxis":              20.0,
    "anaphylaxis_refractory":   15.0,
    "acute_asthma_severe":      30.0,
    "pulmonary_oedema":         30.0,
    "massive_pe":               60.0,
    "default_p1":               60.0,
    "default_p2":              120.0,
    "default_p3":              240.0,
}

CATH_LAB_CONDITIONS = {
    "stemi_anterior", "stemi_inferior", "stemi_posterior",
    "stemi_with_vf_arrest", "stemi_post_cabg", "stemi_cocaine",
    "cardiac_arrest_vf", "cardiac_arrest_pea",
}

STROKE_UNIT_CONDITIONS = {
    "ischaemic_stroke", "ischemic_stroke",
    "hemorrhagic_stroke", "haemorrhagic_stroke_sah",
    "paediatric_stroke", "tia",
}

TRAUMA_CENTRE_CONDITIONS = {
    "polytrauma_blunt", "polytrauma_penetrating",
    "severe_tbi", "blast_injury", "mci_rta",
    "crush_syndrome", "penetrating_trauma",
}

BURNS_UNIT_CONDITIONS = {
    "major_burns_40pct", "major_burns", "inhalation_injury",
    "chemical_burns", "electrical_burns",
}

PAEDIATRIC_CONDITIONS = {
    "febrile_seizure", "choking_infant", "near_drowning",
    "paediatric_cardiac_arrest", "paediatric_stroke",
    "paediatric_trauma", "kawasaki", "intussusception",
}

SURGICAL_CONDITIONS = {
    "polytrauma_blunt", "polytrauma_penetrating", "severe_tbi",
    "appendicitis_perforated", "bowel_obstruction",
    "aortic_dissection", "ectopic_pregnancy_ruptured",
    "gi_bleed_variceal", "blast_injury",
}

BLOOD_BANK_CONDITIONS = {
    "polytrauma_blunt", "polytrauma_penetrating",
    "gi_bleed_variceal", "postpartum_haemorrhage",
    "aortic_dissection", "ectopic_pregnancy_ruptured",
    "major_burns_40pct",
}

ZONE_RESPONSE_TARGETS: Dict[str, float] = {
    "metro_core":         8.0,
    "metro_suburban":    10.0,
    "metro_satellite":   12.0,
    "metro_tier2":       14.0,
    "metro_peri_urban":  14.0,
    "tier2_city":        15.0,
    "tier3_city":        20.0,
    "semi_urban_coastal":18.0,
    "semi_rural":        22.0,
    "rural":             26.0,
    "tribal_rural":      32.0,
    "tribal_forest":     45.0,
    "coastal_rural":     30.0,
    "coastal_semi_urban":28.0,
    "industrial_town":   18.0,
}
DEFAULT_RESPONSE_TARGET = 20.0

class HospitalEvaluator:
    @classmethod
    def score_specialty(
        cls,
        h:          Dict[str, Any],
        patient:    Dict[str, Any],
        condition:  str,
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        on_div: bool  = bool(h.get("on_diversion", False))
        occ:    float = float(h.get("er_occupancy_pct", 0.70))

        required_specs: List[str] = patient.get("required_hospital_specialties", [])
        hospital_specs: List[str] = h.get("specialties", [])

        is_l1:          bool      = bool(h.get("is_level1_trauma", False))
        is_stroke_ctr:  bool      = bool(h.get("is_stroke_centre", False))
        is_paed:        bool      = bool(h.get("is_paediatric_hospital", False))
        is_burns:       bool      = bool(h.get("is_burns_unit", False))
        has_cath_lab:   bool      = bool(h.get("has_cath_lab", False))

        if not required_specs:
            base_score = 0.70
            notes.append("No specialty requirement — generic match")
        else:
            matched_n = 0
            for req in required_specs:
                req_l = req.lower().replace(" ", "_")
                for cap in hospital_specs:
                    cap_l = cap.lower().replace(" ", "_")
                    if req_l in cap_l or cap_l in req_l:
                        matched_n += 1
                        break

            n_req = len(required_specs)
            match_ratio = matched_n / max(n_req, 1)

            if match_ratio == 1.0:
                base_score = 1.00
                notes.append(f"All {n_req} specialties matched")
            elif match_ratio > 0.5:
                base_score = 0.50 + 0.35 * match_ratio
                notes.append(f"Partial specialty match {matched_n}/{n_req}")
            elif match_ratio > 0:
                base_score = 0.15 + 0.35 * match_ratio
                notes.append(f"Weak specialty match {matched_n}/{n_req}")
            else:
                base_score = 0.10
                notes.append(f"No matching specialties (required: {required_specs})")

        occ_modifier = 1.00
        for lo, hi, mod in OCC_BANDS:
            if lo <= occ < hi:
                occ_modifier = mod
                break

        if on_div:
            score = min(base_score * 0.12, 0.10)
            notes.append("HOSPITAL ON DIVERSION — severely penalised")
        else:
            score = base_score * occ_modifier

        bonus = 0.0
        if condition in CATH_LAB_CONDITIONS and has_cath_lab:
            bonus += 0.05
            notes.append("Cath lab available — bonus +0.05")
        if condition in STROKE_UNIT_CONDITIONS and is_stroke_ctr:
            bonus += 0.05
            notes.append("Stroke centre — bonus +0.05")
        if condition in TRAUMA_CENTRE_CONDITIONS and is_l1:
            bonus += 0.05
            notes.append("Level-1 trauma centre — bonus +0.05")
        if condition in PAEDIATRIC_CONDITIONS and is_paed:
            bonus += 0.05
            notes.append("Paediatric hospital — bonus +0.05")
        if condition in BURNS_UNIT_CONDITIONS and is_burns:
            bonus += 0.05
            notes.append("Burns unit — bonus +0.05")

        if required_specs and hospital_specs:
            extras = max(0, len(hospital_specs) - len(required_specs))
            extra_bonus = min(0.06, extras * 0.02)
            if extra_bonus > 0:
                bonus += extra_bonus
                notes.append(f"{extras} extra specialties — bonus +{extra_bonus:.2f}")

        final = ScoringUtils.clamp(score + bonus)
        return final, notes

    @classmethod
    def score_capacity(
        cls,
        h:       Dict[str, Any],
        patient: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        on_div = bool(h.get("on_diversion", False))

        if on_div:
            notes.append("On diversion — capacity score 0")
            return 0.00, notes

        er_occ     = float(h.get("er_occupancy_pct", 0.70))
        icu_total  = int(h.get("icu_beds_total", 20))
        icu_avail  = int(h.get("icu_beds_available", 8))
        ward_total = int(h.get("ward_beds_total", 150))
        ward_avail = int(h.get("ward_beds_available", 40))

        theatre_ok = bool(h.get("theatre_available", True))
        blood_bank = bool(h.get("has_blood_bank", False))

        icu_occ    = 1.0 - (icu_avail  / max(icu_total,  1))
        ward_occ   = 1.0 - (ward_avail / max(ward_total, 1))
        theatre_occ = 0.0 if theatre_ok else 1.0

        mci = (
            WC_ER      * er_occ     +
            WC_ICU     * icu_occ    +
            WC_THEATRE * theatre_occ +
            WC_WARD    * ward_occ
        )
        mci = ScoringUtils.clamp(mci)

        cap_score = 0.05
        for lo, hi, val in MCI_BANDS:
            if lo <= mci < hi:
                cap_score = val
                break

        notes.append(
            f"MCI={mci:.3f} (ER={er_occ:.0%} ICU={icu_occ:.0%} "
            f"Ward={ward_occ:.0%} Theatre={'OK' if theatre_ok else 'BUSY'})"
        )

        severity    = patient.get("severity", "P2")
        needs_icu   = bool(patient.get("requires_icu", severity == "P1"))
        needs_theatre = bool(patient.get("requires_surgery", False))
        needs_blood = bool(patient.get("requires_blood_bank", False))
        condition   = patient.get("condition_code", "unknown")

        penalty = 0.0

        if needs_icu and icu_avail == 0:
            penalty += 0.20
            notes.append("⚠ ICU beds = 0 for P1/MICU patient — penalty −0.20")

        if needs_theatre and not theatre_ok:
            penalty += 0.15
            notes.append("⚠ Theatre unavailable for surgical patient — penalty −0.15")

        if needs_blood and not blood_bank:
            penalty += 0.10
            notes.append("⚠ No blood bank for haemorrhage patient — penalty −0.10")
        elif needs_blood and blood_bank:
            cap_score = min(1.0, cap_score + 0.05)
            notes.append("Blood bank available — bonus +0.05")

        final = ScoringUtils.clamp(cap_score - penalty)
        return final, notes

    @classmethod
    def score_travel(
        cls,
        travel_min:     float,
        zone_type:      str,
        condition:      str,
        severity:       str,
        hotspot_count:  int,
        route_provided: bool,
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []

        target_min = ZONE_RESPONSE_TARGETS.get(zone_type, DEFAULT_RESPONSE_TARGET)
        gh_thresh = GOLDEN_HOUR_THRESHOLDS.get(condition)
        if gh_thresh is None:
            gh_thresh = GOLDEN_HOUR_THRESHOLDS.get(
                f"default_{severity.lower()}", 60.0
            )

        if travel_min <= target_min:
            raw_a = 1.00
        elif travel_min <= target_min * 1.5:
            excess = travel_min - target_min
            raw_a  = 1.00 - 0.40 * (excess / (target_min * 0.5))
        elif travel_min <= 60.0:
            excess  = travel_min - target_min * 1.5
            span    = 60.0 - target_min * 1.5
            raw_a   = 0.60 - 0.40 * (excess / max(span, 1.0))
        else:
            raw_a = {"P1": 0.00, "P2": 0.10, "P3": 0.30}.get(severity, 0.05)
            notes.append(
                f"Golden-hour breach: {travel_min:.0f} min > 60 min "
                f"(severity={severity})"
            )
        raw_a = ScoringUtils.clamp(raw_a)
        notes.append(
            f"Response-time: {travel_min:.1f} min vs target {target_min:.0f} min "
            f"→ A={raw_a:.3f}"
        )

        if gh_thresh <= 0:
            raw_b = 0.50   
        elif travel_min <= gh_thresh:
            raw_b = 1.00
        elif travel_min <= gh_thresh * 1.5:
            raw_b = 0.70
        elif travel_min <= gh_thresh * 2.0:
            raw_b = 0.40
        else:
            raw_b = 0.00
            notes.append(
                f"Well beyond golden hour: {travel_min:.0f} min "
                f"vs {gh_thresh:.0f} min threshold for {condition}"
            )
        raw_b = ScoringUtils.clamp(raw_b)
        notes.append(f"Golden-hour {condition}: {travel_min:.1f}/{gh_thresh:.0f} min → B={raw_b:.3f}")

        if not route_provided:
            raw_c = 0.60   
            notes.append("Route not provided — route quality neutral 0.60")
        elif hotspot_count == 0:
            raw_c = 1.00
        elif hotspot_count == 1:
            raw_c = 0.75
        elif hotspot_count == 2:
            raw_c = 0.50
        else:
            raw_c = max(0.10, 0.50 - 0.10 * (hotspot_count - 2))
            notes.append(
                f"{hotspot_count} congestion hotspots on route — "
                f"significant delay risk"
            )
        raw_c = ScoringUtils.clamp(raw_c)
        notes.append(f"Route quality ({hotspot_count} hotspots) → C={raw_c:.3f}")

        travel_score = (
            WT_RESPONSE * raw_a +
            WT_GOLDEN   * raw_b +
            WT_ROUTE    * raw_c
        )
        travel_score = ScoringUtils.clamp(travel_score)
        notes.append(
            f"Travel weighted: {WT_RESPONSE}×{raw_a:.3f} + "
            f"{WT_GOLDEN}×{raw_b:.3f} + {WT_ROUTE}×{raw_c:.3f} = {travel_score:.3f}"
        )

        return travel_score, notes

    @classmethod
    def composite_score(
        cls,
        h:            Dict[str, Any],
        patient:      Dict[str, Any],
        travel_min:   float,
        zone_type:    str,
        hotspot_count: int = 0,
        route_provided: bool = False,
    ) -> Tuple[float, Dict[str, float], List[str]]:

        condition = patient.get("condition_code", "unknown")
        severity  = patient.get("severity", "P2")

        s_score, s_notes = cls.score_specialty(h, patient, condition)
        c_score, c_notes = cls.score_capacity(h, patient)
        t_score, t_notes = cls.score_travel(
            travel_min, zone_type, condition, severity,
            hotspot_count, route_provided
        )

        composite = (
            W_SPECIALTY * s_score +
            W_CAPACITY  * c_score +
            W_TRAVEL    * t_score
        )

        return ScoringUtils.clamp(composite), {
            "specialty": s_score,
            "capacity":  c_score,
            "travel":    t_score,
        }, s_notes + c_notes + t_notes

class Task2Grader(BaseGrader):

    TASK_ID          = TASK_ID
    TASK_SEED        = TASK_SEED
    TASK_BASELINE    = TASK_BASELINE
    TASK_DIFFICULTY  = "easy"

    COMPONENT_WEIGHTS: Dict[str, float] = {
        "specialty_match": W_SPECIALTY,
        "capacity_check":  W_CAPACITY,
        "travel_time":     W_TRAVEL,
    }

    def __init__(self) -> None:
        super().__init__()
        self._evaluator = HospitalEvaluator()

    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(
                f"WARNING: seed {gi.seed} ≠ task seed {TASK_SEED}. "
                "Determinism not guaranteed."
            )

        patient = self._extract_patient(gi)
        if patient is None:
            result.status        = GraderStatus.INVALID_INPUT
            result.error_message = "Missing patient_summaries in episode_ledger."
            return

        action = self._extract_routing_action(gi)
        if action is None:
            result.add_note("No routing action found — zero score.")
            self._apply_zero_scores(result)
            result.status = GraderStatus.PARTIAL
            return

        agent_hospital_id = self._extract_hospital_id(action)
        travel_min        = self._extract_travel_time(action, gi, patient)
        hotspot_count     = len(action.get("congestion_hotspots") or [])
        route_provided    = bool(action.get("route_path") or action.get("path"))

        hospital_network  = self._resolve_hospital_network(gi)

        agent_h_state     = self._get_hospital_state(
            agent_hospital_id, hospital_network, gi, patient
        )

        s_score, s_notes = HospitalEvaluator.score_specialty(
            agent_h_state, patient, patient.get("condition_code", "unknown")
        )
        self._add_component(
            result, "specialty_match", s_score, W_SPECIALTY,
            " | ".join(s_notes[:3])
        )

        c_score, c_notes = HospitalEvaluator.score_capacity(
            agent_h_state, patient
        )
        self._add_component(
            result, "capacity_check", c_score, W_CAPACITY,
            " | ".join(c_notes[:3])
        )

        zone_type = patient.get("zone_type", "metro_core")
        t_score, t_notes = HospitalEvaluator.score_travel(
            travel_min, zone_type,
            patient.get("condition_code", "unknown"),
            patient.get("severity", "P2"),
            hotspot_count, route_provided,
        )
        self._add_component(
            result, "travel_time", t_score, W_TRAVEL,
            " | ".join(t_notes[:3])
        )

        comparative = self._run_comparative_analysis(
            agent_hospital_id, travel_min, hotspot_count, route_provided,
            patient, hospital_network, zone_type, result,
        )

        proto_bonus = self._apply_protocol_bonuses(
            action, patient, agent_h_state, travel_min, result
        )

        if comparative.get("apply_comparative_penalty", False):
            self._add_penalty(
                result,
                "comparative_routing_penalty",
                abs(COMPARATIVE_PENALTY),
                "Inferior hospital chosen when clearly better option existed",
                rule_ref="EMERGI-ENV Rule T2-C1",
            )
            result.add_note(
                f"Comparative penalty: agent rank {comparative.get('agent_rank', '?')} of "
                f"{comparative.get('total_candidates', '?')} candidates"
            )

        result.extra.update({
            "agent_hospital_id":            agent_hospital_id or "N/A",
            "optimal_hospital_id":          gi.episode_ledger.get(
                                               "optimal_hospital_id", "N/A"),
            "travel_time_min":              round(travel_min, 2),
            "hospital_on_diversion":        bool(agent_h_state.get("on_diversion")),
            "er_occupancy":                 round(float(agent_h_state.get(
                                               "er_occupancy_pct", 0.7)), 3),
            "icu_available":                int(agent_h_state.get(
                                               "icu_beds_available", 0)),
            "theatre_available":            bool(agent_h_state.get(
                                               "theatre_available", True)),
            "blood_bank":                   bool(agent_h_state.get(
                                               "has_blood_bank", False)),
            "is_level1":                    bool(agent_h_state.get(
                                               "is_level1_trauma", False)),
            "specialty_score":              round(s_score, 4),
            "capacity_score":               round(c_score, 4),
            "travel_score":                 round(t_score, 4),
            "protocol_bonus":               round(proto_bonus, 4),
            **{k: v for k, v in comparative.items()
               if k not in ("apply_comparative_penalty",)},
        })

        result.total_patients   = 1
        result.p1_patients      = 1 if patient.get("severity") == "P1" else 0
        result.p1_survival_rate = float(patient.get("final_survival_prob", 0.85))

    def _run_comparative_analysis(
        self,
        agent_hospital_id: Optional[str],
        travel_min:        float,
        hotspot_count:     int,
        route_provided:    bool,
        patient:           Dict[str, Any],
        hospital_network:  Dict[str, Dict[str, Any]],
        zone_type:         str,
        result:            GraderResult,
    ) -> Dict[str, Any]:
        if not hospital_network:
            return {"hospitals_evaluated": 0, "agent_rank": 1}

        candidate_scores: List[Tuple[str, float]] = []

        for hid, h_state in hospital_network.items():
            if hid == agent_hospital_id:
                h_travel = travel_min
            else:
                h_travel = float(h_state.get("estimated_travel_min", travel_min * 1.2))

            composite, subs, _ = HospitalEvaluator.composite_score(
                h_state, patient, h_travel, zone_type,
                hotspot_count if hid == agent_hospital_id else 0,
                route_provided if hid == agent_hospital_id else False,
            )
            candidate_scores.append((hid, composite))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        total = len(candidate_scores)
        agent_rank = next(
            (i + 1 for i, (hid, _) in enumerate(candidate_scores)
             if hid == agent_hospital_id),
            total,
        )
        best_score  = candidate_scores[0][1] if candidate_scores else 0.0
        worst_score = candidate_scores[-1][1] if candidate_scores else 0.0
        agent_score = next(
            (s for hid, s in candidate_scores if hid == agent_hospital_id),
            0.0,
        )

        apply_penalty = (
            agent_score < COMPARATIVE_WORST_THRESHOLD and
            best_score > agent_score + 0.25 and
            total > 1
        )

        if agent_rank == 1 and total > 1:
            result.add_note(
                f"Agent selected OPTIMAL hospital among {total} candidates — excellent routing"
            )
        if 1 < agent_rank <= 3 and total > 3:
            result.add_note(
                f"Agent chose rank {agent_rank} out of {total} candidates — good routing"
            )

        result.extra["candidate_ranking"] = [
            {"rank": i+1, "hospital_id": hid, "score": round(s, 4)}
            for i, (hid, s) in enumerate(candidate_scores[:5])   
        ]

        return {
            "hospitals_evaluated":         total,
            "agent_rank":                  agent_rank,
            "best_available_score":        round(best_score, 4),
            "worst_available_score":       round(worst_score, 4),
            "agent_hospital_score":        round(agent_score, 4),
            "optimal_hospital_id":         candidate_scores[0][0] if candidate_scores else "N/A",
            "score_gap_to_optimal":        round(best_score - agent_score, 4),
            "apply_comparative_penalty":   apply_penalty,
            "total_candidates":            total,
        }

    def _apply_protocol_bonuses(
        self,
        action:    ActionLogEntry,
        patient:   Dict[str, Any],
        h_state:   Dict[str, Any],
        travel_min: float,
        result:    GraderResult,
    ) -> float:
        condition   = patient.get("condition_code", "unknown")
        severity    = patient.get("severity", "P2")
        is_paed     = patient.get("age", 99) < 12

        on_div      = bool(h_state.get("on_diversion", False))
        has_cath    = bool(h_state.get("has_cath_lab", False))
        has_stroke  = bool(h_state.get("is_stroke_centre", False))
        is_l1       = bool(h_state.get("is_level1_trauma", False))
        is_paed_h   = bool(h_state.get("is_paediatric_hospital", False))
        is_burns    = bool(h_state.get("is_burns_unit", False))
        has_blood   = bool(h_state.get("has_blood_bank", False))
        er_occ      = float(h_state.get("er_occupancy_pct", 0.70))

        net = 0.0
        bonuses: List[Tuple[str, float, str]] = []
        penalties_list: List[Tuple[str, float, str]] = []

        if condition in CATH_LAB_CONDITIONS:
            if has_cath:
                bonuses.append(("cath_lab_routing", 0.020, "STEMI → cath-lab hospital"))
            else:
                penalties_list.append((
                    "no_cath_lab", 0.025,
                    "STEMI routed to hospital without cath lab"
                ))

        if condition in STROKE_UNIT_CONDITIONS:
            if has_stroke:
                bonuses.append(("stroke_unit_routing", 0.018, "CVA → stroke-unit hospital"))
            else:
                penalties_list.append((
                    "no_stroke_unit", 0.025,
                    "CVA routed to hospital without stroke unit"
                ))

        if condition in TRAUMA_CENTRE_CONDITIONS:
            if is_l1 and travel_min <= 30.0:
                bonuses.append(("trauma_l1_30min", 0.015,
                                "Major trauma → Level-1 within 30 min"))
            elif is_l1:
                bonuses.append(("trauma_l1_late", 0.005,
                                "Major trauma → Level-1 but > 30 min"))
            else:
                penalties_list.append((
                    "no_trauma_centre", 0.020,
                    "Major trauma NOT routed to Level-1 trauma centre"
                ))

        if condition in BURNS_UNIT_CONDITIONS and is_burns:
            bonuses.append(("burns_unit_routing", 0.012, "Burns → burns unit"))

        if is_paed and is_paed_h:
            bonuses.append(("paed_hospital_routing", 0.015,
                            "Paediatric patient → children's hospital"))
        elif is_paed and not is_paed_h:
            penalties_list.append((
                "non_paed_for_child", 0.015,
                "Paediatric patient routed to non-paediatric hospital"
            ))

        if condition in BLOOD_BANK_CONDITIONS and has_blood:
            bonuses.append(("blood_bank_routing", 0.008,
                            "Haemorrhage patient → hospital with blood bank"))

        if er_occ >= 0.90 and severity == "P1":
            penalties_list.append((
                "high_occ_p1_routing", 0.040,
                f"P1 patient routed to {er_occ:.0%} occupied ER — alternatives exist?"
            ))

        if on_div and severity == "P1":
            penalties_list.append((
                "diverted_p1_routing", 0.030,
                "P1 patient routed to diverted hospital"
            ))

        optimal_travel = float(
            patient.get("optimal_travel_time_min", travel_min + 5.0)
        )
        if travel_min <= optimal_travel + 2.0:
            bonuses.append(("nearest_appropriate", 0.025,
                            "Nearest appropriate hospital chosen"))

        for name, amt, reason in bonuses:
            net += amt
            result.add_note(f"Protocol bonus [{name}]: +{amt:.3f} — {reason}")

        for name, amt, reason in penalties_list:
            net -= amt
            self._add_penalty(result, name, amt, reason, "EMERGI-ENV Protocol v2")
            result.protocol_violations += 1

        capped = ScoringUtils.clamp(net, -PROTOCOL_PENALTY_CAP, PROTOCOL_BONUS_CAP)
        if abs(capped - net) > 1e-6:
            result.add_note(f"Protocol net {net:.3f} capped to {capped:.3f}")

        if capped > 0:
            result.add_component(
                ScoringUtils.build_score_component(
                    "protocol_bonus", capped, 1.0,
                    f"Protocol compliance ({len(bonuses)} bonuses, "
                    f"{len(penalties_list)} violations)"
                )
            )
        elif capped < 0:
            self._add_penalty(
                result, "protocol_net_penalty", abs(capped),
                f"Net protocol penalty ({len(penalties_list)} violations)",
            )

        return capped

    def _extract_patient(self, gi: GraderInput) -> Optional[Dict[str, Any]]:
        summaries = gi.all_patient_summaries()
        if not summaries:
            return None
        p = dict(summaries[0])
        p.setdefault("severity", "P1")
        p.setdefault("condition_code", "unknown")
        p.setdefault("zone_type", "metro_core")
        p.setdefault("required_hospital_specialties", [])
        p.setdefault("age", 45)
        p.setdefault("requires_icu", p.get("severity") == "P1")
        p.setdefault("requires_surgery", p.get("condition_code", "") in SURGICAL_CONDITIONS)
        p.setdefault("requires_blood_bank",
                     p.get("condition_code", "") in BLOOD_BANK_CONDITIONS)
        p.setdefault("optimal_travel_time_min", 12.0)
        p.setdefault("final_survival_prob", 0.85)
        return p

    def _extract_routing_action(
        self, gi: GraderInput
    ) -> Optional[ActionLogEntry]:
        for action_type in ("route", "dispatch", "triage_and_dispatch",
                            "hospital_select", "assign_hospital"):
            for entry in gi.action_log:
                if entry.action_type == action_type:
                    return entry
        return None

    def _extract_hospital_id(
        self, action: ActionLogEntry
    ) -> Optional[str]:
        return (
            action.get("hospital_id")
            or action.get("destination_hospital")
            or action.get("hospital")
            or action.get("dest_hospital_id")
        )

    def _extract_travel_time(
        self,
        action:  ActionLogEntry,
        gi:      GraderInput,
        patient: Dict[str, Any],
    ) -> float:
        v = (
            action.get("travel_time_min")
            or action.get("eta_minutes")
            or action.get("eta")
            or action.get("distance_minutes")
        )
        if v is not None:
            return float(v)

        for obs in reversed(gi.observation_log):
            if "eta_to_hospital_min" in obs:
                return float(obs["eta_to_hospital_min"])

        return float(patient.get("optimal_travel_time_min", 15.0))

    def _resolve_hospital_network(
        self, gi: GraderInput
    ) -> Dict[str, Dict[str, Any]]:
        if gi.final_hospital_state:
            return dict(gi.final_hospital_state)
        snapshot = gi.episode_ledger.get("hospital_snapshot", {})
        if snapshot:
            return dict(snapshot)
        return {}

    def _get_hospital_state(
        self,
        hospital_id:      Optional[str],
        network:          Dict[str, Dict[str, Any]],
        gi:               GraderInput,
        patient:          Dict[str, Any],
    ) -> Dict[str, Any]:
        if hospital_id and hospital_id in network:
            h = dict(network[hospital_id])
        else:
            h = {
                "hospital_id":           hospital_id or "unknown",
                "on_diversion":          patient.get("hospital_on_diversion", False),
                "er_occupancy_pct":      patient.get("hospital_er_occupancy_pct", 0.70),
                "icu_beds_total":        patient.get("icu_beds_total", 20),
                "icu_beds_available":    patient.get("icu_beds_available", 8),
                "ward_beds_total":       150,
                "ward_beds_available":   40,
                "theatre_available":     patient.get("theatre_available", True),
                "has_blood_bank":        patient.get("hospital_has_blood_bank", False),
                "has_cath_lab":          patient.get("hospital_has_cath_lab", False),
                "is_stroke_centre":      patient.get("hospital_is_stroke_centre", False),
                "is_level1_trauma":      patient.get("hospital_is_level1", False),
                "is_paediatric_hospital":patient.get("hospital_is_paediatric", False),
                "is_burns_unit":         patient.get("hospital_is_burns_unit", False),
                "specialties":           patient.get("required_hospital_specialties", []),
                "has_required_specialty":patient.get("hospital_has_specialty", True),
                "estimated_travel_min":  patient.get("optimal_travel_time_min", 15.0),
            }

        h["er_occupancy_pct"]   = float(h.get("er_occupancy_pct", 0.70))
        h["icu_beds_available"] = int(h.get("icu_beds_available", 8))
        return h

    def _apply_zero_scores(self, result: GraderResult) -> None:
        for name, weight in [("specialty_match", W_SPECIALTY),
                              ("capacity_check",  W_CAPACITY),
                              ("travel_time",     W_TRAVEL)]:
            self._add_component(result, name, 0.0, weight, "No routing action taken")
        self._add_penalty(
            result, "no_routing_action", 0.25,
            "Agent issued no routing action — Task 2 is routing-only",
            rule_ref="EMERGI-ENV Rule N2",
        )

GraderRegistry.register(TASK_ID, Task2Grader)
logger.info(
    "Task2Grader registered: task_id=%s baseline=%.2f seed=%d",
    TASK_ID, TASK_BASELINE, TASK_SEED,
)

def _build_hospital_network() -> Dict[str, Dict[str, Any]]:
    return {
        "H01_KEM": {
            "hospital_id": "H01_KEM",
            "name": "KEM Hospital Pune",
            "on_diversion": False,
            "er_occupancy_pct": 0.62,
            "icu_beds_total": 30, "icu_beds_available": 8,
            "ward_beds_total": 200, "ward_beds_available": 55,
            "theatre_available": True,
            "has_blood_bank": True, "has_cath_lab": True,
            "is_stroke_centre": True, "is_level1_trauma": True,
            "is_paediatric_hospital": False, "is_burns_unit": False,
            "specialties": ["cardiology", "cath_lab", "neurology",
                           "stroke_unit", "trauma", "emergency_surgery"],
            "estimated_travel_min": 12.0,
        },
        "H02_RUBY": {
            "hospital_id": "H02_RUBY",
            "name": "Ruby Hall Clinic",
            "on_diversion": False,
            "er_occupancy_pct": 0.78,
            "icu_beds_total": 25, "icu_beds_available": 3,
            "ward_beds_total": 180, "ward_beds_available": 30,
            "theatre_available": True,
            "has_blood_bank": True, "has_cath_lab": True,
            "is_stroke_centre": False, "is_level1_trauma": False,
            "is_paediatric_hospital": False, "is_burns_unit": False,
            "specialties": ["cardiology", "cath_lab", "orthopaedics"],
            "estimated_travel_min": 8.0,
        },
        "H03_SASOON": {
            "hospital_id": "H03_SASOON",
            "name": "Sassoon General Hospital",
            "on_diversion": True,     
            "er_occupancy_pct": 0.96,
            "icu_beds_total": 40, "icu_beds_available": 0,
            "ward_beds_total": 500, "ward_beds_available": 10,
            "theatre_available": False,
            "has_blood_bank": True, "has_cath_lab": False,
            "is_stroke_centre": True, "is_level1_trauma": True,
            "is_paediatric_hospital": False, "is_burns_unit": False,
            "specialties": ["trauma", "general_surgery", "neurology"],
            "estimated_travel_min": 6.0,
        },
        "H04_JEHANGIR": {
            "hospital_id": "H04_JEHANGIR",
            "name": "Jehangir Hospital",
            "on_diversion": False,
            "er_occupancy_pct": 0.55,
            "icu_beds_total": 20, "icu_beds_available": 12,
            "ward_beds_total": 150, "ward_beds_available": 60,
            "theatre_available": True,
            "has_blood_bank": True, "has_cath_lab": False,
            "is_stroke_centre": True, "is_level1_trauma": False,
            "is_paediatric_hospital": False, "is_burns_unit": False,
            "specialties": ["neurology", "stroke_unit", "pulmonology",
                           "obstetrics"],
            "estimated_travel_min": 18.0,
        },
        "H05_RATNA": {
            "hospital_id": "H05_RATNA",
            "name": "Ratna Memorial Children's Hospital",
            "on_diversion": False,
            "er_occupancy_pct": 0.60,
            "icu_beds_total": 15, "icu_beds_available": 6,
            "ward_beds_total": 100, "ward_beds_available": 38,
            "theatre_available": True,
            "has_blood_bank": False, "has_cath_lab": False,
            "is_stroke_centre": False, "is_level1_trauma": False,
            "is_paediatric_hospital": True, "is_burns_unit": False,
            "specialties": ["paediatrics", "nicu", "paediatric_emergency"],
            "estimated_travel_min": 22.0,
        },
        "H06_BURNS": {
            "hospital_id": "H06_BURNS",
            "name": "Pune Burns & Plastics Centre",
            "on_diversion": False,
            "er_occupancy_pct": 0.45,
            "icu_beds_total": 12, "icu_beds_available": 5,
            "ward_beds_total": 80, "ward_beds_available": 30,
            "theatre_available": True,
            "has_blood_bank": False, "has_cath_lab": False,
            "is_stroke_centre": False, "is_level1_trauma": False,
            "is_paediatric_hospital": False, "is_burns_unit": True,
            "specialties": ["burns", "plastic_surgery", "icu"],
            "estimated_travel_min": 28.0,
        },
        "H07_DIST": {
            "hospital_id": "H07_DIST",
            "name": "District General Hospital",
            "on_diversion": False,
            "er_occupancy_pct": 0.88,
            "icu_beds_total": 10, "icu_beds_available": 1,
            "ward_beds_total": 120, "ward_beds_available": 8,
            "theatre_available": True,
            "has_blood_bank": False, "has_cath_lab": False,
            "is_stroke_centre": False, "is_level1_trauma": False,
            "is_paediatric_hospital": False, "is_burns_unit": False,
            "specialties": ["general_medicine", "emergency"],
            "estimated_travel_min": 4.0,
        },
        "H08_PRIV": {
            "hospital_id": "H08_PRIV",
            "name": "City Private Hospital",
            "on_diversion": False,
            "er_occupancy_pct": 0.70,
            "icu_beds_total": 8, "icu_beds_available": 4,
            "ward_beds_total": 60, "ward_beds_available": 18,
            "theatre_available": False,
            "has_blood_bank": False, "has_cath_lab": False,
            "is_stroke_centre": False, "is_level1_trauma": False,
            "is_paediatric_hospital": False, "is_burns_unit": False,
            "specialties": ["general_medicine"],
            "estimated_travel_min": 7.0,
        },
    }

def _build_test_gi(
    hospital_id:     str   = "H01_KEM",
    travel_min:      float = 12.0,
    condition:       str   = "stemi_anterior",
    severity:        str   = "P1",
    specialties:     Optional[List[str]] = None,
    zone_type:       str   = "metro_core",
    age:             int   = 52,
    requires_icu:    bool  = True,
    requires_surg:   bool  = False,
    requires_blood:  bool  = False,
    hotspots:        Optional[List[str]] = None,
    episode_id:      str   = "ep-t2-001",
    network:         Optional[Dict[str, Dict[str, Any]]] = None,
    cath_activated:  bool  = True,
) -> GraderInput:
    if specialties is None:
        specialties = ["cardiology", "cath_lab"]
    network = network or _build_hospital_network()

    return GraderInput(
        task_id      = TASK_ID,
        episode_id   = episode_id,
        seed         = TASK_SEED,
        action_log   = [
            ActionLogEntry(
                step=1,
                action_type="route",
                action_data={
                    "hospital_id":             hospital_id,
                    "travel_time_min":         travel_min,
                    "route_path":              ["Z05", "Z04", hospital_id],
                    "congestion_hotspots":     hotspots or [],
                    "cath_lab_activated":      cath_activated,
                    "stroke_unit_notified":    False,
                    "multi_agency_coordinated":False,
                },
            )
        ],
        episode_ledger={
            "patient_summaries": [
                {
                    "patient_id":                    "P-002",
                    "severity":                      severity,
                    "condition_code":                condition,
                    "zone_id":                       "Z05",
                    "zone_type":                     zone_type,
                    "required_hospital_specialties": specialties,
                    "age":                           age,
                    "requires_icu":                  requires_icu,
                    "requires_surgery":              requires_surg,
                    "requires_blood_bank":           requires_blood,
                    "optimal_travel_time_min":       10.0,
                    "optimal_hospital_id":           "H01_KEM",
                    "final_survival_prob":           0.87,
                    "weight": 1.0, "weighted_reward": 0.87,
                }
            ],
            "hospital_snapshot":   network,
            "optimal_hospital_id": "H01_KEM",
        },
        observation_log    = [],
        episode_steps      = 3,
        total_patients     = 1,
        p1_patients        = 1,
        final_hospital_state = network,
    )

def _run_self_tests() -> None:
    grader   = Task2Grader()
    failures: List[str] = []

    def chk(name: str, cond: bool, msg: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]: {msg}")

    network = _build_hospital_network()

    gi1  = _build_test_gi("H01_KEM", 12.0, "stemi_anterior", network=network)
    r1   = grader.grade(gi1)
    chk("T1_range",    0.0 <= r1.final_score <= 1.0)
    chk("T1_baseline", r1.final_score >= TASK_BASELINE,
        f"score={r1.final_score:.4f} < baseline={TASK_BASELINE}")
    chk("T1_components", len(r1.components) >= 3)

    gi2  = _build_test_gi("H01_KEM", 12.0, "stemi_anterior", network=network)
    r2   = grader.grade(gi2)
    chk("T2_determinism",
        abs(r1.final_score - r2.final_score) < 1e-9,
        f"{r1.final_score} ≠ {r2.final_score}")

    gi3  = _build_test_gi("H03_SASOON", 6.0, "stemi_anterior", network=network)
    r3   = grader.grade(gi3)
    chk("T3_diversion_lower", r3.final_score < r1.final_score,
        f"Diverted {r3.final_score:.4f} should be < optimal {r1.final_score:.4f}")
    chk("T3_range", 0.0 <= r3.final_score <= 1.0)

    gi4  = _build_test_gi("H07_DIST", 4.0, "stemi_anterior", network=network)
    r4   = grader.grade(gi4)
    chk("T4_no_specialty", r4.final_score < r1.final_score)
    chk("T4_range", 0.0 <= r4.final_score <= 1.0)

    gi5  = _build_test_gi(
        "H04_JEHANGIR", 18.0, "ischaemic_stroke",
        specialties=["neurology", "stroke_unit"],
        network=network,
    )
    r5 = grader.grade(gi5)
    chk("T5_stroke_range", 0.0 <= r5.final_score <= 1.0)

    gi6  = _build_test_gi(
        "H05_RATNA", 22.0, "febrile_seizure",
        specialties=["paediatrics", "paediatric_emergency"],
        age=5, zone_type="metro_suburban",
        network=network,
    )
    r6 = grader.grade(gi6)
    chk("T6_paed_range", 0.0 <= r6.final_score <= 1.0)

    gi7  = _build_test_gi(
        "H06_BURNS", 28.0, "major_burns_40pct",
        specialties=["burns", "icu"],
        network=network,
    )
    r7 = grader.grade(gi7)
    chk("T7_burns_range", 0.0 <= r7.final_score <= 1.0)

    gi8  = GraderInput(
        task_id=TASK_ID, episode_id="ep-noact", seed=TASK_SEED,
        action_log=[], episode_ledger={"patient_summaries": [{
            "patient_id": "P-x", "severity": "P1",
            "condition_code": "stemi_anterior",
            "required_hospital_specialties": ["cardiology"],
            "zone_type": "metro_core", "weight": 1.0
        }]},
        observation_log=[], episode_steps=0,
        total_patients=1, p1_patients=1,
    )
    r8 = grader.grade(gi8)
    chk("T8_no_action", r8.final_score == 0.0,
        f"No-action should score 0.0, got {r8.final_score}")
    chk("T8_status", r8.status == GraderStatus.PARTIAL)

    gi9a = _build_test_gi("H01_KEM", 8.0,  network=network)
    gi9b = _build_test_gi("H01_KEM", 85.0, network=network)
    r9a, r9b = grader.grade(gi9a), grader.grade(gi9b)
    chk("T9_travel_sensitivity",
        r9a.final_score > r9b.final_score,
        f"Fast={r9a.final_score:.4f} should > Slow={r9b.final_score:.4f}")

    gi10a = _build_test_gi("H01_KEM", 12.0, hotspots=[],               network=network)
    gi10b = _build_test_gi("H01_KEM", 12.0, hotspots=["Z05","Z04","Z08"], network=network)
    r10a, r10b = grader.grade(gi10a), grader.grade(gi10b)
    chk("T10_hotspot_penalty",
        r10a.final_score >= r10b.final_score,
        f"No hotspot={r10a.final_score:.4f} should ≥ hotspot={r10b.final_score:.4f}")

    chk("T11_comparative", "hospitals_evaluated" in r1.extra,
        "comparative analysis not in extras")
    chk("T11_ranking", r1.extra.get("agent_rank", 99) <= 3,
        f"KEM should rank in top-3, got rank {r1.extra.get('agent_rank')}")

    d = r1.as_dict()
    chk("T12_dict", all(k in d for k in ("final_score","components","penalties")))
    chk("T12_json", len(r1.as_json()) > 200)

    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task2Grader self-test: {len(failures)} failure(s):\n" +
            "\n".join(failures)
        )
    logger.info("Task2Grader self-test PASSED (12 test cases).")

try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task2Grader self-test FAILED at import: %s", _e)
    raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 72)
    print("EMERGI-ENV  ·  Task2Grader  ·  Hospital Route Selection Demo")
    print("=" * 72)

    grader  = Task2Grader()
    network = _build_hospital_network()

    scenarios = [
        ("STEMI → KEM (cath lab, low occ, 12 min)",
         _build_test_gi("H01_KEM",    12.0, "stemi_anterior",  network=network)),
        ("STEMI → Ruby (cath lab, 78% occ, 8 min)",
         _build_test_gi("H02_RUBY",   8.0,  "stemi_anterior",  network=network)),
        ("STEMI → Sassoon (DIVERTED)",
         _build_test_gi("H03_SASOON", 6.0,  "stemi_anterior",  network=network)),
        ("STEMI → District General (no cath lab)",
         _build_test_gi("H07_DIST",   4.0,  "stemi_anterior",  network=network)),
        ("Stroke → Jehangir (stroke unit, 18 min)",
         _build_test_gi("H04_JEHANGIR", 18.0, "ischaemic_stroke",
                        specialties=["neurology","stroke_unit"], network=network)),
        ("Burns → Burns Centre (specialist, 28 min)",
         _build_test_gi("H06_BURNS", 28.0, "major_burns_40pct",
                        specialties=["burns","icu"], network=network)),
        ("Paed seizure → Children's Hospital (22 min)",
         _build_test_gi("H05_RATNA", 22.0, "febrile_seizure",
                        specialties=["paediatrics"], age=5, network=network)),
        ("P1 STEMI → District (nearest, wrong specialty, 4 min)",
         _build_test_gi("H07_DIST",   4.0, "stemi_anterior",
                        hotspots=[], network=network)),
        ("Golden-hour breach: KEM but 85 min travel",
         _build_test_gi("H01_KEM",  85.0, "stemi_anterior", network=network)),
        ("3 congestion hotspots on route",
         _build_test_gi("H01_KEM",  14.0, "stemi_anterior",
                        hotspots=["Z05","Z08","Z04"], network=network)),
    ]

    results_all = []
    for name, gi in scenarios:
        res = grader.grade(gi)
        results_all.append(res)
        beat = "✓" if res.beats_baseline else "✗"
        rank = res.extra.get("agent_rank", "?")
        total_h = res.extra.get("hospitals_evaluated", "?")

        print(f"\n  [{beat}] {name}")
        print(f"       Score={res.final_score:.4f}  base={TASK_BASELINE:.2f}  "
              f"Δ={res.score_delta_vs_baseline:+.4f}  "
              f"rank={rank}/{total_h}")

        for c in res.components:
            if c.name in ("protocol_bonus",):
                continue
            print(f"         {c.name:<18} {c.raw_score:.4f} × {c.weight:.2f} "
                  f"= {c.weighted:.4f}  | {c.notes[:60]}")

        if res.penalties:
            for p in res.penalties[:2]:
                print(f"         PENALTY: {p.name:<28} {p.amount:+.4f}  {p.reason[:50]}")

    print("\n" + "=" * 72)
    beats = sum(1 for r in results_all if r.beats_baseline)
    print(f"  {beats}/{len(results_all)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 72)
    print("\n✅  Task2Grader demo complete.")