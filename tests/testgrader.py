from __future__ import annotations
import copy
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
import pytest
from server.graders.basegrader import (
    BASE_GRADER_VERSION,
    GRADER_TIMEOUT_SECONDS,
    MAX_PENALTY_FRACTION,
    SCORE_CEILING,
    SCORE_FLOOR,
    TASK_BASELINES,
    TASK_SEEDS,
    ActionLogEntry,
    BaseGrader,
    GraderInput,
    GraderPipeline,
    GraderRegistry,
    GraderResult,
    GraderStatus,
    PenaltyRecord,
    ProtocolRuleChecker,
    ScoreComponent,
    ScoringUtils,
)
import server.graders.taskgrader1  
import server.graders.taskgrader2  
import server.graders.taskgrader3  
import server.graders.taskgrader4  
import server.graders.taskgrader5  
import server.graders.taskgrader6  
import server.graders.taskgrader7  
import server.graders.taskgrader8  
import server.graders.taskgrader9  
from server.env import (
    DIFFICULTY_MAP,
    MAX_STEPS_BY_TASK,
    TASK_IDS,
    EmergiEnv,
    make_env,
)
ALL_TASK_IDS: List[str] = sorted(TASK_IDS)
_BASELINE_NOOP_MAX_OVERSHOOT: float = 0.15  
_FLOAT_TOL: float = 1e-9
def _make_action_log(entries: List[Dict[str, Any]]) -> List[ActionLogEntry]:
    return [
        ActionLogEntry(step=i + 1, action_type=e["action_type"], action_data=e)
        for i, e in enumerate(entries)
    ]
def _make_patient_summary(
    *,
    patient_id: str = "P001",
    severity: str = "P1",
    phase: str = "treated",
    dispatch_latency_min: float = 6.0,
    survival_prob: float = 0.85,
    optimal_survival_prob: float = 0.95,
    assigned_priority: str = "P1",
    required_unit_type: str = "MICU",
    assigned_unit: str = "MICU-01",
    assigned_hospital: str = "H01",
    required_hospital_specialties: Optional[List[str]] = None,
    hospital_has_specialty: bool = True,
    hospital_on_diversion: bool = False,
    hospital_er_occupancy_pct: float = 0.65,
    hospital_is_level1: bool = True,
    start_tag: Optional[str] = None,
    ground_truth_start_tag: Optional[str] = None,
    transfer_required: bool = False,
    transfer_initiated: bool = False,
    specialist_hospital_id: Optional[str] = None,
    transfer_window_min: Optional[float] = 60.0,
    transfer_latency_min: Optional[float] = None,
    destination_icu_utilisation: Optional[float] = 0.60,
    requires_extrication: bool = False,
    surge_zone: Optional[str] = None,
    mutual_aid_required: bool = False,
    mci_type: Optional[str] = None,
    weight: float = 3.0,
    weighted_reward: float = 0.72,
    response_time_target_min: float = 8.0,
    optimal_travel_time_min: float = 6.0,
    zone_id: str = "zone_1",
    condition_code: str = "stemi_anterior",
    golden_hour_breach: bool = False,
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "severity": severity,
        "condition_code": condition_code,
        "condition_family": "cardiac",
        "zone_id": zone_id,
        "zone_type": "metro_core",
        "assigned_priority": assigned_priority,
        "required_unit_type": required_unit_type,
        "assigned_unit": assigned_unit,
        "assigned_hospital": assigned_hospital,
        "required_hospital_specialties": required_hospital_specialties or ["cardiology"],
        "dispatch_step": 1,
        "dispatch_latency_min": dispatch_latency_min,
        "arrival_step": 3,
        "treatment_step": 4,
        "phase_at_episode_end": phase,
        "treatment_started": phase == "treated",
        "final_survival_prob": survival_prob,
        "optimal_survival_prob": optimal_survival_prob,
        "golden_hour_breach": golden_hour_breach,
        "weight": weight,
        "weighted_reward": weighted_reward,
        "hospital_on_diversion": hospital_on_diversion,
        "hospital_er_occupancy_pct": hospital_er_occupancy_pct,
        "hospital_has_specialty": hospital_has_specialty,
        "hospital_is_level1": hospital_is_level1,
        "response_time_target_min": response_time_target_min,
        "optimal_travel_time_min": optimal_travel_time_min,
        "start_tag": start_tag,
        "ground_truth_start_tag": ground_truth_start_tag,
        "rpm_score": 2,
        "transfer_required": transfer_required,
        "transfer_initiated": transfer_initiated,
        "specialist_hospital_id": specialist_hospital_id,
        "transfer_window_min": transfer_window_min,
        "transfer_latency_min": transfer_latency_min,
        "destination_icu_utilisation": destination_icu_utilisation,
        "requires_extrication": requires_extrication,
        "requires_police": False,
        "requires_fire": False,
        "surge_zone": surge_zone,
        "mutual_aid_required": mutual_aid_required,
        "mci_type": mci_type,
    }
def _minimal_ledger(
    patient_summaries: Optional[List[Dict]] = None,
    cascade_events: Optional[List] = None,
    mutual_aid_requests: int = 0,
    surge_declared: bool = False,
    protocol_violations: int = 0,
    protocol_bonus_accrued: float = 0.0,
    comm_failures: int = 0,
) -> Dict[str, Any]:
    return {
        "patient_summaries": patient_summaries or [],
        "cascade_events": cascade_events or [],
        "mutual_aid_requests": mutual_aid_requests,
        "surge_declared": surge_declared,
        "protocol_violations": protocol_violations,
        "protocol_bonus_accrued": protocol_bonus_accrued,
        "comm_failures": comm_failures,
        "step_rewards": [],
        "fleet_summary": {},
        "hospital_summary": {},
        "protocol_violations_list": [],
    }
def _build_grader_input(
    task_id: str,
    patient_summaries: Optional[List[Dict]] = None,
    action_log: Optional[List[ActionLogEntry]] = None,
    episode_steps: int = 10,
    total_patients: int = 1,
    p1_patients: int = 1,
    seed: Optional[int] = None,
    **ledger_kwargs,
) -> GraderInput:
    if seed is None:
        seed = TASK_SEEDS.get(task_id, 42)
    ledger = _minimal_ledger(patient_summaries=patient_summaries, **ledger_kwargs)
    return GraderInput(
        task_id=task_id,
        episode_id=str(uuid.uuid4()),
        seed=seed,
        action_log=action_log or _make_action_log([{"action_type": "noop"}]),
        episode_ledger=ledger,
        observation_log=[],
        episode_steps=episode_steps,
        total_patients=total_patients,
        p1_patients=p1_patients,
    )
def _run_noop_episode(task_id: str) -> GraderInput:
    env = make_env()
    env.reset(task_id=task_id, seed=TASK_SEEDS.get(task_id, 42))
    max_s = MAX_STEPS_BY_TASK.get(task_id, 50)
    for _ in range(max_s):
        if env.is_done:
            break
        env.step({"action_type": "noop", "reason": "test_noop"})
    return env.to_grader_input()
class TestGraderRegistry:
    def test_all_nine_tasks_registered(self):
        registered = set(GraderRegistry.all_task_ids())
        assert registered == TASK_IDS, (
            f"Registry missing tasks: {TASK_IDS - registered}. "
            f"Extra tasks: {registered - TASK_IDS}"
        )
    def test_all_task_ids_in_baselines(self):
        for tid in TASK_IDS:
            assert tid in TASK_BASELINES, f"{tid} not in TASK_BASELINES"
            assert 0.0 <= TASK_BASELINES[tid] <= 1.0
    def test_all_task_ids_in_seeds(self):
        for tid in TASK_IDS:
            assert tid in TASK_SEEDS, f"{tid} not in TASK_SEEDS"
            assert TASK_SEEDS[tid] >= 0
    def test_seeds_are_unique(self):
        seeds = list(TASK_SEEDS.values())
        assert len(seeds) == len(set(seeds)), "Duplicate task seeds detected"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_is_registered(self, task_id: str):
        assert GraderRegistry.is_registered(task_id)
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_get_returns_class(self, task_id: str):
        cls = GraderRegistry.get(task_id)
        assert issubclass(cls, BaseGrader)
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_get_instance_returns_grader(self, task_id: str):
        grader = GraderRegistry.get_instance(task_id)
        assert isinstance(grader, BaseGrader)
    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="no grader for task_id"):
            GraderRegistry.get("task_nonexistent_xyz")
    def test_all_task_ids_sorted(self):
        ids = GraderRegistry.all_task_ids()
        assert ids == sorted(ids), "GraderRegistry.all_task_ids() should return sorted list"
    def test_grader_task_id_matches_registry_key(self):
        for tid in ALL_TASK_IDS:
            grader = GraderRegistry.get_instance(tid)
            assert grader.TASK_ID == tid, (
                f"Grader class TASK_ID='{grader.TASK_ID}' != registry key '{tid}'"
            )
    def test_grader_baseline_matches_constants(self):
        for tid in ALL_TASK_IDS:
            grader = GraderRegistry.get_instance(tid)
            expected = TASK_BASELINES[tid]
            assert abs(grader.TASK_BASELINE - expected) < _FLOAT_TOL, (
                f"{tid}: TASK_BASELINE={grader.TASK_BASELINE} != expected {expected}"
            )
    def test_grader_seed_matches_constants(self):
        for tid in ALL_TASK_IDS:
            grader = GraderRegistry.get_instance(tid)
            expected = TASK_SEEDS[tid]
            assert grader.TASK_SEED == expected, (
                f"{tid}: TASK_SEED={grader.TASK_SEED} != expected {expected}"
            )
    def test_grader_difficulty_is_valid(self):
        valid = {"easy", "medium", "hard"}
        for tid in ALL_TASK_IDS:
            grader = GraderRegistry.get_instance(tid)
            assert grader.TASK_DIFFICULTY in valid, (
                f"{tid}: TASK_DIFFICULTY='{grader.TASK_DIFFICULTY}' not in {valid}"
            )
    def test_component_weights_sum_to_one(self):
        for tid in ALL_TASK_IDS:
            grader = GraderRegistry.get_instance(tid)
            if grader.COMPONENT_WEIGHTS:
                total = sum(grader.COMPONENT_WEIGHTS.values())
                assert abs(total - 1.0) < 1e-4, (
                    f"{tid}: COMPONENT_WEIGHTS sum={total:.6f} (expected 1.0)"
                )
class TestScoringUtils:
    def test_clamp_below_floor(self):
        assert ScoringUtils.clamp(-100.0) == 0.0
    def test_clamp_above_ceiling(self):
        assert ScoringUtils.clamp(100.0) == 1.0
    def test_clamp_in_range(self):
        assert ScoringUtils.clamp(0.72) == pytest.approx(0.72)
    def test_clamp_at_boundaries(self):
        assert ScoringUtils.clamp(0.0) == 0.0
        assert ScoringUtils.clamp(1.0) == 1.0
    def test_clamp_custom_bounds(self):
        assert ScoringUtils.clamp(0.5, 0.2, 0.8) == pytest.approx(0.5)
        assert ScoringUtils.clamp(0.1, 0.2, 0.8) == pytest.approx(0.2)
        assert ScoringUtils.clamp(0.9, 0.2, 0.8) == pytest.approx(0.8)
    def test_response_time_perfect(self):
        assert ScoringUtils.response_time_score(5.0, 8.0) == 1.0
    def test_response_time_at_target(self):
        assert ScoringUtils.response_time_score(8.0, 8.0) == 1.0
    def test_response_time_worst_case(self):
        assert ScoringUtils.response_time_score(120.0, 8.0, 120.0) == 0.0
    def test_response_time_between(self):
        score = ScoringUtils.response_time_score(64.0, 8.0, 120.0)
        assert 0.0 < score < 1.0
    def test_response_time_monotone_decreasing(self):
        times = [0, 5, 8, 12, 20, 40, 80, 120]
        scores = [ScoringUtils.response_time_score(t, 8.0, 120.0) for t in times]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Non-monotone at t={times[i]}→{times[i+1]}: {scores[i]:.4f}→{scores[i+1]:.4f}"
            )
    def test_response_time_exponential_shape(self):
        s_linear = ScoringUtils.response_time_score(40.0, 8.0, 120.0, shape="linear")
        s_exp    = ScoringUtils.response_time_score(40.0, 8.0, 120.0, shape="exponential")
        assert 0.0 < s_linear <= 1.0
        assert 0.0 < s_exp    <= 1.0
    def test_response_time_step_shape(self):
        s = ScoringUtils.response_time_score(65.0, 8.0, 120.0, shape="step")
        assert s in (0.0, 0.5)
    def test_unit_exact_match(self):
        assert ScoringUtils.unit_type_score("MICU", ("MICU",)) == 1.0
    def test_unit_over_provision(self):
        s = ScoringUtils.unit_type_score("MICU", ("BLS",))
        assert 0.80 <= s <= 0.90, f"Expected ~0.85, got {s}"
    def test_unit_one_gap(self):
        s = ScoringUtils.unit_type_score("ALS", ("MICU",))
        assert 0.50 <= s <= 0.60
    def test_unit_two_gap(self):
        s = ScoringUtils.unit_type_score("BLS", ("MICU",))
        assert s == pytest.approx(0.20)
    def test_unit_unknown_type(self):
        assert ScoringUtils.unit_type_score("HELICOPTER", ("MICU",)) == 0.0
    def test_triage_exact_immediate(self):
        assert ScoringUtils.triage_accuracy_score("Immediate", "Immediate") == 1.0
    def test_triage_exact_delayed(self):
        assert ScoringUtils.triage_accuracy_score("Delayed", "Delayed") == 1.0
    def test_triage_immediate_as_expectant_zero(self):
        assert ScoringUtils.triage_accuracy_score("Expectant", "Immediate") == 0.0
    def test_triage_one_step_off(self):
        s = ScoringUtils.triage_accuracy_score("Delayed", "Immediate")
        assert 0.40 <= s <= 0.60
    def test_triage_two_steps_off(self):
        s = ScoringUtils.triage_accuracy_score("Minimal", "Immediate")
        assert 0.10 <= s <= 0.30
    def test_hospital_specialty_match_low_occupancy(self):
        assert ScoringUtils.hospital_specialty_score(True, False, 0.60) == 1.0
    def test_hospital_on_diversion(self):
        assert ScoringUtils.hospital_specialty_score(True, True, 0.60) == 0.0
    def test_hospital_no_specialty(self):
        s = ScoringUtils.hospital_specialty_score(False, False, 0.60)
        assert s == pytest.approx(0.20)
    def test_hospital_high_occupancy_penalty(self):
        s_lo = ScoringUtils.hospital_specialty_score(True, False, 0.60)
        s_hi = ScoringUtils.hospital_specialty_score(True, False, 0.92)
        assert s_lo > s_hi
    def test_survival_at_optimal(self):
        assert ScoringUtils.survival_probability_score(0.90, 0.90) == 1.0
    def test_survival_at_worst(self):
        s = ScoringUtils.survival_probability_score(0.05, 0.90)
        assert s == pytest.approx(0.0)
    def test_survival_monotone(self):
        p_opts = [0.95] * 5
        p_acts = [0.95, 0.70, 0.50, 0.30, 0.10]
        scores = [ScoringUtils.survival_probability_score(a, o) for a, o in zip(p_acts, p_opts)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
    def test_weighted_sum_perfect(self):
        scores = {"a": 1.0, "b": 1.0, "c": 1.0}
        weights = {"a": 0.4, "b": 0.3, "c": 0.3}
        assert ScoringUtils.weighted_sum(scores, weights) == pytest.approx(1.0)
    def test_weighted_sum_zero(self):
        scores = {"a": 0.0, "b": 0.0}
        weights = {"a": 0.5, "b": 0.5}
        assert ScoringUtils.weighted_sum(scores, weights) == pytest.approx(0.0)
    def test_weighted_sum_normalises(self):
        scores = {"a": 1.0, "b": 1.0}
        weights = {"a": 2.0, "b": 2.0}  
        result = ScoringUtils.weighted_sum(scores, weights)
        assert result == pytest.approx(1.0)
    def test_weighted_sum_clamped(self):
        scores = {"a": 1.5, "b": 1.5}
        weights = {"a": 0.5, "b": 0.5}
        result = ScoringUtils.weighted_sum(scores, weights)
        assert 0.0 <= result <= 1.0
    def test_validate_weights_ok(self):
        assert ScoringUtils.validate_weights({"a": 0.4, "b": 0.3, "c": 0.3})
    def test_validate_weights_fail(self):
        assert not ScoringUtils.validate_weights({"a": 0.4, "b": 0.3})  
    def test_validate_weights_tolerance(self):
        assert ScoringUtils.validate_weights({"a": 0.4, "b": 0.3, "c": 0.3 + 1e-5})
    def test_normalise_weights_output_sums_to_one(self):
        result = ScoringUtils.normalise_weights({"a": 2.0, "b": 3.0, "c": 5.0})
        assert abs(sum(result.values()) - 1.0) < 1e-9
    def test_normalise_weights_zero_total(self):
        result = ScoringUtils.normalise_weights({"a": 0.0, "b": 0.0})
        assert abs(sum(result.values()) - 1.0) < 1e-9
    def test_linear_decay_at_best(self):
        assert ScoringUtils.linear_decay_score(0.0, 0.0, 120.0) == 1.0
    def test_linear_decay_at_worst(self):
        assert ScoringUtils.linear_decay_score(120.0, 0.0, 120.0) == 0.0
    def test_linear_decay_midpoint(self):
        s = ScoringUtils.linear_decay_score(60.0, 0.0, 120.0)
        assert s == pytest.approx(0.5)
    def test_exp_decay_zero_elapsed(self):
        assert ScoringUtils.exponential_decay_score(0.0) == 1.0
    def test_exp_decay_positive(self):
        s = ScoringUtils.exponential_decay_score(30.0)
        assert 0.0 < s < 1.0
    def test_exp_decay_monotone(self):
        times = [0, 10, 20, 40, 80, 120]
        scores = [ScoringUtils.exponential_decay_score(t) for t in times]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
    def test_boolean_true(self):
        assert ScoringUtils.boolean_score(True) == 1.0
    def test_boolean_false(self):
        assert ScoringUtils.boolean_score(False) == 0.0
    def test_boolean_custom(self):
        assert ScoringUtils.boolean_score(True, if_true=0.8, if_false=0.2) == pytest.approx(0.8)
    def test_count_score_exact(self):
        assert ScoringUtils.count_score(5, 5) == 1.0
    def test_count_score_miss(self):
        s = ScoringUtils.count_score(3, 5)
        assert 0.0 <= s < 1.0
    def test_count_score_zero_target(self):
        assert ScoringUtils.count_score(10, 0) == 1.0
    def test_build_score_component(self):
        comp = ScoringUtils.build_score_component("triage", 0.8, 0.4, "test")
        assert isinstance(comp, ScoreComponent)
        assert comp.raw_score == pytest.approx(0.8)
        assert comp.weight == pytest.approx(0.4)
        assert comp.weighted == pytest.approx(0.32)
        assert comp.notes == "test"
    def test_build_score_component_clamped(self):
        comp = ScoringUtils.build_score_component("x", 1.5, 0.5)
        assert comp.raw_score == pytest.approx(1.0)
    def test_build_penalty_negative(self):
        p = ScoringUtils.build_penalty("test", 0.30, "reason", "rule1")
        assert p.amount < 0.0
        assert p.amount == pytest.approx(-0.30)
    def test_build_penalty_already_negative(self):
        p = ScoringUtils.build_penalty("test", -0.15, "reason")
        assert p.amount == pytest.approx(-0.15)
    def test_input_hash_deterministic(self):
        data = {"task_id": "task1", "seed": 42}
        h1 = ScoringUtils.compute_input_hash(data)
        h2 = ScoringUtils.compute_input_hash(data)
        assert h1 == h2
    def test_input_hash_different_inputs(self):
        h1 = ScoringUtils.compute_input_hash({"a": 1})
        h2 = ScoringUtils.compute_input_hash({"a": 2})
        assert h1 != h2
    def test_input_hash_length(self):
        h = ScoringUtils.compute_input_hash({"x": "y"})
        assert len(h) == 16
class TestProtocolRuleChecker:
    def test_micu_stemi_compliant(self):
        ok, delta, _ = ProtocolRuleChecker.check_micu_for_stemi("MICU", "stemi_anterior")
        assert ok is True
        assert delta >= 0.0
    def test_micu_stemi_violation(self):
        ok, delta, _ = ProtocolRuleChecker.check_micu_for_stemi("BLS", "stemi_anterior")
        assert ok is False
        assert delta < 0.0
    def test_micu_stemi_na(self):
        _, delta, reason = ProtocolRuleChecker.check_micu_for_stemi("BLS", "general_illness")
        assert reason == "n/a"
    def test_micu_stemi_all_variants(self):
        for cond in [
            "stemi_anterior", "stemi_inferior", "stemi_posterior",
            "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg",
        ]:
            ok, delta, _ = ProtocolRuleChecker.check_micu_for_stemi("ALS", cond)
            assert ok is False and delta < 0.0, f"Expected violation for {cond}"
    def test_als_stroke_als_compliant(self):
        ok, delta, _ = ProtocolRuleChecker.check_als_for_stroke("ALS", "ischemic_stroke")
        assert ok is True
    def test_als_stroke_micu_also_compliant(self):
        ok, _, _ = ProtocolRuleChecker.check_als_for_stroke("MICU", "ischemic_stroke")
        assert ok is True
    def test_als_stroke_bls_violation(self):
        ok, delta, _ = ProtocolRuleChecker.check_als_for_stroke("BLS", "hemorrhagic_stroke_sah")
        assert ok is False and delta < 0.0
    def test_als_stroke_na(self):
        _, _, reason = ProtocolRuleChecker.check_als_for_stroke("BLS", "burns_minor")
        assert reason == "n/a"
    def test_no_diversion_not_diverted(self):
        ok, delta, _ = ProtocolRuleChecker.check_no_routing_to_diverted(False, "H01")
        assert ok is True
    def test_no_diversion_diverted(self):
        ok, delta, _ = ProtocolRuleChecker.check_no_routing_to_diverted(True, "H01")
        assert ok is False and delta < 0.0
    def test_cath_lab_activated_bonus(self):
        ok, delta, _ = ProtocolRuleChecker.check_cath_lab_activation(True, "stemi_anterior")
        assert ok is True and delta > 0.0
    def test_cath_lab_not_activated_penalty(self):
        ok, delta, _ = ProtocolRuleChecker.check_cath_lab_activation(False, "stemi_anterior")
        assert ok is False and delta < 0.0
    def test_cath_lab_na_non_stemi(self):
        _, _, reason = ProtocolRuleChecker.check_cath_lab_activation(True, "ischemic_stroke")
        assert reason == "n/a"
    def test_stroke_notified_bonus(self):
        ok, delta, _ = ProtocolRuleChecker.check_stroke_unit_prenotification(True, "ischemic_stroke")
        assert ok is True and delta > 0.0
    def test_stroke_not_notified_penalty(self):
        ok, delta, _ = ProtocolRuleChecker.check_stroke_unit_prenotification(False, "ischemic_stroke")
        assert ok is False and delta < 0.0
    def test_multi_agency_coordinated(self):
        ok, delta, _ = ProtocolRuleChecker.check_multi_agency_for_trapped(
            True, True, "polytrauma_blunt"
        )
        assert ok is True
    def test_multi_agency_missing_penalty(self):
        ok, delta, _ = ProtocolRuleChecker.check_multi_agency_for_trapped(
            False, True, "polytrauma_blunt"
        )
        assert ok is False and delta < 0.0
    def test_multi_agency_not_trapped_na(self):
        _, _, reason = ProtocolRuleChecker.check_multi_agency_for_trapped(
            False, False, "polytrauma_blunt"
        )
        assert reason == "n/a"
    def test_level1_trauma_within_30(self):
        ok, delta, _ = ProtocolRuleChecker.check_level1_trauma_routing(
            True, 20.0, "polytrauma_blunt"
        )
        assert ok is True and delta > 0.0
    def test_level1_trauma_not_level1(self):
        ok, delta, _ = ProtocolRuleChecker.check_level1_trauma_routing(
            False, 20.0, "polytrauma_blunt"
        )
        assert ok is False and delta < 0.0
    def test_level1_trauma_na(self):
        _, _, reason = ProtocolRuleChecker.check_level1_trauma_routing(
            False, 20.0, "general_illness"
        )
        assert reason == "n/a"
    def test_start_protocol_applied_bonus(self):
        ok, delta, _ = ProtocolRuleChecker.check_start_triage_in_mci(True, True)
        assert ok is True and delta > 0.0
    def test_start_protocol_not_applied_penalty(self):
        ok, delta, _ = ProtocolRuleChecker.check_start_triage_in_mci(False, True)
        assert ok is False and delta < 0.0
    def test_start_protocol_not_mci_na(self):
        _, _, reason = ProtocolRuleChecker.check_start_triage_in_mci(False, False)
        assert reason == "n/a"
    def test_mutual_aid_surge_declared_and_requested(self):
        ok, delta, _ = ProtocolRuleChecker.check_mutual_aid_in_surge(True, True)
        assert ok is True
    def test_mutual_aid_surge_declared_not_requested(self):
        ok, delta, _ = ProtocolRuleChecker.check_mutual_aid_in_surge(False, True)
        assert ok is False and delta < 0.0
    def test_mutual_aid_no_surge_na(self):
        _, _, reason = ProtocolRuleChecker.check_mutual_aid_in_surge(False, False)
        assert reason == "n/a"
    def test_run_all_checks_returns_float(self):
        net, penalties, notes = ProtocolRuleChecker.run_all_checks(
            condition_key="stemi_anterior",
            dispatched_unit="MICU",
            hospital_on_diversion=False,
            hospital_id="H01",
            hospital_is_level1=True,
            response_time_min=8.0,
            cath_lab_activated=True,
            stroke_unit_notified=False,
            multi_agency_coordinated=False,
            trapped_victim=False,
            is_mci=False,
            start_protocol_applied=False,
            mutual_aid_requested=False,
            surge_declared=False,
        )
        assert isinstance(net, float)
        assert isinstance(penalties, list)
        assert isinstance(notes, list)
class TestScoreComponent:
    def test_construction_auto_weighted(self):
        comp = ScoreComponent(name="triage", raw_score=0.8, weight=0.4, weighted=0.32)
        assert comp.weighted == pytest.approx(0.32)
    def test_as_dict_keys(self):
        comp = ScoreComponent(name="hospital", raw_score=0.7, weight=0.3, weighted=0.21, notes="ok")
        d = comp.as_dict()
        assert set(d.keys()) == {"name", "raw_score", "weight", "weighted", "max_possible", "notes"}
    def test_as_dict_rounding(self):
        comp = ScoreComponent(name="x", raw_score=0.333333, weight=0.333333, weighted=0.111111)
        d = comp.as_dict()
        assert len(str(d["raw_score"]).split(".")[-1]) <= 5
    def test_immutability(self):
        comp = ScoreComponent(name="test", raw_score=0.5, weight=0.5, weighted=0.25)
        with pytest.raises((AttributeError, TypeError)):
            comp.raw_score = 0.9  
class TestPenaltyRecord:
    def test_penalty_amount_negative(self):
        p = ScoringUtils.build_penalty("div", 0.30, "diverted hospital")
        assert p.amount < 0.0
    def test_as_dict_schema(self):
        p = PenaltyRecord(name="test", amount=-0.10, reason="test reason", rule_ref="rule1")
        d = p.as_dict()
        assert "name" in d and "amount" in d and "reason" in d and "rule_ref" in d
        assert d["amount"] < 0.0
    def test_zero_penalty(self):
        p = PenaltyRecord(name="ok", amount=0.0, reason="compliant")
        d = p.as_dict()
        assert d["amount"] == 0.0
class TestGraderResult:
    def _make_result(self) -> GraderResult:
        r = GraderResult(
            task_id="task1_single_triage",
            episode_id="ep-001",
            seed=42,
            baseline=0.61,
            task_difficulty="easy",
        )
        r.add_component(ScoringUtils.build_score_component("triage", 0.8, 0.4))
        r.add_component(ScoringUtils.build_score_component("unit_type", 1.0, 0.3))
        r.add_component(ScoringUtils.build_score_component("hospital", 0.9, 0.3))
        return r
    def test_finalise_score_in_range(self):
        r = self._make_result()
        r.finalise()
        assert 0.0 <= r.final_score <= 1.0
    def test_finalise_beats_baseline(self):
        r = self._make_result()
        r.finalise()
        assert r.beats_baseline is True
    def test_finalise_with_penalty_reduces_score(self):
        r = self._make_result()
        r.add_penalty(ScoringUtils.build_penalty("div", 0.30, "diverted"))
        r.finalise()
        r2 = self._make_result()
        r2.finalise()
        assert r.final_score < r2.final_score
    def test_finalise_clamps_to_floor(self):
        r = GraderResult(task_id="task1_single_triage", episode_id="x", seed=42, baseline=0.61)
        r.add_penalty(ScoringUtils.build_penalty("huge", 5.0, "massive penalty"))
        r.finalise()
        assert r.final_score == SCORE_FLOOR
    def test_finalise_clamps_to_ceiling(self):
        r = GraderResult(task_id="task1_single_triage", episode_id="x", seed=42, baseline=0.61)
        for _ in range(10):
            r.add_component(ScoringUtils.build_score_component("x", 1.0, 0.1))
        r.finalise()
        assert r.final_score <= SCORE_CEILING
    def test_as_dict_schema(self):
        r = self._make_result()
        r.finalise()
        d = r.as_dict()
        required_keys = {
            "grader_id", "task_id", "episode_id", "seed", "final_score",
            "raw_score", "baseline", "beats_baseline", "delta_vs_baseline",
            "status", "grading_time_ms", "task_difficulty", "components",
            "penalties", "notes", "input_hash",
        }
        assert required_keys.issubset(set(d.keys()))
    def test_as_json_valid(self):
        r = self._make_result()
        r.finalise()
        j = r.as_json()
        parsed = json.loads(j)
        assert parsed["task_id"] == "task1_single_triage"
    def test_summary_line_contains_task_id(self):
        r = self._make_result()
        r.finalise()
        line = r.summary_line()
        assert "task1_single_triage" in line
    def test_score_delta_vs_baseline(self):
        r = self._make_result()
        r.finalise()
        expected_delta = round(r.final_score - r.baseline, 4)
        assert r.score_delta_vs_baseline == pytest.approx(expected_delta)
    def test_component_breakdown_property(self):
        r = self._make_result()
        r.finalise()
        bd = r.component_breakdown
        assert "triage" in bd
        assert all(isinstance(v, float) for v in bd.values())
    def test_add_note(self):
        r = self._make_result()
        r.add_note("test note 1")
        r.add_note("test note 2")
        assert len(r.notes) == 2
    def test_total_penalty_accumulates(self):
        r = self._make_result()
        r.add_penalty(ScoringUtils.build_penalty("p1", 0.10, "reason1"))
        r.add_penalty(ScoringUtils.build_penalty("p2", 0.20, "reason2"))
        assert r.total_penalty == pytest.approx(-0.30)
class TestGraderInput:
    def _make_gi(self) -> GraderInput:
        ps = [
            _make_patient_summary(patient_id="P001", severity="P1"),
            _make_patient_summary(patient_id="P002", severity="P2", weight=2.0, weighted_reward=0.6),
            _make_patient_summary(patient_id="P003", severity="P3", weight=1.0, weighted_reward=0.4),
        ]
        al = _make_action_log([
            {"action_type": "dispatch", "incident_id": "INC001", "unit_type": "MICU"},
            {"action_type": "dispatch", "incident_id": "INC002", "unit_type": "ALS"},
            {"action_type": "noop"},
        ])
        return GraderInput(
            task_id="task4_multi_incident",
            episode_id="ep-test",
            seed=45,
            action_log=al,
            episode_ledger=_minimal_ledger(patient_summaries=ps),
            observation_log=[],
            episode_steps=15,
            total_patients=3,
            p1_patients=1,
        )
    def test_get_actions_by_type_dispatch(self):
        gi = self._make_gi()
        dispatches = gi.get_actions_by_type("dispatch")
        assert len(dispatches) == 2
    def test_get_actions_by_type_noop(self):
        gi = self._make_gi()
        noops = gi.get_actions_by_type("noop")
        assert len(noops) == 1
    def test_get_actions_by_type_empty(self):
        gi = self._make_gi()
        assert gi.get_actions_by_type("surge") == []
    def test_p1_summaries(self):
        gi = self._make_gi()
        p1s = gi.p1_summaries()
        assert len(p1s) == 1
        assert p1s[0]["patient_id"] == "P001"
    def test_all_patient_summaries(self):
        gi = self._make_gi()
        assert len(gi.all_patient_summaries()) == 3
    def test_get_patient_summary_found(self):
        gi = self._make_gi()
        ps = gi.get_patient_summary("P002")
        assert ps is not None
        assert ps["severity"] == "P2"
    def test_get_patient_summary_not_found(self):
        gi = self._make_gi()
        assert gi.get_patient_summary("P999") is None
    def test_cascade_failure_default(self):
        gi = self._make_gi()
        assert gi.cascade_failure_occurred is False
class TestActionLogEntry:
    def test_get_returns_action_data_field(self):
        entry = ActionLogEntry(
            step=1,
            action_type="dispatch",
            action_data={"incident_id": "INC001", "unit_type": "MICU"},
        )
        assert entry.get("incident_id") == "INC001"
        assert entry.get("unit_type") == "MICU"
        assert entry.get("missing_field", "default") == "default"
    def test_observation_snapshot_optional(self):
        entry = ActionLogEntry(step=1, action_type="noop", action_data={})
        assert entry.observation_snapshot is None
class TestGraderDeterminism:
    N_RUNS: int = 3
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_deterministic_same_input(self, task_id: str):
        gi = _build_grader_input(
            task_id=task_id,
            patient_summaries=[_make_patient_summary(patient_id="P001", severity="P1")],
        )
        grader = GraderRegistry.get_instance(task_id)
        scores = [grader.grade(gi).final_score for _ in range(self.N_RUNS)]
        assert len(set(round(s, 10) for s in scores)) == 1, (
            f"{task_id}: non-deterministic scores across {self.N_RUNS} runs: {scores}"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_different_seeds_different_inputs_may_differ(self, task_id: str):
        gi1 = _build_grader_input(task_id=task_id, seed=TASK_SEEDS[task_id])
        gi2 = _build_grader_input(task_id=task_id, seed=TASK_SEEDS[task_id] + 100)
        g = GraderRegistry.get_instance(task_id)
        r1 = g.grade(gi1)
        r2 = g.grade(gi2)
        assert 0.0 <= r1.final_score <= 1.0
        assert 0.0 <= r2.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_input_hash_deterministic(self, task_id: str):
        gi = _build_grader_input(task_id=task_id)
        grader = GraderRegistry.get_instance(task_id)
        results = [grader.grade(gi) for _ in range(2)]
        assert results[0].input_hash == results[1].input_hash
class TestGraderScoreRange:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_score_in_range_minimal_input(self, task_id: str):
        gi = _build_grader_input(task_id=task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert isinstance(result.final_score, float), f"{task_id}: score not float"
        assert 0.0 <= result.final_score <= 1.0, (
            f"{task_id}: score={result.final_score} outside [0,1]"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_score_in_range_no_patients(self, task_id: str):
        gi = _build_grader_input(task_id=task_id, patient_summaries=[], total_patients=0, p1_patients=0)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= result.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_score_in_range_many_patients(self, task_id: str):
        patients = [
            _make_patient_summary(
                patient_id=f"P{i:03d}",
                severity="P1" if i % 3 == 0 else ("P2" if i % 3 == 1 else "P3"),
            )
            for i in range(20)
        ]
        gi = _build_grader_input(
            task_id=task_id, patient_summaries=patients,
            total_patients=20, p1_patients=7,
        )
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= result.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_score_not_nan_or_inf(self, task_id: str):
        gi = _build_grader_input(task_id=task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert not math.isnan(result.final_score), f"{task_id}: score is NaN"
        assert not math.isinf(result.final_score), f"{task_id}: score is Inf"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_status_is_valid_enum_value(self, task_id: str):
        gi = _build_grader_input(task_id=task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert result.status in GraderStatus
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_components_individually_in_range(self, task_id: str):
        gi = _build_grader_input(
            task_id=task_id,
            patient_summaries=[_make_patient_summary()],
        )
        result = GraderRegistry.get_instance(task_id).grade(gi)
        for comp in result.components:
            assert 0.0 <= comp.raw_score <= 1.0, (
                f"{task_id}: component '{comp.name}' raw_score={comp.raw_score} out of range"
            )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_penalty_amounts_are_non_positive(self, task_id: str):
        gi = _build_grader_input(task_id=task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        for pen in result.penalties:
            assert pen.amount <= 0.0, (
                f"{task_id}: penalty '{pen.name}' has positive amount={pen.amount}"
            )
class TestGraderBaselineProximity:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_noop_does_not_greatly_exceed_baseline(self, task_id: str):
        gi = _run_noop_episode(task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        baseline = TASK_BASELINES[task_id]
        assert result.final_score <= baseline + _BASELINE_NOOP_MAX_OVERSHOOT, (
            f"{task_id}: noop score={result.final_score:.4f} exceeds "
            f"baseline={baseline:.2f} by more than {_BASELINE_NOOP_MAX_OVERSHOOT:.2f}"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_grader_returns_status_success_or_partial_on_noop(self, task_id: str):
        gi = _run_noop_episode(task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert result.status in (GraderStatus.SUCCESS, GraderStatus.PARTIAL), (
            f"{task_id}: unexpected status {result.status} on noop episode"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_episode_steps_recorded(self, task_id: str):
        gi = _run_noop_episode(task_id)
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert result.episode_steps > 0
class TestTask1Grader:
    TID = "task1_single_triage"
    def _grade(self, **kwargs) -> GraderResult:
        ps = [_make_patient_summary(**kwargs)]
        gi = _build_grader_input(self.TID, patient_summaries=ps)
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_perfect_dispatch_high_score(self):
        r = self._grade(
            assigned_priority="P1", required_unit_type="MICU",
            assigned_unit="MICU-01", hospital_has_specialty=True,
            hospital_on_diversion=False, hospital_er_occupancy_pct=0.60,
        )
        assert r.final_score > 0.60
    def test_wrong_unit_type_reduces_score(self):
        r_correct = self._grade(assigned_unit="MICU-01", required_unit_type="MICU")
        r_wrong   = self._grade(assigned_unit="BLS-01",  required_unit_type="MICU")
        assert r_correct.final_score >= r_wrong.final_score
    def test_diverted_hospital_reduces_score(self):
        r_ok  = self._grade(hospital_on_diversion=False)
        r_div = self._grade(hospital_on_diversion=True)
        assert r_ok.final_score >= r_div.final_score
    def test_triage_class_mismatch_reduces_score(self):
        r_correct = self._grade(assigned_priority="P1")
        r_wrong   = self._grade(assigned_priority="P3")
        assert r_correct.final_score >= r_wrong.final_score
    def test_has_three_components(self):
        r = self._grade()
        assert len(r.components) >= 3
    def test_grading_time_reasonable(self):
        r = self._grade()
        assert r.grading_time_ms < GRADER_TIMEOUT_SECONDS * 1000
class TestTask2Grader:
    TID = "task2_hospital_route"
    def _grade(self, **kwargs) -> GraderResult:
        ps = [_make_patient_summary(**kwargs)]
        gi = _build_grader_input(self.TID, patient_summaries=ps)
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_specialty_match_bonus(self):
        r_match    = self._grade(hospital_has_specialty=True,  hospital_on_diversion=False)
        r_no_match = self._grade(hospital_has_specialty=False, hospital_on_diversion=False)
        assert r_match.final_score >= r_no_match.final_score
    def test_diverted_hospital_penalty(self):
        r_ok  = self._grade(hospital_on_diversion=False)
        r_div = self._grade(hospital_on_diversion=True)
        assert r_ok.final_score >= r_div.final_score
    def test_high_occupancy_reduces_score(self):
        r_lo = self._grade(hospital_er_occupancy_pct=0.50)
        r_hi = self._grade(hospital_er_occupancy_pct=0.95)
        assert r_lo.final_score >= r_hi.final_score
    def test_specialty_weight_dominant(self):
        r_spec    = self._grade(hospital_has_specialty=True,  hospital_er_occupancy_pct=0.85)
        r_no_spec = self._grade(hospital_has_specialty=False, hospital_er_occupancy_pct=0.30)
        assert r_spec.final_score >= r_no_spec.final_score
class TestTask3Grader:
    TID = "task3_unit_type"
    def _grade(self, *, assigned: str, required: str) -> GraderResult:
        ps = [_make_patient_summary(assigned_unit=f"{assigned}-01", required_unit_type=required)]
        gi = _build_grader_input(self.TID, patient_summaries=ps)
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_micu_micu_high_score(self):
        r = self._grade(assigned="MICU", required="MICU")
        assert r.final_score >= 0.70
    def test_als_als_high_score(self):
        r = self._grade(assigned="ALS", required="ALS")
        assert r.final_score >= 0.70
    def test_bls_micu_penalty(self):
        r_correct = self._grade(assigned="MICU", required="MICU")
        r_wrong   = self._grade(assigned="BLS",  required="MICU")
        assert r_correct.final_score > r_wrong.final_score
    def test_score_clamped_in_range(self):
        for assigned in ("BLS", "ALS", "MICU"):
            for required in ("BLS", "ALS", "MICU"):
                r = self._grade(assigned=assigned, required=required)
                assert 0.0 <= r.final_score <= 1.0
class TestTask4Grader:
    TID = "task4_multi_incident"
    def _multi_grade(self, summaries: List[Dict]) -> GraderResult:
        gi = _build_grader_input(
            self.TID,
            patient_summaries=summaries,
            total_patients=len(summaries),
            p1_patients=sum(1 for s in summaries if s["severity"] == "P1"),
        )
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_all_p1_treated_high_score(self):
        patients = [
            _make_patient_summary(patient_id=f"P{i}", severity="P1", phase="treated",
                                  survival_prob=0.85, weight=3.0, weighted_reward=0.72)
            for i in range(3)
        ]
        r = self._multi_grade(patients)
        assert r.final_score > 0.30
    def test_p1_untreated_lower_score(self):
        p_treated   = [_make_patient_summary(patient_id="P1", severity="P1", phase="treated")]
        p_untreated = [_make_patient_summary(patient_id="P1", severity="P1", phase="waiting",
                                              survival_prob=0.0, weighted_reward=0.0)]
        r_ok  = self._multi_grade(p_treated)
        r_bad = self._multi_grade(p_untreated)
        assert r_ok.final_score >= r_bad.final_score
    def test_weighted_survival_uses_severity_weight(self):
        p_p1 = [_make_patient_summary(patient_id="P1", severity="P1", weight=3.0, weighted_reward=0.72)]
        p_p3 = [_make_patient_summary(patient_id="P1", severity="P3", weight=1.0, weighted_reward=0.72)]
        r_p1 = self._multi_grade(p_p1)
        r_p3 = self._multi_grade(p_p3)
        assert 0.0 <= r_p1.final_score <= 1.0
        assert 0.0 <= r_p3.final_score <= 1.0
class TestTask5Grader:
    TID = "task5_dynamic_rerouting"
    def _grade_with_reroutes(self, *, n_rerouted: int, time_saved: float) -> GraderResult:
        patients = [
            _make_patient_summary(
                patient_id=f"P{i}",
                severity="P1",
                phase="treated" if i < n_rerouted else "en_route",
                optimal_travel_time_min=6.0,
            )
            for i in range(3)
        ]
        al = _make_action_log(
            [{"action_type": "reroute", "unit_id": f"ALS-0{i}", "new_hospital_id": "H02"}
             for i in range(n_rerouted)]
            + [{"action_type": "noop"}]
        )
        gi = _build_grader_input(self.TID, patient_summaries=patients, action_log=al)
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_reroutes_present_non_zero_score(self):
        r = self._grade_with_reroutes(n_rerouted=2, time_saved=8.0)
        assert r.final_score >= 0.0
    def test_no_reroutes_low_score(self):
        r = self._grade_with_reroutes(n_rerouted=0, time_saved=0.0)
        assert 0.0 <= r.final_score <= 1.0
    def test_score_in_range(self):
        for n in range(4):
            r = self._grade_with_reroutes(n_rerouted=n, time_saved=float(n * 5))
            assert 0.0 <= r.final_score <= 1.0
class TestTask6Grader:
    TID = "task6_prepositioning"
    def _grade_preposition(self, *, n_prepositioned: int) -> GraderResult:
        al = _make_action_log(
            [{"action_type": "preposition", "unit_id": f"BLS-0{i}", "target_zone": f"zone_{i+1}"}
             for i in range(n_prepositioned)]
        )
        gi = _build_grader_input(
            self.TID, patient_summaries=[], action_log=al, total_patients=0, p1_patients=0
        )
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_some_prepositioning_valid_score(self):
        r = self._grade_preposition(n_prepositioned=4)
        assert 0.0 <= r.final_score <= 1.0
    def test_no_prepositioning_valid_score(self):
        r = self._grade_preposition(n_prepositioned=0)
        assert 0.0 <= r.final_score <= 1.0
    def test_more_prepositioning_not_worse(self):
        r_few  = self._grade_preposition(n_prepositioned=1)
        r_many = self._grade_preposition(n_prepositioned=6)
        assert 0.0 <= r_few.final_score <= 1.0
        assert 0.0 <= r_many.final_score <= 1.0
class TestTask7Grader:
    TID = "task7_mci_start"
    def _grade_mci(
        self,
        *,
        n_victims: int = 10,
        correct_tags: int = 8,
        wrong_critical: int = 0,
    ) -> GraderResult:
        patients = []
        for i in range(n_victims):
            is_immediate = (i < 4)
            gt_tag = "Immediate" if is_immediate else "Delayed"
            if i < correct_tags:
                applied = gt_tag
            else:
                applied = "Expectant" if (is_immediate and i - correct_tags < wrong_critical) else "Minimal"
            patients.append(_make_patient_summary(
                patient_id=f"P{i:03d}",
                severity="P1" if is_immediate else "P2",
                start_tag=applied,
                ground_truth_start_tag=gt_tag,
                mci_type="road_accident_bus",
                weight=3.0 if is_immediate else 2.0,
                weighted_reward=0.60 if applied == gt_tag else 0.10,
            ))
        al = _make_action_log(
            [{"action_type": "tag", "incident_id": f"P{i:03d}", "tag": patients[i]["start_tag"]}
             for i in range(n_victims)]
        )
        gi = _build_grader_input(
            self.TID,
            patient_summaries=patients,
            action_log=al,
            total_patients=n_victims,
            p1_patients=4,
        )
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_all_correct_tags_high_score(self):
        r = self._grade_mci(n_victims=10, correct_tags=10)
        assert r.final_score > 0.20
    def test_immediate_as_expectant_reduces_score(self):
        r_ok  = self._grade_mci(n_victims=10, correct_tags=10, wrong_critical=0)
        r_bad = self._grade_mci(n_victims=10, correct_tags=8,  wrong_critical=2)
        assert r_ok.final_score >= r_bad.final_score
    def test_zero_victims_handled(self):
        gi = _build_grader_input(self.TID, patient_summaries=[], total_patients=0, p1_patients=0)
        r = GraderRegistry.get_instance(self.TID).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
class TestTask8Grader:
    TID = "task8_transfer_cascade"
    def _grade_transfer(self, *, transfer_initiated: bool, latency: float, icu_util: float) -> GraderResult:
        ps = [
            _make_patient_summary(
                patient_id="P001",
                severity="P1",
                transfer_required=True,
                transfer_initiated=transfer_initiated,
                specialist_hospital_id="H02",
                assigned_hospital="H02" if transfer_initiated else "H01",
                transfer_window_min=60.0,
                transfer_latency_min=latency if transfer_initiated else None,
                destination_icu_utilisation=icu_util,
            )
        ]
        gi = _build_grader_input(self.TID, patient_summaries=ps)
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_timely_transfer_good_score(self):
        r = self._grade_transfer(transfer_initiated=True, latency=20.0, icu_util=0.60)
        assert r.final_score > 0.10
    def test_no_transfer_low_score(self):
        r = self._grade_transfer(transfer_initiated=False, latency=0.0, icu_util=0.60)
        assert 0.0 <= r.final_score <= 1.0
    def test_high_icu_util_reduces_score(self):
        r_lo = self._grade_transfer(transfer_initiated=True, latency=20.0, icu_util=0.60)
        r_hi = self._grade_transfer(transfer_initiated=True, latency=20.0, icu_util=0.98)
        assert r_lo.final_score >= r_hi.final_score
    def test_late_transfer_penalty(self):
        r_early = self._grade_transfer(transfer_initiated=True, latency=10.0, icu_util=0.60)
        r_late  = self._grade_transfer(transfer_initiated=True, latency=80.0, icu_util=0.60)
        assert r_early.final_score >= r_late.final_score
class TestTask9Grader:
    TID = "task9_surge"
    def _grade_surge(
        self,
        *,
        surge_declared: bool = True,
        mutual_aid_requests: int = 4,
        cascade_events: int = 0,
        n_patients: int = 30,
    ) -> GraderResult:
        patients = [
            _make_patient_summary(
                patient_id=f"P{i:03d}",
                severity="P1" if i % 3 == 0 else "P2",
                phase="treated" if i < n_patients // 2 else "waiting",
                mutual_aid_required=True,
                surge_zone="zone_1",
                mci_type="road_accident_bus",
                weight=3.0 if i % 3 == 0 else 2.0,
                weighted_reward=0.50 if i < n_patients // 2 else 0.0,
            )
            for i in range(n_patients)
        ]
        al = _make_action_log(
            ([{"action_type": "declare_surge", "reason": "test"}] if surge_declared else [])
            + [{"action_type": "request_mutual_aid", "n_units": 3, "from_zone": "zone_2"}
               for _ in range(min(mutual_aid_requests, 3))]
        )
        gi = _build_grader_input(
            self.TID,
            patient_summaries=patients,
            action_log=al,
            total_patients=n_patients,
            p1_patients=n_patients // 3,
            surge_declared=surge_declared,
            mutual_aid_requests=mutual_aid_requests,
            cascade_events=[{"type": "failure", "step": i} for i in range(cascade_events)],
        )
        return GraderRegistry.get_instance(self.TID).grade(gi)
    def test_surge_declared_with_mutual_aid(self):
        r = self._grade_surge(surge_declared=True, mutual_aid_requests=4, cascade_events=0)
        assert 0.0 <= r.final_score <= 1.0
    def test_no_surge_lower_score(self):
        r_surge   = self._grade_surge(surge_declared=True,  mutual_aid_requests=4)
        r_no_surge = self._grade_surge(surge_declared=False, mutual_aid_requests=0)
        assert r_surge.final_score >= r_no_surge.final_score
    def test_cascade_reduces_score(self):
        r_ok      = self._grade_surge(cascade_events=0)
        r_cascade = self._grade_surge(cascade_events=3)
        assert r_ok.final_score >= r_cascade.final_score
    def test_score_below_hard_baseline(self):
        gi = _run_noop_episode(self.TID)
        r = GraderRegistry.get_instance(self.TID).grade(gi)
        assert r.final_score <= 0.40, (
            f"Task 9 noop score {r.final_score:.4f} seems too high for a hard task"
        )
class TestGraderEdgeCases:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_empty_action_log(self, task_id: str):
        gi = _build_grader_input(task_id=task_id, action_log=[])
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_all_fields_zero(self, task_id: str):
        ps = [_make_patient_summary(
            survival_prob=0.0, optimal_survival_prob=0.0,
            weighted_reward=0.0, hospital_er_occupancy_pct=0.0,
        )]
        gi = _build_grader_input(task_id=task_id, patient_summaries=ps)
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_all_fields_max(self, task_id: str):
        ps = [_make_patient_summary(
            survival_prob=1.0, optimal_survival_prob=1.0,
            weighted_reward=1.0, hospital_er_occupancy_pct=1.0,
        )]
        gi = _build_grader_input(task_id=task_id, patient_summaries=ps)
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_missing_optional_fields(self, task_id: str):
        gi = GraderInput(
            task_id=task_id,
            episode_id=str(uuid.uuid4()),
            seed=TASK_SEEDS.get(task_id, 42),
            action_log=[],
            episode_ledger={},  
            observation_log=[],
            episode_steps=0,
            total_patients=0,
            p1_patients=0,
        )
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_100_patients(self, task_id: str):
        patients = [
            _make_patient_summary(patient_id=f"P{i:04d}", severity="P1" if i < 10 else "P2")
            for i in range(100)
        ]
        gi = _build_grader_input(
            task_id=task_id, patient_summaries=patients,
            total_patients=100, p1_patients=10,
        )
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_very_long_action_log(self, task_id: str):
        al = _make_action_log([{"action_type": "noop"} for _ in range(500)])
        gi = _build_grader_input(task_id=task_id, action_log=al, episode_steps=500)
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_negative_latency_field(self, task_id: str):
        ps = [_make_patient_summary(dispatch_latency_min=-5.0, optimal_travel_time_min=-1.0)]
        gi = _build_grader_input(task_id=task_id, patient_summaries=ps)
        r = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= r.final_score <= 1.0
class TestGraderPipeline:
    def _run_pipeline(self, task_ids: Optional[List[str]] = None) -> Tuple[Dict, GraderPipeline]:
        tids = task_ids or ALL_TASK_IDS
        pipeline = GraderPipeline(task_ids=tids)
        inputs = {
            tid: _build_grader_input(tid, patient_summaries=[_make_patient_summary()])
            for tid in tids
        }
        results = pipeline.run(inputs)
        return results, pipeline
    def test_pipeline_runs_all_tasks(self):
        results, _ = self._run_pipeline()
        assert set(results.keys()) == set(ALL_TASK_IDS)
    def test_pipeline_all_scores_in_range(self):
        results, _ = self._run_pipeline()
        for tid, res in results.items():
            assert 0.0 <= res.final_score <= 1.0, f"{tid}: score={res.final_score}"
    def test_pipeline_aggregate_in_range(self):
        results, pipeline = self._run_pipeline()
        agg = pipeline.aggregate_score(results)
        assert 0.0 <= agg <= 1.0
    def test_pipeline_weighted_aggregate(self):
        results, pipeline = self._run_pipeline()
        weights = {tid: 1.0 / len(ALL_TASK_IDS) for tid in ALL_TASK_IDS}
        agg = pipeline.aggregate_score(results, weights=weights)
        assert 0.0 <= agg <= 1.0
    def test_pipeline_empty_results(self):
        pipeline = GraderPipeline()
        agg = pipeline.aggregate_score({})
        assert agg == 0.0
    def test_summary_table_contains_all_tasks(self):
        results, pipeline = self._run_pipeline()
        table = pipeline.summary_table(results)
        assert isinstance(table, str)
        for tid in ALL_TASK_IDS:
            assert tid in table, f"{tid} not in summary table"
    def test_summary_table_has_baseline_column(self):
        results, pipeline = self._run_pipeline()
        table = pipeline.summary_table(results)
        assert "Baseline" in table or "baseline" in table.lower()
    def test_pipeline_subset_tasks(self):
        subset = ["task1_single_triage", "task4_multi_incident", "task9_surge"]
        results, pipeline = self._run_pipeline(task_ids=subset)
        assert set(results.keys()) == set(subset)
    def test_pipeline_missing_input_skipped(self):
        pipeline = GraderPipeline(task_ids=ALL_TASK_IDS)
        inputs = {
            "task1_single_triage": _build_grader_input("task1_single_triage")
        }  
        results = pipeline.run(inputs)
        assert "task1_single_triage" in results
        assert len(results) == 1
class TestProtocolComplianceBonus:
    def test_protocol_bonus_cap(self):
        r = GraderResult(task_id="task1_single_triage", episode_id="x", seed=42, baseline=0.61)
        for _ in range(20):
            r.add_component(ScoringUtils.build_score_component("protocol", 1.0, 0.01))
        r.finalise()
        assert r.final_score <= 1.0
    def test_large_penalty_floored(self):
        r = GraderResult(task_id="task1_single_triage", episode_id="x", seed=42, baseline=0.61)
        for _ in range(10):
            r.add_penalty(ScoringUtils.build_penalty(f"p", 1.0, "big"))
        r.finalise()
        assert r.final_score == 0.0
    def test_penalty_sign_always_negative(self):
        p = ScoringUtils.build_penalty("test", 0.50, "reason")
        assert p.amount < 0.0
class TestGraderResultSummaryLine:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_summary_line_contains_task_id(self, task_id: str):
        r = GraderResult(task_id=task_id, episode_id="ep", seed=42, baseline=0.5)
        r.finalise()
        line = r.summary_line()
        assert task_id in line
    def test_summary_line_beats_baseline_marker(self):
        r = GraderResult(task_id="task1_single_triage", episode_id="ep", seed=42, baseline=0.20)
        r.add_component(ScoringUtils.build_score_component("x", 1.0, 1.0))
        r.finalise()
        line = r.summary_line()
        assert "BEATS" in line or "✓" in line
    def test_summary_line_below_baseline_marker(self):
        r = GraderResult(task_id="task1_single_triage", episode_id="ep", seed=42, baseline=0.99)
        r.finalise()
        line = r.summary_line()
        assert "BELOW" in line or "✗" in line
class TestGraderIntegration:
    def _run_and_grade(self, task_id: str, n_actions: int = 5) -> GraderResult:
        env = make_env()
        env.reset(task_id=task_id, seed=TASK_SEEDS[task_id])
        for i in range(n_actions):
            if env.is_done:
                break
            env.step({"action_type": "noop", "reason": "integration_test"})
        gi = env.to_grader_input()
        return GraderRegistry.get_instance(task_id).grade(gi)
    @pytest.mark.parametrize("task_id", ["task1_single_triage", "task3_unit_type"])
    def test_easy_tasks_round_trip(self, task_id: str):
        r = self._run_and_grade(task_id, n_actions=8)
        assert 0.0 <= r.final_score <= 1.0
        assert r.status in (GraderStatus.SUCCESS, GraderStatus.PARTIAL)
    @pytest.mark.parametrize("task_id", ["task4_multi_incident", "task5_dynamic_rerouting"])
    def test_medium_tasks_round_trip(self, task_id: str):
        r = self._run_and_grade(task_id, n_actions=15)
        assert 0.0 <= r.final_score <= 1.0
    @pytest.mark.parametrize("task_id", ["task7_mci_start", "task9_surge"])
    def test_hard_tasks_round_trip(self, task_id: str):
        r = self._run_and_grade(task_id, n_actions=20)
        assert 0.0 <= r.final_score <= 1.0
    def test_grader_input_task_id_matches_env_task_id(self):
        for task_id in ["task1_single_triage", "task6_prepositioning"]:
            env = make_env()
            env.reset(task_id=task_id, seed=TASK_SEEDS[task_id])
            gi = env.to_grader_input()
            assert gi.task_id == task_id
    def test_grader_input_episode_id_non_empty(self):
        env = make_env()
        env.reset(task_id="task2_hospital_route", seed=43)
        gi = env.to_grader_input()
        assert gi.episode_id and gi.episode_id != "unknown"
    def test_grader_input_seed_matches(self):
        task_id = "task1_single_triage"
        seed = TASK_SEEDS[task_id]
        env = make_env()
        env.reset(task_id=task_id, seed=seed)
        gi = env.to_grader_input()
        assert gi.seed == seed
    def test_full_episode_then_grade_all_easy_tasks(self):
        easy = [t for t in ALL_TASK_IDS if DIFFICULTY_MAP.get(t) == "easy"]
        for tid in easy:
            env = make_env()
            env.reset(task_id=tid, seed=TASK_SEEDS[tid])
            max_s = MAX_STEPS_BY_TASK[tid]
            for _ in range(max_s):
                if env.is_done:
                    break
                env.step({"action_type": "noop"})
            assert env.is_done, f"{tid}: episode not done after {max_s} steps"
            gi = env.to_grader_input()
            r = GraderRegistry.get_instance(tid).grade(gi)
            assert 0.0 <= r.final_score <= 1.0, f"{tid}: score {r.final_score} out of range"
class TestGraderPipelineSummaryTable:
    def test_table_is_string(self):
        p = GraderPipeline()
        table = p.summary_table({})
        assert isinstance(table, str)
    def test_table_header_present(self):
        p = GraderPipeline(task_ids=ALL_TASK_IDS)
        inputs = {tid: _build_grader_input(tid) for tid in ALL_TASK_IDS}
        results = p.run(inputs)
        table = p.summary_table(results)
        assert "EMERGI" in table or "Score" in table
    def test_table_line_count_gt_task_count(self):
        p = GraderPipeline(task_ids=ALL_TASK_IDS)
        inputs = {tid: _build_grader_input(tid) for tid in ALL_TASK_IDS}
        results = p.run(inputs)
        table = p.summary_table(results)
        lines = [l for l in table.split("\n") if l.strip()]
        assert len(lines) >= len(ALL_TASK_IDS) + 2
def test_base_grader_version():
    assert BASE_GRADER_VERSION >= 1
def test_score_floor_and_ceiling():
    assert SCORE_FLOOR == 0.0
    assert SCORE_CEILING == 1.0
def test_max_penalty_fraction_in_range():
    assert 0.0 < MAX_PENALTY_FRACTION <= 1.0
def test_grader_timeout_positive():
    assert GRADER_TIMEOUT_SECONDS > 0.0
def test_task_baselines_order():
    easy_avg   = sum(TASK_BASELINES[t] for t in ALL_TASK_IDS if DIFFICULTY_MAP.get(t) == "easy")   / 3
    medium_avg = sum(TASK_BASELINES[t] for t in ALL_TASK_IDS if DIFFICULTY_MAP.get(t) == "medium") / 3
    hard_avg   = sum(TASK_BASELINES[t] for t in ALL_TASK_IDS if DIFFICULTY_MAP.get(t) == "hard")   / 3
    assert easy_avg > medium_avg > hard_avg, (
        f"Baseline ordering wrong: easy={easy_avg:.3f} medium={medium_avg:.3f} hard={hard_avg:.3f}"
    )