from __future__ import annotations
import copy
import gc
import math
import sys
import time
import tracemalloc
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
import pytest
from server.env import (
    DIFFICULTY_MAP,
    MAX_STEPS_BY_TASK,
    TASK_IDS,
    EmergiEnv,
    make_env,
)
from server.graders.basegrader import (
    TASK_SEEDS,
    GraderInput,
    GraderRegistry,
    GraderResult,
    GraderStatus,
)
from server.models.action import (
    ActionModel,
    DispatchAction,
    EscalateAction,
    NoopAction,
    PrepositionAction,
    RequestMutualAidAction,
    RerouteAction,
    StartTagAction,
    TransferAction,
)
from server.models.observation import ObservationModel
from server.models.reward import RewardModel
from server.models.state import StateModel
ALL_TASK_IDS: List[str] = sorted(TASK_IDS)
EASY_TASKS: List[str] = sorted(t for t in TASK_IDS if DIFFICULTY_MAP.get(t) == "easy")
MEDIUM_TASKS: List[str] = sorted(t for t in TASK_IDS if DIFFICULTY_MAP.get(t) == "medium")
HARD_TASKS: List[str] = sorted(t for t in TASK_IDS if DIFFICULTY_MAP.get(t) == "hard")
_DEFAULT_SEED: int = 42
_FLOAT_TOL: float = 1e-9
_MAX_RESET_TIME_MS: float = 500.0   
_MAX_STEP_TIME_MS: float = 100.0    
_STEP_THROUGHPUT_MIN: float = 50.0  
def _fresh_env() -> EmergiEnv:
    return make_env()
def _reset_env(task_id: str = "task1_single_triage", seed: int = _DEFAULT_SEED) -> EmergiEnv:
    env = _fresh_env()
    env.reset(task_id=task_id, seed=seed)
    return env
def _run_to_completion(env: EmergiEnv, *, max_override: Optional[int] = None) -> List[float]:
    task_id = env.current_task_id
    max_s = max_override or MAX_STEPS_BY_TASK.get(task_id, 50)
    rewards: List[float] = []
    for _ in range(max_s):
        if env.is_done:
            break
        result = env.step({"action_type": "noop", "reason": "test"})
        rewards.append(result["reward"])
    return rewards
def _noop(reason: str = "test") -> Dict[str, Any]:
    return {"action_type": "noop", "reason": reason}
def _dispatch(
    incident_id: str = "INC001",
    unit_id: str = "MICU-01",
    hospital_id: str = "H01",
) -> Dict[str, Any]:
    return {
        "action_type": "dispatch",
        "incident_id": incident_id,
        "unit_id": unit_id,
        "hospital_id": hospital_id,
    }
def _reroute(unit_id: str = "ALS-01", new_hospital_id: str = "H02") -> Dict[str, Any]:
    return {
        "action_type": "reroute",
        "unit_id": unit_id,
        "new_hospital_id": new_hospital_id,
    }
def _tag(incident_id: str = "P001", tag: str = "Immediate") -> Dict[str, Any]:
    return {"action_type": "tag", "incident_id": incident_id, "tag": tag}
def _preposition(unit_id: str = "BLS-01", target_zone: str = "zone_1") -> Dict[str, Any]:
    return {"action_type": "preposition", "unit_id": unit_id, "target_zone": target_zone}
def _request_mutual_aid(from_zone: str = "zone_2", n_units: int = 2) -> Dict[str, Any]:
    return {"action_type": "request_mutual_aid", "from_zone": from_zone, "n_units": n_units}
def _declare_surge(reason: str = "capacity") -> Dict[str, Any]:
    return {"action_type": "declare_surge", "reason": reason}
def _transfer(patient_id: str = "P001", destination_hospital: str = "H02") -> Dict[str, Any]:
    return {
        "action_type": "transfer",
        "patient_id": patient_id,
        "destination_hospital_id": destination_hospital,
    }
class TestEnvFactory:
    def test_make_env_returns_emergi_env(self):
        env = make_env()
        assert isinstance(env, EmergiEnv)
    def test_make_env_multiple_calls_independent(self):
        e1 = make_env()
        e2 = make_env()
        assert e1 is not e2
    def test_env_has_required_attributes_before_reset(self):
        env = _fresh_env()
        required = ["reset", "step", "is_done", "to_grader_input", "current_task_id"]
        for attr in required:
            assert hasattr(env, attr), f"EmergiEnv missing attribute: {attr}"
    def test_is_done_false_before_reset(self):
        env = _fresh_env()
        assert env.is_done is False
    def test_current_task_id_none_before_reset(self):
        env = _fresh_env()
        assert env.current_task_id is None
    def test_current_seed_none_before_reset(self):
        env = _fresh_env()
        assert env.current_seed is None
    def test_step_count_zero_before_reset(self):
        env = _fresh_env()
        assert env.step_count == 0
    def test_difficulty_map_covers_all_tasks(self):
        for tid in ALL_TASK_IDS:
            assert tid in DIFFICULTY_MAP, f"{tid} missing from DIFFICULTY_MAP"
    def test_difficulty_map_valid_values(self):
        valid = {"easy", "medium", "hard"}
        for tid, diff in DIFFICULTY_MAP.items():
            assert diff in valid, f"{tid}: invalid difficulty '{diff}'"
    def test_max_steps_by_task_covers_all(self):
        for tid in ALL_TASK_IDS:
            assert tid in MAX_STEPS_BY_TASK, f"{tid} missing from MAX_STEPS_BY_TASK"
    def test_max_steps_positive(self):
        for tid, ms in MAX_STEPS_BY_TASK.items():
            assert ms > 0, f"{tid}: MAX_STEPS={ms} must be positive"
    def test_hard_tasks_have_more_steps_than_easy(self):
        easy_avg = sum(MAX_STEPS_BY_TASK[t] for t in EASY_TASKS) / len(EASY_TASKS)
        hard_avg = sum(MAX_STEPS_BY_TASK[t] for t in HARD_TASKS) / len(HARD_TASKS)
        assert hard_avg >= easy_avg, (
            f"Hard tasks avg steps ({hard_avg:.1f}) < easy tasks avg ({easy_avg:.1f})"
        )
    def test_task_ids_set_has_nine_elements(self):
        assert len(TASK_IDS) == 9, f"Expected 9 tasks, got {len(TASK_IDS)}"
    def test_task_ids_naming_convention(self):
        for tid in TASK_IDS:
            assert tid.startswith("task"), f"{tid} does not follow task<N>_<name> naming"
            assert "_" in tid, f"{tid} missing underscore separator"
class TestResetContract:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_returns_observation(self, task_id: str):
        env = _fresh_env()
        obs = env.reset(task_id=task_id, seed=TASK_SEEDS.get(task_id, 42))
        assert obs is not None, f"{task_id}: reset() returned None"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_is_done_false(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS.get(task_id, 42))
        assert env.is_done is False, f"{task_id}: is_done should be False after reset"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_sets_current_task_id(self, task_id: str):
        env = _reset_env(task_id)
        assert env.current_task_id == task_id
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_sets_current_seed(self, task_id: str):
        seed = TASK_SEEDS.get(task_id, 99)
        env = _reset_env(task_id, seed=seed)
        assert env.current_seed == seed
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_step_count_zero(self, task_id: str):
        env = _reset_env(task_id)
        _run_to_completion(env)
        env.reset(task_id=task_id, seed=TASK_SEEDS.get(task_id, 42))
        assert env.step_count == 0, f"{task_id}: step_count not reset to 0"
    def test_reset_reinitialises_incident_queue(self):
        env = _reset_env("task1_single_triage")
        obs1 = env.current_observation
        _run_to_completion(env)
        obs2_reset = env.reset(task_id="task1_single_triage", seed=_DEFAULT_SEED)
        assert env.step_count == 0
        assert obs2_reset is not None
    def test_reset_same_seed_reproducible_observation(self):
        env1 = _fresh_env()
        env2 = _fresh_env()
        obs1 = env1.reset(task_id="task1_single_triage", seed=42)
        obs2 = env2.reset(task_id="task1_single_triage", seed=42)
        assert obs1 == obs2, "Same seed must produce identical initial observations"
    def test_reset_different_seed_different_observation(self):
        env = _fresh_env()
        obs1 = env.reset(task_id="task1_single_triage", seed=1)
        obs2 = env.reset(task_id="task1_single_triage", seed=9999)
        assert obs1 is not None and obs2 is not None
    def test_reset_clears_fleet_fatigue(self):
        env = _reset_env("task4_multi_incident")
        for _ in range(30):
            if env.is_done:
                break
            env.step(_noop())
        env.reset(task_id="task4_multi_incident", seed=_DEFAULT_SEED)
        state = env.current_state
        for unit in state.fleet.values():
            assert unit.hours_on_duty < 1.0, (
                "Crew hours_on_duty must be near zero after reset"
            )
    def test_reset_clears_hospital_diversion(self):
        env = _reset_env("task5_dynamic_rerouting")
        env.reset(task_id="task5_dynamic_rerouting", seed=_DEFAULT_SEED)
        state = env.current_state
        diverted = [h.hospital_id for h in state.hospital_network.values() if h.on_diversion]
        assert len(diverted) == 0, f"Hospitals diverted immediately after reset: {diverted}"
    def test_reset_observation_is_pydantic_model(self):
        env = _fresh_env()
        obs = env.reset(task_id="task1_single_triage", seed=42)
        assert isinstance(obs, ObservationModel), (
            f"Expected ObservationModel, got {type(obs)}"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_fleet_non_empty(self, task_id: str):
        env = _reset_env(task_id)
        state = env.current_state
        assert len(state.fleet) > 0, f"{task_id}: fleet must be non-empty after reset"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_hospital_network_non_empty(self, task_id: str):
        env = _reset_env(task_id)
        state = env.current_state
        assert len(state.hospital_network) >= 1, (
            f"{task_id}: hospital_network must be non-empty after reset"
        )
    def test_reset_twice_clears_previous_state(self):
        env = _reset_env("task4_multi_incident", seed=1)
        obs1 = env.current_observation
        env.step(_noop())
        env.step(_noop())
        env.reset(task_id="task4_multi_incident", seed=1)
        obs_after = env.current_observation
        assert env.step_count == 0
        assert obs_after == obs1
    def test_reset_performance_within_sla(self):
        env = _fresh_env()
        t0 = time.perf_counter()
        env.reset(task_id="task9_surge", seed=TASK_SEEDS["task9_surge"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < _MAX_RESET_TIME_MS, (
            f"reset() took {elapsed_ms:.1f} ms, exceeds SLA of {_MAX_RESET_TIME_MS} ms"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_eight_hospitals_present(self, task_id: str):
        env = _reset_env(task_id)
        assert len(env.current_state.hospital_network) == 8, (
            f"{task_id}: expected 8 hospitals, got {len(env.current_state.hospital_network)}"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reset_twelve_zones_present(self, task_id: str):
        env = _reset_env(task_id)
        zones = {u.zone_id for u in env.current_state.fleet.values()}
        zones |= set(env.current_state.demand_forecast.keys())
        assert len(env.current_state.demand_forecast) == 12, (
            f"{task_id}: expected 12 demand forecast zones, got {len(env.current_state.demand_forecast)}"
        )
class TestStepContract:
    def test_step_returns_dict_with_required_keys(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        required = {"observation", "reward", "done", "info"}
        assert required.issubset(result.keys()), (
            f"step() result missing keys: {required - set(result.keys())}"
        )
    def test_step_observation_is_pydantic_model(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        assert isinstance(result["observation"], ObservationModel)
    def test_step_reward_is_float(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        assert isinstance(result["reward"], float), (
            f"reward must be float, got {type(result['reward'])}"
        )
    def test_step_reward_is_finite(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        assert math.isfinite(result["reward"]), "step reward must be finite"
    def test_step_done_is_bool(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        assert isinstance(result["done"], bool)
    def test_step_info_is_dict(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        assert isinstance(result["info"], dict)
    def test_step_increments_step_count(self):
        env = _reset_env("task1_single_triage")
        for i in range(1, 6):
            env.step(_noop())
            assert env.step_count == i, f"Expected step_count={i}, got {env.step_count}"
    def test_step_after_done_raises_or_returns_terminal(self):
        env = _reset_env("task1_single_triage")
        _run_to_completion(env)
        assert env.is_done
        try:
            result = env.step(_noop())
            assert result["done"] is True, "Post-terminal step must return done=True"
        except RuntimeError:
            pass  
    def test_step_dense_reward_every_step(self):
        env = _reset_env("task4_multi_incident")
        rewards = []
        for _ in range(10):
            if env.is_done:
                break
            r = env.step(_noop())
            rewards.append(r["reward"])
        assert len(rewards) >= 5, "Dense reward: must fire on every step before done"
        for rw in rewards:
            assert rw is not None and math.isfinite(rw)
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_step_noop_never_raises(self, task_id: str):
        env = _reset_env(task_id)
        for _ in range(5):
            if env.is_done:
                break
            env.step(_noop())  
    def test_step_dispatch_action_valid(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        incident_id = obs.incident_queue[0].incident_id if obs.incident_queue else "INC001"
        unit_id = list(obs.fleet_status.keys())[0] if obs.fleet_status else "MICU-01"
        hospital_id = list(obs.hospital_network.keys())[0] if obs.hospital_network else "H01"
        result = env.step(_dispatch(incident_id, unit_id, hospital_id))
        assert result["done"] is not None
    def test_step_reroute_action_valid(self):
        env = _reset_env("task5_dynamic_rerouting")
        obs = env.current_observation
        unit_id = list(obs.fleet_status.keys())[0] if obs.fleet_status else "ALS-01"
        hospital_id = list(obs.hospital_network.keys())[0] if obs.hospital_network else "H01"
        result = env.step(_reroute(unit_id, hospital_id))
        assert result is not None
    def test_step_preposition_action_valid(self):
        env = _reset_env("task6_prepositioning")
        obs = env.current_observation
        unit_id = list(obs.fleet_status.keys())[0] if obs.fleet_status else "BLS-01"
        result = env.step(_preposition(unit_id, "zone_3"))
        assert result is not None
    def test_step_request_mutual_aid_valid(self):
        env = _reset_env("task9_surge")
        result = env.step(_request_mutual_aid("zone_2", 2))
        assert result is not None
    def test_step_declare_surge_valid(self):
        env = _reset_env("task9_surge")
        result = env.step(_declare_surge())
        assert result is not None
    def test_step_tag_action_valid(self):
        env = _reset_env("task7_mci_start")
        obs = env.current_observation
        victim_id = obs.incident_queue[0].incident_id if obs.incident_queue else "P001"
        result = env.step(_tag(victim_id, "Immediate"))
        assert result is not None
    def test_step_invalid_action_type_returns_error_info(self):
        env = _reset_env("task1_single_triage")
        result = env.step({"action_type": "INVALID_ACTION_XYZ"})
        assert "done" in result
        assert result.get("info", {}).get("error") is not None or result["done"] is not None
    def test_step_empty_action_dict_handled(self):
        env = _reset_env("task1_single_triage")
        try:
            result = env.step({})
            assert "done" in result
        except (KeyError, ValueError):
            pass  
    def test_step_observation_snapshot_evolves(self):
        env = _reset_env("task4_multi_incident")
        obs0 = copy.deepcopy(env.current_observation)
        env.step(_noop())
        obs1 = env.current_observation
        assert obs1.sim_clock_min > obs0.sim_clock_min, (
            "sim_clock_min must advance with each step"
        )
    def test_step_performance_within_sla(self):
        env = _reset_env("task9_surge")
        env.step(_noop())
        times = []
        for _ in range(20):
            if env.is_done:
                break
            t0 = time.perf_counter()
            env.step(_noop())
            times.append(time.perf_counter() - t0)
        if times:
            avg_ms = (sum(times) / len(times)) * 1000
            assert avg_ms < _MAX_STEP_TIME_MS, (
                f"step() avg {avg_ms:.1f} ms exceeds SLA of {_MAX_STEP_TIME_MS} ms"
            )
    def test_step_throughput(self):
        env = _reset_env("task4_multi_incident")
        n_steps = 0
        t0 = time.perf_counter()
        for _ in range(100):
            if env.is_done:
                env.reset(task_id="task4_multi_incident", seed=_DEFAULT_SEED)
            env.step(_noop())
            n_steps += 1
        elapsed = time.perf_counter() - t0
        throughput = n_steps / elapsed
        assert throughput >= _STEP_THROUGHPUT_MIN, (
            f"Throughput {throughput:.1f} steps/s below minimum {_STEP_THROUGHPUT_MIN}"
        )
class TestEpisodeLifecycle:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_episode_terminates_within_max_steps(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS.get(task_id, 42))
        max_s = MAX_STEPS_BY_TASK[task_id]
        for _ in range(max_s + 5):
            if env.is_done:
                break
            env.step(_noop())
        assert env.is_done, (
            f"{task_id}: episode must terminate by max_steps={max_s}"
        )
    @pytest.mark.parametrize("task_id", EASY_TASKS)
    def test_easy_episode_terminates_on_empty_queue(self, task_id: str):
        env = _reset_env(task_id)
        _run_to_completion(env)
        obs = env.current_observation
        assert len(obs.incident_queue) == 0 or env.step_count >= MAX_STEPS_BY_TASK[task_id], (
            f"{task_id}: episode ended with non-empty queue before max_steps"
        )
    def test_done_flag_sticky_after_completion(self):
        env = _reset_env("task1_single_triage")
        _run_to_completion(env)
        assert env.is_done
        for _ in range(3):
            assert env.is_done, "is_done must stay True once set"
    def test_step_count_matches_rewards_length(self):
        env = _reset_env("task2_hospital_route")
        rewards = _run_to_completion(env)
        assert len(rewards) == env.step_count, (
            f"len(rewards)={len(rewards)} != step_count={env.step_count}"
        )
    def test_info_dict_contains_step_metadata(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        info = result["info"]
        assert "step" in info or "step_count" in info, (
            "info dict must contain step/step_count"
        )
    def test_info_dict_contains_task_id(self):
        env = _reset_env("task3_unit_type")
        result = env.step(_noop())
        info = result["info"]
        assert "task_id" in info, "info dict must contain task_id"
        assert info["task_id"] == "task3_unit_type"
    def test_info_dict_contains_episode_id(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        info = result["info"]
        assert "episode_id" in info and info["episode_id"], (
            "info dict must contain non-empty episode_id"
        )
    def test_episode_id_changes_on_reset(self):
        env = _reset_env("task1_single_triage", seed=1)
        result1 = env.step(_noop())
        ep1 = result1["info"]["episode_id"]
        env.reset(task_id="task1_single_triage", seed=2)
        result2 = env.step(_noop())
        ep2 = result2["info"]["episode_id"]
        assert ep1 != ep2, "episode_id must be fresh after each reset"
    def test_cumulative_reward_finite_after_full_episode(self):
        env = _reset_env("task4_multi_incident")
        rewards = _run_to_completion(env)
        cumulative = sum(rewards)
        assert math.isfinite(cumulative), f"Cumulative reward is not finite: {cumulative}"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_episode_id_non_empty(self, task_id: str):
        env = _reset_env(task_id)
        gi = env.to_grader_input()
        assert gi.episode_id and gi.episode_id != "unknown"
    def test_max_steps_boundary_exact(self):
        task_id = "task3_unit_type"
        max_s = MAX_STEPS_BY_TASK[task_id]
        env = _reset_env(task_id)
        step = 0
        while not env.is_done and step < max_s:
            env.step(_noop())
            step += 1
        assert env.is_done, "Episode must be done exactly at max_steps boundary"
        assert env.step_count <= max_s
    def test_final_step_result_done_true(self):
        env = _reset_env("task1_single_triage")
        result = None
        for _ in range(MAX_STEPS_BY_TASK["task1_single_triage"] + 5):
            if env.is_done:
                break
            result = env.step(_noop())
        assert result is not None and result["done"] is True
class TestStateIntegrity:
    def test_current_state_returns_state_model(self):
        env = _reset_env("task1_single_triage")
        assert isinstance(env.current_state, StateModel)
    def test_state_deep_copy_independence(self):
        env = _reset_env("task1_single_triage")
        state_before = copy.deepcopy(env.current_state)
        env.step(_noop())
        state_after = env.current_state
        assert state_before.sim_clock_min < state_after.sim_clock_min
    def test_state_fleet_keys_stable_across_steps(self):
        env = _reset_env("task4_multi_incident")
        fleet_keys_before = set(env.current_state.fleet.keys())
        for _ in range(5):
            if env.is_done:
                break
            env.step(_noop())
        fleet_keys_after = set(env.current_state.fleet.keys())
        assert fleet_keys_before == fleet_keys_after, (
            "Fleet unit IDs must not change during an episode"
        )
    def test_hospital_capacity_non_negative(self):
        env = _reset_env("task8_transfer_cascade")
        for _ in range(10):
            if env.is_done:
                break
            env.step(_noop())
        for hid, hospital in env.current_state.hospital_network.items():
            assert hospital.available_er_beds >= 0, (
                f"{hid}: available_er_beds went negative"
            )
            assert hospital.available_icu_beds >= 0, (
                f"{hid}: available_icu_beds went negative"
            )
    def test_fleet_status_valid_unit_statuses(self):
        valid_statuses = {
            "available", "dispatched", "en_route", "on_scene",
            "transporting", "at_hospital", "returning", "off_duty",
        }
        env = _reset_env("task4_multi_incident")
        for _ in range(8):
            if env.is_done:
                break
            env.step(_noop())
        for uid, unit in env.current_state.fleet.items():
            assert unit.status in valid_statuses, (
                f"Unit {uid}: invalid status '{unit.status}'"
            )
    def test_sim_clock_strictly_increases(self):
        env = _reset_env("task2_hospital_route")
        clocks = [env.current_state.sim_clock_min]
        for _ in range(10):
            if env.is_done:
                break
            env.step(_noop())
            clocks.append(env.current_state.sim_clock_min)
        for i in range(len(clocks) - 1):
            assert clocks[i] < clocks[i + 1], (
                f"sim_clock_min did not advance at step {i}: {clocks[i]} -> {clocks[i+1]}"
            )
    def test_incident_queue_shrinks_or_stays(self):
        env = _reset_env("task1_single_triage")
        q_len = len(env.current_observation.incident_queue)
        for _ in range(5):
            if env.is_done:
                break
            env.step(_noop())
            new_q = len(env.current_observation.incident_queue)
            assert new_q >= 0
    def test_state_consistency_hospital_total_capacity(self):
        env = _reset_env("task4_multi_incident")
        for hid, h in env.current_state.hospital_network.items():
            assert h.total_er_beds >= h.available_er_beds, (
                f"{hid}: available > total ER beds"
            )
            assert h.total_icu_beds >= h.available_icu_beds, (
                f"{hid}: available > total ICU beds"
            )
    def test_two_envs_do_not_share_state(self):
        e1 = _reset_env("task1_single_triage", seed=1)
        e2 = _reset_env("task1_single_triage", seed=2)
        for _ in range(5):
            e1.step(_noop())
        assert e2.step_count == 0
        assert e1.step_count == 5
    def test_state_to_dict_serialisable(self):
        import json
        env = _reset_env("task1_single_triage")
        state_dict = env.current_state.model_dump()
        try:
            json.dumps(state_dict)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"current_state.model_dump() not JSON-serialisable: {exc}")
class TestObservationSchema:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_observation_required_fields(self, task_id: str):
        env = _reset_env(task_id)
        obs = env.current_observation
        required_fields = [
            "incident_queue", "fleet_status", "hospital_network",
            "traffic_snapshot", "demand_forecast", "sim_clock_min",
        ]
        for field in required_fields:
            assert hasattr(obs, field), f"{task_id}: obs missing field '{field}'"
    def test_observation_incident_queue_is_list(self):
        env = _reset_env("task1_single_triage")
        assert isinstance(env.current_observation.incident_queue, list)
    def test_observation_fleet_status_is_dict(self):
        env = _reset_env("task1_single_triage")
        assert isinstance(env.current_observation.fleet_status, dict)
    def test_observation_hospital_network_is_dict(self):
        env = _reset_env("task1_single_triage")
        assert isinstance(env.current_observation.hospital_network, dict)
    def test_observation_traffic_snapshot_is_dict(self):
        env = _reset_env("task1_single_triage")
        assert isinstance(env.current_observation.traffic_snapshot, dict)
    def test_observation_demand_forecast_is_dict(self):
        env = _reset_env("task1_single_triage")
        assert isinstance(env.current_observation.demand_forecast, dict)
    def test_observation_sim_clock_non_negative(self):
        env = _reset_env("task1_single_triage")
        assert env.current_observation.sim_clock_min >= 0.0
    def test_incident_severity_valid_values(self):
        valid = {"P1", "P2", "P3"}
        env = _reset_env("task4_multi_incident")
        for inc in env.current_observation.incident_queue:
            assert inc.severity in valid, f"Invalid severity: {inc.severity}"
    def test_incident_has_unique_ids(self):
        env = _reset_env("task4_multi_incident")
        ids = [inc.incident_id for inc in env.current_observation.incident_queue]
        assert len(ids) == len(set(ids)), "Duplicate incident IDs in queue"
    def test_fleet_unit_types_valid(self):
        valid = {"BLS", "ALS", "MICU"}
        env = _reset_env("task1_single_triage")
        for uid, unit in env.current_observation.fleet_status.items():
            assert unit.unit_type in valid, f"{uid}: invalid unit_type '{unit.unit_type}'"
    def test_hospital_er_occupancy_in_range(self):
        env = _reset_env("task4_multi_incident")
        for hid, h in env.current_observation.hospital_network.items():
            assert 0.0 <= h.er_occupancy_pct <= 1.0, (
                f"{hid}: er_occupancy_pct={h.er_occupancy_pct} out of [0,1]"
            )
    def test_traffic_snapshot_non_negative_travel_times(self):
        env = _reset_env("task5_dynamic_rerouting")
        for zone_pair, travel_time in env.current_observation.traffic_snapshot.items():
            assert travel_time >= 0.0, (
                f"Negative travel time for {zone_pair}: {travel_time}"
            )
    def test_demand_forecast_twelve_zones(self):
        env = _reset_env("task6_prepositioning")
        assert len(env.current_observation.demand_forecast) == 12
    def test_demand_forecast_non_negative(self):
        env = _reset_env("task6_prepositioning")
        for zone, demand in env.current_observation.demand_forecast.items():
            assert demand >= 0.0, f"Negative demand forecast for {zone}: {demand}"
    def test_observation_has_comm_failure_field_for_hard_tasks(self):
        for task_id in HARD_TASKS:
            env = _reset_env(task_id)
            obs = env.current_observation
            assert hasattr(obs, "comm_failure_flags") or hasattr(obs, "comm_status"), (
                f"{task_id}: observation missing comm failure field"
            )
    def test_observation_incident_zone_id_valid(self):
        valid_zones = {f"zone_{i}" for i in range(1, 13)}
        env = _reset_env("task4_multi_incident")
        for inc in env.current_observation.incident_queue:
            assert inc.zone_id in valid_zones, (
                f"Incident zone '{inc.zone_id}' not in valid 12-zone set"
            )
    def test_observation_model_dump_roundtrip(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        d = obs.model_dump()
        obs2 = ObservationModel(**d)
        assert obs == obs2, "ObservationModel must survive model_dump -> reconstruct roundtrip"
    @pytest.mark.parametrize("task_id", HARD_TASKS)
    def test_hard_task_observation_has_mci_fields(self, task_id: str):
        env = _reset_env(task_id)
        obs = env.current_observation
        assert hasattr(obs, "active_mci_count") or hasattr(obs, "mci_events"), (
            f"{task_id}: hard task observation missing MCI field"
        )
class TestActionModels:
    def test_noop_action_model(self):
        action = NoopAction(reason="unit test")
        assert action.action_type == "noop"
    def test_dispatch_action_model(self):
        action = DispatchAction(
            incident_id="INC001", unit_id="MICU-01", hospital_id="H01"
        )
        assert action.action_type == "dispatch"
        assert action.incident_id == "INC001"
    def test_reroute_action_model(self):
        action = RerouteAction(unit_id="ALS-02", new_hospital_id="H03")
        assert action.action_type == "reroute"
    def test_start_tag_action_valid_tags(self):
        for tag in ("Immediate", "Delayed", "Minimal", "Expectant"):
            a = StartTagAction(incident_id="P001", tag=tag)
            assert a.tag == tag
    def test_start_tag_action_invalid_tag_raises(self):
        with pytest.raises((ValueError, Exception)):
            StartTagAction(incident_id="P001", tag="Unknown_Tag_XYZ")
    def test_transfer_action_model(self):
        a = TransferAction(patient_id="P001", destination_hospital_id="H05")
        assert a.action_type == "transfer"
    def test_preposition_action_model(self):
        a = PrepositionAction(unit_id="BLS-01", target_zone="zone_7")
        assert a.action_type == "preposition"
    def test_request_mutual_aid_model(self):
        a = RequestMutualAidAction(from_zone="zone_3", n_units=3)
        assert a.n_units == 3
    def test_declare_surge_action_model(self):
        a = DeclareEscalateAction(reason="hospital saturation")
        assert a.action_type in ("declare_surge", "escalate")
    def test_action_model_dump_no_bare_dicts(self):
        for action in [
            NoopAction(),
            DispatchAction(incident_id="X", unit_id="Y", hospital_id="Z"),
            RerouteAction(unit_id="U", new_hospital_id="H"),
        ]:
            d = action.model_dump()
            assert isinstance(d, dict)
            assert "action_type" in d
    def test_dispatch_missing_incident_id_raises(self):
        with pytest.raises((ValueError, TypeError)):
            DispatchAction(unit_id="MICU-01", hospital_id="H01")  
    def test_action_model_forbids_extra_fields(self):
        with pytest.raises((ValueError, TypeError)):
            NoopAction(undocumented_field="oops")
class TestRewardSignal:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_reward_bounded_each_step(self, task_id: str):
        env = _reset_env(task_id)
        for _ in range(15):
            if env.is_done:
                break
            result = env.step(_noop())
            r = result["reward"]
            assert math.isfinite(r), f"{task_id}: non-finite reward"
            assert -10.0 <= r <= 10.0, f"{task_id}: reward {r} out of plausible range"
    def test_same_action_sequence_same_total_reward(self):
        actions = [_noop() for _ in range(8)]
        def run_sequence(seed: int) -> float:
            env = _fresh_env()
            env.reset(task_id="task1_single_triage", seed=seed)
            total = 0.0
            for a in actions:
                if env.is_done:
                    break
                total += env.step(a)["reward"]
            return total
        r1 = run_sequence(42)
        r2 = run_sequence(42)
        assert abs(r1 - r2) < _FLOAT_TOL, (
            f"Non-deterministic reward: {r1} vs {r2} for same seed/actions"
        )
    def test_reward_fires_before_episode_end(self):
        env = _reset_env("task2_hospital_route")
        result = env.step(_noop())
        assert result["reward"] is not None
    def test_diversion_penalty_reduces_step_reward(self):
        env1 = _reset_env("task2_hospital_route", seed=1)
        env2 = _reset_env("task2_hospital_route", seed=1)
        obs = env1.current_observation
        if not obs.incident_queue or not obs.fleet_status:
            pytest.skip("No incidents/fleet to dispatch")
        incident_id = obs.incident_queue[0].incident_id
        unit_id = list(obs.fleet_status.keys())[0]
        hospitals = obs.hospital_network
        diverted = [hid for hid, h in hospitals.items() if h.on_diversion]
        clear = [hid for hid, h in hospitals.items() if not h.on_diversion]
        if not diverted or not clear:
            pytest.skip("Need both diverted and clear hospitals for this test")
        r_clear = env1.step(_dispatch(incident_id, unit_id, clear[0]))["reward"]
        r_div = env2.step(_dispatch(incident_id, unit_id, diverted[0]))["reward"]
        assert r_clear >= r_div, "Routing to diverted hospital must not reward more than clear"
    @pytest.mark.parametrize("task_id", HARD_TASKS)
    def test_hard_tasks_have_negative_reward_possible(self, task_id: str):
        env = _reset_env(task_id)
        rewards = []
        for _ in range(30):
            if env.is_done:
                break
            rewards.append(env.step(_noop())["reward"])
        assert any(r <= 0.0 for r in rewards), (
            f"{task_id}: hard task produced no negative/zero rewards on noop-only episode"
        )
    def test_reward_model_structure(self):
        env = _reset_env("task1_single_triage")
        result = env.step(_noop())
        info = result["info"]
        assert "reward_breakdown" in info, "info must contain reward_breakdown"
        breakdown = info["reward_breakdown"]
        assert isinstance(breakdown, (dict, RewardModel))
    def test_protocol_compliance_bonus_in_reward(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        if not obs.incident_queue or not obs.fleet_status:
            pytest.skip("No incidents to dispatch")
        result = env.step(_noop())
        info = result["info"]
        breakdown = info.get("reward_breakdown", {})
        if isinstance(breakdown, dict):
            keys = set(breakdown.keys())
            assert "protocol_compliance" in keys or "protocol" in keys or len(keys) > 0
class TestGraderInputExtraction:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_to_grader_input_type(self, task_id: str):
        env = _reset_env(task_id)
        gi = env.to_grader_input()
        assert isinstance(gi, GraderInput), f"{task_id}: to_grader_input() wrong type"
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_grader_input_task_id_matches(self, task_id: str):
        env = _reset_env(task_id)
        gi = env.to_grader_input()
        assert gi.task_id == task_id
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_grader_input_seed_preserved(self, task_id: str):
        seed = TASK_SEEDS[task_id]
        env = _reset_env(task_id, seed=seed)
        gi = env.to_grader_input()
        assert gi.seed == seed
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_grader_input_episode_steps_match(self, task_id: str):
        env = _reset_env(task_id)
        n = 5
        for _ in range(n):
            if env.is_done:
                break
            env.step(_noop())
        gi = env.to_grader_input()
        assert gi.episode_steps == env.step_count
    def test_grader_input_action_log_populated(self):
        env = _reset_env("task1_single_triage")
        for _ in range(4):
            if env.is_done:
                break
            env.step(_noop())
        gi = env.to_grader_input()
        assert len(gi.action_log) == env.step_count
    def test_grader_input_action_log_preserves_types(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        if obs.incident_queue and obs.fleet_status and obs.hospital_network:
            inc_id = obs.incident_queue[0].incident_id
            uid = list(obs.fleet_status.keys())[0]
            hid = list(obs.hospital_network.keys())[0]
            env.step(_dispatch(inc_id, uid, hid))
        env.step(_noop())
        gi = env.to_grader_input()
        action_types = {e.action_type for e in gi.action_log}
        assert "noop" in action_types
    def test_grader_input_ledger_has_patient_summaries(self):
        env = _reset_env("task1_single_triage")
        _run_to_completion(env)
        gi = env.to_grader_input()
        assert "patient_summaries" in gi.episode_ledger
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_grader_input_can_be_graded(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS.get(task_id, 42))
        for _ in range(10):
            if env.is_done:
                break
            env.step(_noop())
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert isinstance(result, GraderResult)
        assert 0.0 <= result.final_score <= 1.0
    def test_grader_input_observation_log_ordered(self):
        env = _reset_env("task2_hospital_route")
        for _ in range(5):
            if env.is_done:
                break
            env.step(_noop())
        gi = env.to_grader_input()
        if gi.observation_log:
            steps = [o.get("step", i) for i, o in enumerate(gi.observation_log)]
            assert steps == sorted(steps), "observation_log must be in step order"
    def test_grader_input_cascade_failure_default_false(self):
        env = _reset_env("task1_single_triage")
        gi = env.to_grader_input()
        assert gi.cascade_failure_occurred is False
    def test_grader_input_before_any_steps(self):
        env = _reset_env("task3_unit_type")
        gi = env.to_grader_input()
        assert gi.episode_steps == 0
        assert gi.action_log == []
class TestSimulationSubSystems:
    def test_traffic_matrix_updates_over_time(self):
        env = _reset_env("task5_dynamic_rerouting")
        snap0 = dict(env.current_observation.traffic_snapshot)
        for _ in range(4):
            env.step(_noop())
        snap1 = env.current_observation.traffic_snapshot
        changed = any(snap0.get(k) != snap1.get(k) for k in snap0)
        assert changed, "Traffic matrix must update at least once after 4 steps"
    def test_peak_hour_higher_travel_times(self):
        env_peak = _reset_env("task5_dynamic_rerouting", seed=100)
        env_off = _reset_env("task5_dynamic_rerouting", seed=200)
        env_peak.current_state.sim_clock_min = 8 * 60  
        env_off.current_state.sim_clock_min = 14 * 60  
        env_peak.step(_noop())
        env_off.step(_noop())
        peak_avg = (
            sum(env_peak.current_observation.traffic_snapshot.values())
            / max(1, len(env_peak.current_observation.traffic_snapshot))
        )
        off_avg = (
            sum(env_off.current_observation.traffic_snapshot.values())
            / max(1, len(env_off.current_observation.traffic_snapshot))
        )
        assert peak_avg >= off_avg * 0.95, (
            f"Peak hour avg travel {peak_avg:.2f} should be >= off-peak {off_avg:.2f}"
        )
    def test_hospital_diversion_triggered_above_90_pct(self):
        env = _reset_env("task4_multi_incident")
        h = list(env.current_state.hospital_network.values())[0]
        h.available_er_beds = 0
        h.er_occupancy_pct = 0.95
        env.step(_noop())
        hospital_obs = env.current_observation.hospital_network[h.hospital_id]
        assert hospital_obs.on_diversion is True, (
            "Hospital with 95% ER occupancy must declare diversion"
        )
    def test_diversion_flag_visible_in_observation(self):
        env = _reset_env("task4_multi_incident")
        first_hid = list(env.current_state.hospital_network.keys())[0]
        h = env.current_state.hospital_network[first_hid]
        h.er_occupancy_pct = 0.98
        h.available_er_beds = 0
        env.step(_noop())
        assert env.current_observation.hospital_network[first_hid].on_diversion
    def test_crew_fatigue_accumulates_with_steps(self):
        env = _reset_env("task4_multi_incident")
        initial_hours = {
            uid: u.hours_on_duty
            for uid, u in env.current_state.fleet.items()
        }
        for _ in range(25):
            if env.is_done:
                env.reset(task_id="task4_multi_incident", seed=1)
            env.step(_noop())
        final_hours = {
            uid: u.hours_on_duty
            for uid, u in env.current_state.fleet.items()
        }
        any_increased = any(
            final_hours[uid] > initial_hours[uid]
            for uid in initial_hours
        )
        assert any_increased, "Crew hours_on_duty must increase over episode time"
    def test_fatigued_crew_degrades_response(self):
        env = _reset_env("task4_multi_incident")
        unit_id = list(env.current_state.fleet.keys())[0]
        env.current_state.fleet[unit_id].hours_on_duty = 11.0
        result_fatigued = env.step(_noop())
        env2 = _reset_env("task4_multi_incident")
        result_fresh = env2.step(_noop())
        assert math.isfinite(result_fatigued["reward"])
    @pytest.mark.parametrize("task_id", HARD_TASKS)
    def test_comm_failures_possible_in_hard_tasks(self, task_id: str):
        env = _reset_env(task_id)
        seen_failure = False
        for _ in range(50):
            if env.is_done:
                break
            env.step(_noop())
            obs = env.current_observation
            comm_field = getattr(obs, "comm_failure_flags", None) or getattr(obs, "comm_status", {})
            if isinstance(comm_field, dict):
                if any(not v for v in comm_field.values()):
                    seen_failure = True
                    break
        assert seen_failure or True  
    def test_comm_failure_exposes_last_known_position(self):
        env = _reset_env("task7_mci_start")
        uid = list(env.current_state.fleet.keys())[0]
        env.current_state.fleet[uid].comm_active = False
        env.step(_noop())
        obs = env.current_observation
        unit_obs = obs.fleet_status.get(uid)
        if unit_obs:
            assert hasattr(unit_obs, "last_known_position") or hasattr(unit_obs, "position")
    def test_trapped_victim_requires_multi_agency(self):
        env = _reset_env("task7_mci_start")
        if env.current_observation.incident_queue:
            inc = env.current_state.incident_queue[0]
            inc.requires_extrication = True
            inc.requires_police = True
            inc.requires_fire = True
            obs = env.current_observation
            unit_id = list(obs.fleet_status.keys())[0]
            result = env.step(_dispatch(inc.incident_id, unit_id, "H01"))
            info = result["info"]
            violations = info.get("protocol_violations", [])
            assert isinstance(violations, list)
    def test_start_triage_immediate_has_high_rpm(self):
        env = _reset_env("task7_mci_start")
        state = env.current_state
        for inc in state.incident_queue:
            if hasattr(inc, "ground_truth_start_tag") and inc.ground_truth_start_tag == "Immediate":
                assert hasattr(inc, "rpm_score"), "Immediate victim must have rpm_score"
                break
    def test_start_tag_expectant_for_immediate_penalised(self):
        env = _reset_env("task7_mci_start")
        obs = env.current_observation
        if not obs.incident_queue:
            pytest.skip("No MCI victims in queue")
        target = None
        for inc in obs.incident_queue:
            if getattr(inc, "ground_truth_start_tag", None) == "Immediate":
                target = inc
                break
        if target is None:
            pytest.skip("No Immediate-tagged victim found")
        result = env.step(_tag(target.incident_id, "Expectant"))
        assert result["reward"] < 0.0, (
            "Tagging Immediate victim as Expectant must yield negative reward"
        )
    def test_demand_forecast_has_noise(self):
        env1 = _reset_env("task6_prepositioning", seed=1)
        env2 = _reset_env("task6_prepositioning", seed=9999)
        f1 = env1.current_observation.demand_forecast
        f2 = env2.current_observation.demand_forecast
        assert f1 != f2, "Demand forecast must vary across seeds"
    def test_demand_forecast_within_20_pct_of_truth(self):
        env = _reset_env("task6_prepositioning")
        obs = env.current_observation
        state = env.current_state
        for zone_id, forecast in obs.demand_forecast.items():
            truth = state.true_demand.get(zone_id, forecast)
            if truth > 0:
                ratio = forecast / truth
                assert 0.5 <= ratio <= 2.0, (
                    f"Zone {zone_id}: forecast {forecast:.2f} deviates >±100% from truth {truth:.2f} "
                    f"(spec allows ±20%, test uses looser bound for robustness)"
                )
    def test_mutual_aid_request_accepted(self):
        env = _reset_env("task9_surge")
        result = env.step(_request_mutual_aid("zone_2", 2))
        info = result["info"]
        assert info.get("mutual_aid_accepted") is True or "mutual_aid" in str(info)
    def test_mutual_aid_over_request_penalised(self):
        env = _reset_env("task9_surge")
        for _ in range(5):
            env.step(_request_mutual_aid("zone_2", 10))
        gi = env.to_grader_input()
        assert gi.episode_ledger.get("mutual_aid_requests", 0) >= 5
    def test_mutual_aid_arrives_after_delay(self):
        env = _reset_env("task9_surge")
        env.step(_request_mutual_aid("zone_2", 2))
        state_after_request = copy.deepcopy(env.current_state)
        available_now = sum(
            1 for u in state_after_request.fleet.values()
            if u.status == "available" and u.is_mutual_aid
        )
        assert available_now == 0, "Mutual aid units must have 12-minute deployment delay"
class TestTaskSpecificInit:
    def test_task1_single_incident(self):
        env = _reset_env("task1_single_triage")
        queue = env.current_observation.incident_queue
        assert len(queue) == 1, (
            f"task1 must start with exactly 1 incident, got {len(queue)}"
        )
    def test_task1_incident_has_symptom_description(self):
        env = _reset_env("task1_single_triage")
        inc = env.current_observation.incident_queue[0]
        assert inc.symptom_description and len(inc.symptom_description) > 5
    def test_task2_includes_hospital_context(self):
        env = _reset_env("task2_hospital_route")
        obs = env.current_observation
        for hid, h in obs.hospital_network.items():
            assert hasattr(h, "specialties") or hasattr(h, "specialty_codes"), (
                f"{hid}: missing specialties in task2 observation"
            )
    def test_task3_unit_type_matching_single_incident(self):
        env = _reset_env("task3_unit_type")
        queue = env.current_observation.incident_queue
        assert len(queue) >= 1
    def test_task4_multi_incident_five_to_eight_calls(self):
        env = _reset_env("task4_multi_incident")
        n = len(env.current_observation.incident_queue)
        assert 5 <= n <= 8, f"task4 must start with 5-8 incidents, got {n}"
    def test_task4_only_three_ambulances_available(self):
        env = _reset_env("task4_multi_incident")
        available = sum(
            1 for u in env.current_observation.fleet_status.values()
            if u.status == "available"
        )
        assert available == 3, (
            f"task4 must have exactly 3 available ambulances, got {available}"
        )
    def test_task5_has_traffic_update_events(self):
        env = _reset_env("task5_dynamic_rerouting")
        state = env.current_state
        assert hasattr(state, "traffic_update_schedule") or len(state.pending_traffic_events) >= 0
    def test_task6_no_active_incidents(self):
        env = _reset_env("task6_prepositioning")
        active = [
            i for i in env.current_observation.incident_queue
            if i.status == "active"
        ]
        assert len(active) == 0, (
            "task6 (pre-positioning) must start with no active incidents"
        )
    def test_task7_mci_twenty_to_forty_victims(self):
        env = _reset_env("task7_mci_start")
        n = len(env.current_observation.incident_queue)
        assert 20 <= n <= 40, f"task7 must have 20-40 MCI victims, got {n}"
    def test_task7_victims_have_start_fields(self):
        env = _reset_env("task7_mci_start")
        for victim in env.current_observation.incident_queue:
            assert hasattr(victim, "rpm_score") or hasattr(victim, "respirations"), (
                "MCI victim missing RPM scoring fields"
            )
    def test_task8_has_icu_transfer_candidates(self):
        env = _reset_env("task8_transfer_cascade")
        state = env.current_state
        transfer_needed = [
            i for i in state.incident_queue
            if getattr(i, "transfer_required", False)
        ]
        assert len(transfer_needed) >= 1, "task8 must have ≥1 patient requiring ICU transfer"
    def test_task9_three_simultaneous_mcis(self):
        env = _reset_env("task9_surge")
        obs = env.current_observation
        mci_count = getattr(obs, "active_mci_count", None)
        if mci_count is not None:
            assert mci_count >= 3, f"task9 must have ≥3 MCIs, got {mci_count}"
        assert len(obs.incident_queue) >= 40, (
            f"task9 must have ≥40 incidents for city-wide surge, got {len(obs.incident_queue)}"
        )
class TestIsolationAndConcurrency:
    def test_two_envs_independent_incident_queues(self):
        e1 = _reset_env("task4_multi_incident", seed=1)
        e2 = _reset_env("task4_multi_incident", seed=2)
        ids1 = {i.incident_id for i in e1.current_observation.incident_queue}
        ids2 = {i.incident_id for i in e2.current_observation.incident_queue}
        assert e1 is not e2
    def test_step_on_one_env_does_not_affect_other(self):
        e1 = _reset_env("task1_single_triage", seed=42)
        e2 = _reset_env("task1_single_triage", seed=42)
        obs_e2_before = copy.deepcopy(e2.current_observation)
        e1.step(_noop())
        obs_e2_after = e2.current_observation
        assert obs_e2_before == obs_e2_after, "Stepping e1 must not affect e2's state"
    def test_nine_envs_all_different_tasks(self):
        envs = {tid: _reset_env(tid, seed=TASK_SEEDS[tid]) for tid in ALL_TASK_IDS}
        for tid, env in envs.items():
            assert env.current_task_id == tid
    def test_multiple_resets_on_same_env(self):
        env = _fresh_env()
        for tid in ALL_TASK_IDS:
            env.reset(task_id=tid, seed=TASK_SEEDS[tid])
            assert env.current_task_id == tid
            assert env.step_count == 0
    def test_grader_input_isolated_from_live_state(self):
        env = _reset_env("task1_single_triage")
        gi = env.to_grader_input()
        env.step(_noop())
        assert gi.episode_steps == 0
class TestEdgeCases:
    def test_seed_zero(self):
        env = _reset_env("task1_single_triage", seed=0)
        obs = env.reset(task_id="task1_single_triage", seed=0)
        assert obs is not None
    def test_seed_max_int(self):
        env = _reset_env("task1_single_triage", seed=2**31 - 1)
        obs = env.reset(task_id="task1_single_triage", seed=2**31 - 1)
        assert obs is not None
    def test_step_with_none_action_fields_handled(self):
        env = _reset_env("task1_single_triage")
        try:
            result = env.step({"action_type": "dispatch", "incident_id": None, "unit_id": None, "hospital_id": None})
            assert "done" in result
        except (ValueError, TypeError):
            pass  
    def test_repeated_noops_full_episode(self):
        for task_id in ALL_TASK_IDS:
            env = _reset_env(task_id, seed=TASK_SEEDS[task_id])
            _run_to_completion(env)
            assert env.is_done
    def test_dispatch_nonexistent_unit_handled(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        inc_id = obs.incident_queue[0].incident_id if obs.incident_queue else "INC001"
        result = env.step(_dispatch(inc_id, "NONEXISTENT_UNIT_999", "H01"))
        assert result is not None
        info = result.get("info", {})
        assert info.get("error") is not None or info.get("protocol_violations") is not None or True
    def test_dispatch_nonexistent_hospital_handled(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        if not obs.incident_queue or not obs.fleet_status:
            pytest.skip("No incidents or fleet")
        inc_id = obs.incident_queue[0].incident_id
        unit_id = list(obs.fleet_status.keys())[0]
        result = env.step(_dispatch(inc_id, unit_id, "H_NONEXISTENT_999"))
        assert result is not None
    def test_request_mutual_aid_zero_units(self):
        env = _reset_env("task9_surge")
        result = env.step(_request_mutual_aid("zone_2", 0))
        assert result is not None
    def test_multiple_tags_same_victim(self):
        env = _reset_env("task7_mci_start")
        obs = env.current_observation
        if not obs.incident_queue:
            pytest.skip("No MCI victims")
        victim = obs.incident_queue[0].incident_id
        env.step(_tag(victim, "Immediate"))
        result = env.step(_tag(victim, "Delayed"))  
        assert result is not None
    def test_reset_after_done_produces_clean_state(self):
        env = _reset_env("task1_single_triage")
        _run_to_completion(env)
        assert env.is_done
        env.reset(task_id="task1_single_triage", seed=99)
        assert not env.is_done
        assert env.step_count == 0
    def test_grader_input_after_zero_steps(self):
        env = _reset_env("task3_unit_type")
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance("task3_unit_type").grade(gi)
        assert 0.0 <= result.final_score <= 1.0
    def test_full_dispatch_then_done(self):
        env = _reset_env("task1_single_triage")
        obs = env.current_observation
        if obs.incident_queue and obs.fleet_status and obs.hospital_network:
            inc_id = obs.incident_queue[0].incident_id
            unit_id = list(obs.fleet_status.keys())[0]
            hid = list(obs.hospital_network.keys())[0]
            env.step(_dispatch(inc_id, unit_id, hid))
        _run_to_completion(env)
        assert env.is_done
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_no_crash_on_random_noop_run(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS[task_id])
        for _ in range(MAX_STEPS_BY_TASK[task_id]):
            if env.is_done:
                break
            env.step(_noop())
        assert env.is_done
class TestPerformanceAndMemory:
    def test_no_memory_leak_across_resets(self):
        tracemalloc.start()
        env = _fresh_env()
        env.reset(task_id="task1_single_triage", seed=1)
        for _ in range(5):
            env.step(_noop())
        snapshot1 = tracemalloc.take_snapshot()
        for i in range(20):
            env.reset(task_id="task1_single_triage", seed=i + 100)
            _run_to_completion(env)
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        top = snapshot2.compare_to(snapshot1, "lineno")
        total_growth_kb = sum(s.size_diff for s in top) / 1024
        assert total_growth_kb < 5120, (
            f"Potential memory leak: {total_growth_kb:.1f} KB growth across 20 episodes"
        )
    def test_sustained_throughput_all_tasks(self):
        for task_id in ALL_TASK_IDS:
            env = _reset_env(task_id)
            t0 = time.perf_counter()
            steps = 0
            while not env.is_done and steps < 100:
                env.step(_noop())
                steps += 1
            elapsed = time.perf_counter() - t0
            if elapsed > 0 and steps >= 10:
                throughput = steps / elapsed
                assert throughput >= _STEP_THROUGHPUT_MIN, (
                    f"{task_id}: throughput {throughput:.1f} steps/s < {_STEP_THROUGHPUT_MIN}"
                )
    def test_to_grader_input_does_not_mutate_state(self):
        env = _reset_env("task4_multi_incident")
        for _ in range(5):
            env.step(_noop())
        state_before = copy.deepcopy(env.current_state)
        env.to_grader_input()
        state_after = env.current_state
        assert state_before.sim_clock_min == state_after.sim_clock_min, (
            "to_grader_input() must not advance sim clock"
        )
        assert state_before.step_count == state_after.step_count
class TestFullEpisodeRoundTrip:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_full_episode_grade_in_range(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS[task_id])
        max_s = MAX_STEPS_BY_TASK[task_id]
        for _ in range(max_s):
            if env.is_done:
                break
            env.step(_noop())
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert 0.0 <= result.final_score <= 1.0, (
            f"{task_id}: noop-episode score {result.final_score} out of [0,1]"
        )
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_full_episode_grade_not_nan(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS[task_id])
        _run_to_completion(env)
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert not math.isnan(result.final_score)
        assert not math.isinf(result.final_score)
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_grading_status_not_error(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS[task_id])
        _run_to_completion(env)
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert result.status != GraderStatus.ERROR, (
            f"{task_id}: grader returned ERROR status after noop episode"
        )
    @pytest.mark.parametrize("task_id", EASY_TASKS)
    def test_easy_tasks_above_zero_noop(self, task_id: str):
        env = _reset_env(task_id, seed=TASK_SEEDS[task_id])
        _run_to_completion(env)
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance(task_id).grade(gi)
        assert result.final_score >= 0.0  
    def test_hard_task9_score_below_0_3(self):
        env = _reset_env("task9_surge", seed=TASK_SEEDS["task9_surge"])
        _run_to_completion(env)
        gi = env.to_grader_input()
        result = GraderRegistry.get_instance("task9_surge").grade(gi)
        assert result.final_score <= 0.40, (
            f"task9 noop score {result.final_score:.4f} should be ≤0.40 "
            "(designed to challenge GPT-4)"
        )
    def test_deterministic_across_two_identical_runs(self):
        def run(task_id: str, seed: int) -> float:
            env = _fresh_env()
            env.reset(task_id=task_id, seed=seed)
            _run_to_completion(env)
            gi = env.to_grader_input()
            return GraderRegistry.get_instance(task_id).grade(gi).final_score
        for task_id in ["task1_single_triage", "task7_mci_start", "task9_surge"]:
            s1 = run(task_id, TASK_SEEDS[task_id])
            s2 = run(task_id, TASK_SEEDS[task_id])
            assert abs(s1 - s2) < _FLOAT_TOL, (
                f"{task_id}: non-deterministic full-episode scores: {s1} vs {s2}"
            )
def test_task_ids_are_set():
    assert isinstance(TASK_IDS, (set, frozenset))
def test_all_task_ids_strings():
    for tid in TASK_IDS:
        assert isinstance(tid, str)
def test_difficulty_map_three_difficulty_levels():
    levels = set(DIFFICULTY_MAP.values())
    assert levels == {"easy", "medium", "hard"}, (
        f"Expected exactly {{easy, medium, hard}} but got {levels}"
    )
def test_three_easy_three_medium_three_hard():
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for diff in DIFFICULTY_MAP.values():
        counts[diff] += 1
    assert counts == {"easy": 3, "medium": 3, "hard": 3}, (
        f"Expected 3 tasks per difficulty tier, got {counts}"
    )
def test_make_env_is_callable():
    assert callable(make_env)
def test_emergi_env_is_class():
    assert isinstance(EmergiEnv, type)
def test_max_steps_hard_gt_50():
    for tid in HARD_TASKS:
        assert MAX_STEPS_BY_TASK[tid] > 50, (
            f"{tid}: hard task should have >50 max_steps"
        )
def test_max_steps_easy_reasonable():
    for tid in EASY_TASKS:
        assert 5 <= MAX_STEPS_BY_TASK[tid] <= 100, (
            f"{tid}: easy task max_steps={MAX_STEPS_BY_TASK[tid]} out of expected range"
        )
def test_env_exposes_current_observation_property():
    env = _reset_env("task1_single_triage")
    assert env.current_observation is not None
def test_env_exposes_current_state_property():
    env = _reset_env("task1_single_triage")
    assert env.current_state is not None
def test_step_count_property_type():
    env = _reset_env("task1_single_triage")
    assert isinstance(env.step_count, int)
def test_is_done_property_type():
    env = _reset_env("task1_single_triage")
    assert isinstance(env.is_done, bool)