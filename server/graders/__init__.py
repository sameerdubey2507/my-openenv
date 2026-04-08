from __future__ import annotations
import importlib
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from server.graders.basegrader import (
    BASE_GRADER_VERSION,
    GRADER_TIMEOUT_SECONDS,
    MAX_PENALTY_FRACTION,
    SCORE_CEILING,
    SCORE_FLOOR,
    TASK_BASELINES,
    TASK_SEEDS,
    ActionLogEntry,
    GraderInput,
    GraderPipeline,
    GraderRegistry,
    GraderResult,
    GraderStatus,
    PenaltyRecord,
    ProtocolRuleChecker,
    ScoreComponent,
    ScoringUtils,
    BaseGrader,
)
logger = logging.getLogger("emergi_env.graders")
_GRADER_MANIFEST: List[Tuple[str, str, str, str]] = [
    (
        "task1_single_triage",
        "server.graders.taskgrader1",
        "Task1Grader",
        "easy",
    ),
    (
        "task2_hospital_route",
        "server.graders.taskgrader2",
        "Task2Grader",
        "easy",
    ),
    (
        "task3_unit_type",
        "server.graders.taskgrader3",
        "Task3Grader",
        "easy",
    ),
    (
        "task4_multi_incident",
        "server.graders.taskgrader4",
        "Task4Grader",
        "medium",
    ),
    (
        "task5_dynamic_rerouting",
        "server.graders.taskgrader5",
        "Task5Grader",
        "medium",
    ),
    (
        "task6_prepositioning",
        "server.graders.taskgrader6",
        "Task6Grader",
        "medium",
    ),
    (
        "task7_mci_start",
        "server.graders.taskgrader7",
        "Task7Grader",
        "hard",
    ),
    (
        "task8_transfer_cascade",
        "server.graders.taskgrader8",
        "Task8Grader",
        "hard",
    ),
    (
        "task9_surge",
        "server.graders.taskgrader9",
        "Task9Grader",
        "hard",
    ),
]
_LOADED_GRADER_CLASSES: Dict[str, Type[BaseGrader]] = {}
_LOAD_ERRORS: Dict[str, str] = {}
_LOAD_WARNINGS: List[str] = []
EASY_TASK_IDS:   List[str] = []
MEDIUM_TASK_IDS: List[str] = []
HARD_TASK_IDS:   List[str] = []
ALL_TASK_IDS:    List[str] = []
DIFFICULTY_MAP:  Dict[str, str] = {}      
BASELINE_MAP:    Dict[str, float] = TASK_BASELINES
SEED_MAP:        Dict[str, int]   = TASK_SEEDS
PACKAGE_VERSION: str = f"2.{len(_GRADER_MANIFEST)}.0"
def _load_all_graders() -> None:
    global EASY_TASK_IDS, MEDIUM_TASK_IDS, HARD_TASK_IDS, ALL_TASK_IDS
    for task_id, module_path, class_name, difficulty in _GRADER_MANIFEST:
        try:
            mod = importlib.import_module(module_path)
            cls: Type[BaseGrader] = getattr(mod, class_name)
            if not (isinstance(cls, type) and issubclass(cls, BaseGrader)):
                raise TypeError(
                    f"{module_path}.{class_name} is not a BaseGrader subclass"
                )
            _LOADED_GRADER_CLASSES[task_id] = cls
            if not GraderRegistry.is_registered(task_id):
                GraderRegistry.register(task_id, cls)
                _LOAD_WARNINGS.append(
                    f"{task_id}: registered from __init__ (missing module-level register)"
                )
            DIFFICULTY_MAP[task_id] = difficulty
            if difficulty == "easy":
                EASY_TASK_IDS.append(task_id)
            elif difficulty == "medium":
                MEDIUM_TASK_IDS.append(task_id)
            elif difficulty == "hard":
                HARD_TASK_IDS.append(task_id)
            logger.debug("Grader loaded: %s ← %s.%s", task_id, module_path, class_name)
        except Exception as exc:  
            _LOAD_ERRORS[task_id] = str(exc)
            logger.error(
                "GRADER LOAD FAILURE — task_id=%s  module=%s  error=%s",
                task_id,
                module_path,
                exc,
                exc_info=True,
            )
    ALL_TASK_IDS = EASY_TASK_IDS + MEDIUM_TASK_IDS + HARD_TASK_IDS
_load_all_graders()
def get_grader(task_id: str) -> BaseGrader:
    if task_id in _LOAD_ERRORS:
        raise KeyError(
            f"Grader for '{task_id}' failed to load: {_LOAD_ERRORS[task_id]}"
        )
    if task_id not in _LOADED_GRADER_CLASSES:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {sorted(_LOADED_GRADER_CLASSES.keys())}"
        )
    return _LOADED_GRADER_CLASSES[task_id]()
def grade_episode(task_id: str, grader_input: GraderInput) -> GraderResult:
    grader = get_grader(task_id)
    return grader.grade(grader_input)
def grade_all(
    inputs: Dict[str, GraderInput],
    *,
    skip_missing: bool = True,
) -> Dict[str, GraderResult]:
    results: Dict[str, GraderResult] = {}
    for task_id in ALL_TASK_IDS:
        if task_id not in inputs:
            if not skip_missing:
                raise RuntimeError(
                    f"grade_all: no input provided for required task '{task_id}'"
                )
            logger.debug("grade_all: skipping %s (no input)", task_id)
            continue
        try:
            results[task_id] = grade_episode(task_id, inputs[task_id])
        except Exception as exc:  
            logger.error("grade_all: error grading %s — %s", task_id, exc, exc_info=True)
            failed = GraderResult(
                task_id=task_id,
                episode_id=inputs[task_id].episode_id,
                seed=inputs[task_id].seed,
                baseline=BASELINE_MAP.get(task_id, 0.0),
                status=GraderStatus.FAILED,
                error_message=str(exc),
                final_score=0.0,
            )
            results[task_id] = failed
    return results
def list_tasks(difficulty: Optional[str] = None) -> List[str]:
    if difficulty is None:
        return list(ALL_TASK_IDS)
    difficulty = difficulty.lower().strip()
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(
            f"difficulty must be 'easy', 'medium', or 'hard'; got '{difficulty}'"
        )
    return [tid for tid in ALL_TASK_IDS if DIFFICULTY_MAP.get(tid) == difficulty]
def aggregate_score(
    results: Dict[str, GraderResult],
    *,
    weights: Optional[Dict[str, float]] = None,
    difficulty_weights: Optional[Dict[str, float]] = None,
) -> float:
    if not results:
        return 0.0
    if weights is None and difficulty_weights is not None:
        tier_counts = {
            d: sum(1 for tid in results if DIFFICULTY_MAP.get(tid) == d)
            for d in ("easy", "medium", "hard")
        }
        weights = {}
        for tid in results:
            tier = DIFFICULTY_MAP.get(tid, "medium")
            tier_w = difficulty_weights.get(tier, 1.0)
            count = tier_counts.get(tier, 1) or 1
            weights[tid] = tier_w / count
    pipeline = GraderPipeline(list(results.keys()))
    return pipeline.aggregate_score(results, weights=weights)
def summary_table(results: Dict[str, GraderResult]) -> str:
    pipeline = GraderPipeline()
    base_table = pipeline.summary_table(results)
    agg = aggregate_score(results)
    beats_count = sum(1 for r in results.values() if r.beats_baseline)
    total = len(results)
    footer_lines = [
        "╠═══════════════════════╩════════╩══════════╩═══════════╩══════════╣",
        f"║  Aggregate score: {agg:.4f}   |  Beats baseline: {beats_count}/{total}          ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
    ]
    return base_table + "\n" + "\n".join(footer_lines)
def health_check() -> Dict[str, Any]:
    loaded_count   = len(_LOADED_GRADER_CLASSES)
    expected_count = len(_GRADER_MANIFEST)
    error_count    = len(_LOAD_ERRORS)
    if error_count == 0:
        status = "healthy"
    elif loaded_count > 0:
        status = "degraded"
    else:
        status = "unhealthy"
    tasks_meta: Dict[str, Dict[str, Any]] = {}
    for task_id, _, _, difficulty in _GRADER_MANIFEST:
        tasks_meta[task_id] = {
            "loaded":     task_id in _LOADED_GRADER_CLASSES,
            "difficulty": difficulty,
            "baseline":   BASELINE_MAP.get(task_id, 0.0),
            "seed":       SEED_MAP.get(task_id, 0),
            "error":      _LOAD_ERRORS.get(task_id),
        }
    return {
        "status":          status,
        "loaded_count":    loaded_count,
        "expected_count":  expected_count,
        "load_errors":     dict(_LOAD_ERRORS),
        "load_warnings":   list(_LOAD_WARNINGS),
        "tasks":           tasks_meta,
        "package_version": PACKAGE_VERSION,
        "grader_version":  BASE_GRADER_VERSION,
    }
def get_task_metadata(task_id: str) -> Dict[str, Any]:
    entry = next(
        ((tid, mp, cn, diff) for tid, mp, cn, diff in _GRADER_MANIFEST if tid == task_id),
        None,
    )
    if entry is None:
        raise KeyError(f"Unknown task_id '{task_id}'")
    tid, module_path, class_name, difficulty = entry
    return {
        "task_id":       tid,
        "module_path":   module_path,
        "class_name":    class_name,
        "difficulty":    difficulty,
        "baseline":      BASELINE_MAP.get(tid, 0.0),
        "seed":          SEED_MAP.get(tid, 0),
        "loaded":        tid in _LOADED_GRADER_CLASSES,
        "error":         _LOAD_ERRORS.get(tid),
    }
def run_smoke_test(*, verbose: bool = False) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    all_passed = True
    for task_id in ALL_TASK_IDS:
        if task_id in _LOAD_ERRORS:
            messages.append(f"SKIP  {task_id}: load error — {_LOAD_ERRORS[task_id]}")
            all_passed = False
            continue
        gi = _build_minimal_grader_input(task_id)
        try:
            grader = get_grader(task_id)
            result = grader.grade(gi)
            score_ok  = 0.0 <= result.final_score <= 1.0
            status_ok = result.status != GraderStatus.FAILED
            if score_ok and status_ok:
                tag = "PASS "
                msg = (
                    f"{tag} {task_id:<26} "
                    f"score={result.final_score:.4f}  "
                    f"status={result.status.value}"
                )
            else:
                tag = "FAIL "
                all_passed = False
                msg = (
                    f"{tag} {task_id:<26} "
                    f"score={result.final_score:.4f} (ok={score_ok})  "
                    f"status={result.status.value} (ok={status_ok})"
                )
            messages.append(msg)
        except Exception as exc:  
            all_passed = False
            messages.append(f"ERROR {task_id}: {exc}")
        if verbose:
            logger.info("smoke_test: %s", messages[-1])
    return all_passed, messages
def _build_minimal_grader_input(task_id: str) -> GraderInput:
    seed = SEED_MAP.get(task_id, 42)
    patient_summary: Dict[str, Any] = {
        "patient_id":                   "smoke-P001",
        "severity":                     "P1",
        "condition_code":               "cardiac_arrest",
        "required_unit_type":           "MICU",
        "required_hospital_specialties": ["cardiology"],
        "zone_type":                    "metro_core",
        "hospital_on_diversion":        False,
        "hospital_er_occupancy_pct":    0.70,
        "hospital_has_specialty":       True,
        "hospital_is_level1":           False,
        "response_time_target_min":     8.0,
        "optimal_travel_time_min":      10.0,
        "final_survival_prob":          0.60,
        "optimal_survival_prob":        0.95,
        "dispatch_latency_min":         None,
        "phase_at_episode_end":         "untreated",
        "weight":                       1.0,
        "weighted_reward":              0.0,
        "start_tag":                    "Immediate",
        "ground_truth_start_tag":       "Immediate",
        "rpm_score":                    11,
        "transfer_required":            False,
        "requires_extrication":         False,
        "requires_police":              False,
        "requires_fire":                False,
        "hospital_id":                  "H01",
        "specialist_hospital_id":       "H02",
        "transfer_window_min":          60.0,
        "transfer_latency_min":         None,
        "destination_icu_utilisation":  0.70,
        "surge_zone":                   "zone_1",
        "mutual_aid_required":          False,
    }
    n_patients = (
        3
        if task_id in ("task4_multi_incident", "task7_mci_start", "task9_surge")
        else 1
    )
    patient_summaries = []
    for i in range(n_patients):
        p = dict(patient_summary)
        p["patient_id"] = f"smoke-P{i+1:03d}"
        patient_summaries.append(p)
    return GraderInput(
        task_id=task_id,
        episode_id="smoke-test-ep-001",
        seed=seed,
        action_log=[],
        episode_ledger={"patient_summaries": patient_summaries},
        observation_log=[],
        episode_steps=10,
        total_patients=n_patients,
        p1_patients=n_patients,
    )
try:
    from server.graders.taskgrader1 import Task1Grader
    from server.graders.taskgrader2 import Task2Grader
    from server.graders.taskgrader3 import Task3Grader
    from server.graders.taskgrader4 import Task4Grader
    from server.graders.taskgrader5 import Task5Grader
    from server.graders.taskgrader6 import Task6Grader
    from server.graders.taskgrader7 import Task7Grader
    from server.graders.taskgrader8 import Task8Grader
    from server.graders.taskgrader9 import Task9Grader
    _DIRECT_IMPORTS_OK = True
except Exception as _imp_err:  
    logger.warning(
        "graders.__init__: direct class imports failed (%s). "
        "get_grader() still works via _LOADED_GRADER_CLASSES.",
        _imp_err,
    )
    _DIRECT_IMPORTS_OK = False
import os as _os
_RUN_SMOKE_AT_IMPORT: bool = (
    _os.getenv("EMERGI_SKIP_SMOKE_TEST", "").lower() not in ("1", "true", "yes")
    and _os.getenv("PYTEST_CURRENT_TEST") is None  
)
if _RUN_SMOKE_AT_IMPORT:
    _t0 = time.monotonic()
    _ok, _msgs = run_smoke_test(verbose=False)
    _elapsed_ms = (time.monotonic() - _t0) * 1000.0
    for _m in _msgs:
        (logger.info if _m.startswith("PASS") else logger.warning)("smoke: %s", _m)
    if not _ok:
        logger.error(
            "EMERGI-ENV graders smoke-test FAILED (%d tasks had errors). "
            "Set EMERGI_SKIP_SMOKE_TEST=1 to suppress at startup.",
            sum(1 for m in _msgs if not m.startswith("PASS")),
        )
    else:
        logger.info(
            "EMERGI-ENV graders smoke-test PASSED — "
            "%d/%d tasks OK in %.1f ms.",
            len(_LOADED_GRADER_CLASSES),
            len(_GRADER_MANIFEST),
            _elapsed_ms,
        )
def _print_banner() -> None:
    loaded   = len(_LOADED_GRADER_CLASSES)
    expected = len(_GRADER_MANIFEST)
    errors   = len(_LOAD_ERRORS)
    status_icon = "✅" if errors == 0 else ("⚠️" if loaded > 0 else "❌")
    banner = (
        f"\n"
        f"╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  EMERGI-ENV  ·  Grader Package  v{PACKAGE_VERSION:<8}                   ║\n"
        f"║  Emergency Medical Intelligence & Resource Governance Env       ║\n"
        f"╠══════════════════════════════════════════════════════════════════╣\n"
        f"║  {status_icon}  Graders loaded : {loaded}/{expected}                                 ║\n"
        f"║  ⚠️  Load errors  : {errors:<2}                                           ║\n"
        f"║  📋 Tasks (easy)  : {len(EASY_TASK_IDS):<2}  │  "
        f"Medium: {len(MEDIUM_TASK_IDS):<2}  │  Hard: {len(HARD_TASK_IDS):<2}          ║\n"
        f"║  🔢 Base version  : {BASE_GRADER_VERSION}                                            ║\n"
        f"╠══════════════════════════════════════════════════════════════════╣\n"
    )
    for task_id, _, _, difficulty in _GRADER_MANIFEST:
        ok   = "✓" if task_id in _LOADED_GRADER_CLASSES else "✗"
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
        base = BASELINE_MAP.get(task_id, 0.0)
        banner += (
            f"║  {ok} {diff_icon} {task_id:<28} baseline={base:.2f}           ║\n"
        )
    banner += "╚══════════════════════════════════════════════════════════════════╝"
    logger.info(banner)
_print_banner()
__all__: List[str] = [
    "PACKAGE_VERSION",
    "BASE_GRADER_VERSION",
    "ALL_TASK_IDS",
    "EASY_TASK_IDS",
    "MEDIUM_TASK_IDS",
    "HARD_TASK_IDS",
    "DIFFICULTY_MAP",
    "BASELINE_MAP",
    "SEED_MAP",
    "SCORE_FLOOR",
    "SCORE_CEILING",
    "TASK_BASELINES",
    "TASK_SEEDS",
    "GRADER_TIMEOUT_SECONDS",
    "MAX_PENALTY_FRACTION",
    "BaseGrader",
    "GraderInput",
    "GraderResult",
    "GraderStatus",
    "GraderRegistry",
    "GraderPipeline",
    "ActionLogEntry",
    "ScoreComponent",
    "PenaltyRecord",
    "ScoringUtils",
    "ProtocolRuleChecker",
    "Task1Grader",
    "Task2Grader",
    "Task3Grader",
    "Task4Grader",
    "Task5Grader",
    "Task6Grader",
    "Task7Grader",
    "Task8Grader",
    "Task9Grader",
    "get_grader",
    "grade_episode",
    "grade_all",
    "list_tasks",
    "aggregate_score",
    "summary_table",
    "health_check",
    "get_task_metadata",
    "run_smoke_test",
]