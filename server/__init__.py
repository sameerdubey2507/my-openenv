from __future__ import annotations
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _PYDANTIC_AVAILABLE = True
except ImportError:  
    _PYDANTIC_AVAILABLE = False
    warnings.warn(
        "pydantic / pydantic-settings not found — ServerSettings unavailable.",
        ImportWarning,
        stacklevel=2,
    )
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  
    _NUMPY_AVAILABLE = False
try:
                                                     
    _STRUCTLOG_AVAILABLE = False
except ImportError:
    _STRUCTLOG_AVAILABLE = False
if TYPE_CHECKING:
    from fastapi import FastAPI
    from server.env import EmergiEnv  
    from server.models.observation import ObservationModel  
    from server.models.action import ActionModel  
    from server.models.state import StateModel  
    from server.models.reward import RewardModel  
__title__: str = "emergi-env"
__description__: str = (
    "Reinforcement-learning environment simulating India's 108/112 "
    "ambulance dispatch system for national AI hackathon (OpenEnv framework)."
)
__version__: str = "1.0.0"
__author__: str = "EMERGI-ENV Team"
__license__: str = "MIT"
__url__: str = "https://huggingface.co/spaces/emergi-env/emergi-env"
def _resolve_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
def _resolve_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
__build_meta__: Dict[str, str] = {
    "version": __version__,
    "git_sha": _resolve_git_sha(),
    "git_branch": _resolve_git_branch(),
    "build_timestamp": datetime.now(tz=timezone.utc).isoformat(),
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "platform": platform.system(),
    "platform_release": platform.release(),
    "hostname": socket.gethostname(),
}
def _configure_stdlib_logging(level: str = "INFO") -> logging.Logger:
    _fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    return logging.getLogger("emergi_env.server")
def _configure_structlog(level: str = "INFO") -> Any:
    pass
_log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
if _STRUCTLOG_AVAILABLE:
    _configure_stdlib_logging(_log_level)  
    logger = _configure_structlog(_log_level)
else:
    _stdlib_logger = _configure_stdlib_logging(_log_level)
    logger = _stdlib_logger  
def _log(level: str, event: str, **kwargs: Any) -> None:
    if _STRUCTLOG_AVAILABLE:
        getattr(logger, level)(event, **kwargs)
    else:
        extras = "  ".join(f"{k}={v!r}" for k, v in kwargs.items())
        msg = f"{event}  {extras}" if extras else event
        getattr(logger, level)(msg)
class RunMode(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
class LogRenderer(str, Enum):
    JSON = "json"
    CONSOLE = "console"
if _PYDANTIC_AVAILABLE:
    class ServerSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="EMERGI_",
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
        host: str = Field(default="0.0.0.0", description="Bind host for uvicorn")
        port: int = Field(default=7860, ge=1024, le=65535, description="Bind port")
        workers: int = Field(default=1, ge=1, le=8, description="Uvicorn worker count")
        reload: bool = Field(default=False, description="Enable hot-reload (dev only)")
        run_mode: RunMode = Field(default=RunMode.PRODUCTION)
        max_steps: int = Field(default=200, ge=10, le=1000)
        num_zones: int = Field(default=12, ge=4, le=50)
        num_hospitals: int = Field(default=8, ge=2, le=30)
        num_ambulances: int = Field(default=15, ge=3, le=100)
        rng_seed: Optional[int] = Field(default=42, description="Global RNG seed; None = random")
        task_id: Optional[int] = Field(default=None, ge=1, le=9, description="Pin to a specific task")
        step_duration_seconds: int = Field(default=60, description="Wall-clock seconds per env step")
        traffic_update_interval: int = Field(default=3, description="Steps between traffic matrix refresh")
        peak_hour_multiplier: float = Field(default=1.45, ge=1.0, le=3.0)
        hospital_diversion_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
        crew_fatigue_hours: float = Field(default=10.0, ge=4.0, le=24.0)
        comms_failure_probability: float = Field(default=0.12, ge=0.0, le=1.0)
        mutual_aid_delay_steps: int = Field(default=12, ge=1, le=60)
        over_request_penalty: float = Field(default=-0.1, le=0.0)
        diversion_penalty: float = Field(default=-0.3, le=0.0)
        wrong_tag_penalty: float = Field(default=-0.5, le=0.0)
        protocol_compliance_bonus: float = Field(default=0.15, ge=0.0)
        api_base_url: str = Field(
            default="https://api-inference.huggingface.co/v1",
            alias="API_BASE_URL",
            description="OpenAI-compatible base URL used by inference.py",
        )
        model_name: str = Field(
            default="meta-llama/Llama-3.1-8B-Instruct",
            alias="MODEL_NAME",
        )
        hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
        inference_timeout: float = Field(default=30.0, ge=5.0, le=300.0)
        inference_max_retries: int = Field(default=3, ge=0, le=10)
        log_level: str = Field(default="INFO")
        log_renderer: LogRenderer = Field(default=LogRenderer.JSON)
        data_dir: Path = Field(
            default=Path(__file__).parent.parent / "data",
            description="Root directory for JSON data files",
        )
        @field_validator("log_level", mode="before")
        @classmethod
        def _validate_log_level(cls, v: str) -> str:
            allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            upper = v.upper()
            if upper not in allowed:
                raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
            return upper
        @field_validator("data_dir", mode="before")
        @classmethod
        def _validate_data_dir(cls, v: Any) -> Path:
            p = Path(v)
            if not p.exists():
                warnings.warn(
                    f"Data directory {p} does not exist — some features may fail.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return p
        @model_validator(mode="after")
        def _cross_validate(self) -> "ServerSettings":
            if self.reload and self.run_mode == RunMode.PRODUCTION:
                warnings.warn(
                    "reload=True is not recommended in production mode.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            if self.workers > 1 and self.rng_seed is not None:
                warnings.warn(
                    "Multi-worker deployments with a fixed rng_seed may "
                    "share RNG state — consider rng_seed=None in that case.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return self
        def as_log_dict(self) -> Dict[str, Any]:
            d = self.model_dump()
            if d.get("hf_token"):
                d["hf_token"] = "***REDACTED***"
            d["data_dir"] = str(d["data_dir"])
            return d
    @lru_cache(maxsize=1)
    def get_settings() -> ServerSettings:
        try:
            settings = ServerSettings()
            _log(
                "info",
                "server_settings_loaded",
                **{k: v for k, v in settings.as_log_dict().items()
                   if k in ("run_mode", "host", "port", "rng_seed",
                             "max_steps", "num_zones", "num_hospitals", "num_ambulances")},
            )
            return settings
        except Exception as exc:
            _log("error", "server_settings_load_failed", error=str(exc))
            raise
else:  
    class ServerSettings:  
        port: int = 7860
        host: str = "0.0.0.0"
        max_steps: int = 200
        rng_seed: int = 42
    def get_settings() -> ServerSettings:  
        return ServerSettings()
_SUBPACKAGE_LOAD_ORDER: Tuple[str, ...] = (
    "server.models",
    "server.medical",
    "server.simulation",
    "server.graders",
    "server.env",
    "server.main",
)
@dataclass(frozen=True)
class OptionalDependency:
    package_name: str
    import_name: str
    available: bool
    version: Optional[str]
    required_by: FrozenSet[str]
    install_hint: str
def _probe_optional(
    package_name: str,
    import_name: Optional[str] = None,
    required_by: FrozenSet[str] = frozenset(),
) -> OptionalDependency:
    _imp = import_name or package_name
    spec = importlib.util.find_spec(_imp)
    available = spec is not None
    version: Optional[str] = None
    if available:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return OptionalDependency(
        package_name=package_name,
        import_name=_imp,
        available=available,
        version=version,
        required_by=required_by,
        install_hint=f"pip install {package_name}",
    )
OPTIONAL_DEPS: Dict[str, OptionalDependency] = {
    dep.package_name: dep
    for dep in [
        _probe_optional("structlog", required_by=frozenset(["server.logging"])),
        _probe_optional("numpy", required_by=frozenset(["server.simulation", "server.graders"])),
        _probe_optional("scipy", required_by=frozenset(["server.medical.survival_curves"])),
        _probe_optional("pandas", required_by=frozenset(["server.simulation.demand_forecaster"])),
        _probe_optional("httpx", required_by=frozenset(["server.main", "inference"])),
        _probe_optional("openai", required_by=frozenset(["inference"])),
        _probe_optional(
            "openenv-core", import_name="openenv",
            required_by=frozenset(["openenv.yaml validation"])
        ),
    ]
}
def check_optional_deps(required_by: Optional[str] = None) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for name, dep in OPTIONAL_DEPS.items():
        if required_by is None or any(required_by in rb for rb in dep.required_by):
            result[name] = dep.available
            if not dep.available:
                _log(
                    "warning",
                    "optional_dep_missing",
                    package=name,
                    install_hint=dep.install_hint,
                    required_by=list(dep.required_by),
                )
    return result
@dataclass
class SubsystemStatus:
    name: str
    healthy: bool
    message: str = ""
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
@dataclass
class PackageHealthReport:
    timestamp_utc: str
    version: str
    build_meta: Dict[str, str]
    overall_healthy: bool
    subsystems: List[SubsystemStatus]
    optional_deps: Dict[str, bool]
    settings_summary: Dict[str, Any]
    python_version: str
    uptime_seconds: float
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "version": self.version,
            "build_meta": self.build_meta,
            "overall_healthy": self.overall_healthy,
            "subsystems": [
                {
                    "name": s.name,
                    "healthy": s.healthy,
                    "message": s.message,
                    "latency_ms": s.latency_ms,
                    "details": s.details,
                }
                for s in self.subsystems
            ],
            "optional_deps": self.optional_deps,
            "settings_summary": self.settings_summary,
            "python_version": self.python_version,
            "uptime_seconds": self.uptime_seconds,
        }
_SERVER_START_TIME: float = time.monotonic()
def get_uptime_seconds() -> float:
    return time.monotonic() - _SERVER_START_TIME
def _probe_data_files() -> SubsystemStatus:
    t0 = time.monotonic()
    try:
        settings = get_settings()
        expected = [
            "city_zones.json",
            "hospital_profiles.json",
            "incident_templates.json",
            "traffic_patterns.json",
            "demand_history.json",
            "survival_params.json",
        ]
        missing = [f for f in expected if not (settings.data_dir / f).exists()]
        elapsed = (time.monotonic() - t0) * 1000
        if missing:
            return SubsystemStatus(
                name="data_files",
                healthy=False,
                message=f"Missing data files: {missing}",
                latency_ms=round(elapsed, 2),
                details={"missing": missing, "data_dir": str(settings.data_dir)},
            )
        return SubsystemStatus(
            name="data_files",
            healthy=True,
            message="All 6 data files present",
            latency_ms=round(elapsed, 2),
            details={"data_dir": str(settings.data_dir), "files_checked": expected},
        )
    except Exception as exc:
        return SubsystemStatus(
            name="data_files",
            healthy=False,
            message=str(exc),
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
def _probe_subpackages() -> SubsystemStatus:
    t0 = time.monotonic()
    failed: List[str] = []
    details: Dict[str, str] = {}
    probes = [
        "server.models",
        "server.medical",
        "server.simulation",
        "server.graders",
    ]
    for mod_name in probes:
        try:
            importlib.import_module(mod_name)
            details[mod_name] = "ok"
        except Exception as exc:
            failed.append(mod_name)
            details[mod_name] = str(exc)
    elapsed = (time.monotonic() - t0) * 1000
    return SubsystemStatus(
        name="subpackages",
        healthy=len(failed) == 0,
        message="All sub-packages importable" if not failed else f"Failed: {failed}",
        latency_ms=round(elapsed, 2),
        details=details,
    )
def _probe_env_class() -> SubsystemStatus:
    t0 = time.monotonic()
    try:
        from server.env import EmergiEnv  
        env = EmergiEnv(task_id=1, seed=0)
        obs = env.reset()
        elapsed = (time.monotonic() - t0) * 1000
        return SubsystemStatus(
            name="emergi_env_class",
            healthy=True,
            message="EmergiEnv.reset() returned observation",
            latency_ms=round(elapsed, 2),
            details={"obs_keys": list(obs.keys()) if isinstance(obs, dict) else str(type(obs))},
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return SubsystemStatus(
            name="emergi_env_class",
            healthy=False,
            message=str(exc),
            latency_ms=round(elapsed, 2),
            details={"traceback": traceback.format_exc()},
        )
def _probe_graders() -> SubsystemStatus:
    t0 = time.monotonic()
    details: Dict[str, Any] = {}
    failed: List[str] = []
    try:
        from server.graders import get_grader  
        for task_id in range(1, 10):
            try:
                grader = get_grader(task_id)
                details[f"task{task_id}"] = type(grader).__name__
            except Exception as exc:
                failed.append(f"task{task_id}")
                details[f"task{task_id}_error"] = str(exc)
    except ImportError as exc:
        failed.append("graders_module")
        details["import_error"] = str(exc)
    elapsed = (time.monotonic() - t0) * 1000
    return SubsystemStatus(
        name="graders",
        healthy=len(failed) == 0,
        message="All 9 graders instantiated" if not failed else f"Failed: {failed}",
        latency_ms=round(elapsed, 2),
        details=details,
    )
def _probe_numpy() -> SubsystemStatus:
    t0 = time.monotonic()
    if not _NUMPY_AVAILABLE:
        return SubsystemStatus(
            name="numpy",
            healthy=False,
            message="numpy not installed",
        )
    try:
        arr = np.random.default_rng(42).random((12, 12))
        assert arr.shape == (12, 12)
        elapsed = (time.monotonic() - t0) * 1000
        return SubsystemStatus(
            name="numpy",
            healthy=True,
            message=f"numpy {np.__version__} functional",
            latency_ms=round(elapsed, 2),
            details={"version": np.__version__},
        )
    except Exception as exc:
        return SubsystemStatus(
            name="numpy",
            healthy=False,
            message=str(exc),
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
def health_check(
    *,
    probe_env: bool = True,
    probe_graders: bool = True,
) -> PackageHealthReport:
    subsystems: List[SubsystemStatus] = [
        _probe_data_files(),
        _probe_subpackages(),
        _probe_numpy(),
    ]
    if probe_env:
        subsystems.append(_probe_env_class())
    if probe_graders:
        subsystems.append(_probe_graders())
    overall = all(s.healthy for s in subsystems)
    try:
        settings = get_settings()
        settings_summary = settings.as_log_dict()
    except Exception:
        settings_summary = {"error": "settings unavailable"}
    report = PackageHealthReport(
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
        version=__version__,
        build_meta=__build_meta__,
        overall_healthy=overall,
        subsystems=subsystems,
        optional_deps=check_optional_deps(),
        settings_summary=settings_summary,
        python_version=platform.python_version(),
        uptime_seconds=round(get_uptime_seconds(), 3),
    )
    _level = "info" if overall else "error"
    _log(
        _level,
        "health_check_complete",
        overall_healthy=overall,
        failed=[s.name for s in subsystems if not s.healthy],
        uptime_seconds=report.uptime_seconds,
    )
    return report
class _LazyProxy:
    _cache: Dict[str, Any] = {}
    _registry: Dict[str, Tuple[str, str]] = {
        "EmergiEnv":         ("server.env",         "EmergiEnv"),
        "create_app":        ("server.main",         "create_app"),
        "ObservationModel":  ("server.models.observation", "ObservationModel"),
        "ActionModel":       ("server.models.action",      "ActionModel"),
        "StateModel":        ("server.models.state",       "StateModel"),
        "RewardModel":       ("server.models.reward",      "RewardModel"),
    }
    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls._cache:
            if name not in cls._registry:
                raise AttributeError(f"server has no attribute {name!r}")
            module_path, obj_name = cls._registry[name]
            try:
                mod = importlib.import_module(module_path)
                cls._cache[name] = getattr(mod, obj_name)
                _log("debug", "lazy_import_resolved", attr=name, module=module_path)
            except Exception as exc:
                raise ImportError(
                    f"Failed to lazily import {name} from {module_path}: {exc}"
                ) from exc
        return cls._cache[name]
def __getattr__(name: str) -> Any:
    return _LazyProxy.get(name)
class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"
@dataclass(frozen=True)
class TaskMeta:
    task_id: int
    name: str
    difficulty: Difficulty
    baseline_score: float
    grader_module: str
    description: str
    max_steps_override: Optional[int] = None
TASK_REGISTRY: Dict[int, TaskMeta] = {
    1: TaskMeta(
        task_id=1,
        name="single_call_triage_dispatch",
        difficulty=Difficulty.EASY,
        baseline_score=0.61,
        grader_module="server.graders.task1_grader",
        description=(
            "Agent reads one symptom description, classifies severity (P1/P2/P3), "
            "selects correct ambulance type, routes to appropriate hospital."
        ),
    ),
    2: TaskMeta(
        task_id=2,
        name="hospital_route_selection",
        difficulty=Difficulty.EASY,
        baseline_score=0.72,
        grader_module="server.graders.task2_grader",
        description=(
            "Agent picks best hospital given patient symptoms, location, and "
            "hospital network state."
        ),
    ),
    3: TaskMeta(
        task_id=3,
        name="unit_type_matching",
        difficulty=Difficulty.EASY,
        baseline_score=0.68,
        grader_module="server.graders.task3_grader",
        description="Pure medical knowledge: agent selects BLS/ALS/MICU for given scenario.",
    ),
    4: TaskMeta(
        task_id=4,
        name="multi_incident_queue",
        difficulty=Difficulty.MEDIUM,
        baseline_score=0.44,
        grader_module="server.graders.task4_grader",
        description=(
            "5-8 simultaneous calls, only 3 ambulances available. "
            "Agent must prioritise under resource scarcity."
        ),
    ),
    5: TaskMeta(
        task_id=5,
        name="dynamic_rerouting",
        difficulty=Difficulty.MEDIUM,
        baseline_score=0.38,
        grader_module="server.graders.task5_grader",
        description=(
            "Mid-episode traffic matrix updates and hospital diversion flags "
            "force agent to reroute active units."
        ),
    ),
    6: TaskMeta(
        task_id=6,
        name="pre_positioning",
        difficulty=Difficulty.MEDIUM,
        baseline_score=0.42,
        grader_module="server.graders.task6_grader",
        description=(
            "No active incidents — agent repositions fleet based on demand "
            "forecast heatmap to minimise future response times."
        ),
    ),
    7: TaskMeta(
        task_id=7,
        name="mass_casualty_incident",
        difficulty=Difficulty.HARD,
        baseline_score=0.29,
        grader_module="server.graders.task7_grader",
        description=(
            "20-40 simultaneous victims; agent applies START triage protocol "
            "(Immediate/Delayed/Minimal/Expectant). Wrong tag I→E = -0.5 penalty."
        ),
        max_steps_override=400,
    ),
    8: TaskMeta(
        task_id=8,
        name="inter_hospital_transfer_cascade",
        difficulty=Difficulty.HARD,
        baseline_score=0.24,
        grader_module="server.graders.task8_grader",
        description=(
            "ICU patients need specialist care; agent manages transfers with "
            "bed utilisation constraints."
        ),
    ),
    9: TaskMeta(
        task_id=9,
        name="city_wide_surge",
        difficulty=Difficulty.HARD,
        baseline_score=0.17,
        grader_module="server.graders.task9_grader",
        description=(
            "3 simultaneous MCIs, hospital saturation, comms failures. "
            "Agent must request mutual aid and declare surge. Designed to score < 0.3 "
            "even for GPT-4."
        ),
        max_steps_override=500,
    ),
}
def get_task_meta(task_id: int) -> TaskMeta:
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_id={task_id!r}. Valid range: 1–{len(TASK_REGISTRY)}."
        )
    return TASK_REGISTRY[task_id]
def list_tasks(difficulty: Optional[Difficulty] = None) -> List[TaskMeta]:
    tasks = list(TASK_REGISTRY.values())
    if difficulty is not None:
        tasks = [t for t in tasks if t.difficulty == difficulty]
    return tasks
class ActionType(str, Enum):
    DISPATCH         = "dispatch"
    REROUTE          = "reroute"
    ESCALATE         = "escalate"
    TAG              = "tag"           
    TRANSFER         = "transfer"      
    REQUEST_MUTUAL_AID = "request_mutual_aid"
    DECLARE_SURGE    = "declare_surge"
    CREW_SWAP        = "crew_swap"
    NOOP             = "noop"
class AmbulanceType(str, Enum):
    BLS  = "BLS"    
    ALS  = "ALS"    
    MICU = "MICU"   
class TriagePriority(str, Enum):
    P1 = "P1"   
    P2 = "P2"   
    P3 = "P3"   
class StartTag(str, Enum):
    IMMEDIATE  = "Immediate"
    DELAYED    = "Delayed"
    MINIMAL    = "Minimal"
    EXPECTANT  = "Expectant"
ZONE_NAMES: Tuple[str, ...] = (
    "Shivajinagar",    
    "Kothrud",         
    "Hadapsar",        
    "Wakad",           
    "Kharadi",         
    "Baner",           
    "Viman_Nagar",     
    "Pimpri",          
    "Chinchwad",       
    "Katraj",          
    "Undri",           
    "Wagholi",         
)
NUM_ZONES: int = len(ZONE_NAMES)
HOSPITAL_NAMES: Tuple[str, ...] = (
    "Ruby_Hall_Clinic",          
    "KEM_Hospital",              
    "Jehangir_Hospital",         
    "Sahyadri_Nagar",            
    "Deenanath_Mangeshkar",      
    "Columbia_Asia",             
    "Aditya_Birla_Memorial",     
    "PCMC_Civil",                
)
NUM_HOSPITALS: int = len(HOSPITAL_NAMES)
GOLDEN_HOUR_MINUTES: int = 60        
PEAK_HOURS: Tuple[Tuple[int, int], ...] = ((8, 10), (17, 19))  
MAX_CREW_DUTY_HOURS: float = 10.0
MUTUAL_AID_DELAY_STEPS: int = 12
COMMS_FAILURE_PROB: float = 0.12
DIVERSION_THRESHOLD: float = 0.90   
class RewardConstants:
    DIVERSION_PENALTY: float        = -0.30
    WRONG_TAG_PENALTY: float        = -0.50   
    OVER_REQUEST_PENALTY: float     = -0.10
    NOOP_UNKNOWN_UNIT_PENALTY: float = -0.20
    PROTOCOL_COMPLIANCE_BONUS: float = +0.15
    CREW_FATIGUE_SCALE: float       = 0.15    
_MIN_PYTHON: Tuple[int, int] = (3, 10)
if sys.version_info < _MIN_PYTHON:  
    raise RuntimeError(
        f"emergi-env requires Python {'.'.join(map(str, _MIN_PYTHON))}+, "
        f"got {platform.python_version()}"
    )
_log(
    "info",
    "emergi_env_server_package_loaded",
    version=__version__,
    git_sha=__build_meta__["git_sha"],
    python=platform.python_version(),
    num_zones=NUM_ZONES,
    num_hospitals=NUM_HOSPITALS,
    tasks_registered=len(TASK_REGISTRY),
    structlog_available=_STRUCTLOG_AVAILABLE,
    numpy_available=_NUMPY_AVAILABLE,
)
__all__: List[str] = [
    "__title__",
    "__version__",
    "__author__",
    "__license__",
    "__build_meta__",
    "ServerSettings",
    "get_settings",
    "RunMode",
    "LogRenderer",
    "logger",
    "PackageHealthReport",
    "SubsystemStatus",
    "health_check",
    "get_uptime_seconds",
    "OptionalDependency",
    "OPTIONAL_DEPS",
    "check_optional_deps",
    "TaskMeta",
    "TASK_REGISTRY",
    "Difficulty",
    "get_task_meta",
    "list_tasks",
    "ActionType",
    "AmbulanceType",
    "TriagePriority",
    "StartTag",
    "ZONE_NAMES",
    "NUM_ZONES",
    "HOSPITAL_NAMES",
    "NUM_HOSPITALS",
    "GOLDEN_HOUR_MINUTES",
    "PEAK_HOURS",
    "MAX_CREW_DUTY_HOURS",
    "MUTUAL_AID_DELAY_STEPS",
    "COMMS_FAILURE_PROB",
    "DIVERSION_THRESHOLD",
    "RewardConstants",
    "_SUBPACKAGE_LOAD_ORDER",
    "EmergiEnv",
    "create_app",
    "ObservationModel",
    "ActionModel",
    "StateModel",
    "RewardModel",
]