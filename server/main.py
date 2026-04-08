from __future__ import annotations
import asyncio
import concurrent.futures
import json
import logging
import os
from pathlib import Path as FilePath
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,

    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from server.env import (
    DIFFICULTY_MAP,
    ENV_VERSION,
    MAX_STEPS_BY_TASK,
    MINUTES_PER_STEP,
    TASK_IDS,
    EmergiEnv,
    make_env,
)
from server.graders.basegrader import (
    TASK_BASELINES,
    TASK_SEEDS,
    GraderInput,
    GraderPipeline,
    GraderRegistry,
    GraderResult,
    GraderStatus,
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("emergi_env.main")
APP_VERSION: str = "2.0.0"
APP_TITLE: str = "EMERGI-ENV: Emergency Medical Intelligence & Resource Governance Environment"
APP_DESCRIPTION: str = (
    "National-level AI hackathon RL environment simulating India's 108/112 "
    "ambulance dispatch system across a Pune-inspired 12-zone city grid with "
    "8 hospitals, three ambulance types, 9 graded tasks, and 10 advanced "
    "simulation features including golden-hour survival curves, START triage, "
    "crew fatigue, hospital diversion, mutual aid, and cascade failure detection."
)
SERVER_PORT: int = int(os.getenv("PORT", "7860"))
MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "50"))
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL", "3600"))
LEADERBOARD_TOP_K: int = 10
MAX_REPLAY_EVENTS: int = 2_000
GRADER_THREAD_POOL_WORKERS: int = 4
_DEFAULT_SESSION_ID: str = "default"
class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task1_single_triage",
        description="One of the 9 task IDs (task1_single_triage … task9_surge).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed. None → use canonical task seed from TASK_SEEDS.",
    )
    session_id: str = Field(
        default=_DEFAULT_SESSION_ID,
        description="Session handle. Use 'default' for single-agent workflows.",
    )
    scenario_override: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional partial scenario dict to merge into task scenario.",
    )
    @field_validator("task_id")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in TASK_IDS:
            raise ValueError(
                f"Unknown task_id '{v}'. Valid: {sorted(TASK_IDS)}"
            )
        return v
class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(
        ...,
        description=(
            "Action dict. Must contain 'action_type'. "
            "See GET /docs/schema for the full action schema."
        ),
    )
    session_id: str = Field(default=_DEFAULT_SESSION_ID)
class GradeRequest(BaseModel):
    session_id: str = Field(default=_DEFAULT_SESSION_ID)
    force_complete: bool = Field(
        default=False,
        description="Grade even if episode is still running (uses current ledger).",
    )
class BatchGradeRequest(BaseModel):
    task_ids: List[str] = Field(
        default_factory=lambda: sorted(TASK_IDS),
        description="Subset of task IDs to grade. Defaults to all 9.",
    )
    seeds: Optional[Dict[str, int]] = Field(
        default=None,
        description="Per-task seed overrides. Missing keys use TASK_SEEDS defaults.",
    )
    max_steps_override: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Override max_steps for all tasks (useful for fast CI runs).",
    )
    @field_validator("task_ids")
    @classmethod
    def validate_tasks(cls, v: List[str]) -> List[str]:
        unknown = set(v) - TASK_IDS
        if unknown:
            raise ValueError(f"Unknown task_ids: {unknown}")
        return v
class LeaderboardEntry(BaseModel):
    rank: int
    episode_id: str
    session_id: str
    score: float
    baseline: float
    beats_baseline: bool
    delta: float
    task_difficulty: str
    total_steps: int
    p1_survival_rate: float
    submitted_at: str
class SessionInfoResponse(BaseModel):
    session_id: str
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    done: bool
    episode_reward: float
    queue_length: int
    active_patients: int
    resolved_patients: int
    created_at: str
    last_active_at: str
    graded: bool
    last_score: Optional[float]
class HealthResponse(BaseModel):
    status: str
    version: str
    env_version: str
    uptime_seconds: float
    active_sessions: int
    total_episodes: int
    total_steps: int
    grader_registry_size: int
    registered_tasks: List[str]
    timestamp: str
class ValidateResponse(BaseModel):
    passed: bool
    checks: Dict[str, bool]
    failures: List[str]
    warnings: List[str]
    timestamp: str
class ProtocolRuleInfo(BaseModel):
    rule_id: str
    description: str
    penalty_per_violation: float
    bonus_per_correct: float
    applies_to_tasks: List[str]
    condition_codes: List[str]
class SchemaResponse(BaseModel):
    action_types: List[str]
    observation_fields: List[str]
    task_valid_actions: Dict[str, List[str]]
    severity_levels: List[str]
    unit_types: List[str]
    triage_tags: List[str]
    zone_ids: List[str]
    hospital_ids: List[str]
class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time: float = time.monotonic()
        self._counters: Dict[str, int] = defaultdict(int)
        self._floats: Dict[str, float] = defaultdict(float)
        self._request_latencies: Deque[float] = deque(maxlen=1_000)
        self._score_history: Deque[Tuple[str, float]] = deque(maxlen=500)
    def inc(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counters[key] += n
    def add_float(self, key: str, val: float) -> None:
        with self._lock:
            self._floats[key] += val
    def record_latency(self, ms: float) -> None:
        with self._lock:
            self._request_latencies.append(ms)
    def record_score(self, task_id: str, score: float) -> None:
        with self._lock:
            self._score_history.append((task_id, score))
    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lats = list(self._request_latencies)
            avg_lat = sum(lats) / len(lats) if lats else 0.0
            p95 = sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0
            per_task: Dict[str, Dict[str, Any]] = defaultdict(
                lambda: {"count": 0, "total_score": 0.0, "best": 0.0}
            )
            for tid, sc in self._score_history:
                per_task[tid]["count"] += 1
                per_task[tid]["total_score"] += sc
                per_task[tid]["best"] = max(per_task[tid]["best"], sc)
            task_stats = {}
            for tid, d in per_task.items():
                task_stats[tid] = {
                    "graded_episodes": d["count"],
                    "avg_score": round(d["total_score"] / d["count"], 4),
                    "best_score": round(d["best"], 4),
                    "baseline": TASK_BASELINES.get(tid, 0.0),
                    "beats_baseline_pct": round(
                        sum(
                            1 for t, s in self._score_history
                            if t == tid and s > TASK_BASELINES.get(t, 0.0)
                        ) / max(d["count"], 1) * 100,
                        1,
                    ),
                }
            return {
                "uptime_seconds": round(self.uptime_seconds, 2),
                "total_requests": self._counters.get("total_requests", 0),
                "total_resets": self._counters.get("total_resets", 0),
                "total_steps": self._counters.get("total_steps", 0),
                "total_graded_episodes": self._counters.get("total_graded", 0),
                "total_cascade_events": self._counters.get("cascade_events", 0),
                "avg_request_latency_ms": round(avg_lat, 2),
                "p95_request_latency_ms": round(p95, 2),
                "action_type_counts": {
                    k.replace("action_", ""): v
                    for k, v in self._counters.items()
                    if k.startswith("action_")
                },
                "per_task_stats": task_stats,
            }
class LeaderboardStore:
    def __init__(self, top_k: int = LEADERBOARD_TOP_K) -> None:
        self._top_k = top_k
        self._lock = threading.Lock()
        self._boards: Dict[str, List[LeaderboardEntry]] = {
            tid: [] for tid in TASK_IDS
        }
    def submit(self, task_id: str, entry: LeaderboardEntry) -> None:
        with self._lock:
            board = self._boards.setdefault(task_id, [])
            board.append(entry)
            board.sort(key=lambda e: e.score, reverse=True)
            self._boards[task_id] = board[: self._top_k]
    def get(self, task_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            if task_id:
                return {
                    task_id: [
                        e.model_dump() for e in self._boards.get(task_id, [])
                    ]
                }
            return {
                tid: [e.model_dump() for e in entries]
                for tid, entries in self._boards.items()
            }
class WebSocketBus:
    def __init__(self) -> None:
        self._clients: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
    async def subscribe(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._clients[session_id].add(ws)
        logger.info("WS subscribe: session=%s  total_clients=%d", session_id, len(self._clients[session_id]))
    async def unsubscribe(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._clients[session_id].discard(ws)
    async def broadcast(self, session_id: str, payload: Dict[str, Any]) -> None:
        msg = json.dumps(payload, default=str)
        dead: Set[WebSocket] = set()
        async with self._lock:
            targets = set(self._clients.get(session_id, set()))
        for ws in targets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._clients[session_id] -= dead
class _SessionRecord:
    __slots__ = (
        "env", "task_id", "episode_id", "created_at", "last_active_at",
        "graded", "last_score", "replay_log",
    )
    def __init__(self, env: EmergiEnv) -> None:
        self.env: EmergiEnv = env
        self.task_id: str = ""
        self.episode_id: str = ""
        self.created_at: float = time.monotonic()
        self.last_active_at: float = time.monotonic()
        self.graded: bool = False
        self.last_score: Optional[float] = None
        self.replay_log: Deque[Dict[str, Any]] = deque(maxlen=MAX_REPLAY_EVENTS)
    def touch(self) -> None:
        self.last_active_at = time.monotonic()
    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_active_at

    def to_grader_input(self):
        from server.graders.basegrader import ActionLogEntry, GraderInput
        action_log = []
        obs_log = []
        for event in self.replay_log:
            if event['event'] == 'step':
                action_log.append(
                    ActionLogEntry(
                        step=event['step'],
                        action_type=event.get('action', {}).get('action_type', 'noop'),
                        action_data=event.get('action', {}),
                    )
                )
            elif event['event'] == 'observation':
                obs_log.append(event.get('data', event))
        
        state = self.env.get_state()
        all_incidents = list(state.get('active_incidents', {}).values()) + state.get('resolved', []) + state.get('incident_queue', [])
        p1_count = sum(1 for inc in all_incidents if inc.get('severity') == 'P1')
        
        fleet = state.get("fleet", [])
        final_pos = {u['unit_id']: u.get('zone', u.get('current_zone', 'Z1')) for u in fleet}
        types_map = {u['unit_id']: u.get('unit_type', 'ALS').upper() for u in fleet}
        
        initial_pos = {}
        if obs_log and "fleet_status" in obs_log[0]:
            for u in obs_log[0]["fleet_status"]:
                initial_pos[u.get('unit_id')] = u.get("zone", u.get("current_zone", "Z1"))
        else:
            initial_pos = {u['unit_id']: "Z1" for u in fleet}

        ledger = {
            'patient_summaries': all_incidents,
            'final_fleet_positions': final_pos,
            'initial_fleet_positions': initial_pos,
            'unit_types': types_map,
            'demand_heatmap': state.get('demand_heatmap', {}),
            'fleet_size': len(fleet),
        }
        
        return GraderInput(
            task_id=self.task_id,
            episode_id=self.episode_id,
            seed=state.get('seed', 42),
            action_log=action_log,
            episode_ledger=ledger,
            observation_log=obs_log,
            episode_steps=self.env.step_count,
            total_patients=len(all_incidents),
            p1_patients=p1_count,
        )
    def to_info(self, session_id: str) -> SessionInfoResponse:
        env = self.env
        return SessionInfoResponse(
            session_id=session_id,
            task_id=self.task_id,
            episode_id=self.episode_id,
            step=env.step_count,
            max_steps=MAX_STEPS_BY_TASK.get(self.task_id, 50),
            done=env.is_done,
            episode_reward=round(env._episode_reward, 4),
            queue_length=env.incident_queue_length,
            active_patients=env.active_patient_count,
            resolved_patients=len(env._resolved),
            created_at=datetime.fromtimestamp(
                self.created_at, tz=timezone.utc
            ).isoformat(),
            last_active_at=datetime.fromtimestamp(
                self.last_active_at, tz=timezone.utc
            ).isoformat(),
            graded=self.graded,
            last_score=self.last_score,
        )
class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionRecord] = {}
        self._lock = asyncio.Lock()
    async def get_or_create(self, session_id: str) -> _SessionRecord:
        async with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= MAX_SESSIONS:
                    self._evict_oldest()
                self._sessions[session_id] = _SessionRecord(make_env())
                logger.info("Session created: %s  (total=%d)", session_id, len(self._sessions))
            rec = self._sessions[session_id]
            rec.touch()
            return rec
    async def get(self, session_id: str) -> Optional[_SessionRecord]:
        async with self._lock:
            rec = self._sessions.get(session_id)
            if rec:
                rec.touch()
            return rec
    async def require(self, session_id: str, context: str = "endpoint") -> _SessionRecord:
        rec = await self.get(session_id)
        if rec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "session_not_found",
                    "session_id": session_id,
                    "hint": f"Call POST /reset first to initialise session '{session_id}'.",
                    "context": context,
                },
            )
        return rec
    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            return existed
    async def list_all(self) -> List[Tuple[str, _SessionRecord]]:
        async with self._lock:
            return list(self._sessions.items())
    async def cleanup_expired(self) -> int:
        async with self._lock:
            expired = [
                sid for sid, rec in self._sessions.items()
                if rec.age_seconds > SESSION_TTL_SECONDS
            ]
            for sid in expired:
                del self._sessions[sid]
                logger.info("Session expired: %s", sid)
            return len(expired)
    def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions, key=lambda s: self._sessions[s].last_active_at)
        del self._sessions[oldest]
        logger.warning("Session evicted (capacity): %s", oldest)
    @property
    def count(self) -> int:
        return len(self._sessions)
_sessions: SessionManager
_metrics: MetricsCollector
_leaderboard: LeaderboardStore
_ws_bus: WebSocketBus
_grader_pool: concurrent.futures.ThreadPoolExecutor
_server_start: float
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sessions, _metrics, _leaderboard, _ws_bus, _grader_pool, _server_start
    _server_start = time.monotonic()
    _sessions = SessionManager()
    _metrics = MetricsCollector()
    _leaderboard = LeaderboardStore()
    _ws_bus = WebSocketBus()
    _grader_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=GRADER_THREAD_POOL_WORKERS,
        thread_name_prefix="grader",
    )

    registered = GraderRegistry.all_task_ids()
    logger.info(
        "EMERGI-ENV server v%s starting on port %d — "
        "env_version=%s  graders_registered=%d  tasks=%s",
        APP_VERSION,
        SERVER_PORT,
        ENV_VERSION,
        len(registered),
        registered,
    )
    await _sessions.get_or_create(_DEFAULT_SESSION_ID)
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(300)
            pruned = await _sessions.cleanup_expired()
            if pruned:
                logger.info("Session cleanup: pruned %d expired sessions", pruned)
    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield  
    cleanup_task.cancel()
    _grader_pool.shutdown(wait=False)
    logger.info("EMERGI-ENV server shutting down.")
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url=None,                                                               
    redoc_url=None,
    openapi_tags=[
        {"name": "openenv",   "description": "Required by OpenEnv Phase-1 validator"},
        {"name": "grading",   "description": "Episode grading and batch evaluation"},
        {"name": "analytics", "description": "Metrics, leaderboard, and replay"},
        {"name": "sessions",  "description": "Session lifecycle management"},
        {"name": "schema",    "description": "Observation, action, and protocol schemas"},
        {"name": "realtime",  "description": "WebSocket real-time event stream"},
    ],
)

_DOCS_HTML = FilePath(__file__).parent.parent / "docs" / "emergi_docs.html"

@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_docs():
    if _DOCS_HTML.exists():
        return HTMLResponse(content=_DOCS_HTML.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/openapi.json")

@app.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
async def custom_redoc():
    _redoc_html = FilePath(__file__).parent.parent / "docs" / "emergi_redoc.html"
    if _redoc_html.exists():
        return HTMLResponse(content=_redoc_html.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/openapi.json")

@app.get("/swagger", response_class=HTMLResponse, include_in_schema=False)
async def swagger_ui():
    from fastapi.openapi.docs import get_swagger_ui_html
    return get_swagger_ui_html(openapi_url="/openapi.json", title="EMERGI-ENV · Swagger UI")

@app.get("/health-dashboard", response_class=HTMLResponse, include_in_schema=False)
async def health_dashboard():
    _h = FilePath(__file__).parent.parent / "docs" / "emergi_health.html"
    if _h.exists():
        return HTMLResponse(content=_h.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/health")

@app.get("/openapi-viewer", response_class=HTMLResponse, include_in_schema=False)
async def openapi_viewer():
    _o = FilePath(__file__).parent.parent / "docs" / "emergi_openapi.html"
    if _o.exists():
        return HTMLResponse(content=_o.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/openapi.json")

@app.get("/emergi_openapi.html", response_class=HTMLResponse, include_in_schema=False)
async def openapi_viewer_direct():
    _o = FilePath(__file__).parent.parent / "docs" / "emergi_openapi.html"
    if _o.exists():
        return HTMLResponse(content=_o.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/openapi.json")

@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    _l = FilePath(__file__).parent.parent / "docs" / "emergi_login.html"
    if _l.exists():
        return HTMLResponse(content=_l.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/automation", response_class=HTMLResponse, include_in_schema=False)
async def automation_dashboard():
    _a = FilePath(__file__).parent.parent / "docs" / "emergi_automation.html"
    if _a.exists():
        return HTMLResponse(content=_a.read_text(encoding="utf-8"))
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

_STATIC_DIR = FilePath(__file__).parent.parent / "static" / "dist"
if _STATIC_DIR.is_dir():
    _ASSETS_DIR = _STATIC_DIR / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_INDEX_PATH = _STATIC_DIR / "index.html"

@app.get("/", include_in_schema=False)
async def root_dashboard():
    if _INDEX_PATH.exists():
        return FileResponse(str(_INDEX_PATH), media_type="text/html")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
app.add_middleware(GZipMiddleware, minimum_size=1_024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Episode-Id", "X-Step", "X-Score", "X-Request-Id", "X-Elapsed-Ms"],
)
@app.middleware("http")
async def _telemetry_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request.state.request_id = req_id
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    _metrics.inc("total_requests")
    _metrics.record_latency(elapsed_ms)
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.2f}"
    return response
@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    _metrics.inc("http_errors")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
@app.exception_handler(Exception)
async def _generic_exc_handler(request: Request, exc: Exception):
    _metrics.inc("internal_errors")
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
def _safe_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(i) for i in obj]
    return obj
def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    d = _safe_dict(obs)
    if not isinstance(d, dict):
        d = {"raw": str(obs)}
    return d
def _build_step_response(
    obs: Any,
    reward: float,
    done: bool,
    info: Dict[str, Any],
    episode_id: str,
    step: int,
) -> Dict[str, Any]:
    return {
        "observation": _obs_to_dict(obs),
        "reward": round(reward, 6),
        "done": done,
        "info": info,
        "episode_id": episode_id,
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
def _run_grade_sync(grader_input: GraderInput) -> GraderResult:
    if not GraderRegistry.is_registered(grader_input.task_id):
        raise HTTPException(
            status_code=422,
            detail=f"No grader registered for task_id '{grader_input.task_id}'",
        )
    grader = GraderRegistry.get_instance(grader_input.task_id)
    return grader.grade(grader_input)
async def _grade_async(grader_input: GraderInput) -> GraderResult:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_grader_pool, _run_grade_sync, grader_input)
@app.post(
    "/reset",
    tags=["openenv"],
    summary="Reset / initialise an episode (OpenEnv required)",
    response_description="Initial observation from the environment",
    status_code=status.HTTP_200_OK,
)
async def reset_episode(body: ResetRequest) -> JSONResponse:
    t0 = time.monotonic()
    rec = await _sessions.get_or_create(body.session_id)
    try:
        obs = rec.env.reset(
            task_id=body.task_id,
            seed=body.seed,
            scenario_override=body.scenario_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rec.task_id = body.task_id
    rec.episode_id = rec.env.episode_id
    rec.graded = False
    rec.last_score = None
    rec.replay_log.clear()
    rec.replay_log.append({
        "event": "reset",
        "task_id": body.task_id,
        "seed": body.seed or TASK_SEEDS.get(body.task_id, 42),
        "episode_id": rec.episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _metrics.inc("total_resets")
    _metrics.inc(f"resets_{body.task_id}")
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    obs_dict = _obs_to_dict(obs)
    response_body = {
        "observation": obs_dict,
        "episode_id": rec.episode_id,
        "task_id": body.task_id,
        "task_difficulty": DIFFICULTY_MAP.get(body.task_id, "unknown"),
        "seed": body.seed or TASK_SEEDS.get(body.task_id, 42),
        "max_steps": MAX_STEPS_BY_TASK.get(body.task_id, 50),
        "baseline_score": TASK_BASELINES.get(body.task_id, 0.0),
        "session_id": body.session_id,
        "reset_latency_ms": round(elapsed_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env_version": ENV_VERSION,
        "schema_version": 7,
    }
    return JSONResponse(
        content=response_body,
        headers={
            "X-Episode-Id": rec.episode_id,
            "X-Task-Id": body.task_id,
            "X-Elapsed-Ms": f"{elapsed_ms:.2f}",
        },
    )
@app.post(
    "/step",
    tags=["openenv"],
    summary="Submit one action and advance simulation (OpenEnv required)",
    status_code=status.HTTP_200_OK,
)
async def step_episode(body: StepRequest) -> JSONResponse:
    rec = await _sessions.require(body.session_id, context="step")
    env = rec.env
    if env.is_done:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "episode_done",
                "episode_id": env.episode_id,
                "hint": "Call POST /reset to start a new episode.",
                "final_reward": round(env._episode_reward, 4),
            },
        )
    t0 = time.monotonic()
    try:
        obs, reward, done, info = env.step(body.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("env.step() raised: %s", exc)
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    rec.replay_log.append({
        "event": "step",
        "step": env.step_count,
        "action": body.action,
        "reward": reward,
        "done": done,
        "sim_clock_min": round(env._sim_clock_min, 2),
        "queue_len": env.incident_queue_length,
        "active_patients": env.active_patient_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _metrics.inc("total_steps")
    action_type = str(body.action.get("action_type", "unknown"))
    _metrics.inc(f"action_{action_type}")
    if info.get("cascade_events"):
        _metrics.inc("cascade_events", len(info["cascade_events"]))
    resp = _build_step_response(obs, reward, done, info, env.episode_id, env.step_count)
    resp["session_id"] = body.session_id
    resp["step_latency_ms"] = round(elapsed_ms, 2)
    resp["sim_clock_min"] = round(env._sim_clock_min, 2)
    await _ws_bus.broadcast(
        body.session_id,
        {
            "event": "step",
            "step": env.step_count,
            "reward": reward,
            "done": done,
            "queue_len": env.incident_queue_length,
            "active": env.active_patient_count,
        },
    )
    return JSONResponse(
        content=resp,
        headers={
            "X-Episode-Id": env.episode_id,
            "X-Step": str(env.step_count),
            "X-Elapsed-Ms": f"{elapsed_ms:.2f}",
        },
    )
@app.get(
    "/state",
    tags=["openenv"],
    summary="Full internal episode state (OpenEnv required)",
    status_code=status.HTTP_200_OK,
)
async def get_state(
    session_id: str = Query(default=_DEFAULT_SESSION_ID, description="Session handle"),
) -> JSONResponse:
    rec = await _sessions.require(session_id, context="get_state")
    t0 = time.monotonic()
    try:
        state = rec.env.get_state()
    except Exception as exc:
        logger.exception("get_state() failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(
        content={
            "state": _safe_dict(state),
            "session_id": session_id,
            "episode_id": rec.episode_id,
            "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

@app.get(
    "/episode/ledger",
    tags=["openenv"],
    summary="Patient outcome and fleet ledger (OpenEnv required)",
    status_code=status.HTTP_200_OK,
)
async def get_episode_ledger(
    session_id: str = Query(default=_DEFAULT_SESSION_ID, description="Session handle"),
) -> JSONResponse:
    rec = await _sessions.require(session_id, context="get_episode_ledger")
    
    grader_input = rec.to_grader_input()
    ledger_data = grader_input.episode_ledger
    
    return JSONResponse(
        status_code=200,
        content={
            "session_id": session_id,
            "episode_id": rec.episode_id,
            **ledger_data
        }
    )
@app.get(
    "/tasks",
    tags=["openenv"],
    summary="Metadata for all 9 tasks (OpenEnv required)",
    status_code=status.HTTP_200_OK,
)
async def list_tasks() -> JSONResponse:
    tasks = []
    difficulty_order = {"easy": 1, "medium": 2, "hard": 3}
    for tid in sorted(TASK_IDS, key=lambda t: (difficulty_order.get(DIFFICULTY_MAP.get(t, "easy"), 0), t)):
        tasks.append({
            "task_id": tid,
            "difficulty": DIFFICULTY_MAP.get(tid, "unknown"),
            "seed": TASK_SEEDS.get(tid, 42),
            "baseline_score": TASK_BASELINES.get(tid, 0.0),
            "max_steps": MAX_STEPS_BY_TASK.get(tid, 50),
            "minutes_per_step": MINUTES_PER_STEP,
            "episode_duration_min": MAX_STEPS_BY_TASK.get(tid, 50) * MINUTES_PER_STEP,
            "grader_registered": GraderRegistry.is_registered(tid),
            "description": _TASK_DESCRIPTIONS.get(tid, ""),
            "valid_actions": _TASK_VALID_ACTIONS.get(tid, []),
            "reward_weights": _TASK_REWARD_WEIGHTS.get(tid, {}),
            "grading_components": _TASK_GRADING_COMPONENTS.get(tid, {}),
        })
    return JSONResponse(
        content={
            "tasks": tasks,
            "total": len(tasks),
            "env_version": ENV_VERSION,
            "schema_version": 7,
            "minutes_per_step": MINUTES_PER_STEP,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.get(
    "/health",
    tags=["openenv"],
    summary="Liveness and readiness probe (OpenEnv required)",
    status_code=status.HTTP_200_OK,
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        env_version=ENV_VERSION,
        uptime_seconds=round(time.monotonic() - _server_start, 2),
        active_sessions=_sessions.count,
        total_episodes=_metrics._counters.get("total_resets", 0),
        total_steps=_metrics._counters.get("total_steps", 0),
        grader_registry_size=len(GraderRegistry.all_task_ids()),
        registered_tasks=GraderRegistry.all_task_ids(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
@app.get(
    "/tasks/{task_id}",
    tags=["openenv"],
    summary="Single-task metadata with full grading schema",
    status_code=status.HTTP_200_OK,
)
async def get_task(
    task_id: str = Path(..., description="One of the 9 task IDs"),
) -> JSONResponse:
    if task_id not in TASK_IDS:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown_task", "task_id": task_id, "valid": sorted(TASK_IDS)},
        )
    return JSONResponse(
        content={
            "task_id": task_id,
            "difficulty": DIFFICULTY_MAP.get(task_id, "unknown"),
            "seed": TASK_SEEDS.get(task_id, 42),
            "baseline_score": TASK_BASELINES.get(task_id, 0.0),
            "max_steps": MAX_STEPS_BY_TASK.get(task_id, 50),
            "minutes_per_step": MINUTES_PER_STEP,
            "episode_duration_min": MAX_STEPS_BY_TASK.get(task_id, 50) * MINUTES_PER_STEP,
            "grader_registered": GraderRegistry.is_registered(task_id),
            "description": _TASK_DESCRIPTIONS.get(task_id, ""),
            "valid_actions": _TASK_VALID_ACTIONS.get(task_id, []),
            "reward_weights": _TASK_REWARD_WEIGHTS.get(task_id, {}),
            "grading_components": _TASK_GRADING_COMPONENTS.get(task_id, {}),
            "features_active": _TASK_FEATURES.get(task_id, []),
            "mci_active": task_id in ("task7_mci_start", "task9_surge"),
            "comm_failures_active": task_id in ("task7_mci_start", "task8_transfer_cascade", "task9_surge"),
            "mutual_aid_active": task_id in ("task7_mci_start", "task9_surge"),
            "cascade_risk": task_id == "task9_surge",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.post(
    "/grade",
    tags=["grading"],
    summary="Grade the current episode for a session",
    status_code=status.HTTP_200_OK,
)
async def grade_episode(body: GradeRequest) -> JSONResponse:
    rec = await _sessions.require(body.session_id, context="grade")
    env = rec.env
    if not env.is_done and not body.force_complete:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "episode_not_done",
                "episode_id": env.episode_id,
                "step": env.step_count,
                "hint": "Set force_complete=true to grade mid-episode, or call more steps.",
            },
        )
    if not rec.task_id:
        raise HTTPException(
            status_code=422,
            detail={"error": "no_episode_initialised", "hint": "Call POST /reset first."},
        )
    t0 = time.monotonic()
    try:
        grader_input: GraderInput = rec.to_grader_input()
        result: GraderResult = await _grade_async(grader_input)
    except Exception as exc:
        logger.exception("Grading failed for session %s: %s", body.session_id, exc)
        raise HTTPException(status_code=500, detail=f"Grading error: {exc}") from exc
    grading_ms = (time.monotonic() - t0) * 1000.0
    rec.graded = True
    rec.last_score = result.final_score
    _metrics.inc("total_graded")
    _metrics.record_score(env.task_id, result.final_score)
    _leaderboard.submit(
        env.task_id,
        LeaderboardEntry(
            rank=0,
            episode_id=env.episode_id,
            session_id=body.session_id,
            score=result.final_score,
            baseline=result.baseline,
            beats_baseline=result.beats_baseline,
            delta=result.score_delta_vs_baseline,
            task_difficulty=result.task_difficulty,
            total_steps=result.episode_steps,
            p1_survival_rate=result.p1_survival_rate,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    await _ws_bus.broadcast(
        body.session_id,
        {
            "event": "graded",
            "final_score": result.final_score,
            "beats_baseline": result.beats_baseline,
            "status": result.status.value,
        },
    )
    result_dict = result.as_dict()
    result_dict["grading_latency_ms"] = round(grading_ms, 2)
    result_dict["session_id"] = body.session_id
    return JSONResponse(
        content=result_dict,
        headers={
            "X-Score": f"{result.final_score:.4f}",
            "X-Episode-Id": env.episode_id,
        },
    )
@app.post(
    "/grade/batch",
    tags=["grading"],
    summary="Run GraderPipeline across multiple tasks (batch evaluation)",
    status_code=status.HTTP_200_OK,
)
async def grade_batch(body: BatchGradeRequest) -> JSONResponse:
    t0 = time.monotonic()
    task_ids = body.task_ids
    seeds = body.seeds or {}
    async def _run_one_task(tid: str) -> Tuple[str, GraderResult]:
        env = make_env()
        seed = seeds.get(tid, TASK_SEEDS.get(tid, 42))
        max_steps = body.max_steps_override or MAX_STEPS_BY_TASK.get(tid, 50)
        try:
            env.reset(task_id=tid, seed=seed)
            for _ in range(max_steps):
                if env.is_done:
                    break
                env.step({"action_type": "noop", "reason": "batch_grade_baseline"})
            gi = env.to_grader_input()
            result = await _grade_async(gi)
        except Exception as exc:
            logger.exception("Batch grade failed for %s: %s", tid, exc)
            result = GraderResult(
                task_id=tid,
                episode_id=env.episode_id if env.episode_id else "unknown",
                seed=seed,
                baseline=TASK_BASELINES.get(tid, 0.0),
                status=GraderStatus.FAILED,
                error_message=str(exc),
                task_difficulty=DIFFICULTY_MAP.get(tid, "unknown"),
            )
            result.finalise()
        return tid, result
    coros = [_run_one_task(tid) for tid in task_ids]
    pairs: List[Tuple[str, GraderResult]] = await asyncio.gather(*coros)
    results = {tid: res for tid, res in pairs}
    pipeline = GraderPipeline(task_ids=task_ids)
    aggregate = pipeline.aggregate_score(results)
    table = pipeline.summary_table(results)
    total_ms = (time.monotonic() - t0) * 1000.0
    _metrics.inc("total_graded", len(results))
    for tid, res in results.items():
        _metrics.record_score(tid, res.final_score)
    return JSONResponse(
        content={
            "aggregate_score": round(aggregate, 4),
            "task_results": {tid: res.as_dict() for tid, res in results.items()},
            "summary_table": table,
            "tasks_evaluated": len(results),
            "total_latency_ms": round(total_ms, 2),
            "all_scores_in_range": all(
                0.0 <= res.final_score <= 1.0 for res in results.values()
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.get(
    "/episode/ledger",
    tags=["analytics"],
    summary="Full episode ledger: patient summaries, fleet, hospital stats",
    status_code=status.HTTP_200_OK,
)
async def get_episode_ledger(
    session_id: str = Query(default=_DEFAULT_SESSION_ID),
) -> JSONResponse:
    rec = await _sessions.require(session_id, context="episode_ledger")
    try:
        ledger = rec.env.get_episode_ledger()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(
        content={
            "ledger": ledger,
            "session_id": session_id,
            "episode_id": rec.episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.get(
    "/metrics",
    tags=["analytics"],
    summary="Server-wide request and episode telemetry",
    status_code=status.HTTP_200_OK,
)
async def get_metrics() -> JSONResponse:
    return JSONResponse(
        content={
            "metrics": _metrics.snapshot(),
            "active_sessions": _sessions.count,
            "registered_graders": GraderRegistry.all_task_ids(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.get(
    "/replay",
    tags=["analytics"],
    summary="NDJSON event-stream replay of the episode action log",
    status_code=status.HTTP_200_OK,
)
async def replay_episode(
    session_id: str = Query(default=_DEFAULT_SESSION_ID),
    format: str = Query(default="ndjson", description="'ndjson' or 'json'"),
) -> Response:
    rec = await _sessions.require(session_id, context="replay")
    events = list(rec.replay_log)
    if format == "json":
        return JSONResponse(
            content={
                "episode_id": rec.episode_id,
                "task_id": rec.task_id,
                "session_id": session_id,
                "event_count": len(events),
                "events": events,
            }
        )
    async def _stream():
        for ev in events:
            yield json.dumps(ev, default=str) + "\n"
    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={"X-Episode-Id": rec.episode_id, "X-Event-Count": str(len(events))},
    )
@app.get(
    "/scenarios",
    tags=["analytics"],
    summary="Seed-locked scenario catalogue for all 9 tasks",
    status_code=status.HTTP_200_OK,
)
async def list_scenarios() -> JSONResponse:
    scenarios = [
        {
            "task_id": "task1_single_triage",
            "seed": 42,
            "name": "Acute STEMI — Single Dispatch",
            "description": "One high-acuity cardiac patient in metro core zone. Agent classifies severity, selects MICU, routes to cath-lab hospital.",
            "n_incidents": 1,
            "n_ambulances": 6,
            "n_hospitals": 8,
            "key_challenges": ["correct_unit_type", "cath_lab_activation", "golden_hour"],
        },
        {
            "task_id": "task2_hospital_route",
            "seed": 43,
            "name": "Stroke Routing with Diversion Flags",
            "description": "P1 stroke patient. Two hospitals on diversion. Agent must route to stroke-unit-equipped hospital within travel budget.",
            "n_incidents": 1,
            "n_ambulances": 6,
            "n_hospitals": 8,
            "key_challenges": ["diversion_awareness", "specialty_matching", "travel_time"],
            "diverted_hospitals": 2,
        },
        {
            "task_id": "task3_unit_type",
            "seed": 44,
            "name": "Unit Type Classification",
            "description": "Pure medical knowledge. Agent selects BLS/ALS/MICU for a random condition family.",
            "n_incidents": 1,
            "n_ambulances": 6,
            "n_hospitals": 8,
            "key_challenges": ["condition_to_unit_mapping", "protocol_compliance"],
        },
        {
            "task_id": "task4_multi_incident",
            "seed": 45,
            "name": "5–8 Simultaneous Calls, 3 Ambulances",
            "description": "Fleet scarcity forces prioritisation. Weighted survival scoring penalises missed P1s heavily.",
            "n_incidents_range": [5, 8],
            "n_ambulances": 3,
            "n_hospitals": 8,
            "key_challenges": ["resource_scarcity", "p1_prioritisation", "weighted_survival"],
        },
        {
            "task_id": "task5_dynamic_rerouting",
            "seed": 46,
            "name": "Mid-Episode Traffic Spike + Diversion Flip",
            "description": "3 active transports are disrupted by a sudden traffic spike (step 5) and a diversion flag flip (step 8).",
            "n_incidents": 3,
            "n_ambulances": 5,
            "n_hospitals": 8,
            "key_challenges": ["reroute_triggers", "traffic_awareness", "time_saved_scoring"],
            "traffic_spike_step": 5,
            "diversion_flip_step": 8,
        },
        {
            "task_id": "task6_prepositioning",
            "seed": 47,
            "name": "Demand Forecast Pre-Positioning",
            "description": "No active incidents. Agent uses 12-hour demand heatmap to pre-position 9 units. Reward is delayed until scheduled incidents arrive.",
            "n_incidents": 0,
            "n_ambulances": 9,
            "n_hospitals": 8,
            "key_challenges": ["demand_forecast", "delayed_reward", "zone_coverage"],
            "scheduled_incidents": 12,
        },
        {
            "task_id": "task7_mci_start",
            "seed": 48,
            "name": "Mass Casualty Incident — START Triage",
            "description": "20–40 victims from bus crash / building collapse. Agent applies START RPM triage, dispatches, spreads across hospitals. Wrong Immediate→Expectant tag: -0.5 penalty.",
            "n_incidents_range": [20, 40],
            "n_ambulances": 13,
            "n_hospitals": 8,
            "key_challenges": ["start_triage_protocol", "hospital_spread", "comm_failures", "wrong_tag_penalty"],
            "comm_failure_prob": 0.12,
        },
        {
            "task_id": "task8_transfer_cascade",
            "seed": 49,
            "name": "ICU Transfer Cascade",
            "description": "5–8 patients at non-specialist hospitals need inter-hospital transfer. ICU beds near capacity. Transfer window: 30–90 min.",
            "n_incidents_range": [5, 8],
            "n_ambulances": 4,
            "n_hospitals": 8,
            "key_challenges": ["specialist_routing", "icu_capacity", "transfer_timing", "comm_failures"],
            "comm_failure_prob": 0.12,
        },
        {
            "task_id": "task9_surge",
            "seed": 50,
            "name": "City-Wide Surge — 3 Simultaneous MCIs",
            "description": "Bus crash + building collapse + factory explosion. All hospitals near capacity. Agent must declare surge, request mutual aid, prevent cascade collapse. Designed to score < 0.30 even for GPT-4.",
            "n_incidents_total": "37–53",
            "n_ambulances": 12,
            "n_hospitals": 8,
            "key_challenges": ["surge_declaration", "mutual_aid", "cascade_prevention", "comm_failures", "hospital_saturation"],
            "comm_failure_prob": 0.12,
            "cascade_failure_risk": "high",
        },
    ]
    return JSONResponse(content={"scenarios": scenarios, "total": len(scenarios)})
@app.get(
    "/leaderboard",
    tags=["analytics"],
    summary="Per-task top-10 score leaderboard",
    status_code=status.HTTP_200_OK,
)
async def get_leaderboard(
    task_id: Optional[str] = Query(
        default=None,
        description="Filter to a single task. Omit for all tasks.",
    ),
) -> JSONResponse:
    if task_id:
        if task_id.isdigit():
            search_prefix = f"task{task_id}"
        else:
            search_prefix = task_id.split("_")[0]

        for t in TASK_IDS:
            if t == task_id or t.startswith(search_prefix + "_"):
                task_id = t
                break

    if task_id and task_id not in TASK_IDS:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown_task", "valid": sorted(TASK_IDS)},
        )
    boards = _leaderboard.get(task_id)
    for tid, entries in boards.items():
        for i, e in enumerate(entries, start=1):
            e["rank"] = i
    return JSONResponse(
        content={
            "leaderboard": boards,
            "queried_task": task_id or "all",
            "top_k": LEADERBOARD_TOP_K,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.get(
    "/sessions",
    tags=["sessions"],
    summary="List all active sessions",
    status_code=status.HTTP_200_OK,
)
async def list_sessions() -> JSONResponse:
    pairs = await _sessions.list_all()
    infos = [rec.to_info(sid).model_dump() for sid, rec in pairs]
    return JSONResponse(
        content={
            "sessions": infos,
            "total": len(infos),
            "max_sessions": MAX_SESSIONS,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.delete(
    "/sessions/{session_id}",
    tags=["sessions"],
    summary="Force-close and delete a session",
    status_code=status.HTTP_200_OK,
)
async def delete_session(
    session_id: str = Path(..., description="Session ID to delete"),
) -> JSONResponse:
    if session_id == _DEFAULT_SESSION_ID:
        raise HTTPException(
            status_code=403,
            detail="Cannot delete the default session. Use POST /reset to reinitialise it.",
        )
    existed = await _sessions.delete(session_id)
    if not existed:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "session_id": session_id},
        )
    return JSONResponse(
        content={
            "deleted": True,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.post(
    "/validate",
    tags=["openenv"],
    summary="Self-validate all OpenEnv contract invariants",
    status_code=status.HTTP_200_OK,
    response_model=ValidateResponse,
)
async def validate_contract() -> ValidateResponse:
    checks: Dict[str, bool] = {}
    failures: List[str] = []
    warnings: List[str] = []
    registered = set(GraderRegistry.all_task_ids())
    checks["all_9_graders_registered"] = registered == TASK_IDS
    if registered != TASK_IDS:
        missing = TASK_IDS - registered
        failures.append(f"Missing graders: {sorted(missing)}")
    try:
        env = make_env()
        obs = env.reset(task_id="task1_single_triage", seed=42)
        obs_dict = _obs_to_dict(obs)
        checks["reset_returns_observation"] = bool(obs_dict)
    except Exception as exc:
        checks["reset_returns_observation"] = False
        failures.append(f"reset() raised: {exc}")
    try:
        obs2, reward, done, info = env.step({"action_type": "noop"})
        checks["step_returns_correct_types"] = (
            isinstance(reward, float)
            and isinstance(done, bool)
            and isinstance(info, dict)
            and obs2 is not None
        )
        checks["reward_in_range"] = -1.0 <= reward <= 1.0
    except Exception as exc:
        checks["step_returns_correct_types"] = False
        checks["reward_in_range"] = False
        failures.append(f"step() raised: {exc}")
    try:
        env2 = make_env()
        env2.reset(task_id="task3_unit_type", seed=44)
        max_s = MAX_STEPS_BY_TASK["task3_unit_type"]
        for _ in range(max_s + 5):
            if env2.is_done:
                break
            env2.step({"action_type": "noop"})
        checks["done_after_max_steps"] = env2.is_done
    except Exception as exc:
        checks["done_after_max_steps"] = False
        failures.append(f"done-after-max-steps check raised: {exc}")
    scores_ok = True
    for tid in sorted(TASK_IDS)[:3]:  
        try:
            env3 = make_env()
            env3.reset(task_id=tid, seed=TASK_SEEDS[tid])
            
            rec = _SessionRecord(env=env3)
            rec.task_id = tid
            rec.episode_id = env3.episode_id
            
            ms = MAX_STEPS_BY_TASK[tid]
            for step_idx in range(ms):
                if env3.is_done:
                    break
                obs, reward, done, info = env3.step({"action_type": "noop"})
                
                rec.replay_log.append({
                    "event": "step",
                    "step": step_idx + 1,
                    "action": {"action_type": "noop"},
                    "reward": reward,
                    "done": done,
                })
                rec.replay_log.append({
                    "event": "observation",
                    "step": step_idx + 1,
                    "data": _obs_to_dict(obs),
                })
            
            gi = rec.to_grader_input()
            res = await _grade_async(gi)
            if not (0.0 <= res.final_score <= 1.0):
                scores_ok = False
                failures.append(f"{tid}: score {res.final_score} outside [0,1]")
        except Exception as exc:
            scores_ok = False
            failures.append(f"{tid}: grader raised {exc}")
    checks["grader_scores_in_range"] = scores_ok
    try:
        env4 = make_env()
        env4.reset(task_id="task1_single_triage", seed=42)
        ep1 = env4.episode_id
        env4.reset(task_id="task2_hospital_route", seed=43)
        ep2 = env4.episode_id
        checks["consecutive_resets_independent"] = ep1 != ep2 and env4.task_id == "task2_hospital_route"
    except Exception as exc:
        checks["consecutive_resets_independent"] = False
        failures.append(f"consecutive resets check raised: {exc}")
    passed = all(checks.values())
    return ValidateResponse(
        passed=passed,
        checks=checks,
        failures=failures,
        warnings=warnings,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
@app.get(
    "/protocol/rules",
    tags=["schema"],
    summary="Full EMS protocol rule catalogue with penalties and bonuses",
    status_code=status.HTTP_200_OK,
)
async def get_protocol_rules() -> JSONResponse:
    rules = [
        ProtocolRuleInfo(
            rule_id="micu_for_stemi",
            description="MICU must be dispatched for all STEMI conditions. BLS/ALS dispatch incurs penalty.",
            penalty_per_violation=-0.020,
            bonus_per_correct=0.0,
            applies_to_tasks=["task1_single_triage", "task4_multi_incident", "task7_mci_start", "task9_surge"],
            condition_codes=["stemi_anterior", "stemi_inferior", "stemi_posterior", "stemi_with_vf_arrest", "stemi_cocaine", "stemi_post_cabg"],
        ),
        ProtocolRuleInfo(
            rule_id="als_for_stroke",
            description="ALS or MICU minimum for all stroke presentations.",
            penalty_per_violation=-0.015,
            bonus_per_correct=0.0,
            applies_to_tasks=["task2_hospital_route", "task4_multi_incident", "task9_surge"],
            condition_codes=["ischemic_stroke", "ischemic_stroke_wake_up", "hemorrhagic_stroke_sah", "paediatric_stroke"],
        ),
        ProtocolRuleInfo(
            rule_id="no_routing_to_diverted",
            description="Dispatching to a hospital on diversion incurs -0.30 per incident.",
            penalty_per_violation=-0.030,
            bonus_per_correct=0.0,
            applies_to_tasks=list(TASK_IDS),
            condition_codes=[],
        ),
        ProtocolRuleInfo(
            rule_id="cath_lab_activation",
            description="Pre-alerting a cath-lab hospital for STEMI earns +0.018 protocol bonus.",
            penalty_per_violation=-0.022,
            bonus_per_correct=0.018,
            applies_to_tasks=["task1_single_triage", "task4_multi_incident", "task7_mci_start", "task9_surge"],
            condition_codes=["stemi_anterior", "stemi_inferior", "cardiac_arrest_vf"],
        ),
        ProtocolRuleInfo(
            rule_id="stroke_unit_prenotification",
            description="Pre-alerting stroke unit for ischemic/haemorrhagic stroke earns +0.015.",
            penalty_per_violation=-0.028,
            bonus_per_correct=0.015,
            applies_to_tasks=["task2_hospital_route", "task4_multi_incident", "task9_surge"],
            condition_codes=["ischemic_stroke", "hemorrhagic_stroke_sah"],
        ),
        ProtocolRuleInfo(
            rule_id="multi_agency_trapped",
            description="Trapped victims require Police + Fire coordination before EMS dispatch.",
            penalty_per_violation=-0.025,
            bonus_per_correct=0.015,
            applies_to_tasks=["task7_mci_start", "task9_surge"],
            condition_codes=["polytrauma_blunt", "mci_rta", "blast_injury", "crush_syndrome"],
        ),
        ProtocolRuleInfo(
            rule_id="level1_trauma_routing",
            description="Major trauma must go to Level-1 trauma centre within 30 min.",
            penalty_per_violation=-0.025,
            bonus_per_correct=0.015,
            applies_to_tasks=["task2_hospital_route", "task4_multi_incident", "task7_mci_start", "task9_surge"],
            condition_codes=["polytrauma_blunt", "polytrauma_penetrating", "severe_tbi", "blast_injury"],
        ),
        ProtocolRuleInfo(
            rule_id="start_triage_mci",
            description="START triage (RPM → Immediate/Delayed/Minimal/Expectant) must be applied in MCI. Wrong Immediate→Expectant tag: -0.50.",
            penalty_per_violation=-0.035,
            bonus_per_correct=0.020,
            applies_to_tasks=["task7_mci_start", "task9_surge"],
            condition_codes=[],
        ),
        ProtocolRuleInfo(
            rule_id="mutual_aid_in_surge",
            description="Surge declaration without mutual aid request incurs -0.05. Correct mutual aid earns +0.02.",
            penalty_per_violation=-0.050,
            bonus_per_correct=0.020,
            applies_to_tasks=["task9_surge"],
            condition_codes=[],
        ),
    ]
    return JSONResponse(
        content={
            "protocol_rules": [r.model_dump() for r in rules],
            "total_rules": len(rules),
            "max_protocol_bonus": 0.15,
            "critical_penalty_tags": {
                "immediate_as_expectant": -0.50,
                "diverted_hospital_routing": -0.30,
                "single_ems_trapped_victim": -0.25,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.get(
    "/docs/schema",
    tags=["schema"],
    summary="Observation and action JSON schema for all 9 tasks",
    status_code=status.HTTP_200_OK,
)
async def get_schema() -> JSONResponse:
    return JSONResponse(
        content={
            "action_types": [
                "dispatch", "reroute", "escalate", "tag", "transfer",
                "request_mutual_aid", "preposition", "crew_swap",
                "declare_surge", "hospital_bypass", "noop",
            ],
            "task_valid_actions": {
                "task1_single_triage":    ["dispatch", "noop"],
                "task2_hospital_route":   ["dispatch", "noop"],
                "task3_unit_type":        ["dispatch", "noop"],
                "task4_multi_incident":   ["dispatch", "escalate", "crew_swap", "noop"],
                "task5_dynamic_rerouting":["dispatch", "reroute", "hospital_bypass", "crew_swap", "noop"],
                "task6_prepositioning":   ["preposition", "noop"],
                "task7_mci_start":        ["dispatch", "tag", "escalate", "crew_swap", "request_mutual_aid", "noop"],
                "task8_transfer_cascade": ["transfer", "reroute", "hospital_bypass", "crew_swap", "noop"],
                "task9_surge":            ["dispatch", "reroute", "escalate", "tag", "transfer", "request_mutual_aid", "preposition", "crew_swap", "declare_surge", "hospital_bypass", "noop"],
            },
            "action_schemas": {
                "dispatch": {
                    "action_type": "dispatch",
                    "incident_id": "<str: INC-XXXX>",
                    "unit_id": "<str: BLS-01 | ALS-02 | MICU-01>",
                    "unit_type": "<str: BLS | ALS | MICU>",
                    "hospital_id": "<str: H01–H08>",
                    "assigned_priority": "<str: P1 | P2 | P3>",
                    "reasoning": "<str: optional>",
                },
                "reroute": {
                    "action_type": "reroute",
                    "unit_id": "<str>",
                    "new_hospital_id": "<str: H01–H08>",
                    "reason": "<str: optional>",
                },
                "tag": {
                    "action_type": "tag",
                    "incident_id": "<str>",
                    "tag": "<str: Immediate | Delayed | Minimal | Expectant>",
                },
                "transfer": {
                    "action_type": "transfer",
                    "patient_id": "<str>",
                    "dest_hospital_id": "<str: H01–H08>",
                },
                "request_mutual_aid": {
                    "action_type": "request_mutual_aid",
                    "n_units": "<int: 1–8>",
                    "from_zone": "<str: zone_1–zone_12>",
                    "unit_type": "<str: BLS | ALS | MICU>",
                },
                "preposition": {
                    "action_type": "preposition",
                    "unit_id": "<str>",
                    "target_zone": "<str: zone_1–zone_12>",
                },
                "crew_swap": {
                    "action_type": "crew_swap",
                    "unit_id": "<str>",
                },
                "declare_surge": {
                    "action_type": "declare_surge",
                    "reason": "<str: optional>",
                },
                "escalate": {
                    "action_type": "escalate",
                    "incident_id": "<str>",
                    "new_unit_type": "<str: ALS | MICU>",
                },
                "hospital_bypass": {
                    "action_type": "hospital_bypass",
                    "unit_id": "<str>",
                    "bypassed_hospital_id": "<str>",
                    "new_hospital_id": "<str>",
                },
                "noop": {
                    "action_type": "noop",
                    "reason": "<str: optional>",
                },
            },
            "observation_top_level_fields": [
                "task_id", "episode_id", "step", "sim_clock_min",
                "incident_queue", "fleet_status", "hospital_network",
                "traffic_snapshot", "demand_forecast",
                "queue_length", "active_patients", "resolved_patients",
                "surge_declared", "mci_active",
            ],
            "incident_fields": [
                "incident_id", "symptom_description", "zone_id", "zone_type",
                "severity_hint", "condition_family", "requires_extrication",
                "requires_police", "requires_fire", "arrival_time_step",
            ],
            "unit_fields": [
                "unit_id", "unit_type", "status", "zone_id",
                "last_known_zone", "last_seen_step", "hours_on_duty",
                "fatigue_flag", "comm_ok", "current_patient_id",
            ],
            "hospital_fields": [
                "hospital_id", "name", "zone_id", "specialties",
                "er_occupancy_pct", "icu_utilisation_pct", "total_beds",
                "icu_beds", "on_diversion", "has_cath_lab",
                "has_helipad", "is_level1_trauma", "pre_alerted",
            ],
            "severity_levels": ["P1", "P2", "P3", "P0"],
            "unit_types": ["BLS", "ALS", "MICU"],
            "triage_tags": ["Immediate", "Delayed", "Minimal", "Expectant"],
            "zone_ids": [f"zone_{i}" for i in range(1, 13)],
            "hospital_ids": [f"H0{i}" if i < 10 else f"H{i}" for i in range(1, 9)],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await _ws_bus.subscribe(session_id, websocket)
    logger.info("WS connected: session=%s", session_id)
    try:
        rec = await _sessions.get(session_id)
        if rec:
            await websocket.send_text(json.dumps({
                "event": "connected",
                "session_id": session_id,
                "task_id": rec.task_id,
                "episode_id": rec.episode_id,
                "step": rec.env.step_count,
                "done": rec.env.is_done,
            }, default=str))
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data.strip() == "ping":
                    await websocket.send_text(json.dumps({"event": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "event": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }))
    except Exception as exc:
        logger.warning("WS error for session %s: %s", session_id, exc)
    finally:
        await _ws_bus.unsubscribe(session_id, websocket)

@app.on_event("startup")
async def startup_event():
    import socket
    logger.info("*" * 60)
    logger.info("   EMERGI-ENV FRONTEND IS NOW LIVE!   ")
    logger.info("   Open the following link in your browser:")
    logger.info("   👉 http://127.0.0.1:7860👈")
    logger.info("*" * 60)

_TASK_DESCRIPTIONS: Dict[str, str] = {
    "task1_single_triage": (
        "Single-call triage and dispatch. Agent reads one cardiac emergency "
        "symptom description, classifies severity (P1/P2/P3), selects the "
        "correct ambulance type (MICU for STEMI), and routes to the "
        "appropriate hospital. Grader: triage class (0.4) + unit type (0.3) "
        "+ hospital match (0.3). Baseline: 0.61."
    ),
    "task2_hospital_route": (
        "Hospital route selection. Agent picks the best hospital given patient "
        "symptoms, location, and network state including diversion flags. "
        "Grader: specialty match (0.5) + capacity check (0.3) + travel time (0.2). "
        "Baseline: 0.72."
    ),
    "task3_unit_type": (
        "Pure medical knowledge test. Agent selects BLS/ALS/MICU for a given "
        "clinical scenario from 5 condition families. Binary exact-match grader "
        "plus protocol compliance bonus. Baseline: 0.68."
    ),
    "task4_multi_incident": (
        "Multi-incident queue. 5–8 simultaneous calls, only 3 ambulances. "
        "Agent must prioritise P1s under resource scarcity. Grader: weighted "
        "survival score across all patients with delay penalties. Baseline: 0.44."
    ),
    "task5_dynamic_rerouting": (
        "Dynamic rerouting. Mid-episode traffic spike (step 5) and hospital "
        "diversion flip (step 8) force rerouting of active units. Grader: "
        "reroute correctness (0.5) + net time saved vs no-reroute (0.5). "
        "Baseline: 0.38."
    ),
    "task6_prepositioning": (
        "Pre-positioning. No active incidents. Agent repositions fleet using "
        "12-hour demand forecast heatmap. Delayed reward only — scored when "
        "scheduled incidents arrive. Baseline: 0.42."
    ),
    "task7_mci_start": (
        "Mass Casualty Incident. 20–40 simultaneous victims from bus crash or "
        "building collapse. Agent applies START triage (RPM → Immediate/Delayed/"
        "Minimal/Expectant), dispatches correctly, and spreads patients across "
        "hospitals. Wrong Immediate→Expectant tag: -0.50 penalty. Comm failures "
        "active (p=0.12/step/unit). Baseline: 0.29."
    ),
    "task8_transfer_cascade": (
        "Inter-hospital transfer cascade. ICU patients at non-specialist hospitals "
        "require specialist care. Agent manages transfers under bed-utilisation "
        "constraints and 30–90 min transfer windows. Comm failures active. "
        "Baseline: 0.24."
    ),
    "task9_surge": (
        "City-wide surge. Three simultaneous MCIs (bus crash + building collapse + "
        "factory explosion), hospital saturation, comm failures. Agent must declare "
        "surge, request mutual aid, and prevent cascade collapse. Designed to score "
        "< 0.30 even for GPT-4. Baseline: 0.17."
    ),
}
_TASK_VALID_ACTIONS: Dict[str, List[str]] = {
    "task1_single_triage":    ["dispatch", "noop"],
    "task2_hospital_route":   ["dispatch", "noop"],
    "task3_unit_type":        ["dispatch", "noop"],
    "task4_multi_incident":   ["dispatch", "escalate", "crew_swap", "noop"],
    "task5_dynamic_rerouting":["dispatch", "reroute", "hospital_bypass", "crew_swap", "noop"],
    "task6_prepositioning":   ["preposition", "noop"],
    "task7_mci_start":        ["dispatch", "tag", "escalate", "crew_swap", "request_mutual_aid", "noop"],
    "task8_transfer_cascade": ["transfer", "reroute", "hospital_bypass", "crew_swap", "noop"],
    "task9_surge":            ["dispatch", "reroute", "escalate", "tag", "transfer", "request_mutual_aid", "preposition", "crew_swap", "declare_surge", "hospital_bypass", "noop"],
}
_TASK_REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "task1_single_triage":    {"triage_accuracy": 0.40, "hospital_routing": 0.30, "response_time": 0.15, "protocol_compliance": 0.10, "survival": 0.05},
    "task2_hospital_route":   {"hospital_routing": 0.50, "triage_accuracy": 0.20, "response_time": 0.15, "diversion_avoidance": 0.10, "protocol_compliance": 0.05},
    "task3_unit_type":        {"triage_accuracy": 0.60, "protocol_compliance": 0.25, "response_time": 0.10, "survival": 0.05},
    "task4_multi_incident":   {"survival": 0.35, "response_time": 0.25, "triage_accuracy": 0.15, "protocol_compliance": 0.10, "hospital_routing": 0.08, "crew_fatigue_management": 0.04, "noop_penalty": 0.03},
    "task5_dynamic_rerouting":{"response_time": 0.30, "hospital_routing": 0.25, "diversion_avoidance": 0.20, "survival": 0.12, "protocol_compliance": 0.08, "crew_fatigue_management": 0.05},
    "task6_prepositioning":   {"preposition_efficiency": 0.60, "response_time": 0.25, "protocol_compliance": 0.10, "survival": 0.05},
    "task7_mci_start":        {"survival": 0.30, "triage_accuracy": 0.25, "response_time": 0.15, "protocol_compliance": 0.12, "hospital_routing": 0.10, "multi_agency_coordination": 0.05, "comms_resilience": 0.03},
    "task8_transfer_cascade": {"transfer_timeliness": 0.40, "hospital_routing": 0.25, "survival": 0.20, "protocol_compliance": 0.10, "comms_resilience": 0.05},
    "task9_surge":            {"survival": 0.25, "surge_management": 0.20, "cascade_avoidance": 0.20, "mutual_aid_efficiency": 0.15, "response_time": 0.10, "protocol_compliance": 0.05, "comms_resilience": 0.05},
}
_TASK_GRADING_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "task1_single_triage":    {"triage_class": 0.4, "unit_type_match": 0.3, "hospital_match": 0.3},
    "task2_hospital_route":   {"specialty_match": 0.5, "capacity_check": 0.3, "travel_time": 0.2},
    "task3_unit_type":        {"unit_exact_match": 1.0},
    "task4_multi_incident":   {"weighted_survival_score": 0.6, "delay_penalties": 0.2, "p1_treatment_rate": 0.2},
    "task5_dynamic_rerouting":{"reroute_correctness": 0.5, "net_time_saved": 0.5},
    "task6_prepositioning":   {"avg_response_time_vs_random": 0.7, "zone_coverage_score": 0.3},
    "task7_mci_start":        {"start_accuracy": 0.4, "response_score": 0.35, "hospital_spread": 0.25},
    "task8_transfer_cascade": {"transfer_appropriateness": 0.4, "timing_score": 0.3, "utilisation_efficiency": 0.3},
    "task9_surge":            {"system_survival": 0.4, "cascade_avoidance": 0.3, "mutual_aid_score": 0.3},
}
_TASK_FEATURES: Dict[str, List[str]] = {
    "task1_single_triage":    ["golden_hour_curves", "protocol_compliance"],
    "task2_hospital_route":   ["golden_hour_curves", "hospital_diversion", "protocol_compliance"],
    "task3_unit_type":        ["protocol_compliance"],
    "task4_multi_incident":   ["golden_hour_curves", "crew_fatigue", "protocol_compliance"],
    "task5_dynamic_rerouting":["golden_hour_curves", "time_varying_traffic", "hospital_diversion", "crew_fatigue"],
    "task6_prepositioning":   ["demand_forecast", "time_varying_traffic"],
    "task7_mci_start":        ["start_triage", "golden_hour_curves", "comm_failures", "multi_agency", "mutual_aid", "protocol_compliance"],
    "task8_transfer_cascade": ["golden_hour_curves", "comm_failures", "crew_fatigue", "hospital_diversion", "protocol_compliance"],
    "task9_surge":            ["start_triage", "golden_hour_curves", "comm_failures", "multi_agency", "mutual_aid", "hospital_diversion", "cascade_failure", "demand_forecast", "crew_fatigue", "protocol_compliance"],
}
if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
        access_log=True,
        reload=False,
        workers=1,                  
        loop="asyncio",
        timeout_keep_alive=75,
        limit_concurrency=100,
    )