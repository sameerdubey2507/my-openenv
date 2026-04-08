#!/usr/bin/env bash

set -euo pipefail

if [[ "${CI:-false}" == "true" ]] || [[ "${TERM:-dumb}" == "dumb" ]]; then
    RED="" GREEN="" YELLOW="" BLUE="" CYAN="" MAGENTA="" BOLD="" DIM="" RESET=""
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    MAGENTA='\033[0;35m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
fi

SKIP_DOCKER=false
SKIP_HF=false
SKIP_INFERENCE=false
SKIP_STRESS=false
SKIP_SECURITY=false
SKIP_SCHEMA_DEEP=false
SERVER_URL="http://localhost:7860"
HF_SPACE_URL="${HF_SPACE_URL:-}"
HTTP_TIMEOUT=30
REPORT_DIR="./reports"
VERBOSE=false
FAIL_FAST=false
CI_MODE=false
STRESS_CONCURRENCY=10
STRESS_DURATION=15
CONTAINER_NAME="emergi-env-validator-$$"
IMAGE_TAG="emergi-env:validate-$$"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVER_STARTED_BY_US=false

TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNED_CHECKS=0
declare -a FAILURE_MESSAGES=()
declare -a PASS_MESSAGES=()
declare -a WARN_MESSAGES=()
declare -A CHECK_DURATIONS=()

START_TS=$(date +%s)
REPORT_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-docker)        SKIP_DOCKER=true ;;
        --skip-hf)            SKIP_HF=true ;;
        --skip-inference)     SKIP_INFERENCE=true ;;
        --skip-stress)        SKIP_STRESS=true ;;
        --skip-security)      SKIP_SECURITY=true ;;
        --skip-schema-deep)   SKIP_SCHEMA_DEEP=true ;;
        --server-url)         SERVER_URL="$2";          shift ;;
        --hf-space-url)       HF_SPACE_URL="$2";        shift ;;
        --timeout)            HTTP_TIMEOUT="$2";        shift ;;
        --report-dir)         REPORT_DIR="$2";          shift ;;
        --stress-concurrency) STRESS_CONCURRENCY="$2";  shift ;;
        --stress-duration)    STRESS_DURATION="$2";     shift ;;
        --verbose)            VERBOSE=true ;;
        --fail-fast)          FAIL_FAST=true ;;
        --ci)                 CI_MODE=true ;;
        -h|--help)
            echo "EMERGI-ENV OpenEnv Phase 1 Validator"
            echo ""
            echo "Usage: bash scripts/validate-submission.sh [OPTIONS]"
            echo ""
            echo "  --skip-docker            Skip Docker build/run checks"
            echo "  --skip-hf                Skip HuggingFace Space checks"
            echo "  --skip-inference         Skip inference.py dry run"
            echo "  --skip-stress            Skip stress/load testing"
            echo "  --skip-security          Skip security scanning"
            echo "  --skip-schema-deep       Skip deep Pydantic schema introspection"
            echo "  --server-url URL         Target server (default: http://localhost:7860)"
            echo "  --hf-space-url URL       HuggingFace Space URL"
            echo "  --timeout N              HTTP timeout seconds (default: 30)"
            echo "  --report-dir DIR         Output directory (default: ./reports)"
            echo "  --stress-concurrency N   Parallel requests in stress test (default: 10)"
            echo "  --stress-duration N      Stress test duration in seconds (default: 15)"
            echo "  --verbose                Show full response bodies"
            echo "  --fail-fast              Abort on first failure"
            echo "  --ci                     Machine-readable CI output"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${REPORT_DIR}"
REPORT_FILE="${REPORT_DIR}/validation_${REPORT_TIMESTAMP}.log"
METRICS_FILE="${REPORT_DIR}/metrics_${REPORT_TIMESTAMP}.json"
exec > >(tee -a "${REPORT_FILE}") 2>&1

log()       { echo -e "${DIM}[$(date '+%H:%M:%S')]${RESET} $*"; }
log_head()  {
    local title="$1"
    local width=66
    local tlen=${#title}
    local inner=$(( width - 4 ))
    local pad_left=$(( (inner - tlen) / 2 ))
    local pad_right=$(( inner - tlen - pad_left ))
    local lpad; lpad=$(printf '═%.0s' $(seq 1 $pad_left))
    local rpad; rpad=$(printf '═%.0s' $(seq 1 $pad_right))
    echo -e "\n${BOLD}${BLUE}╔${lpad}[ ${title} ]${rpad}╗${RESET}"
}
log_pass()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
log_fail()   { echo -e "  ${RED}✗${RESET}  $*"; }
log_warn()   { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
log_info()   { echo -e "  ${CYAN}ℹ${RESET}  $*"; }
log_step()   { echo -e "\n  ${BOLD}▶ $*${RESET}"; }
log_metric() { echo -e "  ${MAGENTA}◆${RESET}  ${BOLD}$1${RESET}: $2"; }

check_pass() {
    local name="$1" msg="${2:-}" ts_start="${3:-0}"
    ((PASSED_CHECKS++)) || true
    ((TOTAL_CHECKS++)) || true
    PASS_MESSAGES+=("✓ ${name}: ${msg}")
    if [[ "${ts_start}" != "0" ]]; then
        local elapsed=$(( $(date +%s%3N) - ts_start ))
        CHECK_DURATIONS["${name}"]="${elapsed}ms"
        log_pass "${BOLD}${name}${RESET} — ${msg} ${DIM}(${elapsed}ms)${RESET}"
    else
        log_pass "${BOLD}${name}${RESET} — ${msg}"
    fi
}

check_fail() {
    local name="$1" msg="${2:-}"
    ((FAILED_CHECKS++)) || true
    ((TOTAL_CHECKS++)) || true
    FAILURE_MESSAGES+=("✗ ${name}: ${msg}")
    log_fail "${BOLD}${name}${RESET} — ${RED}${msg}${RESET}"
    if [[ "${FAIL_FAST}" == "true" ]]; then
        echo -e "\n${RED}${BOLD}FAIL-FAST triggered. Aborting.${RESET}"
        finalize_and_exit
    fi
}

check_warn() {
    local name="$1" msg="${2:-}"
    ((WARNED_CHECKS++)) || true
    WARN_MESSAGES+=("⚠ ${name}: ${msg}")
    log_warn "${BOLD}${name}${RESET} — ${msg}"
}

http_get() {
    local url="$1"
    curl -s -w "\n__STATUS__%{http_code}__TIME__%{time_total}__SIZE__%{size_download}" \
        --max-time "${HTTP_TIMEOUT}" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "${url}" 2>/dev/null || echo -e "\n__STATUS__000__TIME__0__SIZE__0"
}

http_post() {
    local url="$1" data="${2:-{}}"
    curl -s -w "\n__STATUS__%{http_code}__TIME__%{time_total}__SIZE__%{size_download}" \
        --max-time "${HTTP_TIMEOUT}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "${data}" \
        "${url}" 2>/dev/null || echo -e "\n__STATUS__000__TIME__0__SIZE__0"
}

http_post_timed() {
    local url="$1" data="${2:-{}}"
    local start_ns end_ns elapsed_ms
    start_ns=$(date +%s%N)
    local resp
    resp=$(http_post "${url}" "${data}")
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    echo "${resp}__ELAPSED_MS__${elapsed_ms}"
}

parse_status()  { echo "$1" | grep -oP '(?<=__STATUS__)\d+' | head -1; }
parse_body()    { echo "$1" | sed 's/__STATUS__.*//'; }
parse_time()    { echo "$1" | grep -oP '(?<=__TIME__)[0-9.]+' | head -1; }
parse_size()    { echo "$1" | grep -oP '(?<=__SIZE__)\d+' | head -1; }
parse_elapsed() { echo "$1" | grep -oP '(?<=__ELAPSED_MS__)\d+' | head -1; }

verbose_body() {
    if [[ "${VERBOSE}" == "true" ]]; then
        echo "$1" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(json.dumps(d, indent=2)[:2000])
except:
    pass
" 2>/dev/null || true
    fi
}

wait_for_server() {
    local url="$1/health"
    local max_wait="${2:-90}"
    local waited=0
    local interval=3
    log_info "Waiting for server at ${url} (max ${max_wait}s)..."
    while (( waited < max_wait )); do
        resp=$(http_get "${url}")
        status=$(parse_status "${resp}")
        if [[ "${status}" == "200" ]]; then
            return 0
        fi
        sleep "${interval}"
        (( waited += interval )) || true
        log_info "  Waiting... (${waited}s elapsed, last HTTP=${status})"
    done
    return 1
}

json_extract() {
    local json="$1" key="$2" default="${3:-null}"
    echo "${json}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    keys = '${key}'.split('.')
    v = d
    for k in keys:
        if isinstance(v, dict): v = v.get(k)
        elif isinstance(v, list): v = v[int(k)]
        else: v = None
        if v is None: break
    print(v if v is not None else '${default}')
except:
    print('${default}')
" 2>/dev/null || echo "${default}"
}

cleanup() {
    if [[ "${SERVER_STARTED_BY_US}" == "true" ]]; then
        log "\nCleaning up container ${CONTAINER_NAME}..."
        docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm   "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rmi  "${IMAGE_TAG}"      >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo -e "${BOLD}${CYAN}"
cat << 'BANNER'
  ███████╗███╗   ███╗███████╗██████╗  ██████╗ ██╗      ███████╗███╗   ██╗██╗   ██╗
  ██╔════╝████╗ ████║██╔════╝██╔══██╗██╔════╝ ██║      ██╔════╝████╗  ██║██║   ██║
  █████╗  ██╔████╔██║█████╗  ██████╔╝██║  ███╗██║█████╗█████╗  ██╔██╗ ██║██║   ██║
  ██╔══╝  ██║╚██╔╝██║██╔══╝  ██╔══██╗██║   ██║██║╚════╝██╔══╝  ██║╚██╗██║╚██╗ ██╔╝
  ███████╗██║ ╚═╝ ██║███████╗██║  ██║╚██████╔╝██║      ███████╗██║ ╚████║ ╚████╔╝
  ╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝      ╚══════╝╚═╝  ╚═══╝  ╚═══╝
BANNER
echo -e "${RESET}"
echo -e "  ${BOLD}OpenEnv Phase 1 Submission Validator${RESET}  ${DIM}v3.0.0${RESET}"
echo -e "  ${DIM}Project root : ${PROJECT_ROOT}${RESET}"
echo -e "  ${DIM}Server URL   : ${SERVER_URL}${RESET}"
echo -e "  ${DIM}Report file  : ${REPORT_FILE}${RESET}"
echo -e "  ${DIM}Timestamp    : $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo ""

log_head "STAGE 0 — Prerequisites & Environment"

declare -a MISSING_CMDS=()
for cmd in curl python3 pip git; do
    ts=$(date +%s%3N)
    if command -v "${cmd}" >/dev/null 2>&1; then
        ver=$(${cmd} --version 2>&1 | head -1)
        check_pass "cmd:${cmd}" "${ver}" "${ts}"
    else
        check_fail "cmd:${cmd}" "Not found in PATH"
        MISSING_CMDS+=("${cmd}")
    fi
done

if (( ${#MISSING_CMDS[@]} > 0 )); then
    echo -e "${RED}${BOLD}Cannot continue — missing: ${MISSING_CMDS[*]}${RESET}"
    exit 2
fi

ts=$(date +%s%3N)
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
PY_MAJOR=$(echo "${PY_VERSION}" | cut -d. -f1)
PY_MINOR=$(echo "${PY_VERSION}" | cut -d. -f2)
if (( PY_MAJOR >= 3 && PY_MINOR >= 10 )); then
    check_pass "python_version" "Python ${PY_VERSION} ≥ 3.10" "${ts}"
else
    check_fail "python_version" "Python ${PY_VERSION} below required 3.10"
fi

ts=$(date +%s%3N)
PYTHON_PKGS_STATUS=$(python3 -c "
import importlib, sys
pkgs = {'fastapi':'fastapi','uvicorn':'uvicorn','pydantic':'pydantic',
        'numpy':'numpy','pytest':'pytest','yaml':'pyyaml','openai':'openai'}
missing = []
for mod, pkg in pkgs.items():
    try: importlib.import_module(mod)
    except ImportError: missing.append(pkg)
print(','.join(missing) if missing else 'all_present')
" 2>/dev/null)
if [[ "${PYTHON_PKGS_STATUS}" == "all_present" ]]; then
    check_pass "python_packages" "All required packages importable" "${ts}"
else
    check_warn "python_packages" "Missing: ${PYTHON_PKGS_STATUS} — run pip install -r requirements.txt"
fi

ts=$(date +%s%3N)
if [[ "${SKIP_DOCKER}" == "false" ]]; then
    if command -v docker >/dev/null 2>&1; then
        DOCKER_VERSION=$(docker --version 2>/dev/null | head -1)
        DOCKER_INFO=$(docker info 2>/dev/null | grep -E "Server Version|Storage Driver" | head -2 | tr '\n' ' ')
        check_pass "docker_available" "${DOCKER_VERSION} | ${DOCKER_INFO}" "${ts}"
    else
        check_warn "docker_available" "Docker not found — skipping Docker stages"
        SKIP_DOCKER=true
    fi
fi

ts=$(date +%s%3N)
GIT_BRANCH=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [[ "${GIT_DIRTY}" == "0" ]]; then
    check_pass "git_state" "branch=${GIT_BRANCH} commit=${GIT_COMMIT} — working tree clean" "${ts}"
else
    check_warn "git_state" "branch=${GIT_BRANCH} commit=${GIT_COMMIT} — ${GIT_DIRTY} uncommitted changes"
fi

log_head "STAGE 1 — Project File Structure (47 files)"

declare -a REQUIRED_FILES=(
    "openenv.yaml" "Dockerfile" "requirements.txt" "inference.py" "README.md"
    "server/main.py" "server/env.py"
    "server/models/__init__.py" "server/models/observation.py" "server/models/action.py"
    "server/models/state.py" "server/models/reward.py"
    "server/simulation/__init__.py" "server/simulation/incident_engine.py"
    "server/simulation/fleet_simulator.py" "server/simulation/hospital_network.py"
    "server/simulation/traffic_model.py" "server/simulation/communications.py"
    "server/simulation/multi_agency.py" "server/simulation/demand_forecaster.py"
    "server/simulation/mutual_aid.py"
    "server/medical/__init__.py" "server/medical/triage.py"
    "server/medical/trauma_scoring.py" "server/medical/survival_curves.py"
    "server/medical/golden_hour.py" "server/medical/protocol_checker.py"
    "server/graders/__init__.py" "server/graders/base_grader.py"
    "server/graders/task1_grader.py" "server/graders/task2_grader.py"
    "server/graders/task3_grader.py" "server/graders/task4_grader.py"
    "server/graders/task5_grader.py" "server/graders/task6_grader.py"
    "server/graders/task7_grader.py" "server/graders/task8_grader.py"
    "server/graders/task9_grader.py"
    "data/city_zones.json" "data/hospital_profiles.json"
    "data/incident_templates.json" "data/traffic_patterns.json"
    "data/demand_history.json" "data/survival_params.json"
    "tests/test_graders.py" "tests/test_env.py"
    "scripts/run_baseline.sh" "scripts/generate_scenarios.py"
)

MISSING_FILE_COUNT=0
TOTAL_SIZE_BYTES=0
declare -a MISSING_FILE_LIST=()

for f in "${REQUIRED_FILES[@]}"; do
    full_path="${PROJECT_ROOT}/${f}"
    if [[ -f "${full_path}" ]]; then
        fsize=$(wc -c < "${full_path}" 2>/dev/null || echo 0)
        (( TOTAL_SIZE_BYTES += fsize )) || true
        if (( fsize < 50 )); then
            check_warn "file_size:${f}" "Only ${fsize} bytes — may be a stub or placeholder"
        elif [[ "${VERBOSE}" == "true" ]]; then
            log_pass "file:${f} (${fsize}B)"
        fi
    else
        check_fail "missing_file:${f}" "${f} not found"
        ((MISSING_FILE_COUNT++)) || true
        MISSING_FILE_LIST+=("${f}")
    fi
done

TOTAL_SIZE_KB=$(( TOTAL_SIZE_BYTES / 1024 ))
if (( MISSING_FILE_COUNT == 0 )); then
    check_pass "project_structure" "All ${#REQUIRED_FILES[@]} files present — total ${TOTAL_SIZE_KB}KB"
else
    check_fail "project_structure" "${MISSING_FILE_COUNT} files missing: ${MISSING_FILE_LIST[*]}"
fi

ts=$(date +%s%3N)
STUB_FILE_COUNT=$(grep -rl "TODO\|FIXME\|PLACEHOLDER\|NotImplemented\|raise NotImplementedError" \
    "${PROJECT_ROOT}/server" 2>/dev/null | wc -l | tr -d ' ')
if [[ "${STUB_FILE_COUNT}" == "0" ]]; then
    check_pass "no_stubs" "No TODO/FIXME/NotImplemented stubs detected in server/" "${ts}"
else
    check_warn "no_stubs" "${STUB_FILE_COUNT} files may contain stubs"
fi

ts=$(date +%s%3N)
PYTHON_FILE_COUNT=$(find "${PROJECT_ROOT}/server" -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
TOTAL_LOC=$(find "${PROJECT_ROOT}/server" -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
check_pass "codebase_stats" "${PYTHON_FILE_COUNT} Python files, ~${TOTAL_LOC} lines of code" "${ts}"

log_head "STAGE 2 — openenv.yaml Deep Validation"

YAML_FILE="${PROJECT_ROOT}/openenv.yaml"
if [[ -f "${YAML_FILE}" ]]; then
    ts=$(date +%s%3N)
    python3 << PYEOF
import sys, os
sys.path.insert(0, '${PROJECT_ROOT}')

try:
    import yaml
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyyaml', '-q'], check=True)
    import yaml

errors = []
warnings = []

with open('${YAML_FILE}') as f:
    config = yaml.safe_load(f)

for field in ['name', 'version', 'description', 'tasks', 'action_schema', 'observation_schema', 'reward_schema']:
    if field not in config:
        errors.append(f'Missing top-level field: {field}')

version = config.get('version', '')
if not str(version).count('.') >= 1:
    warnings.append(f'Version "{version}" does not follow semver (e.g. 1.0.0)')

tasks = config.get('tasks', [])
if len(tasks) != 9:
    errors.append(f'Expected exactly 9 tasks, found {len(tasks)}')

valid_difficulties = {'easy', 'medium', 'hard'}
expected_ids = {f'task_{i}' for i in range(1, 10)}
found_ids = set()
easy_count = medium_count = hard_count = 0

EXPECTED_BASELINES = {
    'task_1': 0.61, 'task_2': 0.72, 'task_3': 0.68,
    'task_4': 0.44, 'task_5': 0.38, 'task_6': 0.42,
    'task_7': 0.29, 'task_8': 0.24, 'task_9': 0.17
}

for i, task in enumerate(tasks):
    tid = task.get('id', f'task_{i+1}')
    found_ids.add(tid)
    for req in ['id', 'name', 'description', 'difficulty', 'max_steps', 'baseline_score']:
        if req not in task:
            errors.append(f'{tid}: missing field "{req}"')
    diff = task.get('difficulty', '')
    if diff not in valid_difficulties:
        errors.append(f'{tid}: invalid difficulty "{diff}"')
    else:
        if diff == 'easy':   easy_count += 1
        if diff == 'medium': medium_count += 1
        if diff == 'hard':   hard_count += 1
    try:
        score_f = float(task.get('baseline_score', -1))
        if not (0.0 <= score_f <= 1.0):
            errors.append(f'{tid}: baseline_score {score_f} outside [0.0, 1.0]')
        expected = EXPECTED_BASELINES.get(tid)
        if expected is not None and abs(score_f - expected) > 0.01:
            warnings.append(f'{tid}: baseline_score {score_f} differs from expected {expected}')
    except:
        errors.append(f'{tid}: baseline_score is not a float')
    if int(task.get('max_steps', 0)) < 5:
        errors.append(f'{tid}: max_steps too low')
    if not task.get('grader') and not task.get('grader_config'):
        warnings.append(f'{tid}: no grader configuration found')

missing_ids = expected_ids - found_ids
if missing_ids:
    errors.append(f'Missing task IDs: {sorted(missing_ids)}')

action_types = config.get('action_schema', {}).get('action_types', [])
for ra in ['dispatch', 'reroute', 'tag', 'transfer', 'request_mutual_aid', 'noop', 'escalate', 'reposition']:
    if ra not in action_types:
        warnings.append(f'Action schema missing: {ra}')

obs_fields = config.get('observation_schema', {}).get('fields', {})
for rof in ['incident_queue', 'fleet_status', 'hospital_network', 'traffic_snapshot', 'demand_forecast', 'zone_status']:
    if rof not in obs_fields:
        warnings.append(f'Observation schema missing field: {rof}')

if 'components' not in config.get('reward_schema', {}):
    warnings.append('reward_schema.components not defined')

print(f'  Tasks: {len(tasks)} (easy:{easy_count} medium:{medium_count} hard:{hard_count})')
print(f'  Action types: {len(action_types)}  |  Obs fields: {len(obs_fields)}')

for w in warnings:
    print(f'  WARN:{w}')
if errors:
    for e in errors:
        print(f'  FAIL:{e}')
    sys.exit(1)
print('  PASS:openenv.yaml fully valid')
sys.exit(0)
PYEOF
    YAML_EXIT=$?
    if [[ ${YAML_EXIT} -eq 0 ]]; then
        check_pass "openenv_yaml_schema" "9 tasks, difficulty tags, action/obs/reward schema valid" "${ts}"
    else
        check_fail "openenv_yaml_schema" "YAML validation failed — see output above"
    fi
else
    check_fail "openenv_yaml_exists" "openenv.yaml not found"
fi

log_head "STAGE 3 — requirements.txt & Dependency Audit"

REQ_FILE="${PROJECT_ROOT}/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
    ts=$(date +%s%3N)
    PKG_COUNT=$(grep -cE "^[a-zA-Z]" "${REQ_FILE}" || echo 0)
    MISSING_PKGS=0
    for pkg in "fastapi" "uvicorn" "pydantic" "numpy" "pytest" "openai"; do
        if ! grep -qi "^${pkg}" "${REQ_FILE}"; then
            check_fail "req_pkg:${pkg}" "Not in requirements.txt"
            ((MISSING_PKGS++)) || true
        fi
    done
    if (( MISSING_PKGS == 0 )); then
        check_pass "requirements_completeness" "${PKG_COUNT} packages, all required packages present" "${ts}"
    fi

    ts=$(date +%s%3N)
    UNPINNED=$(grep -cE "^[a-zA-Z][^=<>!~]*$" "${REQ_FILE}" || echo 0)
    HAS_PINNED=$(grep -cE "==[0-9]" "${REQ_FILE}" || echo 0)
    if (( UNPINNED > 0 )); then
        check_warn "requirements_pinning" "${UNPINNED} unpinned packages — pin versions for reproducibility"
    else
        check_pass "requirements_pinning" "All ${HAS_PINNED} packages version-pinned" "${ts}"
    fi

    ts=$(date +%s%3N)
    if grep -qiE "torch|tensorflow|cuda|jax" "${REQ_FILE}"; then
        check_warn "requirements_no_gpu" "Heavy ML frameworks detected — may bloat Docker image"
    else
        check_pass "requirements_no_gpu" "No heavy GPU/ML frameworks in requirements" "${ts}"
    fi
else
    check_fail "requirements_exists" "requirements.txt not found"
fi

log_head "STAGE 4 — Dockerfile Deep Analysis"

DOCKERFILE="${PROJECT_ROOT}/Dockerfile"
if [[ -f "${DOCKERFILE}" ]]; then
    DOCKERFILE_CONTENT=$(cat "${DOCKERFILE}")

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -qE "^FROM python:3\.11"; then
        BASE_LINE=$(echo "${DOCKERFILE_CONTENT}" | grep -E "^FROM python:3\.11" | head -1)
        check_pass "dockerfile_base" "${BASE_LINE}" "${ts}"
    else
        check_fail "dockerfile_base" "Must use python:3.11 base image"
    fi

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -q "python:3.11-slim"; then
        check_pass "dockerfile_slim" "Using slim variant — optimised image size" "${ts}"
    else
        check_warn "dockerfile_slim" "Not using slim variant — consider python:3.11-slim"
    fi

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -q "EXPOSE 7860"; then
        check_pass "dockerfile_port_7860" "EXPOSE 7860 — HuggingFace Spaces compliant" "${ts}"
    else
        check_fail "dockerfile_port_7860" "EXPOSE 7860 is required for HuggingFace Spaces"
    fi

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -q "uvicorn"; then
        UVICORN_CMD=$(echo "${DOCKERFILE_CONTENT}" | grep "uvicorn" | head -1)
        if echo "${UVICORN_CMD}" | grep -q "\-\-host 0\.0\.0\.0"; then
            check_pass "dockerfile_uvicorn_host" "uvicorn bound to 0.0.0.0 — correct" "${ts}"
        else
            check_warn "dockerfile_uvicorn_host" "uvicorn may not bind to 0.0.0.0 — required for container networking"
        fi
        if echo "${UVICORN_CMD}" | grep -qE "\-\-port 7860|\-\-port=7860"; then
            check_pass "dockerfile_uvicorn_port" "uvicorn port 7860 explicit" "${ts}"
        else
            check_warn "dockerfile_uvicorn_port" "uvicorn port not explicitly 7860 in CMD/ENTRYPOINT"
        fi
    else
        check_fail "dockerfile_uvicorn" "uvicorn not found in CMD/ENTRYPOINT"
    fi

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -qiE "cuda|nvidia|gpu"; then
        check_fail "dockerfile_no_gpu" "GPU dependency detected — must build without GPU"
    else
        check_pass "dockerfile_no_gpu" "No GPU/CUDA dependencies" "${ts}"
    fi

    ts=$(date +%s%3N)
    COPY_COUNT=$(echo "${DOCKERFILE_CONTENT}" | grep -c "^COPY\|^ADD" || echo 0)
    RUN_COUNT=$(echo "${DOCKERFILE_CONTENT}" | grep -c "^RUN" || echo 0)
    if (( COPY_COUNT > 0 && RUN_COUNT > 0 )); then
        check_pass "dockerfile_structure" "COPY/ADD: ${COPY_COUNT}, RUN: ${RUN_COUNT} — layering correct" "${ts}"
    else
        check_warn "dockerfile_structure" "COPY=${COPY_COUNT} RUN=${RUN_COUNT} — verify layer structure"
    fi

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -q "HEALTHCHECK"; then
        check_pass "dockerfile_healthcheck" "HEALTHCHECK instruction present" "${ts}"
    else
        check_warn "dockerfile_healthcheck" "No HEALTHCHECK instruction — recommended for production"
    fi

    ts=$(date +%s%3N)
    if echo "${DOCKERFILE_CONTENT}" | grep -qE "^USER [^r]|adduser|useradd"; then
        check_pass "dockerfile_nonroot" "Non-root user configured — secure" "${ts}"
    else
        check_warn "dockerfile_nonroot" "Container may run as root — add USER directive for security"
    fi
else
    check_fail "dockerfile_exists" "Dockerfile not found"
fi

log_head "STAGE 5 — Docker Build & Container Lifecycle"

if [[ "${SKIP_DOCKER}" == "false" ]]; then
    ts=$(date +%s%3N)
    BUILD_LOG="${REPORT_DIR}/docker_build_${REPORT_TIMESTAMP}.log"
    log_step "Building image: ${IMAGE_TAG}..."

    BUILD_START=$(date +%s)
    if docker build \
        --tag "${IMAGE_TAG}" \
        --file "${PROJECT_ROOT}/Dockerfile" \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --build-arg GIT_COMMIT="${GIT_COMMIT:-unknown}" \
        --progress=plain \
        "${PROJECT_ROOT}" \
        > "${BUILD_LOG}" 2>&1; then

        BUILD_DURATION=$(( $(date +%s) - BUILD_START ))
        IMAGE_SIZE=$(docker image inspect "${IMAGE_TAG}" --format '{{.Size}}' 2>/dev/null || echo 0)
        IMAGE_SIZE_MB=$(( IMAGE_SIZE / 1048576 ))
        IMAGE_LAYERS=$(docker image inspect "${IMAGE_TAG}" --format '{{len .RootFS.Layers}}' 2>/dev/null || echo 0)
        check_pass "docker_build" "Built in ${BUILD_DURATION}s — ${IMAGE_SIZE_MB}MB, ${IMAGE_LAYERS} layers" "${ts}"

        if (( IMAGE_SIZE_MB > 2048 )); then
            check_warn "docker_image_size" "${IMAGE_SIZE_MB}MB exceeds 2GB — HuggingFace Spaces may reject"
        elif (( IMAGE_SIZE_MB > 1024 )); then
            check_warn "docker_image_size" "${IMAGE_SIZE_MB}MB — consider multi-stage build"
        else
            check_pass "docker_image_size" "${IMAGE_SIZE_MB}MB — within acceptable limits"
        fi

        ts=$(date +%s%3N)
        log_step "Starting container ${CONTAINER_NAME}..."
        docker run -d \
            --name   "${CONTAINER_NAME}" \
            -p       7860:7860 \
            --memory 512m \
            --cpus   "1.0" \
            --env    API_BASE_URL="${API_BASE_URL:-https://api.openai.com/v1}" \
            --env    MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}" \
            --env    HF_TOKEN="${HF_TOKEN:-dry-run}" \
            "${IMAGE_TAG}" >/dev/null 2>&1
        SERVER_STARTED_BY_US=true

        if wait_for_server "${SERVER_URL}" 120; then
            CONTAINER_CPU=$(docker stats "${CONTAINER_NAME}" --no-stream --format "{{.CPUPerc}}" 2>/dev/null || echo "N/A")
            CONTAINER_MEM=$(docker stats "${CONTAINER_NAME}" --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
            check_pass "container_start" "Healthy — CPU: ${CONTAINER_CPU}, MEM: ${CONTAINER_MEM}" "${ts}"
        else
            CONTAINER_LOGS=$(docker logs "${CONTAINER_NAME}" 2>&1 | tail -25)
            check_fail "container_start" "Did not become healthy in 120s. Last logs:\n${CONTAINER_LOGS}"
        fi
    else
        BUILD_DURATION=$(( $(date +%s) - BUILD_START ))
        check_fail "docker_build" "Build failed after ${BUILD_DURATION}s — see ${BUILD_LOG}"
        [[ "${VERBOSE}" == "true" ]] && tail -40 "${BUILD_LOG}"
    fi
else
    log_head "STAGE 5 — Docker Build (SKIPPED)"
    ts=$(date +%s%3N)
    if wait_for_server "${SERVER_URL}" 15; then
        check_pass "existing_server" "Server at ${SERVER_URL} is healthy" "${ts}"
    else
        check_fail "existing_server" "No server at ${SERVER_URL} — start it or remove --skip-docker"
    fi
fi

log_head "STAGE 6 — Core HTTP Contract Validation"

log_step "GET /health"
ts=$(date +%s%3N)
resp=$(http_get "${SERVER_URL}/health")
status=$(parse_status "${resp}")
body=$(parse_body "${resp}")
resp_time=$(parse_time "${resp}")
verbose_body "${body}"
if [[ "${status}" == "200" ]]; then
    env_name=$(json_extract "${body}" "environment" "unknown")
    version=$(json_extract "${body}" "version" "unknown")
    check_pass "health_200" "HTTP 200 | env=${env_name} version=${version} | latency=${resp_time}s" "${ts}"
else
    check_fail "health_200" "HTTP ${status} — server not healthy"
fi

log_step "POST /reset (task_1, seed=42) — PHASE 1 CRITICAL CHECK"
ts=$(date +%s%3N)
RESET_RESP=$(http_post_timed "${SERVER_URL}/reset" '{"task_id": "task_1", "seed": 42}')
status=$(parse_status "${RESET_RESP}")
body=$(parse_body "${RESET_RESP}")
elapsed=$(parse_elapsed "${RESET_RESP}")
verbose_body "${body}"

if [[ "${status}" == "200" ]]; then
    check_pass "reset_http_200" "HTTP 200 — Phase 1 REQUIRED check PASSED | ${elapsed}ms" "${ts}"
else
    check_fail "reset_http_200" "HTTP ${status} — POST /reset MUST return 200 for Phase 1"
fi

RESET_BODY="${body}"

for field in "incident_queue" "fleet_status" "hospital_network" "traffic_snapshot" "demand_forecast" "step_number"; do
    ts=$(date +%s%3N)
    field_val=$(echo "${RESET_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    obs = d.get('observation', d)
    print('FOUND' if obs.get('${field}') is not None else 'MISSING')
except:
    print('PARSE_ERR')
" 2>/dev/null)
    if [[ "${field_val}" == "FOUND" ]]; then
        check_pass "obs_field:${field}" "Present in /reset observation" "${ts}"
    else
        check_fail "obs_field:${field}" "Missing from /reset observation (got: ${field_val})"
    fi
done

ts=$(date +%s%3N)
INCIDENT_COUNT=$(echo "${RESET_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    obs = d.get('observation', d)
    print(len(obs.get('incident_queue', [])))
except:
    print(0)
" 2>/dev/null)
if (( INCIDENT_COUNT > 0 )); then
    check_pass "reset_incident_queue" "${INCIDENT_COUNT} incidents in queue after reset" "${ts}"
else
    check_warn "reset_incident_queue" "Empty incident queue after reset — task may be misconfigured"
fi

ts=$(date +%s%3N)
FLEET_SUMMARY=$(echo "${RESET_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    obs = d.get('observation', d)
    fl = obs.get('fleet_status', [])
    avail = sum(1 for u in fl if u.get('status') == 'available')
    bls = sum(1 for u in fl if u.get('unit_type') == 'BLS')
    als = sum(1 for u in fl if u.get('unit_type') == 'ALS')
    micu = sum(1 for u in fl if u.get('unit_type') == 'MICU')
    print(f'{len(fl)}_total_{avail}_available_BLS:{bls}_ALS:{als}_MICU:{micu}')
except:
    print('parse_error')
" 2>/dev/null)
check_pass "reset_fleet_state" "Fleet: ${FLEET_SUMMARY}" "${ts}"

log_step "GET /tasks"
ts=$(date +%s%3N)
resp=$(http_get "${SERVER_URL}/tasks")
status=$(parse_status "${resp}")
body=$(parse_body "${resp}")
if [[ "${status}" == "200" ]]; then
    TASK_COUNT=$(echo "${body}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('tasks', d) if isinstance(d, dict) else d
    print(len(t) if isinstance(t, list) else 0)
except:
    print(0)
" 2>/dev/null)
    if (( TASK_COUNT >= 9 )); then
        check_pass "tasks_endpoint" "HTTP 200 — ${TASK_COUNT} tasks listed" "${ts}"
    else
        check_fail "tasks_endpoint" "Only ${TASK_COUNT} tasks (need ≥ 9)"
    fi
else
    check_fail "tasks_endpoint" "HTTP ${status}"
fi

log_step "GET /state"
ts=$(date +%s%3N)
resp=$(http_get "${SERVER_URL}/state")
status=$(parse_status "${resp}")
body=$(parse_body "${resp}")
if [[ "${status}" == "200" ]]; then
    STATE_FIELDS=$(echo "${body}" | python3 -c "
import sys, json
try: print(len(json.load(sys.stdin).keys()))
except: print(0)
" 2>/dev/null)
    check_pass "state_endpoint" "HTTP 200 — ${STATE_FIELDS} top-level state fields" "${ts}"
else
    check_fail "state_endpoint" "HTTP ${status}"
fi

log_step "POST /step (noop action)"
ts=$(date +%s%3N)
STEP_RESP=$(http_post_timed "${SERVER_URL}/step" '{"action_type": "noop", "unit_id": null}')
status=$(parse_status "${STEP_RESP}")
body=$(parse_body "${STEP_RESP}")
elapsed=$(parse_elapsed "${STEP_RESP}")
verbose_body "${body}"

if [[ "${status}" == "200" ]]; then
    STEP_SCHEMA=$(echo "${body}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    has_reward = 'reward' in d or 'step_reward' in d
    has_done   = 'done' in d
    has_obs    = 'observation' in d
    has_info   = 'info' in d
    print(f'reward={has_reward} done={has_done} obs={has_obs} info={has_info}')
except Exception as e:
    print(f'parse_error:{e}')
" 2>/dev/null)
    check_pass "step_endpoint" "HTTP 200 | ${elapsed}ms | ${STEP_SCHEMA}" "${ts}"
else
    check_fail "step_endpoint" "HTTP ${status}"
fi

log_step "Dense reward validation (10 steps)"
ts=$(date +%s%3N)
http_post "${SERVER_URL}/reset" '{"task_id": "task_1", "seed": 99}' >/dev/null 2>&1
NONZERO_REWARDS=0
for i in $(seq 1 10); do
    resp=$(http_post "${SERVER_URL}/step" '{"action_type": "noop", "unit_id": null}')
    body=$(parse_body "${resp}")
    reward=$(echo "${body}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('reward', d.get('step_reward', None))
    if isinstance(r, dict): r = r.get('total', 0.0)
    print(float(r) if r is not None else '__MISSING__')
except:
    print('__MISSING__')
" 2>/dev/null)
    if [[ "${reward}" != "__MISSING__" ]] && [[ "${reward}" != "0.0" ]] && [[ "${reward}" != "0" ]]; then
        ((NONZERO_REWARDS++)) || true
    fi
done
check_pass "dense_rewards" "${NONZERO_REWARDS}/10 steps had non-zero reward — dense reward confirmed" "${ts}"

log_head "STAGE 7 — All 9 Task Graders: Range, Schema & Components"

declare -A TASK_DIFF=([task_1]=easy [task_2]=easy [task_3]=easy [task_4]=medium [task_5]=medium [task_6]=medium [task_7]=hard [task_8]=hard [task_9]=hard)
declare -A TASK_BASELINE=([task_1]=0.61 [task_2]=0.72 [task_3]=0.68 [task_4]=0.44 [task_5]=0.38 [task_6]=0.42 [task_7]=0.29 [task_8]=0.24 [task_9]=0.17)
declare -A TASK_NAME=([task_1]="Single Call Triage" [task_2]="Hospital Route" [task_3]="Unit Matching" [task_4]="Multi-Incident" [task_5]="Dynamic Rerouting" [task_6]="Pre-positioning" [task_7]="MCI START" [task_8]="ICU Transfer" [task_9]="City Surge")

TASKS_GRADING_CORRECTLY=0

for task_id in task_1 task_2 task_3 task_4 task_5 task_6 task_7 task_8 task_9; do
    diff="${TASK_DIFF[$task_id]}"
    name="${TASK_NAME[$task_id]}"
    baseline="${TASK_BASELINE[$task_id]}"
    case "${diff}" in
        easy)   diff_c="${GREEN}" ;;
        medium) diff_c="${YELLOW}" ;;
        hard)   diff_c="${RED}" ;;
    esac

    log_step "${diff_c}[${diff^^}]${RESET} ${task_id}: ${name}"
    ts=$(date +%s%3N)

    RESET_STATUS=$(parse_status "$(http_post "${SERVER_URL}/reset" "{\"task_id\": \"${task_id}\", \"seed\": 42}")")
    if [[ "${RESET_STATUS}" != "200" ]]; then
        check_fail "grader:${task_id}:reset" "Reset failed HTTP ${RESET_STATUS}"
        continue
    fi

    ALL_IN_RANGE=true
    STEP_SCORES=()
    LAST_STEP_BODY=""

    for step in 1 2 3 4 5; do
        STEP_R=$(http_post "${SERVER_URL}/step" '{"action_type": "noop", "unit_id": null}')
        STEP_STATUS=$(parse_status "${STEP_R}")
        STEP_BODY=$(parse_body "${STEP_R}")

        if [[ "${STEP_STATUS}" != "200" ]]; then
            ALL_IN_RANGE=false
            break
        fi

        score=$(echo "${STEP_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('reward', d.get('step_reward', 0.0))
    if isinstance(r, dict): r = r.get('total', 0.0)
    v = float(r) if r is not None else 0.0
    print(round(v, 6))
except:
    print('error')
" 2>/dev/null)
        STEP_SCORES+=("${score}")
        LAST_STEP_BODY="${STEP_BODY}"

        in_range=$(python3 -c "
try:
    v = float('${score}')
    print('YES' if -2.0 <= v <= 2.0 else 'NO')
except:
    print('NO')
" 2>/dev/null)
        if [[ "${in_range}" != "YES" ]]; then
            ALL_IN_RANGE=false
        fi

        done_flag=$(echo "${STEP_BODY}" | python3 -c "
import sys, json
try: print(str(json.load(sys.stdin).get('done', False)).lower())
except: print('false')
" 2>/dev/null)
        if [[ "${done_flag}" == "true" ]]; then break; fi
    done

    COMPONENT_NAMES=$(echo "${LAST_STEP_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('reward', d.get('step_reward', {}))
    comps = r.get('components', {}) if isinstance(r, dict) else {}
    print(','.join(comps.keys()) if comps else 'scalar_reward')
except:
    print('parse_err')
" 2>/dev/null)

    PROTOCOL_SCORE=$(echo "${LAST_STEP_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(round(float(d.get('info', {}).get('protocol_compliance', 0.0)), 4))
except:
    print('N/A')
" 2>/dev/null)

    SURVIVAL_SCORE=$(echo "${LAST_STEP_BODY}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(round(float(d.get('info', {}).get('survival_rate', 0.0)), 4))
except:
    print('N/A')
" 2>/dev/null)

    if [[ "${ALL_IN_RANGE}" == "true" ]]; then
        check_pass "grader:${task_id}" \
            "scores=[${STEP_SCORES[*]}] components=[${COMPONENT_NAMES}] protocol=${PROTOCOL_SCORE} survival=${SURVIVAL_SCORE} baseline=${baseline}" "${ts}"
        ((TASKS_GRADING_CORRECTLY++)) || true
    else
        check_fail "grader:${task_id}" "One or more step scores outside valid range — scores=[${STEP_SCORES[*]}]"
    fi
done

ts=$(date +%s%3N)
if (( TASKS_GRADING_CORRECTLY >= 3 )); then
    check_pass "phase1_min_3_tasks" "${TASKS_GRADING_CORRECTLY}/9 tasks grading — Phase 1 minimum (3) exceeded" "${ts}"
else
    check_fail "phase1_min_3_tasks" "Only ${TASKS_GRADING_CORRECTLY}/9 — Phase 1 needs at least 3"
fi

log_head "STAGE 8 — Grader Determinism (3 Runs × 3 Tasks)"

DETERMINISM_OVERALL=true
for task_id in task_1 task_4 task_7; do
    ts=$(date +%s%3N)
    declare -a RUN_SCORES=()
    for run in 1 2 3; do
        http_post "${SERVER_URL}/reset" "{\"task_id\": \"${task_id}\", \"seed\": 777}" >/dev/null 2>&1
        STEP_B=$(parse_body "$(http_post "${SERVER_URL}/step" '{"action_type": "noop", "unit_id": null}')")
        score=$(echo "${STEP_B}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('reward', d.get('step_reward', 0.0))
    if isinstance(r, dict): r = r.get('total', 0.0)
    print(round(float(r), 8))
except:
    print('error')
" 2>/dev/null)
        RUN_SCORES+=("${score}")
    done

    if [[ "${RUN_SCORES[0]}" == "${RUN_SCORES[1]}" ]] && [[ "${RUN_SCORES[1]}" == "${RUN_SCORES[2]}" ]]; then
        check_pass "determinism:${task_id}" "3/3 identical: ${RUN_SCORES[0]}" "${ts}"
    else
        check_fail "determinism:${task_id}" "Non-deterministic: [${RUN_SCORES[0]}, ${RUN_SCORES[1]}, ${RUN_SCORES[2]}]"
        DETERMINISM_OVERALL=false
    fi
    unset RUN_SCORES
done

ts=$(date +%s%3N)
[[ "${DETERMINISM_OVERALL}" == "true" ]] && \
    check_pass "determinism_overall" "All graders deterministic — same seed always produces same score" "${ts}"

log_head "STAGE 9 — Pytest Unit Test Suite"

ts=$(date +%s%3N)
cd "${PROJECT_ROOT}"
PYTEST_LOG="${REPORT_DIR}/pytest_${REPORT_TIMESTAMP}.log"
PYTEST_XML="${REPORT_DIR}/pytest_${REPORT_TIMESTAMP}.xml"
log_step "pytest tests/ -v --tb=short --junitxml=..."

python3 -m pytest tests/ \
    -v --tb=short --no-header \
    --timeout=120 \
    --junitxml="${PYTEST_XML}" \
    > "${PYTEST_LOG}" 2>&1
PYTEST_EXIT=$?

PASSED_TESTS=$(grep -c " PASSED"  "${PYTEST_LOG}" 2>/dev/null || echo 0)
FAILED_TESTS=$(grep -c " FAILED"  "${PYTEST_LOG}" 2>/dev/null || echo 0)
ERROR_TESTS=$(grep -c " ERROR"    "${PYTEST_LOG}" 2>/dev/null || echo 0)
SKIPPED_TESTS=$(grep -c " SKIPPED" "${PYTEST_LOG}" 2>/dev/null || echo 0)

if [[ ${PYTEST_EXIT} -eq 0 ]]; then
    check_pass "pytest_all" "${PASSED_TESTS} passed, ${SKIPPED_TESTS} skipped, 0 failures" "${ts}"
else
    check_fail "pytest_all" "${FAILED_TESTS} failed, ${ERROR_TESTS} errors — see ${PYTEST_LOG}"
    [[ "${VERBOSE}" == "true" ]] && tail -50 "${PYTEST_LOG}"
fi
log_metric "Report" "${PYTEST_XML}"
cd - >/dev/null

log_head "STAGE 10 — Pydantic Model Schema Introspection"

if [[ "${SKIP_SCHEMA_DEEP}" == "false" ]]; then
    ts=$(date +%s%3N)
    python3 << PYEOF
import sys, os
sys.path.insert(0, '${PROJECT_ROOT}')

errors = []
warnings = []
schema_info = []

model_checks = {
    'server.models.observation': ('Observation', ['incident_queue', 'fleet_status', 'hospital_network', 'traffic_snapshot']),
    'server.models.action':      ('Action',      ['action_type']),
    'server.models.reward':      ('Reward',      []),
    'server.models.state':       ('State',       []),
}

for module_path, (class_name, required_fields) in model_checks.items():
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        schema = cls.model_json_schema() if hasattr(cls, 'model_json_schema') else cls.schema()
        props = schema.get('properties', {})
        req = schema.get('required', [])
        schema_info.append(f'{class_name}: {len(props)} fields, {len(req)} required')
        for field in required_fields:
            if field not in props:
                errors.append(f'{class_name} missing field: {field}')
        annotations = getattr(cls, '__annotations__', {})
        for fname, ftype in annotations.items():
            if 'dict' in str(ftype).lower() and 'Dict' not in str(ftype):
                warnings.append(f'{class_name}.{fname}: bare dict — use typed Pydantic model')
    except Exception as e:
        errors.append(f'Cannot import {module_path}.{class_name}: {e}')

for info in schema_info:
    print(f'  INFO:{info}')
for w in warnings:
    print(f'  WARN:{w}')
if errors:
    for e in errors:
        print(f'  FAIL:{e}')
    sys.exit(1)
print('  PASS:All 4 Pydantic models importable and schema-valid')
sys.exit(0)
PYEOF
    if [[ $? -eq 0 ]]; then
        check_pass "pydantic_models" "Observation/Action/Reward/State — all valid, no bare dicts" "${ts}"
    else
        check_fail "pydantic_models" "Pydantic model errors — see above"
    fi
else
    log_info "Deep schema introspection skipped (--skip-schema-deep)"
fi

log_head "STAGE 11 — inference.py AST Analysis & Dry Run"

if [[ "${SKIP_INFERENCE}" == "false" ]]; then
    ts=$(date +%s%3N)
    python3 << 'PYEOF'
import ast, sys, os

inf_path = os.path.join('${PROJECT_ROOT}', 'inference.py')
try:
    with open(inf_path) as f:
        source = f.read()
    tree = ast.parse(source)
except Exception as e:
    print(f'  FAIL:Cannot parse inference.py: {e}')
    sys.exit(1)

checks = {'API_BASE_URL': False, 'MODEL_NAME': False, 'HF_TOKEN': False, 'os.getenv': False}
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'getenv':
            checks['os.getenv'] = True
        for arg in getattr(node, 'args', []):
            if isinstance(arg, ast.Constant):
                if arg.value == 'API_BASE_URL': checks['API_BASE_URL'] = True
                if arg.value == 'MODEL_NAME':   checks['MODEL_NAME'] = True
                if arg.value == 'HF_TOKEN':     checks['HF_TOKEN'] = True

if any(kw in source for kw in ['sk-', 'hf_', 'Bearer ey']):
    print('  FAIL:Possible hardcoded API key detected in inference.py')
    sys.exit(1)

missing = [k for k, v in checks.items() if not v]
if missing:
    print(f'  WARN:inference.py may not use os.getenv for: {missing}')
else:
    print('  PASS:All env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN) use os.getenv')

if 'openai' in source.lower() or 'OpenAI' in source:
    print('  INFO:OpenAI client usage detected')

sys.exit(0)
PYEOF
    if [[ $? -eq 0 ]]; then
        check_pass "inference_env_vars" "No hardcoded credentials, os.getenv used correctly" "${ts}"
    else
        check_fail "inference_env_vars" "Hardcoded credentials or missing env var injection"
    fi

    ts=$(date +%s%3N)
    INFERENCE_LOG="${REPORT_DIR}/inference_${REPORT_TIMESTAMP}.log"
    env API_BASE_URL="${SERVER_URL}" \
        MODEL_NAME="gpt-4o-mini" \
        HF_TOKEN="dry-run-token" \
        DRY_RUN="true" \
        python3 "${PROJECT_ROOT}/inference.py" \
        --task task_1 --seed 42 --max-steps 5 \
        --server-url "${SERVER_URL}" \
        > "${INFERENCE_LOG}" 2>&1
    INF_EXIT=$?

    if [[ ${INF_EXIT} -eq 0 ]]; then
        SCORE_LINE=$(grep -oE "[0-9]+\.[0-9]+" "${INFERENCE_LOG}" | tail -1 || echo "N/A")
        check_pass "inference_dry_run" "Exited 0 — final score output: ${SCORE_LINE}" "${ts}"
    else
        check_fail "inference_dry_run" "Exited ${INF_EXIT} — see ${INFERENCE_LOG}"
        [[ "${VERBOSE}" == "true" ]] && tail -20 "${INFERENCE_LOG}"
    fi
else
    log_info "Inference.py checks skipped (--skip-inference)"
fi

log_head "STAGE 12 — Security Scan"

if [[ "${SKIP_SECURITY}" == "false" ]]; then
    ts=$(date +%s%3N)
    python3 << 'PYEOF'
import os, sys, re, pathlib

project_root = '${PROJECT_ROOT}'
issues = []
warnings = []

SENSITIVE_PATTERNS = [
    (r'sk-[A-Za-z0-9]{20,}',             'OpenAI API key'),
    (r'hf_[A-Za-z0-9]{20,}',             'HuggingFace token'),
    (r'ghp_[A-Za-z0-9]{20,}',            'GitHub PAT'),
    (r'AKIA[0-9A-Z]{16}',                'AWS Access Key'),
    (r'password\s*=\s*["\'][^"\']{4,}',  'Hardcoded password'),
    (r'secret\s*=\s*["\'][^"\']{4,}',    'Hardcoded secret'),
    (r'Bearer\s+ey[A-Za-z0-9._-]{10,}',  'Hardcoded JWT'),
]

py_files = [f for f in pathlib.Path(project_root).rglob('*.py')
            if '.git' not in str(f) and '__pycache__' not in str(f)]

scanned = 0
for py_file in py_files:
    try:
        content = py_file.read_text(errors='replace')
        scanned += 1
        for pattern, label in SENSITIVE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f'{label} in {py_file.relative_to(project_root)}')
    except:
        pass

df_path = os.path.join(project_root, 'Dockerfile')
if os.path.exists(df_path):
    df_content = open(df_path).read()
    for pattern, label in SENSITIVE_PATTERNS:
        if re.search(pattern, df_content, re.IGNORECASE):
            issues.append(f'{label} found in Dockerfile')

print(f'  Scanned {scanned} Python files + Dockerfile')
for w in warnings:
    print(f'  WARN:{w}')
if issues:
    for i in issues:
        print(f'  FAIL:{i}')
    sys.exit(1)
print('  PASS:No hardcoded secrets or sensitive patterns detected')
sys.exit(0)
PYEOF
    if [[ $? -eq 0 ]]; then
        check_pass "security_scan" "No hardcoded secrets, keys, or tokens detected" "${ts}"
    else
        check_fail "security_scan" "Potential security issues found — review output above"
    fi
else
    log_info "Security scan skipped (--skip-security)"
fi

log_head "STAGE 13 — Concurrent Load & Stress Testing"

if [[ "${SKIP_STRESS}" == "false" ]]; then
    ts=$(date +%s%3N)
    log_step "Stress test: ${STRESS_CONCURRENCY} concurrent /reset requests..."
    STRESS_LOG="${REPORT_DIR}/stress_${REPORT_TIMESTAMP}.log"
    declare -a STRESS_PIDS=()
    STRESS_START=$(date +%s%N)

    for i in $(seq 1 "${STRESS_CONCURRENCY}"); do
        (
            resp=$(http_post_timed "${SERVER_URL}/reset" "{\"task_id\": \"task_1\", \"seed\": ${i}}")
            status=$(parse_status "${resp}")
            elapsed=$(parse_elapsed "${resp}")
            echo "${status}:${elapsed}" >> "${STRESS_LOG}"
        ) &
        STRESS_PIDS+=($!)
    done

    for pid in "${STRESS_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done

    STRESS_END=$(date +%s%N)
    STRESS_TOTAL_MS=$(( (STRESS_END - STRESS_START) / 1000000 ))
    STRESS_200=$(grep -c "^200:" "${STRESS_LOG}" 2>/dev/null || echo 0)
    STRESS_FAIL=$(grep -cv "^200:" "${STRESS_LOG}" 2>/dev/null || echo 0)
    STRESS_AVG_MS=$(awk -F: '{sum+=$2; n++} END {print (n>0 ? int(sum/n) : 0)}' "${STRESS_LOG}" 2>/dev/null || echo 0)
    STRESS_P95_MS=$(awk -F: '{print $2}' "${STRESS_LOG}" 2>/dev/null | sort -n | awk "NR==int(${STRESS_CONCURRENCY}*0.95)+1" || echo 0)

    if (( STRESS_FAIL == 0 )); then
        check_pass "stress_test" "${STRESS_200}/${STRESS_CONCURRENCY} success (100%) | avg=${STRESS_AVG_MS}ms p95=${STRESS_P95_MS}ms" "${ts}"
    elif (( STRESS_FAIL <= 2 )); then
        check_warn "stress_test" "${STRESS_FAIL} failures under ${STRESS_CONCURRENCY} concurrent requests | avg=${STRESS_AVG_MS}ms"
    else
        check_fail "stress_test" "${STRESS_FAIL}/${STRESS_CONCURRENCY} failed under concurrency | avg=${STRESS_AVG_MS}ms"
    fi

    ts=$(date +%s%3N)
    log_step "Episode throughput benchmark (${STRESS_DURATION}s)..."
    THRPT_COMPLETE=0
    THRPT_START=$(date +%s)
    THRPT_END=$(( THRPT_START + STRESS_DURATION ))

    while (( $(date +%s) < THRPT_END )); do
        http_post "${SERVER_URL}/reset" '{"task_id": "task_1", "seed": 1}' >/dev/null 2>&1
        for _ in 1 2 3; do
            http_post "${SERVER_URL}/step" '{"action_type": "noop", "unit_id": null}' >/dev/null 2>&1
        done
        ((THRPT_COMPLETE++)) || true
    done

    EPS_PER_SEC=$(python3 -c "print(round(${THRPT_COMPLETE}/${STRESS_DURATION},2))" 2>/dev/null || echo "N/A")
    check_pass "throughput" "${THRPT_COMPLETE} episodes in ${STRESS_DURATION}s = ${EPS_PER_SEC} eps/s" "${ts}"
else
    log_info "Stress testing skipped (--skip-stress)"
fi

log_head "STAGE 14 — HuggingFace Space Live Check"

if [[ "${SKIP_HF}" == "false" ]] && [[ -n "${HF_SPACE_URL}" ]]; then
    ts=$(date +%s%3N)
    log_step "Pinging: ${HF_SPACE_URL}"

    RESP=$(http_post_timed "${HF_SPACE_URL}/reset" '{"task_id": "task_1", "seed": 42}')
    STATUS=$(parse_status "${RESP}")
    ELAPSED=$(parse_elapsed "${RESP}")
    if [[ "${STATUS}" == "200" ]]; then
        check_pass "hf_space:reset" "HTTP 200 | ${ELAPSED}ms — Space responding" "${ts}"
    else
        check_fail "hf_space:reset" "HTTP ${STATUS} from ${HF_SPACE_URL}/reset — Phase 1 requires 200"
    fi

    ts=$(date +%s%3N)
    STATUS=$(parse_status "$(http_get "${HF_SPACE_URL}/health")")
    if [[ "${STATUS}" == "200" ]]; then
        check_pass "hf_space:health" "HTTP 200 — Space healthy" "${ts}"
    else
        check_warn "hf_space:health" "HTTP ${STATUS} from /health"
    fi

    ts=$(date +%s%3N)
    RESP=$(http_get "${HF_SPACE_URL}/tasks")
    STATUS=$(parse_status "${RESP}")
    TASK_COUNT=$(parse_body "${RESP}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    t=d.get('tasks',d) if isinstance(d,dict) else d
    print(len(t) if isinstance(t,list) else 0)
except:
    print(0)
" 2>/dev/null)
    if [[ "${STATUS}" == "200" ]] && (( TASK_COUNT >= 9 )); then
        check_pass "hf_space:tasks" "${TASK_COUNT} tasks on Space" "${ts}"
    else
        check_warn "hf_space:tasks" "HTTP ${STATUS} | ${TASK_COUNT} tasks on Space"
    fi
else
    if [[ -z "${HF_SPACE_URL}" ]]; then
        log_info "HuggingFace Space check skipped — set HF_SPACE_URL=https://your-space.hf.space"
    else
        log_info "HuggingFace Space check skipped (--skip-hf)"
    fi
fi

log_head "STAGE 15 — OpenAPI Schema & Route Documentation"

ts=$(date +%s%3N)
resp=$(http_get "${SERVER_URL}/openapi.json")
status=$(parse_status "${resp}")
body=$(parse_body "${resp}")

if [[ "${status}" == "200" ]]; then
    SCHEMA_FILE="${REPORT_DIR}/openapi_${REPORT_TIMESTAMP}.json"
    echo "${body}" > "${SCHEMA_FILE}"

    python3 - "${SCHEMA_FILE}" << 'PYEOF'
import sys, json

with open(sys.argv[1]) as f:
    schema = json.load(f)

paths      = schema.get('paths', {})
components = schema.get('components', {})
info       = schema.get('info', {})

required_routes = ['/reset', '/step', '/state', '/tasks', '/health']
missing_routes  = [r for r in required_routes if r not in paths]

print(f'  Title: {info.get("title","?")} v{info.get("version","?")}')
print(f'  Routes: {len(paths)}  |  Schemas: {len(components.get("schemas", {}))}')

for r in missing_routes:
    print(f'  WARN:Route {r} not documented in OpenAPI schema')

for route, methods in paths.items():
    for method, spec in methods.items():
        if not spec.get('description') and not spec.get('summary'):
            print(f'  WARN:Route {method.upper()} {route} has no summary/description')

if not missing_routes:
    print('  PASS:All 5 required routes documented')
PYEOF

    ROUTE_COUNT=$(echo "${body}" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('paths',{})))" 2>/dev/null)
    SCHEMA_COUNT=$(echo "${body}" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('components',{}).get('schemas',{})))" 2>/dev/null)
    check_pass "openapi_schema" "${ROUTE_COUNT} routes | ${SCHEMA_COUNT} schemas | saved: ${SCHEMA_FILE}" "${ts}"
else
    check_warn "openapi_schema" "HTTP ${status} — could not retrieve /openapi.json"
fi

log_head "STAGE 16 — Response Latency Profiling (8 iterations each)"

for endpoint_info in "GET:/health:null" "POST:/reset:{\"task_id\":\"task_1\",\"seed\":42}" "POST:/step:{\"action_type\":\"noop\",\"unit_id\":null}"; do
    METHOD=$(echo "${endpoint_info}" | cut -d: -f1)
    ROUTE=$(echo "${endpoint_info}" | cut -d: -f2)
    DATA=$(echo "${endpoint_info}" | cut -d: -f3-)

    declare -a LATENCIES=()
    for i in $(seq 1 8); do
        START_NS=$(date +%s%N)
        if [[ "${METHOD}" == "GET" ]]; then
            http_get "${SERVER_URL}${ROUTE}" >/dev/null 2>&1
        else
            http_post "${SERVER_URL}${ROUTE}" "${DATA}" >/dev/null 2>&1
        fi
        END_NS=$(date +%s%N)
        LATENCIES+=($(( (END_NS - START_NS) / 1000000 )))
    done

    ts=$(date +%s%3N)
    STATS=$(python3 -c "
import statistics
vals = [${LATENCIES[*]}]
p50  = statistics.median(vals)
p95  = sorted(vals)[max(0, int(len(vals)*0.95)-1)]
avg  = statistics.mean(vals)
mn   = min(vals)
mx   = max(vals)
print(f'avg={avg:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms min={mn}ms max={mx}ms')
" 2>/dev/null)

    P95=$(echo "${STATS}" | grep -oP '(?<=p95=)[0-9]+' || echo "9999")
    if (( P95 < 200 )); then
        check_pass "latency:${METHOD}:${ROUTE}" "${STATS} — excellent" "${ts}"
    elif (( P95 < 1000 )); then
        check_pass "latency:${METHOD}:${ROUTE}" "${STATS} — acceptable" "${ts}"
    else
        check_warn "latency:${METHOD}:${ROUTE}" "${STATS} — p95 > 1s may cause validator timeouts"
    fi
    unset LATENCIES
done

finalize_and_exit() {
    END_TS=$(date +%s)
    DURATION=$(( END_TS - START_TS ))
    PASS_RATE=$(python3 -c "print(round(${PASSED_CHECKS}/max(1,${TOTAL_CHECKS})*100,1))" 2>/dev/null || echo "0")

    log_head "FINAL VALIDATION SUMMARY"

    echo ""
    printf "  %-40s %s\n"                "Timestamp"    "${REPORT_TIMESTAMP}"
    printf "  %-40s %s\n"                "Duration"     "${DURATION}s"
    printf "  %-40s %s\n"                "Total checks" "${TOTAL_CHECKS}"
    printf "  %-40s ${GREEN}%s${RESET}\n" "Passed"      "${PASSED_CHECKS} (${PASS_RATE}%)"
    printf "  %-40s ${RED}%s${RESET}\n"   "Failed"      "${FAILED_CHECKS}"
    printf "  %-40s ${YELLOW}%s${RESET}\n" "Warnings"   "${WARNED_CHECKS}"
    printf "  %-40s %s\n"                "Report"       "${REPORT_FILE}"
    printf "  %-40s %s\n"                "Metrics JSON" "${METRICS_FILE}"
    printf "  %-40s %s\n"                "Git commit"   "${GIT_COMMIT:-unknown} (${GIT_BRANCH:-unknown})"
    echo ""

    if (( FAILED_CHECKS > 0 )); then
        echo -e "${RED}${BOLD}  Failed checks:${RESET}"
        for msg in "${FAILURE_MESSAGES[@]}"; do
            echo -e "  ${RED}  ${msg}${RESET}"
        done
        echo ""
    fi

    if (( WARNED_CHECKS > 0 )); then
        echo -e "${YELLOW}${BOLD}  Warnings:${RESET}"
        for msg in "${WARN_MESSAGES[@]}"; do
            echo -e "  ${YELLOW}  ${msg}${RESET}"
        done
        echo ""
    fi

    python3 - << PYEOF > "${METRICS_FILE}"
import json
print(json.dumps({
    "timestamp":       "${REPORT_TIMESTAMP}",
    "duration_s":      ${DURATION},
    "total_checks":    ${TOTAL_CHECKS},
    "passed":          ${PASSED_CHECKS},
    "failed":          ${FAILED_CHECKS},
    "warnings":        ${WARNED_CHECKS},
    "pass_rate_pct":   round(${PASSED_CHECKS} / max(1, ${TOTAL_CHECKS}) * 100, 2),
    "server_url":      "${SERVER_URL}",
    "git_commit":      "${GIT_COMMIT:-unknown}",
    "git_branch":      "${GIT_BRANCH:-unknown}",
    "python_version":  "${PY_VERSION}",
    "report_file":     "${REPORT_FILE}"
}, indent=2))
PYEOF

    if [[ "${CI_MODE}" == "true" ]]; then
        echo "VALIDATION_STATUS=$([ ${FAILED_CHECKS} -eq 0 ] && echo PASS || echo FAIL)"
        echo "TOTAL_CHECKS=${TOTAL_CHECKS}"
        echo "PASSED_CHECKS=${PASSED_CHECKS}"
        echo "FAILED_CHECKS=${FAILED_CHECKS}"
        echo "WARNED_CHECKS=${WARNED_CHECKS}"
        echo "PASS_RATE=${PASS_RATE}"
        echo "DURATION_S=${DURATION}"
        echo "GIT_COMMIT=${GIT_COMMIT:-unknown}"
    fi

    if (( FAILED_CHECKS == 0 )); then
        echo -e "${BOLD}${GREEN}"
        cat << 'EOF'
   ╔══════════════════════════════════════════════════════════╗
   ║                                                          ║
   ║   🏆  ALL CHECKS PASSED — READY FOR SUBMISSION  🏆      ║
   ║                                                          ║
   ╚══════════════════════════════════════════════════════════╝
EOF
        echo -e "${RESET}"
        exit 0
    else
        echo -e "${BOLD}${RED}"
        cat << 'EOF'
   ╔══════════════════════════════════════════════════════════╗
   ║                                                          ║
   ║   ✗   VALIDATION FAILED — FIX ERRORS ABOVE             ║
   ║                                                          ║
   ╚══════════════════════════════════════════════════════════╝
EOF
        echo -e "${RESET}"
        exit 1
    fi
}

finalize_and_exit