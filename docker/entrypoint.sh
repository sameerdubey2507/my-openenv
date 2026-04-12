#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[1;33m'
BLU='\033[0;34m'
CYN='\033[0;36m'
NC='\033[0m'
log()  { echo -e "${BLU}[EMERGI-ENV]${NC} $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"; }
ok()   { echo -e "${GRN}[EMERGI-ENV]${NC} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ✅ $*"; }
warn() { echo -e "${YEL}[EMERGI-ENV]${NC} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ⚠️  $*"; }
fail() { echo -e "${RED}[EMERGI-ENV]${NC} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ❌ $*" >&2; }
echo -e "${CYN}"
cat <<'BANNER'
  ███████╗███╗   ███╗███████╗██████╗  ██████╗ ██╗      ███████╗███╗   ██╗██╗   ██╗
  ██╔════╝████╗ ████║██╔════╝██╔══██╗██╔════╝ ██║      ██╔════╝████╗  ██║██║   ██║
  █████╗  ██╔████╔██║█████╗  ██████╔╝██║  ███╗██║█████╗█████╗  ██╔██╗ ██║██║   ██║
  ██╔══╝  ██║╚██╔╝██║██╔══╝  ██╔══██╗██║   ██║██║╚════╝██╔══╝  ██║╚██╗██║╚██╗ ██╔╝
  ███████╗██║ ╚═╝ ██║███████╗██║  ██║╚██████╔╝██║      ███████╗██║ ╚████║ ╚████╔╝
  ╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝      ╚══════╝╚═╝  ╚═══╝  ╚═══╝
  Emergency Medical Intelligence & Resource Governance Environment  |  v${APP_VERSION:-1.4.0}
BANNER
echo -e "${NC}"
log "Starting EMERGI-ENV | env=${ENVIRONMENT:-production} | port=${PORT:-7860} | commit=${GIT_COMMIT:-unknown}"
_shutdown() {
    warn "Caught shutdown signal — initiating graceful shutdown..."
    if [ -n "${APP_PID:-}" ]; then
        kill -SIGTERM "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
    log "EMERGI-ENV shutdown complete."
    exit 0
}
trap _shutdown SIGTERM SIGINT SIGQUIT
log "Checking required directories..."
for dir in /app/logs /app/tmp /var/log/emergi-env /var/run/emergi-env; do
    if [ ! -d "$dir" ]; then
        warn "Directory $dir missing — creating..."
        mkdir -p "$dir" || { fail "Cannot create $dir"; exit 1; }
    fi
done
ok "Directory structure OK"
log "Validating Python environment..."
python3 -c "
import sys
required = ['fastapi', 'uvicorn', 'pydantic', 'numpy']
missing = []
for mod in required:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f'MISSING MODULES: {missing}', file=sys.stderr)
    sys.exit(1)
print(f'Python {sys.version} — all required modules importable')
" || { fail "Python environment check failed"; exit 1; }
ok "Python environment OK"
log "Validating data files..."
DATA_FILES=(
    "peopledataset.json"
    "hospitalprofiles.json"
    "templateincident.json"
    "trafficpattern.json"
    "demandhistory.json"
    "survivalparam.json"
)
MISSING_DATA=0
for f in "${DATA_FILES[@]}"; do
    fpath="/app/data/${f}"
    if [ ! -f "$fpath" ]; then
        fail "Missing data file: $fpath"
        MISSING_DATA=$((MISSING_DATA + 1))
    else
        python3 -c "import json; json.load(open('$fpath'))" 2>/dev/null || {
            fail "Invalid JSON: $fpath"; MISSING_DATA=$((MISSING_DATA + 1));
        }
    fi
done
if [ "$MISSING_DATA" -gt 0 ]; then
    fail "${MISSING_DATA} data file(s) are missing or corrupt. Cannot start."
    exit 1
fi
ok "All data files valid"
if [ ! -f "/app/openenv.yaml" ]; then
    fail "openenv.yaml not found — required for OpenEnv Phase 1 validation"
    exit 1
fi
ok "openenv.yaml present"
log "Importing server.app:app..."
python3 -c "from server.app import app; print(f'FastAPI app loaded: {app.title}')" || {
    fail "server.app:app failed to import — check for syntax errors"
    exit 1
}
ok "FastAPI application imported successfully"
TARGET_PORT="${PORT:-7860}"
log "Checking port ${TARGET_PORT} availability..."
if nc -z localhost "${TARGET_PORT}" 2>/dev/null; then
    warn "Port ${TARGET_PORT} is already in use — another process may be running"
else
    ok "Port ${TARGET_PORT} is available"
fi
PID_FILE="/var/run/emergi-env/emergi-env.pid"
log "─────────────────────────────────────────────────────"
log "Runtime Configuration:"
log "  APP_VERSION     : ${APP_VERSION:-1.4.0}"
log "  ENVIRONMENT     : ${ENVIRONMENT:-production}"
log "  HOST            : ${HOST:-0.0.0.0}"
log "  PORT            : ${PORT:-7860}"
log "  WORKERS         : ${WORKERS:-1}"
log "  LOG_LEVEL       : ${LOG_LEVEL:-info}"
log "  AGENT_MODE      : ${AGENT_MODE:-hybrid}"
log "  MODEL_NAME      : ${MODEL_NAME:-https://api.groq.com/openai/v1}"
log "  API_BASE_URL    : ${API_BASE_URL:-http://localhost:7860}"
log "  GIT_COMMIT      : ${GIT_COMMIT:-unknown}"
log "─────────────────────────────────────────────────────"
log "🚀 Launching EMERGI-ENV server on port ${PORT:-7860}..."
exec "$@" &
APP_PID=$!
echo "${APP_PID}" > "${PID_FILE}" 2>/dev/null || true
ok "Server process started (PID=${APP_PID})"
log "Waiting for application to become healthy..."
HEALTH_URL="http://localhost:${PORT:-7860}/health"
MAX_WAIT=60
WAITED=0
SLEEP_INTERVAL=3
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    HTTP_STATUS=$(curl --silent --output /dev/null --write-out "%{http_code}" \
        --max-time 5 "${HEALTH_URL}" 2>/dev/null || echo "000")
    if [ "${HTTP_STATUS}" = "200" ]; then
        ok "Health check passed (HTTP ${HTTP_STATUS}) after ${WAITED}s"
        break
    fi
    log "Health check pending (HTTP ${HTTP_STATUS}, ${WAITED}s elapsed)..."
    sleep "${SLEEP_INTERVAL}"
    WAITED=$((WAITED + SLEEP_INTERVAL))
done
if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    fail "Server did not become healthy within ${MAX_WAIT}s"
    fail "Last health check URL: ${HEALTH_URL}"
    warn "Continuing anyway — container runtime health check will determine fate"
fi
ok "EMERGI-ENV is LIVE — Simulating India 108/112 EMS Network"
log "HuggingFace Space: https://huggingface.co/spaces/sameerdubey25/emergi-env"
wait "${APP_PID}"
EXIT_CODE=$?
if [ "${EXIT_CODE}" -ne 0 ]; then
    fail "Server exited with code ${EXIT_CODE}"
fi
log "Server process exited (code=${EXIT_CODE})"
exit "${EXIT_CODE}"