
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend


COPY static/package.json static/package-lock.json ./
RUN npm ci --prefer-offline


COPY static/ ./
RUN npm run build


FROM python:3.11-slim-bookworm AS builder

ARG APP_USER=emergi
ARG APP_UID=10001
ARG APP_GID=10001

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /build/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel \
    --no-cache-dir \
    --wheel-dir /wheels \
    --requirement /build/requirements.txt

RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel && \
    /venv/bin/pip install \
    --no-cache-dir \
    --no-index \
    --find-links /wheels \
    --requirement /build/requirements.txt

# ── Stage 3 : Data validator ──────────────────────────────────────
FROM python:3.11-slim-bookworm AS data-validator

COPY data/ /validate/data/
RUN python3 - <<'PYEOF'
import json, sys, os
required_files = [
    "peopledataset.json",
    "hospitalprofiles.json",
    "templateincident.json",
    "trafficpattern.json",
    "demandhistory.json",
    "survivalparam.json",
]
base = "/validate/data"
errors = []
for fname in required_files:
    fpath = os.path.join(base, fname)
    if not os.path.exists(fpath):
        errors.append(f"MISSING: {fname}")
        continue
    try:
        with open(fpath) as f:
            data = json.load(f)
        size = os.path.getsize(fpath)
        print(f"  OK  {fname:40s}  ({size:,} bytes)")
    except json.JSONDecodeError as e:
        errors.append(f"INVALID JSON in {fname}: {e}")
if errors:
    print("\nData validation FAILED:")
    for e in errors:
        print(f"   {e}")
    sys.exit(1)
print(f"\nAll {len(required_files)} data files validated.")
PYEOF

# ── Stage 4 : Production ──────────────────────────────────────────
FROM python:3.11-slim-bookworm AS production

LABEL org.opencontainers.image.title="EMERGI-ENV"
LABEL org.opencontainers.image.description="Emergency Medical Intelligence & Resource Governance Environment"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ARG PORT=7860
ARG WORKERS=1
ARG APP_USER=emergi
ARG APP_UID=10001
ARG APP_GID=10001
ARG LOG_LEVEL=info
ARG ENVIRONMENT=production
ARG APP_VERSION=2.0.0
ARG GIT_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata \
    PATH="/venv/bin:$PATH" \
    VIRTUAL_ENV="/venv" \
    PORT=${PORT} \
    HOST="0.0.0.0" \
    WORKERS=${WORKERS} \
    LOG_LEVEL=${LOG_LEVEL} \
    ENVIRONMENT=${ENVIRONMENT} \
    APP_VERSION=${APP_VERSION} \
    GIT_COMMIT=${GIT_COMMIT} \
    APP_HOME="/app" \
    LOG_DIR="/var/log/emergi-env" \
    RUN_DIR="/var/run/emergi-env" \
    TMP_DIR="/app/tmp" \
    SERVER_URL="http://localhost:7860" \
    API_BASE_URL="https://router.huggingface.co/v1" \
    MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" \
    AGENT_MODE="hybrid" \
    MAX_LLM_TOKENS="1000" \
    LLM_TEMPERATURE="0.2" \
    REQUEST_TIMEOUT="30.0" \
    LLM_TIMEOUT="45.0" \
    UVICORN_LOOP="auto" \
    UVICORN_BACKLOG="2048" \
    SPACE_ID="emergi-env/emergi-env"

# System deps (curl + netcat for healthcheck/entrypoint)
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    tzdata curl netcat-traditional jq procps && \
    ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && \
    echo "Asia/Kolkata" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Non-root user
RUN groupadd --gid ${APP_GID} ${APP_USER} && \
    useradd \
    --uid ${APP_UID} \
    --gid ${APP_GID} \
    --no-create-home \
    --shell /sbin/nologin \
    ${APP_USER}

# Directory structure
RUN mkdir -p \
    /app/server \
    /app/data \
    /app/scripts \
    /app/logs \
    /app/tmp \
    /app/static/dist \
    /var/log/emergi-env \
    /var/run/emergi-env && \
    chown -R ${APP_USER}:${APP_USER} \
    /app /var/log/emergi-env /var/run/emergi-env

WORKDIR /app

# Python venv from builder
COPY --from=builder   --chown=${APP_USER}:${APP_USER} /venv /venv

# Validated data files
COPY --from=data-validator --chown=${APP_USER}:${APP_USER} /validate/data /app/data

# Built React frontend  ← this is the key fix
COPY --from=frontend-builder --chown=${APP_USER}:${APP_USER} /frontend/dist /app/static/dist

# Application source
COPY --chown=${APP_USER}:${APP_USER} server/      /app/server/

COPY --chown=${APP_USER}:${APP_USER} docs /app/docs
COPY --chown=${APP_USER}:${APP_USER} scripts/     /app/scripts/
COPY --chown=${APP_USER}:${APP_USER} inference.py /app/inference.py
COPY --chown=${APP_USER}:${APP_USER} openenv.yaml /app/openenv.yaml
COPY --chown=${APP_USER}:${APP_USER} README.md    /app/README.md

# Copy emergi_docs.html if it exists (optional — fallback handled in main.py)
COPY --chown=${APP_USER}:${APP_USER} docker/entrypoint.sh    /entrypoint.sh
COPY --chown=${APP_USER}:${APP_USER} docker/gunicorn.conf.py /app/gunicorn.conf.py

RUN chmod +x /entrypoint.sh && \
    chmod +x /app/scripts/*.sh 2>/dev/null || true && \
    python -m compileall -q /app/server/ 2>/dev/null || true && \
    chown -R ${APP_USER}:${APP_USER} /app /var/log/emergi-env /var/run/emergi-env && \
    chmod -R 755 /app && \
    chmod -R 777 /app/logs /app/tmp

USER ${APP_USER}

EXPOSE ${PORT}

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl --silent --fail --max-time 8 \
    "http://localhost:${PORT}/health" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status') in ['ok','degraded'] else 1)" \
    || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "server.app:app", \
    "--host", "0.0.0.0", \
    "--port", "7860", \
    "--workers", "1", \
    "--loop", "auto", \
    "--http", "auto", \
    "--log-level", "info", \
    "--access-log", \
    "--use-colors", \
    "--timeout-graceful-shutdown", "30", \
    "--backlog", "2048"]