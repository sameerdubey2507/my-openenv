#!/usr/bin/env bash

set -euo pipefail

if [[ "${CI:-false}" == "true" ]] || [[ "${TERM:-dumb}" == "dumb" ]]; then
    RED="" GREEN="" YELLOW="" BLUE="" CYAN="" MAGENTA="" BOLD="" DIM="" RESET="" BLINK=""
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
    BLINK='\033[5m'
fi

SERVER_URL="${SERVER_URL:-http://localhost:7860}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
API_BASE_URL="${API_BASE_URL:-https://api.openai.com/v1}"
HF_TOKEN="${HF_TOKEN:-}"
REPORT_DIR="${REPORT_DIR:-./reports/baseline}"
MAX_STEPS_EASY="${MAX_STEPS_EASY:-20}"
MAX_STEPS_MEDIUM="${MAX_STEPS_MEDIUM:-40}"
MAX_STEPS_HARD="${MAX_STEPS_HARD:-60}"
SEEDS="${SEEDS:-42,123,777,2024,9999}"
TASKS="${TASKS:-task_1,task_2,task_3,task_4,task_5,task_6,task_7,task_8,task_9}"
WARMUP_EPISODES="${WARMUP_EPISODES:-3}"
HTTP_TIMEOUT=45
PARALLEL_TASKS=false
EXPORT_CSV=true
EXPORT_JSON=true
EXPORT_MARKDOWN=true
VERBOSE=false
COMPARE_BASELINE=true
SHOW_PROGRESS=true
FAIL_BELOW="${FAIL_BELOW:-0.0}"
AGENT_MODE="${AGENT_MODE:-random}"
SKIP_WARMUP=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_DIR="${REPORT_DIR}/${RUN_TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-url)      SERVER_URL="$2";        shift ;;
        --model)           MODEL_NAME="$2";        shift ;;
        --api-base)        API_BASE_URL="$2";      shift ;;
        --hf-token)        HF_TOKEN="$2";          shift ;;
        --report-dir)      REPORT_DIR="$2";        shift ;;
        --seeds)           SEEDS="$2";             shift ;;
        --tasks)           TASKS="$2";             shift ;;
        --max-steps-easy)  MAX_STEPS_EASY="$2";    shift ;;
        --max-steps-medium)MAX_STEPS_MEDIUM="$2";  shift ;;
        --max-steps-hard)  MAX_STEPS_HARD="$2";    shift ;;
        --warmup)          WARMUP_EPISODES="$2";   shift ;;
        --agent-mode)      AGENT_MODE="$2";        shift ;;
        --fail-below)      FAIL_BELOW="$2";        shift ;;
        --parallel)        PARALLEL_TASKS=true     ;;
        --no-csv)          EXPORT_CSV=false        ;;
        --no-json)         EXPORT_JSON=false       ;;
        --no-markdown)     EXPORT_MARKDOWN=false   ;;
        --verbose)         VERBOSE=true            ;;
        --no-compare)      COMPARE_BASELINE=false  ;;
        --no-progress)     SHOW_PROGRESS=false     ;;
        --skip-warmup)     SKIP_WARMUP=true        ;;
        -h|--help)
            echo "EMERGI-ENV Baseline Runner"
            echo ""
            echo "Usage: bash scripts/run_baseline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --server-url URL         Server URL (default: http://localhost:7860)"
            echo "  --model NAME             Model name for inference.py"
            echo "  --api-base URL           API base URL"
            echo "  --hf-token TOKEN         HuggingFace token"
            echo "  --seeds LIST             Comma-separated seeds (default: 42,123,777,2024,9999)"
            echo "  --tasks LIST             Comma-separated task IDs (default: all 9)"
            echo "  --max-steps-easy N       Max steps for easy tasks (default: 20)"
            echo "  --max-steps-medium N     Max steps for medium tasks (default: 40)"
            echo "  --max-steps-hard N       Max steps for hard tasks (default: 60)"
            echo "  --warmup N               Warmup episodes before recording (default: 3)"
            echo "  --agent-mode MODE        Agent strategy: random|greedy|protocol|llm"
            echo "  --fail-below SCORE       Exit 1 if any task mean < this threshold"
            echo "  --parallel               Run tasks in parallel (experimental)"
            echo "  --no-csv/json/markdown   Disable specific export formats"
            echo "  --verbose                Show per-step details"
            echo "  --no-compare             Skip baseline comparison"
            echo "  --skip-warmup            Skip warmup phase"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${REPORT_DIR}"
MAIN_LOG="${REPORT_DIR}/run_baseline.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

declare -A TASK_DIFFICULTY=(
    [task_1]="easy"   [task_2]="easy"   [task_3]="easy"
    [task_4]="medium" [task_5]="medium" [task_6]="medium"
    [task_7]="hard"   [task_8]="hard"   [task_9]="hard"
)

declare -A TASK_NAME=(
    [task_1]="Single Call Triage & Dispatch"
    [task_2]="Hospital Route Selection"
    [task_3]="Unit Type Matching"
    [task_4]="Multi-Incident Queue"
    [task_5]="Dynamic Rerouting"
    [task_6]="Fleet Pre-positioning"
    [task_7]="Mass Casualty Incident"
    [task_8]="Inter-hospital Transfer Cascade"
    [task_9]="City-wide Surge Response"
)

declare -A TASK_BASELINE=(
    [task_1]="0.61" [task_2]="0.72" [task_3]="0.68"
    [task_4]="0.44" [task_5]="0.38" [task_6]="0.42"
    [task_7]="0.29" [task_8]="0.24" [task_9]="0.17"
)

declare -A TASK_MAX_STEPS=(
    [task_1]="${MAX_STEPS_EASY}"   [task_2]="${MAX_STEPS_EASY}"   [task_3]="${MAX_STEPS_EASY}"
    [task_4]="${MAX_STEPS_MEDIUM}" [task_5]="${MAX_STEPS_MEDIUM}" [task_6]="${MAX_STEPS_MEDIUM}"
    [task_7]="${MAX_STEPS_HARD}"   [task_8]="${MAX_STEPS_HARD}"   [task_9]="${MAX_STEPS_HARD}"
)

declare -A TASK_SCORES_MEAN=()
declare -A TASK_SCORES_STD=()
declare -A TASK_SCORES_MIN=()
declare -A TASK_SCORES_MAX=()
declare -A TASK_SCORES_MEDIAN=()
declare -A TASK_EPISODES_COMPLETED=()
declare -A TASK_STEPS_MEAN=()
declare -A TASK_PROTOCOL_COMPLIANCE=()
declare -A TASK_SURVIVAL_RATE=()
declare -A TASK_DIVERSION_HITS=()
declare -A TASK_MUTUAL_AID_CALLS=()
declare -A TASK_FATIGUE_EVENTS=()
declare -A TASK_COMM_FAILURES=()
declare -A TASK_TOTAL_REWARD_PER_STEP=()
declare -A TASK_WALL_TIME_S=()

IFS=',' read -ra SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -ra TASK_ARRAY <<< "${TASKS}"
NUM_SEEDS=${#SEED_ARRAY[@]}
NUM_TASKS=${#TASK_ARRAY[@]}

http_post() {
    local url="$1" data="${2:-{}}"
    curl -s -w "\n__STATUS__%{http_code}" \
        --max-time "${HTTP_TIMEOUT}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "${data}" \
        "${url}" 2>/dev/null || echo -e "\n__STATUS__000"
}

http_get() {
    local url="$1"
    curl -s -w "\n__STATUS__%{http_code}" \
        --max-time "${HTTP_TIMEOUT}" \
        -H "Content-Type: application/json" \
        "${url}" 2>/dev/null || echo -e "\n__STATUS__000"
}

parse_status() { echo "$1" | grep '__STATUS__' | sed 's/__STATUS__//' | tr -d '[:space:]'; }
parse_body()   { echo "$1" | sed '/__STATUS__/d'; }

log_ts()   { echo -e "${DIM}[$(date '+%H:%M:%S')]${RESET} $*"; }
log_head() {
    echo -e "\n${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${BLUE}║  $*$(printf '%*s' $((60 - ${#1})) '')║${RESET}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════╝${RESET}"
}
log_sub()  { echo -e "\n  ${BOLD}${CYAN}── $* ──${RESET}"; }
log_ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
log_err()  { echo -e "  ${RED}✗${RESET}  $*"; }
log_warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
log_info() { echo -e "  ${DIM}ℹ  $*${RESET}"; }
log_metric(){ echo -e "  ${MAGENTA}◆${RESET}  ${BOLD}$1${RESET}: $2"; }

progress_bar() {
    local current="$1" total="$2" label="${3:-}"
    if [[ "${SHOW_PROGRESS}" == "true" ]]; then
        local pct=$(( current * 100 / total ))
        local filled=$(( current * 40 / total ))
        local bar=""
        for ((i=0; i<filled; i++));   do bar+="█"; done
        for ((i=filled; i<40; i++));  do bar+="░"; done
        printf "\r  ${CYAN}[${bar}]${RESET} ${BOLD}%3d%%${RESET} %s" "${pct}" "${label}"
    fi
}

py_stats() {
    python3 - "$@" << 'PYEOF'
import sys, json, math, statistics

scores = [float(x) for x in sys.argv[1:] if x.replace('.','',1).replace('-','',1).lstrip('-').replace('.','',1).isdigit() or (x.replace('-','',1).replace('.','',1).isdigit())]
scores = []
for x in sys.argv[1:]:
    try:
        scores.append(float(x))
    except:
        pass

if not scores:
    print("mean=0.0 std=0.0 min=0.0 max=0.0 median=0.0")
    sys.exit(0)

mean_v    = statistics.mean(scores)
std_v     = statistics.pstdev(scores) if len(scores) > 1 else 0.0
min_v     = min(scores)
max_v     = max(scores)
median_v  = statistics.median(scores)
print(f"mean={mean_v:.6f} std={std_v:.6f} min={min_v:.6f} max={max_v:.6f} median={median_v:.6f}")
PYEOF
}

extract_py() {
    local json_str="$1" key="$2" default="${3:-0.0}"
    echo "${json_str}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    keys = '${key}'.split('.')
    v = d
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k, ${default})
        elif isinstance(v, list):
            v = v[int(k)] if int(k) < len(v) else ${default}
        else:
            v = ${default}
            break
    print(v)
except:
    print(${default})
" 2>/dev/null || echo "${default}"
}

agent_action() {
    local task_id="$1" obs_json="$2" step_num="$3" seed="$4"
    python3 - "${task_id}" "${step_num}" "${seed}" << PYEOF
import sys, json, random, math

task_id  = sys.argv[1]
step_num = int(sys.argv[2])
seed_val = int(sys.argv[3])

rng = random.Random(seed_val * 1000 + step_num * 13 + hash(task_id) % 997)

obs_raw = ""
for line in sys.stdin:
    obs_raw += line
try:
    obs = json.loads(obs_raw) if obs_raw.strip() else {}
except:
    obs = {}

mode = "${AGENT_MODE}"

def get_highest_priority_incident(obs):
    iq = obs.get("incident_queue", [])
    if not iq:
        return None
    priority_map = {"P1": 3, "P2": 2, "P3": 1}
    iq_sorted = sorted(iq, key=lambda x: priority_map.get(x.get("priority", "P3"), 0), reverse=True)
    return iq_sorted[0] if iq_sorted else None

def get_available_unit(obs):
    fleet = obs.get("fleet_status", [])
    available = [u for u in fleet if u.get("status") == "available"]
    return available[0].get("unit_id") if available else None

def get_best_hospital(obs, incident):
    hospitals = obs.get("hospital_network", [])
    available_h = [h for h in hospitals if not h.get("on_diversion", False) and h.get("available_beds", 0) > 0]
    if not available_h:
        available_h = hospitals
    if not available_h:
        return "H1"
    return available_h[0].get("hospital_id", "H1")

def unit_type_for_priority(priority, condition=""):
    condition_lower = condition.lower()
    if "stemi" in condition_lower or "cardiac" in condition_lower or "stroke" in condition_lower:
        return "MICU"
    if "trauma" in condition_lower or "polytrauma" in condition_lower or "mci" in condition_lower:
        return "ALS"
    if priority == "P1":
        return rng.choice(["ALS", "MICU"])
    if priority == "P2":
        return "ALS"
    return "BLS"

if mode == "noop":
    action = {"action_type": "noop", "unit_id": None}

elif mode == "random":
    action_types = ["dispatch", "noop", "reroute"] if step_num % 3 != 0 else ["noop"]
    chosen = rng.choice(action_types)
    if chosen == "dispatch":
        incident = get_highest_priority_incident(obs)
        unit     = get_available_unit(obs)
        if incident and unit:
            hospital = get_best_hospital(obs, incident)
            action = {
                "action_type": "dispatch",
                "unit_id": unit,
                "incident_id": incident.get("incident_id", "I1"),
                "destination_hospital": hospital,
                "priority_override": None
            }
        else:
            action = {"action_type": "noop", "unit_id": None}
    else:
        action = {"action_type": "noop", "unit_id": None}

elif mode == "greedy":
    incident = get_highest_priority_incident(obs)
    unit     = get_available_unit(obs)
    if incident and unit:
        hospital = get_best_hospital(obs, incident)
        action = {
            "action_type": "dispatch",
            "unit_id": unit,
            "incident_id": incident.get("incident_id", "I1"),
            "destination_hospital": hospital,
            "priority_override": None
        }
    else:
        fleet = obs.get("fleet_status", [])
        repositionable = [u for u in fleet if u.get("status") == "available"]
        if repositionable and task_id == "task_6":
            forecast = obs.get("demand_forecast", {})
            zones = list(forecast.keys()) if isinstance(forecast, dict) else []
            if zones:
                best_zone = max(zones, key=lambda z: forecast.get(z, 0))
                action = {
                    "action_type": "reposition",
                    "unit_id": repositionable[0].get("unit_id"),
                    "target_zone": best_zone
                }
            else:
                action = {"action_type": "noop", "unit_id": None}
        else:
            action = {"action_type": "noop", "unit_id": None}

elif mode == "protocol":
    incident = get_highest_priority_incident(obs)
    unit     = get_available_unit(obs)

    if task_id == "task_7":
        victims = obs.get("incident_queue", [])
        mci_victims = [v for v in victims if not v.get("tagged", False)]
        if mci_victims:
            victim = mci_victims[0]
            rpm = victim.get("rpm_score", {})
            resp = rpm.get("respirations", 16)
            pulse = rpm.get("pulse", 90)
            mental = rpm.get("mental_status", "obeys")
            if resp == 0:
                tag = "Expectant"
            elif resp < 10 or resp > 29:
                tag = "Immediate"
            elif pulse == 0:
                tag = "Expectant"
            elif pulse < 50:
                tag = "Immediate"
            elif mental != "obeys":
                tag = "Immediate"
            else:
                tag = "Delayed"
            action = {
                "action_type": "tag",
                "victim_id": victim.get("incident_id", victim.get("victim_id", "V1")),
                "triage_tag": tag,
                "rpm_assessment": rpm
            }
        elif incident and unit:
            hospital = get_best_hospital(obs, incident)
            action = {
                "action_type": "dispatch",
                "unit_id": unit,
                "incident_id": incident.get("incident_id", "I1"),
                "destination_hospital": hospital,
                "priority_override": None
            }
        else:
            action = {"action_type": "noop", "unit_id": None}

    elif task_id == "task_8":
        transfers = [i for i in obs.get("incident_queue", []) if i.get("requires_transfer", False)]
        if transfers and unit:
            t = transfers[0]
            hospitals = obs.get("hospital_network", [])
            specialist_h = [h for h in hospitals
                            if t.get("required_specialty", "") in h.get("specialties", [])
                            and not h.get("on_diversion", False)
                            and h.get("icu_beds_available", 0) > 0]
            dest = specialist_h[0].get("hospital_id", "H1") if specialist_h else "H1"
            action = {
                "action_type": "transfer",
                "unit_id": unit,
                "patient_id": t.get("incident_id", "P1"),
                "from_hospital": t.get("current_hospital", "H1"),
                "to_hospital": dest,
                "urgency": "urgent"
            }
        elif incident and unit:
            hospital = get_best_hospital(obs, incident)
            action = {
                "action_type": "dispatch",
                "unit_id": unit,
                "incident_id": incident.get("incident_id", "I1"),
                "destination_hospital": hospital,
                "priority_override": None
            }
        else:
            action = {"action_type": "noop", "unit_id": None}

    elif task_id == "task_9":
        total_incidents = len(obs.get("incident_queue", []))
        fleet_available = sum(1 for u in obs.get("fleet_status", []) if u.get("status") == "available")
        hospital_saturated = sum(1 for h in obs.get("hospital_network", []) if h.get("on_diversion", False))

        if total_incidents > 15 and fleet_available < 3 and step_num % 8 == 0:
            zones = [z.get("zone_id", "Z1") for z in obs.get("zone_status", [{"zone_id": "Z1"}])[:2]]
            action = {
                "action_type": "request_mutual_aid",
                "requesting_zone": obs.get("current_zone", "Z1"),
                "target_zones": zones[:2],
                "units_requested": min(3, total_incidents // 5),
                "priority_level": "P1"
            }
        elif hospital_saturated >= 5 and step_num % 12 == 0:
            action = {
                "action_type": "escalate",
                "escalation_type": "surge_declaration",
                "affected_zones": ["Z1", "Z2", "Z3"],
                "mutual_aid_requested": True,
                "command_post_activated": True
            }
        elif incident and unit:
            hospital = get_best_hospital(obs, incident)
            action = {
                "action_type": "dispatch",
                "unit_id": unit,
                "incident_id": incident.get("incident_id", "I1"),
                "destination_hospital": hospital,
                "priority_override": None
            }
        else:
            action = {"action_type": "noop", "unit_id": None}

    elif task_id == "task_5":
        fleet = obs.get("fleet_status", [])
        traffic = obs.get("traffic_snapshot", {})
        reroutable = [u for u in fleet if u.get("status") == "en_route"]
        if reroutable:
            unit_to_reroute = reroutable[0]
            current_dest = unit_to_reroute.get("destination_hospital", "H1")
            hospitals = obs.get("hospital_network", [])
            alt_hospitals = [h for h in hospitals
                             if h.get("hospital_id") != current_dest
                             and not h.get("on_diversion", False)
                             and h.get("available_beds", 0) > 0]
            if alt_hospitals:
                new_dest = alt_hospitals[0].get("hospital_id", "H2")
                action = {
                    "action_type": "reroute",
                    "unit_id": unit_to_reroute.get("unit_id"),
                    "new_destination": new_dest,
                    "reason": "traffic_diversion"
                }
            else:
                action = {"action_type": "noop", "unit_id": None}
        elif incident and unit:
            hospital = get_best_hospital(obs, incident)
            action = {
                "action_type": "dispatch",
                "unit_id": unit,
                "incident_id": incident.get("incident_id", "I1"),
                "destination_hospital": hospital,
                "priority_override": None
            }
        else:
            action = {"action_type": "noop", "unit_id": None}

    else:
        if incident and unit:
            condition = incident.get("chief_complaint", incident.get("condition", ""))
            priority  = incident.get("priority", "P2")
            unit_type = unit_type_for_priority(priority, condition)
            hospital  = get_best_hospital(obs, incident)

            fleet = obs.get("fleet_status", [])
            typed_units = [u for u in fleet
                           if u.get("status") == "available"
                           and u.get("unit_type") == unit_type]
            chosen_unit = typed_units[0].get("unit_id") if typed_units else unit

            is_trapped = incident.get("requires_extrication", False) or "trapped" in condition.lower()
            if is_trapped:
                action = {
                    "action_type": "escalate",
                    "escalation_type": "multi_agency",
                    "incident_id": incident.get("incident_id", "I1"),
                    "agencies_required": ["police", "fire", "ems"],
                    "unit_id": chosen_unit,
                    "destination_hospital": hospital
                }
            else:
                action = {
                    "action_type": "dispatch",
                    "unit_id": chosen_unit,
                    "incident_id": incident.get("incident_id", "I1"),
                    "destination_hospital": hospital,
                    "priority_override": priority if priority == "P1" else None
                }
        else:
            action = {"action_type": "noop", "unit_id": None}
else:
    action = {"action_type": "noop", "unit_id": None}

print(json.dumps(action))
PYEOF
}

run_episode() {
    local task_id="$1" seed="$2"
    local max_steps="${TASK_MAX_STEPS[$task_id]}"
    local ep_log="${REPORT_DIR}/episodes/${task_id}_seed${seed}.jsonl"
    mkdir -p "$(dirname "${ep_log}")"

    local reset_resp reset_status reset_body
    reset_resp=$(http_post "${SERVER_URL}/reset" "{\"task_id\": \"${task_id}\", \"seed\": ${seed}}")
    reset_status=$(parse_status "${reset_resp}")
    reset_body=$(parse_body "${reset_resp}")

    if [[ "${reset_status}" != "200" ]]; then
        echo "ERROR:reset_failed:${reset_status}"
        return 1
    fi

    local ep_reward=0.0
    local ep_steps=0
    local ep_protocol_score=0.0
    local ep_survival_rate=0.0
    local ep_diversion_hits=0
    local ep_mutual_aid=0
    local ep_fatigue=0
    local ep_comm_fail=0
    local current_obs="${reset_body}"

    echo "${current_obs}" >> "${ep_log}"

    local done_flag="false"
    local step=0

    while [[ "${done_flag}" == "false" ]] && (( step < max_steps )); do
        local obs_for_agent
        obs_for_agent=$(echo "${current_obs}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    obs = d.get('observation', d)
    print(json.dumps(obs))
except:
    print('{}')
" 2>/dev/null)

        local action_json
        action_json=$(echo "${obs_for_agent}" | agent_action "${task_id}" "${obs_for_agent}" "${step}" "${seed}")

        if [[ -z "${action_json}" ]] || ! echo "${action_json}" | python3 -c "import sys,json; json.load(sys.stdin)" >/dev/null 2>&1; then
            action_json='{"action_type": "noop", "unit_id": null}'
        fi

        local step_resp step_status step_body
        step_resp=$(http_post "${SERVER_URL}/step" "${action_json}")
        step_status=$(parse_status "${step_resp}")
        step_body=$(parse_body "${step_resp}")

        if [[ "${step_status}" != "200" ]]; then
            [[ "${VERBOSE}" == "true" ]] && log_warn "Step ${step} failed HTTP ${step_status}"
            break
        fi

        local step_reward
        step_reward=$(echo "${step_body}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get('reward', d.get('step_reward', 0.0))
    if isinstance(r, dict):
        r = r.get('total', r.get('value', 0.0))
    print(float(r))
except:
    print(0.0)
" 2>/dev/null)

        ep_reward=$(python3 -c "print(round(${ep_reward} + ${step_reward}, 6))" 2>/dev/null || echo "${ep_reward}")

        local step_meta
        step_meta=$(echo "${step_body}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    info = d.get('info', {})
    print(json.dumps({
        'protocol_compliance': float(info.get('protocol_compliance', 0.0)),
        'survival_rate':       float(info.get('survival_rate', 0.0)),
        'diversion_hit':       int(info.get('diversion_penalty_triggered', 0)),
        'mutual_aid_calls':    int(info.get('mutual_aid_calls', 0)),
        'fatigue_events':      int(info.get('crew_fatigue_events', 0)),
        'comm_failures':       int(info.get('comm_failures', 0))
    }))
except:
    print('{\"protocol_compliance\":0,\"survival_rate\":0,\"diversion_hit\":0,\"mutual_aid_calls\":0,\"fatigue_events\":0,\"comm_failures\":0}')
" 2>/dev/null)

        ep_protocol_score=$(python3 -c "
import json
m = json.loads('$(echo "${step_meta}" | sed "s/'/\\\'/g")')
print(round(${ep_protocol_score} + m.get('protocol_compliance', 0.0), 6))
" 2>/dev/null || echo "${ep_protocol_score}")

        ep_diversion_hits=$(python3 -c "
import json
m = json.loads('$(echo "${step_meta}" | sed "s/'/\\\'/g")')
print(${ep_diversion_hits} + int(m.get('diversion_hit', 0)))
" 2>/dev/null || echo "${ep_diversion_hits}")

        ep_mutual_aid=$(python3 -c "
import json
m = json.loads('$(echo "${step_meta}" | sed "s/'/\\\'/g")')
print(${ep_mutual_aid} + int(m.get('mutual_aid_calls', 0)))
" 2>/dev/null || echo "${ep_mutual_aid}")

        ep_fatigue=$(python3 -c "
import json
m = json.loads('$(echo "${step_meta}" | sed "s/'/\\\'/g")')
print(${ep_fatigue} + int(m.get('fatigue_events', 0)))
" 2>/dev/null || echo "${ep_fatigue}")

        ep_comm_fail=$(python3 -c "
import json
m = json.loads('$(echo "${step_meta}" | sed "s/'/\\\'/g")')
print(${ep_comm_fail} + int(m.get('comm_failures', 0)))
" 2>/dev/null || echo "${ep_comm_fail}")

        done_flag=$(echo "${step_body}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(str(d.get('done', False)).lower())
except:
    print('false')
" 2>/dev/null)

        current_obs="${step_body}"
        echo "{\"step\": ${step}, \"action\": $(echo "${action_json}"), \"reward\": ${step_reward}, \"done\": ${done_flag}}" >> "${ep_log}"

        ((step++)) || true
        [[ "${SHOW_PROGRESS}" == "true" ]] && progress_bar "${step}" "${max_steps}" "${task_id} seed=${seed}"
    done

    local final_score
    final_score=$(echo "${current_obs}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    s = d.get('episode_score', d.get('score', None))
    if s is not None:
        print(round(float(s), 6))
        sys.exit(0)
    info = d.get('info', {})
    s = info.get('episode_score', info.get('final_score', None))
    if s is not None:
        print(round(float(s), 6))
        sys.exit(0)
    max_s = ${TASK_MAX_STEPS[$task_id]}
    r_per_step = float('${ep_reward}') / max(${step}, 1)
    normalized = max(0.0, min(1.0, (r_per_step + 1.0) / 2.0))
    print(round(normalized, 6))
except Exception as e:
    print(0.0)
" 2>/dev/null)

    local survival_rate_final
    survival_rate_final=$(echo "${current_obs}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    info = d.get('info', {})
    print(round(float(info.get('survival_rate', 0.0)), 4))
except:
    print(0.0)
" 2>/dev/null)

    echo "SCORE:${final_score}|STEPS:${step}|PROTOCOL:${ep_protocol_score}|SURVIVAL:${survival_rate_final}|DIVERT:${ep_diversion_hits}|MUTUAL:${ep_mutual_aid}|FATIGUE:${ep_fatigue}|COMM:${ep_comm_fail}"
}

run_task() {
    local task_id="$1"
    local task_log="${REPORT_DIR}/tasks/${task_id}.log"
    mkdir -p "$(dirname "${task_log}")"

    local difficulty="${TASK_DIFFICULTY[$task_id]}"
    local task_name="${TASK_NAME[$task_id]}"
    local baseline="${TASK_BASELINE[$task_id]}"

    diff_color=""
    case "${difficulty}" in
        easy)   diff_color="${GREEN}" ;;
        medium) diff_color="${YELLOW}" ;;
        hard)   diff_color="${RED}" ;;
    esac

    log_sub "${diff_color}[${difficulty^^}]${RESET} ${BOLD}${task_id}${RESET}: ${task_name}"
    log_info "Seeds: ${SEEDS}  |  Max steps: ${TASK_MAX_STEPS[$task_id]}  |  Agent: ${AGENT_MODE}"

    if [[ "${SKIP_WARMUP}" == "false" ]] && (( WARMUP_EPISODES > 0 )); then
        log_info "Warming up with ${WARMUP_EPISODES} episodes..."
        for ((w=1; w<=WARMUP_EPISODES; w++)); do
            http_post "${SERVER_URL}/reset" "{\"task_id\": \"${task_id}\", \"seed\": $((w * 99999))}" >/dev/null 2>&1
            http_post "${SERVER_URL}/step" '{"action_type": "noop", "unit_id": null}' >/dev/null 2>&1
        done
    fi

    local task_wall_start
    task_wall_start=$(date +%s)

    declare -a scores_arr=()
    declare -a steps_arr=()
    declare -a protocol_arr=()
    declare -a survival_arr=()
    declare -a divert_arr=()
    declare -a mutual_arr=()
    declare -a fatigue_arr=()
    declare -a comm_arr=()

    local ep_num=0
    for seed in "${SEED_ARRAY[@]}"; do
        ((ep_num++)) || true
        [[ "${SHOW_PROGRESS}" == "true" ]] && echo -ne "  Episode ${ep_num}/${NUM_SEEDS} (seed=${seed}) "

        local ep_result
        ep_result=$(run_episode "${task_id}" "${seed}" 2>/dev/null)

        [[ "${SHOW_PROGRESS}" == "true" ]] && echo ""

        local ep_score ep_steps ep_protocol ep_survival ep_divert ep_mutual ep_fatigue ep_comm
        ep_score=$(echo "${ep_result}"    | grep -oP '(?<=SCORE:)[^|]+' || echo "0.0")
        ep_steps=$(echo "${ep_result}"    | grep -oP '(?<=STEPS:)[^|]+' || echo "0")
        ep_protocol=$(echo "${ep_result}" | grep -oP '(?<=PROTOCOL:)[^|]+' || echo "0.0")
        ep_survival=$(echo "${ep_result}" | grep -oP '(?<=SURVIVAL:)[^|]+' || echo "0.0")
        ep_divert=$(echo "${ep_result}"   | grep -oP '(?<=DIVERT:)[^|]+' || echo "0")
        ep_mutual=$(echo "${ep_result}"   | grep -oP '(?<=MUTUAL:)[^|]+' || echo "0")
        ep_fatigue=$(echo "${ep_result}"  | grep -oP '(?<=FATIGUE:)[^|]+' || echo "0")
        ep_comm=$(echo "${ep_result}"     | grep -oP '(?<=COMM:)[^|]+' || echo "0")

        ep_score=$(python3 -c "
s = float('${ep_score}' or '0.0')
print(round(max(0.0, min(1.0, s)), 6))
" 2>/dev/null || echo "0.0")

        scores_arr+=("${ep_score}")
        steps_arr+=("${ep_steps}")
        protocol_arr+=("${ep_protocol}")
        survival_arr+=("${ep_survival}")
        divert_arr+=("${ep_divert}")
        mutual_arr+=("${ep_mutual}")
        fatigue_arr+=("${ep_fatigue}")
        comm_arr+=("${ep_comm}")

        echo "{\"task_id\": \"${task_id}\", \"seed\": ${seed}, \"score\": ${ep_score}, \"steps\": ${ep_steps}}" >> "${task_log}"

        if [[ "${VERBOSE}" == "true" ]]; then
            log_info "  seed=${seed}: score=${ep_score} steps=${ep_steps} protocol=${ep_protocol} survival=${ep_survival}"
        fi
    done

    local task_wall_end
    task_wall_end=$(date +%s)
    local wall_time=$(( task_wall_end - task_wall_start ))

    local stats_str
    stats_str=$(py_stats "${scores_arr[@]}")

    local mean_score std_score min_score max_score median_score
    mean_score=$(echo "${stats_str}"   | grep -oP '(?<=mean=)[^ ]+')
    std_score=$(echo "${stats_str}"    | grep -oP '(?<=std=)[^ ]+')
    min_score=$(echo "${stats_str}"    | grep -oP '(?<=min=)[^ ]+')
    max_score=$(echo "${stats_str}"    | grep -oP '(?<=max=)[^ ]+')
    median_score=$(echo "${stats_str}" | grep -oP '(?<=median=)[^ ]+')

    local steps_mean
    steps_mean=$(py_stats "${steps_arr[@]}" | grep -oP '(?<=mean=)[^ ]+')

    local protocol_mean
    protocol_mean=$(py_stats "${protocol_arr[@]}" | grep -oP '(?<=mean=)[^ ]+')

    local survival_mean
    survival_mean=$(py_stats "${survival_arr[@]}" | grep -oP '(?<=mean=)[^ ]+')

    local total_diverts=0 total_mutual=0 total_fatigue=0 total_comm=0
    for v in "${divert_arr[@]}"; do  ((total_diverts += v)) || true; done
    for v in "${mutual_arr[@]}";  do ((total_mutual  += v)) || true; done
    for v in "${fatigue_arr[@]}"; do ((total_fatigue += v)) || true; done
    for v in "${comm_arr[@]}";    do ((total_comm    += v)) || true; done

    TASK_SCORES_MEAN[$task_id]="${mean_score}"
    TASK_SCORES_STD[$task_id]="${std_score}"
    TASK_SCORES_MIN[$task_id]="${min_score}"
    TASK_SCORES_MAX[$task_id]="${max_score}"
    TASK_SCORES_MEDIAN[$task_id]="${median_score}"
    TASK_EPISODES_COMPLETED[$task_id]="${ep_num}"
    TASK_STEPS_MEAN[$task_id]="${steps_mean}"
    TASK_PROTOCOL_COMPLIANCE[$task_id]="${protocol_mean}"
    TASK_SURVIVAL_RATE[$task_id]="${survival_mean}"
    TASK_DIVERSION_HITS[$task_id]="${total_diverts}"
    TASK_MUTUAL_AID_CALLS[$task_id]="${total_mutual}"
    TASK_FATIGUE_EVENTS[$task_id]="${total_fatigue}"
    TASK_COMM_FAILURES[$task_id]="${total_comm}"
    TASK_WALL_TIME_S[$task_id]="${wall_time}"

    local vs_baseline=""
    if [[ "${COMPARE_BASELINE}" == "true" ]]; then
        vs_baseline=$(python3 -c "
mean  = float('${mean_score}')
base  = float('${baseline}')
delta = mean - base
arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '═')
color_open  = '\033[0;32m' if delta >= 0 else '\033[0;31m'
color_close = '\033[0m'
print(f'{color_open}{arrow} {delta:+.4f} vs baseline {base}{color_close}')
" 2>/dev/null)
    fi

    log_ok "${BOLD}${task_id}${RESET} mean=${BOLD}${mean_score}${RESET} ± ${std_score}  [${min_score}, ${max_score}]  median=${median_score}  ${vs_baseline}"
    log_metric "Steps"       "avg ${steps_mean} / ${TASK_MAX_STEPS[$task_id]}"
    log_metric "Protocol"    "${protocol_mean}"
    log_metric "Survival"    "${survival_mean}"
    log_metric "Diversions"  "${total_diverts} hits  |  Mutual aid calls: ${total_mutual}"
    log_metric "Crew fatigue"  "${total_fatigue} events  |  Comm failures: ${total_comm}"
    log_metric "Wall time"   "${wall_time}s for ${ep_num} episodes"
}

check_server() {
    local resp status
    resp=$(http_get "${SERVER_URL}/health")
    status=$(parse_status "${resp}")
    if [[ "${status}" != "200" ]]; then
        echo -e "${RED}${BOLD}ERROR: Server not reachable at ${SERVER_URL}/health (HTTP ${status})${RESET}"
        echo -e "${DIM}Start the server with: uvicorn server.main:app --host 0.0.0.0 --port 7860${RESET}"
        exit 3
    fi
    local version_info
    version_info=$(parse_body "${resp}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f\"{d.get('version','?')} — {d.get('environment','EMERGI-ENV')}\")
except:
    print('OK')
" 2>/dev/null)
    log_ok "Server healthy: ${version_info}"
}

print_header_table() {
    echo -e "\n${BOLD}${CYAN}"
    printf "  %-10s %-35s %-8s %-8s %-8s %-8s %-10s %-8s\n" \
        "TASK" "NAME" "DIFF" "MEAN" "±STD" "BASELINE" "VS_BASE" "MEDIAN"
    printf "  %s\n" "$(printf '─%.0s' {1..100})"
    echo -e "${RESET}"

    for task_id in "${TASK_ARRAY[@]}"; do
        local diff="${TASK_DIFFICULTY[$task_id]}"
        local name="${TASK_NAME[$task_id]}"
        local mean="${TASK_SCORES_MEAN[$task_id]:-N/A}"
        local std="${TASK_SCORES_STD[$task_id]:-N/A}"
        local median="${TASK_SCORES_MEDIAN[$task_id]:-N/A}"
        local base="${TASK_BASELINE[$task_id]}"

        local diff_c=""
        case "${diff}" in
            easy)   diff_c="${GREEN}" ;;
            medium) diff_c="${YELLOW}" ;;
            hard)   diff_c="${RED}" ;;
        esac

        local vs_base delta_str score_c=""
        if [[ "${mean}" != "N/A" ]] && [[ "${COMPARE_BASELINE}" == "true" ]]; then
            vs_base=$(python3 -c "
delta = float('${mean}') - float('${base}')
print(f'{delta:+.4f}')
" 2>/dev/null)
            delta_f=$(python3 -c "print(float('${mean}') - float('${base}'))" 2>/dev/null || echo "0")
            if python3 -c "exit(0 if float('${delta_f}') >= 0 else 1)" 2>/dev/null; then
                score_c="${GREEN}"
            else
                score_c="${RED}"
            fi
        else
            vs_base="N/A"
        fi

        printf "  ${BOLD}%-10s${RESET} %-35s ${diff_c}%-8s${RESET} ${score_c}%-8s${RESET} %-8s %-8s ${score_c}%-10s${RESET} %-8s\n" \
            "${task_id}" "${name:0:35}" "${diff}" "${mean}" "±${std}" "${base}" "${vs_base}" "${median}"
    done
    printf "  %s\n\n" "$(printf '─%.0s' {1..100})"
}

export_csv() {
    local csv_file="${REPORT_DIR}/baseline_results.csv"
    {
        echo "task_id,task_name,difficulty,mean_score,std_score,min_score,max_score,median_score,baseline_score,vs_baseline,episodes,mean_steps,protocol_compliance,survival_rate,diversion_hits,mutual_aid_calls,fatigue_events,comm_failures,wall_time_s,agent_mode,model,timestamp"
        for task_id in "${TASK_ARRAY[@]}"; do
            local mean="${TASK_SCORES_MEAN[$task_id]:-0.0}"
            local base="${TASK_BASELINE[$task_id]}"
            local vs_base
            vs_base=$(python3 -c "print(round(float('${mean}') - float('${base}'), 6))" 2>/dev/null || echo "0.0")
            echo "${task_id},\"${TASK_NAME[$task_id]}\",${TASK_DIFFICULTY[$task_id]},${mean},${TASK_SCORES_STD[$task_id]:-0.0},${TASK_SCORES_MIN[$task_id]:-0.0},${TASK_SCORES_MAX[$task_id]:-0.0},${TASK_SCORES_MEDIAN[$task_id]:-0.0},${base},${vs_base},${TASK_EPISODES_COMPLETED[$task_id]:-0},${TASK_STEPS_MEAN[$task_id]:-0.0},${TASK_PROTOCOL_COMPLIANCE[$task_id]:-0.0},${TASK_SURVIVAL_RATE[$task_id]:-0.0},${TASK_DIVERSION_HITS[$task_id]:-0},${TASK_MUTUAL_AID_CALLS[$task_id]:-0},${TASK_FATIGUE_EVENTS[$task_id]:-0},${TASK_COMM_FAILURES[$task_id]:-0},${TASK_WALL_TIME_S[$task_id]:-0},${AGENT_MODE},${MODEL_NAME},${RUN_TIMESTAMP}"
        done
    } > "${csv_file}"
    log_ok "CSV exported: ${csv_file}"
}

export_json() {
    local json_file="${REPORT_DIR}/baseline_results.json"
    python3 - << PYEOF > "${json_file}"
import json, os

tasks = []
task_ids     = "${TASKS}".split(",")
names        = {$(for t in "${!TASK_NAME[@]}"; do echo "\"${t}\": \"${TASK_NAME[$t]}\","; done)}
difficulties = {$(for t in "${!TASK_DIFFICULTY[@]}"; do echo "\"${t}\": \"${TASK_DIFFICULTY[$t]}\","; done)}
baselines    = {$(for t in "${!TASK_BASELINE[@]}"; do echo "\"${t}\": ${TASK_BASELINE[$t]},"; done)}
means        = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_SCORES_MEAN[$t]:-0.0},"; done)}
stds         = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_SCORES_STD[$t]:-0.0},"; done)}
mins_        = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_SCORES_MIN[$t]:-0.0},"; done)}
maxs         = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_SCORES_MAX[$t]:-0.0},"; done)}
medians      = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_SCORES_MEDIAN[$t]:-0.0},"; done)}
episodes     = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_EPISODES_COMPLETED[$t]:-0},"; done)}
steps        = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_STEPS_MEAN[$t]:-0.0},"; done)}
protocols    = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_PROTOCOL_COMPLIANCE[$t]:-0.0},"; done)}
survivals    = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_SURVIVAL_RATE[$t]:-0.0},"; done)}
diversions   = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_DIVERSION_HITS[$t]:-0},"; done)}
mutuals      = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_MUTUAL_AID_CALLS[$t]:-0},"; done)}
fatigues     = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_FATIGUE_EVENTS[$t]:-0},"; done)}
comms        = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_COMM_FAILURES[$t]:-0},"; done)}
wall_times   = {$(for t in "${TASK_ARRAY[@]}"; do echo "\"${t}\": ${TASK_WALL_TIME_S[$t]:-0},"; done)}

for tid in task_ids:
    tid = tid.strip()
    if not tid:
        continue
    mean_s = means.get(tid, 0.0)
    base_s = baselines.get(tid, 0.0)
    tasks.append({
        "task_id":             tid,
        "task_name":           names.get(tid, tid),
        "difficulty":          difficulties.get(tid, "unknown"),
        "mean_score":          round(float(mean_s), 6),
        "std_score":           round(float(stds.get(tid, 0.0)), 6),
        "min_score":           round(float(mins_.get(tid, 0.0)), 6),
        "max_score":           round(float(maxs.get(tid, 0.0)), 6),
        "median_score":        round(float(medians.get(tid, 0.0)), 6),
        "baseline_score":      float(base_s),
        "delta_vs_baseline":   round(float(mean_s) - float(base_s), 6),
        "beats_baseline":      float(mean_s) >= float(base_s),
        "episodes_completed":  int(episodes.get(tid, 0)),
        "mean_steps":          round(float(steps.get(tid, 0.0)), 2),
        "protocol_compliance": round(float(protocols.get(tid, 0.0)), 4),
        "survival_rate":       round(float(survivals.get(tid, 0.0)), 4),
        "diversion_hits":      int(diversions.get(tid, 0)),
        "mutual_aid_calls":    int(mutuals.get(tid, 0)),
        "fatigue_events":      int(fatigues.get(tid, 0)),
        "comm_failures":       int(comms.get(tid, 0)),
        "wall_time_s":         int(wall_times.get(tid, 0))
    })

all_means  = [t["mean_score"] for t in tasks]
total_eps  = sum(t["episodes_completed"] for t in tasks)
beats      = sum(1 for t in tasks if t["beats_baseline"])
easy_mean  = round(sum(t["mean_score"] for t in tasks if t["difficulty"] == "easy") /
                   max(1, sum(1 for t in tasks if t["difficulty"] == "easy")), 6)
med_mean   = round(sum(t["mean_score"] for t in tasks if t["difficulty"] == "medium") /
                   max(1, sum(1 for t in tasks if t["difficulty"] == "medium")), 6)
hard_mean  = round(sum(t["mean_score"] for t in tasks if t["difficulty"] == "hard") /
                   max(1, sum(1 for t in tasks if t["difficulty"] == "hard")), 6)
overall    = round(sum(all_means) / max(1, len(all_means)), 6)

report = {
    "run_metadata": {
        "timestamp":          "${RUN_TIMESTAMP}",
        "agent_mode":         "${AGENT_MODE}",
        "model_name":         "${MODEL_NAME}",
        "server_url":         "${SERVER_URL}",
        "seeds":              "${SEEDS}",
        "tasks_evaluated":    len(tasks),
        "total_episodes":     total_eps,
        "tasks_beating_baseline": beats,
    },
    "aggregate": {
        "overall_mean":      overall,
        "easy_tier_mean":    easy_mean,
        "medium_tier_mean":  med_mean,
        "hard_tier_mean":    hard_mean,
        "beats_baseline_count": beats,
        "beats_baseline_pct":   round(beats / max(1, len(tasks)) * 100, 1)
    },
    "tasks": tasks
}
print(json.dumps(report, indent=2))
PYEOF
    log_ok "JSON exported: ${json_file}"
}

export_markdown() {
    local md_file="${REPORT_DIR}/baseline_report.md"
    python3 - << PYEOF > "${md_file}"
import json, datetime

with open("${REPORT_DIR}/baseline_results.json") as f:
    report = json.load(f)

meta = report["run_metadata"]
agg  = report["aggregate"]
tasks = report["tasks"]

lines = []
lines.append("# EMERGI-ENV Baseline Run Report")
lines.append("")
lines.append(f"**Run timestamp**: {meta['timestamp']}")
lines.append(f"**Agent mode**: \`{meta['agent_mode']}\`  |  **Model**: \`{meta['model_name']}\`")
lines.append(f"**Server**: {meta['server_url']}  |  **Seeds**: \`{meta['seeds']}\`")
lines.append(f"**Tasks evaluated**: {meta['tasks_evaluated']}  |  **Total episodes**: {meta['total_episodes']}")
lines.append("")
lines.append("## Aggregate Scores")
lines.append("")
lines.append("| Tier | Mean Score |")
lines.append("|------|------------|")
lines.append(f"| 🟢 Easy   | {agg['easy_tier_mean']:.4f} |")
lines.append(f"| 🟡 Medium | {agg['medium_tier_mean']:.4f} |")
lines.append(f"| 🔴 Hard   | {agg['hard_tier_mean']:.4f} |")
lines.append(f"| **Overall** | **{agg['overall_mean']:.4f}** |")
lines.append("")
lines.append(f"**Beats baseline**: {agg['beats_baseline_count']}/{meta['tasks_evaluated']} tasks ({agg['beats_baseline_pct']}%)")
lines.append("")
lines.append("## Per-Task Results")
lines.append("")
lines.append("| Task | Name | Diff | Mean | ±Std | Min | Max | Baseline | Δ | Beats? |")
lines.append("|------|------|------|------|------|-----|-----|----------|---|--------|")

for t in tasks:
    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(t["difficulty"], "⚪")
    beats_str = "✅" if t["beats_baseline"] else "❌"
    delta_str = f"+{t['delta_vs_baseline']:.4f}" if t['delta_vs_baseline'] >= 0 else f"{t['delta_vs_baseline']:.4f}"
    lines.append(f"| {t['task_id']} | {t['task_name']} | {diff_emoji} {t['difficulty']} | **{t['mean_score']:.4f}** | ±{t['std_score']:.4f} | {t['min_score']:.4f} | {t['max_score']:.4f} | {t['baseline_score']:.2f} | {delta_str} | {beats_str} |")

lines.append("")
lines.append("## Advanced Metrics")
lines.append("")
lines.append("| Task | Protocol Compliance | Survival Rate | Diversion Hits | Mutual Aid | Fatigue Events | Comm Failures |")
lines.append("|------|---------------------|---------------|----------------|------------|----------------|---------------|")

for t in tasks:
    lines.append(f"| {t['task_id']} | {t['protocol_compliance']:.4f} | {t['survival_rate']:.4f} | {t['diversion_hits']} | {t['mutual_aid_calls']} | {t['fatigue_events']} | {t['comm_failures']} |")

lines.append("")
lines.append("## Episode-Level Details")
lines.append("")
lines.append("| Task | Episodes | Mean Steps | Wall Time |")
lines.append("|------|----------|------------|-----------|")
for t in tasks:
    lines.append(f"| {t['task_id']} | {t['episodes_completed']} | {t['mean_steps']:.1f} | {t['wall_time_s']}s |")

lines.append("")
lines.append("---")
lines.append(f"*Generated by EMERGI-ENV baseline runner on {datetime.datetime.utcnow().isoformat()}Z*")

print("\n".join(lines))
PYEOF
    log_ok "Markdown report exported: ${md_file}"
}

compute_aggregate_summary() {
    log_head "AGGREGATE RESULTS"

    python3 - << PYEOF
import json, sys

try:
    with open("${REPORT_DIR}/baseline_results.json") as f:
        report = json.load(f)
except:
    sys.exit(0)

agg   = report["aggregate"]
meta  = report["run_metadata"]
tasks = report["tasks"]

print()
print(f"  {'Overall mean score':<30} {agg['overall_mean']:.4f}")
print(f"  {'Easy tier mean':<30} {agg['easy_tier_mean']:.4f}")
print(f"  {'Medium tier mean':<30} {agg['medium_tier_mean']:.4f}")
print(f"  {'Hard tier mean':<30} {agg['hard_tier_mean']:.4f}")
print(f"  {'Beats baseline':<30} {agg['beats_baseline_count']}/{meta['tasks_evaluated']} ({agg['beats_baseline_pct']}%)")
print(f"  {'Total episodes run':<30} {meta['total_episodes']}")
print()

difficulty_weighted = (
    agg['easy_tier_mean']   * 0.20 +
    agg['medium_tier_mean'] * 0.35 +
    agg['hard_tier_mean']   * 0.45
)
print(f"  {'Weighted composite score':<30} {difficulty_weighted:.4f}")
print(f"  {'(easy×0.2 + med×0.35 + hard×0.45)'}")
print()

all_protocols  = [t["protocol_compliance"] for t in tasks]
all_survivals  = [t["survival_rate"] for t in tasks]
all_diverts    = sum(t["diversion_hits"] for t in tasks)
all_mutual     = sum(t["mutual_aid_calls"] for t in tasks)
all_fatigue    = sum(t["fatigue_events"] for t in tasks)
all_comm       = sum(t["comm_failures"] for t in tasks)

print(f"  {'Avg protocol compliance':<30} {sum(all_protocols)/max(1,len(all_protocols)):.4f}")
print(f"  {'Avg survival rate':<30} {sum(all_survivals)/max(1,len(all_survivals)):.4f}")
print(f"  {'Total diversion hits':<30} {all_diverts}")
print(f"  {'Total mutual aid calls':<30} {all_mutual}")
print(f"  {'Total fatigue events':<30} {all_fatigue}")
print(f"  {'Total comm failures':<30} {all_comm}")
print()

worst_task  = min(tasks, key=lambda t: t["mean_score"])
best_task   = max(tasks, key=lambda t: t["mean_score"])
most_impact = max(tasks, key=lambda t: abs(t["delta_vs_baseline"]))

print(f"  {'Best performing task':<30} {best_task['task_id']} ({best_task['mean_score']:.4f})")
print(f"  {'Worst performing task':<30} {worst_task['task_id']} ({worst_task['mean_score']:.4f})")
print(f"  {'Biggest delta from baseline':<30} {most_impact['task_id']} ({most_impact['delta_vs_baseline']:+.4f})")
print()
PYEOF
}

fail_threshold_check() {
    if python3 -c "exit(1 if float('${FAIL_BELOW}') <= 0.0 else 0)" 2>/dev/null; then
        return 0
    fi

    local below_threshold=false
    for task_id in "${TASK_ARRAY[@]}"; do
        local mean="${TASK_SCORES_MEAN[$task_id]:-0.0}"
        if python3 -c "exit(0 if float('${mean}') < float('${FAIL_BELOW}') else 1)" 2>/dev/null; then
            log_err "Task ${task_id} mean score ${mean} < fail threshold ${FAIL_BELOW}"
            below_threshold=true
        fi
    done

    if [[ "${below_threshold}" == "true" ]]; then
        echo -e "\n${RED}${BOLD}FAIL: One or more tasks scored below --fail-below ${FAIL_BELOW}${RESET}"
        exit 1
    fi
}

echo -e "${BOLD}${CYAN}"
cat << 'BANNER'
  ██████╗  █████╗ ███████╗███████╗██╗     ██╗███╗   ██╗███████╗
  ██╔══██╗██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║██╔════╝
  ██████╔╝███████║███████╗█████╗  ██║     ██║██╔██╗ ██║█████╗
  ██╔══██╗██╔══██║╚════██║██╔══╝  ██║     ██║██║╚██╗██║██╔══╝
  ██████╔╝██║  ██║███████║███████╗███████╗██║██║ ╚████║███████╗
  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝
BANNER
echo -e "${RESET}"
echo -e "${BOLD}  EMERGI-ENV  •  Baseline Score Runner${RESET}  ${DIM}India 108/112 EMS RL Environment${RESET}"
echo ""
echo -e "  ${DIM}Agent mode  : ${BOLD}${AGENT_MODE}${RESET}"
echo -e "  ${DIM}Tasks       : ${BOLD}${TASKS}${RESET}"
echo -e "  ${DIM}Seeds       : ${BOLD}${SEEDS}${RESET}  (${NUM_SEEDS} episodes/task)"
echo -e "  ${DIM}Server      : ${BOLD}${SERVER_URL}${RESET}"
echo -e "  ${DIM}Report dir  : ${BOLD}${REPORT_DIR}${RESET}"
echo -e "  ${DIM}Model       : ${BOLD}${MODEL_NAME}${RESET}"
echo ""

log_head "SERVER CONNECTIVITY CHECK"
check_server

log_head "RUNNING ALL ${NUM_TASKS} TASKS × ${NUM_SEEDS} SEEDS"

GLOBAL_START=$(date +%s)

if [[ "${PARALLEL_TASKS}" == "true" ]]; then
    declare -a PIDS=()
    for task_id in "${TASK_ARRAY[@]}"; do
        run_task "${task_id}" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done
else
    for task_id in "${TASK_ARRAY[@]}"; do
        run_task "${task_id}"
    done
fi

GLOBAL_END=$(date +%s)
GLOBAL_DURATION=$(( GLOBAL_END - GLOBAL_START ))

log_head "SCORE TABLE"
print_header_table

[[ "${EXPORT_JSON}"     == "true" ]] && export_json
[[ "${EXPORT_CSV}"      == "true" ]] && export_csv
[[ "${EXPORT_MARKDOWN}" == "true" ]] && export_markdown

compute_aggregate_summary

log_head "RUN COMPLETE"
echo -e "  ${DIM}Total wall time    : ${BOLD}${GLOBAL_DURATION}s${RESET}"
echo -e "  ${DIM}Report directory   : ${BOLD}${REPORT_DIR}${RESET}"
echo -e "  ${DIM}Episodes run       : ${BOLD}$(( NUM_TASKS * NUM_SEEDS ))${RESET}"
echo ""

fail_threshold_check

echo -e "${BOLD}${GREEN}"
cat << 'EOF'
   ╔══════════════════════════════════════════════════════╗
   ║  ✓  EMERGI-ENV BASELINE RUN COMPLETE                ║
   ╚══════════════════════════════════════════════════════╝
EOF
echo -e "${RESET}"
exit 0