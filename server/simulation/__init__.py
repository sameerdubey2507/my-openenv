from __future__ import annotations
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .trafficmodel       import (
    TrafficModel,
    TrafficSnapshot,
    RouteResult,
    ZoneNode,
    RoadLink,
    CongestionEvent,
    WeatherCondition,
    CongestionCause,
)
from .hospitalnetwork    import (
    HospitalNetwork,
    Hospital,
    InterHospitalTransfer,
    BloodBank,
    AdmissionType,
    DiversionStatus,
    HospitalTier,
    TransferStatus,
    DIVERSION_ER_THRESHOLD,
    SURGE_CAPACITY_FACTOR,
)
from .incidentengine     import (
    IncidentEngine,
    Incident,
    MCIScene,
    RPMScore,
    SurvivalCurveParams,
    IncidentPriority,
    IncidentStatus,
    StartTag,
    UnitType          as IncidentUnitType,
    AgencyRequirement,
    MCIType,
    INCIDENT_TEMPLATES,
)
from .fleetsimulator     import (
    FleetSimulator,
    AmbulanceUnit,
    CrewFatigueState,
    DispatchRecord,
    FleetMetrics,
    UnitStatus,
    UnitType          as FleetUnitType,
    MutualAidRequest  as FleetMutualAidRequest,
    GOLDEN_HOUR_WINDOWS,
    FATIGUE_ONSET_HOURS,
    MUTUAL_AID_DELAY_MINS,
)
from .communication      import (
    CommunicationsManager,
    CommMessage,
    UnitCommState,
    RelayTower,
    JammingEvent,
    ChannelStats,
    CommChannel,
    MessageType,
    MessagePriority,
    RelayTowerStatus,
    TASK_COMM_FAILURE_PROB,
)
from .demandforecaster   import (
    DemandForecaster,
    DemandForecast,
    DemandHeatmap,
    SurgeAlert,
    MCICluster,
    PrepositionRecommendation,
    ZoneDemandState,
    SurgeLevel,
    MCIRiskLevel,
)
from .multiagency        import (
    MultiAgencyCoordinator,
    MultiAgencyScene,
    AgencyUnit,
    CommandPost,
    CoordinationViolation,
    AgencyType,
    SceneStage,
    CoordinationViolationType,
)
from .mutualaid          import (
    MutualAidCoordinator,
    MutualAidRequest,
    AidUnit,
    AidTreaty,
    ConvoyDispatch,
    CascadeEvent,
    AidSurgeDeclaration,
    PreStagedPosition,
    MutualAidLedger,
    ZoneCapacity,
    AidTier,
    AidStatus,
    AidUnitType,
    TreatyType,
    CascadeStage,
    OVER_REQUEST_PENALTY,
    UNDER_REQUEST_TASK9_PENALTY,
    TIER_DELAY_MINUTES,
)
SIMULATION_VERSION: str   = "2.0.0"
SIMULATION_AUTHOR:  str   = "EMERGI-ENV Team"
SIMULATION_DESC:    str   = (
    "India 108/112 EMS reinforcement-learning environment — "
    "Pune-inspired 12-zone city, 8 hospitals, 9 tasks."
)
_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent.parent / "data"
logger = logging.getLogger(__name__)
STEP_DURATION_MINUTES: float = 3.0
TASK_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "name":              "Single Call Triage & Dispatch",
        "difficulty":        "easy",
        "max_steps":         20,
        "sim_start_min":     480.0,
        "weather":           "clear",
        "comm_failure":      False,
        "mci_enabled":       False,
        "mutual_aid_enabled":False,
        "traffic_dynamic":   False,
        "crew_fatigue":      False,
        "diversion_enabled": False,
        "baseline_score":    0.61,
        "grader_weights":    {"triage_class": 0.4, "unit_type": 0.3, "hospital_match": 0.3},
        "description": (
            "Agent reads one symptom description, classifies severity "
            "(P1/P2/P3), selects correct ambulance type, routes to appropriate hospital."
        ),
    },
    2: {
        "name":              "Hospital Route Selection",
        "difficulty":        "easy",
        "max_steps":         20,
        "sim_start_min":     510.0,
        "weather":           "clear",
        "comm_failure":      False,
        "mci_enabled":       False,
        "mutual_aid_enabled":False,
        "traffic_dynamic":   True,
        "crew_fatigue":      False,
        "diversion_enabled": True,
        "baseline_score":    0.72,
        "grader_weights":    {"specialty_match": 0.5, "capacity_check": 0.3, "travel_time": 0.2},
        "description": (
            "Agent picks best hospital given patient symptoms, location, "
            "and hospital network state."
        ),
    },
    3: {
        "name":              "Unit Type Matching",
        "difficulty":        "easy",
        "max_steps":         15,
        "sim_start_min":     480.0,
        "weather":           "clear",
        "comm_failure":      False,
        "mci_enabled":       False,
        "mutual_aid_enabled":False,
        "traffic_dynamic":   False,
        "crew_fatigue":      False,
        "diversion_enabled": False,
        "baseline_score":    0.68,
        "grader_weights":    {"exact_unit_match": 1.0},
        "description": (
            "Pure medical knowledge test: agent selects BLS / ALS / MICU "
            "for given scenario. Binary exact-match grader."
        ),
    },
    4: {
        "name":              "Multi-Incident Queue",
        "difficulty":        "medium",
        "max_steps":         60,
        "sim_start_min":     510.0,
        "weather":           "haze",
        "comm_failure":      False,
        "mci_enabled":       False,
        "mutual_aid_enabled":True,
        "traffic_dynamic":   True,
        "crew_fatigue":      True,
        "diversion_enabled": True,
        "baseline_score":    0.44,
        "grader_weights":    {"weighted_survival": 1.0},
        "description": (
            "5–8 simultaneous calls, only 3 ambulances available. "
            "Agent must prioritise. Grader: weighted survival score with delay penalties."
        ),
    },
    5: {
        "name":              "Dynamic Re-routing",
        "difficulty":        "medium",
        "max_steps":         50,
        "sim_start_min":     480.0,
        "weather":           "rain",
        "comm_failure":      False,
        "mci_enabled":       False,
        "mutual_aid_enabled":True,
        "traffic_dynamic":   True,
        "crew_fatigue":      True,
        "diversion_enabled": True,
        "baseline_score":    0.38,
        "grader_weights":    {"reroute_correctness": 0.5, "net_time_saved": 0.5},
        "description": (
            "Mid-episode traffic matrix updates and hospital diversion flags "
            "force agent to reroute active units."
        ),
    },
    6: {
        "name":              "Pre-positioning",
        "difficulty":        "medium",
        "max_steps":         80,
        "sim_start_min":     360.0,
        "weather":           "clear",
        "comm_failure":      False,
        "mci_enabled":       False,
        "mutual_aid_enabled":True,
        "traffic_dynamic":   True,
        "crew_fatigue":      False,
        "diversion_enabled": False,
        "baseline_score":    0.42,
        "grader_weights":    {"avg_response_time_vs_baseline": 1.0},
        "description": (
            "No active incidents. Agent repositions fleet based on demand "
            "forecast heatmap to minimise future response times. Delayed reward only."
        ),
    },
    7: {
        "name":              "Mass Casualty Incident (MCI)",
        "difficulty":        "hard",
        "max_steps":         100,
        "sim_start_min":     540.0,
        "weather":           "rain",
        "comm_failure":      True,
        "mci_enabled":       True,
        "mutual_aid_enabled":True,
        "traffic_dynamic":   True,
        "crew_fatigue":      True,
        "diversion_enabled": True,
        "baseline_score":    0.29,
        "grader_weights": {
            "start_accuracy": 0.50,
            "response_speed": 0.30,
            "hospital_spread":0.20,
        },
        "description": (
            "20–40 simultaneous victims. Agent applies START triage protocol "
            "(RPM scoring). Wrong Immediate→Expectant tag = -0.5 penalty."
        ),
    },
    8: {
        "name":              "Inter-Hospital Transfer Cascade",
        "difficulty":        "hard",
        "max_steps":         100,
        "sim_start_min":     600.0,
        "weather":           "clear",
        "comm_failure":      True,
        "mci_enabled":       False,
        "mutual_aid_enabled":True,
        "traffic_dynamic":   True,
        "crew_fatigue":      True,
        "diversion_enabled": True,
        "baseline_score":    0.24,
        "grader_weights": {
            "transfer_appropriateness": 0.50,
            "timing":                   0.30,
            "bed_utilisation":          0.20,
        },
        "description": (
            "ICU patients need specialist care. Agent manages transfers "
            "under bed utilisation constraints."
        ),
    },
    9: {
        "name":              "City-Wide Surge",
        "difficulty":        "hard",
        "max_steps":         120,
        "sim_start_min":     720.0,
        "weather":           "storm",
        "comm_failure":      True,
        "mci_enabled":       True,
        "mutual_aid_enabled":True,
        "traffic_dynamic":   True,
        "crew_fatigue":      True,
        "diversion_enabled": True,
        "baseline_score":    0.17,
        "grader_weights": {
            "system_survival":   0.40,
            "cascade_avoidance": 0.35,
            "mutual_aid_use":    0.25,
        },
        "description": (
            "3 simultaneous MCIs, hospital saturation, communications failures. "
            "Agent must request mutual aid and declare surge to avoid cascade collapse. "
            "Designed to score < 0.30 even for GPT-4."
        ),
    },
}
DEFAULT_ACTIVE_ZONES: List[str] = [
    "Z01", "Z02", "Z03", "Z04", "Z05", "Z06",
    "Z07", "Z08", "Z09", "Z13", "Z14", "Z27",
]
@dataclass
class SimulationBundle:
    traffic:    TrafficModel
    hospitals:  HospitalNetwork
    incidents:  IncidentEngine
    fleet:      FleetSimulator
    comms:      CommunicationsManager
    forecast:   DemandForecaster
    agency:     MultiAgencyCoordinator
    mutual_aid: MutualAidCoordinator
    seed:             int
    task_id:          int
    active_zone_ids:  List[str]
    sim_start_min:    float
    task_cfg:         Dict[str, Any]
    sim_time_min:     float    = field(default=480.0, init=False)
    step_count:       int      = field(default=0,     init=False)
    episode_id:       str      = field(default_factory=lambda: str(uuid.uuid4())[:8],
                                       init=False)
    _initialised:     bool     = field(default=False, init=False, repr=False)
    _cumulative_reward: float  = field(default=0.0,  init=False, repr=False)
    _step_rewards:      List[float] = field(default_factory=list, init=False, repr=False)
    _done:              bool   = field(default=False, init=False, repr=False)
    _done_reason:       str    = field(default="",    init=False, repr=False)
    def reset(
        self,
        task_id:          Optional[int]   = None,
        sim_time_minutes: Optional[float] = None,
        weather:          Optional[str]   = None,
        active_zone_ids:  Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if task_id is not None:
            self.task_id  = task_id
            self.task_cfg = TASK_CONFIGS.get(task_id, TASK_CONFIGS[1])
        cfg = self.task_cfg
        t0  = sim_time_minutes if sim_time_minutes is not None else cfg["sim_start_min"]
        w   = weather          if weather          is not None else cfg["weather"]
        zones = active_zone_ids if active_zone_ids is not None else self.active_zone_ids
        self.sim_time_min    = t0
        self.step_count      = 0
        self.episode_id      = str(uuid.uuid4())[:8]
        self._cumulative_reward = 0.0
        self._step_rewards   = []
        self._done           = False
        self._done_reason    = ""
        self.active_zone_ids = zones
        self.traffic.reset(
            sim_time_minutes = t0,
            active_zone_ids  = zones,
            weather          = w,
        )
        logger.debug("[SIM] traffic reset — %d zones", len(zones))
        self.hospitals.reset(
            task_id          = self.task_id,
            sim_time_minutes = t0,
            traffic_model    = self.traffic,
        )
        logger.debug("[SIM] hospitals reset — %d hospitals",
                     len(self.hospitals.hospitals))
        initial_incidents = self.incidents.reset(
            active_zone_ids  = zones,
            task_id          = self.task_id,
            sim_time_minutes = t0,
        )
        logger.debug("[SIM] incidents reset — %d initial", len(initial_incidents))
        self.fleet.reset(
            active_zone_ids  = zones,
            task_id          = self.task_id,
            sim_time_minutes = t0,
            traffic_model    = self.traffic,
            hospital_network = self.hospitals,
        )
        logger.debug("[SIM] fleet reset — %d units", len(self.fleet.units))
        unit_ids = list(self.fleet.units.keys())
        self.comms.reset(
            unit_ids         = unit_ids,
            active_zone_ids  = zones,
            task_id          = self.task_id,
            sim_time_minutes = t0,
            weather          = w,
        )
        logger.debug("[SIM] comms reset — %d units registered", len(unit_ids))
        self.forecast.reset(
            active_zone_ids  = zones,
            task_id          = self.task_id,
            sim_time_minutes = t0,
            weather          = w,
        )
        logger.debug("[SIM] demand forecaster reset")
        self.agency.reset(
            active_zone_ids  = zones,
            task_id          = self.task_id,
            sim_time_minutes = t0,
        )
        logger.debug("[SIM] multi-agency coordinator reset")
        self.mutual_aid.reset(
            active_zone_ids   = zones,
            task_id           = self.task_id,
            sim_time_minutes  = t0,
            traffic_model     = self.traffic,
            fleet_simulator   = self.fleet,
            hospital_network  = self.hospitals,
            incident_engine   = self.incidents,
            demand_forecaster = self.forecast,
            comms_manager     = self.comms,
        )
        logger.debug("[SIM] mutual-aid coordinator reset")
        self._initialised = True
        logger.info(
            "[SIM] Episode %s reset — task=%d seed=%d time=%.0f zones=%d",
            self.episode_id, self.task_id, self.seed, t0, len(zones),
        )
        return self.get_combined_observation()
    def step(
        self,
        active_incident_zones: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._initialised:
            raise RuntimeError("Call bundle.reset() before bundle.step()")
        self.sim_time_min += STEP_DURATION_MINUTES
        self.step_count   += 1
        step_reward        = 0.0
        if active_incident_zones is None:
            active_incident_zones = list({
                inc.zone_id for inc in self.incidents.active_incidents.values()
            })
        self.traffic.step(active_incident_zones=active_incident_zones)
        weather = self.traffic.weather.value
        new_incidents = self.incidents.step(
            sim_time_minutes = self.sim_time_min,
            traffic_od       = self.traffic._od_matrix,
        )
        if new_incidents:
            self.forecast.observe_incidents_bulk([
                inc.to_observation_dict() for inc in new_incidents
            ])
        _, hosp_reward = self.hospitals.step(self.sim_time_min)
        step_reward   += hosp_reward
        unit_positions = self._build_unit_position_map()
        _, fleet_reward = self.fleet.step(
            sim_time_minutes      = self.sim_time_min,
            active_incident_zones = active_incident_zones,
        )
        step_reward += fleet_reward
        if self.task_cfg.get("comm_failure", False):
            _, comm_reward = self.comms.step(
                sim_time_minutes       = self.sim_time_min,
                unit_positions         = unit_positions,
                weather                = weather,
                active_incident_zones  = active_incident_zones,
            )
            step_reward += comm_reward
        else:
            self.comms.update_weather(weather)
        _, forecast_reward = self.forecast.step(
            sim_time_minutes      = self.sim_time_min,
            weather               = weather,
            active_incident_zones = active_incident_zones,
        )
        step_reward += forecast_reward
        _, agency_reward = self.agency.step(self.sim_time_min)
        step_reward += agency_reward
        if self.task_cfg.get("mutual_aid_enabled", False):
            fleet_avail = self._build_fleet_available_map()
            _, aid_reward = self.mutual_aid.step(
                sim_time_minutes        = self.sim_time_min,
                fleet_available         = fleet_avail,
                hospital_diverted_count = self.hospitals.count_diverted(),
                p1_queue_count          = self.incidents.p1_count,
            )
            step_reward += aid_reward
        done, reason = self._check_done()
        self._done        = done
        self._done_reason = reason
        self._cumulative_reward += step_reward
        self._step_rewards.append(step_reward)
        info: Dict[str, Any] = {
            "step":               self.step_count,
            "sim_time_min":       round(self.sim_time_min, 1),
            "weather":            weather,
            "new_incidents":      len(new_incidents),
            "active_incidents":   self.incidents.total_active,
            "diverted_hospitals": self.hospitals.count_diverted(),
            "p1_count":           self.incidents.p1_count,
            "fleet_available":    self.fleet.get_fleet_observation()["available_count"],
            "cumulative_reward":  round(self._cumulative_reward, 4),
            "done_reason":        reason,
        }
        return self.get_combined_observation(), step_reward, done, info
    def get_combined_observation(self) -> Dict[str, Any]:
        fleet_obs   = self.fleet.get_fleet_observation()
        hosp_obs    = self.hospitals.get_network_observation()
        traffic_obs = self.traffic.get_snapshot()
        forecast_obs= self.forecast.get_observation_dict()
        comms_obs   = self.comms.get_comms_observation()
        agency_obs  = self.agency.get_coordinator_observation()
        aid_obs     = self.mutual_aid.get_observation()
        return {
            "episode_id":         self.episode_id,
            "task_id":            self.task_id,
            "step_count":         self.step_count,
            "sim_time_min":       round(self.sim_time_min, 1),
            "max_steps":          self.task_cfg.get("max_steps", 60),
            "done":               self._done,
            "done_reason":        self._done_reason,
            "cumulative_reward":  round(self._cumulative_reward, 4),
            "incident_queue":     self.incidents.get_queue_observation(),
            "mci_scenes":         self.incidents.get_mci_scenes(),
            "queue_empty":        self.incidents.queue_is_empty,
            "p1_count":           self.incidents.p1_count,
            "mean_deterioration": round(self.incidents.mean_deterioration, 4),
            "fleet_status":       fleet_obs,
            "hospital_network":   hosp_obs,
            "traffic_snapshot": {
                "weather":          traffic_obs.weather,
                "weather_mult":     traffic_obs.weather_multiplier,
                "peak_active":      traffic_obs.peak_active,
                "od_matrix":        traffic_obs.od_matrix,
                "zone_congestion":  traffic_obs.zone_congestion_levels,
                "congestion_events":traffic_obs.congestion_events,
                "step_count":       traffic_obs.step_count,
            },
            "demand_forecast": {
                "heatmap":          forecast_obs.get("heatmap", {}),
                "top_demand_zones": forecast_obs.get("top_demand_zones", []),
                "active_surges":    forecast_obs.get("active_surge_alerts", []),
                "mci_clusters":     forecast_obs.get("mci_clusters_at_risk", []),
                "forecasts":        forecast_obs.get("forecasts", {}),
                "peak_active":      forecast_obs.get("peak_active", False),
            },
            "communications": {
                "units_in_blackout":   comms_obs.get("units_in_blackout", 0),
                "mean_rssi":           comms_obs.get("mean_rssi", 1.0),
                "jammed_zones":        comms_obs.get("jammed_zones", []),
                "channel_degraded":    comms_obs.get("channel_degraded", False),
                "relay_towers_failed": comms_obs.get("relay_towers_failed", 0),
                "unit_comm_states":    comms_obs.get("unit_comm_states", {}),
            },
            "multi_agency": {
                "active_scenes":       agency_obs.get("active_scenes", 0),
                "scenes":              agency_obs.get("scenes", []),
                "violations_total":    agency_obs.get("violations_total", 0),
                "agency_availability": agency_obs.get("agency_availability", {}),
            },
            "mutual_aid": {
                "cascade_stage":    aid_obs.get("cascade_stage", "none"),
                "active_requests":  aid_obs.get("active_requests", 0),
                "active_units":     aid_obs.get("active_units", 0),
                "active_convoys":   aid_obs.get("active_convoys", 0),
                "active_surges":    aid_obs.get("active_surges", 0),
                "requests":         aid_obs.get("requests", []),
                "zone_capacities":  aid_obs.get("zone_capacities", {}),
                "ledger_snapshot":  aid_obs.get("ledger_snapshot", {}),
            },
        }
    def get_full_state(self) -> Dict[str, Any]:
        base = self.get_combined_observation()
        base["_ground_truth"] = {
            "incidents_full": {
                inc_id: inc.to_full_state_dict()
                for inc_id, inc in self.incidents.active_incidents.items()
            },
            "hospitals_full": {
                hid: h.to_full_state_dict()
                for hid, h in self.hospitals.hospitals.items()
            },
            "fleet_full":     self.fleet.get_all_units_full_state(),
            "comms_full":     self.comms.get_full_comm_state(),
            "mutual_aid_analytics": self.mutual_aid.get_episode_analytics(),
            "agency_analytics":     self.agency.get_episode_analytics(),
            "forecast_accuracy":    self.forecast.get_forecast_accuracy_estimate(),
        }
        return base
    def get_episode_summary(self) -> Dict[str, Any]:
        return {
            "episode_id":       self.episode_id,
            "task_id":          self.task_id,
            "task_name":        self.task_cfg.get("name", ""),
            "seed":             self.seed,
            "total_steps":      self.step_count,
            "sim_time_min":     round(self.sim_time_min, 1),
            "cumulative_reward":round(self._cumulative_reward, 4),
            "done_reason":      self._done_reason,
            "fleet_metrics":    self.fleet.get_metrics(),
            "hospital_analytics":self.hospitals.get_episode_analytics(),
            "incident_stats": {
                "total_active":    self.incidents.total_active,
                "total_resolved":  len(self.incidents.resolved_incidents),
                "mean_deterioration": round(self.incidents.mean_deterioration, 4),
            },
            "comms_analytics":   self.comms.get_episode_analytics(),
            "forecast_analytics":self.forecast.get_episode_analytics(),
            "agency_analytics":  self.agency.get_episode_analytics(),
            "mutual_aid_analytics":self.mutual_aid.get_episode_analytics(),
            "cascade_assessment":self.mutual_aid.get_cascade_assessment(),
        }
    def _check_done(self) -> Tuple[bool, str]:
        max_steps = self.task_cfg.get("max_steps", 60)
        if self.step_count >= max_steps:
            return True, "max_steps_reached"
        if self.incidents.queue_is_empty and self.step_count > 3:
            return True, "queue_empty"
        return False, ""
    def _build_unit_position_map(self) -> Dict[str, Tuple[str, float, float]]:
        positions: Dict[str, Tuple[str, float, float]] = {}
        for uid, unit in self.fleet.units.items():
            positions[uid] = (unit.current_zone_id, unit.lat, unit.lon)
        return positions
    def _build_fleet_available_map(self) -> Dict[str, int]:
        avail: Dict[str, int] = {z: 0 for z in self.active_zone_ids}
        for unit in self.fleet.units.values():
            if unit.is_available:
                avail[unit.current_zone_id] = avail.get(unit.current_zone_id, 0) + 1
        return avail
    @property
    def is_done(self) -> bool:
        return self._done
    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward
    @property
    def active_hospitals(self) -> Dict[str, Hospital]:
        return self.hospitals.hospitals
    @property
    def active_incidents(self) -> Dict[str, Incident]:
        return self.incidents.active_incidents
    @property
    def fleet_units(self) -> Dict[str, AmbulanceUnit]:
        return self.fleet.units
    def get_hospital(self, hospital_id: str) -> Optional[Hospital]:
        return self.hospitals.hospitals.get(hospital_id)
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        return self.incidents.get_incident(incident_id)
    def get_unit(self, unit_id: str) -> Optional[AmbulanceUnit]:
        return self.fleet.units.get(unit_id)
    def get_travel_time(self, from_zone: str, to_zone: str) -> float:
        return self.traffic.get_travel_time(from_zone, to_zone)
    def get_best_hospital_for_condition(
        self,
        from_zone:      str,
        condition_code: str,
        priority:       str = "P2",
    ) -> Tuple[Optional[str], float, float]:
        return self.hospitals.get_best_hospital(from_zone, condition_code, priority)
    def describe(self) -> Dict[str, Any]:
        return {
            "episode_id":       self.episode_id,
            "task_id":          self.task_id,
            "task_name":        self.task_cfg.get("name", ""),
            "step":             self.step_count,
            "sim_time_min":     round(self.sim_time_min, 1),
            "done":             self._done,
            "cumulative_reward":round(self._cumulative_reward, 4),
            "incidents":        self.incidents.describe(),
            "fleet":            self.fleet.describe(),
            "hospitals":        self.hospitals.describe(),
            "traffic":          self.traffic.describe(),
            "comms":            self.comms.describe(),
            "forecast":         self.forecast.describe(),
            "agency":           self.agency.describe(),
            "mutual_aid":       self.mutual_aid.describe(),
        }
class SimulationFactory:
    def __init__(self, seed: int = 42) -> None:
        self._seed         = seed
        self._task_id      = 1
        self._zones        = list(DEFAULT_ACTIVE_ZONES)
        self._weather      = None           
        self._start_min    = None
        self._warm_up_steps= 0
        self._log_level    = logging.WARNING
    def with_task(self, task_id: int) -> "SimulationFactory":
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"task_id must be 1-9, got {task_id}")
        self._task_id = task_id
        return self
    def with_zones(self, zone_ids: List[str]) -> "SimulationFactory":
        self._zones = list(zone_ids)
        return self
    def with_weather(self, weather: str) -> "SimulationFactory":
        self._weather = weather
        return self
    def with_start_time(self, sim_time_minutes: float) -> "SimulationFactory":
        self._start_min = sim_time_minutes
        return self
    def with_warm_up(self, steps: int) -> "SimulationFactory":
        self._warm_up_steps = steps
        return self
    def with_log_level(self, level: int) -> "SimulationFactory":
        self._log_level = level
        return self
    def build(self) -> SimulationBundle:
        logging.basicConfig(level=self._log_level)
        cfg  = TASK_CONFIGS.get(self._task_id, TASK_CONFIGS[1])
        t0   = self._start_min if self._start_min is not None else cfg["sim_start_min"]
        w    = self._weather   if self._weather   is not None else cfg["weather"]
        rng_master = random.Random(self._seed)
        def child_seed() -> int:
            return rng_master.randint(0, 2**31)
        traffic   = TrafficModel(seed=child_seed())
        hospitals = HospitalNetwork(seed=child_seed(), traffic_model=traffic)
        incidents = IncidentEngine(seed=child_seed())
        fleet     = FleetSimulator(
            seed=child_seed(),
            traffic_model    = traffic,
            hospital_network = hospitals,
        )
        comms     = CommunicationsManager(seed=child_seed(), task_id=self._task_id)
        forecast  = DemandForecaster(seed=child_seed())
        agency    = MultiAgencyCoordinator(seed=child_seed(), task_id=self._task_id)
        mutual    = MutualAidCoordinator(
            seed              = child_seed(),
            task_id           = self._task_id,
            traffic_model     = traffic,
            fleet_simulator   = fleet,
            hospital_network  = hospitals,
            incident_engine   = incidents,
            demand_forecaster = forecast,
            comms_manager     = comms,
        )
        bundle = SimulationBundle(
            traffic        = traffic,
            hospitals      = hospitals,
            incidents      = incidents,
            fleet          = fleet,
            comms          = comms,
            forecast       = forecast,
            agency         = agency,
            mutual_aid     = mutual,
            seed           = self._seed,
            task_id        = self._task_id,
            active_zone_ids= self._zones,
            sim_start_min  = t0,
            task_cfg       = cfg,
        )
        bundle.reset(
            task_id          = self._task_id,
            sim_time_minutes = t0,
            weather          = w,
            active_zone_ids  = self._zones,
        )
        if self._warm_up_steps > 0:
            warm_up(bundle, steps=self._warm_up_steps, silent=True)
        return bundle
def build_simulation(
    seed:       int  = 42,
    task_id:    int  = 1,
    zone_ids:   Optional[List[str]] = None,
    weather:    Optional[str] = None,
    warm_up_steps: int = 0,
) -> SimulationBundle:
    factory = (
        SimulationFactory(seed=seed)
        .with_task(task_id)
        .with_warm_up(warm_up_steps)
    )
    if zone_ids:
        factory = factory.with_zones(zone_ids)
    if weather:
        factory = factory.with_weather(weather)
    return factory.build()
def warm_up(
    bundle: SimulationBundle,
    steps:  int  = 5,
    silent: bool = False,
) -> SimulationBundle:
    for _ in range(steps):
        bundle.step()
    bundle.step_count        = 0
    bundle._cumulative_reward = 0.0
    bundle._step_rewards      = []
    bundle._done              = False
    bundle._done_reason       = ""
    if not silent:
        logger.info(
            "[SIM] warm-up complete — %d steps, "
            "traffic mean OD: %.1f min, hospitals diverted: %d",
            steps,
            bundle.traffic._mean_od(),
            bundle.hospitals.count_diverted(),
        )
    return bundle
def run_random_episode(
    seed:    int = 42,
    task_id: int = 4,
    verbose: bool = True,
) -> Tuple[SimulationBundle, Dict[str, Any]]:
    bundle = build_simulation(seed=seed, task_id=task_id)
    max_steps = bundle.task_cfg.get("max_steps", 60)
    if verbose:
        print(f"\n[random episode] task={task_id} seed={seed} max_steps={max_steps}")
    while not bundle.is_done:
        _, reward, done, info = bundle.step()
        if verbose and bundle.step_count % 10 == 0:
            print(
                f"  step={bundle.step_count:>3} "
                f"reward={reward:+.4f} "
                f"incidents={info['active_incidents']} "
                f"diverted={info['diverted_hospitals']} "
                f"p1={info['p1_count']}"
            )
        if bundle.step_count >= max_steps:
            break
    summary = bundle.get_episode_summary()
    if verbose:
        print(f"\n[random episode] done — {bundle._done_reason}")
        print(f"  total_reward = {summary['cumulative_reward']:.4f}")
        print(f"  total_steps  = {summary['total_steps']}")
    return bundle, summary
def get_task_info(task_id: int) -> Dict[str, Any]:
    cfg = TASK_CONFIGS.get(task_id)
    if cfg is None:
        raise ValueError(f"Unknown task_id={task_id}. Valid: 1-9")
    return {
        "task_id":        task_id,
        "name":           cfg["name"],
        "difficulty":     cfg["difficulty"],
        "baseline_score": cfg["baseline_score"],
        "max_steps":      cfg["max_steps"],
        "grader_weights": cfg["grader_weights"],
        "description":    cfg["description"],
        "features": {
            "comm_failure":       cfg["comm_failure"],
            "mci_enabled":        cfg["mci_enabled"],
            "mutual_aid_enabled": cfg["mutual_aid_enabled"],
            "traffic_dynamic":    cfg["traffic_dynamic"],
            "crew_fatigue":       cfg["crew_fatigue"],
            "diversion_enabled":  cfg["diversion_enabled"],
        },
    }
def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "task_id":        tid,
            "name":           cfg["name"],
            "difficulty":     cfg["difficulty"],
            "baseline_score": cfg["baseline_score"],
        }
        for tid, cfg in sorted(TASK_CONFIGS.items())
    ]
def validate_bundle(bundle: SimulationBundle) -> Dict[str, Any]:
    report: Dict[str, Any] = {"ok": True, "checks": {}}
    def check(name: str, fn) -> None:
        try:
            result = fn()
            report["checks"][name] = {"pass": True, "detail": result}
        except Exception as exc:
            report["checks"][name] = {"pass": False, "error": str(exc)}
            report["ok"] = False
    check("traffic_zones",
          lambda: len(bundle.traffic.all_zones))
    check("hospital_count",
          lambda: len(bundle.hospitals.hospitals))
    check("fleet_units",
          lambda: len(bundle.fleet.units))
    check("incident_engine",
          lambda: bundle.incidents.describe())
    check("comms_units",
          lambda: len(bundle.comms.unit_states))
    check("forecast_zones",
          lambda: len(bundle.forecast.zone_states))
    check("agency_pools",
          lambda: len(bundle.agency.agency_pools))
    check("mutual_aid_treaties",
          lambda: len(bundle.mutual_aid.treaties))
    check("observation_schema",
          lambda: list(bundle.get_combined_observation().keys()))
    return report
__all__ = [
    "SIMULATION_VERSION",
    "SIMULATION_AUTHOR",
    "SIMULATION_DESC",
    "TASK_CONFIGS",
    "DEFAULT_ACTIVE_ZONES",
    "STEP_DURATION_MINUTES",
    "SimulationBundle",
    "SimulationFactory",
    "build_simulation",
    "warm_up",
    "run_random_episode",
    "get_task_info",
    "list_tasks",
    "validate_bundle",
    "TrafficModel",
    "TrafficSnapshot",
    "RouteResult",
    "ZoneNode",
    "RoadLink",
    "CongestionEvent",
    "WeatherCondition",
    "CongestionCause",
    "HospitalNetwork",
    "Hospital",
    "InterHospitalTransfer",
    "BloodBank",
    "AdmissionType",
    "DiversionStatus",
    "HospitalTier",
    "TransferStatus",
    "DIVERSION_ER_THRESHOLD",
    "SURGE_CAPACITY_FACTOR",
    "IncidentEngine",
    "Incident",
    "MCIScene",
    "RPMScore",
    "SurvivalCurveParams",
    "IncidentPriority",
    "IncidentStatus",
    "StartTag",
    "IncidentUnitType",
    "AgencyRequirement",
    "MCIType",
    "INCIDENT_TEMPLATES",
    "FleetSimulator",
    "AmbulanceUnit",
    "CrewFatigueState",
    "DispatchRecord",
    "FleetMetrics",
    "UnitStatus",
    "FleetUnitType",
    "FleetMutualAidRequest",
    "GOLDEN_HOUR_WINDOWS",
    "FATIGUE_ONSET_HOURS",
    "MUTUAL_AID_DELAY_MINS",
    "CommunicationsManager",
    "CommMessage",
    "UnitCommState",
    "RelayTower",
    "JammingEvent",
    "ChannelStats",
    "CommChannel",
    "MessageType",
    "MessagePriority",
    "RelayTowerStatus",
    "TASK_COMM_FAILURE_PROB",
    "DemandForecaster",
    "DemandForecast",
    "DemandHeatmap",
    "SurgeAlert",
    "MCICluster",
    "PrepositionRecommendation",
    "ZoneDemandState",
    "SurgeLevel",
    "MCIRiskLevel",
    "MultiAgencyCoordinator",
    "MultiAgencyScene",
    "AgencyUnit",
    "CommandPost",
    "CoordinationViolation",
    "AgencyType",
    "SceneStage",
    "CoordinationViolationType",
    "MutualAidCoordinator",
    "MutualAidRequest",
    "AidUnit",
    "AidTreaty",
    "ConvoyDispatch",
    "CascadeEvent",
    "AidSurgeDeclaration",
    "PreStagedPosition",
    "MutualAidLedger",
    "ZoneCapacity",
    "AidTier",
    "AidStatus",
    "AidUnitType",
    "TreatyType",
    "CascadeStage",
    "OVER_REQUEST_PENALTY",
    "UNDER_REQUEST_TASK9_PENALTY",
    "TIER_DELAY_MINUTES",
]
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("=" * 72)
    print(f"EMERGI-ENV  ·  Simulation Package  v{SIMULATION_VERSION}")
    print("=" * 72)
    print("\n✓  Task list:")
    for t in list_tasks():
        print(f"   Task {t['task_id']:>2}: [{t['difficulty']:>6}] "
              f"{t['name']:<40} baseline={t['baseline_score']:.2f}")
    print("\n── Building Task 1 (easy) ──")
    t0 = time.perf_counter()
    bundle1 = build_simulation(seed=42, task_id=1)
    print(f"✓  Build time: {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"   Episode ID: {bundle1.episode_id}")
    report = validate_bundle(bundle1)
    print(f"✓  Validation: {'PASS' if report['ok'] else 'FAIL'}")
    for name, res in report["checks"].items():
        status = "✓" if res["pass"] else "✗"
        detail = res.get("detail", res.get("error", ""))
        print(f"   {status}  {name}: {detail}")
    print("\n── Stepping Task 1 (5 steps) ──")
    for s in range(5):
        obs, rwd, done, info = bundle1.step()
        print(f"  step={s+1} rwd={rwd:+.4f} "
              f"incidents={info['active_incidents']} "
              f"done={done}")
    print(f"  cumulative_reward: {bundle1.cumulative_reward:.4f}")
    d = bundle1.describe()
    print(f"\n✓  describe() keys: {list(d.keys())}")
    print("\n── Building Task 4 (medium, warm-up=3) ──")
    bundle4 = (
        SimulationFactory(seed=7)
        .with_task(4)
        .with_warm_up(3)
        .build()
    )
    print(f"✓  Task 4 ready — "
          f"incidents={bundle4.incidents.total_active} "
          f"fleet={len(bundle4.fleet.units)} units")
    print("\n── Task 9 info ──")
    info9 = get_task_info(9)
    print(f"  name:           {info9['name']}")
    print(f"  baseline_score: {info9['baseline_score']}")
    print(f"  features:       {info9['features']}")
    print("\n── Random episode Task 4 (10 steps) ──")
    bundle_r = build_simulation(seed=99, task_id=4)
    for _ in range(10):
        _, _, done, _ = bundle_r.step()
        if done:
            break
    summary = bundle_r.get_episode_summary()
    print(f"✓  Episode summary keys: {list(summary.keys())}")
    print(f"   fleet_metrics:  {summary['fleet_metrics']}")
    print(f"\n✅  server.simulation package smoke-test PASSED")
    print(f"    version={SIMULATION_VERSION}  tasks={len(TASK_CONFIGS)}  "
          f"subsystems=8  exports={len(__all__)}")