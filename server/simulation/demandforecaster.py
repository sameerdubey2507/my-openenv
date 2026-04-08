from __future__ import annotations
import json
import math
import random
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import numpy as np

_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent.parent / "data"

STEP_DURATION_MINUTES: float = 3.0
STEPS_PER_HOUR:        int   = int(60 / STEP_DURATION_MINUTES)          
LOOK_AHEAD_HOURS:      float = 12.0
LOOK_AHEAD_STEPS:      int   = int(LOOK_AHEAD_HOURS * STEPS_PER_HOUR)   

NOISE_FRACTION:          float = 0.20   
NOISE_CLIP_SIGMA:        float = 2.5    
ALPHA_LEVEL:             float = 0.30   
ALPHA_TREND:             float = 0.10   
ALPHA_SEASONAL:          float = 0.15   
HOLT_DAMPING:            float = 0.92   
BAYESIAN_PRIOR_WEIGHT:   float = 0.60   
BAYESIAN_OBS_WEIGHT:     float = 0.40   

CI_Z_90:                 float = 1.645  
CI_Z_95:                 float = 1.960  

BASELINE_RATE_PER_HOUR:  float = 0.18   
MAX_RATE_PER_HOUR:       float = 8.0    
SEASON_PERIODS:          int   = int(24 * STEPS_PER_HOUR)  

SURGE_THRESHOLD_SIGMA:   float = 2.0    
SURGE_CRITICAL_SIGMA:    float = 3.5    
SURGE_COOLDOWN_STEPS:    int   = 6      
SURGE_ROLLING_WINDOW:    int   = 40     

MCI_RISK_RATE_MULT:      float = 3.5    
MCI_CRITICAL_RATE_MULT:  float = 6.0    
MCI_CLUSTER_RADIUS_HOPS: int   = 2      

SPILLOVER_ATTENUATION:   float = 0.35   
SPILLOVER_HOP2_MULT:     float = 0.15   
PREPOSITION_TOP_K:       int   = 4      
PREPOSITION_HORIZON_MIN: float = 45.0   
COVERAGE_RADIUS_HOPS:    int   = 2      

TEMPORAL_BANDS: Dict[str, Tuple[float, float]] = {
    "00-06": (0.0,  6.0),
    "06-12": (6.0, 12.0),
    "12-18": (12.0, 18.0),
    "18-24": (18.0, 24.0),
}

PEAK_HOUR_PROFILES: List[Tuple[float, float, float]] = [
    (0.0,  5.0, 0.42),   
    (5.0,  7.0, 0.75),   
    (7.0,  9.0, 1.35),   
    (8.0, 10.0, 1.65),   
    (10.0,12.0, 1.20),   
    (12.0,13.0, 1.15),   
    (13.0,17.0, 1.05),   
    (17.0,19.0, 1.70),   
    (19.0,21.0, 1.25),   
    (21.0,23.0, 0.95),   
    (23.0,24.0, 0.65),   
]

ZONE_TYPE_DEMAND_MULT: Dict[str, float] = {
    "urban":      1.40,
    "suburban":   1.00,
    "peri_urban": 0.80,
    "rural":      0.50,
    "industrial": 1.10,
    "highway":    0.90,
}

WEATHER_DEMAND_MULT: Dict[str, float] = {
    "clear": 1.00,
    "haze":  1.04,
    "rain":  1.28,
    "fog":   1.12,
    "storm": 1.52,
}

CATEGORY_DEMAND_WEIGHTS: Dict[str, float] = {
    "cardiac":     1.00,
    "stroke":      0.78,
    "trauma":      1.25,
    "respiratory": 0.92,
    "obstetric":   0.55,
    "paediatric":  0.72,
    "neurological":0.68,
    "burns":       0.38,
    "industrial":  0.42,
    "psychiatric": 0.60,
    "medical":     1.18,
    "water_rescue":0.25,
    "routine":     1.60,
}

RISK_KEY_MAP: Dict[str, float] = {
    "cardiac_arrest_rate_per_100k": 0.35,
    "trauma_rate_per_100k":         0.30,
    "call_volume_per_day":          0.25,
    "industrial_hazard_risk":       0.10,
}

INDUSTRIAL_RISK_SCORES: Dict[str, float] = {
    "very_high": 4.0, "high": 3.0, "medium": 2.0,
    "low": 1.0, "very_low": 0.5,
}

class SurgeLevel(str, Enum):
    NONE     = "none"
    ELEVATED = "elevated"
    HIGH     = "high"
    CRITICAL = "critical"

class MCIRiskLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRITICAL = "critical"

@dataclass
class DemandObservation:
    obs_id:         str
    zone_id:        str
    category:       str
    priority:       str
    observed_at_min: float
    observed_at_step: int
    is_mci:         bool = False
    victim_count:   int  = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_id":          self.obs_id,
            "zone_id":         self.zone_id,
            "category":        self.category,
            "priority":        self.priority,
            "observed_at_min": round(self.observed_at_min, 1),
            "is_mci":          self.is_mci,
            "victim_count":    self.victim_count,
        }

@dataclass
class HoltState:
    level:  float   
    trend:  float   
    seasonal: List[float] = field(default_factory=list)   

    def forecast(self, h_steps: int, damp: float = HOLT_DAMPING) -> float:
        phi_sum = sum(damp ** m for m in range(1, h_steps + 1))
        seasonal_idx = (h_steps % len(self.seasonal)) if self.seasonal else 0
        s_factor = self.seasonal[seasonal_idx] if self.seasonal else 1.0
        raw = (self.level + phi_sum * self.trend) * max(s_factor, 0.05)
        return max(0.0, raw)

    def update(self, y: float, hour: float) -> None:
        hour_idx = int(hour) % 24
        s = self.seasonal[hour_idx] if self.seasonal else 1.0
        s_safe = max(s, 0.01)
        prev_level = self.level
        self.level = ALPHA_LEVEL * (y / s_safe) + (1 - ALPHA_LEVEL) * (self.level + self.trend)
        self.trend = ALPHA_TREND * (self.level - prev_level) + (1 - ALPHA_TREND) * HOLT_DAMPING * self.trend
        if self.seasonal:
            new_s = ALPHA_SEASONAL * (y / max(self.level, 0.01)) + (1 - ALPHA_SEASONAL) * s
            self.seasonal[hour_idx] = max(0.01, new_s)

@dataclass
class ZoneDemandState:
    zone_id:          str
    zone_type:        str
    population:       int
    pop_density:      float
    holt:             HoltState
    recent_rates:     Deque[float] = field(default_factory=lambda: deque(maxlen=SURGE_ROLLING_WINDOW))
    posterior_rate_hour: List[float] = field(default_factory=lambda: [BASELINE_RATE_PER_HOUR] * 24)
    category_weights:    Dict[str, float] = field(default_factory=dict)
    obs_buffer:          List[DemandObservation] = field(default_factory=list)
    obs_counts_by_step:  Deque[int] = field(default_factory=lambda: deque(maxlen=SURGE_ROLLING_WINDOW))
    surge_level:          SurgeLevel = SurgeLevel.NONE
    surge_since_step:     Optional[int] = None
    last_surge_alert_step: int = -SURGE_COOLDOWN_STEPS
    mci_risk_level:       MCIRiskLevel = MCIRiskLevel.LOW
    mci_risk_score:       float = 0.0
    total_observed:       int   = 0
    total_p1_observed:    int   = 0
    peak_step_count:      int   = 0
    steps_with_zero:      int   = 0
    spillover_demand:     float = 0.0

    _forecast_cache:      Optional[List[float]] = field(default=None, repr=False)
    _forecast_ci_lo:      Optional[List[float]] = field(default=None, repr=False)
    _forecast_ci_hi:      Optional[List[float]] = field(default=None, repr=False)
    _cache_valid_step:    int = -1

    def current_rate(self, hour: float) -> float:
        hour_idx = int(hour) % 24
        base     = self.posterior_rate_hour[hour_idx]
        return max(0.0, min(MAX_RATE_PER_HOUR, base))

    def add_observation(self, obs: DemandObservation) -> None:
        self.obs_buffer.append(obs)
        self.total_observed += 1
        if obs.priority == "P1":
            self.total_p1_observed += 1

    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "zone_id":           self.zone_id,
            "zone_type":         self.zone_type,
            "current_level":     round(self.holt.level, 4),
            "current_trend":     round(self.holt.trend, 4),
            "surge_level":       self.surge_level.value,
            "mci_risk_level":    self.mci_risk_level.value,
            "mci_risk_score":    round(self.mci_risk_score, 4),
            "recent_mean_rate":  round(float(np.mean(list(self.recent_rates)) if self.recent_rates else 0.0), 4),
            "spillover_demand":  round(self.spillover_demand, 4),
            "total_observed":    self.total_observed,
            "total_p1_observed": self.total_p1_observed,
        }

@dataclass
class DemandForecast:
    zone_id:        str
    generated_at_step: int
    generated_at_min:  float
    look_ahead_hours:  float
    point_forecast: List[float]      
    ci_lower_90:    List[float]      
    ci_upper_90:    List[float]      
    hourly_forecast:    List[float]  
    hourly_ci_lo:       List[float]
    hourly_ci_hi:       List[float]
    peak_step:       int   = 0
    peak_rate:       float = 0.0
    peak_hour:       float = 0.0
    total_expected:  float = 0.0
    risk_score:      float = 0.0   

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id":            self.zone_id,
            "generated_at_step":  self.generated_at_step,
            "generated_at_min":   round(self.generated_at_min, 1),
            "look_ahead_hours":   self.look_ahead_hours,
            "hourly_forecast":    [round(v, 3) for v in self.hourly_forecast],
            "hourly_ci_lo":       [round(v, 3) for v in self.hourly_ci_lo],
            "hourly_ci_hi":       [round(v, 3) for v in self.hourly_ci_hi],
            "peak_step":          self.peak_step,
            "peak_rate":          round(self.peak_rate, 3),
            "peak_hour":          round(self.peak_hour, 2),
            "total_expected":     round(self.total_expected, 3),
            "risk_score":         round(self.risk_score, 4),
        }

    def to_compact_dict(self) -> Dict[str, Any]:
        return {
            "zone_id":        self.zone_id,
            "hourly_forecast":[round(v, 3) for v in self.hourly_forecast],
            "peak_hour":      round(self.peak_hour, 1),
            "total_expected": round(self.total_expected, 2),
            "risk_score":     round(self.risk_score, 4),
        }

@dataclass
class SurgeAlert:
    alert_id:    str
    zone_id:     str
    level:       SurgeLevel
    z_score:     float
    observed_rate:  float
    expected_rate:  float
    detected_at_step: int
    detected_at_min:  float
    resolved_at_step: Optional[int] = None
    active:      bool  = True
    contributing_zones: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id":           self.alert_id,
            "zone_id":            self.zone_id,
            "level":              self.level.value,
            "z_score":            round(self.z_score, 3),
            "observed_rate":      round(self.observed_rate, 3),
            "expected_rate":      round(self.expected_rate, 3),
            "detected_at_step":   self.detected_at_step,
            "detected_at_min":    round(self.detected_at_min, 1),
            "resolved_at_step":   self.resolved_at_step,
            "active":             self.active,
            "contributing_zones": self.contributing_zones,
        }

@dataclass
class MCICluster:
    cluster_id:    str
    epicentre_zone: str
    member_zones:  List[str]
    risk_level:    MCIRiskLevel
    aggregate_rate: float
    risk_score:    float
    detected_at_step: int
    detected_at_min:  float
    recommended_mutual_aid: bool = False
    estimated_victims:  int  = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id":             self.cluster_id,
            "epicentre_zone":         self.epicentre_zone,
            "member_zones":           self.member_zones,
            "risk_level":             self.risk_level.value,
            "aggregate_rate":         round(self.aggregate_rate, 3),
            "risk_score":             round(self.risk_score, 4),
            "detected_at_step":       self.detected_at_step,
            "detected_at_min":        round(self.detected_at_min, 1),
            "recommended_mutual_aid": self.recommended_mutual_aid,
            "estimated_victims":      self.estimated_victims,
        }

@dataclass
class PrepositionRecommendation:
    rec_id:           str
    zone_id:          str
    priority_rank:    int
    demand_score:     float      
    coverage_zones:   List[str]  
    expected_demand_covered: float
    recommended_unit_type:   str   
    recommended_count:       int
    justification:    str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rec_id":                  self.rec_id,
            "zone_id":                 self.zone_id,
            "priority_rank":           self.priority_rank,
            "demand_score":            round(self.demand_score, 4),
            "coverage_zones":          self.coverage_zones,
            "expected_demand_covered": round(self.expected_demand_covered, 3),
            "recommended_unit_type":   self.recommended_unit_type,
            "recommended_count":       self.recommended_count,
            "justification":           self.justification,
        }

@dataclass
class DemandHeatmap:
    step:          int
    sim_time_min:  float
    zone_ids:      List[str]
    demand_scores: List[float]    
    raw_rates:     List[float]    
    surge_flags:   List[bool]
    mci_risk_flags: List[bool]
    weather:       str
    peak_active:   bool

    def to_observation_array(self) -> "np.ndarray":
        return np.array(self.demand_scores, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":          self.step,
            "sim_time_min":  round(self.sim_time_min, 1),
            "zone_count":    len(self.zone_ids),
            "zones":         self.zone_ids,
            "demand_scores": [round(v, 4) for v in self.demand_scores],
            "raw_rates":     [round(v, 4) for v in self.raw_rates],
            "surge_flags":   self.surge_flags,
            "mci_risk_flags":self.mci_risk_flags,
            "weather":       self.weather,
            "peak_active":   self.peak_active,
        }

class DemandForecaster:
    def __init__(
        self,
        seed:      int  = 42,
        zone_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.seed    = seed
        self.rng     = random.Random(seed)
        self.np_rng  = np.random.RandomState(seed)

        self._zone_meta:   Dict[str, Any]       = zone_data or {}
        self._adjacency:   Dict[str, List[str]] = {}
        self.zone_states:  Dict[str, ZoneDemandState] = {}
        self.active_zone_ids: List[str] = []

        self._historical_baseline: Dict[str, Dict[str, float]] = {}
        self._history_loaded: bool = False

        self._surge_alerts:    Dict[str, SurgeAlert]  = {}
        self._alert_seq:       int = 0
        self.active_surges:    Dict[str, SurgeAlert]  = {}

        self._mci_clusters:    List[MCICluster] = []
        self._cluster_seq:     int = 0

        self._forecasts:       Dict[str, DemandForecast] = {}
        self._forecast_dirty:  bool = True
        self._last_heatmap:    Optional[DemandHeatmap] = None

        self._obs_log:         List[DemandObservation] = []
        self._obs_seq:         int = 0
        self._step_rewards:    List[float] = []
        self._total_reward:    float = 0.0
        self._surge_alert_log: List[SurgeAlert] = []
        self._total_obs:       int = 0

        self.sim_time_min:     float = 480.0
        self.step_count:       int   = 0
        self.task_id:          int   = 1
        self._weather:         str   = "clear"

        self._load_zone_data()
        self._load_historical_baseline()

    def reset(
        self,
        active_zone_ids:  List[str],
        task_id:          int   = 1,
        sim_time_minutes: float = 480.0,
        weather:          str   = "clear",
    ) -> Dict[str, Any]:
        self.task_id       = task_id
        self.sim_time_min  = sim_time_minutes
        self.step_count    = 0
        self._weather      = weather

        self.rng    = random.Random(self.seed + task_id * 23)
        self.np_rng = np.random.RandomState(self.seed + task_id * 23)

        self.zone_states    = {}
        self._surge_alerts  = {}
        self._alert_seq     = 0
        self.active_surges  = {}
        self._mci_clusters  = []
        self._cluster_seq   = 0
        self._forecasts     = {}
        self._forecast_dirty = True
        self._last_heatmap  = None
        self._obs_log       = []
        self._obs_seq       = 0
        self._step_rewards  = []
        self._total_reward  = 0.0
        self._surge_alert_log = []
        self._total_obs     = 0

        for zid in active_zone_ids:
            if zid not in self._zone_meta:
                self._zone_meta[zid] = self._synthetic_zone_meta(zid)

        self.active_zone_ids = list(active_zone_ids)

        for zid in self.active_zone_ids:
            self.zone_states[zid] = self._build_zone_state(zid, sim_time_minutes)

        active_set = set(self.active_zone_ids)
        for zid in self.active_zone_ids:
            raw_adj = self._zone_meta.get(zid, {}).get("adjacent_zones", [])
            self._adjacency[zid] = [n for n in raw_adj if n in active_set]

        self._refresh_all_forecasts()
        self._update_heatmap()

        return self.get_observation_dict()

    def step(
        self,
        sim_time_minutes: float,
        weather:          str = "clear",
        active_incident_zones: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        self.sim_time_min = sim_time_minutes
        self.step_count  += 1
        self._weather     = weather

        step_reward        = 0.0

        for zid in self.active_zone_ids:
            state = self.zone_states[zid]
            obs_this_step = len(state.obs_buffer)
            state.obs_counts_by_step.append(obs_this_step)

            hour = self._hour_of_day()
            step_rate = obs_this_step / STEP_DURATION_MINUTES * 60.0  
            state.recent_rates.append(step_rate)

            state.holt.update(obs_this_step, hour)
            self._bayesian_update(state, obs_this_step, hour)

            reward_delta = self._detect_surge(state)
            step_reward += reward_delta

            self._update_mci_risk(state)

            if obs_this_step > 0:
                state.peak_step_count = max(state.peak_step_count, obs_this_step)
            else:
                state.steps_with_zero += 1

            state.obs_buffer.clear()

        self._propagate_spillover()

        if self.step_count % 5 == 0 or self._forecast_dirty:
            self._refresh_all_forecasts()
            self._forecast_dirty = False

        if self.step_count % 10 == 0:
            self._scan_mci_clusters()

        step_reward += self._resolve_stale_surges()
        self._update_heatmap()
        self._step_rewards.append(step_reward)
        self._total_reward += step_reward

        return self.get_observation_dict(), step_reward

    def observe_incident(
        self,
        zone_id:   str,
        category:  str,
        priority:  str,
        is_mci:    bool = False,
        victim_count: int = 1,
    ) -> Optional[DemandObservation]:
        state = self.zone_states.get(zone_id)
        if state is None:
            return None

        self._obs_seq += 1
        obs = DemandObservation(
            obs_id          = f"OBS-{self._obs_seq:06d}",
            zone_id         = zone_id,
            category        = category,
            priority        = priority,
            observed_at_min = self.sim_time_min,
            observed_at_step= self.step_count,
            is_mci          = is_mci,
            victim_count    = victim_count,
        )
        state.add_observation(obs)
        self._obs_log.append(obs)
        self._total_obs += 1
        self._forecast_dirty = True
        return obs

    def observe_incidents_bulk(
        self,
        incidents: List[Dict[str, Any]],
    ) -> int:
        count = 0
        for inc in incidents:
            if "zone_id" not in inc:
                continue
            self.observe_incident(
                zone_id      = inc["zone_id"],
                category     = inc.get("incident_category", "medical"),
                priority     = inc.get("reported_priority", "P2"),
                is_mci       = inc.get("is_mci", False),
                victim_count = inc.get("victim_count", 1),
            )
            count += 1
        return count

    def get_zone_forecast(
        self,
        zone_id:       str,
        noise:         bool = True,
        horizon_hours: float = LOOK_AHEAD_HOURS,
    ) -> Optional[DemandForecast]:
        fc = self._forecasts.get(zone_id)
        if fc is None:
            fc = self._compute_zone_forecast(zone_id, noise=noise, horizon_hours=horizon_hours)
            self._forecasts[zone_id] = fc
        return fc

    def get_all_forecasts(
        self,
        noise: bool = True,
    ) -> Dict[str, DemandForecast]:
        if self._forecast_dirty:
            self._refresh_all_forecasts(noise=noise)
        return dict(self._forecasts)

    def get_demand_heatmap(self) -> DemandHeatmap:
        if self._last_heatmap is None:
            self._update_heatmap()
        return self._last_heatmap  

    def get_heatmap_observation(self) -> Dict[str, Any]:
        hm = self.get_demand_heatmap()
        return hm.to_dict()

    def get_observation_dict(self) -> Dict[str, Any]:
        hm    = self.get_demand_heatmap()
        surge = self._get_active_surge_obs()
        mci   = [c.to_dict() for c in self._mci_clusters if c.risk_level != MCIRiskLevel.LOW]

        zone_obs = {
            zid: state.to_observation_dict()
            for zid, state in self.zone_states.items()
        }

        forecast_compact = {
            zid: fc.to_compact_dict()
            for zid, fc in self._forecasts.items()
        }

        return {
            "step_count":           self.step_count,
            "sim_time_min":         round(self.sim_time_min, 1),
            "hour_of_day":          round(self._hour_of_day(), 2),
            "weather":              self._weather,
            "peak_active":          self._is_peak_hour(),
            "task_id":              self.task_id,
            "active_zones":         len(self.active_zone_ids),
            "heatmap":              hm.to_dict(),
            "zone_states":          zone_obs,
            "forecasts":            forecast_compact,
            "active_surge_alerts":  surge,
            "mci_clusters_at_risk": mci,
            "top_demand_zones":     self.get_hotspot_zones(k=5),
            "demand_forecast_noise":NOISE_FRACTION,
            "total_observed_this_episode": self._total_obs,
        }

    def get_hotspot_zones(
        self,
        k:              int = 5,
        horizon_minutes: float = PREPOSITION_HORIZON_MIN,
    ) -> List[Dict[str, Any]]:
        horizon_steps = int(horizon_minutes / STEP_DURATION_MINUTES)
        scores: List[Tuple[str, float]] = []

        for zid, state in self.zone_states.items():
            fc = self._forecasts.get(zid)
            if fc is None:
                scores.append((zid, 0.0))
                continue

            demand = sum(fc.point_forecast[:horizon_steps])

            surge_mult = {
                SurgeLevel.NONE:     1.0,
                SurgeLevel.ELEVATED: 1.3,
                SurgeLevel.HIGH:     1.6,
                SurgeLevel.CRITICAL: 2.0,
            }[state.surge_level]

            mci_mult = {
                MCIRiskLevel.LOW:      1.0,
                MCIRiskLevel.MODERATE: 1.2,
                MCIRiskLevel.HIGH:     1.5,
                MCIRiskLevel.CRITICAL: 2.0,
            }[state.mci_risk_level]

            weighted = demand * surge_mult * mci_mult
            scores.append((zid, weighted))

        scores.sort(key=lambda x: -x[1])

        result = []
        for rank, (zid, score) in enumerate(scores[:k], 1):
            state = self.zone_states[zid]
            result.append({
                "rank":          rank,
                "zone_id":       zid,
                "demand_score":  round(float(score), 3),
                "surge_level":   state.surge_level.value,
                "mci_risk":      state.mci_risk_level.value,
            })
        return result

    def get_preposition_recommendations(
        self,
        fleet_state:       Optional[Dict[str, Any]] = None,
        n_recommendations: int = PREPOSITION_TOP_K,
        horizon_minutes:   float = PREPOSITION_HORIZON_MIN,
    ) -> List[PrepositionRecommendation]:
        horizon_steps = int(horizon_minutes / STEP_DURATION_MINUTES)
        zone_demand: Dict[str, float] = {}

        for zid in self.active_zone_ids:
            fc = self._forecasts.get(zid)
            if fc:
                zone_demand[zid] = sum(fc.point_forecast[:horizon_steps])
            else:
                zone_demand[zid] = 0.0

        candidate_scores: List[Tuple[str, float, List[str], float]] = []

        for anchor_zid in self.active_zone_ids:
            coverage = self._get_coverage_zones(anchor_zid, COVERAGE_RADIUS_HOPS)
            covered_demand = 0.0

            for cov_zid in coverage:
                state = self.zone_states.get(cov_zid)
                if state is None:
                    continue
                d = zone_demand.get(cov_zid, 0.0)
                pop_w = min(2.0, math.log1p(state.pop_density) / math.log1p(10000))

                surge_bonus = {
                    SurgeLevel.NONE:     0.0,
                    SurgeLevel.ELEVATED: 0.20,
                    SurgeLevel.HIGH:     0.50,
                    SurgeLevel.CRITICAL: 1.00,
                }[state.surge_level]

                mci_bonus = {
                    MCIRiskLevel.LOW:      0.0,
                    MCIRiskLevel.MODERATE: 0.25,
                    MCIRiskLevel.HIGH:     0.60,
                    MCIRiskLevel.CRITICAL: 1.20,
                }[state.mci_risk_level]

                covered_demand += d * pop_w * (1.0 + surge_bonus + mci_bonus)

            anchor_state = self.zone_states[anchor_zid]
            zone_type_mult = ZONE_TYPE_DEMAND_MULT.get(anchor_state.zone_type, 1.0)
            score = covered_demand * zone_type_mult

            candidate_scores.append((anchor_zid, score, coverage, covered_demand))

        candidate_scores.sort(key=lambda x: -x[1])

        recommendations: List[PrepositionRecommendation] = []
        seen_coverage: Set[str] = set()
        rec_id_seq = 0

        for anchor_zid, score, coverage, covered_demand in candidate_scores:
            if len(recommendations) >= n_recommendations:
                break

            novel = [z for z in coverage if z not in seen_coverage]
            if not novel and recommendations:
                continue

            state = self.zone_states[anchor_zid]
            unit_type = self._recommend_unit_type(anchor_zid, horizon_steps)
            unit_count = max(1, int(math.ceil(
                sum(zone_demand.get(z, 0) for z in coverage[:3]) / 3.0
            )))
            unit_count = min(unit_count, 3)

            justification = (
                f"Zone {anchor_zid} covers {len(coverage)} zones with "
                f"{covered_demand:.1f} expected incidents "
                f"in next {horizon_minutes:.0f} min"
            )
            if state.surge_level != SurgeLevel.NONE:
                justification += f"; SURGE={state.surge_level.value}"
            if state.mci_risk_level != MCIRiskLevel.LOW:
                justification += f"; MCI_RISK={state.mci_risk_level.value}"

            rec_id_seq += 1
            rec = PrepositionRecommendation(
                rec_id                 = f"REC-{rec_id_seq:04d}",
                zone_id                = anchor_zid,
                priority_rank          = len(recommendations) + 1,
                demand_score           = round(score, 4),
                coverage_zones         = coverage,
                expected_demand_covered= round(covered_demand, 3),
                recommended_unit_type  = unit_type,
                recommended_count      = unit_count,
                justification          = justification,
            )
            recommendations.append(rec)
            seen_coverage.update(coverage)

        return recommendations

    def get_surge_alerts(self, active_only: bool = True) -> List[Dict[str, Any]]:
        alerts = list(self._surge_alerts.values())
        if active_only:
            alerts = [a for a in alerts if a.active]
        return [a.to_dict() for a in alerts]

    def get_mci_clusters(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._mci_clusters]

    def get_zone_demand_matrix(self) -> "np.ndarray":
        n = len(self.active_zone_ids)
        matrix = np.zeros((n, LOOK_AHEAD_STEPS), dtype=np.float32)
        for i, zid in enumerate(self.active_zone_ids):
            fc = self._forecasts.get(zid)
            if fc:
                matrix[i, :] = np.array(fc.point_forecast[:LOOK_AHEAD_STEPS],
                                        dtype=np.float32)
        return matrix

    def get_hourly_zone_matrix(self) -> "np.ndarray":
        n = len(self.active_zone_ids)
        matrix = np.zeros((n, int(LOOK_AHEAD_HOURS)), dtype=np.float32)
        for i, zid in enumerate(self.active_zone_ids):
            fc = self._forecasts.get(zid)
            if fc:
                for h, val in enumerate(fc.hourly_forecast):
                    if h < int(LOOK_AHEAD_HOURS):
                        matrix[i, h] = val
        return matrix

    def get_episode_analytics(self) -> Dict[str, Any]:
        total_obs    = self._total_obs
        n_surges     = len(self._surge_alert_log)
        n_mci_risk   = sum(
            1 for c in self._mci_clusters
            if c.risk_level in (MCIRiskLevel.HIGH, MCIRiskLevel.CRITICAL)
        )

        by_zone: Dict[str, Dict] = {}
        for zid, state in self.zone_states.items():
            by_zone[zid] = {
                "total_observed":      state.total_observed,
                "total_p1_observed":   state.total_p1_observed,
                "peak_step_count":     state.peak_step_count,
                "final_surge_level":   state.surge_level.value,
                "final_mci_risk":      state.mci_risk_level.value,
                "holt_level":          round(state.holt.level, 4),
                "holt_trend":          round(state.holt.trend, 4),
            }

        return {
            "episode_steps":            self.step_count,
            "total_incidents_observed": total_obs,
            "surge_alerts_total":       n_surges,
            "mci_risk_clusters":        n_mci_risk,
            "zones_ever_surged":        len(set(a.zone_id for a in self._surge_alert_log)),
            "total_step_reward":        round(self._total_reward, 4),
            "mean_step_reward":         round(
                self._total_reward / max(self.step_count, 1), 4
            ),
            "weather":                  self._weather,
            "zone_analytics":           by_zone,
        }

    def get_forecast_accuracy_estimate(self) -> Dict[str, float]:
        mape_vals: List[float] = []
        rmse_vals: List[float] = []

        for zid, state in self.zone_states.items():
            if len(state.obs_counts_by_step) < 5:
                continue
            actuals   = list(state.obs_counts_by_step)
            fc        = self._forecasts.get(zid)
            if fc is None:
                continue

            n = min(len(actuals), 10)
            for i in range(n):
                actual  = actuals[-(n - i)]
                predict = fc.point_forecast[i] if i < len(fc.point_forecast) else 0.0
                if actual > 0:
                    mape_vals.append(abs(actual - predict) / max(actual, 1e-3))
                rmse_vals.append((actual - predict) ** 2)

        return {
            "mape": round(float(np.mean(mape_vals)) if mape_vals else 0.0, 4),
            "rmse": round(float(np.sqrt(np.mean(rmse_vals))) if rmse_vals else 0.0, 4),
            "n_samples": len(mape_vals),
        }

    def describe(self) -> Dict[str, Any]:
        hour = self._hour_of_day()
        surging = [zid for zid, s in self.zone_states.items()
                   if s.surge_level != SurgeLevel.NONE]
        mci_risk = [zid for zid, s in self.zone_states.items()
                    if s.mci_risk_level != MCIRiskLevel.LOW]

        return {
            "step":            self.step_count,
            "sim_time_min":    round(self.sim_time_min, 1),
            "hour":            f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
            "task_id":         self.task_id,
            "active_zones":    len(self.active_zone_ids),
            "total_observed":  self._total_obs,
            "surging_zones":   surging,
            "mci_risk_zones":  mci_risk,
            "active_surges":   len([a for a in self._surge_alerts.values() if a.active]),
            "mci_clusters":    len(self._mci_clusters),
            "weather":         self._weather,
            "peak_active":     self._is_peak_hour(),
            "total_reward":    round(self._total_reward, 4),
        }

    def update_weather(self, weather: str) -> None:
        self._weather = weather
        self._forecast_dirty = True

    def _build_zone_state(self, zone_id: str, sim_time_min: float) -> ZoneDemandState:
        z = self._zone_meta.get(zone_id, {})
        zone_type   = z.get("zone_type", "urban")
        population  = int(z.get("population", 500_000))
        pop_density = float(z.get("population_density_per_sqkm", 200))

        seasonal = self._build_seasonal_array(zone_id)
        hour      = (sim_time_min / 60.0) % 24.0
        hour_idx  = int(hour) % 24
        baseline_rate = self._get_historical_rate(zone_id, hour_idx)

        holt = HoltState(
            level    = baseline_rate,
            trend    = 0.0,
            seasonal = seasonal,
        )

        posterior = []
        for h in range(24):
            rate = self._get_historical_rate(zone_id, h)
            rate *= ZONE_TYPE_DEMAND_MULT.get(zone_type, 1.0)
            density_mult = min(2.5, math.log1p(pop_density) / math.log1p(2000))
            rate *= density_mult
            rate_per_step = (rate / 60.0) * STEP_DURATION_MINUTES
            posterior.append(max(0.0, rate_per_step))

        cat_weights = {cat: CATEGORY_DEMAND_WEIGHTS.get(cat, 1.0)
                       for cat in CATEGORY_DEMAND_WEIGHTS}

        state = ZoneDemandState(
            zone_id          = zone_id,
            zone_type        = zone_type,
            population       = population,
            pop_density      = pop_density,
            holt             = holt,
            posterior_rate_hour = posterior,
            category_weights = cat_weights,
        )
        return state

    def _build_seasonal_array(self, zone_id: str) -> List[float]:
        z = self._zone_meta.get(zone_id, {})
        weights = z.get("demand_heatmap_weights", {
            "00-06": 0.12, "06-12": 0.28, "12-18": 0.32, "18-24": 0.28
        })
        band_mean = {
            "00-06": 0.12, "06-12": 0.28, "12-18": 0.32, "18-24": 0.28
        }
        seasonal = []
        for h in range(24):
            band = self._hour_to_band(h)
            w = weights.get(band, band_mean.get(band, 0.25))
            tod_mult = self._tod_multiplier(float(h))
            seasonal.append(max(0.1, w * 4.0 * tod_mult))

        mean_s = sum(seasonal) / 24.0
        if mean_s > 0:
            seasonal = [s / mean_s for s in seasonal]
        return seasonal

    def _bayesian_update(
        self,
        state:         ZoneDemandState,
        obs_count:     int,
        hour:          float,
    ) -> None:
        hour_idx = int(hour) % 24
        prior_rate = self._get_historical_rate_per_step(state.zone_id, hour_idx)

        alpha_prior = max(prior_rate * 10.0, 0.01)   
        beta_prior  = 10.0
        alpha_post  = alpha_prior + obs_count
        beta_post   = beta_prior + 1.0
        posterior_mean = alpha_post / beta_post

        blend = BAYESIAN_OBS_WEIGHT if self.step_count > 10 else 0.1
        blended = (1.0 - blend) * prior_rate + blend * posterior_mean

        weather_mult = WEATHER_DEMAND_MULT.get(self._weather, 1.0)
        tod_mult = self._tod_multiplier(hour)

        updated = blended * weather_mult * tod_mult
        updated = max(0.0, min(MAX_RATE_PER_HOUR / 60.0 * STEP_DURATION_MINUTES, updated))

        old_val = state.posterior_rate_hour[hour_idx]
        state.posterior_rate_hour[hour_idx] = (
            (1 - ALPHA_LEVEL) * old_val + ALPHA_LEVEL * updated
        )

    def _refresh_all_forecasts(self, noise: bool = True) -> None:
        for zid in self.active_zone_ids:
            fc = self._compute_zone_forecast(zid, noise=noise)
            self._forecasts[zid] = fc

    def _compute_zone_forecast(
        self,
        zone_id:       str,
        noise:         bool  = True,
        horizon_hours: float = LOOK_AHEAD_HOURS,
    ) -> DemandForecast:
        state    = self.zone_states.get(zone_id)
        n_steps  = int(horizon_hours * STEPS_PER_HOUR)
        hour_now = self._hour_of_day()

        point_forecast: List[float] = []
        ci_lo_90:       List[float] = []
        ci_hi_90:       List[float] = []

        spillover = state.spillover_demand if state else 0.0

        for step_offset in range(n_steps):
            future_minutes = self.sim_time_min + step_offset * STEP_DURATION_MINUTES
            future_hour    = (future_minutes / 60.0) % 24.0
            hour_idx       = int(future_hour) % 24

            if state:
                h = step_offset + 1
                holt_fc = state.holt.forecast(h)
                posterior = state.posterior_rate_hour[hour_idx]
                base = (BAYESIAN_PRIOR_WEIGHT * holt_fc +
                        BAYESIAN_OBS_WEIGHT   * posterior)
            else:
                base = BASELINE_RATE_PER_HOUR / 60.0 * STEP_DURATION_MINUTES

            weather_mult = WEATHER_DEMAND_MULT.get(self._weather, 1.0)
            tod_mult     = self._tod_multiplier(future_hour)
            zone_type_m  = ZONE_TYPE_DEMAND_MULT.get(
                state.zone_type if state else "urban", 1.0
            )

            rate = base * weather_mult * tod_mult * zone_type_m + spillover
            spillover *= 0.85
            rate = max(0.0, rate)

            if noise:
                sigma  = rate * NOISE_FRACTION
                raw_noise = self.np_rng.normal(0.0, sigma)
                clipped   = np.clip(raw_noise, -NOISE_CLIP_SIGMA * sigma,
                                               NOISE_CLIP_SIGMA * sigma)
                noisy_rate = max(0.0, rate + clipped)
            else:
                noisy_rate = rate

            se_rate = math.sqrt(max(noisy_rate, 0.01))
            lo_90   = max(0.0, noisy_rate - CI_Z_90 * se_rate)
            hi_90   = noisy_rate + CI_Z_90 * se_rate

            point_forecast.append(round(noisy_rate, 5))
            ci_lo_90.append(round(lo_90, 5))
            ci_hi_90.append(round(hi_90, 5))

        hourly_forecast, hourly_lo, hourly_hi = self._aggregate_hourly(
            point_forecast, ci_lo_90, ci_hi_90
        )

        peak_step = int(np.argmax(point_forecast)) if point_forecast else 0
        peak_rate = point_forecast[peak_step] if point_forecast else 0.0
        peak_hour = (hour_now + peak_step * STEP_DURATION_MINUTES / 60.0) % 24.0
        total_expected = sum(point_forecast)

        baseline_total = (BASELINE_RATE_PER_HOUR / 60.0 * STEP_DURATION_MINUTES
                          * n_steps)
        risk_score = min(1.0, total_expected / max(baseline_total * 5, 0.01))

        return DemandForecast(
            zone_id            = zone_id,
            generated_at_step  = self.step_count,
            generated_at_min   = self.sim_time_min,
            look_ahead_hours   = horizon_hours,
            point_forecast     = point_forecast,
            ci_lower_90        = ci_lo_90,
            ci_upper_90        = ci_hi_90,
            hourly_forecast    = hourly_forecast,
            hourly_ci_lo       = hourly_lo,
            hourly_ci_hi       = hourly_hi,
            peak_step          = peak_step,
            peak_rate          = peak_rate,
            peak_hour          = peak_hour,
            total_expected     = total_expected,
            risk_score         = risk_score,
        )

    @staticmethod
    def _aggregate_hourly(
        point: List[float],
        lo:    List[float],
        hi:    List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        hourly_fc:  List[float] = []
        hourly_lo:  List[float] = []
        hourly_hi:  List[float] = []

        per_hour = STEPS_PER_HOUR
        total    = len(point)
        n_hours  = int(math.ceil(total / per_hour))

        for h in range(n_hours):
            lo_idx = h * per_hour
            hi_idx = min((h + 1) * per_hour, total)
            if lo_idx >= total:
                break
            hourly_fc.append(round(sum(point[lo_idx:hi_idx]), 4))
            hourly_lo.append(round(sum(lo[lo_idx:hi_idx]),    4))
            hourly_hi.append(round(sum(hi[lo_idx:hi_idx]),    4))

        return hourly_fc, hourly_lo, hourly_hi

    def _detect_surge(self, state: ZoneDemandState) -> float:
        if len(state.recent_rates) < 5:
            return 0.0

        rates  = np.array(list(state.recent_rates))
        mu     = float(np.mean(rates[:-1]))      
        sigma  = float(np.std(rates[:-1])) + 1e-4   
        latest = rates[-1]

        z      = (latest - mu) / sigma
        new_level: SurgeLevel

        if z >= SURGE_CRITICAL_SIGMA:
            new_level = SurgeLevel.CRITICAL
        elif z >= SURGE_THRESHOLD_SIGMA * 1.5:
            new_level = SurgeLevel.HIGH
        elif z >= SURGE_THRESHOLD_SIGMA:
            new_level = SurgeLevel.ELEVATED
        else:
            new_level = SurgeLevel.NONE

        old_level = state.surge_level
        state.surge_level = new_level
        reward = 0.0

        if (new_level != SurgeLevel.NONE and
                self.step_count - state.last_surge_alert_step >= SURGE_COOLDOWN_STEPS):
            state.last_surge_alert_step = self.step_count
            if new_level != old_level:
                self._emit_surge_alert(state.zone_id, new_level, z, latest, mu)
                reward -= 0.02

        if old_level != SurgeLevel.NONE and new_level == SurgeLevel.NONE:
            self._resolve_zone_surges(state.zone_id)
            reward += 0.01   

        return reward

    def _emit_surge_alert(
        self,
        zone_id:       str,
        level:         SurgeLevel,
        z_score:       float,
        observed_rate: float,
        expected_rate: float,
    ) -> SurgeAlert:
        self._alert_seq += 1
        alert = SurgeAlert(
            alert_id          = f"SURGE-{self._alert_seq:04d}",
            zone_id           = zone_id,
            level             = level,
            z_score           = round(z_score, 3),
            observed_rate     = round(observed_rate, 4),
            expected_rate     = round(expected_rate, 4),
            detected_at_step  = self.step_count,
            detected_at_min   = self.sim_time_min,
            contributing_zones= self._adjacency.get(zone_id, [])[:3],
        )
        self._surge_alerts[alert.alert_id] = alert
        self.active_surges[zone_id]        = alert
        self._surge_alert_log.append(alert)
        return alert

    def _resolve_stale_surges(self) -> float:
        reward = 0.0
        for alert in list(self._surge_alerts.values()):
            if not alert.active:
                continue
            state = self.zone_states.get(alert.zone_id)
            if state and state.surge_level == SurgeLevel.NONE:
                alert.active            = False
                alert.resolved_at_step  = self.step_count
                self.active_surges.pop(alert.zone_id, None)
                reward += 0.005
        return reward

    def _resolve_zone_surges(self, zone_id: str) -> None:
        for alert in self._surge_alerts.values():
            if alert.zone_id == zone_id and alert.active:
                alert.active           = False
                alert.resolved_at_step = self.step_count
        self.active_surges.pop(zone_id, None)

    def _update_mci_risk(self, state: ZoneDemandState) -> None:
        if not state.recent_rates:
            state.mci_risk_level = MCIRiskLevel.LOW
            state.mci_risk_score = 0.0
            return

        hour     = self._hour_of_day()
        baseline = self._get_historical_rate_per_step(state.zone_id, int(hour) % 24)
        baseline = max(baseline, 1e-4)

        recent   = float(np.mean(list(state.recent_rates)[-5:]))
        rate_mult = recent / baseline

        adj_mean = 0.0
        adj_zones = self._adjacency.get(state.zone_id, [])
        for adj_zid in adj_zones:
            adj_state = self.zone_states.get(adj_zid)
            if adj_state and adj_state.recent_rates:
                adj_rate = float(np.mean(list(adj_state.recent_rates)[-3:]))
                adj_base = self._get_historical_rate_per_step(adj_zid, int(hour) % 24)
                adj_mean += adj_rate / max(adj_base, 1e-4)
        if adj_zones:
            adj_mean /= len(adj_zones)

        combined_mult = rate_mult * 0.70 + adj_mean * 0.30
        surge_boost = {
            SurgeLevel.NONE:     0.0,
            SurgeLevel.ELEVATED: 0.5,
            SurgeLevel.HIGH:     1.5,
            SurgeLevel.CRITICAL: 3.0,
        }[state.surge_level]

        score = (combined_mult - 1.0) + surge_boost
        score = max(0.0, score)
        norm_score = 1.0 - math.exp(-score / 3.0)
        state.mci_risk_score = round(norm_score, 4)

        if combined_mult >= MCI_CRITICAL_RATE_MULT or norm_score > 0.80:
            state.mci_risk_level = MCIRiskLevel.CRITICAL
        elif combined_mult >= MCI_RISK_RATE_MULT or norm_score > 0.55:
            state.mci_risk_level = MCIRiskLevel.HIGH
        elif combined_mult >= MCI_RISK_RATE_MULT * 0.6 or norm_score > 0.30:
            state.mci_risk_level = MCIRiskLevel.MODERATE
        else:
            state.mci_risk_level = MCIRiskLevel.LOW

    def _scan_mci_clusters(self) -> None:
        self._mci_clusters.clear()
        visited: Set[str] = set()

        high_risk_zones = [
            zid for zid, state in self.zone_states.items()
            if state.mci_risk_level in (MCIRiskLevel.HIGH, MCIRiskLevel.CRITICAL)
        ]

        for seed_zone in high_risk_zones:
            if seed_zone in visited:
                continue

            cluster_zones = self._bfs_cluster(seed_zone, MCI_CLUSTER_RADIUS_HOPS)
            visited.update(cluster_zones)

            aggregate_rate = sum(
                float(np.mean(list(self.zone_states[z].recent_rates)))
                if self.zone_states[z].recent_rates else 0.0
                for z in cluster_zones
                if z in self.zone_states
            )

            seed_state = self.zone_states.get(seed_zone)
            cluster_score = seed_state.mci_risk_score if seed_state else 0.0
            if cluster_score < 0.30:
                continue

            risk_level: MCIRiskLevel
            if cluster_score >= 0.80:
                risk_level = MCIRiskLevel.CRITICAL
            elif cluster_score >= 0.55:
                risk_level = MCIRiskLevel.HIGH
            elif cluster_score >= 0.30:
                risk_level = MCIRiskLevel.MODERATE
            else:
                risk_level = MCIRiskLevel.LOW

            estimated_victims = max(1, int(aggregate_rate * STEPS_PER_HOUR * 0.3))
            mutual_aid_needed = (risk_level == MCIRiskLevel.CRITICAL or
                                 len(cluster_zones) >= 3)

            self._cluster_seq += 1
            cluster = MCICluster(
                cluster_id       = f"MCI-CLST-{self._cluster_seq:04d}",
                epicentre_zone   = seed_zone,
                member_zones     = cluster_zones,
                risk_level       = risk_level,
                aggregate_rate   = round(aggregate_rate, 4),
                risk_score       = round(cluster_score, 4),
                detected_at_step = self.step_count,
                detected_at_min  = self.sim_time_min,
                recommended_mutual_aid = mutual_aid_needed,
                estimated_victims= estimated_victims,
            )
            self._mci_clusters.append(cluster)

    def _bfs_cluster(self, seed: str, max_hops: int) -> List[str]:
        visited: Set[str] = {seed}
        queue:   Deque[Tuple[str, int]] = deque([(seed, 0)])
        result: List[str] = [seed]

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for neighbour in self._adjacency.get(current, []):
                if neighbour in visited:
                    continue
                nb_state = self.zone_states.get(neighbour)
                if nb_state and nb_state.mci_risk_level != MCIRiskLevel.LOW:
                    visited.add(neighbour)
                    queue.append((neighbour, depth + 1))
                    result.append(neighbour)

        return result

    def _propagate_spillover(self) -> None:
        for state in self.zone_states.values():
            state.spillover_demand = 0.0

        for zid, state in self.zone_states.items():
            if not state.obs_counts_by_step:
                continue
            recent_obs = list(state.obs_counts_by_step)[-3:]
            mean_obs   = float(np.mean(recent_obs)) if recent_obs else 0.0
            if mean_obs < 0.01:
                continue

            for adj in self._adjacency.get(zid, []):
                adj_state = self.zone_states.get(adj)
                if adj_state:
                    adj_state.spillover_demand += mean_obs * SPILLOVER_ATTENUATION
                for adj2 in self._adjacency.get(adj, []):
                    if adj2 == zid:
                        continue
                    adj2_state = self.zone_states.get(adj2)
                    if adj2_state:
                        adj2_state.spillover_demand += (mean_obs *
                                                        SPILLOVER_ATTENUATION *
                                                        SPILLOVER_HOP2_MULT)

    def _update_heatmap(self) -> None:
        zone_ids:       List[str]   = []
        demand_scores:  List[float] = []
        raw_rates:      List[float] = []
        surge_flags:    List[bool]  = []
        mci_risk_flags: List[bool]  = []

        for zid in self.active_zone_ids:
            state = self.zone_states[zid]
            fc    = self._forecasts.get(zid)

            hour = self._hour_of_day()
            hour_idx = int(hour) % 24
            raw_rate = (state.posterior_rate_hour[hour_idx] /
                        STEP_DURATION_MINUTES * 60.0)  
            raw_rates.append(round(raw_rate, 4))

            norm_score = fc.risk_score if fc else 0.0
            demand_scores.append(round(float(norm_score), 4))
            zone_ids.append(zid)
            surge_flags.append(state.surge_level != SurgeLevel.NONE)
            mci_risk_flags.append(state.mci_risk_level in
                                  (MCIRiskLevel.HIGH, MCIRiskLevel.CRITICAL))

        self._last_heatmap = DemandHeatmap(
            step           = self.step_count,
            sim_time_min   = self.sim_time_min,
            zone_ids       = zone_ids,
            demand_scores  = demand_scores,
            raw_rates      = raw_rates,
            surge_flags    = surge_flags,
            mci_risk_flags = mci_risk_flags,
            weather        = self._weather,
            peak_active    = self._is_peak_hour(),
        )

    def _load_historical_baseline(self) -> None:
        path = DATA_DIR / "demand_history.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                self._parse_demand_history(raw)
                self._history_loaded = True
                return
            except Exception:
                pass

        self._build_synthetic_baseline()
        self._history_loaded = True

    def _parse_demand_history(self, raw: Dict[str, Any]) -> None:
        zones_raw = raw.get("zones", raw)
        for zid, zdata in zones_raw.items():
            hourly: Dict[str, float] = {}
            if isinstance(zdata, dict):
                if "hour_rates" in zdata:
                    rates_list = zdata["hour_rates"]
                    for h, r in enumerate(rates_list[:24]):
                        hourly[str(h)] = float(r)
                else:
                    for h in range(24):
                        key = str(h).zfill(2)
                        if key in zdata:
                            hourly[key] = float(zdata[key])
                        elif str(h) in zdata:
                            hourly[str(h)] = float(zdata[str(h)])
            if hourly:
                self._historical_baseline[zid] = hourly

    def _build_synthetic_baseline(self) -> None:
        for zid, z in self._zone_meta.items():
            risk = z.get("risk_profile", {})
            zone_type = z.get("zone_type", "urban")
            pop_density = float(z.get("population_density_per_sqkm", 200))

            card_rate  = float(risk.get("cardiac_arrest_rate_per_100k", 12)) / 100_000
            trauma_rate = float(risk.get("trauma_rate_per_100k", 25)) / 100_000
            call_vol   = float(risk.get("call_volume_per_day", 20))

            daily_base = (
                card_rate * pop_density * RISK_KEY_MAP["cardiac_arrest_rate_per_100k"]
                + trauma_rate * pop_density * RISK_KEY_MAP["trauma_rate_per_100k"]
                + call_vol / 24.0 * RISK_KEY_MAP["call_volume_per_day"]
            )
            daily_base = max(0.5, daily_base * ZONE_TYPE_DEMAND_MULT.get(zone_type, 1.0))

            hourly: Dict[str, float] = {}
            for h in range(24):
                tod = self._tod_multiplier(float(h))
                hourly[str(h)] = round(daily_base / 24.0 * tod, 6)
            self._historical_baseline[zid] = hourly

    @staticmethod
    def _synthetic_zone_meta(zone_id: str) -> Dict[str, Any]:
        idx = int("".join(c for c in zone_id if c.isdigit()) or "5")
        return {
            "zone_id":   zone_id,
            "name":      f"Zone {zone_id}",
            "lat":       18.52 + (idx % 5) * 0.05,
            "lon":       73.85 + (idx % 4) * 0.06,
            "zone_type": "urban" if idx <= 6 else "suburban",
            "area_sq_km":  35.0,
            "population":  600_000 - idx * 20_000,
            "population_density_per_sqkm": max(200, 8000 - idx * 500),
            "adjacent_zones": [],
            "road_network": {
                "avg_speed_kmh": 38.0,
                "peak_congestion_multiplier": 1.35,
                "road_quality_score": 0.60,
                "ghat_sections": False,
            },
            "geography": {
                "coastal": False, "hilly": False,
                "flood_prone": False, "terrain_difficulty": 0.2,
                "forest_terrain": False,
            },
            "demand_heatmap_weights": {
                "00-06": 0.10, "06-12": 0.28, "12-18": 0.34, "18-24": 0.28,
            },
            "risk_profile": {
                "cardiac_arrest_rate_per_100k": max(5, 14 - idx),
                "trauma_rate_per_100k":         max(8, 28 - idx * 2),
                "call_volume_per_day":          max(5, 25 - idx),
                "industrial_hazard_risk":       "low",
            },
        }

    def _load_zone_data(self) -> None:
        if self._zone_meta:
            return
        path = DATA_DIR / "city_zones.json"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for z in raw.get("zones", []):
            self._zone_meta[z["zone_id"]] = z
        adj_raw = raw.get("zone_adjacency_matrix", {})
        for zid, neighbours in adj_raw.items():
            self._adjacency[zid] = [n for n in neighbours
                                    if n in self._zone_meta]

    def _get_historical_rate(self, zone_id: str, hour_idx: int) -> float:
        z_hist = self._historical_baseline.get(zone_id, {})
        for key in (str(hour_idx), str(hour_idx).zfill(2)):
            if key in z_hist:
                return z_hist[key]
        return BASELINE_RATE_PER_HOUR

    def _get_historical_rate_per_step(self, zone_id: str, hour_idx: int) -> float:
        per_hour = self._get_historical_rate(zone_id, hour_idx)
        return per_hour / 60.0 * STEP_DURATION_MINUTES

    def _tod_multiplier(self, hour: float) -> float:
        for start, end, mult in PEAK_HOUR_PROFILES:
            if start <= hour < end:
                return mult
        return 1.0

    def _hour_of_day(self) -> float:
        return (self.sim_time_min / 60.0) % 24.0

    def _is_peak_hour(self) -> bool:
        h = self._hour_of_day()
        return (8.0 <= h < 10.0) or (17.0 <= h < 19.0)

    @staticmethod
    def _hour_to_band(hour: int) -> str:
        if hour < 6:   return "00-06"
        if hour < 12:  return "06-12"
        if hour < 18:  return "12-18"
        return "18-24"

    def _get_coverage_zones(self, anchor_zone: str, max_hops: int) -> List[str]:
        visited: Set[str] = {anchor_zone}
        queue: Deque[Tuple[str, int]] = deque([(anchor_zone, 0)])
        result = [anchor_zone]

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for nb in self._adjacency.get(current, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
                    result.append(nb)
        return result

    def _recommend_unit_type(self, zone_id: str, horizon_steps: int) -> str:
        state = self.zone_states.get(zone_id)
        if state is None:
            return "ALS"

        zone_obs = [o for o in self._obs_log if o.zone_id == zone_id]
        if not zone_obs:
            zone_type = state.zone_type
            return "MICU" if zone_type in ("urban", "industrial") else "ALS"

        p1_count = sum(1 for o in zone_obs if o.priority == "P1")
        p2_count = sum(1 for o in zone_obs if o.priority == "P2")
        total    = max(len(zone_obs), 1)
        p1_frac  = p1_count / total

        if p1_frac >= 0.40:
            return "MICU"
        if p1_frac >= 0.20 or p2_count / total >= 0.50:
            return "ALS"
        return "BLS"

    def _get_active_surge_obs(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._surge_alerts.values() if a.active]

    def print_zone_table(self, max_zones: int = 12) -> None:
        header = (
            f"{'ZONE':>6} {'TYPE':>10} {'RATE/H':>7} {'HOLT_L':>7} "
            f"{'SURGE':>10} {'MCI_RISK':>10} {'OBS':>5} {'SPILL':>6}"
        )
        print(header)
        print("-" * len(header))
        for zid in list(self.active_zone_ids)[:max_zones]:
            state = self.zone_states[zid]
            hour  = self._hour_of_day()
            rate  = state.current_rate(hour)
            rate_h = rate / STEP_DURATION_MINUTES * 60.0
            print(
                f"{zid:>6} {state.zone_type:>10} "
                f"{rate_h:>7.3f} "
                f"{state.holt.level:>7.4f} "
                f"{state.surge_level.value:>10} "
                f"{state.mci_risk_level.value:>10} "
                f"{state.total_observed:>5} "
                f"{state.spillover_demand:>6.3f}"
            )

    def print_forecast_table(self, max_zones: int = 8) -> None:
        header = (
            f"{'ZONE':>6} "
            + "".join(f"{'H'+str(h+1):>7}" for h in range(min(6, int(LOOK_AHEAD_HOURS))))
            + f" {'TOTAL':>7} {'PEAK_H':>7} {'RISK':>6}"
        )
        print(header)
        print("-" * len(header))
        for zid in list(self.active_zone_ids)[:max_zones]:
            fc = self._forecasts.get(zid)
            if fc is None:
                continue
            hourly_str = "".join(
                f"{v:>7.3f}" for v in fc.hourly_forecast[:min(6, int(LOOK_AHEAD_HOURS))]
            )
            print(
                f"{zid:>6} {hourly_str} "
                f"{fc.total_expected:>7.3f} "
                f"{fc.peak_hour:>7.2f} "
                f"{fc.risk_score:>6.4f}"
            )

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 72)
    print("EMERGI-ENV  ·  DemandForecaster smoke-test")
    print("=" * 72)

    forecaster = DemandForecaster(seed=42)
    active_zones = [
        "Z01", "Z02", "Z03", "Z04", "Z05", "Z06",
        "Z07", "Z08", "Z09", "Z13", "Z14", "Z27",
    ]

    obs = forecaster.reset(
        active_zone_ids  = active_zones,
        task_id          = 1,
        sim_time_minutes = 510.0,    
        weather          = "clear",
    )

    print(f"\n✓  Reset OK — {obs['active_zones']} zones | "
          f"hour={obs['hour_of_day']:.2f} | peak={obs['peak_active']}")

    fc_z05 = forecaster.get_zone_forecast("Z05")
    if fc_z05:
        print(f"\n✓  Z05 Forecast (12h):")
        print(f"   Total expected:  {fc_z05.total_expected:.3f} incidents")
        print(f"   Peak at hour:    {fc_z05.peak_hour:.2f}")
        print(f"   Risk score:      {fc_z05.risk_score:.4f}")
        print(f"   Hourly(h1-h6):   "
              f"{[round(float(v), 2) for v in fc_z05.hourly_forecast[:6]]}")
        print(f"   CI lo/hi h1:     [{fc_z05.hourly_ci_lo[0]:.3f}, "
              f"{fc_z05.hourly_ci_hi[0]:.3f}]")

    hotspots = forecaster.get_hotspot_zones(k=5)
    print(f"\n✓  Top-5 hotspot zones:")
    for hs in hotspots:
        print(f"score={hs['demand_score']:.3f} | "
              f"surge={hs['surge_level']} | "
              f"mci_risk={hs['mci_risk']}")

    print(f"\n  Injecting synthetic incidents ...")
    incidents = [
        {"zone_id": "Z05", "incident_category": "cardiac", "reported_priority": "P1"},
        {"zone_id": "Z05", "incident_category": "trauma",  "reported_priority": "P1"},
        {"zone_id": "Z06", "incident_category": "stroke",  "reported_priority": "P2"},
        {"zone_id": "Z02", "incident_category": "routine", "reported_priority": "P3"},
        {"zone_id": "Z05", "incident_category": "cardiac", "reported_priority": "P1"},
        {"zone_id": "Z05", "incident_category": "burns",   "reported_priority": "P1",
         "is_mci": True, "victim_count": 15},
    ]

    registered = forecaster.observe_incidents_bulk(incidents)
    print(f"  Registered {registered} observations")

    print(f"\n  Stepping 40 steps (Task 1, weather: clear→rain):")
    total_rwd = 0.0
    for s in range(40):
        weather = "rain" if s >= 20 else "clear"
        sim_t   = 510.0 + (s + 1) * STEP_DURATION_MINUTES
        if s % 3 == 0:
            zone_choice = active_zones[s % len(active_zones)]
            forecaster.observe_incident(zone_choice, "cardiac", "P1")
        if s == 15:
            for _ in range(6):
                forecaster.observe_incident("Z05", "trauma", "P1", is_mci=False)

        obs2, rwd = forecaster.step(sim_t, weather=weather)
        total_rwd += rwd

        if s % 8 == 0:
            surges = obs2["active_surge_alerts"]
            mci_cl = obs2["mci_clusters_at_risk"]
            print(f"  Step {s+1:02d}: "
                  f"hour={obs2['hour_of_day']:.1f} | "
                  f"surge_alerts={len(surges)} | "
                  f"mci_clusters={len(mci_cl)} | "
                  f"rwd={rwd:+.4f} | "
                  f"weather={obs2['weather']}")

    print(f"\n  Total step reward (40 steps): {total_rwd:+.4f}")

    print(f"\n  Zone demand state table:")
    forecaster.print_zone_table(max_zones=8)

    print(f"\n  12-hour demand forecast table:")
    forecaster.print_forecast_table(max_zones=6)

    recs = forecaster.get_preposition_recommendations(n_recommendations=4)
    print(f"\n✓  Pre-positioning recommendations:")
    for rec in recs:
        print(f"   score={rec.demand_score:.4f} | "
              f"unit={rec.recommended_unit_type} x{rec.recommended_count} | "
              f"covers={rec.coverage_zones[:3]}")

    all_surges = forecaster.get_surge_alerts(active_only=False)
    print(f"\n✓  Surge alerts generated: {len(all_surges)}")
    for sa in all_surges[:4]:
        print(f"   {sa['alert_id']}: zone={sa['zone_id']} "
              f"level={sa['level']} z={sa['z_score']:.2f} "
              f"active={sa['active']}")

    clusters = forecaster.get_mci_clusters()
    print(f"\n✓  MCI clusters detected: {len(clusters)}")
    for cl in clusters[:3]:
        print(f"   {cl['cluster_id']}: epicentre={cl['epicentre_zone']} "
              f"risk={cl['risk_level']} members={cl['member_zones'][:3]} "
              f"estimated_victims={cl['estimated_victims']}")

    matrix = forecaster.get_zone_demand_matrix()
    print(f"\n✓  Demand matrix shape: {matrix.shape} (zones × forecast_steps)")
    if matrix.size > 0:
        print(f"   Max value: {float(matrix.max()):.4f} | "
              f"Mean: {float(matrix.mean()):.4f}")
    else:
        print("   (empty matrix — no active zones)")

    hourly_mat = forecaster.get_hourly_zone_matrix()
    print(f"✓  Hourly matrix shape: {hourly_mat.shape} (zones × 12 hours)")

    hm = forecaster.get_demand_heatmap()
    print(f"\n✓  Heatmap: {len(hm.zone_ids)} zones | "
          f"max_score={max(hm.demand_scores):.4f} | "
          f"surge_zones={sum(hm.surge_flags)} | "
          f"mci_risk_zones={sum(hm.mci_risk_flags)}")

    acc = forecaster.get_forecast_accuracy_estimate()
    print(f"\n✓  Forecast accuracy estimate: MAPE={acc['mape']:.4f} "
          f"RMSE={acc['rmse']:.4f} n={acc['n_samples']}")

    print(f"\n── Task 6 — Pre-positioning scenario ──")
    forecaster.reset(active_zone_ids=active_zones, task_id=6,
                     sim_time_minutes=600.0, weather="clear")
    for s in range(10):
        forecaster.step(600.0 + s * STEP_DURATION_MINUTES)
    recs6 = forecaster.get_preposition_recommendations(n_recommendations=4,
                                                       horizon_minutes=60.0)
    print(f"✓  Task 6 recommendations ({len(recs6)} zones):")
    for r in recs6:
        print(f"   Zone={r.zone_id} | {r.justification[:70]}")

    print(f"\n── Task 9 — City-wide surge scenario ──")
    forecaster.reset(active_zone_ids=active_zones, task_id=9,
                     sim_time_minutes=720.0, weather="storm")
    for _ in range(25):
        forecaster.observe_incident("Z05", "trauma",  "P1", is_mci=True, victim_count=30)
        forecaster.observe_incident("Z02", "cardiac", "P1", is_mci=True, victim_count=20)
        forecaster.observe_incident("Z08", "burns",   "P1", is_mci=True, victim_count=15)
    for s in range(15):
        obs9, rwd9 = forecaster.step(720.0 + s * STEP_DURATION_MINUTES, weather="storm")

    print(f"✓  Task 9 final state:")
    print(f"   Surges active:   {len(obs9['active_surge_alerts'])}")
    print(f"   MCI clusters:    {len(obs9['mci_clusters_at_risk'])}")
    print(f"   Top demand zone: {obs9['top_demand_zones'][0] if obs9['top_demand_zones'] else 'N/A'}")
    forecaster.print_zone_table(max_zones=6)

    analytics = forecaster.get_episode_analytics()
    print(f"\n  Episode analytics (Task 9):")
    for k, v in analytics.items():
        if not isinstance(v, dict):
            print(f"   {k}: {v}")

    print(f"\n  Describe: {forecaster.describe()}")
    print("\n✅  DemandForecaster smoke-test PASSED")