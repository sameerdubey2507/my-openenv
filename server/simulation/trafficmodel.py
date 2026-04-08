from __future__ import annotations
import heapq
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import numpy as np
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent.parent / "data"
STEP_DURATION_MINUTES: float = 3.0        
PEAK_AM = (8.0, 10.0)
PEAK_PM = (17.0, 19.0)
SHOULDER_WINDOW = 0.75                    
OFF_PEAK_MULT = 1.00
SHOULDER_MULT = 1.12
NIGHT_MULT    = 0.85                      
SEC_CONGESTION_MULT    = 1.25            
SEC_CONGESTION_DECAY   = 5              
SEC_CONGESTION_RADIUS  = 1              
BPR_ALPHA = 0.15                        
BPR_BETA  = 4.0                         
ETA_SIGMA_FRACTION = 0.15               
TERRAIN_SPEED_PENALTY = {               
    (0.00, 0.20): 1.00,
    (0.20, 0.40): 1.08,
    (0.40, 0.60): 1.18,
    (0.60, 0.80): 1.30,
    (0.80, 1.00): 1.50,
}
GHAT_EXTRA_MULT       = 1.20
FOREST_EXTRA_MULT     = 1.15
SEASONAL_CLOSURE_MULT = 1.40
REMOTE_ACCESS_MULT    = 1.25
WEATHER_BASE_MULT = {
    "clear":  1.00,
    "haze":   1.05,
    "rain":   1.25,
    "fog":    1.35,
    "storm":  1.55,
}
WEATHER_COASTAL_EXTRA  = {"rain": 0.15, "storm": 0.30, "fog": 0.10}
WEATHER_HILLY_EXTRA    = {"rain": 0.10, "storm": 0.20, "fog": 0.15}
WEATHER_FLOOD_EXTRA    = {"rain": 0.20, "storm": 0.40}
class WeatherCondition(str, Enum):
    CLEAR = "clear"
    HAZE  = "haze"
    RAIN  = "rain"
    FOG   = "fog"
    STORM = "storm"
class CongestionCause(str, Enum):
    EMS_INCIDENT  = "ems_incident"
    ACCIDENT      = "accident"
    VIP_CONVOY    = "vip_convoy"
    ROAD_WORK     = "road_work"
    FLOOD_CLOSURE = "flood_closure"
@dataclass
class ZoneNode:
    zone_id:                    str
    name:                       str
    lat:                        float
    lon:                        float
    zone_type:                  str             
    area_sq_km:                 float
    population:                 int
    population_density:         float           
    adjacent_zone_ids:          List[str]
    avg_speed_kmh:              float           
    peak_congestion_multiplier: float           
    road_quality_score:         float           
    coastal:                    bool
    hilly:                      bool
    flood_prone:                bool
    terrain_difficulty:         float           
    ghat_roads:                 bool
    forest_terrain:             bool
    seasonal_closure:           bool
    remote_access:              bool
    demand_heatmap_weights:     Dict[str, float]
    terrain_multiplier:         float = field(init=False)
    def __post_init__(self) -> None:
        self.terrain_multiplier = self._compute_terrain_mult()
    def _compute_terrain_mult(self) -> float:
        mult = 1.0
        for (lo, hi), m in TERRAIN_SPEED_PENALTY.items():
            if lo <= self.terrain_difficulty < hi:
                mult = m
                break
        if self.ghat_roads:
            mult *= GHAT_EXTRA_MULT
        if self.forest_terrain:
            mult *= FOREST_EXTRA_MULT
        if self.seasonal_closure:
            mult *= SEASONAL_CLOSURE_MULT
        if self.remote_access:
            mult *= REMOTE_ACCESS_MULT
        return mult
@dataclass
class RoadLink:
    from_id:          str
    to_id:            str
    distance_km:      float
    base_speed_kmh:   float          
    terrain_mult:     float          
    capacity_vph:     float          
    current_flow_vph: float = 0.0
    congestion_factor: float = 0.0  
    @property
    def base_travel_minutes(self) -> float:
        return (self.distance_km / self.base_speed_kmh) * 60.0 * self.terrain_mult
    @property
    def current_travel_minutes(self) -> float:
        v_c = min(self.current_flow_vph / max(self.capacity_vph, 1.0), 2.0)
        bpr_factor = 1.0 + BPR_ALPHA * (v_c ** BPR_BETA)
        return self.base_travel_minutes * bpr_factor * (1.0 + self.congestion_factor)
@dataclass
class CongestionEvent:
    event_id:        str
    origin_zone_id:  str
    affected_ids:    List[str]       
    multiplier:      float           
    remaining_steps: int
    cause:           CongestionCause
    severity:        float           
@dataclass
class RouteResult:
    from_id:          str
    to_id:            str
    path:             List[str]      
    total_distance_km: float
    base_eta_minutes:  float         
    current_eta_minutes: float       
    eta_lower_bound:  float          
    eta_upper_bound:  float          
    via_zones:        List[str]      
    hotspot_zones:    List[str]      
@dataclass
class TrafficSnapshot:
    sim_time_minutes:      float
    hour_of_day:           float
    weather:               str
    weather_multiplier:    float
    peak_active:           bool
    active_zone_ids:       List[str]
    od_matrix:             Dict[str, Dict[str, float]]  
    zone_congestion_levels: Dict[str, float]             
    congestion_events:     List[Dict]
    step_count:            int
    matrix_last_updated:   int       
class TrafficModel:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng   = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.all_zones:    Dict[str, ZoneNode] = {}
        self.all_links:    Dict[Tuple[str, str], RoadLink] = {}
        self.adjacency:    Dict[str, List[str]] = {}
        self._anchor_times: Dict[Tuple[str, str], float] = {}
        self.active_zone_ids: List[str] = []
        self.sim_time_minutes: float = 480.0   
        self.step_count: int = 0
        self.weather: WeatherCondition = WeatherCondition.CLEAR
        self._weather_multiplier: float = 1.0
        self._zone_weather_mults: Dict[str, float] = {}
        self.congestion_events: List[CongestionEvent] = []
        self._event_seq: int = 0
        self._od_matrix: Dict[str, Dict[str, float]] = {}
        self._base_od:   Dict[str, Dict[str, float]] = {}
        self._zone_congestion: Dict[str, float] = {}
        self._full_base_shortest: Dict[str, Dict[str, Tuple[float, List[str]]]] = {}
        self._next_weather_change: int = self.rng.randint(15, 50)
        self._load_zones()
        self._parse_anchor_times()
        self._build_links()
        self._precompute_full_base_shortest_paths()
    def _load_zones(self) -> None:
        path = DATA_DIR / "city_zones.json"
        if not path.exists():
            raise FileNotFoundError(
                f"city_zones.json not found at {path}. "
                "Ensure the data/ directory is populated before starting the server."
            )
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for z in raw["zones"]:
            rn  = z.get("road_network", {})
            geo = z.get("geography", {})
            dhw = z.get("demand_heatmap_weights", {
                "00-06": 0.12, "06-12": 0.28, "12-18": 0.32, "18-24": 0.28
            })
            adj = z.get("adjacent_zones", [])
            adj = [a for a in adj if a.startswith("Z") and a[1:].isdigit()]
            node = ZoneNode(
                zone_id                    = z["zone_id"],
                name                       = z["name"],
                lat                        = z["lat"],
                lon                        = z["lon"],
                zone_type                  = z.get("zone_type", "rural"),
                area_sq_km                 = float(z.get("area_sq_km", 1000)),
                population                 = int(z.get("population", 500000)),
                population_density         = float(z.get("population_density_per_sqkm", 200)),
                adjacent_zone_ids          = adj,
                avg_speed_kmh              = float(rn.get("avg_speed_kmh", 40)),
                peak_congestion_multiplier = float(rn.get("peak_congestion_multiplier", 1.3)),
                road_quality_score         = float(rn.get("road_quality_score", 0.55)),
                coastal                    = bool(geo.get("coastal", False)),
                hilly                      = bool(geo.get("hilly", False)),
                flood_prone                = bool(geo.get("flood_prone", False)),
                terrain_difficulty         = float(geo.get("terrain_difficulty", 0.2)),
                ghat_roads                 = bool(rn.get("ghat_sections", False) or
                                                  geo.get("ghat_roads", False)),
                forest_terrain             = bool(geo.get("forest_terrain", False)),
                seasonal_closure           = bool(rn.get("seasonal_road_closures", False)),
                remote_access              = bool(geo.get("remote_access", False)),
                demand_heatmap_weights     = dhw,
            )
            self.all_zones[node.zone_id] = node
            self.adjacency[node.zone_id] = adj
        raw_adj = raw.get("zone_adjacency_matrix", {})
        for zid, neighbours in raw_adj.items():
            if zid in self.all_zones:
                clean = [n for n in neighbours if n in self.all_zones]
                self.adjacency[zid] = clean
                self.all_zones[zid].adjacent_zone_ids = clean
    def _parse_anchor_times(self) -> None:
        path = DATA_DIR / "city_zones.json"
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        anchor_raw = raw.get("inter_zone_travel_times_minutes", {})
        for key, minutes in anchor_raw.items():
            parts = key.split("_")
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                if a in self.all_zones and b in self.all_zones:
                    self._anchor_times[(a, b)] = float(minutes)
                    self._anchor_times[(b, a)] = float(minutes)  
    def _build_links(self) -> None:
        for from_id, neighbours in self.adjacency.items():
            if from_id not in self.all_zones:
                continue
            fz = self.all_zones[from_id]
            for to_id in neighbours:
                if to_id not in self.all_zones:
                    continue
                tz = self.all_zones[to_id]
                dist_km = self._haversine_km(fz.lat, fz.lon, tz.lat, tz.lon)
                dist_km = max(dist_km, 2.0)  
                harmonic_speed = (2 * fz.avg_speed_kmh * tz.avg_speed_kmh /
                                  (fz.avg_speed_kmh + tz.avg_speed_kmh + 1e-9))
                quality_factor = min(fz.road_quality_score, tz.road_quality_score)
                quality_mult   = 0.6 + 0.4 * quality_factor  
                effective_speed = harmonic_speed * quality_mult
                if (from_id, to_id) in self._anchor_times:
                    anchor_min = self._anchor_times[(from_id, to_id)]
                    calibrated = (dist_km / (anchor_min / 60.0))
                    effective_speed = calibrated  
                terrain_mult = max(fz.terrain_multiplier, tz.terrain_multiplier)
                lanes = 4 if effective_speed > 45 else 2
                capacity = CAPACITY_VPH_PER_LANE * lanes * quality_factor
                link = RoadLink(
                    from_id          = from_id,
                    to_id            = to_id,
                    distance_km      = dist_km,
                    base_speed_kmh   = max(effective_speed, 5.0),
                    terrain_mult     = terrain_mult,
                    capacity_vph     = max(capacity, 200.0),
                )
                self.all_links[(from_id, to_id)] = link
    def _precompute_full_base_shortest_paths(self) -> None:
        for src in self.all_zones:
            self._full_base_shortest[src] = self._dijkstra(
                src,
                set(self.all_zones.keys()),
                use_current=False,
            )
    def _dijkstra(
        self,
        source: str,
        allowed_zones: Set[str],
        use_current: bool = True,
    ) -> Dict[str, Tuple[float, List[str]]]:
        INF = float("inf")
        dist: Dict[str, float] = {z: INF for z in allowed_zones}
        prev: Dict[str, Optional[str]] = {z: None for z in allowed_zones}
        dist[source] = 0.0
        heap: List[Tuple[float, str]] = [(0.0, source)]
        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist[u]:
                continue
            for v in self.adjacency.get(u, []):
                if v not in allowed_zones:
                    continue
                link = self.all_links.get((u, v))
                if link is None:
                    continue
                w = link.current_travel_minutes if use_current else link.base_travel_minutes
                alt = dist[u] + w
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))
        results: Dict[str, Tuple[float, List[str]]] = {}
        for z in allowed_zones:
            path: List[str] = []
            cur: Optional[str] = z
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            results[z] = (dist[z], path if path[0] == source else [])
        return results
    def reset(
        self,
        sim_time_minutes: float = 480.0,
        active_zone_ids: Optional[List[str]] = None,
        weather: Optional[str] = None,
    ) -> None:
        self.sim_time_minutes = sim_time_minutes
        self.step_count       = 0
        self.congestion_events.clear()
        self._event_seq = 0
        if active_zone_ids:
            self.active_zone_ids = [z for z in active_zone_ids if z in self.all_zones]
        else:
            self.active_zone_ids = list(self.all_zones.keys())
        self.weather = WeatherCondition(weather) if weather else WeatherCondition.CLEAR
        self._recompute_zone_weather_multipliers()
        for link in self.all_links.values():
            link.current_flow_vph = 0.0
            link.congestion_factor = 0.0
        self._zone_congestion = {z: 0.0 for z in self.active_zone_ids}
        self._rebuild_od_matrix()
        self._build_base_od()
        self._next_weather_change = self.step_count + self.rng.randint(15, 50)
    def step(
        self,
        active_incident_zones: Optional[List[str]] = None,
    ) -> None:
        self.sim_time_minutes += STEP_DURATION_MINUTES
        self.step_count       += 1
        self._decay_congestion_events()
        if active_incident_zones:
            self._spawn_incident_congestion(active_incident_zones)
        self._maybe_spawn_random_disturbance()
        if self.step_count >= self._next_weather_change:
            self._evolve_weather()
            self._next_weather_change = self.step_count + self.rng.randint(20, 60)
        self._rebuild_od_matrix()
        self._update_zone_congestion()
    def _rebuild_od_matrix(self) -> None:
        hour = self._hour_of_day()
        self._update_link_congestion_factors(hour)
        active_set: Set[str] = set(self.active_zone_ids)
        self._od_matrix = {}
        for src in self.active_zone_ids:
            sp = self._dijkstra(src, active_set, use_current=True)
            row: Dict[str, float] = {}
            for dst in self.active_zone_ids:
                base_min = sp[dst][0] if dst in sp else float("inf")
                dst_zone = self.all_zones.get(dst)
                w_mult = self._zone_weather_mults.get(dst, self._weather_multiplier)
                row[dst] = base_min * w_mult
            self._od_matrix[src] = row
    def _build_base_od(self) -> None:
        active_set: Set[str] = set(self.active_zone_ids)
        self._base_od = {}
        for src in self.active_zone_ids:
            sp = self._dijkstra(src, active_set, use_current=False)
            self._base_od[src] = {
                dst: sp[dst][0] if dst in sp else float("inf")
                for dst in self.active_zone_ids
            }
    def _update_link_congestion_factors(self, hour: float) -> None:
        event_zone_mults: Dict[str, float] = {}
        for ev in self.congestion_events:
            decay = ev.remaining_steps / max(SEC_CONGESTION_DECAY, 1)
            eff   = 1.0 + (ev.multiplier - 1.0) * decay
            for zid in ev.affected_ids:
                event_zone_mults[zid] = max(event_zone_mults.get(zid, 1.0), eff)
        active_set = set(self.active_zone_ids)
        for (from_id, to_id), link in self.all_links.items():
            if from_id not in active_set and to_id not in active_set:
                continue
            fz = self.all_zones.get(from_id)
            tz = self.all_zones.get(to_id)
            if fz is None or tz is None:
                continue
            tod_mult = self._tod_speed_multiplier(fz, hour)
            peak_mult = 1.0
            if self._is_peak_hour(hour):
                peak_mult = (fz.peak_congestion_multiplier * 0.6 +
                             tz.peak_congestion_multiplier * 0.4)
                peak_mult = peak_mult * tod_mult
            else:
                peak_mult = tod_mult
            ev_mult = max(
                event_zone_mults.get(from_id, 1.0),
                event_zone_mults.get(to_id,   1.0),
            )
            combined = peak_mult * ev_mult
            cf = min((combined - 1.0) / 1.8, 1.0)
            link.congestion_factor = max(cf, 0.0)
            link.current_flow_vph = link.capacity_vph * (cf ** 0.5)
    def _tod_speed_multiplier(self, zone: ZoneNode, hour: float) -> float:
        weights = zone.demand_heatmap_weights
        band_map = {
            "00-06": (0,  6),
            "06-12": (6,  12),
            "12-18": (12, 18),
            "18-24": (18, 24),
        }
        current_w = 0.28  
        next_w    = 0.28
        frac      = 0.0
        for band, (lo, hi) in band_map.items():
            if lo <= hour < hi:
                current_w = weights.get(band, 0.25)
                next_band = list(band_map.keys())[(list(band_map.keys()).index(band) + 1) % 4]
                next_w    = weights.get(next_band, 0.25)
                frac      = (hour - lo) / (hi - lo)
                break
        interp_w = current_w * (1 - frac) + next_w * frac
        avg_w = sum(weights.values()) / max(len(weights), 1)
        ratio = interp_w / max(avg_w, 0.01)
        return max(NIGHT_MULT, min(ratio, zone.peak_congestion_multiplier))
    @staticmethod
    def _is_peak_hour(hour: float) -> bool:
        return (PEAK_AM[0] <= hour < PEAK_AM[1]) or (PEAK_PM[0] <= hour < PEAK_PM[1])
    def _update_zone_congestion(self) -> None:
        for zid in self.active_zone_ids:
            max_cf = 0.0
            for neighbour in self.adjacency.get(zid, []):
                link = self.all_links.get((zid, neighbour))
                if link:
                    max_cf = max(max_cf, link.congestion_factor)
            self._zone_congestion[zid] = max_cf
    def _spawn_incident_congestion(self, incident_zones: List[str]) -> None:
        existing_origins = {ev.origin_zone_id for ev in self.congestion_events
                            if ev.cause == CongestionCause.EMS_INCIDENT}
        for zid in incident_zones:
            if zid in existing_origins or zid not in self.all_zones:
                continue
            affected = [zid] + [
                n for n in self.adjacency.get(zid, [])
                if n in set(self.active_zone_ids)
            ]
            zone = self.all_zones[zid]
            severity = min(zone.population_density / 20000.0, 1.0)
            mult = 1.0 + (SEC_CONGESTION_MULT - 1.0) * (0.5 + 0.5 * severity)
            ev = CongestionEvent(
                event_id        = f"ev_{self._event_seq:04d}",
                origin_zone_id  = zid,
                affected_ids    = affected,
                multiplier      = mult,
                remaining_steps = SEC_CONGESTION_DECAY,
                cause           = CongestionCause.EMS_INCIDENT,
                severity        = severity,
            )
            self.congestion_events.append(ev)
            self._event_seq += 1
    def _maybe_spawn_random_disturbance(self) -> None:
        if self.rng.random() > 0.025:
            return
        cause = self.rng.choice([
            CongestionCause.ACCIDENT,
            CongestionCause.VIP_CONVOY,
            CongestionCause.ROAD_WORK,
        ])
        if not self.active_zone_ids:
            return
        origin = self.rng.choice(self.active_zone_ids)
        affected = [origin] + self.adjacency.get(origin, [])[:2]
        severity = self.rng.uniform(0.2, 0.8)
        mult = {
            CongestionCause.ACCIDENT:   1.0 + 0.4 * severity,
            CongestionCause.VIP_CONVOY: 1.0 + 0.6 * severity,
            CongestionCause.ROAD_WORK:  1.0 + 0.3 * severity,
        }[cause]
        steps = self.rng.randint(3, 8)
        ev = CongestionEvent(
            event_id        = f"ev_{self._event_seq:04d}",
            origin_zone_id  = origin,
            affected_ids    = affected,
            multiplier      = mult,
            remaining_steps = steps,
            cause           = cause,
            severity        = severity,
        )
        self.congestion_events.append(ev)
        self._event_seq += 1
    def _decay_congestion_events(self) -> None:
        alive = []
        for ev in self.congestion_events:
            ev.remaining_steps -= 1
            if ev.remaining_steps > 0:
                alive.append(ev)
        self.congestion_events = alive
    def spawn_flood_closure(self, zone_id: str, duration_steps: int = 20) -> str:
        if zone_id not in self.all_zones:
            raise ValueError(f"Unknown zone: {zone_id}")
        affected = [zone_id] + self.adjacency.get(zone_id, [])
        ev = CongestionEvent(
            event_id        = f"flood_{self._event_seq:04d}",
            origin_zone_id  = zone_id,
            affected_ids    = affected,
            multiplier      = SEASONAL_CLOSURE_MULT,
            remaining_steps = duration_steps,
            cause           = CongestionCause.FLOOD_CLOSURE,
            severity        = 1.0,
        )
        self.congestion_events.append(ev)
        self._event_seq += 1
        return ev.event_id
    def _evolve_weather(self) -> None:
        transitions = {
            WeatherCondition.CLEAR: {
                WeatherCondition.CLEAR: 0.65,
                WeatherCondition.HAZE:  0.18,
                WeatherCondition.RAIN:  0.12,
                WeatherCondition.FOG:   0.04,
                WeatherCondition.STORM: 0.01,
            },
            WeatherCondition.HAZE: {
                WeatherCondition.CLEAR: 0.40,
                WeatherCondition.HAZE:  0.35,
                WeatherCondition.RAIN:  0.15,
                WeatherCondition.FOG:   0.08,
                WeatherCondition.STORM: 0.02,
            },
            WeatherCondition.RAIN: {
                WeatherCondition.CLEAR: 0.25,
                WeatherCondition.HAZE:  0.15,
                WeatherCondition.RAIN:  0.40,
                WeatherCondition.FOG:   0.10,
                WeatherCondition.STORM: 0.10,
            },
            WeatherCondition.FOG: {
                WeatherCondition.CLEAR: 0.30,
                WeatherCondition.HAZE:  0.20,
                WeatherCondition.RAIN:  0.15,
                WeatherCondition.FOG:   0.30,
                WeatherCondition.STORM: 0.05,
            },
            WeatherCondition.STORM: {
                WeatherCondition.CLEAR: 0.10,
                WeatherCondition.HAZE:  0.10,
                WeatherCondition.RAIN:  0.45,
                WeatherCondition.FOG:   0.05,
                WeatherCondition.STORM: 0.30,
            },
        }
        probs = transitions[self.weather]
        conditions = list(probs.keys())
        weights    = [probs[c] for c in conditions]
        self.weather = self.rng.choices(conditions, weights=weights, k=1)[0]
        self._recompute_zone_weather_multipliers()
    def _recompute_zone_weather_multipliers(self) -> None:
        base = WEATHER_BASE_MULT.get(self.weather.value, 1.0)
        self._weather_multiplier = base
        for zid, zone in self.all_zones.items():
            z_mult = base
            cond = self.weather.value
            if zone.coastal:
                z_mult += WEATHER_COASTAL_EXTRA.get(cond, 0.0)
            if zone.hilly or zone.ghat_roads:
                z_mult += WEATHER_HILLY_EXTRA.get(cond, 0.0)
            if zone.flood_prone:
                z_mult += WEATHER_FLOOD_EXTRA.get(cond, 0.0)
            self._zone_weather_mults[zid] = z_mult
    def force_weather(self, condition: str) -> None:
        self.weather = WeatherCondition(condition)
        self._recompute_zone_weather_multipliers()
    def get_eta(
        self,
        from_id: str,
        to_id:   str,
        with_uncertainty: bool = True,
    ) -> RouteResult:
        if from_id not in self.all_zones or to_id not in self.all_zones:
            raise ValueError(f"Unknown zone(s): {from_id}, {to_id}")
        active_set = set(self.active_zone_ids)
        if from_id not in active_set:
            active_set.add(from_id)
        if to_id not in active_set:
            active_set.add(to_id)
        sp_current = self._dijkstra(from_id, active_set, use_current=True)
        current_min, path = sp_current.get(to_id, (float("inf"), []))
        dst_weather = self._zone_weather_mults.get(to_id, self._weather_multiplier)
        current_eta = current_min * dst_weather
        base_sp = self._full_base_shortest.get(from_id, {})
        base_eta, base_path = base_sp.get(to_id, (float("inf"), []))
        total_dist = sum(
            self.all_links[(path[i], path[i+1])].distance_km
            for i in range(len(path) - 1)
            if (path[i], path[i+1]) in self.all_links
        ) if len(path) > 1 else 0.0
        hotspots = [
            zid for zid in path
            if self._zone_congestion.get(zid, 0.0) > 0.5
        ]
        sigma = current_eta * ETA_SIGMA_FRACTION if with_uncertainty else 0.0
        lo = max(current_eta - sigma, 0.0)
        hi = current_eta + sigma
        via = [self.all_zones[z].name for z in path[1:-1] if z in self.all_zones]
        return RouteResult(
            from_id            = from_id,
            to_id              = to_id,
            path               = path,
            total_distance_km  = total_dist,
            base_eta_minutes   = base_eta,
            current_eta_minutes= current_eta,
            eta_lower_bound    = lo,
            eta_upper_bound    = hi,
            via_zones          = via,
            hotspot_zones      = hotspots,
        )
    def get_travel_time(self, from_id: str, to_id: str) -> float:
        if from_id == to_id:
            return 0.0
        row = self._od_matrix.get(from_id, {})
        return row.get(to_id, float("inf"))
    def get_best_hospital_by_eta(
        self,
        from_id:      str,
        hospital_ids: List[str],
        exclude_ids:  Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        excluded = set(exclude_ids or [])
        best_id  = None
        best_eta = float("inf")
        for hid in hospital_ids:
            if hid in excluded:
                continue
            eta = self.get_travel_time(from_id, hid)
            if eta < best_eta:
                best_eta = eta
                best_id  = hid
        if best_id is None:
            raise RuntimeError(
                f"No reachable hospital from {from_id} among {hospital_ids}"
            )
        return best_id, best_eta
    def get_zone_congestion(self, zone_id: str) -> float:
        return self._zone_congestion.get(zone_id, 0.0)
    def get_active_congestion_events(self) -> List[Dict]:
        return [
            {
                "event_id":       ev.event_id,
                "origin_zone":    ev.origin_zone_id,
                "affected_zones": ev.affected_ids,
                "cause":          ev.cause.value,
                "severity":       ev.severity,
                "remaining_steps":ev.remaining_steps,
                "multiplier":     round(ev.multiplier, 3),
            }
            for ev in self.congestion_events
        ]
    def get_snapshot(self) -> TrafficSnapshot:
        od_rounded: Dict[str, Dict[str, float]] = {}
        for src, row in self._od_matrix.items():
            od_rounded[src] = {
                dst: round(min(v, 9999.0), 2)
                for dst, v in row.items()
            }
        return TrafficSnapshot(
            sim_time_minutes       = self.sim_time_minutes,
            hour_of_day            = self._hour_of_day(),
            weather                = self.weather.value,
            weather_multiplier     = round(self._weather_multiplier, 3),
            peak_active            = self._is_peak_hour(self._hour_of_day()),
            active_zone_ids        = list(self.active_zone_ids),
            od_matrix              = od_rounded,
            zone_congestion_levels = {
                zid: round(v, 3)
                for zid, v in self._zone_congestion.items()
            },
            congestion_events      = self.get_active_congestion_events(),
            step_count             = self.step_count,
            matrix_last_updated    = self.step_count,
        )
    def get_od_matrix_array(self) -> np.ndarray:
        n = len(self.active_zone_ids)
        m = np.full((n, n), 9999.0, dtype=np.float32)
        for i, src in enumerate(self.active_zone_ids):
            for j, dst in enumerate(self.active_zone_ids):
                v = self._od_matrix.get(src, {}).get(dst, 9999.0)
                m[i, j] = min(v, 9999.0)
        return m
    def get_demand_forecast(self, look_ahead_hours: float = 12.0) -> Dict[str, List[float]]:
        forecast: Dict[str, List[float]] = {}
        band_map = {"00-06": (0, 6), "06-12": (6, 12), "12-18": (12, 18), "18-24": (18, 24)}
        for zid in self.active_zone_ids:
            zone = self.all_zones[zid]
            weights = zone.demand_heatmap_weights
            hour_now = self._hour_of_day()
            hourly_vals: List[float] = []
            for h_offset in range(int(look_ahead_hours)):
                target_hour = (hour_now + h_offset) % 24
                band_w = 0.25  
                for band, (lo, hi) in band_map.items():
                    if lo <= target_hour < hi:
                        band_w = weights.get(band, 0.25)
                        break
                noise = self.np_rng.uniform(-0.20, 0.20) * band_w
                hourly_vals.append(max(0.0, round(band_w + noise, 4)))
            forecast[zid] = hourly_vals
        return forecast
    def _hour_of_day(self) -> float:
        return (self.sim_time_minutes / 60.0) % 24.0
    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    def describe(self) -> Dict:
        hour = self._hour_of_day()
        return {
            "sim_time_minutes":    self.sim_time_minutes,
            "hour_of_day":         f"{int(hour):02d}:{int((hour%1)*60):02d}",
            "step_count":          self.step_count,
            "weather":             self.weather.value,
            "peak_active":         self._is_peak_hour(hour),
            "active_zones":        len(self.active_zone_ids),
            "total_zones":         len(self.all_zones),
            "total_links":         len(self.all_links),
            "congestion_events":   len(self.congestion_events),
            "mean_od_minutes":     self._mean_od(),
            "max_zone_congestion": max(self._zone_congestion.values(), default=0.0),
        }
    def _mean_od(self) -> float:
        vals = [
            v for row in self._od_matrix.values()
            for v in row.values()
            if v < 9000
        ]
        return round(sum(vals) / max(len(vals), 1), 2)
    def print_od_table(self, max_zones: int = 8) -> None:
        zones = self.active_zone_ids[:max_zones]
        header = "FROM/TO".ljust(12) + "".join(f"{z:<10}" for z in zones)
        print(header)
        print("-" * len(header))
        for src in zones:
            row_str = f"{self.all_zones[src].name[:11]:<12}"
            for dst in zones:
                v = self._od_matrix.get(src, {}).get(dst, float("inf"))
                row_str += f"{v:>8.1f}  "
            print(row_str)
CAPACITY_VPH_PER_LANE: float = 1800.0  
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 60)
    print("EMERGI-ENV  ·  TrafficModel smoke-test")
    print("=" * 60)
    model = TrafficModel(seed=42)
    print(f"✓  Loaded {len(model.all_zones)} zones, {len(model.all_links)} directed links")
    print(f"✓  Anchor travel times seeded: {len(model._anchor_times)//2} pairs")
    active = ["Z01","Z02","Z03","Z04","Z05","Z06","Z07","Z08",
              "Z09","Z13","Z14","Z27"]
    model.reset(sim_time_minutes=510, active_zone_ids=active, weather="clear")
    print(f"✓  Reset OK — {len(model.active_zone_ids)} active zones, hour=08:30")
    r1 = model.get_eta("Z05", "Z09")
    print(f"\n  Z05 (Pune) → Z09 (Nashik)")
    print(f"    Base ETA:    {r1.base_eta_minutes:.1f} min")
    print(f"    Current ETA: {r1.current_eta_minutes:.1f} min")
    print(f"    Bounds:      [{r1.eta_lower_bound:.1f}, {r1.eta_upper_bound:.1f}]")
    print(f"    Path:        {' → '.join(r1.path)}")
    r2 = model.get_eta("Z01", "Z27")
    print(f"\n  Z01 (Mumbai City) → Z27 (Nagpur)")
    print(f"    Current ETA: {r2.current_eta_minutes:.1f} min  |  dist: {r2.total_distance_km:.0f} km")
    print("\n  Stepping through AM peak (510→630 min):")
    for step in range(40):
        model.step(active_incident_zones=["Z05"] if step % 10 == 0 else None)
    desc = model.describe()
    print(f"    Hour: {desc['hour_of_day']}  |  Peak: {desc['peak_active']}  "
          f"|  Events: {desc['congestion_events']}  |  Mean OD: {desc['mean_od_minutes']} min")
    model.force_weather("storm")
    model.step()
    r3 = model.get_eta("Z05", "Z09")
    print(f"\n  Storm ETA Z05→Z09: {r3.current_eta_minutes:.1f} min  "
          f"(was {r1.current_eta_minutes:.1f} clear)")
    print("\n  OD sub-matrix (first 5 active zones):")
    model.print_od_table(max_zones=5)
    snap = model.get_snapshot()
    print(f"\n  Snapshot: weather={snap.weather}, peak={snap.peak_active}, "
          f"events={len(snap.congestion_events)}")
    fc = model.get_demand_forecast(look_ahead_hours=12)
    print(f"  Demand forecast: {len(fc)} zones × 12 hours")
    print(f"  Z05 forecast: {fc.get('Z05', [])[:6]}")
    print("\n✅  TrafficModel smoke-test PASSED")