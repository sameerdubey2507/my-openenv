from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from server.graders.basegrader import (
    ActionLogEntry,
    BaseGrader,
    GraderInput,
    GraderResult,
    GraderRegistry,
    GraderStatus,
    PenaltyRecord,
    ProtocolRuleChecker,
    ScoreComponent,
    ScoringUtils,
    TASK_BASELINES,
    TASK_SEEDS,
)
logger = logging.getLogger("emergi_env.graders.task6")
TASK_ID       = "task6_prepositioning"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]
W_ZONE_COVERAGE          = 0.25
W_DEMAND_ALIGNMENT       = 0.30
W_UNIT_TYPE_DISTRIBUTION = 0.20
W_GOLDEN_HOUR_READINESS  = 0.15
W_REPOSITIONING_EFFICIENCY = 0.10
assert abs(
    W_ZONE_COVERAGE + W_DEMAND_ALIGNMENT + W_UNIT_TYPE_DISTRIBUTION +
    W_GOLDEN_HOUR_READINESS + W_REPOSITIONING_EFFICIENCY - 1.0
) < 1e-9, "Task6 component weights must sum to 1.0"
PB_MICU_MULTI_ZONE       = 0.03
PB_NO_CLUSTERING         = 0.02
PB_TOP3_DEMAND_COVERAGE  = 0.02
PB_MUTUAL_AID_EDGE       = 0.02
PROTOCOL_BONUS_CAP       = 0.10
HP_CRITICAL_CLUSTERING   = 0.15    
HP_ZERO_MICU_COVERAGE    = 0.10    
HP_UNCOVERED_HIGH_DEMAND = 0.05    
DEMAND_HIGH_THRESHOLD     = 0.70
DEMAND_CRITICAL_THRESHOLD = 0.80
MICU_MULTI_ZONE_THRESHOLD = 4
CLUSTERING_ZONE_THRESHOLD = 3        
MAX_FLEET_PCT_PER_ZONE    = 0.40     
ZONE_TYPES: Dict[str, str] = {
    "Z1":  "metro_core",      "Z2":  "metro_core",
    "Z3":  "metro_suburban",  "Z4":  "metro_suburban",
    "Z5":  "metro_suburban",  "Z6":  "metro_suburban",
    "Z7":  "metro_satellite", "Z8":  "metro_satellite",
    "Z9":  "metro_satellite", "Z10": "semi_urban",
    "Z11": "semi_urban",      "Z12": "rural",
}
EDGE_ZONES: FrozenSet[str] = frozenset({"Z1", "Z6", "Z7", "Z12"})
HIGH_RISK_ZONE_TYPES: FrozenSet[str] = frozenset({"metro_core", "metro_suburban"})
UNIT_TYPES: Tuple[str, ...] = ("BLS", "ALS", "MICU")
IDEAL_RATIOS: Dict[str, Dict[str, float]] = {
    "metro_core":      {"BLS": 0.30, "ALS": 0.40, "MICU": 0.30},
    "metro_suburban":  {"BLS": 0.35, "ALS": 0.40, "MICU": 0.25},
    "metro_satellite": {"BLS": 0.45, "ALS": 0.35, "MICU": 0.20},
    "semi_urban":      {"BLS": 0.55, "ALS": 0.35, "MICU": 0.10},
    "rural":           {"BLS": 0.70, "ALS": 0.25, "MICU": 0.05},
}
@dataclass
class RepositionAction:
    unit_id:        str
    unit_type:      str
    from_zone:      str
    to_zone:        str
    step:           int
    action_idx:     int
    demand_at_dest: float = 0.0    
    @property
    def moved(self) -> bool:
        return self.from_zone != self.to_zone
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id":        self.unit_id,
            "unit_type":      self.unit_type,
            "from_zone":      self.from_zone,
            "to_zone":        self.to_zone,
            "step":           self.step,
            "moved":          self.moved,
            "demand_at_dest": round(self.demand_at_dest, 3),
        }
@dataclass
class FleetSnapshot:
    positions:      Dict[str, str]           
    unit_types:     Dict[str, str]           
    zone_units:     Dict[str, List[str]]     
    zone_by_type:   Dict[str, Dict[str, int]]  
    total_units:    int
    zones_covered:  Set[str]
    @classmethod
    def build(
        cls,
        positions:  Dict[str, str],
        unit_types: Dict[str, str],
    ) -> "FleetSnapshot":
        zone_units: Dict[str, List[str]] = defaultdict(list)
        zone_by_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"BLS": 0, "ALS": 0, "MICU": 0}
        )
        for uid, zone in positions.items():
            zone_units[zone].append(uid)
            utype = unit_types.get(uid, "ALS")
            zone_by_type[zone][utype] = zone_by_type[zone].get(utype, 0) + 1
        return cls(
            positions=positions,
            unit_types=unit_types,
            zone_units=dict(zone_units),
            zone_by_type=dict(zone_by_type),
            total_units=len(positions),
            zones_covered=set(zone_units.keys()),
        )
    def count_by_type(self, utype: str) -> int:
        return sum(1 for t in self.unit_types.values() if t.upper() == utype.upper())
    def zone_micu_count(self, zone: str) -> int:
        return self.zone_by_type.get(zone, {}).get("MICU", 0)
    def zone_als_or_micu_count(self, zone: str) -> int:
        zbt = self.zone_by_type.get(zone, {})
        return zbt.get("ALS", 0) + zbt.get("MICU", 0)
    def fraction_in_zone(self, zone: str) -> float:
        if self.total_units == 0:
            return 0.0
        return len(self.zone_units.get(zone, [])) / self.total_units
@dataclass
class DemandProfile:
    zone_demand:     Dict[str, float]  
    top3_zones:      List[str]
    critical_zones:  List[str]         
    @classmethod
    def build(cls, raw: Dict[str, float]) -> "DemandProfile":
        filled = {f"Z{i}": raw.get(f"Z{i}", 0.3) for i in range(1, 13)}
        sorted_zones = sorted(filled, key=lambda z: filled[z], reverse=True)
        critical = [z for z in sorted_zones if filled[z] >= DEMAND_CRITICAL_THRESHOLD]
        return cls(
            zone_demand=filled,
            top3_zones=sorted_zones[:3],
            critical_zones=critical,
        )
    def demand(self, zone: str) -> float:
        return self.zone_demand.get(zone, 0.3)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_demand":    {z: round(v, 3) for z, v in self.zone_demand.items()},
            "top3_zones":     self.top3_zones,
            "critical_zones": self.critical_zones,
        }
class ZoneCoverageScorer:
    BANDS: List[Tuple[int, float]] = [
        (12, 1.00), (10, 0.95), (9, 0.88),
        (8,  0.80), (7, 0.70), (6, 0.58),
        (5,  0.44), (4, 0.30), (3, 0.15),
        (2,  0.06), (1, 0.0),
    ]
    @classmethod
    def score(
        cls,
        snap:   FleetSnapshot,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        n_covered = len(snap.zones_covered)
        total     = 12
        raw = 0.0
        for threshold, val in cls.BANDS:
            if n_covered >= threshold:
                raw = val
                break
        edge_covered = snap.zones_covered & EDGE_ZONES
        edge_bonus = len(edge_covered) * 0.015
        raw = ScoringUtils.clamp(raw + edge_bonus)
        result.add_note(
            f"ZoneCoverage: {n_covered}/{total} zones "
            f"({len(edge_covered)}/{len(EDGE_ZONES)} edge zones) → {raw:.4f}"
        )
        return raw, {
            "zones_covered":       n_covered,
            "total_zones":         total,
            "edge_zones_covered":  list(edge_covered),
            "coverage_fraction":   round(n_covered / total, 4),
            "raw_score":           round(raw, 4),
        }
class DemandAlignmentScorer:
    ALS_MICU_DEMAND_MULTIPLIER = 1.40
    MIN_HIGH_DEMAND_UNITS      = 3    
    @classmethod
    def score(
        cls,
        snap:   FleetSnapshot,
        demand: DemandProfile,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if snap.total_units == 0:
            return 0.0, {"reason": "empty_fleet"}
        total_weight = 0.0
        weighted_sum = 0.0
        per_zone_detail: List[Dict] = []
        for zone in sorted(set(list(snap.zone_units.keys()) + list(demand.zone_demand.keys()))):
            d = demand.demand(zone)
            units_here = snap.zone_units.get(zone, [])
            for uid in units_here:
                utype = snap.unit_types.get(uid, "BLS")
                mult  = cls.ALS_MICU_DEMAND_MULTIPLIER if utype in ("ALS", "MICU") else 1.0
                contribution = d * mult
                weighted_sum += contribution
                total_weight += mult
        raw = weighted_sum / total_weight if total_weight > 0 else 0.0
        raw = ScoringUtils.clamp(raw)
        units_in_high = sum(
            len(snap.zone_units.get(z, []))
            for z in demand.zone_demand
            if demand.demand(z) >= DEMAND_HIGH_THRESHOLD
        )
        if units_in_high < cls.MIN_HIGH_DEMAND_UNITS:
            shortfall_penalty = (cls.MIN_HIGH_DEMAND_UNITS - units_in_high) * 0.04
            raw = ScoringUtils.clamp(raw - shortfall_penalty)
            result.add_note(
                f"DemandAlignment: only {units_in_high} units in high-demand zones "
                f"(min {cls.MIN_HIGH_DEMAND_UNITS}) → penalty {shortfall_penalty:.3f}"
            )
        result.add_note(
            f"DemandAlignment: weighted demand score {raw:.4f}, "
            f"units in high-demand zones: {units_in_high}"
        )
        return raw, {
            "weighted_demand_score":  round(raw, 4),
            "units_in_high_demand":   units_in_high,
            "top3_demand_zones":      demand.top3_zones,
            "critical_zones":         demand.critical_zones,
        }
class UnitTypeDistributionScorer:
    @classmethod
    def score(
        cls,
        snap:   FleetSnapshot,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if snap.total_units == 0:
            return 0.0, {"reason": "empty_fleet"}
        zone_scores: List[float] = []
        zone_details: List[Dict] = []
        for zone in snap.zones_covered:
            zone_type = ZONE_TYPES.get(zone, "semi_urban")
            ideal     = IDEAL_RATIOS.get(zone_type, IDEAL_RATIOS["semi_urban"])
            units     = snap.zone_units.get(zone, [])
            n         = len(units)
            if n == 0:
                continue
            actual: Dict[str, float] = {
                "BLS":  sum(1 for u in units if snap.unit_types.get(u) == "BLS") / n,
                "ALS":  sum(1 for u in units if snap.unit_types.get(u) == "ALS") / n,
                "MICU": sum(1 for u in units if snap.unit_types.get(u) == "MICU") / n,
            }
            l1 = sum(abs(actual.get(t, 0.0) - ideal.get(t, 0.0)) for t in UNIT_TYPES)
            zone_score = ScoringUtils.clamp(1.0 - l1 / 2.0)
            zone_scores.append(zone_score)
            zone_details.append({
                "zone": zone, "type": zone_type,
                "actual": {t: round(v, 3) for t, v in actual.items()},
                "ideal":  {t: round(v, 3) for t, v in ideal.items()},
                "l1_dist": round(l1, 4), "score": round(zone_score, 4),
            })
        if not zone_scores:
            return 0.30, {"reason": "no_positioned_zones"}
        raw = sum(zone_scores) / len(zone_scores)
        result.add_note(
            f"UnitTypeDistribution: avg zone score {raw:.4f} over "
            f"{len(zone_scores)} zones"
        )
        return ScoringUtils.clamp(raw), {
            "avg_zone_score": round(raw, 4),
            "zone_details":   zone_details,
        }
class GoldenHourReadinessScorer:
    ZONE_ADJACENCY: Dict[str, List[str]] = {
        "Z1": ["Z2","Z4"],    "Z2": ["Z1","Z3","Z5"],
        "Z3": ["Z2","Z4","Z6"], "Z4": ["Z1","Z3","Z7"],
        "Z5": ["Z2","Z6","Z8"], "Z6": ["Z3","Z5","Z9"],
        "Z7": ["Z4","Z8","Z10"],"Z8": ["Z5","Z7","Z11"],
        "Z9": ["Z6","Z10","Z12"],"Z10":["Z7","Z9","Z11"],
        "Z11":["Z8","Z10","Z12"],"Z12":["Z9","Z11"],
    }
    @classmethod
    def score(
        cls,
        snap:   FleetSnapshot,
        demand: DemandProfile,
        result: GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        high_risk_zones = [
            z for z, t in ZONE_TYPES.items()
            if t in HIGH_RISK_ZONE_TYPES
        ]
        if not high_risk_zones:
            return 0.80, {"reason": "no_high_risk_zones"}
        zone_scores: List[float] = []
        zone_details: List[Dict] = []
        for zone in high_risk_zones:
            d = demand.demand(zone)
            als_micu_here   = snap.zone_als_or_micu_count(zone)
            adjacent_backup = any(
                snap.zone_als_or_micu_count(adj) > 0
                for adj in cls.ZONE_ADJACENCY.get(zone, [])
            )
            if als_micu_here >= 2:
                base = 1.00
            elif als_micu_here == 1:
                base = 0.75
            elif adjacent_backup:
                base = 0.45
            else:
                base = 0.05
            weighted = base * (0.5 + 0.5 * d)
            zone_scores.append(weighted)
            zone_details.append({
                "zone": zone, "demand": round(d, 3),
                "als_micu_count": als_micu_here,
                "adjacent_backup": adjacent_backup,
                "zone_score": round(weighted, 4),
            })
        if not zone_scores:
            return 0.50, {"reason": "no_scored_zones"}
        raw = sum(zone_scores) / len(zone_scores)
        uncovered = sum(1 for z in zone_details
                        if z["als_micu_count"] == 0 and not z["adjacent_backup"])
        if uncovered > 0:
            result.add_note(
                f"GoldenHourReadiness: {uncovered} high-risk zone(s) have no "
                f"ALS/MICU and no adjacent backup"
            )
        return ScoringUtils.clamp(raw), {
            "high_risk_zones_checked": len(high_risk_zones),
            "uncovered_zones":         uncovered,
            "avg_readiness":           round(raw, 4),
            "zone_details":            zone_details,
        }
class RepositioningEfficiencyScorer:
    DEMAND_DIRECTED_THRESHOLD = 0.60    
    MICU_HIGH_DEMAND_BONUS    = 0.05
    WASTED_MOVE_PENALTY       = 0.03
    @classmethod
    def score(
        cls,
        actions:         List[RepositionAction],
        demand:          DemandProfile,
        initial_coverage: Set[str],
        result:          GraderResult,
    ) -> Tuple[float, Dict[str, Any]]:
        if not actions:
            result.add_note("RepositioningEfficiency: no reposition actions taken")
            return 0.10, {"reason": "no_actions", "action_count": 0}
        actual_moves = [a for a in actions if a.moved]
        if not actual_moves:
            result.add_note("RepositioningEfficiency: all reposition actions stayed in place")
            return 0.20, {"reason": "all_noop", "action_count": len(actions)}
        bonus         = 0.0
        penalty       = 0.0
        per_action:   List[Dict] = []
        newly_covered = set()
        for act in actual_moves:
            d_dest = demand.demand(act.to_zone)
            is_high_demand  = d_dest >= cls.DEMAND_DIRECTED_THRESHOLD
            new_zone_credit = act.to_zone not in initial_coverage and act.to_zone not in newly_covered
            move_bonus   = 0.0
            move_penalty = 0.0
            if is_high_demand:
                move_bonus += d_dest * 0.08
                if act.unit_type == "MICU":
                    move_bonus += cls.MICU_HIGH_DEMAND_BONUS
            else:
                move_penalty += cls.WASTED_MOVE_PENALTY * (1.0 - d_dest)
            if new_zone_credit:
                move_bonus += 0.04
                newly_covered.add(act.to_zone)
            bonus   += move_bonus
            penalty += move_penalty
            per_action.append({
                "unit_id":    act.unit_id,
                "unit_type":  act.unit_type,
                "to_zone":    act.to_zone,
                "demand":     round(d_dest, 3),
                "move_bonus": round(move_bonus, 4),
                "move_penalty": round(move_penalty, 4),
            })
        n   = len(actual_moves)
        raw = ScoringUtils.clamp(0.50 + bonus / n - penalty / n)
        result.add_note(
            f"RepositioningEfficiency: {len(actual_moves)} real moves, "
            f"total_bonus={bonus:.3f}, total_penalty={penalty:.3f} → {raw:.4f}"
        )
        return raw, {
            "actual_moves":       n,
            "newly_covered_zones": list(newly_covered),
            "per_action_detail":  per_action,
        }
class Task6PenaltyEngine:
    @staticmethod
    def apply(
        snap:   FleetSnapshot,
        demand: DemandProfile,
        result: GraderResult,
    ) -> float:
        total = 0.0
        if len(snap.zones_covered) <= CLUSTERING_ZONE_THRESHOLD:
            result.add_penalty(PenaltyRecord(
                name="critical_clustering",
                amount=-HP_CRITICAL_CLUSTERING,
                reason=(
                    f"Fleet clustered in only {len(snap.zones_covered)} zones — "
                    "entire region left unprotected"
                ),
                rule_ref="EMERGI-ENV Rule T6-H1",
            ))
            total -= HP_CRITICAL_CLUSTERING
            result.add_note(
                f"HARD PENALTY: critical clustering "
                f"({len(snap.zones_covered)} zones ≤ {CLUSTERING_ZONE_THRESHOLD})"
            )
        micu_count = snap.count_by_type("MICU")
        if micu_count == 0:
            result.add_penalty(PenaltyRecord(
                name="zero_micu_coverage",
                amount=-HP_ZERO_MICU_COVERAGE,
                reason="No MICU unit repositioned — zero advanced-life-support coverage",
                rule_ref="EMERGI-ENV Rule T6-H2",
            ))
            total -= HP_ZERO_MICU_COVERAGE
            result.add_note("HARD PENALTY: zero MICU repositioned")
        for zone in demand.critical_zones:
            if zone not in snap.zones_covered:
                result.add_penalty(PenaltyRecord(
                    name=f"uncovered_critical_{zone}",
                    amount=-HP_UNCOVERED_HIGH_DEMAND,
                    reason=(
                        f"Zone {zone} has demand {demand.demand(zone):.2f} "
                        f"(≥ {DEMAND_CRITICAL_THRESHOLD}) but zero units assigned"
                    ),
                    rule_ref="EMERGI-ENV Rule T6-H3",
                ))
                total -= HP_UNCOVERED_HIGH_DEMAND
                result.add_note(f"HARD PENALTY: critical zone {zone} uncovered")
        return total
class Task6BonusEngine:
    @staticmethod
    def apply(
        snap:    FleetSnapshot,
        demand:  DemandProfile,
        actions: List[RepositionAction],
        result:  GraderResult,
    ) -> float:
        total = 0.0
        micu_zones = {
            zone for zone, bt in snap.zone_by_type.items()
            if bt.get("MICU", 0) > 0
        }
        if len(micu_zones) >= MICU_MULTI_ZONE_THRESHOLD:
            result.add_note(
                f"Protocol bonus: MICU in {len(micu_zones)} distinct zones "
                f"(+{PB_MICU_MULTI_ZONE:.2f})"
            )
            total += PB_MICU_MULTI_ZONE
        max_frac = max(
            (snap.fraction_in_zone(z) for z in snap.zones_covered),
            default=0.0,
        )
        if max_frac <= MAX_FLEET_PCT_PER_ZONE:
            result.add_note(
                f"Protocol bonus: max zone concentration {max_frac:.1%} "
                f"≤ {MAX_FLEET_PCT_PER_ZONE:.0%} (+{PB_NO_CLUSTERING:.2f})"
            )
            total += PB_NO_CLUSTERING
        top3_covered = sum(
            1 for z in demand.top3_zones if z in snap.zones_covered
        )
        if top3_covered == 3:
            result.add_note(
                f"Protocol bonus: all top-3 demand zones covered "
                f"(+{PB_TOP3_DEMAND_COVERAGE:.2f})"
            )
            total += PB_TOP3_DEMAND_COVERAGE
        edge_covered = EDGE_ZONES & snap.zones_covered
        if len(edge_covered) == len(EDGE_ZONES):
            result.add_note(
                f"Protocol bonus: all {len(EDGE_ZONES)} edge/mutual-aid zones "
                f"covered (+{PB_MUTUAL_AID_EDGE:.2f})"
            )
            total += PB_MUTUAL_AID_EDGE
        return min(total, PROTOCOL_BONUS_CAP)
def _parse_reposition_actions(gi: GraderInput, demand: DemandProfile) -> List[RepositionAction]:
    actions: List[RepositionAction] = []
    for idx, entry in enumerate(gi.action_log):
        if entry.action_type not in ("reposition", "pre_position", "move_unit"):
            continue
        uid       = entry.get("unit_id") or entry.get("ambulance_id") or ""
        utype     = (entry.get("unit_type") or "ALS").upper()
        from_zone = entry.get("from_zone") or entry.get("current_zone") or "Z1"
        to_zone   = (
            entry.get("to_zone")
            or entry.get("zone")
            or entry.get("target_zone")
            or from_zone
        )
        actions.append(RepositionAction(
            unit_id=uid, unit_type=utype,
            from_zone=from_zone, to_zone=to_zone,
            step=entry.step, action_idx=idx,
            demand_at_dest=demand.demand(to_zone),
        ))
    return actions
def _parse_fleet_snapshot(gi: GraderInput, actions: List[RepositionAction]) -> Tuple[FleetSnapshot, Set[str]]:
    ledger = gi.episode_ledger
    final_pos  = ledger.get("final_fleet_positions", {})
    unit_types = ledger.get("unit_types", {})
    if final_pos:
        snap = FleetSnapshot.build(final_pos, unit_types)
        initial_raw = ledger.get("initial_fleet_positions", {})
        initial_cov = set(initial_raw.values()) if initial_raw else set()
        return snap, initial_cov
    if gi.observation_log:
        last_obs = gi.observation_log[-1]
        fleet_raw = last_obs.get("fleet_status", [])
        positions: Dict[str, str] = {}
        types_map: Dict[str, str] = {}
        for unit in fleet_raw:
            uid = unit.get("unit_id", "")
            positions[uid] = unit.get("zone", unit.get("current_zone", "Z1"))
            types_map[uid] = (unit.get("unit_type", "ALS")).upper()
        if positions:
            snap = FleetSnapshot.build(positions, types_map)
            first_obs = gi.observation_log[0]
            fleet_init = first_obs.get("fleet_status", [])
            initial_cov = {u.get("zone","Z1") for u in fleet_init if u.get("zone")}
            return snap, initial_cov
    if actions:
        positions = {}
        types_map = {}
        by_unit: Dict[str, List[RepositionAction]] = defaultdict(list)
        for a in actions:
            by_unit[a.unit_id].append(a)
        initial_cov: Set[str] = set()
        for uid, acts in by_unit.items():
            acts_sorted = sorted(acts, key=lambda x: x.step)
            initial_cov.add(acts_sorted[0].from_zone)
            positions[uid] = acts_sorted[-1].to_zone
            types_map[uid] = acts_sorted[0].unit_type
        if positions:
            snap = FleetSnapshot.build(positions, types_map)
            return snap, initial_cov
    default_fleet_size = ledger.get("fleet_size", 10)
    positions = {f"U{i+1:03d}": "Z1" for i in range(default_fleet_size)}
    types_map = {
        f"U{i+1:03d}": (["BLS","ALS","MICU"][i % 3]) for i in range(default_fleet_size)
    }
    return FleetSnapshot.build(positions, types_map), {"Z1"}
def _parse_demand_profile(gi: GraderInput) -> DemandProfile:
    ledger = gi.episode_ledger
    if "demand_heatmap" in ledger:
        return DemandProfile.build(ledger["demand_heatmap"])
    if gi.observation_log:
        obs = gi.observation_log[0]
        raw = obs.get("demand_forecast") or obs.get("demand_heatmap") or {}
        if raw:
            return DemandProfile.build(raw)
    import random
    rng = random.Random(gi.seed)
    return DemandProfile.build({f"Z{i}": rng.uniform(0.1, 0.9) for i in range(1, 13)})
class Task6Grader(BaseGrader):
    TASK_ID         = TASK_ID
    TASK_SEED       = TASK_SEED
    TASK_BASELINE   = TASK_BASELINE
    TASK_DIFFICULTY = "medium"
    COMPONENT_WEIGHTS: Dict[str, float] = {
        "zone_coverage":             W_ZONE_COVERAGE,
        "demand_alignment":          W_DEMAND_ALIGNMENT,
        "unit_type_distribution":    W_UNIT_TYPE_DISTRIBUTION,
        "golden_hour_readiness":     W_GOLDEN_HOUR_READINESS,
        "repositioning_efficiency":  W_REPOSITIONING_EFFICIENCY,
    }
    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(
                f"WARNING: episode seed {gi.seed} ≠ task seed {TASK_SEED}. "
                "Grading proceeds but determinism not guaranteed."
            )
        demand  = _parse_demand_profile(gi)
        actions = _parse_reposition_actions(gi, demand)
        snap, initial_coverage = _parse_fleet_snapshot(gi, actions)
        result.extra["demand_profile"]   = demand.to_dict()
        result.extra["fleet_total"]      = snap.total_units
        result.extra["zones_covered"]    = sorted(snap.zones_covered)
        result.extra["reposition_count"] = len(actions)
        result.extra["initial_coverage"] = sorted(initial_coverage)
        if snap.total_units == 0:
            result.status        = GraderStatus.INVALID_INPUT
            result.error_message = (
                "Fleet snapshot is empty. "
                "Check episode_ledger.final_fleet_positions or observation_log."
            )
            result.final_score = 0.0
            return
        cov_score, cov_detail = ZoneCoverageScorer.score(snap, result)
        self._add_component(
            result, "zone_coverage", cov_score, W_ZONE_COVERAGE,
            f"{len(snap.zones_covered)}/12 zones covered"
        )
        result.extra["zone_coverage_detail"] = cov_detail
        da_score, da_detail = DemandAlignmentScorer.score(snap, demand, result)
        self._add_component(
            result, "demand_alignment", da_score, W_DEMAND_ALIGNMENT,
            f"weighted demand alignment {da_score:.4f}"
        )
        result.extra["demand_alignment_detail"] = da_detail
        utd_score, utd_detail = UnitTypeDistributionScorer.score(snap, result)
        self._add_component(
            result, "unit_type_distribution", utd_score, W_UNIT_TYPE_DISTRIBUTION,
            f"avg zone type-mix L1 alignment {utd_score:.4f}"
        )
        result.extra["unit_type_distribution_detail"] = utd_detail
        ghr_score, ghr_detail = GoldenHourReadinessScorer.score(snap, demand, result)
        self._add_component(
            result, "golden_hour_readiness", ghr_score, W_GOLDEN_HOUR_READINESS,
            f"high-risk zone ALS/MICU readiness {ghr_score:.4f}"
        )
        result.extra["golden_hour_readiness_detail"] = ghr_detail
        re_score, re_detail = RepositioningEfficiencyScorer.score(
            actions, demand, initial_coverage, result
        )
        self._add_component(
            result, "repositioning_efficiency", re_score, W_REPOSITIONING_EFFICIENCY,
            f"{len([a for a in actions if a.moved])} real moves → {re_score:.4f}"
        )
        result.extra["repositioning_efficiency_detail"] = re_detail
        penalty_total = Task6PenaltyEngine.apply(snap, demand, result)
        result.extra["hard_penalty_total"] = round(penalty_total, 4)
        proto_bonus = Task6BonusEngine.apply(snap, demand, actions, result)
        result.extra["protocol_bonus_total"] = round(proto_bonus, 4)
        result.extra["_proto_bonus"]   = proto_bonus
        result.extra["_hard_penalty"]  = penalty_total
        result.total_patients   = 0
        result.p1_patients      = 0
        result.p1_survival_rate = 1.0   
    def _finalise_score_override(self, raw_weighted: float, result: GraderResult) -> float:
        bonus   = result.extra.pop("_proto_bonus", 0.0)
        penalty = result.extra.pop("_hard_penalty", 0.0)
        return ScoringUtils.clamp(raw_weighted + bonus + penalty)
def _build_test_input(
    zones_covered:    int     = 9,
    micu_count:       int     = 3,
    high_demand_units:int     = 5,
    reposition_count: int     = 8,
    on_edge_zones:    bool    = True,
    clustering:       bool    = False,
) -> GraderInput:
    import random
    rng = random.Random(TASK_SEED)
    all_zones  = [f"Z{i}" for i in range(1, 13)]
    sel_zones  = (all_zones[:3] * 5) if clustering else all_zones[:zones_covered]
    unit_types = (
        ["MICU"] * micu_count +
        ["ALS"]  * ((zones_covered * 2) // 3) +
        ["BLS"]  * max(1, zones_covered)
    )
    fleet_size = len(unit_types)
    final_pos  = {f"U{i+1:03d}": sel_zones[i % len(sel_zones)] for i in range(fleet_size)}
    types_map  = {f"U{i+1:03d}": unit_types[i] for i in range(fleet_size)}
    demand_raw = {z: rng.uniform(0.1, 0.9) for z in all_zones}
    for z in all_zones[:3]:
        demand_raw[z] = rng.uniform(0.72, 0.95)
    action_log: List[ActionLogEntry] = []
    for i in range(reposition_count):
        uid    = f"U{(i % fleet_size) + 1:03d}"
        fzone  = sel_zones[i % len(sel_zones)]
        tzone  = all_zones[(i + 3) % 12]
        action_log.append(ActionLogEntry(
            step=i + 1, 
            action_type="reposition",
             action_data={
        "ambulance_id": f"AMB-{i+1:02d}",
        "target_zone":  f"zone_{(i % 12) + 1}",
    },
       
        ))
    return GraderInput(
        task_id=TASK_ID,
        episode_id=f"ep-t6-test-{zones_covered}z",
        seed=TASK_SEED,
        action_log=action_log,
        episode_ledger={
            "final_fleet_positions": final_pos,
            "initial_fleet_positions": {f"U{i+1:03d}": "Z1" for i in range(fleet_size)},
            "unit_types": types_map,
            "demand_heatmap": demand_raw,
            "fleet_size": fleet_size,
        },
        observation_log=[],
        episode_steps=reposition_count,
        total_patients=0,
        p1_patients=0,
    )
def _run_self_tests() -> None:
    grader   = Task6Grader()
    failures: List[str] = []
    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
    gi1  = _build_test_input()
    res1 = grader.grade(gi1)
    check("T1_range",    0.0 <= res1.final_score <= 1.0, f"score={res1.final_score}")
    check("T1_baseline", res1.final_score >= TASK_BASELINE,
          f"score {res1.final_score:.4f} < baseline {TASK_BASELINE}")
    check("T1_components", len(res1.components) >= 5)
    gi2  = _build_test_input()
    res2 = grader.grade(gi2)
    check("T2_determinism",
          abs(res1.final_score - res2.final_score) < 1e-9,
          f"{res1.final_score} ≠ {res2.final_score}")
    good_res = grader.grade(_build_test_input(zones_covered=10, micu_count=4))
    poor_res = grader.grade(_build_test_input(zones_covered=3,  micu_count=0))
    check("T3_coverage_ordering",
          good_res.final_score > poor_res.final_score,
          f"good={good_res.final_score:.4f} should > poor={poor_res.final_score:.4f}")
    clust = grader.grade(_build_test_input(clustering=True, zones_covered=3))
    check("T4_clustering_penalty",
          any("critical_clustering" in p.name for p in clust.penalties),
          "Missing critical_clustering penalty")
    check("T4_range", 0.0 <= clust.final_score <= 1.0)
    no_micu = grader.grade(_build_test_input(micu_count=0))
    check("T5_zero_micu_penalty",
          any("zero_micu" in p.name for p in no_micu.penalties),
          "Missing zero_micu_coverage penalty")
    check("T5_range", 0.0 <= no_micu.final_score <= 1.0)
    active_res  = grader.grade(_build_test_input(reposition_count=12))
    passive_res = grader.grade(_build_test_input(reposition_count=0))
    check("T6_actions_help",
          active_res.final_score >= passive_res.final_score - 0.05,
          f"active={active_res.final_score:.4f} passive={passive_res.final_score:.4f}")
    d = res1.as_dict()
    check("T7_dict_keys",
          all(k in d for k in ("final_score", "components", "penalties", "notes")))
    check("T7_json_valid", len(res1.as_json()) > 50)
    check("T7_summary",    TASK_ID in res1.summary_line())
    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task6Grader self-test: {len(failures)} failure(s):\n" +
            "\n".join(failures)
        )
    logger.info("Task6Grader self-test PASSED (7 test cases).")
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task6Grader self-test FAILED at import: %s", _e)
    raise
if __name__ == "__main__":
    import json as _json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    print("=" * 72)
    print("EMERGI-ENV  ·  Task6Grader (Pre-positioning)  ·  Interactive demo")
    print("=" * 72)
    grader = Task6Grader()
    scenarios = [
        ("Optimal — 10 zones, 4 MICU, high demand coverage",
         _build_test_input(zones_covered=10, micu_count=4, reposition_count=12)),
        ("Good — 8 zones, 3 MICU",
         _build_test_input(zones_covered=8, micu_count=3, reposition_count=8)),
        ("Mediocre — 6 zones, 2 MICU, few actions",
         _build_test_input(zones_covered=6, micu_count=2, reposition_count=3)),
        ("Poor — 4 zones, 1 MICU",
         _build_test_input(zones_covered=4, micu_count=1, reposition_count=2)),
        ("Critical clustering — all units in Z1",
         _build_test_input(clustering=True, zones_covered=3, micu_count=0)),
        ("Zero MICU — ALS/BLS only, 9 zones",
         _build_test_input(zones_covered=9, micu_count=0, reposition_count=6)),
    ]
    results_list = []
    for name, gi in scenarios:
        res = grader.grade(gi)
        results_list.append(res)
        beats = "✓" if res.beats_baseline else "✗"
        print(f"\n  [{beats}] {name}")
        print(f"       Score:     {res.final_score:.4f}  "
              f"(baseline {TASK_BASELINE:.2f}, Δ={res.score_delta_vs_baseline:+.4f})")
        print(f"       Status:    {res.status.value}")
        print(f"       Components:")
        for c in res.components:
            print(f"         {c.name:<28} raw={c.raw_score:.4f}  "
                  f"w={c.weight:.2f}  weighted={c.weighted:.4f}  | {c.notes}")
        if res.penalties:
            print(f"       Penalties ({len(res.penalties)}):")
            for p in res.penalties:
                print(f"         {p.name:<30} {p.amount:+.4f}  | {p.reason}")
        for n in res.notes[:3]:
            print(f"         NOTE: {n}")
    print("\n" + "=" * 72)
    beats = sum(1 for r in results_list if r.beats_baseline)
    print(f"  {beats}/{len(results_list)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print("=" * 72)
    print("\n✅  Task6Grader demo complete.")