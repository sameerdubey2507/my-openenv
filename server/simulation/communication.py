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
TASK_COMM_FAILURE_PROB: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.00,   
    4: 0.00, 5: 0.00, 6: 0.00,   
    7: 0.12, 8: 0.12, 9: 0.12,   
}
COMM_RESTORE_PROB_BASE:    float = 0.25   
COMM_RESTORE_MIN_STEPS:    int   = 1      
BLACKOUT_MAX_STEPS:        int   = 20     
NOOP_UNKNOWN_UNIT_PENALTY: float = -0.05  
NOOP_GRACE_STEPS:          int   = 2      
MAX_MSG_BUFFER_PER_UNIT:   int   = 50     
MAX_MSG_PER_STEP_GLOBAL:   int   = 120    
BANDWIDTH_OVERLOAD_PENALTY:float = -0.02  
RSSI_THRESHOLD_DIGITAL:    float = 0.35   
RSSI_THRESHOLD_ANALOGUE:   float = 0.15   
RSSI_URBAN_BASE:           float = 0.95
RSSI_SUBURBAN_BASE:        float = 0.85
RSSI_RURAL_BASE:           float = 0.65
RSSI_TERRAIN_PENALTY: Dict[str, float] = {
    "ghat":    0.20,
    "forest":  0.15,
    "flood":   0.10,
    "hilly":   0.12,
}
RSSI_WEATHER_PENALTY: Dict[str, float] = {
    "clear": 0.00,
    "haze":  0.02,
    "rain":  0.08,
    "fog":   0.10,
    "storm": 0.20,
}
RELAY_TOWER_COVERAGE_RADIUS: int   = 2
RELAY_TOWER_BOOST:           float = 0.20  
RELAY_FAIL_PROB:             float = 0.01  
JAMMING_PROB_TASK9:          float = 0.05   
JAMMING_DURATION_STEPS:      int   = 6
JAMMING_RSSI_SUPPRESS:       float = 0.85   
CHANNEL_DEGRADED_FAILURE_MULT: float = 2.5  
CHANNEL_DEGRADED_THRESHOLD:    int   = 5    
class CommChannel(str, Enum):
    DIGITAL_CAD  = "digital_cad"    
    ANALOGUE_VHF = "analogue_vhf"   
    DEGRADED     = "degraded"       
    BLACKOUT     = "blackout"       
class MessageType(str, Enum):
    DISPATCH          = "dispatch"
    REROUTE           = "reroute"
    STATUS_UPDATE     = "status_update"
    POSITION_UPDATE   = "position_update"
    PRE_ALERT         = "pre_alert"
    MUTUAL_AID        = "mutual_aid"
    CREW_SWAP         = "crew_swap"
    SCENE_REPORT      = "scene_report"
    INCIDENT_ESCALATE = "incident_escalate"
    ACK               = "ack"
    NOOP_PENALTY      = "noop_penalty"
    SYSTEM_ALERT      = "system_alert"
class MessagePriority(str, Enum):
    EMERGENCY = "emergency"   
    HIGH      = "high"        
    NORMAL    = "normal"      
    LOW       = "low"         
class RelayTowerStatus(str, Enum):
    OPERATIONAL  = "operational"
    DEGRADED     = "degraded"
    FAILED       = "failed"
@dataclass
class CommMessage:
    message_id:    str
    message_type:  MessageType
    priority:      MessagePriority
    sender_id:     str            
    recipient_id:  str            
    payload:       Dict[str, Any]
    created_at_min: float
    delivered_at_min: Optional[float] = None
    acknowledged:  bool = False
    retries:       int  = 0
    max_retries:   int  = 3
    ttl_minutes:   float = 30.0   
    requires_ack:  bool  = False
    channel_used:  Optional[CommChannel] = None
    @property
    def is_expired(self) -> bool:
        return (self.delivered_at_min or 0.0) > (self.created_at_min + self.ttl_minutes)
    @property
    def age_minutes(self) -> float:
        return max(0.0, (self.delivered_at_min or 0.0) - self.created_at_min)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id":     self.message_id,
            "type":           self.message_type.value,
            "priority":       self.priority.value,
            "sender_id":      self.sender_id,
            "recipient_id":   self.recipient_id,
            "created_at_min": round(self.created_at_min, 1),
            "delivered_at_min": round(self.delivered_at_min, 1)
                               if self.delivered_at_min else None,
            "acknowledged":   self.acknowledged,
            "retries":        self.retries,
            "channel_used":   self.channel_used.value if self.channel_used else None,
        }
@dataclass
class UnitCommState:
    unit_id:              str
    channel:              CommChannel = CommChannel.DIGITAL_CAD
    rssi:                 float = 1.0          
    blackout_active:      bool  = False
    blackout_since_step:  int   = 0
    blackout_steps:       int   = 0            
    restore_eligible_step: int  = 0            
    last_known_zone_id:   Optional[str] = None
    last_known_lat:       float = 0.0
    last_known_lon:       float = 0.0
    last_comm_at_min:     float = 0.0
    last_comm_step:       int   = 0
    noop_penalty_steps:   int   = 0            
    noop_grace_remaining: int   = NOOP_GRACE_STEPS
    message_buffer:       Deque[CommMessage] = field(
        default_factory=lambda: deque(maxlen=MAX_MSG_BUFFER_PER_UNIT)
    )
    pending_acks:         Dict[str, CommMessage] = field(default_factory=dict)
    messages_sent:        int  = 0
    messages_received:    int  = 0
    messages_dropped:     int  = 0
    blackout_event_count: int  = 0
    total_blackout_min:   float = 0.0
    @property
    def position_age_minutes(self) -> float:
        return 0.0  
    @property
    def is_position_fresh(self) -> bool:
        return not self.blackout_active
    def to_observation_dict(self, sim_time_min: float) -> Dict[str, Any]:
        age = round(sim_time_min - self.last_comm_at_min, 1) if self.blackout_active else 0.0
        return {
            "unit_id":             self.unit_id,
            "channel":             self.channel.value,
            "rssi":                round(self.rssi, 3),
            "blackout_active":     self.blackout_active,
            "blackout_steps":      self.blackout_steps,
            "last_known_zone_id":  self.last_known_zone_id if self.blackout_active else None,
            "last_known_lat":      round(self.last_known_lat, 5) if self.blackout_active else None,
            "last_known_lon":      round(self.last_known_lon, 5) if self.blackout_active else None,
            "position_age_min":    age,
            "last_comm_at_min":    round(self.last_comm_at_min, 1),
            "buffered_messages":   len(self.message_buffer),
            "noop_penalty_steps":  self.noop_penalty_steps,
        }
@dataclass
class RelayTower:
    tower_id:      str
    zone_id:       str
    lat:           float
    lon:           float
    covered_zones: List[str]          
    status:        RelayTowerStatus = RelayTowerStatus.OPERATIONAL
    rssi_boost:    float = RELAY_TOWER_BOOST
    failed_at_step: Optional[int] = None
    repair_eta_steps: int = 0
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tower_id":      self.tower_id,
            "zone_id":       self.zone_id,
            "status":        self.status.value,
            "covered_zones": self.covered_zones,
            "rssi_boost":    round(self.rssi_boost, 3),
        }
@dataclass
class JammingEvent:
    event_id:        str
    affected_zones:  List[str]
    started_at_step: int
    duration_steps:  int
    rssi_suppress:   float = JAMMING_RSSI_SUPPRESS
    cause:           str   = "unknown"         
    @property
    def remaining_steps(self) -> int:
        return max(0, self.duration_steps - 0)  
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":       self.event_id,
            "affected_zones": self.affected_zones,
            "duration_steps": self.duration_steps,
            "rssi_suppress":  self.rssi_suppress,
            "cause":          self.cause,
        }
@dataclass
class ChannelStats:
    step:                 int
    messages_attempted:   int   = 0
    messages_delivered:   int   = 0
    messages_buffered:    int   = 0
    messages_dropped:     int   = 0
    bandwidth_overloaded: bool  = False
    units_in_blackout:    int   = 0
    active_jamming_zones: int   = 0
    mean_rssi:            float = 1.0
class CommunicationsManager:
    def __init__(
        self,
        seed:      int  = 42,
        task_id:   int  = 1,
        zone_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.seed    = seed
        self.task_id = task_id
        self.rng     = random.Random(seed)
        self.np_rng  = np.random.RandomState(seed)
        self.unit_states:   Dict[str, UnitCommState] = {}
        self.relay_towers:  Dict[str, RelayTower]  = {}
        self._tower_seq:    int = 0
        self.jamming_events: List[JammingEvent] = []
        self._jam_seq:       int = 0
        self._msg_seq:         int = 0
        self._global_outbox:   List[CommMessage] = []
        self._delivered_log:   List[CommMessage] = []
        self._consecutive_overload_steps: int   = 0
        self._channel_degraded:           bool  = False
        self._bandwidth_penalty_this_step: float = 0.0
        self._zone_meta:  Dict[str, Any] = zone_data or {}
        self._weather:    str = "clear"   
        self._active_zones: List[str] = []
        self.sim_time_min: float = 480.0
        self.step_count:   int   = 0
        self._step_stats:   List[ChannelStats] = []
        self._reward_log:   List[float]        = []
        self._total_blackout_penalties:   float = 0.0
        self._total_noop_penalties:       float = 0.0
        self._total_bandwidth_penalties:  float = 0.0
        self._total_jamming_penalties:    float = 0.0
        self._total_restore_bonuses:      float = 0.0
        self._load_zone_meta()
    def reset(
        self,
        unit_ids:         List[str],
        active_zone_ids:  List[str],
        task_id:          int   = 1,
        sim_time_minutes: float = 480.0,
        weather:          str   = "clear",
    ) -> Dict[str, Any]:
        self.task_id       = task_id
        self.sim_time_min  = sim_time_minutes
        self.step_count    = 0
        self._weather      = weather
        self._active_zones = list(active_zone_ids)
        self.rng    = random.Random(self.seed + task_id * 31)
        self.np_rng = np.random.RandomState(self.seed + task_id * 31)
        self.unit_states    = {}
        self.relay_towers   = {}
        self._tower_seq     = 0
        self.jamming_events = []
        self._jam_seq       = 0
        self._global_outbox = []
        self._delivered_log = []
        self._step_stats    = []
        self._reward_log    = []
        self._consecutive_overload_steps = 0
        self._channel_degraded = False
        self._total_blackout_penalties   = 0.0
        self._total_noop_penalties       = 0.0
        self._total_bandwidth_penalties  = 0.0
        self._total_jamming_penalties    = 0.0
        self._total_restore_bonuses      = 0.0
        for uid in unit_ids:
            self._register_unit(uid)
        self._deploy_relay_towers(active_zone_ids)
        for uid, state in self.unit_states.items():
            state.rssi = self._compute_rssi(
                zone_id    = state.last_known_zone_id or "Z05",
                weather    = self._weather,
            )
            state.last_comm_at_min = sim_time_minutes
            state.last_comm_step   = 0
        return self.get_comms_observation()
    def step(
        self,
        sim_time_minutes: float,
        unit_positions:   Dict[str, Tuple[str, float, float]],
        weather:          str  = "clear",
        active_incident_zones: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        self.sim_time_min = sim_time_minutes
        self.step_count  += 1
        self._weather     = weather
        step_reward        = 0.0
        self._bandwidth_penalty_this_step = 0.0
        self._tick_relay_towers()
        self._tick_jamming_events()
        if self.task_id == 9:
            step_reward += self._maybe_spawn_jamming(active_incident_zones or [])
        self._update_unit_positions(unit_positions)
        failure_prob = TASK_COMM_FAILURE_PROB.get(self.task_id, 0.0)
        if self._channel_degraded:
            failure_prob = min(1.0, failure_prob * CHANNEL_DEGRADED_FAILURE_MULT)
        for uid, state in self.unit_states.items():
            zone = (unit_positions.get(uid, (None,))[0]
                    or state.last_known_zone_id
                    or "Z05")
            state.rssi = self._compute_rssi(zone, weather)
            state.channel = self._rssi_to_channel(state.rssi)
            if state.blackout_active:
                step_reward += self._tick_blackout_unit(state)
            else:
                effective_prob = self._effective_failure_prob(
                    failure_prob, state.rssi, zone
                )
                if self.rng.random() < effective_prob:
                    self._enter_blackout(state, unit_positions.get(uid))
                    step_reward -= 0.02   
                else:
                    pos = unit_positions.get(uid)
                    if pos:
                        state.last_known_zone_id = pos[0]
                        state.last_known_lat      = pos[1]
                        state.last_known_lon      = pos[2]
                    state.last_comm_at_min = sim_time_minutes
                    state.last_comm_step   = self.step_count
                    step_reward += self._flush_buffer(state)
        stat = self._route_messages()
        self._step_stats.append(stat)
        if stat.bandwidth_overloaded:
            self._consecutive_overload_steps += 1
            if self._consecutive_overload_steps >= CHANNEL_DEGRADED_THRESHOLD:
                if not self._channel_degraded:
                    self._channel_degraded = True
            penalty = BANDWIDTH_OVERLOAD_PENALTY
            step_reward += penalty
            self._total_bandwidth_penalties += abs(penalty)
            self._bandwidth_penalty_this_step = penalty
        else:
            self._consecutive_overload_steps = max(
                0, self._consecutive_overload_steps - 1
            )
            if self._consecutive_overload_steps == 0:
                self._channel_degraded = False
        self._reward_log.append(step_reward)
        return self.get_comms_observation(), step_reward
    def send_message(
        self,
        sender_id:    str,
        recipient_id: str,
        message_type: MessageType,
        priority:     MessagePriority,
        payload:      Dict[str, Any],
        requires_ack: bool = False,
        ttl_minutes:  float = 30.0,
    ) -> Tuple[bool, str, Optional[CommMessage]]:
        msg = CommMessage(
            message_id    = f"MSG-{self._msg_seq:06d}",
            message_type  = message_type,
            priority      = priority,
            sender_id     = sender_id,
            recipient_id  = recipient_id,
            payload       = payload,
            created_at_min= self.sim_time_min,
            requires_ack  = requires_ack,
            ttl_minutes   = ttl_minutes,
        )
        self._msg_seq += 1
        sender_state = self.unit_states.get(sender_id)
        if sender_state and sender_state.blackout_active:
            if len(sender_state.message_buffer) < MAX_MSG_BUFFER_PER_UNIT:
                sender_state.message_buffer.append(msg)
                return True, "buffered_sender_blackout", msg
            else:
                sender_state.messages_dropped += 1
                return False, "buffer_full_sender_blackout", None
        self._global_outbox.append(msg)
        if sender_state:
            sender_state.messages_sent += 1
        return True, "queued", msg
    def send_dispatch(
        self,
        unit_id:         str,
        incident_id:     str,
        incident_zone:   str,
        hospital_id:     str,
        priority:        str,
        condition_code:  str,
    ) -> Tuple[bool, str]:
        ok, reason, msg = self.send_message(
            sender_id    = "DISPATCH_CENTRE",
            recipient_id = unit_id,
            message_type = MessageType.DISPATCH,
            priority     = (MessagePriority.EMERGENCY
                            if priority == "P1"
                            else MessagePriority.HIGH),
            payload = {
                "incident_id":    incident_id,
                "incident_zone":  incident_zone,
                "hospital_id":    hospital_id,
                "priority":       priority,
                "condition_code": condition_code,
            },
            requires_ack = True,
            ttl_minutes  = 15.0,
        )
        return ok, reason
    def send_reroute(
        self,
        unit_id:        str,
        new_hospital_id: str,
        reason:         str,
    ) -> Tuple[bool, str]:
        ok, reason_str, _ = self.send_message(
            sender_id    = "DISPATCH_CENTRE",
            recipient_id = unit_id,
            message_type = MessageType.REROUTE,
            priority     = MessagePriority.HIGH,
            payload = {
                "new_hospital_id": new_hospital_id,
                "reason":          reason,
            },
            requires_ack = True,
            ttl_minutes  = 10.0,
        )
        return ok, reason_str
    def send_pre_alert(
        self,
        unit_id:       str,
        hospital_id:   str,
        condition_code: str,
        eta_minutes:   float,
    ) -> Tuple[bool, str]:
        ok, reason_str, _ = self.send_message(
            sender_id    = unit_id,
            recipient_id = "DISPATCH_CENTRE",    
            message_type = MessageType.PRE_ALERT,
            priority     = MessagePriority.HIGH,
            payload = {
                "hospital_id":    hospital_id,
                "condition_code": condition_code,
                "eta_minutes":    round(eta_minutes, 1),
            },
            ttl_minutes = 20.0,
        )
        return ok, reason_str
    def acknowledge_message(self, message_id: str, ack_by: str) -> bool:
        state = self.unit_states.get(ack_by)
        if state and message_id in state.pending_acks:
            state.pending_acks[message_id].acknowledged = True
            del state.pending_acks[message_id]
            return True
        for msg in self._delivered_log:
            if msg.message_id == message_id:
                msg.acknowledged = True
                return True
        return False
    def record_noop_on_unit(self, unit_id: str) -> float:
        state = self.unit_states.get(unit_id)
        if state is None:
            return 0.0
        if not state.blackout_active:
            return 0.0   
        if state.noop_grace_remaining > 0:
            state.noop_grace_remaining -= 1
            return 0.0
        penalty = NOOP_UNKNOWN_UNIT_PENALTY
        state.noop_penalty_steps += 1
        self._total_noop_penalties += abs(penalty)
        return penalty
    def spawn_jamming_event(
        self,
        affected_zones: List[str],
        duration_steps: int = JAMMING_DURATION_STEPS,
        cause:          str = "surge_interference",
    ) -> str:
        ev = JammingEvent(
            event_id        = f"JAM-{self._jam_seq:04d}",
            affected_zones  = list(affected_zones),
            started_at_step = self.step_count,
            duration_steps  = duration_steps,
            cause           = cause,
        )
        self.jamming_events.append(ev)
        self._jam_seq += 1
        return ev.event_id
    def cancel_jamming_event(self, event_id: str) -> bool:
        before = len(self.jamming_events)
        self.jamming_events = [j for j in self.jamming_events
                               if j.event_id != event_id]
        return len(self.jamming_events) < before
    def register_unit(self, unit_id: str, zone_id: str = "Z05",
                      lat: float = 18.52, lon: float = 73.85) -> None:
        if unit_id not in self.unit_states:
            state = UnitCommState(unit_id=unit_id)
            state.last_known_zone_id = zone_id
            state.last_known_lat     = lat
            state.last_known_lon     = lon
            state.last_comm_at_min   = self.sim_time_min
            state.rssi = self._compute_rssi(zone_id, self._weather)
            state.channel = self._rssi_to_channel(state.rssi)
            self.unit_states[unit_id] = state
    def deregister_unit(self, unit_id: str) -> bool:
        return self.unit_states.pop(unit_id, None) is not None
    def is_unit_reachable(self, unit_id: str) -> bool:
        state = self.unit_states.get(unit_id)
        if state is None:
            return False
        return not state.blackout_active
    def get_unit_channel(self, unit_id: str) -> CommChannel:
        state = self.unit_states.get(unit_id)
        if state is None:
            return CommChannel.BLACKOUT
        return state.channel
    def get_unit_rssi(self, unit_id: str) -> float:
        state = self.unit_states.get(unit_id)
        return state.rssi if state else 0.0
    def get_last_known_position(
        self, unit_id: str
    ) -> Optional[Tuple[str, float, float, float]]:
        state = self.unit_states.get(unit_id)
        if state is None or not state.blackout_active:
            return None
        age = round(self.sim_time_min - state.last_comm_at_min, 1)
        return (
            state.last_known_zone_id or "UNKNOWN",
            state.last_known_lat,
            state.last_known_lon,
            age,
        )
    def get_units_in_blackout(self) -> List[str]:
        return [uid for uid, s in self.unit_states.items() if s.blackout_active]
    def get_units_on_degraded_channel(self) -> List[str]:
        return [
            uid for uid, s in self.unit_states.items()
            if s.channel == CommChannel.DEGRADED and not s.blackout_active
        ]
    def force_blackout(self, unit_id: str) -> bool:
        state = self.unit_states.get(unit_id)
        if state is None:
            return False
        self._enter_blackout(state, None)
        return True
    def force_restore(self, unit_id: str) -> bool:
        state = self.unit_states.get(unit_id)
        if state is None:
            return False
        self._restore_comm(state)
        return True
    def update_weather(self, weather: str) -> None:
        self._weather = weather
        for uid, state in self.unit_states.items():
            zone = state.last_known_zone_id or "Z05"
            state.rssi    = self._compute_rssi(zone, weather)
            state.channel = self._rssi_to_channel(state.rssi)
    def get_comms_observation(self) -> Dict[str, Any]:
        units_obs = {
            uid: state.to_observation_dict(self.sim_time_min)
            for uid, state in self.unit_states.items()
        }
        blackout_count = sum(1 for s in self.unit_states.values() if s.blackout_active)
        mean_rssi = (
            sum(s.rssi for s in self.unit_states.values())
            / max(len(self.unit_states), 1)
        )
        jammed_zones: Set[str] = set()
        for jev in self.jamming_events:
            jammed_zones.update(jev.affected_zones)
        return {
            "task_id":              self.task_id,
            "failure_probability":  TASK_COMM_FAILURE_PROB.get(self.task_id, 0.0),
            "channel_degraded":     self._channel_degraded,
            "weather":              self._weather,
            "step_count":           self.step_count,
            "sim_time_min":         round(self.sim_time_min, 1),
            "units_in_blackout":    blackout_count,
            "mean_rssi":            round(mean_rssi, 3),
            "jammed_zones":         sorted(jammed_zones),
            "active_jamming_events":len(self.jamming_events),
            "relay_towers":         [t.to_dict() for t in self.relay_towers.values()],
            "relay_towers_failed":  sum(
                1 for t in self.relay_towers.values()
                if t.status == RelayTowerStatus.FAILED
            ),
            "unit_comm_states":     units_obs,
            "pending_outbox":       len(self._global_outbox),
        }
    def get_full_comm_state(self) -> Dict[str, Any]:
        base = self.get_comms_observation()
        base.update({
            "total_blackout_penalties":  round(self._total_blackout_penalties,  4),
            "total_noop_penalties":      round(self._total_noop_penalties,      4),
            "total_bandwidth_penalties": round(self._total_bandwidth_penalties, 4),
            "total_jamming_penalties":   round(self._total_jamming_penalties,   4),
            "total_restore_bonuses":     round(self._total_restore_bonuses,     4),
            "messages_delivered_total":  len(self._delivered_log),
            "messages_dropped_total":    sum(
                s.messages_dropped for s in self.unit_states.values()
            ),
            "channel_stats_last5":       [
                {
                    "step":                s.step,
                    "delivered":          s.messages_delivered,
                    "dropped":            s.messages_dropped,
                    "blackout_units":     s.units_in_blackout,
                    "bandwidth_overload": s.bandwidth_overloaded,
                    "mean_rssi":          round(s.mean_rssi, 3),
                }
                for s in self._step_stats[-5:]
            ],
            "jamming_events":            [j.to_dict() for j in self.jamming_events],
        })
        return base
    def get_episode_analytics(self) -> Dict[str, Any]:
        total_sent    = sum(s.messages_sent    for s in self.unit_states.values())
        total_recv    = sum(s.messages_received for s in self.unit_states.values())
        total_dropped = sum(s.messages_dropped  for s in self.unit_states.values())
        total_delivered = len(self._delivered_log)
        delivery_rate = total_delivered / max(total_sent, 1)
        total_blackouts  = sum(s.blackout_event_count for s in self.unit_states.values())
        total_blk_min    = sum(s.total_blackout_min   for s in self.unit_states.values())
        mean_rssi_ep     = (
            sum(s.rssi for s in self.unit_states.values())
            / max(len(self.unit_states), 1)
        )
        noop_steps_total = sum(s.noop_penalty_steps for s in self.unit_states.values())
        return {
            "total_messages_sent":          total_sent,
            "total_messages_received":      total_recv,
            "total_messages_dropped":       total_dropped,
            "total_messages_delivered":     total_delivered,
            "delivery_rate":                round(delivery_rate, 4),
            "total_blackout_events":        total_blackouts,
            "total_blackout_minutes":       round(total_blk_min, 1),
            "mean_rssi_episode":            round(mean_rssi_ep, 3),
            "noop_penalty_steps_total":     noop_steps_total,
            "total_blackout_penalties":     round(self._total_blackout_penalties,  4),
            "total_noop_penalties":         round(self._total_noop_penalties,      4),
            "total_bandwidth_penalties":    round(self._total_bandwidth_penalties, 4),
            "total_jamming_penalties":      round(self._total_jamming_penalties,   4),
            "total_restore_bonuses":        round(self._total_restore_bonuses,     4),
            "net_comm_reward":              round(
                self._total_restore_bonuses
                - self._total_blackout_penalties
                - self._total_noop_penalties
                - self._total_bandwidth_penalties
                - self._total_jamming_penalties,
                4
            ),
        }
    def describe(self) -> Dict[str, Any]:
        blk = self.get_units_in_blackout()
        return {
            "step":             self.step_count,
            "sim_time_min":     round(self.sim_time_min, 1),
            "task_id":          self.task_id,
            "registered_units": len(self.unit_states),
            "units_in_blackout":len(blk),
            "blackout_ids":     blk,
            "channel_degraded": self._channel_degraded,
            "jamming_events":   len(self.jamming_events),
            "relay_towers":     len(self.relay_towers),
            "relay_failed":     sum(
                1 for t in self.relay_towers.values()
                if t.status == RelayTowerStatus.FAILED
            ),
            "failure_prob":     TASK_COMM_FAILURE_PROB.get(self.task_id, 0.0),
            "weather":          self._weather,
        }
    def _enter_blackout(
        self,
        state:    UnitCommState,
        position: Optional[Tuple[str, float, float]],
    ) -> None:
        if state.blackout_active:
            return  
        state.blackout_active     = True
        state.blackout_since_step = self.step_count
        state.restore_eligible_step = self.step_count + COMM_RESTORE_MIN_STEPS
        state.noop_grace_remaining  = NOOP_GRACE_STEPS
        state.blackout_event_count += 1
        state.channel = CommChannel.BLACKOUT
        if position:
            state.last_known_zone_id = position[0]
            state.last_known_lat     = position[1]
            state.last_known_lon     = position[2]
        state.last_comm_at_min = self.sim_time_min
        state.last_comm_step   = self.step_count
    def _tick_blackout_unit(self, state: UnitCommState) -> float:
        reward = 0.0
        state.blackout_steps      += 1
        state.total_blackout_min  += STEP_DURATION_MINUTES
        self._total_blackout_penalties += 0.01   
        steps_blacked_out = self.step_count - state.blackout_since_step
        if steps_blacked_out >= BLACKOUT_MAX_STEPS:
            self._restore_comm(state)
            reward += 0.05  
            self._total_restore_bonuses += 0.05
            return reward
        if self.step_count >= state.restore_eligible_step:
            restore_prob = COMM_RESTORE_PROB_BASE
            zone_rssi = self._compute_rssi(
                state.last_known_zone_id or "Z05", self._weather
            )
            restore_prob *= (0.5 + 0.5 * zone_rssi)   
            if self._unit_in_relay_coverage(state.last_known_zone_id or "Z05"):
                restore_prob = min(1.0, restore_prob + 0.10)
            if self.rng.random() < restore_prob:
                self._restore_comm(state)
                reward += 0.08
                self._total_restore_bonuses += 0.08
        return reward
    def _restore_comm(self, state: UnitCommState) -> None:
        state.blackout_active  = False
        zone = state.last_known_zone_id or "Z05"
        state.rssi    = self._compute_rssi(zone, self._weather)
        state.channel = self._rssi_to_channel(state.rssi)
        state.last_comm_at_min = self.sim_time_min
        state.last_comm_step   = self.step_count
    def _flush_buffer(self, state: UnitCommState) -> float:
        if not state.message_buffer:
            return 0.0
        reward = 0.0
        flushed = 0
        while state.message_buffer:
            msg = state.message_buffer.popleft()
            if self.sim_time_min > msg.created_at_min + msg.ttl_minutes:
                state.messages_dropped += 1
                continue
            msg.delivered_at_min = self.sim_time_min
            msg.channel_used     = state.channel
            state.messages_received += 1
            self._delivered_log.append(msg)
            flushed += 1
        if flushed > 0:
            reward += 0.01 * min(flushed, 5)   
        return reward
    def _route_messages(self) -> ChannelStats:
        stat = ChannelStats(step=self.step_count)
        messages_this_step = len(self._global_outbox)
        stat.messages_attempted = messages_this_step
        stat.bandwidth_overloaded = messages_this_step > MAX_MSG_PER_STEP_GLOBAL
        outbox     = list(self._global_outbox)
        self._global_outbox = []
        priority_order = {
            MessagePriority.EMERGENCY: 0,
            MessagePriority.HIGH:      1,
            MessagePriority.NORMAL:    2,
            MessagePriority.LOW:       3,
        }
        outbox.sort(key=lambda m: priority_order.get(m.priority, 9))
        budget = MAX_MSG_PER_STEP_GLOBAL
        processed: List[CommMessage] = []
        dropped_bw: List[CommMessage] = []
        for msg in outbox:
            if budget > 0:
                processed.append(msg)
                budget -= 1
            else:
                if msg.priority in (MessagePriority.EMERGENCY, MessagePriority.HIGH):
                    processed.append(msg)   
                else:
                    dropped_bw.append(msg)
                    stat.messages_dropped += 1
        for msg in dropped_bw:
            recip_state = self.unit_states.get(msg.recipient_id)
            if recip_state:
                recip_state.messages_dropped += 1
        for msg in processed:
            recip_state = self.unit_states.get(msg.recipient_id)
            if recip_state:
                if recip_state.blackout_active:
                    if len(recip_state.message_buffer) < MAX_MSG_BUFFER_PER_UNIT:
                        recip_state.message_buffer.append(msg)
                        stat.messages_buffered += 1
                    else:
                        recip_state.messages_dropped += 1
                        stat.messages_dropped += 1
                else:
                    msg.delivered_at_min = self.sim_time_min
                    msg.channel_used     = recip_state.channel
                    recip_state.messages_received += 1
                    self._delivered_log.append(msg)
                    stat.messages_delivered += 1
                    if msg.requires_ack:
                        recip_state.pending_acks[msg.message_id] = msg
            else:
                msg.delivered_at_min = self.sim_time_min
                self._delivered_log.append(msg)
                stat.messages_delivered += 1
        stat.units_in_blackout     = sum(
            1 for s in self.unit_states.values() if s.blackout_active
        )
        stat.active_jamming_zones  = len({
            z for jev in self.jamming_events for z in jev.affected_zones
        })
        stat.mean_rssi = (
            sum(s.rssi for s in self.unit_states.values())
            / max(len(self.unit_states), 1)
        )
        return stat
    def _compute_rssi(self, zone_id: str, weather: str) -> float:
        z = self._zone_meta.get(zone_id, {})
        zone_type = z.get("zone_type", "urban")
        base_rssi_map = {
            "urban":      RSSI_URBAN_BASE,
            "suburban":   RSSI_SUBURBAN_BASE,
            "peri_urban": 0.78,
            "rural":      RSSI_RURAL_BASE,
            "industrial": 0.80,
            "highway":    0.75,
        }
        rssi = base_rssi_map.get(zone_type, 0.80)
        rn = z.get("road_network", {})
        geo = z.get("geography", {})
        road_q = float(rn.get("road_quality_score", 0.55))
        rssi  *= (0.7 + 0.3 * road_q)   
        if rn.get("ghat_sections", False) or geo.get("ghat_roads", False):
            rssi -= RSSI_TERRAIN_PENALTY["ghat"]
        if geo.get("forest_terrain", False):
            rssi -= RSSI_TERRAIN_PENALTY["forest"]
        if geo.get("flood_prone", False):
            rssi -= RSSI_TERRAIN_PENALTY["flood"]
        if geo.get("hilly", False):
            rssi -= RSSI_TERRAIN_PENALTY["hilly"]
        rssi -= RSSI_WEATHER_PENALTY.get(weather, 0.0)
        if geo.get("coastal", False) and weather in ("storm", "rain"):
            rssi -= 0.05
        if self._unit_in_relay_coverage(zone_id):
            rssi += RELAY_TOWER_BOOST
        for jev in self.jamming_events:
            if zone_id in jev.affected_zones:
                rssi *= (1.0 - jev.rssi_suppress)
                break   
        return max(0.0, min(1.0, rssi))
    @staticmethod
    def _rssi_to_channel(rssi: float) -> CommChannel:
        if rssi >= RSSI_THRESHOLD_DIGITAL:
            return CommChannel.DIGITAL_CAD
        if rssi >= RSSI_THRESHOLD_ANALOGUE:
            return CommChannel.ANALOGUE_VHF
        if rssi > 0.05:
            return CommChannel.DEGRADED
        return CommChannel.BLACKOUT
    def _effective_failure_prob(
        self,
        base_prob: float,
        rssi:      float,
        zone_id:   str,
    ) -> float:
        if base_prob <= 0.0:
            return 0.0
        rssi_mult = 3.0 - 2.7 * rssi   
        prob = base_prob * rssi_mult
        for jev in self.jamming_events:
            if zone_id in jev.affected_zones:
                prob = min(1.0, prob * 4.0)
                break
        return min(1.0, max(0.0, prob))
    def _deploy_relay_towers(self, active_zone_ids: List[str]) -> None:
        self.relay_towers = {}
        if not active_zone_ids:
            return
        adjacency: Dict[str, List[str]] = {}
        for zid in active_zone_ids:
            z = self._zone_meta.get(zid, {})
            adj = z.get("adjacent_zones", [])
            adj = [a for a in adj if a in set(active_zone_ids)]
            adjacency[zid] = adj
        tower_zones: List[str] = []
        for zid in active_zone_ids:
            z = self._zone_meta.get(zid, {})
            zone_type = z.get("zone_type", "urban")
            pop_density = float(z.get("population_density_per_sqkm", 100))
            if zone_type in ("urban", "industrial") or pop_density > 5000:
                tower_zones.append(zid)
        if len(tower_zones) < 3:
            sorted_by_pop = sorted(
                active_zone_ids,
                key=lambda z: float(
                    self._zone_meta.get(z, {}).get("population_density_per_sqkm", 0)
                ),
                reverse=True,
            )
            for zid in sorted_by_pop:
                if zid not in tower_zones:
                    tower_zones.append(zid)
                if len(tower_zones) >= 3:
                    break
        for zid in tower_zones:
            z     = self._zone_meta.get(zid, {})
            lat   = float(z.get("lat", 18.52))
            lon   = float(z.get("lon", 73.85))
            covered = [zid] + adjacency.get(zid, [])[:RELAY_TOWER_COVERAGE_RADIUS]
            self._tower_seq += 1
            tid = f"TOWER-{self._tower_seq:03d}"
            tower = RelayTower(
                tower_id      = tid,
                zone_id       = zid,
                lat           = lat,
                lon           = lon,
                covered_zones = covered,
            )
            self.relay_towers[tid] = tower
    def _tick_relay_towers(self) -> None:
        for tower in self.relay_towers.values():
            if tower.status == RelayTowerStatus.OPERATIONAL:
                if self.rng.random() < RELAY_FAIL_PROB:
                    tower.status         = RelayTowerStatus.FAILED
                    tower.failed_at_step = self.step_count
                    tower.repair_eta_steps = self.rng.randint(5, 15)
            elif tower.status == RelayTowerStatus.FAILED:
                if tower.repair_eta_steps > 0:
                    tower.repair_eta_steps -= 1
                else:
                    tower.status = RelayTowerStatus.OPERATIONAL
            elif tower.status == RelayTowerStatus.DEGRADED:
                if self.rng.random() < 0.05:
                    tower.status = RelayTowerStatus.FAILED
                    tower.repair_eta_steps = self.rng.randint(3, 10)
                elif self.rng.random() < 0.20:
                    tower.status = RelayTowerStatus.OPERATIONAL
    def _unit_in_relay_coverage(self, zone_id: str) -> bool:
        for tower in self.relay_towers.values():
            if (tower.status == RelayTowerStatus.OPERATIONAL
                    and zone_id in tower.covered_zones):
                return True
        return False
    def _tick_jamming_events(self) -> None:
        alive = []
        for jev in self.jamming_events:
            elapsed = self.step_count - jev.started_at_step
            if elapsed < jev.duration_steps:
                alive.append(jev)
        self.jamming_events = alive
    def _maybe_spawn_jamming(self, active_incident_zones: List[str]) -> float:
        if self.task_id != 9:
            return 0.0
        if self.rng.random() > JAMMING_PROB_TASK9:
            return 0.0
        if active_incident_zones:
            origin = self.rng.choice(active_incident_zones)
        elif self._active_zones:
            origin = self.rng.choice(self._active_zones)
        else:
            return 0.0
        z_meta  = self._zone_meta.get(origin, {})
        adj     = z_meta.get("adjacent_zones", [])
        affected = [origin] + adj[:2]  
        affected = [z for z in affected if z in set(self._active_zones)]
        if not affected:
            return 0.0
        jev = JammingEvent(
            event_id        = f"JAM-{self._jam_seq:04d}",
            affected_zones  = affected,
            started_at_step = self.step_count,
            duration_steps  = JAMMING_DURATION_STEPS,
            cause           = "surge_interference",
        )
        self.jamming_events.append(jev)
        self._jam_seq += 1
        penalty = 0.0
        for uid, state in self.unit_states.items():
            zone = state.last_known_zone_id or "Z05"
            if zone in affected and not state.blackout_active:
                self._enter_blackout(state, None)
                penalty -= 0.05
                self._total_jamming_penalties += 0.05
        return penalty
    def _update_unit_positions(
        self,
        unit_positions: Dict[str, Tuple[str, float, float]],
    ) -> None:
        for uid, pos in unit_positions.items():
            state = self.unit_states.get(uid)
            if state is None:
                self.register_unit(uid, pos[0], pos[1], pos[2])
                continue
            if not state.blackout_active:
                state.last_known_zone_id = pos[0]
                state.last_known_lat     = pos[1]
                state.last_known_lon     = pos[2]
                state.last_comm_at_min   = self.sim_time_min
                state.last_comm_step     = self.step_count
    def _register_unit(
        self,
        unit_id:  str,
        zone_id:  str = "Z05",
        lat:      float = 18.52,
        lon:      float = 73.85,
    ) -> None:
        state = UnitCommState(unit_id=unit_id)
        state.last_known_zone_id = zone_id
        state.last_known_lat     = lat
        state.last_known_lon     = lon
        state.last_comm_at_min   = self.sim_time_min
        state.last_comm_step     = 0
        self.unit_states[unit_id] = state
    def _load_zone_meta(self) -> None:
        if self._zone_meta:
            return
        path = DATA_DIR / "city_zones.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for z in raw.get("zones", []):
                self._zone_meta[z["zone_id"]] = z
    @staticmethod
    def _haversine_km(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        R  = 6371.0
        φ1 = math.radians(lat1); φ2 = math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    def print_status_table(self) -> None:
        header = (
            f"{'UNIT':>14} {'CHANNEL':>14} {'RSSI':>6} "
            f"{'BLK':>4} {'BLK_S':>6} {'LAST_ZONE':>10} "
            f"{'BUF':>5} {'NOOP':>5}"
        )
        print(header)
        print("-" * len(header))
        for uid, state in self.unit_states.items():
            print(
                f"{uid:>14} {state.channel.value:>14} "
                f"{state.rssi:>6.3f} "
                f"{'YES' if state.blackout_active else 'no':>4} "
                f"{state.blackout_steps:>6} "
                f"{state.last_known_zone_id or 'N/A':>10} "
                f"{len(state.message_buffer):>5} "
                f"{state.noop_penalty_steps:>5}"
            )
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 72)
    print("EMERGI-ENV  ·  CommunicationsManager smoke-test")
    print("=" * 72)
    mgr = CommunicationsManager(seed=42, task_id=7)
    active_zones = [
        "Z01", "Z02", "Z03", "Z04", "Z05",
        "Z06", "Z07", "Z08", "Z09",
    ]
    unit_ids = [
        "AMB-M001", "AMB-M002", "AMB-A001", "AMB-A002",
        "AMB-A003", "AMB-B001", "AMB-B002", "AMB-B003",
    ]
    obs = mgr.reset(
        unit_ids         = unit_ids,
        active_zone_ids  = active_zones,
        task_id          = 7,
        sim_time_minutes = 540.0,
        weather          = "rain",
    )
    print(f"\n✓  Registered {len(unit_ids)} units")
    print(f"   Relay towers deployed: {obs['relay_towers'].__len__()}")
    print(f"   Jammed zones at start: {obs['jammed_zones']}")
    print(f"   Mean RSSI: {obs['mean_rssi']:.3f}")
    positions = {
        "AMB-M001": ("Z05", 18.517, 73.856),
        "AMB-M002": ("Z06", 18.533, 73.882),
        "AMB-A001": ("Z03", 18.530, 73.880),
        "AMB-A002": ("Z08", 18.504, 73.824),
        "AMB-A003": ("Z02", 18.621, 73.800),
        "AMB-B001": ("Z07", 18.480, 73.900),
        "AMB-B002": ("Z09", 18.516, 73.845),
        "AMB-B003": ("Z04", 18.558, 73.809),
    }
    ok, reason = mgr.send_dispatch(
        unit_id        = "AMB-M001",
        incident_id    = "INC-0001",
        incident_zone  = "Z05",
        hospital_id    = "H01",
        priority       = "P1",
        condition_code = "STEMI",
    )
    print(f"\n✓  Dispatch to AMB-M001: {ok} — {reason}")
    ok2, r2 = mgr.send_reroute("AMB-A002", "H03", "diversion")
    print(f"✓  Reroute to AMB-A002: {ok2} — {r2}")
    ok3, r3 = mgr.send_pre_alert("AMB-M002", "H02", "POLYTRAUMA", 12.5)
    print(f"✓  Pre-alert from AMB-M002: {ok3} — {r3}")
    print(f"\n  Stepping 25 steps (Task 7, failure_prob=0.12, weather='rain'):")
    total_rwd = 0.0
    for s in range(25):
        new_time = 540.0 + (s + 1) * STEP_DURATION_MINUTES
        obs2, rwd = mgr.step(
            sim_time_minutes     = new_time,
            unit_positions       = positions,
            weather              = "rain" if s < 15 else "storm",
            active_incident_zones= ["Z05", "Z06"],
        )
        total_rwd += rwd
        blk = obs2["units_in_blackout"]
        if blk > 0 or s % 5 == 0:
            print(f"  Step {s+1:02d}: blackout={blk} units | "
                  f"rssi={obs2['mean_rssi']:.3f} | "
                  f"jam_zones={obs2['jammed_zones']} | "
                  f"rwd={rwd:+.4f}")
    print(f"\n  Total comm reward over 25 steps: {total_rwd:+.4f}")
    blk_units = mgr.get_units_in_blackout()
    if blk_units:
        uid_blk = blk_units[0]
        p = mgr.record_noop_on_unit(uid_blk)
        print(f"\n✓  noop_penalty on blacked-out {uid_blk}: {p}")
        for _ in range(NOOP_GRACE_STEPS + 1):
            p2 = mgr.record_noop_on_unit(uid_blk)
        print(f"   After grace period exhausted: {p2}")
    mgr.force_blackout("AMB-B003")
    print(f"\n✓  Forced blackout on AMB-B003 — reachable: {mgr.is_unit_reachable('AMB-B003')}")
    mgr.force_restore("AMB-B003")
    print(f"✓  Forced restore on AMB-B003 — reachable: {mgr.is_unit_reachable('AMB-B003')}")
    mgr2 = CommunicationsManager(seed=42, task_id=9)
    mgr2.reset(
        unit_ids        = unit_ids,
        active_zone_ids = active_zones,
        task_id         = 9,
        weather         = "storm",
    )
    jam_id = mgr2.spawn_jamming_event(["Z05", "Z06"], duration_steps=4,
                                      cause="task9_surge")
    print(f"\n✓  Task 9 jamming event spawned: {jam_id}")
    obs3 = mgr2.get_comms_observation()
    print(f"   Jammed zones: {obs3['jammed_zones']}")
    z05_rssi = mgr2._compute_rssi("Z05", "storm")
    print(f"   Z05 RSSI under jamming+storm: {z05_rssi:.3f}")
    print(f"\n  Final unit comm-state table:")
    mgr.print_status_table()
    analytics = mgr.get_episode_analytics()
    print(f"\n  Episode analytics:")
    for k, v in analytics.items():
        print(f"   {k}: {v}")
    full = mgr.get_full_comm_state()
    print(f"\n  Full state keys: {list(full.keys())}")
    print(f"  Channel stats last 3 steps: {full['channel_stats_last5'][-3:]}")
    print("\n✅  CommunicationsManager smoke-test PASSED")