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
DIVERSION_ER_THRESHOLD:      float = 0.90   
DIVERSION_CLEAR_THRESHOLD:   float = 0.80   
DIVERSION_PENALTY:           float = -0.30  
DIVERSION_REROUTE_DELAY:     float = 5.0    
ICU_TRIGGER_FRACTION:        float = 0.85   
SYSTEM_OVERLOAD_DIVERTED_N:  int   = 4      
SURGE_CAPACITY_FACTOR:       float = 1.25   
BED_STAY_PARAMS: Dict[str, Tuple[float, float]] = {
    "icu":      (480.0, 120.0),    
    "hdu":      (300.0,  60.0),    
    "general":  (180.0,  45.0),    
    "burns":    (600.0, 150.0),    
    "neonatal": (360.0,  90.0),    
    "psych":    (240.0,  60.0),    
}
STAFF_SURGE_ONSET_STEPS:  int   = 24       
STAFF_THROUGHPUT_PENALTY: float = 0.15    
PROTOCOL_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "STEMI":            {"specialty": "cardiology", "resource": "cath_lab",     "window_min": 90},
    "CARDIAC_ARREST":   {"specialty": "resuscitation", "resource": None,        "window_min": 15},
    "ISCHAEMIC_STROKE": {"specialty": "stroke_unit",   "resource": "ct_scanner","window_min": 270},
    "HAEMORRHAGIC_STROKE":{"specialty":"neurosurgery", "resource": "ct_scanner","window_min": 60},
    "POLYTRAUMA":       {"specialty": "trauma",        "resource": None,        "window_min": 60},
    "HEAD_INJURY_SEVERE":{"specialty":"neurosurgery",  "resource": "ct_scanner","window_min": 120},
    "MAJOR_BURNS":      {"specialty": "burns",         "resource": None,        "window_min": 60},
    "MASSIVE_PE":       {"specialty": "cardiology",    "resource": "ct_scanner","window_min": 60},
    "ECLAMPSIA":        {"specialty": "obstetrics",    "resource": None,        "window_min": 45},
    "PPH":              {"specialty": "obstetrics",    "resource": "blood_bank","window_min": 60},
    "NEAR_DROWNING":    {"specialty": "paediatrics",   "resource": None,        "window_min": 30},
    "SEPTIC_SHOCK":     {"specialty": "icu",           "resource": None,        "window_min": 60},
    "ORGANOPHOSPHATE_POISONING": {"specialty": "toxicology", "resource": None,  "window_min": 30},
}
BLOOD_GROUP_PREVALENCE: Dict[str, float] = {
    "O+": 0.36, "A+": 0.28, "B+": 0.22, "AB+": 0.06,
    "O-": 0.02, "A-": 0.02, "B-": 0.02, "AB-": 0.02,
}
BLOOD_UNITS_RESTOCK_STEPS: int = 40   
class HospitalTier(str, Enum):
    TERTIARY      = "tertiary"       
    SECONDARY     = "secondary"      
    PRIMARY       = "primary"        
    OVERFLOW      = "overflow"       
class DiversionStatus(str, Enum):
    OPEN          = "open"
    DIVERTED      = "diverted"
    CONDITIONAL   = "conditional"    
    SURGE_OPEN    = "surge_open"     
class TransferStatus(str, Enum):
    PENDING       = "pending"
    ACCEPTED      = "accepted"
    EN_ROUTE      = "en_route"
    ARRIVED       = "arrived"
    REJECTED      = "rejected"
    CANCELLED     = "cancelled"
class AdmissionType(str, Enum):
    EMERGENCY     = "emergency"
    ICU_DIRECT    = "icu_direct"
    TRANSFER_IN   = "transfer_in"
    WALK_IN       = "walk_in"
@dataclass
class BedBlock:
    bed_id:         str
    patient_id:     str
    bed_type:       str         
    admitted_at_min: float
    expected_discharge_min: float
    priority:       str         
    condition_code: str
    is_transfer:    bool = False
    @property
    def is_ready_for_discharge(self) -> bool:
        return False  
@dataclass
class ResourceSlot:
    slot_id:       str
    resource_type: str           
    total_slots:   int
    in_use:        int = 0
    queue_length:  int = 0
    mean_wait_min: float = 0.0
    busy_until_min: Dict[int, float] = field(default_factory=dict)  
    @property
    def available_slots(self) -> int:
        return max(0, self.total_slots - self.in_use)
    @property
    def utilisation_pct(self) -> float:
        return 100.0 * self.in_use / max(self.total_slots, 1)
    def book(self, duration_min: float, sim_time_min: float) -> Tuple[bool, float]:
        for idx in range(self.total_slots):
            busy_until = self.busy_until_min.get(idx, 0.0)
            if busy_until <= sim_time_min:
                self.busy_until_min[idx] = sim_time_min + duration_min
                self.in_use = sum(
                    1 for bt in self.busy_until_min.values()
                    if bt > sim_time_min
                )
                return True, 0.0
        self.queue_length += 1
        min_wait = min(
            bt - sim_time_min
            for bt in self.busy_until_min.values()
        )
        self.mean_wait_min = max(self.mean_wait_min, min_wait)
        return False, min_wait
    def tick(self, sim_time_min: float) -> None:
        self.in_use = sum(
            1 for bt in self.busy_until_min.values()
            if bt > sim_time_min
        )
        self.queue_length = max(0, self.queue_length - 1) if self.in_use < self.total_slots else self.queue_length
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type":   self.resource_type,
            "total_slots":     self.total_slots,
            "in_use":          self.in_use,
            "available":       self.available_slots,
            "queue_length":    self.queue_length,
            "utilisation_pct": round(self.utilisation_pct, 1),
        }
@dataclass
class BloodBank:
    hospital_id:    str
    inventory:      Dict[str, int] = field(default_factory=dict)
    critical_below: int = 4        
    last_restock_step: int = 0
    def __post_init__(self) -> None:
        if not self.inventory:
            self.inventory = {
                bg: max(4, round(BLOOD_GROUP_PREVALENCE[bg] * 80))
                for bg in BLOOD_GROUP_PREVALENCE
            }
    @property
    def total_units(self) -> int:
        return sum(self.inventory.values())
    @property
    def critical_groups(self) -> List[str]:
        return [bg for bg, n in self.inventory.items() if n < self.critical_below]
    def consume(self, blood_group: str, units: int = 2) -> Tuple[bool, int]:
        available = self.inventory.get(blood_group, 0)
        consumed  = min(available, units)
        self.inventory[blood_group] = available - consumed
        return consumed == units, consumed
    def restock(self, rng: random.Random) -> None:
        for bg in self.inventory:
            addition = max(0, round(rng.gauss(
                BLOOD_GROUP_PREVALENCE[bg] * 40,
                BLOOD_GROUP_PREVALENCE[bg] * 10,
            )))
            self.inventory[bg] += addition
    def is_compatible_emergency(self, blood_group: str) -> bool:
        if self.inventory.get(blood_group, 0) >= 2:
            return True
        return self.inventory.get("O-", 0) >= 2
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hospital_id":    self.hospital_id,
            "total_units":    self.total_units,
            "inventory":      dict(self.inventory),
            "critical_groups":self.critical_groups,
        }
@dataclass
class AdmissionQueueEntry:
    patient_id:     str
    incident_id:    str
    priority:       str
    condition_code: str
    bed_type_needed: str
    admitted_at_min: float      
    waiting_since_min: float
    unit_id:        str         
@dataclass
class InterHospitalTransfer:
    transfer_id:         str
    patient_id:          str
    incident_id:         str
    from_hospital_id:    str
    to_hospital_id:      str
    condition_code:      str
    priority:            str
    specialty_needed:    str
    requested_at_min:    float
    accepted_at_min:     Optional[float] = None
    departed_at_min:     Optional[float] = None
    arrived_at_min:      Optional[float] = None
    status:              TransferStatus = TransferStatus.PENDING
    transport_unit_id:   Optional[str]  = None
    transfer_eta_min:    float = 0.0
    rejection_reason:    Optional[str]  = None
    was_appropriate:     bool  = True    
    time_critical:       bool  = False   
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfer_id":       self.transfer_id,
            "patient_id":        self.patient_id,
            "incident_id":       self.incident_id,
            "from_hospital_id":  self.from_hospital_id,
            "to_hospital_id":    self.to_hospital_id,
            "condition_code":    self.condition_code,
            "priority":          self.priority,
            "specialty_needed":  self.specialty_needed,
            "status":            self.status.value,
            "requested_at_min":  round(self.requested_at_min, 1),
            "accepted_at_min":   round(self.accepted_at_min, 1) if self.accepted_at_min else None,
            "arrived_at_min":    round(self.arrived_at_min,  1) if self.arrived_at_min  else None,
            "transfer_eta_min":  round(self.transfer_eta_min, 1),
            "was_appropriate":   self.was_appropriate,
            "time_critical":     self.time_critical,
        }
@dataclass
class Hospital:
    hospital_id:     str
    name:            str
    zone_id:         str
    lat:             float
    lon:             float
    tier:            HospitalTier
    specialties:     Dict[str, float]   
    capacity_icu:      int
    capacity_hdu:      int
    capacity_general:  int
    capacity_burns:    int
    capacity_neonatal: int
    capacity_psych:    int
    occupied_icu:      int = 0
    occupied_hdu:      int = 0
    occupied_general:  int = 0
    occupied_burns:    int = 0
    occupied_neonatal: int = 0
    occupied_psych:    int = 0
    diversion_status:  DiversionStatus = DiversionStatus.OPEN
    diversion_since_step: int = 0
    conditional_min_priority: str = "P1"  
    surge_active:      bool = False
    surge_activated_at_step: int = 0
    continuous_surge_steps: int = 0     
    staff_throughput_mult:  float = 1.0  
    resources:         Dict[str, ResourceSlot] = field(default_factory=dict)
    blood_bank:        Optional[BloodBank] = None
    has_helipad:       bool = False
    trauma_level:      int  = 2          
    bed_blocks:        List[BedBlock]             = field(default_factory=list)
    admission_queue:   List[AdmissionQueueEntry]  = field(default_factory=list)
    active_transfers_in:  List[str]               = field(default_factory=list)
    active_transfers_out: List[str]               = field(default_factory=list)
    trauma_activated:  bool  = False
    trauma_alert_level: int  = 0   
    total_admissions:  int   = 0
    total_transfers_in: int  = 0
    total_transfers_out: int = 0
    total_diverted:    int   = 0
    peak_er_load:      float = 0.0
    step_admission_log: List[Dict] = field(default_factory=list)
    step_reward_delta: float = 0.0
    @property
    def er_load(self) -> float:
        cap = max(self.capacity_general + self.capacity_hdu, 1)
        occ = self.occupied_general + self.occupied_hdu
        return min(1.0, occ / cap)
    @property
    def icu_load(self) -> float:
        cap = max(self.capacity_icu, 1)
        return min(1.0, self.occupied_icu / cap)
    @property
    def is_diverted(self) -> bool:
        return self.diversion_status == DiversionStatus.DIVERTED
    @property
    def available_icu_beds(self) -> int:
        cap = self.capacity_icu
        if self.surge_active:
            cap = int(cap * SURGE_CAPACITY_FACTOR)
        return max(0, cap - self.occupied_icu)
    @property
    def available_general_beds(self) -> int:
        cap = self.capacity_general
        if self.surge_active:
            cap = int(cap * SURGE_CAPACITY_FACTOR)
        return max(0, cap - self.occupied_general)
    @property
    def available_hdu_beds(self) -> int:
        cap = self.capacity_hdu
        if self.surge_active:
            cap = int(cap * SURGE_CAPACITY_FACTOR)
        return max(0, cap - self.occupied_hdu)
    @property
    def available_burns_beds(self) -> int:
        return max(0, self.capacity_burns - self.occupied_burns)
    @property
    def available_neonatal_beds(self) -> int:
        return max(0, self.capacity_neonatal - self.occupied_neonatal)
    @property
    def available_psych_beds(self) -> int:
        return max(0, self.capacity_psych - self.occupied_psych)
    def has_specialty(self, specialty: str) -> bool:
        return self.specialties.get(specialty, 0.0) > 0.5
    def specialty_proficiency(self, specialty: str) -> float:
        return self.specialties.get(specialty, 0.0)
    def can_accept(self, condition_code: str, priority: str = "P2") -> Tuple[bool, str]:
        if self.diversion_status == DiversionStatus.DIVERTED:
            return False, "hospital_diverted"
        if self.diversion_status == DiversionStatus.CONDITIONAL:
            prio_order = {"P1": 1, "P0": 0, "P2": 2, "P3": 3}
            if prio_order.get(priority, 9) > prio_order.get(self.conditional_min_priority, 1):
                return False, "conditional_priority_restriction"
        req = PROTOCOL_REQUIREMENTS.get(condition_code, {})
        specialty_needed = req.get("specialty")
        if specialty_needed and not self.has_specialty(specialty_needed):
            return False, f"missing_specialty:{specialty_needed}"
        resource_needed = req.get("resource")
        if resource_needed:
            slot = self.resources.get(resource_needed)
            if slot and slot.available_slots == 0 and slot.queue_length > 3:
                return False, f"resource_overloaded:{resource_needed}"
        if self.available_general_beds <= 0 and self.available_hdu_beds <= 0:
            return False, "no_beds_available"
        return True, "ok"
    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "hospital_id":         self.hospital_id,
            "name":                self.name,
            "zone_id":             self.zone_id,
            "tier":                self.tier.value,
            "diversion_status":    self.diversion_status.value,
            "surge_active":        self.surge_active,
            "er_load_pct":         round(self.er_load * 100, 1),
            "icu_load_pct":        round(self.icu_load * 100, 1),
            "available_icu":       self.available_icu_beds,
            "available_hdu":       self.available_hdu_beds,
            "available_general":   self.available_general_beds,
            "available_burns":     self.available_burns_beds,
            "available_neonatal":  self.available_neonatal_beds,
            "specialties":         list(self.specialties.keys()),
            "has_helipad":         self.has_helipad,
            "trauma_level":        self.trauma_level,
            "trauma_activated":    self.trauma_activated,
            "resources":           {k: v.to_dict() for k, v in self.resources.items()},
            "blood_bank_critical": (
                self.blood_bank.critical_groups if self.blood_bank else []
            ),
            "queue_length":        len(self.admission_queue),
            "staff_throughput_mult": round(self.staff_throughput_mult, 3),
        }
    def to_full_state_dict(self) -> Dict[str, Any]:
        base = self.to_observation_dict()
        base.update({
            "total_admissions":  self.total_admissions,
            "total_transfers_in": self.total_transfers_in,
            "total_transfers_out":self.total_transfers_out,
            "total_diverted":    self.total_diverted,
            "peak_er_load":      round(self.peak_er_load * 100, 1),
            "blood_bank":        self.blood_bank.to_dict() if self.blood_bank else None,
            "occupied_beds": {
                "icu":     self.occupied_icu,
                "hdu":     self.occupied_hdu,
                "general": self.occupied_general,
                "burns":   self.occupied_burns,
                "neonatal":self.occupied_neonatal,
                "psych":   self.occupied_psych,
            },
            "capacity_beds": {
                "icu":     self.capacity_icu,
                "hdu":     self.capacity_hdu,
                "general": self.capacity_general,
                "burns":   self.capacity_burns,
                "neonatal":self.capacity_neonatal,
                "psych":   self.capacity_psych,
            },
            "active_transfers_in":  len(self.active_transfers_in),
            "active_transfers_out": len(self.active_transfers_out),
            "specialty_proficiencies": dict(self.specialties),
        })
        return base
class HospitalNetwork:
    def __init__(
        self,
        seed: int = 42,
        traffic_model: Optional[Any] = None,
    ) -> None:
        self.seed          = seed
        self.rng           = random.Random(seed)
        self.np_rng        = np.random.RandomState(seed)
        self._traffic      = traffic_model
        self.hospitals:    Dict[str, Hospital] = {}
        self.hospital_order: List[str]         = []
        self.transfers:    Dict[str, InterHospitalTransfer] = {}
        self._xfer_seq:    int = 0
        self.overflow_hospitals_active: bool = False
        self.overflow_pool:  List[Hospital]  = []
        self.sim_time_min: float = 480.0
        self.step_count:   int   = 0
        self.task_id:      int   = 1
        self._diversion_events: List[Dict]  = []
        self._admission_log:    List[Dict]  = []
        self._step_rewards:     List[float] = []
        self._specialty_admission_counts: Dict[str, int] = defaultdict(int)
        self._load_hospitals()
    def reset(
        self,
        task_id:          int   = 1,
        sim_time_minutes: float = 480.0,
        traffic_model:    Optional[Any] = None,
    ) -> Dict[str, Any]:
        if traffic_model:
            self._traffic = traffic_model
        self.task_id       = task_id
        self.sim_time_min  = sim_time_minutes
        self.step_count    = 0
        self.transfers     = {}
        self._xfer_seq     = 0
        self._diversion_events.clear()
        self._admission_log.clear()
        self._step_rewards.clear()
        self._specialty_admission_counts.clear()
        self.overflow_hospitals_active = False
        self.overflow_pool.clear()
        self.rng    = random.Random(self.seed + task_id * 13)
        self.np_rng = np.random.RandomState(self.seed + task_id * 13)
        self._load_hospitals()
        return self.get_network_observation()
    def step(self, sim_time_minutes: float) -> Tuple[Dict[str, Any], float]:
        self.sim_time_min = sim_time_minutes
        self.step_count  += 1
        step_reward = 0.0
        for hid in self.hospital_order:
            hosp = self.hospitals[hid]
            hosp.step_reward_delta = 0.0
            self._tick_bed_discharges(hosp)
            self._tick_admission_queue(hosp)
            self._tick_diversion_protocol(hosp)
            self._tick_staff_fatigue(hosp)
            self._tick_resources(hosp)
            self._tick_blood_bank(hosp)
            step_reward += hosp.step_reward_delta
        for hosp in self.overflow_pool:
            self._tick_bed_discharges(hosp)
            self._tick_diversion_protocol(hosp)
        step_reward += self._tick_transfers()
        step_reward += self._check_system_overload()
        self._step_rewards.append(step_reward)
        return self.get_network_observation(), step_reward
    def admit_patient(
        self,
        hospital_id:    str,
        incident_id:    str,
        priority:       str,
        condition_code: str,
        unit_id:        str,
        admission_type: AdmissionType = AdmissionType.EMERGENCY,
    ) -> Tuple[bool, str, float]:
        hosp = self.hospitals.get(hospital_id)
        if hosp is None:
            hosp = next((h for h in self.overflow_pool if h.hospital_id == hospital_id), None)
        if hosp is None:
            return False, "unknown_hospital", 0.0
        if hosp.is_diverted:
            hosp.step_reward_delta += DIVERSION_PENALTY
            hosp.total_diverted    += 1
            self._diversion_events.append({
                "step":        self.step_count,
                "hospital_id": hospital_id,
                "incident_id": incident_id,
                "priority":    priority,
            })
            return False, "hospital_diverted", DIVERSION_REROUTE_DELAY
        req = PROTOCOL_REQUIREMENTS.get(condition_code, {})
        specialty_needed = req.get("specialty")
        protocol_ok = True
        if specialty_needed and not hosp.has_specialty(specialty_needed):
            hosp.step_reward_delta -= 0.15
            protocol_ok = False
        bed_type = self._priority_to_bed_type(priority, condition_code)
        avail = self._available_bed_count(hosp, bed_type)
        if avail <= 0:
            if bed_type == "icu" and hosp.available_hdu_beds > 0:
                bed_type = "hdu"
                hosp.step_reward_delta -= 0.05  
            elif hosp.available_general_beds > 0:
                bed_type = "general"
                hosp.step_reward_delta -= 0.05
            else:
                entry = AdmissionQueueEntry(
                    patient_id       = f"PAT-{incident_id}",
                    incident_id      = incident_id,
                    priority         = priority,
                    condition_code   = condition_code,
                    bed_type_needed  = bed_type,
                    admitted_at_min  = self.sim_time_min,
                    waiting_since_min= self.sim_time_min,
                    unit_id          = unit_id,
                )
                hosp.admission_queue.append(entry)
                return False, "queued_no_bed", 0.0
        resource_needed = req.get("resource")
        wait_time = 0.0
        if resource_needed:
            slot = hosp.resources.get(resource_needed)
            if slot:
                duration = self._resource_duration(resource_needed, condition_code)
                booked, wait_time = slot.book(duration, self.sim_time_min)
                if not booked:
                    hosp.step_reward_delta -= 0.05
                    wait_time = max(wait_time, 0.0)
        mean_stay, sigma_stay = BED_STAY_PARAMS.get(bed_type, (180.0, 45.0))
        stay = float(self.np_rng.normal(mean_stay, sigma_stay))
        stay = max(mean_stay * 0.3, stay)
        stay /= hosp.staff_throughput_mult     
        block = BedBlock(
            bed_id               = f"BED-{uuid.uuid4().hex[:6].upper()}",
            patient_id           = f"PAT-{incident_id}",
            bed_type             = bed_type,
            admitted_at_min      = self.sim_time_min,
            expected_discharge_min = self.sim_time_min + stay,
            priority             = priority,
            condition_code       = condition_code,
            is_transfer          = (admission_type == AdmissionType.TRANSFER_IN),
        )
        hosp.bed_blocks.append(block)
        self._increment_occupancy(hosp, bed_type, 1)
        hosp.total_admissions += 1
        if admission_type == AdmissionType.TRANSFER_IN:
            hosp.total_transfers_in += 1
        if specialty_needed:
            self._specialty_admission_counts[specialty_needed] += 1
        if protocol_ok:
            hosp.step_reward_delta += 0.08
        window = req.get("window_min")
        if window:
            hosp.step_reward_delta += 0.05
        hosp.peak_er_load = max(hosp.peak_er_load, hosp.er_load)
        self._admission_log.append({
            "step":          self.step_count,
            "hospital_id":   hospital_id,
            "incident_id":   incident_id,
            "priority":      priority,
            "condition_code":condition_code,
            "bed_type":      bed_type,
            "protocol_ok":   protocol_ok,
            "wait_time":     round(wait_time, 1),
        })
        handover_delay = self._compute_handover_delay(hosp, priority)
        return True, "admitted", handover_delay + wait_time
    def request_transfer(
        self,
        from_hospital_id: str,
        to_hospital_id:   str,
        incident_id:      str,
        condition_code:   str,
        priority:         str,
        specialty_needed: str,
        transport_unit_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[InterHospitalTransfer]]:
        from_hosp = self.hospitals.get(from_hospital_id)
        to_hosp   = self.hospitals.get(to_hospital_id)
        if from_hosp is None or to_hosp is None:
            return False, "unknown_hospital", None
        ok, reason = to_hosp.can_accept(condition_code, priority)
        if not ok:
            return False, reason, None
        if specialty_needed and not to_hosp.has_specialty(specialty_needed):
            return False, f"destination_lacks_{specialty_needed}", None
        eta = self._compute_transfer_eta(from_hospital_id, to_hospital_id)
        appropriate = to_hosp.specialty_proficiency(specialty_needed) > 0.7
        xfer_id = f"XFR-{self._xfer_seq:04d}"
        self._xfer_seq += 1
        xfer = InterHospitalTransfer(
            transfer_id         = xfer_id,
            patient_id          = f"PAT-{incident_id}",
            incident_id         = incident_id,
            from_hospital_id    = from_hospital_id,
            to_hospital_id      = to_hospital_id,
            condition_code      = condition_code,
            priority            = priority,
            specialty_needed    = specialty_needed,
            requested_at_min    = self.sim_time_min,
            accepted_at_min     = self.sim_time_min,
            status              = TransferStatus.ACCEPTED,
            transport_unit_id   = transport_unit_id,
            transfer_eta_min    = eta,
            was_appropriate     = appropriate,
            time_critical       = (priority == "P1"),
        )
        self.transfers[xfer_id] = xfer
        from_hosp.active_transfers_out.append(xfer_id)
        to_hosp.active_transfers_in.append(xfer_id)
        from_hosp.total_transfers_out += 1
        return True, "accepted", xfer
    def mark_transfer_departed(self, transfer_id: str) -> bool:
        xfer = self.transfers.get(transfer_id)
        if xfer is None or xfer.status != TransferStatus.ACCEPTED:
            return False
        xfer.status           = TransferStatus.EN_ROUTE
        xfer.departed_at_min  = self.sim_time_min
        return True
    def mark_transfer_arrived(self, transfer_id: str) -> Tuple[bool, str]:
        xfer = self.transfers.get(transfer_id)
        if xfer is None:
            return False, "unknown_transfer"
        if xfer.status not in (TransferStatus.ACCEPTED, TransferStatus.EN_ROUTE):
            return False, f"invalid_status:{xfer.status.value}"
        xfer.status         = TransferStatus.ARRIVED
        xfer.arrived_at_min = self.sim_time_min
        admitted, reason, _ = self.admit_patient(
            hospital_id    = xfer.to_hospital_id,
            incident_id    = xfer.incident_id,
            priority       = xfer.priority,
            condition_code = xfer.condition_code,
            unit_id        = xfer.transport_unit_id or "TRANSFER",
            admission_type = AdmissionType.TRANSFER_IN,
        )
        return admitted, reason
    def cancel_transfer(self, transfer_id: str, reason: str = "cancelled") -> bool:
        xfer = self.transfers.get(transfer_id)
        if xfer is None:
            return False
        xfer.status           = TransferStatus.CANCELLED
        xfer.rejection_reason = reason
        hosp_from = self.hospitals.get(xfer.from_hospital_id)
        hosp_to   = self.hospitals.get(xfer.to_hospital_id)
        if hosp_from and transfer_id in hosp_from.active_transfers_out:
            hosp_from.active_transfers_out.remove(transfer_id)
        if hosp_to and transfer_id in hosp_to.active_transfers_in:
            hosp_to.active_transfers_in.remove(transfer_id)
        return True
    def activate_surge(self, hospital_id: str) -> Tuple[bool, str]:
        hosp = self.hospitals.get(hospital_id)
        if hosp is None:
            return False, "unknown_hospital"
        if hosp.surge_active:
            return False, "surge_already_active"
        hosp.surge_active            = True
        hosp.surge_activated_at_step = self.step_count
        hosp.diversion_status        = DiversionStatus.SURGE_OPEN
        return True, f"surge_capacity_active_at_{hospital_id}"
    def deactivate_surge(self, hospital_id: str) -> bool:
        hosp = self.hospitals.get(hospital_id)
        if hosp is None:
            return False
        hosp.surge_active     = False
        hosp.diversion_status = DiversionStatus.OPEN
        return True
    def activate_overflow_hospitals(self) -> List[str]:
        if self.overflow_hospitals_active:
            return [h.hospital_id for h in self.overflow_pool]
        overflow_configs = [
            {
                "hospital_id": "H_OVF1", "name": "Overflow Centre Alpha",
                "zone_id": "Z04", "lat": 18.55, "lon": 73.80,
                "tier": HospitalTier.OVERFLOW,
                "specialties": {"emergency": 0.7, "general_surgery": 0.6},
                "capacity_icu": 5, "capacity_hdu": 10, "capacity_general": 30,
                "capacity_burns": 0, "capacity_neonatal": 0, "capacity_psych": 0,
            },
            {
                "hospital_id": "H_OVF2", "name": "Overflow Centre Beta",
                "zone_id": "Z07", "lat": 18.48, "lon": 73.90,
                "tier": HospitalTier.OVERFLOW,
                "specialties": {"emergency": 0.7, "orthopaedics": 0.6},
                "capacity_icu": 4, "capacity_hdu": 8, "capacity_general": 25,
                "capacity_burns": 0, "capacity_neonatal": 0, "capacity_psych": 0,
            },
            {
                "hospital_id": "H_OVF3", "name": "Field Hospital Gamma",
                "zone_id": "Z11", "lat": 18.60, "lon": 73.95,
                "tier": HospitalTier.OVERFLOW,
                "specialties": {"emergency": 0.6},
                "capacity_icu": 2, "capacity_hdu": 5, "capacity_general": 20,
                "capacity_burns": 0, "capacity_neonatal": 0, "capacity_psych": 0,
            },
        ]
        new_ids = []
        for cfg in overflow_configs:
            hosp = self._build_hospital_from_dict(cfg)
            self.overflow_pool.append(hosp)
            new_ids.append(hosp.hospital_id)
        self.overflow_hospitals_active = True
        return new_ids
    def issue_pre_alert(
        self,
        hospital_id:  str,
        condition_code: str,
        priority:     str,
        eta_minutes:  float,
    ) -> Tuple[bool, str]:
        hosp = self.hospitals.get(hospital_id)
        if hosp is None:
            return False, "unknown_hospital"
        if hosp.is_diverted:
            return False, "hospital_diverted"
        req = PROTOCOL_REQUIREMENTS.get(condition_code, {})
        if req.get("specialty") in ("trauma", "neurosurgery"):
            if not hosp.trauma_activated:
                hosp.trauma_activated  = True
                hosp.trauma_alert_level = 2 if priority == "P1" else 1
        resource = req.get("resource")
        if resource and resource in hosp.resources:
            duration = self._resource_duration(resource, condition_code)
            hosp.resources[resource].book(duration, self.sim_time_min + eta_minutes)
        return True, f"pre_alert_issued:{condition_code}@{hospital_id}"
    def get_best_hospital(
        self,
        from_zone_id:   str,
        condition_code: str,
        priority:       str = "P2",
        exclude_ids:    Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float, float]:
        excluded = set(exclude_ids or [])
        req      = PROTOCOL_REQUIREMENTS.get(condition_code, {})
        specialty_needed = req.get("specialty")
        best_id    = None
        best_score = -1.0
        best_eta   = float("inf")
        all_hospitals = list(self.hospitals.values())
        if self.overflow_hospitals_active:
            all_hospitals += self.overflow_pool
        for hosp in all_hospitals:
            if hosp.hospital_id in excluded:
                continue
            ok, _ = hosp.can_accept(condition_code, priority)
            if not ok:
                continue
            sp_score = hosp.specialty_proficiency(specialty_needed) if specialty_needed else 0.6
            total_cap = max(
                hosp.capacity_general + hosp.capacity_hdu + hosp.capacity_icu, 1
            )
            avail = (
                hosp.available_general_beds +
                hosp.available_hdu_beds +
                hosp.available_icu_beds
            )
            cap_score = avail / total_cap
            eta = self._compute_eta_to_hospital(from_zone_id, hosp.hospital_id)
            eta_score = 1.0 / (1.0 + eta / 30.0)
            composite = sp_score * 0.5 + cap_score * 0.3 + eta_score * 0.2
            if composite > best_score:
                best_score = composite
                best_id    = hosp.hospital_id
                best_eta   = eta
        return best_id, best_eta, max(0.0, best_score)
    def get_hospitals_by_specialty(
        self,
        specialty: str,
        min_proficiency: float = 0.6,
        open_only: bool = True,
    ) -> List[Hospital]:
        result = [
            h for h in self.hospitals.values()
            if h.specialty_proficiency(specialty) >= min_proficiency
            and (not open_only or not h.is_diverted)
        ]
        result.sort(key=lambda h: -h.specialty_proficiency(specialty))
        return result
    def get_diversion_status_all(self) -> Dict[str, str]:
        return {
            hid: h.diversion_status.value
            for hid, h in self.hospitals.items()
        }
    def count_diverted(self) -> int:
        return sum(1 for h in self.hospitals.values() if h.is_diverted)
    def is_system_in_overload(self) -> bool:
        return self.count_diverted() >= SYSTEM_OVERLOAD_DIVERTED_N
    def get_network_observation(self) -> Dict[str, Any]:
        hospitals_obs = [h.to_observation_dict() for h in self._iter_hospitals()]
        if self.overflow_hospitals_active:
            hospitals_obs += [h.to_observation_dict() for h in self.overflow_pool]
        diverted_count = self.count_diverted()
        return {
            "hospital_count":       len(self.hospitals),
            "diverted_count":       diverted_count,
            "system_overload":      self.is_system_in_overload(),
            "overflow_active":      self.overflow_hospitals_active,
            "active_transfers":     sum(
                1 for x in self.transfers.values()
                if x.status in (TransferStatus.ACCEPTED, TransferStatus.EN_ROUTE)
            ),
            "hospitals":            hospitals_obs,
            "specialty_scarcity":   self._compute_specialty_scarcity(),
            "sim_time_min":         self.sim_time_min,
            "step_count":           self.step_count,
        }
    def get_hospital_observation(self, hospital_id: str) -> Optional[Dict[str, Any]]:
        hosp = self.hospitals.get(hospital_id)
        return hosp.to_observation_dict() if hosp else None
    def get_hospital_full_state(self, hospital_id: str) -> Optional[Dict[str, Any]]:
        hosp = self.hospitals.get(hospital_id)
        return hosp.to_full_state_dict() if hosp else None
    def get_all_transfers(self) -> List[Dict[str, Any]]:
        return [x.to_dict() for x in self.transfers.values()]
    def get_active_transfers(self) -> List[Dict[str, Any]]:
        return [
            x.to_dict() for x in self.transfers.values()
            if x.status in (TransferStatus.ACCEPTED, TransferStatus.EN_ROUTE)
        ]
    def get_episode_analytics(self) -> Dict[str, Any]:
        total_admit  = sum(h.total_admissions     for h in self._iter_hospitals())
        total_xfr_ok = sum(
            1 for x in self.transfers.values()
            if x.status == TransferStatus.ARRIVED
        )
        total_xfr    = len(self.transfers)
        xfr_success  = total_xfr_ok / max(total_xfr, 1)
        appropriate_xfrs = sum(
            1 for x in self.transfers.values()
            if x.status == TransferStatus.ARRIVED and x.was_appropriate
        )
        appropriate_rate = appropriate_xfrs / max(total_xfr_ok, 1)
        mean_er = sum(h.er_load for h in self._iter_hospitals()) / max(len(self.hospitals), 1)
        peak_er = max((h.peak_er_load for h in self._iter_hospitals()), default=0.0)
        return {
            "total_admissions":        total_admit,
            "total_diverted":          sum(h.total_diverted for h in self._iter_hospitals()),
            "total_transfers":         total_xfr,
            "transfer_success_rate":   round(xfr_success,     3),
            "transfer_appropriate_rate":round(appropriate_rate,3),
            "diversion_events":        len(self._diversion_events),
            "mean_er_load_pct":        round(mean_er * 100, 1),
            "peak_er_load_pct":        round(peak_er * 100, 1),
            "system_overload_count":   sum(
                1 for h in self._iter_hospitals()
                if h.er_load >= DIVERSION_ER_THRESHOLD
            ),
            "specialty_utilisation":   dict(self._specialty_admission_counts),
            "specialty_scarcity":      self._compute_specialty_scarcity(),
        }
    def _tick_bed_discharges(self, hosp: Hospital) -> None:
        remaining: List[BedBlock] = []
        for block in hosp.bed_blocks:
            if self.sim_time_min >= block.expected_discharge_min:
                self._decrement_occupancy(hosp, block.bed_type, 1)
            else:
                remaining.append(block)
        hosp.bed_blocks = remaining
    def _tick_admission_queue(self, hosp: Hospital) -> None:
        if not hosp.admission_queue:
            return
        still_waiting: List[AdmissionQueueEntry] = []
        for entry in sorted(hosp.admission_queue,
                            key=lambda e: {"P1": 0, "P0": 0, "P2": 1, "P3": 2}.get(e.priority, 3)):
            avail = self._available_bed_count(hosp, entry.bed_type_needed)
            if avail > 0:
                mean_stay, sig = BED_STAY_PARAMS.get(entry.bed_type_needed, (180.0, 45.0))
                stay = max(mean_stay * 0.3, float(self.np_rng.normal(mean_stay, sig)))
                block = BedBlock(
                    bed_id               = f"BED-{uuid.uuid4().hex[:6].upper()}",
                    patient_id           = entry.patient_id,
                    bed_type             = entry.bed_type_needed,
                    admitted_at_min      = self.sim_time_min,
                    expected_discharge_min = self.sim_time_min + stay,
                    priority             = entry.priority,
                    condition_code       = entry.condition_code,
                )
                hosp.bed_blocks.append(block)
                self._increment_occupancy(hosp, entry.bed_type_needed, 1)
                hosp.total_admissions += 1
                wait = self.sim_time_min - entry.waiting_since_min
                if wait > 30 and entry.priority == "P1":
                    hosp.step_reward_delta -= 0.10   
            else:
                wait = self.sim_time_min - entry.waiting_since_min
                if entry.priority == "P1" and wait > 20:
                    hosp.step_reward_delta -= 0.05
                still_waiting.append(entry)
        hosp.admission_queue = still_waiting
    def _tick_diversion_protocol(self, hosp: Hospital) -> None:
        er = hosp.er_load
        icu = hosp.icu_load
        if hosp.diversion_status == DiversionStatus.OPEN:
            if er >= DIVERSION_ER_THRESHOLD or icu >= ICU_TRIGGER_FRACTION:
                hosp.diversion_status      = DiversionStatus.DIVERTED
                hosp.diversion_since_step  = self.step_count
                hosp.continuous_surge_steps += 1
        elif hosp.diversion_status == DiversionStatus.DIVERTED:
            if er < DIVERSION_CLEAR_THRESHOLD and icu < 0.75:
                hosp.diversion_status = DiversionStatus.OPEN
            else:
                hosp.continuous_surge_steps += 1
                hosp.step_reward_delta -= 0.02
        elif hosp.diversion_status == DiversionStatus.SURGE_OPEN:
            if er < 0.70:
                hosp.surge_active     = False
                hosp.diversion_status = DiversionStatus.OPEN
        hosp.peak_er_load = max(hosp.peak_er_load, er)
    def _tick_staff_fatigue(self, hosp: Hospital) -> None:
        if hosp.continuous_surge_steps >= STAFF_SURGE_ONSET_STEPS:
            excess = hosp.continuous_surge_steps - STAFF_SURGE_ONSET_STEPS
            degradation = min(STAFF_THROUGHPUT_PENALTY,
                              0.01 * excess / STAFF_SURGE_ONSET_STEPS)
            hosp.staff_throughput_mult = max(
                1.0 - STAFF_THROUGHPUT_PENALTY,
                1.0 - degradation
            )
        else:
            hosp.staff_throughput_mult = min(
                1.0,
                hosp.staff_throughput_mult + 0.01
            )
        if not hosp.is_diverted:
            hosp.continuous_surge_steps = max(0, hosp.continuous_surge_steps - 1)
    def _tick_resources(self, hosp: Hospital) -> None:
        for slot in hosp.resources.values():
            slot.tick(self.sim_time_min)
    def _tick_blood_bank(self, hosp: Hospital) -> None:
        if hosp.blood_bank is None:
            return
        if self.step_count - hosp.blood_bank.last_restock_step >= BLOOD_UNITS_RESTOCK_STEPS:
            hosp.blood_bank.restock(self.rng)
            hosp.blood_bank.last_restock_step = self.step_count
    def _tick_transfers(self) -> float:
        reward = 0.0
        for xfer in self.transfers.values():
            if xfer.status != TransferStatus.EN_ROUTE:
                continue
            if xfer.departed_at_min is None:
                continue
            elapsed = self.sim_time_min - xfer.departed_at_min
            if elapsed >= xfer.transfer_eta_min:
                ok, reason = self.mark_transfer_arrived(xfer.transfer_id)
                if ok:
                    reward += 0.10 if xfer.was_appropriate else 0.02
                    if xfer.time_critical:
                        total_time = self.sim_time_min - xfer.requested_at_min
                        if total_time <= 60:
                            reward += 0.15
                        else:
                            reward -= 0.10
        return reward
    def _check_system_overload(self) -> float:
        n_diverted = self.count_diverted()
        if n_diverted >= SYSTEM_OVERLOAD_DIVERTED_N:
            return -0.20
        return 0.0
    def _load_hospitals(self) -> None:
        self.hospitals    = {}
        self.hospital_order = []
        path = DATA_DIR / "hospital_profiles.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            hospitals_raw = raw.get("hospitals", raw) if isinstance(raw, dict) else raw
            if isinstance(hospitals_raw, dict):
                hospitals_raw = list(hospitals_raw.values())
            for h_raw in hospitals_raw:
                hosp = self._build_hospital_from_dict(h_raw)
                self.hospitals[hosp.hospital_id]  = hosp
                self.hospital_order.append(hosp.hospital_id)
        else:
            self._build_default_hospitals()
    def _build_hospital_from_dict(self, raw: Dict[str, Any]) -> Hospital:
        hid  = raw.get("hospital_id") or raw.get("id", f"H{len(self.hospitals)+1:02d}")
        name = raw.get("name", hid)
        zone = raw.get("zone_id", "Z05")
        lat  = float(raw.get("lat", 18.52))
        lon  = float(raw.get("lon", 73.85))
        tier_str = raw.get("tier_name") or raw.get("tier", "secondary")
        tier_map = {"tertiary": 1, "secondary": 2, "primary": 3, "overflow": 4}
        if isinstance(tier_str, int):
            tier_name = {1:"tertiary",2:"secondary",3:"primary"}.get(tier_str, "secondary")
        else:
            tier_name = str(tier_str).lower()
        tier = {
            "tertiary": HospitalTier.TERTIARY,
            "secondary": HospitalTier.SECONDARY,
            "primary":   HospitalTier.PRIMARY,
            "overflow":  HospitalTier.OVERFLOW,
        }.get(tier_name, HospitalTier.SECONDARY)
        raw_sp = raw.get("specialties", [])
        if isinstance(raw_sp, list):
            specialties = {sp: self._default_proficiency(sp, tier) for sp in raw_sp}
        else:
            specialties = {k: float(v) for k, v in raw_sp.items()}
        specialties.setdefault("emergency", 0.75)
        cap = raw.get("capacity", raw.get("beds", {}))
        if isinstance(cap, int):
            cap = {"general": cap}
        hosp = Hospital(
            hospital_id      = hid,
            name             = name,
            zone_id          = zone,
            lat              = lat,
            lon              = lon,
            tier             = tier,
            specialties      = specialties,
            capacity_icu     = int(cap.get("icu",     raw.get("icu_beds",      self._default_icu(tier)))),
            capacity_hdu     = int(cap.get("hdu",     raw.get("hdu_beds",      self._default_hdu(tier)))),
            capacity_general = int(cap.get("general", raw.get("general_beds",  self._default_gen(tier)))),
            capacity_burns   = int(cap.get("burns",   raw.get("burns_beds",    self._default_burns(tier, specialties)))),
            capacity_neonatal= int(cap.get("neonatal",raw.get("neonatal_beds", self._default_nicu(tier, specialties)))),
            capacity_psych   = int(cap.get("psych",   raw.get("psych_beds",    self._default_psych(tier, specialties)))),
            has_helipad      = bool(raw.get("helipad", False)),
            trauma_level     = int(raw.get("trauma_level", 2 if tier != HospitalTier.TERTIARY else 1)),
        )
        hosp.resources = self._build_resources(raw, tier, specialties)
        if tier in (HospitalTier.TERTIARY, HospitalTier.SECONDARY):
            hosp.blood_bank = BloodBank(hospital_id=hid)
        self._seed_initial_occupancy(hosp)
        return hosp
    def _build_resources(
        self,
        raw:         Dict,
        tier:        HospitalTier,
        specialties: Dict[str, float],
    ) -> Dict[str, ResourceSlot]:
        resources: Dict[str, ResourceSlot] = {}
        has_cath = raw.get("cath_lab", "cardiology" in specialties and specialties["cardiology"] > 0.7)
        if has_cath:
            resources["cath_lab"] = ResourceSlot(
                slot_id      = "cl_01",
                resource_type= "cath_lab",
                total_slots  = 2 if tier == HospitalTier.TERTIARY else 1,
            )
        has_ct = raw.get("ct_scanner", tier in (HospitalTier.TERTIARY, HospitalTier.SECONDARY))
        if has_ct:
            resources["ct_scanner"] = ResourceSlot(
                slot_id      = "ct_01",
                resource_type= "ct_scanner",
                total_slots  = 2 if tier == HospitalTier.TERTIARY else 1,
            )
        has_mri = raw.get("mri", tier == HospitalTier.TERTIARY)
        if has_mri:
            resources["mri"] = ResourceSlot(
                slot_id      = "mri_01",
                resource_type= "mri",
                total_slots  = 1,
            )
        or_count = raw.get("or_suites", {
            HospitalTier.TERTIARY:  4,
            HospitalTier.SECONDARY: 2,
            HospitalTier.PRIMARY:   1,
            HospitalTier.OVERFLOW:  0,
        }.get(tier, 1))
        if or_count > 0:
            resources["or_suite"] = ResourceSlot(
                slot_id      = "or_01",
                resource_type= "or_suite",
                total_slots  = or_count,
            )
        if raw.get("helipad", False):
            resources["helipad"] = ResourceSlot(
                slot_id      = "hp_01",
                resource_type= "helipad",
                total_slots  = 2,
            )
        return resources
    def _seed_initial_occupancy(self, hosp: Hospital) -> None:
        load_frac = self.np_rng.uniform(0.50, 0.75)
        tier_load = {
            HospitalTier.TERTIARY:  0.70,
            HospitalTier.SECONDARY: 0.60,
            HospitalTier.PRIMARY:   0.45,
            HospitalTier.OVERFLOW:  0.20,
        }
        load_frac = self.np_rng.uniform(
            tier_load.get(hosp.tier, 0.55) - 0.10,
            tier_load.get(hosp.tier, 0.55) + 0.10,
        )
        for bed_type, capacity in [
            ("icu",      hosp.capacity_icu),
            ("hdu",      hosp.capacity_hdu),
            ("general",  hosp.capacity_general),
            ("burns",    hosp.capacity_burns),
            ("neonatal", hosp.capacity_neonatal),
            ("psych",    hosp.capacity_psych),
        ]:
            occupied = int(capacity * load_frac)
            self._increment_occupancy(hosp, bed_type, occupied)
            mean_stay, sig = BED_STAY_PARAMS.get(bed_type, (180.0, 45.0))
            for _ in range(occupied):
                discharge_offset = float(self.np_rng.uniform(30, mean_stay + sig))
                hosp.bed_blocks.append(BedBlock(
                    bed_id               = f"BED-{uuid.uuid4().hex[:6].upper()}",
                    patient_id           = f"EXISTING-{uuid.uuid4().hex[:4]}",
                    bed_type             = bed_type,
                    admitted_at_min      = self.sim_time_min - mean_stay / 2,
                    expected_discharge_min = self.sim_time_min + discharge_offset,
                    priority             = "P2",
                    condition_code       = "EXISTING",
                ))
    def _build_default_hospitals(self) -> None:
        defaults = [
            {
                "hospital_id": "H01", "name": "Sassoon General Hospital",
                "zone_id": "Z05", "lat": 18.517, "lon": 73.856,
                "tier": "tertiary",
                "specialties": ["cardiology","neurology","stroke_unit","trauma","orthopaedics",
                                "burns","obstetrics","paediatrics","icu","neurosurgery",
                                "resuscitation","cath_lab","ct_scanner","blood_bank",
                                "general_surgery","emergency","toxicology","nephrology"],
                "capacity": {"icu":40,"hdu":30,"general":200,"burns":20,"neonatal":15,"psych":15},
                "cath_lab": True, "ct_scanner": True, "mri": True, "helipad": True,
                "trauma_level": 1,
            },
            {
                "hospital_id": "H02", "name": "KEM Hospital",
                "zone_id": "Z05", "lat": 18.519, "lon": 73.850,
                "tier": "tertiary",
                "specialties": ["cardiology","neurology","stroke_unit","trauma","orthopaedics",
                                "obstetrics","paediatrics","icu","neurosurgery","resuscitation",
                                "general_surgery","emergency","gastroenterology","pulmonology"],
                "capacity": {"icu":30,"hdu":25,"general":160,"burns":10,"neonatal":12,"psych":10},
                "cath_lab": True, "ct_scanner": True, "mri": True, "helipad": False,
                "trauma_level": 1,
            },
            {
                "hospital_id": "H03", "name": "Jehangir Hospital",
                "zone_id": "Z06", "lat": 18.533, "lon": 73.882,
                "tier": "secondary",
                "specialties": ["cardiology","orthopaedics","emergency","general_surgery",
                                "obstetrics","paediatrics","icu"],
                "capacity": {"icu":20,"hdu":15,"general":100,"burns":0,"neonatal":8,"psych":5},
                "cath_lab": True, "ct_scanner": True, "mri": False, "helipad": False,
                "trauma_level": 2,
            },
            {
                "hospital_id": "H04", "name": "Ruby Hall Clinic",
                "zone_id": "Z03", "lat": 18.530, "lon": 73.880,
                "tier": "secondary",
                "specialties": ["cardiology","stroke_unit","neurology","emergency",
                                "orthopaedics","general_surgery","icu","nephrology"],
                "capacity": {"icu":18,"hdu":12,"general":90,"burns":0,"neonatal":6,"psych":0},
                "cath_lab": True, "ct_scanner": True, "mri": True, "helipad": False,
                "trauma_level": 2,
            },
            {
                "hospital_id": "H05", "name": "Deenanath Mangeshkar Hospital",
                "zone_id": "Z08", "lat": 18.504, "lon": 73.824,
                "tier": "secondary",
                "specialties": ["cardiology","neurosurgery","trauma","emergency",
                                "orthopaedics","obstetrics","paediatrics","icu"],
                "capacity": {"icu":22,"hdu":18,"general":120,"burns":5,"neonatal":10,"psych":5},
                "cath_lab": True, "ct_scanner": True, "mri": False, "helipad": True,
                "trauma_level": 2,
            },
            {
                "hospital_id": "H06", "name": "Poona Hospital",
                "zone_id": "Z09", "lat": 18.516, "lon": 73.845,
                "tier": "secondary",
                "specialties": ["emergency","orthopaedics","general_surgery",
                                "obstetrics","icu","pulmonology"],
                "capacity": {"icu":12,"hdu":10,"general":80,"burns":0,"neonatal":5,"psych":8},
                "cath_lab": False, "ct_scanner": True, "mri": False, "helipad": False,
                "trauma_level": 2,
            },
            {
                "hospital_id": "H07", "name": "Pimpri-Chinchwad Municipal Hospital",
                "zone_id": "Z02", "lat": 18.621, "lon": 73.800,
                "tier": "secondary",
                "specialties": ["emergency","trauma","orthopaedics","general_surgery","icu",
                                "obstetrics","paediatrics","burns"],
                "capacity": {"icu":15,"hdu":12,"general":100,"burns":8,"neonatal":6,"psych":5},
                "cath_lab": False, "ct_scanner": True, "mri": False, "helipad": False,
                "trauma_level": 2,
            },
            {
                "hospital_id": "H08", "name": "Aundh District Hospital",
                "zone_id": "Z04", "lat": 18.558, "lon": 73.809,
                "tier": "primary",
                "specialties": ["emergency","general_surgery","obstetrics","paediatrics"],
                "capacity": {"icu":6,"hdu":5,"general":50,"burns":0,"neonatal":3,"psych":0},
                "cath_lab": False, "ct_scanner": False, "mri": False, "helipad": False,
                "trauma_level": 3,
            },
        ]
        for d in defaults:
            hosp = self._build_hospital_from_dict(d)
            self.hospitals[hosp.hospital_id] = hosp
            self.hospital_order.append(hosp.hospital_id)
    def _iter_hospitals(self):
        for hid in self.hospital_order:
            yield self.hospitals[hid]
    def _increment_occupancy(self, hosp: Hospital, bed_type: str, n: int) -> None:
        attr = f"occupied_{bed_type}"
        cap  = getattr(hosp, f"capacity_{bed_type}", 0)
        current = getattr(hosp, attr, 0)
        surge_cap = int(cap * SURGE_CAPACITY_FACTOR) if hosp.surge_active else cap
        setattr(hosp, attr, min(current + n, surge_cap))
    def _decrement_occupancy(self, hosp: Hospital, bed_type: str, n: int) -> None:
        attr = f"occupied_{bed_type}"
        current = getattr(hosp, attr, 0)
        setattr(hosp, attr, max(0, current - n))
    def _available_bed_count(self, hosp: Hospital, bed_type: str) -> int:
        bed_map = {
            "icu":      hosp.available_icu_beds,
            "hdu":      hosp.available_hdu_beds,
            "general":  hosp.available_general_beds,
            "burns":    hosp.available_burns_beds,
            "neonatal": hosp.available_neonatal_beds,
            "psych":    hosp.available_psych_beds,
        }
        return bed_map.get(bed_type, hosp.available_general_beds)
    @staticmethod
    def _priority_to_bed_type(priority: str, condition_code: str) -> str:
        icu_conditions = {
            "STEMI", "CARDIAC_ARREST", "HAEMORRHAGIC_STROKE", "POLYTRAUMA",
            "SEPTIC_SHOCK", "MAJOR_BURNS", "ECLAMPSIA", "NEAR_DROWNING",
            "ORGANOPHOSPHATE_POISONING", "MASSIVE_PE", "PPH",
        }
        if condition_code in icu_conditions:
            return "icu"
        if condition_code in ("ISCHAEMIC_STROKE", "HEAD_INJURY_SEVERE", "PULMONARY_OEDEMA"):
            return "hdu"
        if priority == "P1":
            return "hdu"
        if priority in ("P2", "P3"):
            return "general"
        return "general"
    @staticmethod
    def _resource_duration(resource_type: str, condition_code: str) -> float:
        durations: Dict[str, float] = {
            "cath_lab":   90.0,
            "ct_scanner": 20.0,
            "mri":        45.0,
            "or_suite":  120.0,
            "helipad":    15.0,
        }
        base = durations.get(resource_type, 30.0)
        if resource_type == "cath_lab" and condition_code == "STEMI":
            base = 75.0
        return base
    def _compute_handover_delay(self, hosp: Hospital, priority: str) -> float:
        base = 8.0
        load_extra = max(0.0, (hosp.er_load - 0.60) * 20.0)  
        base = min(20.0, base + load_extra)
        if priority == "P1" and hosp.trauma_activated:
            base = max(3.0, base - 3.0)  
        return base * (1.0 / hosp.staff_throughput_mult)
    def _compute_eta_to_hospital(self, from_zone: str, hospital_id: str) -> float:
        hosp = self.hospitals.get(hospital_id)
        if hosp is None:
            return 15.0
        if self._traffic is not None:
            try:
                return self._traffic.get_travel_time(from_zone, hosp.zone_id)
            except Exception:
                pass
        return 8.0
    def _compute_transfer_eta(self, from_hid: str, to_hid: str) -> float:
        from_h = self.hospitals.get(from_hid)
        to_h   = self.hospitals.get(to_hid)
        if not from_h or not to_h:
            return 20.0
        if self._traffic is not None:
            try:
                return self._traffic.get_travel_time(from_h.zone_id, to_h.zone_id)
            except Exception:
                pass
        return self._haversine_min(from_h.lat, from_h.lon, to_h.lat, to_h.lon)
    def _compute_specialty_scarcity(self) -> Dict[str, float]:
        specialty_open: Dict[str, int]  = defaultdict(int)
        specialty_total: Dict[str, int] = defaultdict(int)
        for hosp in self._iter_hospitals():
            for sp, prof in hosp.specialties.items():
                if prof > 0.5:
                    specialty_total[sp] += 1
                    if not hosp.is_diverted:
                        specialty_open[sp] += 1
        scarcity: Dict[str, float] = {}
        for sp, total in specialty_total.items():
            if total > 0:
                fraction_open = specialty_open[sp] / total
                scarcity[sp]  = round(1.0 - fraction_open, 3)   
        return scarcity
    def _default_proficiency(self, specialty: str, tier: HospitalTier) -> float:
        tier_mult = {
            HospitalTier.TERTIARY:  1.0,
            HospitalTier.SECONDARY: 0.75,
            HospitalTier.PRIMARY:   0.55,
            HospitalTier.OVERFLOW:  0.60,
        }.get(tier, 0.70)
        return round(
            min(1.0, tier_mult + self.np_rng.uniform(-0.05, 0.05)), 2
        )
    @staticmethod
    def _default_icu(tier: HospitalTier) -> int:
        return {HospitalTier.TERTIARY: 30, HospitalTier.SECONDARY: 15,
                HospitalTier.PRIMARY:  5,  HospitalTier.OVERFLOW:  4}.get(tier, 10)
    @staticmethod
    def _default_hdu(tier: HospitalTier) -> int:
        return {HospitalTier.TERTIARY: 20, HospitalTier.SECONDARY: 10,
                HospitalTier.PRIMARY:  4,  HospitalTier.OVERFLOW:  6}.get(tier, 8)
    @staticmethod
    def _default_gen(tier: HospitalTier) -> int:
        return {HospitalTier.TERTIARY: 150, HospitalTier.SECONDARY: 80,
                HospitalTier.PRIMARY:  40,  HospitalTier.OVERFLOW:  25}.get(tier, 60)
    @staticmethod
    def _default_burns(tier: HospitalTier, sp: Dict) -> int:
        return 12 if "burns" in sp else 0
    @staticmethod
    def _default_nicu(tier: HospitalTier, sp: Dict) -> int:
        return 10 if "neonatal" in sp or "paediatrics" in sp else 0
    @staticmethod
    def _default_psych(tier: HospitalTier, sp: Dict) -> int:
        return 8 if "psychiatry" in sp else 0
    @staticmethod
    def _haversine_min(lat1: float, lon1: float,
                       lat2: float, lon2: float,
                       avg_speed_kmh: float = 40.0) -> float:
        R   = 6371.0
        φ1  = math.radians(lat1);  φ2  = math.radians(lat2)
        dφ  = math.radians(lat2 - lat1)
        dλ  = math.radians(lon2 - lon1)
        a   = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        km  = R * 2 * math.asin(math.sqrt(a))
        return (km / avg_speed_kmh) * 60.0
    def describe(self) -> Dict[str, Any]:
        diverted = [hid for hid, h in self.hospitals.items() if h.is_diverted]
        return {
            "sim_time_min":    self.sim_time_min,
            "step_count":      self.step_count,
            "task_id":         self.task_id,
            "hospital_count":  len(self.hospitals),
            "diverted":        diverted,
            "system_overload": self.is_system_in_overload(),
            "active_transfers":sum(1 for x in self.transfers.values()
                                   if x.status in (TransferStatus.ACCEPTED,
                                                   TransferStatus.EN_ROUTE)),
            "overflow_active": self.overflow_hospitals_active,
            "analytics":       self.get_episode_analytics(),
        }
    def print_status_table(self) -> None:
        header = (
            f"{'HOSP':>6} {'NAME':>30} {'TIER':>10} {'DIV':>10} "
            f"{'ER%':>5} {'ICU%':>5} {'GEN_AV':>7} {'ICU_AV':>7}"
        )
        print(header)
        print("-" * len(header))
        for hosp in self._iter_hospitals():
            print(
                f"{hosp.hospital_id:>6} {hosp.name[:30]:>30} "
                f"{hosp.tier.value:>10} "
                f"{hosp.diversion_status.value:>10} "
                f"{hosp.er_load*100:>5.1f} "
                f"{hosp.icu_load*100:>5.1f} "
                f"{hosp.available_general_beds:>7} "
                f"{hosp.available_icu_beds:>7}"
            )
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 72)
    print("EMERGI-ENV  ·  HospitalNetwork smoke-test")
    print("=" * 72)
    net = HospitalNetwork(seed=42)
    obs = net.reset(task_id=4, sim_time_minutes=480.0)
    print(f"\n✓  Loaded {obs['hospital_count']} hospitals")
    print(f"   Diverted at start: {obs['diverted_count']}")
    print(f"   System overload:   {obs['system_overload']}")
    print()
    net.print_status_table()
    best_id, eta, score = net.get_best_hospital(
        from_zone_id   = "Z06",
        condition_code = "STEMI",
        priority       = "P1",
    )
    print(f"\n✓  Best STEMI hospital from Z06: {best_id} "
          f"(ETA={eta:.1f} min, score={score:.3f})")
    ok, msg = net.issue_pre_alert(best_id, "STEMI", "P1", eta_minutes=12.0)
    print(f"✓  Pre-alert: {ok} — {msg}")
    ok2, reason2, delay2 = net.admit_patient(
        hospital_id    = best_id,
        incident_id    = "INC-0001",
        priority       = "P1",
        condition_code = "STEMI",
        unit_id        = "AMB-M001",
    )
    print(f"✓  Admit STEMI to {best_id}: {ok2} — {reason2} "
          f"(handover delay={delay2:.1f} min)")
    h01 = net.hospitals.get("H01")
    if h01:
        h01.occupied_general = h01.capacity_general
        h01.occupied_hdu     = h01.capacity_hdu
        print(f"\n  Forcing H01 ER overload → ER load = {h01.er_load*100:.1f}%")
    print(f"\n  Stepping 20 steps:")
    total_rwd = 0.0
    for s in range(20):
        _, rwd = net.step(480.0 + (s + 1) * STEP_DURATION_MINUTES)
        total_rwd += rwd
    print(f"  Cumulative step reward: {total_rwd:.4f}")
    print(f"  H01 diversion: {net.hospitals['H01'].diversion_status.value if 'H01' in net.hospitals else 'N/A'}")
    print(f"  Diverted hospitals: {net.count_diverted()}")
    xfr_ok, xfr_msg, xfr = net.request_transfer(
        from_hospital_id = "H06",
        to_hospital_id   = "H01",
        incident_id      = "INC-0042",
        condition_code   = "HAEMORRHAGIC_STROKE",
        priority         = "P1",
        specialty_needed = "neurosurgery",
        transport_unit_id= "AMB-A007",
    )
    print(f"\n✓  Transfer request: {xfr_ok} — {xfr_msg}")
    if xfr:
        net.mark_transfer_departed(xfr.transfer_id)
        print(f"   Transfer departed: {xfr.transfer_id} ETA {xfr.transfer_eta_min:.1f} min")
    net.activate_surge("H05")
    overflow_ids = net.activate_overflow_hospitals()
    print(f"\n✓  Surge activated at H05")
    print(f"✓  Overflow hospitals: {overflow_ids}")
    analytics = net.get_episode_analytics()
    print(f"\n  Episode analytics:")
    for k, v in analytics.items():
        if not isinstance(v, dict):
            print(f"   {k}: {v}")
    print("\n✅  HospitalNetwork smoke-test PASSED")