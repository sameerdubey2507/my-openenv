from __future__ import annotations
import json
import math
import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import numpy as np
_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent.parent / "data"
MAX_ACTIVE_INCIDENTS   = 80      
MCI_VICTIM_MIN         = 20
MCI_VICTIM_MAX         = 40
DUPLICATE_MERGE_RADIUS = 0.5    
DUPLICATE_MERGE_WINDOW = 2      
CLUSTER_SPAWN_PROB     = 0.18   
STEP_DURATION_MIN      = 3.0   
DECAY_STEEP      = "steep"       
DECAY_LINEAR     = "linear"      
DECAY_SIGMOID    = "sigmoid"     
DECAY_FLAT       = "flat"        
DECAY_EXPONENTIAL = "exponential"  
class IncidentPriority(str, Enum):
    P1 = "P1"   
    P2 = "P2"   
    P3 = "P3"   
    P0 = "P0"   
class StartTag(str, Enum):
    IMMEDIATE  = "Immediate"   
    DELAYED    = "Delayed"     
    MINIMAL    = "Minimal"     
    EXPECTANT  = "Expectant"   
class UnitType(str, Enum):
    BLS  = "BLS"    
    ALS  = "ALS"    
    MICU = "MICU"   
class IncidentStatus(str, Enum):
    REPORTED     = "reported"
    TRIAGED      = "triaged"
    DISPATCHED   = "dispatched"
    ON_SCENE     = "on_scene"
    TRANSPORTING = "transporting"
    RESOLVED     = "resolved"
    CANCELLED    = "cancelled"
    DUPLICATE    = "duplicate"
    ESCALATED    = "escalated"     
class AgencyRequirement(str, Enum):
    EMS_ONLY      = "ems_only"
    POLICE_EMS    = "police_ems"          
    FIRE_EMS      = "fire_ems"            
    ALL_THREE     = "all_three"           
    HAZMAT        = "hazmat"              
    COAST_GUARD   = "coast_guard"         
class MCIType(str, Enum):
    ROAD_TRAFFIC_ACCIDENT = "road_traffic_accident"
    BUILDING_COLLAPSE     = "building_collapse"
    INDUSTRIAL_EXPLOSION  = "industrial_explosion"
    STAMPEDE              = "stampede"
    TRAIN_ACCIDENT        = "train_accident"
    CHEMICAL_SPILL        = "chemical_spill"
    FLOOD_RESCUE          = "flood_rescue"
    FIRE_MULTI_VICTIM     = "fire_multi_victim"
    FESTIVAL_CROWD        = "festival_crowd"
    TERRORIST_ATTACK      = "terrorist_attack"
@dataclass
class RPMScore:
    respiration_rate:     int    
    pulse_present:        bool
    pulse_rate:           int    
    radial_pulse:         bool   
    obeys_commands:       bool   
    capillary_refill_sec: float  
    gcs:                  int    
    is_ambulatory:        bool = False  
    @property
    def start_tag(self) -> StartTag:
        if self.is_ambulatory:
            return StartTag.MINIMAL
        if self.respiration_rate == 0:
            return StartTag.EXPECTANT
        if self.respiration_rate > 30 or self.respiration_rate < 10:
            return StartTag.IMMEDIATE
        if not self.radial_pulse or self.capillary_refill_sec > 2.0:
            return StartTag.IMMEDIATE
        if not self.obeys_commands:
            return StartTag.IMMEDIATE
        return StartTag.DELAYED
    @property
    def priority(self) -> IncidentPriority:
        tag = self.start_tag
        return {
            StartTag.IMMEDIATE: IncidentPriority.P1,
            StartTag.DELAYED:   IncidentPriority.P2,
            StartTag.MINIMAL:   IncidentPriority.P3,
            StartTag.EXPECTANT: IncidentPriority.P0,
        }[tag]
@dataclass
class SurvivalCurveParams:
    decay_kind:          str      
    baseline_survival:   float   
    golden_hour_minutes: float   
    min_survival:        float   
    decay_rate:          float   
    condition_code:      str     
@dataclass
class Incident:
    incident_id:    str
    call_number:    str                
    zone_id:        str
    zone_name:      str
    lat:            float
    lon:            float
    symptom_description:    str        
    caller_reported_count:  int        
    reported_priority:      str        
    incident_category:      str        
    agency_flags:           List[str]  
    is_mci:                 bool
    mci_type:               Optional[str]
    victim_count:           int        
    status:                 IncidentStatus
    reported_at_step:       int
    dispatched_at_step:     Optional[int]
    resolved_at_step:       Optional[int]
    assigned_unit_ids:      List[str]  
    assigned_hospital_id:   Optional[str]
    ground_truth_priority:             IncidentPriority
    ground_truth_start_tag:            Optional[StartTag]      
    ground_truth_unit_type:            UnitType
    ground_truth_hospital_specialty:   List[str]               
    ground_truth_agency:               AgencyRequirement
    ground_truth_protocol:             List[str]               
    survival_curve:                    SurvivalCurveParams
    rpm_score:                         Optional[RPMScore]       
    time_since_reported_min:   float = 0.0
    deterioration_level:       float = 0.0   
    current_survival_prob:     float = 1.0
    is_duplicate:              bool  = False
    duplicate_of:              Optional[str] = None
    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "incident_id":           self.incident_id,
            "call_number":           self.call_number,
            "zone_id":               self.zone_id,
            "zone_name":             self.zone_name,
            "lat":                   round(self.lat, 5),
            "lon":                   round(self.lon, 5),
            "symptom_description":   self.symptom_description,
            "caller_reported_count": self.caller_reported_count,
            "reported_priority":     self.reported_priority,
            "incident_category":     self.incident_category,
            "agency_flags":          self.agency_flags,
            "is_mci":                self.is_mci,
            "mci_type":              self.mci_type,
            "victim_count":          self.victim_count,
            "status":                self.status.value,
            "reported_at_step":      self.reported_at_step,
            "dispatched_at_step":    self.dispatched_at_step,
            "resolved_at_step":      self.resolved_at_step,
            "assigned_unit_ids":     self.assigned_unit_ids,
            "assigned_hospital_id":  self.assigned_hospital_id,
            "time_since_reported_min": round(self.time_since_reported_min, 1),
            "deterioration_level":   round(self.deterioration_level, 3),
        }
    def to_full_state_dict(self) -> Dict[str, Any]:
        base = self.to_observation_dict()
        base.update({
            "ground_truth_priority":           self.ground_truth_priority.value,
            "ground_truth_start_tag":          self.ground_truth_start_tag.value
                                               if self.ground_truth_start_tag else None,
            "ground_truth_unit_type":          self.ground_truth_unit_type.value,
            "ground_truth_hospital_specialty": self.ground_truth_hospital_specialty,
            "ground_truth_agency":             self.ground_truth_agency.value,
            "ground_truth_protocol":           self.ground_truth_protocol,
            "survival_curve":                  {
                "decay_kind":         self.survival_curve.decay_kind,
                "baseline_survival":  self.survival_curve.baseline_survival,
                "golden_hour_minutes":self.survival_curve.golden_hour_minutes,
                "min_survival":       self.survival_curve.min_survival,
                "decay_rate":         self.survival_curve.decay_rate,
                "condition_code":     self.survival_curve.condition_code,
            },
            "current_survival_prob": round(self.current_survival_prob, 4),
            "is_duplicate":          self.is_duplicate,
        })
        return base
@dataclass
class MCIScene:
    scene_id:      str
    mci_type:      MCIType
    zone_id:       str
    total_victims: int
    victims:       List[Incident] = field(default_factory=list)
    spawned_at_step: int = 0
    is_active:     bool = True
    @property
    def victim_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {
            "Immediate": 0, "Delayed": 0, "Minimal": 0, "Expectant": 0
        }
        for v in self.victims:
            if v.ground_truth_start_tag:
                counts[v.ground_truth_start_tag.value] += 1
        return counts
    @property
    def immediate_count(self) -> int:
        return self.victim_breakdown["Immediate"]
INCIDENT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "CARD_001", "category": "cardiac", "subcategory": "stemi",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["cardiology", "cath_lab"],
        "agency": "ems_only",
        "protocols": ["STEMI_ACTIVATION", "ECG_TRANSMISSION", "ASPIRIN_300MG"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.96, "golden_min": 90.0,
                     "min_surv": 0.25, "rate": 0.018, "code": "STEMI"},
        "weight": 4.2,
        "descriptions": [
            "Male patient, approximately 55 years, severe chest pain radiating to left arm, sweating profusely, onset 20 minutes ago. Wife reporting he is pale and confused.",
            "58-year-old male, sudden crushing chest pain, pain score 9/10, shortness of breath, diaphoresis. History of hypertension.",
            "Caller reports husband collapsed, chest pain started 30 minutes back, now becoming unconscious. Diabetic patient.",
        ],
        "rpm_range": {"rr": (10, 28), "pulse": True, "hr": (50, 110), "gcs": (10, 15)},
    },
    {
        "id": "CARD_002", "category": "cardiac", "subcategory": "cardiac_arrest",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["cardiology", "resuscitation"],
        "agency": "ems_only",
        "protocols": ["CPR_PROTOCOL", "DEFIB_READY", "ADRENALINE_IV"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.65, "golden_min": 10.0,
                     "min_surv": 0.05, "rate": 0.08, "code": "CARDIAC_ARREST"},
        "weight": 3.1,
        "descriptions": [
            "Patient found unconscious, not breathing, no pulse detected by bystander. CPR in progress by caller.",
            "Elderly male collapsed in market, bystander performing CPR, 112 called 5 minutes ago. Patient unresponsive.",
            "Female patient, 62 years, collapsed at home. No breathing detected. Family member on call, CPR instructions given.",
        ],
        "rpm_range": {"rr": (0, 6), "pulse": False, "hr": (0, 40), "gcs": (3, 6)},
    },
    {
        "id": "CARD_003", "category": "cardiac", "subcategory": "unstable_angina",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["cardiology"],
        "agency": "ems_only",
        "protocols": ["ECG_12LEAD", "NITRO_SL", "ASPIRIN_300MG"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.97, "golden_min": 120.0,
                     "min_surv": 0.70, "rate": 0.003, "code": "UNSTABLE_ANGINA"},
        "weight": 3.8,
        "descriptions": [
            "Male, 48 years, chest discomfort on exertion since this morning, slightly better at rest. No diaphoresis.",
            "Patient with known heart disease, chest pain 6/10, came on after climbing stairs. Wife worried, patient stable.",
        ],
        "rpm_range": {"rr": (12, 22), "pulse": True, "hr": (60, 100), "gcs": (14, 15)},
    },
    {
        "id": "CARD_004", "category": "cardiac", "subcategory": "svt",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["cardiology"],
        "agency": "ems_only",
        "protocols": ["ECG_12LEAD", "VAGAL_MANOEUVRE", "ADENOSINE_IV"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.99, "golden_min": 180.0,
                     "min_surv": 0.90, "rate": 0.001, "code": "SVT"},
        "weight": 2.0,
        "descriptions": [
            "Young female, 32 years, palpitations, heart racing suddenly, feeling faint. BP not measured.",
            "Patient reports heart fluttering very fast for past 15 minutes, no chest pain, no loss of consciousness.",
        ],
        "rpm_range": {"rr": (14, 22), "pulse": True, "hr": (150, 220), "gcs": (14, 15)},
    },
    {
        "id": "STRO_001", "category": "stroke", "subcategory": "ischaemic_stroke",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["neurology", "stroke_unit", "ct_scanner"],
        "agency": "ems_only",
        "protocols": ["FAST_PROTOCOL", "THROMBOLYSIS_WINDOW", "NIL_BY_MOUTH"],
        "survival": {"kind": DECAY_SIGMOID, "baseline": 0.94, "golden_min": 270.0,
                     "min_surv": 0.35, "rate": 0.012, "code": "ISCHAEMIC_STROKE"},
        "weight": 3.5,
        "descriptions": [
            "Male patient, 62 years, sudden onset slurred speech and right-sided weakness 45 minutes ago. FAST positive.",
            "Elderly female, face drooping on one side, cannot raise both arms, speech garbled. Onset approximately 1 hour ago.",
            "Patient found with sudden confusion, cannot speak properly, left arm weak. Neighbour called. No trauma.",
        ],
        "rpm_range": {"rr": (12, 24), "pulse": True, "hr": (60, 100), "gcs": (8, 14)},
    },
    {
        "id": "STRO_002", "category": "stroke", "subcategory": "haemorrhagic_stroke",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["neurosurgery", "ct_scanner", "icu"],
        "agency": "ems_only",
        "protocols": ["BP_CONTROL", "NEURO_ALERT", "NIL_BY_MOUTH"],
        "survival": {"kind": DECAY_SIGMOID, "baseline": 0.82, "golden_min": 60.0,
                     "min_surv": 0.15, "rate": 0.025, "code": "HAEMORRHAGIC_STROKE"},
        "weight": 1.8,
        "descriptions": [
            "Severe sudden headache described as 'worst headache of my life', patient now semiconscious. Vomiting.",
            "Patient collapsed with severe headache, now unresponsive, BP reportedly very high. History of hypertension.",
        ],
        "rpm_range": {"rr": (8, 20), "pulse": True, "hr": (40, 80), "gcs": (6, 12)},
    },
    {
        "id": "STRO_003", "category": "stroke", "subcategory": "tia",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["neurology"],
        "agency": "ems_only",
        "protocols": ["FAST_PROTOCOL", "ASPIRIN_300MG"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.99, "golden_min": 240.0,
                     "min_surv": 0.85, "rate": 0.001, "code": "TIA"},
        "weight": 2.2,
        "descriptions": [
            "Patient had brief episode of weakness in right hand lasting 10 minutes, now resolved. Family worried.",
        ],
        "rpm_range": {"rr": (12, 20), "pulse": True, "hr": (60, 90), "gcs": (14, 15)},
    },
    {
        "id": "TRAU_001", "category": "trauma", "subcategory": "polytrauma_rta",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["trauma", "emergency_surgery", "orthopaedics"],
        "agency": "fire_ems",
        "protocols": ["MAJOR_TRAUMA_PROTOCOL", "NEAREST_TRAUMA_30MIN",
                      "SPINAL_PRECAUTIONS", "MASSIVE_HAEMORRHAGE"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.90, "golden_min": 60.0,
                     "min_surv": 0.20, "rate": 0.012, "code": "POLYTRAUMA"},
        "weight": 5.0,
        "descriptions": [
            "Major road traffic accident, two-wheeler hit by truck on highway. Patient unconscious, significant blood loss, possible leg fracture.",
            "RTA on NH48 — bus rollover, multiple patients. Caller reporting at least 3 unconscious victims, trapped under vehicle.",
            "High-speed collision, patient ejected from vehicle. GCS appears low, deformity to chest and pelvis noted.",
        ],
        "rpm_range": {"rr": (6, 35), "pulse": True, "hr": (40, 130), "gcs": (3, 10)},
    },
    {
        "id": "TRAU_002", "category": "trauma", "subcategory": "head_injury",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["neurosurgery", "ct_scanner"],
        "agency": "ems_only",
        "protocols": ["HEAD_INJURY_PROTOCOL", "SPINAL_PRECAUTIONS", "AIRWAY_MANAGEMENT"],
        "survival": {"kind": DECAY_SIGMOID, "baseline": 0.88, "golden_min": 120.0,
                     "min_surv": 0.30, "rate": 0.015, "code": "HEAD_INJURY_SEVERE"},
        "weight": 4.2,
        "descriptions": [
            "Fall from construction scaffolding, 3rd floor. Patient unconscious, head injury visible, helmet not worn.",
            "Two-wheeler accident, patient hit head on divider, brief loss of consciousness, now drowsy and confused.",
        ],
        "rpm_range": {"rr": (10, 28), "pulse": True, "hr": (50, 120), "gcs": (6, 13)},
    },
    {
        "id": "TRAU_003", "category": "trauma", "subcategory": "penetrating_trauma",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["trauma", "emergency_surgery"],
        "agency": "police_ems",
        "protocols": ["PENETRATING_TRAUMA_PROTOCOL", "TOURNIQUET_APPLY",
                      "CRIME_SCENE_PRESERVE"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.85, "golden_min": 45.0,
                     "min_surv": 0.15, "rate": 0.018, "code": "PENETRATING_TRAUMA"},
        "weight": 2.0,
        "descriptions": [
            "Stabbing incident — male, 28 years, stab wound to abdomen, significant bleeding. Police informed.",
            "Gunshot wound to chest, patient conscious but deteriorating rapidly. Police on scene.",
        ],
        "rpm_range": {"rr": (8, 32), "pulse": True, "hr": (60, 140), "gcs": (8, 15)},
    },
    {
        "id": "TRAU_004", "category": "trauma", "subcategory": "fall_elderly",
        "priority": "P2", "unit_type": "BLS",
        "specialties": ["orthopaedics", "emergency"],
        "agency": "ems_only",
        "protocols": ["FALL_ASSESSMENT", "SPINAL_PRECAUTIONS", "PAIN_MANAGEMENT"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.97, "golden_min": 120.0,
                     "min_surv": 0.75, "rate": 0.002, "code": "FALL_ELDERLY"},
        "weight": 6.0,
        "descriptions": [
            "Elderly woman, 74 years, fell in bathroom, unable to get up, hip pain. Alert and oriented.",
            "Old man fell from chair at home, knee and wrist pain. Not unconscious. Daughter calling.",
            "72-year-old male, slip and fall near temple steps. Right leg pain, limping. No head injury reported.",
        ],
        "rpm_range": {"rr": (14, 22), "pulse": True, "hr": (70, 100), "gcs": (14, 15)},
    },
    {
        "id": "TRAU_005", "category": "trauma", "subcategory": "trapped_victim_rta",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["trauma", "emergency_surgery"],
        "agency": "all_three",
        "protocols": ["ENTRAPMENT_PROTOCOL", "FIRE_EXTRICATION_WAIT",
                      "MAJOR_TRAUMA_PROTOCOL", "SPINAL_PRECAUTIONS"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.88, "golden_min": 60.0,
                     "min_surv": 0.20, "rate": 0.015, "code": "POLYTRAUMA"},
        "weight": 2.5,
        "descriptions": [
            "Truck accident — driver trapped inside crushed cab, conscious but unable to move. Fire brigade needed for extrication.",
            "Car vs divider collision, patient trapped, airbags deployed, fuel leak reported. Police and fire required.",
        ],
        "rpm_range": {"rr": (12, 30), "pulse": True, "hr": (60, 130), "gcs": (8, 15)},
    },
    {
        "id": "TRAU_006", "category": "trauma", "subcategory": "minor_rta",
        "priority": "P3", "unit_type": "BLS",
        "specialties": ["emergency"],
        "agency": "ems_only",
        "protocols": ["WOUND_CARE", "PAIN_MANAGEMENT"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.999, "golden_min": 300.0,
                     "min_surv": 0.95, "rate": 0.0002, "code": "MINOR_TRAUMA"},
        "weight": 8.0,
        "descriptions": [
            "Minor two-wheeler accident, patient has road rash and wrist pain. Alert, no head injury, requesting ambulance.",
            "Cyclist fell, knee laceration, walking wounded. Wants to be checked at hospital.",
        ],
        "rpm_range": {"rr": (14, 20), "pulse": True, "hr": (70, 95), "gcs": (15, 15)},
    },
    {
        "id": "RESP_001", "category": "respiratory", "subcategory": "acute_asthma",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["pulmonology", "emergency"],
        "agency": "ems_only",
        "protocols": ["ASTHMA_PROTOCOL", "NEBULISATION", "O2_15LPM"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.93, "golden_min": 30.0,
                     "min_surv": 0.50, "rate": 0.035, "code": "ACUTE_ASTHMA"},
        "weight": 3.5,
        "descriptions": [
            "Child, 8 years, severe breathlessness, wheezing audible over phone. Known asthmatic, inhaler not working.",
            "Adult female, 35, acute severe asthma attack. Cannot speak in full sentences, cyanotic lips.",
        ],
        "rpm_range": {"rr": (28, 45), "pulse": True, "hr": (100, 140), "gcs": (10, 14)},
    },
    {
        "id": "RESP_002", "category": "respiratory", "subcategory": "copd_exacerbation",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["pulmonology", "emergency"],
        "agency": "ems_only",
        "protocols": ["COPD_PROTOCOL", "CONTROLLED_O2", "NEBULISATION"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.95, "golden_min": 90.0,
                     "min_surv": 0.60, "rate": 0.006, "code": "COPD_EXACERBATION"},
        "weight": 3.0,
        "descriptions": [
            "Elderly male, known COPD, increased breathlessness past 2 days, today much worse. On home oxygen.",
        ],
        "rpm_range": {"rr": (22, 34), "pulse": True, "hr": (90, 120), "gcs": (13, 15)},
    },
    {
        "id": "RESP_003", "category": "respiratory", "subcategory": "pulmonary_oedema",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["cardiology", "icu"],
        "agency": "ems_only",
        "protocols": ["PULMONARY_OEDEMA_PROTOCOL", "FRUSEMIDE_IV", "SITTING_POSITION"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.90, "golden_min": 60.0,
                     "min_surv": 0.30, "rate": 0.022, "code": "PULMONARY_OEDEMA"},
        "weight": 2.0,
        "descriptions": [
            "Patient cannot breathe lying flat, pink frothy sputum, extreme breathlessness. Known heart failure patient.",
        ],
        "rpm_range": {"rr": (28, 40), "pulse": True, "hr": (100, 130), "gcs": (12, 15)},
    },
    {
        "id": "OBST_001", "category": "obstetric", "subcategory": "labour_imminent",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["obstetrics", "nicu"],
        "agency": "ems_only",
        "protocols": ["OBSTETRIC_EMERGENCY", "BIRTH_KIT_PREPARE", "NEONATAL_RESUS_READY"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.98, "golden_min": 30.0,
                     "min_surv": 0.80, "rate": 0.005, "code": "IMMINENT_DELIVERY"},
        "weight": 2.8,
        "descriptions": [
            "Pregnant woman, 38 weeks, contractions every 2 minutes, urge to push. Husband calling, first baby.",
            "Labour pains very strong, water broken, patient says she can feel the baby coming. Cannot get to hospital in time.",
        ],
        "rpm_range": {"rr": (16, 26), "pulse": True, "hr": (80, 110), "gcs": (15, 15)},
    },
    {
        "id": "OBST_002", "category": "obstetric", "subcategory": "eclampsia",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["obstetrics", "icu", "nicu"],
        "agency": "ems_only",
        "protocols": ["ECLAMPSIA_PROTOCOL", "MAGNESIUM_IV", "LEFT_LATERAL_POSITION"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.88, "golden_min": 45.0,
                     "min_surv": 0.30, "rate": 0.028, "code": "ECLAMPSIA"},
        "weight": 1.2,
        "descriptions": [
            "Pregnant woman, 34 weeks, had a seizure at home, now semiconscious. High BP known during pregnancy.",
        ],
        "rpm_range": {"rr": (10, 24), "pulse": True, "hr": (80, 120), "gcs": (7, 13)},
    },
    {
        "id": "OBST_003", "category": "obstetric", "subcategory": "postpartum_haemorrhage",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["obstetrics", "blood_bank", "icu"],
        "agency": "ems_only",
        "protocols": ["PPH_PROTOCOL", "IV_ACCESS_2X", "OXYTOCIN_IV"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.91, "golden_min": 60.0,
                     "min_surv": 0.25, "rate": 0.020, "code": "PPH"},
        "weight": 1.0,
        "descriptions": [
            "New mother, delivered 6 hours ago at home, now excessive bleeding not stopping. Midwife very worried.",
        ],
        "rpm_range": {"rr": (18, 30), "pulse": True, "hr": (100, 140), "gcs": (10, 15)},
    },
    {
        "id": "PAED_001", "category": "paediatric", "subcategory": "febrile_seizure",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["paediatrics", "emergency"],
        "agency": "ems_only",
        "protocols": ["PAED_SEIZURE_PROTOCOL", "DIAZEPAM_PR", "AIRWAY_MANAGEMENT"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.97, "golden_min": 20.0,
                     "min_surv": 0.70, "rate": 0.015, "code": "FEBRILE_SEIZURE"},
        "weight": 3.2,
        "descriptions": [
            "Child, 3 years, high fever and now shaking/convulsing. Parents very panicked. First seizure.",
            "Toddler, 18 months, eyes rolled back, body stiff and jerking, high temperature. Mother calling.",
        ],
        "rpm_range": {"rr": (24, 45), "pulse": True, "hr": (120, 180), "gcs": (6, 12)},
    },
    {
        "id": "PAED_002", "category": "paediatric", "subcategory": "choking_infant",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["paediatrics", "emergency"],
        "agency": "ems_only",
        "protocols": ["PAED_CHOKING_PROTOCOL", "BACK_BLOWS_5X", "AIRWAY_MANAGEMENT"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.88, "golden_min": 5.0,
                     "min_surv": 0.20, "rate": 0.12, "code": "AIRWAY_OBSTRUCTION"},
        "weight": 1.5,
        "descriptions": [
            "Baby, 9 months, choking on food, turning blue, not crying. Mother panicking on call.",
        ],
        "rpm_range": {"rr": (0, 10), "pulse": True, "hr": (60, 160), "gcs": (3, 10)},
    },
    {
        "id": "PAED_003", "category": "paediatric", "subcategory": "drowning_near",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["paediatrics", "icu", "emergency"],
        "agency": "ems_only",
        "protocols": ["NEAR_DROWNING_PROTOCOL", "C_SPINE_PRECAUTIONS", "O2_HIGH_FLOW"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.82, "golden_min": 30.0,
                     "min_surv": 0.25, "rate": 0.040, "code": "NEAR_DROWNING"},
        "weight": 1.0,
        "descriptions": [
            "Child pulled from well, not breathing initially, now coughing. Wet, cold, semiconscious.",
        ],
        "rpm_range": {"rr": (4, 20), "pulse": True, "hr": (40, 100), "gcs": (6, 13)},
    },
    {
        "id": "NEUR_001", "category": "neurological", "subcategory": "epilepsy_status",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["neurology", "emergency"],
        "agency": "ems_only",
        "protocols": ["STATUS_EPILEPTICUS_PROTOCOL", "DIAZEPAM_IV", "AIRWAY_MANAGEMENT"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.92, "golden_min": 30.0,
                     "min_surv": 0.55, "rate": 0.028, "code": "STATUS_EPILEPTICUS"},
        "weight": 2.5,
        "descriptions": [
            "Known epileptic, seizure not stopping, 10 minutes now, not regaining consciousness between fits.",
        ],
        "rpm_range": {"rr": (8, 30), "pulse": True, "hr": (80, 150), "gcs": (3, 10)},
    },
    {
        "id": "NEUR_002", "category": "neurological", "subcategory": "meningitis",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["neurology", "icu", "infectious_disease"],
        "agency": "ems_only",
        "protocols": ["MENINGITIS_PROTOCOL", "ISOLATION_PRECAUTIONS", "ANTIBIOTICS_ASAP"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.87, "golden_min": 120.0,
                     "min_surv": 0.30, "rate": 0.010, "code": "BACTERIAL_MENINGITIS"},
        "weight": 0.8,
        "descriptions": [
            "Young adult, severe headache, stiff neck, photophobia, high fever. Petechial rash noticed on arm.",
        ],
        "rpm_range": {"rr": (14, 26), "pulse": True, "hr": (80, 120), "gcs": (8, 14)},
    },
    {
        "id": "BURN_001", "category": "burns", "subcategory": "major_burns",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["burns", "icu", "plastic_surgery"],
        "agency": "fire_ems",
        "protocols": ["BURNS_MAJOR_PROTOCOL", "FLUID_RESUSCITATION", "COOL_RUNNING_WATER"],
        "survival": {"kind": DECAY_EXPONENTIAL, "baseline": 0.78, "golden_min": 60.0,
                     "min_surv": 0.10, "rate": 0.030, "code": "MAJOR_BURNS"},
        "weight": 1.5,
        "descriptions": [
            "LPG cylinder explosion at home, 40% body burns, patient screaming, bystander says skin peeling off.",
            "Factory fire, worker with extensive burns to face, chest, arms. Fire brigade extinguished fire.",
        ],
        "rpm_range": {"rr": (18, 36), "pulse": True, "hr": (100, 150), "gcs": (10, 15)},
    },
    {
        "id": "BURN_002", "category": "burns", "subcategory": "inhalation_injury",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["emergency", "pulmonology", "icu"],
        "agency": "fire_ems",
        "protocols": ["INHALATION_PROTOCOL", "O2_HIGH_FLOW", "AIRWAY_WATCH"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.85, "golden_min": 45.0,
                     "min_surv": 0.35, "rate": 0.025, "code": "INHALATION_INJURY"},
        "weight": 1.0,
        "descriptions": [
            "Building fire, patient rescued, singed eyebrows and nasal hair, hoarse voice, coughing black sputum.",
        ],
        "rpm_range": {"rr": (22, 40), "pulse": True, "hr": (100, 140), "gcs": (11, 15)},
    },
    {
        "id": "HAZM_001", "category": "industrial", "subcategory": "chemical_exposure",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["toxicology", "icu", "emergency"],
        "agency": "hazmat",
        "protocols": ["HAZMAT_PROTOCOL", "DECONTAMINATION_FIRST",
                      "ORGANOPHOSPHATE_ANTIDOTE", "SCENE_SAFETY"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.80, "golden_min": 30.0,
                     "min_surv": 0.15, "rate": 0.045, "code": "ORGANOPHOSPHATE_POISONING"},
        "weight": 0.8,
        "descriptions": [
            "MIDC industrial area — chemical leak, multiple workers with eye burning, difficulty breathing, excessive salivation.",
            "Pesticide factory accident, worker collapsed, pinpoint pupils, muscle twitching. Hazmat team needed.",
        ],
        "rpm_range": {"rr": (5, 30), "pulse": True, "hr": (40, 80), "gcs": (5, 12)},
    },
    {
        "id": "HAZM_002", "category": "industrial", "subcategory": "crush_injury_industrial",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["trauma", "orthopaedics", "renal"],
        "agency": "fire_ems",
        "protocols": ["CRUSH_SYNDROME_PROTOCOL", "IV_SALINE_AGGRESSIVE",
                      "TOURNIQUET_CONSIDER"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.88, "golden_min": 90.0,
                     "min_surv": 0.40, "rate": 0.010, "code": "CRUSH_INJURY"},
        "weight": 1.2,
        "descriptions": [
            "Construction site collapse, worker pinned under concrete slab for 2 hours. Just freed by rescue team.",
        ],
        "rpm_range": {"rr": (14, 28), "pulse": True, "hr": (80, 130), "gcs": (10, 15)},
    },
    {
        "id": "PSYC_001", "category": "psychiatric", "subcategory": "self_harm",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["emergency", "psychiatry"],
        "agency": "police_ems",
        "protocols": ["SELF_HARM_PROTOCOL", "WOUND_CARE", "PSYCHIATRIC_CONSULT",
                      "POLICE_NOTIFY"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.97, "golden_min": 120.0,
                     "min_surv": 0.80, "rate": 0.003, "code": "SELF_HARM"},
        "weight": 1.8,
        "descriptions": [
            "Young male, 22 years, self-inflicted wrist laceration, police called by family. Still conscious.",
        ],
        "rpm_range": {"rr": (14, 22), "pulse": True, "hr": (70, 100), "gcs": (13, 15)},
    },
    {
        "id": "PSYC_002", "category": "psychiatric", "subcategory": "overdose_intentional",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["emergency", "toxicology", "psychiatry"],
        "agency": "police_ems",
        "protocols": ["OVERDOSE_PROTOCOL", "NALOXONE_CONSIDER", "ACTIVATED_CHARCOAL"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.90, "golden_min": 60.0,
                     "min_surv": 0.40, "rate": 0.022, "code": "DRUG_OVERDOSE"},
        "weight": 1.5,
        "descriptions": [
            "Young female, took entire strip of sleeping tablets 1 hour ago, now drowsy and unresponsive to voice.",
        ],
        "rpm_range": {"rr": (8, 18), "pulse": True, "hr": (50, 90), "gcs": (6, 12)},
    },
    {
        "id": "PSYC_003", "category": "psychiatric", "subcategory": "acute_psychosis",
        "priority": "P2", "unit_type": "BLS",
        "specialties": ["psychiatry"],
        "agency": "police_ems",
        "protocols": ["ACUTE_PSYCHOSIS_PROTOCOL", "POLICE_ASSIST", "SAFE_TRANSPORT"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.99, "golden_min": 300.0,
                     "min_surv": 0.90, "rate": 0.0005, "code": "ACUTE_PSYCHOSIS"},
        "weight": 1.5,
        "descriptions": [
            "Man behaving erratically in public, threatening passers-by, not making sense. Police requested.",
        ],
        "rpm_range": {"rr": (16, 24), "pulse": True, "hr": (80, 110), "gcs": (14, 15)},
    },
    {
        "id": "MED_001", "category": "medical", "subcategory": "diabetic_emergency",
        "priority": "P2", "unit_type": "BLS",
        "specialties": ["emergency"],
        "agency": "ems_only",
        "protocols": ["HYPOGLYCAEMIA_PROTOCOL", "GLUCAGON_IM", "DEXTROSE_IV"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.98, "golden_min": 60.0,
                     "min_surv": 0.75, "rate": 0.004, "code": "HYPOGLYCAEMIA"},
        "weight": 5.0,
        "descriptions": [
            "Diabetic patient, found confused and sweating. Last meal 8 hours ago, took insulin this morning.",
            "Known diabetic, unresponsive, family says he had missed meals, blood sugar reading 38 on home device.",
            "Old lady with diabetes, very weak, shaking, not able to speak properly. Son calling.",
        ],
        "rpm_range": {"rr": (14, 22), "pulse": True, "hr": (80, 110), "gcs": (8, 14)},
    },
    {
        "id": "MED_002", "category": "medical", "subcategory": "anaphylaxis",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["emergency", "allergy"],
        "agency": "ems_only",
        "protocols": ["ANAPHYLAXIS_PROTOCOL", "ADRENALINE_IM", "SUPINE_LEGS_RAISED"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.94, "golden_min": 20.0,
                     "min_surv": 0.30, "rate": 0.055, "code": "ANAPHYLAXIS"},
        "weight": 1.5,
        "descriptions": [
            "Bee sting 10 minutes ago, now patient has severe swelling of face and throat, difficulty breathing.",
            "Ate prawns at restaurant, now face swollen, throat tightening, hives all over body.",
        ],
        "rpm_range": {"rr": (24, 40), "pulse": True, "hr": (100, 150), "gcs": (11, 15)},
    },
    {
        "id": "MED_003", "category": "medical", "subcategory": "snake_bite",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["toxicology", "emergency"],
        "agency": "ems_only",
        "protocols": ["SNAKE_BITE_PROTOCOL", "IMMOBILISE_LIMB", "ANTIVENOM_ASSESS"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.87, "golden_min": 180.0,
                     "min_surv": 0.35, "rate": 0.008, "code": "SNAKE_BITE_VENOMOUS"},
        "weight": 2.5,
        "descriptions": [
            "Farmer bitten by snake in field, 30 minutes ago. Swelling spreading up arm. Cannot describe snake.",
            "Child bitten by snake near home, bite mark visible, limb swelling, becoming drowsy.",
        ],
        "rpm_range": {"rr": (10, 26), "pulse": True, "hr": (60, 110), "gcs": (10, 15)},
    },
    {
        "id": "MED_004", "category": "medical", "subcategory": "heat_stroke",
        "priority": "P1", "unit_type": "ALS",
        "specialties": ["emergency", "icu"],
        "agency": "ems_only",
        "protocols": ["HEAT_STROKE_PROTOCOL", "RAPID_COOLING", "IV_FLUIDS"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.88, "golden_min": 45.0,
                     "min_surv": 0.30, "rate": 0.030, "code": "HEAT_STROKE"},
        "weight": 2.0,
        "descriptions": [
            "Outdoor worker, collapsed in afternoon heat, skin hot and dry, confused. Temp reportedly very high.",
            "Old man found collapsed on road in peak summer, not sweating, temperature over 40 by bystander's estimate.",
        ],
        "rpm_range": {"rr": (20, 35), "pulse": True, "hr": (100, 140), "gcs": (8, 13)},
    },
    {
        "id": "MED_005", "category": "medical", "subcategory": "septic_shock",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["icu", "infectious_disease", "emergency"],
        "agency": "ems_only",
        "protocols": ["SEPSIS_6_PROTOCOL", "IV_FLUIDS_AGGRESSIVE", "ANTIBIOTICS_1H",
                      "BLOOD_CULTURE_BEFORE_ABX"],
        "survival": {"kind": DECAY_EXPONENTIAL, "baseline": 0.82, "golden_min": 60.0,
                     "min_surv": 0.15, "rate": 0.020, "code": "SEPTIC_SHOCK"},
        "weight": 2.5,
        "descriptions": [
            "Elderly diabetic, fever for 3 days, today very confused, cold and clammy, low BP reported.",
        ],
        "rpm_range": {"rr": (22, 38), "pulse": True, "hr": (100, 140), "gcs": (8, 13)},
    },
    {
        "id": "MED_006", "category": "medical", "subcategory": "gi_bleed",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["gastroenterology", "emergency", "blood_bank"],
        "agency": "ems_only",
        "protocols": ["GI_BLEED_PROTOCOL", "IV_ACCESS_2X", "BLOOD_GROUP_CROSSMATCH"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.93, "golden_min": 120.0,
                     "min_surv": 0.50, "rate": 0.007, "code": "UPPER_GI_BLEED"},
        "weight": 2.5,
        "descriptions": [
            "Patient vomiting large amounts of blood, known alcoholic, very weak and dizzy. Three episodes.",
        ],
        "rpm_range": {"rr": (18, 28), "pulse": True, "hr": (100, 140), "gcs": (12, 15)},
    },
    {
        "id": "MED_007", "category": "medical", "subcategory": "pulmonary_embolism",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["cardiology", "pulmonology", "icu"],
        "agency": "ems_only",
        "protocols": ["PE_PROTOCOL", "THROMBOLYSIS_CONSIDER", "O2_HIGH_FLOW"],
        "survival": {"kind": DECAY_STEEP, "baseline": 0.86, "golden_min": 60.0,
                     "min_surv": 0.25, "rate": 0.025, "code": "MASSIVE_PE"},
        "weight": 1.2,
        "descriptions": [
            "Post-operative patient, sudden severe breathlessness, chest pain, collapse. Recent hip surgery.",
        ],
        "rpm_range": {"rr": (24, 38), "pulse": True, "hr": (110, 150), "gcs": (11, 15)},
    },
    {
        "id": "MED_008", "category": "medical", "subcategory": "renal_failure_acute",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["nephrology", "emergency"],
        "agency": "ems_only",
        "protocols": ["AKI_PROTOCOL", "IV_FLUIDS", "STOP_NEPHROTOXINS"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.95, "golden_min": 180.0,
                     "min_surv": 0.65, "rate": 0.004, "code": "ACUTE_KIDNEY_INJURY"},
        "weight": 1.5,
        "descriptions": [
            "Patient not passed urine for 24 hours, swollen ankles, breathless, known kidney disease.",
        ],
        "rpm_range": {"rr": (18, 28), "pulse": True, "hr": (80, 110), "gcs": (13, 15)},
    },
    {
        "id": "WATR_001", "category": "water_rescue", "subcategory": "flood_trapped",
        "priority": "P1", "unit_type": "MICU",
        "specialties": ["emergency", "trauma"],
        "agency": "all_three",
        "protocols": ["FLOOD_RESCUE_PROTOCOL", "HYPOTHERMIA_PREVENT",
                      "NEAR_DROWNING_PROTOCOL"],
        "survival": {"kind": DECAY_LINEAR, "baseline": 0.90, "golden_min": 90.0,
                     "min_surv": 0.35, "rate": 0.010, "code": "FLOOD_RESCUE"},
        "weight": 0.8,
        "descriptions": [
            "Family of 4 trapped on rooftop by flood waters, including infant and elderly. State disaster response coordinating.",
        ],
        "rpm_range": {"rr": (12, 26), "pulse": True, "hr": (80, 120), "gcs": (12, 15)},
    },
    {
        "id": "ROUT_001", "category": "routine", "subcategory": "abdominal_pain",
        "priority": "P3", "unit_type": "BLS",
        "specialties": ["emergency", "gastroenterology"],
        "agency": "ems_only",
        "protocols": ["ABDOMINAL_PAIN_ASSESSMENT"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.999, "golden_min": 300.0,
                     "min_surv": 0.95, "rate": 0.0002, "code": "ABDOMINAL_PAIN_ROUTINE"},
        "weight": 7.0,
        "descriptions": [
            "Patient complaining of stomach pain since morning, no vomiting, passing gas, walking.",
            "Mild tummy ache, took antacid, not better, requesting ambulance to go to hospital.",
        ],
        "rpm_range": {"rr": (14, 20), "pulse": True, "hr": (70, 90), "gcs": (15, 15)},
    },
    {
        "id": "ROUT_002", "category": "routine", "subcategory": "general_weakness",
        "priority": "P3", "unit_type": "BLS",
        "specialties": ["emergency"],
        "agency": "ems_only",
        "protocols": ["GENERAL_ASSESSMENT"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.999, "golden_min": 300.0,
                     "min_surv": 0.97, "rate": 0.0001, "code": "GENERAL_WEAKNESS"},
        "weight": 8.0,
        "descriptions": [
            "Old man not feeling well, general weakness since yesterday. Wants to go to hospital for checkup.",
            "Patient feeling dizzy and tired, no specific complaint, no chest pain or breathing difficulty.",
        ],
        "rpm_range": {"rr": (14, 20), "pulse": True, "hr": (65, 90), "gcs": (15, 15)},
    },
    {
        "id": "ROUT_003", "category": "routine", "subcategory": "interhospital_transfer",
        "priority": "P2", "unit_type": "ALS",
        "specialties": ["varies"],
        "agency": "ems_only",
        "protocols": ["TRANSFER_PROTOCOL", "PATIENT_HANDOVER_DOCS"],
        "survival": {"kind": DECAY_FLAT, "baseline": 0.99, "golden_min": 240.0,
                     "min_surv": 0.88, "rate": 0.001, "code": "STABLE_TRANSFER"},
        "weight": 3.0,
        "descriptions": [
            "Stable patient requiring transfer from district hospital to tertiary centre for specialist care.",
        ],
        "rpm_range": {"rr": (14, 22), "pulse": True, "hr": (65, 95), "gcs": (14, 15)},
    },
]
MCI_VICTIM_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    MCIType.ROAD_TRAFFIC_ACCIDENT.value: {
        "Immediate": 0.25, "Delayed": 0.40, "Minimal": 0.30, "Expectant": 0.05
    },
    MCIType.BUILDING_COLLAPSE.value: {
        "Immediate": 0.30, "Delayed": 0.35, "Minimal": 0.25, "Expectant": 0.10
    },
    MCIType.INDUSTRIAL_EXPLOSION.value: {
        "Immediate": 0.35, "Delayed": 0.30, "Minimal": 0.25, "Expectant": 0.10
    },
    MCIType.STAMPEDE.value: {
        "Immediate": 0.15, "Delayed": 0.30, "Minimal": 0.50, "Expectant": 0.05
    },
    MCIType.TRAIN_ACCIDENT.value: {
        "Immediate": 0.28, "Delayed": 0.38, "Minimal": 0.28, "Expectant": 0.06
    },
    MCIType.CHEMICAL_SPILL.value: {
        "Immediate": 0.40, "Delayed": 0.30, "Minimal": 0.22, "Expectant": 0.08
    },
    MCIType.FLOOD_RESCUE.value: {
        "Immediate": 0.20, "Delayed": 0.35, "Minimal": 0.40, "Expectant": 0.05
    },
    MCIType.FIRE_MULTI_VICTIM.value: {
        "Immediate": 0.35, "Delayed": 0.30, "Minimal": 0.25, "Expectant": 0.10
    },
    MCIType.FESTIVAL_CROWD.value: {
        "Immediate": 0.10, "Delayed": 0.25, "Minimal": 0.60, "Expectant": 0.05
    },
    MCIType.TERRORIST_ATTACK.value: {
        "Immediate": 0.38, "Delayed": 0.32, "Minimal": 0.22, "Expectant": 0.08
    },
}
MCI_RPM_BY_TAG: Dict[str, Dict[str, Any]] = {
    "Immediate": {
        "rr_options": [(0, 0, 0.15), (5, 9, 0.25), (31, 40, 0.60)],  
        "pulse_absent_prob": 0.20,
        "pulse_range": (40, 130),
        "obeys_commands_prob": 0.30,
        "capillary_refill_range": (2.5, 5.0),
        "gcs_range": (3, 9),
    },
    "Delayed": {
        "rr_options": [(10, 29, 1.0)],
        "pulse_absent_prob": 0.02,
        "pulse_range": (60, 110),
        "obeys_commands_prob": 0.85,
        "capillary_refill_range": (2.0, 3.5),
        "gcs_range": (9, 13),
    },
    "Minimal": {
        "rr_options": [(12, 22, 1.0)],
        "pulse_absent_prob": 0.0,
        "pulse_range": (65, 100),
        "obeys_commands_prob": 1.0,
        "capillary_refill_range": (1.0, 2.0),
        "gcs_range": (13, 15),
    },
    "Expectant": {
        "rr_options": [(0, 0, 0.70), (4, 8, 0.30)],
        "pulse_absent_prob": 0.80,
        "pulse_range": (0, 40),
        "obeys_commands_prob": 0.0,
        "capillary_refill_range": (4.0, 6.0),
        "gcs_range": (3, 5),
    },
}
class IncidentEngine:
    def __init__(
        self,
        seed: int = 42,
        zone_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.seed    = seed
        self.rng     = random.Random(seed)
        self.np_rng  = np.random.RandomState(seed)
        self._zone_data: Dict[str, Any] = zone_data or {}
        self._load_zone_data()
        self._templates: List[Dict] = INCIDENT_TEMPLATES
        self._template_by_id: Dict[str, Dict] = {t["id"]: t for t in self._templates}
        self._template_weights: List[float] = [t["weight"] for t in self._templates]
        self._total_weight: float = sum(self._template_weights)
        self.active_zone_ids: List[str]   = []
        self.active_incidents: Dict[str, Incident]  = {}   
        self.resolved_incidents: List[Incident]     = []
        self.active_mci_scenes: Dict[str, MCIScene] = {}
        self._call_counter:    int   = 0
        self._step_count:      int   = 0
        self._sim_time_min:    float = 480.0
        self._task_id:         int   = 1
        self._recent_calls: Dict[str, List[Tuple[float, float, float]]] = {}
    def _load_zone_data(self) -> None:
        if self._zone_data:
            return
        path = DATA_DIR / "city_zones.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for z in raw.get("zones", []):
                self._zone_data[z["zone_id"]] = z
    def reset(
        self,
        active_zone_ids: List[str],
        task_id:          int = 1,
        sim_time_minutes: float = 480.0,
        preload_incidents: Optional[List[Dict]] = None,
    ) -> List[Incident]:
        self.active_zone_ids     = active_zone_ids
        self._task_id            = task_id
        self._sim_time_min       = sim_time_minutes
        self._step_count         = 0
        self._call_counter       = 0
        self.active_incidents    = {}
        self.resolved_incidents  = []
        self.active_mci_scenes   = {}
        self._recent_calls       = {z: [] for z in active_zone_ids}
        self.rng    = random.Random(self.seed + task_id * 1000)
        self.np_rng = np.random.RandomState(self.seed + task_id * 1000)
        if preload_incidents:
            return self._load_fixed_incidents(preload_incidents)
        initial = self._spawn_initial_incidents()
        for inc in initial:
            self.active_incidents[inc.incident_id] = inc
        return initial
    def step(
        self,
        sim_time_minutes: float,
        traffic_od: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Incident]:
        self._sim_time_min = sim_time_minutes
        self._step_count  += 1
        self._age_incidents()
        self._auto_resolve_stale()
        new_incidents = self._spawn_step_incidents()
        new_incidents += self._spawn_mci_cluster_incidents()
        for inc in new_incidents:
            self.active_incidents[inc.incident_id] = inc
        return new_incidents
    def _spawn_initial_incidents(self) -> List[Incident]:
        incidents: List[Incident] = []
        task = self._task_id
        if task == 1:
            incidents.append(self._generate_single_incident(
                template_id="CARD_001", zone_id=self._pick_zone(),
            ))
        elif task == 2:
            incidents.append(self._generate_single_incident(
                template_id="TRAU_001", zone_id=self._pick_zone(),
            ))
        elif task == 3:
            incidents.append(self._generate_single_incident(
                template_id=self.rng.choice(["CARD_001", "STRO_001", "RESP_001"]),
                zone_id=self._pick_zone(),
            ))
        elif task == 4:
            n = self.rng.randint(5, 8)
            for _ in range(n):
                incidents.append(self._generate_random_incident())
        elif task == 5:
            n = self.rng.randint(3, 5)
            for _ in range(n):
                incidents.append(self._generate_random_incident())
        elif task == 6:
            pass
        elif task == 7:
            mci_type = self.rng.choice(list(MCIType))
            scene = self._spawn_mci_scene(
                mci_type=mci_type,
                zone_id=self._pick_zone(),
                victim_count=self.rng.randint(MCI_VICTIM_MIN, MCI_VICTIM_MAX),
            )
            incidents.extend(scene.victims)
        elif task == 8:
            n = self.rng.randint(4, 6)
            for _ in range(n):
                incidents.append(self._generate_single_incident(
                    template_id="ROUT_003", zone_id=self._pick_zone(),
                ))
        elif task == 9:
            for _ in range(3):
                mci_type = self.rng.choice(list(MCIType))
                zone = self._pick_zone()
                victim_count = self.rng.randint(15, 40)
                scene = self._spawn_mci_scene(mci_type, zone, victim_count)
                incidents.extend(scene.victims)
        return incidents
    def _spawn_step_incidents(self) -> List[Incident]:
        if len(self.active_incidents) >= MAX_ACTIVE_INCIDENTS:
            return []
        arrival_rates = {1: 0.0, 2: 0.0, 3: 0.0,   
                         4: 0.3, 5: 0.3, 6: 0.2,
                         7: 0.0, 8: 0.1, 9: 0.0}
        base_rate = arrival_rates.get(self._task_id, 0.2)
        hour = (self._sim_time_min / 60.0) % 24.0
        tod_mult = self._hour_demand_multiplier(hour)
        rate = base_rate * tod_mult
        n = self.np_rng.poisson(rate)
        new_incidents = []
        for _ in range(min(n, 3)):   
            inc = self._generate_random_incident()
            if not self._is_duplicate(inc):
                new_incidents.append(inc)
                self._register_recent_call(inc)
        return new_incidents
    def _spawn_mci_cluster_incidents(self) -> List[Incident]:
        secondaries: List[Incident] = []
        if not self.active_mci_scenes:
            return secondaries
        for scene in self.active_mci_scenes.values():
            if not scene.is_active:
                continue
            if self.rng.random() < CLUSTER_SPAWN_PROB:
                template = self.rng.choice([
                    t for t in self._templates
                    if t["priority"] in ("P2", "P3") and
                    t["category"] not in ("cardiac",)
                ])
                inc = self._generate_single_incident(
                    template_id=template["id"],
                    zone_id=scene.zone_id,
                )
                if not self._is_duplicate(inc):
                    secondaries.append(inc)
        return secondaries
    def _generate_random_incident(self) -> Incident:
        template = self.rng.choices(
            self._templates,
            weights=self._template_weights,
            k=1,
        )[0]
        zone_id = self._pick_zone_by_risk(template["category"])
        return self._generate_single_incident(template["id"], zone_id)
    def _generate_single_incident(
        self,
        template_id: str,
        zone_id: str,
    ) -> Incident:
        tpl   = self._template_by_id[template_id]
        zone  = self._zone_data.get(zone_id, {})
        zname = zone.get("name", zone_id)
        base_lat = zone.get("lat", 18.52)
        base_lon = zone.get("lon", 73.85)
        lat = base_lat + self.np_rng.normal(0, 0.015)
        lon = base_lon + self.np_rng.normal(0, 0.015)
        descs = tpl["descriptions"]
        desc  = self.rng.choice(descs)
        true_count = 1
        caller_count = true_count  
        true_prio = tpl["priority"]
        noise = self.rng.random()
        if noise < 0.10:   
            reported_prio = {"P1": "P2", "P2": "P3", "P3": "P3"}.get(true_prio, true_prio)
        elif noise < 0.18: 
            reported_prio = {"P3": "P2", "P2": "P1", "P1": "P1"}.get(true_prio, true_prio)
        else:
            reported_prio = true_prio
        agency = AgencyRequirement(tpl["agency"])
        flags  = self._agency_to_flags(agency)
        sp = tpl["survival"]
        survival_curve = SurvivalCurveParams(
            decay_kind          = sp["kind"],
            baseline_survival   = sp["baseline"],
            golden_hour_minutes = sp["golden_min"],
            min_survival        = sp["min_surv"],
            decay_rate          = sp["rate"],
            condition_code      = sp["code"],
        )
        rpm = self._generate_rpm(tpl)
        self._call_counter += 1
        call_num = f"INC-{self._call_counter:04d}"
        incident = Incident(
            incident_id    = str(uuid.uuid4())[:8],
            call_number    = call_num,
            zone_id        = zone_id,
            zone_name      = zname,
            lat            = round(lat, 5),
            lon            = round(lon, 5),
            symptom_description    = desc,
            caller_reported_count  = caller_count,
            reported_priority      = reported_prio,
            incident_category      = tpl["category"],
            agency_flags           = flags,
            is_mci                 = False,
            mci_type               = None,
            victim_count           = 1,
            status                 = IncidentStatus.REPORTED,
            reported_at_step       = self._step_count,
            dispatched_at_step     = None,
            resolved_at_step       = None,
            assigned_unit_ids      = [],
            assigned_hospital_id   = None,
            ground_truth_priority            = IncidentPriority(true_prio),
            ground_truth_start_tag           = None,
            ground_truth_unit_type           = UnitType(tpl["unit_type"]),
            ground_truth_hospital_specialty  = tpl["specialties"],
            ground_truth_agency              = agency,
            ground_truth_protocol            = tpl["protocols"],
            survival_curve                   = survival_curve,
            rpm_score                        = rpm,
        )
        return incident
    def _spawn_mci_scene(
        self,
        mci_type: MCIType,
        zone_id:  str,
        victim_count: int,
    ) -> MCIScene:
        scene_id = f"MCI-{str(uuid.uuid4())[:6].upper()}"
        dist     = MCI_VICTIM_DISTRIBUTIONS.get(
            mci_type.value,
            {"Immediate": 0.25, "Delayed": 0.35, "Minimal": 0.35, "Expectant": 0.05}
        )
        zone     = self._zone_data.get(zone_id, {})
        base_lat = zone.get("lat", 18.52)
        base_lon = zone.get("lon", 73.85)
        zname    = zone.get("name", zone_id)
        tags: List[StartTag] = []
        for tag_str, fraction in dist.items():
            n = max(0, round(fraction * victim_count))
            tags.extend([StartTag(tag_str)] * n)
        while len(tags) < victim_count:
            tags.append(StartTag.MINIMAL)
        self.rng.shuffle(tags)
        tags = tags[:victim_count]
        scene = MCIScene(
            scene_id       = scene_id,
            mci_type       = mci_type,
            zone_id        = zone_id,
            total_victims  = victim_count,
            spawned_at_step= self._step_count,
        )
        mci_tpl_map = {
            MCIType.ROAD_TRAFFIC_ACCIDENT:  "TRAU_001",
            MCIType.BUILDING_COLLAPSE:      "TRAU_005",
            MCIType.INDUSTRIAL_EXPLOSION:   "HAZM_001",
            MCIType.CHEMICAL_SPILL:         "HAZM_001",
            MCIType.STAMPEDE:               "TRAU_006",
            MCIType.TRAIN_ACCIDENT:         "TRAU_001",
            MCIType.FLOOD_RESCUE:           "WATR_001",
            MCIType.FIRE_MULTI_VICTIM:      "BURN_001",
            MCIType.FESTIVAL_CROWD:         "TRAU_006",
            MCIType.TERRORIST_ATTACK:       "TRAU_003",
        }
        base_tpl_id = mci_tpl_map.get(mci_type, "TRAU_001")
        base_tpl    = self._template_by_id[base_tpl_id]
        for i, tag in enumerate(tags):
            lat = base_lat + self.np_rng.normal(0, 0.006)
            lon = base_lon + self.np_rng.normal(0, 0.006)
            rpm  = self._generate_rpm_for_tag(tag)
            prio = rpm.priority
            victim_desc = self._generate_mci_victim_description(
                mci_type, tag, i + 1, victim_count
            )
            sp = base_tpl["survival"]
            tag_surv_mods = {
                StartTag.IMMEDIATE:  (0.00,  0.00),   
                StartTag.DELAYED:    (0.05, -0.005),
                StartTag.MINIMAL:    (0.08, -0.010),
                StartTag.EXPECTANT:  (-0.30, 0.020),
            }
            b_delta, r_delta = tag_surv_mods[tag]
            survival_curve = SurvivalCurveParams(
                decay_kind          = sp["kind"],
                baseline_survival   = max(0.05, sp["baseline"] + b_delta),
                golden_hour_minutes = sp["golden_min"],
                min_survival        = sp["min_surv"],
                decay_rate          = sp["rate"] + r_delta,
                condition_code      = sp["code"] + f"_MCI_{tag.value.upper()}",
            )
            self._call_counter += 1
            victim = Incident(
                incident_id   = f"{scene_id}-V{i+1:03d}",
                call_number   = f"INC-{self._call_counter:04d}",
                zone_id       = zone_id,
                zone_name     = zname,
                lat           = round(lat, 5),
                lon           = round(lon, 5),
                symptom_description   = victim_desc,
                caller_reported_count = victim_count,
                reported_priority     = self._tag_to_reported_priority(tag),
                incident_category     = base_tpl["category"],
                agency_flags          = self._mci_agency_flags(mci_type),
                is_mci                = True,
                mci_type              = mci_type.value,
                victim_count          = victim_count,
                status                = IncidentStatus.REPORTED,
                reported_at_step      = self._step_count,
                dispatched_at_step    = None,
                resolved_at_step      = None,
                assigned_unit_ids     = [],
                assigned_hospital_id  = None,
                ground_truth_priority           = prio,
                ground_truth_start_tag          = tag,
                ground_truth_unit_type          = self._start_tag_to_unit(tag),
                ground_truth_hospital_specialty = base_tpl["specialties"],
                ground_truth_agency             = self._mci_agency_requirement(mci_type),
                ground_truth_protocol           = base_tpl["protocols"],
                survival_curve                  = survival_curve,
                rpm_score                       = rpm,
                current_survival_prob           = survival_curve.baseline_survival,
            )
            scene.victims.append(victim)
        self.active_mci_scenes[scene_id] = scene
        return scene
    def _age_incidents(self) -> None:
        for inc in self.active_incidents.values():
            inc.time_since_reported_min += STEP_DURATION_MIN
            if inc.ground_truth_priority != IncidentPriority.P1:
                continue
            sc = inc.survival_curve
            t  = inc.time_since_reported_min
            if sc.decay_kind == DECAY_STEEP:
                if t < sc.golden_hour_minutes:
                    prob = sc.baseline_survival - sc.decay_rate * 0.2 * t
                else:
                    excess = t - sc.golden_hour_minutes
                    prob   = sc.baseline_survival * 0.5 - sc.decay_rate * 3.0 * excess
            elif sc.decay_kind == DECAY_LINEAR:
                prob = sc.baseline_survival - sc.decay_rate * t
            elif sc.decay_kind == DECAY_SIGMOID:
                mid  = sc.golden_hour_minutes
                k    = sc.decay_rate
                prob = sc.baseline_survival / (1 + math.exp(k * 60 * (t - mid) / mid))
            elif sc.decay_kind == DECAY_EXPONENTIAL:
                prob = sc.baseline_survival * math.exp(-sc.decay_rate * t)
            else:  
                prob = sc.baseline_survival - sc.decay_rate * t
            inc.current_survival_prob = max(sc.min_survival, min(1.0, prob))
            margin = sc.baseline_survival - sc.min_survival
            if margin > 0:
                lost   = sc.baseline_survival - inc.current_survival_prob
                inc.deterioration_level = min(1.0, lost / margin)
    def _auto_resolve_stale(self) -> None:
        to_resolve: List[str] = []
        for inc_id, inc in self.active_incidents.items():
            if inc.status == IncidentStatus.TRANSPORTING:
                if inc.time_since_reported_min > 45:
                    to_resolve.append(inc_id)
            elif inc.status == IncidentStatus.REPORTED:
                if inc.ground_truth_priority == IncidentPriority.P3:
                    if inc.time_since_reported_min > 180:
                        inc.status = IncidentStatus.CANCELLED
                        to_resolve.append(inc_id)
        for inc_id in to_resolve:
            inc = self.active_incidents.pop(inc_id)
            if inc.status != IncidentStatus.CANCELLED:
                inc.status = IncidentStatus.RESOLVED
                inc.resolved_at_step = self._step_count
            self.resolved_incidents.append(inc)
    def update_incident_status(
        self,
        incident_id: str,
        new_status:  IncidentStatus,
        unit_ids:    Optional[List[str]] = None,
        hospital_id: Optional[str] = None,
    ) -> Optional[Incident]:
        inc = self.active_incidents.get(incident_id)
        if inc is None:
            return None
        inc.status = new_status
        if unit_ids is not None:
            inc.assigned_unit_ids = unit_ids
        if hospital_id is not None:
            inc.assigned_hospital_id = hospital_id
        if new_status == IncidentStatus.DISPATCHED:
            inc.dispatched_at_step = self._step_count
        if new_status in (IncidentStatus.RESOLVED, IncidentStatus.CANCELLED):
            inc.resolved_at_step = self._step_count
            self.resolved_incidents.append(
                self.active_incidents.pop(incident_id)
            )
        return inc
    def escalate_to_mci(
        self,
        incident_id: str,
        mci_type:    MCIType,
        total_victims: int,
    ) -> Optional[MCIScene]:
        inc = self.active_incidents.get(incident_id)
        if inc is None:
            return None
        zone_id = inc.zone_id
        del self.active_incidents[incident_id]
        scene = self._spawn_mci_scene(mci_type, zone_id, total_victims)
        for v in scene.victims:
            self.active_incidents[v.incident_id] = v
        return scene
    def get_active_queue(
        self,
        zone_filter: Optional[List[str]] = None,
        priority_filter: Optional[List[str]] = None,
        status_filter: Optional[List[IncidentStatus]] = None,
    ) -> List[Incident]:
        incidents = list(self.active_incidents.values())
        if zone_filter:
            zset = set(zone_filter)
            incidents = [i for i in incidents if i.zone_id in zset]
        if priority_filter:
            pset = set(priority_filter)
            incidents = [i for i in incidents
                         if i.ground_truth_priority.value in pset]
        if status_filter:
            sset = set(status_filter)
            incidents = [i for i in incidents if i.status in sset]
        prio_order = {"P1": 0, "P2": 1, "P3": 2, "P0": 3}
        incidents.sort(
            key=lambda i: (
                prio_order.get(i.ground_truth_priority.value, 9),
                -i.deterioration_level,
                i.reported_at_step,
            )
        )
        return incidents
    def get_queue_observation(
        self,
        zone_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            inc.to_observation_dict()
            for inc in self.get_active_queue(zone_filter=zone_filter)
        ]
    def get_mci_scenes(self) -> List[Dict[str, Any]]:
        return [
            {
                "scene_id":       sc.scene_id,
                "mci_type":       sc.mci_type.value,
                "zone_id":        sc.zone_id,
                "total_victims":  sc.total_victims,
                "victim_breakdown": sc.victim_breakdown,
                "victims_pending": sum(
                    1 for v in sc.victims
                    if v.incident_id in self.active_incidents and
                       self.active_incidents[v.incident_id].status
                       not in (IncidentStatus.RESOLVED, IncidentStatus.CANCELLED)
                ),
                "is_active":      sc.is_active,
                "spawned_at_step":sc.spawned_at_step,
            }
            for sc in self.active_mci_scenes.values()
        ]
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        return self.active_incidents.get(incident_id)
    def get_incident_full_state(self, incident_id: str) -> Optional[Dict]:
        inc = self.active_incidents.get(incident_id)
        return inc.to_full_state_dict() if inc else None
    @property
    def queue_is_empty(self) -> bool:
        return len(self.active_incidents) == 0
    @property
    def total_active(self) -> int:
        return len(self.active_incidents)
    @property
    def p1_count(self) -> int:
        return sum(
            1 for i in self.active_incidents.values()
            if i.ground_truth_priority == IncidentPriority.P1
        )
    @property
    def mean_deterioration(self) -> float:
        vals = [i.deterioration_level for i in self.active_incidents.values()]
        return sum(vals) / max(len(vals), 1)
    def build_task1_scenario(self, seed_offset: int = 0) -> Incident:
        rng_bak = random.Random(self.seed + seed_offset)
        tpl_id  = rng_bak.choice([
            "CARD_001", "STRO_001", "TRAU_001", "RESP_001",
            "CARD_002", "RESP_003", "TRAU_002", "OBST_001",
        ])
        zone = rng_bak.choice(self.active_zone_ids) if self.active_zone_ids else "Z05"
        inc  = self._generate_single_incident(tpl_id, zone)
        self.active_incidents[inc.incident_id] = inc
        return inc
    def build_task4_scenario(self) -> List[Incident]:
        n_total = self.rng.randint(5, 8)
        p1_ids  = ["CARD_001", "CARD_002", "STRO_001", "TRAU_001"]
        p2_ids  = ["CARD_003", "TRAU_004", "NEUR_001", "MED_001"]
        p3_ids  = ["ROUT_001", "ROUT_002", "TRAU_006"]
        incidents = []
        for tid in self.rng.sample(p1_ids, 2):
            incidents.append(self._generate_single_incident(tid, self._pick_zone()))
        for _ in range(n_total - 2):
            tid = self.rng.choice(p2_ids + p3_ids)
            incidents.append(self._generate_single_incident(tid, self._pick_zone()))
        self.rng.shuffle(incidents)
        for inc in incidents:
            self.active_incidents[inc.incident_id] = inc
        return incidents
    def build_task7_mci_scenario(self) -> MCIScene:
        mci_type = self.rng.choice([
            MCIType.ROAD_TRAFFIC_ACCIDENT,
            MCIType.BUILDING_COLLAPSE,
            MCIType.INDUSTRIAL_EXPLOSION,
            MCIType.TRAIN_ACCIDENT,
        ])
        zone_id      = self._pick_zone()
        victim_count = self.rng.randint(MCI_VICTIM_MIN, MCI_VICTIM_MAX)
        scene        = self._spawn_mci_scene(mci_type, zone_id, victim_count)
        for v in scene.victims:
            self.active_incidents[v.incident_id] = v
        return scene
    def build_task9_surge_scenario(self) -> List[MCIScene]:
        scenes  = []
        zones_picked: Set[str] = set()
        mci_types = self.rng.sample(list(MCIType), 3)
        for mt in mci_types:
            zone = self._pick_zone()
            while zone in zones_picked and len(self.active_zone_ids) > 3:
                zone = self._pick_zone()
            zones_picked.add(zone)
            vc   = self.rng.randint(15, 40)
            sc   = self._spawn_mci_scene(mt, zone, vc)
            for v in sc.victims:
                self.active_incidents[v.incident_id] = v
            scenes.append(sc)
        return scenes
    def _generate_rpm(self, template: Dict) -> RPMScore:
        r = template.get("rpm_range", {})
        rr_lo, rr_hi = r.get("rr", (12, 22))
        rr = self.rng.randint(rr_lo, rr_hi)
        pulse_present = r.get("pulse", True)
        if isinstance(pulse_present, bool):
            p_present = pulse_present
        else:
            p_present = self.rng.random() > 0.1  
        hr_lo, hr_hi = r.get("hr", (70, 100))
        hr = self.rng.randint(hr_lo, hr_hi) if p_present else 0
        gcs_lo, gcs_hi = r.get("gcs", (13, 15))
        gcs = self.rng.randint(gcs_lo, gcs_hi)
        obeys = gcs >= 10
        cap_refill = self.np_rng.uniform(1.5, 3.5) if p_present else 4.5
        radial = cap_refill <= 2.0 or rr in range(10, 30)
        return RPMScore(
            respiration_rate     = rr,
            pulse_present        = p_present,
            pulse_rate           = hr,
            radial_pulse         = radial,
            obeys_commands       = obeys,
            capillary_refill_sec = round(cap_refill, 1),
            gcs                  = gcs,
        )
    def _generate_rpm_for_tag(self, tag: StartTag) -> RPMScore:
        spec = MCI_RPM_BY_TAG[tag.value]
        rr_options  = spec["rr_options"]
        chosen_rr   = self.rng.choices(
            rr_options, weights=[o[2] for o in rr_options], k=1
        )[0]
        rr = self.rng.randint(chosen_rr[0], max(chosen_rr[0], chosen_rr[1]))
        pulse_absent_prob = spec["pulse_absent_prob"]
        pulse_present = self.rng.random() > pulse_absent_prob
        hr_lo, hr_hi  = spec["pulse_range"]
        hr = self.rng.randint(hr_lo, max(hr_lo, hr_hi)) if pulse_present else 0
        obeys = self.rng.random() < spec["obeys_commands_prob"]
        cfr_lo, cfr_hi = spec["capillary_refill_range"]
        cap_refill = self.np_rng.uniform(cfr_lo, cfr_hi)
        radial = cap_refill <= 2.0
        gcs_lo, gcs_hi = spec["gcs_range"]
        gcs = self.rng.randint(gcs_lo, max(gcs_lo, gcs_hi))
        rpm = RPMScore(
            respiration_rate     = rr,
            pulse_present        = pulse_present,
            pulse_rate           = hr,
            radial_pulse         = radial,
            obeys_commands       = obeys,
            capillary_refill_sec = round(cap_refill, 1),
            gcs                  = gcs,
        )
        generated_tag = rpm.start_tag
        if generated_tag != tag:
            rpm = self._force_correct_rpm(rpm, tag)
        return rpm
    def _force_correct_rpm(self, rpm: RPMScore, target: StartTag) -> RPMScore:
        if target == StartTag.EXPECTANT:
            rpm.respiration_rate = 0
            rpm.pulse_present    = False
            rpm.pulse_rate       = 0
            rpm.is_ambulatory    = False
        elif target == StartTag.IMMEDIATE:
            rpm.is_ambulatory    = False
            rpm.respiration_rate = self.rng.choice([5, 35])
            rpm.capillary_refill_sec = 3.5
            rpm.radial_pulse         = False
        elif target == StartTag.MINIMAL:
            rpm.is_ambulatory        = True   
            rpm.respiration_rate     = self.rng.randint(12, 22)
            rpm.pulse_present        = True
            rpm.pulse_rate           = self.rng.randint(65, 100)
            rpm.obeys_commands       = True
            rpm.capillary_refill_sec = 1.5
            rpm.radial_pulse         = True
        elif target == StartTag.DELAYED:
            rpm.is_ambulatory        = False  
            rpm.respiration_rate     = self.rng.randint(10, 29)
            rpm.pulse_present        = True
            rpm.capillary_refill_sec = 1.8
            rpm.radial_pulse         = True
            rpm.obeys_commands       = True
        return rpm
    def _generate_mci_victim_description(
        self,
        mci_type: MCIType,
        tag:      StartTag,
        victim_n: int,
        total:    int,
    ) -> str:
        scene_descriptions = {
            MCIType.ROAD_TRAFFIC_ACCIDENT: f"Major road accident scene — {total} affected. Vehicle {victim_n}:",
            MCIType.BUILDING_COLLAPSE:     f"Building collapse site — {total} trapped/injured. Victim {victim_n}:",
            MCIType.INDUSTRIAL_EXPLOSION:  f"Factory explosion — {total} casualties. Casualty {victim_n}:",
            MCIType.STAMPEDE:              f"Stampede scene — {total} victims. Person {victim_n}:",
            MCIType.TRAIN_ACCIDENT:        f"Train derailment — {total} injured. Passenger {victim_n}:",
            MCIType.CHEMICAL_SPILL:        f"Chemical spill incident — {total} exposed. Victim {victim_n}:",
            MCIType.FLOOD_RESCUE:          f"Flood rescue — {total} rescued. Survivor {victim_n}:",
            MCIType.FIRE_MULTI_VICTIM:     f"Multi-victim fire scene — {total} affected. Patient {victim_n}:",
            MCIType.FESTIVAL_CROWD:        f"Festival crowd incident — {total} casualties. Person {victim_n}:",
            MCIType.TERRORIST_ATTACK:      f"Mass casualty event — {total} victims. Victim {victim_n}:",
        }
        prefix = scene_descriptions.get(mci_type, f"MCI scene — victim {victim_n}/{total}:")
        tag_descriptions = {
            StartTag.IMMEDIATE:  "NOT breathing/resp >30 or capillary refill >2 sec or not following commands. IMMEDIATE priority.",
            StartTag.DELAYED:    "Breathing 10-29/min, pulse present, capillary refill <2 sec, follows commands. DELAYED.",
            StartTag.MINIMAL:    "Walking wounded, minor injuries, can follow commands. MINIMAL.",
            StartTag.EXPECTANT:  "Not breathing after airway repositioning, or unsurvivable injuries given current resources. EXPECTANT.",
        }
        return f"{prefix} {tag_descriptions[tag]}"
    def _pick_zone(self) -> str:
        if not self.active_zone_ids:
            return "Z05"
        return self.rng.choice(self.active_zone_ids)
    def _pick_zone_by_risk(self, category: str) -> str:
        if not self.active_zone_ids:
            return "Z05"
        cat_risk_key = {
            "cardiac":    "cardiac_arrest_rate_per_100k",
            "trauma":     "trauma_rate_per_100k",
            "stroke":     "cardiac_arrest_rate_per_100k",
            "burns":      "industrial_hazard_risk",
            "industrial": "industrial_hazard_risk",
            "respiratory":"trauma_rate_per_100k",
        }
        risk_key = cat_risk_key.get(category, "call_volume_per_day")
        weights: List[float] = []
        for zid in self.active_zone_ids:
            z    = self._zone_data.get(zid, {})
            risk = z.get("risk_profile", {})
            if risk_key == "industrial_hazard_risk":
                val = {"very_high": 4, "high": 3, "medium": 2, "low": 1, "very_low": 0.5}.get(
                    str(risk.get(risk_key, "low")), 1.0
                )
            else:
                val = float(risk.get(risk_key, 10.0))
            density = float(z.get("population_density_per_sqkm", 100)) / 1000.0
            weights.append(max(val * density, 0.1))
        return self.rng.choices(self.active_zone_ids, weights=weights, k=1)[0]
    def _hour_demand_multiplier(self, hour: float) -> float:
        if 8 <= hour < 10 or 17 <= hour < 19:
            return 1.6
        if 6 <= hour < 8 or 10 <= hour < 12 or 19 <= hour < 21:
            return 1.2
        if 0 <= hour < 5:
            return 0.5
        return 1.0
    def _agency_to_flags(self, agency: AgencyRequirement) -> List[str]:
        flag_map = {
            AgencyRequirement.EMS_ONLY:   [],
            AgencyRequirement.POLICE_EMS: ["police"],
            AgencyRequirement.FIRE_EMS:   ["fire"],
            AgencyRequirement.ALL_THREE:  ["police", "fire"],
            AgencyRequirement.HAZMAT:     ["fire", "hazmat"],
            AgencyRequirement.COAST_GUARD:["coast_guard"],
        }
        return flag_map.get(agency, [])
    def _mci_agency_flags(self, mci_type: MCIType) -> List[str]:
        heavy = {MCIType.BUILDING_COLLAPSE, MCIType.INDUSTRIAL_EXPLOSION,
                 MCIType.CHEMICAL_SPILL, MCIType.TERRORIST_ATTACK,
                 MCIType.TRAIN_ACCIDENT}
        police = {MCIType.STAMPEDE, MCIType.FESTIVAL_CROWD, MCIType.TERRORIST_ATTACK}
        flags = ["fire"] if mci_type in heavy else []
        if mci_type in police:
            flags.append("police")
        return list(set(flags))
    def _mci_agency_requirement(self, mci_type: MCIType) -> AgencyRequirement:
        all_three = {MCIType.BUILDING_COLLAPSE, MCIType.INDUSTRIAL_EXPLOSION,
                     MCIType.TERRORIST_ATTACK, MCIType.TRAIN_ACCIDENT}
        fire_ems  = {MCIType.FIRE_MULTI_VICTIM, MCIType.CHEMICAL_SPILL, MCIType.FLOOD_RESCUE}
        police    = {MCIType.STAMPEDE, MCIType.FESTIVAL_CROWD}
        if mci_type in all_three:
            return AgencyRequirement.ALL_THREE
        if mci_type in fire_ems:
            return AgencyRequirement.FIRE_EMS
        if mci_type in police:
            return AgencyRequirement.POLICE_EMS
        return AgencyRequirement.EMS_ONLY
    def _start_tag_to_unit(self, tag: StartTag) -> UnitType:
        return {
            StartTag.IMMEDIATE: UnitType.MICU,
            StartTag.DELAYED:   UnitType.ALS,
            StartTag.MINIMAL:   UnitType.BLS,
            StartTag.EXPECTANT: UnitType.BLS,
        }[tag]
    def _tag_to_reported_priority(self, tag: StartTag) -> str:
        return {
            StartTag.IMMEDIATE: "P1",
            StartTag.DELAYED:   "P2",
            StartTag.MINIMAL:   "P3",
            StartTag.EXPECTANT: "P0",
        }[tag]
    def _is_duplicate(self, inc: Incident) -> bool:
        recent = self._recent_calls.get(inc.zone_id, [])
        for (lat, lon, t) in recent:
            if abs(self._sim_time_min - t) < DUPLICATE_MERGE_WINDOW:
                dist = self._approx_dist_km(inc.lat, inc.lon, lat, lon)
                if dist < DUPLICATE_MERGE_RADIUS:
                    inc.is_duplicate = True
                    return True
        return False
    def _register_recent_call(self, inc: Incident) -> None:
        cache = self._recent_calls.setdefault(inc.zone_id, [])
        cache.append((inc.lat, inc.lon, self._sim_time_min))
        if len(cache) > 10:
            self._recent_calls[inc.zone_id] = cache[-10:]
    def _load_fixed_incidents(self, specs: List[Dict]) -> List[Incident]:
        incidents = []
        for spec in specs:
            inc = self._generate_single_incident(
                spec["template_id"], spec.get("zone_id", self._pick_zone())
            )
            incidents.append(inc)
            self.active_incidents[inc.incident_id] = inc
        return incidents
    @staticmethod
    def _approx_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dlat = (lat1 - lat2) * 111.0
        dlon = (lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
        return math.sqrt(dlat**2 + dlon**2)
    def describe(self) -> Dict[str, Any]:
        queue = self.get_active_queue()
        p_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for inc in queue:
            p_counts[inc.ground_truth_priority.value] += 1
        return {
            "step_count":       self._step_count,
            "sim_time_min":     self._sim_time_min,
            "task_id":          self._task_id,
            "active_incidents": self.total_active,
            "resolved_total":   len(self.resolved_incidents),
            "priority_breakdown": p_counts,
            "mci_scenes":       len(self.active_mci_scenes),
            "mean_deterioration": round(self.mean_deterioration, 3),
            "queue_empty":      self.queue_is_empty,
        }
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("=" * 65)
    print("EMERGI-ENV  ·  IncidentEngine smoke-test")
    print("=" * 65)
    engine = IncidentEngine(seed=42)
    active_zones = ["Z01", "Z02", "Z03", "Z04", "Z05", "Z06",
                    "Z07", "Z08", "Z09", "Z13", "Z14", "Z27"]
    engine.reset(active_zones, task_id=1, sim_time_minutes=480)
    inc1 = engine.build_task1_scenario()
    print(f"\nTask 1 — single incident:")
    print(f"  ID:       {inc1.incident_id}")
    print(f"  Category: {inc1.incident_category}")
    print(f"  Symptom:  {inc1.symptom_description[:70]}…")
    print(f"  GT Prio:  {inc1.ground_truth_priority.value}")
    print(f"  GT Unit:  {inc1.ground_truth_unit_type.value}")
    print(f"  GT Spec:  {inc1.ground_truth_hospital_specialty}")
    print(f"  RPM tag:  {inc1.rpm_score.start_tag.value if inc1.rpm_score else 'N/A'}")
    engine.reset(active_zones, task_id=4, sim_time_minutes=510)
    incidents4 = engine.build_task4_scenario()
    print(f"\nTask 4 — {len(incidents4)} simultaneous incidents:")
    for inc in incidents4:
        print(f"  {inc.call_number:>10} | {inc.incident_category:<14} | "
              f"{inc.ground_truth_priority.value} | {inc.ground_truth_unit_type.value}")
    print(f"  Describe: {engine.describe()}")
    engine.reset(active_zones, task_id=7, sim_time_minutes=540)
    scene7 = engine.build_task7_mci_scenario()
    print(f"\nTask 7 — MCI scene '{scene7.scene_id}':")
    print(f"  Type:      {scene7.mci_type.value}")
    print(f"  Victims:   {scene7.total_victims}")
    print(f"  Breakdown: {scene7.victim_breakdown}")
    for v in scene7.victims[:5]:
        rpm = v.rpm_score
        if rpm:
            gen_tag = rpm.start_tag
            expected = v.ground_truth_start_tag
            match = "✓" if gen_tag == expected else "✗"
            print(f"  [{match}] Victim {v.incident_id[-3:]} | expected={expected.value} "
                  f"| generated={gen_tag.value} | RR={rpm.respiration_rate} GCS={rpm.gcs}")
    engine.reset(active_zones, task_id=4, sim_time_minutes=480)
    engine.build_task4_scenario()
    print(f"\nStepping engine 20 steps:")
    for s in range(20):
        new = engine.step(480 + s * 3.0)
        if new:
            print(f"  Step {s+1:02d}: {len(new)} new incident(s) spawned")
    print(f"  Final state: {engine.describe()}")
    engine.reset(active_zones, task_id=9, sim_time_minutes=600)
    scenes9 = engine.build_task9_surge_scenario()
    print(f"\nTask 9 — {len(scenes9)} simultaneous MCI scenes:")
    for sc in scenes9:
        print(f"  {sc.scene_id} | {sc.mci_type.value} | zone={sc.zone_id} "
              f"| victims={sc.total_victims} | {sc.victim_breakdown}")
    print("\n✅  IncidentEngine smoke-test PASSED")