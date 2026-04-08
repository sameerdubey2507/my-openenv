from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generator,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    class _NumpyStub:
        def __getattr__(self, _: str) -> Any:
            raise ImportError("numpy is required. pip install numpy")
    np = _NumpyStub()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("emergi.scenario_gen")

GENERATOR_VERSION = "1.0.0"
SCHEMA_VERSION    = 3
SEED_RANGE        = (42, 9999)
DEFAULT_SEEDS_PER_TASK = 8
MAX_SEEDS_PER_TASK     = 500

TASK_META: Dict[int, Dict[str, Any]] = {
    1: {
        "name":        "Single Call Triage & Dispatch",
        "difficulty":  "easy",
        "baseline":    0.61,
        "max_steps":   10,
        "min_incidents": 1,
        "max_incidents": 1,
        "min_ambulances": 4,
        "max_ambulances": 6,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "valid_actions": ["dispatch", "noop"],
        "grader_weights": {"triage_class": 0.4, "unit_type": 0.3, "hospital_match": 0.3},
    },
    2: {
        "name":        "Hospital Route Selection",
        "difficulty":  "easy",
        "baseline":    0.72,
        "max_steps":   10,
        "min_incidents": 1,
        "max_incidents": 1,
        "min_ambulances": 4,
        "max_ambulances": 6,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "valid_actions": ["dispatch", "noop"],
        "grader_weights": {"specialty_match": 0.5, "capacity_check": 0.3, "travel_time": 0.2},
    },
    3: {
        "name":        "Unit Type Matching",
        "difficulty":  "easy",
        "baseline":    0.68,
        "max_steps":   5,
        "min_incidents": 1,
        "max_incidents": 1,
        "min_ambulances": 6,
        "max_ambulances": 8,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "valid_actions": ["dispatch", "noop"],
        "grader_weights": {"exact_unit_match": 1.0},
    },
    4: {
        "name":        "Multi-Incident Queue Prioritisation",
        "difficulty":  "medium",
        "baseline":    0.44,
        "max_steps":   20,
        "min_incidents": 5,
        "max_incidents": 8,
        "min_ambulances": 3,
        "max_ambulances": 3,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "valid_actions": ["dispatch", "escalate", "crew_swap", "noop"],
        "grader_weights": {"weighted_survival": 0.7, "delay_penalty": 0.3},
    },
    5: {
        "name":        "Dynamic Rerouting",
        "difficulty":  "medium",
        "baseline":    0.38,
        "max_steps":   25,
        "min_incidents": 3,
        "max_incidents": 5,
        "min_ambulances": 4,
        "max_ambulances": 5,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "traffic_disruption": True,
        "hospital_diversion": True,
        "valid_actions": ["dispatch", "reroute", "hospital_bypass", "crew_swap", "noop"],
        "grader_weights": {"reroute_correctness": 0.5, "net_time_saved": 0.5},
    },
    6: {
        "name":        "Pre-Positioning (Demand Forecast)",
        "difficulty":  "medium",
        "baseline":    0.42,
        "max_steps":   15,
        "min_incidents": 0,
        "max_incidents": 0,
        "min_ambulances": 6,
        "max_ambulances": 8,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "demand_forecast": True,
        "valid_actions": ["preposition", "noop"],
        "grader_weights": {"avg_response_time_vs_baseline": 1.0},
    },
    7: {
        "name":        "Mass Casualty Incident (MCI)",
        "difficulty":  "hard",
        "baseline":    0.29,
        "max_steps":   40,
        "min_incidents": 20,
        "max_incidents": 40,
        "min_ambulances": 6,
        "max_ambulances": 10,
        "mci":         True,
        "surge":       False,
        "comm_failure": False,
        "start_triage": True,
        "valid_actions": ["dispatch", "tag", "escalate", "crew_swap", "request_mutual_aid", "noop"],
        "grader_weights": {"start_accuracy": 0.4, "response": 0.3, "hospital_spread": 0.3},
        "wrong_tag_penalty": -0.5,
    },
    8: {
        "name":        "Inter-Hospital Transfer Cascade",
        "difficulty":  "hard",
        "baseline":    0.24,
        "max_steps":   35,
        "min_incidents": 4,
        "max_incidents": 8,
        "min_ambulances": 4,
        "max_ambulances": 6,
        "mci":         False,
        "surge":       False,
        "comm_failure": False,
        "icu_transfers": True,
        "valid_actions": ["transfer", "reroute", "hospital_bypass", "crew_swap", "noop"],
        "grader_weights": {"transfer_appropriateness": 0.4, "timing": 0.3, "utilisation": 0.3},
    },
    9: {
        "name":        "City-Wide Surge (3× MCI + Comms Failure)",
        "difficulty":  "hard",
        "baseline":    0.17,
        "max_steps":   60,
        "min_incidents": 40,
        "max_incidents": 80,
        "min_ambulances": 8,
        "max_ambulances": 12,
        "mci":         True,
        "surge":       True,
        "comm_failure": True,
        "comm_failure_prob": 0.12,
        "mutual_aid_required": True,
        "valid_actions": [
            "dispatch", "reroute", "escalate", "tag", "transfer",
            "request_mutual_aid", "preposition", "crew_swap",
            "declare_surge", "hospital_bypass", "noop",
        ],
        "grader_weights": {
            "system_survival": 0.5,
            "cascade_avoidance": 0.3,
            "mutual_aid_efficiency": 0.2,
        },
    },
}

ZONES: List[Dict[str, Any]] = [
    {"zone_id": "Z01", "name": "Shivajinagar-Deccan",    "type": "residential_dense", "lat": 18.5308, "lon": 73.8475, "population": 210000},
    {"zone_id": "Z02", "name": "Koregaon Park-Kalyani",  "type": "mixed_commercial",  "lat": 18.5362, "lon": 73.8937, "population": 155000},
    {"zone_id": "Z03", "name": "Hadapsar-Mundhwa",       "type": "industrial",        "lat": 18.5018, "lon": 73.9280, "population": 320000},
    {"zone_id": "Z04", "name": "Kothrud-Warje",          "type": "residential_dense", "lat": 18.5074, "lon": 73.8063, "population": 280000},
    {"zone_id": "Z05", "name": "Shivajinagar Station",   "type": "commercial_hub",    "lat": 18.5236, "lon": 73.8738, "population": 190000},
    {"zone_id": "Z06", "name": "Yerawada-Nagar Road",    "type": "mixed_commercial",  "lat": 18.5534, "lon": 73.9050, "population": 240000},
    {"zone_id": "Z07", "name": "Pimpri-Chinchwad",       "type": "industrial",        "lat": 18.6298, "lon": 73.7997, "population": 420000},
    {"zone_id": "Z08", "name": "Katraj-Kondhwa",         "type": "residential_dense", "lat": 18.4528, "lon": 73.8614, "population": 260000},
    {"zone_id": "Z09", "name": "Baner-Balewadi",         "type": "mixed_commercial",  "lat": 18.5590, "lon": 73.7760, "population": 175000},
    {"zone_id": "Z10", "name": "Wadgaonsheri",           "type": "mixed_commercial",  "lat": 18.5625, "lon": 73.9300, "population": 145000},
    {"zone_id": "Z11", "name": "Bhosari-Talawade",       "type": "industrial",        "lat": 18.6450, "lon": 73.8400, "population": 185000},
    {"zone_id": "Z12", "name": "Sinhagad-Dhayari",      "type": "peri_urban",         "lat": 18.4450, "lon": 73.7900, "population": 120000},
]

ZONE_IDS      = [z["zone_id"] for z in ZONES]
ZONE_LOOKUP   = {z["zone_id"]: z for z in ZONES}

ZONE_ADJACENCY: Dict[str, List[str]] = {
    "Z01": ["Z02", "Z04", "Z05", "Z09"],
    "Z02": ["Z01", "Z05", "Z06", "Z10"],
    "Z03": ["Z05", "Z06", "Z08", "Z10"],
    "Z04": ["Z01", "Z05", "Z09", "Z12"],
    "Z05": ["Z01", "Z02", "Z03", "Z04", "Z06", "Z08"],
    "Z06": ["Z02", "Z03", "Z05", "Z10", "Z11"],
    "Z07": ["Z09", "Z11"],
    "Z08": ["Z03", "Z05", "Z12"],
    "Z09": ["Z01", "Z04", "Z07", "Z11"],
    "Z10": ["Z02", "Z03", "Z06", "Z11"],
    "Z11": ["Z06", "Z07", "Z09", "Z10"],
    "Z12": ["Z04", "Z08"],
}

HOSPITALS: List[Dict[str, Any]] = [
    {
        "hospital_id": "H01", "name": "Sassoon General Hospital",
        "zone_id": "Z05", "tier": "level_1_trauma", "type": "government_tertiary",
        "er_beds": 80, "icu_beds": 120, "total_beds": 1400,
        "specialties": ["trauma_centre","cardiac_cath_lab","stroke_unit","neurosurgery",
                        "burns_unit","toxicology","obstetrics","paediatric_emergency"],
        "on_diversion": False, "er_occupancy": 0.72, "icu_occupancy": 0.85,
        "has_helipad": True, "lat": 18.5236, "lon": 73.8738,
    },
    {
        "hospital_id": "H02", "name": "KEM Hospital Rasta Peth",
        "zone_id": "Z05", "tier": "level_2_trauma", "type": "government_secondary",
        "er_beds": 40, "icu_beds": 60, "total_beds": 650,
        "specialties": ["trauma_centre","obstetrics","paediatric_emergency","toxicology"],
        "on_diversion": False, "er_occupancy": 0.68, "icu_occupancy": 0.78,
        "has_helipad": False, "lat": 18.5195, "lon": 73.8600,
    },
    {
        "hospital_id": "H03", "name": "Ruby Hall Clinic",
        "zone_id": "Z02", "tier": "level_2_trauma", "type": "private_tertiary",
        "er_beds": 35, "icu_beds": 80, "total_beds": 500,
        "specialties": ["cardiac_cath_lab","stroke_unit","neurosurgery","cardiac_surgery"],
        "on_diversion": False, "er_occupancy": 0.65, "icu_occupancy": 0.70,
        "has_helipad": True, "lat": 18.5362, "lon": 73.8937,
    },
    {
        "hospital_id": "H04", "name": "Deenanath Mangeshkar Hospital",
        "zone_id": "Z04", "tier": "level_2_trauma", "type": "private_tertiary",
        "er_beds": 30, "icu_beds": 90, "total_beds": 600,
        "specialties": ["cardiac_cath_lab","neurosurgery","transplant_centre","renal_dialysis"],
        "on_diversion": False, "er_occupancy": 0.58, "icu_occupancy": 0.75,
        "has_helipad": False, "lat": 18.5074, "lon": 73.8063,
    },
    {
        "hospital_id": "H05", "name": "PCMC Aundh Civil Hospital",
        "zone_id": "Z09", "tier": "level_2_trauma", "type": "government_secondary",
        "er_beds": 25, "icu_beds": 40, "total_beds": 350,
        "specialties": ["trauma_centre","obstetrics","toxicology"],
        "on_diversion": False, "er_occupancy": 0.80, "icu_occupancy": 0.88,
        "has_helipad": False, "lat": 18.5590, "lon": 73.7760,
    },
    {
        "hospital_id": "H06", "name": "Yerawada District Hospital",
        "zone_id": "Z06", "tier": "level_3_community", "type": "government_district",
        "er_beds": 20, "icu_beds": 20, "total_beds": 200,
        "specialties": ["obstetrics","paediatric_emergency"],
        "on_diversion": False, "er_occupancy": 0.75, "icu_occupancy": 0.80,
        "has_helipad": False, "lat": 18.5534, "lon": 73.9050,
    },
    {
        "hospital_id": "H07", "name": "Pimpri-Chinchwad Municipal Hospital",
        "zone_id": "Z07", "tier": "level_2_trauma", "type": "government_secondary",
        "er_beds": 30, "icu_beds": 45, "total_beds": 400,
        "specialties": ["trauma_centre","burns_unit","toxicology","obstetrics"],
        "on_diversion": False, "er_occupancy": 0.70, "icu_occupancy": 0.82,
        "has_helipad": False, "lat": 18.6298, "lon": 73.7997,
    },
    {
        "hospital_id": "H08", "name": "Inlaks & Budhrani Hospital",
        "zone_id": "Z03", "tier": "level_2_trauma", "type": "private_tertiary",
        "er_beds": 25, "icu_beds": 55, "total_beds": 350,
        "specialties": ["cardiac_cath_lab","stroke_unit","renal_dialysis","neurosurgery"],
        "on_diversion": False, "er_occupancy": 0.62, "icu_occupancy": 0.72,
        "has_helipad": True, "lat": 18.5018, "lon": 73.9280,
    },
]
HOSPITAL_IDS    = [h["hospital_id"] for h in HOSPITALS]
HOSPITAL_LOOKUP = {h["hospital_id"]: h for h in HOSPITALS}

UNIT_TYPES = ["BLS", "ALS", "MICU"]

SEVERITY_LEVELS  = ["P1", "P2", "P3"]
START_TAGS       = ["Immediate", "Delayed", "Minimal", "Expectant"]
P1_START_TAGS    = ["Immediate"]
P2_START_TAGS    = ["Delayed"]
P3_START_TAGS    = ["Minimal"]
P0_START_TAGS    = ["Expectant"]

MICU_CONDITIONS = [
    "stemi_anterior", "stemi_inferior", "stemi_with_vf_arrest", "cardiac_arrest_vf",
    "severe_tbi", "polytrauma_blunt", "polytrauma_penetrating", "anaphylaxis_severe",
    "status_epilepticus_refractory", "tension_pneumothorax", "ruptured_aaa",
    "eclampsia_severe", "septic_shock", "respiratory_failure_copd",
    "blast_injury", "crush_syndrome",
]
ALS_CONDITIONS = [
    "ischemic_stroke", "hemorrhagic_stroke_sah", "stemi_cocaine",
    "obstructed_airway_adult", "major_haemorrhage", "preeclampsia_severe",
    "acute_asthma_severe", "diabetic_ketoacidosis", "acute_psychosis_violence",
    "chest_trauma", "splenic_laceration", "gsw_shoulder", "heat_stroke",
    "carbon_monoxide", "paediatric_febrile_seizure",
]
BLS_CONDITIONS = [
    "fracture_lower_limb", "soft_tissue_injury", "mild_laceration",
    "syncope_vasovagal", "mild_allergic_reaction", "anxiety_panic_attack",
    "minor_burns_superficial", "nausea_vomiting_dehydration",
    "urinary_tract_infection_elderly", "minor_head_injury_gcs15",
]

SPECIALTY_CONDITIONS: Dict[str, str] = {
    "stemi_anterior": "cardiac_cath_lab",
    "stemi_inferior": "cardiac_cath_lab",
    "stemi_with_vf_arrest": "cardiac_cath_lab",
    "cardiac_arrest_vf": "cardiac_cath_lab",
    "stemi_cocaine": "cardiac_cath_lab",
    "ischemic_stroke": "stroke_unit",
    "hemorrhagic_stroke_sah": "stroke_unit",
    "ischemic_stroke_wake_up": "stroke_unit",
    "severe_tbi": "neurosurgery",
    "skull_base_fracture": "neurosurgery",
    "polytrauma_blunt": "trauma_centre",
    "polytrauma_penetrating": "trauma_centre",
    "blast_injury": "trauma_centre",
    "crush_syndrome": "trauma_centre",
    "mci_rta": "trauma_centre",
    "gsw_shoulder": "trauma_centre",
    "chest_trauma": "trauma_centre",
    "eclampsia_severe": "obstetrics",
    "preeclampsia_severe": "obstetrics",
    "obstetric_haemorrhage": "obstetrics",
    "paediatric_febrile_seizure": "paediatric_emergency",
    "paediatric_stroke": "paediatric_emergency",
    "carbon_monoxide": "toxicology",
    "organophosphate_poisoning": "toxicology",
    "anaphylaxis_severe": "trauma_centre",
    "ruptured_aaa": "trauma_centre",
    "septic_shock": "trauma_centre",
    "major_haemorrhage": "trauma_centre",
    "burns_major": "burns_unit",
    "burns_inhalation": "burns_unit",
}

PEAK_MORNING = (8, 10)
PEAK_EVENING = (17, 19)

DAYS_OF_WEEK = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "public_holiday",
]
SEASONS = ["summer", "monsoon", "winter", "spring"]

WEATHER_PROFILES: Dict[str, Dict[str, Any]] = {
    "clear":        {"traffic_mult": 1.0, "incident_mult": 1.0, "comm_degradation": 0.0},
    "heavy_rain":   {"traffic_mult": 1.45, "incident_mult": 1.30, "comm_degradation": 0.08},
    "fog":          {"traffic_mult": 1.35, "incident_mult": 1.15, "comm_degradation": 0.05},
    "flood":        {"traffic_mult": 2.10, "incident_mult": 1.80, "comm_degradation": 0.15},
    "extreme_heat": {"traffic_mult": 1.05, "incident_mult": 1.40, "comm_degradation": 0.02},
    "dust_storm":   {"traffic_mult": 1.20, "incident_mult": 1.25, "comm_degradation": 0.12},
}

INCIDENT_TEMPLATE_POOL: List[Dict[str, Any]] = [
    {
        "template_id": "T001", "condition": "stemi_anterior",
        "call_description": "58-year-old male, sudden severe crushing chest pain radiating to left arm and jaw, started 25 minutes ago. Sweating profusely. BP 90/60, pulse 48 irregular. Wife says he looks grey.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "cardiac_cath_lab",
        "survival_curve": "sigmoid", "golden_hour_min": 90,
        "rpm": {"respirations": "normal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T002", "condition": "cardiac_arrest_vf",
        "call_description": "42-year-old male collapsed in office. Unresponsive, not breathing. Co-worker performing CPR. Bystander AED arrived — shocked once, no pulse. Duration unknown.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "cardiac_cath_lab",
        "survival_curve": "exponential", "golden_hour_min": 30,
        "rpm": {"respirations": "absent", "pulse": "absent", "mental_status": "unresponsive"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T003", "condition": "polytrauma_blunt",
        "call_description": "RTA on Pune-Nashik highway. 29-year-old male vs truck, unbelted. GCS 10, BP 80/50, HR 128, distended abdomen, bilateral femur deformity. Multiple bystanders attempting to move victim.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "trauma_centre",
        "survival_curve": "linear", "golden_hour_min": 60,
        "rpm": {"respirations": "normal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": True, "trapped": False, "hazmat": False,
        "agencies_required": ["Police"],
    },
    {
        "template_id": "T004", "condition": "severe_tbi",
        "call_description": "Construction worker fell from 4th floor scaffolding. Unresponsive, GCS 6. Bilateral periorbital haematoma. Right pupil fixed and dilated. BP 170/100 rising (Cushing reflex).",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "neurosurgery",
        "survival_curve": "step_cliff", "golden_hour_min": 60,
        "rpm": {"respirations": "normal", "pulse": "present_strong", "mental_status": "unresponsive"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T005", "condition": "anaphylaxis_severe",
        "call_description": "22-year-old female stung by wasp at Empress Garden. Throat closing, cannot speak, stridor audible over phone. Urticaria spreading. SpO2 falling rapidly.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "trauma_centre",
        "survival_curve": "exponential", "golden_hour_min": 20,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T006", "condition": "eclampsia_severe",
        "call_description": "32-week pregnant 26-year-old. Generalised tonic-clonic seizures ongoing 4 min. BP 180/120, severe headache before onset. Foetal movements reduced.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "obstetrics",
        "survival_curve": "sigmoid", "golden_hour_min": 60,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_strong", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T007", "condition": "blast_injury",
        "call_description": "Explosion at Hadapsar chemical plant. Multiple victims. 35-year-old male, blast wave and fragment injuries. Burns to 40% BSA. Tympanic membrane rupture. GCS 9.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "trauma_centre",
        "survival_curve": "linear", "golden_hour_min": 60,
        "rpm": {"respirations": "normal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": True, "trapped": False, "hazmat": True,
        "agencies_required": ["Police", "Fire"],
    },
    {
        "template_id": "T008", "condition": "septic_shock",
        "call_description": "68-year-old diabetic female. Fever 40.1°C, BP 75/40, HR 138, RR 28, confused. Indwelling urinary catheter. Caregiver reports 2-day history of confusion and reduced urine output.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "trauma_centre",
        "survival_curve": "exponential", "golden_hour_min": 60,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T009", "condition": "ischemic_stroke",
        "call_description": "72-year-old male, sudden onset right-sided weakness and inability to speak. Wife says it started exactly 55 minutes ago. Atrial fibrillation known, on anticoagulants.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "ALS",
        "required_specialty": "stroke_unit",
        "survival_curve": "sigmoid", "golden_hour_min": 270,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T010", "condition": "hemorrhagic_stroke_sah",
        "call_description": "49-year-old female, 'worst headache of my life' — thunderclap onset. Now confused, photophobic. GCS 12, BP 200/110. History of polycystic kidney disease.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "ALS",
        "required_specialty": "stroke_unit",
        "survival_curve": "exponential", "golden_hour_min": 120,
        "rpm": {"respirations": "normal", "pulse": "present_strong", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T011", "condition": "major_haemorrhage",
        "call_description": "20-year-old male, stabbed in left flank at Shivajinagar. Active arterial bleeding from abdomen. BP 70/40, HR 145, GCS 13. Bystanders applying pressure.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "ALS",
        "required_specialty": "trauma_centre",
        "survival_curve": "linear", "golden_hour_min": 45,
        "rpm": {"respirations": "normal", "pulse": "present_weak", "mental_status": "alert"},
        "multi_agency": True, "trapped": False, "hazmat": False,
        "agencies_required": ["Police"],
    },
    {
        "template_id": "T012", "condition": "acute_asthma_severe",
        "call_description": "17-year-old male, severe bronchospasm. Can only speak one word at a time. Peak flow <30% predicted. Using accessory muscles. First reliever not working. SpO2 88%.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "ALS",
        "required_specialty": "paediatric_emergency",
        "survival_curve": "step_cliff", "golden_hour_min": 30,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_strong", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T013", "condition": "chest_trauma",
        "call_description": "Construction accident — rebar through right chest. Open pneumothorax. Sucking chest wound. RR 32, SpO2 84%. Conscious but very distressed. Colleagues holding wound.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "ALS",
        "required_specialty": "trauma_centre",
        "survival_curve": "step_cliff", "golden_hour_min": 30,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_normal", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T014", "condition": "preeclampsia_severe",
        "call_description": "30-week pregnant 24-year-old. BP 160/105. Severe headache, blurred vision. No seizures yet. Pedal oedema +++. Reduced foetal movement. Urine output decreasing.",
        "severity": "P2", "start_tag": "Delayed", "correct_unit": "ALS",
        "required_specialty": "obstetrics",
        "survival_curve": "sigmoid", "golden_hour_min": 120,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T015", "condition": "diabetic_ketoacidosis",
        "call_description": "19-year-old Type 1 diabetic. Kussmaul breathing, fruity breath. BS 520 mg/dL, confused but rousable. Vomiting × 8h. First episode.",
        "severity": "P2", "start_tag": "Delayed", "correct_unit": "ALS",
        "required_specialty": "trauma_centre",
        "survival_curve": "linear", "golden_hour_min": 120,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_normal", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T016", "condition": "organophosphate_poisoning",
        "call_description": "55-year-old farmer. Found collapsed in field, empty pesticide bottle nearby. Pinpoint pupils, SLUDGE syndrome, excessive secretions. GCS 9, HR 42.",
        "severity": "P2", "start_tag": "Delayed", "correct_unit": "ALS",
        "required_specialty": "toxicology",
        "survival_curve": "exponential", "golden_hour_min": 90,
        "rpm": {"respirations": "present_abnormal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": False, "trapped": False, "hazmat": True,
        "agencies_required": ["Fire"],
    },
    {
        "template_id": "T017", "condition": "fracture_lower_limb",
        "call_description": "67-year-old female, slipped on wet floor. Suspected fractured hip. Cannot weight bear. Pain 8/10. Alert and oriented. Stable vitals. No neurovascular compromise.",
        "severity": "P2", "start_tag": "Delayed", "correct_unit": "BLS",
        "required_specialty": "trauma_centre",
        "survival_curve": "linear", "golden_hour_min": 240,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T018", "condition": "anxiety_panic_attack",
        "call_description": "28-year-old female at Deccan bus stop. Hyperventilating, dizziness, tingling hands. No past medical history. Breathing into bag partially effective. No chest pain. SpO2 99%.",
        "severity": "P3", "start_tag": "Minimal", "correct_unit": "BLS",
        "required_specialty": None,
        "survival_curve": None, "golden_hour_min": None,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T019", "condition": "minor_burns_superficial",
        "call_description": "34-year-old male, cooking fire. Superficial burns to right forearm and hand, <5% BSA. Painful, blistering. No airway involvement. Alert, walking.",
        "severity": "P3", "start_tag": "Minimal", "correct_unit": "BLS",
        "required_specialty": None,
        "survival_curve": None, "golden_hour_min": None,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T020", "condition": "syncope_vasovagal",
        "call_description": "19-year-old student collapsed at exam hall. Brief loss of consciousness, now recovered fully. Prodrome of nausea and diaphoresis. BP 118/72. GCS 15. No trauma in fall.",
        "severity": "P3", "start_tag": "Minimal", "correct_unit": "BLS",
        "required_specialty": None,
        "survival_curve": None, "golden_hour_min": None,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "alert"},
        "multi_agency": False, "trapped": False, "hazmat": False,
    },
    {
        "template_id": "T021", "condition": "mci_rta",
        "call_description": "Bus-truck collision on NH-48. Multiple victims. Driver trapped. 4 confirmed P1, 3 P2. Scene unsecured. Fuel leakage. Police 6 minutes out.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "MICU",
        "required_specialty": "trauma_centre",
        "survival_curve": "linear", "golden_hour_min": 60,
        "rpm": {"respirations": "normal", "pulse": "present_weak", "mental_status": "altered"},
        "multi_agency": True, "trapped": True, "hazmat": False,
        "agencies_required": ["Police", "Fire"],
    },
    {
        "template_id": "T022", "condition": "carbon_monoxide",
        "call_description": "Family of 4 found unconscious in flat in Kothrud. Neighbour smelled gas. CO detector beeping. All unresponsive. Suspected generator exhaust in enclosed space.",
        "severity": "P1", "start_tag": "Immediate", "correct_unit": "ALS",
        "required_specialty": "toxicology",
        "survival_curve": "exponential", "golden_hour_min": 60,
        "rpm": {"respirations": "normal", "pulse": "present_normal", "mental_status": "unresponsive"},
        "multi_agency": True, "trapped": False, "hazmat": True,
        "agencies_required": ["Fire"],
    },
]

T_IDX_BY_SEVERITY: Dict[str, List[int]] = {"P1": [], "P2": [], "P3": []}
for _i, _t in enumerate(INCIDENT_TEMPLATE_POOL):
    T_IDX_BY_SEVERITY[_t["severity"]].append(_i)

T_IDX_BY_UNIT: Dict[str, List[int]] = {"MICU": [], "ALS": [], "BLS": []}
for _i, _t in enumerate(INCIDENT_TEMPLATE_POOL):
    T_IDX_BY_UNIT[_t["correct_unit"]].append(_i)

SURVIVAL_CURVE_PARAMS: Dict[str, Dict[str, Any]] = {
    "sigmoid":       {"survival_at_zero": 0.97, "survival_floor": 0.18, "k": 0.10, "midpoint": 90},
    "exponential":   {"survival_at_zero": 0.98, "survival_floor": 0.08, "decay_rate": 0.035},
    "linear":        {"survival_at_zero": 0.95, "survival_floor": 0.10, "linear_rate": 0.012},
    "step_cliff":    {"survival_at_zero": 0.95, "survival_floor": 0.15, "cliff_time": 30, "cliff_width": 10},
}

class ScenarioSuite(str, Enum):
    STANDARD    = "standard"
    ADVERSARIAL = "adversarial"
    CURRICULUM  = "curriculum"
    BENCHMARK   = "benchmark"
    STRESS_TEST = "stress_test"
    REGRESSION  = "regression"
    MCI_FOCUS   = "mci_focus"

class OutputFormat(str, Enum):
    JSON  = "json"
    JSONL = "jsonl"
    CSV   = "csv"
    TSV   = "tsv"
    ALL   = "all"

@dataclass(frozen=True)
class AugmentationSpec:
    weather:          str   = "clear"
    hour_of_day:      int   = 10
    day_of_week:      str   = "monday"
    season:           str   = "winter"
    traffic_incident: bool  = False
    hospital_under_surge: bool = False
    crew_fatigue_pre:     bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_peak_morning(self) -> bool:
        return PEAK_MORNING[0] <= self.hour_of_day < PEAK_MORNING[1]

    @property
    def is_peak_evening(self) -> bool:
        return PEAK_EVENING[0] <= self.hour_of_day < PEAK_EVENING[1]

    @property
    def is_peak(self) -> bool:
        return self.is_peak_morning or self.is_peak_evening

    @property
    def base_traffic_multiplier(self) -> float:
        hour_mult = {
            0:0.55,1:0.48,2:0.44,3:0.42,4:0.45,5:0.58,
            6:0.82,7:1.38,8:1.72,9:1.85,10:1.55,11:1.28,
            12:1.32,13:1.25,14:1.18,15:1.22,16:1.48,17:1.88,
            18:1.95,19:1.78,20:1.42,21:1.15,22:0.88,23:0.68,
        }.get(self.hour_of_day, 1.0)
        weather_mult = WEATHER_PROFILES.get(self.weather, {}).get("traffic_mult", 1.0)
        incident_add = 0.25 if self.traffic_incident else 0.0
        return round(hour_mult * weather_mult + incident_add, 4)

    @property
    def incident_rate_multiplier(self) -> float:
        return WEATHER_PROFILES.get(self.weather, {}).get("incident_mult", 1.0)

@dataclass(frozen=True)
class ScenarioConfig:
    task_id:    int
    seed:       int
    suite:      ScenarioSuite = ScenarioSuite.STANDARD
    augment:    AugmentationSpec = field(default_factory=AugmentationSpec)

    def __post_init__(self) -> None:
        if self.task_id not in TASK_META:
            raise ValueError(f"Invalid task_id: {self.task_id}")
        if not (SEED_RANGE[0] <= self.seed <= SEED_RANGE[1]):
            raise ValueError(f"Seed {self.seed} out of range {SEED_RANGE}")

    @property
    def scenario_id(self) -> str:
        aug_sig = (
            f"{self.augment.weather}_{self.augment.hour_of_day:02d}"
            f"_{self.augment.day_of_week[:3]}"
        )
        return f"task{self.task_id:02d}_{self.suite.value}_seed{self.seed:04d}_{aug_sig}"

    @property
    def content_hash(self) -> str:
        raw = json.dumps(
            {
                "task_id": self.task_id,
                "seed": self.seed,
                "suite": self.suite.value,
                "augment": self.augment.to_dict(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def task_meta(self) -> Dict[str, Any]:
        return TASK_META[self.task_id]

    @property
    def difficulty(self) -> str:
        return self.task_meta["difficulty"]

    @property
    def baseline_score(self) -> float:
        return self.task_meta["baseline"]

@dataclass
class GeneratedIncident:
    incident_id:   str
    template_id:   str
    call_number:   str
    condition:     str
    severity:      str
    start_tag:     str
    correct_unit:  str
    required_specialty: Optional[str]
    zone_id:       str
    lat:           float
    lon:           float
    call_description: str
    rpm:           Dict[str, str]
    multi_agency:  bool
    trapped:       bool
    hazmat:        bool
    agencies_required: List[str]
    arrival_minute: float
    survival_curve: Optional[str]
    golden_hour_min: Optional[float]
    survival_at_zero: float
    current_survival_prob: float
    ground_truth:  Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in asdict(self).items()}

@dataclass
class GeneratedUnit:
    unit_id:       str
    unit_type:     str
    zone_id:       str
    lat:           float
    lon:           float
    status:        str
    hours_on_duty: float
    crew_fatigued: bool
    comms_active:  bool
    fuel_pct:      float

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}

@dataclass
class GeneratedHospital:
    hospital_id:     str
    name:            str
    zone_id:         str
    tier:            str
    er_occupancy:    float
    icu_occupancy:   float
    on_diversion:    bool
    diversion_reason: Optional[str]
    available_er_beds: int
    available_icu_beds: int
    specialties:     List[str]
    accepts_patients: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}

@dataclass
class TrafficMatrix:
    matrix:          Dict[str, Dict[str, float]]
    peak_active:     bool
    traffic_multiplier: float
    active_slowdowns: List[Dict[str, Any]]

    def get(self, from_zone: str, to_zone: str) -> float:
        return self.matrix.get(from_zone, {}).get(to_zone, 30.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": self.matrix,
            "peak_active": self.peak_active,
            "traffic_multiplier": round(self.traffic_multiplier, 4),
            "active_slowdowns": self.active_slowdowns,
        }

@dataclass
class MutualAidContext:
    available_zones:    List[str]
    units_available:    int
    request_delay_min:  float
    over_request_threshold: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ScenarioMetadata:
    scenario_id:       str
    content_hash:      str
    generator_version: str
    schema_version:    int
    task_id:           int
    task_name:         str
    difficulty:        str
    seed:              int
    suite:             str
    augmentation:      Dict[str, Any]
    baseline_score:    float
    generated_at_utc:  str
    valid_actions:     List[str]
    grader_weights:    Dict[str, float]
    episode_config:    Dict[str, Any]
    statistical_signature: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ScenarioPack:
    metadata:          ScenarioMetadata
    incident_queue:    List[GeneratedIncident]
    fleet_status:      List[GeneratedUnit]
    hospital_network:  List[GeneratedHospital]
    traffic:           TrafficMatrix
    demand_forecast:   Dict[str, Any]
    mutual_aid:        MutualAidContext
    comms_failure_prob: float
    ground_truth:      Dict[str, Any]
    agent_prompt_context: str
    validation_result: Optional[Dict[str, Any]] = None

    def to_dict(self, include_ground_truth: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "metadata":         self.metadata.to_dict(),
            "incident_queue":   [i.to_dict() for i in self.incident_queue],
            "fleet_status":     [u.to_dict() for u in self.fleet_status],
            "hospital_network": [h.to_dict() for h in self.hospital_network],
            "traffic":          self.traffic.to_dict(),
            "demand_forecast":  self.demand_forecast,
            "mutual_aid":       self.mutual_aid.to_dict(),
            "comms_failure_prob": round(self.comms_failure_prob, 4),
            "agent_prompt_context": self.agent_prompt_context,
        }
        if include_ground_truth:
            d["ground_truth"] = self.ground_truth
        if self.validation_result is not None:
            d["validation_result"] = self.validation_result
        return d

    def to_json(self, include_ground_truth: bool = False, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(include_ground_truth=include_ground_truth),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )

    @property
    def scenario_id(self) -> str:
        return self.metadata.scenario_id

    @property
    def incident_count(self) -> int:
        return len(self.incident_queue)

    @property
    def p1_count(self) -> int:
        return sum(1 for i in self.incident_queue if i.severity == "P1")

    @property
    def p2_count(self) -> int:
        return sum(1 for i in self.incident_queue if i.severity == "P2")

    @property
    def p3_count(self) -> int:
        return sum(1 for i in self.incident_queue if i.severity == "P3")

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj))
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Not serializable: {type(obj)}")

class DeterministicRNG:

    def __init__(self, root_seed: int, task_id: int) -> None:
        self._root = root_seed
        self._task = task_id
        self._streams: Dict[str, np.random.Generator] = {}
        self._py_streams: Dict[str, random.Random] = {}

    def _derive_seed(self, stream_name: str) -> int:
        raw = f"{self._root}:{self._task}:{stream_name}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return int(digest[:16], 16) % (2**32)

    def np_stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            self._streams[name] = np.random.default_rng(self._derive_seed(name))
        return self._streams[name]

    def py_stream(self, name: str) -> random.Random:
        if name not in self._py_streams:
            rng = random.Random()
            rng.seed(self._derive_seed(name))
            self._py_streams[name] = rng
        return self._py_streams[name]

    def uniform(self, low: float, high: float, stream: str = "default") -> float:
        return float(self.np_stream(stream).uniform(low, high))

    def integers(self, low: int, high: int, stream: str = "default") -> int:
        return int(self.np_stream(stream).integers(low, high + 1))

    def choice(self, seq: Sequence, stream: str = "default") -> Any:
        return seq[int(self.np_stream(stream).integers(0, len(seq)))]

    def choices(self, seq: Sequence, k: int, stream: str = "default") -> List:
        idxs = self.np_stream(stream).integers(0, len(seq), size=k)
        return [seq[i] for i in idxs]

    def shuffle(self, lst: List, stream: str = "default") -> List:
        copy = lst[:]
        self.py_stream(stream).shuffle(copy)
        return copy

    def normal(self, mu: float, sigma: float, stream: str = "default") -> float:
        return float(self.np_stream(stream).normal(mu, sigma))

    def clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def bernoulli(self, p: float, stream: str = "default") -> bool:
        return self.uniform(0.0, 1.0, stream) < p

    def sample_without_replace(self, seq: Sequence, k: int, stream: str = "default") -> List:
        shuffled = self.shuffle(list(seq), stream)
        return shuffled[:k]

class AugmentationPipeline:

    @staticmethod
    def random_augment(rng: DeterministicRNG, task_id: int) -> AugmentationSpec:
        difficulty = TASK_META[task_id]["difficulty"]

        if difficulty == "hard":
            weather_pool = ["clear", "clear", "heavy_rain", "fog", "extreme_heat"]
            hour_pool = list(range(0, 24))
            hospital_surge = rng.bernoulli(0.45, "augment")
            traffic_incident = rng.bernoulli(0.50, "augment")
            crew_fatigue_pre = rng.bernoulli(0.60, "augment")
        elif difficulty == "medium":
            weather_pool = ["clear", "clear", "clear", "heavy_rain", "fog"]
            hour_pool = list(range(6, 22))
            hospital_surge = rng.bernoulli(0.25, "augment")
            traffic_incident = rng.bernoulli(0.35, "augment")
            crew_fatigue_pre = rng.bernoulli(0.35, "augment")
        else:
            weather_pool = ["clear", "clear", "clear", "clear", "heavy_rain"]
            hour_pool = list(range(8, 20))
            hospital_surge = rng.bernoulli(0.10, "augment")
            traffic_incident = rng.bernoulli(0.15, "augment")
            crew_fatigue_pre = rng.bernoulli(0.10, "augment")

        return AugmentationSpec(
            weather=rng.choice(weather_pool, "augment"),
            hour_of_day=rng.choice(hour_pool, "augment"),
            day_of_week=rng.choice(DAYS_OF_WEEK, "augment"),
            season=rng.choice(SEASONS, "augment"),
            traffic_incident=traffic_incident,
            hospital_under_surge=hospital_surge,
            crew_fatigue_pre=crew_fatigue_pre,
        )

    @staticmethod
    def adversarial_augment(rng: DeterministicRNG) -> AugmentationSpec:
        return AugmentationSpec(
            weather=rng.choice(["heavy_rain", "fog", "flood", "dust_storm"], "adv_aug"),
            hour_of_day=rng.choice([7, 8, 9, 17, 18, 19], "adv_aug"),
            day_of_week=rng.choice(["monday", "friday", "public_holiday"], "adv_aug"),
            season=rng.choice(["monsoon", "summer"], "adv_aug"),
            traffic_incident=True,
            hospital_under_surge=True,
            crew_fatigue_pre=True,
        )

    @staticmethod
    def apply_to_traffic(
        base_matrix: Dict[str, Dict[str, float]],
        augment: AugmentationSpec,
    ) -> Dict[str, Dict[str, float]]:
        mult = augment.base_traffic_multiplier
        return {
            from_z: {
                to_z: round(t * mult, 2)
                for to_z, t in row.items()
            }
            for from_z, row in base_matrix.items()
        }

    @staticmethod
    def apply_to_hospitals(
        hospitals: List[Dict[str, Any]],
        augment: AugmentationSpec,
        rng: DeterministicRNG,
    ) -> List[Dict[str, Any]]:
        if not augment.hospital_under_surge:
            return hospitals
        result = []
        for h in hospitals:
            h = dict(h)
            surge_boost = rng.uniform(0.05, 0.18, "hosp_aug")
            h["er_occupancy"] = min(0.99, round(h["er_occupancy"] + surge_boost, 3))
            h["icu_occupancy"] = min(0.99, round(h["icu_occupancy"] + surge_boost * 0.8, 3))
            h["on_diversion"] = h["er_occupancy"] >= 0.90
            h["diversion_reason"] = "er_capacity" if h["on_diversion"] else None
            result.append(h)
        return result

    @staticmethod
    def comm_failure_prob(base_prob: float, augment: AugmentationSpec) -> float:
        weather_deg = WEATHER_PROFILES.get(augment.weather, {}).get("comm_degradation", 0.0)
        return min(0.40, round(base_prob + weather_deg, 4))

class TrafficMatrixBuilder:

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    @classmethod
    def build_base_matrix(cls, rng: DeterministicRNG) -> Dict[str, Dict[str, float]]:
        matrix: Dict[str, Dict[str, float]] = {}
        for z_from in ZONES:
            row: Dict[str, float] = {}
            for z_to in ZONES:
                if z_from["zone_id"] == z_to["zone_id"]:
                    row[z_to["zone_id"]] = 0.0
                else:
                    dist_km = cls._haversine(
                        z_from["lat"], z_from["lon"],
                        z_to["lat"],  z_to["lon"],
                    )
                    base_min = (dist_km / 18.0) * 60.0
                    noise    = rng.uniform(-0.15, 0.15, "traffic_base")
                    row[z_to["zone_id"]] = round(max(2.0, base_min * (1 + noise)), 2)
            matrix[z_from["zone_id"]] = row
        return matrix

    @staticmethod
    def generate_slowdowns(
        rng: DeterministicRNG,
        augment: AugmentationSpec,
        incident_zones: List[str],
    ) -> List[Dict[str, Any]]:
        if not augment.traffic_incident and not incident_zones:
            return []
        slowdowns: List[Dict[str, Any]] = []
        n_slowdowns = rng.integers(1, 3, "slowdowns")
        zones_pool = incident_zones if incident_zones else ZONE_IDS
        for _i in range(n_slowdowns):
            zone = rng.choice(zones_pool, "slowdowns")
            slowdowns.append({
                "zone_id": zone,
                "adjacent_zones": ZONE_ADJACENCY.get(zone, []),
                "delay_pct": round(rng.uniform(0.15, 0.45, "slowdowns"), 3),
                "caused_by": "active_incident" if zone in incident_zones else "external",
                "expires_in_steps": rng.integers(3, 10, "slowdowns"),
            })
        return slowdowns

class IncidentEngine:

    def __init__(self, config: ScenarioConfig, rng: DeterministicRNG) -> None:
        self.config = config
        self.rng    = rng
        self.meta   = config.task_meta

    def generate(self) -> List[GeneratedIncident]:
        tid = self.config.task_id
        if tid == 1: return self._task1()
        if tid == 2: return self._task2()
        if tid == 3: return self._task3()
        if tid == 4: return self._task4()
        if tid == 5: return self._task5()
        if tid == 6: return []
        if tid == 7: return self._task7_mci()
        if tid == 8: return self._task8_transfers()
        if tid == 9: return self._task9_surge()
        raise ValueError(f"Unknown task_id: {tid}")

    def _task1(self) -> List[GeneratedIncident]:
        idx = T_IDX_BY_SEVERITY["P1"][
            self.rng.integers(0, len(T_IDX_BY_SEVERITY["P1"]) - 1, "t1_pick")
        ]
        template = INCIDENT_TEMPLATE_POOL[idx]
        return [self._make_incident(template, zone_id=self.rng.choice(ZONE_IDS, "t1_zone"))]

    def _task2(self) -> List[GeneratedIncident]:
        specialty_templates = [
            t for t in INCIDENT_TEMPLATE_POOL if t.get("required_specialty")
        ]
        template = self.rng.choice(specialty_templates, "t2_pick")
        return [self._make_incident(template, zone_id=self.rng.choice(ZONE_IDS, "t2_zone"))]

    def _task3(self) -> List[GeneratedIncident]:
        micu_als_pool = [
            INCIDENT_TEMPLATE_POOL[i]
            for i in T_IDX_BY_UNIT["MICU"] + T_IDX_BY_UNIT["ALS"]
        ]
        template = self.rng.choice(micu_als_pool, "t3_pick")
        return [self._make_incident(template, zone_id=self.rng.choice(ZONE_IDS, "t3_zone"))]

    def _task4(self) -> List[GeneratedIncident]:
        n = self.rng.integers(
            self.meta["min_incidents"], self.meta["max_incidents"], "t4_n"
        )
        incidents: List[GeneratedIncident] = []
        p1_indices = self.rng.sample_without_replace(T_IDX_BY_SEVERITY["P1"], 2, "t4_p1")
        for idx in p1_indices:
            incidents.append(self._make_incident(
                INCIDENT_TEMPLATE_POOL[idx],
                zone_id=self.rng.choice(ZONE_IDS, "t4_zone"),
                stagger_minute=self.rng.uniform(0, 2, "t4_stagger"),
            ))
        remaining = n - 2
        pool_p2 = [INCIDENT_TEMPLATE_POOL[i] for i in T_IDX_BY_SEVERITY["P2"]]
        pool_p3 = [INCIDENT_TEMPLATE_POOL[i] for i in T_IDX_BY_SEVERITY["P3"]]
        for j in range(remaining):
            if j % 3 == 0:
                t = self.rng.choice(pool_p3, "t4_mix")
            else:
                t = self.rng.choice(pool_p2, "t4_mix")
            incidents.append(self._make_incident(
                t,
                zone_id=self.rng.choice(ZONE_IDS, "t4_zone"),
                stagger_minute=self.rng.uniform(0, 5, "t4_stagger"),
            ))
        return incidents

    def _task5(self) -> List[GeneratedIncident]:
        n = self.rng.integers(
            self.meta["min_incidents"], self.meta["max_incidents"], "t5_n"
        )
        incidents: List[GeneratedIncident] = []
        reroute_template = self.rng.choice(
            [t for t in INCIDENT_TEMPLATE_POOL
             if t.get("required_specialty") in ["cardiac_cath_lab", "stroke_unit"]],
            "t5_primary"
        )
        incidents.append(self._make_incident(
            reroute_template,
            zone_id=self.rng.choice(["Z05", "Z02", "Z01"], "t5_prim_zone"),
            reroute_trap=True,
        ))
        for _ in range(n - 1):
            t = self.rng.choice(INCIDENT_TEMPLATE_POOL, "t5_others")
            incidents.append(self._make_incident(
                t, zone_id=self.rng.choice(ZONE_IDS, "t5_zone"),
            ))
        return incidents

    def _task7_mci(self) -> List[GeneratedIncident]:
        n = self.rng.integers(
            self.meta["min_incidents"], self.meta["max_incidents"], "t7_n"
        )
        n_immediate  = max(1, int(n * self.rng.uniform(0.15, 0.25, "t7_dist")))
        n_delayed    = max(1, int(n * self.rng.uniform(0.25, 0.35, "t7_dist")))
        n_expectant  = max(1, int(n * self.rng.uniform(0.08, 0.15, "t7_dist")))
        n_minimal    = n - n_immediate - n_delayed - n_expectant

        incidents: List[GeneratedIncident] = []
        mci_zones = self.rng.sample_without_replace(ZONE_IDS, 2, "t7_zones")

        p1_pool = [INCIDENT_TEMPLATE_POOL[i] for i in T_IDX_BY_SEVERITY["P1"]]
        for _ in range(n_immediate):
            t = self.rng.choice(p1_pool, "t7_imm")
            incidents.append(self._make_incident(
                t, zone_id=self.rng.choice(mci_zones, "t7_zone"),
                start_tag_override="Immediate", mci=True,
            ))

        p2_pool = [INCIDENT_TEMPLATE_POOL[i] for i in T_IDX_BY_SEVERITY["P2"]]
        for _ in range(n_delayed):
            t = self.rng.choice(p2_pool + p1_pool[:3], "t7_del")
            incidents.append(self._make_incident(
                t, zone_id=self.rng.choice(mci_zones, "t7_zone"),
                start_tag_override="Delayed", mci=True,
            ))

        expectant_rpm = {
            "respirations": "absent",
            "pulse": "absent",
            "mental_status": "unresponsive",
        }
        for _ in range(n_expectant):
            t = self.rng.choice(p1_pool, "t7_exp")
            t = dict(t)
            t["rpm"] = expectant_rpm
            incidents.append(self._make_incident(
                t, zone_id=self.rng.choice(mci_zones, "t7_zone"),
                start_tag_override="Expectant", mci=True,
            ))

        p3_pool = [INCIDENT_TEMPLATE_POOL[i] for i in T_IDX_BY_SEVERITY["P3"]]
        for _ in range(max(0, n_minimal)):
            t = self.rng.choice(p3_pool + p2_pool[:2], "t7_min")
            incidents.append(self._make_incident(
                t, zone_id=self.rng.choice(mci_zones, "t7_zone"),
                start_tag_override="Minimal", mci=True,
            ))

        return incidents

    def _task8_transfers(self) -> List[GeneratedIncident]:
        n = self.rng.integers(
            self.meta["min_incidents"], self.meta["max_incidents"], "t8_n"
        )
        transfer_conditions = [
            "stemi_anterior", "severe_tbi", "ischemic_stroke",
            "hemorrhagic_stroke_sah", "septic_shock", "ruptured_aaa",
        ]
        incidents: List[GeneratedIncident] = []
        for _ in range(n):
            cond = self.rng.choice(transfer_conditions, "t8_cond")
            matching = [t for t in INCIDENT_TEMPLATE_POOL if t["condition"] == cond]
            template = self.rng.choice(matching if matching else INCIDENT_TEMPLATE_POOL, "t8_tmpl")
            inc = self._make_incident(
                template,
                zone_id=self.rng.choice(ZONE_IDS, "t8_zone"),
                icu_transfer=True,
            )
            incidents.append(inc)
        return incidents

    def _task9_surge(self) -> List[GeneratedIncident]:
        n = self.rng.integers(
            self.meta["min_incidents"], self.meta["max_incidents"], "t9_n"
        )
        per_mci = n // 3
        remainder = n % 3
        mci_zones = [
            ("Z03", "Z05"),
            ("Z07", "Z11"),
            ("Z08", "Z12"),
        ]
        incidents: List[GeneratedIncident] = []
        for _mci_idx, zones in enumerate(mci_zones):
            n_this = per_mci + (1 if _mci_idx < remainder else 0)
            sub_meta = {**self.meta, "min_incidents": n_this, "max_incidents": n_this}
            sub_cfg  = ScenarioConfig(
                task_id=7,
                seed=self.config.seed + _mci_idx,
                suite=self.config.suite,
                augment=self.config.augment,
            )
            sub_rng = DeterministicRNG(self.config.seed + _mci_idx * 17, task_id=9)
            sub_engine = IncidentEngine(sub_cfg, sub_rng)
            sub_incidents = sub_engine._task7_mci()
            for inc in sub_incidents:
                inc.zone_id = self.rng.choice(list(zones), "t9_zone")
                zone_info = ZONE_LOOKUP.get(inc.zone_id, ZONES[0])
                inc.lat = round(zone_info["lat"] + self.rng.uniform(-0.005, 0.005, "t9_jitter"), 6)
                inc.lon = round(zone_info["lon"] + self.rng.uniform(-0.005, 0.005, "t9_jitter"), 6)
            incidents.extend(sub_incidents)
        return incidents

    def _make_incident(
        self,
        template: Dict[str, Any],
        zone_id: str,
        stagger_minute: float = 0.0,
        start_tag_override: Optional[str] = None,
        reroute_trap: bool = False,
        mci: bool = False,
        icu_transfer: bool = False,
    ) -> GeneratedIncident:
        zone_info = ZONE_LOOKUP.get(zone_id, ZONES[0])
        lat = round(zone_info["lat"] + self.rng.uniform(-0.008, 0.008, "inc_jitter"), 6)
        lon = round(zone_info["lon"] + self.rng.uniform(-0.008, 0.008, "inc_jitter"), 6)

        incident_id  = f"INC-{self.config.task_id:02d}-{uuid.UUID(int=self.rng.integers(0, 2**32, 'inc_uuid') * 2**96).hex[:8].upper()}"
        call_number  = f"108-PUN-{self.rng.integers(1000, 9999, 'call_num'):04d}"

        final_start_tag = start_tag_override or template.get("start_tag", "Immediate")
        survival_at_zero = self._survival_at_zero(template, final_start_tag)

        elapsed = self.rng.uniform(0, 5, "elapsed")
        current_survival = self._compute_survival(
            template.get("survival_curve"),
            survival_at_zero,
            elapsed,
        )

        rpm = dict(template.get("rpm", {}))

        if final_start_tag == "Expectant":
            rpm = {"respirations": "absent", "pulse": "absent", "mental_status": "unresponsive"}
            current_survival = 0.05

        ground_truth = {
            "correct_triage":    final_start_tag,
            "correct_unit":      template.get("correct_unit", "ALS"),
            "correct_specialty": template.get("required_specialty"),
            "survival_at_zero":  survival_at_zero,
            "reroute_trap":      reroute_trap,
            "icu_transfer":      icu_transfer,
            "mci_victim":        mci,
            "penalty_if_wrong_tag": -0.5 if (
                final_start_tag == "Immediate" and self.config.task_id in (7, 9)
            ) else 0.0,
        }

        return GeneratedIncident(
            incident_id=incident_id,
            template_id=template.get("template_id", "T000"),
            call_number=call_number,
            condition=template.get("condition", "unknown"),
            severity=self._start_tag_to_severity(final_start_tag),
            start_tag=final_start_tag,
            correct_unit=template.get("correct_unit", "ALS"),
            required_specialty=template.get("required_specialty"),
            zone_id=zone_id,
            lat=lat,
            lon=lon,
            call_description=template.get("call_description", ""),
            rpm=rpm,
            multi_agency=template.get("multi_agency", False),
            trapped=template.get("trapped", False),
            hazmat=template.get("hazmat", False),
            agencies_required=template.get("agencies_required", []),
            arrival_minute=round(stagger_minute, 2),
            survival_curve=template.get("survival_curve"),
            golden_hour_min=template.get("golden_hour_min"),
            survival_at_zero=round(survival_at_zero, 6),
            current_survival_prob=round(current_survival, 6),
            ground_truth=ground_truth,
        )

    @staticmethod
    def _start_tag_to_severity(tag: str) -> str:
        return {"Immediate": "P1", "Delayed": "P2", "Minimal": "P3", "Expectant": "P0"}.get(tag, "P2")

    @staticmethod
    def _survival_at_zero(template: Dict[str, Any], start_tag: str) -> float:
        base = {"Immediate": 0.95, "Delayed": 0.82, "Minimal": 0.99, "Expectant": 0.05}
        params = SURVIVAL_CURVE_PARAMS.get(template.get("survival_curve", ""), {})
        return params.get("survival_at_zero", base.get(start_tag, 0.90))

    @staticmethod
    def _compute_survival(curve: Optional[str], s0: float, t_min: float) -> float:
        if curve is None:
            return s0
        p = SURVIVAL_CURVE_PARAMS.get(curve, {})
        floor = p.get("survival_floor", 0.10)
        if curve == "exponential":
            return max(floor, s0 * math.exp(-p.get("decay_rate", 0.035) * t_min))
        if curve == "linear":
            return max(floor, s0 - p.get("linear_rate", 0.012) * t_min)
        if curve == "sigmoid":
            k, mid = p.get("k", 0.10), p.get("midpoint", 90)
            return floor + (s0 - floor) / (1 + math.exp(k * (t_min - mid)))
        if curve == "step_cliff":
            cliff = p.get("cliff_time", 30)
            if t_min < cliff:
                return s0
            return max(floor, s0 * 0.40)
        return s0

class FleetBuilder:

    UNIT_PREFIXES = {"BLS": "BLS", "ALS": "ALS", "MICU": "MCU"}

    def __init__(self, config: ScenarioConfig, rng: DeterministicRNG) -> None:
        self.config = config
        self.rng    = rng
        self.meta   = config.task_meta

    def build(self) -> List[GeneratedUnit]:
        meta = self.meta
        n = self.rng.integers(meta["min_ambulances"], meta["max_ambulances"], "fleet_n")
        units: List[GeneratedUnit] = []

        composition = self._composition_for_task()
        zone_pool   = self._starting_zones()

        for idx, unit_type in enumerate(composition[:n]):
            zone_id = zone_pool[idx % len(zone_pool)]
            zone    = ZONE_LOOKUP.get(zone_id, ZONES[0])
            hours   = self._hours_on_duty()
            comms   = self._comms_active()

            unit_num = f"{self.UNIT_PREFIXES[unit_type]}-{idx+1:02d}"
            units.append(GeneratedUnit(
                unit_id=unit_num,
                unit_type=unit_type,
                zone_id=zone_id,
                lat=round(zone["lat"] + self.rng.uniform(-0.003, 0.003, "fleet_jitter"), 6),
                lon=round(zone["lon"] + self.rng.uniform(-0.003, 0.003, "fleet_jitter"), 6),
                status="available",
                hours_on_duty=round(hours, 2),
                crew_fatigued=hours >= 10.0,
                comms_active=comms,
                fuel_pct=round(self.rng.uniform(0.55, 1.0, "fleet_fuel"), 3),
            ))
        return units

    def _composition_for_task(self) -> List[str]:
        tid = self.config.task_id
        n_max = self.meta["max_ambulances"]
        if tid in (1, 2, 3):
            base = ["MICU", "ALS", "ALS", "BLS", "BLS", "BLS"]
        elif tid == 4:
            base = ["MICU", "ALS", "BLS"]
        elif tid in (5, 6):
            base = ["MICU", "ALS", "ALS", "BLS", "BLS"]
        elif tid == 7:
            base = ["MICU", "MICU", "MICU", "ALS", "ALS", "ALS", "ALS", "BLS", "BLS", "BLS"]
        elif tid == 8:
            base = ["MICU", "MICU", "ALS", "ALS", "BLS", "BLS"]
        else:
            base = ["MICU", "MICU", "MICU", "ALS", "ALS", "ALS", "ALS", "BLS", "BLS", "BLS", "BLS", "BLS"]
        return (base * 3)[:n_max]

    def _starting_zones(self) -> List[str]:
        pops = [z["population"] for z in ZONES]
        total_pop = sum(pops)
        weights = [p / total_pop for p in pops]
        shuffled = self.rng.shuffle(ZONE_IDS[:], "fleet_zones")
        n_max = self.meta["max_ambulances"]
        chosen: List[str] = []
        for _ in range(n_max):
            r = self.rng.uniform(0.0, 1.0, "fleet_weight")
            cumul = 0.0
            for z_id, w in zip(ZONE_IDS, weights):
                cumul += w
                if r <= cumul:
                    chosen.append(z_id)
                    break
            else:
                chosen.append(shuffled[0])
        return chosen

    def _hours_on_duty(self) -> float:
        if self.config.augment.crew_fatigue_pre:
            return self.rng.uniform(6.0, 12.5, "crew_hours")
        return self.rng.uniform(0.5, 8.0, "crew_hours")

    def _comms_active(self) -> bool:
        base_fail = self.config.task_meta.get("comm_failure_prob", 0.0)
        fail_prob = AugmentationPipeline.comm_failure_prob(base_fail, self.config.augment)
        return not self.rng.bernoulli(fail_prob, "comms")

class HospitalNetworkBuilder:

    def __init__(self, config: ScenarioConfig, rng: DeterministicRNG) -> None:
        self.config = config
        self.rng    = rng

    def build(self) -> List[GeneratedHospital]:
        hospitals_raw = AugmentationPipeline.apply_to_hospitals(
            HOSPITALS, self.config.augment, self.rng
        )
        result: List[GeneratedHospital] = []
        for h in hospitals_raw:
            er_occ  = h["er_occupancy"]
            icu_occ = h["icu_occupancy"]

            er_noise  = self.rng.uniform(-0.04, 0.04, "hosp_noise")
            icu_noise = self.rng.uniform(-0.04, 0.04, "hosp_noise")
            er_occ    = max(0.10, min(0.99, er_occ  + er_noise))
            icu_occ   = max(0.10, min(0.99, icu_occ + icu_noise))

            on_div = h.get("on_diversion", False) or er_occ >= 0.90
            avail_er  = max(0, int(h["er_beds"]  * (1 - er_occ)))
            avail_icu = max(0, int(h["icu_beds"] * (1 - icu_occ)))

            result.append(GeneratedHospital(
                hospital_id=h["hospital_id"],
                name=h["name"],
                zone_id=h["zone_id"],
                tier=h["tier"],
                er_occupancy=round(er_occ, 3),
                icu_occupancy=round(icu_occ, 3),
                on_diversion=on_div,
                diversion_reason="er_capacity" if on_div else None,
                available_er_beds=avail_er,
                available_icu_beds=avail_icu,
                specialties=h["specialties"],
                accepts_patients=not on_div,
            ))
        return result

class DemandForecastBuilder:

    ZONE_BASE_DEMAND: Dict[str, float] = {
        "Z01": 1.20, "Z02": 0.95, "Z03": 1.35, "Z04": 1.10,
        "Z05": 1.25, "Z06": 1.05, "Z07": 1.45, "Z08": 1.15,
        "Z09": 0.90, "Z10": 0.85, "Z11": 1.00, "Z12": 0.70,
    }

    HOUR_MULTIPLIER: Dict[int, float] = {
        0:0.55,1:0.48,2:0.44,3:0.42,4:0.45,5:0.58,
        6:0.82,7:1.38,8:1.72,9:1.85,10:1.55,11:1.28,
        12:1.32,13:1.25,14:1.18,15:1.22,16:1.48,17:1.88,
        18:1.95,19:1.78,20:1.42,21:1.15,22:0.88,23:0.68,
    }

    def __init__(self, config: ScenarioConfig, rng: DeterministicRNG) -> None:
        self.config = config
        self.rng    = rng
        self.noise_pct = 0.20

    def build(self) -> Dict[str, Any]:
        start_hour = self.config.augment.hour_of_day
        horizon    = 12
        task_uses_forecast = self.config.task_id in (4, 5, 6, 7, 8, 9)

        zones_forecast: Dict[str, Any] = {}
        for zone_id in ZONE_IDS:
            base = self.ZONE_BASE_DEMAND.get(zone_id, 1.0)
            hourly_forecast: List[Dict[str, Any]] = []
            for h_offset in range(horizon):
                hour = (start_hour + h_offset) % 24
                hour_mult = self.HOUR_MULTIPLIER.get(hour, 1.0)
                noise     = self.rng.normal(0, self.noise_pct * 0.5, "demand_noise")
                noisy_idx = max(0.1, base * hour_mult * (1 + noise))
                weather_mult = self.config.augment.incident_rate_multiplier
                hourly_forecast.append({
                    "hour_offset": h_offset,
                    "wall_clock_hour": hour,
                    "demand_index": round(noisy_idx * weather_mult, 4),
                    "forecast_calls_per_step": round(noisy_idx * weather_mult * 2.5, 2),
                    "confidence": round(1.0 - min(0.5, h_offset * 0.04), 3),
                })
            zones_forecast[zone_id] = {
                "zone_id": zone_id,
                "zone_name": ZONE_LOOKUP[zone_id]["name"],
                "base_demand_index": base,
                "hourly_forecast": hourly_forecast,
                "peak_demand_hour_offset": int(
                    max(range(horizon), key=lambda i: hourly_forecast[i]["demand_index"])
                ),
                "total_forecast_calls": round(
                    sum(f["forecast_calls_per_step"] for f in hourly_forecast), 1
                ),
            }

        return {
            "forecast_horizon_hours": horizon,
            "start_hour": start_hour,
            "noise_pct": self.noise_pct,
            "weather": self.config.augment.weather,
            "task_uses_forecast": task_uses_forecast,
            "zones": zones_forecast,
            "hot_zones": sorted(
                zones_forecast.keys(),
                key=lambda z: zones_forecast[z]["total_forecast_calls"],
                reverse=True,
            )[:4],
        }

class MutualAidBuilder:

    def __init__(self, config: ScenarioConfig, rng: DeterministicRNG) -> None:
        self.config = config
        self.rng    = rng

    def build(self) -> MutualAidContext:
        meta = self.config.task_meta
        if not meta.get("mutual_aid_required") and self.config.task_id < 7:
            return MutualAidContext(
                available_zones=[],
                units_available=0,
                request_delay_min=12.0,
                over_request_threshold=0,
            )

        available_zones = self.rng.sample_without_replace(ZONE_IDS, 4, "mutual_zones")
        units_per_zone  = self.rng.integers(1, 3, "mutual_units")
        total_units     = len(available_zones) * units_per_zone
        return MutualAidContext(
            available_zones=available_zones,
            units_available=total_units,
            request_delay_min=12.0,
            over_request_threshold=max(1, total_units // 2),
        )

class GroundTruthAssembler:

    def __init__(
        self,
        config: ScenarioConfig,
        incidents: List[GeneratedIncident],
        units: List[GeneratedUnit],
        hospitals: List[GeneratedHospital],
    ) -> None:
        self.config    = config
        self.incidents = incidents
        self.units     = units
        self.hospitals = hospitals

    def assemble(self) -> Dict[str, Any]:
        return {
            "task_id":       self.config.task_id,
            "seed":          self.config.seed,
            "schema_version": SCHEMA_VERSION,
            "incidents":     {i.incident_id: i.ground_truth for i in self.incidents},
            "optimal_dispatch": self._optimal_dispatch(),
            "hospital_scores":  self._hospital_specialty_scores(),
            "grader_inputs":    self._grader_inputs(),
        }

    def _optimal_dispatch(self) -> Dict[str, Any]:
        optimal: Dict[str, Any] = {}
        for inc in self.incidents:
            best_unit = self._best_unit_for(inc)
            best_hosp = self._best_hospital_for(inc)
            optimal[inc.incident_id] = {
                "unit_type":        inc.correct_unit,
                "best_unit_id":     best_unit,
                "hospital_id":      best_hosp,
                "start_tag":        inc.start_tag,
                "required_specialty": inc.required_specialty,
                "golden_hour_min":  inc.golden_hour_min,
            }
        return optimal

    def _best_unit_for(self, inc: GeneratedIncident) -> Optional[str]:
        matching = [u for u in self.units
                    if u.unit_type == inc.correct_unit and u.status == "available"]
        if not matching:
            tier_up = {"BLS": "ALS", "ALS": "MICU", "MICU": "MICU"}
            next_t  = tier_up.get(inc.correct_unit, "ALS")
            matching = [u for u in self.units if u.unit_type == next_t and u.status == "available"]
        if not matching:
            return None
        return matching[0].unit_id

    def _best_hospital_for(self, inc: GeneratedIncident) -> Optional[str]:
        spec = inc.required_specialty
        if spec is None:
            accepting = [h for h in self.hospitals if h.accepts_patients]
            return accepting[0].hospital_id if accepting else None
        matching = [h for h in self.hospitals
                    if spec in h.specialties and h.accepts_patients]
        if not matching:
            matching = [h for h in self.hospitals if h.accepts_patients]
        if not matching:
            return None
        return min(matching, key=lambda h: h.er_occupancy).hospital_id

    def _hospital_specialty_scores(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for h in self.hospitals:
            spec_score     = min(1.0, len(h.specialties) / 8.0)
            capacity_score = 1.0 - h.er_occupancy
            scores[h.hospital_id] = round((spec_score + capacity_score) / 2.0, 4)
        return scores

    def _grader_inputs(self) -> Dict[str, Any]:
        tid = self.config.task_id
        gi: Dict[str, Any] = {"task_id": tid}

        if tid in (1, 2, 3):
            if self.incidents:
                inc = self.incidents[0]
                gi["correct_triage"] = inc.start_tag
                gi["correct_unit"]   = inc.correct_unit
                gi["correct_hospital"] = self._best_hospital_for(inc)

        elif tid == 7:
            gi["victim_tags"] = {
                i.incident_id: i.start_tag for i in self.incidents
            }
            gi["immediate_count"] = sum(
                1 for i in self.incidents if i.start_tag == "Immediate"
            )
            gi["expectant_count"] = sum(
                1 for i in self.incidents if i.start_tag == "Expectant"
            )
            gi["wrong_immediate_as_expectant_penalty"] = -0.5

        elif tid == 9:
            gi["cascade_threshold_p1_unaddressed"] = max(
                5, int(len([i for i in self.incidents if i.severity == "P1"]) * 0.4)
            )
            gi["surge_declaration_required"] = True
            gi["mutual_aid_required"] = True

        return gi

class AgentPromptBuilder:

    def __init__(
        self,
        config: ScenarioConfig,
        incidents: List[GeneratedIncident],
        units: List[GeneratedUnit],
        hospitals: List[GeneratedHospital],
        augment: AugmentationSpec,
    ) -> None:
        self.config    = config
        self.incidents = incidents
        self.units     = units
        self.hospitals = hospitals
        self.augment   = augment

    def build(self) -> str:
        meta = self.config.task_meta
        lines: List[str] = [
            "═" * 70,
            f"EMERGI-ENV  ·  Task {self.config.task_id}: {meta['name']}",
            f"Difficulty: {meta['difficulty'].upper()}  |  Seed: {self.config.seed}",
            f"Baseline Score: {meta['baseline']:.2f}  |  Suite: {self.config.suite.value}",
            "═" * 70,
            "",
            "OPERATIONAL CONTEXT",
            "───────────────────",
            f"Time of Day : {self.augment.hour_of_day:02d}:00  "
            f"({'PEAK HOUR ⚠' if self.augment.is_peak else 'Off-peak'})",
            f"Day         : {self.augment.day_of_week.title()}",
            f"Weather     : {self.augment.weather.replace('_', ' ').title()}",
            f"Season      : {self.augment.season.title()}",
            f"Traffic Mult: {self.augment.base_traffic_multiplier:.2f}×",
            "",
        ]

        lines += [
            f"INCIDENT QUEUE ({len(self.incidents)} total)",
            "─" * 40,
        ]
        p1 = [i for i in self.incidents if i.severity == "P1"]
        p2 = [i for i in self.incidents if i.severity == "P2"]
        p3 = [i for i in self.incidents if i.severity == "P3"]
        p0 = [i for i in self.incidents if i.start_tag == "Expectant"]
        lines += [
            f"  P1 (Immediate) : {len(p1)}",
            f"  P2 (Delayed)   : {len(p2)}",
            f"  P3 (Minimal)   : {len(p3)}",
            f"  P0 (Expectant) : {len(p0)}",
            "",
        ]

        sorted_inc = sorted(
            self.incidents,
            key=lambda i: {"P1": 0, "P0": 0, "P2": 1, "P3": 2}.get(i.severity, 1)
        )
        lines.append("TOP PRIORITY INCIDENTS")
        lines.append("─" * 40)
        for inc in sorted_inc[:3]:
            trap = " [REROUTE TRAP]" if inc.ground_truth.get("reroute_trap") else ""
            ma   = " [MULTI-AGENCY]" if inc.multi_agency else ""
            haz  = " ⚠HAZMAT" if inc.hazmat else ""
            lines += [
                f"  [{inc.severity}] {inc.incident_id} | {inc.condition.upper()} | "
                f"Zone {inc.zone_id}{trap}{ma}{haz}",
                f"    → {inc.call_description[:80]}...",
                f"    START Tag: {inc.start_tag} | "
                f"Survival: {inc.current_survival_prob:.0%} | "
                f"Unit: {inc.correct_unit}",
                "",
            ]

        lines += [
            f"FLEET STATUS ({len(self.units)} units)",
            "─" * 40,
        ]
        for unit in self.units:
            fatigue = " [FATIGUED]" if unit.crew_fatigued else ""
            comms   = " [COMMS LOST]" if not unit.comms_active else ""
            lines.append(
                f"  {unit.unit_id:8s} | Zone {unit.zone_id} | {unit.unit_type:4s} | "
                f"Duty: {unit.hours_on_duty:.1f}h | Fuel: {unit.fuel_pct:.0%}"
                f"{fatigue}{comms}"
            )
        lines.append("")

        lines += [
            "HOSPITAL NETWORK",
            "─" * 40,
        ]
        for h in self.hospitals:
            div = " ⛔ ON DIVERSION" if h.on_diversion else ""
            lines.append(
                f"  {h.hospital_id} {h.name[:28]:28s} | Zone {h.zone_id} | "
                f"ER: {h.er_occupancy:.0%} | ICU: {h.icu_occupancy:.0%}"
                f"{div}"
            )
        lines.append("")

        lines += [
            "VALID ACTIONS FOR THIS TASK",
            "─" * 40,
            "  " + " | ".join(meta["valid_actions"]),
            "",
            "GRADER WEIGHTS",
            "─" * 40,
        ]
        for k, v in meta["grader_weights"].items():
            lines.append(f"  {k}: {v:.0%}")

        lines.append("═" * 70)
        return "\n".join(lines)

class ScenarioGenerator:

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.rng    = DeterministicRNG(config.seed, config.task_id)

    def generate(self) -> ScenarioPack:
        logger.debug(
            "Generating scenario: task=%d seed=%d suite=%s",
            self.config.task_id, self.config.seed, self.config.suite.value,
        )

        incidents = IncidentEngine(self.config, self.rng).generate()

        units = FleetBuilder(self.config, self.rng).build()

        hospitals = HospitalNetworkBuilder(self.config, self.rng).build()

        base_matrix = TrafficMatrixBuilder.build_base_matrix(self.rng)
        aug_matrix  = AugmentationPipeline.apply_to_traffic(
            base_matrix, self.config.augment
        )
        slowdowns = TrafficMatrixBuilder.generate_slowdowns(
            self.rng, self.config.augment,
            incident_zones=[i.zone_id for i in incidents],
        )
        traffic = TrafficMatrix(
            matrix=aug_matrix,
            peak_active=self.config.augment.is_peak,
            traffic_multiplier=self.config.augment.base_traffic_multiplier,
            active_slowdowns=slowdowns,
        )

        demand_forecast = DemandForecastBuilder(self.config, self.rng).build()

        mutual_aid = MutualAidBuilder(self.config, self.rng).build()

        gt = GroundTruthAssembler(self.config, incidents, units, hospitals).assemble()

        prompt = AgentPromptBuilder(
            self.config, incidents, units, hospitals, self.config.augment
        ).build()

        stat_sig = self._statistical_signature(incidents, units, hospitals)

        comm_prob = AugmentationPipeline.comm_failure_prob(
            self.config.task_meta.get("comm_failure_prob", 0.0),
            self.config.augment,
        )
        meta = ScenarioMetadata(
            scenario_id=self.config.scenario_id,
            content_hash=self.config.content_hash,
            generator_version=GENERATOR_VERSION,
            schema_version=SCHEMA_VERSION,
            task_id=self.config.task_id,
            task_name=self.config.task_meta["name"],
            difficulty=self.config.difficulty,
            seed=self.config.seed,
            suite=self.config.suite.value,
            augmentation=self.config.augment.to_dict(),
            baseline_score=self.config.baseline_score,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            valid_actions=self.config.task_meta["valid_actions"],
            grader_weights=self.config.task_meta["grader_weights"],
            episode_config={
                "max_steps":         self.config.task_meta["max_steps"],
                "timestep_minutes":  3,
                "max_fleet_size":    len(units),
                "comm_failure_prob": comm_prob,
                "mutual_aid_delay_min": 12,
                "diversion_threshold": 0.90,
                "crew_fatigue_threshold_hours": 10,
                "protocol_compliance_max_bonus": 0.15,
            },
            statistical_signature=stat_sig,
        )

        pack = ScenarioPack(
            metadata=meta,
            incident_queue=incidents,
            fleet_status=units,
            hospital_network=hospitals,
            traffic=traffic,
            demand_forecast=demand_forecast,
            mutual_aid=mutual_aid,
            comms_failure_prob=comm_prob,
            ground_truth=gt,
            agent_prompt_context=prompt,
        )
        return pack

    @staticmethod
    def _statistical_signature(
        incidents: List[GeneratedIncident],
        units: List[GeneratedUnit],
        hospitals: List[GeneratedHospital],
    ) -> Dict[str, Any]:
        if not incidents:
            return {
                "incident_count": 0, "p1_count": 0, "p2_count": 0, "p3_count": 0,
                "mean_survival": 1.0, "fleet_available_pct": 1.0,
                "hospitals_on_diversion": 0, "hospitals_available": len(hospitals),
                "multi_agency_incidents": 0, "trapped_incidents": 0,
                "hazmat_incidents": 0, "micu_required": 0,
            }
        sev_counts = Counter(i.severity for i in incidents)
        micu_req   = sum(1 for i in incidents if i.correct_unit == "MICU")
        return {
            "incident_count":         len(incidents),
            "p1_count":               sev_counts.get("P1", 0),
            "p2_count":               sev_counts.get("P2", 0),
            "p3_count":               sev_counts.get("P3", 0),
            "p0_expectant_count":     sum(1 for i in incidents if i.start_tag == "Expectant"),
            "mean_survival":          round(
                sum(i.current_survival_prob for i in incidents) / len(incidents), 4
            ),
            "min_survival":           round(min(i.current_survival_prob for i in incidents), 4),
            "fleet_size":             len(units),
            "fleet_available_pct":    round(
                sum(1 for u in units if u.status == "available") / max(1, len(units)), 4
            ),
            "fleet_micu_count":       sum(1 for u in units if u.unit_type == "MICU"),
            "fleet_fatigued_count":   sum(1 for u in units if u.crew_fatigued),
            "fleet_comms_lost_count": sum(1 for u in units if not u.comms_active),
            "hospitals_on_diversion": sum(1 for h in hospitals if h.on_diversion),
            "hospitals_available":    sum(1 for h in hospitals if h.accepts_patients),
            "multi_agency_incidents": sum(1 for i in incidents if i.multi_agency),
            "trapped_incidents":      sum(1 for i in incidents if i.trapped),
            "hazmat_incidents":       sum(1 for i in incidents if i.hazmat),
            "micu_required":          micu_req,
            "resource_deficit":       max(0, sev_counts.get("P1", 0) - micu_req),
        }

class ValidationResult(NamedTuple):
    passed: bool
    errors: List[str]
    warnings: List[str]
    score_range_ok: bool
    determinism_ok: bool
    medical_rules_ok: bool

class GroundTruthValidator:

    SEVERITY_TAG_MAP = {
        "P1": ["Immediate"],
        "P2": ["Delayed"],
        "P3": ["Minimal"],
        "P0": ["Expectant"],
    }

    TAG_SEVERITY_MAP = {
        "Immediate": "P1", "Delayed": "P2", "Minimal": "P3", "Expectant": "P0",
    }

    @classmethod
    def validate(cls, pack: ScenarioPack) -> ValidationResult:
        errors:   List[str] = []
        warnings: List[str] = []

        cls._check_severity_tag_consistency(pack, errors, warnings)
        cls._check_hospital_routing(pack, errors, warnings)
        cls._check_fleet_adequacy(pack, errors, warnings)
        cls._check_rpm_start_consistency(pack, errors, warnings)
        cls._check_multi_agency(pack, errors, warnings)
        cls._check_grader_weights(pack, errors, warnings)
        cls._check_score_fields(pack, errors)
        cls._check_determinism(pack, errors)

        score_range_ok   = not any("score" in e.lower() for e in errors)
        determinism_ok   = not any("determinism" in e.lower() for e in errors)
        medical_rules_ok = not any("medical" in e.lower() or "rpm" in e.lower() for e in errors)

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score_range_ok=score_range_ok,
            determinism_ok=determinism_ok,
            medical_rules_ok=medical_rules_ok,
        )

    @classmethod
    def _check_severity_tag_consistency(
        cls, pack: ScenarioPack, errors: List[str], warnings: List[str]
    ) -> None:
        for inc in pack.incident_queue:
            expected_sev = cls.TAG_SEVERITY_MAP.get(inc.start_tag)
            if expected_sev and inc.severity != expected_sev:
                errors.append(
                    f"MEDICAL: {inc.incident_id} — severity={inc.severity} "
                    f"but start_tag={inc.start_tag} → expected severity={expected_sev}"
                )
            if not (0.0 <= inc.current_survival_prob <= 1.0):
                errors.append(
                    f"SCORE: {inc.incident_id} — current_survival_prob "
                    f"{inc.current_survival_prob} out of [0.0, 1.0]"
                )
            if not (0.0 <= inc.survival_at_zero <= 1.0):
                errors.append(
                    f"SCORE: {inc.incident_id} — survival_at_zero "
                    f"{inc.survival_at_zero} out of [0.0, 1.0]"
                )

    @classmethod
    def _check_hospital_routing(
        cls, pack: ScenarioPack, errors: List[str], warnings: List[str]
    ) -> None:
        accepting = [h for h in pack.hospital_network if h.accepts_patients]
        if not accepting:
            errors.append(
                "HOSPITAL: All hospitals on diversion — no valid routing target. "
                "Reduce er_occupancy or augment.hospital_under_surge."
            )
        required_specs = {
            i.required_specialty
            for i in pack.incident_queue
            if i.required_specialty
        }
        for spec in required_specs:
            reachable = [h for h in accepting if spec in h.specialties]
            if not reachable:
                warnings.append(
                    f"ROUTING: Specialty '{spec}' not available in any non-diverted hospital. "
                    "Agent will be penalised for hospital mismatch but this is valid hard scenario."
                )

    @classmethod
    def _check_fleet_adequacy(
        cls, pack: ScenarioPack, errors: List[str], warnings: List[str]
    ) -> None:
        available_units = [u for u in pack.fleet_status if u.status == "available"]
        micu_available  = sum(1 for u in available_units if u.unit_type == "MICU")
        micu_required   = sum(1 for i in pack.incident_queue if i.correct_unit == "MICU")

        if not available_units:
            errors.append("FLEET: No available units. Scenario is unresolvable.")
        if micu_required > 0 and micu_available == 0:
            warnings.append(
                f"FLEET: {micu_required} MICU-required incidents but 0 MICU available. "
                "Acceptable for hard tasks (agent must request mutual aid)."
            )
        if len(available_units) == 0 and len(pack.incident_queue) > 0:
            errors.append("FLEET: Incident queue non-empty but fleet completely unavailable.")

    @classmethod
    def _check_rpm_start_consistency(
        cls, pack: ScenarioPack, errors: List[str], warnings: List[str]
    ) -> None:
        for inc in pack.incident_queue:
            rpm = inc.rpm
            if not rpm:
                continue
            resp = rpm.get("respirations", "")
            pulse = rpm.get("pulse", "")
            ms    = rpm.get("mental_status", "")

            derived = cls._rpm_to_start_tag(resp, pulse, ms)
            actual  = inc.start_tag

            if actual == "Expectant":
                if resp != "absent" or pulse != "absent":
                    warnings.append(
                        f"RPM: {inc.incident_id} tagged Expectant but RPM is not all-absent. "
                        "This is allowed for near-death scenarios."
                    )
                continue

            if derived != actual:
                warnings.append(
                    f"RPM: {inc.incident_id} — RPM suggests '{derived}' "
                    f"but scenario uses '{actual}'. "
                    "Verify clinical template is intentionally ambiguous."
                )

    @staticmethod
    def _rpm_to_start_tag(resp: str, pulse: str, mental: str) -> str:
        if resp == "absent":
            return "Expectant"
        if pulse == "absent":
            return "Expectant"
        if mental == "unresponsive":
            return "Immediate"
        if resp == "present_abnormal" or pulse == "present_weak":
            return "Immediate"
        if mental == "altered":
            return "Delayed"
        return "Minimal"

    @classmethod
    def _check_multi_agency(
        cls, pack: ScenarioPack, errors: List[str], warnings: List[str]
    ) -> None:
        for inc in pack.incident_queue:
            if inc.trapped and not inc.multi_agency:
                errors.append(
                    f"MEDICAL: {inc.incident_id} — trapped=True but multi_agency=False. "
                    "Trapped victims always require Fire extrication."
                )
            if inc.hazmat and "Fire" not in inc.agencies_required:
                errors.append(
                    f"MEDICAL: {inc.incident_id} — hazmat=True but Fire not in agencies_required."
                )

    @classmethod
    def _check_grader_weights(
        cls, pack: ScenarioPack, errors: List[str], warnings: List[str]
    ) -> None:
        weights = pack.metadata.grader_weights
        total   = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"GRADER: Weights sum to {total:.4f} ≠ 1.0 for task {pack.metadata.task_id}. "
                f"Weights: {weights}"
            )

    @classmethod
    def _check_score_fields(cls, pack: ScenarioPack, errors: List[str]) -> None:
        baseline = pack.metadata.baseline_score
        if not (0.0 <= baseline <= 1.0):
            errors.append(f"SCORE: baseline_score {baseline} out of [0.0, 1.0]")

    @classmethod
    def _check_determinism(cls, pack: ScenarioPack, errors: List[str]) -> None:
        config   = ScenarioConfig(
            task_id=pack.metadata.task_id,
            seed=pack.metadata.seed,
            suite=ScenarioSuite(pack.metadata.suite),
            augment=AugmentationSpec(**pack.metadata.augmentation),
        )
        repack = ScenarioGenerator(config).generate()
        if repack.metadata.content_hash != pack.metadata.content_hash:
            errors.append(
                f"DETERMINISM: Re-generated scenario has different content hash. "
                f"Original: {pack.metadata.content_hash}, "
                f"Regenerated: {repack.metadata.content_hash}"
            )

class StatisticsEngine:

    def __init__(self, packs: List[ScenarioPack]) -> None:
        self.packs = packs

    def run(self) -> Dict[str, Any]:
        return {
            "corpus_summary":     self._corpus_summary(),
            "severity_dist":      self._severity_distribution(),
            "condition_coverage": self._condition_coverage(),
            "zone_coverage":      self._zone_coverage(),
            "hospital_coverage":  self._hospital_coverage(),
            "survival_stats":     self._survival_statistics(),
            "fleet_stats":        self._fleet_statistics(),
            "calibration":        self._difficulty_calibration(),
            "anomalies":          self._anomaly_detection(),
            "traffic_stats":      self._traffic_statistics(),
        }

    def _corpus_summary(self) -> Dict[str, Any]:
        by_task: Dict[int, int] = Counter(p.metadata.task_id for p in self.packs)
        by_diff: Dict[str, int] = Counter(p.metadata.difficulty for p in self.packs)
        return {
            "total_scenarios":   len(self.packs),
            "by_task":           dict(sorted(by_task.items())),
            "by_difficulty":     dict(by_diff),
            "total_incidents":   sum(p.incident_count for p in self.packs),
            "total_p1":          sum(p.p1_count for p in self.packs),
            "unique_conditions": len(self._all_conditions()),
            "unique_seeds":      len({p.metadata.seed for p in self.packs}),
        }

    def _severity_distribution(self) -> Dict[str, Any]:
        all_sev = [i.severity for p in self.packs for i in p.incident_queue]
        total   = len(all_sev) or 1
        counts  = Counter(all_sev)
        return {
            "P0_expectant": {"count": counts["P0"], "pct": round(counts["P0"] / total, 4)},
            "P1_immediate": {"count": counts["P1"], "pct": round(counts["P1"] / total, 4)},
            "P2_delayed":   {"count": counts["P2"], "pct": round(counts["P2"] / total, 4)},
            "P3_minimal":   {"count": counts["P3"], "pct": round(counts["P3"] / total, 4)},
            "total":        total,
        }

    def _condition_coverage(self) -> Dict[str, Any]:
        conditions = self._all_conditions()
        all_possible = {t["condition"] for t in INCIDENT_TEMPLATE_POOL}
        return {
            "covered":    sorted(conditions),
            "covered_pct": round(len(conditions) / max(1, len(all_possible)), 4),
            "uncovered":  sorted(all_possible - conditions),
        }

    def _zone_coverage(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for p in self.packs:
            for i in p.incident_queue:
                counts[i.zone_id] += 1
        return dict(sorted(counts.items()))

    def _hospital_coverage(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for p in self.packs:
            gt = p.ground_truth.get("optimal_dispatch", {})
            for v in gt.values():
                if v.get("hospital_id"):
                    counts[v["hospital_id"]] += 1
        return dict(sorted(counts.items()))

    def _survival_statistics(self) -> Dict[str, Any]:
        surv = [i.current_survival_prob
                for p in self.packs for i in p.incident_queue]
        if not surv:
            return {}
        arr = np.array(surv)
        return {
            "mean":   round(float(arr.mean()), 4),
            "std":    round(float(arr.std()),  4),
            "min":    round(float(arr.min()),  4),
            "max":    round(float(arr.max()),  4),
            "p25":    round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.percentile(arr, 50)), 4),
            "p75":    round(float(np.percentile(arr, 75)), 4),
            "p90":    round(float(np.percentile(arr, 90)), 4),
        }

    def _fleet_statistics(self) -> Dict[str, Any]:
        micu = als = bls = fatigued = comms_lost = 0
        for p in self.packs:
            for u in p.fleet_status:
                if u.unit_type == "MICU": micu += 1
                elif u.unit_type == "ALS": als  += 1
                else:                      bls  += 1
                if u.crew_fatigued:   fatigued   += 1
                if not u.comms_active: comms_lost += 1
        total = micu + als + bls or 1
        return {
            "MICU_pct":       round(micu / total, 4),
            "ALS_pct":        round(als  / total, 4),
            "BLS_pct":        round(bls  / total, 4),
            "fatigued_pct":   round(fatigued   / total, 4),
            "comms_lost_pct": round(comms_lost / total, 4),
        }

    def _difficulty_calibration(self) -> Dict[str, Any]:
        by_difficulty: Dict[str, List[ScenarioPack]] = defaultdict(list)
        for p in self.packs:
            by_difficulty[p.metadata.difficulty].append(p)

        calibration: Dict[str, Any] = {}
        for diff, diff_packs in by_difficulty.items():
            if not diff_packs:
                continue
            sig_list = [p.metadata.statistical_signature for p in diff_packs]
            p1_pcts     = [s.get("p1_count", 0) / max(1, s.get("incident_count", 1))
                           for s in sig_list]
            surv_means  = [s.get("mean_survival", 1.0)     for s in sig_list if "mean_survival" in s]
            div_counts  = [s.get("hospitals_on_diversion", 0) for s in sig_list]

            calibration[diff] = {
                "scenario_count":     len(diff_packs),
                "mean_p1_pct":        round(sum(p1_pcts) / len(p1_pcts), 4)    if p1_pcts    else None,
                "mean_survival":      round(sum(surv_means) / len(surv_means), 4) if surv_means else None,
                "mean_diversions":    round(sum(div_counts) / len(div_counts), 4) if div_counts else None,
                "baseline_expected":  {
                    "easy":   "0.61-0.72",
                    "medium": "0.38-0.44",
                    "hard":   "0.17-0.29",
                }[diff],
            }
        return calibration

    def _anomaly_detection(self) -> List[Dict[str, Any]]:
        anomalies: List[Dict[str, Any]] = []
        for p in self.packs:
            for i in p.incident_queue:
                if i.current_survival_prob < 0 or i.current_survival_prob > 1:
                    anomalies.append({
                        "type": "survival_out_of_range",
                        "scenario_id": p.scenario_id,
                        "incident_id": i.incident_id,
                        "value": i.current_survival_prob,
                    })
            if p.traffic.traffic_multiplier > 3.0:
                anomalies.append({
                    "type": "extreme_traffic",
                    "scenario_id": p.scenario_id,
                    "multiplier": p.traffic.traffic_multiplier,
                })
            if all(h.on_diversion for h in p.hospital_network):
                anomalies.append({
                    "type": "all_hospitals_diverted",
                    "scenario_id": p.scenario_id,
                })
        return anomalies

    def _traffic_statistics(self) -> Dict[str, Any]:
        mults = [p.traffic.traffic_multiplier for p in self.packs]
        if not mults:
            return {}
        arr = np.array(mults)
        return {
            "mean":   round(float(arr.mean()), 4),
            "std":    round(float(arr.std()),  4),
            "min":    round(float(arr.min()),  4),
            "max":    round(float(arr.max()),  4),
            "peak_pct": round(sum(1 for p in self.packs if p.traffic.peak_active) / len(self.packs), 4),
        }

    def _all_conditions(self) -> Set[str]:
        return {i.condition for p in self.packs for i in p.incident_queue}

    def to_markdown_report(self, stats: Dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines: List[str] = [
            "# EMERGI-ENV Scenario Generation Report",
            f"Generated: {now} | Generator v{GENERATOR_VERSION}",
            "",
            "## Corpus Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
        ]
        cs = stats["corpus_summary"]
        for k, v in cs.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
        lines += [
            "",
            "## Severity Distribution",
            "",
            "| Severity | Count | Percentage |",
            "|----------|-------|------------|",
        ]
        sd = stats["severity_dist"]
        for sev, data in sd.items():
            if sev == "total": continue
            if isinstance(data, dict):
                lines.append(f"| {sev} | {data['count']} | {data['pct']:.1%} |")
        lines += [
            "",
            "## Difficulty Calibration",
            "",
            "| Difficulty | Scenarios | Mean P1% | Mean Survival | Baseline Expected |",
            "|------------|-----------|----------|---------------|-------------------|",
        ]
        for diff, cal in stats["calibration"].items():
            p1 = f"{cal['mean_p1_pct']:.0%}" if cal["mean_p1_pct"] is not None else "N/A"
            surv = f"{cal['mean_survival']:.2%}" if cal["mean_survival"] is not None else "N/A"
            lines.append(
                f"| {diff.title()} | {cal['scenario_count']} | {p1} | "
                f"{surv} | {cal['baseline_expected']} |"
            )
        lines += [
            "",
            "## Survival Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in stats["survival_stats"].items():
            lines.append(f"| {k.title()} | {v:.4f} |")
        lines += [
            "",
            "## Anomalies",
            f"Total anomalies detected: **{len(stats['anomalies'])}**",
        ]
        if stats["anomalies"]:
            lines.append("")
            for a in stats["anomalies"][:10]:
                lines.append(f"- `{a['type']}` in `{a['scenario_id']}`")
            if len(stats["anomalies"]) > 10:
                lines.append(f"- ... and {len(stats['anomalies']) - 10} more")
        lines += ["", "## Zone Coverage", ""]
        for z, cnt in stats["zone_coverage"].items():
            bar = "█" * min(cnt, 20)
            lines.append(f"- {z}: {bar} ({cnt})")
        lines += ["", "---", f"*EMERGI-ENV Scenario Generator v{GENERATOR_VERSION}*"]
        return "\n".join(lines)

class ScenarioCatalog:

    def __init__(self, catalog_path: Path) -> None:
        self.path    = catalog_path
        self.entries: List[Dict[str, Any]] = []
        if catalog_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            logger.info("Loaded catalog: %d entries from %s", len(self.entries), self.path)
        except Exception as exc:
            logger.warning("Failed to load catalog %s: %s", self.path, exc)
            self.entries = []

    def add(self, pack: ScenarioPack, file_path: Path) -> None:
        entry = {
            "scenario_id":     pack.scenario_id,
            "task_id":         pack.metadata.task_id,
            "task_name":       pack.metadata.task_name,
            "difficulty":      pack.metadata.difficulty,
            "seed":            pack.metadata.seed,
            "suite":           pack.metadata.suite,
            "augmentation":    pack.metadata.augmentation,
            "file_path":       str(file_path.name),
            "content_hash":    pack.metadata.content_hash,
            "incident_count":  pack.incident_count,
            "p1_count":        pack.p1_count,
            "baseline_score":  pack.metadata.baseline_score,
            "validation_passed": pack.validation_result is not None and pack.validation_result.get("passed", False),
        }
        existing = next(
            (i for i, e in enumerate(self.entries) if e["scenario_id"] == pack.scenario_id),
            None
        )
        if existing is not None:
            self.entries[existing] = entry
        else:
            self.entries.append(entry)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version":           "1.0",
            "generator_version": GENERATOR_VERSION,
            "generated_at_utc":  datetime.now(timezone.utc).isoformat(),
            "total_entries":     len(self.entries),
            "entries":           sorted(self.entries, key=lambda e: (e["task_id"], e["seed"])),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        logger.info("Saved catalog: %d entries → %s", len(self.entries), self.path)

    def filter(
        self,
        task_id:    Optional[int]   = None,
        difficulty: Optional[str]   = None,
        suite:      Optional[str]   = None,
    ) -> List[Dict[str, Any]]:
        result = self.entries
        if task_id    is not None: result = [e for e in result if e["task_id"] == task_id]
        if difficulty is not None: result = [e for e in result if e["difficulty"] == difficulty]
        if suite      is not None: result = [e for e in result if e["suite"] == suite]
        return result

class OutputFormatter:

    CSV_FIELDS = [
        "scenario_id", "task_id", "difficulty", "seed", "suite", "weather",
        "hour_of_day", "incident_count", "p1_count", "p2_count", "p3_count",
        "fleet_size", "hospitals_on_diversion", "mean_survival",
        "comms_failure_prob", "traffic_multiplier", "baseline_score",
        "content_hash", "validation_passed",
    ]

    def __init__(self, output_dir: Path, fmt: OutputFormat = OutputFormat.JSON) -> None:
        self.output_dir = output_dir
        self.fmt        = fmt
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_pack(
        self,
        pack: ScenarioPack,
        include_ground_truth: bool = False,
    ) -> Path:
        if self.fmt == OutputFormat.JSON:
            return self._write_json(pack, include_ground_truth)
        if self.fmt == OutputFormat.JSONL:
            return self._write_jsonl(pack, include_ground_truth)
        if self.fmt in (OutputFormat.CSV, OutputFormat.TSV):
            return self._write_flat(pack)
        json_path = self._write_json(pack, include_ground_truth)
        self._write_jsonl(pack, include_ground_truth)
        self._write_flat(pack)
        return json_path

    def _write_json(self, pack: ScenarioPack, include_gt: bool = False) -> Path:
        path = self.output_dir / f"{pack.scenario_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(pack.to_json(include_ground_truth=include_gt, indent=2))
        logger.debug("Wrote JSON: %s", path)
        return path

    def _write_jsonl(self, pack: ScenarioPack, include_gt: bool = False) -> Path:
        path = self.output_dir / "scenarios.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    pack.to_dict(include_ground_truth=include_gt),
                    sort_keys=True,
                    default=_json_default,
                ) + "\n"
            )
        return path

    def _write_flat(self, pack: ScenarioPack) -> Path:
        sep  = "\t" if self.fmt == OutputFormat.TSV else ","
        ext  = "tsv" if self.fmt == OutputFormat.TSV else "csv"
        path = self.output_dir / f"scenarios.{ext}"

        row = {
            "scenario_id":          pack.scenario_id,
            "task_id":              pack.metadata.task_id,
            "difficulty":           pack.metadata.difficulty,
            "seed":                 pack.metadata.seed,
            "suite":                pack.metadata.suite,
            "weather":              pack.metadata.augmentation.get("weather", "clear"),
            "hour_of_day":          pack.metadata.augmentation.get("hour_of_day", 10),
            "incident_count":       pack.incident_count,
            "p1_count":             pack.p1_count,
            "p2_count":             pack.p2_count,
            "p3_count":             pack.p3_count,
            "fleet_size":           len(pack.fleet_status),
            "hospitals_on_diversion": sum(1 for h in pack.hospital_network if h.on_diversion),
            "mean_survival":        pack.metadata.statistical_signature.get("mean_survival", ""),
            "comms_failure_prob":   pack.comms_failure_prob,
            "traffic_multiplier":   pack.traffic.traffic_multiplier,
            "baseline_score":       pack.metadata.baseline_score,
            "content_hash":         pack.metadata.content_hash,
            "validation_passed":    pack.validation_result.get("passed", False)
                                    if pack.validation_result else "",
        }
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS, delimiter=sep)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return path

    def write_report(self, report_md: str) -> Path:
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"generation_report_{ts}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info("Wrote report: %s", path)
        return path

def _seed_range_for_suite(suite: ScenarioSuite, n_seeds: int) -> List[int]:
    if suite == ScenarioSuite.BENCHMARK:
        return [42, 137, 256, 512, 1024, 2048, 4096, 8192][:n_seeds]
    if suite == ScenarioSuite.REGRESSION:
        return [42, 100, 999][:n_seeds]
    if suite == ScenarioSuite.ADVERSARIAL:
        primes = [103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
                  157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
        return primes[:n_seeds]
    step = max(1, (SEED_RANGE[1] - SEED_RANGE[0]) // n_seeds)
    return [SEED_RANGE[0] + i * step for i in range(n_seeds)]

def build_curriculum_schedule(n_seeds: int) -> List[ScenarioConfig]:
    configs: List[ScenarioConfig] = []
    seeds = _seed_range_for_suite(ScenarioSuite.CURRICULUM, n_seeds)

    for seed in seeds[:max(1, n_seeds // 4)]:
        configs.append(ScenarioConfig(task_id=3, seed=seed, suite=ScenarioSuite.CURRICULUM))

    for seed in seeds[:max(1, n_seeds // 3)]:
        configs.append(ScenarioConfig(task_id=1, seed=seed, suite=ScenarioSuite.CURRICULUM))
        configs.append(ScenarioConfig(task_id=2, seed=seed, suite=ScenarioSuite.CURRICULUM))

    for seed in seeds[:max(1, n_seeds // 2)]:
        configs.append(ScenarioConfig(task_id=4, seed=seed, suite=ScenarioSuite.CURRICULUM))
        configs.append(ScenarioConfig(task_id=5, seed=seed, suite=ScenarioSuite.CURRICULUM))
        configs.append(ScenarioConfig(task_id=6, seed=seed, suite=ScenarioSuite.CURRICULUM))

    for seed in seeds:
        configs.append(ScenarioConfig(task_id=7, seed=seed, suite=ScenarioSuite.CURRICULUM))
        configs.append(ScenarioConfig(task_id=8, seed=seed, suite=ScenarioSuite.CURRICULUM))

    for seed in seeds[max(0, n_seeds - 3):]:
        configs.append(ScenarioConfig(task_id=9, seed=seed, suite=ScenarioSuite.CURRICULUM))

    return configs

def build_adversarial_suite(task_ids: List[int], n_seeds: int) -> List[ScenarioConfig]:
    configs: List[ScenarioConfig] = []
    seeds = _seed_range_for_suite(ScenarioSuite.ADVERSARIAL, n_seeds)
    rng_base = DeterministicRNG(99999, task_id=0)

    for task_id in task_ids:
        for seed in seeds:
            adv_rng    = DeterministicRNG(seed, task_id)
            adv_augment = AugmentationPipeline.adversarial_augment(adv_rng)
            configs.append(ScenarioConfig(
                task_id=task_id,
                seed=seed,
                suite=ScenarioSuite.ADVERSARIAL,
                augment=adv_augment,
            ))
    return configs

def build_stress_test_suite(task_ids: List[int], n_seeds: int) -> List[ScenarioConfig]:
    max_stress = AugmentationSpec(
        weather="flood",
        hour_of_day=18,
        day_of_week="monday",
        season="monsoon",
        traffic_incident=True,
        hospital_under_surge=True,
        crew_fatigue_pre=True,
    )
    seeds = _seed_range_for_suite(ScenarioSuite.STRESS_TEST, n_seeds)
    return [
        ScenarioConfig(task_id=tid, seed=s, suite=ScenarioSuite.STRESS_TEST, augment=max_stress)
        for tid in task_ids for s in seeds
    ]

def build_mci_focus_suite(n_seeds: int) -> List[ScenarioConfig]:
    seeds  = _seed_range_for_suite(ScenarioSuite.MCI_FOCUS, n_seeds)
    hours  = [7, 8, 12, 17, 18, 22, 2]
    weathers = ["clear", "heavy_rain", "fog", "extreme_heat"]
    configs: List[ScenarioConfig] = []
    for seed in seeds:
        for task_id in [7, 8, 9]:
            rng   = DeterministicRNG(seed, task_id)
            augm  = AugmentationSpec(
                weather=rng.choice(weathers, "mci_aug"),
                hour_of_day=rng.choice(hours, "mci_aug"),
                day_of_week=rng.choice(DAYS_OF_WEEK[:6], "mci_aug"),
                season=rng.choice(SEASONS, "mci_aug"),
                traffic_incident=rng.bernoulli(0.5, "mci_aug"),
                hospital_under_surge=rng.bernoulli(0.4, "mci_aug"),
                crew_fatigue_pre=rng.bernoulli(0.5, "mci_aug"),
            )
            configs.append(ScenarioConfig(
                task_id=task_id, seed=seed, suite=ScenarioSuite.MCI_FOCUS, augment=augm,
            ))
    return configs

class BatchGenerator:

    def __init__(
        self,
        output_dir:   Path,
        fmt:          OutputFormat = OutputFormat.JSON,
        validate:     bool         = True,
        include_gt:   bool         = False,
        run_stats:    bool         = True,
    ) -> None:
        self.output_dir  = output_dir
        self.formatter   = OutputFormatter(output_dir, fmt)
        self.catalog     = ScenarioCatalog(output_dir / "catalog.json")
        self.validate    = validate
        self.include_gt  = include_gt
        self.run_stats   = run_stats
        self.generated:  List[ScenarioPack] = []

    def run(self, configs: List[ScenarioConfig]) -> List[ScenarioPack]:
        total     = len(configs)
        t_start   = time.perf_counter()
        logger.info("Starting batch generation: %d scenarios", total)

        for idx, cfg in enumerate(configs, 1):
            try:
                pack = self._generate_one(cfg)
                self.generated.append(pack)

                file_path = self.formatter.write_pack(pack, include_ground_truth=self.include_gt)
                self.catalog.add(pack, file_path)

                pct = idx / total * 100
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / idx) * (total - idx)
                logger.info(
                    "[%d/%d | %.0f%%] %s  "
                    "| P1=%d P2=%d P3=%d | Validation: %s | ETA: %.0fs",
                    idx, total, pct, cfg.scenario_id,
                    pack.p1_count, pack.p2_count, pack.p3_count,
                    "✓" if (pack.validation_result or {}).get("passed") else "⚠",
                    eta,
                )
            except Exception as exc:
                logger.error(
                    "Failed generating %s: %s", cfg.scenario_id, exc, exc_info=True
                )

        self.catalog.save()
        t_elapsed = time.perf_counter() - t_start
        logger.info(
            "Batch complete: %d/%d generated in %.1fs (%.2f/s)",
            len(self.generated), total, t_elapsed,
            len(self.generated) / max(1, t_elapsed),
        )

        if self.run_stats and self.generated:
            self._run_statistics()

        return self.generated

    def _generate_one(self, cfg: ScenarioConfig) -> ScenarioPack:
        gen  = ScenarioGenerator(cfg)
        pack = gen.generate()

        if self.validate:
            result = GroundTruthValidator.validate(pack)
            pack.validation_result = {
                "passed":          result.passed,
                "errors":          result.errors,
                "warnings":        result.warnings,
                "score_range_ok":  result.score_range_ok,
                "determinism_ok":  result.determinism_ok,
                "medical_rules_ok": result.medical_rules_ok,
            }
            if result.errors:
                logger.warning(
                    "Validation FAILED for %s: %s",
                    cfg.scenario_id, "; ".join(result.errors[:3])
                )
            if result.warnings:
                logger.debug(
                    "Validation warnings for %s: %s",
                    cfg.scenario_id, "; ".join(result.warnings[:2])
                )
        return pack

    def _run_statistics(self) -> None:
        engine = StatisticsEngine(self.generated)
        stats  = engine.run()
        report = engine.to_markdown_report(stats)

        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True, default=_json_default)

        report_path = self.formatter.write_report(report)

        logger.info("Statistics saved to %s and %s", stats_path, report_path)
        self._print_summary(stats)

    @staticmethod
    def _print_summary(stats: Dict[str, Any]) -> None:
        cs = stats["corpus_summary"]
        print("\n" + "═" * 60)
        print(f"  EMERGI-ENV  ·  Generation Summary")
        print("═" * 60)
        print(f"  Total scenarios : {cs['total_scenarios']}")
        print(f"  Total incidents : {cs['total_incidents']}")
        print(f"  Total P1        : {cs['total_p1']}")
        print(f"  Unique conditions: {cs['unique_conditions']}")
        print(f"  Anomalies       : {len(stats.get('anomalies', []))}")
        print("")
        print("  Difficulty Calibration:")
        for diff, cal in stats["calibration"].items():
            p1  = f"{cal['mean_p1_pct']:.0%}" if cal["mean_p1_pct"] is not None else "N/A"
            surv = f"{cal['mean_survival']:.0%}" if cal["mean_survival"] is not None else "N/A"
            print(
                f"    {diff:6s}  |  {cal['scenario_count']:3d} scenarios  |  "
                f"Mean P1: {p1:5s}  |  Mean Survival: {surv}"
            )
        print("═" * 60 + "\n")

def validate_catalog(catalog_path: Path, scenario_dir: Path) -> None:
    if not catalog_path.exists():
        logger.error("Catalog not found: %s", catalog_path)
        sys.exit(1)

    with open(catalog_path) as f:
        catalog = json.load(f)

    entries  = catalog.get("entries", [])
    passed   = failed = warned = 0
    logger.info("Validating %d catalog entries...", len(entries))

    for entry in entries:
        fpath = scenario_dir / entry["file_path"]
        if not fpath.exists():
            logger.warning("Missing file: %s", fpath)
            failed += 1
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            cfg = ScenarioConfig(
                task_id=entry["task_id"],
                seed=entry["seed"],
                suite=ScenarioSuite(entry["suite"]),
                augment=AugmentationSpec(**entry.get("augmentation", {})),
            )
            regen_hash = cfg.content_hash
            if regen_hash != entry["content_hash"]:
                logger.error(
                    "DETERMINISM FAIL: %s — hash mismatch", entry["scenario_id"]
                )
                failed += 1
            else:
                logger.debug("PASS: %s", entry["scenario_id"])
                passed += 1
                if entry.get("validation_warnings"):
                    warned += 1
        except Exception as exc:
            logger.error("ERROR validating %s: %s", entry.get("scenario_id"), exc)
            failed += 1

    total = len(entries)
    print(f"\nValidation Complete: {passed}/{total} passed | {failed} failed | {warned} warned")
    if failed > 0:
        sys.exit(1)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_scenarios.py",
        description=textwrap.dedent("""\
            EMERGI-ENV Seed-Based Scenario Generator
            ─────────────────────────────────────────
            Generates deterministic, richly-annotated scenario packs
            for all 9 tasks across 3 difficulty tiers.

            Output: data/scenarios/ (default) or --output-dir
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sel = parser.add_argument_group("Scenario Selection")
    sel.add_argument(
        "--tasks", "-t", nargs="+", type=int,
        choices=list(range(1, 10)), metavar="TASK_ID",
        default=list(range(1, 10)),
        help="Task IDs to generate (default: all 1-9)",
    )
    sel.add_argument(
        "--seeds", "-s", type=int, default=DEFAULT_SEEDS_PER_TASK,
        help=f"Number of seeds per task (default: {DEFAULT_SEEDS_PER_TASK}, max: {MAX_SEEDS_PER_TASK})",
    )
    sel.add_argument(
        "--seed-list", nargs="+", type=int, metavar="SEED",
        help="Explicit seed list (overrides --seeds)",
    )
    sel.add_argument(
        "--suite", choices=[s.value for s in ScenarioSuite],
        default=ScenarioSuite.STANDARD.value,
        help="Scenario suite strategy (default: standard)",
    )
    sel.add_argument(
        "--all", action="store_true",
        help="Generate all 9 tasks with default seeds (equivalent to --tasks 1 2 3 4 5 6 7 8 9)",
    )

    aug = parser.add_argument_group("Augmentation")
    aug.add_argument(
        "--weather",
        choices=list(WEATHER_PROFILES.keys()),
        default=None,
        help="Force specific weather for all scenarios (default: random per scenario)",
    )
    aug.add_argument(
        "--hour", type=int, default=None,
        choices=list(range(24)),
        help="Force specific hour of day (0-23)",
    )
    aug.add_argument(
        "--no-augment", action="store_true",
        help="Disable all augmentation (use baseline AugmentationSpec)",
    )

    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("data/scenarios"),
        help="Output directory (default: data/scenarios)",
    )
    out.add_argument(
        "--format", "-f", choices=[f.value for f in OutputFormat],
        default=OutputFormat.JSON.value,
        help="Output format (default: json)",
    )
    out.add_argument(
        "--include-ground-truth", action="store_true",
        help="Include hidden ground truth in output (for debugging only — do not share with agents!)",
    )

    modes = parser.add_argument_group("Operation Modes")
    modes.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing catalog (no generation)",
    )
    modes.add_argument(
        "--catalog", type=Path,
        default=None,
        help="Catalog path for --validate-only (default: <output-dir>/catalog.json)",
    )
    modes.add_argument(
        "--stats", action="store_true",
        help="Run statistical analysis after generation",
    )
    modes.add_argument(
        "--no-validate", action="store_true",
        help="Skip per-scenario validation (faster)",
    )
    modes.add_argument(
        "--dry-run", action="store_true",
        help="Generate but do not write files (useful for testing)",
    )
    modes.add_argument(
        "--calibrate", action="store_true",
        help="Run calibration report and print to stdout",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress INFO logging (errors only)",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"EMERGI-ENV Scenario Generator v{GENERATOR_VERSION}",
    )

    return parser

def resolve_configs(args: argparse.Namespace) -> List[ScenarioConfig]:
    task_ids = list(range(1, 10)) if args.all else args.tasks
    suite    = ScenarioSuite(args.suite)

    if args.seed_list:
        seeds = [s for s in args.seed_list if SEED_RANGE[0] <= s <= SEED_RANGE[1]]
        if not seeds:
            raise ValueError(f"No valid seeds in range {SEED_RANGE}")
    else:
        n = min(args.seeds, MAX_SEEDS_PER_TASK)
        seeds = _seed_range_for_suite(suite, n)

    if suite == ScenarioSuite.CURRICULUM:
        return build_curriculum_schedule(len(seeds))
    if suite == ScenarioSuite.ADVERSARIAL:
        return build_adversarial_suite(task_ids, len(seeds))
    if suite == ScenarioSuite.STRESS_TEST:
        return build_stress_test_suite(task_ids, len(seeds))
    if suite == ScenarioSuite.MCI_FOCUS:
        return build_mci_focus_suite(len(seeds))

    configs: List[ScenarioConfig] = []
    for task_id in task_ids:
        for seed in seeds:
            rng_cfg = DeterministicRNG(seed, task_id)

            if args.no_augment:
                augment = AugmentationSpec()
            elif args.weather or args.hour is not None:
                default_aug = AugmentationPipeline.random_augment(rng_cfg, task_id)
                augment = AugmentationSpec(
                    weather=args.weather or default_aug.weather,
                    hour_of_day=args.hour if args.hour is not None else default_aug.hour_of_day,
                    day_of_week=default_aug.day_of_week,
                    season=default_aug.season,
                    traffic_incident=default_aug.traffic_incident,
                    hospital_under_surge=default_aug.hospital_under_surge,
                    crew_fatigue_pre=default_aug.crew_fatigue_pre,
                )
            else:
                augment = AugmentationPipeline.random_augment(rng_cfg, task_id)

            configs.append(ScenarioConfig(
                task_id=task_id, seed=seed, suite=suite, augment=augment,
            ))
    return configs

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    logger.info("EMERGI-ENV Scenario Generator v%s", GENERATOR_VERSION)
    logger.info("Output dir: %s", args.output_dir)

    if args.validate_only:
        catalog_path = args.catalog or (args.output_dir / "catalog.json")
        validate_catalog(catalog_path, args.output_dir)
        return 0

    try:
        configs = resolve_configs(args)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    logger.info(
        "Resolved %d scenario configs | Suite: %s | Tasks: %s",
        len(configs),
        args.suite,
        sorted({c.task_id for c in configs}),
    )

    if args.dry_run:
        logger.info("DRY RUN — generating but not writing files")
        ok = error_count = 0
        for cfg in configs:
            try:
                pack = ScenarioGenerator(cfg).generate()
                logger.debug("DRY RUN OK: %s | incidents=%d", cfg.scenario_id, pack.incident_count)
                ok += 1
            except Exception as exc:
                logger.error("DRY RUN FAIL: %s — %s", cfg.scenario_id, exc)
                error_count += 1
        print(f"Dry run: {ok} OK | {error_count} errors")
        return 0 if error_count == 0 else 1

    batch = BatchGenerator(
        output_dir=args.output_dir,
        fmt=OutputFormat(args.format),
        validate=not args.no_validate,
        include_gt=args.include_ground_truth,
        run_stats=args.stats,
    )
    packs = batch.run(configs)

    if args.calibrate and packs:
        engine = StatisticsEngine(packs)
        stats  = engine.run()
        print("\n" + engine.to_markdown_report(stats))

    logger.info("Done. Generated %d scenario packs in %s", len(packs), args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())