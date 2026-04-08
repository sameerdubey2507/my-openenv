import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto, unique
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__version__: str = "2.0.0"
__author__: str = "EMERGI-ENV Team"
__description__: str = (
    "Emergency Medical Intelligence & Resource Governance Environment — "
    "Maharashtra 108/112 EMS Reinforcement Learning Simulation"
)
__openenv_spec_version__: str = "1.0"
__schema_version__: int = 7
__min_python__: str = "3.11"

logger = logging.getLogger("emergi_env.models")

TIMESTEP_MINUTES: int = 3
MAX_STEPS_PER_EPISODE: int = 100
NUM_ZONES: int = 36
ACTIVE_ZONES_PER_EPISODE: int = 12
NUM_HOSPITALS: int = 28
FLEET_SIZE_DEFAULT: int = 24
INCIDENT_QUEUE_MIN: int = 1
INCIDENT_QUEUE_MAX: int = 45

TRAFFIC_UPDATE_EVERY_N_STEPS: int = 3
PEAK_MORNING_HOURS: Tuple[int, ...] = (7, 8, 9)
PEAK_EVENING_HOURS: Tuple[int, ...] = (17, 18, 19)
PEAK_HOUR_MULTIPLIER_MAX: float = 1.95
SECONDARY_SLOWDOWN_MULTIPLIER: float = 1.40
SECONDARY_SLOWDOWN_DURATION_STEPS: int = 6

DIVERSION_THRESHOLD_PCT: float = 0.90
DIVERSION_PENALTY: float = -0.30
DIVERSION_REDIRECT_DELAY_MIN: int = 12
DIVERSION_FLAG_RESET_STEPS: int = 8

CREW_FATIGUE_THRESHOLD_HOURS: float = 10.0
CREW_SWAP_DEPLOY_DELAY_MIN: int = 8

COMMS_FAILURE_PROBABILITY_PER_STEP: float = 0.12

MUTUAL_AID_DELAY_MIN: int = 12
MUTUAL_AID_OVER_REQUEST_PENALTY: float = -0.10

PROTOCOL_COMPLIANCE_MAX_BONUS: float = 0.15
WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY: float = -0.50
SCORE_FLOOR: float = 0.0
SCORE_CEILING: float = 1.0

DEMAND_FORECAST_HORIZON_HOURS: int = 12
DEMAND_FORECAST_NOISE_PCT: float = 0.20

@unique
class SeverityLevel(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P0 = "P0"

    @property
    def target_response_minutes(self) -> int:
        return {
            SeverityLevel.P1: 8,
            SeverityLevel.P2: 30,
            SeverityLevel.P3: 120,
            SeverityLevel.P0: 999,
        }[self]

    @property
    def start_tag(self) -> "TriageTag":
        return {
            SeverityLevel.P1: TriageTag.IMMEDIATE,
            SeverityLevel.P2: TriageTag.DELAYED,
            SeverityLevel.P3: TriageTag.MINIMAL,
            SeverityLevel.P0: TriageTag.EXPECTANT,
        }[self]

    @property
    def colour(self) -> str:
        return {
            SeverityLevel.P1: "RED",
            SeverityLevel.P2: "YELLOW",
            SeverityLevel.P3: "GREEN",
            SeverityLevel.P0: "BLACK",
        }[self]

@unique
class TriageTag(str, Enum):
    IMMEDIATE  = "Immediate"
    DELAYED    = "Delayed"
    MINIMAL    = "Minimal"
    EXPECTANT  = "Expectant"

    @property
    def wrong_tag_as_immediate_penalty(self) -> float:
        return WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY if self == TriageTag.EXPECTANT else 0.0

    @classmethod
    def from_rpm(cls, respirations: str, pulse: str, mental_status: str) -> "TriageTag":
        r_ok = respirations not in ("absent",)
        p_ok = pulse not in ("absent",)
        m_ok = mental_status in ("alert",)

        if not r_ok:
            return cls.EXPECTANT
        if not p_ok:
            return cls.IMMEDIATE
        if respirations in ("present_rapid", "present_abnormal") or not m_ok:
            return cls.IMMEDIATE
        if not m_ok:
            return cls.DELAYED
        return cls.MINIMAL

@unique
class UnitType(str, Enum):
    BLS  = "BLS"
    ALS  = "ALS"
    MICU = "MICU"

    @property
    def crew_size(self) -> int:
        return {UnitType.BLS: 2, UnitType.ALS: 3, UnitType.MICU: 4}[self]

    @property
    def dispatch_cost(self) -> int:
        return {UnitType.BLS: 1, UnitType.ALS: 2, UnitType.MICU: 4}[self]

    @property
    def capabilities(self) -> FrozenSet[str]:
        base = frozenset({"cpr", "oxygen", "splinting", "transport", "aed"})
        als_extra = frozenset({
            "iv_access", "intubation", "cardiac_monitoring",
            "drug_administration", "12_lead_ecg",
        })
        micu_extra = frozenset({
            "ventilator", "ecmo_prep", "advanced_hemodynamics",
            "iabp", "rsi", "thoracostomy", "transcardiac_pacing",
        })
        if self == UnitType.BLS:
            return base
        if self == UnitType.ALS:
            return base | als_extra
        return base | als_extra | micu_extra

    @property
    def can_use_unpaved_road(self) -> bool:
        return self == UnitType.BLS

    @property
    def max_speed_urban_kmh(self) -> int:
        return {UnitType.BLS: 70, UnitType.ALS: 75, UnitType.MICU: 80}[self]

    @property
    def max_speed_highway_kmh(self) -> int:
        return {UnitType.BLS: 100, UnitType.ALS: 110, UnitType.MICU: 120}[self]

    def can_handle_condition(self, required_capability: str) -> bool:
        return required_capability in self.capabilities

@unique
class UnitStatus(str, Enum):
    AVAILABLE         = "available"
    DISPATCHED        = "dispatched"
    ON_SCENE          = "on_scene"
    TRANSPORTING      = "transporting"
    AT_HOSPITAL       = "at_hospital"
    RETURNING         = "returning"
    OUT_OF_SERVICE    = "out_of_service"
    MUTUAL_AID_DEPLOY = "mutual_aid_deploy"
    CREW_SWAP_PENDING = "crew_swap_pending"
    COMMS_LOST        = "comms_lost"

@unique
class ActionType(str, Enum):
    DISPATCH          = "dispatch"
    REROUTE           = "reroute"
    ESCALATE          = "escalate"
    TAG               = "tag"
    TRANSFER          = "transfer"
    REQUEST_MUTUAL_AID= "request_mutual_aid"
    PREPOSITION       = "preposition"
    CREW_SWAP         = "crew_swap"
    DECLARE_SURGE     = "declare_surge"
    HOSPITAL_BYPASS   = "hospital_bypass"
    NOOP              = "noop"

@unique
class IncidentType(str, Enum):
    STEMI              = "STEMI"
    STROKE             = "Stroke"
    POLYTRAUMA         = "Polytrauma"
    RESPIRATORY        = "Respiratory_Failure"
    BURNS              = "Burns"
    OBSTETRIC          = "Obstetric_Emergency"
    SEPSIS             = "Sepsis"
    ANAPHYLAXIS        = "Anaphylaxis"
    DROWNING           = "Drowning"
    TOXICOLOGY         = "Toxicology"
    PAEDIATRIC         = "Paediatric_Emergency"
    TRAUMA_MINOR       = "Trauma_Minor"
    TRAUMA_MODERATE    = "Trauma_Moderate"
    TRAUMA_MAJOR       = "Trauma_Major"
    CARDIAC_ARREST     = "Cardiac_Arrest"
    PSYCHIATRIC        = "Psychiatric_Emergency"
    HYPOGLYCAEMIA      = "Hypoglycaemia"
    SPINAL             = "Spinal_Trauma"
    HEAT_STROKE        = "Heat_Stroke"
    MCI_RTA            = "MCI_RTA"
    MCI_INDUSTRIAL     = "MCI_Industrial"
    MCI_NATURAL        = "MCI_Natural_Disaster"
    SNAKEBITE          = "Snake_Bite"
    RENAL              = "Renal_Emergency"
    ABDOMINAL          = "Abdominal_Emergency"
    AORTIC             = "Aortic_Emergency"
    ARRHYTHMIA         = "Cardiac_Arrhythmia"
    CRUSH              = "Crush_Injury"

@unique
class DecayModel(str, Enum):
    EXPONENTIAL        = "exponential"
    LINEAR             = "linear"
    SIGMOID            = "sigmoid"
    STEP_CLIFF         = "step_cliff"
    BIMODAL            = "bimodal"
    PLATEAU_THEN_DROP  = "plateau_then_drop"
    RAPID_INITIAL      = "rapid_initial"
    TREATMENT_RESPONSIVE = "treatment_responsive"

@unique
class HospitalTier(str, Enum):
    LEVEL_1_TRAUMA = "level_1_trauma"
    LEVEL_2_TRAUMA = "level_2_trauma"
    LEVEL_3_BASIC  = "level_3_basic"

@unique
class HospitalType(str, Enum):
    GOVERNMENT_TERTIARY  = "government_tertiary"
    GOVERNMENT_SECONDARY = "government_secondary"
    GOVERNMENT_DISTRICT  = "government_district"
    MUNICIPAL_TERTIARY   = "municipal_tertiary"
    PRIVATE_TERTIARY     = "private_tertiary"
    PRIVATE_SECONDARY    = "private_secondary"

@unique
class ZoneType(str, Enum):
    METRO_CORE         = "metro_core"
    METRO_SUBURBAN     = "metro_suburban"
    METRO_SATELLITE    = "metro_satellite"
    METRO_PERI_URBAN   = "metro_peri_urban"
    METRO_TIER2        = "metro_tier2"
    TIER2_CITY         = "tier2_city"
    TIER3_CITY         = "tier3_city"
    SEMI_URBAN         = "semi_urban"
    SEMI_URBAN_COASTAL = "semi_urban_coastal"
    SEMI_RURAL         = "semi_rural"
    RURAL              = "rural"
    COASTAL_RURAL      = "coastal_rural"
    COASTAL_SEMI_URBAN = "coastal_semi_urban"
    TRIBAL_RURAL       = "tribal_rural"
    TRIBAL_FOREST      = "tribal_forest"
    INDUSTRIAL_TOWN    = "industrial_town"

@unique
class Season(str, Enum):
    SUMMER       = "summer"
    MONSOON      = "monsoon"
    POST_MONSOON = "post_monsoon"
    WINTER       = "winter"

@unique
class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"

@unique
class TaskID(str, Enum):
    T1  = "task1_single_triage"
    T2  = "task2_hospital_route"
    T3  = "task3_unit_type"
    T4  = "task4_multi_incident"
    T5  = "task5_dynamic_rerouting"
    T6  = "task6_prepositioning"
    T7  = "task7_mci_start"
    T8  = "task8_transfer_cascade"
    T9  = "task9_surge"

    @property
    def difficulty(self) -> TaskDifficulty:
        easy   = {TaskID.T1, TaskID.T2, TaskID.T3}
        medium = {TaskID.T4, TaskID.T5, TaskID.T6}
        if self in easy:
            return TaskDifficulty.EASY
        if self in medium:
            return TaskDifficulty.MEDIUM
        return TaskDifficulty.HARD

    @property
    def baseline_score(self) -> float:
        return {
            TaskID.T1: 0.61,
            TaskID.T2: 0.72,
            TaskID.T3: 0.68,
            TaskID.T4: 0.44,
            TaskID.T5: 0.38,
            TaskID.T6: 0.42,
            TaskID.T7: 0.29,
            TaskID.T8: 0.24,
            TaskID.T9: 0.17,
        }[self]

    @property
    def max_incident_queue_size(self) -> int:
        return {
            TaskID.T1: 1,
            TaskID.T2: 1,
            TaskID.T3: 1,
            TaskID.T4: 8,
            TaskID.T5: 5,
            TaskID.T6: 0,
            TaskID.T7: 40,
            TaskID.T8: 12,
            TaskID.T9: 45,
        }[self]

    @property
    def comms_failure_active(self) -> bool:
        return self in {TaskID.T7, TaskID.T8, TaskID.T9}

    @property
    def mutual_aid_required(self) -> bool:
        return self in {TaskID.T9}

    @property
    def seed(self) -> int:
        return {
            TaskID.T1: 42,
            TaskID.T2: 43,
            TaskID.T3: 44,
            TaskID.T4: 45,
            TaskID.T5: 46,
            TaskID.T6: 47,
            TaskID.T7: 48,
            TaskID.T8: 49,
            TaskID.T9: 50,
        }[self]

@unique
class AgencyType(str, Enum):
    EMS    = "EMS"
    POLICE = "Police"
    FIRE   = "Fire"
    NDRF   = "NDRF"

@unique
class ProtocolRule(str, Enum):
    MICU_FOR_STEMI               = "micu_for_stemi"
    ALS_FOR_STROKE               = "als_for_stroke"
    LEVEL1_TRAUMA_WITHIN_30_MIN  = "nearest_level1_trauma_within_30_min"
    NO_ROUTING_TO_DIVERTED       = "no_routing_to_diverted_hospital"
    MULTI_AGENCY_TRAPPED         = "multi_agency_for_trapped_victim"
    CATH_LAB_ACTIVATION_STEMI    = "cath_lab_activation_for_stemi"
    STROKE_UNIT_FOR_STROKE       = "stroke_unit_for_stroke"
    BURNS_UNIT_FOR_MAJOR_BURNS   = "burns_unit_for_major_burns"
    PAEDIATRIC_ED_FOR_PAEDS      = "paediatric_ed_for_paediatric_emergency"
    NOOP_UNKNOWN_UNIT_PENALTY    = "noop_on_unknown_unit_penalty"
    START_TRIAGE_IN_MCI          = "start_triage_in_mci"
    MUTUAL_AID_IN_SURGE          = "mutual_aid_in_surge"

    @property
    def bonus_per_correct(self) -> float:
        bonuses = {
            ProtocolRule.MICU_FOR_STEMI:              0.015,
            ProtocolRule.ALS_FOR_STROKE:              0.012,
            ProtocolRule.LEVEL1_TRAUMA_WITHIN_30_MIN: 0.015,
            ProtocolRule.NO_ROUTING_TO_DIVERTED:      0.010,
            ProtocolRule.MULTI_AGENCY_TRAPPED:        0.015,
            ProtocolRule.CATH_LAB_ACTIVATION_STEMI:   0.018,
            ProtocolRule.STROKE_UNIT_FOR_STROKE:      0.015,
            ProtocolRule.BURNS_UNIT_FOR_MAJOR_BURNS:  0.012,
            ProtocolRule.PAEDIATRIC_ED_FOR_PAEDS:     0.012,
            ProtocolRule.NOOP_UNKNOWN_UNIT_PENALTY:   0.008,
            ProtocolRule.START_TRIAGE_IN_MCI:         0.020,
            ProtocolRule.MUTUAL_AID_IN_SURGE:         0.020,
        }
        return bonuses[self]

    @property
    def penalty_per_violation(self) -> float:
        penalties = {
            ProtocolRule.MICU_FOR_STEMI:              -0.020,
            ProtocolRule.ALS_FOR_STROKE:              -0.015,
            ProtocolRule.LEVEL1_TRAUMA_WITHIN_30_MIN: -0.025,
            ProtocolRule.NO_ROUTING_TO_DIVERTED:      -0.030,
            ProtocolRule.MULTI_AGENCY_TRAPPED:        -0.025,
            ProtocolRule.CATH_LAB_ACTIVATION_STEMI:   -0.022,
            ProtocolRule.STROKE_UNIT_FOR_STROKE:      -0.028,
            ProtocolRule.BURNS_UNIT_FOR_MAJOR_BURNS:  -0.020,
            ProtocolRule.PAEDIATRIC_ED_FOR_PAEDS:     -0.018,
            ProtocolRule.NOOP_UNKNOWN_UNIT_PENALTY:   -0.015,
            ProtocolRule.START_TRIAGE_IN_MCI:         -0.035,
            ProtocolRule.MUTUAL_AID_IN_SURGE:         -0.050,
        }
        return penalties[self]

ZoneID          = str
HospitalID      = str
UnitID          = str
IncidentID      = str
TemplateID      = str
StepNumber      = int
Score           = float
Minutes         = float
Probability     = float
TrafficMatrix   = Dict[ZoneID, Dict[ZoneID, Minutes]]
DemandHeatmap   = Dict[ZoneID, Dict[str, float]]

class HospitalSpecialty:
    TRAUMA_CENTRE           = "trauma_centre"
    CARDIAC_CATH_LAB        = "cardiac_cath_lab"
    STROKE_UNIT             = "stroke_unit"
    NEUROSURGERY            = "neurosurgery"
    CARDIOTHORACIC_SURGERY  = "cardiothoracic_surgery"
    BURNS_UNIT              = "burns_unit"
    TOXICOLOGY              = "toxicology"
    OBSTETRICS              = "obstetrics"
    PAEDIATRIC_EMERGENCY    = "paediatric_emergency"
    PLASTIC_SURGERY         = "plastic_surgery"
    RENAL_DIALYSIS          = "renal_dialysis"
    TRANSPLANT_CENTRE       = "transplant_centre"
    HYPERBARIC_CHAMBER      = "hyperbaric_chamber"
    LEVEL_1_TRAUMA          = "level_1_trauma"

    ALL: ClassVar[FrozenSet[str]] = frozenset({
        TRAUMA_CENTRE, CARDIAC_CATH_LAB, STROKE_UNIT, NEUROSURGERY,
        CARDIOTHORACIC_SURGERY, BURNS_UNIT, TOXICOLOGY, OBSTETRICS,
        PAEDIATRIC_EMERGENCY, PLASTIC_SURGERY, RENAL_DIALYSIS,
        TRANSPLANT_CENTRE, HYPERBARIC_CHAMBER, LEVEL_1_TRAUMA,
    })

class HospitalInfrastructure:
    HELIPAD                = "helipad"
    BLOOD_BANK             = "blood_bank"
    LAB_24H                = "24h_lab"
    RADIOLOGY_24H          = "24h_radiology"
    CT_SCANNER             = "ct_scanner"
    MRI                    = "mri"
    ANGIOGRAPHY_SUITE      = "angiography_suite"
    ECMO_CAPABILITY        = "ecmo_capability"
    THROMBOLYSIS_CAPABILITY= "thrombolysis_capability"

CONDITION_REQUIRED_UNIT: Dict[str, UnitType] = {
    "stemi_anterior":        UnitType.MICU,
    "stemi_inferior":        UnitType.MICU,
    "stemi_with_vf_arrest":  UnitType.MICU,
    "stemi_cocaine":         UnitType.MICU,
    "stemi_post_cabg":       UnitType.MICU,
    "ischemic_stroke":           UnitType.ALS,
    "ischemic_stroke_wake_up":   UnitType.ALS,
    "hemorrhagic_stroke_sah":    UnitType.ALS,
    "paediatric_stroke":         UnitType.ALS,
    "polytrauma_blunt":          UnitType.MICU,
    "polytrauma_penetrating":    UnitType.ALS,
    "severe_tbi":                UnitType.MICU,
    "crush_syndrome":            UnitType.MICU,
    "chest_trauma":              UnitType.ALS,
    "splenic_laceration":        UnitType.ALS,
    "respiratory_failure":       UnitType.ALS,
    "paediatric_respiratory":    UnitType.ALS,
    "ards":                      UnitType.ALS,
    "pulmonary_embolism":        UnitType.ALS,
    "massive_haemoptysis":       UnitType.MICU,
    "acute_pulmonary_oedema":    UnitType.MICU,
    "decompression_sickness":    UnitType.ALS,
    "burns_major":               UnitType.ALS,
    "burns_moderate":            UnitType.ALS,
    "burns_electrical":          UnitType.ALS,
    "chemical_burns":            UnitType.ALS,
    "septic_shock":              UnitType.ALS,
    "meningococcal_sepsis":      UnitType.MICU,
    "necrotising_fasciitis":     UnitType.ALS,
    "cardiac_arrest_vf":         UnitType.MICU,
    "cardiac_arrest_pea":        UnitType.MICU,
    "neonatal_resuscitation":    UnitType.ALS,
    "hyperkalaemia":             UnitType.MICU,
    "lightning_strike":          UnitType.ALS,
    "hypoglycaemia":             UnitType.ALS,
    "dka":                       UnitType.ALS,
    "heat_stroke":               UnitType.ALS,
    "heat_exhaustion":           UnitType.BLS,
    "submersion":                UnitType.ALS,
    "near_drowning":             UnitType.BLS,
    "organophosphate_poisoning": UnitType.ALS,
    "opioid_overdose":           UnitType.ALS,
    "toxic_inhalation":          UnitType.ALS,
    "carbon_monoxide":           UnitType.ALS,
    "botulism_outbreak":         UnitType.ALS,
    "phosphine_poisoning":       UnitType.MICU,
    "eclampsia":                 UnitType.MICU,
    "obstetric_hemorrhage":      UnitType.ALS,
    "uterine_rupture":           UnitType.ALS,
    "normal_delivery":           UnitType.BLS,
    "aaa_rupture":               UnitType.MICU,
    "aortic_dissection":         UnitType.MICU,
    "pancreatitis_severe":       UnitType.ALS,
    "perforated_viscus":         UnitType.ALS,
    "mci_rta":                   UnitType.MICU,
    "blast_injury":              UnitType.MICU,
    "mci_natural_disaster":      UnitType.MICU,
    "minor_trauma":              UnitType.BLS,
    "minor_head_trauma":         UnitType.BLS,
    "psychiatric_crisis":        UnitType.ALS,
    "expected_death":            UnitType.BLS,
    "snakebite_hemotoxic":       UnitType.ALS,
    "snakebite_neurotoxic":      UnitType.ALS,
    "rhabdomyolysis":            UnitType.ALS,
    "urosepsis":                 UnitType.ALS,
    "complete_heart_block":      UnitType.MICU,
    "wpw_svt":                   UnitType.MICU,
}

CONDITION_REQUIRED_SPECIALTY: Dict[str, str] = {
    "stemi_anterior":            HospitalSpecialty.CARDIAC_CATH_LAB,
    "stemi_inferior":            HospitalSpecialty.CARDIAC_CATH_LAB,
    "stemi_with_vf_arrest":      HospitalSpecialty.CARDIAC_CATH_LAB,
    "stemi_cocaine":             HospitalSpecialty.CARDIAC_CATH_LAB,
    "stemi_post_cabg":           HospitalSpecialty.CARDIAC_CATH_LAB,
    "ischemic_stroke":           HospitalSpecialty.STROKE_UNIT,
    "ischemic_stroke_wake_up":   HospitalSpecialty.STROKE_UNIT,
    "hemorrhagic_stroke_sah":    HospitalSpecialty.NEUROSURGERY,
    "polytrauma_blunt":          HospitalSpecialty.LEVEL_1_TRAUMA,
    "polytrauma_penetrating":    HospitalSpecialty.LEVEL_1_TRAUMA,
    "severe_tbi":                HospitalSpecialty.NEUROSURGERY,
    "crush_syndrome":            HospitalSpecialty.LEVEL_1_TRAUMA,
    "burns_major":               HospitalSpecialty.BURNS_UNIT,
    "burns_moderate":            HospitalSpecialty.BURNS_UNIT,
    "burns_electrical":          HospitalSpecialty.BURNS_UNIT,
    "chemical_burns":            HospitalSpecialty.BURNS_UNIT,
    "meningococcal_sepsis":      HospitalSpecialty.PAEDIATRIC_EMERGENCY,
    "neonatal_resuscitation":    HospitalSpecialty.PAEDIATRIC_EMERGENCY,
    "eclampsia":                 HospitalSpecialty.OBSTETRICS,
    "obstetric_hemorrhage":      HospitalSpecialty.OBSTETRICS,
    "uterine_rupture":           HospitalSpecialty.OBSTETRICS,
    "aaa_rupture":               HospitalSpecialty.CARDIOTHORACIC_SURGERY,
    "aortic_dissection":         HospitalSpecialty.CARDIOTHORACIC_SURGERY,
    "hyperkalaemia":             HospitalSpecialty.RENAL_DIALYSIS,
    "rhabdomyolysis":            HospitalSpecialty.RENAL_DIALYSIS,
    "organophosphate_poisoning": HospitalSpecialty.TOXICOLOGY,
    "opioid_overdose":           HospitalSpecialty.TOXICOLOGY,
    "decompression_sickness":    HospitalSpecialty.HYPERBARIC_CHAMBER,
    "paediatric_respiratory":    HospitalSpecialty.PAEDIATRIC_EMERGENCY,
    "mci_rta":                   HospitalSpecialty.LEVEL_1_TRAUMA,
    "blast_injury":              HospitalSpecialty.LEVEL_1_TRAUMA,
    "mci_natural_disaster":      HospitalSpecialty.LEVEL_1_TRAUMA,
}

class RPMScore:
    RESPIRATIONS_ABSENT        = "absent"
    RESPIRATIONS_ABNORMAL      = "present_abnormal"
    RESPIRATIONS_RAPID         = "present_rapid"
    RESPIRATIONS_NORMAL        = "normal"

    PULSE_ABSENT               = "absent"
    PULSE_WEAK                 = "present_weak"
    PULSE_RAPID                = "present_rapid"
    PULSE_SLOW                 = "present_slow"
    PULSE_IRREGULAR            = "present_irregular"
    PULSE_NORMAL               = "present_normal"
    PULSE_STRONG               = "present_strong"

    MENTAL_UNRESPONSIVE        = "unresponsive"
    MENTAL_ALTERED             = "altered"
    MENTAL_ALERT               = "alert"

    MODIFIERS: Dict[str, float] = {
        "respirations_absent":   -0.35,
        "respirations_abnormal": -0.15,
        "respirations_normal":    0.00,
        "pulse_absent":          -0.45,
        "pulse_weak":            -0.20,
        "pulse_rapid":           -0.12,
        "pulse_slow":            -0.15,
        "pulse_normal":           0.00,
        "mental_unresponsive":   -0.28,
        "mental_altered":        -0.12,
        "mental_alert":           0.00,
    }

    @classmethod
    def compute_modifier(
        cls,
        respirations: str,
        pulse: str,
        mental_status: str,
    ) -> float:
        r_key = f"respirations_{respirations.replace('present_', '').split('_')[0]}"
        p_key = f"pulse_{pulse.replace('present_', '')}"
        m_key = f"mental_{mental_status.replace('present_', '')}"

        r_mod = cls.MODIFIERS.get(f"respirations_{respirations}", 0.0)
        p_mod = cls.MODIFIERS.get(f"pulse_{pulse}", 0.0)
        m_mod = cls.MODIFIERS.get(f"mental_{mental_status}", 0.0)

        return max(-1.0, r_mod + p_mod + m_mod)

class SurvivalCurve:

    @staticmethod
    def exponential(
        t: float,
        survival_at_zero: float,
        survival_floor: float,
        decay_rate: float,
    ) -> float:
        p = survival_floor + (survival_at_zero - survival_floor) * math.exp(-decay_rate * t)
        return float(max(survival_floor, min(survival_at_zero, p)))

    @staticmethod
    def linear(
        t: float,
        survival_at_zero: float,
        survival_floor: float,
        linear_rate: float,
    ) -> float:
        p = survival_at_zero - linear_rate * t
        return float(max(survival_floor, min(survival_at_zero, p)))

    @staticmethod
    def sigmoid(
        t: float,
        survival_at_zero: float,
        survival_floor: float,
        sigmoid_k: float,
        sigmoid_midpoint: float,
    ) -> float:
        p = survival_floor + (survival_at_zero - survival_floor) / (
            1.0 + math.exp(sigmoid_k * (t - sigmoid_midpoint))
        )
        return float(max(survival_floor, min(survival_at_zero, p)))

    @staticmethod
    def step_cliff(
        t: float,
        survival_at_zero: float,
        p_before_cliff: float,
        p_after_cliff: float,
        cliff_time_minutes: float,
        cliff_width_minutes: float,
    ) -> float:
        if t < cliff_time_minutes:
            return float(p_before_cliff)
        if t > cliff_time_minutes + cliff_width_minutes:
            return float(p_after_cliff)
        alpha = (t - cliff_time_minutes) / cliff_width_minutes
        return float(p_before_cliff + alpha * (p_after_cliff - p_before_cliff))

    @staticmethod
    def plateau_then_drop(
        t: float,
        survival_at_zero: float,
        plateau_end_minutes: float,
        survival_floor: float,
        decay_rate: float,
    ) -> float:
        if t <= plateau_end_minutes:
            return float(survival_at_zero)
        return SurvivalCurve.exponential(
            t - plateau_end_minutes, survival_at_zero, survival_floor, decay_rate
        )

    @staticmethod
    def rapid_initial(
        t: float,
        survival_at_zero: float,
        survival_floor: float,
        rapid_decay_rate: float,
        rapid_phase_end: float,
        slow_decay_rate: float,
    ) -> float:
        if t <= rapid_phase_end:
            p = survival_floor + (survival_at_zero - survival_floor) * math.exp(-rapid_decay_rate * t)
        else:
            p_at_rapid_end = survival_floor + (survival_at_zero - survival_floor) * math.exp(
                -rapid_decay_rate * rapid_phase_end
            )
            p = survival_floor + (p_at_rapid_end - survival_floor) * math.exp(
                -slow_decay_rate * (t - rapid_phase_end)
            )
        return float(max(survival_floor, min(survival_at_zero, p)))

    @classmethod
    def compute(cls, model: DecayModel, t: float, params: Dict[str, Any]) -> float:
        if model == DecayModel.EXPONENTIAL:
            return cls.exponential(
                t,
                params["survival_at_zero_min"],
                params["survival_floor"],
                params["decay_rate"],
            )
        if model == DecayModel.LINEAR:
            return cls.linear(
                t,
                params["survival_at_zero_min"],
                params["survival_floor"],
                params["linear_rate"],
            )
        if model == DecayModel.SIGMOID:
            return cls.sigmoid(
                t,
                params["survival_at_zero_min"],
                params["survival_floor"],
                params["sigmoid_k"],
                params["sigmoid_midpoint"],
            )
        if model == DecayModel.STEP_CLIFF:
            return cls.step_cliff(
                t,
                params["survival_at_zero_min"],
                params["p_before_cliff"],
                params["p_after_cliff"],
                params["cliff_time_minutes"],
                params["cliff_width_minutes"],
            )
        if model == DecayModel.PLATEAU_THEN_DROP:
            return cls.plateau_then_drop(
                t,
                params["survival_at_zero_min"],
                params["plateau_end_minutes"],
                params["survival_floor"],
                params["decay_rate"],
            )
        if model == DecayModel.RAPID_INITIAL:
            return cls.rapid_initial(
                t,
                params["survival_at_zero_min"],
                params["survival_floor"],
                params["rapid_decay_rate"],
                params["rapid_phase_end"],
                params["slow_decay_rate"],
            )
        return cls.exponential(
            t,
            params.get("survival_at_zero_min", 0.9),
            params.get("survival_floor", 0.3),
            params.get("decay_rate", 0.02),
        )

class _ModelRegistry:

    _registry: Dict[str, Type[BaseModel]] = {}
    _locked: bool = False

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        if cls._locked:
            raise RuntimeError(
                f"ModelRegistry is locked. Cannot register '{name}' after lock."
            )
        if name in cls._registry:
            logger.warning("ModelRegistry: overwriting existing registration for '%s'", name)
        cls._registry[name] = model_class
        logger.debug("ModelRegistry: registered '%s'", name)

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        if name not in cls._registry:
            raise KeyError(
                f"ModelRegistry: '{name}' not found. "
                f"Registered models: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def all_names(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def lock(cls) -> None:
        cls._locked = True
        logger.info("ModelRegistry locked with %d models.", len(cls._registry))

    @classmethod
    def unlock(cls) -> None:
        cls._locked = False

    @classmethod
    def schema_version(cls) -> int:
        return __schema_version__

ModelRegistry = _ModelRegistry()

class EmergiBaseModel(BaseModel):

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        populate_by_name=True,
        extra="forbid",
        frozen=False,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmergiBaseModel":
        return cls.model_validate(data)

    @classmethod
    def schema_version(cls) -> int:
        return __schema_version__

class ImmutableEmergiModel(EmergiBaseModel):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

class GeoCoordinate(ImmutableEmergiModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (WGS84)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (WGS84)")

    def distance_km(self, other: "GeoCoordinate") -> float:
        R = 6371.0
        phi1, phi2 = math.radians(self.lat), math.radians(other.lat)
        dphi = math.radians(other.lat - self.lat)
        dlambda = math.radians(other.lon - self.lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class CapacitySnapshot(EmergiBaseModel):
    total_beds: int             = Field(..., ge=0)
    er_beds: int                = Field(..., ge=0)
    icu_beds: int               = Field(..., ge=0)
    ccu_beds: int               = Field(..., ge=0)
    trauma_bays: int            = Field(..., ge=0)
    ventilators: int            = Field(..., ge=0)
    er_occupancy_pct: float     = Field(..., ge=0.0, le=1.0)
    icu_occupancy_pct: float    = Field(..., ge=0.0, le=1.0)
    total_occupancy_pct: float  = Field(..., ge=0.0, le=1.0)
    on_diversion: bool          = Field(default=False)
    diversion_reason: Optional[str] = Field(default=None)

    @property
    def er_available_beds(self) -> int:
        return max(0, int(self.er_beds * (1.0 - self.er_occupancy_pct)))

    @property
    def icu_available_beds(self) -> int:
        return max(0, int(self.icu_beds * (1.0 - self.icu_occupancy_pct)))

    @property
    def is_near_capacity(self) -> bool:
        return self.er_occupancy_pct >= DIVERSION_THRESHOLD_PCT

    @property
    def surge_capacity_available(self) -> bool:
        return self.total_occupancy_pct < 0.95

class PatientVitals(ImmutableEmergiModel):
    respirations: str = Field(
        ...,
        description="Respiration status: normal | present_rapid | present_abnormal | absent",
    )
    pulse: str = Field(
        ...,
        description=(
            "Pulse status: present_strong | present_normal | present_weak | "
            "present_rapid | present_slow | present_irregular | absent"
        ),
    )
    mental_status: str = Field(
        ...,
        description="Mental status: alert | altered | unresponsive",
    )
    gcs: Optional[int] = Field(default=None, ge=3, le=15)
    spo2_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    bp_systolic: Optional[int] = Field(default=None, ge=0, le=300)
    bp_diastolic: Optional[int] = Field(default=None, ge=0, le=200)
    heart_rate: Optional[int] = Field(default=None, ge=0, le=300)
    respiratory_rate: Optional[int] = Field(default=None, ge=0, le=80)
    temperature_celsius: Optional[float] = Field(default=None, ge=28.0, le=45.0)

    @property
    def rpm_modifier(self) -> float:
        return RPMScore.compute_modifier(
            self.respirations, self.pulse, self.mental_status
        )

    @property
    def derived_triage_tag(self) -> TriageTag:
        return TriageTag.from_rpm(self.respirations, self.pulse, self.mental_status)

    @property
    def in_shock(self) -> bool:
        if self.bp_systolic is not None and self.bp_systolic < 90:
            return True
        if self.heart_rate is not None and self.heart_rate > 120:
            return True
        return self.pulse in ("present_weak",)

    @property
    def hypoxic(self) -> bool:
        if self.spo2_pct is not None:
            return self.spo2_pct < 94.0
        return self.respirations in ("present_abnormal", "present_rapid", "absent")

class CrewFatigueState(EmergiBaseModel):
    unit_id: UnitID
    hours_on_duty: float            = Field(default=0.0, ge=0.0)
    swap_requested: bool            = Field(default=False)
    swap_eta_steps: Optional[int]   = Field(default=None, ge=0)
    total_missions_this_shift: int  = Field(default=0, ge=0)

    @property
    def fatigued(self) -> bool:
        return self.hours_on_duty >= CREW_FATIGUE_THRESHOLD_HOURS

    @property
    def fatigue_penalty(self) -> float:
        if not self.fatigued:
            return 0.0
        excess = self.hours_on_duty - CREW_FATIGUE_THRESHOLD_HOURS
        return float(min(0.30, excess / 4.0 * 0.30))

    def advance_time(self, steps: int = 1) -> None:
        self.hours_on_duty += steps * TIMESTEP_MINUTES / 60.0

    def complete_mission(self) -> None:
        self.total_missions_this_shift += 1

class CommunicationsState(EmergiBaseModel):
    unit_id: UnitID
    comms_active: bool                          = Field(default=True)
    last_known_position: Optional[GeoCoordinate]= Field(default=None)
    last_known_step: Optional[int]              = Field(default=None, ge=0)
    failure_count_this_episode: int             = Field(default=0, ge=0)

    @property
    def position_stale_steps(self) -> Optional[int]:
        if self.comms_active or self.last_known_step is None:
            return None
        return None 

    @property
    def has_last_known_position(self) -> bool:
        return self.last_known_position is not None

class MutualAidRequest(EmergiBaseModel):
    request_id: str             = Field(default_factory=lambda: str(uuid.uuid4()))
    requesting_zone: ZoneID
    providing_zone: ZoneID
    unit_type: UnitType
    units_requested: int        = Field(..., ge=1, le=10)
    eta_steps: int              = Field(default=4, ge=1) 
    fulfilled: bool             = Field(default=False)
    cancelled: bool             = Field(default=False)
    step_requested: StepNumber  = Field(..., ge=0)

    @property
    def delay_minutes(self) -> float:
        return self.eta_steps * TIMESTEP_MINUTES

def clamp_score(value: float) -> Score:
    return float(max(SCORE_FLOOR, min(SCORE_CEILING, value)))

def weighted_sum_score(
    components: Dict[str, float],
    weights: Dict[str, float],
) -> Score:
    total = 0.0
    for name, weight in weights.items():
        raw = components.get(name, 0.0)
        total += weight * float(raw)
    return clamp_score(total)

def normalise_response_time(
    actual_minutes: float,
    target_minutes: float,
    worst_case_minutes: float = 120.0,
) -> Score:
    if actual_minutes <= target_minutes:
        return 1.0
    if actual_minutes >= worst_case_minutes:
        return 0.0
    return clamp_score(
        1.0 - (actual_minutes - target_minutes) / (worst_case_minutes - target_minutes)
    )

def survival_probability_delta(p_before: float, p_after: float) -> float:
    delta = p_after - p_before
    return max(-1.0, min(0.0, delta))

@lru_cache(maxsize=512)
def get_episode_id() -> str:
    return str(uuid.uuid4())

def new_episode_id() -> str:
    return str(uuid.uuid4())

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def validate_zone_id(zone_id: str) -> bool:
    if not isinstance(zone_id, str):
        return False
    if not zone_id.startswith("Z"):
        return False
    try:
        n = int(zone_id[1:])
        return 1 <= n <= NUM_ZONES
    except ValueError:
        return False

def validate_hospital_id(hospital_id: str) -> bool:
    if not isinstance(hospital_id, str):
        return False
    if not hospital_id.startswith("H"):
        return False
    try:
        n = int(hospital_id[1:])
        return 1 <= n <= NUM_HOSPITALS
    except ValueError:
        return False

def validate_unit_id(unit_id: str) -> bool:
    if not isinstance(unit_id, str):
        return False
    parts = unit_id.split("-")
    if len(parts) != 2:
        return False
    unit_type_str, num_str = parts
    if unit_type_str not in {u.value for u in UnitType}:
        return False
    try:
        int(num_str)
        return True
    except ValueError:
        return False

def score_in_range(score: float) -> bool:
    return SCORE_FLOOR <= score <= SCORE_CEILING

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "__openenv_spec_version__",
    "__schema_version__",

    "TIMESTEP_MINUTES",
    "MAX_STEPS_PER_EPISODE",
    "NUM_ZONES",
    "ACTIVE_ZONES_PER_EPISODE",
    "NUM_HOSPITALS",
    "FLEET_SIZE_DEFAULT",
    "INCIDENT_QUEUE_MIN",
    "INCIDENT_QUEUE_MAX",
    "TRAFFIC_UPDATE_EVERY_N_STEPS",
    "PEAK_MORNING_HOURS",
    "PEAK_EVENING_HOURS",
    "PEAK_HOUR_MULTIPLIER_MAX",
    "SECONDARY_SLOWDOWN_MULTIPLIER",
    "SECONDARY_SLOWDOWN_DURATION_STEPS",
    "DIVERSION_THRESHOLD_PCT",
    "DIVERSION_PENALTY",
    "DIVERSION_REDIRECT_DELAY_MIN",
    "DIVERSION_FLAG_RESET_STEPS",
    "CREW_FATIGUE_THRESHOLD_HOURS",
    "CREW_SWAP_DEPLOY_DELAY_MIN",
    "COMMS_FAILURE_PROBABILITY_PER_STEP",
    "MUTUAL_AID_DELAY_MIN",
    "MUTUAL_AID_OVER_REQUEST_PENALTY",
    "PROTOCOL_COMPLIANCE_MAX_BONUS",
    "WRONG_TAG_IMMEDIATE_AS_EXPECTANT_PENALTY",
    "SCORE_FLOOR",
    "SCORE_CEILING",
    "DEMAND_FORECAST_HORIZON_HOURS",
    "DEMAND_FORECAST_NOISE_PCT",

    "SeverityLevel",
    "TriageTag",
    "UnitType",
    "UnitStatus",
    "ActionType",
    "IncidentType",
    "DecayModel",
    "HospitalTier",
    "HospitalType",
    "ZoneType",
    "Season",
    "TaskDifficulty",
    "TaskID",
    "AgencyType",
    "ProtocolRule",

    "ZoneID",
    "HospitalID",
    "UnitID",
    "IncidentID",
    "TemplateID",
    "StepNumber",
    "Score",
    "Minutes",
    "Probability",
    "TrafficMatrix",
    "DemandHeatmap",

    "HospitalSpecialty",
    "HospitalInfrastructure",
    "RPMScore",

    "CONDITION_REQUIRED_UNIT",
    "CONDITION_REQUIRED_SPECIALTY",

    "SurvivalCurve",
    "ModelRegistry",

    "EmergiBaseModel",
    "ImmutableEmergiModel",

    "GeoCoordinate",
    "CapacitySnapshot",
    "PatientVitals",
    "CrewFatigueState",
    "CommunicationsState",
    "MutualAidRequest",

    "clamp_score",
    "weighted_sum_score",
    "normalise_response_time",
    "survival_probability_delta",
    "new_episode_id",
    "utc_now_iso",
    "validate_zone_id",
    "validate_hospital_id",
    "validate_unit_id",
    "score_in_range",
]

def _self_test() -> None:
    seeds = [t.seed for t in TaskID]
    assert len(seeds) == len(set(seeds)), "Duplicate TaskID seeds detected!"

    for t in TaskID:
        assert 0.0 <= t.baseline_score <= 1.0, f"Invalid baseline for {t}"

    assert UnitType.BLS.dispatch_cost < UnitType.ALS.dispatch_cost < UnitType.MICU.dispatch_cost

    for t_val in [0, 30, 60, 90, 180, 360]:
        p = SurvivalCurve.exponential(t_val, 0.97, 0.28, 0.055)
        assert 0.0 <= p <= 1.0, f"SurvivalCurve out of bounds at t={t_val}"

    for rule in ProtocolRule:
        assert rule.penalty_per_violation < 0.0, f"Protocol penalty must be negative: {rule}"
        assert rule.bonus_per_correct > 0.0, f"Protocol bonus must be positive: {rule}"

    assert clamp_score(1.5) == 1.0
    assert clamp_score(-0.5) == 0.0
    assert clamp_score(0.72) == 0.72

    tag = TriageTag.from_rpm("absent", "present_weak", "alert")
    assert tag == TriageTag.EXPECTANT

    tag2 = TriageTag.from_rpm("normal", "present_normal", "alert")
    assert tag2 == TriageTag.MINIMAL

    logger.debug("server.models self-test passed.")

_self_test()

logger.info(
    "EMERGI-ENV server.models v%s loaded — schema_version=%d, %d conditions mapped.",
    __version__,
    __schema_version__,
    len(CONDITION_REQUIRED_UNIT),
)