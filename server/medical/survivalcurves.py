from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
try:
    from server.medical.triage import (  
        TriageTag,
        RPMScore,
        RevisedTraumaScore,
        RespirationStatus,
        PulseStatus,
        MentalStatus,
    )
    _TRIAGE_AVAILABLE = True
except ImportError:
    _TRIAGE_AVAILABLE = False
    TriageTag = None   
    RPMScore  = None   
logger = logging.getLogger("emergi_env.medical.survival_curves")
SURVIVAL_CURVES_VERSION: int = 4
SIMULATION_STEP_DURATION_MIN: float = 3.0   
MAX_EPISODE_STEPS: int = 25
MAX_EPISODE_MINUTES: float = MAX_EPISODE_STEPS * SIMULATION_STEP_DURATION_MIN
SURVIVAL_FLOOR: float = 0.01
UNDER_TRIAGE_SURVIVAL_PENALTY: float = 0.82   
CRITICAL_UNDER_TRIAGE_PENALTY: float = 0.55   
GOLDEN_HOUR_COMPLIANCE_BONUS: float = 0.12
class CurveModelType(str, Enum):
    EXPONENTIAL        = "exponential"        
    SIGMOID_DECAY      = "sigmoid_decay"      
    BIPHASIC           = "biphasic"           
    STEPPED            = "stepped"            
    LINEAR             = "linear"             
    PLATEAU_THEN_DECAY = "plateau_then_decay" 
    FLAT               = "flat"               
class SeverityTier(str, Enum):
    P0 = "P0"   
    P1 = "P1"   
    P2 = "P2"   
    P3 = "P3"   
class TransportUrgencyClass(str, Enum):
    LOAD_AND_GO   = "load_and_go"    
    STAY_AND_PLAY = "stay_and_play"  
    TIME_CRITICAL = "time_critical"  
    ELECTIVE      = "elective"       
class GoldenWindowType(str, Enum):
    PLATINUM_TEN   = "platinum_10_min"   
    GOLDEN_HOUR    = "golden_hour"       
    SILVER_HOUR    = "silver_hour"       
    BRONZE_WINDOW  = "bronze_window"     
    EXTENDED       = "extended"          
    NONE           = "none"              
@dataclass(frozen=True)
class GoldenHourWindow:
    window_type: GoldenWindowType
    open_at_min: float
    closes_at_min: float
    optimal_end_min: float
    compliance_threshold_min: float
    description: str
    @property
    def duration_min(self) -> float:
        return self.closes_at_min - self.open_at_min
    def is_open(self, elapsed_min: float) -> bool:
        return self.open_at_min <= elapsed_min < self.closes_at_min
    def compliance_fraction_remaining(self, elapsed_min: float) -> float:
        if elapsed_min <= self.open_at_min:
            return 1.0
        if elapsed_min >= self.closes_at_min:
            return 0.0
        span = self.closes_at_min - self.open_at_min
        elapsed_in_window = elapsed_min - self.open_at_min
        return max(0.0, 1.0 - (elapsed_in_window / span))
@dataclass(frozen=True)
class SurvivalParameters:
    condition_key:       str
    display_name:        str
    icd10_codes:         Tuple[str, ...]   
    evidence_source:     str               
    curve_model:         CurveModelType
    initial_survival_prob: float           
    decay_lambda:        float = 0.0       
    sigmoid_inflection_min: float = 0.0   
    sigmoid_steepness:   float = 0.0       
    phase1_lambda:       float = 0.0
    phase2_lambda:       float = 0.0
    biphasic_transition_min: float = 0.0
    step_times_min:      Tuple[float, ...] = ()
    step_penalties:      Tuple[float, ...] = ()   
    linear_slope_per_min: float = 0.0     
    plateau_end_min:     float = 0.0
    plateau_decay_lambda: float = 0.0
    survival_floor:      float = SURVIVAL_FLOOR
    severity_tier:       SeverityTier = SeverityTier.P1
    transport_urgency:   TransportUrgencyClass = TransportUrgencyClass.LOAD_AND_GO
    golden_window:       Optional[GoldenHourWindow] = None
    correct_unit_types:  Tuple[str, ...] = ("ALS",)
    required_specialties: Tuple[str, ...] = ()
    time_to_irreversible_min: float = 60.0   
    per_patient_reward_weight: float = 1.0   
    paediatric_curve_variant: Optional[str] = None
    elderly_penalty_factor:  float = 1.0    
    def __post_init__(self) -> None:
        assert 0.0 < self.initial_survival_prob <= 1.0, (
            f"{self.condition_key}: initial_survival_prob must be (0, 1]"
        )
        assert 0.0 <= self.survival_floor < self.initial_survival_prob, (
            f"{self.condition_key}: floor must be < initial"
        )
        if self.curve_model == CurveModelType.STEPPED:
            assert len(self.step_times_min) == len(self.step_penalties), (
                f"{self.condition_key}: stepped model requires matching times/penalties"
            )
@dataclass
class PatientSurvivalState:
    patient_id:       str
    incident_id:      str
    condition_key:    str
    severity:         SeverityTier
    params:           SurvivalParameters
    age_years:        Optional[int]
    step_registered:  int
    time_registered_min: float
    current_step:     int          = field(default=0)
    elapsed_minutes:  float        = field(default=0.0)
    current_survival_prob: float   = field(default=1.0)
    survival_integral: float       = field(default=0.0)   
    dispatched_unit:  Optional[str] = field(default=None)
    routed_hospital:  Optional[str] = field(default=None)
    treated:          bool          = field(default=False)
    treatment_time_min: Optional[float] = field(default=None)
    deteriorated:     bool          = field(default=False)
    outcome:          Optional[str] = field(default=None)  
    golden_window_compliance: Optional[bool] = field(default=None)
    unit_type_correct: Optional[bool] = field(default=None)
    survival_history: List[Tuple[float, float]] = field(default_factory=list)  
    step_history:     List[Tuple[int, float, float]] = field(default_factory=list)
    def __post_init__(self) -> None:
        self.current_survival_prob = self.params.initial_survival_prob
        self.survival_history.append((0.0, self.current_survival_prob))
    @property
    def is_critical(self) -> bool:
        return self.severity == SeverityTier.P1
    @property
    def total_reward_contribution(self) -> float:
        ideal = self.params.initial_survival_prob * self.elapsed_minutes
        if ideal <= 0:
            return self.current_survival_prob
        return min(1.0, self.survival_integral / ideal)
    @property
    def is_in_golden_window(self) -> bool:
        gw = self.params.golden_window
        if gw is None:
            return False
        return gw.is_open(self.elapsed_minutes)
    @property
    def age_adjusted_lambda(self) -> float:
        base = self.params.decay_lambda
        if self.age_years is not None and self.age_years >= 65:
            return base * self.params.elderly_penalty_factor
        return base
@dataclass(frozen=True)
class SurvivalSnapshot:
    patient_id:          str
    condition_key:       str
    elapsed_min:         float
    survival_probability: float
    survival_integral:   float
    golden_window_open:  bool
    golden_window_compliance: float   
    time_to_irreversible: float       
    reward_score:        float        
    clinical_notes:      str
class SurvivalProbabilityCalculator:
    @staticmethod
    def compute(
        params: SurvivalParameters,
        elapsed_min: float,
        age_years: Optional[int] = None,
        dispatch_delay_min: float = 0.0,
        under_triage_penalty: float = 1.0,
    ) -> float:
        t = elapsed_min + dispatch_delay_min
        lambda_adj = params.decay_lambda
        if age_years is not None and age_years >= 65:
            lambda_adj *= params.elderly_penalty_factor
        model = params.curve_model
        P0 = params.initial_survival_prob
        floor = params.survival_floor
        if model == CurveModelType.EXPONENTIAL:
            prob = P0 * math.exp(-lambda_adj * t)
        elif model == CurveModelType.SIGMOID_DECAY:
            k = params.sigmoid_steepness
            t50 = params.sigmoid_inflection_min
            prob = floor + (P0 - floor) / (1.0 + math.exp(k * (t - t50)))
        elif model == CurveModelType.BIPHASIC:
            t_trans = params.biphasic_transition_min
            if t <= t_trans:
                prob = P0 * math.exp(-params.phase1_lambda * t)
            else:
                P_trans = P0 * math.exp(-params.phase1_lambda * t_trans)
                prob = P_trans * math.exp(-params.phase2_lambda * (t - t_trans))
        elif model == CurveModelType.STEPPED:
            prob = P0
            for step_t, penalty in zip(params.step_times_min, params.step_penalties):
                if t >= step_t:
                    prob *= penalty
        elif model == CurveModelType.LINEAR:
            prob = P0 - params.linear_slope_per_min * t
        elif model == CurveModelType.PLATEAU_THEN_DECAY:
            if t <= params.plateau_end_min:
                prob = P0
            else:
                prob = P0 * math.exp(
                    -params.plateau_decay_lambda * (t - params.plateau_end_min)
                )
        elif model == CurveModelType.FLAT:
            prob = P0
        else:
            prob = P0
        prob = max(floor, min(P0, prob))
        prob *= under_triage_penalty
        prob = max(floor, prob)
        return prob
    @staticmethod
    def integrate(
        params: SurvivalParameters,
        from_min: float,
        to_min: float,
        age_years: Optional[int] = None,
        steps: int = 60,
    ) -> float:
        if to_min <= from_min:
            return 0.0
        h = (to_min - from_min) / steps
        total = 0.0
        for i in range(steps + 1):
            t = from_min + i * h
            p = SurvivalProbabilityCalculator.compute(params, t, age_years)
            if i == 0 or i == steps:
                total += p
            elif i % 2 == 1:
                total += 4 * p
            else:
                total += 2 * p
        return total * h / 3.0
    @staticmethod
    def compute_per_minute_loss(
        params: SurvivalParameters,
        at_min: float,
        age_years: Optional[int] = None,
    ) -> float:
        delta = 0.5
        p_before = SurvivalProbabilityCalculator.compute(params, max(0, at_min - delta), age_years)
        p_after  = SurvivalProbabilityCalculator.compute(params, at_min + delta, age_years)
        return (p_before - p_after) / (2 * delta)   
    @staticmethod
    def minutes_until_threshold(
        params: SurvivalParameters,
        threshold: float,
        age_years: Optional[int] = None,
        max_search_min: float = 300.0,
        resolution_min: float = 0.5,
    ) -> Optional[float]:
        p0 = SurvivalProbabilityCalculator.compute(params, 0.0, age_years)
        if p0 < threshold:
            return 0.0
        t = 0.0
        while t < max_search_min:
            p = SurvivalProbabilityCalculator.compute(params, t, age_years)
            if p < threshold:
                return t
            t += resolution_min
        return None
class GoldenHourWindows:
    CARDIAC_ARREST = GoldenHourWindow(
        window_type=GoldenWindowType.PLATINUM_TEN,
        open_at_min=0.0, closes_at_min=10.0, optimal_end_min=4.0,
        compliance_threshold_min=6.0,
        description="VF cardiac arrest: CPR within 4 min, shock within 6 min"
    )
    STEMI = GoldenHourWindow(
        window_type=GoldenWindowType.SILVER_HOUR,
        open_at_min=0.0, closes_at_min=90.0, optimal_end_min=60.0,
        compliance_threshold_min=90.0,
        description="STEMI door-to-balloon ≤90 min target (ESC 2023)"
    )
    STROKE_THROMBOLYSIS = GoldenHourWindow(
        window_type=GoldenWindowType.BRONZE_WINDOW,
        open_at_min=0.0, closes_at_min=270.0, optimal_end_min=60.0,
        compliance_threshold_min=270.0,
        description="Ischemic stroke tPA window: 4.5h from onset (IST-3)"
    )
    MAJOR_TRAUMA = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=60.0, optimal_end_min=20.0,
        compliance_threshold_min=60.0,
        description="Polytrauma golden hour — definitive care within 60 min"
    )
    ANAPHYLAXIS = GoldenHourWindow(
        window_type=GoldenWindowType.PLATINUM_TEN,
        open_at_min=0.0, closes_at_min=30.0, optimal_end_min=5.0,
        compliance_threshold_min=10.0,
        description="Anaphylaxis: adrenaline within minutes, airway rapidly closes"
    )
    ECLAMPSIA = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=30.0, optimal_end_min=10.0,
        compliance_threshold_min=20.0,
        description="Eclampsia: MgSO4 loading within 20 min"
    )
    PAEDIATRIC_CRITICAL = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=60.0, optimal_end_min=15.0,
        compliance_threshold_min=30.0,
        description="Paediatric critical: definitive care within 30 min optimal"
    )
    BURNS = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=120.0, optimal_end_min=60.0,
        compliance_threshold_min=120.0,
        description="Major burns: Parkland fluids + airway within 2h"
    )
    SEPSIS_HOUR_1 = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=60.0, optimal_end_min=30.0,
        compliance_threshold_min=60.0,
        description="Sepsis Hour-1 Bundle: antibiotics within 60 min"
    )
    AAA_RUPTURE = GoldenHourWindow(
        window_type=GoldenWindowType.PLATINUM_TEN,
        open_at_min=0.0, closes_at_min=30.0, optimal_end_min=15.0,
        compliance_threshold_min=20.0,
        description="Ruptured AAA: permissive hypotension, OR within 30 min"
    )
    SNAKEBITE = GoldenHourWindow(
        window_type=GoldenWindowType.EXTENDED,
        open_at_min=0.0, closes_at_min=120.0, optimal_end_min=60.0,
        compliance_threshold_min=90.0,
        description="Snakebite: ASV within 2h — haemotoxic"
    )
    NEONATAL = GoldenHourWindow(
        window_type=GoldenWindowType.PLATINUM_TEN,
        open_at_min=0.0, closes_at_min=10.0, optimal_end_min=3.0,
        compliance_threshold_min=5.0,
        description="Neonatal arrest: NRP within 5 min"
    )
    OBSTETRIC_HAEMORRHAGE = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=30.0, optimal_end_min=10.0,
        compliance_threshold_min=20.0,
        description="PPH / placental abruption: oxytocin + OR within 30 min"
    )
    PULMONARY_EMBOLISM = GoldenHourWindow(
        window_type=GoldenWindowType.SILVER_HOUR,
        open_at_min=0.0, closes_at_min=90.0, optimal_end_min=45.0,
        compliance_threshold_min=90.0,
        description="Massive PE: anticoagulation / thrombolysis within 90 min"
    )
    ORGANOPHOSPHATE = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=60.0, optimal_end_min=20.0,
        compliance_threshold_min=30.0,
        description="OP poisoning: atropine within 30 min critical"
    )
    PAEDIATRIC_CHOKING = GoldenHourWindow(
        window_type=GoldenWindowType.PLATINUM_TEN,
        open_at_min=0.0, closes_at_min=15.0, optimal_end_min=4.0,
        compliance_threshold_min=8.0,
        description="Severe paediatric airway obstruction: 4-minute brain death window"
    )
    DROWNING = GoldenHourWindow(
        window_type=GoldenWindowType.PLATINUM_TEN,
        open_at_min=0.0, closes_at_min=30.0, optimal_end_min=8.0,
        compliance_threshold_min=15.0,
        description="Drowning cardiac arrest: CPR + ACLS within 15 min"
    )
    CRUSH_SYNDROME = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=90.0, optimal_end_min=30.0,
        compliance_threshold_min=60.0,
        description="Crush syndrome: IV fluids BEFORE extrication, dialysis within 6h"
    )
    STANDARD_CRITICAL = GoldenHourWindow(
        window_type=GoldenWindowType.GOLDEN_HOUR,
        open_at_min=0.0, closes_at_min=60.0, optimal_end_min=30.0,
        compliance_threshold_min=60.0,
        description="Standard P1 critical: definitive care within 60 min"
    )
    STANDARD_URGENT = GoldenHourWindow(
        window_type=GoldenWindowType.EXTENDED,
        open_at_min=0.0, closes_at_min=180.0, optimal_end_min=60.0,
        compliance_threshold_min=120.0,
        description="Standard P2 urgent: ED within 3h"
    )
class SurvivalCurveRegistry:
    _REGISTRY: Dict[str, SurvivalParameters] = {}
    _built: bool = False
    @classmethod
    def _cardiac(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="stemi_anterior",
                display_name="STEMI — Anterior Wall",
                icd10_codes=("I21.0",),
                evidence_source="ESC 2023 STEMI Guidelines; Keeley NEJM 2003",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.88,
                phase1_lambda=0.0085,    
                phase2_lambda=0.0022,    
                biphasic_transition_min=30.0,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STEMI,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.3,
            ),
            SurvivalParameters(
                condition_key="stemi_inferior",
                display_name="STEMI — Inferior Wall",
                icd10_codes=("I21.1",),
                evidence_source="ESC 2023; AHA ACC 2013",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.91,
                phase1_lambda=0.0065,
                phase2_lambda=0.0018,
                biphasic_transition_min=30.0,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STEMI,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=150.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.25,
            ),
            SurvivalParameters(
                condition_key="stemi_posterior",
                display_name="STEMI — Posterior Wall",
                icd10_codes=("I21.2",),
                evidence_source="ESC 2023",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.89,
                decay_lambda=0.0055,
                survival_floor=0.14,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STEMI,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.25,
            ),
            SurvivalParameters(
                condition_key="stemi_with_vf_arrest",
                display_name="STEMI + VF Cardiac Arrest",
                icd10_codes=("I21.0", "I49.0"),
                evidence_source="ERC 2021 Resuscitation Guidelines — 10%/min VF rule",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.78,
                decay_lambda=0.0380,    
                survival_floor=0.02,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.CARDIAC_ARREST,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=10.0,
                per_patient_reward_weight=2.0,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="stemi_cocaine",
                display_name="STEMI — Cocaine-Induced Vasospasm",
                icd10_codes=("I21.A9", "T40.5X1A"),
                evidence_source="AHA Cocaine Cardiovascular Complications 2022",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.85,
                sigmoid_inflection_min=45.0,
                sigmoid_steepness=0.10,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STEMI,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="stemi_post_cabg",
                display_name="STEMI — Post-CABG Graft Thrombosis",
                icd10_codes=("I21.A9",),
                evidence_source="ACC/AHA CABG guidelines 2022",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.82,
                phase1_lambda=0.0090,
                phase2_lambda=0.0025,
                biphasic_transition_min=25.0,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STEMI,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=100.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.35,
            ),
            SurvivalParameters(
                condition_key="cardiac_arrest_vf",
                display_name="Cardiac Arrest — Ventricular Fibrillation",
                icd10_codes=("I46.0",),
                evidence_source="ERC Resuscitation Guidelines 2021 — shockable rhythms",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.75,
                decay_lambda=0.0400,
                survival_floor=0.02,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.CARDIAC_ARREST,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=8.0,
                per_patient_reward_weight=2.0,
                elderly_penalty_factor=1.6,
                paediatric_curve_variant="cardiac_arrest_vf_paediatric",
            ),
            SurvivalParameters(
                condition_key="cardiac_arrest_pea",
                display_name="Cardiac Arrest — PEA / Asystole",
                icd10_codes=("I46.9",),
                evidence_source="ERC 2021 — non-shockable rhythms (PEA survival ~5-10%)",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.32,
                decay_lambda=0.0600,
                survival_floor=0.01,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.CARDIAC_ARREST,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=6.0,
                per_patient_reward_weight=2.0,
                elderly_penalty_factor=1.8,
            ),
            SurvivalParameters(
                condition_key="cardiac_arrest_vf_paediatric",
                display_name="Paediatric Cardiac Arrest — VF",
                icd10_codes=("I46.0",),
                evidence_source="ERC Paediatric Life Support 2021",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.68,
                decay_lambda=0.0500,
                survival_floor=0.02,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.CARDIAC_ARREST,
                correct_unit_types=("ALS", "MICU"),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=8.0,
                per_patient_reward_weight=2.2,
            ),
            SurvivalParameters(
                condition_key="neonatal_resuscitation",
                display_name="Neonatal Cardiac Arrest",
                icd10_codes=("P28.5",),
                evidence_source="ILCOR NRP 2020 — neonatal resuscitation protocol",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.60,
                decay_lambda=0.0750,
                survival_floor=0.02,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.NEONATAL,
                correct_unit_types=("ALS", "MICU"),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=5.0,
                per_patient_reward_weight=2.5,
            ),
            SurvivalParameters(
                condition_key="lightning_strike",
                display_name="Lightning Strike — Cardiac Arrest",
                icd10_codes=("T75.09XA",),
                evidence_source="Wilderness Medicine Society Lightning Injury 2020",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.88,   
                decay_lambda=0.0500,
                survival_floor=0.05,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.CARDIAC_ARREST,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=10.0,
                per_patient_reward_weight=1.8,
            ),
            SurvivalParameters(
                condition_key="expected_death",
                display_name="Expected Death — Palliative",
                icd10_codes=("Z51.5",),
                evidence_source="N/A — palliative context",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.05,
                survival_floor=0.01,
                severity_tier=SeverityTier.P0,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=(),
                time_to_irreversible_min=9999.0,
                per_patient_reward_weight=0.1,
            ),
            SurvivalParameters(
                condition_key="aortic_dissection",
                display_name="Aortic Dissection — Type A",
                icd10_codes=("I71.01",),
                evidence_source="IRAD registry; ESC aortic guidelines 2014 — 1–2%/hr mortality untreated",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.86,
                decay_lambda=0.0120,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.AAA_RUPTURE,
                correct_unit_types=("MICU",),
                required_specialties=("cardiothoracic_surgery",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.6,
                elderly_penalty_factor=1.35,
            ),
            SurvivalParameters(
                condition_key="aaa_rupture",
                display_name="Ruptured Abdominal Aortic Aneurysm",
                icd10_codes=("I71.3",),
                evidence_source="EVAR-2 trial; Earnshaw Vascular 2020 — 80% in-hospital mortality without surgery",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.72,
                sigmoid_inflection_min=20.0,
                sigmoid_steepness=0.18,
                survival_floor=0.05,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.AAA_RUPTURE,
                correct_unit_types=("MICU",),
                required_specialties=("cardiothoracic_surgery",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.8,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="wpw_svt",
                display_name="WPW Syndrome — Rapid SVT",
                icd10_codes=("I45.6",),
                evidence_source="AHA SVT Management 2015 — avoid adenosine in WPW",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.94,
                plateau_end_min=45.0,
                plateau_decay_lambda=0.0120,
                survival_floor=0.20,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.2,
            ),
            SurvivalParameters(
                condition_key="complete_heart_block",
                display_name="Complete (3rd Degree) Heart Block",
                icd10_codes=("I44.2",),
                evidence_source="AHA ACLS 2020 — unstable bradycardia",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.92,
                plateau_end_min=30.0,
                plateau_decay_lambda=0.0200,
                survival_floor=0.20,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.2,
                elderly_penalty_factor=1.3,
            ),
            SurvivalParameters(
                condition_key="brugada_event",
                display_name="Brugada Syndrome — Suspected VF Event",
                icd10_codes=("I45.81",),
                evidence_source="ESC Brugada Guidelines 2015",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.90,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0080,
                survival_floor=0.25,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.0,
            ),
            SurvivalParameters(
                condition_key="fast_af",
                display_name="Fast Atrial Fibrillation",
                icd10_codes=("I48.91",),
                evidence_source="AHA Afib Management 2019",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.96,
                survival_floor=0.40,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=240.0,
                per_patient_reward_weight=0.8,
            ),
            SurvivalParameters(
                condition_key="pacemaker_failure",
                display_name="Pacemaker Failure — Bradycardia",
                icd10_codes=("Z45.010",),
                evidence_source="AHA Device Therapy 2018",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.94,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0080,
                survival_floor=0.30,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=0.9,
                elderly_penalty_factor=1.2,
            ),
            SurvivalParameters(
                condition_key="hypocalcaemia",
                display_name="Post-Thyroidectomy Hypocalcaemia",
                icd10_codes=("E83.51",),
                evidence_source="ATA Thyroid Surgery Complications 2015",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.97,
                plateau_end_min=90.0,
                plateau_decay_lambda=0.0060,
                survival_floor=0.40,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=180.0,
                per_patient_reward_weight=0.7,
            ),
            SurvivalParameters(
                condition_key="hyperkalaemia",
                display_name="Hyperkalaemia — ECG Changes / Dialysis Patient",
                icd10_codes=("E87.5",),
                evidence_source="KDIGO 2012; AHA ACLS 2020 — sine wave = imminent standstill",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.82,
                sigmoid_inflection_min=30.0,
                sigmoid_steepness=0.12,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("renal_dialysis",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.3,
            ),
            SurvivalParameters(
                condition_key="acute_pulmonary_oedema",
                display_name="Acute Cardiogenic Pulmonary Oedema",
                icd10_codes=("I50.1",),
                evidence_source="ESC Heart Failure Guidelines 2021",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.84,
                decay_lambda=0.0140,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("cardiac_cath_lab",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.4,
            ),
        ]
    @classmethod
    def _stroke(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="ischemic_stroke",
                display_name="Acute Ischemic Stroke — Known Onset",
                icd10_codes=("I63.9",),
                evidence_source="Saver JAMA 2006 — 1.9M neurons/min; IST-3 Lancet 2012",
                curve_model=CurveModelType.LINEAR,
                initial_survival_prob=0.90,
                linear_slope_per_min=0.00085,
                survival_floor=0.25,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STROKE_THROMBOLYSIS,
                correct_unit_types=("ALS",),
                required_specialties=("stroke_unit",),
                time_to_irreversible_min=270.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.3,
            ),
            SurvivalParameters(
                condition_key="ischemic_stroke_wake_up",
                display_name="Ischemic Stroke — Wake-Up (Unknown Onset)",
                icd10_codes=("I63.9",),
                evidence_source="WAKE-UP trial NEJM 2018; MRI DWI-FLAIR mismatch",
                curve_model=CurveModelType.LINEAR,
                initial_survival_prob=0.86,
                linear_slope_per_min=0.00090,
                survival_floor=0.22,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STROKE_THROMBOLYSIS,
                correct_unit_types=("ALS",),
                required_specialties=("stroke_unit",),
                time_to_irreversible_min=270.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.35,
            ),
            SurvivalParameters(
                condition_key="hemorrhagic_stroke_sah",
                display_name="Subarachnoid Haemorrhage",
                icd10_codes=("I60.9",),
                evidence_source="ISAT trial; Connolly Stroke 2012 — rebleed 4% in first 24h",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.78,
                sigmoid_inflection_min=60.0,
                sigmoid_steepness=0.065,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="meningitis_cryptococcal",
                display_name="Cryptococcal Meningitis (HIV)",
                icd10_codes=("B45.1",),
                evidence_source="WHO HIV Guidelines 2019; ACTA Trial 2019",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.72,
                decay_lambda=0.0058,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="paediatric_stroke",
                display_name="Paediatric Arterial Ischaemic Stroke",
                icd10_codes=("G45.9",),
                evidence_source="AHA Paediatric Stroke Guidelines 2019",
                curve_model=CurveModelType.LINEAR,
                initial_survival_prob=0.92,
                linear_slope_per_min=0.00070,
                survival_floor=0.30,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STROKE_THROMBOLYSIS,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=270.0,
                per_patient_reward_weight=1.6,
            ),
        ]
    @classmethod
    def _trauma(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="polytrauma_blunt",
                display_name="Blunt Polytrauma — High Energy",
                icd10_codes=("T07",),
                evidence_source="MTOS Major Trauma Outcome Study; TARN Data 2022",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.80,
                phase1_lambda=0.0120,
                phase2_lambda=0.0040,
                biphasic_transition_min=20.0,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.8,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="polytrauma_penetrating",
                display_name="Penetrating Polytrauma",
                icd10_codes=("T07",),
                evidence_source="TRAUMA Study Group 2020 — penetrating vs blunt",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.78,
                decay_lambda=0.0150,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.9,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="penetrating_chest",
                display_name="Penetrating Chest Trauma",
                icd10_codes=("S21.1",),
                evidence_source="Cothren JACS 2009 — haemopneumothorax management",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.82,
                sigmoid_inflection_min=25.0,
                sigmoid_steepness=0.14,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("ALS",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.6,
            ),
            SurvivalParameters(
                condition_key="severe_tbi",
                display_name="Severe Traumatic Brain Injury (GCS ≤8)",
                icd10_codes=("S06.9",),
                evidence_source="IMPACT study; Brain Trauma Foundation 2016 guidelines",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.72,
                phase1_lambda=0.0100,
                phase2_lambda=0.0030,
                biphasic_transition_min=30.0,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.8,
                elderly_penalty_factor=1.6,
            ),
            SurvivalParameters(
                condition_key="skull_base_fracture",
                display_name="Skull Base Fracture",
                icd10_codes=("S02.1",),
                evidence_source="EAST TBI guidelines 2018",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.84,
                plateau_end_min=30.0,
                plateau_decay_lambda=0.0080,
                survival_floor=0.20,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="subdural_haematoma",
                display_name="Acute Subdural Haematoma",
                icd10_codes=("S06.4",),
                evidence_source="Bullock Neurosurgery 2006 — 'talk and die' phenomenon",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.88,
                plateau_end_min=40.0,
                plateau_decay_lambda=0.0180,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.6,
            ),
            SurvivalParameters(
                condition_key="cervical_spinal_injury",
                display_name="Cervical Spinal Cord Injury",
                icd10_codes=("S14.0",),
                evidence_source="NACRS registry; Wing JNEURO 2018",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.85,
                sigmoid_inflection_min=50.0,
                sigmoid_steepness=0.080,
                survival_floor=0.30,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.5,
            ),
            SurvivalParameters(
                condition_key="thoracic_spinal_injury",
                display_name="Thoracic / Lumbar Spinal Cord Injury",
                icd10_codes=("S24.0",),
                evidence_source="STASCIS trial 2012 — early decompression benefit",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.90,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0050,
                survival_floor=0.35,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="chest_trauma",
                display_name="Multiple Rib Fractures / Haemothorax",
                icd10_codes=("S22.4", "S27.1"),
                evidence_source="Eastern Association for Surgery of Trauma 2017",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.85,
                decay_lambda=0.0090,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("ALS",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="splenic_laceration",
                display_name="Splenic Laceration — Haemodynamically Unstable",
                icd10_codes=("S36.01",),
                evidence_source="Trauma NOM vs OM 2020 — spleen salvage rates",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.87,
                phase1_lambda=0.0100,
                phase2_lambda=0.0030,
                biphasic_transition_min=25.0,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("ALS",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="crush_syndrome",
                display_name="Crush Syndrome / Rhabdomyolysis",
                icd10_codes=("T79.5",),
                evidence_source="Sever NEJM 1941; WHO EMS guidelines earthquake injuries",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.80,
                sigmoid_inflection_min=60.0,
                sigmoid_steepness=0.055,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.CRUSH_SYNDROME,
                correct_unit_types=("MICU",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="traumatic_amputation",
                display_name="Traumatic Amputation",
                icd10_codes=("T11.1",),
                evidence_source="EAST guidelines; replantation window 6h",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.88,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0060,
                survival_floor=0.30,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("ALS",),
                required_specialties=("plastic_surgery",),
                time_to_irreversible_min=360.0,
                per_patient_reward_weight=1.1,
            ),
            SurvivalParameters(
                condition_key="degloving_crush",
                display_name="Degloving + Crush Injury",
                icd10_codes=("S60.0",),
                evidence_source="EAST; BAPRAS replantation guidelines 2020",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.87,
                plateau_end_min=90.0,
                plateau_decay_lambda=0.0050,
                survival_floor=0.25,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("plastic_surgery",),
                time_to_irreversible_min=360.0,
                per_patient_reward_weight=1.0,
            ),
            SurvivalParameters(
                condition_key="femur_fracture",
                display_name="Femur Fracture — Tachycardia / Blood Loss",
                icd10_codes=("S72.0",),
                evidence_source="TARN 2020 — long bone fracture haemorrhage",
                curve_model=CurveModelType.LINEAR,
                initial_survival_prob=0.92,
                linear_slope_per_min=0.00080,
                survival_floor=0.35,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=0.9,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="brachial_plexus",
                display_name="Brachial Plexus Injury",
                icd10_codes=("S14.3",),
                evidence_source="Hentz JBJS 2009",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.98,
                survival_floor=0.60,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=180.0,
                per_patient_reward_weight=0.7,
            ),
            SurvivalParameters(
                condition_key="open_fracture",
                display_name="Open Fracture — Contaminated",
                icd10_codes=("S82.2",),
                evidence_source="EAST; antibiotics within 6h window",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.96,
                plateau_end_min=120.0,
                plateau_decay_lambda=0.0020,
                survival_floor=0.55,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=360.0,
                per_patient_reward_weight=0.7,
            ),
            SurvivalParameters(
                condition_key="facial_trauma",
                display_name="Panfacial Fractures",
                icd10_codes=("S09.8",),
                evidence_source="ATLS facial trauma 2018",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.95,
                survival_floor=0.55,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("neurosurgery",),
                time_to_irreversible_min=240.0,
                per_patient_reward_weight=0.7,
            ),
            SurvivalParameters(
                condition_key="gsw_shoulder",
                display_name="Gunshot Wound — Shoulder",
                icd10_codes=("S40.01XA",),
                evidence_source="EAST penetrating extremity 2012",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.96,
                plateau_end_min=90.0,
                plateau_decay_lambda=0.0030,
                survival_floor=0.40,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=180.0,
                per_patient_reward_weight=0.8,
            ),
            SurvivalParameters(
                condition_key="non_accidental_injury",
                display_name="Non-Accidental Injury (Child Abuse)",
                icd10_codes=("T74.1XXA",),
                evidence_source="NICE child abuse guidelines 2017",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.85,
                plateau_end_min=30.0,
                plateau_decay_lambda=0.0100,
                survival_floor=0.18,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=2.0,
            ),
            SurvivalParameters(
                condition_key="minor_trauma",
                display_name="Minor Trauma / Walking Wounded",
                icd10_codes=("S00",),
                evidence_source="N/A — minimal mortality risk",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.90,
                severity_tier=SeverityTier.P3,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.3,
            ),
            SurvivalParameters(
                condition_key="minor_head_trauma",
                display_name="Minor Head Injury — Brief LOC",
                icd10_codes=("S09.90XA",),
                evidence_source="PECARN head injury rule 2009",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.99,
                plateau_end_min=180.0,
                plateau_decay_lambda=0.0008,
                survival_floor=0.70,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=480.0,
                per_patient_reward_weight=0.5,
                elderly_penalty_factor=1.3,
            ),
            SurvivalParameters(
                condition_key="nof_fracture",
                display_name="Neck of Femur Fracture — Elderly",
                icd10_codes=("S72.00",),
                evidence_source="SIGN 2009; AO Foundation — 30-day mortality 10-14%",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.96,
                plateau_end_min=240.0,
                plateau_decay_lambda=0.0010,
                survival_floor=0.65,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=720.0,
                per_patient_reward_weight=0.6,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="colles_fracture",
                display_name="Colles Fracture — Distal Radius",
                icd10_codes=("S52.50",),
                evidence_source="N/A — negligible mortality",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.95,
                severity_tier=SeverityTier.P3,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.2,
            ),
            SurvivalParameters(
                condition_key="acl_rupture",
                display_name="ACL Rupture — Haemarthrosis",
                icd10_codes=("S83.51",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.96,
                severity_tier=SeverityTier.P3,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.2,
            ),
            SurvivalParameters(
                condition_key="mci_rta",
                display_name="MCI — Road Traffic Accident",
                icd10_codes=("T07",),
                evidence_source="WHO Global Road Safety 2022 — MCI protocols",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.75,
                phase1_lambda=0.0140,
                phase2_lambda=0.0035,
                biphasic_transition_min=20.0,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU", "ALS", "BLS"),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.6,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="blast_injury",
                display_name="Blast / Explosion Injury",
                icd10_codes=("T70.8",),
                evidence_source="Military trauma registry; blast triage protocols",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.72,
                sigmoid_inflection_min=25.0,
                sigmoid_steepness=0.120,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.9,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="mine_collapse",
                display_name="Mine Collapse / Structural Entrapment",
                icd10_codes=("W35",),
                evidence_source="WHO Disaster Medical Response 2018",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.78,
                sigmoid_inflection_min=90.0,
                sigmoid_steepness=0.040,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.CRUSH_SYNDROME,
                correct_unit_types=("MICU",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.6,
            ),
            SurvivalParameters(
                condition_key="mine_rescue_standby",
                display_name="Mine Rescue — Standby (Unknown Injuries)",
                icd10_codes=("W35",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.85,
                survival_floor=0.20,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.CRUSH_SYNDROME,
                correct_unit_types=("MICU", "ALS"),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.0,
            ),
            SurvivalParameters(
                condition_key="mci_natural_disaster",
                display_name="MCI — Natural Disaster",
                icd10_codes=("T07",),
                evidence_source="WHO Disaster Field Operations 2018",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.72,
                phase1_lambda=0.0160,
                phase2_lambda=0.0040,
                biphasic_transition_min=30.0,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.MAJOR_TRAUMA,
                correct_unit_types=("MICU",),
                required_specialties=("level_1_trauma",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.8,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="rhabdomyolysis",
                display_name="Exertional Rhabdomyolysis — AKI",
                icd10_codes=("T79.6",),
                evidence_source="Shapiro Muscle Nerve 2012",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.92,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0060,
                survival_floor=0.25,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("renal_dialysis",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.1,
            ),
        ]
    @classmethod
    def _burns(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="burns_major",
                display_name="Major Burns ≥40% TBSA + Inhalation Injury",
                icd10_codes=("T31.40",),
                evidence_source="Osler JACS 2010 Baux score survival model; Belgian Burn Registry",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.62,
                step_times_min=(30.0, 60.0, 120.0),
                step_penalties=(0.88, 0.82, 0.76),   
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.BURNS,
                correct_unit_types=("ALS",),
                required_specialties=("burns_unit",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.8,
            ),
            SurvivalParameters(
                condition_key="burns_moderate",
                display_name="Moderate Burns 15–40% TBSA",
                icd10_codes=("T31.30",),
                evidence_source="Belgian Burn Registry 2020",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.84,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0055,
                survival_floor=0.18,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.BURNS,
                correct_unit_types=("ALS",),
                required_specialties=("burns_unit",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.6,
            ),
            SurvivalParameters(
                condition_key="burns_electrical",
                display_name="Electrical Burns",
                icd10_codes=("T75.4",),
                evidence_source="ABA Electrical burns protocol 2018",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.90,
                plateau_end_min=120.0,
                plateau_decay_lambda=0.0040,
                survival_floor=0.30,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("burns_unit",),
                time_to_irreversible_min=180.0,
                per_patient_reward_weight=0.9,
            ),
            SurvivalParameters(
                condition_key="chemical_burns",
                display_name="Chemical Burns — Acid/Alkali",
                icd10_codes=("T54.1",),
                evidence_source="ABA Chemical Burns 2018",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.86,
                sigmoid_inflection_min=40.0,
                sigmoid_steepness=0.100,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.BURNS,
                correct_unit_types=("ALS",),
                required_specialties=("burns_unit",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.4,
            ),
        ]
    @classmethod
    def _respiratory(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="respiratory_failure",
                display_name="Acute Respiratory Failure",
                icd10_codes=("J96.0",),
                evidence_source="ACCP mechanical ventilation guidelines 2017",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.84,
                decay_lambda=0.0120,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="ards",
                display_name="ARDS — Acute Respiratory Distress Syndrome",
                icd10_codes=("J80",),
                evidence_source="ARDSNet NEJM 2000; WHO COVID ARDS severity 2021",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.80,
                sigmoid_inflection_min=25.0,
                sigmoid_steepness=0.100,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="paediatric_respiratory",
                display_name="Paediatric Respiratory Failure / Bronchiolitis",
                icd10_codes=("J21.9",),
                evidence_source="PICU respiratory outcomes; BRONCHUS trial 2019",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.88,
                decay_lambda=0.0130,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.6,
            ),
            SurvivalParameters(
                condition_key="pulmonary_embolism",
                display_name="Massive Pulmonary Embolism",
                icd10_codes=("I26.01",),
                evidence_source="PEITHO trial 2014; ESC PE guidelines 2019",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.83,
                sigmoid_inflection_min=50.0,
                sigmoid_steepness=0.080,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PULMONARY_EMBOLISM,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.3,
            ),
            SurvivalParameters(
                condition_key="massive_haemoptysis",
                display_name="Massive Haemoptysis — TB / CF",
                icd10_codes=("R04.2",),
                evidence_source="Sakr JCRM 2010 — mortality >50% if not bronchoscoped",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.82,
                decay_lambda=0.0160,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("cardiothoracic_surgery",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="decompression_sickness",
                display_name="Decompression Sickness — Type II",
                icd10_codes=("T70.3",),
                evidence_source="Undersea Hyperbaric Medical Society 2019",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.89,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0080,
                survival_floor=0.20,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("hyperbaric_chamber",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.1,
            ),
            SurvivalParameters(
                condition_key="massive_pleural_effusion",
                display_name="Massive Pleural Effusion — Mesothelioma",
                icd10_codes=("J90",),
                evidence_source="N/A — palliative context",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.85,
                decay_lambda=0.0100,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("cardiothoracic_surgery",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.1,
                elderly_penalty_factor=1.4,
            ),
        ]
    @classmethod
    def _obstetric(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="obstetric_hemorrhage",
                display_name="Obstetric Haemorrhage — PPH / Abruption",
                icd10_codes=("O72.1", "O45.0"),
                evidence_source="WOMAN Trial Lancet 2017; WHO PPH guidelines",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.88,
                sigmoid_inflection_min=20.0,
                sigmoid_steepness=0.180,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.OBSTETRIC_HAEMORRHAGE,
                correct_unit_types=("MICU", "ALS"),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=2.0,
            ),
            SurvivalParameters(
                condition_key="eclampsia",
                display_name="Eclampsia — Tonic-Clonic Seizure",
                icd10_codes=("O15.0",),
                evidence_source="MAGPIE trial Lancet 2002; WHO MgSO4 protocol",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.90,
                decay_lambda=0.0200,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.ECLAMPSIA,
                correct_unit_types=("MICU", "ALS"),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=1.9,
            ),
            SurvivalParameters(
                condition_key="uterine_rupture",
                display_name="Uterine Rupture — Scar Dehiscence",
                icd10_codes=("O71.1",),
                evidence_source="Zwart BJOG 2009 — 0.07/1000 births, high maternal mortality",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.84,
                sigmoid_inflection_min=15.0,
                sigmoid_steepness=0.200,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.OBSTETRIC_HAEMORRHAGE,
                correct_unit_types=("ALS",),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=2.0,
            ),
            SurvivalParameters(
                condition_key="obstetric_trauma",
                display_name="Trauma in Pregnancy",
                icd10_codes=("O26.5",),
                evidence_source="EAST trauma in pregnancy 2010",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.93,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0055,
                survival_floor=0.30,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="normal_delivery",
                display_name="Active Labour — Imminent Delivery",
                icd10_codes=("O80",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.85,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.5,
            ),
            SurvivalParameters(
                condition_key="precipitate_delivery",
                display_name="Precipitate Delivery — Imminent Birth",
                icd10_codes=("O62.3",),
                evidence_source="N/A",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.96,
                sigmoid_inflection_min=15.0,
                sigmoid_steepness=0.120,
                survival_floor=0.40,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.ECLAMPSIA,
                correct_unit_types=("ALS",),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=1.5,
            ),
            SurvivalParameters(
                condition_key="neonatal_emergency",
                display_name="Neonatal Emergency — Congenital / ALTE",
                icd10_codes=("P29.8",),
                evidence_source="ILCOR NRP 2020",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.70,
                decay_lambda=0.0600,
                survival_floor=0.03,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.NEONATAL,
                correct_unit_types=("MICU", "ALS"),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=8.0,
                per_patient_reward_weight=2.2,
            ),
        ]
    @classmethod
    def _sepsis(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="septic_shock",
                display_name="Septic Shock",
                icd10_codes=("A41.9",),
                evidence_source="SSC Hour-1 Bundle 2018; PROCESS trial NEJM 2014",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.76,
                decay_lambda=0.0075,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.5,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="sepsis_surgical",
                display_name="Sepsis — Surgical Source",
                icd10_codes=("T81.4",),
                evidence_source="SSC 2018",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.80,
                decay_lambda=0.0065,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="necrotising_fasciitis",
                display_name="Necrotising Fasciitis",
                icd10_codes=("M72.6",),
                evidence_source="Sarani JACS 2009 — 1h delay = 12% increased mortality",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.74,
                sigmoid_inflection_min=40.0,
                sigmoid_steepness=0.120,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.6,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="neutropenic_sepsis",
                display_name="Neutropenic Sepsis — Oncology Patient",
                icd10_codes=("D70.1", "A41.9"),
                evidence_source="ESMO febrile neutropenia guidelines 2016",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.80,
                decay_lambda=0.0080,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="meningococcal_sepsis",
                display_name="Meningococcal Septicaemia",
                icd10_codes=("A39.2",),
                evidence_source="Ninis BMJ 2005 — antibiotics before hospital reduces mortality 2-3x",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.80,
                decay_lambda=0.0180,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS", "MICU"),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=2.0,
                paediatric_curve_variant="meningococcal_paediatric",
            ),
            SurvivalParameters(
                condition_key="meningococcal_paediatric",
                display_name="Meningococcal Sepsis — Paediatric",
                icd10_codes=("A39.2",),
                evidence_source="GOSH Meningococcal outcomes 2020",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.82,
                decay_lambda=0.0200,
                survival_floor=0.05,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("MICU", "ALS"),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=2.4,
            ),
            SurvivalParameters(
                condition_key="weil_disease",
                display_name="Weil's Disease (Leptospirosis) — Multi-Organ Failure",
                icd10_codes=("A27.0",),
                evidence_source="WHO Leptospirosis guidelines 2018; Rajapakse 2015",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.72,
                decay_lambda=0.0090,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("renal_dialysis",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.5,
            ),
            SurvivalParameters(
                condition_key="weil_disease_severe",
                display_name="Weil's Disease — Pulmonary Haemorrhage",
                icd10_codes=("A27.0",),
                evidence_source="WHO 2018",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.65,
                decay_lambda=0.0120,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("renal_dialysis",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.8,
            ),
            SurvivalParameters(
                condition_key="transplant_sepsis",
                display_name="Post-Transplant Sepsis",
                icd10_codes=("T86.0", "A41.9"),
                evidence_source="AST transplant infectious disease guidelines 2019",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.74,
                decay_lambda=0.0080,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("transplant_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="urosepsis",
                display_name="Obstructive Urosepsis",
                icd10_codes=("N10", "A41.9"),
                evidence_source="EAU Urological Infections 2023",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.80,
                decay_lambda=0.0075,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("renal_dialysis",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="sbp_peritonitis",
                display_name="Spontaneous Bacterial Peritonitis",
                icd10_codes=("K65.2",),
                evidence_source="EASL cirrhosis guidelines 2018",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.72,
                decay_lambda=0.0090,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.SEPSIS_HOUR_1,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="enteric_fever",
                display_name="Enteric Fever Outbreak (Typhoid / Shigella)",
                icd10_codes=("A01.0",),
                evidence_source="WHO enteric fever treatment 2018",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.85,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0060,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.2,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="sepsis_malnutrition",
                display_name="Sepsis with Severe Acute Malnutrition (Paediatric)",
                icd10_codes=("E43", "A41.9"),
                evidence_source="WHO SAM Treatment 2013 — ReSoMal protocol",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.68,
                decay_lambda=0.0100,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("BLS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.5,
            ),
            SurvivalParameters(
                condition_key="severe_malaria",
                display_name="Severe Falciparum Malaria",
                icd10_codes=("B50.0",),
                evidence_source="AQUAMAT trial Lancet 2010 — IV artesunate vs quinine",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.78,
                decay_lambda=0.0080,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.4,
                paediatric_curve_variant="severe_malaria",
            ),
            SurvivalParameters(
                condition_key="dengue_severe",
                display_name="Severe Dengue — Plasma Leakage / Shock",
                icd10_codes=("A97.2",),
                evidence_source="WHO Dengue Guidelines 2012; SEARO guidelines",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.84,
                plateau_end_min=45.0,
                plateau_decay_lambda=0.0120,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="pneumocystis_pneumonia",
                display_name="PCP Pneumonia — HIV/AIDS",
                icd10_codes=("B59",),
                evidence_source="ACTG 021 study 1990; WHO HIV treatment 2021",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.78,
                decay_lambda=0.0080,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.3,
                elderly_penalty_factor=1.4,
            ),
        ]
    @classmethod
    def _toxicology(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="organophosphate_poisoning",
                display_name="Organophosphate (Pesticide) Poisoning",
                icd10_codes=("T60.0X1A",),
                evidence_source="WHO OP poisoning treatment 2020; IPCS guidelines",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.76,
                sigmoid_inflection_min=25.0,
                sigmoid_steepness=0.150,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.ORGANOPHOSPHATE,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.6,
            ),
            SurvivalParameters(
                condition_key="opioid_overdose",
                display_name="Opioid Overdose — Respiratory Arrest",
                icd10_codes=("T40.2X1A",),
                evidence_source="WHO opioid overdose guidelines 2014; NALOXONE efficacy data",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.88,
                decay_lambda=0.0250,
                survival_floor=0.05,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.CARDIAC_ARREST,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=10.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="toxic_inhalation",
                display_name="Toxic Gas Inhalation (Chlorine/Industrial)",
                icd10_codes=("T59.4X1A",),
                evidence_source="NIOSH chemical emergency guidelines",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.80,
                sigmoid_inflection_min=30.0,
                sigmoid_steepness=0.100,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="carbon_monoxide",
                display_name="Carbon Monoxide Poisoning",
                icd10_codes=("T58.01XA",),
                evidence_source="Weaver NEJM 2002 — HBO reduces neurological sequelae",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.88,
                step_times_min=(30.0, 60.0, 120.0),
                step_penalties=(0.90, 0.85, 0.78),
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("hyperbaric_chamber",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="cbrn_suspected",
                display_name="CBRN Exposure — Suspected Biological/Chemical",
                icd10_codes=("T65.91XA",),
                evidence_source="CHEMPACK programme; NDMA India CBRN protocols",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.75,
                sigmoid_inflection_min=20.0,
                sigmoid_steepness=0.120,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.STAY_AND_PLAY,
                golden_window=GoldenHourWindows.ORGANOPHOSPHATE,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.6,
            ),
            SurvivalParameters(
                condition_key="paraquat_exposure",
                display_name="Paraquat Skin Exposure",
                icd10_codes=("T60.3X1A",),
                evidence_source="Vale BMJ 1987 — delayed multi-organ failure",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.78,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0080,
                survival_floor=0.08,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=240.0,
                per_patient_reward_weight=1.2,
            ),
            SurvivalParameters(
                condition_key="phosphine_poisoning",
                display_name="Zinc Phosphide / Phosphine Mass Poisoning",
                icd10_codes=("T57.1X1A",),
                evidence_source="Chugh IJM 2006 — Indian zinc phosphide poisoning data",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.62,
                sigmoid_inflection_min=20.0,
                sigmoid_steepness=0.160,
                survival_floor=0.04,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.8,
            ),
            SurvivalParameters(
                condition_key="mdma_toxicity",
                display_name="MDMA Toxidrome — Serotonin Syndrome / Hyperthermia",
                icd10_codes=("T43.621A",),
                evidence_source="Hall Lancet 1996; serotonin syndrome treatment protocols",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.84,
                step_times_min=(20.0, 40.0, 60.0),
                step_penalties=(0.88, 0.83, 0.78),
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="botulism_outbreak",
                display_name="Botulism — Mass Outbreak",
                icd10_codes=("A05.1",),
                evidence_source="CDC botulism treatment guidelines 2021",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.82,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0070,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("MICU",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="botulism_foodborne",
                display_name="Foodborne Botulism",
                icd10_codes=("A05.1",),
                evidence_source="CDC 2021",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.84,
                plateau_end_min=90.0,
                plateau_decay_lambda=0.0060,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="paediatric_poisoning",
                display_name="Paediatric Poisoning / Overdose",
                icd10_codes=("T39.1X1A",),
                evidence_source="TISS paediatric toxicology; AAP guidelines 2019",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.88,
                decay_lambda=0.0120,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.6,
            ),
            SurvivalParameters(
                condition_key="hydrocarbon_ingestion",
                display_name="Hydrocarbon Aspiration (Kerosene)",
                icd10_codes=("T52.0X1A",),
                evidence_source="AAP kerosene guidelines 2003",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.85,
                decay_lambda=0.0140,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=45.0,
                per_patient_reward_weight=1.5,
            ),
            SurvivalParameters(
                condition_key="alcohol_withdrawal_seizure",
                display_name="Alcohol Withdrawal Seizure / DTs",
                icd10_codes=("F10.231",),
                evidence_source="CIWA protocol; Ely Alcohol Withdrawal 2019",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.88,
                sigmoid_inflection_min=15.0,
                sigmoid_steepness=0.120,
                survival_floor=0.20,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.2,
            ),
            SurvivalParameters(
                condition_key="snakebite_hemotoxic",
                display_name="Snakebite — Hemotoxic (Russell's Viper)",
                icd10_codes=("T63.029A",),
                evidence_source="WHO Snakebite Management 2010; Isbister Lancet 2015",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.84,
                plateau_end_min=30.0,
                plateau_decay_lambda=0.0100,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.SNAKEBITE,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="snakebite_neurotoxic",
                display_name="Snakebite — Neurotoxic (Cobra)",
                icd10_codes=("T63.041A",),
                evidence_source="Warrell Toxicon 2010; ASV India protocol",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.80,
                sigmoid_inflection_min=30.0,
                sigmoid_steepness=0.120,
                survival_floor=0.06,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("toxicology",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.5,
            ),
        ]
    @classmethod
    def _drowning(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="submersion",
                display_name="Drowning — Cardiac Arrest",
                icd10_codes=("T75.1",),
                evidence_source="ERC Drowning 2021; Szpilman Circulation 2012",
                curve_model=CurveModelType.BIPHASIC,
                initial_survival_prob=0.72,
                phase1_lambda=0.0450,
                phase2_lambda=0.0100,
                biphasic_transition_min=8.0,
                survival_floor=0.03,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.DROWNING,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=1.8,
                paediatric_curve_variant="submersion",
            ),
            SurvivalParameters(
                condition_key="near_drowning",
                display_name="Near-Drowning — Aspiration Pneumonitis Risk",
                icd10_codes=("T75.1",),
                evidence_source="Szpilman classification 2012",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.94,
                plateau_end_min=60.0,
                plateau_decay_lambda=0.0060,
                survival_floor=0.35,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=240.0,
                per_patient_reward_weight=0.8,
            ),
            SurvivalParameters(
                condition_key="drowning_cold_water",
                display_name="Cold Water Drowning (Hypothermic Protection)",
                icd10_codes=("T75.1", "T68"),
                evidence_source="ERC 2021 — 'not dead until warm and dead'",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.76,
                sigmoid_inflection_min=30.0,
                sigmoid_steepness=0.060,
                survival_floor=0.05,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.DROWNING,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.5,
            ),
        ]
    @classmethod
    def _anaphylaxis(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="anaphylaxis",
                display_name="Severe Anaphylaxis",
                icd10_codes=("T78.2",),
                evidence_source="EAACI Anaphylaxis Guidelines 2021; Sheikh JACI 2014",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.92,
                decay_lambda=0.0350,
                survival_floor=0.04,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.ANAPHYLAXIS,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=10.0,
                per_patient_reward_weight=1.6,
                paediatric_curve_variant="anaphylaxis",
            ),
            SurvivalParameters(
                condition_key="minor_allergy",
                display_name="Mild Allergic Reaction — Urticaria Only",
                icd10_codes=("T78.1",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.92,
                severity_tier=SeverityTier.P3,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.2,
            ),
        ]
    @classmethod
    def _metabolic(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="hypoglycaemia",
                display_name="Severe Hypoglycaemia — Unconscious",
                icd10_codes=("E16.0",),
                evidence_source="ACCORD trial; ADA Standards of Care 2023",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.92,
                decay_lambda=0.0180,
                survival_floor=0.10,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=1.2,
            ),
            SurvivalParameters(
                condition_key="dka",
                display_name="Diabetic Ketoacidosis",
                icd10_codes=("E11.10",),
                evidence_source="JBDS DKA guidelines 2021; BSPED paediatric DKA",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.90,
                plateau_end_min=45.0,
                plateau_decay_lambda=0.0090,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.2,
                paediatric_curve_variant="dka",
            ),
            SurvivalParameters(
                condition_key="heat_stroke",
                display_name="Heat Stroke — Dry/Anhidrotic",
                icd10_codes=("T67.0",),
                evidence_source="Epstein NEJM 1992 — target <39°C in 30 min; Indian MOHFW 2015",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.82,
                step_times_min=(15.0, 30.0, 60.0),
                step_penalties=(0.88, 0.80, 0.72),
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.4,
                elderly_penalty_factor=1.6,
            ),
            SurvivalParameters(
                condition_key="heat_exhaustion",
                display_name="Heat Exhaustion",
                icd10_codes=("T67.3",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.98,
                survival_floor=0.75,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=360.0,
                per_patient_reward_weight=0.4,
                elderly_penalty_factor=1.3,
            ),
        ]
    @classmethod
    def _paediatric(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="paediatric_choking",
                display_name="Severe Paediatric Airway Obstruction",
                icd10_codes=("T17.9",),
                evidence_source="ERC Paediatric BLS 2021 — 4-min brain death window",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.88,
                decay_lambda=0.0750,
                survival_floor=0.03,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CHOKING,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=4.0,
                per_patient_reward_weight=2.4,
            ),
            SurvivalParameters(
                condition_key="paediatric_ingestion",
                display_name="Paediatric Foreign Body Ingestion — Button Battery",
                icd10_codes=("T18.1",),
                evidence_source="Litovitz Ped 2010 — oesophageal battery caustic within 2h",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.92,
                step_times_min=(30.0, 60.0, 120.0),
                step_penalties=(0.96, 0.88, 0.75),
                survival_floor=0.30,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.4,
            ),
            SurvivalParameters(
                condition_key="paediatric_head_trauma",
                display_name="Paediatric Head Trauma — Possible NAI",
                icd10_codes=("S09.8",),
                evidence_source="PECARN 2009; NICE head injury guidelines 2019",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.88,
                plateau_end_min=20.0,
                plateau_decay_lambda=0.0120,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.8,
            ),
            SurvivalParameters(
                condition_key="paediatric_joint",
                display_name="Paediatric Septic Arthritis",
                icd10_codes=("M00.9",),
                evidence_source="Kocher JBJS 2004 — Kocher criteria",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.98,
                plateau_end_min=180.0,
                plateau_decay_lambda=0.0015,
                survival_floor=0.65,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=360.0,
                per_patient_reward_weight=0.7,
            ),
            SurvivalParameters(
                condition_key="status_epilepticus",
                display_name="Status Epilepticus — >5 Minutes",
                icd10_codes=("G41.9",),
                evidence_source="ERC Status Epilepticus 2020 — RAMPART trial",
                curve_model=CurveModelType.SIGMOID_DECAY,
                initial_survival_prob=0.90,
                sigmoid_inflection_min=15.0,
                sigmoid_steepness=0.130,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=30.0,
                per_patient_reward_weight=1.5,
                paediatric_curve_variant="status_epilepticus",
            ),
            SurvivalParameters(
                condition_key="febrile_seizure",
                display_name="Febrile Seizure — Prolonged",
                icd10_codes=("R56.00",),
                evidence_source="AAP Febrile Seizure Guidelines 2011",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.97,
                plateau_end_min=10.0,
                plateau_decay_lambda=0.0120,
                survival_floor=0.25,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.PAEDIATRIC_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=20.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="sickle_cell_crisis",
                display_name="Sickle Cell Crisis — Acute Chest Syndrome",
                icd10_codes=("D57.01",),
                evidence_source="ASH Sickle Cell Guidelines 2020",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.88,
                decay_lambda=0.0100,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.4,
                paediatric_curve_variant="sickle_cell_crisis",
            ),
        ]
    @classmethod
    def _abdominal(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="pancreatitis_severe",
                display_name="Severe Acute Pancreatitis",
                icd10_codes=("K85.1",),
                evidence_source="IAP/APA Guidelines 2013",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.84,
                decay_lambda=0.0070,
                survival_floor=0.15,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.2,
                elderly_penalty_factor=1.4,
            ),
            SurvivalParameters(
                condition_key="perforated_viscus",
                display_name="Perforated Viscus — Peritonitis",
                icd10_codes=("K63.1",),
                evidence_source="Peptic Ulcer Perforation Outcomes; CAPES study 2020",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.84,
                decay_lambda=0.0075,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.2,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="bowel_obstruction",
                display_name="Bowel Obstruction with Perforation Signs",
                icd10_codes=("K56.6",),
                evidence_source="ACS 2020",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.85,
                plateau_end_min=45.0,
                plateau_decay_lambda=0.0080,
                survival_floor=0.12,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=90.0,
                per_patient_reward_weight=1.2,
                elderly_penalty_factor=1.5,
            ),
            SurvivalParameters(
                condition_key="appendicitis",
                display_name="Appendicitis — Pre-Perforation",
                icd10_codes=("K37",),
                evidence_source="CODA trial 2020 — 6–12h perforation window",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.98,
                plateau_end_min=360.0,
                plateau_decay_lambda=0.0015,
                survival_floor=0.60,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=720.0,
                per_patient_reward_weight=0.6,
            ),
            SurvivalParameters(
                condition_key="intussusception",
                display_name="Intussusception — Paediatric",
                icd10_codes=("K56.1",),
                evidence_source="Gilbertson AAP 2007 — reduction window 6h",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.94,
                step_times_min=(30.0, 60.0, 120.0),
                step_penalties=(0.95, 0.88, 0.78),
                survival_floor=0.20,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("paediatric_emergency",),
                time_to_irreversible_min=120.0,
                per_patient_reward_weight=1.3,
            ),
            SurvivalParameters(
                condition_key="ovarian_torsion",
                display_name="Ovarian Torsion",
                icd10_codes=("N83.5",),
                evidence_source="Sasaki JMIG 2018 — 6h organ viability window",
                curve_model=CurveModelType.STEPPED,
                initial_survival_prob=0.96,
                step_times_min=(60.0, 120.0, 240.0),
                step_penalties=(0.92, 0.82, 0.65),   
                survival_floor=0.40,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=("obstetrics",),
                time_to_irreversible_min=360.0,
                per_patient_reward_weight=1.0,
            ),
            SurvivalParameters(
                condition_key="renal_colic",
                display_name="Renal Colic — Uncomplicated",
                icd10_codes=("N23",),
                evidence_source="N/A — no mortality risk if afebrile",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.90,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.3,
            ),
        ]
    @classmethod
    def _psychiatric(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="psychiatric_crisis",
                display_name="Psychiatric Emergency — Suicidal Ideation",
                icd10_codes=("F43.2",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.96,
                survival_floor=0.65,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("ALS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.6,
            ),
            SurvivalParameters(
                condition_key="self_harm",
                display_name="Self-Harm — Superficial Lacerations",
                icd10_codes=("X78.1",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.98,
                survival_floor=0.85,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.5,
            ),
            SurvivalParameters(
                condition_key="psychiatric_wandering",
                display_name="Dementia Patient — Wandering",
                icd10_codes=("F03",),
                evidence_source="N/A",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.90,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=("trauma_centre",),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.3,
            ),
        ]
    @classmethod
    def _defaults(cls) -> List[SurvivalParameters]:
        return [
            SurvivalParameters(
                condition_key="_default_p1_critical",
                display_name="Default P1 Critical (Fallback)",
                icd10_codes=(),
                evidence_source="EMERGI-ENV default parameterisation",
                curve_model=CurveModelType.EXPONENTIAL,
                initial_survival_prob=0.82,
                decay_lambda=0.0100,
                survival_floor=0.08,
                severity_tier=SeverityTier.P1,
                transport_urgency=TransportUrgencyClass.LOAD_AND_GO,
                golden_window=GoldenHourWindows.STANDARD_CRITICAL,
                correct_unit_types=("ALS",),
                required_specialties=(),
                time_to_irreversible_min=60.0,
                per_patient_reward_weight=1.0,
            ),
            SurvivalParameters(
                condition_key="_default_p2_urgent",
                display_name="Default P2 Urgent (Fallback)",
                icd10_codes=(),
                evidence_source="EMERGI-ENV default",
                curve_model=CurveModelType.PLATEAU_THEN_DECAY,
                initial_survival_prob=0.95,
                plateau_end_min=90.0,
                plateau_decay_lambda=0.0040,
                survival_floor=0.35,
                severity_tier=SeverityTier.P2,
                transport_urgency=TransportUrgencyClass.TIME_CRITICAL,
                golden_window=GoldenHourWindows.STANDARD_URGENT,
                correct_unit_types=("BLS",),
                required_specialties=(),
                time_to_irreversible_min=180.0,
                per_patient_reward_weight=0.6,
            ),
            SurvivalParameters(
                condition_key="_default_p3_minor",
                display_name="Default P3 Minor (Fallback)",
                icd10_codes=(),
                evidence_source="EMERGI-ENV default",
                curve_model=CurveModelType.FLAT,
                initial_survival_prob=0.99,
                survival_floor=0.90,
                severity_tier=SeverityTier.P3,
                transport_urgency=TransportUrgencyClass.ELECTIVE,
                golden_window=None,
                correct_unit_types=("BLS",),
                required_specialties=(),
                time_to_irreversible_min=999.0,
                per_patient_reward_weight=0.2,
            ),
        ]
    @classmethod
    def _build_registry(cls) -> None:
        all_params: List[SurvivalParameters] = []
        for builder in [
            cls._cardiac,
            cls._stroke,
            cls._trauma,
            cls._burns,
            cls._respiratory,
            cls._obstetric,
            cls._sepsis,
            cls._toxicology,
            cls._drowning,
            cls._anaphylaxis,
            cls._metabolic,
            cls._paediatric,
            cls._abdominal,
            cls._psychiatric,
            cls._defaults,
        ]:
            all_params.extend(builder())
        cls._REGISTRY = {p.condition_key: p for p in all_params}
        cls._built = True
        logger.info(
            "SurvivalCurveRegistry built — %d conditions registered.",
            len(cls._REGISTRY),
        )
    @classmethod
    def get(cls, condition_key: str) -> SurvivalParameters:
        if not cls._built:
            cls._build_registry()
        if condition_key in cls._REGISTRY:
            return cls._REGISTRY[condition_key]
        for key in cls._REGISTRY:
            if condition_key.startswith(key) or key.startswith(condition_key):
                logger.debug("SurvivalCurveRegistry: fuzzy match '%s' → '%s'", condition_key, key)
                return cls._REGISTRY[key]
        lk = condition_key.lower()
        if any(w in lk for w in ("arrest", "shock", "severe", "major", "rupture", "blast")):
            logger.warning("SurvivalCurveRegistry: unknown '%s', using P1 default", condition_key)
            return cls._REGISTRY["_default_p1_critical"]
        if any(w in lk for w in ("minor", "stable", "walking", "fracture")):
            return cls._REGISTRY["_default_p3_minor"]
        logger.warning("SurvivalCurveRegistry: unknown '%s', using P2 default", condition_key)
        return cls._REGISTRY["_default_p2_urgent"]
    @classmethod
    def all_keys(cls) -> List[str]:
        if not cls._built:
            cls._build_registry()
        return sorted(cls._REGISTRY.keys())
    @classmethod
    def keys_by_severity(cls, tier: SeverityTier) -> List[str]:
        if not cls._built:
            cls._build_registry()
        return [k for k, p in cls._REGISTRY.items() if p.severity_tier == tier]
    @classmethod
    def keys_by_model(cls, model: CurveModelType) -> List[str]:
        if not cls._built:
            cls._build_registry()
        return [k for k, p in cls._REGISTRY.items() if p.curve_model == model]
    @classmethod
    def registry_stats(cls) -> Dict[str, Any]:
        if not cls._built:
            cls._build_registry()
        from collections import Counter
        tiers = Counter(p.severity_tier for p in cls._REGISTRY.values())
        models = Counter(p.curve_model for p in cls._REGISTRY.values())
        return {
            "total": len(cls._REGISTRY),
            "by_severity": dict(tiers),
            "by_model": dict(models),
        }
class SurvivalCurveEngine:
    def __init__(self, rng_seed: int = 42) -> None:
        self._calculator = SurvivalProbabilityCalculator()
        self._registry = SurvivalCurveRegistry
        self._patients: Dict[str, PatientSurvivalState] = {}
        logger.info("SurvivalCurveEngine initialised (seed=%d)", rng_seed)
    def register_patient(
        self,
        patient_id: str,
        incident_id: str,
        condition_key: str,
        severity: str = "P1",
        age_years: Optional[int] = None,
        step_registered: int = 0,
        time_registered_min: float = 0.0,
    ) -> PatientSurvivalState:
        params = self._registry.get(condition_key)
        tier = SeverityTier(severity) if severity in {t.value for t in SeverityTier} \
               else SeverityTier.P1
        state = PatientSurvivalState(
            patient_id=patient_id,
            incident_id=incident_id,
            condition_key=condition_key,
            severity=tier,
            params=params,
            age_years=age_years,
            step_registered=step_registered,
            time_registered_min=time_registered_min,
        )
        self._patients[patient_id] = state
        logger.debug(
            "SurvivalCurveEngine: registered %s | condition=%s | P(0)=%.3f",
            patient_id, condition_key, params.initial_survival_prob,
        )
        return state
    def update_step(
        self,
        patient_id: str,
        step: int,
        dispatch_delay_additional_min: float = 0.0,
    ) -> float:
        state = self._patients.get(patient_id)
        if state is None:
            logger.warning("SurvivalCurveEngine.update_step: unknown patient %s", patient_id)
            return 0.5
        elapsed_since_reg = (step - state.step_registered) * SIMULATION_STEP_DURATION_MIN
        state.elapsed_minutes = state.time_registered_min + elapsed_since_reg
        state.current_step = step
        unit_penalty = self._get_unit_penalty(state)
        prob = self._calculator.compute(
            params=state.params,
            elapsed_min=state.elapsed_minutes,
            age_years=state.age_years,
            dispatch_delay_min=dispatch_delay_additional_min,
            under_triage_penalty=unit_penalty,
        )
        state.current_survival_prob = prob
        if state.survival_history:
            prev_t, prev_p = state.survival_history[-1]
            dt = state.elapsed_minutes - prev_t
            state.survival_integral += 0.5 * (prev_p + prob) * dt
        state.survival_history.append((state.elapsed_minutes, prob))
        state.step_history.append((step, state.elapsed_minutes, prob))
        gw = state.params.golden_window
        if gw is not None and state.golden_window_compliance is None:
            if not gw.is_open(state.elapsed_minutes):
                state.golden_window_compliance = (
                    state.dispatched_unit is not None
                    and state.elapsed_minutes <= gw.closes_at_min
                )
        return prob
    def update_all(self, step: int) -> Dict[str, float]:
        return {pid: self.update_step(pid, step) for pid in self._patients}
    def compute_survival(
        self,
        patient_id: str,
        elapsed_min: Optional[float] = None,
    ) -> float:
        state = self._patients.get(patient_id)
        if state is None:
            return 0.5
        t = elapsed_min if elapsed_min is not None else state.elapsed_minutes
        return self._calculator.compute(
            params=state.params,
            elapsed_min=t,
            age_years=state.age_years,
        )
    def step_survival_integral(
        self,
        patient_id: str,
        from_min: float,
        to_min: float,
    ) -> float:
        state = self._patients.get(patient_id)
        if state is None:
            return 0.0
        return self._calculator.integrate(
            params=state.params,
            from_min=from_min,
            to_min=to_min,
            age_years=state.age_years,
        )
    def snapshot(self, patient_id: str) -> SurvivalSnapshot:
        state = self._patients.get(patient_id)
        if state is None:
            return SurvivalSnapshot(
                patient_id=patient_id, condition_key="unknown",
                elapsed_min=0, survival_probability=0.5,
                survival_integral=0, golden_window_open=False,
                golden_window_compliance=0, time_to_irreversible=999,
                reward_score=0.5, clinical_notes="patient not registered",
            )
        gw = state.params.golden_window
        gw_compliance = (
            gw.compliance_fraction_remaining(state.elapsed_minutes)
            if gw else 1.0
        )
        time_to_irrev = max(
            0.0,
            state.params.time_to_irreversible_min - state.elapsed_minutes
        )
        return SurvivalSnapshot(
            patient_id=patient_id,
            condition_key=state.condition_key,
            elapsed_min=state.elapsed_minutes,
            survival_probability=state.current_survival_prob,
            survival_integral=state.survival_integral,
            golden_window_open=state.is_in_golden_window,
            golden_window_compliance=gw_compliance,
            time_to_irreversible=time_to_irrev,
            reward_score=state.total_reward_contribution,
            clinical_notes=self._generate_clinical_notes(state),
        )
    def episode_reward_contribution(
        self,
        patient_id: str,
        normalise: bool = True,
    ) -> float:
        state = self._patients.get(patient_id)
        if state is None:
            return 0.0
        base = state.total_reward_contribution
        gw = state.params.golden_window
        if gw is not None and state.golden_window_compliance is True:
            base = min(1.0, base + GOLDEN_HOUR_COMPLIANCE_BONUS)
        if normalise:
            return min(1.0, max(0.0, base))
        return base * state.params.per_patient_reward_weight
    def mark_dispatched(
        self,
        patient_id: str,
        unit_type: str,
        step: int,
    ) -> None:
        state = self._patients.get(patient_id)
        if state is None:
            return
        state.dispatched_unit = unit_type.upper()
        correct = state.params.correct_unit_types
        state.unit_type_correct = (unit_type.upper() in correct)
        logger.debug(
            "SurvivalCurveEngine: %s dispatched %s — correct=%s",
            patient_id, unit_type, state.unit_type_correct,
        )
    def mark_treated(
        self,
        patient_id: str,
        hospital_id: str,
        step: int,
    ) -> None:
        state = self._patients.get(patient_id)
        if state is None:
            return
        state.treated = True
        state.treatment_time_min = state.elapsed_minutes
        state.outcome = "survived"
        gw = state.params.golden_window
        if gw is not None:
            state.golden_window_compliance = gw.is_open(state.elapsed_minutes)
        logger.debug(
            "SurvivalCurveEngine: %s treated at %s at %.1f min (P=%.3f)",
            patient_id, hospital_id, state.elapsed_minutes,
            state.current_survival_prob,
        )
    def mark_deceased(self, patient_id: str) -> None:
        state = self._patients.get(patient_id)
        if state:
            state.outcome = "deceased"
            state.current_survival_prob = SURVIVAL_FLOOR
    def get_all_snapshots(self) -> Dict[str, SurvivalSnapshot]:
        return {pid: self.snapshot(pid) for pid in self._patients}
    def critical_patients(self) -> List[str]:
        return [
            pid for pid, s in self._patients.items()
            if s.severity == SeverityTier.P1 and not s.treated
        ]
    def patients_near_irreversible(
        self,
        within_steps: int = 2,
    ) -> List[Tuple[str, float]]:
        window = within_steps * SIMULATION_STEP_DURATION_MIN
        result = []
        for pid, state in self._patients.items():
            if state.treated:
                continue
            remaining = state.params.time_to_irreversible_min - state.elapsed_minutes
            if 0 < remaining <= window:
                result.append((pid, remaining))
        return sorted(result, key=lambda x: x[1])
    def total_episode_weighted_reward(self) -> float:
        total_weight = sum(
            s.params.per_patient_reward_weight for s in self._patients.values()
        )
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            self.episode_reward_contribution(pid, normalise=True)
            * self._patients[pid].params.per_patient_reward_weight
            for pid in self._patients
        )
        return min(1.0, weighted_sum / total_weight)
    def survival_rate_by_severity(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {
            t.value: {"count": 0, "treated": 0, "avg_prob": 0.0, "total_integral": 0.0}
            for t in SeverityTier
        }
        for state in self._patients.values():
            tier = state.severity.value
            stats[tier]["count"] += 1
            if state.treated:
                stats[tier]["treated"] += 1
            stats[tier]["avg_prob"] += state.current_survival_prob
            stats[tier]["total_integral"] += state.survival_integral
        for tier in stats:
            n = stats[tier]["count"]
            if n > 0:
                stats[tier]["avg_prob"] /= n
        return stats
    def per_minute_loss_at_current(self, patient_id: str) -> float:
        state = self._patients.get(patient_id)
        if state is None:
            return 0.0
        return self._calculator.compute_per_minute_loss(
            params=state.params,
            at_min=state.elapsed_minutes,
            age_years=state.age_years,
        )
    def time_until_critical_threshold(
        self,
        patient_id: str,
        threshold: float = 0.50,
    ) -> Optional[float]:
        state = self._patients.get(patient_id)
        if state is None:
            return None
        absolute_t = self._calculator.minutes_until_threshold(
            params=state.params,
            threshold=threshold,
            age_years=state.age_years,
        )
        if absolute_t is None:
            return None
        return max(0.0, absolute_t - state.elapsed_minutes)
    def reset(self) -> None:
        self._patients.clear()
        logger.debug("SurvivalCurveEngine.reset()")
    def get_state(self, patient_id: str) -> Optional[PatientSurvivalState]:
        return self._patients.get(patient_id)
    def patient_ids(self) -> List[str]:
        return list(self._patients.keys())
    def _get_unit_penalty(self, state: PatientSurvivalState) -> float:
        if state.dispatched_unit is None:
            return 1.0
        correct = state.params.correct_unit_types
        dispatched = state.dispatched_unit
        if dispatched in correct:
            return 1.0
        unit_ranks = {"BLS": 1, "ALS": 2, "MICU": 3}
        d_rank = unit_ranks.get(dispatched, 1)
        c_ranks = [unit_ranks.get(u, 1) for u in correct]
        min_correct = min(c_ranks)
        max_correct = max(c_ranks)
        if d_rank < min_correct:
            gap = min_correct - d_rank
            if min_correct == 3 and d_rank == 1:
                return CRITICAL_UNDER_TRIAGE_PENALTY
            return UNDER_TRIAGE_SURVIVAL_PENALTY ** gap
        return 1.0
    @staticmethod
    def _generate_clinical_notes(state: PatientSurvivalState) -> str:
        gw = state.params.golden_window
        gw_str = "N/A"
        if gw:
            remaining = max(0.0, gw.closes_at_min - state.elapsed_minutes)
            gw_str = f"{remaining:.1f} min remaining in {gw.window_type.value}"
        urgency_str = state.params.transport_urgency.value
        return (
            f"{state.params.display_name} | "
            f"P(t={state.elapsed_minutes:.1f})={state.current_survival_prob:.3f} | "
            f"Window: {gw_str} | "
            f"Urgency: {urgency_str} | "
            f"Correct units: {','.join(state.params.correct_unit_types)}"
        )
def _self_test() -> None:
    registry = SurvivalCurveRegistry
    registry._build_registry()
    stats = registry.registry_stats()
    assert stats["total"] >= 100, f"Expected ≥100 conditions, got {stats['total']}"
    stemi = registry.get("stemi_anterior")
    assert stemi.curve_model == CurveModelType.BIPHASIC
    p0  = SurvivalProbabilityCalculator.compute(stemi, 0.0)
    p30 = SurvivalProbabilityCalculator.compute(stemi, 30.0)
    p90 = SurvivalProbabilityCalculator.compute(stemi, 90.0)
    assert p0 > p30 > p90, f"STEMI curve not monotonically decreasing: {p0:.3f} {p30:.3f} {p90:.3f}"
    assert p0 == stemi.initial_survival_prob, f"P(0) mismatch: {p0} vs {stemi.initial_survival_prob}"
    assert p90 >= stemi.survival_floor
    vf = registry.get("cardiac_arrest_vf")
    p0_vf = SurvivalProbabilityCalculator.compute(vf, 0.0)
    p10_vf = SurvivalProbabilityCalculator.compute(vf, 10.0)
    assert p0_vf - p10_vf >= 0.24, f"VF arrest curve too flat: lost only {p0_vf - p10_vf:.3f} in 10 min"
    hs = registry.get("heat_stroke")
    assert hs.curve_model == CurveModelType.STEPPED
    p_before = SurvivalProbabilityCalculator.compute(hs, 14.0)
    p_after  = SurvivalProbabilityCalculator.compute(hs, 16.0)
    assert p_after < p_before, "Heat stroke stepped model not decreasing at step boundary"
    engine = SurvivalCurveEngine(rng_seed=42)
    state = engine.register_patient("P001", "INC001", "stemi_anterior", "P1", age_years=60)
    assert state.current_survival_prob == stemi.initial_survival_prob
    engine.mark_dispatched("P001", "MICU", step=0)
    p_s1 = engine.update_step("P001", step=1)
    p_s3 = engine.update_step("P001", step=3)
    assert p_s1 >= p_s3, "Survival should not increase without treatment"
    snap = engine.snapshot("P001")
    assert 0.0 <= snap.survival_probability <= 1.0
    assert snap.golden_window_open   
    engine.mark_treated("P001", "H01", step=3)
    reward = engine.episode_reward_contribution("P001")
    assert 0.0 <= reward <= 1.0, f"Reward out of bounds: {reward}"
    engine2 = SurvivalCurveEngine()
    engine2.register_patient("P002", "INC001", "stemi_anterior", "P1", age_years=58)
    engine2.mark_dispatched("P002", "BLS", step=0)   
    engine2.update_step("P002", step=1)
    state2 = engine2.get_state("P002")
    engine3 = SurvivalCurveEngine()
    engine3.register_patient("P003", "INC001", "stemi_anterior", "P1", age_years=58)
    engine3.mark_dispatched("P003", "MICU", step=0)  
    engine3.update_step("P003", step=1)
    state3 = engine3.get_state("P003")
    assert state2.current_survival_prob < state3.current_survival_prob, \
        "Wrong unit type should reduce survival probability"
    integral = SurvivalProbabilityCalculator.integrate(stemi, 0.0, 30.0)
    assert 0 < integral < 30.0, f"Integral out of expected range: {integral}"
    gw = GoldenHourWindows.STEMI
    assert gw.is_open(60.0)
    assert not gw.is_open(100.0)
    assert gw.compliance_fraction_remaining(45.0) == pytest_approx(0.5, abs_tol=0.1)
    unk = registry.get("unknown_condition_xyz")
    assert unk is not None
    engine.reset()
    for i in range(5):
        engine.register_patient(f"P{i:03d}", "INC001", "stemi_anterior", "P1")
        engine.mark_dispatched(f"P{i:03d}", "MICU", step=0)
        for s in range(3):
            engine.update_step(f"P{i:03d}", s)
        engine.mark_treated(f"P{i:03d}", "H01", step=3)
    total_reward = engine.total_episode_weighted_reward()
    assert 0.0 <= total_reward <= 1.0
    logger.info(
        "survival_curves.py self-test PASSED — %d conditions, all assertions verified.",
        stats["total"],
    )
def pytest_approx(val: float, abs_tol: float = 0.05) -> "_ApproxHelper":
    class _ApproxHelper:
        def __init__(self, v: float, a: float) -> None:
            self.v, self.a = v, a
        def __eq__(self, other: float) -> bool:
            return abs(other - self.v) <= self.a
    return _ApproxHelper(val, abs_tol)
SurvivalCurveRegistry._build_registry()
_self_test()
logger.info(
    "EMERGI-ENV server.medical.survival_curves v%d loaded — "
    "%d conditions, 6 curve models, %d golden windows, "
    "SurvivalCurveEngine ready.",
    SURVIVAL_CURVES_VERSION,
    len(SurvivalCurveRegistry._REGISTRY),
    len([gw for gw in dir(GoldenHourWindows) if not gw.startswith("_")]),
)
__all__ = [
    "CurveModelType",
    "SeverityTier",
    "TransportUrgencyClass",
    "GoldenWindowType",
    "GoldenHourWindow",
    "GoldenHourWindows",
    "SurvivalParameters",
    "PatientSurvivalState",
    "SurvivalSnapshot",
    "SurvivalProbabilityCalculator",
    "SurvivalCurveRegistry",
    "SurvivalCurveEngine",
    "SURVIVAL_CURVES_VERSION",
    "SIMULATION_STEP_DURATION_MIN",
    "SURVIVAL_FLOOR",
    "UNDER_TRIAGE_SURVIVAL_PENALTY",
    "CRITICAL_UNDER_TRIAGE_PENALTY",
    "GOLDEN_HOUR_COMPLIANCE_BONUS",
]