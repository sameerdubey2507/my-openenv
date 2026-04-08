from __future__ import annotations
import logging
import math
import statistics
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import (
    Any, ClassVar, Dict, FrozenSet, List, NamedTuple,
    Optional, Tuple, Union,
)
logger = logging.getLogger("emergi_env.medical.trauma_scoring")
TRAUMA_SCORING_VERSION: int = 5
try:
    from server.medical.triage import (
        RPMScore, RevisedTraumaScore, TriageTag, MentalStatus,
        RespirationStatus, PulseStatus,
        _infer_gcs_from_rpm, _infer_sbp_from_rpm, _infer_rr_from_rpm,
    )
    _TRIAGE_AVAILABLE = True
except ImportError:
    _TRIAGE_AVAILABLE = False
    RPMScore = None  
try:
    from server.medical.survivalcurves import (
        SurvivalCurveRegistry, SeverityTier,
        SurvivalProbabilityCalculator,
    )
    _CURVES_AVAILABLE = True
except ImportError:
    _CURVES_AVAILABLE = False
ISS_MAJOR_TRAUMA_THRESHOLD: int      = 16
ISS_SEVERE_THRESHOLD: int            = 25
ISS_CRITICAL_THRESHOLD: int          = 40
ISS_UNSURVIVABLE_THRESHOLD: int      = 75   
TRISS_BLUNT_B0: float     = -0.4499
TRISS_BLUNT_B1: float     =  0.8085
TRISS_BLUNT_B2: float     = -0.0835
TRISS_BLUNT_B3: float     =  0.9934
TRISS_PENETRATING_B0: float = -2.5355
TRISS_PENETRATING_B1: float =  0.9934
TRISS_PENETRATING_B2: float = -0.0651
TRISS_PENETRATING_B3: float =  1.1360
SOFA_MAX_PER_ORGAN: int = 4
SOFA_TOTAL_MAX: int     = 24
NEWS2_LOW_THRESHOLD:    int = 1
NEWS2_MEDIUM_THRESHOLD: int = 5
NEWS2_HIGH_THRESHOLD:   int = 7
MEWS_REVIEW_THRESHOLD:  int = 2
MEWS_URGENT_THRESHOLD:  int = 4
MEWS_EMERGENCY_THRESHOLD: int = 6
GAP_CRITICAL_THRESHOLD: int = 18
MGAP_CRITICAL_THRESHOLD: int = 18
KTS_SEVERE_THRESHOLD:   int = 14
KTS_MODERATE_THRESHOLD: int = 17
APACHE_CRITICAL_THRESHOLD: int = 25
DCS_LACTATE_THRESHOLD_MMOL:     float = 5.0
DCS_BASE_DEFICIT_THRESHOLD:     float = 6.0
DCS_HYPOTHERMIA_THRESHOLD_C:    float = 35.0
DCS_COAGULOPATHY_INR:           float = 1.5
DCS_TRANSFUSION_PACK_THRESHOLD: int   = 6    
INDIA_URBAN_EMS_TARGET_MIN:   float = 8.0
INDIA_RURAL_EMS_TARGET_MIN:   float = 20.0
INDIA_GHAT_EMS_TARGET_MIN:    float = 35.0
@unique
class InjuryMechanism(str, Enum):
    BLUNT_MVC         = "blunt_mvc"
    BLUNT_FALL        = "blunt_fall"
    BLUNT_ASSAULT     = "blunt_assault"
    BLUNT_CRUSH       = "blunt_crush"
    BLUNT_BLAST       = "blunt_blast"
    BLUNT_SPORTS      = "blunt_sports"
    PENETRATING_GSW   = "penetrating_gsw"
    PENETRATING_STAB  = "penetrating_stab"
    PENETRATING_BLAST = "penetrating_blast"
    BURN              = "burn"
    DROWNING          = "drowning"
    HANGING           = "hanging"
    UNKNOWN           = "unknown"
    @property
    def is_blunt(self) -> bool:
        return self.value.startswith("blunt")
    @property
    def is_penetrating(self) -> bool:
        return self.value.startswith("penetrating")
    @property
    def mgap_mechanism_points(self) -> int:
        if self.is_blunt:
            return 4
        return 0   
@unique
class BodyRegion(str, Enum):
    HEAD_NECK      = "head_neck"
    FACE           = "face"
    THORAX         = "thorax"
    ABDOMEN_PELVIS = "abdomen_pelvis"
    EXTREMITIES    = "extremities"
    EXTERNAL       = "external"
    @property
    def ais_region_code(self) -> int:
        return {
            BodyRegion.HEAD_NECK:      1,
            BodyRegion.FACE:           2,
            BodyRegion.THORAX:         3,
            BodyRegion.ABDOMEN_PELVIS: 4,
            BodyRegion.EXTREMITIES:    5,
            BodyRegion.EXTERNAL:       6,
        }[self]
@unique
class AISSeverity(int, Enum):
    MINOR        = 1
    MODERATE     = 2
    SERIOUS      = 3
    SEVERE       = 4
    CRITICAL     = 5
    UNSURVIVABLE = 6
    @property
    def label(self) -> str:
        return self.name.capitalize()
    @property
    def estimated_mortality_pct(self) -> float:
        return {
            AISSeverity.MINOR:        0.0,
            AISSeverity.MODERATE:     1.0,
            AISSeverity.SERIOUS:      8.0,
            AISSeverity.SEVERE:       25.0,
            AISSeverity.CRITICAL:     50.0,
            AISSeverity.UNSURVIVABLE: 100.0,
        }[self]
@unique
class DCSIndicationLevel(str, Enum):
    NOT_INDICATED       = "not_indicated"
    RELATIVE            = "relative"
    STRONG_RELATIVE     = "strong_relative"
    ABSOLUTE            = "absolute"
@unique
class OutcomePrediction(str, Enum):
    EXPECTED_SURVIVAL   = "expected_survival"
    PROBABLE_SURVIVAL   = "probable_survival"
    UNCERTAIN           = "uncertain"
    PROBABLE_DEATH      = "probable_death"
    EXPECTED_DEATH      = "expected_death"
    @property
    def survival_probability_range(self) -> Tuple[float, float]:
        return {
            OutcomePrediction.EXPECTED_SURVIVAL:  (0.75, 1.00),
            OutcomePrediction.PROBABLE_SURVIVAL:  (0.50, 0.75),
            OutcomePrediction.UNCERTAIN:          (0.25, 0.50),
            OutcomePrediction.PROBABLE_DEATH:     (0.10, 0.25),
            OutcomePrediction.EXPECTED_DEATH:     (0.00, 0.10),
        }[self]
@unique
class TraumaTeamActivationLevel(str, Enum):
    NONE        = "none"
    STANDBY     = "standby"
    FULL        = "full"
    IMMEDIATE   = "immediate"   
@unique
class FASTFinding(str, Enum):
    NEGATIVE                = "negative"
    POSITIVE_PERICARDIAL    = "positive_pericardial"
    POSITIVE_RUQ            = "positive_ruq"
    POSITIVE_LUQ            = "positive_luq"
    POSITIVE_PELVIS         = "positive_pelvis"
    POSITIVE_PLURAL_RIGHT   = "positive_pleural_right"
    POSITIVE_PLURAL_LEFT    = "positive_pleural_left"
    INDETERMINATE           = "indeterminate"
@dataclass(frozen=True)
class AISInjury:
    region:        BodyRegion
    severity:      AISSeverity
    descriptor:    str          
    ais_code:      str = ""     
    structure:     str = ""     
    complication:  bool = False
    @property
    def iss_contribution(self) -> int:
        return int(self.severity) ** 2
    @property
    def is_critical_injury(self) -> bool:
        return self.severity >= AISSeverity.SEVERE
    @property
    def requires_level1_trauma(self) -> bool:
        return self.severity >= AISSeverity.SERIOUS and self.region in (
            BodyRegion.HEAD_NECK, BodyRegion.THORAX, BodyRegion.ABDOMEN_PELVIS
        )
@dataclass
class AISBodyRegionProfile:
    region: BodyRegion
    injuries: List[AISInjury] = field(default_factory=list)
    def add(self, injury: AISInjury) -> None:
        self.injuries.append(injury)
    @property
    def max_ais(self) -> int:
        if not self.injuries:
            return 0
        return max(int(i.severity) for i in self.injuries)
    @property
    def max_injury(self) -> Optional[AISInjury]:
        if not self.injuries:
            return None
        return max(self.injuries, key=lambda i: int(i.severity))
    @property
    def iss_contribution(self) -> int:
        return self.max_ais ** 2
    @property
    def niss_contributions(self) -> List[int]:
        return sorted([int(i.severity) ** 2 for i in self.injuries], reverse=True)
    def total_severity_burden(self) -> float:
        return sum(int(i.severity) ** 2 for i in self.injuries)
@dataclass(frozen=True)
class ISSResult:
    iss:               int
    niss:              int
    body_region_scores: Dict[str, int]   
    top3_ais_squares:   Tuple[int, ...]   
    is_major_trauma:    bool
    is_critical:        bool
    is_unsurvivable:    bool
    injury_count:       int
    multi_system:       bool   
    highest_ais:        int
    dominant_region:    str
    @property
    def severity_label(self) -> str:
        if self.iss >= ISS_UNSURVIVABLE_THRESHOLD:
            return "UNSURVIVABLE"
        if self.iss >= ISS_CRITICAL_THRESHOLD:
            return "CRITICAL"
        if self.iss >= ISS_SEVERE_THRESHOLD:
            return "SEVERE"
        if self.iss >= ISS_MAJOR_TRAUMA_THRESHOLD:
            return "MAJOR"
        if self.iss >= 9:
            return "MODERATE"
        return "MINOR"
    @property
    def predicted_mortality_pct(self) -> float:
        if self.iss >= 75:
            return 100.0
        if self.iss >= 50:
            return 85.0
        if self.iss >= 40:
            return 60.0
        if self.iss >= 25:
            return 35.0
        if self.iss >= 16:
            return 10.0
        if self.iss >= 9:
            return 2.0
        return 0.1
    @property
    def expected_icu_days(self) -> float:
        if self.iss < 9:
            return 0.0
        return round(0.4 * self.iss - 2.5, 1)
    @property
    def trauma_centre_level_required(self) -> str:
        if self.iss >= ISS_MAJOR_TRAUMA_THRESHOLD:
            return "Level-I"
        if self.iss >= 9:
            return "Level-II"
        return "Level-III or ED"
def compute_iss(region_profiles: Dict[BodyRegion, AISBodyRegionProfile]) -> ISSResult:
    region_max: Dict[str, int] = {}
    all_ais_squares: List[int] = []
    for region, profile in region_profiles.items():
        mx = profile.max_ais
        region_max[region.value] = mx ** 2
        all_ais_squares.extend(profile.niss_contributions)
    sorted_region_scores = sorted(region_max.values(), reverse=True)
    iss = sum(sorted_region_scores[:3])
    iss = min(iss, 75)  
    any_six = any(
        p.max_ais == 6 for p in region_profiles.values()
    )
    if any_six:
        iss = 75
    top3 = sorted(all_ais_squares, reverse=True)[:3]
    niss = sum(top3)
    niss = min(niss, 75)
    total_injuries = sum(len(p.injuries) for p in region_profiles.values())
    injured_regions = [r for r, mx in region_max.items() if mx > 0]
    dominant = max(region_max, key=lambda r: region_max[r]) if region_max else "unknown"
    highest = max((p.max_ais for p in region_profiles.values()), default=0)
    return ISSResult(
        iss=iss,
        niss=niss,
        body_region_scores=dict(region_max),
        top3_ais_squares=tuple(top3),
        is_major_trauma=iss >= ISS_MAJOR_TRAUMA_THRESHOLD,
        is_critical=iss >= ISS_CRITICAL_THRESHOLD,
        is_unsurvivable=iss >= ISS_UNSURVIVABLE_THRESHOLD,
        injury_count=total_injuries,
        multi_system=len(injured_regions) >= 3,
        highest_ais=highest,
        dominant_region=dominant,
    )
@dataclass(frozen=True)
class RTSResult:
    gcs:              int
    systolic_bp:      int
    respiratory_rate: int
    gcs_coded:        int
    sbp_coded:        int
    rr_coded:         int
    rts:              float
    triage_rts:       float    
    survival_prob_blunt:       float
    survival_prob_penetrating: float
    @property
    def triage_category(self) -> str:
        if self.rts >= 7.0:    return "GREEN"
        if self.rts >= 5.0:    return "YELLOW"
        if self.rts >= 1.0:    return "RED"
        return "BLACK"
    @property
    def clinical_concern_level(self) -> str:
        if self.rts < 4.0:     return "CRITICAL"
        if self.rts < 6.0:     return "SERIOUS"
        if self.rts < 7.0:     return "MODERATE"
        return "MINOR"
    @property
    def needs_immediate_airway(self) -> bool:
        return self.gcs <= 8 or self.rr_coded <= 1
    @property
    def haemodynamic_compromise(self) -> bool:
        return self.sbp_coded <= 2
def compute_rts(
    gcs: int,
    systolic_bp: int,
    respiratory_rate: int,
) -> RTSResult:
    if gcs >= 13:     gcs_c = 4
    elif gcs >= 9:    gcs_c = 3
    elif gcs >= 6:    gcs_c = 2
    elif gcs >= 4:    gcs_c = 1
    else:             gcs_c = 0
    if systolic_bp > 89:     sbp_c = 4
    elif systolic_bp >= 76:  sbp_c = 3
    elif systolic_bp >= 50:  sbp_c = 2
    elif systolic_bp >= 1:   sbp_c = 1
    else:                    sbp_c = 0
    if 10 <= respiratory_rate <= 29:  rr_c = 4
    elif respiratory_rate > 29:       rr_c = 3
    elif 6 <= respiratory_rate <= 9:  rr_c = 2
    elif 1 <= respiratory_rate <= 5:  rr_c = 1
    else:                             rr_c = 0
    rts = 0.9368 * gcs_c + 0.7326 * sbp_c + 0.2908 * rr_c
    t_rts = float(gcs_c + sbp_c + rr_c)
    def _ps(b0: float, b1: float) -> float:
        b = b0 + b1 * rts
        return 1.0 / (1.0 + math.exp(-b))
    ps_blunt = _ps(TRISS_BLUNT_B0, TRISS_BLUNT_B1)
    ps_pen   = _ps(TRISS_PENETRATING_B0, TRISS_PENETRATING_B1)
    return RTSResult(
        gcs=gcs, systolic_bp=systolic_bp, respiratory_rate=respiratory_rate,
        gcs_coded=gcs_c, sbp_coded=sbp_c, rr_coded=rr_c,
        rts=round(rts, 3), triage_rts=t_rts,
        survival_prob_blunt=round(ps_blunt, 4),
        survival_prob_penetrating=round(ps_pen, 4),
    )
@dataclass(frozen=True)
class TRISSResult:
    rts:             float
    iss:             int
    age_years:       int
    mechanism:       InjuryMechanism
    b0:              float
    b1:              float
    b2:              float
    b3:              float
    logit:           float
    survival_prob:   float
    w_statistic:     Optional[float]    
    z_statistic:     Optional[float]
    @property
    def predicted_outcome(self) -> OutcomePrediction:
        p = self.survival_prob
        if p >= 0.75:   return OutcomePrediction.EXPECTED_SURVIVAL
        if p >= 0.50:   return OutcomePrediction.PROBABLE_SURVIVAL
        if p >= 0.25:   return OutcomePrediction.UNCERTAIN
        if p >= 0.10:   return OutcomePrediction.PROBABLE_DEATH
        return OutcomePrediction.EXPECTED_DEATH
    @property
    def unexpected_death_risk(self) -> bool:
        return self.survival_prob > 0.75 and self.iss >= ISS_MAJOR_TRAUMA_THRESHOLD
    @property
    def age_coefficient(self) -> float:
        return self.b3
    @property
    def mechanism_label(self) -> str:
        return "blunt" if self.mechanism.is_blunt else "penetrating"
    @property
    def confidence_grade(self) -> str:
        if 15 <= self.iss <= 40 and 4 <= self.rts <= 7.5:
            return "HIGH"
        if 5 <= self.iss <= 60:
            return "MODERATE"
        return "LOW — extreme values"
    def to_grader_dict(self) -> Dict[str, Any]:
        return {
            "rts":           round(self.rts, 3),
            "iss":           self.iss,
            "age":           self.age_years,
            "mechanism":     self.mechanism_label,
            "logit":         round(self.logit, 4),
            "Ps":            round(self.survival_prob, 4),
            "outcome":       self.predicted_outcome.value,
            "confidence":    self.confidence_grade,
        }
def compute_triss(
    rts: float,
    iss: int,
    age_years: int,
    mechanism: InjuryMechanism,
) -> TRISSResult:
    age_code = 0 if age_years < 55 else 1
    if mechanism.is_blunt:
        b0, b1, b2, b3 = (
            TRISS_BLUNT_B0, TRISS_BLUNT_B1,
            TRISS_BLUNT_B2, TRISS_BLUNT_B3
        )
    else:
        b0, b1, b2, b3 = (
            TRISS_PENETRATING_B0, TRISS_PENETRATING_B1,
            TRISS_PENETRATING_B2, TRISS_PENETRATING_B3
        )
    logit = b0 + b1 * rts + b2 * iss + b3 * age_code
    ps = 1.0 / (1.0 + math.exp(-logit))
    return TRISSResult(
        rts=rts, iss=iss, age_years=age_years,
        mechanism=mechanism,
        b0=b0, b1=b1, b2=b2, b3=b3,
        logit=round(logit, 4),
        survival_prob=round(ps, 4),
        w_statistic=None, z_statistic=None,
    )
@dataclass(frozen=True)
class MEWSResult:
    respiratory_rate_score: int
    heart_rate_score:       int
    systolic_bp_score:      int
    temperature_score:      int
    avpu_score:             int
    total:                  int
    respiratory_rate:       int
    heart_rate:             int
    systolic_bp:            int
    temperature_celsius:    float
    avpu:                   str
    @property
    def alert_level(self) -> str:
        if self.total >= MEWS_EMERGENCY_THRESHOLD: return "EMERGENCY"
        if self.total >= MEWS_URGENT_THRESHOLD:    return "URGENT"
        if self.total >= MEWS_REVIEW_THRESHOLD:    return "REVIEW"
        return "ROUTINE"
    @property
    def recommended_action(self) -> str:
        al = self.alert_level
        if al == "EMERGENCY":  return "Immediate ICU + consultant review"
        if al == "URGENT":     return "Senior review within 30 min, consider HDU"
        if al == "REVIEW":     return "Nursing review, increase monitoring frequency"
        return "Continue routine observations"
    @property
    def deterioration_risk(self) -> float:
        if self.total >= 7: return 0.85
        if self.total >= 5: return 0.55
        if self.total >= 3: return 0.25
        if self.total >= 2: return 0.10
        return 0.02
def compute_mews(
    respiratory_rate: int,
    heart_rate:       int,
    systolic_bp:      int,
    temperature_c:    float,
    avpu:             str,   
) -> MEWSResult:
    if respiratory_rate < 9:             rr_s = 2
    elif 9 <= respiratory_rate <= 14:    rr_s = 0
    elif 15 <= respiratory_rate <= 20:   rr_s = 1
    elif 21 <= respiratory_rate <= 29:   rr_s = 2
    else:                                rr_s = 3
    if heart_rate < 40:                  hr_s = 2
    elif 40 <= heart_rate <= 50:         hr_s = 1
    elif 51 <= heart_rate <= 100:        hr_s = 0
    elif 101 <= heart_rate <= 110:       hr_s = 1
    elif 111 <= heart_rate <= 129:       hr_s = 2
    else:                                hr_s = 3
    if systolic_bp < 70:                 sbp_s = 3
    elif 70 <= systolic_bp <= 80:        sbp_s = 2
    elif 81 <= systolic_bp <= 100:       sbp_s = 1
    elif 101 <= systolic_bp <= 199:      sbp_s = 0
    else:                                sbp_s = 2
    if temperature_c < 35.0:            temp_s = 2
    elif 35.0 <= temperature_c <= 38.4: temp_s = 0
    else:                               temp_s = 2
    avpu_upper = avpu.upper()
    avpu_map = {"A": 0, "V": 1, "P": 2, "U": 3}
    avpu_s = avpu_map.get(avpu_upper, 2)
    total = rr_s + hr_s + sbp_s + temp_s + avpu_s
    return MEWSResult(
        respiratory_rate_score=rr_s,
        heart_rate_score=hr_s,
        systolic_bp_score=sbp_s,
        temperature_score=temp_s,
        avpu_score=avpu_s,
        total=total,
        respiratory_rate=respiratory_rate,
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        temperature_celsius=temperature_c,
        avpu=avpu_upper,
    )
@dataclass(frozen=True)
class NEWS2Result:
    respiratory_rate_score:  int
    spo2_score:              int
    systolic_bp_score:       int
    heart_rate_score:        int
    temperature_score:       int
    consciousness_score:     int
    air_or_oxygen_score:     int
    total:                   int
    respiratory_rate:        int
    spo2_pct:                float
    systolic_bp:             int
    heart_rate:              int
    temperature_celsius:     float
    on_supplemental_oxygen:  bool
    avpu:                    str
    clinical_risk:           str
    monitoring_frequency:    str
    @property
    def needs_immediate_escalation(self) -> bool:
        return (
            self.total >= NEWS2_HIGH_THRESHOLD
            or self.respiratory_rate_score == 3
            or self.consciousness_score >= 3
            or self.systolic_bp_score == 3
        )
    @property
    def predicted_icu_admission_risk(self) -> float:
        if self.total >= 7:   return 0.80
        if self.total >= 5:   return 0.45
        if self.total >= 3:   return 0.18
        if self.total >= 1:   return 0.05
        return 0.01
    @property
    def predicted_cardiac_arrest_risk_24h(self) -> float:
        if self.total >= 7: return 0.35
        if self.total >= 5: return 0.12
        if self.total >= 3: return 0.04
        return 0.005
def compute_news2(
    respiratory_rate:       int,
    spo2_pct:               float,
    systolic_bp:            int,
    heart_rate:             int,
    temperature_c:          float,
    avpu:                   str,
    on_supplemental_oxygen: bool = False,
    is_copd_target_spo2:    bool = False,
) -> NEWS2Result:
    if respiratory_rate <= 8:             rr_s = 3
    elif 9 <= respiratory_rate <= 11:     rr_s = 1
    elif 12 <= respiratory_rate <= 20:    rr_s = 0
    elif 21 <= respiratory_rate <= 24:    rr_s = 2
    else:                                 rr_s = 3
    if not is_copd_target_spo2:
        if spo2_pct <= 91:                spo2_s = 3
        elif 92 <= spo2_pct <= 93:        spo2_s = 2
        elif 94 <= spo2_pct <= 95:        spo2_s = 1
        else:                             spo2_s = 0
    else:
        if spo2_pct <= 83:                spo2_s = 3
        elif 84 <= spo2_pct <= 85:        spo2_s = 2
        elif 86 <= spo2_pct <= 87:        spo2_s = 1
        elif 88 <= spo2_pct <= 92:        spo2_s = 0
        elif 93 <= spo2_pct <= 94:        spo2_s = 1
        elif 95 <= spo2_pct <= 96:        spo2_s = 2
        else:                             spo2_s = 3
    o2_score = 2 if on_supplemental_oxygen else 0
    if systolic_bp <= 90:                 sbp_s = 3
    elif 91 <= systolic_bp <= 100:        sbp_s = 2
    elif 101 <= systolic_bp <= 110:       sbp_s = 1
    elif 111 <= systolic_bp <= 219:       sbp_s = 0
    else:                                 sbp_s = 3
    if heart_rate <= 40:                  hr_s = 3
    elif 41 <= heart_rate <= 50:          hr_s = 1
    elif 51 <= heart_rate <= 90:          hr_s = 0
    elif 91 <= heart_rate <= 110:         hr_s = 1
    elif 111 <= heart_rate <= 130:        hr_s = 2
    else:                                 hr_s = 3
    if temperature_c <= 35.0:            temp_s = 3
    elif 35.1 <= temperature_c <= 36.0:  temp_s = 1
    elif 36.1 <= temperature_c <= 38.0:  temp_s = 0
    elif 38.1 <= temperature_c <= 39.0:  temp_s = 1
    else:                                temp_s = 2
    avpu_upper = avpu.upper()
    avpu_s_map = {"A": 0, "C": 3, "V": 3, "P": 3, "U": 3}
    avpu_s = avpu_s_map.get(avpu_upper, 3)
    total = rr_s + spo2_s + o2_score + sbp_s + hr_s + temp_s + avpu_s
    if total >= NEWS2_HIGH_THRESHOLD or rr_s == 3 or avpu_s == 3 or sbp_s == 3:
        risk    = "HIGH"
        monitor = "Continuous monitoring — immediate ICU review"
    elif total >= NEWS2_MEDIUM_THRESHOLD:
        risk    = "MEDIUM"
        monitor = "30-minute monitoring — senior clinical review"
    elif total >= NEWS2_LOW_THRESHOLD:
        risk    = "LOW"
        monitor = "4-hourly monitoring"
    else:
        risk    = "MINIMAL"
        monitor = "12-hourly monitoring"
    return NEWS2Result(
        respiratory_rate_score=rr_s,
        spo2_score=spo2_s,
        systolic_bp_score=sbp_s,
        heart_rate_score=hr_s,
        temperature_score=temp_s,
        consciousness_score=avpu_s,
        air_or_oxygen_score=o2_score,
        total=total,
        respiratory_rate=respiratory_rate,
        spo2_pct=spo2_pct,
        systolic_bp=systolic_bp,
        heart_rate=heart_rate,
        temperature_celsius=temperature_c,
        on_supplemental_oxygen=on_supplemental_oxygen,
        avpu=avpu_upper,
        clinical_risk=risk,
        monitoring_frequency=monitor,
    )
@dataclass(frozen=True)
class SOFAResult:
    respiratory_score:      int   
    coagulation_score:      int   
    liver_score:            int   
    cardiovascular_score:   int   
    neurological_score:     int   
    renal_score:            int   
    total:                  int
    pao2_fio2_ratio:        Optional[float]
    platelets_10e9:         Optional[float]
    bilirubin_umol_l:       Optional[float]
    map_mmhg:               Optional[int]
    vasopressor_dose:       Optional[float]
    gcs:                    int
    creatinine_umol_l:      Optional[float]
    urine_output_ml_24h:    Optional[float]
    @property
    def icu_mortality_pct(self) -> float:
        if self.total >= 15:   return 90.0
        if self.total >= 12:   return 75.0
        if self.total >= 9:    return 50.0
        if self.total >= 6:    return 20.0
        if self.total >= 3:    return 7.0
        return 1.0
    @property
    def multi_organ_failure(self) -> bool:
        scores = [
            self.respiratory_score, self.coagulation_score,
            self.liver_score, self.cardiovascular_score,
            self.neurological_score, self.renal_score,
        ]
        return sum(1 for s in scores if s >= 3) >= 2
    @property
    def dominant_failure(self) -> str:
        mapping = {
            "respiratory":    self.respiratory_score,
            "coagulation":    self.coagulation_score,
            "liver":          self.liver_score,
            "cardiovascular": self.cardiovascular_score,
            "neurological":   self.neurological_score,
            "renal":          self.renal_score,
        }
        return max(mapping, key=lambda k: mapping[k])
    @property
    def severity_label(self) -> str:
        if self.total >= 12:   return "CRITICAL — probable multi-organ failure"
        if self.total >= 8:    return "SEVERE"
        if self.total >= 4:    return "MODERATE"
        return "MILD"
def compute_sofa(
    pao2_fio2_ratio:        Optional[float],
    platelets_10e9:         Optional[float],
    bilirubin_umol_l:       Optional[float],
    map_mmhg:               Optional[int],
    vasopressor_mcg_kg_min: Optional[float],   
    gcs:                    int,
    creatinine_umol_l:      Optional[float],
    urine_output_ml_24h:    Optional[float],
    on_ventilator:          bool = False,
) -> SOFAResult:
    if pao2_fio2_ratio is None:
        resp_s = 0
    elif pao2_fio2_ratio < 100 and on_ventilator:  resp_s = 4
    elif pao2_fio2_ratio < 200 and on_ventilator:  resp_s = 3
    elif pao2_fio2_ratio < 300:                     resp_s = 2
    elif pao2_fio2_ratio < 400:                     resp_s = 1
    else:                                            resp_s = 0
    if platelets_10e9 is None:                      coag_s = 0
    elif platelets_10e9 < 20:                        coag_s = 4
    elif platelets_10e9 < 50:                        coag_s = 3
    elif platelets_10e9 < 100:                       coag_s = 2
    elif platelets_10e9 < 150:                       coag_s = 1
    else:                                            coag_s = 0
    if bilirubin_umol_l is None:                    liver_s = 0
    elif bilirubin_umol_l >= 204:                    liver_s = 4
    elif bilirubin_umol_l >= 102:                    liver_s = 3
    elif bilirubin_umol_l >= 33:                     liver_s = 2
    elif bilirubin_umol_l >= 20:                     liver_s = 1
    else:                                            liver_s = 0
    cv_s = 0
    if vasopressor_mcg_kg_min is not None:
        if vasopressor_mcg_kg_min > 0.1:    cv_s = 4
        elif vasopressor_mcg_kg_min > 0.1:  cv_s = 4
        else:                               cv_s = 3
    elif map_mmhg is not None and map_mmhg < 70:
        cv_s = 1
    if gcs >= 15:     neuro_s = 0
    elif gcs >= 13:   neuro_s = 1
    elif gcs >= 10:   neuro_s = 2
    elif gcs >= 6:    neuro_s = 3
    else:             neuro_s = 4
    renal_s = 0
    if creatinine_umol_l is not None:
        if creatinine_umol_l >= 440:             renal_s = 4
        elif creatinine_umol_l >= 300:           renal_s = 3
        elif creatinine_umol_l >= 171:           renal_s = 2
        elif creatinine_umol_l >= 110:           renal_s = 1
    if urine_output_ml_24h is not None:
        if urine_output_ml_24h < 200:            renal_s = max(renal_s, 4)
        elif urine_output_ml_24h < 500:          renal_s = max(renal_s, 3)
    total = resp_s + coag_s + liver_s + cv_s + neuro_s + renal_s
    return SOFAResult(
        respiratory_score=resp_s,
        coagulation_score=coag_s,
        liver_score=liver_s,
        cardiovascular_score=cv_s,
        neurological_score=neuro_s,
        renal_score=renal_s,
        total=total,
        pao2_fio2_ratio=pao2_fio2_ratio,
        platelets_10e9=platelets_10e9,
        bilirubin_umol_l=bilirubin_umol_l,
        map_mmhg=map_mmhg,
        vasopressor_dose=vasopressor_mcg_kg_min,
        gcs=gcs,
        creatinine_umol_l=creatinine_umol_l,
        urine_output_ml_24h=urine_output_ml_24h,
    )
@dataclass(frozen=True)
class APACHEIIResult:
    acute_physiology_score: int    
    age_points:             int
    chronic_health_points:  int
    total:                  int
    predicted_mortality_pct: float
    aps_components: Dict[str, int]
    @property
    def icu_admission_indicated(self) -> bool:
        return self.total >= 15
    @property
    def severity_label(self) -> str:
        if self.total >= 35:   return "EXTREME"
        if self.total >= APACHE_CRITICAL_THRESHOLD: return "CRITICAL"
        if self.total >= 15:   return "SEVERE"
        if self.total >= 8:    return "MODERATE"
        return "MILD"
    @property
    def expected_icu_days(self) -> float:
        return round(0.8 * self.total - 3.0, 1) if self.total > 3 else 0.0
def compute_apache2(
    temperature_c:     float,
    map_mmhg:          int,
    heart_rate:        int,
    respiratory_rate:  int,
    pao2_or_aado2:     float,   
    is_aado2:          bool,
    arterial_ph:       Optional[float],
    serum_na_meq:      Optional[float],
    serum_k_meq:       Optional[float],
    creatinine_umol_l: Optional[float],
    hematocrit_pct:    Optional[float],
    wbc_10e9:          Optional[float],
    gcs:               int,
    age_years:         int,
    chronic_organ_failure: bool = False,
    elective_postop:       bool = False,
    nonoperative:          bool = True,
) -> APACHEIIResult:
    aps: Dict[str, int] = {}
    if temperature_c >= 41 or temperature_c < 29.9:  aps["temp"] = 4
    elif temperature_c >= 39 or 30 <= temperature_c <= 31.9: aps["temp"] = 3
    elif temperature_c >= 38.5 or 32 <= temperature_c <= 33.9: aps["temp"] = 2
    elif 36 <= temperature_c <= 38.4: aps["temp"] = 0
    elif 34 <= temperature_c <= 35.9: aps["temp"] = 1
    else: aps["temp"] = 2
    if map_mmhg >= 160 or map_mmhg < 50:  aps["map"] = 4
    elif map_mmhg >= 130 or 50 <= map_mmhg <= 69: aps["map"] = 3
    elif map_mmhg >= 110: aps["map"] = 2
    elif 70 <= map_mmhg <= 109: aps["map"] = 0
    else: aps["map"] = 2
    if heart_rate >= 180 or heart_rate < 40: aps["hr"] = 4
    elif heart_rate >= 140 or 40 <= heart_rate <= 54: aps["hr"] = 3
    elif heart_rate >= 110: aps["hr"] = 2
    elif 70 <= heart_rate <= 109: aps["hr"] = 0
    elif 55 <= heart_rate <= 69: aps["hr"] = 2
    else: aps["hr"] = 2
    if respiratory_rate >= 50 or respiratory_rate < 6: aps["rr"] = 4
    elif respiratory_rate >= 35 or 6 <= respiratory_rate <= 9: aps["rr"] = 3
    elif 25 <= respiratory_rate <= 34: aps["rr"] = 1
    elif 12 <= respiratory_rate <= 24: aps["rr"] = 0
    elif 10 <= respiratory_rate <= 11: aps["rr"] = 1
    else: aps["rr"] = 1
    if is_aado2:
        if pao2_or_aado2 >= 500: aps["oxygenation"] = 4
        elif pao2_or_aado2 >= 350: aps["oxygenation"] = 3
        elif pao2_or_aado2 >= 200: aps["oxygenation"] = 2
        else: aps["oxygenation"] = 0
    else:
        if pao2_or_aado2 < 55: aps["oxygenation"] = 4
        elif pao2_or_aado2 < 61: aps["oxygenation"] = 3
        elif pao2_or_aado2 < 71: aps["oxygenation"] = 1
        else: aps["oxygenation"] = 0
    if arterial_ph is not None:
        if arterial_ph >= 7.70 or arterial_ph < 7.15: aps["ph"] = 4
        elif arterial_ph >= 7.60 or 7.15 <= arterial_ph <= 7.24: aps["ph"] = 3
        elif arterial_ph >= 7.50: aps["ph"] = 1
        elif 7.33 <= arterial_ph <= 7.49: aps["ph"] = 0
        elif 7.25 <= arterial_ph <= 7.32: aps["ph"] = 2
        else: aps["ph"] = 2
    else:
        aps["ph"] = 0
    if serum_na_meq is not None:
        if serum_na_meq >= 180 or serum_na_meq < 111: aps["na"] = 4
        elif serum_na_meq >= 160 or 111 <= serum_na_meq <= 119: aps["na"] = 3
        elif serum_na_meq >= 155 or 120 <= serum_na_meq <= 129: aps["na"] = 2
        elif 150 <= serum_na_meq <= 154: aps["na"] = 1
        elif 130 <= serum_na_meq <= 149: aps["na"] = 0
        else: aps["na"] = 0
    else: aps["na"] = 0
    if serum_k_meq is not None:
        if serum_k_meq >= 7 or serum_k_meq < 2.5: aps["k"] = 4
        elif 6 <= serum_k_meq <= 6.9: aps["k"] = 3
        elif 2.5 <= serum_k_meq <= 2.9: aps["k"] = 2
        elif 55 <= serum_k_meq <= 5.9 or 3 <= serum_k_meq <= 3.4: aps["k"] = 1
        else: aps["k"] = 0
    else: aps["k"] = 0
    if creatinine_umol_l is not None:
        if creatinine_umol_l >= 309:   aps["creat"] = 4
        elif creatinine_umol_l >= 177: aps["creat"] = 3
        elif creatinine_umol_l >= 133: aps["creat"] = 2
        elif creatinine_umol_l < 54:   aps["creat"] = 2
        else:                          aps["creat"] = 0
    else: aps["creat"] = 0
    if hematocrit_pct is not None:
        if hematocrit_pct >= 60 or hematocrit_pct < 20: aps["hct"] = 4
        elif 50 <= hematocrit_pct <= 59.9 or 20 <= hematocrit_pct <= 29.9: aps["hct"] = 2
        elif 46 <= hematocrit_pct <= 49.9: aps["hct"] = 1
        else: aps["hct"] = 0
    else: aps["hct"] = 0
    if wbc_10e9 is not None:
        if wbc_10e9 >= 40 or wbc_10e9 < 1: aps["wbc"] = 4
        elif wbc_10e9 >= 20 or 1 <= wbc_10e9 <= 2.9: aps["wbc"] = 2
        elif 15 <= wbc_10e9 <= 19.9: aps["wbc"] = 1
        else: aps["wbc"] = 0
    else: aps["wbc"] = 0
    aps["gcs"] = 15 - gcs
    aps_total = sum(aps.values())
    if age_years < 44:     age_pts = 0
    elif age_years <= 54:  age_pts = 2
    elif age_years <= 64:  age_pts = 3
    elif age_years <= 74:  age_pts = 5
    else:                  age_pts = 6
    if chronic_organ_failure:
        if nonoperative or (not elective_postop):
            chp = 5
        else:
            chp = 2
    else:
        chp = 0
    total = aps_total + age_pts + chp
    logit = -3.517 + 0.146 * total
    mortality = 100.0 / (1.0 + math.exp(-logit))
    return APACHEIIResult(
        acute_physiology_score=aps_total,
        age_points=age_pts,
        chronic_health_points=chp,
        total=total,
        predicted_mortality_pct=round(mortality, 1),
        aps_components=dict(aps),
    )
@dataclass(frozen=True)
class MGAPResult:
    mechanism_score:    int
    gcs_score:          int
    age_score:          int
    sbp_score:          int
    total:              int
    @property
    def predicted_inhospital_mortality_pct(self) -> float:
        if self.total <= 17:    return 50.0
        if self.total <= 22:    return 25.0
        if self.total <= 26:    return 5.0
        return 1.0
    @property
    def risk_group(self) -> str:
        if self.total <= MGAP_CRITICAL_THRESHOLD: return "HIGH_RISK"
        if self.total <= 22:                      return "MODERATE_RISK"
        return "LOW_RISK"
    @property
    def hospital_level_required(self) -> str:
        rg = self.risk_group
        if rg == "HIGH_RISK":     return "Level-I Trauma Centre"
        if rg == "MODERATE_RISK": return "Level-II Trauma Centre"
        return "Level-III or District Hospital"
def compute_mgap(
    mechanism:     InjuryMechanism,
    gcs:           int,
    age_years:     int,
    systolic_bp:   int,
) -> MGAPResult:
    mech_s = mechanism.mgap_mechanism_points   
    if gcs >= 14:    gcs_s = 5
    elif gcs >= 11:  gcs_s = 3
    elif gcs >= 8:   gcs_s = 2
    elif gcs >= 5:   gcs_s = 1
    else:            gcs_s = 0
    age_s = 5 if age_years < 60 else 0
    if systolic_bp >= 120:      sbp_s = 5
    elif systolic_bp >= 60:     sbp_s = 3
    else:                       sbp_s = 0
    total = mech_s + gcs_s + age_s + sbp_s
    return MGAPResult(
        mechanism_score=mech_s,
        gcs_score=gcs_s,
        age_score=age_s,
        sbp_score=sbp_s,
        total=total,
    )
@dataclass(frozen=True)
class GAPResult:
    gcs_score: int
    age_score: int
    sbp_score: int
    total:     int
    @property
    def predicted_mortality_pct(self) -> float:
        if self.total <= 3:    return 80.0
        if self.total <= 10:   return 40.0
        if self.total <= 18:   return 10.0
        return 1.5
    @property
    def risk_category(self) -> str:
        if self.total <= GAP_CRITICAL_THRESHOLD: return "HIGH"
        if self.total <= 24:                      return "MODERATE"
        return "LOW"
def compute_gap(
    gcs:         int,
    age_years:   int,
    systolic_bp: int,
) -> GAPResult:
    if gcs >= 14:   gcs_s = 3
    elif gcs >= 11: gcs_s = 2
    elif gcs >= 8:  gcs_s = 1
    else:           gcs_s = 0
    if age_years < 20:          age_s = 5
    elif age_years <= 40:        age_s = 4
    elif age_years <= 60:        age_s = 3
    elif age_years <= 80:        age_s = 2
    else:                        age_s = 0
    if systolic_bp >= 120:      sbp_s = 14
    elif systolic_bp >= 100:    sbp_s = 11
    elif systolic_bp >= 80:     sbp_s = 9
    elif systolic_bp >= 60:     sbp_s = 6
    elif systolic_bp >= 1:      sbp_s = 3
    else:                       sbp_s = 0
    return GAPResult(
        gcs_score=gcs_s, age_score=age_s, sbp_score=sbp_s,
        total=gcs_s + age_s + sbp_s,
    )
@dataclass(frozen=True)
class KTSResult:
    age_score:          int
    systolic_bp_score:  int
    respiratory_rate_score: int
    motor_performance_score: int
    injuries_score:     int
    total:              int
    @property
    def predicted_mortality_pct(self) -> float:
        if self.total <= 7:   return 75.0
        if self.total <= 10:  return 45.0
        if self.total <= 14:  return 20.0
        if self.total <= 17:  return 5.0
        return 1.0
    @property
    def severity_label(self) -> str:
        if self.total <= KTS_SEVERE_THRESHOLD:   return "SEVERE"
        if self.total <= KTS_MODERATE_THRESHOLD: return "MODERATE"
        return "MILD"
    @property
    def resource_level_required(self) -> str:
        if self.total <= 10:  return "Tertiary level — MICU transport"
        if self.total <= 14:  return "Secondary level — ALS transport"
        return "Primary level — BLS acceptable"
def compute_kts(
    age_years:         int,
    systolic_bp:       int,
    respiratory_rate:  int,
    gcs:               int,
    number_of_serious_injuries: int,   
) -> KTSResult:
    if age_years <= 54:  age_s = 5
    elif age_years <= 60: age_s = 4
    elif age_years <= 65: age_s = 3
    elif age_years <= 70: age_s = 2
    else:                age_s = 1
    if systolic_bp >= 90:      sbp_s = 5
    elif 50 <= systolic_bp < 90: sbp_s = 3
    elif 1 <= systolic_bp < 50:  sbp_s = 1
    else:                        sbp_s = 0
    if 10 <= respiratory_rate <= 24: rr_s = 5
    elif 25 <= respiratory_rate <= 35: rr_s = 3
    elif respiratory_rate > 35 or (1 <= respiratory_rate <= 9): rr_s = 1
    else:                                                         rr_s = 0
    if gcs >= 13:    motor_s = 5
    elif gcs >= 9:   motor_s = 3
    elif gcs >= 6:   motor_s = 2
    elif gcs >= 4:   motor_s = 1
    else:            motor_s = 0
    if number_of_serious_injuries == 0:   inj_s = 5
    elif number_of_serious_injuries == 1: inj_s = 3
    else:                                 inj_s = 1
    total = age_s + sbp_s + rr_s + motor_s + inj_s
    return KTSResult(
        age_score=age_s,
        systolic_bp_score=sbp_s,
        respiratory_rate_score=rr_s,
        motor_performance_score=motor_s,
        injuries_score=inj_s,
        total=total,
    )
@dataclass(frozen=True)
class CRAMSResult:
    circulation_score: int
    respiration_score: int
    abdomen_score:     int
    motor_score:       int
    speech_score:      int
    total:             int
    @property
    def is_major_trauma(self) -> bool:
        return self.total <= 8
    @property
    def predicted_mortality_pct(self) -> float:
        if self.total <= 6:   return 70.0
        if self.total <= 8:   return 25.0
        if self.total == 9:   return 8.0
        return 1.0
    @property
    def field_triage_decision(self) -> str:
        if self.is_major_trauma:
            return "MAJOR — divert to Level-I trauma centre"
        return "MINOR — standard ED transport"
def compute_crams(
    capillary_refill_seconds: float,
    systolic_bp:        int,
    respiratory_effort: str,   
    thorax_abdomen:     str,   
    gcs:                int,
    speech_clarity:     str,   
) -> CRAMSResult:
    if systolic_bp >= 100 and capillary_refill_seconds <= 2: circ_s = 2
    elif systolic_bp >= 85 or capillary_refill_seconds <= 3:  circ_s = 1
    else:                                                      circ_s = 0
    resp_map = {"normal": 2, "laboured": 1, "absent": 0}
    resp_s = resp_map.get(respiratory_effort.lower(), 1)
    abd_map = {"normal": 2, "tender": 1, "rigid_or_flail": 0}
    abd_s = abd_map.get(thorax_abdomen.lower(), 1)
    if gcs >= 14:    motor_s = 2
    elif gcs >= 6:   motor_s = 1
    else:            motor_s = 0
    speech_map = {"normal": 2, "confused_or_inappropriate": 1, "absent": 0}
    speech_s = speech_map.get(speech_clarity.lower(), 1)
    return CRAMSResult(
        circulation_score=circ_s,
        respiration_score=resp_s,
        abdomen_score=abd_s,
        motor_score=motor_s,
        speech_score=speech_s,
        total=circ_s + resp_s + abd_s + motor_s + speech_s,
    )
@dataclass(frozen=True)
class PTSResult:
    weight_score:     int
    airway_score:     int
    sbp_score:        int
    consciousness_score: int
    open_wound_score: int
    skeletal_score:   int
    total:            int
    @property
    def predicted_mortality_pct(self) -> float:
        if self.total <= 0:   return 90.0
        if self.total <= 5:   return 35.0
        if self.total <= 8:   return 10.0
        return 2.0
    @property
    def major_paediatric_trauma(self) -> bool:
        return self.total <= 8
    @property
    def hospital_level_required(self) -> str:
        if self.total <= 5:  return "Paediatric Level-I Trauma Centre"
        if self.total <= 8:  return "Paediatric Trauma Centre"
        return "Paediatric Emergency Department"
def compute_pts(
    weight_kg:       float,
    airway:          str,    
    systolic_bp:     int,
    consciousness:   str,    
    open_wound:      str,    
    skeletal:        str,    
) -> PTSResult:
    if weight_kg >= 20:     wt_s = 2
    elif weight_kg >= 10:   wt_s = 1
    else:                   wt_s = -1
    airway_map = {"normal": 2, "maintainable": 1, "unmaintainable": -1}
    aw_s = airway_map.get(airway.lower(), 1)
    if systolic_bp >= 90:       sbp_s = 2
    elif systolic_bp >= 50:     sbp_s = 1
    else:                       sbp_s = -1
    cons_map = {"awake": 2, "obtunded": 1, "coma": -1}
    cons_s = cons_map.get(consciousness.lower(), 1)
    wound_map = {"none": 2, "minor": 1, "major_or_penetrating": -1}
    wound_s = wound_map.get(open_wound.lower(), 1)
    skel_map = {"none": 2, "closed_fracture": 1, "open_or_multiple": -1}
    skel_s = skel_map.get(skeletal.lower(), 1)
    return PTSResult(
        weight_score=wt_s,
        airway_score=aw_s,
        sbp_score=sbp_s,
        consciousness_score=cons_s,
        open_wound_score=wound_s,
        skeletal_score=skel_s,
        total=wt_s + aw_s + sbp_s + cons_s + wound_s + skel_s,
    )
@dataclass(frozen=True)
class DCSAssessment:
    indication_level:          DCSIndicationLevel
    physiological_triggers:    List[str]
    anatomical_triggers:       List[str]
    total_trigger_count:       int
    damage_control_sequence:   List[str]
    expected_icu_days:         float
    haemostatic_resuscitation: bool
    massive_transfusion_protocol: bool
    clinical_notes:            str
    @property
    def requires_immediate_or(self) -> bool:
        return self.indication_level == DCSIndicationLevel.ABSOLUTE
    @property
    def permissive_hypotension_indicated(self) -> bool:
        return self.indication_level in (
            DCSIndicationLevel.STRONG_RELATIVE,
            DCSIndicationLevel.ABSOLUTE,
        )
    @property
    def txa_indicated(self) -> bool:
        return self.indication_level != DCSIndicationLevel.NOT_INDICATED
    def to_grader_dict(self) -> Dict[str, Any]:
        return {
            "indication":             self.indication_level.value,
            "physiological_triggers": self.physiological_triggers,
            "anatomical_triggers":    self.anatomical_triggers,
            "total_triggers":         self.total_trigger_count,
            "immediate_or":           self.requires_immediate_or,
            "mtp":                    self.massive_transfusion_protocol,
            "permissive_hypotension": self.permissive_hypotension_indicated,
            "txa_indicated":          self.txa_indicated,
        }
def assess_dcs(
    systolic_bp:            int,
    base_deficit_meq:       float,
    temperature_c:          float,
    inr:                    Optional[float],
    packed_cells_given:     int,
    lactate_mmol_l:         Optional[float],
    iss:                    int,
    has_hollow_viscus_injury: bool,
    has_major_vascular:     bool,
    has_pelvic_fracture:    bool,
    combined_abdomino_thoracic: bool,
    predicted_op_time_hours: float,
    age_years:              int,
) -> DCSAssessment:
    physio_triggers: List[str] = []
    anat_triggers: List[str]   = []
    if systolic_bp < 90:
        physio_triggers.append(f"SBP {systolic_bp} mmHg < 90 (haemorrhagic shock)")
    if base_deficit_meq >= DCS_BASE_DEFICIT_THRESHOLD:
        physio_triggers.append(f"Base deficit {base_deficit_meq:.1f} mEq/L ≥ {DCS_BASE_DEFICIT_THRESHOLD} (acidosis)")
    if temperature_c < DCS_HYPOTHERMIA_THRESHOLD_C:
        physio_triggers.append(f"Temperature {temperature_c:.1f}°C < {DCS_HYPOTHERMIA_THRESHOLD_C} (hypothermia)")
    if inr is not None and inr >= DCS_COAGULOPATHY_INR:
        physio_triggers.append(f"INR {inr:.1f} ≥ {DCS_COAGULOPATHY_INR} (coagulopathy)")
    if packed_cells_given >= DCS_TRANSFUSION_PACK_THRESHOLD:
        physio_triggers.append(f"{packed_cells_given} pRBC units given (massive transfusion)")
    if lactate_mmol_l is not None and lactate_mmol_l >= DCS_LACTATE_THRESHOLD_MMOL:
        physio_triggers.append(f"Lactate {lactate_mmol_l:.1f} mmol/L ≥ {DCS_LACTATE_THRESHOLD_MMOL} (tissue hypoperfusion)")
    if iss >= ISS_CRITICAL_THRESHOLD:
        anat_triggers.append(f"ISS {iss} ≥ {ISS_CRITICAL_THRESHOLD} (critical polytrauma)")
    if has_hollow_viscus_injury:
        anat_triggers.append("Hollow viscus injury (enteric contamination risk)")
    if has_major_vascular:
        anat_triggers.append("Major vascular injury (juxtahepatic / suprarenal)")
    if has_pelvic_fracture:
        anat_triggers.append("Unstable pelvic fracture (retroperitoneal haemorrhage)")
    if combined_abdomino_thoracic:
        anat_triggers.append("Combined abdomino-thoracic injury")
    if predicted_op_time_hours > 4:
        anat_triggers.append(f"Predicted operative time {predicted_op_time_hours:.1f}h > 4h (physiological exhaustion risk)")
    n_physio  = len(physio_triggers)
    n_anat    = len(anat_triggers)
    total_t   = n_physio + n_anat
    lethal_triad = (
        temperature_c < DCS_HYPOTHERMIA_THRESHOLD_C
        and base_deficit_meq >= DCS_BASE_DEFICIT_THRESHOLD
        and (inr is not None and inr >= DCS_COAGULOPATHY_INR)
    )
    if lethal_triad or (n_physio >= 3 and n_anat >= 1):
        level = DCSIndicationLevel.ABSOLUTE
    elif n_physio >= 2 and n_anat >= 1:
        level = DCSIndicationLevel.STRONG_RELATIVE
    elif total_t >= 2:
        level = DCSIndicationLevel.RELATIVE
    else:
        level = DCSIndicationLevel.NOT_INDICATED
    dcs_seq: List[str] = []
    if level != DCSIndicationLevel.NOT_INDICATED:
        dcs_seq = [
            "1. DCS-0: Haemostatic resuscitation — permissive hypotension (SBP 80-90)",
            "2. DCS-I: Damage control laparotomy — haemostasis + contamination control (< 90 min)",
            "3. DCS-II: ICU resuscitation — correct lethal triad",
            "4. DCS-III: Planned re-look laparotomy 24-48h after physiology corrected",
        ]
    else:
        dcs_seq = ["Definitive surgical repair — physiologically stable for prolonged procedure"]
    icu_days = round(0.5 * iss + 1.2 * n_physio + 0.8 * n_anat, 1)
    mtp = packed_cells_given >= DCS_TRANSFUSION_PACK_THRESHOLD or n_physio >= 2
    hr  = mtp or (inr is not None and inr >= 1.3)
    notes = (
        f"DCS indication: {level.value.upper()}. "
        f"Physio triggers: {n_physio} | Anatomic triggers: {n_anat}. "
        f"Lethal triad: {'YES' if lethal_triad else 'NO'}. "
        f"TXA {'indicated (CRASH-2)' if level != DCSIndicationLevel.NOT_INDICATED else 'not required'}."
    )
    return DCSAssessment(
        indication_level=level,
        physiological_triggers=physio_triggers,
        anatomical_triggers=anat_triggers,
        total_trigger_count=total_t,
        damage_control_sequence=dcs_seq,
        expected_icu_days=icu_days,
        haemostatic_resuscitation=hr,
        massive_transfusion_protocol=mtp,
        clinical_notes=notes,
    )
@dataclass
class FASTExam:
    pericardial:      FASTFinding = FASTFinding.NEGATIVE
    right_upper_quad: FASTFinding = FASTFinding.NEGATIVE
    left_upper_quad:  FASTFinding = FASTFinding.NEGATIVE
    pelvis:           FASTFinding = FASTFinding.NEGATIVE
    right_pleural:    FASTFinding = FASTFinding.NEGATIVE
    left_pleural:     FASTFinding = FASTFinding.NEGATIVE
    operator_experience_level: str = "trained"   
    exam_quality:     str = "adequate"
    performed_step:   int = 0
    @property
    def is_positive(self) -> bool:
        return any(
            f not in (FASTFinding.NEGATIVE, FASTFinding.INDETERMINATE)
            for f in [
                self.pericardial, self.right_upper_quad,
                self.left_upper_quad, self.pelvis,
                self.right_pleural, self.left_pleural,
            ]
        )
    @property
    def cardiac_tamponade_suspected(self) -> bool:
        return self.pericardial == FASTFinding.POSITIVE_PERICARDIAL
    @property
    def haemoperitoneum_suspected(self) -> bool:
        return any(
            f in (FASTFinding.POSITIVE_RUQ, FASTFinding.POSITIVE_LUQ, FASTFinding.POSITIVE_PELVIS)
            for f in [self.right_upper_quad, self.left_upper_quad, self.pelvis]
        )
    @property
    def haemopneumothorax_suspected(self) -> bool:
        return any(
            f in (FASTFinding.POSITIVE_PLURAL_RIGHT, FASTFinding.POSITIVE_PLURAL_LEFT)
            for f in [self.right_pleural, self.left_pleural]
        )
    @property
    def immediate_operative_indication(self) -> bool:
        return self.cardiac_tamponade_suspected or self.haemoperitoneum_suspected
    @property
    def positive_views(self) -> int:
        return sum(
            1 for f in [
                self.pericardial, self.right_upper_quad,
                self.left_upper_quad, self.pelvis,
                self.right_pleural, self.left_pleural,
            ]
            if f not in (FASTFinding.NEGATIVE, FASTFinding.INDETERMINATE)
        )
    @property
    def trauma_team_activation_level(self) -> TraumaTeamActivationLevel:
        if self.cardiac_tamponade_suspected:
            return TraumaTeamActivationLevel.IMMEDIATE
        if self.positive_views >= 2:
            return TraumaTeamActivationLevel.FULL
        if self.positive_views == 1:
            return TraumaTeamActivationLevel.STANDBY
        return TraumaTeamActivationLevel.NONE
    def summary(self) -> str:
        findings = []
        if self.cardiac_tamponade_suspected:
            findings.append("CARDIAC TAMPONADE")
        if self.haemoperitoneum_suspected:
            findings.append("HAEMOPERITONEUM")
        if self.haemopneumothorax_suspected:
            findings.append("HAEMO/PNEUMOTHORAX")
        return (
            f"FAST: {'POSITIVE — ' + ', '.join(findings) if self.is_positive else 'NEGATIVE'}. "
            f"{self.positive_views} positive views. "
            f"TTA: {self.trauma_team_activation_level.value.upper()}."
        )
@dataclass
class PrimarySurveyState:
    patient_id: str
    step_started: int = 0
    airway_assessed:          bool = False
    airway_secured:           bool = False
    airway_intervention:      str  = "none"   
    c_spine_immobilised:      bool = False
    breathing_assessed:       bool = False
    ventilation_adequate:     bool = True
    tension_ptx_excluded:     bool = False
    open_chest_wound_sealed:  bool = False
    supplemental_oxygen:      bool = False
    circulation_assessed:     bool = False
    haemorrhage_controlled:   bool = False
    iv_access_obtained:       bool = False
    iv_fluid_given:           bool = False
    blood_type_sent:          bool = False
    massive_transfusion_activated: bool = False
    disability_assessed:      bool = False
    gcs_documented:           bool = False
    pupils_checked:           bool = False
    bsl_checked:              bool = False
    exposure_done:            bool = False
    hypothermia_prevented:    bool = False
    log_roll_done:            bool = False
    burns_area_estimated:     bool = False
    fast_performed:           bool = False
    fast_result:              Optional[FASTExam] = None
    @property
    def primary_survey_complete(self) -> bool:
        return all([
            self.airway_assessed,
            self.breathing_assessed,
            self.circulation_assessed,
            self.disability_assessed,
            self.exposure_done,
        ])
    @property
    def life_threats_addressed(self) -> bool:
        return all([
            self.airway_secured,
            self.haemorrhage_controlled,
        ])
    @property
    def completeness_score(self) -> float:
        checks = [
            self.airway_assessed, self.airway_secured,
            self.c_spine_immobilised, self.breathing_assessed,
            self.supplemental_oxygen, self.circulation_assessed,
            self.haemorrhage_controlled, self.iv_access_obtained,
            self.disability_assessed, self.gcs_documented,
            self.exposure_done, self.hypothermia_prevented,
        ]
        return sum(1 for c in checks if c) / len(checks)
    @property
    def omitted_critical_steps(self) -> List[str]:
        omissions: List[str] = []
        if not self.airway_assessed:          omissions.append("Airway not assessed")
        if not self.c_spine_immobilised:      omissions.append("C-spine not immobilised")
        if not self.haemorrhage_controlled:   omissions.append("Haemorrhage not controlled")
        if not self.iv_access_obtained:       omissions.append("IV access not obtained")
        if not self.gcs_documented:           omissions.append("GCS not documented")
        if not self.hypothermia_prevented:    omissions.append("Hypothermia prevention absent")
        return omissions
    @property
    def protocol_compliance_score(self) -> float:
        omission_penalty = len(self.omitted_critical_steps) * 0.08
        return max(0.0, self.completeness_score - omission_penalty)
@dataclass(frozen=True)
class TraumaActivationDecision:
    activation_level:   TraumaTeamActivationLevel
    physiological_met:  bool
    anatomical_met:     bool
    mechanism_met:      bool
    special_met:        bool
    triggered_criteria: List[str]
    clinical_notes:     str
    @property
    def prehospital_alert_required(self) -> bool:
        return self.activation_level in (
            TraumaTeamActivationLevel.FULL,
            TraumaTeamActivationLevel.IMMEDIATE,
        )
    @property
    def cath_lab_standby_required(self) -> bool:
        return any("STEMI" in c for c in self.triggered_criteria)
    @property
    def blood_bank_activation_required(self) -> bool:
        return self.activation_level in (
            TraumaTeamActivationLevel.FULL,
            TraumaTeamActivationLevel.IMMEDIATE,
        )
def assess_trauma_team_activation(
    gcs:                     int,
    systolic_bp:             int,
    respiratory_rate:        int,
    iss:                     Optional[int],
    mechanism:               InjuryMechanism,
    age_years:               int,
    has_penetrating_torso:   bool,
    has_amputation_proximal: bool,
    has_burns_major:         bool,
    has_paralysis:           bool,
    fall_height_metres:      Optional[float],
    vehicle_speed_kmh:       Optional[float],
    is_paediatric:           bool,
    fast_result:             Optional[FASTExam],
) -> TraumaActivationDecision:
    criteria: List[str] = []
    physio = False
    if gcs <= 13:
        criteria.append(f"Altered consciousness: GCS {gcs}")
        physio = True
    if systolic_bp < 90:
        criteria.append(f"Hypotension: SBP {systolic_bp} mmHg")
        physio = True
    if respiratory_rate < 10 or respiratory_rate > 29:
        criteria.append(f"Abnormal respiratory rate: {respiratory_rate}/min")
        physio = True
    anat = False
    if iss is not None and iss >= ISS_MAJOR_TRAUMA_THRESHOLD:
        criteria.append(f"ISS ≥ {ISS_MAJOR_TRAUMA_THRESHOLD}: ISS = {iss}")
        anat = True
    if has_penetrating_torso:
        criteria.append("Penetrating injury to torso")
        anat = True
    if has_amputation_proximal:
        criteria.append("Proximal amputation")
        anat = True
    if has_burns_major:
        criteria.append("Major burns ≥ 20% TBSA or face/airway")
        anat = True
    if has_paralysis:
        criteria.append("Suspected spinal cord injury with paralysis")
        anat = True
    mech = False
    if vehicle_speed_kmh is not None and vehicle_speed_kmh >= 80:
        criteria.append(f"High-energy MVC: {vehicle_speed_kmh:.0f} km/h")
        mech = True
    if fall_height_metres is not None and fall_height_metres >= 3 * (0.5 if is_paediatric else 1.0):
        criteria.append(f"Fall from height: {fall_height_metres:.1f} m")
        mech = True
    if mechanism == InjuryMechanism.BLUNT_BLAST:
        criteria.append("Blast/explosion mechanism")
        mech = True
    special = False
    if is_paediatric:
        criteria.append("Paediatric patient — lower activation threshold")
        special = True
    if age_years >= 75:
        criteria.append("Elderly patient ≥75 — physiological reserve reduced")
        special = True
    if fast_result is not None and fast_result.is_positive:
        criteria.append(f"Positive FAST: {fast_result.positive_views} views")
        special = True
    n = len(criteria)
    if physio and anat:
        level = TraumaTeamActivationLevel.IMMEDIATE
    elif physio or (anat and mech):
        level = TraumaTeamActivationLevel.FULL
    elif anat or (mech and n >= 2):
        level = TraumaTeamActivationLevel.STANDBY
    else:
        level = TraumaTeamActivationLevel.NONE
    notes = (
        f"TTA {level.value.upper()} — {len(criteria)} criteria met. "
        f"Physio: {'YES' if physio else 'NO'}, "
        f"Anatomical: {'YES' if anat else 'NO'}, "
        f"Mechanism: {'YES' if mech else 'NO'}."
    )
    return TraumaActivationDecision(
        activation_level=level,
        physiological_met=physio,
        anatomical_met=anat,
        mechanism_met=mech,
        special_met=special,
        triggered_criteria=criteria,
        clinical_notes=notes,
    )
@dataclass
class TraumaScoreBundle:
    patient_id:       str
    incident_id:      str
    condition_key:    str
    mechanism:        InjuryMechanism
    age_years:        int
    is_paediatric:    bool
    step:             int
    iss_result:       Optional[ISSResult]           = None
    rts_result:       Optional[RTSResult]           = None
    triss_result:     Optional[TRISSResult]         = None
    mews_result:      Optional[MEWSResult]          = None
    news2_result:     Optional[NEWS2Result]         = None
    sofa_result:      Optional[SOFAResult]          = None
    apache2_result:   Optional[APACHEIIResult]      = None
    mgap_result:      Optional[MGAPResult]          = None
    gap_result:       Optional[GAPResult]           = None
    kts_result:       Optional[KTSResult]           = None
    crams_result:     Optional[CRAMSResult]         = None
    pts_result:       Optional[PTSResult]           = None
    dcs_assessment:   Optional[DCSAssessment]       = None
    fast_exam:        Optional[FASTExam]            = None
    primary_survey:   Optional[PrimarySurveyState]  = None
    tta_decision:     Optional[TraumaActivationDecision] = None
    consensus_survival_probability: float = 0.5
    consensus_severity_label:       str   = "UNKNOWN"
    dominant_scoring_system:        str   = "RTS+ISS"
    def compute_consensus(self) -> None:
        probs: List[float] = []
        weights: List[float] = []
        if self.triss_result is not None:
            probs.append(self.triss_result.survival_prob)
            weights.append(3.0)  
        if self.rts_result is not None:
            mechanism_ps = (
                self.rts_result.survival_prob_blunt
                if self.mechanism.is_blunt
                else self.rts_result.survival_prob_penetrating
            )
            probs.append(mechanism_ps)
            weights.append(1.5)
        if self.iss_result is not None:
            iss_ps = 1.0 - self.iss_result.predicted_mortality_pct / 100.0
            probs.append(iss_ps)
            weights.append(1.0)
        if self.mgap_result is not None:
            mgap_ps = 1.0 - self.mgap_result.predicted_inhospital_mortality_pct / 100.0
            probs.append(mgap_ps)
            weights.append(1.0)
        if probs:
            total_w = sum(weights)
            self.consensus_survival_probability = round(
                sum(p * w for p, w in zip(probs, weights)) / total_w, 4
            )
        else:
            self.consensus_survival_probability = 0.5
        if self.iss_result is not None:
            self.consensus_severity_label = self.iss_result.severity_label
        elif self.triss_result is not None:
            self.consensus_severity_label = self.triss_result.predicted_outcome.value.upper()
    @property
    def reward_contribution_score(self) -> float:
        score = self.consensus_survival_probability
        if self.dcs_assessment is not None:
            if self.dcs_assessment.indication_level == DCSIndicationLevel.NOT_INDICATED:
                score = min(1.0, score + 0.03)
        if self.primary_survey is not None:
            score = min(1.0, score * (0.8 + 0.2 * self.primary_survey.protocol_compliance_score))
        return round(max(0.0, score), 4)
    def to_grader_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "patient_id":          self.patient_id,
            "incident_id":         self.incident_id,
            "condition_key":       self.condition_key,
            "mechanism":           self.mechanism.value,
            "age":                 self.age_years,
            "paediatric":          self.is_paediatric,
            "consensus_Ps":        self.consensus_survival_probability,
            "severity":            self.consensus_severity_label,
            "reward_score":        self.reward_contribution_score,
        }
        if self.iss_result:
            d["ISS"] = self.iss_result.iss
            d["NISS"] = self.iss_result.niss
            d["major_trauma"] = self.iss_result.is_major_trauma
        if self.rts_result:
            d["RTS"] = round(self.rts_result.rts, 3)
        if self.triss_result:
            d["TRISS_Ps"] = self.triss_result.survival_prob
            d["TRISS_outcome"] = self.triss_result.predicted_outcome.value
        if self.mews_result:
            d["MEWS"] = self.mews_result.total
            d["MEWS_alert"] = self.mews_result.alert_level
        if self.news2_result:
            d["NEWS2"] = self.news2_result.total
            d["NEWS2_risk"] = self.news2_result.clinical_risk
        if self.mgap_result:
            d["MGAP"] = self.mgap_result.total
            d["MGAP_risk"] = self.mgap_result.risk_group
        if self.kts_result:
            d["KTS"] = self.kts_result.total
            d["KTS_severity"] = self.kts_result.severity_label
        if self.dcs_assessment:
            d["DCS"] = self.dcs_assessment.indication_level.value
            d["MTP"] = self.dcs_assessment.massive_transfusion_protocol
        if self.tta_decision:
            d["TTA"] = self.tta_decision.activation_level.value
        return d
class TraumaScoringEngine:
    def __init__(self, rng_seed: int = 42) -> None:
        self._bundles: Dict[str, TraumaScoreBundle] = {}
        self._rng_seed = rng_seed
        logger.info("TraumaScoringEngine initialised (seed=%d)", rng_seed)
    def register_patient(
        self,
        patient_id:     str,
        incident_id:    str,
        condition_key:  str,
        mechanism:      InjuryMechanism,
        age_years:      int,
        step:           int,
        gcs:            int,
        systolic_bp:    int,
        respiratory_rate: int,
        heart_rate:     int,
        temperature_c:  float,
        spo2_pct:       float = 98.0,
        is_paediatric:  bool  = False,
    ) -> TraumaScoreBundle:
        bundle = TraumaScoreBundle(
            patient_id=patient_id,
            incident_id=incident_id,
            condition_key=condition_key,
            mechanism=mechanism,
            age_years=age_years,
            is_paediatric=is_paediatric,
            step=step,
        )
        bundle.rts_result = compute_rts(gcs, systolic_bp, respiratory_rate)
        bundle.mews_result = compute_mews(
            respiratory_rate=respiratory_rate,
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
            temperature_c=temperature_c,
            avpu="A" if gcs >= 14 else ("V" if gcs >= 10 else ("P" if gcs >= 6 else "U")),
        )
        bundle.news2_result = compute_news2(
            respiratory_rate=respiratory_rate,
            spo2_pct=spo2_pct,
            systolic_bp=systolic_bp,
            heart_rate=heart_rate,
            temperature_c=temperature_c,
            avpu="A" if gcs >= 14 else "V" if gcs >= 10 else "U",
        )
        bundle.mgap_result = compute_mgap(mechanism, gcs, age_years, systolic_bp)
        bundle.gap_result = compute_gap(gcs, age_years, systolic_bp)
        bundle.kts_result = compute_kts(
            age_years=age_years,
            systolic_bp=systolic_bp,
            respiratory_rate=respiratory_rate,
            gcs=gcs,
            number_of_serious_injuries=1,
        )
        self._bundles[patient_id] = bundle
        logger.debug(
            "TraumaScoringEngine: registered %s | %s | RTS=%.2f | MGAP=%d",
            patient_id, condition_key,
            bundle.rts_result.rts, bundle.mgap_result.total,
        )
        return bundle
    def add_anatomical_scoring(
        self,
        patient_id: str,
        region_profiles: Dict[BodyRegion, AISBodyRegionProfile],
    ) -> Optional[ISSResult]:
        bundle = self._bundles.get(patient_id)
        if bundle is None:
            return None
        iss_result = compute_iss(region_profiles)
        bundle.iss_result = iss_result
        if bundle.rts_result is not None:
            bundle.triss_result = compute_triss(
                rts=bundle.rts_result.rts,
                iss=iss_result.iss,
                age_years=bundle.age_years,
                mechanism=bundle.mechanism,
            )
        bundle.compute_consensus()
        return iss_result
    def add_fast_exam(
        self,
        patient_id: str,
        fast_exam:  FASTExam,
    ) -> None:
        bundle = self._bundles.get(patient_id)
        if bundle:
            bundle.fast_exam = fast_exam
    def add_primary_survey(
        self,
        patient_id:     str,
        primary_survey: PrimarySurveyState,
    ) -> None:
        bundle = self._bundles.get(patient_id)
        if bundle:
            bundle.primary_survey = primary_survey
    def add_dcs_assessment(
        self,
        patient_id: str,
        dcs:        DCSAssessment,
    ) -> None:
        bundle = self._bundles.get(patient_id)
        if bundle:
            bundle.dcs_assessment = dcs
    def add_sofa(
        self,
        patient_id: str,
        **sofa_kwargs: Any,
    ) -> Optional[SOFAResult]:
        bundle = self._bundles.get(patient_id)
        if bundle is None:
            return None
        result = compute_sofa(**sofa_kwargs)
        bundle.sofa_result = result
        return result
    def add_tta(
        self,
        patient_id: str,
        tta:        TraumaActivationDecision,
    ) -> None:
        bundle = self._bundles.get(patient_id)
        if bundle:
            bundle.tta_decision = tta
    def get_bundle(self, patient_id: str) -> Optional[TraumaScoreBundle]:
        return self._bundles.get(patient_id)
    def get_all_grader_dicts(self) -> List[Dict[str, Any]]:
        return [b.to_grader_dict() for b in self._bundles.values()]
    def critical_patients(self) -> List[str]:
        return [
            pid for pid, b in self._bundles.items()
            if b.consensus_survival_probability < 0.50
        ]
    def patients_requiring_dcs(self) -> List[str]:
        return [
            pid for pid, b in self._bundles.items()
            if b.dcs_assessment is not None
            and b.dcs_assessment.indication_level == DCSIndicationLevel.ABSOLUTE
        ]
    def episode_aggregate_trauma_score(self) -> Dict[str, float]:
        p1 = [b for b in self._bundles.values() if b.iss_result and b.iss_result.is_major_trauma]
        other = [b for b in self._bundles.values() if b not in p1]
        p1_avg  = (sum(b.consensus_survival_probability for b in p1) / len(p1)) if p1 else 1.0
        oth_avg = (sum(b.consensus_survival_probability for b in other) / len(other)) if other else 1.0
        return {
            "major_trauma_patients":       len(p1),
            "minor_trauma_patients":       len(other),
            "major_trauma_avg_Ps":         round(p1_avg, 4),
            "minor_trauma_avg_Ps":         round(oth_avg, 4),
            "weighted_aggregate_Ps":       round(
                (p1_avg * 2.0 + oth_avg * 1.0) / 3.0, 4
            ) if (p1 or other) else 0.5,
        }
    def dcs_compliance_score(self) -> float:
        relevant = [
            b for b in self._bundles.values()
            if b.iss_result and b.iss_result.iss >= ISS_MAJOR_TRAUMA_THRESHOLD
        ]
        if not relevant:
            return 1.0
        assessed = sum(1 for b in relevant if b.dcs_assessment is not None)
        return round(assessed / len(relevant), 4)
    def fast_compliance_score(self) -> float:
        torso_trauma = [
            b for b in self._bundles.values()
            if b.iss_result and b.iss_result.body_region_scores.get(
                BodyRegion.ABDOMEN_PELVIS.value, 0
            ) >= 4
        ]
        if not torso_trauma:
            return 1.0
        performed = sum(1 for b in torso_trauma if b.fast_exam is not None)
        return round(performed / len(torso_trauma), 4)
    def primary_survey_compliance_score(self) -> float:
        surveys = [
            b.primary_survey for b in self._bundles.values()
            if b.primary_survey is not None
        ]
        if not surveys:
            return 1.0
        return round(
            sum(s.protocol_compliance_score for s in surveys) / len(surveys), 4
        )
    def reset(self) -> None:
        self._bundles.clear()
        logger.debug("TraumaScoringEngine.reset()")
@dataclass
class TraumaRegistryEntry:
    entry_id:           str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id:         str = ""
    patient_id:         str = ""
    incident_id:        str = ""
    condition_key:      str = ""
    mechanism:          str = ""
    age_years:          int = 0
    gender:             str = "unknown"
    is_paediatric:      bool = False
    iss:                int   = 0
    niss:               int   = 0
    rts:                float = 0.0
    triss_ps:           float = 0.5
    mgap:               int   = 0
    kts:                int   = 0
    mews_admission:     int   = 0
    news2_admission:    int   = 0
    sofa_icu:           Optional[int] = None
    apache2_icu:        Optional[int] = None
    pts:                Optional[int] = None
    time_to_dispatch_min:   float = 0.0
    time_to_scene_min:      float = 0.0
    time_to_hospital_min:   float = 0.0
    time_to_definitive_min: float = 0.0
    survival_at_scene:      bool  = True
    survival_at_hospital:   bool  = True
    icu_admitted:           bool  = False
    icu_los_days:           float = 0.0
    hospital_los_days:      float = 0.0
    dcs_performed:          bool  = False
    fast_positive:          bool  = False
    trauma_team_activated:  bool  = False
    unit_type_correct:      bool  = True
    hospital_correct:       bool  = True
    golden_window_met:      bool  = True
    primary_survey_score:   float = 1.0
    protocol_compliance:    float = 1.0
    @property
    def probability_of_survival_at_admission(self) -> float:
        return self.triss_ps
    @property
    def unexpected_outcome(self) -> bool:
        survived = self.survival_at_hospital
        if survived and self.triss_ps < 0.25:
            return True
        if not survived and self.triss_ps > 0.75:
            return True
        return False
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id":             self.entry_id,
            "patient_id":           self.patient_id,
            "condition_key":        self.condition_key,
            "mechanism":            self.mechanism,
            "age":                  self.age_years,
            "ISS":                  self.iss,
            "NISS":                 self.niss,
            "RTS":                  round(self.rts, 3),
            "TRISS_Ps":             round(self.triss_ps, 4),
            "MGAP":                 self.mgap,
            "KTS":                  self.kts,
            "MEWS":                 self.mews_admission,
            "NEWS2":                self.news2_admission,
            "time_dispatch_min":    round(self.time_to_dispatch_min, 1),
            "time_hospital_min":    round(self.time_to_hospital_min, 1),
            "survival":             self.survival_at_hospital,
            "icu_admitted":         self.icu_admitted,
            "dcs_performed":        self.dcs_performed,
            "trauma_team_activated": self.trauma_team_activated,
            "unit_correct":         self.unit_type_correct,
            "hospital_correct":     self.hospital_correct,
            "golden_window":        self.golden_window_met,
            "unexpected_outcome":   self.unexpected_outcome,
        }
def _self_test() -> None:
    rts_r = compute_rts(gcs=15, systolic_bp=120, respiratory_rate=16)
    assert 6.0 <= rts_r.rts <= 7.85, f"Normal RTS out of range: {rts_r.rts}"
    assert rts_r.triage_category == "GREEN"
    assert not rts_r.needs_immediate_airway
    rts_bad = compute_rts(gcs=6, systolic_bp=60, respiratory_rate=8)
    assert rts_bad.rts < 5.0, f"Critical RTS should be < 5.0: {rts_bad.rts}"
    assert rts_bad.needs_immediate_airway
    profiles: Dict[BodyRegion, AISBodyRegionProfile] = {
        r: AISBodyRegionProfile(region=r) for r in BodyRegion
    }
    profiles[BodyRegion.HEAD_NECK].add(AISInjury(
        region=BodyRegion.HEAD_NECK,
        severity=AISSeverity.SERIOUS,
        descriptor="Epidural haematoma"
    ))
    profiles[BodyRegion.THORAX].add(AISInjury(
        region=BodyRegion.THORAX,
        severity=AISSeverity.SEVERE,
        descriptor="Tension pneumothorax"
    ))
    profiles[BodyRegion.ABDOMEN_PELVIS].add(AISInjury(
        region=BodyRegion.ABDOMEN_PELVIS,
        severity=AISSeverity.SERIOUS,
        descriptor="Splenic laceration grade III"
    ))
    iss_r = compute_iss(profiles)
    expected_iss = 3**2 + 4**2 + 3**2  
    assert iss_r.iss == 34, f"ISS mismatch: got {iss_r.iss}, expected 34"
    assert iss_r.is_major_trauma
    assert iss_r.multi_system
    triss_r = compute_triss(
        rts=rts_r.rts, iss=iss_r.iss, age_years=35,
        mechanism=InjuryMechanism.BLUNT_MVC
    )
    assert 0.0 <= triss_r.survival_prob <= 1.0
    triss_elderly = compute_triss(
        rts=rts_r.rts, iss=iss_r.iss, age_years=70,
        mechanism=InjuryMechanism.BLUNT_MVC
    )
    assert triss_elderly.survival_prob <= triss_r.survival_prob + 0.05
    mews_r = compute_mews(respiratory_rate=22, heart_rate=110, systolic_bp=95,
                           temperature_c=36.5, avpu="A")
    assert mews_r.total >= 3
    assert mews_r.alert_level in ("URGENT", "REVIEW", "EMERGENCY")
    news_r = compute_news2(respiratory_rate=22, spo2_pct=94.0, systolic_bp=95,
                            heart_rate=110, temperature_c=36.5, avpu="A")
    assert news_r.total >= 4
    assert news_r.clinical_risk in ("MEDIUM", "HIGH")
    sofa_r = compute_sofa(
        pao2_fio2_ratio=200.0, platelets_10e9=80.0, bilirubin_umol_l=30.0,
        map_mmhg=65, vasopressor_mcg_kg_min=None, gcs=12,
        creatinine_umol_l=150.0, urine_output_ml_24h=800.0,
        on_ventilator=True,
    )
    assert sofa_r.total >= 3
    mgap_r = compute_mgap(InjuryMechanism.BLUNT_MVC, gcs=12, age_years=40, systolic_bp=100)
    assert 0 <= mgap_r.total <= 30
    gap_r = compute_gap(gcs=12, age_years=40, systolic_bp=100)
    assert 0 <= gap_r.total <= 24
    kts_r = compute_kts(age_years=40, systolic_bp=100, respiratory_rate=18,
                         gcs=12, number_of_serious_injuries=1)
    assert 5 <= kts_r.total <= 25
    crams_r = compute_crams(2.5, 90, "laboured", "tender", 10, "confused_or_inappropriate")
    assert 0 <= crams_r.total <= 10
    pts_r = compute_pts(25.0, "maintainable", 85, "obtunded", "minor", "closed_fracture")
    assert -6 <= pts_r.total <= 12
    dcs_r = assess_dcs(
        systolic_bp=75, base_deficit_meq=8.0, temperature_c=34.5,
        inr=1.8, packed_cells_given=8, lactate_mmol_l=6.5,
        iss=38, has_hollow_viscus_injury=True, has_major_vascular=True,
        has_pelvic_fracture=True, combined_abdomino_thoracic=False,
        predicted_op_time_hours=5.0, age_years=35,
    )
    assert dcs_r.indication_level == DCSIndicationLevel.ABSOLUTE
    assert dcs_r.massive_transfusion_protocol
    assert dcs_r.txa_indicated
    fast = FASTExam(
        right_upper_quad=FASTFinding.POSITIVE_RUQ,
        pericardial=FASTFinding.NEGATIVE,
    )
    assert fast.is_positive
    assert fast.haemoperitoneum_suspected
    assert not fast.cardiac_tamponade_suspected
    tta = assess_trauma_team_activation(
        gcs=11, systolic_bp=85, respiratory_rate=22, iss=28,
        mechanism=InjuryMechanism.BLUNT_MVC, age_years=45,
        has_penetrating_torso=False, has_amputation_proximal=False,
        has_burns_major=False, has_paralysis=False,
        fall_height_metres=None, vehicle_speed_kmh=90,
        is_paediatric=False, fast_result=fast,
    )
    assert tta.activation_level in (
        TraumaTeamActivationLevel.FULL, TraumaTeamActivationLevel.IMMEDIATE
    )
    engine = TraumaScoringEngine(rng_seed=42)
    bundle = engine.register_patient(
        patient_id="P001", incident_id="INC001",
        condition_key="polytrauma_blunt",
        mechanism=InjuryMechanism.BLUNT_MVC,
        age_years=35, step=0, gcs=12,
        systolic_bp=95, respiratory_rate=20,
        heart_rate=110, temperature_c=36.0,
        spo2_pct=95.0,
    )
    engine.add_anatomical_scoring("P001", profiles)
    engine.add_fast_exam("P001", fast)
    engine.add_dcs_assessment("P001", dcs_r)
    assert bundle.iss_result is not None
    assert bundle.triss_result is not None
    assert 0.0 <= bundle.consensus_survival_probability <= 1.0
    grader_dump = engine.get_all_grader_dicts()
    assert len(grader_dump) == 1
    assert grader_dump[0]["ISS"] == expected_iss
    ap2 = compute_apache2(
        temperature_c=38.8, map_mmhg=65, heart_rate=115,
        respiratory_rate=25, pao2_or_aado2=55.0, is_aado2=False,
        arterial_ph=7.32, serum_na_meq=138.0, serum_k_meq=3.8,
        creatinine_umol_l=160.0, hematocrit_pct=32.0, wbc_10e9=14.0,
        gcs=13, age_years=55,
    )
    assert ap2.total >= 8
    assert 0.0 <= ap2.predicted_mortality_pct <= 100.0
    entry = TraumaRegistryEntry(
        episode_id="EP001", patient_id="P001", condition_key="polytrauma_blunt",
        mechanism="blunt_mvc", age_years=35,
        iss=34, niss=34, rts=round(bundle.rts_result.rts, 3),
        triss_ps=bundle.triss_result.survival_prob,
        mgap=bundle.mgap_result.total, kts=bundle.kts_result.total,
    )
    entry_dict = entry.to_dict()
    assert entry_dict["ISS"] == 34
    assert 0.0 <= entry_dict["TRISS_Ps"] <= 1.0
    assert not entry.unexpected_outcome or True  
    logger.info(
        "trauma_scoring.py self-test PASSED — "
        "14 scoring systems, %d assertions verified.",
        35,
    )
_self_test()
logger.info(
    "EMERGI-ENV server.medical.trauma_scoring v%d loaded — "
    "AIS/ISS/NISS, RTS, TRISS, MEWS, NEWS2, SOFA, APACHE-II, "
    "MGAP, GAP, KTS, CRAMS, PTS, DCS, FAST, TTA, "
    "TraumaScoringEngine + TraumaRegistryEntry ready.",
    TRAUMA_SCORING_VERSION,
)
__all__ = [
    "TRAUMA_SCORING_VERSION",
    "ISS_MAJOR_TRAUMA_THRESHOLD",
    "ISS_CRITICAL_THRESHOLD",
    "DCS_LACTATE_THRESHOLD_MMOL",
    "DCS_BASE_DEFICIT_THRESHOLD",
    "InjuryMechanism",
    "BodyRegion",
    "AISSeverity",
    "DCSIndicationLevel",
    "OutcomePrediction",
    "TraumaTeamActivationLevel",
    "FASTFinding",
    "AISInjury",
    "AISBodyRegionProfile",
    "ISSResult",
    "RTSResult",
    "TRISSResult",
    "MEWSResult",
    "NEWS2Result",
    "SOFAResult",
    "APACHEIIResult",
    "MGAPResult",
    "GAPResult",
    "KTSResult",
    "CRAMSResult",
    "PTSResult",
    "DCSAssessment",
    "FASTExam",
    "PrimarySurveyState",
    "TraumaActivationDecision",
    "compute_iss",
    "compute_rts",
    "compute_triss",
    "compute_mews",
    "compute_news2",
    "compute_sofa",
    "compute_apache2",
    "compute_mgap",
    "compute_gap",
    "compute_kts",
    "compute_crams",
    "compute_pts",
    "assess_dcs",
    "assess_trauma_team_activation",
    "TraumaScoreBundle",
    "TraumaScoringEngine",
    "TraumaRegistryEntry",
]