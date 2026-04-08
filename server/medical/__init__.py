from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
logger = logging.getLogger("emergi_env.medical")
MEDICAL_MODULE_VERSION: Tuple[int, int, int] = (2, 0, 0)
MEDICAL_MODULE_VERSION_STR: str = ".".join(str(x) for x in MEDICAL_MODULE_VERSION)
SCHEMA_VERSION: int = 7               
MEDICAL_MODULE_READY: bool = False        
_import_errors: Dict[str, str] = {}
try:
    from server.medical.triage import (  
        CbrnContaminationType,
        IndianEMSTriageMapper,
        JumpSTARTProtocolEngine,
        MCITriageSupervisor,
        MentalStatus,
        NACAScore,
        PatientDeteriorationTracker,
        PulseStatus,
        RPMScore,
        RespirationStatus,
        RevisedTraumaScore,
        SALTProtocolEngine,
        STARTProtocolEngine,
        TriageDecision,
        TriageEngine,
        TriageGroundTruthDatabase,
        TriageProtocol,
        TriageTag,
        TRIAGE_VERSION,
        PAEDIATRIC_AGE_THRESHOLD,
        TRIAGE_EXACT_MATCH_SCORE,
        TRIAGE_ADJACENT_MISMATCH_SCORE,
        TRIAGE_DISTANT_MISMATCH_SCORE,
        TRIAGE_CRITICAL_MISMATCH_SCORE,
        TRIAGE_CRITICAL_MISMATCH_PENALTY,
        DETERIORATION_STEPS,
        _infer_gcs_from_rpm,
        _infer_sbp_from_rpm,
        _infer_rr_from_rpm,
    )
    _TRIAGE_OK = True
    logger.debug("server.medical: triage v%d loaded", TRIAGE_VERSION)
except Exception as exc:  
    _TRIAGE_OK = False
    _import_errors["triage"] = str(exc)
    logger.error("server.medical: FAILED to import triage — %s", exc)
try:
    from server.medical.survivalcurves import (  
        CurveModelType,
        GoldenHourWindow,
        GoldenHourWindows,
        GoldenWindowType,
        PatientSurvivalState,
        SeverityTier,
        SurvivalCurveEngine,
        SurvivalCurveRegistry,
        SurvivalParameters,
        SurvivalProbabilityCalculator,
        SurvivalSnapshot,
        TransportUrgencyClass,
        SURVIVAL_CURVES_VERSION,
        SIMULATION_STEP_DURATION_MIN,
        SURVIVAL_FLOOR,
        UNDER_TRIAGE_SURVIVAL_PENALTY,
        CRITICAL_UNDER_TRIAGE_PENALTY,
        GOLDEN_HOUR_COMPLIANCE_BONUS,
    )
    _CURVES_OK = True
    logger.debug("server.medical: survival_curves v%d loaded", SURVIVAL_CURVES_VERSION)
except Exception as exc:  
    _CURVES_OK = False
    _import_errors["survival_curves"] = str(exc)
    logger.error("server.medical: FAILED to import survival_curves — %s", exc)
try:
    from server.medical.goldenhour import (  
        AlertSeverity,
        AlertType,
        ConditionGoldenHourPolicy,
        ConditionGoldenHourPolicyRegistry,
        DenseRewardCalculator,
        DispatchQualityGrade,
        EpisodeGoldenHourLedger,
        EpisodeRewardSummary,
        GoldenHourAlert,
        GoldenHourAlertQueue,
        GoldenHourComplianceGrader,
        GoldenHourEngine,
        GoldenHourPhase,
        GoldenHourTracker,
        GoldenHourViolationDetector,
        MCIGoldenHourCoordinator,
        MCIVictimState,
        PatientGoldenHourRecord,
        StepRewardComponents,
        SurvivalRewardIntegrator,
        GOLDEN_HOUR_VERSION,
        DISPATCH_LATENCY_PENALTY_PER_MIN,
        CORRECT_UNIT_TYPE_BONUS,
        GOLDEN_WINDOW_STILL_OPEN_BONUS,
        GOLDEN_WINDOW_BREACHED_PENALTY,
        P1_UNASSIGNED_STEP_PENALTY,
        P1_DISPATCH_BONUS,
        PLATINUM_10_BONUS,
        MCI_MIN_HOSPITAL_SPREAD,
        MCI_MAX_VICTIMS_PER_HOSPITAL,
        STEP_REWARD_MIN,
        STEP_REWARD_MAX,
        MAX_EPISODE_DURATION_MIN,
        ALERT_CRITICAL_SURVIVAL_THRESHOLD,
        ALERT_WARNING_SURVIVAL_THRESHOLD,
    )
    _GOLDEN_OK = True
    logger.debug("server.medical: golden_hour v%d loaded", GOLDEN_HOUR_VERSION)
except Exception as exc:  
    _GOLDEN_OK = False
    _import_errors["golden_hour"] = str(exc)
    logger.error("server.medical: FAILED to import golden_hour — %s", exc)
try:
    from server.medical.traumascoring import (  
        AISBodyRegionProfile,
        AISInjury,
        AISSeverity,
        APACHEIIResult,
        BodyRegion,
        CRAMSResult,
        DCSAssessment,
        DCSIndicationLevel,
        FASTExam,
        FASTFinding,
        GAPResult,
        ISSResult,
        InjuryMechanism,
        KTSResult,
        MGAPResult,
        MEWSResult,
        NEWS2Result,
        OutcomePrediction,
        PTSResult,
        PrimarySurveyState,
        RTSResult,
        SOFAResult,
        TRISSResult,
        TraumaActivationDecision,
        TraumaRegistryEntry,
        TraumaScoreBundle,
        TraumaScoringEngine,
        TraumaTeamActivationLevel,
        TRAUMA_SCORING_VERSION,
        ISS_MAJOR_TRAUMA_THRESHOLD,
        ISS_SEVERE_THRESHOLD,
        ISS_CRITICAL_THRESHOLD,
        DCS_LACTATE_THRESHOLD_MMOL,
        DCS_BASE_DEFICIT_THRESHOLD,
        compute_iss,
        compute_rts,
        compute_triss,
        compute_mews,
        compute_news2,
        compute_sofa,
        compute_apache2,
        compute_mgap,
        compute_gap,
        compute_kts,
        compute_crams,
        compute_pts,
        assess_dcs,
        assess_trauma_team_activation,
    )
    _TRAUMA_OK = True
    logger.debug("server.medical: trauma_scoring v%d loaded", TRAUMA_SCORING_VERSION)
except Exception as exc:  
    _TRAUMA_OK = False
    _import_errors["trauma_scoring"] = str(exc)
    logger.error("server.medical: FAILED to import trauma_scoring — %s", exc)
try:
    from server.medical.protocolchecker import (
        ProtocolCheckResult,
        ProtocolCheckBatch,
        EpisodeProtocolRecord,
        ProtocolCheckerEngine,
        ProtocolRuleDefinition,
        make_protocol_checker,
        ViolationSeverity,
        DispatchProtocolChecker,
        TriageProtocolChecker,
        HospitalRoutingChecker,
        TransferProtocolChecker,
        MutualAidProtocolChecker,
        SurgeProtocolChecker,
        MultiAgencyProtocolChecker,
        CrewFatigueProtocolChecker,
        PrepositionProtocolChecker,
        ProtocolComplianceScoreCalculator,
        PROTOCOL_CHECKER_VERSION,
        CALL_TO_DISPATCH_TARGET_MIN,
        SCENE_TIME_LOAD_AND_GO_MAX_MIN,
    )
    _PROTOCOL_OK = True
    logger.debug(
        "server.medical: protocol_checker v%d loaded", PROTOCOL_CHECKER_VERSION
    )
except Exception as exc:  
    _PROTOCOL_OK = False
    _import_errors["protocol_checker"] = str(exc)
    logger.error("server.medical: FAILED to import protocol_checker — %s", exc)
@dataclass(frozen=True)
class MedicalModuleHealth:
    triage_ok:          bool
    survival_curves_ok: bool
    golden_hour_ok:     bool
    trauma_scoring_ok:  bool
    protocol_checker_ok: bool
    import_errors:      Dict[str, str]
    schema_version:     int = SCHEMA_VERSION
    module_version:     str = MEDICAL_MODULE_VERSION_STR
    @property
    def all_ok(self) -> bool:
        return all([
            self.triage_ok,
            self.survival_curves_ok,
            self.golden_hour_ok,
            self.trauma_scoring_ok,
            self.protocol_checker_ok,
        ])
    @property
    def critical_systems_ok(self) -> bool:
        return self.triage_ok and self.survival_curves_ok and self.golden_hour_ok
    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_ok":               self.all_ok,
            "critical_systems_ok":  self.critical_systems_ok,
            "triage":               self.triage_ok,
            "survival_curves":      self.survival_curves_ok,
            "golden_hour":          self.golden_hour_ok,
            "trauma_scoring":       self.trauma_scoring_ok,
            "protocol_checker":     self.protocol_checker_ok,
            "import_errors":        self.import_errors,
            "schema_version":       self.schema_version,
            "module_version":       self.module_version,
        }
    def __str__(self) -> str:
        marks = {True: "✓", False: "✗"}
        lines = [
            f"MedicalModuleHealth v{self.module_version}",
            f"  triage          {marks[self.triage_ok]}",
            f"  survival_curves {marks[self.survival_curves_ok]}",
            f"  golden_hour     {marks[self.golden_hour_ok]}",
            f"  trauma_scoring  {marks[self.trauma_scoring_ok]}",
            f"  protocol_checker{marks[self.protocol_checker_ok]}",
        ]
        if self.import_errors:
            lines.append("  errors: " + str(self.import_errors))
        return "\n".join(lines)
MODULE_HEALTH = MedicalModuleHealth(
    triage_ok=_TRIAGE_OK,
    survival_curves_ok=_CURVES_OK,
    golden_hour_ok=_GOLDEN_OK,
    trauma_scoring_ok=_TRAUMA_OK,
    protocol_checker_ok=_PROTOCOL_OK,
    import_errors=dict(_import_errors),
)
TIMESTEP_MINUTES:       float = 3.0    
MAX_EPISODE_STEPS:      int   = 100    
NUM_ZONES:              int   = 36     
NUM_HOSPITALS:          int   = 28     
FLEET_SIZE:             int   = 24     
TASK_SEEDS: Dict[str, int] = {f"T{i}": 41 + i for i in range(1, 10)}
@unique
class PatientLifecycleState(str, Enum):
    INCIDENT_CREATED   = "incident_created"
    TRIAGE_PENDING     = "triage_pending"
    TRIAGE_COMPLETE    = "triage_complete"
    AWAITING_DISPATCH  = "awaiting_dispatch"
    UNIT_DISPATCHED    = "unit_dispatched"
    UNIT_ON_SCENE      = "unit_on_scene"
    PATIENT_LOADED     = "patient_loaded"
    IN_TRANSPORT       = "in_transport"
    ARRIVED_HOSPITAL   = "arrived_hospital"
    TREATMENT_STARTED  = "treatment_started"
    TREATMENT_COMPLETE = "treatment_complete"
    TRANSFERRED        = "transferred"
    DISCHARGED         = "discharged"
    DECEASED           = "deceased"
    EXPECTANT_COMFORT  = "expectant_comfort"
    @property
    def is_terminal(self) -> bool:
        return self in (
            PatientLifecycleState.TREATMENT_COMPLETE,
            PatientLifecycleState.TRANSFERRED,
            PatientLifecycleState.DISCHARGED,
            PatientLifecycleState.DECEASED,
            PatientLifecycleState.EXPECTANT_COMFORT,
        )
    @property
    def reward_emitting(self) -> bool:
        return not self.is_terminal and self != PatientLifecycleState.INCIDENT_CREATED
     
_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    PatientLifecycleState.INCIDENT_CREATED:   frozenset([PatientLifecycleState.TRIAGE_PENDING]),
    PatientLifecycleState.TRIAGE_PENDING:     frozenset([
        PatientLifecycleState.TRIAGE_COMPLETE,
        PatientLifecycleState.DECEASED,
        PatientLifecycleState.EXPECTANT_COMFORT,
    ]),
    PatientLifecycleState.TRIAGE_COMPLETE:    frozenset([PatientLifecycleState.AWAITING_DISPATCH, PatientLifecycleState.UNIT_DISPATCHED]),
    PatientLifecycleState.AWAITING_DISPATCH:  frozenset([
        PatientLifecycleState.UNIT_DISPATCHED,
        PatientLifecycleState.DECEASED,
    ]),
    PatientLifecycleState.UNIT_DISPATCHED:    frozenset([
        PatientLifecycleState.UNIT_ON_SCENE,
        PatientLifecycleState.DECEASED,
    ]),
    PatientLifecycleState.UNIT_ON_SCENE:      frozenset([
        PatientLifecycleState.PATIENT_LOADED,
        PatientLifecycleState.TREATMENT_STARTED,  
        PatientLifecycleState.DECEASED,
    ]),
    PatientLifecycleState.PATIENT_LOADED:     frozenset([PatientLifecycleState.IN_TRANSPORT]),
    PatientLifecycleState.IN_TRANSPORT:       frozenset([
        PatientLifecycleState.ARRIVED_HOSPITAL,
        PatientLifecycleState.DECEASED,
    ]),
    PatientLifecycleState.ARRIVED_HOSPITAL:   frozenset([PatientLifecycleState.TREATMENT_STARTED]),
    PatientLifecycleState.TREATMENT_STARTED:  frozenset([
        PatientLifecycleState.TREATMENT_COMPLETE,
        PatientLifecycleState.TRANSFERRED,
        PatientLifecycleState.DECEASED,
    ]),
    PatientLifecycleState.TREATMENT_COMPLETE: frozenset([PatientLifecycleState.DISCHARGED]),
    PatientLifecycleState.TRANSFERRED:        frozenset([
        PatientLifecycleState.TREATMENT_STARTED,
        PatientLifecycleState.DECEASED,
    ]),
    PatientLifecycleState.DISCHARGED:         frozenset(),
    PatientLifecycleState.DECEASED:           frozenset(),
    PatientLifecycleState.EXPECTANT_COMFORT:  frozenset([PatientLifecycleState.DECEASED]),
}
PATIENT_LIFECYCLE_VALID_TRANSITIONS = _TRANSITIONS
@dataclass
class UnifiedPatientRecord:
    patient_id:       str
    incident_id:      str
    condition_key:    str
    victim_index:     int = 0
    age_years:        Optional[int]  = None
    gender:           str            = "unknown"
    is_paediatric:    bool           = False
    is_mci_victim:    bool           = False
    lifecycle_state:  PatientLifecycleState = PatientLifecycleState.INCIDENT_CREATED
    step_registered:  int = 0
    step_triaged:     Optional[int] = None
    step_dispatched:  Optional[int] = None
    step_on_scene:    Optional[int] = None
    step_at_hospital: Optional[int] = None
    step_treated:     Optional[int] = None
    triage_tag:        Optional["TriageTag"]      = None
    ground_truth_tag:  Optional["TriageTag"]      = None
    triage_correct:    Optional[bool]             = None
    triage_critical_mismatch: bool               = False
    triage_protocol_used: Optional[str]          = None
    rpm:               Optional["RPMScore"]       = None
    severity:          str = "P1"
    rts_score:         Optional[float]    = None
    iss_score:         Optional[int]      = None
    niss_score:        Optional[int]      = None
    triss_ps:          Optional[float]    = None
    mews_score:        Optional[int]      = None
    news2_score:       Optional[int]      = None
    mgap_score:        Optional[int]      = None
    dispatched_unit_id:   Optional[str]  = None
    dispatched_unit_type: Optional[str]  = None
    destination_hospital: Optional[str]  = None
    unit_type_correct:    Optional[bool] = None
    dispatch_latency_min: float          = 0.0
    cath_lab_activated:       bool = False
    stroke_unit_notified:     bool = False
    trauma_activation_sent:   bool = False
    multi_agency_coordinated: bool = False
    routed_to_diverted_hospital: bool = False
    pre_hospital_intervention: Optional[str] = None
    survival_prob_initial:     float         = 1.0
    survival_prob_current:     float         = 1.0
    survival_prob_at_dispatch: Optional[float] = None
    survival_prob_at_treatment: Optional[float] = None
    survival_integral:         float         = 0.0
    elapsed_minutes:           float         = 0.0
    golden_window_compliant:   Optional[bool] = None
    platinum_window_achieved:  bool          = False
    time_to_irreversible_remaining: float    = 999.0
    protocol_compliance_score: float = 0.0
    protocol_violations:       List[str] = field(default_factory=list)
    protocol_bonuses_earned:   List[str] = field(default_factory=list)
    cumulative_reward:         float = 0.0
    cumulative_protocol_bonus: float = 0.0
    cumulative_penalty:        float = 0.0
    cbrn_type:             str  = "none"
    decontamination_done:  bool = False
    outcome:               Optional[str] = None   
    unexpected_outcome:    bool          = False
    def transition_to(self, new_state: PatientLifecycleState) -> bool:
        allowed = PATIENT_LIFECYCLE_VALID_TRANSITIONS.get(self.lifecycle_state, frozenset())
        if new_state not in allowed:
            logger.warning(
                "UnifiedPatientRecord %s: invalid transition %s → %s",
                self.patient_id, self.lifecycle_state.value, new_state.value,
            )
            return False
        self.lifecycle_state = new_state
        return True
    @property
    def is_p1_critical(self) -> bool:
        return self.severity == "P1"
    @property
    def is_terminal(self) -> bool:
        return self.lifecycle_state.is_terminal
    @property
    def total_reward(self) -> float:
        return self.cumulative_reward + self.cumulative_protocol_bonus + self.cumulative_penalty
    @property
    def dispatch_latency_steps(self) -> Optional[int]:
        if self.step_dispatched is None:
            return None
        return self.step_dispatched - self.step_registered
    def to_observation_dict(self) -> Dict[str, Any]:
        return {
            "patient_id":                  self.patient_id,
            "incident_id":                 self.incident_id,
            "condition_key":               self.condition_key,
            "victim_index":                self.victim_index,
            "age_years":                   self.age_years,
            "is_paediatric":               self.is_paediatric,
            "lifecycle_state":             self.lifecycle_state.value,
            "triage_tag":                  self.triage_tag.value if self.triage_tag else None,
            "triage_colour":               self.triage_tag.colour if self.triage_tag else None,
            "indian_ems_priority":         self.triage_tag.indian_ems_priority if self.triage_tag else None,
            "severity":                    self.severity,
            "dispatched_unit_type":        self.dispatched_unit_type,
            "destination_hospital":        self.destination_hospital,
            "unit_type_correct":           self.unit_type_correct,
            "dispatch_latency_min":        round(self.dispatch_latency_min, 2),
            "survival_prob_current":       round(self.survival_prob_current, 4),
            "elapsed_minutes":             round(self.elapsed_minutes, 2),
            "golden_window_compliant":     self.golden_window_compliant,
            "time_to_irreversible_min":    round(self.time_to_irreversible_remaining, 1),
            "cath_lab_activated":          self.cath_lab_activated,
            "stroke_unit_notified":        self.stroke_unit_notified,
            "trauma_activation_sent":      self.trauma_activation_sent,
            "protocol_compliance_score":   round(self.protocol_compliance_score, 4),
            "cbrn_type":                   self.cbrn_type,
            "decontamination_done":        self.decontamination_done,
        }
    def to_grader_dict(self) -> Dict[str, Any]:
        d = self.to_observation_dict()
        d.update({
            "ground_truth_tag":            self.ground_truth_tag.value if self.ground_truth_tag else None,
            "triage_correct":              self.triage_correct,
            "triage_critical_mismatch":    self.triage_critical_mismatch,
            "survival_prob_initial":       round(self.survival_prob_initial, 4),
            "survival_prob_at_dispatch":   (
                round(self.survival_prob_at_dispatch, 4)
                if self.survival_prob_at_dispatch is not None else None
            ),
            "survival_prob_at_treatment":  (
                round(self.survival_prob_at_treatment, 4)
                if self.survival_prob_at_treatment is not None else None
            ),
            "survival_integral":           round(self.survival_integral, 4),
            "platinum_window_achieved":    self.platinum_window_achieved,
            "outcome":                     self.outcome,
            "unexpected_outcome":          self.unexpected_outcome,
            "total_reward":                round(self.total_reward, 4),
            "rts_score":                   self.rts_score,
            "iss_score":                   self.iss_score,
            "triss_ps":                    self.triss_ps,
            "protocol_violations":         self.protocol_violations,
            "protocol_bonuses_earned":     self.protocol_bonuses_earned,
        })
        return d
@unique
class IncidentType(str, Enum):
    SINGLE_PATIENT     = "single_patient"
    MULTI_CASUALTY     = "multi_casualty"          
    MASS_CASUALTY      = "mass_casualty"           
    OBSTETRIC          = "obstetric"
    CARDIAC            = "cardiac"
    STROKE             = "stroke"
    TRAUMA_PENETRATING = "trauma_penetrating"
    TRAUMA_BLUNT       = "trauma_blunt"
    HAZMAT_CBRN        = "hazmat_cbrn"
    INDUSTRIAL         = "industrial"
    NATURAL_DISASTER   = "natural_disaster"
    WATER_RESCUE       = "water_rescue"
    NEONATAL           = "neonatal"
    PSYCHIATRIC        = "psychiatric"
@dataclass
class IncidentRecord:
    incident_id:      str
    incident_type:    IncidentType
    zone_id:          str
    grid_x:           int
    grid_y:           int
    step_reported:    int
    patient_ids:      List[str]                     = field(default_factory=list)
    is_mci:           bool                          = False
    mci_victim_count: int                           = 0
    multi_agency_required: bool                     = False
    police_requested: bool                          = False
    fire_requested:   bool                          = False
    ndrf_requested:   bool                          = False
    surge_activated:  bool                          = False
    closed:           bool                          = False
    step_closed:      Optional[int]                 = None
    @property
    def patient_count(self) -> int:
        return len(self.patient_ids)
    @property
    def is_mass_casualty(self) -> bool:
        return self.patient_count >= 10 or self.mci_victim_count >= 10
class MedicalEngine:
    def __init__(
        self,
        episode_id: str,
        task_id:    str  = "T1",
        rng_seed:   int  = 42,
    ) -> None:
        self.episode_id = episode_id
        self.task_id    = task_id
        self.rng_seed   = rng_seed
        if _TRIAGE_OK:
            self._triage_engine = TriageEngine(rng_seed=rng_seed)
        if _CURVES_OK:
            self._survival_engine = SurvivalCurveEngine(rng_seed=rng_seed)
        if _GOLDEN_OK:
            self._golden_engine = GoldenHourEngine(
                episode_id=episode_id,
                task_id=task_id,
                rng_seed=rng_seed,
            )
        if _TRAUMA_OK:
            self._trauma_engine = TraumaScoringEngine(rng_seed=rng_seed)
        if _PROTOCOL_OK:
            self._protocol_checker = make_protocol_checker(self.episode_id, self.task_id)
        self._patients:   Dict[str, UnifiedPatientRecord] = {}
        self._incidents:  Dict[str, IncidentRecord]       = {}
        self._current_step:      int   = 0
        self._episode_start_ts:  float = time.monotonic()
        self._step_rewards:      List[Tuple[int, float]] = []
        self._cascade_failure:   bool  = False
        self._surge_declared:    bool  = False
        logger.info(
            "MedicalEngine: episode=%s task=%s seed=%d | modules=%s",
            episode_id, task_id, rng_seed,
            "T+S+G+R+P" if MODULE_HEALTH.all_ok else str(MODULE_HEALTH.import_errors),
        )
    def register_incident(
        self,
        incident_id:        str,
        incident_type:      Union[IncidentType, str],
        zone_id:            str,
        grid_x:             int,
        grid_y:             int,
        step:               int,
        multi_agency:       bool = False,
    ) -> IncidentRecord:
        if isinstance(incident_type, str):
            incident_type = IncidentType(incident_type)
        rec = IncidentRecord(
            incident_id=incident_id,
            incident_type=incident_type,
            zone_id=zone_id,
            grid_x=grid_x,
            grid_y=grid_y,
            step_reported=step,
            multi_agency_required=multi_agency,
        )
        self._incidents[incident_id] = rec
        logger.debug(
            "MedicalEngine: registered incident %s [%s] zone=%s step=%d",
            incident_id, incident_type.value, zone_id, step,
        )
        return rec
    def register_patient(
        self,
        patient_id:    str,
        incident_id:   str,
        condition_key: str,
        severity:      str,
        step:          int,
        rpm:           Optional["RPMScore"]  = None,
        age_years:     Optional[int]         = None,
        gender:        str                   = "unknown",
        is_paediatric: bool                  = False,
        is_mci:        bool                  = False,
        victim_index:  int                   = 0,
        time_offset_min: float               = 0.0,
    ) -> UnifiedPatientRecord:
        gt_tag = None
        if _TRIAGE_OK:
            gt_tag = self._triage_engine.ground_truth_tag(condition_key)
        survival_prob_initial = 0.80
        if _CURVES_OK:
            params = SurvivalCurveRegistry.get(condition_key)
            survival_prob_initial = params.initial_survival_prob
        rec = UnifiedPatientRecord(
            patient_id=patient_id,
            incident_id=incident_id,
            condition_key=condition_key,
            victim_index=victim_index,
            age_years=age_years,
            gender=gender,
            is_paediatric=is_paediatric,
            is_mci_victim=is_mci,
            severity=severity,
            step_registered=step,
            lifecycle_state=PatientLifecycleState.TRIAGE_PENDING,
            ground_truth_tag=gt_tag,
            rpm=rpm,
            survival_prob_initial=survival_prob_initial,
            survival_prob_current=survival_prob_initial,
        )
        self._patients[patient_id] = rec
        if _CURVES_OK:
            self._survival_engine.register_patient(
                patient_id=patient_id,
                incident_id=incident_id,
                condition_key=condition_key,
                severity=severity,
                age_years=age_years,
                step_registered=step,
                time_registered_min=time_offset_min,
            )
        if _GOLDEN_OK:
            self._golden_engine.register_patient(
                patient_id=patient_id,
                incident_id=incident_id,
                condition_key=condition_key,
                severity=severity,
                step=step,
                time_offset_min=time_offset_min,
                age_years=age_years,
                is_paediatric=is_paediatric,
                is_mci=is_mci,
                victim_index=victim_index,
                rpm=rpm,
                ground_truth_tag=gt_tag,
            )
        if incident_id in self._incidents:
            self._incidents[incident_id].patient_ids.append(patient_id)
        logger.debug(
            "MedicalEngine: registered patient %s [%s] sev=%s P0=%.3f",
            patient_id, condition_key, severity, survival_prob_initial,
        )
        return rec
    def apply_triage(
        self,
        patient_id:    str,
        rpm:           "RPMScore",
        condition_key: Optional[str] = None,
        step:          int           = 0,
        use_salt:      bool          = False,
        cbrn_type:     str           = "none",
    ) -> Tuple[Optional["TriageDecision"], float, str]:
        rec = self._patients.get(patient_id)
        if rec is None or not _TRIAGE_OK:
            return None, 0.0, "patient_not_found_or_triage_unavailable"
        ck = condition_key or rec.condition_key
        decision = self._triage_engine.triage_single(
            patient_id=patient_id,
            incident_id=rec.incident_id,
            rpm=rpm,
            condition_key=ck,
            victim_index=rec.victim_index,
            step=step,
        )
        gt_tag = rec.ground_truth_tag
        is_correct = (
            decision.assigned_tag == gt_tag if gt_tag else None
        )
        is_critical_mismatch = self._triage_engine.is_critical_triage_mismatch(
            decision.assigned_tag.value, ck
        )
        rec.triage_tag              = decision.assigned_tag
        rec.triage_protocol_used    = decision.protocol_used.value
        rec.triage_correct          = is_correct
        rec.triage_critical_mismatch = is_critical_mismatch
        rec.step_triaged            = step
        rec.rpm                     = rpm
        score, explanation = self._triage_engine.score_triage_decision(
            decision.assigned_tag.value, ck
        )
        if is_critical_mismatch:
            score = max(0.0, score + TRIAGE_CRITICAL_MISMATCH_PENALTY)
            rec.cumulative_penalty += abs(TRIAGE_CRITICAL_MISMATCH_PENALTY)
        rec.protocol_compliance_score += score * 0.05   
        rec.transition_to(PatientLifecycleState.TRIAGE_COMPLETE)
        if _GOLDEN_OK:
            self._golden_engine.mark_triage(
                patient_id=patient_id,
                assigned_tag=decision.assigned_tag,
                is_correct=bool(is_correct),
                critical_mismatch=is_critical_mismatch,
            )
        rts = self._triage_engine.compute_rts(rpm)
        rec.rts_score = rts.rts
        logger.debug(
            "MedicalEngine.apply_triage: %s → %s (correct=%s mismatch=%s) score=%.3f",
            patient_id, decision.assigned_tag.value, is_correct, is_critical_mismatch, score,
        )
        return decision, score, explanation
    def apply_dispatch(
        self,
        patient_id:    str,
        unit_id:       str,
        unit_type:     str,
        hospital_id:   str,
        step:          int,
        cath_lab_activated:      bool = False,
        stroke_unit_notified:    bool = False,
        trauma_activation_sent:  bool = False,
        multi_agency_coordinated: bool = False,
        routed_to_diverted:      bool = False,
    ) -> Tuple[float, str]:
        rec = self._patients.get(patient_id)
        if rec is None:
            return 0.0, "patient_not_found"
        dispatch_quality = DispatchQualityGrade.GOOD
        unit_type_correct = None
        if _CURVES_OK and _GOLDEN_OK:
            params = SurvivalCurveRegistry.get(rec.condition_key)
            sev_tier = SeverityTier(rec.severity) if rec.severity in {
                t.value for t in SeverityTier
            } else SeverityTier.P1
            calc = DenseRewardCalculator()
            dispatch_quality = calc.compute_dispatch_quality_grade(
                dispatched_unit_type=unit_type,
                condition_key=rec.condition_key,
                severity=sev_tier,
                elapsed_at_dispatch=rec.elapsed_minutes,
                params=params,
            )
            unit_type_correct = unit_type.upper() in params.correct_unit_types
        rec.dispatched_unit_id         = unit_id
        rec.dispatched_unit_type       = unit_type.upper()
        rec.destination_hospital       = hospital_id
        rec.unit_type_correct          = unit_type_correct
        rec.dispatch_latency_min       = (step - rec.step_registered) * TIMESTEP_MINUTES
        rec.cath_lab_activated         = cath_lab_activated
        rec.stroke_unit_notified       = stroke_unit_notified
        rec.trauma_activation_sent     = trauma_activation_sent
        rec.multi_agency_coordinated   = multi_agency_coordinated
        rec.routed_to_diverted_hospital = routed_to_diverted
        rec.survival_prob_at_dispatch  = rec.survival_prob_current
        rec.step_dispatched            = step
        rec.transition_to(PatientLifecycleState.UNIT_DISPATCHED)
        if _GOLDEN_OK:
            self._golden_engine.mark_dispatched(
                patient_id=patient_id,
                unit_id=unit_id,
                unit_type=unit_type,
                hospital_id=hospital_id,
                step=step,
                dispatch_quality=dispatch_quality,
                unit_type_correct=unit_type_correct,
                cath_lab_activated=cath_lab_activated,
                stroke_unit_notified=stroke_unit_notified,
                trauma_activation_sent=trauma_activation_sent,
                multi_agency_coordinated=multi_agency_coordinated,
                routed_to_diverted=routed_to_diverted,
            )
        if _CURVES_OK:
            self._survival_engine.mark_dispatched(
                patient_id=patient_id, unit_type=unit_type, step=step
            )
        protocol_score = 0.0
        explanation    = "no_protocol_checker"
        if _PROTOCOL_OK:
            result = self._protocol_checker.check_dispatch_action(
                incident_id=rec.incident_id,
                condition_key=rec.condition_key,
                severity=rec.severity,
                dispatched_unit_type=unit_type,
                hospital_id=hospital_id,
                hospital_on_diversion=routed_to_diverted,
                cath_lab_activated=cath_lab_activated,
                stroke_unit_notified=stroke_unit_notified,
                trauma_activation_sent=trauma_activation_sent,
                elapsed_minutes=rec.elapsed_minutes,
                step=step,
                unit_id=unit_id,
                police_on_scene=multi_agency_coordinated,
                fire_on_scene=multi_agency_coordinated,
            )
            protocol_score = result.total_reward_delta
            explanation    = result.summary_str() if result.results else "no_checks"
            rec.protocol_compliance_score += result.compliance_rate * 0.3
            rec.cumulative_protocol_bonus += result.total_bonus_earned
            rec.cumulative_penalty        += abs(result.total_penalty_incurred)
            for r in result.results:
                if r.is_violation:
                    rec.protocol_violations.append(r.rule_key)
                if r.reward_delta > 0:
                    rec.protocol_bonuses_earned.append(r.rule_key)
        logger.debug(
            "MedicalEngine.apply_dispatch: %s → %s/%s quality=%s correct=%s score=%.3f",
            patient_id, unit_id, hospital_id,
            dispatch_quality.value if _GOLDEN_OK else "?",
            unit_type_correct, protocol_score,
        )
        return protocol_score, explanation
    def mark_on_scene(self, patient_id: str, step: int) -> None:
        rec = self._patients.get(patient_id)
        if rec:
            rec.step_on_scene = step
            rec.transition_to(PatientLifecycleState.UNIT_ON_SCENE)
        if _GOLDEN_OK:
            self._golden_engine.mark_on_scene(patient_id, step)
    def mark_transporting(self, patient_id: str, step: int) -> None:
        rec = self._patients.get(patient_id)
        if rec:
            rec.transition_to(PatientLifecycleState.IN_TRANSPORT)
        if _GOLDEN_OK:
            self._golden_engine.mark_transporting(patient_id, step)
    def mark_treated(
        self,
        patient_id:  str,
        hospital_id: str,
        step:        int,
        outcome:     str = "survived",
    ) -> None:
        rec = self._patients.get(patient_id)
        if rec:
            rec.step_treated                = step
            rec.survival_prob_at_treatment  = rec.survival_prob_current
            rec.outcome                     = outcome
            rec.transition_to(PatientLifecycleState.TREATMENT_STARTED)
            rec.transition_to(PatientLifecycleState.TREATMENT_COMPLETE)
            if rec.triss_ps is not None:
                survived = outcome == "survived"
                rec.unexpected_outcome = (
                    (survived and rec.triss_ps < 0.25)
                    or (not survived and rec.triss_ps > 0.75)
                )
        if _CURVES_OK:
            self._survival_engine.mark_treated(patient_id, hospital_id, step)
        if _GOLDEN_OK:
            self._golden_engine.mark_treated(patient_id, hospital_id, step)
        if _TRIAGE_OK:
            self._triage_engine.mark_treated(patient_id)
        logger.debug(
            "MedicalEngine.mark_treated: %s at %s step=%d outcome=%s",
            patient_id, hospital_id, step, outcome,
        )
    def mark_deceased(self, patient_id: str, step: int) -> None:
        rec = self._patients.get(patient_id)
        if rec:
            rec.outcome = "deceased"
            rec.lifecycle_state = PatientLifecycleState.DECEASED
        if _CURVES_OK:
            self._survival_engine.mark_deceased(patient_id)
    def tick(self, step: int) -> Tuple[float, List["GoldenHourAlert"]]:
        self._current_step = step
        if _CURVES_OK:
            survival_updates = self._survival_engine.update_all(step)
            for pid, new_prob in survival_updates.items():
                rec = self._patients.get(pid)
                if rec and not rec.is_terminal:
                    old_prob = rec.survival_prob_current
                    rec.survival_prob_current = new_prob
                    rec.elapsed_minutes = (
                        (step - rec.step_registered) * TIMESTEP_MINUTES
                    )
                    rec.survival_integral += 0.5 * (old_prob + new_prob) * TIMESTEP_MINUTES
                    if _CURVES_OK:
                        params = SurvivalCurveRegistry.get(rec.condition_key)
                        rec.time_to_irreversible_remaining = max(
                            0.0,
                            params.time_to_irreversible_min - rec.elapsed_minutes,
                        )
        if _TRIAGE_OK:
            escalations = self._triage_engine.tick(step)
            for pid, old_tag, new_tag in escalations:
                rec = self._patients.get(pid)
                if rec:
                    rec.triage_tag = new_tag
                    logger.warning(
                        "MedicalEngine: patient %s deteriorated %s→%s at step %d",
                        pid, old_tag.value, new_tag.value, step,
                    )
        step_reward = 0.0
        new_alerts  = []
        if _GOLDEN_OK:
            step_reward, new_alerts = self._golden_engine.tick(step)
        self._step_rewards.append((step, step_reward))
        if not self._cascade_failure:
            self._check_cascade_failure()
        return step_reward, new_alerts
    def _check_cascade_failure(self) -> None:
        p1_patients = [
            r for r in self._patients.values()
            if r.severity == "P1" and not r.is_terminal
        ]
        if len(p1_patients) < 3:
            return
        unassigned = [r for r in p1_patients if r.dispatched_unit_id is None]
        fraction = len(unassigned) / len(p1_patients)
        if fraction >= 0.80:
            self._cascade_failure = True
            if _GOLDEN_OK:
                self._golden_engine.set_cascade_failure(True)
            logger.error(
                "MedicalEngine: CASCADE FAILURE — %.0f%% of P1 patients unassigned "
                "(step=%d)",
                fraction * 100, self._current_step,
            )
    def declare_surge(self, step: int, requesting_agency: str = "108") -> float:
        self._surge_declared = True
        if _PROTOCOL_OK:
            result = self._protocol_checker.check_surge_action(
                incident_ids=list(self._incidents.keys()),
                cascade_risk_fraction=self._compute_cascade_risk(),
                mutual_aid_activated=True,
                step=step,
                requesting_agency=requesting_agency,
            )
            bonus = result.bonus
        else:
            bonus = 0.025
        if self._cascade_failure and self._surge_declared:
            self._cascade_failure = False
            if _GOLDEN_OK:
                self._golden_engine.set_cascade_failure(False)
            logger.info("MedicalEngine: cascade failure CLEARED by surge declaration")
        return bonus
    def _compute_cascade_risk(self) -> float:
        p1 = [r for r in self._patients.values() if r.severity == "P1" and not r.is_terminal]
        if not p1:
            return 0.0
        unassigned = [r for r in p1 if r.dispatched_unit_id is None]
        return len(unassigned) / max(1, len(p1))
    def close_episode(self) -> Dict[str, Any]:
        ledger_dict: Dict[str, Any] = {
            "episode_id":        self.episode_id,
            "task_id":           self.task_id,
            "schema_version":    SCHEMA_VERSION,
            "total_steps":       self._current_step,
            "total_patients":    len(self._patients),
            "total_incidents":   len(self._incidents),
            "cascade_failure":   self._cascade_failure,
            "surge_declared":    self._surge_declared,
            "module_health":     MODULE_HEALTH.to_dict(),
            "elapsed_real_sec":  round(time.monotonic() - self._episode_start_ts, 3),
        }
        if _GOLDEN_OK:
            gh_ledger = self._golden_engine.close_episode()
            ledger_dict["golden_hour_ledger"] = gh_ledger.as_grader_input()
        else:
            ledger_dict["golden_hour_ledger"] = {}
        if _TRAUMA_OK:
            ledger_dict["trauma_aggregate"] = (
                self._trauma_engine.episode_aggregate_trauma_score()
            )
        ledger_dict["patients"] = [
            r.to_grader_dict() for r in self._patients.values()
        ]
        if _PROTOCOL_OK:
            proto_summary = self._protocol_checker.episode_summary()
            ledger_dict["protocol_summary"] = proto_summary.to_dict()
        else:
            ledger_dict["protocol_summary"] = {}
        p1_treated = [
            r for r in self._patients.values()
            if r.severity == "P1" and r.outcome == "survived"
        ]
        p1_total = [r for r in self._patients.values() if r.severity == "P1"]
        ledger_dict["p1_survival_rate"] = (
            len(p1_treated) / max(1, len(p1_total))
        )
        total_reward = sum(r.total_reward for r in self._patients.values())
        max_possible = sum(r.survival_prob_initial for r in self._patients.values())
        ledger_dict["normalised_episode_reward"] = round(
            min(1.0, max(0.0, total_reward / max(1.0, max_possible))), 4
        )
        ledger_dict["cascade_failure_penalty"] = -0.30 if self._cascade_failure else 0.0
        logger.info(
            "MedicalEngine.close_episode: ep=%s reward=%.4f p1_survival=%.2f%% "
            "patients=%d cascade=%s",
            self.episode_id,
            ledger_dict["normalised_episode_reward"],
            ledger_dict["p1_survival_rate"] * 100,
            len(self._patients),
            self._cascade_failure,
        )
        return ledger_dict
    def get_patient(self, patient_id: str) -> Optional[UnifiedPatientRecord]:
        return self._patients.get(patient_id)
    def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        return self._incidents.get(incident_id)
    def get_alerts(self) -> List[Dict[str, Any]]:
        if _GOLDEN_OK:
            return self._golden_engine.get_alerts_for_observation()
        return []
    def get_survival_prob(self, patient_id: str) -> float:
        rec = self._patients.get(patient_id)
        return rec.survival_prob_current if rec else 0.5
    def get_all_observation_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_observation_dict() for r in self._patients.values()]
    def active_p1_patients(self) -> List[UnifiedPatientRecord]:
        return [
            r for r in self._patients.values()
            if r.severity == "P1" and not r.is_terminal
        ]
    def unassigned_p1_patients(self) -> List[UnifiedPatientRecord]:
        return [
            r for r in self._patients.values()
            if r.severity == "P1"
            and not r.is_terminal
            and r.dispatched_unit_id is None
        ]
    def current_stats(self) -> Dict[str, Any]:
        all_recs = list(self._patients.values())
        active_p1 = self.active_p1_patients()
        return {
            "step":              self._current_step,
            "total_patients":    len(all_recs),
            "active_p1":         len(active_p1),
            "unassigned_p1":     len(self.unassigned_p1_patients()),
            "treated":           sum(1 for r in all_recs if r.outcome == "survived"),
            "deceased":          sum(1 for r in all_recs if r.outcome == "deceased"),
            "critical_alerts":   (
                self._golden_engine.get_critical_alerts_count() if _GOLDEN_OK else 0
            ),
            "cascade_risk":      round(self._compute_cascade_risk(), 3),
            "cascade_failure":   self._cascade_failure,
            "surge_declared":    self._surge_declared,
            "avg_p1_survival":   round(
                sum(r.survival_prob_current for r in active_p1) / max(1, len(active_p1)), 4
            ),
        }
    def reset(
        self,
        episode_id: Optional[str] = None,
        task_id:    Optional[str] = None,
        rng_seed:   Optional[int] = None,
    ) -> None:
        self.episode_id    = episode_id or str(uuid.uuid4())
        self.task_id       = task_id    or self.task_id
        self.rng_seed      = rng_seed   or self.rng_seed
        self._patients.clear()
        self._incidents.clear()
        self._current_step       = 0
        self._episode_start_ts   = time.monotonic()
        self._step_rewards.clear()
        self._cascade_failure    = False
        self._surge_declared     = False
        if _TRIAGE_OK:
            self._triage_engine.reset()
        if _CURVES_OK:
            self._survival_engine.reset()
        if _GOLDEN_OK:
            self._golden_engine.reset(episode_id=self.episode_id, task_id=self.task_id)
        if _TRAUMA_OK:
            self._trauma_engine.reset()
        if _PROTOCOL_OK:
            self._protocol_checker.reset()
        logger.debug("MedicalEngine.reset() → episode=%s", self.episode_id)
def get_module_registry_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if _CURVES_OK:
        stats["survival_registry"] = SurvivalCurveRegistry.registry_stats()
    if _TRIAGE_OK:
        stats["triage_conditions"] = len(TriageGroundTruthDatabase._GROUND_TRUTH)
    if _GOLDEN_OK:
        stats["gh_policy_count"] = len(ConditionGoldenHourPolicyRegistry.all_keys())
    if _PROTOCOL_OK:
        stats["protocol_rules"] = len(ClinicalGuidelineDatabase.all_rules())
    return stats
def get_condition_keys() -> List[str]:
    if _CURVES_OK:
        return SurvivalCurveRegistry.all_keys()
    if _TRIAGE_OK:
        return TriageGroundTruthDatabase.all_condition_keys()
    return []
def _self_test() -> None:
    t0 = time.monotonic()
    assert MODULE_HEALTH.critical_systems_ok, (
        f"Critical medical systems not loaded: {MODULE_HEALTH.import_errors}"
    )
    engine = MedicalEngine(episode_id="self-test-001", task_id="T1", rng_seed=42)
    engine.register_incident(
        incident_id="INC001",
        incident_type=IncidentType.CARDIAC,
        zone_id="Z01", grid_x=5, grid_y=3, step=0,
    )
    rpm_stemi = RPMScore(
        respirations="normal", pulse="present_weak",
        mental_status="alert", heart_rate=48, systolic_bp=90, gcs_score=15,
    )
    rec = engine.register_patient(
        patient_id="P001", incident_id="INC001",
        condition_key="stemi_anterior", severity="P1", step=0,
        rpm=rpm_stemi, age_years=58,
    )
    assert rec.survival_prob_initial > 0.5, "STEMI initial survival should be >50%"
    rpm_minor = RPMScore(
        respirations="normal", pulse="present_strong",
        mental_status="alert", can_walk=True,
    )
    engine.register_patient(
        patient_id="P002", incident_id="INC001",
        condition_key="minor_trauma", severity="P3", step=0,
        rpm=rpm_minor, age_years=25,
    )
    d1, score1, _ = engine.apply_triage("P001", rpm_stemi, step=0)
    assert d1 is not None
    assert d1.assigned_tag == TriageTag.IMMEDIATE, f"STEMI triage wrong: {d1.assigned_tag}"
    assert score1 > 0.0
    d2, score2, _ = engine.apply_triage("P002", rpm_minor, step=0)
    assert d2 is not None
    assert d2.assigned_tag == TriageTag.MINIMAL
    reward_s1, alerts_s1 = engine.tick(step=1)
    assert any(
        a.alert_type == AlertType.P1_UNASSIGNED for a in alerts_s1
    ), "Should fire P1_UNASSIGNED for unassigned P1 at step 1"
    p_score, expl = engine.apply_dispatch(
        patient_id="P001", unit_id="MICU-01", unit_type="MICU",
        hospital_id="H01", step=1,
        cath_lab_activated=True,
    )
    assert 0.0 <= p_score <= 1.0
    for s in range(2, 5):
        engine.tick(step=s)
    engine.mark_on_scene("P001", step=2)
    engine.mark_transporting("P001", step=3)
    engine.mark_treated("P001", "H01", step=4)
    engine.mark_treated("P002", "H03", step=4)
    rec1 = engine.get_patient("P001")
    assert rec1 is not None
    assert rec1.outcome == "survived"
    assert rec1.cath_lab_activated is True
    assert rec1.lifecycle_state == PatientLifecycleState.TREATMENT_COMPLETE
    stats = engine.current_stats()
    assert stats["treated"] == 2
    assert stats["active_p1"] == 0
    ledger = engine.close_episode()
    assert 0.0 <= ledger["normalised_episode_reward"] <= 1.0
    assert ledger["p1_survival_rate"] == 1.0
    assert not ledger["cascade_failure"]
    engine2 = MedicalEngine(episode_id="self-test-002", task_id="T4", rng_seed=43)
    for i in range(5):
        engine2.register_patient(
            patient_id=f"PX{i:03d}", incident_id="INCX",
            condition_key="cardiac_arrest_vf", severity="P1", step=0,
        )
    engine2.tick(step=1)
    engine2.tick(step=2)
    assert engine2._compute_cascade_risk() == 1.0
    engine2._check_cascade_failure()
    assert engine2._cascade_failure, "Should trigger cascade with 100% P1 unassigned"
    engine2.declare_surge(step=2)
    assert not engine2._cascade_failure, "Surge should clear cascade failure"
    rs = get_module_registry_stats()
    assert rs.get("triage_conditions", 0) >= 100 or rs.get("survival_registry", {}).get("total", 0) >= 100
    elapsed = round(time.monotonic() - t0, 3)
    logger.info(
        "server.medical __init__ self-test PASSED in %.3fs — "
        "MedicalEngine + %d sub-modules + all assertions OK",
        elapsed,
        sum([_TRIAGE_OK, _CURVES_OK, _GOLDEN_OK, _TRAUMA_OK, _PROTOCOL_OK]),
    )
if MODULE_HEALTH.critical_systems_ok:
    try:
        _self_test()
        MEDICAL_MODULE_READY = True
    except Exception as _self_test_exc:
        logger.error(
            "server.medical self-test FAILED: %s", _self_test_exc, exc_info=True
        )
        MEDICAL_MODULE_READY = False
else:
    logger.warning(
        "server.medical: critical systems not available — "
        "skipping self-test. Errors: %s",
        _import_errors,
    )
    MEDICAL_MODULE_READY = False
logger.info(
    "EMERGI-ENV server.medical v%s loaded — "
    "ready=%s | triage=%s | survival=%s | golden_hour=%s | "
    "trauma=%s | protocol=%s | schema_v%d",
    MEDICAL_MODULE_VERSION_STR,
    MEDICAL_MODULE_READY,
    _TRIAGE_OK, _CURVES_OK, _GOLDEN_OK, _TRAUMA_OK, _PROTOCOL_OK,
    SCHEMA_VERSION,
)
__all__ = [
    "MEDICAL_MODULE_VERSION",
    "MEDICAL_MODULE_VERSION_STR",
    "SCHEMA_VERSION",
    "MEDICAL_MODULE_READY",
    "MODULE_HEALTH",
    "MedicalModuleHealth",
    "TIMESTEP_MINUTES",
    "MAX_EPISODE_STEPS",
    "NUM_ZONES",
    "NUM_HOSPITALS",
    "FLEET_SIZE",
    "TASK_SEEDS",
    "PatientLifecycleState",
    "UnifiedPatientRecord",
    "IncidentType",
    "IncidentRecord",
    "MedicalEngine",
    "get_module_registry_stats",
    "get_condition_keys",
    "TriageTag",
    "TriageEngine",
    "TriageDecision",
    "RPMScore",
    "RPMScore",
    "RevisedTraumaScore",
    "MentalStatus",
    "RespirationStatus",
    "PulseStatus",
    "NACAScore",
    "CbrnContaminationType",
    "TriageProtocol",
    "MCITriageSupervisor",
    "PatientDeteriorationTracker",
    "TriageGroundTruthDatabase",
    "IndianEMSTriageMapper",
    "TRIAGE_CRITICAL_MISMATCH_PENALTY",
    "DETERIORATION_STEPS",
    "PAEDIATRIC_AGE_THRESHOLD",
    "SeverityTier",
    "SurvivalCurveRegistry",
    "SurvivalCurveEngine",
    "SurvivalParameters",
    "PatientSurvivalState",
    "SurvivalProbabilityCalculator",
    "GoldenHourWindows",
    "GoldenHourWindow",
    "GoldenWindowType",
    "TransportUrgencyClass",
    "CurveModelType",
    "SIMULATION_STEP_DURATION_MIN",
    "SURVIVAL_FLOOR",
    "GOLDEN_HOUR_COMPLIANCE_BONUS",
    "GoldenHourEngine",
    "GoldenHourTracker",
    "GoldenHourAlert",
    "GoldenHourAlertQueue",
    "GoldenHourComplianceGrader",
    "GoldenHourPhase",
    "AlertType",
    "AlertSeverity",
    "DispatchQualityGrade",
    "DenseRewardCalculator",
    "EpisodeGoldenHourLedger",
    "EpisodeRewardSummary",
    "StepRewardComponents",
    "PatientGoldenHourRecord",
    "MCIGoldenHourCoordinator",
    "ConditionGoldenHourPolicyRegistry",
    "PLATINUM_10_BONUS",
    "P1_UNASSIGNED_STEP_PENALTY",
    "STEP_REWARD_MIN",
    "STEP_REWARD_MAX",
    "TraumaScoringEngine",
    "TraumaScoreBundle",
    "TraumaRegistryEntry",
    "InjuryMechanism",
    "BodyRegion",
    "AISSeverity",
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
    "FASTFinding",
    "PrimarySurveyState",
    "TraumaActivationDecision",
    "TraumaTeamActivationLevel",
    "DCSIndicationLevel",
    "OutcomePrediction",
    "ISS_MAJOR_TRAUMA_THRESHOLD",
    "ISS_CRITICAL_THRESHOLD",
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
    "ProtocolCheckResult",
    "ProtocolCheckBatch",
    "EpisodeProtocolRecord",
    "ProtocolCheckerEngine",
    "ProtocolRuleDefinition",
    "ViolationSeverity",
    "DispatchProtocolChecker",
    "TriageProtocolChecker",
    "HospitalRoutingChecker",
    "TransferProtocolChecker",
    "MutualAidProtocolChecker",
    "SurgeProtocolChecker",
    "MultiAgencyProtocolChecker",
    "CrewFatigueProtocolChecker",
    "PrepositionProtocolChecker",
    "ProtocolComplianceScoreCalculator",
    "PROTOCOL_CHECKER_VERSION",
    "CALL_TO_DISPATCH_TARGET_MIN",
    "SCENE_TIME_LOAD_AND_GO_MAX_MIN",
]