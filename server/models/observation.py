from __future__ import annotations

import logging
import math
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydantic import Field, computed_field, field_validator, model_validator

from server.models import (
    EmergiBaseModel,
    ImmutableEmergiModel,
    ModelRegistry,
    AgencyType,
    DecayModel,
    HospitalTier,
    HospitalType,
    IncidentType,
    ProtocolRule,
    Season,
    SeverityLevel,
    TaskDifficulty,
    TaskID,
    TriageTag,
    UnitStatus,
    UnitType,
    ZoneType,
    DemandHeatmap,
    HospitalID,
    IncidentID,
    Minutes,
    Probability,
    Score,
    StepNumber,
    TemplateID,
    TrafficMatrix,
    UnitID,
    ZoneID,
    CapacitySnapshot,
    CommunicationsState,
    CrewFatigueState,
    GeoCoordinate,
    MutualAidRequest,
    PatientVitals,
    ACTIVE_ZONES_PER_EPISODE,
    COMMS_FAILURE_PROBABILITY_PER_STEP,
    CREW_FATIGUE_THRESHOLD_HOURS,
    DEMAND_FORECAST_HORIZON_HOURS,
    DEMAND_FORECAST_NOISE_PCT,
    DIVERSION_PENALTY,
    DIVERSION_THRESHOLD_PCT,
    FLEET_SIZE_DEFAULT,
    MAX_STEPS_PER_EPISODE,
    MUTUAL_AID_DELAY_MIN,
    NUM_HOSPITALS,
    NUM_ZONES,
    PEAK_EVENING_HOURS,
    PEAK_MORNING_HOURS,
    PROTOCOL_COMPLIANCE_MAX_BONUS,
    SCORE_CEILING,
    SCORE_FLOOR,
    TIMESTEP_MINUTES,
    TRAFFIC_UPDATE_EVERY_N_STEPS,
    HospitalSpecialty,
    RPMScore,
    clamp_score,
    utc_now_iso,
    __schema_version__,
)

logger = logging.getLogger("emergi_env.models.observation")

class SimulationClock(EmergiBaseModel):
    step: StepNumber = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    episode_minute: float = Field(..., ge=0.0)
    wall_clock_hour: int = Field(..., ge=0, le=23)
    wall_clock_minute: int = Field(..., ge=0, le=59)
    day_of_week: str = Field(...,)
    season: Season
    is_peak_morning: bool = Field(default=False)
    is_peak_evening: bool = Field(default=False)
    next_traffic_update_in_steps: int = Field(default=0, ge=0, le=TRAFFIC_UPDATE_EVERY_N_STEPS)
    steps_remaining: int = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    episode_id: str = Field(..., min_length=36, max_length=36)
    task_id: TaskID
    task_seed: int = Field(..., ge=42, le=50)
    real_time_iso: str = Field(default_factory=utc_now_iso)

    @computed_field
    @property
    def elapsed_hours(self) -> float:
        return self.episode_minute / 60.0

    @computed_field
    @property
    def time_pressure_index(self) -> float:
        peak_factor = 1.0 if (self.is_peak_morning or self.is_peak_evening) else 0.5
        episode_progress = 1.0 - (self.steps_remaining / MAX_STEPS_PER_EPISODE)
        return clamp_score(0.5 * peak_factor + 0.5 * episode_progress)

    @computed_field
    @property
    def current_traffic_multiplier_estimate(self) -> float:
        hour_multipliers = {
            0: 0.55, 1: 0.48, 2: 0.44, 3: 0.42, 4: 0.45, 5: 0.58,
            6: 0.82, 7: 1.38, 8: 1.72, 9: 1.85, 10: 1.55, 11: 1.28,
            12: 1.32, 13: 1.25, 14: 1.18, 15: 1.22, 16: 1.48, 17: 1.88,
            18: 1.95, 19: 1.78, 20: 1.42, 21: 1.15, 22: 0.88, 23: 0.68,
        }
        return hour_multipliers.get(self.wall_clock_hour, 1.0)

    @field_validator("day_of_week")
    @classmethod
    def validate_day(cls, v: str) -> str:
        allowed = {
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "public_holiday", "election_day", "bandh_declared",
        }
        if v.lower() not in allowed:
            raise ValueError(f"day_of_week must be one of {allowed}, got '{v}'")
        return v.lower()

    @model_validator(mode="after")
    def validate_clock_consistency(self) -> "SimulationClock":
        expected_minute = self.step * TIMESTEP_MINUTES
        if abs(self.episode_minute - expected_minute) > 0.01:
            raise ValueError(
                f"episode_minute ({self.episode_minute}) inconsistent with "
                f"step ({self.step}) × TIMESTEP_MINUTES ({TIMESTEP_MINUTES}) = {expected_minute}"
            )
        if self.step + self.steps_remaining > MAX_STEPS_PER_EPISODE:
            raise ValueError(
                f"step ({self.step}) + steps_remaining ({self.steps_remaining}) "
                f"exceeds MAX_STEPS_PER_EPISODE ({MAX_STEPS_PER_EPISODE})"
            )
        if self.task_seed != TaskID(self.task_id).seed:
            raise ValueError(
                f"task_seed {self.task_seed} does not match "
                f"expected seed for {self.task_id}"
            )
        return self

class RPMObservation(ImmutableEmergiModel):
    record_type: Literal["rpm"] = "rpm"
    respirations: str = Field(...,)
    pulse: str = Field(...,)
    mental_status: str = Field(...,)

    @computed_field
    @property
    def rpm_score(self) -> int:
        r = 1 if self.respirations in ("normal", "present_normal") else 0
        p = 1 if self.pulse in ("present_strong", "present_normal") else 0
        m = 1 if self.mental_status == "alert" else 0
        return r + p + m

    @computed_field
    @property
    def derived_tag(self) -> str:
        return TriageTag.from_rpm(
            self.respirations, self.pulse, self.mental_status
        ).value

    @computed_field
    @property
    def survival_modifier(self) -> float:
        return RPMScore.compute_modifier(
            self.respirations, self.pulse, self.mental_status
        )

class MultiAgencyRequirement(ImmutableEmergiModel):
    record_type: Literal["multi_agency_req"] = "multi_agency_req"
    agencies_required: List[str] = Field(...,)
    trapped: bool = Field(default=False)
    hazmat: bool = Field(default=False)
    hazmat_class: Optional[str] = Field(default=None)
    police_scene_cleared: bool = Field(default=False)
    fire_extrication_complete: bool = Field(default=False)
    treatment_unblocked: bool = Field(default=False)

    @computed_field
    @property
    def ems_can_treat(self) -> bool:
        if "Police" in self.agencies_required and not self.police_scene_cleared:
            return False
        if self.trapped and not self.fire_extrication_complete:
            return False
        return True

    @computed_field
    @property
    def blocking_agencies(self) -> List[str]:
        blocking = []
        if "Police" in self.agencies_required and not self.police_scene_cleared:
            blocking.append("Police")
        if ("Fire" in self.agencies_required or self.trapped) and not self.fire_extrication_complete:
            blocking.append("Fire")
        return blocking

class IncidentRecord(EmergiBaseModel):
    record_type: Literal["incident"] = "incident"

    incident_id: IncidentID = Field(...,)
    template_id: TemplateID = Field(...,)
    call_number: str = Field(...,)

    condition: str = Field(...,)
    incident_type: IncidentType
    severity: SeverityLevel
    symptom_description: str = Field(..., min_length=20, max_length=800,)
    protocol_notes: str = Field(..., min_length=10, max_length=600,)
    rpm: RPMObservation

    zone_id: ZoneID = Field(...,)
    location_lat: float = Field(..., ge=-90.0, le=90.0)
    location_lon: float = Field(..., ge=-180.0, le=180.0)
    location_description: str = Field(default="",)

    step_reported: StepNumber = Field(..., ge=0)
    minutes_since_reported: float = Field(default=0.0, ge=0.0)
    golden_hour_minutes: int = Field(..., ge=1,)
    minutes_remaining_golden_hour: float = Field(default_factory=float,)

    victim_count: int = Field(default=1, ge=1, le=50)
    victims_tagged: int = Field(default=0, ge=0)
    victims_immediate: int = Field(default=0, ge=0)
    victims_delayed: int = Field(default=0, ge=0)
    victims_minimal: int = Field(default=0, ge=0)
    victims_expectant: int = Field(default=0, ge=0)
    mci_declared: bool = Field(default=False)

    multi_agency: Optional[MultiAgencyRequirement] = Field(default=None)

    units_assigned: List[UnitID] = Field(default_factory=list)
    units_on_scene: List[UnitID] = Field(default_factory=list)
    dispatched: bool = Field(default=False)
    scene_contact_made: bool = Field(default=False)
    en_route_eta_minutes: Optional[float] = Field(default=None, ge=0.0,)
    nearest_available_unit_eta_minutes: float = Field(..., ge=0.0,)

    requires_micu: bool = Field(default=False)
    requires_als_minimum: bool = Field(default=False)
    hazmat_present: bool = Field(default=False)
    trapped_victim: bool = Field(default=False)
    paediatric: bool = Field(default=False)
    obstetric: bool = Field(default=False)
    psychiatric_risk: bool = Field(default=False)
    dnr_present: bool = Field(default=False)

    @computed_field
    @property
    def is_mci(self) -> bool:
        return self.victim_count > 1 or self.mci_declared

    @computed_field
    @property
    def golden_hour_exceeded(self) -> bool:
        return self.minutes_since_reported > self.golden_hour_minutes

    @computed_field
    @property
    def golden_hour_fraction_elapsed(self) -> float:
        if self.golden_hour_minutes <= 0:
            return 0.0
        return self.minutes_since_reported / self.golden_hour_minutes

    @computed_field
    @property
    def urgency_score(self) -> float:
        severity_weight = {
            SeverityLevel.P1.value: 1.0,
            SeverityLevel.P2.value: 0.65,
            SeverityLevel.P3.value: 0.30,
            SeverityLevel.P0.value: 0.0,
        }.get(self.severity, 0.5)

        gh_penalty = min(1.0, self.golden_hour_fraction_elapsed)
        rpm_mod = abs(self.rpm.survival_modifier)

        raw = (0.5 * severity_weight) + (0.3 * gh_penalty) + (0.2 * rpm_mod)
        return clamp_score(raw)

    @computed_field
    @property
    def victims_untagged(self) -> int:
        return max(0, self.victim_count - self.victims_tagged)

    @computed_field
    @property
    def requires_multi_agency(self) -> bool:
        return self.multi_agency is not None and len(self.multi_agency.agencies_required) > 1

    @computed_field
    @property
    def ems_blocked_by_agency(self) -> bool:
        if self.multi_agency is None:
            return False
        return not self.multi_agency.ems_can_treat

    @computed_field
    @property
    def triage_summary(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.condition} | Zone {self.zone_id} | "
            f"Victims: {self.victim_count} | "
            f"GH elapsed: {self.golden_hour_fraction_elapsed:.0%} | "
            f"Dispatched: {self.dispatched}"
        )

    @field_validator("victim_count")
    @classmethod
    def victim_count_consistency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("victim_count must be >= 1")
        return v

    @model_validator(mode="after")
    def validate_triage_counts(self) -> "IncidentRecord":
        total_tagged = (
            self.victims_immediate
            + self.victims_delayed
            + self.victims_minimal
            + self.victims_expectant
        )
        if total_tagged != self.victims_tagged:
            raise ValueError(
                f"Sum of triage categories ({total_tagged}) != victims_tagged ({self.victims_tagged})"
            )
        if self.victims_tagged > self.victim_count:
            raise ValueError(
                f"victims_tagged ({self.victims_tagged}) > victim_count ({self.victim_count})"
            )
        return self

class UnitLocationObservation(EmergiBaseModel):
    record_type: Literal["unit_location"] = "unit_location"
    is_current: bool = Field(...,)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    zone_id: ZoneID = Field(...,)
    last_update_step: StepNumber = Field(..., ge=0)
    position_age_steps: int = Field(default=0, ge=0,)

    @computed_field
    @property
    def position_age_minutes(self) -> float:
        return self.position_age_steps * TIMESTEP_MINUTES

    @computed_field
    @property
    def position_reliability(self) -> float:
        if self.is_current:
            return 1.0
        return clamp_score(1.0 - (self.position_age_steps * 0.15))

    def as_geo_coordinate(self) -> GeoCoordinate:
        return GeoCoordinate(lat=self.lat, lon=self.lon)

class UnitMissionContext(EmergiBaseModel):
    record_type: Literal["unit_mission"] = "unit_mission"
    incident_id: Optional[IncidentID] = Field(default=None)
    destination_hospital_id: Optional[HospitalID] = Field(default=None)
    destination_zone_id: Optional[ZoneID] = Field(default=None)
    eta_to_destination_minutes: Optional[float] = Field(default=None, ge=0.0)
    eta_available_minutes: Optional[float] = Field(default=None, ge=0.0,)
    patient_on_board: bool = Field(default=False)
    patient_severity: Optional[str] = Field(default=None)
    distance_to_destination_km: Optional[float] = Field(default=None, ge=0.0)
    can_be_rerouted: bool = Field(default=True,)

    @computed_field
    @property
    def is_reroutable(self) -> bool:
        return self.can_be_rerouted and not self.patient_on_board

class UnitStatusRecord(EmergiBaseModel):
    record_type: Literal["unit"] = "unit"

    unit_id: UnitID
    unit_type: UnitType
    registration: str = Field(...,)

    status: UnitStatus
    location: UnitLocationObservation
    mission: Optional[UnitMissionContext] = Field(default=None)

    crew_fatigue: CrewFatigueState
    hours_on_duty: float = Field(..., ge=0.0)
    crew_swap_available: bool = Field(default=True)

    comms_state: CommunicationsState
    radio_channel: str = Field(default="CH1")

    missions_completed_this_episode: int = Field(default=0, ge=0)
    total_distance_km_this_episode: float = Field(default=0.0, ge=0.0)
    avg_response_time_minutes: Optional[float] = Field(default=None, ge=0.0)

    home_zone: ZoneID
    current_zone: ZoneID
    staging_post_id: Optional[str] = Field(default=None)

    is_mutual_aid_unit: bool = Field(default=False)
    mutual_aid_source_zone: Optional[ZoneID] = Field(default=None)
    mutual_aid_eta_steps_remaining: Optional[int] = Field(default=None, ge=0)

    @computed_field
    @property
    def is_available(self) -> bool:
        return self.status == UnitStatus.AVAILABLE

    @computed_field
    @property
    def is_fatigued(self) -> bool:
        return self.hours_on_duty >= CREW_FATIGUE_THRESHOLD_HOURS

    @computed_field
    @property
    def fatigue_penalty_factor(self) -> float:
        return self.crew_fatigue.fatigue_penalty

    @computed_field
    @property
    def comms_active(self) -> bool:
        return self.comms_state.comms_active

    @computed_field
    @property
    def is_deployable(self) -> bool:
        if self.status not in (UnitStatus.AVAILABLE,):
            return False
        if self.mutual_aid_eta_steps_remaining is not None and self.mutual_aid_eta_steps_remaining > 0:
            return False
        return True

    @computed_field
    @property
    def effective_speed_multiplier(self) -> float:
        return max(0.70, 1.0 - self.fatigue_penalty_factor)

    @computed_field
    @property
    def capability_summary(self) -> List[str]:
        return sorted(UnitType(self.unit_type).capabilities)

    @computed_field
    @property
    def operational_summary(self) -> str:
        fatigue_str = f"FATIGUED({self.hours_on_duty:.1f}h)" if self.is_fatigued else "fresh"
        comms_str = "COMMS_OK" if self.comms_active else "COMMS_LOST"
        return (
            f"[{self.unit_id}] {self.unit_type} | {self.status} | "
            f"Zone {self.current_zone} | {fatigue_str} | {comms_str}"
        )

class HospitalCapabilityObservation(EmergiBaseModel):
    record_type: Literal["hospital_capability"] = "hospital_capability"

    trauma_centre: bool = Field(default=False)
    cardiac_cath_lab: bool = Field(default=False)
    stroke_unit: bool = Field(default=False)
    neurosurgery: bool = Field(default=False)
    cardiothoracic_surgery: bool = Field(default=False)
    burns_unit: bool = Field(default=False)
    toxicology: bool = Field(default=False)
    obstetrics: bool = Field(default=False)
    paediatric_emergency: bool = Field(default=False)
    plastic_surgery: bool = Field(default=False)
    renal_dialysis: bool = Field(default=False)
    transplant_centre: bool = Field(default=False)
    hyperbaric_chamber: bool = Field(default=False)
    level_1_trauma: bool = Field(default=False)

    helipad: bool = Field(default=False)
    blood_bank: bool = Field(default=False)
    lab_24h: bool = Field(default=False)
    radiology_24h: bool = Field(default=False)
    ct_scanner: bool = Field(default=False)
    mri: bool = Field(default=False)
    angiography_suite: bool = Field(default=False)
    ecmo_capability: bool = Field(default=False)
    thrombolysis_capability: bool = Field(default=False)

    @computed_field
    @property
    def specialty_set(self) -> List[str]:
        result = []
        for spec in HospitalSpecialty.ALL:
            if getattr(self, spec.replace("-", "_"), False):
                result.append(spec)
        return sorted(result)

    def has_specialty(self, specialty: str) -> bool:
        attr = specialty.replace("-", "_")
        return bool(getattr(self, attr, False))

    def can_accept_condition(self, required_specialty: str) -> bool:
        return self.has_specialty(required_specialty)

    @computed_field
    @property
    def capability_tier_score(self) -> float:
        spec_flags = [
            self.trauma_centre, self.cardiac_cath_lab, self.stroke_unit,
            self.neurosurgery, self.cardiothoracic_surgery, self.burns_unit,
            self.toxicology, self.obstetrics, self.paediatric_emergency,
            self.plastic_surgery, self.renal_dialysis, self.transplant_centre,
            self.hyperbaric_chamber, self.level_1_trauma,
        ]
        infra_flags = [
            self.helipad, self.blood_bank, self.lab_24h, self.radiology_24h,
            self.ct_scanner, self.mri, self.angiography_suite,
            self.ecmo_capability, self.thrombolysis_capability,
        ]
        spec_score = sum(spec_flags) / len(spec_flags)
        infra_score = sum(infra_flags) / len(infra_flags)
        return clamp_score(0.6 * spec_score + 0.4 * infra_score)

class HospitalPerformanceObservation(EmergiBaseModel):
    record_type: Literal["hospital_performance"] = "hospital_performance"

    door_to_doctor_min: float = Field(..., ge=0.0)
    door_to_ct_min: float = Field(..., ge=0.0)
    door_to_balloon_min: float = Field(..., ge=0.0)
    door_to_needle_min: float = Field(..., ge=0.0)
    trauma_activation_time_min: float = Field(..., ge=0.0)

    emergency_physicians_on_duty: int = Field(..., ge=0)
    trauma_surgeons_on_call: int = Field(..., ge=0)
    icu_nurses_per_shift: int = Field(..., ge=0)

    @computed_field
    @property
    def stemi_readiness_score(self) -> float:
        if self.door_to_balloon_min >= 999:
            return 0.0
        return clamp_score(1.0 - (self.door_to_balloon_min / 90.0))

    @computed_field
    @property
    def stroke_readiness_score(self) -> float:
        if self.door_to_needle_min >= 999:
            return 0.0
        return clamp_score(1.0 - (self.door_to_needle_min / 60.0))

class HospitalStatusRecord(EmergiBaseModel):
    record_type: Literal["hospital"] = "hospital"

    hospital_id: HospitalID
    name: str
    short_name: str
    hospital_type: HospitalType
    tier: HospitalTier
    zone_id: ZoneID
    district: str
    location: GeoCoordinate

    capacity: CapacitySnapshot

    capabilities: HospitalCapabilityObservation
    performance: HospitalPerformanceObservation

    routing_priorities: Dict[str, int] = Field(default_factory=dict,)

    surge_capacity_additional_beds: int = Field(default=0, ge=0)
    mutual_aid_capacity: int = Field(default=0, ge=0)
    base_score_weight: float = Field(default=1.0, ge=0.0, le=1.0)

    diversion_flag_active: bool = Field(default=False)
    diversion_reason: Optional[str] = Field(default=None)
    diversion_reset_in_steps: Optional[int] = Field(default=None, ge=0)
    consecutive_diversion_steps: int = Field(default=0, ge=0)

    active_incoming_transports: int = Field(default=0, ge=0,)
    admitted_this_episode: int = Field(default=0, ge=0)

    @computed_field
    @property
    def is_on_diversion(self) -> bool:
        return self.diversion_flag_active or self.capacity.on_diversion

    @computed_field
    @property
    def effective_er_occupancy(self) -> float:
        raw = self.capacity.er_occupancy_pct
        transport_load = self.active_incoming_transports * 0.02
        return min(1.0, raw + transport_load)

    @computed_field
    @property
    def accepts_patients(self) -> bool:
        if self.is_on_diversion:
            return False
        return self.effective_er_occupancy < DIVERSION_THRESHOLD_PCT

    @computed_field
    @property
    def capacity_pressure(self) -> float:
        return clamp_score(self.effective_er_occupancy / DIVERSION_THRESHOLD_PCT)

    @computed_field
    @property
    def routing_score(self) -> float:
        if not self.accepts_patients:
            return 0.0
        cap_score = self.capabilities.capability_tier_score
        cap_pressure = 1.0 - self.capacity_pressure
        return clamp_score(
            self.base_score_weight * (0.6 * cap_score + 0.4 * cap_pressure)
        )

    def specialty_match_score(self, required_specialty: str) -> float:
        if not self.capabilities.has_specialty(required_specialty):
            return 0.0
        if not self.accepts_patients:
            return 0.0
        return clamp_score(1.0 - self.capacity_pressure * 0.5)

    def get_routing_priority(self, condition: str) -> int:
        return self.routing_priorities.get(condition, 9)

    @computed_field
    @property
    def surge_capacity_remaining(self) -> int:
        if self.capacity.total_occupancy_pct >= 0.95:
            return 0
        return self.surge_capacity_additional_beds

    @computed_field
    @property
    def hospital_summary(self) -> str:
        divert_str = "ON_DIVERSION" if self.is_on_diversion else "ACCEPTING"
        return (
            f"[{self.hospital_id}] {self.short_name} | "
            f"Tier: {self.tier} | Zone: {self.zone_id} | "
            f"ER: {self.capacity.er_occupancy_pct:.0%} | {divert_str} | "
            f"Score: {self.routing_score:.2f}"
        )

class ActiveSlowdown(ImmutableEmergiModel):
    slowdown_id: str
    affected_zones: List[ZoneID]
    multiplier: float = Field(..., ge=1.0, le=5.0)
    steps_remaining: int = Field(..., ge=0)
    cause: str = Field(...,)
    requires_police_clearance: bool = Field(default=False)
    requires_fire_clearance: bool = Field(default=False)
    ems_exempt: bool = Field(default=False,)

    @computed_field
    @property
    def minutes_remaining(self) -> float:
        return self.steps_remaining * TIMESTEP_MINUTES

    @computed_field
    @property
    def severity_label(self) -> str:
        if self.multiplier >= 2.5:
            return "SEVERE"
        if self.multiplier >= 1.7:
            return "MODERATE"
        return "MILD"

class BottleneckObservation(ImmutableEmergiModel):
    bottleneck_id: str
    zone_id: ZoneID
    name: str
    affected_zones: List[ZoneID]
    currently_blocked: bool
    current_multiplier: float = Field(..., ge=1.0, le=6.0)
    ems_bypass_available: bool
    bypass_time_delta_minutes: float = Field(default=0.0, ge=0.0)

    @computed_field
    @property
    def is_critical(self) -> bool:
        return self.current_multiplier >= 1.8 and not self.ems_bypass_available

class TrafficSnapshot(EmergiBaseModel):
    record_type: Literal["traffic"] = "traffic"

    od_matrix: Dict[ZoneID, Dict[ZoneID, float]] = Field(...,)

    last_updated_step: StepNumber = Field(..., ge=0)
    next_update_step: StepNumber = Field(..., ge=0)
    current_global_multiplier: float = Field(..., ge=0.3, le=5.0)

    active_slowdowns: List[ActiveSlowdown] = Field(default_factory=list)
    bottlenecks: List[BottleneckObservation] = Field(default_factory=list)

    closed_routes: List[Tuple[ZoneID, ZoneID]] = Field(default_factory=list,)
    ghat_closures: List[str] = Field(default_factory=list,)

    green_corridor_active: bool = Field(default=False)
    green_corridor_route: Optional[List[ZoneID]] = Field(default=None)
    green_corridor_expires_step: Optional[int] = Field(default=None)
    active_ems_corridors: List[str] = Field(default_factory=list,)

    special_event_active: Optional[str] = Field(default=None,)
    event_traffic_multiplier: float = Field(default=1.0, ge=0.5, le=4.0)

    def get_travel_time(
        self,
        origin: ZoneID,
        destination: ZoneID,
        unit_type: Optional[UnitType] = None,
    ) -> float:
        if (origin, destination) in self.closed_routes:
            return math.inf

        row = self.od_matrix.get(origin)
        if row is None:
            return math.inf
        base_time = row.get(destination, math.inf)
        if base_time == math.inf or math.isnan(base_time):
            return math.inf

        if unit_type is not None:
            unit_type_enum = UnitType(unit_type) if isinstance(unit_type, str) else unit_type
            speed_mod = {
                UnitType.BLS: 1.08,
                UnitType.ALS: 1.04,
                UnitType.MICU: 1.00,
            }.get(unit_type_enum, 1.0)
            base_time *= speed_mod

        slowdown_factor = 1.0
        for slowdown in self.active_slowdowns:
            if (origin in slowdown.affected_zones or destination in slowdown.affected_zones):
                if not slowdown.ems_exempt:
                    slowdown_factor = max(slowdown_factor, slowdown.multiplier)

        return base_time * slowdown_factor

    @computed_field
    @property
    def num_active_slowdowns(self) -> int:
        return len(self.active_slowdowns)

    @computed_field
    @property
    def num_closed_routes(self) -> int:
        return len(self.closed_routes)

    @computed_field
    @property
    def worst_slowdown_multiplier(self) -> float:
        if not self.active_slowdowns:
            return 1.0
        return max(s.multiplier for s in self.active_slowdowns)

    @computed_field
    @property
    def is_peak_hour(self) -> bool:
        return self.current_global_multiplier >= 1.5

    @computed_field
    @property
    def traffic_congestion_index(self) -> float:
        mult_factor = clamp_score((self.current_global_multiplier - 0.4) / (2.0 - 0.4))
        slowdown_factor = clamp_score(self.num_active_slowdowns / 10.0)
        closure_factor = clamp_score(self.num_closed_routes / 5.0)
        return clamp_score(0.5 * mult_factor + 0.3 * slowdown_factor + 0.2 * closure_factor)

    def fastest_hospital_route(
        self,
        origin_zone: ZoneID,
        hospital_zones: Dict[HospitalID, ZoneID],
        exclude_hospitals: Optional[Set[HospitalID]] = None,
    ) -> Optional[Tuple[HospitalID, float]]:
        exclude = exclude_hospitals or set()
        best_time = math.inf
        best_hospital: Optional[HospitalID] = None

        for hosp_id, hosp_zone in hospital_zones.items():
            if hosp_id in exclude:
                continue
            t = self.get_travel_time(origin_zone, hosp_zone)
            if t < best_time:
                best_time = t
                best_hospital = hosp_id

        if best_hospital is None:
            return None
        return (best_hospital, best_time)

class ZoneDemandForecast(ImmutableEmergiModel):
    zone_id: ZoneID
    zone_type: ZoneType
    base_call_volume_per_day: float = Field(..., ge=0.0)

    forecast_00_03h: float = Field(..., ge=0.0,)
    forecast_03_06h: float = Field(..., ge=0.0,)
    forecast_06_09h: float = Field(..., ge=0.0,)
    forecast_09_12h: float = Field(..., ge=0.0,)

    noise_applied_pct: float = Field(default=DEMAND_FORECAST_NOISE_PCT, ge=0.0, le=0.5,)
    confidence_interval_pct: float = Field(default=0.80, ge=0.0, le=1.0,)

    cardiac_arrest_risk: float = Field(default=0.0, ge=0.0)
    trauma_risk: float = Field(default=0.0, ge=0.0)
    mci_risk: float = Field(default=0.0, ge=0.0)
    flood_risk: str = Field(default="low")
    industrial_hazard_risk: str = Field(default="low")

    avg_historical_response_min: float = Field(default=15.0, ge=0.0)
    target_response_min: int = Field(default=10, ge=1)

    @computed_field
    @property
    def peak_forecast_slot(self) -> str:
        slots = {
            "00_03h": self.forecast_00_03h,
            "03_06h": self.forecast_03_06h,
            "06_09h": self.forecast_06_09h,
            "09_12h": self.forecast_09_12h,
        }
        return max(slots, key=slots.get)

    @computed_field
    @property
    def total_forecast_demand_index(self) -> float:
        return (
            self.forecast_00_03h
            + self.forecast_03_06h
            + self.forecast_06_09h
            + self.forecast_09_12h
        )

    @computed_field
    @property
    def response_time_gap(self) -> float:
        return max(0.0, self.avg_historical_response_min - self.target_response_min)

    @computed_field
    @property
    def priority_score(self) -> float:
        demand_score = clamp_score(self.total_forecast_demand_index / 8.0)
        gap_score = clamp_score(self.response_time_gap / 30.0)
        risk_score = clamp_score(self.cardiac_arrest_risk + self.mci_risk)
        return clamp_score(0.4 * demand_score + 0.35 * gap_score + 0.25 * risk_score)

class DemandForecastObservation(EmergiBaseModel):
    record_type: Literal["demand_forecast"] = "demand_forecast"

    zone_forecasts: List[ZoneDemandForecast] = Field(...,)
    forecast_generated_step: StepNumber = Field(..., ge=0)
    forecast_horizon_hours: int = Field(default=DEMAND_FORECAST_HORIZON_HOURS)
    noise_applied: bool = Field(default=True)
    noise_std_pct: float = Field(default=DEMAND_FORECAST_NOISE_PCT)

    highest_demand_zone: Optional[ZoneID] = Field(default=None)
    highest_mci_risk_zone: Optional[ZoneID] = Field(default=None)
    expected_peak_hour: Optional[int] = Field(default=None, ge=0, le=23)

    @computed_field
    @property
    def zone_count(self) -> int:
        return len(self.zone_forecasts)

    @computed_field
    @property
    def top_priority_zones(self) -> List[ZoneID]:
        sorted_zones = sorted(
            self.zone_forecasts,
            key=lambda z: z.priority_score,
            reverse=True,
        )
        return [z.zone_id for z in sorted_zones[:5]]

    def get_zone_forecast(self, zone_id: ZoneID) -> Optional[ZoneDemandForecast]:
        for z in self.zone_forecasts:
            if z.zone_id == zone_id:
                return z
        return None

    def demand_rank(self, zone_id: ZoneID) -> int:
        sorted_zones = sorted(
            self.zone_forecasts,
            key=lambda z: z.total_forecast_demand_index,
            reverse=True,
        )
        for i, z in enumerate(sorted_zones):
            if z.zone_id == zone_id:
                return i + 1
        return len(sorted_zones) + 1

class AgencyResourceObservation(ImmutableEmergiModel):
    agency: AgencyType
    units_available: int = Field(..., ge=0)
    units_on_scene: int = Field(..., ge=0)
    response_time_estimate_min: float = Field(..., ge=0.0)
    currently_dispatched_incidents: List[IncidentID] = Field(default_factory=list)

    @computed_field
    @property
    def is_available(self) -> bool:
        return self.units_available > 0

    @computed_field
    @property
    def utilisation_rate(self) -> float:
        total = self.units_available + self.units_on_scene
        if total == 0:
            return 0.0
        return self.units_on_scene / total

class MultiAgencyStatusObservation(EmergiBaseModel):
    record_type: Literal["multi_agency"] = "multi_agency"

    agencies: Dict[str, AgencyResourceObservation] = Field(...,)

    coordinated_incidents: List[IncidentID] = Field(default_factory=list,)

    blocked_incidents: List[IncidentID] = Field(default_factory=list,)

    @computed_field
    @property
    def police_available(self) -> bool:
        police = self.agencies.get("Police")
        return police is not None and police.is_available

    @computed_field
    @property
    def fire_available(self) -> bool:
        fire = self.agencies.get("Fire")
        return fire is not None and fire.is_available

    @computed_field
    @property
    def ndrf_available(self) -> bool:
        ndrf = self.agencies.get("NDRF")
        return ndrf is not None and ndrf.is_available

    @computed_field
    @property
    def num_blocked_incidents(self) -> int:
        return len(self.blocked_incidents)

class MutualAidStatusObservation(EmergiBaseModel):
    record_type: Literal["mutual_aid_status"] = "mutual_aid_status"

    active_requests: List[MutualAidRequest] = Field(default_factory=list)
    fulfilled_requests: int = Field(default=0, ge=0)
    cancelled_requests: int = Field(default=0, ge=0)
    over_request_penalty_incurred: float = Field(default=0.0, le=0.0)

    zone_mutual_aid_available: Dict[ZoneID, int] = Field(default_factory=dict,)

    @computed_field
    @property
    def active_request_count(self) -> int:
        return len([r for r in self.active_requests if not r.cancelled and not r.fulfilled])

    @computed_field
    @property
    def units_en_route_via_mutual_aid(self) -> int:
        return len([r for r in self.active_requests if not r.fulfilled and not r.cancelled])

class SurgeStatusObservation(EmergiBaseModel):
    record_type: Literal["surge_status"] = "surge_status"

    surge_declared: bool = Field(default=False)
    surge_declared_step: Optional[int] = Field(default=None)
    surge_level: int = Field(default=0, ge=0, le=3,)

    hospitals_on_diversion: List[HospitalID] = Field(default_factory=list)
    hospitals_at_capacity: List[HospitalID] = Field(default_factory=list)
    system_er_occupancy_avg: float = Field(default=0.0, ge=0.0, le=1.0)
    system_icu_occupancy_avg: float = Field(default=0.0, ge=0.0, le=1.0)

    mutual_aid: MutualAidStatusObservation

    cascade_failure_risk: float = Field(default=0.0, ge=0.0, le=1.0,)
    cascade_failure_occurred: bool = Field(default=False)
    steps_until_cascade: Optional[int] = Field(default=None, ge=0,)

    simultaneous_mci_count: int = Field(default=0, ge=0)
    comms_failure_count: int = Field(default=0, ge=0,)

    @computed_field
    @property
    def system_under_stress(self) -> bool:
        return (
            self.system_er_occupancy_avg >= 0.80
            or len(self.hospitals_on_diversion) >= 3
            or self.simultaneous_mci_count >= 2
        )

    @computed_field
    @property
    def surge_required(self) -> bool:
        return (
            self.simultaneous_mci_count >= 3
            or self.system_er_occupancy_avg >= 0.85
            or self.cascade_failure_risk >= 0.50
            or len(self.hospitals_on_diversion) >= 4
        )

    @computed_field
    @property
    def diversion_penalty_exposure(self) -> float:
        return len(self.hospitals_on_diversion) * abs(DIVERSION_PENALTY)

class ProtocolViolationRecord(ImmutableEmergiModel):
    rule: ProtocolRule
    incident_id: Optional[IncidentID] = Field(default=None)
    unit_id: Optional[UnitID] = Field(default=None)
    hospital_id: Optional[HospitalID] = Field(default=None)
    step: StepNumber
    penalty_applied: float = Field(..., le=0.0)
    description: str

class ProtocolComplianceTracker(EmergiBaseModel):
    record_type: Literal["compliance"] = "compliance"

    total_correct_actions: int = Field(default=0, ge=0)
    total_violations: int = Field(default=0, ge=0)
    compliance_bonus_earned: float = Field(default=0.0, ge=0.0, le=PROTOCOL_COMPLIANCE_MAX_BONUS)
    penalty_total: float = Field(default=0.0, le=0.0)

    rule_correct_counts: Dict[str, int] = Field(default_factory=dict,)
    rule_violation_counts: Dict[str, int] = Field(default_factory=dict,)

    recent_violations: List[ProtocolViolationRecord] = Field(default_factory=list, max_length=10,)

    mciu_stemi_correct_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    start_triage_correct_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    diversion_violation_count: int = Field(default=0, ge=0)
    multi_agency_violation_count: int = Field(default=0, ge=0)

    @computed_field
    @property
    def compliance_rate(self) -> float:
        total = self.total_correct_actions + self.total_violations
        if total == 0:
            return 1.0
        return clamp_score(self.total_correct_actions / total)

    @computed_field
    @property
    def max_remaining_bonus(self) -> float:
        return max(0.0, PROTOCOL_COMPLIANCE_MAX_BONUS - self.compliance_bonus_earned)

    @computed_field
    @property
    def net_compliance_score(self) -> float:
        return clamp_score(self.compliance_bonus_earned + self.penalty_total)

class EpisodeMetaObservation(EmergiBaseModel):
    record_type: Literal["episode_meta"] = "episode_meta"

    task_id: TaskID
    task_difficulty: TaskDifficulty
    baseline_score: float = Field(..., ge=0.0, le=1.0)
    current_episode_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_vs_baseline: float = Field(default=0.0,)

    total_incidents_this_episode: int = Field(default=0, ge=0)
    incidents_resolved: int = Field(default=0, ge=0)
    incidents_active: int = Field(default=0, ge=0)
    incidents_failed: int = Field(default=0, ge=0,)
    p1_incidents_total: int = Field(default=0, ge=0)
    p1_incidents_resolved: int = Field(default=0, ge=0)

    fleet_utilisation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_response_time_minutes: float = Field(default=0.0, ge=0.0)
    units_available_count: int = Field(default=0, ge=0)
    units_fatigued_count: int = Field(default=0, ge=0)
    units_comms_lost_count: int = Field(default=0, ge=0)

    cumulative_survival_reward: float = Field(default=0.0)
    cumulative_protocol_reward: float = Field(default=0.0)
    cumulative_diversion_penalty: float = Field(default=0.0, le=0.0)
    cumulative_multi_agency_penalty: float = Field(default=0.0, le=0.0)

    cascade_failure_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    surge_declared: bool = Field(default=False)
    mutual_aid_activated: bool = Field(default=False)

    @computed_field
    @property
    def p1_resolution_rate(self) -> float:
        if self.p1_incidents_total == 0:
            return 1.0
        return clamp_score(self.p1_incidents_resolved / self.p1_incidents_total)

    @computed_field
    @property
    def incident_resolution_rate(self) -> float:
        if self.total_incidents_this_episode == 0:
            return 1.0
        return clamp_score(self.incidents_resolved / self.total_incidents_this_episode)

    @computed_field
    @property
    def beating_baseline(self) -> bool:
        return self.current_episode_score > self.baseline_score

    @computed_field
    @property
    def total_cumulative_reward(self) -> float:
        return (
            self.cumulative_survival_reward
            + self.cumulative_protocol_reward
            + self.cumulative_diversion_penalty
            + self.cumulative_multi_agency_penalty
        )

class EpisodeObservation(EmergiBaseModel):
    schema_version: int = Field(default=__schema_version__,)
    observation_step: StepNumber = Field(..., ge=0, le=MAX_STEPS_PER_EPISODE)
    episode_done: bool = Field(default=False,)
    termination_reason: Optional[str] = Field(default=None,)

    clock: SimulationClock
    incident_queue: List[IncidentRecord] = Field(default_factory=list,)
    fleet_status: List[UnitStatusRecord] = Field(default_factory=list,)
    hospital_network: List[HospitalStatusRecord] = Field(default_factory=list,)
    traffic_snapshot: TrafficSnapshot
    demand_forecast: DemandForecastObservation
    multi_agency_status: MultiAgencyStatusObservation
    surge_status: SurgeStatusObservation
    compliance_tracker: ProtocolComplianceTracker
    episode_meta: EpisodeMetaObservation

    legal_actions: List[str] = Field(default_factory=list,)
    action_mask: Dict[str, bool] = Field(default_factory=dict,)

    available_unit_ids: List[UnitID] = Field(default_factory=list,)
    p1_incident_ids: List[IncidentID] = Field(default_factory=list,)
    divert_hospital_ids: List[HospitalID] = Field(default_factory=list,)
    active_slowdown_zones: List[ZoneID] = Field(default_factory=list,)

    @computed_field
    @property
    def queue_length(self) -> int:
        return len(self.incident_queue)

    @computed_field
    @property
    def p1_queue_length(self) -> int:
        return len([i for i in self.incident_queue if i.severity == SeverityLevel.P1.value])

    @computed_field
    @property
    def unassigned_p1_count(self) -> int:
        return len([
            i for i in self.incident_queue
            if i.severity == SeverityLevel.P1.value and not i.dispatched
        ])

    @computed_field
    @property
    def fleet_available_count(self) -> int:
        return len([u for u in self.fleet_status if u.is_available])

    @computed_field
    @property
    def fleet_deployable_count(self) -> int:
        return len([u for u in self.fleet_status if u.is_deployable])

    @computed_field
    @property
    def resource_deficit(self) -> int:
        return self.unassigned_p1_count - self.fleet_deployable_count

    @computed_field
    @property
    def system_stress_level(self) -> str:
        if self.resource_deficit >= 5 or self.surge_status.cascade_failure_risk >= 0.7:
            return "critical"
        if self.resource_deficit >= 3 or self.surge_status.system_under_stress:
            return "high"
        if self.resource_deficit >= 1:
            return "elevated"
        return "normal"

    @computed_field
    @property
    def mci_active(self) -> bool:
        return any(i.is_mci for i in self.incident_queue)

    @computed_field
    @property
    def comms_failure_active(self) -> bool:
        return any(not u.comms_active for u in self.fleet_status)

    @computed_field
    @property
    def highest_urgency_incident(self) -> Optional[str]:
        unassigned = [i for i in self.incident_queue if not i.dispatched]
        if not unassigned:
            return None
        return max(unassigned, key=lambda i: i.urgency_score).incident_id

    @computed_field
    @property
    def observation_summary(self) -> str:
        return (
            f"Step {self.observation_step}/{MAX_STEPS_PER_EPISODE} | "
            f"Task: {self.clock.task_id} | "
            f"Queue: {self.queue_length} incidents ({self.p1_queue_length} P1) | "
            f"Fleet: {self.fleet_available_count}/{len(self.fleet_status)} available | "
            f"Stress: {self.system_stress_level} | "
            f"Score: {self.episode_meta.current_episode_score:.3f} "
            f"(baseline {self.episode_meta.baseline_score:.2f}) | "
            f"Traffic: {self.traffic_snapshot.traffic_congestion_index:.0%} congestion"
        )

    @model_validator(mode="after")
    def validate_observation_consistency(self) -> "EpisodeObservation":
        if self.schema_version != __schema_version__:
            raise ValueError(
                f"Observation schema_version {self.schema_version} "
                f"!= current __schema_version__ {__schema_version__}. "
                "Regenerate observation from env.py."
            )
        if self.observation_step != self.clock.step:
            raise ValueError(
                f"observation_step ({self.observation_step}) "
                f"!= clock.step ({self.clock.step})"
            )
        if self.clock.task_id != self.episode_meta.task_id:
            raise ValueError(
                f"clock.task_id ({self.clock.task_id}) "
                f"!= episode_meta.task_id ({self.episode_meta.task_id})"
            )
        if self.episode_meta.incidents_active != len(self.incident_queue):
            raise ValueError(
                f"episode_meta.incidents_active ({self.episode_meta.incidents_active}) "
                f"!= len(incident_queue) ({len(self.incident_queue)})"
            )
        actual_available = {u.unit_id for u in self.fleet_status if u.is_available}
        declared_available = set(self.available_unit_ids)
        if actual_available != declared_available:
            raise ValueError(
                f"available_unit_ids mismatch. "
                f"Actual: {actual_available}, Declared: {declared_available}"
            )
        return self

    def get_incident(self, incident_id: IncidentID) -> Optional[IncidentRecord]:
        for inc in self.incident_queue:
            if inc.incident_id == incident_id:
                return inc
        return None

    def get_unit(self, unit_id: UnitID) -> Optional[UnitStatusRecord]:
        for unit in self.fleet_status:
            if unit.unit_id == unit_id:
                return unit
        return None

    def get_hospital(self, hospital_id: HospitalID) -> Optional[HospitalStatusRecord]:
        for hosp in self.hospital_network:
            if hosp.hospital_id == hospital_id:
                return hosp
        return None

    def available_units_of_type(self, unit_type: UnitType) -> List[UnitStatusRecord]:
        return [
            u for u in self.fleet_status
            if UnitType(u.unit_type) == unit_type and u.is_deployable
        ]

    def hospitals_with_specialty(self, specialty: str) -> List[HospitalStatusRecord]:
        return [
            h for h in self.hospital_network
            if h.capabilities.has_specialty(specialty) and h.accepts_patients
        ]

    def best_hospital_for_condition(
        self,
        condition_key: str,
        required_specialty: str,
        origin_zone: ZoneID,
    ) -> Optional[HospitalStatusRecord]:
        candidates = self.hospitals_with_specialty(required_specialty)
        if not candidates:
            return None

        def score_hospital(h: HospitalStatusRecord) -> float:
            priority = h.get_routing_priority(condition_key)
            priority_score = clamp_score(1.0 - (priority - 1) / 8.0)
            capacity_score = 1.0 - h.capacity_pressure
            travel_min = self.traffic_snapshot.get_travel_time(origin_zone, h.zone_id)
            travel_score = clamp_score(1.0 - travel_min / 120.0)
            return 0.4 * priority_score + 0.3 * capacity_score + 0.3 * travel_score

        return max(candidates, key=score_hospital)

    def nearest_available_unit(
        self,
        incident_zone: ZoneID,
        unit_type: Optional[UnitType] = None,
    ) -> Optional[UnitStatusRecord]:
        candidates = [u for u in self.fleet_status if u.is_deployable]
        if unit_type is not None:
            unit_type_enum = UnitType(unit_type) if isinstance(unit_type, str) else unit_type
            candidates = [u for u in candidates if UnitType(u.unit_type) == unit_type_enum]

        if not candidates:
            return None

        def eta(unit: UnitStatusRecord) -> float:
            return self.traffic_snapshot.get_travel_time(unit.current_zone, incident_zone)

        return min(candidates, key=eta)

ModelRegistry.register("SimulationClock", SimulationClock)
ModelRegistry.register("RPMObservation", RPMObservation)
ModelRegistry.register("MultiAgencyRequirement", MultiAgencyRequirement)
ModelRegistry.register("IncidentRecord", IncidentRecord)
ModelRegistry.register("UnitLocationObservation", UnitLocationObservation)
ModelRegistry.register("UnitMissionContext", UnitMissionContext)
ModelRegistry.register("UnitStatusRecord", UnitStatusRecord)
ModelRegistry.register("HospitalCapabilityObservation", HospitalCapabilityObservation)
ModelRegistry.register("HospitalPerformanceObservation", HospitalPerformanceObservation)
ModelRegistry.register("HospitalStatusRecord", HospitalStatusRecord)
ModelRegistry.register("ActiveSlowdown", ActiveSlowdown)
ModelRegistry.register("BottleneckObservation", BottleneckObservation)
ModelRegistry.register("TrafficSnapshot", TrafficSnapshot)
ModelRegistry.register("ZoneDemandForecast", ZoneDemandForecast)
ModelRegistry.register("DemandForecastObservation", DemandForecastObservation)
ModelRegistry.register("AgencyResourceObservation", AgencyResourceObservation)
ModelRegistry.register("MultiAgencyStatusObservation", MultiAgencyStatusObservation)
ModelRegistry.register("MutualAidStatusObservation", MutualAidStatusObservation)
ModelRegistry.register("SurgeStatusObservation", SurgeStatusObservation)
ModelRegistry.register("ProtocolViolationRecord", ProtocolViolationRecord)
ModelRegistry.register("ProtocolComplianceTracker", ProtocolComplianceTracker)
ModelRegistry.register("EpisodeMetaObservation", EpisodeMetaObservation)
ModelRegistry.register("EpisodeObservation", EpisodeObservation)
AmbulanceObs        = UnitStatusRecord
DemandForecastObs   = DemandForecastObservation
HospitalObs         = HospitalStatusRecord
IncidentObs         = IncidentRecord
ObservationModel    = EpisodeObservation
TrafficSnapshotObs  = TrafficSnapshot
logger.info(
    "observation.py: registered %d observation models.",
    23,
)

__all__ = [
    "SimulationClock",
    "AmbulanceObs",
    "DemandForecastObs",
    "HospitalObs",
    "IncidentObs",
    "ObservationModel",
    "TrafficSnapshotObs",
    "RPMObservation",
    "MultiAgencyRequirement",
    "IncidentRecord",
    "UnitLocationObservation",
    "UnitMissionContext",
    "UnitStatusRecord",
    "HospitalCapabilityObservation",
    "HospitalPerformanceObservation",
    "HospitalStatusRecord",
    "ActiveSlowdown",
    "BottleneckObservation",
    "TrafficSnapshot",
    "ZoneDemandForecast",
    "DemandForecastObservation",
    "AgencyResourceObservation",
    "MultiAgencyStatusObservation",
    "MutualAidStatusObservation",
    "SurgeStatusObservation",
    "ProtocolViolationRecord",
    "ProtocolComplianceTracker",
    "EpisodeMetaObservation",
    "EpisodeObservation",
]