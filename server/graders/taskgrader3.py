from __future__ import annotations
import logging
import math
import re
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
logger = logging.getLogger("emergi_env.graders.task3")
TASK_ID       = "task3_unit_type"
TASK_SEED     = TASK_SEEDS[TASK_ID]
TASK_BASELINE = TASK_BASELINES[TASK_ID]
W_ACCURACY     = 0.55
W_JUSTIFICATION= 0.20
W_EQUIPMENT    = 0.15
W_EFFICIENCY   = 0.10
UNIT_RANK: Dict[str, int] = {"BLS": 1, "ALS": 2, "MICU": 3}
UNIT_NAMES = ("BLS", "ALS", "MICU")
ACCURACY_MATRIX: Dict[Tuple[int, int], float] = {
    (1, 1): 1.00,   
    (2, 2): 1.00,   
    (3, 3): 1.00,   
    (2, 1): 0.82,   
    (3, 1): 0.65,   
    (3, 2): 0.82,   
    (1, 2): 0.45,   
    (1, 3): 0.10,   
    (2, 3): 0.45,   
}
PROTOCOL_BONUS_CAP   = 0.15
DEMOGRAPHIC_MOD_CAP  = 0.10
EFFICIENCY_CAP       = 0.10
HP_BLS_CARDIAC_ARREST     = 0.40
HP_BLS_STEMI              = 0.35
HP_BLS_ECLAMPSIA          = 0.30
HP_BLS_RESP_FAILURE       = 0.30
HP_BLS_SEPTIC_SHOCK       = 0.25
HP_BLS_STATUS_EPILEPTICUS  = 0.25
HP_ALS_POST_ARREST_VASOP   = 0.20
HP_BLS_PAED_SEIZURE        = 0.15
CONF_EXCELLENT_DELTA = 0.10
CONF_GOOD_DELTA      = 0.20
CONF_EXCELLENT_BONUS = 0.05
CONF_GOOD_BONUS      = 0.02
@dataclass(frozen=True)
class EquipmentItem:
    name:     str
    priority: str   
    unit_min: str   
@dataclass(frozen=True)
class ConditionSpec:
    condition_code:     str
    category:           str
    primary_unit:       str                    
    acceptable_alts:    Tuple[str, ...]        
    score_reductions:   Tuple[float, ...]      
    equipment_critical_a: Tuple[str, ...]      
    equipment_critical_b: Tuple[str, ...]      
    equipment_important:  Tuple[str, ...]      
    equipment_contra:     Tuple[str, ...]      
    protocol_codes:     Tuple[str, ...]        
    upgrade_if:         Tuple[str, ...]        
    downgrade_if:       Tuple[str, ...]        
    time_critical_min:  float                  
    notes:              str                    
def _build_condition_db() -> Dict[str, ConditionSpec]:
    db: Dict[str, ConditionSpec] = {}
    def add(c: ConditionSpec) -> None:
        db[c.condition_code] = c
    add(ConditionSpec(
        "cardiac_arrest_vf", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.25,),
        equipment_critical_a=("defibrillator", "adrenaline_1mg_iv", "amiodarone_iv",
                              "airway_management_rsi", "iv_access_2x"),
        equipment_critical_b=("capnography", "12lead_ecg", "cpr_mechanical_device"),
        equipment_important=("ultrasound_point_of_care",),
        equipment_contra=(), protocol_codes=("AHA_ACLS_VF", "ROSC_BUNDLE"),
        upgrade_if=(), downgrade_if=("bystander_rosc_confirmed",),
        time_critical_min=10.0,
        notes="VF cardiac arrest: immediate defibrillation, ACLS drugs, airway. MICU for post-ROSC care.",
    ))
    add(ConditionSpec(
        "cardiac_arrest_pea", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("adrenaline_1mg_iv", "airway_management_rsi",
                              "iv_access_2x", "point_of_care_ultrasound"),
        equipment_critical_b=("capnography", "12lead_ecg"),
        equipment_important=("atropine_iv",),
        equipment_contra=(), protocol_codes=("AHA_ACLS_PEA", "REVERSIBLE_CAUSES"),
        upgrade_if=(), downgrade_if=("bystander_rosc_confirmed",),
        time_critical_min=10.0,
        notes="PEA: reversible cause identification critical. POCUS for tamponade/tension PTX.",
    ))
    add(ConditionSpec(
        "cardiac_arrest_asystole", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("adrenaline_1mg_iv", "airway_management_rsi",
                              "iv_access_2x"),
        equipment_critical_b=("capnography", "atropine_iv"),
        equipment_important=("transcutaneous_pacing",),
        equipment_contra=(), protocol_codes=("AHA_ACLS_ASYSTOLE",),
        upgrade_if=(), downgrade_if=(),
        time_critical_min=10.0,
        notes="Asystole: poorest prognosis. MICU for family support, palliative decisions.",
    ))
    add(ConditionSpec(
        "stemi_anterior", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.18,),
        equipment_critical_a=("12lead_ecg", "aspirin_300mg", "cath_lab_activation",
                              "iv_access_2x", "defibrillator"),
        equipment_critical_b=("heparin_iv", "nitroglycerin_sl", "morphine_iv",
                              "oxygen_if_spo2_low"),
        equipment_important=("clopidogrel_300mg", "ticagrelor"),
        equipment_contra=("nitroglycerin_if_hypotensive",),
        protocol_codes=("STEMI_ACTIVATION_PROTOCOL", "CATH_LAB_BYPASS",
                        "DOOR_TO_BALLOON_90MIN"),
        upgrade_if=("haemodynamic_instability", "cardiogenic_shock", "vf_onset"),
        downgrade_if=(),
        time_critical_min=90.0,
        notes="Anterior STEMI: highest risk. MICU for cath-lab activation, anti-platelet loading, "
              "defib readiness. Door-to-balloon ≤90 min target.",
    ))
    add(ConditionSpec(
        "stemi_inferior", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.18,),
        equipment_critical_a=("12lead_ecg", "aspirin_300mg", "cath_lab_activation",
                              "iv_access_2x", "defibrillator"),
        equipment_critical_b=("atropine_iv", "nitroglycerin_sl"),
        equipment_important=("right_sided_ecg_leads",),
        equipment_contra=("nitroglycerin_if_rv_infarct",),
        protocol_codes=("STEMI_ACTIVATION_PROTOCOL", "RV_INFARCT_SCREEN"),
        upgrade_if=("rv_infarct_confirmed", "haemodynamic_instability"),
        downgrade_if=(),
        time_critical_min=90.0,
        notes="Inferior STEMI: RV infarct risk — nitro may cause severe hypotension.",
    ))
    add(ConditionSpec(
        "stemi_posterior", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.18,),
        equipment_critical_a=("12lead_ecg", "posterior_ecg_leads", "aspirin_300mg",
                              "cath_lab_activation", "iv_access_2x"),
        equipment_critical_b=("defibrillator", "anticoagulation"),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("STEMI_ACTIVATION_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=90.0,
        notes="Posterior STEMI often missed on standard 12-lead. V7-V9 leads required.",
    ))
    add(ConditionSpec(
        "nstemi_unstable_angina", "cardiac",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("12lead_ecg", "aspirin_300mg", "iv_access"),
        equipment_critical_b=("nitroglycerin_sl", "oxygen_if_spo2_low", "defibrillator"),
        equipment_important=("heparin_iv", "morphine_iv"),
        equipment_contra=(),
        protocol_codes=("NSTEMI_ACS_PROTOCOL",),
        upgrade_if=("dynamic_ecg_changes", "troponin_positive_field",
                    "haemodynamic_instability"),
        downgrade_if=("pain_resolved", "vitals_stable"),
        time_critical_min=120.0,
        notes="NSTEMI/UA: ALS sufficient unless haemodynamically unstable.",
    ))
    add(ConditionSpec(
        "cardiogenic_shock", "cardiac",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("vasopressors_noradrenaline", "iv_access_2x",
                              "arterial_line_prep", "12lead_ecg",
                              "inotropes_dobutamine", "airway_management_rsi"),
        equipment_critical_b=("urinary_catheter", "point_of_care_ultrasound",
                              "defibrillator"),
        equipment_important=("iabp_prep",),
        equipment_contra=(),
        protocol_codes=("CARDIOGENIC_SHOCK_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="Cardiogenic shock: mortality >50%. MICU mandatory — vasopressors, inotropes, IABP prep.",
    ))
    add(ConditionSpec(
        "hypertensive_emergency", "cardiac",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("iv_access", "blood_pressure_monitor_continuous",
                              "labetalol_iv"),
        equipment_critical_b=("12lead_ecg", "nitroprusside"),
        equipment_important=("urinary_catheter",),
        equipment_contra=("rapid_bp_reduction_target",),   
        protocol_codes=("HYPERTENSIVE_EMERGENCY_PROTOCOL",),
        upgrade_if=("end_organ_damage_confirmed", "aortic_dissection_suspected"),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Hypertensive emergency: BP >180/120 with end-organ damage. ALS for IV labetalol.",
    ))
    add(ConditionSpec(
        "acute_heart_failure_pulmonary_oedema", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("cpap_bipap", "frusemide_iv", "iv_access",
                              "oxygen_high_flow", "nitrates_iv"),
        equipment_critical_b=("12lead_ecg", "defibrillator"),
        equipment_important=("morphine_iv_low_dose",),
        equipment_contra=(),
        protocol_codes=("ACUTE_HF_PROTOCOL", "SITTING_POSITION"),
        upgrade_if=("intubation_needed", "cardiogenic_shock"),
        downgrade_if=("responding_to_cpap",),
        time_critical_min=30.0,
        notes="Pulmonary oedema: CPAP/BiPAP critical. MICU has full respiratory support stack.",
    ))
    add(ConditionSpec(
        "svt_haemodynamically_stable", "cardiac",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.15,),
        equipment_critical_a=("12lead_ecg", "adenosine_iv", "iv_access"),
        equipment_critical_b=("defibrillator", "valsalva_manoeuvre"),
        equipment_important=("verapamil_iv",),
        equipment_contra=("adenosine_if_wpw",),
        protocol_codes=("SVT_PROTOCOL",),
        upgrade_if=("haemodynamic_compromise", "syncope"),
        downgrade_if=("self_terminated",),
        time_critical_min=180.0,
        notes="Stable SVT: ALS for chemical cardioversion. WPW: avoid AV-nodal agents.",
    ))
    add(ConditionSpec(
        "svt_haemodynamically_unstable", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("synchronised_cardioversion", "sedation_midazolam",
                              "12lead_ecg", "iv_access"),
        equipment_critical_b=("defibrillator", "airway_management"),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("SVT_UNSTABLE_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=20.0,
        notes="Unstable SVT: synchronised cardioversion. Sedation requires MICU airway safety.",
    ))
    add(ConditionSpec(
        "complete_heart_block", "cardiac",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("transcutaneous_pacing", "atropine_iv", "iv_access",
                              "12lead_ecg"),
        equipment_critical_b=("adrenaline_infusion", "defibrillator"),
        equipment_important=("dopamine_infusion",),
        equipment_contra=(),
        protocol_codes=("BRADYCARDIA_PROTOCOL",),
        upgrade_if=("haemodynamic_instability",),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="CHB: transcutaneous pacing. MICU for pacing oversight and drug infusions.",
    ))
    add(ConditionSpec(
        "ischaemic_stroke_thrombolysis_candidate", "stroke",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("stroke_unit_prenotification", "blood_glucose_monitor",
                              "12lead_ecg", "iv_access", "nihss_assessment"),
        equipment_critical_b=("oxygen_target_spo2_95", "ct_request_prearrival"),
        equipment_important=("antiemetics",),
        equipment_contra=("nitroglycerin_if_normotensive", "aggressive_bp_lowering"),
        protocol_codes=("FAST_PROTOCOL", "STROKE_BYPASS", "THROMBOLYSIS_WINDOW"),
        upgrade_if=("gcs_dropping", "herniation_signs"),
        downgrade_if=(),
        time_critical_min=270.0,   
        notes="Ischaemic stroke: pre-notify stroke team, BSL check (hypoglycaemia mimic), "
              "tPA window. MICU for neurological deterioration risk.",
    ))
    add(ConditionSpec(
        "ischaemic_stroke_als_window", "stroke",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("blood_glucose_monitor", "iv_access", "nihss_assessment"),
        equipment_critical_b=("oxygen_if_spo2_low", "12lead_ecg"),
        equipment_important=(),
        equipment_contra=("aggressive_bp_lowering",),
        protocol_codes=("FAST_PROTOCOL",),
        upgrade_if=("gcs_less_than_9", "airway_compromise"),
        downgrade_if=(),
        time_critical_min=270.0,
        notes="Stroke beyond tPA window: ALS for monitoring, BSL, airway readiness.",
    ))
    add(ConditionSpec(
        "haemorrhagic_stroke_ich", "stroke",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("airway_management_rsi", "iv_access_2x",
                              "controlled_bp_labetalol", "mannitol_20pct"),
        equipment_critical_b=("capnography", "point_of_care_ultrasound"),
        equipment_important=("seizure_prophylaxis",),
        equipment_contra=("tpa", "aspirin", "anticoagulants"),
        protocol_codes=("ICH_PROTOCOL", "NEURO_ALERT_ACTIVATION"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="ICH: rapid neurological decline, herniation risk. MICU for RSI, ICP management.",
    ))
    add(ConditionSpec(
        "haemorrhagic_stroke_sah", "stroke",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("airway_management_rsi", "iv_access_2x",
                              "nimodipine_enteral", "seizure_control"),
        equipment_critical_b=("controlled_bp_management", "vomiting_management"),
        equipment_important=(),
        equipment_contra=("aggressive_bp_lowering", "anticoagulants"),
        protocol_codes=("SAH_PROTOCOL", "NEUROSURGERY_ALERT"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="SAH: worst-headache-of-life. MICU for airway, seizure, vasospasm prevention.",
    ))
    add(ConditionSpec(
        "tia_transient", "stroke",
        primary_unit="ALS", acceptable_alts=("BLS",), score_reductions=(0.25,),
        equipment_critical_a=("blood_glucose_monitor", "12lead_ecg", "iv_access"),
        equipment_critical_b=("aspirin_300mg",),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("FAST_PROTOCOL", "TIA_RISK_STRATIFICATION"),
        upgrade_if=("crescendo_tia", "symptoms_recurring"),
        downgrade_if=("fully_resolved_symptoms", "abcd2_score_low"),
        time_critical_min=240.0,
        notes="TIA: ALS for monitoring recurrence and aspirin administration.",
    ))
    add(ConditionSpec(
        "status_epilepticus", "stroke",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.12,),
        equipment_critical_a=("diazepam_iv_im", "airway_management",
                              "blood_glucose_monitor", "iv_access",
                              "lorazepam_iv"),
        equipment_critical_b=("oxygen_high_flow", "phenytoin_iv"),
        equipment_important=("thiamine_if_alcoholic",),
        equipment_contra=(),
        protocol_codes=("STATUS_EPILEPTICUS_PROTOCOL",),
        upgrade_if=("refractory_status", "airway_compromise", "post_ictal_gcs_low"),
        downgrade_if=("seizure_terminated_pre_arrival",),
        time_critical_min=30.0,
        notes="Status epilepticus: benzodiazepines 1st line. MICU if refractory or airway lost.",
    ))
    add(ConditionSpec(
        "refractory_status_epilepticus", "stroke",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("phenobarbitone_iv", "propofol_infusion", "rsi_kit",
                              "airway_management_rsi", "eeg_continuous"),
        equipment_critical_b=("capnography", "iv_access_2x"),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("RSE_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=15.0,
        notes="Refractory SE: anaesthetic agents, intubation required. MICU mandatory.",
    ))
    add(ConditionSpec(
        "meningitis_suspected", "stroke",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("iv_access", "isolation_precautions",
                              "blood_cultures_if_able", "ceftriaxone_iv"),
        equipment_critical_b=("dexamethasone_iv", "oxygen_if_needed"),
        equipment_important=(),
        equipment_contra=("lumbar_puncture_field",),
        protocol_codes=("MENINGITIS_PROTOCOL", "ANTIBIOTICS_WITHIN_1H"),
        upgrade_if=("septic_shock", "petechial_rash", "gcs_below_10"),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Suspected meningitis: IV antibiotics within 1h. ALS for IV access and monitoring.",
    ))
    add(ConditionSpec(
        "raised_icp_herniation", "stroke",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("rsi_kit", "airway_management_rsi",
                              "mannitol_20pct", "controlled_ventilation",
                              "head_30_degrees_elevation"),
        equipment_critical_b=("controlled_bp", "seizure_prophylaxis"),
        equipment_important=("hypertonic_saline",),
        equipment_contra=("neck_flexion", "hypocapnia_below_35"),
        protocol_codes=("HERNIATION_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=20.0,
        notes="Herniation: Cushing's triad, uncal herniation. RSI, controlled hyperventilation.",
    ))
    add(ConditionSpec(
        "acute_severe_asthma", "respiratory",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.12,),
        equipment_critical_a=("salbutamol_nebuliser", "ipratropium_nebuliser",
                              "oxygen_high_flow", "iv_access",
                              "methylprednisolone_iv"),
        equipment_critical_b=("magnesium_sulphate_iv", "capnography"),
        equipment_important=("adrenaline_im_if_near_fatal",),
        equipment_contra=("morphine_iv", "sedation_without_intubation"),
        protocol_codes=("ACUTE_ASTHMA_PROTOCOL", "PEFR_TARGET"),
        upgrade_if=("near_fatal_features", "silent_chest", "spo2_below_88",
                    "life_threatening_features"),
        downgrade_if=("mild_episode", "responding_to_neb"),
        time_critical_min=30.0,
        notes="Acute severe asthma: back-to-back nebs, IV steroids, Mg. MICU if near-fatal.",
    ))
    add(ConditionSpec(
        "near_fatal_asthma", "respiratory",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("rsi_kit", "airway_management_rsi",
                              "ketamine_iv", "adrenaline_im",
                              "heliox_if_available"),
        equipment_critical_b=("magnesium_sulphate_iv", "salbutamol_iv"),
        equipment_important=(),
        equipment_contra=("sedation_without_airway_secured",),
        protocol_codes=("NEAR_FATAL_ASTHMA_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=10.0,
        notes="Near-fatal asthma: silent chest, cyanosis, exhaustion. RSI often needed.",
    ))
    add(ConditionSpec(
        "copd_exacerbation_moderate", "respiratory",
        primary_unit="ALS", acceptable_alts=("BLS",), score_reductions=(0.25,),
        equipment_critical_a=("salbutamol_nebuliser", "ipratropium_nebuliser",
                              "oxygen_controlled_28pct_35pct", "iv_access"),
        equipment_critical_b=("methylprednisolone_iv", "capnography"),
        equipment_important=(),
        equipment_contra=("high_flow_oxygen_uncontrolled", "sedation"),
        protocol_codes=("COPD_EXACERBATION_PROTOCOL", "CONTROLLED_O2"),
        upgrade_if=("respiratory_failure_type2", "spo2_below_85",
                    "confusion_new_onset"),
        downgrade_if=("mild_exacerbation", "good_response"),
        time_critical_min=90.0,
        notes="COPD: controlled O2 critical — hypercapnic drive. ALS for O2 titration and IV.",
    ))
    add(ConditionSpec(
        "copd_respiratory_failure", "respiratory",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("cpap_bipap", "capnography", "iv_access",
                              "controlled_oxygen", "aminophylline_iv"),
        equipment_critical_b=("rsi_kit",),
        equipment_important=(),
        equipment_contra=("high_flow_o2_without_monitoring",),
        protocol_codes=("COPD_NIV_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="Type 2 respiratory failure: NIV/BiPAP first, RSI if failing. MICU for ventilator.",
    ))
    add(ConditionSpec(
        "tension_pneumothorax", "respiratory",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("needle_decompression_14g", "iv_access",
                              "chest_drain_kit"),
        equipment_critical_b=("oxygen_high_flow", "defibrillator"),
        equipment_important=(),
        equipment_contra=("positive_pressure_ventilation_without_decompression",),
        protocol_codes=("TENSION_PTX_PROTOCOL",),
        upgrade_if=("haemothorax", "bilateral_ptx"),
        downgrade_if=(),
        time_critical_min=5.0,
        notes="Tension PTX: needle decompression 2nd ICS MCL. Life-saving. ALS minimum.",
    ))
    add(ConditionSpec(
        "pulmonary_embolism_massive", "respiratory",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("thrombolysis_alteplase", "iv_access_2x",
                              "anticoagulation_heparin", "oxygen_high_flow"),
        equipment_critical_b=("point_of_care_ultrasound", "vasopressors"),
        equipment_important=("surgical_thrombectomy_consult",),
        equipment_contra=("anticoagulants_if_haemorrhagic_risk",),
        protocol_codes=("MASSIVE_PE_PROTOCOL", "THROMBOLYSIS_PE"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Massive PE: haemodynamic collapse, RV failure. Thrombolysis if cardiac arrest imminent.",
    ))
    add(ConditionSpec(
        "pulmonary_embolism_sub_massive", "respiratory",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("iv_access", "anticoagulation_heparin",
                              "oxygen_high_flow"),
        equipment_critical_b=("12lead_ecg", "point_of_care_echo"),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("PE_PROTOCOL",),
        upgrade_if=("haemodynamic_compromise",),
        downgrade_if=(),
        time_critical_min=120.0,
        notes="Sub-massive PE: anticoagulation, monitoring. ALS sufficient unless decompensating.",
    ))
    add(ConditionSpec(
        "anaphylaxis_moderate", "respiratory",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("adrenaline_im_0.5mg", "iv_access",
                              "chlorphenamine_iv", "hydrocortisone_iv",
                              "oxygen_high_flow"),
        equipment_critical_b=("salbutamol_nebuliser", "iv_fluid_bolus"),
        equipment_important=(),
        equipment_contra=("glucagon_if_beta_blocker",),
        protocol_codes=("ANAPHYLAXIS_PROTOCOL",),
        upgrade_if=("refractory_hypotension", "airway_oedema_severe",
                    "cardiac_arrest_anaphylaxis"),
        downgrade_if=(),
        time_critical_min=20.0,
        notes="Anaphylaxis: IM adrenaline is life-saving. ALS for IV steroids and antihistamines.",
    ))
    add(ConditionSpec(
        "anaphylaxis_severe_airway", "respiratory",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("adrenaline_im_1mg", "rsi_kit",
                              "airway_management_rsi", "iv_access_2x",
                              "adrenaline_infusion"),
        equipment_critical_b=("cricothyroidotomy_kit", "iv_fluid_bolus"),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("ANAPHYLAXIS_SEVERE_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=5.0,
        notes="Severe anaphylaxis with airway compromise: can't intubate/ventilate → surgical airway.",
    ))
    add(ConditionSpec(
        "polytrauma_blunt_high_energy", "trauma",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("massive_haemorrhage_protocol", "tourniquet",
                              "iv_access_2x_large_bore", "spinal_precautions",
                              "rsi_kit", "pelvic_binder"),
        equipment_critical_b=("blood_products_preparation", "txa_iv",
                              "extended_fast_ultrasound"),
        equipment_important=("wound_packing_haemostatic",),
        equipment_contra=(),
        protocol_codes=("MAJOR_TRAUMA_PROTOCOL", "DAMAGE_CONTROL_RESUSCITATION",
                        "NEAREST_TRAUMA_30MIN"),
        upgrade_if=("airway_compromise", "haemodynamic_instability"),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="High-energy polytrauma: damage control, blood products, TXA within 3h. MICU.",
    ))
    add(ConditionSpec(
        "polytrauma_penetrating", "trauma",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("haemorrhage_control", "tourniquet", "wound_packing",
                              "iv_access_2x", "txa_iv", "blood_products"),
        equipment_critical_b=("rsi_kit", "decompression_needle"),
        equipment_important=(),
        equipment_contra=("permissive_hypotension_target_90",),
        protocol_codes=("PENETRATING_TRAUMA_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=45.0,
        notes="Penetrating trauma: permissive hypotension, haemorrhage control, scoop and run.",
    ))
    add(ConditionSpec(
        "severe_tbi_gcs_below_9", "trauma",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("rsi_kit", "airway_management_rsi",
                              "mannitol_or_hypertonic_saline",
                              "spinal_precautions",
                              "controlled_ventilation_paco2_35_40"),
        equipment_critical_b=("iv_access_2x", "capnography", "blood_glucose"),
        equipment_important=("seizure_prophylaxis",),
        equipment_contra=("hypotension_syst_below_90",
                          "hypoxia_spo2_below_90",
                          "fever_allow"),
        protocol_codes=("SEVERE_TBI_PROTOCOL", "AIRWAY_EARLY"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Severe TBI: early intubation, avoid secondary injury (hypotension, hypoxia).",
    ))
    add(ConditionSpec(
        "moderate_head_injury_gcs_9_13", "trauma",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("spinal_precautions", "iv_access",
                              "oxygen_if_spo2_low", "blood_glucose"),
        equipment_critical_b=("12lead_ecg",),
        equipment_important=("seizure_watch",),
        equipment_contra=(),
        protocol_codes=("HEAD_INJURY_PROTOCOL",),
        upgrade_if=("gcs_dropping", "bilateral_dilated_pupils"),
        downgrade_if=("gcs_stable_above_14",),
        time_critical_min=120.0,
        notes="Moderate TBI: close monitoring, C-spine. ALS for airway readiness.",
    ))
    add(ConditionSpec(
        "minor_head_injury_gcs_14_15", "trauma",
        primary_unit="BLS", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("spinal_precautions", "assessment_gcs"),
        equipment_critical_b=("ice_pack",),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("MINOR_HEAD_INJURY_PROTOCOL",),
        upgrade_if=("loss_of_consciousness_confirmed", "amnesia_present",
                    "anticoagulant_patient"),
        downgrade_if=(),
        time_critical_min=240.0,
        notes="Minor head injury: NICE criteria for CT. BLS if GCS 15, no LOC, no amnesia.",
    ))
    add(ConditionSpec(
        "spinal_injury_suspected", "trauma",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("spinal_precautions", "log_roll_technique",
                              "cervical_collar", "spinal_board_if_needed"),
        equipment_critical_b=("iv_access", "neurogenic_shock_management"),
        equipment_important=(),
        equipment_contra=("hard_cervical_collar_penetrating_trauma",),
        protocol_codes=("SPINAL_INJURY_PROTOCOL",),
        upgrade_if=("complete_cord_injury", "neurogenic_shock"),
        downgrade_if=("mechanism_not_significant",),
        time_critical_min=120.0,
        notes="Spinal injury: manual in-line stabilisation, log roll. ALS for neuro monitoring.",
    ))
    add(ConditionSpec(
        "major_burns_40_pct", "trauma",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("iv_access_2x", "fluid_resuscitation_parkland",
                              "airway_assessment_inhalation",
                              "cool_running_water_first_20min"),
        equipment_critical_b=("rsi_kit_if_inhalation", "morphine_iv",
                              "dressings_non_adherent"),
        equipment_important=("urinary_catheter",),
        equipment_contra=("ice_application", "butter_application"),
        protocol_codes=("BURNS_MAJOR_PROTOCOL", "PARKLAND_FORMULA"),
        upgrade_if=("inhalation_injury", "facial_burns", "circumferential_burns"),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Major burns: airway oedema progresses rapidly. MICU for early RSI decision.",
    ))
    add(ConditionSpec(
        "minor_burns_less_than_10", "trauma",
        primary_unit="BLS", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("cool_running_water_first_20min", "dressings"),
        equipment_critical_b=("pain_assessment",),
        equipment_important=("analgesia_paracetamol",),
        equipment_contra=("ice", "butter", "toothpaste"),
        protocol_codes=("MINOR_BURNS_PROTOCOL",),
        upgrade_if=("face_hands_genitalia", "under_5_years", "chemical_burn"),
        downgrade_if=(),
        time_critical_min=300.0,
        notes="Minor burns < 10% BSA, superficial. BLS with cooling and dressings.",
    ))
    add(ConditionSpec(
        "crush_injury_syndrome", "trauma",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("iv_fluid_aggressive_saline",
                              "ecg_monitor_hyperkalaemia",
                              "iv_access_2x", "calcium_gluconate_iv"),
        equipment_critical_b=("tourniquet_if_limb_threatening",
                              "urine_output_monitoring"),
        equipment_important=("bicarbonate_iv",),
        equipment_contra=("rapid_release_without_iv_access",),
        protocol_codes=("CRUSH_SYNDROME_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Crush syndrome: reperfusion hyperkalaemia can cause VF on release. Pre-treat.",
    ))
    add(ConditionSpec(
        "minor_extremity_trauma", "trauma",
        primary_unit="BLS", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("splints", "bandages", "ice_pack"),
        equipment_critical_b=("assessment_neurovascular_status",),
        equipment_important=("analgesia_paracetamol",),
        equipment_contra=(),
        protocol_codes=("MINOR_TRAUMA_PROTOCOL",),
        upgrade_if=("neurovascular_compromise", "open_fracture"),
        downgrade_if=(),
        time_critical_min=300.0,
        notes="Closed limb fracture, minor lacerations. BLS with immobilisation.",
    ))
    add(ConditionSpec(
        "imminent_delivery_uncomplicated", "obstetric",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("obstetric_kit_delivery", "neonatal_resus_kit",
                              "iv_access", "oxytocin_prepared"),
        equipment_critical_b=("warmed_blankets_neonate", "bulb_syringe",
                              "cord_clamps"),
        equipment_important=("pethidine_if_allowed",),
        equipment_contra=(),
        protocol_codes=("OBSTETRIC_EMERGENCY_PROTOCOL", "NEONATAL_RESUS_READY"),
        upgrade_if=("meconium_stained", "breech_presentation",
                    "cord_prolapse", "pre_term_below_34wk"),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="Uncomplicated imminent delivery: ALS with full obstetric and neonatal kit.",
    ))
    add(ConditionSpec(
        "eclampsia", "obstetric",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("magnesium_sulphate_iv_loading",
                              "iv_access_2x",
                              "labetalol_iv_bp_control",
                              "left_lateral_position",
                              "airway_management_ready"),
        equipment_critical_b=("diazepam_rectal", "calcium_gluconate_antidote"),
        equipment_important=("foetal_monitoring_preparation",),
        equipment_contra=("diazepam_if_mgso4_given",),
        protocol_codes=("ECLAMPSIA_PROTOCOL", "MG_SULPHATE_LOADING"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="Eclampsia: MgSO4 loading dose, controlled BP, airway. MICU for foetal viability.",
    ))
    add(ConditionSpec(
        "postpartum_haemorrhage", "obstetric",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("iv_access_2x_large_bore",
                              "oxytocin_iv_infusion",
                              "bimanual_uterine_compression",
                              "blood_products_request",
                              "ergometrine_im"),
        equipment_critical_b=("misoprostol_pr", "tranexamic_acid_iv",
                              "fluid_resuscitation"),
        equipment_important=("aortic_compression_if_no_blood",),
        equipment_contra=(),
        protocol_codes=("PPH_PROTOCOL", "MASSIVE_HAEMORRHAGE_OBSTETRIC"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="PPH: >500ml post-delivery. TXA within 3h. Blood products, uterotonic drugs.",
    ))
    add(ConditionSpec(
        "ectopic_pregnancy_ruptured", "obstetric",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("iv_access_2x", "fluid_resuscitation",
                              "blood_products", "txa_iv"),
        equipment_critical_b=("point_of_care_ultrasound",),
        equipment_important=(),
        equipment_contra=("delay_for_diagnosis",),
        protocol_codes=("ECTOPIC_PROTOCOL", "HAEMORRHAGIC_SHOCK_PROTOCOL"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=20.0,
        notes="Ruptured ectopic: intra-abdominal haemorrhage. Scoop and run to theatre.",
    ))
    add(ConditionSpec(
        "cord_prolapse", "obstetric",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("knee_chest_position",
                              "cord_keep_moist_warm",
                              "iv_access",
                              "obstetric_emergency_notification"),
        equipment_critical_b=("tocolysis_salbutamol_if_contracting",),
        equipment_important=(),
        equipment_contra=("cord_replacement_field",),
        protocol_codes=("CORD_PROLAPSE_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=10.0,
        notes="Cord prolapse: minutes to foetal death. Manual elevation, theatre within 30 min.",
    ))
    add(ConditionSpec(
        "paediatric_cardiac_arrest", "paediatric",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("paed_sized_defib_pads", "paed_drug_dosing_chart",
                              "io_access", "paed_bvm",
                              "adrenaline_paed_dose"),
        equipment_critical_b=("paed_airway_sizes", "atropine_paed"),
        equipment_important=(),
        equipment_contra=("adult_dose_adrenaline",),
        protocol_codes=("PAED_ACLS_PROTOCOL", "WEIGHT_BASED_DOSING"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=5.0,
        notes="Paediatric arrest: weight-based dosing critical. IO access preferred. MICU.",
    ))
    add(ConditionSpec(
        "paediatric_respiratory_failure", "paediatric",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.12,),
        equipment_critical_a=("paed_bvm", "paed_airway_adjuncts",
                              "salbutamol_paed_neb", "oxygen_paed"),
        equipment_critical_b=("paed_sized_iv_io", "adrenaline_paed_im"),
        equipment_important=(),
        equipment_contra=("adult_cpap_mask",),
        protocol_codes=("PAED_RESPIRATORY_PROTOCOL",),
        upgrade_if=("age_below_2", "spo2_below_90", "wob_severe"),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="Paediatric respiratory failure: age-appropriate equipment critical.",
    ))
    add(ConditionSpec(
        "febrile_seizure_simple", "paediatric",
        primary_unit="ALS", acceptable_alts=("BLS",), score_reductions=(0.25,),
        equipment_critical_a=("blood_glucose_paed", "rectal_diazepam",
                              "buccal_midazolam", "fever_management"),
        equipment_critical_b=("paed_airway", "oxygen_paed"),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("PAED_SEIZURE_PROTOCOL",),
        upgrade_if=("prolonged_greater_15min", "complex_features",
                    "age_below_18_months"),
        downgrade_if=("self_terminated_under_5min",),
        time_critical_min=20.0,
        notes="Simple febrile seizure: rectal diazepam or buccal midazolam. ALS for drug admin.",
    ))
    add(ConditionSpec(
        "paediatric_airway_foreign_body", "paediatric",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("back_blows_chest_thrusts_technique",
                              "paed_airway_adjuncts",
                              "magill_forceps"),
        equipment_critical_b=("suction", "paed_bvm"),
        equipment_important=(),
        equipment_contra=("blind_finger_sweep",),
        protocol_codes=("PAED_CHOKING_PROTOCOL",),
        upgrade_if=("complete_obstruction", "cyanosis", "age_below_1"),
        downgrade_if=(),
        time_critical_min=5.0,
        notes="Paediatric FB airway: back blows/chest thrusts. Magill forceps if visible.",
    ))
    add(ConditionSpec(
        "neonatal_resuscitation", "paediatric",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("neonatal_resus_kit", "neonatal_bvm",
                              "warm_dry_stimulate",
                              "neonatal_sized_laryngoscope",
                              "surfactant_preparation"),
        equipment_critical_b=("paed_pulse_oximetry", "glucose_infusion_d10"),
        equipment_important=("transport_incubator",),
        equipment_contra=("adult_airway_sizes",),
        protocol_codes=("NEONATAL_RESUS_PROTOCOL", "NRP_ALGORITHM"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=5.0,
        notes="Neonatal resus: golden minute concept. MICU with neonatal specialist.",
    ))
    add(ConditionSpec(
        "organophosphate_poisoning", "toxicology",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("atropine_high_dose_iv", "pralidoxime_iv",
                              "decontamination",
                              "airway_management_rsi",
                              "iv_access_2x"),
        equipment_critical_b=("seizure_management", "capnography"),
        equipment_important=(),
        equipment_contra=("morphine", "suxamethonium"),
        protocol_codes=("CHOLINERGIC_TOXIDROME_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="OPC: SLUDGE syndrome. Atropine doses needed can be enormous. MICU for airway.",
    ))
    add(ConditionSpec(
        "opioid_overdose_respiratory_depression", "toxicology",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("naloxone_im_iv_intranasal", "bvm",
                              "oxygen_high_flow"),
        equipment_critical_b=("iv_access", "airway_adjuncts"),
        equipment_important=(),
        equipment_contra=("naloxone_if_chronic_user_high_dose",),
        protocol_codes=("OPIOID_OVERDOSE_PROTOCOL",),
        upgrade_if=("respiratory_arrest", "polysubstance"),
        downgrade_if=("response_to_naloxone",),
        time_critical_min=10.0,
        notes="Opioid OD: naloxone reversal. Cautious dosing in dependent patients.",
    ))
    add(ConditionSpec(
        "tricyclic_antidepressant_overdose", "toxicology",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.15,),
        equipment_critical_a=("sodium_bicarbonate_iv",
                              "iv_access_2x",
                              "seizure_management_benzodiazepine",
                              "ecg_qrs_monitoring",
                              "airway_management"),
        equipment_critical_b=("defibrillator", "lipid_emulsion_20pct"),
        equipment_important=(),
        equipment_contra=("flumazenil", "physostigmine", "class_1a_antiarrhythmics"),
        protocol_codes=("TCA_OVERDOSE_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="TCA OD: QRS >100ms = NaHCO3. Rapid deterioration. MICU for arrhythmia management.",
    ))
    add(ConditionSpec(
        "paracetamol_overdose_acute", "toxicology",
        primary_unit="ALS", acceptable_alts=("BLS",), score_reductions=(0.25,),
        equipment_critical_a=("nac_n_acetylcysteine_iv", "iv_access",
                              "blood_glucose", "activated_charcoal_if_within_1h"),
        equipment_critical_b=("antiemetics_metoclopramide",),
        equipment_important=(),
        equipment_contra=("ipecac",),
        protocol_codes=("PARACETAMOL_OD_PROTOCOL",),
        upgrade_if=("hepatic_failure_signs", "coagulopathy"),
        downgrade_if=("ingestion_over_24h_old_stable",),
        time_critical_min=240.0,
        notes="Paracetamol OD: NAC within 8h for max benefit. ALS for IV NAC initiation.",
    ))
    add(ConditionSpec(
        "snake_bite_neurotoxic", "toxicology",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.12,),
        equipment_critical_a=("pressure_immobilisation_bandage",
                              "iv_access",
                              "antivenom_preparation",
                              "airway_monitoring"),
        equipment_critical_b=("atropine_iv", "neostigmine_iv"),
        equipment_important=(),
        equipment_contra=("tourniquet", "incision_suction"),
        protocol_codes=("SNAKE_BITE_PROTOCOL", "ANTIVENOM_ADMINISTRATION"),
        upgrade_if=("respiratory_paralysis", "ptosis_bilateral"),
        downgrade_if=(),
        time_critical_min=120.0,
        notes="Neurotoxic snake bite: pressure immobilisation, antivenom. ALS for airway monitoring.",
    ))
    add(ConditionSpec(
        "septic_shock", "sepsis",
        primary_unit="MICU", acceptable_alts=("ALS",), score_reductions=(0.18,),
        equipment_critical_a=("iv_access_2x", "fluid_bolus_crystalloid_30ml_kg",
                              "vasopressors_noradrenaline",
                              "antibiotics_within_1h",
                              "blood_cultures"),
        equipment_critical_b=("oxygen_high_flow", "lactate_monitoring",
                              "urinary_catheter"),
        equipment_important=("hydrocortisone_if_refractory",),
        equipment_contra=(),
        protocol_codes=("SEPSIS_6_PROTOCOL", "SURVIVING_SEPSIS"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Septic shock: antibiotics within 1h, fluid resuscitation, vasopressors. MICU.",
    ))
    add(ConditionSpec(
        "severe_sepsis_no_shock", "sepsis",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("iv_access", "fluid_bolus",
                              "oxygen_if_spo2_low"),
        equipment_critical_b=("blood_cultures", "antibiotics_preparation"),
        equipment_important=("lactate_testing",),
        equipment_contra=(),
        protocol_codes=("SEPSIS_PROTOCOL",),
        upgrade_if=("lactate_above_4", "hypotension_developed"),
        downgrade_if=(),
        time_critical_min=90.0,
        notes="Severe sepsis without shock: ALS for IV access, oxygen, monitoring.",
    ))
    add(ConditionSpec(
        "hypoglycaemia_responsive", "medical",
        primary_unit="BLS", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("blood_glucose_monitor", "oral_glucose_gel",
                              "biscuits_if_conscious"),
        equipment_critical_b=(),
        equipment_important=(),
        equipment_contra=("iv_dextrose_if_oral_route_possible",),
        protocol_codes=("HYPOGLYCAEMIA_PROTOCOL",),
        upgrade_if=("unconscious", "unable_to_swallow"),
        downgrade_if=(),
        time_critical_min=60.0,
        notes="Responsive hypoglycaemia: oral glucose. BLS sufficient if swallowing intact.",
    ))
    add(ConditionSpec(
        "hypoglycaemia_unconscious", "medical",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("dextrose_50pct_iv", "iv_access",
                              "blood_glucose_monitor",
                              "glucagon_im"),
        equipment_critical_b=("airway_protection",),
        equipment_important=(),
        equipment_contra=("oral_glucose_if_unconscious",),
        protocol_codes=("HYPOGLYCAEMIA_UNCONSCIOUS_PROTOCOL",),
        upgrade_if=("prolonged_hypoglycaemia", "sulphonylurea_ingestion"),
        downgrade_if=(),
        time_critical_min=30.0,
        notes="Unconscious hypoglycaemia: IV dextrose or IM glucagon. ALS for IV access.",
    ))
    add(ConditionSpec(
        "diabetic_ketoacidosis", "medical",
        primary_unit="ALS", acceptable_alts=("MICU",), score_reductions=(0.10,),
        equipment_critical_a=("iv_access_2x", "fluid_resuscitation_0.9_nacl",
                              "blood_glucose", "potassium_monitoring"),
        equipment_critical_b=("insulin_infusion_prep", "ecg_monitoring"),
        equipment_important=("antiemetics",),
        equipment_contra=("bicarbonate_routine",),
        protocol_codes=("DKA_PROTOCOL",),
        upgrade_if=("gcs_below_12", "haemodynamic_instability",
                    "euglycaemic_dka"),
        downgrade_if=(),
        time_critical_min=120.0,
        notes="DKA: fluid first, then insulin. Potassium critical. ALS for IV management.",
    ))
    add(ConditionSpec(
        "general_weakness_stable_elderly", "medical",
        primary_unit="BLS", acceptable_alts=("ALS",), score_reductions=(0.25,),
        equipment_critical_a=("vital_signs_assessment",),
        equipment_critical_b=("blood_glucose_check",),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("GENERAL_ASSESSMENT",),
        upgrade_if=("abnormal_vitals", "syncope_suspected"),
        downgrade_if=(),
        time_critical_min=300.0,
        notes="Stable elderly general weakness: BLS with monitoring.",
    ))
    add(ConditionSpec(
        "abdominal_pain_stable", "medical",
        primary_unit="BLS", acceptable_alts=("ALS",), score_reductions=(0.20,),
        equipment_critical_a=("pain_assessment",),
        equipment_critical_b=(),
        equipment_important=("pain_score_monitoring",),
        equipment_contra=(),
        protocol_codes=("ABDOMINAL_PAIN_ASSESSMENT",),
        upgrade_if=("rigidity_guarding", "haemodynamic_instability",
                    "suspected_aaa"),
        downgrade_if=(),
        time_critical_min=300.0,
        notes="Stable abdominal pain: BLS transport. ALS if vascular/surgical emergency suspected.",
    ))
    add(ConditionSpec(
        "aortic_aneurysm_rupture_suspected", "medical",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("iv_access_2x_large_bore",
                              "permissive_hypotension_target_70_80",
                              "blood_products", "txa_iv"),
        equipment_critical_b=("point_of_care_ultrasound",),
        equipment_important=(),
        equipment_contra=("aggressive_fluid_resuscitation",),
        protocol_codes=("VASCULAR_EMERGENCY_PROTOCOL",),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=15.0,
        notes="AAA rupture: immediate vascular surgery. Permissive hypotension — avoid diluting clot.",
    ))
    add(ConditionSpec(
        "icu_transfer_ventilated", "transfer",
        primary_unit="MICU", acceptable_alts=(), score_reductions=(),
        equipment_critical_a=("transport_ventilator", "vasopressors_infusion",
                              "iv_access_arterial_line",
                              "monitoring_ecg_spo2_etco2",
                              "emergency_drug_kit"),
        equipment_critical_b=("spare_o2_cylinder", "suction"),
        equipment_important=("sedation_infusion", "blood_products"),
        equipment_contra=(),
        protocol_codes=("ICU_TRANSFER_PROTOCOL", "TRANSPORT_MEDICINE"),
        upgrade_if=(),
        downgrade_if=(),
        time_critical_min=120.0,
        notes="Ventilated ICU transfer: full MICU kit. Anticipate deterioration en route.",
    ))
    add(ConditionSpec(
        "stable_transfer_post_procedure", "transfer",
        primary_unit="ALS", acceptable_alts=("BLS",), score_reductions=(0.20,),
        equipment_critical_a=("monitoring_ecg_spo2", "iv_access",
                              "transfer_documentation"),
        equipment_critical_b=(),
        equipment_important=(),
        equipment_contra=(),
        protocol_codes=("TRANSFER_PROTOCOL",),
        upgrade_if=("recent_intervention", "anticoagulated"),
        downgrade_if=("completely_stable_mobile",),
        time_critical_min=240.0,
        notes="Stable post-procedure transfer: ALS for monitoring, IV access.",
    ))
    return db
_CONDITION_DB: Dict[str, ConditionSpec] = _build_condition_db()
logger.info("Condition DB loaded: %d condition specifications", len(_CONDITION_DB))
@dataclass
class JustificationRubric:
    RUBRIC: Dict[str, Tuple[str, float]] = field(default_factory=lambda: {
        "primary_indication":   ("primary", 0.25),
        "main_reason":          ("primary", 0.25),
        "because":              ("primary", 0.20),
        "vital_signs":          ("secondary", 0.20),
        "gcs":                  ("secondary", 0.10),
        "spo2":                 ("secondary", 0.10),
        "blood_pressure":       ("secondary", 0.10),
        "heart_rate":           ("secondary", 0.10),
        "deteriorating":        ("vitals_interp", 0.15),
        "unstable":             ("vitals_interp", 0.15),
        "critical":             ("vitals_interp", 0.10),
        "stable":               ("vitals_interp", 0.10),
        "improving":            ("vitals_interp", 0.08),
        "protocol":             ("protocol_ref", 0.15),
        "guideline":            ("protocol_ref", 0.12),
        "acls":                 ("protocol_ref", 0.15),
        "als":                  ("protocol_ref", 0.12),
        "micu":                 ("protocol_ref", 0.12),
        "bls":                  ("protocol_ref", 0.08),
        "contraindicated":      ("contra", 0.10),
        "avoid":                ("contra", 0.08),
        "not_recommended":      ("contra", 0.06),
        "upgrade":              ("upgrade_reason", 0.10),
        "downgrade":            ("upgrade_reason", 0.10),
        "higher_level":         ("upgrade_reason", 0.08),
        "lower_level":          ("upgrade_reason", 0.08),
    })
    def score(
        self,
        justification: Optional[str],
        correct_unit:  str,
        chosen_unit:   str,
        condition:     Optional[ConditionSpec],
    ) -> Tuple[float, str]:
        if justification is None:
            if correct_unit == chosen_unit:
                return 0.50, "No justification (correct unit) — partial credit 0.50"
            return 0.15, "No justification (wrong unit) — minimal credit 0.15"
        j_lower = justification.lower()
        component_scores: Dict[str, float] = {}
        for keyword, (component, contrib) in self.RUBRIC.items():
            if keyword.replace("_", " ") in j_lower or keyword in j_lower:
                component_scores[component] = max(
                    component_scores.get(component, 0.0), contrib
                )
        if condition and condition.condition_code.replace("_", " ") in j_lower:
            component_scores["primary"] = min(
                0.25, component_scores.get("primary", 0.0) + 0.10
            )
        if condition:
            for proto in condition.protocol_codes:
                if proto.lower() in j_lower:
                    component_scores["protocol_ref"] = min(
                        0.15, component_scores.get("protocol_ref", 0.0) + 0.05
                    )
        if condition:
            equip_mentions = sum(
                1 for eq in (condition.equipment_critical_a + condition.equipment_critical_b)
                if eq.replace("_", " ") in j_lower or eq in j_lower
            )
            if equip_mentions >= 2:
                component_scores["secondary"] = min(
                    0.20, component_scores.get("secondary", 0.0) + 0.05
                )
        len_bonus = min(0.05, len(justification) / 2000.0)
        raw = (
            component_scores.get("primary", 0.0) +
            component_scores.get("secondary", 0.0) * 0.80 +
            component_scores.get("vitals_interp", 0.0) * 0.80 +
            component_scores.get("protocol_ref", 0.0) * 0.60 +
            component_scores.get("contra", 0.0) * 0.50 +
            component_scores.get("upgrade_reason", 0.0) * 0.50 +
            len_bonus
        )
        if correct_unit == chosen_unit:
            raw = min(1.0, raw * 1.3)
        note = (
            f"Justification: {len(justification)} chars, "
            f"components={list(component_scores.keys())}, "
            f"raw={raw:.3f}"
        )
        return ScoringUtils.clamp(raw), note
class EquipmentChecker:
    @staticmethod
    def score(
        declared:  Optional[List[str]],
        condition: Optional[ConditionSpec],
        unit_type: str,
    ) -> Tuple[float, str]:
        if condition is None:
            return 0.50, "No condition spec — equipment neutral 0.50"
        if not declared:
            defaults = {"BLS": 0.40, "ALS": 0.60, "MICU": 0.70}
            score = defaults.get(unit_type.upper(), 0.40)
            return score, f"No equipment declared — default credit for {unit_type}: {score:.2f}"
        decl_set = {d.lower().replace(" ", "_") for d in declared}
        ca_hits = sum(
            1 for eq in condition.equipment_critical_a
            if any(eq.lower() in d or d in eq.lower() for d in decl_set)
        )
        ca_total = max(len(condition.equipment_critical_a), 1)
        ca_score = 0.40 * (ca_hits / ca_total)
        cb_hits = sum(
            1 for eq in condition.equipment_critical_b
            if any(eq.lower() in d or d in eq.lower() for d in decl_set)
        )
        cb_total = max(len(condition.equipment_critical_b), 1)
        cb_score = 0.30 * (cb_hits / cb_total)
        imp_hits = sum(
            1 for eq in condition.equipment_important
            if any(eq.lower() in d or d in eq.lower() for d in decl_set)
        )
        imp_total = max(len(condition.equipment_important), 1)
        imp_score = 0.20 * (imp_hits / max(imp_total, 1))
        contra_hits = sum(
            1 for eq in condition.equipment_contra
            if any(eq.lower() in d or d in eq.lower() for d in decl_set)
        )
        contra_penalty = 0.10 * (contra_hits / max(len(condition.equipment_contra), 1))
        raw = ca_score + cb_score + imp_score - contra_penalty
        note = (
            f"Equipment: CA={ca_hits}/{ca_total} CB={cb_hits}/{cb_total} "
            f"Imp={imp_hits}/{imp_total} Contra={contra_hits} "
            f"→ {raw:.3f}"
        )
        return ScoringUtils.clamp(raw), note
class ClinicalModifierEngine:
    @staticmethod
    def compute(
        dispatched_unit: str,
        required_unit:   str,
        patient:         Dict[str, Any],
        condition:       Optional[ConditionSpec],
        result:          GraderResult,
    ) -> float:
        net = 0.0
        age        = int(patient.get("age", 45))
        is_pregnant= bool(patient.get("is_pregnant", False))
        severity   = patient.get("severity", "P2")
        gcs        = int(patient.get("gcs", 15))
        spo2       = float(patient.get("spo2_pct", 98.0))
        systolic   = float(patient.get("systolic_bp", 120.0))
        heart_rate = float(patient.get("heart_rate", 80.0))
        rr         = float(patient.get("rr_per_min", 18.0))
        gcs_trend  = str(patient.get("gcs_trend", "stable"))   
        disp       = dispatched_unit.upper()
        req        = required_unit.upper()
        if age < 2:
            if disp == "BLS":
                net -= 0.20
                result.add_note(
                    f"Infant age {age} years: BLS insufficient — "
                    "ALS minimum for all acute presentations of patients < 2 years"
                )
        elif age < 12:
            if disp == "BLS" and severity in ("P1", "P2"):
                net -= 0.10
                result.add_note(
                    f"Paediatric age {age}: BLS for P1/P2 presentation — "
                    "ALS minimum recommended for children < 12 with acute illness"
                )
            elif disp == "MICU" and req == "MICU":
                net += 0.05
                result.add_note("Paediatric P1 dispatched MICU — appropriate bonus")
        if is_pregnant and disp == "BLS":
            net -= 0.15
            result.add_note(
                "Pregnant patient dispatched BLS — obstetric emergencies require "
                "ALS minimum (foetal monitoring capability, obstetric drug access)"
            )
        if age > 75 and disp == "BLS":
            cond_code = patient.get("condition_code", "")
            if any(kw in cond_code for kw in ("cardiac", "stemi", "arrest", "stroke")):
                net -= 0.10
                result.add_note(
                    f"Age {age} with cardiac/stroke presentation: "
                    "BLS inadequate — elderly have higher complication rates"
                )
        if gcs_trend == "falling" and gcs < 13 and disp == "BLS":
            net -= 0.12
            result.add_note(
                f"GCS trending down (current {gcs}) with BLS dispatch — "
                "deteriorating consciousness requires airway-capable crew"
            )
        if spo2 < 85.0 and disp == "BLS":
            net -= 0.15
            result.add_note(
                f"SpO2 {spo2:.0f}% < 85% with BLS — "
                "severe hypoxia requires ALS for CPAP/BiPAP and IV access"
            )
        elif spo2 < 90.0 and disp == "BLS" and severity == "P1":
            net -= 0.08
            result.add_note(
                f"SpO2 {spo2:.0f}% with P1 severity and BLS — "
                "consider ALS for supplemental O2 titration"
            )
        if systolic < 90.0 and disp == "BLS":
            net -= 0.20
            result.add_note(
                f"BP {systolic:.0f} systolic < 90 with BLS — "
                "haemodynamic shock requires IV fluids and vasopressors (ALS/MICU)"
            )
        elif systolic < 100.0 and disp == "BLS" and severity == "P1":
            net -= 0.10
            result.add_note(
                f"BP {systolic:.0f} borderline with P1 and BLS — upgrade considered"
            )
        if heart_rate > 150 and disp == "BLS":
            net -= 0.10
            result.add_note(
                f"HR {heart_rate:.0f} > 150 with BLS — "
                "supraventricular/ventricular tachycardia requires ALS"
            )
        if (rr < 8.0 or rr > 35.0) and disp == "BLS":
            net -= 0.15
            result.add_note(
                f"RR {rr:.0f}/min with BLS — "
                "respiratory failure or apnoea requires airway-capable crew"
            )
        vitals_ok = (
            spo2 >= 95.0 and
            90.0 <= systolic <= 160.0 and
            50 <= heart_rate <= 120 and
            8.0 <= rr <= 25.0 and
            gcs >= 14 and
            gcs_trend != "falling"
        )
        if vitals_ok and disp == req:
            net += 0.05
            result.add_note("All vitals normal with correct unit — clinical stability bonus")
        return ScoringUtils.clamp(net, -0.30, 0.15)
class ResourceEfficiencyScorer:
    _MATRIX: Dict[Tuple[str, str], float] = {
        ("MICU", "P1"): 1.00,
        ("ALS",  "P1"): 0.80,
        ("BLS",  "P1"): 0.10,
        ("ALS",  "P2"): 1.00,
        ("MICU", "P2"): 0.75,
        ("BLS",  "P2"): 0.30,
        ("BLS",  "P3"): 1.00,
        ("ALS",  "P3"): 0.80,
        ("MICU", "P3"): 0.40,
        ("BLS",  "P0"): 1.00,
        ("ALS",  "P0"): 0.80,
        ("MICU", "P0"): 0.60,
    }
    @classmethod
    def score(
        cls,
        dispatched: str,
        severity:   str,
        note_cb:    Any,   
    ) -> Tuple[float, str]:
        key = (dispatched.upper(), severity.upper())
        raw = cls._MATRIX.get(key, 0.50)
        note = f"Efficiency: {dispatched} for {severity} → {raw:.2f}"
        return raw, note
_HARD_PENALTIES: List[Tuple[str, str, str, float, str]] = [
    ("cardiac_arrest", "BLS",
     "BLS for cardiac arrest — MICU mandatory",
     HP_BLS_CARDIAC_ARREST, "EMERGI-ENV Rule H3-1"),
    ("stemi", "BLS",
     "BLS for STEMI — MICU required for cath-lab activation",
     HP_BLS_STEMI, "EMERGI-ENV Rule H3-2"),
    ("eclampsia", "BLS",
     "BLS for eclampsia — MICU mandatory for MgSO4 and airway",
     HP_BLS_ECLAMPSIA, "EMERGI-ENV Rule H3-3"),
    ("respiratory_failure", "BLS",
     "BLS for respiratory failure (SpO2<85%) — ALS minimum for CPAP",
     HP_BLS_RESP_FAILURE, "EMERGI-ENV Rule H3-4"),
    ("septic_shock", "BLS",
     "BLS for septic shock — ALS/MICU for vasopressors",
     HP_BLS_SEPTIC_SHOCK, "EMERGI-ENV Rule H3-5"),
    ("status_epilepticus", "BLS",
     "BLS for status epilepticus — ALS for IV benzodiazepines",
     HP_BLS_STATUS_EPILEPTICUS, "EMERGI-ENV Rule H3-6"),
    ("postresuscitation", "ALS",
     "ALS post-cardiac-arrest — MICU mandatory for vasopressors/cooling",
     HP_ALS_POST_ARREST_VASOP, "EMERGI-ENV Rule H3-7"),
    ("paediatric_cardiac_arrest", "BLS",
     "BLS for paediatric cardiac arrest — MICU required",
     HP_BLS_PAED_SEIZURE, "EMERGI-ENV Rule H3-8"),
]
def _check_hard_penalties(
    condition_code: str,
    dispatched:     str,
    result:         GraderResult,
) -> float:
    total = 0.0
    disp  = dispatched.upper()
    cond  = condition_code.lower()
    for pattern, unit_match, reason, amount, rule_ref in _HARD_PENALTIES:
        if pattern in cond and disp == unit_match.upper():
            result.add_penalty(PenaltyRecord(
                name=f"hard_{pattern[:25]}",
                amount=-amount,
                reason=reason,
                rule_ref=rule_ref,
            ))
            result.critical_mismatches += 1
            result.add_note(f"⚠ HARD PENALTY [{rule_ref}]: {reason}")
            total += amount
    return total
class Task3Grader(BaseGrader):
    TASK_ID          = TASK_ID
    TASK_SEED        = TASK_SEED
    TASK_BASELINE    = TASK_BASELINE
    TASK_DIFFICULTY  = "easy"
    COMPONENT_WEIGHTS: Dict[str, float] = {
        "unit_accuracy":         W_ACCURACY,
        "clinical_justification":W_JUSTIFICATION,
        "equipment_adequacy":    W_EQUIPMENT,
        "resource_efficiency":   W_EFFICIENCY,
    }
    def __init__(self) -> None:
        super().__init__()
        self._justification_rubric = JustificationRubric()
        self._equipment_checker    = EquipmentChecker()
        self._modifier_engine      = ClinicalModifierEngine()
        self._efficiency_scorer    = ResourceEfficiencyScorer()
    def _grade_impl(self, gi: GraderInput, result: GraderResult) -> None:
        if gi.seed != TASK_SEED:
            result.add_note(f"Seed mismatch {gi.seed} ≠ {TASK_SEED}")
        patient = self._extract_patient(gi)
        if patient is None:
            result.status        = GraderStatus.INVALID_INPUT
            result.error_message = "Missing patient_summaries in episode_ledger."
            return
        action = self._extract_unit_action(gi)
        if action is None:
            result.add_note("No unit-type action found — zero score.")
            self._apply_zero_scores(result)
            result.status = GraderStatus.PARTIAL
            return
        condition_code  = patient.get("condition_code", "unknown")
        required_unit   = patient.get("required_unit_type", "ALS").upper()
        severity        = patient.get("severity", "P2")
        condition_spec  = _CONDITION_DB.get(condition_code)
        if condition_spec:
            required_unit = condition_spec.primary_unit
        dispatched_unit = self._parse_unit_type(action)
        if dispatched_unit is None:
            result.add_note("Could not parse unit_type from action — score 0.")
            self._apply_zero_scores(result)
            result.status = GraderStatus.PARTIAL
            return
        result.extra["condition_code"]  = condition_code
        result.extra["required_unit"]   = required_unit
        result.extra["dispatched_unit"] = dispatched_unit
        result.extra["condition_in_db"] = condition_spec is not None
        acc_score, acc_note = self._score_accuracy(
            dispatched_unit, required_unit, condition_spec, patient
        )
        self._add_component(
            result, "unit_accuracy", acc_score, W_ACCURACY, acc_note
        )
        justification = (
            action.get("justification")
            or action.get("reasoning")
            or action.get("rationale")
            or action.get("clinical_reason")
        )
        just_score, just_note = self._justification_rubric.score(
            justification, required_unit, dispatched_unit, condition_spec
        )
        self._add_component(
            result, "clinical_justification", just_score, W_JUSTIFICATION, just_note
        )
        declared_equipment: Optional[List[str]] = (
            action.get("required_equipment")
            or action.get("equipment")
            or action.get("kit_list")
        )
        equip_score, equip_note = self._equipment_checker.score(
            declared_equipment, condition_spec, dispatched_unit
        )
        self._add_component(
            result, "equipment_adequacy", equip_score, W_EQUIPMENT, equip_note
        )
        eff_score, eff_note = self._efficiency_scorer.score(
            dispatched_unit, severity, result.add_note
        )
        self._add_component(
            result, "resource_efficiency", eff_score, W_EFFICIENCY, eff_note
        )
        demo_mod = self._modifier_engine.compute(
            dispatched_unit, required_unit, patient, condition_spec, result
        )
        if abs(demo_mod) > 1e-6:
            if demo_mod < 0:
                self._add_penalty(
                    result, "demographic_vital_modifier",
                    abs(demo_mod),
                    f"Clinical modifier (demographics/vitals): {demo_mod:.3f}",
                    rule_ref="EMERGI-ENV ClinMod v2",
                )
            else:
                result.add_component(
                    ScoringUtils.build_score_component(
                        "clinical_modifier_bonus", demo_mod, 1.0,
                        f"Demographics/vitals bonus: +{demo_mod:.3f}"
                    )
                )
        result.extra["demographic_modifier"] = round(demo_mod, 4)
        proto_bonus = self._apply_protocol_checks_task3(
            action, patient, dispatched_unit, required_unit,
            condition_spec, result
        )
        result.extra["protocol_net"] = round(proto_bonus, 4)
        hard_total = _check_hard_penalties(condition_code, dispatched_unit, result)
        if hard_total > 0:
            result.status = GraderStatus.PARTIAL
        result.extra["hard_penalty_total"] = round(hard_total, 4)
        self._check_upgrade_downgrade(
            dispatched_unit, required_unit, condition_spec, patient, result
        )
        confidence = action.get("confidence")
        if confidence is not None:
            conf_bonus = self._score_confidence_calibration(
                float(confidence), acc_score, result
            )
            result.extra["confidence"] = float(confidence)
            result.extra["confidence_bonus"] = round(conf_bonus, 4)
        result.total_patients   = 1
        result.p1_patients      = 1 if severity == "P1" else 0
        result.p1_survival_rate = float(patient.get("final_survival_prob", 0.85))
        result.extra["condition_spec_available"] = condition_spec is not None
        result.extra["condition_categories"] = (
            condition_spec.category if condition_spec else "unknown"
        )
    def _score_accuracy(
        self,
        dispatched:     str,
        required:       str,
        spec:           Optional[ConditionSpec],
        patient:        Dict[str, Any],
    ) -> Tuple[float, str]:
        d_rank = UNIT_RANK.get(dispatched, 0)
        r_rank = UNIT_RANK.get(required, 2)
        base = ACCURACY_MATRIX.get((d_rank, r_rank), 0.05)
        alt_score = base
        if spec and dispatched in spec.acceptable_alts:
            idx = list(spec.acceptable_alts).index(dispatched)
            reduction = spec.score_reductions[idx] if idx < len(spec.score_reductions) else 0.10
            alt_score = max(base, 1.0 - reduction)
            base = alt_score
        n_patients = int(patient.get("patient_count", 1))
        if n_patients >= 3 and dispatched == "BLS":
            base = max(0.10, base - 0.15)
        elapsed = float(patient.get("time_elapsed_min", 0.0))
        if spec and elapsed > spec.time_critical_min * 0.8 and dispatched != required:
            base = max(0.05, base - 0.10)
        note = (
            f"Unit: {dispatched} (req: {required}) | "
            f"Rank: {d_rank}vs{r_rank} | "
            f"Base: {base:.3f}"
        )
        return ScoringUtils.clamp(base), note
    def _apply_protocol_checks_task3(
        self,
        action:         ActionLogEntry,
        patient:        Dict[str, Any],
        dispatched:     str,
        required:       str,
        spec:           Optional[ConditionSpec],
        result:         GraderResult,
    ) -> float:
        cond = patient.get("condition_code", "unknown")
        net  = 0.0
        ok, delta, reason = ProtocolRuleChecker.check_micu_for_stemi(dispatched, cond)
        if reason != "n/a":
            net += delta
            result.add_note(reason)
            if delta < 0:
                self._add_penalty(result, "stemi_unit_protocol",
                                  abs(delta), reason, "EMERGI-ENV Protocol v2")
                result.protocol_violations += 1
        ok2, delta2, reason2 = ProtocolRuleChecker.check_als_for_stroke(dispatched, cond)
        if reason2 != "n/a":
            net += delta2
            result.add_note(reason2)
            if delta2 < 0:
                self._add_penalty(result, "stroke_unit_protocol",
                                  abs(delta2), reason2, "EMERGI-ENV Protocol v2")
                result.protocol_violations += 1
        is_mci = bool(patient.get("is_mci", False))
        start  = bool(action.get("start_protocol_applied", False))
        ok3, delta3, reason3 = ProtocolRuleChecker.check_start_triage_in_mci(start, is_mci)
        if reason3 != "n/a":
            net += delta3
            result.add_note(reason3)
        trapped   = bool(patient.get("requires_extrication", False))
        multi_ag  = bool(action.get("multi_agency_coordinated", False))
        ok4, delta4, reason4 = ProtocolRuleChecker.check_multi_agency_for_trapped(
            multi_ag, trapped, cond
        )
        if reason4 != "n/a":
            net += delta4
            result.add_note(reason4)
            if delta4 < 0:
                self._add_penalty(result, "multi_agency_protocol",
                                  abs(delta4), reason4, "EMERGI-ENV Protocol v2")
                result.protocol_violations += 1
        if dispatched == required:
            net += 0.015
            result.add_note(f"Correct unit {dispatched} — protocol compliance bonus +0.015")
        capped = ScoringUtils.clamp(net, -PROTOCOL_BONUS_CAP, PROTOCOL_BONUS_CAP)
        if capped > 0:
            result.add_component(ScoringUtils.build_score_component(
                "protocol_bonus", capped, 1.0,
                f"Protocol net bonus: {net:.3f} → capped {capped:.3f}"
            ))
        elif capped < 0:
            self._add_penalty(result, "protocol_net_penalty", abs(capped),
                              f"Protocol net penalty: {net:.3f}")
        return capped
    def _check_upgrade_downgrade(
        self,
        dispatched: str,
        required:   str,
        spec:       Optional[ConditionSpec],
        patient:    Dict[str, Any],
        result:     GraderResult,
    ) -> None:
        if spec is None:
            return
        active_modifiers = {
            k.lower().replace(" ", "_"): v
            for k, v in patient.items()
            if isinstance(v, bool) and v
        }
        for upgrade_cond in spec.upgrade_if:
            uc = upgrade_cond.lower().replace(" ", "_")
            if active_modifiers.get(uc, False):
                d_rank = UNIT_RANK.get(dispatched, 1)
                r_rank = UNIT_RANK.get(required, 2)
                if d_rank >= r_rank + 1:
                    result.add_note(
                        f"Upgrade condition '{upgrade_cond}' active — "
                        f"agent correctly dispatched {dispatched}"
                    )
                else:
                    result.add_note(
                        f"Upgrade condition '{upgrade_cond}' active — "
                        f"agent did not upgrade to {required} (dispatched {dispatched})"
                    )
        for down_cond in spec.downgrade_if:
            dc = down_cond.lower().replace(" ", "_")
            if active_modifiers.get(dc, False):
                d_rank = UNIT_RANK.get(dispatched, 1)
                r_rank = UNIT_RANK.get(required, 2)
                if d_rank <= r_rank:
                    result.add_note(
                        f"Downgrade condition '{down_cond}' active — "
                        f"agent efficiently dispatched {dispatched}"
                    )
    def _score_confidence_calibration(
        self,
        confidence:     float,
        accuracy_score: float,
        result:         GraderResult,
    ) -> float:
        conf  = ScoringUtils.clamp(confidence)
        delta = abs(conf - accuracy_score)
        if delta < CONF_EXCELLENT_DELTA:
            bonus = CONF_EXCELLENT_BONUS
            result.add_note(
                f"Confidence calibration excellent: conf={conf:.2f} vs "
                f"accuracy={accuracy_score:.2f} Δ={delta:.3f} → +{bonus:.3f}"
            )
        elif delta < CONF_GOOD_DELTA:
            bonus = CONF_GOOD_BONUS
            result.add_note(
                f"Confidence calibration good: Δ={delta:.3f} → +{bonus:.3f}"
            )
        else:
            bonus = 0.0
            result.add_note(
                f"Confidence miscalibrated: Δ={delta:.3f} > {CONF_GOOD_DELTA}"
            )
        if bonus > 0:
            result.add_component(ScoringUtils.build_score_component(
                "confidence_calibration", bonus, 1.0,
                f"Confidence bonus: +{bonus:.3f}"
            ))
        return bonus
    def _extract_patient(self, gi: GraderInput) -> Optional[Dict[str, Any]]:
        summaries = gi.all_patient_summaries()
        if not summaries:
            return None
        p = dict(summaries[0])
        p.setdefault("severity", "P1")
        p.setdefault("condition_code", "unknown")
        p.setdefault("required_unit_type", "ALS")
        p.setdefault("age", 45)
        p.setdefault("is_pregnant", False)
        p.setdefault("patient_count", 1)
        p.setdefault("time_elapsed_min", 0.0)
        p.setdefault("gcs", 15)
        p.setdefault("spo2_pct", 98.0)
        p.setdefault("systolic_bp", 120.0)
        p.setdefault("heart_rate", 80.0)
        p.setdefault("rr_per_min", 18.0)
        p.setdefault("gcs_trend", "stable")
        p.setdefault("is_mci", False)
        p.setdefault("requires_extrication", False)
        p.setdefault("final_survival_prob", 0.85)
        return p
    def _extract_unit_action(
        self, gi: GraderInput
    ) -> Optional[ActionLogEntry]:
        for action_type in ("unit_select", "dispatch", "triage",
                            "triage_and_dispatch", "unit_assignment"):
            for entry in gi.action_log:
                if entry.action_type == action_type:
                    return entry
        return None
    def _parse_unit_type(
        self, action: ActionLogEntry
    ) -> Optional[str]:
        raw = (
            action.get("unit_type")
            or action.get("ambulance_type")
            or action.get("unit")
            or action.get("dispatch_type")
        )
        if raw is None:
            return None
        raw = str(raw).upper().strip()
        aliases = {
            "BASIC": "BLS", "BLS": "BLS", "EMT": "BLS",
            "ADVANCED": "ALS", "ALS": "ALS", "PARAMEDIC": "ALS",
            "MOBILE_ICU": "MICU", "MICU": "MICU", "ICU": "MICU",
            "CRITICAL_CARE": "MICU", "CCT": "MICU",
        }
        return aliases.get(raw, raw if raw in UNIT_RANK else None)
    def _apply_zero_scores(self, result: GraderResult) -> None:
        for name, weight in [
            ("unit_accuracy", W_ACCURACY),
            ("clinical_justification", W_JUSTIFICATION),
            ("equipment_adequacy", W_EQUIPMENT),
            ("resource_efficiency", W_EFFICIENCY),
        ]:
            self._add_component(result, name, 0.0, weight, "No action taken")
        self._add_penalty(result, "no_action", 0.20,
                         "Agent issued no unit-type action",
                         rule_ref="EMERGI-ENV Rule N3")
GraderRegistry.register(TASK_ID, Task3Grader)
logger.info(
    "Task3Grader registered: task_id=%s baseline=%.2f seed=%d db_size=%d",
    TASK_ID, TASK_BASELINE, TASK_SEED, len(_CONDITION_DB),
)
def _mk_gi(
    unit:       str   = "MICU",
    condition:  str   = "stemi_anterior",
    severity:   str   = "P1",
    req_unit:   str   = "MICU",
    age:        int   = 52,
    justification: Optional[str] = None,
    equipment:  Optional[List[str]] = None,
    confidence: Optional[float] = None,
    spo2:       float = 98.0,
    systolic:   float = 120.0,
    hr:         float = 80.0,
    gcs:        int   = 15,
    gcs_trend:  str   = "stable",
    is_pregnant:bool  = False,
    episode_id: str   = "ep-t3-001",
    seed:       int   = TASK_SEED,
) -> GraderInput:
    action_data: Dict[str, Any] = {"unit_type": unit}
    if justification:
        action_data["justification"] = justification
    if equipment:
        action_data["required_equipment"] = equipment
    if confidence is not None:
        action_data["confidence"] = confidence
    return GraderInput(
        task_id=TASK_ID, episode_id=episode_id, seed=seed,
        action_log=[ActionLogEntry(
            step=1, action_type="unit_select",
            action_data=action_data,
        )],
        episode_ledger={"patient_summaries": [{
            "patient_id": "P-003",
            "severity": severity,
            "condition_code": condition,
            "required_unit_type": req_unit,
            "age": age,
            "is_pregnant": is_pregnant,
            "spo2_pct": spo2,
            "systolic_bp": systolic,
            "heart_rate": hr,
            "gcs": gcs,
            "gcs_trend": gcs_trend,
            "rr_per_min": 18.0,
            "patient_count": 1,
            "time_elapsed_min": 0.0,
            "is_mci": False,
            "requires_extrication": False,
            "final_survival_prob": 0.88,
            "weight": 1.0,
        }]},
        observation_log=[], episode_steps=2,
        total_patients=1, p1_patients=1 if severity=="P1" else 0,
    )
def _run_self_tests() -> None:
    grader   = Task3Grader()
    failures: List[str] = []
    def chk(name: str, cond: bool, msg: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]: {msg}")
    r1 = grader.grade(_mk_gi("MICU", "stemi_anterior", "P1", "MICU"))
    chk("T1_range",    0.0 <= r1.final_score <= 1.0)
    chk("T1_baseline", r1.final_score >= TASK_BASELINE,
        f"{r1.final_score:.4f} < {TASK_BASELINE}")
    chk("T1_components", len(r1.components) >= 4)
    r2 = grader.grade(_mk_gi("MICU", "stemi_anterior", "P1", "MICU"))
    chk("T2_determinism", abs(r1.final_score - r2.final_score) < 1e-9,
        f"{r1.final_score} ≠ {r2.final_score}")
    r3 = grader.grade(_mk_gi("BLS", "stemi_anterior", "P1", "MICU"))
    chk("T3_bls_stemi_lower", r3.final_score < r1.final_score)
    chk("T3_hard_penalty",
        any("stemi" in p.name for p in r3.penalties),
        "Missing STEMI hard penalty")
    chk("T3_range", 0.0 <= r3.final_score <= 1.0)
    r4 = grader.grade(_mk_gi("ALS", "stemi_anterior", "P1", "MICU"))
    chk("T4_als_stemi",
        r4.final_score < r1.final_score,
        f"ALS {r4.final_score:.4f} should be < MICU {r1.final_score:.4f}")
    chk("T4_als_better_than_bls", r4.final_score > r3.final_score,
        f"ALS {r4.final_score:.4f} should be > BLS {r3.final_score:.4f}")
    chk("T4_range", 0.0 <= r4.final_score <= 1.0)
    r5 = grader.grade(_mk_gi("BLS", "general_weakness_stable_elderly", "P3", "BLS"))
    chk("T5_bls_for_p3", 0.0 <= r5.final_score <= 1.0)
    chk("T5_bls_p3_good", r5.final_score >= TASK_BASELINE,
        f"BLS for P3 {r5.final_score:.4f} should beat baseline")
    r6 = grader.grade(_mk_gi("BLS", "febrile_seizure_simple", "P1", "ALS", age=1))
    chk("T6_infant_bls_penalised",
        any("demographic" in p.name for p in r6.penalties),
        "Infant BLS should trigger demographic penalty")
    chk("T6_range", 0.0 <= r6.final_score <= 1.0)
    r7 = grader.grade(_mk_gi("BLS", "imminent_delivery_uncomplicated",
                             "P1", "ALS", is_pregnant=True))
    chk("T7_pregnant_bls", r7.final_score < r1.final_score)
    chk("T7_range", 0.0 <= r7.final_score <= 1.0)
    r8a = grader.grade(_mk_gi("MICU", "stemi_anterior"))
    r8b = grader.grade(_mk_gi("MICU", "stemi_anterior",
        justification=(
            "STEMI anterior — MICU required for cath_lab activation protocol. "
            "Patient vitals: BP 90/60, HR 115, GCS 14 (stable). "
            "12-lead ECG shows ST elevation V1-V4. "
            "IV access established, aspirin 300mg given. Defibrillator on standby."
        ),
        equipment=["12lead_ecg", "aspirin_300mg", "defibrillator",
                   "iv_access_2x", "cath_lab_activation"],
    ))
    j_comp_a = next(c for c in r8a.components if "justification" in c.name)
    j_comp_b = next(c for c in r8b.components if "justification" in c.name)
    chk("T8_justification_higher",
        j_comp_b.raw_score > j_comp_a.raw_score,
        f"With justification {j_comp_b.raw_score:.4f} should > without {j_comp_a.raw_score:.4f}")
    r9 = grader.grade(_mk_gi("MICU", "stemi_anterior", confidence=0.95))
    chk("T9_conf_range", 0.0 <= r9.final_score <= 1.0)
    chk("T9_conf_key", "confidence" in r9.extra)
    r10 = grader.grade(_mk_gi("BLS", "copd_respiratory_failure", "P1", "MICU",
                              spo2=82.0, systolic=100.0))
    chk("T10_hypoxia_penalty",
        any("demographic" in p.name or "modifier" in p.name for p in r10.penalties),
        "SpO2<85% with BLS should trigger vital-sign penalty")
    chk("T10_range", 0.0 <= r10.final_score <= 1.0)
    r11 = grader.grade(_mk_gi("BLS", "septic_shock", "P1", "MICU",
                              systolic=75.0))
    chk("T11_shock_bls_penalised",
        r11.final_score < r1.final_score)
    chk("T11_range", 0.0 <= r11.final_score <= 1.0)
    gi_na = GraderInput(
        task_id=TASK_ID, episode_id="ep-na", seed=TASK_SEED,
        action_log=[], episode_ledger={"patient_summaries": [{
            "patient_id": "P-x", "severity": "P1",
            "condition_code": "stemi_anterior",
            "required_unit_type": "MICU", "weight": 1.0,
        }]},
        observation_log=[], episode_steps=0,
        total_patients=1, p1_patients=1,
    )
    r12 = grader.grade(gi_na)
    chk("T12_no_action_zero", r12.final_score == 0.0)
    chk("T12_status", r12.status == GraderStatus.PARTIAL)
    chk("T13_db_size", len(_CONDITION_DB) >= 40,
        f"DB has {len(_CONDITION_DB)} entries, expected ≥40")
    d = r1.as_dict()
    chk("T14_dict", all(k in d for k in ("final_score","components","penalties")))
    chk("T14_json", len(r1.as_json()) > 200)
    if failures:
        for f in failures:
            logger.error(f)
        raise AssertionError(
            f"Task3Grader: {len(failures)} failure(s):\n" + "\n".join(failures)
        )
    logger.info("Task3Grader self-test PASSED (14 test cases, DB=%d specs).",
                len(_CONDITION_DB))
try:
    _run_self_tests()
except Exception as _e:
    logger.error("Task3Grader self-test FAILED: %s", _e)
    raise
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    print("=" * 72)
    print("EMERGI-ENV  ·  Task3Grader  ·  Unit Type Matching Demo")
    print(f"  Condition DB: {len(_CONDITION_DB)} specifications")
    print("=" * 72)
    grader = Task3Grader()
    scenarios = [
        ("STEMI anterior — MICU (correct)",
         _mk_gi("MICU", "stemi_anterior", "P1", "MICU",
                justification="STEMI cath_lab_activation required. MICU for ACLS protocol.",
                equipment=["12lead_ecg","aspirin_300mg","defibrillator","iv_access_2x"],
                confidence=0.92)),
        ("STEMI anterior — ALS (acceptable alt)",
         _mk_gi("ALS", "stemi_anterior", "P1", "MICU", confidence=0.80)),
        ("STEMI anterior — BLS (critical failure)",
         _mk_gi("BLS", "stemi_anterior", "P1", "MICU")),
        ("Cardiac arrest VF — MICU (correct)",
         _mk_gi("MICU", "cardiac_arrest_vf", "P1", "MICU",
                justification="VF arrest: defibrillation, adrenaline, airway. ACLS protocol.")),
        ("Status epilepticus — ALS (correct)",
         _mk_gi("ALS", "status_epilepticus", "P1", "ALS",
                justification="Diazepam IV, airway monitoring. ALS for IV benzodiazepines.")),
        ("Status epilepticus — BLS (hard penalty)",
         _mk_gi("BLS", "status_epilepticus", "P1", "ALS")),
        ("General weakness P3 — BLS (correct, efficient)",
         _mk_gi("BLS", "general_weakness_stable_elderly", "P3", "BLS")),
        ("General weakness P3 — MICU (over-resource)",
         _mk_gi("MICU", "general_weakness_stable_elderly", "P3", "BLS")),
        ("Infant seizure — BLS (demographic penalty)",
         _mk_gi("BLS", "febrile_seizure_simple", "P1", "ALS", age=1)),
        ("Paediatric seizure — ALS + good justification",
         _mk_gi("ALS", "febrile_seizure_simple", "P1", "ALS", age=3,
                justification="Febrile seizure: rectal diazepam or buccal midazolam required. "
                              "ALS for drug administration and paed airway.",
                equipment=["rectal_diazepam","buccal_midazolam","paed_airway"],
                confidence=0.88)),
        ("Eclampsia — BLS (critical)",
         _mk_gi("BLS", "eclampsia", "P1", "MICU", is_pregnant=True)),
        ("COPD resp failure — MICU (correct)",
         _mk_gi("MICU", "copd_respiratory_failure", "P1", "MICU",
                spo2=84.0, hr=118.0,
                justification="Type 2 respiratory failure, SpO2 84%, BiPAP/CPAP required. MICU.")),
        ("Septic shock — MICU (correct)",
         _mk_gi("MICU", "septic_shock", "P1", "MICU",
                systolic=82.0, hr=130.0,
                justification="Septic shock: vasopressors, Sepsis-6 protocol. MICU mandatory.")),
        ("Opioid OD — ALS (correct)",
         _mk_gi("ALS", "opioid_overdose_respiratory_depression", "P2", "ALS",
                equipment=["naloxone_im_iv_intranasal","bvm","oxygen_high_flow"])),
        ("Snake bite neurotoxic — ALS (correct)",
         _mk_gi("ALS", "snake_bite_neurotoxic", "P1", "ALS")),
    ]
    results_all = []
    for name, gi in scenarios:
        res = grader.grade(gi)
        results_all.append(res)
        beat = "✓" if res.beats_baseline else "✗"
        disp = res.extra.get("dispatched_unit", "?")
        req  = res.extra.get("required_unit", "?")
        print(f"\n  [{beat}] {name}")
        print(f"       Score={res.final_score:.4f}  base={TASK_BASELINE:.2f}  "
              f"Δ={res.score_delta_vs_baseline:+.4f}  "
              f"dispatch={disp} (req={req})  status={res.status.value}")
        for c in res.components:
            if c.name in ("protocol_bonus","confidence_calibration","clinical_modifier_bonus"):
                continue
            print(f"         {c.name:<26} {c.raw_score:.4f} × {c.weight:.2f} "
                  f"= {c.weighted:.4f}")
        if res.penalties:
            for p in res.penalties[:2]:
                print(f"         PENALTY {p.name:<28} {p.amount:+.4f}  {p.reason[:50]}")
    print("\n" + "=" * 72)
    beats = sum(1 for r in results_all if r.beats_baseline)
    print(f"  {beats}/{len(results_all)} scenarios beat baseline {TASK_BASELINE:.2f}")
    print(f"  Condition DB: {len(_CONDITION_DB)} specifications across "
          f"{len(set(s.category for s in _CONDITION_DB.values()))} categories")
    print("=" * 72)
    print("\n✅  Task3Grader demo complete.")