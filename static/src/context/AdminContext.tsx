import { createContext, useContext, useReducer, ReactNode, Dispatch } from 'react';

export interface AmbulanceUnit {
  id: string;
  type: 'MICU' | 'ALS' | 'BLS';
  driverName: string;
  driverBadge: string;
  status: 'AVAILABLE' | 'EN_ROUTE' | 'ON_SCENE' | 'WITHDRAWN' | 'SUSPENDED' | 'OFFLINE' | 'TERMINATED';
  zone: string;
  assignedIncidentId: string | null;
  lat: number;
  lng: number;
  eta: number;
  fatigue: number;
  rating: number;
  compliance: number;
  dispatchesToday: number;
  avgResponseTime: number;
  severity: 'IMMEDIATE' | 'URGENT' | 'ROUTINE' | 'STANDBY';
  equipmentStatus: { o2Tank: boolean; defibrillator: boolean; drugKit: boolean };
  suspensionUntil: number | null;
  speedHistory: number[];
  routeWaypoints: [number, number][];
  currentWaypointIndex: number;
}

export interface Incident {
  id: string;
  priority: 'ALPHA' | 'BETA' | 'GAMMA';
  type: string;
  title: string;
  location: string;
  lat: number;
  lng: number;
  vitals: { hr: number; bp: string; spo2: number };
  time: string;
  eta: number;
  responder: string;
  assignedUnitId: string | null;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: 'M' | 'F' | 'Other';
  bloodType: string;
  allergies: string[];
  emergencyContact: string;
  admittedAt: number;
  admittingUnitId: string;
  hospital: string;
  condition: string;
  icd10: string;
  iss: number;
  status: 'STABLE' | 'CRITICAL' | 'ICU' | 'DISCHARGED' | 'DECEASED';
  vitals: { hr: number; bpSystolic: number; bpDiastolic: number; spO2: number; gcs: number };
  triageTag: 'IMMEDIATE' | 'DELAYED' | 'MINIMAL' | 'EXPECTANT';
  treatmentLog: { time: number; action: string; by: string }[];
  medications: { name: string; dose: string; freq: string; lastGiven: string }[];
  ward: string;
  physician: string;
  dischargeDate: number | null;
  outcome: string | null;
}

export interface FeedbackEntry {
  id: string;
  timestamp: number;
  adminName: string;
  unitId: string;
  rating: number;
  categories: string[];
  comment: string;
}

export interface AuditEntry {
  id: string;
  timestamp: number;
  adminName: string;
  action: 'SUSPEND' | 'TERMINATE' | 'REINSTATE' | 'DISPATCH' | 'RECALL' | 'SEVERITY_CHANGE';
  unitId: string;
  driverName: string;
  reason: string;
  details: string;
}

export interface CommandEntry {
  id: string;
  timestamp: number;
  raw: string;
  type: 'VOICE' | 'TEXT';
  parsed: string;
  status: 'EXECUTED' | 'FAILED' | 'UNRECOGNIZED';
}

export interface ToastItem {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
  timeout: number;
}

export interface AdminState {
  isAuthenticated: boolean;
  units: AmbulanceUnit[];
  incidents: Incident[];
  patients: Patient[];
  feedbacks: FeedbackEntry[];
  auditLog: AuditEntry[];
  commandHistory: CommandEntry[];
  voiceActive: boolean;
  selectedUnitId: string | null;
  selectedPatientId: string | null;
  callTrackerUnitId: string | null;
  mapLayers: { incidents: boolean; ambulances: boolean; hospitals: boolean; routes: boolean };
  aiSummaries: Record<string, string>;
  aiLoading: Record<string, boolean>;
  sessionStartTime: number;
  toasts: ToastItem[];
}

export type AdminAction =
  | { type: 'LOGIN' }
  | { type: 'LOGOUT' }
  | { type: 'UPDATE_UNIT'; id: string; updates: Partial<AmbulanceUnit> }
  | { type: 'DISPATCH_UNIT'; unitId: string; incidentId: string }
  | { type: 'RECALL_UNIT'; unitId: string; reason: string; reassignTo?: string }
  | { type: 'SUSPEND_UNIT'; unitId: string; until: number; reason: string }
  | { type: 'TERMINATE_UNIT'; unitId: string }
  | { type: 'REINSTATE_UNIT'; unitId: string }
  | { type: 'SET_SEVERITY'; unitId: string; severity: AmbulanceUnit['severity'] }
  | { type: 'ADD_FEEDBACK'; feedback: FeedbackEntry }
  | { type: 'ADD_AUDIT'; entry: AuditEntry }
  | { type: 'ADD_COMMAND'; entry: CommandEntry }
  | { type: 'SET_SELECTED_UNIT'; id: string | null }
  | { type: 'SET_SELECTED_PATIENT'; id: string | null }
  | { type: 'SET_CALL_TRACKER'; id: string | null }
  | { type: 'TOGGLE_MAP_LAYER'; layer: keyof AdminState['mapLayers'] }
  | { type: 'SET_AI_SUMMARY'; patientId: string; summary: string }
  | { type: 'SET_AI_LOADING'; patientId: string; loading: boolean }
  | { type: 'UPDATE_VITALS' }
  | { type: 'TICK_ETA' }
  | { type: 'MOVE_UNITS' }
  | { type: 'SET_VOICE_ACTIVE'; active: boolean }
  | { type: 'UPDATE_PATIENT'; id: string; updates: Partial<Patient> }
  | { type: 'ADD_TOAST'; toast: ToastItem }
  | { type: 'REMOVE_TOAST'; id: string }
  | { type: 'UPDATE_INCIDENT'; id: string; updates: Partial<Incident> };

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function genWaypoints(start: [number, number], end: [number, number], n = 6): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const jx = (Math.random() - 0.5) * 0.008;
    const jy = (Math.random() - 0.5) * 0.008;
    pts.push([lerp(start[0], end[0], t) + (i > 0 && i < n ? jx : 0), lerp(start[1], end[1], t) + (i > 0 && i < n ? jy : 0)]);
  }
  return pts;
}

let _id = 1000;
function uid() { return `${++_id}`; }

const now = Date.now();

export const HOSPITALS = [
  { id: 'H1', name: 'KEM Hospital', lat: 18.9966, lng: 72.8422, beds: 120, icuBeds: 18, load: 78 },
  { id: 'H2', name: 'Nanavati Hospital', lat: 19.0669, lng: 72.8367, beds: 200, icuBeds: 32, load: 65 },
  { id: 'H3', name: 'Lilavati Hospital', lat: 19.0510, lng: 72.8283, beds: 180, icuBeds: 24, load: 72 },
  { id: 'H4', name: 'Jupiter Hospital Thane', lat: 19.2088, lng: 72.9769, beds: 150, icuBeds: 20, load: 58 },
  { id: 'H5', name: 'Kokilaben Hospital', lat: 19.1269, lng: 72.8250, beds: 250, icuBeds: 40, load: 82 },
];

const initialUnits: AmbulanceUnit[] = [
  { id: 'MICU-01', type: 'MICU', driverName: 'Rajesh Verma', driverBadge: 'MH-4421', status: 'EN_ROUTE', zone: 'Bandra', assignedIncidentId: 'INC-3867', lat: 19.0596, lng: 72.8295, eta: 252, fatigue: 62, rating: 4.8, compliance: 97, dispatchesToday: 5, avgResponseTime: 6.2, severity: 'IMMEDIATE', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [65, 58, 72, 60, 68, 55, 70, 62, 58, 64], routeWaypoints: genWaypoints([19.0596, 72.8295], [19.0178, 72.8478]), currentWaypointIndex: 1 },
  { id: 'MICU-02', type: 'MICU', driverName: 'Sunita Desai', driverBadge: 'MH-4422', status: 'EN_ROUTE', zone: 'Andheri', assignedIncidentId: 'INC-4115', lat: 19.1136, lng: 72.8697, eta: 180, fatigue: 45, rating: 4.5, compliance: 94, dispatchesToday: 3, avgResponseTime: 5.8, severity: 'IMMEDIATE', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [70, 62, 58, 75, 68, 60, 72, 65, 58, 70], routeWaypoints: genWaypoints([19.1136, 72.8697], [19.0726, 72.8795]), currentWaypointIndex: 2 },
  { id: 'ALS-11', type: 'ALS', driverName: 'Mohammed Khan', driverBadge: 'MH-3301', status: 'EN_ROUTE', zone: 'Kurla', assignedIncidentId: 'INC-4079', lat: 19.0726, lng: 72.8795, eta: 390, fatigue: 38, rating: 4.2, compliance: 91, dispatchesToday: 4, avgResponseTime: 7.1, severity: 'URGENT', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: false }, suspensionUntil: null, speedHistory: [55, 48, 62, 50, 58, 45, 60, 52, 48, 55], routeWaypoints: genWaypoints([19.0726, 72.8795], [19.0596, 72.8295]), currentWaypointIndex: 0 },
  { id: 'ALS-12', type: 'ALS', driverName: 'Priya Patil', driverBadge: 'MH-3302', status: 'AVAILABLE', zone: 'Dadar', assignedIncidentId: null, lat: 19.0178, lng: 72.8478, eta: 0, fatigue: 22, rating: 4.6, compliance: 96, dispatchesToday: 2, avgResponseTime: 5.5, severity: 'STANDBY', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], routeWaypoints: [], currentWaypointIndex: 0 },
  { id: 'ALS-13', type: 'ALS', driverName: 'Vikram Joshi', driverBadge: 'MH-3303', status: 'EN_ROUTE', zone: 'Borivali', assignedIncidentId: 'INC-4144', lat: 19.2288, lng: 72.8561, eta: 480, fatigue: 55, rating: 3.9, compliance: 88, dispatchesToday: 6, avgResponseTime: 8.3, severity: 'IMMEDIATE', equipmentStatus: { o2Tank: true, defibrillator: false, drugKit: true }, suspensionUntil: null, speedHistory: [45, 52, 48, 55, 42, 58, 50, 45, 52, 48], routeWaypoints: genWaypoints([19.2288, 72.8561], [19.2088, 72.9769]), currentWaypointIndex: 1 },
  { id: 'ALS-14', type: 'ALS', driverName: 'Anita Sharma', driverBadge: 'MH-3304', status: 'ON_SCENE', zone: 'Thane', assignedIncidentId: 'INC-4102', lat: 19.2183, lng: 72.9781, eta: 0, fatigue: 70, rating: 4.1, compliance: 93, dispatchesToday: 4, avgResponseTime: 6.8, severity: 'ROUTINE', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [0, 0, 52, 48, 55, 42, 0, 0, 0, 0], routeWaypoints: [], currentWaypointIndex: 0 },
  { id: 'BLS-21', type: 'BLS', driverName: 'Deepak Gupta', driverBadge: 'MH-2201', status: 'AVAILABLE', zone: 'Navi Mumbai', assignedIncidentId: null, lat: 19.0330, lng: 73.0297, eta: 0, fatigue: 15, rating: 4.3, compliance: 95, dispatchesToday: 1, avgResponseTime: 9.2, severity: 'STANDBY', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], routeWaypoints: [], currentWaypointIndex: 0 },
  { id: 'BLS-22', type: 'BLS', driverName: 'Kavita Reddy', driverBadge: 'MH-2202', status: 'EN_ROUTE', zone: 'Bandra', assignedIncidentId: 'INC-4122', lat: 19.0540, lng: 72.8340, eta: 540, fatigue: 48, rating: 3.7, compliance: 82, dispatchesToday: 3, avgResponseTime: 10.5, severity: 'URGENT', equipmentStatus: { o2Tank: true, defibrillator: false, drugKit: true }, suspensionUntil: null, speedHistory: [40, 45, 38, 48, 42, 50, 44, 40, 45, 38], routeWaypoints: genWaypoints([19.0540, 72.8340], [19.1136, 72.8697]), currentWaypointIndex: 2 },
  { id: 'BLS-23', type: 'BLS', driverName: 'Ramesh Nair', driverBadge: 'MH-2203', status: 'AVAILABLE', zone: 'Kurla', assignedIncidentId: null, lat: 19.0700, lng: 72.8820, eta: 0, fatigue: 10, rating: 4.0, compliance: 90, dispatchesToday: 0, avgResponseTime: 8.0, severity: 'STANDBY', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], routeWaypoints: [], currentWaypointIndex: 0 },
  { id: 'BLS-24', type: 'BLS', driverName: 'Sneha Kulkarni', driverBadge: 'MH-2204', status: 'EN_ROUTE', zone: 'Dadar', assignedIncidentId: 'INC-4130', lat: 19.0200, lng: 72.8500, eta: 720, fatigue: 35, rating: 4.4, compliance: 92, dispatchesToday: 2, avgResponseTime: 11.0, severity: 'ROUTINE', equipmentStatus: { o2Tank: true, defibrillator: true, drugKit: false }, suspensionUntil: null, speedHistory: [35, 42, 38, 45, 40, 48, 44, 35, 42, 38], routeWaypoints: genWaypoints([19.0200, 72.8500], [19.0330, 73.0297]), currentWaypointIndex: 1 },
  { id: 'ALS-15', type: 'ALS', driverName: 'Amit Tiwari', driverBadge: 'MH-3305', status: 'AVAILABLE', zone: 'Andheri', assignedIncidentId: null, lat: 19.1190, lng: 72.8650, eta: 0, fatigue: 28, rating: 2.8, compliance: 68, dispatchesToday: 1, avgResponseTime: 12.0, severity: 'STANDBY', equipmentStatus: { o2Tank: false, defibrillator: true, drugKit: true }, suspensionUntil: null, speedHistory: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], routeWaypoints: [], currentWaypointIndex: 0 },
  { id: 'BLS-25', type: 'BLS', driverName: 'Fatima Sheikh', driverBadge: 'MH-2205', status: 'SUSPENDED', zone: 'Borivali', assignedIncidentId: null, lat: 19.2300, lng: 72.8580, eta: 0, fatigue: 90, rating: 2.2, compliance: 55, dispatchesToday: 0, avgResponseTime: 15.0, severity: 'STANDBY', equipmentStatus: { o2Tank: false, defibrillator: false, drugKit: false }, suspensionUntil: now + 8 * 3600 * 1000, speedHistory: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], routeWaypoints: [], currentWaypointIndex: 0 },
];

const initialIncidents: Incident[] = [
  { id: 'INC-3867', priority: 'ALPHA', type: 'CARDIAC', title: 'Cardiac Arrest — STEMI', location: 'Bandra West, Hill Road', lat: 19.0540, lng: 72.8260, vitals: { hr: 0, bp: '0/0', spo2: 0 }, time: '0.4s', eta: 252, responder: 'MICU-01', assignedUnitId: 'MICU-01' },
  { id: 'INC-4079', priority: 'BETA', type: 'TRAUMA', title: 'Multi-Vehicle MVA', location: 'Western Express Hwy, Kurla', lat: 19.0750, lng: 72.8720, vitals: { hr: 135, bp: '85/50', spo2: 91 }, time: '1.2m', eta: 390, responder: 'ALS-11', assignedUnitId: 'ALS-11' },
  { id: 'INC-4102', priority: 'GAMMA', type: 'MEDICAL', title: 'Minor Fall — Elderly', location: 'Thane Station Road', lat: 19.1860, lng: 72.9750, vitals: { hr: 88, bp: '130/80', spo2: 97 }, time: '3.5m', eta: 0, responder: 'ALS-14', assignedUnitId: 'ALS-14' },
  { id: 'INC-4115', priority: 'ALPHA', type: 'TRAUMA', title: 'GSW — Abdomen', location: 'Andheri Market Yard', lat: 19.1180, lng: 72.8650, vitals: { hr: 142, bp: '70/40', spo2: 88 }, time: '0.1s', eta: 180, responder: 'MICU-02', assignedUnitId: 'MICU-02' },
  { id: 'INC-4122', priority: 'BETA', type: 'MEDICAL', title: 'Respiratory Distress', location: 'Bandra East, BKC', lat: 19.0650, lng: 72.8600, vitals: { hr: 110, bp: '145/90', spo2: 85 }, time: '4.5m', eta: 540, responder: 'BLS-22', assignedUnitId: 'BLS-22' },
  { id: 'INC-4130', priority: 'GAMMA', type: 'MEDICAL', title: 'Lift Assist', location: 'Navi Mumbai Vashi Sec-17', lat: 19.0760, lng: 72.9980, vitals: { hr: 75, bp: '120/80', spo2: 98 }, time: '8.2m', eta: 720, responder: 'BLS-24', assignedUnitId: 'BLS-24' },
  { id: 'INC-4144', priority: 'ALPHA', type: 'TRAUMA', title: 'Industrial Burn — 40% BSA', location: 'Borivali MIDC Zone', lat: 19.2350, lng: 72.8620, vitals: { hr: 125, bp: '90/60', spo2: 92 }, time: '1.1s', eta: 480, responder: 'ALS-13', assignedUnitId: 'ALS-13' },
  { id: 'INC-4158', priority: 'BETA', type: 'MEDICAL', title: 'Ischaemic Stroke', location: 'Dadar TT Circle', lat: 19.0160, lng: 72.8440, vitals: { hr: 98, bp: '180/110', spo2: 94 }, time: '2.8m', eta: 0, responder: 'UNASSIGNED', assignedUnitId: null },
];

const initialPatients: Patient[] = [
  { id: 'PT-001', name: 'Raj Sharma', age: 58, gender: 'M', bloodType: 'O+', allergies: ['Penicillin'], emergencyContact: '+91 98201 45678', admittedAt: now - 3600000, admittingUnitId: 'MICU-01', hospital: 'KEM Hospital', condition: 'ST-Elevation MI (STEMI)', icd10: 'I21.0', iss: 25, status: 'CRITICAL', vitals: { hr: 42, bpSystolic: 80, bpDiastolic: 50, spO2: 88, gcs: 12 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 3500000, action: 'IV Access established — 18G L antecubital', by: 'Rajesh Verma' }, { time: now - 3400000, action: 'Aspirin 325mg PO administered', by: 'Rajesh Verma' }, { time: now - 3200000, action: '12-Lead ECG — STEMI confirmed', by: 'Dr. Malhotra' }], medications: [{ name: 'Aspirin', dose: '325mg', freq: 'STAT', lastGiven: '14:20' }, { name: 'Heparin', dose: '5000U IV', freq: 'Continuous', lastGiven: '14:35' }, { name: 'Morphine', dose: '4mg IV', freq: 'PRN', lastGiven: '14:45' }], ward: 'CCU-3', physician: 'Dr. S. Malhotra', dischargeDate: null, outcome: null },
  { id: 'PT-002', name: 'Priya Patil', age: 34, gender: 'F', bloodType: 'A+', allergies: [], emergencyContact: '+91 98765 43210', admittedAt: now - 7200000, admittingUnitId: 'ALS-11', hospital: 'Lilavati Hospital', condition: 'Polytrauma — MVA', icd10: 'S09.90', iss: 18, status: 'ICU', vitals: { hr: 112, bpSystolic: 95, bpDiastolic: 60, spO2: 93, gcs: 10 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 7100000, action: 'C-Spine immobilization', by: 'Mohammed Khan' }, { time: now - 7000000, action: 'Bilateral chest decompression', by: 'Mohammed Khan' }, { time: now - 6800000, action: 'RSI + ETT placement', by: 'Dr. Iyer' }], medications: [{ name: 'Propofol', dose: '150mcg/kg/min', freq: 'Continuous', lastGiven: '13:00' }, { name: 'Fentanyl', dose: '50mcg/hr', freq: 'Continuous', lastGiven: '13:00' }], ward: 'SICU-1', physician: 'Dr. R. Iyer', dischargeDate: null, outcome: null },
  { id: 'PT-003', name: 'Mohammed Khan', age: 72, gender: 'M', bloodType: 'B+', allergies: ['Sulfa drugs'], emergencyContact: '+91 87654 32109', admittedAt: now - 14400000, admittingUnitId: 'ALS-12', hospital: 'Nanavati Hospital', condition: 'Ischaemic Stroke — MCA territory', icd10: 'G45.9', iss: 9, status: 'ICU', vitals: { hr: 88, bpSystolic: 165, bpDiastolic: 95, spO2: 96, gcs: 9 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 14300000, action: 'Cincinnati Stroke Scale positive', by: 'Priya Patil' }, { time: now - 14200000, action: 'tPA administered', by: 'Dr. Kapoor' }], medications: [{ name: 'Alteplase (tPA)', dose: '0.9mg/kg IV', freq: 'STAT', lastGiven: '10:00' }, { name: 'Labetalol', dose: '10mg IV', freq: 'Q15min PRN', lastGiven: '11:30' }], ward: 'Stroke Unit', physician: 'Dr. A. Kapoor', dischargeDate: null, outcome: null },
  { id: 'PT-004', name: 'Sunita Desai', age: 28, gender: 'F', bloodType: 'AB-', allergies: ['Latex'], emergencyContact: '+91 76543 21098', admittedAt: now - 21600000, admittingUnitId: 'ALS-13', hospital: 'Kokilaben Hospital', condition: 'Burns 30% BSA — Industrial', icd10: 'T31.3', iss: 22, status: 'CRITICAL', vitals: { hr: 128, bpSystolic: 85, bpDiastolic: 55, spO2: 91, gcs: 14 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 21500000, action: 'Parkland formula resuscitation initiated', by: 'Vikram Joshi' }, { time: now - 21400000, action: 'Wound debridement', by: 'Dr. Fernandes' }], medications: [{ name: 'Ringer Lactate', dose: '4ml/kg/%BSA', freq: 'Continuous', lastGiven: '08:00' }, { name: 'Morphine PCA', dose: '1mg/dose', freq: 'Q10min PRN', lastGiven: '13:30' }], ward: 'Burns ICU', physician: 'Dr. L. Fernandes', dischargeDate: null, outcome: null },
  { id: 'PT-005', name: 'Arun Mehta', age: 65, gender: 'M', bloodType: 'O-', allergies: ['Codeine'], emergencyContact: '+91 65432 10987', admittedAt: now - 28800000, admittingUnitId: 'BLS-22', hospital: 'Jupiter Hospital Thane', condition: 'Respiratory Failure — COPD Exacerbation', icd10: 'J96.00', iss: 4, status: 'STABLE', vitals: { hr: 92, bpSystolic: 135, bpDiastolic: 85, spO2: 94, gcs: 15 }, triageTag: 'DELAYED', treatmentLog: [{ time: now - 28700000, action: 'Nebulized Albuterol + Ipratropium', by: 'Kavita Reddy' }, { time: now - 28600000, action: 'BiPAP initiated', by: 'Dr. Shah' }], medications: [{ name: 'Methylprednisolone', dose: '125mg IV', freq: 'Q6H', lastGiven: '12:00' }, { name: 'Albuterol', dose: '2.5mg Neb', freq: 'Q4H', lastGiven: '13:00' }], ward: 'Ward-7B', physician: 'Dr. N. Shah', dischargeDate: null, outcome: null },
  { id: 'PT-006', name: 'Lakshmi Iyer', age: 45, gender: 'F', bloodType: 'A-', allergies: [], emergencyContact: '+91 54321 09876', admittedAt: now - 36000000, admittingUnitId: 'BLS-24', hospital: 'Lilavati Hospital', condition: 'Diabetic Ketoacidosis', icd10: 'E11.10', iss: 1, status: 'STABLE', vitals: { hr: 105, bpSystolic: 110, bpDiastolic: 70, spO2: 97, gcs: 15 }, triageTag: 'DELAYED', treatmentLog: [{ time: now - 35900000, action: 'Insulin drip initiated', by: 'Sneha Kulkarni' }], medications: [{ name: 'Insulin Regular', dose: '0.1U/kg/hr', freq: 'Continuous', lastGiven: '12:00' }, { name: 'NS 0.9%', dose: '500ml/hr', freq: 'Continuous', lastGiven: '12:00' }], ward: 'Ward-4A', physician: 'Dr. P. Rao', dischargeDate: null, outcome: null },
  { id: 'PT-007', name: 'Dinesh Chandra', age: 52, gender: 'M', bloodType: 'B-', allergies: ['Aspirin'], emergencyContact: '+91 43210 98765', admittedAt: now - 43200000, admittingUnitId: 'ALS-14', hospital: 'KEM Hospital', condition: 'Acute Pancreatitis', icd10: 'K85.9', iss: 2, status: 'STABLE', vitals: { hr: 98, bpSystolic: 128, bpDiastolic: 78, spO2: 96, gcs: 15 }, triageTag: 'DELAYED', treatmentLog: [{ time: now - 43100000, action: 'IV fluid resuscitation', by: 'Anita Sharma' }, { time: now - 42000000, action: 'CT abdomen ordered', by: 'Dr. Kulkarni' }], medications: [{ name: 'Pantoprazole', dose: '40mg IV', freq: 'Q12H', lastGiven: '08:00' }, { name: 'Ondansetron', dose: '4mg IV', freq: 'Q8H PRN', lastGiven: '10:00' }], ward: 'Ward-3C', physician: 'Dr. V. Kulkarni', dischargeDate: null, outcome: null },
  { id: 'PT-008', name: 'Neha Singh', age: 19, gender: 'F', bloodType: 'O+', allergies: [], emergencyContact: '+91 32109 87654', admittedAt: now - 50400000, admittingUnitId: 'BLS-21', hospital: 'Nanavati Hospital', condition: 'Anaphylaxis — Insect Sting', icd10: 'T63.441A', iss: 3, status: 'DISCHARGED', vitals: { hr: 78, bpSystolic: 120, bpDiastolic: 75, spO2: 99, gcs: 15 }, triageTag: 'MINIMAL', treatmentLog: [{ time: now - 50300000, action: 'Epinephrine 0.3mg IM', by: 'Deepak Gupta' }, { time: now - 50200000, action: 'Diphenhydramine 50mg IV', by: 'Dr. Menon' }], medications: [{ name: 'EpiPen', dose: '0.3mg', freq: 'PRN', lastGiven: 'Discharged' }], ward: 'Discharged', physician: 'Dr. K. Menon', dischargeDate: now - 25200000, outcome: 'Full recovery' },
  { id: 'PT-009', name: 'Vinod Pillai', age: 41, gender: 'M', bloodType: 'A+', allergies: ['Morphine'], emergencyContact: '+91 21098 76543', admittedAt: now - 57600000, admittingUnitId: 'MICU-02', hospital: 'Kokilaben Hospital', condition: 'Tension Pneumothorax', icd10: 'J93.0', iss: 16, status: 'ICU', vitals: { hr: 118, bpSystolic: 90, bpDiastolic: 55, spO2: 89, gcs: 13 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 57500000, action: 'Needle decompression — 2nd ICS R', by: 'Sunita Desai' }, { time: now - 57400000, action: 'Chest tube insertion', by: 'Dr. Agarwal' }], medications: [{ name: 'Ketamine', dose: '1mg/kg IV', freq: 'PRN', lastGiven: '06:00' }, { name: 'Ceftriaxone', dose: '2g IV', freq: 'Q24H', lastGiven: '08:00' }], ward: 'TICU-2', physician: 'Dr. S. Agarwal', dischargeDate: null, outcome: null },
  { id: 'PT-010', name: 'Geeta Rao', age: 67, gender: 'F', bloodType: 'B+', allergies: ['Iodine'], emergencyContact: '+91 10987 65432', admittedAt: now - 64800000, admittingUnitId: 'ALS-11', hospital: 'Jupiter Hospital Thane', condition: 'Hip Fracture — Fall', icd10: 'S72.001A', iss: 9, status: 'STABLE', vitals: { hr: 82, bpSystolic: 140, bpDiastolic: 85, spO2: 97, gcs: 15 }, triageTag: 'DELAYED', treatmentLog: [{ time: now - 64700000, action: 'Splint applied, pain management', by: 'Mohammed Khan' }], medications: [{ name: 'Tramadol', dose: '50mg IV', freq: 'Q6H', lastGiven: '12:00' }, { name: 'Enoxaparin', dose: '40mg SC', freq: 'Q24H', lastGiven: '08:00' }], ward: 'Ortho-2', physician: 'Dr. D. Patel', dischargeDate: null, outcome: null },
  { id: 'PT-011', name: 'Sanjay Mishra', age: 55, gender: 'M', bloodType: 'AB+', allergies: [], emergencyContact: '+91 09876 54321', admittedAt: now - 72000000, admittingUnitId: 'BLS-23', hospital: 'Lilavati Hospital', condition: 'Acute Appendicitis', icd10: 'K35.80', iss: 2, status: 'DISCHARGED', vitals: { hr: 76, bpSystolic: 125, bpDiastolic: 80, spO2: 99, gcs: 15 }, triageTag: 'DELAYED', treatmentLog: [{ time: now - 71000000, action: 'Laparoscopic appendectomy', by: 'Dr. Reddy' }], medications: [{ name: 'Amoxicillin', dose: '500mg PO', freq: 'Q8H', lastGiven: 'Discharged' }], ward: 'Discharged', physician: 'Dr. M. Reddy', dischargeDate: now - 36000000, outcome: 'Post-op recovery — uneventful' },
  { id: 'PT-012', name: 'Aarti Choudhary', age: 31, gender: 'F', bloodType: 'O-', allergies: ['NSAIDs'], emergencyContact: '+91 98765 12345', admittedAt: now - 5400000, admittingUnitId: 'BLS-22', hospital: 'KEM Hospital', condition: 'Eclampsia — 36wks Gestation', icd10: 'O15.0', iss: 9, status: 'CRITICAL', vitals: { hr: 115, bpSystolic: 170, bpDiastolic: 110, spO2: 93, gcs: 11 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 5300000, action: 'MgSO4 loading dose 4g IV', by: 'Kavita Reddy' }, { time: now - 5200000, action: 'Emergency C-section prep', by: 'Dr. Jha' }], medications: [{ name: 'MgSO4', dose: '1g/hr IV', freq: 'Continuous', lastGiven: '14:00' }, { name: 'Labetalol', dose: '20mg IV', freq: 'Q10min PRN', lastGiven: '14:30' }], ward: 'OB-ICU', physician: 'Dr. S. Jha', dischargeDate: null, outcome: null },
  { id: 'PT-013', name: 'Ravi Kumar', age: 8, gender: 'M', bloodType: 'A+', allergies: [], emergencyContact: '+91 87654 23456', admittedAt: now - 10800000, admittingUnitId: 'ALS-12', hospital: 'Nanavati Hospital', condition: 'Drowning — Near', icd10: 'T75.1', iss: 12, status: 'ICU', vitals: { hr: 130, bpSystolic: 85, bpDiastolic: 50, spO2: 90, gcs: 8 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 10700000, action: 'BVM ventilation initiated', by: 'Priya Patil' }, { time: now - 10600000, action: 'Intubated — 5.0 ETT', by: 'Dr. Deshmukh' }], medications: [{ name: 'Dexamethasone', dose: '0.6mg/kg IV', freq: 'Q6H', lastGiven: '12:00' }], ward: 'PICU', physician: 'Dr. A. Deshmukh', dischargeDate: null, outcome: null },
  { id: 'PT-014', name: 'Meena Bose', age: 78, gender: 'F', bloodType: 'B-', allergies: ['ACE inhibitors'], emergencyContact: '+91 76543 34567', admittedAt: now - 86400000, admittingUnitId: 'MICU-01', hospital: 'Lilavati Hospital', condition: 'Congestive Heart Failure — Acute', icd10: 'I50.9', iss: 4, status: 'STABLE', vitals: { hr: 95, bpSystolic: 150, bpDiastolic: 90, spO2: 92, gcs: 15 }, triageTag: 'DELAYED', treatmentLog: [{ time: now - 86300000, action: 'Furosemide 40mg IV', by: 'Rajesh Verma' }, { time: now - 86200000, action: 'CPAP initiated', by: 'Dr. Malhotra' }], medications: [{ name: 'Furosemide', dose: '40mg IV', freq: 'Q12H', lastGiven: '08:00' }, { name: 'Nitroglycerin', dose: '10mcg/min IV', freq: 'Continuous', lastGiven: '08:00' }], ward: 'CCU-1', physician: 'Dr. S. Malhotra', dischargeDate: null, outcome: null },
  { id: 'PT-015', name: 'Arjun Thakur', age: 22, gender: 'M', bloodType: 'O+', allergies: [], emergencyContact: '+91 65432 45678', admittedAt: now - 1800000, admittingUnitId: 'ALS-13', hospital: 'Kokilaben Hospital', condition: 'Stab Wound — Left Thorax', icd10: 'S21.101A', iss: 20, status: 'CRITICAL', vitals: { hr: 138, bpSystolic: 75, bpDiastolic: 45, spO2: 86, gcs: 11 }, triageTag: 'IMMEDIATE', treatmentLog: [{ time: now - 1700000, action: 'Occlusive dressing applied', by: 'Vikram Joshi' }, { time: now - 1600000, action: 'Massive transfusion protocol activated', by: 'Dr. Fernandes' }], medications: [{ name: 'Tranexamic Acid', dose: '1g IV', freq: 'STAT', lastGiven: '14:50' }, { name: 'O-Neg pRBC', dose: '2U', freq: 'STAT', lastGiven: '14:55' }], ward: 'Trauma-OR', physician: 'Dr. L. Fernandes', dischargeDate: null, outcome: null },
];

const initialFeedbacks: FeedbackEntry[] = [
  { id: uid(), timestamp: now - 86400000, adminName: 'CHIEF DISPATCHER', unitId: 'MICU-01', rating: 5, categories: ['Response Time', 'Patient Handling'], comment: 'Exceptional response to cardiac arrest case. Gold standard.' },
  { id: uid(), timestamp: now - 72000000, adminName: 'CHIEF DISPATCHER', unitId: 'ALS-11', rating: 4, categories: ['Protocol Compliance', 'Communication'], comment: 'Good MVA response, minor documentation delays.' },
  { id: uid(), timestamp: now - 57600000, adminName: 'CHIEF DISPATCHER', unitId: 'BLS-22', rating: 3, categories: ['Response Time'], comment: 'Response time needs improvement. Consider retraining.' },
  { id: uid(), timestamp: now - 43200000, adminName: 'CHIEF DISPATCHER', unitId: 'ALS-14', rating: 4, categories: ['Patient Handling', 'Vehicle Maintenance'], comment: 'Solid performance on Thane scene calls.' },
  { id: uid(), timestamp: now - 28800000, adminName: 'CHIEF DISPATCHER', unitId: 'BLS-25', rating: 2, categories: ['Protocol Compliance', 'Communication'], comment: 'Multiple protocol violations. Under review.' },
  { id: uid(), timestamp: now - 14400000, adminName: 'CHIEF DISPATCHER', unitId: 'MICU-02', rating: 5, categories: ['Response Time', 'Protocol Compliance', 'Patient Handling'], comment: 'Outstanding performance across all metrics.' },
  { id: uid(), timestamp: now - 7200000, adminName: 'CHIEF DISPATCHER', unitId: 'ALS-15', rating: 2, categories: ['Response Time', 'Communication'], comment: 'Consistently slow response times. Verbal warning issued.' },
  { id: uid(), timestamp: now - 3600000, adminName: 'CHIEF DISPATCHER', unitId: 'ALS-12', rating: 5, categories: ['Patient Handling', 'Protocol Compliance'], comment: 'Textbook near-drowning rescue. Commendation filed.' },
];

const initialState: AdminState = {
  isAuthenticated: false,
  units: initialUnits,
  incidents: initialIncidents,
  patients: initialPatients,
  feedbacks: initialFeedbacks,
  auditLog: [
    { id: uid(), timestamp: now - 86400000, adminName: 'CHIEF DISPATCHER', action: 'SUSPEND', unitId: 'BLS-25', driverName: 'Fatima Sheikh', reason: 'Protocol Violation', details: 'Suspended for 8 hours due to repeated protocol violations' },
  ],
  commandHistory: [],
  voiceActive: false,
  selectedUnitId: null,
  selectedPatientId: null,
  callTrackerUnitId: null,
  mapLayers: { incidents: true, ambulances: true, hospitals: true, routes: true },
  aiSummaries: {},
  aiLoading: {},
  sessionStartTime: now,
  toasts: [],
};

function adminReducer(state: AdminState, action: AdminAction): AdminState {
  switch (action.type) {
    case 'LOGIN':
      return { ...state, isAuthenticated: true, sessionStartTime: Date.now() };
    case 'LOGOUT':
      return { ...state, isAuthenticated: false };
    case 'UPDATE_UNIT':
      return { ...state, units: state.units.map(u => u.id === action.id ? { ...u, ...action.updates } : u) };
    case 'DISPATCH_UNIT': {
      const inc = state.incidents.find(i => i.id === action.incidentId);
      return {
        ...state,
        units: state.units.map(u => u.id === action.unitId ? { ...u, status: 'EN_ROUTE' as const, assignedIncidentId: action.incidentId, eta: 300, severity: inc?.priority === 'ALPHA' ? 'IMMEDIATE' as const : 'URGENT' as const, routeWaypoints: inc ? genWaypoints([u.lat, u.lng], [inc.lat, inc.lng]) : u.routeWaypoints, currentWaypointIndex: 0 } : u),
        incidents: state.incidents.map(i => i.id === action.incidentId ? { ...i, assignedUnitId: action.unitId, responder: action.unitId, eta: 300 } : i),
      };
    }
    case 'RECALL_UNIT': {
      const unit = state.units.find(u => u.id === action.unitId);
      const nearestHospital = HOSPITALS[0];
      return {
        ...state,
        units: state.units.map(u => u.id === action.unitId ? {
          ...u, status: 'WITHDRAWN' as const, assignedIncidentId: null, eta: 480,
          routeWaypoints: genWaypoints([u.lat, u.lng], [nearestHospital.lat, nearestHospital.lng]),
          currentWaypointIndex: 0,
        } : u),
        incidents: state.incidents.map(i => i.assignedUnitId === action.unitId ? {
          ...i, assignedUnitId: action.reassignTo || null, responder: action.reassignTo || 'UNASSIGNED',
        } : i),
        auditLog: [{ id: uid(), timestamp: Date.now(), adminName: 'CHIEF DISPATCHER', action: 'RECALL', unitId: action.unitId, driverName: unit?.driverName || '', reason: action.reason, details: `Recalled from ${unit?.assignedIncidentId || 'field'}${action.reassignTo ? ` — Reassigned to ${action.reassignTo}` : ''}` }, ...state.auditLog],
      };
    }
    case 'SUSPEND_UNIT': {
      const unit = state.units.find(u => u.id === action.unitId);
      return {
        ...state,
        units: state.units.map(u => u.id === action.unitId ? { ...u, status: 'SUSPENDED' as const, assignedIncidentId: null, suspensionUntil: action.until } : u),
        auditLog: [{ id: uid(), timestamp: Date.now(), adminName: 'CHIEF DISPATCHER', action: 'SUSPEND', unitId: action.unitId, driverName: unit?.driverName || '', reason: action.reason, details: `Suspended until ${new Date(action.until).toLocaleString()}` }, ...state.auditLog],
      };
    }
    case 'TERMINATE_UNIT': {
      const unit = state.units.find(u => u.id === action.unitId);
      return {
        ...state,
        units: state.units.map(u => u.id === action.unitId ? { ...u, status: 'TERMINATED' as const, assignedIncidentId: null } : u),
        auditLog: [{ id: uid(), timestamp: Date.now(), adminName: 'CHIEF DISPATCHER', action: 'TERMINATE', unitId: action.unitId, driverName: unit?.driverName || '', reason: 'Permanent termination', details: 'Unit permanently removed from active registry' }, ...state.auditLog],
      };
    }
    case 'REINSTATE_UNIT': {
      const unit = state.units.find(u => u.id === action.unitId);
      return {
        ...state,
        units: state.units.map(u => u.id === action.unitId ? { ...u, status: 'AVAILABLE' as const, suspensionUntil: null, fatigue: Math.max(u.fatigue - 20, 0) } : u),
        auditLog: [{ id: uid(), timestamp: Date.now(), adminName: 'CHIEF DISPATCHER', action: 'REINSTATE', unitId: action.unitId, driverName: unit?.driverName || '', reason: 'Reinstatement', details: 'Unit reinstated to active duty' }, ...state.auditLog],
      };
    }
    case 'SET_SEVERITY': {
      const unit = state.units.find(u => u.id === action.unitId);
      return {
        ...state,
        units: state.units.map(u => u.id === action.unitId ? { ...u, severity: action.severity } : u),
        auditLog: [{ id: uid(), timestamp: Date.now(), adminName: 'CHIEF DISPATCHER', action: 'SEVERITY_CHANGE', unitId: action.unitId, driverName: unit?.driverName || '', reason: `Severity changed to ${action.severity}`, details: '' }, ...state.auditLog],
      };
    }
    case 'ADD_FEEDBACK': {
      const newRatings = [...state.feedbacks, action.feedback].filter(f => f.unitId === action.feedback.unitId);
      const avgRating = newRatings.reduce((s, f) => s + f.rating, 0) / newRatings.length;
      return {
        ...state,
        feedbacks: [action.feedback, ...state.feedbacks],
        units: state.units.map(u => u.id === action.feedback.unitId ? { ...u, rating: Math.round(avgRating * 10) / 10 } : u),
      };
    }
    case 'ADD_AUDIT':
      return { ...state, auditLog: [action.entry, ...state.auditLog] };
    case 'ADD_COMMAND':
      return { ...state, commandHistory: [action.entry, ...state.commandHistory].slice(0, 50) };
    case 'SET_SELECTED_UNIT':
      return { ...state, selectedUnitId: action.id };
    case 'SET_SELECTED_PATIENT':
      return { ...state, selectedPatientId: action.id };
    case 'SET_CALL_TRACKER':
      return { ...state, callTrackerUnitId: action.id };
    case 'TOGGLE_MAP_LAYER':
      return { ...state, mapLayers: { ...state.mapLayers, [action.layer]: !state.mapLayers[action.layer] } };
    case 'SET_AI_SUMMARY':
      return { ...state, aiSummaries: { ...state.aiSummaries, [action.patientId]: action.summary } };
    case 'SET_AI_LOADING':
      return { ...state, aiLoading: { ...state.aiLoading, [action.patientId]: action.loading } };
    case 'UPDATE_VITALS':
      return {
        ...state,
        incidents: state.incidents.map(i => ({
          ...i,
          vitals: {
            hr: i.vitals.hr === 0 ? 0 : Math.max(30, i.vitals.hr + Math.floor(Math.random() * 5) - 2),
            bp: i.vitals.bp,
            spo2: i.vitals.spo2 === 0 ? 0 : Math.min(100, Math.max(70, i.vitals.spo2 + Math.floor(Math.random() * 3) - 1)),
          }
        })),
        patients: state.patients.map(p => p.status === 'DISCHARGED' || p.status === 'DECEASED' ? p : {
          ...p,
          vitals: {
            ...p.vitals,
            hr: Math.max(30, p.vitals.hr + Math.floor(Math.random() * 5) - 2),
            spO2: Math.min(100, Math.max(70, p.vitals.spO2 + Math.floor(Math.random() * 3) - 1)),
            bpSystolic: Math.max(60, p.vitals.bpSystolic + Math.floor(Math.random() * 5) - 2),
          }
        }),
      };
    case 'TICK_ETA':
      return {
        ...state,
        units: state.units.map(u => u.status === 'EN_ROUTE' || u.status === 'WITHDRAWN' ? { ...u, eta: Math.max(0, u.eta - 1), fatigue: Math.min(100, u.fatigue + 0.01) } : u),
        incidents: state.incidents.map(i => i.eta > 0 ? { ...i, eta: Math.max(0, i.eta - 1) } : i),
      };
    case 'MOVE_UNITS':
      return {
        ...state,
        units: state.units.map(u => {
          if ((u.status !== 'EN_ROUTE' && u.status !== 'WITHDRAWN') || u.routeWaypoints.length < 2) return u;
          const nextIdx = Math.min(u.currentWaypointIndex + 1, u.routeWaypoints.length - 1);
          const curr = u.routeWaypoints[u.currentWaypointIndex] || [u.lat, u.lng];
          const next = u.routeWaypoints[nextIdx] || curr;
          const speed = u.eta > 0 ? Math.min(0.15, 1 / u.eta) : 0.05;
          const newLat = lerp(curr[0], next[0], speed) + (Math.random() - 0.5) * 0.0002;
          const newLng = lerp(curr[1], next[1], speed) + (Math.random() - 0.5) * 0.0002;
          const dist = Math.sqrt((newLat - next[0]) ** 2 + (newLng - next[1]) ** 2);
          const newSpeed = 40 + Math.random() * 40;
          return {
            ...u,
            lat: newLat, lng: newLng,
            currentWaypointIndex: dist < 0.001 ? nextIdx : u.currentWaypointIndex,
            speedHistory: [...u.speedHistory.slice(-9), newSpeed],
          };
        }),
      };
    case 'SET_VOICE_ACTIVE':
      return { ...state, voiceActive: action.active };
    case 'UPDATE_PATIENT':
      return { ...state, patients: state.patients.map(p => p.id === action.id ? { ...p, ...action.updates } : p) };
    case 'ADD_TOAST':
      return { ...state, toasts: [...state.toasts, action.toast] };
    case 'REMOVE_TOAST':
      return { ...state, toasts: state.toasts.filter(t => t.id !== action.id) };
    case 'UPDATE_INCIDENT':
      return { ...state, incidents: state.incidents.map(i => i.id === action.id ? { ...i, ...action.updates } : i) };
    default:
      return state;
  }
}

interface AdminContextValue {
  state: AdminState;
  dispatch: Dispatch<AdminAction>;
}

const AdminContext = createContext<AdminContextValue | null>(null);

export function AdminProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(adminReducer, initialState);
  return (
    <AdminContext.Provider value={{ state, dispatch }}>
      {children}
    </AdminContext.Provider>
  );
}

export function useAdmin(): AdminContextValue {
  const ctx = useContext(AdminContext);
  if (!ctx) throw new Error('useAdmin must be used within AdminProvider');
  return ctx;
}

export { genWaypoints };
