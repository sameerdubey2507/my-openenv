import { useEffect, useRef, useState, useCallback } from 'react';
import { Network, Activity, Navigation, Radio, Map as MapIcon, Crosshair, Users, AlertTriangle, Shield, CheckCircle, BarChart3, Lock, LogIn, Clock, Eye, EyeOff, Truck, X, ChevronRight, ZoomIn, ZoomOut, Locate } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';
import GlassCard from '../shared/GlassCard';
import StatusChip from '../shared/StatusChip';
import PulseButton from '../shared/PulseButton';
import { useAdmin, HOSPITALS } from '../../context/AdminContext';
import type { AmbulanceUnit, Incident } from '../../context/AdminContext';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler, RadialLinearScale
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler, RadialLinearScale);

interface Props { sim: any; }

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const typeColors: Record<string, string> = { MICU: '#A2C9FF', ALS: '#7DDC7A', BLS: '#FFB74D' };
const prioColors: Record<string, string> = { ALPHA: '#FF544C', BETA: '#FFB74D', GAMMA: '#7DDC7A' };

function makeAmbIcon(type: string) {
  const c = typeColors[type] || '#A2C9FF';
  return L.divIcon({
    className: '',
    iconSize: [14, 14],
    iconAnchor: [7, 7],
    html: `<div style="width:14px;height:14px;background:${c};border:2px solid ${c};border-radius:3px;box-shadow:0 0 8px ${c};animation:ambPulse 1.5s infinite"></div>`,
  });
}

function makeIncIcon(priority: string) {
  const c = prioColors[priority] || '#FF544C';
  return L.divIcon({
    className: '',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    html: `<div style="position:relative;width:20px;height:20px"><div style="position:absolute;inset:0;border-radius:50%;background:${c};opacity:0.9"></div><div style="position:absolute;inset:-4px;border-radius:50%;border:2px solid ${c};animation:ringExpand 1.8s infinite"></div></div>`,
  });
}

function SystemClock() {
  const [t, setT] = useState(new Date());
  const [blink, setBlink] = useState(true);
  useEffect(() => { const i = setInterval(() => { setT(new Date()); setBlink(b => !b); }, 1000); return () => clearInterval(i); }, []);
  const hh = String(t.getHours()).padStart(2, '0');
  const mm = String(t.getMinutes()).padStart(2, '0');
  const ss = String(t.getSeconds()).padStart(2, '0');
  return (
    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--primary)', letterSpacing: 2 }}>
      {hh}<span style={{ opacity: blink ? 1 : 0.2 }}>:</span>{mm}<span style={{ opacity: blink ? 1 : 0.2 }}>:</span>{ss}
    </span>
  );
}

function FlyToUnit({ lat, lng }: { lat: number; lng: number }) {
  const map = useMap();
  useEffect(() => { if (lat && lng) map.flyTo([lat, lng], 14, { duration: 1 }); }, [lat, lng, map]);
  return null;
}

function CallTracker({ unit, onClose }: { unit: AmbulanceUnit; onClose: () => void }) {
  const [gps, setGps] = useState({ lat: unit.lat, lng: unit.lng });
  const [speed, setSpeed] = useState(60);
  const [dist, setDist] = useState(2400);
  const sparkRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const i = setInterval(() => {
      setGps(g => ({ lat: g.lat + (Math.random() - 0.5) * 0.001, lng: g.lng + (Math.random() - 0.5) * 0.001 }));
      setSpeed(40 + Math.floor(Math.random() * 40));
      setDist(d => Math.max(0, d - Math.floor(Math.random() * 80)));
    }, 2000);
    return () => clearInterval(i);
  }, []);

  useEffect(() => {
    const c = sparkRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.strokeStyle = typeColors[unit.type]; ctx.lineWidth = 1.5; ctx.beginPath();
    unit.speedHistory.forEach((v, i) => { const x = (i / 9) * c.width; const y = c.height - (v / 100) * c.height; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
  }, [unit.speedHistory, unit.type]);

  const etaM = Math.floor(unit.eta / 60); const etaS = unit.eta % 60;
  return (
    <motion.div initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ type: 'spring', damping: 25 }}
      style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: 280, background: 'var(--bg-surface-low)', borderLeft: '1px solid var(--ghost-border)', zIndex: 1000, padding: 'var(--space-4)', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: typeColors[unit.type] }}>{unit.id}</span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--on-surface-variant)', cursor: 'pointer' }}><X size={16} /></button>
      </div>
      <StatusChip variant={unit.type === 'MICU' ? 'delta' : unit.type === 'ALS' ? 'gamma' : 'beta'}>{unit.type}</StatusChip>
      <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>Driver: <span style={{ color: 'var(--on-surface)' }}>{unit.driverName}</span></div>
      <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--on-surface-variant)' }}>GPS: {gps.lat.toFixed(4)}, {gps.lng.toFixed(4)}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-2)' }}>
        <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-2)', textAlign: 'center' }}>
          <div className="text-label">SPEED</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--primary)' }}>{speed} km/h</div>
        </div>
        <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-2)', textAlign: 'center' }}>
          <div className="text-label">DISTANCE</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--secondary)' }}>{dist}m</div>
        </div>
      </div>
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-3)', textAlign: 'center' }}>
        <div className="text-label">ETA COUNTDOWN</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.5rem', fontWeight: 800, color: unit.eta < 120 ? '#FF544C' : 'var(--primary)' }}>{String(etaM).padStart(2, '0')}:{String(etaS).padStart(2, '0')}</div>
      </div>
      <div>
        <div className="text-label" style={{ marginBottom: 4 }}>SPEED (LAST 10)</div>
        <canvas ref={sparkRef} width={240} height={40} style={{ width: '100%', height: 40, background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)' }} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)', marginTop: 'auto' }}>
        <PulseButton variant="danger" size="sm" onClick={onClose}>WITHDRAW UNIT</PulseButton>
        <PulseButton variant="ghost" size="sm" onClick={onClose}>CONTACT DRIVER</PulseButton>
      </div>
    </motion.div>
  );
}

function WithdrawModal({ unit, incidents, availableUnits, onConfirm, onClose }: {
  unit: AmbulanceUnit; incidents: Incident[]; availableUnits: AmbulanceUnit[];
  onConfirm: (reason: string, reassignTo: string | undefined) => void; onClose: () => void;
}) {
  const [reason, setReason] = useState('More critical case');
  const [reassignTo, setReassignTo] = useState('');
  const inc = incidents.find(i => i.id === unit.assignedIncidentId);
  return createPortal(
    <AnimatePresence>
      <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} style={{ position: 'absolute', inset: 0, background: 'rgba(5,5,7,0.85)', backdropFilter: 'blur(8px)' }} />
        <motion.div initial={{ opacity: 0, scale: 0.92, y: 20 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.92 }}
          style={{ position: 'relative', width: '100%', maxWidth: 500, background: 'var(--bg-surface-high)', borderRadius: 'var(--radius-xl)', border: '1px solid var(--ghost-border)', padding: 'var(--space-6)', zIndex: 1 }}>
          <h3 style={{ margin: '0 0 var(--space-4)', color: '#FF544C' }}>⚠ RECALL UNIT {unit.id}?</h3>
          {inc && <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-3)', fontSize: '0.8rem' }}>
            <div>Currently responding to: <strong>{inc.id}</strong> — {inc.title}</div>
            <div style={{ color: '#FF544C', marginTop: 4 }}>Warning: This will leave {inc.id} unresponded.</div>
          </div>}
          <div style={{ marginBottom: 'var(--space-3)' }}>
            <label className="text-label">Reason for recall</label>
            <select value={reason} onChange={e => setReason(e.target.value)} style={{ width: '100%', marginTop: 4, padding: 'var(--space-2)', background: 'var(--bg-base)', color: 'var(--on-surface)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-md)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              {['More critical case', 'Hospital override', 'Driver fatigue', 'Equipment failure', 'Manual override'].map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <div style={{ marginBottom: 'var(--space-4)' }}>
            <label className="text-label">Reassign to</label>
            <select value={reassignTo} onChange={e => setReassignTo(e.target.value)} style={{ width: '100%', marginTop: 4, padding: 'var(--space-2)', background: 'var(--bg-base)', color: 'var(--on-surface)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-md)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              <option value="">— No reassignment —</option>
              {availableUnits.map(u => <option key={u.id} value={u.id}>{u.id} ({u.type}) — {u.driverName}</option>)}
            </select>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-3)', justifyContent: 'flex-end' }}>
            <PulseButton variant="ghost" onClick={onClose}>CANCEL</PulseButton>
            <PulseButton variant="danger" onClick={() => onConfirm(reason, reassignTo || undefined)}>CONFIRM WITHDRAWAL</PulseButton>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>, document.body
  );
}

function AdminLoginModal({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) {
  const [user, setUser] = useState(''); const [pass, setPass] = useState('');
  const [showPass, setShowPass] = useState(false); const [error, setError] = useState(false);
  const [success, setSuccess] = useState(false); const [shake, setShake] = useState(false);

  const handleAuth = () => {
    if (user === 'emergent' && pass === 'emergent123') {
      setSuccess(true); setError(false);
      setTimeout(() => { onSuccess(); onClose(); }, 1200);
    } else {
      setError(true); setShake(true);
      setTimeout(() => setShake(false), 500);
    }
  };

  return createPortal(
    <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} style={{ position: 'absolute', inset: 0, background: 'rgba(5,5,7,0.9)', backdropFilter: 'blur(12px)' }} />
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={shake ? { x: [-8, 8, -8, 8, 0], opacity: 1, scale: 1 } : { opacity: 1, scale: 1 }} transition={{ duration: 0.3 }}
        style={{ position: 'relative', width: 400, background: 'var(--bg-surface-high)', borderRadius: 'var(--radius-xl)', border: '1px solid var(--ghost-border)', padding: 'var(--space-6)', zIndex: 1 }}>
        <div style={{ textAlign: 'center', marginBottom: 'var(--space-5)' }}>
          <Shield size={32} style={{ color: 'var(--primary)', marginBottom: 'var(--space-2)' }} />
          <h3 style={{ margin: 0 }}>SECURE ADMIN ACCESS</h3>
        </div>
        {success ? (
          <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} style={{ textAlign: 'center', padding: 'var(--space-6)' }}>
            <CheckCircle size={48} style={{ color: 'var(--secondary)' }} />
            <div style={{ marginTop: 'var(--space-3)', color: 'var(--secondary)', fontFamily: 'var(--font-mono)' }}>ACCESS GRANTED — LOADING ADMIN PANEL...</div>
          </motion.div>
        ) : (
          <>
            <div style={{ marginBottom: 'var(--space-3)' }}>
              <input value={user} onChange={e => setUser(e.target.value)} placeholder="Username" autoComplete="off"
                style={{ width: '100%', padding: 'var(--space-3)', background: 'var(--bg-base)', color: 'var(--on-surface)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-md)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', outline: 'none', boxSizing: 'border-box' }}
                onFocus={e => e.target.style.borderColor = '#A2C9FF'} onBlur={e => e.target.style.borderColor = ''} />
            </div>
            <div style={{ marginBottom: 'var(--space-4)', position: 'relative' }}>
              <input value={pass} onChange={e => setPass(e.target.value)} type={showPass ? 'text' : 'password'} placeholder="Password"
                style={{ width: '100%', padding: 'var(--space-3)', background: 'var(--bg-base)', color: 'var(--on-surface)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-md)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', outline: 'none', boxSizing: 'border-box' }}
                onFocus={e => e.target.style.borderColor = '#A2C9FF'} onBlur={e => e.target.style.borderColor = ''}
                onKeyDown={e => e.key === 'Enter' && handleAuth()} />
              <button onClick={() => setShowPass(!showPass)} style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--on-surface-variant)', cursor: 'pointer' }}>
                {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {error && <motion.div initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} style={{ background: 'rgba(255,84,76,0.15)', color: '#FF544C', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', fontSize: '0.75rem', textAlign: 'center', marginBottom: 'var(--space-3)', fontFamily: 'var(--font-mono)' }}>AUTHENTICATION FAILED — INVALID CREDENTIALS</motion.div>}
            <PulseButton variant="primary" onClick={handleAuth} style={{ width: '100%' }}>
              <LogIn size={16} /> AUTHENTICATE
            </PulseButton>
            <div style={{ textAlign: 'center', marginTop: 'var(--space-3)', fontSize: '0.65rem', color: 'var(--on-surface-dim)', fontFamily: 'var(--font-mono)' }}>ENCRYPTED CHANNEL — AES-256</div>
          </>
        )}
      </motion.div>
    </div>, document.body
  );
}

const chartOptions: any = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false }, tooltip: { backgroundColor: '#0a0d14', borderColor: '#A2C9FF', borderWidth: 1, titleFont: { family: 'JetBrains Mono' }, bodyFont: { family: 'JetBrains Mono' } } },
  scales: {
    x: { grid: { color: 'rgba(120,130,160,0.1)' }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } },
    y: { grid: { color: 'rgba(120,130,160,0.1)' }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } },
  },
  animation: { duration: 800, easing: 'easeInOutQuart' as const },
  interaction: { mode: 'nearest' as const, axis: 'x' as const, intersect: false },
};

export default function CommandCenter({ sim }: Props) {
  const { state, dispatch } = useAdmin();
  const { units, incidents, mapLayers } = state;
  const [loginOpen, setLoginOpen] = useState(false);
  const [withdrawUnit, setWithdrawUnit] = useState<AmbulanceUnit | null>(null);
  const [incidentFilter, setIncidentFilter] = useState<string>('ALL');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [flyTarget, setFlyTarget] = useState<{ lat: number; lng: number } | null>(null);
  const [velData, setVelData] = useState({ labels: ['-60m', '-50m', '-40m', '-30m', '-20m', '-10m', 'NOW'], p1: [2, 4, 3, 7, 5, 12, 18], p2: [15, 18, 14, 22, 28, 25, 30] });
  const [sortByLoad, setSortByLoad] = useState(false);
  const [recalculating, setRecalculating] = useState(false);

  const dispatched = units.filter(u => u.status === 'EN_ROUTE' || u.status === 'ON_SCENE').length;
  const available = units.filter(u => u.status === 'AVAILABLE').length;
  const enRouteUnits = units.filter(u => u.status === 'EN_ROUTE' || u.status === 'ON_SCENE' || u.status === 'WITHDRAWN');
  const activeUnits = units.filter(u => u.status !== 'TERMINATED');
  const availableForReassign = units.filter(u => u.status === 'AVAILABLE');

  useEffect(() => {
    const etaI = setInterval(() => dispatch({ type: 'TICK_ETA' }), 1000);
    const moveI = setInterval(() => dispatch({ type: 'MOVE_UNITS' }), 500);
    const vitI = setInterval(() => dispatch({ type: 'UPDATE_VITALS' }), 3000);
    const chartI = setInterval(() => {
      setVelData(prev => ({
        labels: [...prev.labels.slice(1), 'NOW'],
        p1: [...prev.p1.slice(1), Math.floor(Math.random() * 15) + 5],
        p2: [...prev.p2.slice(1), Math.floor(Math.random() * 20) + 15],
      }));
    }, 10000);
    return () => { clearInterval(etaI); clearInterval(moveI); clearInterval(vitI); clearInterval(chartI); };
  }, [dispatch]);

  const velocityData = {
    labels: velData.labels,
    datasets: [
      { label: 'P1 (Critical)', data: velData.p1, borderColor: '#FF544C', backgroundColor: 'rgba(255,84,76,0.1)', fill: true, tension: 0.4 },
      { label: 'P2 (Urgent)', data: velData.p2, borderColor: '#FFB74D', borderDash: [5, 5], tension: 0.4 },
    ],
  };

  const allocLabels = ['Z1-Bandra', 'Z2-Andheri', 'Z3-Kurla', 'Z4-Dadar', 'Z5-Thane'];
  const allocRaw = [12, 8, 24, 15, 38];

  const sortedIndices = sortByLoad
    ? [...allocLabels.keys()].sort((a, b) => allocRaw[b] - allocRaw[a])
    : [...allocLabels.keys()];

  const allocData = {
    labels: sortedIndices.map(i => allocLabels[i]),
    datasets: [{
      label: 'Active Deployments',
      data: sortedIndices.map(i => allocRaw[i]),
      backgroundColor: (ctx: any) => ctx.raw > 30 ? '#FF544C' : ctx.raw > 15 ? '#FFB74D' : '#7DDC7A',
      borderRadius: 4
    }],
  };

  const filteredIncidents = incidentFilter === 'ALL' ? incidents : incidents.filter(i => i.priority === incidentFilter);

  const handleWithdraw = useCallback((reason: string, reassignTo?: string) => {
    if (!withdrawUnit) return;
    dispatch({ type: 'RECALL_UNIT', unitId: withdrawUnit.id, reason, reassignTo });
    setWithdrawUnit(null);
  }, [withdrawUnit, dispatch]);

  const handleRecalculate = () => {
    setRecalculating(true);
    setTimeout(() => { setRecalculating(false); }, 1500);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)', paddingBottom: 'var(--space-8)' }}>
      {}
      <style>{`
        @keyframes ringExpand { 0% { transform: scale(1); opacity: 0.8; } 100% { transform: scale(2.5); opacity: 0; } }
        @keyframes ambPulse { 0%,100% { box-shadow: 0 0 4px currentColor; } 50% { box-shadow: 0 0 12px currentColor; } }
        @keyframes marchingAnts { to { stroke-dashoffset: -20; } }
        .leaflet-container { background: var(--bg-void) !important; border-radius: var(--radius-md); }
        .leaflet-control-zoom { display: none !important; }
        .leaflet-tile-pane { filter: brightness(0.7) saturate(0.8); }
      `}</style>

      {}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', borderBottom: '1px solid var(--ghost-border)', paddingBottom: 'var(--space-3)', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
            <Activity size={28} style={{ color: 'var(--primary)' }} />
            <h1 style={{ margin: 0, fontSize: '1.8rem', letterSpacing: 2 }}>COMMAND CENTER OVERWATCH</h1>
          </div>
          <div className="text-dim" style={{ fontSize: '0.9rem' }}>MAHARASHTRA GROUP TACTICAL SECTOR - LIVE TELEMETRY</div>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'center', flexWrap: 'wrap' }}>
          <SystemClock />
          <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-full)', padding: '4px 12px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', display: 'flex', gap: 8 }}>
            <span>DISPATCHED: <span style={{ color: '#FF544C', fontWeight: 700 }}>{dispatched}</span></span>
            <span style={{ color: 'var(--ghost-border)' }}>|</span>
            <span>AVAILABLE: <span style={{ color: '#7DDC7A', fontWeight: 700 }}>{available}</span></span>
          </div>
          <StatusChip variant="active" pulse>RL ENGINE: ACTIVE</StatusChip>
          <StatusChip variant="gamma">DEFCON 4</StatusChip>
        </div>
      </div>

      {}
      <AnimatePresence>
        {recalculating && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            style={{ position: 'fixed', inset: 0, zIndex: 8000, background: 'linear-gradient(90deg, transparent, rgba(162,201,255,0.08), transparent)', backgroundSize: '200% 100%', animation: 'shimmer 1.5s ease-in-out infinite', display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)', fontSize: '1.2rem' }}>RECALCULATING OPTIMAL ROUTES...</div>
          </motion.div>
        )}
      </AnimatePresence>

      {}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.2fr)', gap: 'var(--space-4)' }}>
        <GlassCard>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <MapIcon size={18} style={{ color: 'var(--primary)' }} />
              <h3 style={{ margin: 0 }}>LIVE DEPLOYMENT GRID</h3>
            </div>
          </div>
          <div style={{ width: '100%', height: 420, borderRadius: 'var(--radius-md)', position: 'relative', overflow: 'hidden', border: '1px solid rgba(162,201,255,0.1)' }}>
            <MapContainer center={[19.076, 72.8777]} zoom={12} style={{ width: '100%', height: '100%' }} zoomControl={false} attributionControl={false}>
              <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
              {flyTarget && <FlyToUnit lat={flyTarget.lat} lng={flyTarget.lng} />}

              {}
              {mapLayers.hospitals && HOSPITALS.map(h => (
                <Circle key={h.id} center={[h.lat, h.lng]} radius={3000} pathOptions={{ color: 'rgba(162,201,255,0.2)', fillColor: 'rgba(162,201,255,0.06)', fillOpacity: 0.6 }}>
                  <Popup><div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    <strong>{h.name}</strong><br />Beds: {h.beds} | ICU: {h.icuBeds}<br />Load: {h.load}%
                  </div></Popup>
                </Circle>
              ))}

              {}
              {mapLayers.incidents && incidents.map(inc => (
                <Marker key={inc.id} position={[inc.lat, inc.lng]} icon={makeIncIcon(inc.priority)}>
                  <Popup><div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', minWidth: 160 }}>
                    <strong style={{ color: prioColors[inc.priority] }}>{inc.id}</strong> — {inc.title}<br />
                    HR: {inc.vitals.hr} | BP: {inc.vitals.bp} | SpO2: {inc.vitals.spo2}%<br />
                    ETA: {Math.floor(inc.eta / 60)}m {inc.eta % 60}s
                  </div></Popup>
                </Marker>
              ))}

              {}
              {mapLayers.ambulances && enRouteUnits.map(u => (
                <Marker key={u.id} position={[u.lat, u.lng]} icon={makeAmbIcon(u.type)}
                  eventHandlers={{ click: () => dispatch({ type: 'SET_CALL_TRACKER', id: u.id }) }}>
                  <Popup><div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    <strong style={{ color: typeColors[u.type] }}>{u.id}</strong> — {u.driverName}<br />
                    Status: {u.status} | ETA: {Math.floor(u.eta / 60)}m
                  </div></Popup>
                </Marker>
              ))}

              {}
              {mapLayers.routes && enRouteUnits.filter(u => u.assignedIncidentId).map(u => {
                const inc = incidents.find(i => i.id === u.assignedIncidentId);
                if (!inc) return null;
                return <Polyline key={`route-${u.id}`} positions={[[u.lat, u.lng], [inc.lat, inc.lng]]} pathOptions={{ color: typeColors[u.type], weight: 2, dashArray: '8 4', opacity: 0.6 }} />;
              })}
            </MapContainer>

            {}
            <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 999, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {(['incidents', 'ambulances', 'hospitals', 'routes'] as const).map(layer => (
                <button key={layer} onClick={() => dispatch({ type: 'TOGGLE_MAP_LAYER', layer })}
                  style={{ background: mapLayers[layer] ? 'rgba(162,201,255,0.2)' : 'rgba(5,5,7,0.7)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', color: mapLayers[layer] ? 'var(--primary)' : 'var(--on-surface-dim)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', cursor: 'pointer', textTransform: 'uppercase' }}>
                  {layer}
                </button>
              ))}
            </div>
            <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 999, display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(5,5,7,0.7)', padding: '3px 8px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--ghost-border)' }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#7DDC7A', animation: 'pulse-glow 3s infinite' }} />
              <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: '#7DDC7A' }}>GPS SYNCED</span>
            </div>

            {}
            <AnimatePresence>
              {state.callTrackerUnitId && (() => {
                const u = units.find(x => x.id === state.callTrackerUnitId);
                return u ? <CallTracker unit={u} onClose={() => dispatch({ type: 'SET_CALL_TRACKER', id: null })} /> : null;
              })()}
            </AnimatePresence>
          </div>
        </GlassCard>

        {}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          <GlassCard>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
              <BarChart3 size={18} style={{ color: 'var(--priority-beta)' }} />
              <h3 style={{ margin: 0 }}>HOURLY INCIDENT VELOCITY</h3>
            </div>
            <div style={{ height: 160, width: '100%' }}>
              <Line data={velocityData} options={chartOptions} />
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-3)', marginTop: 'var(--space-3)' }}>
              {[{ label: 'TOTAL TODAY', val: '47', color: 'var(--primary)' }, { label: 'AVG RESPONSE', val: '06:42', color: 'var(--secondary)' }, { label: 'P1 RESOLUTION', val: '94%', color: '#FF544C' }].map(s => (
                <div key={s.label} style={{ flex: 1, background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-2)', textAlign: 'center' }}>
                  <div className="text-label">{s.label}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: s.color, fontSize: '1rem' }}>{s.val}</div>
                </div>
              ))}
            </div>
          </GlassCard>
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <Network size={18} style={{ color: 'var(--secondary)' }} />
                <h3 style={{ margin: 0 }}>REGIONAL UNIT ALLOCATION</h3>
              </div>
              <button onClick={() => setSortByLoad(!sortByLoad)} style={{ background: 'var(--bg-surface)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-sm)', padding: '2px 8px', color: 'var(--on-surface-variant)', fontSize: '0.65rem', fontFamily: 'var(--font-mono)', cursor: 'pointer' }}>SORT BY LOAD ↕</button>
            </div>
            <div style={{ height: 140, width: '100%' }}>
              <Bar data={allocData} options={{ ...chartOptions, indexAxis: 'y' as const }} />
            </div>
          </GlassCard>
        </div>
      </div>

      {}
      <GlassCard>
        <h3 style={{ margin: '0 0 var(--space-3)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}><Truck size={18} style={{ color: 'var(--primary)' }} /> AMBULANCE FLEET STATUS</h3>
        <div style={{ display: 'flex', gap: 'var(--space-3)', overflowX: 'auto', paddingBottom: 'var(--space-2)' }}>
          {activeUnits.map(u => {
            const statusColor = u.status === 'AVAILABLE' ? '#7DDC7A' : u.status === 'EN_ROUTE' ? '#A2C9FF' : u.status === 'ON_SCENE' ? '#FFB74D' : u.status === 'WITHDRAWN' ? '#FF544C' : u.status === 'SUSPENDED' ? '#8A919E' : '#404752';
            return (
              <motion.div key={u.id} layout style={{ minWidth: 140, background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-3)', borderTop: `2px solid ${statusColor}`, cursor: 'pointer', flexShrink: 0 }}
                onClick={() => { setFlyTarget({ lat: u.lat, lng: u.lng }); dispatch({ type: 'SET_CALL_TRACKER', id: u.id }); }}
                whileHover={{ scale: 1.02 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.8rem', marginBottom: 4 }}>{u.id}</div>
                <div style={{ fontSize: '0.6rem', color: statusColor, fontWeight: 600, marginBottom: 4, textTransform: 'uppercase' }}>{u.status.replace('_', ' ')}</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', marginBottom: 4 }}>{u.driverName}</div>
                {u.status === 'EN_ROUTE' && (
                  <>
                    <div className="progress-bar" style={{ marginBottom: 4 }}><div className="progress-bar-fill" style={{ width: `${Math.max(5, 100 - (u.eta / 6))}%`, background: typeColors[u.type] }} /></div>
                    <div style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: typeColors[u.type] }}>ETA {Math.floor(u.eta / 60)}:{String(u.eta % 60).padStart(2, '0')}</div>
                  </>
                )}
                {u.status === 'AVAILABLE' && <PulseButton variant="primary" size="sm" style={{ width: '100%', fontSize: '0.65rem', marginTop: 4 }}>DEPLOY</PulseButton>}
                {u.status === 'EN_ROUTE' && <PulseButton variant="danger" size="sm" onClick={e => { e.stopPropagation(); setWithdrawUnit(u); }} style={{ width: '100%', fontSize: '0.65rem', marginTop: 4 }}>WITHDRAW</PulseButton>}
                <div style={{ marginTop: 6 }}>
                  <div className="text-label" style={{ fontSize: '0.55rem' }}>FATIGUE</div>
                  <div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${u.fatigue}%`, background: u.fatigue > 80 ? '#FF544C' : u.fatigue > 40 ? '#FFB74D' : '#7DDC7A' }} /></div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </GlassCard>

      {}
      <GlassCard>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <Crosshair size={22} style={{ color: 'var(--primary)' }} />
            <h2 style={{ margin: 0 }}>ACTIVE INCIDENT MATRIX</h2>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
            {['ALL', 'ALPHA', 'BETA', 'GAMMA'].map(f => (
              <button key={f} onClick={() => setIncidentFilter(f)}
                style={{ background: incidentFilter === f ? 'rgba(162,201,255,0.15)' : 'transparent', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-full)', padding: '2px 10px', color: f === 'ALL' ? 'var(--on-surface)' : prioColors[f] || 'var(--on-surface)', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', cursor: 'pointer', fontWeight: incidentFilter === f ? 700 : 400 }}>
                {f}
              </button>
            ))}
            <div style={{ textAlign: 'right', marginLeft: 'var(--space-4)' }}>
              <div className="text-label">Active Dispatches</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--secondary)' }}>{dispatched} <span style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)' }}>/ {units.length} UNITS</span></div>
            </div>
            <PulseButton variant="primary" size="sm" onClick={handleRecalculate}>FORCE RECALCULATION</PulseButton>
          </div>
        </div>

        <table className="data-table" style={{ width: '100%' }}>
          <thead><tr>
            <th>ID / TIME</th><th>PRIORITY / TYPE</th><th>GEOLOCATION</th><th>LIVE VITALS (HR | BP | SpO2)</th><th>RESPONDING UNIT(S)</th><th>ETA</th><th>ACTIONS</th>
          </tr></thead>
          <tbody>
            {filteredIncidents.map((inc, i) => (
              <motion.tr key={inc.id} layout initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
                style={{ borderBottom: '1px solid var(--ghost-border)', cursor: 'pointer', background: expandedRow === inc.id ? 'rgba(162,201,255,0.04)' : 'transparent' }}
                onClick={() => setExpandedRow(expandedRow === inc.id ? null : inc.id)}>
                <td style={{ padding: 'var(--space-3)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{inc.id}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>T0 + {inc.time}</div>
                </td>
                <td style={{ padding: 'var(--space-3)' }}>
                  <StatusChip variant={inc.priority === 'ALPHA' ? 'alpha' : inc.priority === 'BETA' ? 'beta' : 'gamma'} pulse={inc.priority === 'ALPHA'}>{inc.priority} - {inc.type}</StatusChip>
                </td>
                <td style={{ padding: 'var(--space-3)' }}>
                  <div style={{ fontWeight: 600 }}>{inc.title}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', display: 'flex', alignItems: 'center', gap: 4 }}><Navigation size={10} /> {inc.location}</div>
                </td>
                <td style={{ padding: 'var(--space-3)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                  {inc.vitals.hr === 0 ? <span style={{ color: 'var(--tertiary)' }}>CARDIAC ASYSTOLE</span> : (
                    <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
                      <motion.span animate={{ color: inc.vitals.hr > 120 ? '#FF544C' : '#E0E0FF' }} style={{ fontWeight: 600 }}>{inc.vitals.hr}</motion.span>
                      <span style={{ color: 'var(--on-surface-dim)' }}>|</span>
                      <span>{inc.vitals.bp}</span>
                      <span style={{ color: 'var(--on-surface-dim)' }}>|</span>
                      <span style={{ color: inc.vitals.spo2 < 90 ? '#FF544C' : 'var(--primary)', animation: inc.vitals.spo2 < 90 ? 'pulse-glow-red 1s infinite' : 'none' }}>{inc.vitals.spo2}%</span>
                    </div>
                  )}
                </td>
                <td style={{ padding: 'var(--space-3)', fontWeight: 600, color: inc.responder === 'UNASSIGNED' ? '#FF544C' : 'var(--secondary)' }}>{inc.responder}</td>
                <td style={{ padding: 'var(--space-3)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: inc.eta < 180 ? '#FF544C' : 'var(--on-surface)' }}>
                  {inc.eta > 0 ? `${Math.floor(inc.eta / 60)}:${String(inc.eta % 60).padStart(2, '0')}` : '—'}
                </td>
                <td style={{ padding: 'var(--space-3)' }} onClick={e => e.stopPropagation()}>
                  <div style={{ display: 'flex', gap: 4 }}>
                    <button onClick={() => { setFlyTarget({ lat: inc.lat, lng: inc.lng }); if (inc.assignedUnitId) dispatch({ type: 'SET_CALL_TRACKER', id: inc.assignedUnitId }); }}
                      style={{ background: 'var(--bg-surface)', border: '1px solid var(--ghost-border)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', color: 'var(--primary)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', cursor: 'pointer' }}>TRACK</button>
                    {inc.assignedUnitId && <button onClick={() => { const u = units.find(x => x.id === inc.assignedUnitId); if (u) setWithdrawUnit(u); }}
                      style={{ background: 'rgba(255,84,76,0.1)', border: '1px solid rgba(255,84,76,0.3)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', color: '#FF544C', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', cursor: 'pointer' }}>WITHDRAW</button>}
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </GlassCard>

      {}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-4)' }}>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 6 }}><AlertTriangle size={16} /> ENVIRONMENTAL RISK</h3>
          </div>
          <div style={{ padding: 'var(--space-3)', background: 'var(--bg-base)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ color: 'var(--priority-beta)', fontWeight: 700, fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="20" height="20" viewBox="0 0 20 20"><path d="M10 2L8 8H4L7 12L5 18L10 14L15 18L13 12L16 8H12L10 2Z" fill="none" stroke="#FFB74D" strokeWidth="1.5"><animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite"/></path></svg>
              HEAVY PRECIPITATION
            </div>
            <div style={{ color: 'var(--on-surface-variant)', fontSize: '0.8rem', marginTop: 4 }}>+45% predicted MVC rate. Sub-optimal flight conditions.</div>
          </div>
        </GlassCard>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 6 }}><Radio size={16} /> SATELLITE TRAFFIC</h3>
          </div>
          <div style={{ padding: 'var(--space-3)', background: 'rgba(255,183,77,0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(255,183,77,0.2)' }}>
            <div style={{ color: 'var(--priority-beta)', fontWeight: 700, fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="20" height="20" viewBox="0 0 20 20">
                <path d="M10 14a2 2 0 100-4 2 2 0 000 4z" fill="#FFB74D"/>
                <path d="M6 10a4 4 0 018 0" fill="none" stroke="#FFB74D" strokeWidth="1.5" opacity="0.6"><animate attributeName="opacity" values="0.3;1;0.3" dur="1.5s" repeatCount="indefinite"/></path>
                <path d="M3 10a7 7 0 0114 0" fill="none" stroke="#FFB74D" strokeWidth="1.5" opacity="0.3"><animate attributeName="opacity" values="0.1;0.7;0.1" dur="1.5s" repeatCount="indefinite" begin="0.3s"/></path>
              </svg>
              ELEVATED (INDEX 8.2)
            </div>
            <div style={{ color: 'var(--on-surface-variant)', fontSize: '0.8rem', marginTop: 4 }}>Chokepoint at NH-48. Rerouting algorithms active.</div>
          </div>
        </GlassCard>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 6 }}><Users size={16} /> SPECIAL EVENTS</h3>
          </div>
          <div style={{ padding: 'var(--space-3)', background: 'rgba(125,220,122,0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(125,220,122,0.2)' }}>
            <div style={{ color: 'var(--secondary)', fontWeight: 700, fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="20" height="20" viewBox="0 0 20 20">
                <circle cx="10" cy="10" r="8" fill="none" stroke="#7DDC7A" strokeWidth="1.5"/>
                <path d="M6 10l3 3 5-6" fill="none" stroke="#7DDC7A" strokeWidth="2"><animate attributeName="stroke-dashoffset" from="20" to="0" dur="1s" fill="freeze"/></path>
              </svg>
              ALL CLEAR
            </div>
            <div style={{ color: 'var(--on-surface-variant)', fontSize: '0.8rem', marginTop: 4 }}>No mass gatherings affecting central tactical zones.</div>
          </div>
        </GlassCard>
      </div>

      {}
      {loginOpen && <AdminLoginModal onSuccess={() => dispatch({ type: 'LOGIN' })} onClose={() => setLoginOpen(false)} />}
      {withdrawUnit && <WithdrawModal unit={withdrawUnit} incidents={incidents} availableUnits={availableForReassign} onConfirm={handleWithdraw} onClose={() => setWithdrawUnit(null)} />}
    </div>
  );
}
