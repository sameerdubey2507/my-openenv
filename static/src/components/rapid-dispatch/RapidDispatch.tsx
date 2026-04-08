import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, User, MapPin, Truck, Clock, Shield, UserCheck, AlertTriangle, ChevronRight, X, HeartPulse, Activity } from 'lucide-react';
import GlassCard from '../shared/GlassCard';
import StatusChip from '../shared/StatusChip';
import PulseButton from '../shared/PulseButton';

interface Props { sim: any; }

const advancedVictims = [
  { id: '#0921-A', status: 'Tension Pneumothorax, SpO2 82%, GCS 9', tag: 'IMMEDIATE', triage: 'START_01', sector: 'SEC-4 HIGH-RISE', color: '#FF544C', hr: 130, bp: '80/50', etco2: 24, eta: '4 min' },
  { id: '#0921-B', status: 'Open Femur Fx, Controlled Bleed, GCS 15', tag: 'DELAYED', triage: 'START_02', sector: 'SEC-4 LOADING DOCK', color: '#FFB74D', hr: 110, bp: '100/70', etco2: 35, eta: '12 min' },
  { id: '#0921-C', status: 'Lacerations (Arms), Walking Wounded', tag: 'MINIMAL', triage: 'START_03', sector: 'SEC-4 TRIAGE ALPHA', color: '#7DDC7A', hr: 85, bp: '120/80', etco2: 40, eta: 'N/A' },
  { id: '#0922-D', status: 'Flail Chest, Unresponsive, GCS 6', tag: 'IMMEDIATE', triage: 'START_04', sector: 'SEC-1 TUNNEL', color: '#FF544C', hr: 150, bp: '70/40', etco2: 18, eta: '7 min' },
  { id: '#0922-E', status: 'Blunt Force Trauma (Lower Ext), GCS 13', tag: 'DELAYED', triage: 'START_05', sector: 'SEC-1 TUNNEL ENT', color: '#FFB74D', hr: 105, bp: '105/65', etco2: 32, eta: '15 min' },
  { id: '#0923-F', status: 'Smoke Inhalation, Mild Distress', tag: 'MINIMAL', triage: 'START_06', sector: 'SEC-2 EAST WING', color: '#7DDC7A', hr: 95, bp: '130/85', etco2: 38, eta: 'N/A' },
];

const unitTypes = [
  { type: 'BLS', label: 'BASIC LIFE SUPPORT', avail: 14, icon: <Truck size={20} />, color: 'var(--on-surface-variant)' },
  { type: 'ALS', label: 'ADV. LIFE SUPPORT', avail: 5, icon: <Shield size={20} />, color: 'var(--primary)' },
  { type: 'MICU', label: 'MOBILE INTENSIVE', avail: 2, icon: <HeartPulse size={20} />, color: 'var(--tertiary)' },
  { type: 'HELO', label: 'AIR AMBULANCE', avail: 1, icon: <Activity size={20} />, color: 'var(--secondary)' },
];

const crews = [
  { name: 'CREW ALPHA-01', level: 'CRITICAL', duty: '09H 12M', progress: 92, color: 'var(--tertiary)' },
  { name: 'CREW BRAVO-04', level: 'ELEVATED', duty: '07H 45M', progress: 75, color: 'var(--priority-beta)' },
  { name: 'CREW GAMMA-12', level: 'OPTIMAL', duty: '04H 30M', progress: 45, color: 'var(--secondary)' },
];

function InterRegionalMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = c.offsetWidth * 2; c.height = c.offsetHeight * 2; ctx.scale(2, 2);
    const w = c.offsetWidth, h = c.offsetHeight;

    let frame: number, t = 0;
    const draw = () => {
      t += 0.02;
      ctx.clearRect(0, 0, w, h);

      const regions = [{x: w*0.5, y: h*0.5, r: 40, c: '#A2C9FF', n: 'CORE'}, {x: w*0.15, y: h*0.2, r: 25, c: '#FFB74D', n: 'NORTH'}, {x: w*0.85, y: h*0.8, r: 30, c: '#7DDC7A', n: 'METRO_SOUTH'}];

      regions.forEach((rg, i) => {
        ctx.beginPath();
        ctx.arc(rg.x, rg.y, rg.r + Math.sin(t*2+i)*2, 0, Math.PI*2);
        ctx.strokeStyle = rg.c + '33'; ctx.lineWidth = 2; ctx.stroke();
        ctx.beginPath(); ctx.arc(rg.x, rg.y, 4, 0, Math.PI*2); ctx.fillStyle = rg.c; ctx.fill();
        ctx.fillStyle = '#C0C7D4'; ctx.font = '8px Inter'; ctx.fillText(rg.n, rg.x - 15, rg.y + 15);
      });

      ctx.strokeStyle = 'rgba(162, 201, 255, 0.1)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(regions[1].x, regions[1].y); ctx.lineTo(regions[0].x, regions[0].y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(regions[2].x, regions[2].y); ctx.lineTo(regions[0].x, regions[0].y); ctx.stroke();

      const progress = (t * 0.3) % 1;
      const lx = regions[2].x + (regions[0].x - regions[2].x) * progress;
      const ly = regions[2].y + (regions[0].y - regions[2].y) * progress;
      ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI*2); ctx.fillStyle = '#7DDC7A'; ctx.fill();

      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, []);
  return <canvas ref={canvasRef} style={{ width: '100%', height: '100%', position: 'absolute', inset: 0 }} />;
}

export default function RapidDispatch({ sim }: Props) {
  const [selectedUnit, setSelectedUnit] = useState<string | null>(null);
  const [dispatchingId, setDispatchingId] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState<any>(null);

  const handleDispatch = async (victimId: string, unitType: string) => {
    setDispatchingId(victimId);
    await sim.stepAction({ action_type: 'dispatch', incident_id: victimId, unit_type: unitType, hospital_id: 'H01', assigned_priority: 'P1' });
    setTimeout(() => setDispatchingId(null), 1000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', paddingBottom: 'var(--space-6)' }}>
      {}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
          <Zap size={22} style={{ color: 'var(--primary)' }} />
          <h2 style={{ margin: 0 }}>NATIONAL DISPATCH COMMAND</h2>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          <StatusChip variant="alpha" pulse>12 IMMEDIATE</StatusChip>
          <StatusChip variant="beta">8 DELAYED</StatusChip>
          <StatusChip variant="gamma">24 STABLE</StatusChip>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: 'var(--space-5)' }}>
        {}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
               <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <AlertTriangle size={18} style={{ color: 'var(--tertiary)' }} />
                <h3 style={{ margin: 0 }}>TRIAGE QUEUE (MCI PROTOCOL ACTIVE)</h3>
              </div>
              <span className="text-label">SORT: ACUITY ▼</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', maxHeight: '700px', overflowY: 'auto', paddingRight: 'var(--space-2)' }}>
              {advancedVictims.map((v, i) => (
                <motion.div
                  key={v.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                  style={{
                    background: 'var(--bg-surface)', borderRadius: 'var(--radius-lg)', padding: 'var(--space-4)',
                    borderLeft: `3px solid ${v.color}`, display: 'flex', justifyContent: 'space-between'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
                      <StatusChip variant={v.tag === 'IMMEDIATE' ? 'alpha' : v.tag === 'DELAYED' ? 'beta' : 'gamma'}>{v.tag}</StatusChip>
                      <span style={{ fontWeight: 800, fontSize: '0.95rem', letterSpacing: 1 }}>VICTIM {v.id}</span>
                      <span style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)' }}>{v.triage}</span>
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', marginBottom: 'var(--space-1)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      <User size={14} style={{ marginRight: 6, verticalAlign: 'text-bottom' }} />{v.status}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <MapPin size={12} /> {v.sector}
                    </div>

                    {}
                    <div style={{ display: 'flex', gap: 'var(--space-4)', marginTop: 'var(--space-3)', paddingTop: 'var(--space-2)', borderTop: '1px solid var(--ghost-border)' }}>
                      <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)' }}>HR: <span style={{ color: v.hr > 120 ? 'var(--tertiary)' : 'var(--on-surface)'}}>{v.hr}</span></div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)' }}>BP: <span style={{ color: 'var(--on-surface)'}}>{v.bp}</span></div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)' }}>ETCO2: <span style={{ color: 'var(--on-surface)'}}>{v.etco2}</span></div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', alignItems: 'flex-end', marginLeft: 'var(--space-4)' }}>
                    <PulseButton variant="ghost" size="sm" onClick={() => setProfileOpen(v)} style={{ width: '120px', justifyContent: 'space-between' }}>
                      PROFILE <ChevronRight size={14} />
                    </PulseButton>
                    <PulseButton variant="primary" size="sm" loading={dispatchingId === v.id} onClick={() => handleDispatch(v.id, selectedUnit || 'ALS')} style={{ width: '120px' }}>
                      DISPATCH
                    </PulseButton>
                  </div>
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </div>

        {}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          {}
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <Truck size={18} style={{ color: 'var(--primary)' }} /> GLOBAL ASSET TERMINAL
              </h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-3)' }}>
              {unitTypes.map((u) => (
                <div
                  key={u.type}
                  onClick={() => setSelectedUnit(u.type)}
                  style={{
                    background: selectedUnit === u.type ? 'var(--bg-surface-high)' : 'var(--bg-surface)',
                    borderRadius: 'var(--radius-md)', padding: 'var(--space-3)', cursor: 'pointer',
                    border: selectedUnit === u.type ? `1px solid ${u.color}` : '1px solid var(--ghost-border)',
                    transition: 'all 0.2s', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ color: u.color }}>{u.icon}</div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{u.avail} READY</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600 }}>{u.label}</div>
                </div>
              ))}
            </div>
          </GlassCard>

          {}
          <GlassCard>
             <h3 style={{ margin: '0 0 var(--space-3) 0' }}>MUTUAL AID TOPOGRAPHY</h3>
             <div style={{ background: 'var(--bg-void)', borderRadius: 'var(--radius-lg)', height: 220, marginBottom: 'var(--space-4)', position: 'relative', overflow: 'hidden', border: '1px solid var(--ghost-border)' }}>
               <InterRegionalMap />
               <div style={{ position: 'absolute', top: 8, left: 8, fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', textShadow: '0 0 4px var(--secondary)' }}>LIVE SAT-LINK</div>
             </div>

             {[
              { id: 'METRO_SOUTH R-12', eta: '09:42 ETA', req: '2X ALS UNITS', color: 'var(--priority-beta)' },
              { id: 'NORTH_PORT M-04', eta: '02:15 ETA', req: '1X MICU SUPPORT', color: 'var(--secondary)' },
              { id: 'EAST_DISTRICT E-99', eta: '18:00 ETA', req: 'MASS EVAC BUS', color: 'var(--primary)' },
            ].map((aid, i) => (
              <div key={i} style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-3)', marginBottom: 'var(--space-2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>{aid.id}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)', marginTop: 2 }}>{aid.req}</div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                  <StatusChip variant={i === 0 ? 'beta' : 'gamma'}>{aid.eta}</StatusChip>
                  <span style={{ fontSize: '0.65rem', color: 'var(--primary)', cursor: 'pointer' }}>OVERRIDE</span>
                </div>
              </div>
            ))}
          </GlassCard>

          {}
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <UserCheck size={18} style={{ color: 'var(--tertiary)' }} /> FLEET EXHAUSTION
              </h3>
            </div>
            {crews.map(c => (
              <div key={c.name} style={{ marginBottom: 'var(--space-3)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 6 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{c.name}</span>
                  <span style={{ color: c.color }}>{c.duty} / 10H</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill" style={{ width: `${c.progress}%`, background: c.color }} />
                </div>
              </div>
            ))}
          </GlassCard>
        </div>
      </div>

      {}
      <AnimatePresence>
        {profileOpen && (
          <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 'var(--space-4)' }}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setProfileOpen(null)}
              style={{ position: 'absolute', inset: 0, background: 'rgba(5, 5, 7, 0.8)', backdropFilter: 'blur(8px)' }} />

            <motion.div initial={{ opacity: 0, y: 30, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 20, scale: 0.95 }}
              style={{ position: 'relative', width: '100%', maxWidth: '800px', background: 'var(--bg-surface-high)', borderRadius: 'var(--radius-xl)', border: '1px solid var(--ghost-border)', boxShadow: 'var(--glow-card)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>

              <div style={{ padding: 'var(--space-4) var(--space-6)', borderBottom: '1px solid var(--ghost-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-surface)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                   <StatusChip variant={profileOpen.tag === 'IMMEDIATE' ? 'alpha' : 'beta'}>{profileOpen.tag}</StatusChip>
                   <h2 style={{ margin: 0, fontSize: '1.2rem', fontFamily: 'var(--font-mono)' }}>PROFILE: {profileOpen.id}</h2>
                </div>
                <button onClick={() => setProfileOpen(null)} style={{ background: 'transparent', border: 'none', color: 'var(--on-surface-variant)', cursor: 'pointer' }}><X size={24} /></button>
              </div>

              <div style={{ padding: 'var(--space-6)', display: 'flex', gap: 'var(--space-6)', flexDirection: 'column' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-4)' }}>
                  <div style={{ background: 'var(--bg-base)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
                    <div className="text-label" style={{ marginBottom: 4 }}>HEART RATE</div>
                    <div className="text-metric" style={{ color: profileOpen.hr > 120 ? 'var(--tertiary)' : 'var(--on-surface)' }}>{profileOpen.hr} <span style={{ fontSize: '0.8rem' }}>BPM</span></div>
                  </div>
                  <div style={{ background: 'var(--bg-base)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
                    <div className="text-label" style={{ marginBottom: 4 }}>BLOOD PRESSURE</div>
                    <div className="text-metric">{profileOpen.bp}</div>
                  </div>
                  <div style={{ background: 'var(--bg-base)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
                    <div className="text-label" style={{ marginBottom: 4 }}>ETCO2</div>
                    <div className="text-metric" style={{ color: 'var(--secondary)' }}>{profileOpen.etco2} <span style={{ fontSize: '0.8rem' }}>mmHg</span></div>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', borderLeft: `4px solid ${profileOpen.color}` }}>
                  <h4 style={{ margin: '0 0 var(--space-2) 0', color: 'var(--on-surface-variant)' }}>CLINICAL NARRATIVE</h4>
                  <p style={{ margin: 0, fontSize: '0.85rem', lineHeight: 1.6 }}>{profileOpen.status}. Detected at {profileOpen.sector}. Initial assessment indicates requiring immediate surgical intervention. Recommended transfer to Level 1 Trauma Center. Predicted degradation slope is steep. ETA to catastrophic failure if untreated: {profileOpen.eta}.</p>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 'var(--space-3)', marginTop: 'var(--space-4)' }}>
                  <PulseButton variant="ghost" onClick={() => setProfileOpen(null)}>CLOSE</PulseButton>
                  <PulseButton variant="primary" icon={<Truck size={16} />} onClick={() => { handleDispatch(profileOpen.id, selectedUnit || 'ALS'); setProfileOpen(null); }}>
                    OVERRIDE & DISPATCH NOW
                  </PulseButton>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
