import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Database, CheckCircle, Activity, Filter, Download, ChevronLeft, ChevronRight } from 'lucide-react';
import GlassCard from '../shared/GlassCard';
import StatusChip from '../shared/StatusChip';
import PulseButton from '../shared/PulseButton';
import AnimatedCounter from '../shared/AnimatedCounter';
interface Props { sim: any; }
const vitalUnits = [
  { id: 'ECHO-12', label: 'Pulse', value: 74, unit: 'BPM', status: 'STABLE', color: 'var(--secondary)', waveType: 'ecg' },
  { id: 'DELTA-09', label: 'Resp', value: 28, unit: '/min', status: 'ELEVATED', color: 'var(--tertiary)', waveType: 'resp' },
  { id: 'KILO-04', label: 'SpO2', value: 99, unit: '%', status: 'OPTIMAL', color: 'var(--primary)', waveType: 'spo2' },
];
const incidents = Array.from({ length: 45 }, (_, i) => ({
  id: `#EM-2024-${8119 + i}`,
  protocol: ['CARDIAC ARREST', 'PENETRATING TRAUMA', 'MASS CASUALTY (START)', 'RESPIRATORY DISTRESS', 'STROKE (CVA)', 'SEVERE BURN (3RD DEG)'][i % 6],
  unit: ['M1', 'A2', 'Z0', 'M3', 'H1', 'A5'][i % 6],
  unitName: ['Med-Response Delta', 'Tactical Unit Echo', 'Civil Logistics', 'Med-Response Alpha', 'Aero-Med Evac', 'Ground Transport Beta'][i % 6],
  time: `12:${(i * 3).toString().padStart(2, '0')}:${(i * 7 % 60).toString().padStart(2, '0')} UTC`,
  compliance: Math.random() > 0.2,
  outcome: Math.random() > 0.15 ? (Math.random() > 0.5 ? 'SURVIVAL (ROSC)' : 'STABILIZED') : 'MORTALITY',
  outcomeColor: '',
})).map(inc => ({
  ...inc,
  outcomeColor: inc.outcome.includes('SURV') ? 'var(--secondary)' : inc.outcome === 'STABILIZED' ? 'var(--primary)' : 'var(--tertiary)'
}));
function VitalWaveform({ type, color }: { type: string; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const w = canvas.offsetWidth, h = canvas.offsetHeight;
    let frame: number, t = 0;
    const draw = () => {
      t += 0.03;
      ctx.clearRect(0, 0, w, h);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      for (let x = 0; x <= w; x++) {
        let y = h / 2;
        const p = (x / w + t) % 1;
        if (type === 'ecg') {
          if (p > 0.3 && p < 0.35) y = h * 0.2;
          else if (p > 0.35 && p < 0.38) y = h * 0.8;
          else if (p > 0.38 && p < 0.42) y = h * 0.1;
          else if (p > 0.42 && p < 0.45) y = h * 0.6;
          else y = h / 2 + Math.sin(x * 0.02 + t) * 2;
        } else if (type === 'resp') {
          y = h / 2 + Math.sin(p * Math.PI * 4 + t * 2) * h * 0.3;
        } else {
          y = h / 2 + Math.sin(p * Math.PI * 6 + t) * h * 0.15 + Math.random() * 2;
        }
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.strokeStyle = color + '30';
      ctx.lineWidth = 4;
      ctx.stroke();
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [type, color]);
  return <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />;
}
export default function BioArchives({ sim }: Props) {
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', paddingBottom: 'var(--space-6)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
        <Database size={24} style={{ color: 'var(--primary)' }} />
        <h2 style={{ margin: 0 }}>NATIONAL BIO-ARCHIVES DATABASE</h2>
      </div>
      { }
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 'var(--space-4)' }}>
        <GlassCard hover3d>
          <span className="text-label">PERFORMANCE INDEX</span>
          <div style={{ fontSize: '0.7rem', color: 'var(--secondary)', marginBottom: 'var(--space-2)' }}>+2.4% VS. PREV CYCLE</div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 'var(--space-3)' }}>
            <motion.div
              animate={{ rotateY: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity }}
              style={{
                fontSize: '5rem', fontWeight: 900, color: 'var(--secondary)',
                textShadow: '0 0 30px rgba(125, 220, 122, 0.3)',
                fontFamily: 'var(--font-mono)',
              }}
            >
              A+
            </motion.div>
            <CheckCircle size={28} style={{ color: 'var(--secondary)' }} />
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', marginTop: 'var(--space-1)' }}>OPTIMIZED</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 'var(--space-2)', fontSize: '0.65rem' }}>
            <span>SURVIVAL BENCHMARK</span>
            <span style={{ color: 'var(--secondary)' }}>98.2%</span>
          </div>
          <div className="progress-bar progress-green" style={{ marginTop: 'var(--space-1)' }}>
            <div className="progress-bar-fill" style={{ width: '98.2%' }} />
          </div>
          <div style={{ marginTop: 'var(--space-4)', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--ghost-border)' }}>
            <h4 style={{ margin: '0 0 var(--space-2) 0', fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>DEMOGRAPHIC DISTRIBUTION</h4>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--on-surface)', marginBottom: 4 }}>
              <span>GERIATRIC (65+)</span> <span>42%</span>
            </div>
            <div className="progress-bar" style={{ marginBottom: 'var(--space-2)', height: 4 }}>
              <div className="progress-bar-fill" style={{ width: '42%', background: 'var(--primary)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--on-surface)', marginBottom: 4 }}>
              <span>ADULT (18-64)</span> <span>51%</span>
            </div>
            <div className="progress-bar" style={{ marginBottom: 'var(--space-2)', height: 4 }}>
              <div className="progress-bar-fill" style={{ width: '51%', background: 'var(--secondary)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--on-surface)', marginBottom: 4 }}>
              <span>PEDIATRIC (0-17)</span> <span>7%</span>
            </div>
            <div className="progress-bar" style={{ height: 4 }}>
              <div className="progress-bar-fill" style={{ width: '7%', background: 'var(--tertiary)' }} />
            </div>
          </div>
        </GlassCard>
        <GlassCard>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
            <div>
              <h3 style={{ margin: 0 }}>BIO-METRIC SYNC: LIVE TRANSIT</h3>
              <span style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)' }}>Real-time physiological telemetry from tactical units</span>
            </div>
            <PulseButton variant="ghost" size="sm" onClick={() => sim.loadState()}>⋮</PulseButton>
          </div>
          <div className="grid-3" style={{ gap: 'var(--space-3)' }}>
            {vitalUnits.map((v, i) => (
              <motion.div key={v.id} initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.15 }}
                style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-lg)', padding: 'var(--space-3)', overflow: 'hidden' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                  <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--on-surface-variant)' }}>UNIT {v.id}</span>
                  <StatusChip variant={v.status === 'STABLE' ? 'gamma' : v.status === 'ELEVATED' ? 'alpha' : 'delta'}>{v.status}</StatusChip>
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-1)' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>{v.label}:</span>
                  <AnimatedCounter value={v.value} decimals={0} suffix="" />
                  <span style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)' }}>{v.unit}</span>
                </div>
                <div style={{ height: 50, marginTop: 'var(--space-2)' }}>
                  <VitalWaveform type={v.waveType} color={v.color.includes('var') ? (v.waveType === 'ecg' ? '#7DDC7A' : v.waveType === 'resp' ? '#FF544C' : '#A2C9FF') : v.color} />
                </div>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      </div>
      { }
      <GlassCard>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
          <div>
            <h3 style={{ margin: 0 }}>Historical Incident Registry</h3>
            <span style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)' }}>ARCHIVE BATCH 54,212 • LIFECYCLE 2024.Q4</span>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <PulseButton variant="secondary" size="sm" icon={<Filter size={14} />} onClick={() => { }}>FILTER LOGS</PulseButton>
            <PulseButton variant="secondary" size="sm" icon={<Download size={14} />} onClick={() => sim.loadLedger()}>EXPORT DATASET</PulseButton>
          </div>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>INCIDENT ID</th><th>PROTOCOL CATEGORY</th><th>ASSIGNED FLEET UNIT</th><th>UTC TIMESTAMP</th><th>COMPLIANCE</th><th>POST-ACTION OUTCOME</th>
            </tr>
          </thead>
          <tbody>
            {incidents.slice(0, 15).map((inc, i) => (
              <motion.tr key={inc.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: Math.min(i * 0.03, 0.5) }}>
                <td><span className="text-mono" style={{ fontSize: '0.8rem', fontWeight: 600 }}>{inc.id}</span></td>
                <td><StatusChip variant={inc.protocol.includes('CARDIAC') ? 'alpha' : inc.protocol.includes('TRAUMA') ? 'beta' : inc.protocol.includes('MASS') ? 'alpha' : 'alpha'}>{inc.protocol}</StatusChip></td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                    <span style={{ background: 'var(--bg-surface-high)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>{inc.unit}</span>
                    {inc.unitName}
                  </div>
                </td>
                <td className="text-mono" style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)' }}>{inc.time}</td>
                <td>{inc.compliance ? <CheckCircle size={16} style={{ color: 'var(--secondary)' }} /> : <Activity size={16} style={{ color: 'var(--tertiary)' }} />}</td>
                <td style={{ color: inc.outcomeColor, fontWeight: 600, fontSize: '0.8rem' }}>{inc.outcome}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
        { }
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'var(--space-4)' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>VISUALIZING 25 / 54,212 ARCHIVAL ENTRIES</span>
          <div style={{ display: 'flex', gap: 'var(--space-1)', alignItems: 'center' }}>
            <PulseButton variant="ghost" size="sm" onClick={() => setPage(Math.max(1, page - 1))}><ChevronLeft size={14} /></PulseButton>
            {[1, 2, 3].map(p => (
              <PulseButton key={p} variant={page === p ? 'primary' : 'ghost'} size="sm" onClick={() => setPage(p)}>{p}</PulseButton>
            ))}
            <span style={{ color: 'var(--on-surface-variant)', fontSize: '0.75rem' }}>... 2168</span>
            <PulseButton variant="ghost" size="sm" onClick={() => setPage(page + 1)}><ChevronRight size={14} /></PulseButton>
          </div>
        </div>
      </GlassCard>
      { }
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)', marginTop: 'var(--space-4)' }}>
        <GlassCard>
          <h3 style={{ margin: '0 0 var(--space-4) 0', fontSize: '1.1rem' }}>GEOSPATIAL ANOMALY CLUSTERING</h3>
          <div style={{ height: 300, background: 'var(--bg-void)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem' }}>Automated detection of localized incident spikes deviating from historical baselines.</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 'var(--space-3)' }}>
              <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} style={{ background: 'rgba(255,84,76,0.1)', borderLeft: '3px solid var(--tertiary)', padding: 'var(--space-3)', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <strong style={{ color: 'var(--tertiary)' }}>SECTOR 4 - RESPIRATORY SURGE</strong>
                  <motion.div animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1, repeat: Infinity }}>
                    <StatusChip variant="alpha">ACTIVE</StatusChip>
                  </motion.div>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>+340% increase in acute respiratory distress over 12 hours. Probable industrial particulate release.</div>
              </motion.div>
              <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} style={{ background: 'rgba(255,183,77,0.1)', borderLeft: '3px solid var(--priority-beta)', padding: 'var(--space-3)', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <strong style={{ color: 'var(--priority-beta)' }}>HIGHWAY 48 - MVC POCKET</strong>
                  <StatusChip variant="beta">MONITORING</StatusChip>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>Repeated multi-vehicle collisions at interchange 12. Alerted local traffic control.</div>
              </motion.div>
            </div>
          </div>
        </GlassCard>
        <GlassCard>
          <h3 style={{ margin: '0 0 var(--space-4) 0', fontSize: '1.1rem' }}>PREDICTIVE RESOURCE EXHAUSTION</h3>
          <div style={{ height: 300, display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
            <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem', marginBottom: 0 }}>Monte Carlo simulations (N=10,000) projecting likely supply chain failures over next 72h.</p>
            <div style={{ position: 'relative' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                <span>UNIVERSAL O-NEG BLOOD UNITS</span>
                <span style={{ color: 'var(--tertiary)', fontWeight: 700 }}>98% PROBABILITY OF EXHAUSTION</span>
              </div>
              <div className="progress-bar">
                <motion.div initial={{ width: 0 }} animate={{ width: '98%' }} transition={{ duration: 1.5, delay: 0.2, type: 'spring' }} className="progress-bar-fill" style={{ background: 'var(--tertiary)' }} />
              </div>
            </div>
            <div style={{ position: 'relative' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                <span>LEVEL 1 NEURO-SURGICAL BEDS</span>
                <span style={{ color: 'var(--priority-beta)', fontWeight: 700 }}>64% PROBABILITY OF EXHAUSTION</span>
              </div>
              <div className="progress-bar">
                <motion.div initial={{ width: 0 }} animate={{ width: '64%' }} transition={{ duration: 1.5, delay: 0.4, type: 'spring' }} className="progress-bar-fill" style={{ background: 'var(--priority-beta)' }} />
              </div>
            </div>
            <div style={{ position: 'relative' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                <span>ALS FIELD RESPONDERS</span>
                <span style={{ color: 'var(--secondary)', fontWeight: 700 }}>12% PROBABILITY OF EXHAUSTION</span>
              </div>
              <div className="progress-bar">
                <motion.div initial={{ width: 0 }} animate={{ width: '12%' }} transition={{ duration: 1.5, delay: 0.6, type: 'spring' }} className="progress-bar-fill" style={{ background: 'var(--secondary)' }} />
              </div>
            </div>
          </div>
        </GlassCard>
      </div>
      { }
      <GlassCard>
        <h3 style={{ margin: '0 0 var(--space-4) 0', fontSize: '1.1rem' }}>72H INCIDENT DISTRIBUTIONS</h3>
        <div style={{ height: 250, width: '100%', position: 'relative' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', height: '100%', gap: '4px', paddingTop: 'var(--space-4)' }}>
            {Array.from({ length: 48 }).map((_, i) => {
              const val1 = Math.random() * 40 + 10;
              const val2 = Math.random() * 20 + 5;
              return (
                <motion.div key={i} initial={{ height: 0 }} animate={{ height: '100%' }} transition={{ delay: i * 0.02 }} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', gap: 2 }}>
                  <div style={{ height: `${val2}%`, background: 'var(--priority-beta)', borderRadius: 2, opacity: 0.8 }} />
                  <div style={{ height: `${val1}%`, background: 'var(--tertiary)', borderRadius: 2, opacity: 0.8 }} />
                </motion.div>
              );
            })}
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--on-surface-variant)', fontSize: '0.7rem', marginTop: 'var(--space-2)' }}>
          <span>T-72h</span><span>T-36h</span><span>NOW</span>
        </div>
      </GlassCard>
    </div>
  );
}
