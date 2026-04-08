import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Map, Radio, AlertTriangle, TrendingUp, ArrowRight, RotateCw } from 'lucide-react';
import GlassCard from '../shared/GlassCard';
import StatusChip from '../shared/StatusChip';
import PulseButton from '../shared/PulseButton';
import AnimatedCounter from '../shared/AnimatedCounter';

interface Props { sim: any; }

const hospitals = [
  { name: 'ST. JUDE', capacity: 64, status: 'NOMINAL', color: 'var(--secondary)' },
  { name: 'METRO GEN', capacity: 92, status: 'HEAVY', color: 'var(--priority-beta)' },
  { name: 'NORTH REACH', capacity: 41, status: 'FLOW', color: 'var(--secondary)' },
  { name: 'CITY TRAUMA', capacity: 98, status: 'CRITICAL', color: 'var(--tertiary)' },
];

const signals = [
  { title: 'SIGNAL PHASE ALPHA', text: 'Broadway/5th green-wave initiated. Projected recovery: 4.2m.', status: 'EXECUTED', time: '12:05 AGO', variant: 'alpha' as const },
  { title: 'GRIDLOCK WARNING', text: 'Unexpected density in Sector 09. Suggested buffer: +15%.', status: '', time: '', variant: 'beta' as const },
  { title: 'BASELINE FORECAST', text: 'Peak flow tapering in 38m. Reverting to A1.', status: '', time: '', variant: 'gamma' as const },
];

export default function GridRouting({ sim }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [saturation, setSaturation] = useState(84.2);

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
      t += 0.01;
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = 'rgba(162, 201, 255, 0.04)';
      ctx.lineWidth = 0.5;
      for (let x = 0; x < w; x += 30) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y < h; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

      const continents = [];
      for (let i = 0; i < 200; i++) {
        continents.push({
          x: (Math.sin(i * 0.7) * 0.3 + 0.5) * w + Math.cos(i * 0.13) * w * 0.2,
          y: (Math.cos(i * 0.4) * 0.25 + 0.5) * h + Math.sin(i * 0.17) * h * 0.15,
        });
      }
      continents.forEach(p => {
        if (p.x < 0 || p.x > w || p.y < 0 || p.y > h) return;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(162, 201, 255, 0.15)';
        ctx.fill();
      });

      const routes = [
        { x1: w * 0.2, y1: h * 0.4, x2: w * 0.7, y2: h * 0.3 },
        { x1: w * 0.3, y1: h * 0.6, x2: w * 0.8, y2: h * 0.5 },
      ];
      routes.forEach(r => {
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = 'rgba(51, 148, 241, 0.4)';
        ctx.lineWidth = 2;
        ctx.moveTo(r.x1, r.y1);
        ctx.quadraticCurveTo((r.x1 + r.x2) / 2, r.y1 - 40, r.x2, r.y2);
        ctx.stroke();
        ctx.setLineDash([]);

        const prog = (Math.sin(t * 0.5) + 1) / 2;
        const cx = r.x1 + (r.x2 - r.x1) * prog;
        const cy = r.y1 + (r.y2 - r.y1) * prog - Math.sin(prog * Math.PI) * 40;
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#3394F1';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(51, 148, 241, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      ctx.beginPath();
      ctx.arc(w * 0.65, h * 0.25, 8 + Math.sin(t * 2) * 2, 0, Math.PI * 2);
      ctx.fillStyle = 'var(--tertiary)';
      ctx.fillStyle = '#FF544C';
      ctx.fill();

      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      {}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <Map size={22} style={{ color: 'var(--primary)' }} />
            <h2 style={{ margin: 0 }}>GRID ROUTING MATRIX</h2>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>
            Sector 7G: Morning Peak Operations
          </span>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          <StatusChip variant="gamma">CLEAR</StatusChip>
          <StatusChip variant="beta">HEAVY</StatusChip>
          <StatusChip variant="alpha">BLOCKED</StatusChip>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 'var(--space-4)' }}>
        {}
        <GlassCard hover3d>
          <div style={{ position: 'relative', borderRadius: 'var(--radius-lg)', overflow: 'hidden', background: 'var(--bg-void)', height: 380 }}>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
            {}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              style={{
                position: 'absolute', top: 'var(--space-4)', left: 'var(--space-4)',
                background: 'rgba(0, 7, 103, 0.8)', backdropFilter: 'blur(12px)',
                borderRadius: 'var(--radius-lg)', padding: 'var(--space-4)', minWidth: 200,
                border: '1px solid rgba(162, 201, 255, 0.08)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-2)' }}>
                <span style={{ color: 'var(--secondary)', fontWeight: 700, fontSize: '0.8rem', letterSpacing: '0.05em' }}>SECTOR INTELLIGENCE</span>
                <span className="text-label">ZONE 04</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-2)' }}>
                <span style={{ fontSize: '0.8rem' }}>Saturation Level</span>
                <span style={{ fontWeight: 700, color: 'var(--secondary)' }}>{saturation}%</span>
              </div>
              <div className="progress-bar progress-green">
                <div className="progress-bar-fill" style={{ width: `${saturation}%` }} />
              </div>
              <div style={{ marginTop: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 'var(--space-1)', fontSize: '0.7rem', color: 'var(--on-surface-variant)' }}>
                <Radio size={12} style={{ color: 'var(--primary)' }} /> RESPONSE LATENCY <span style={{ color: 'var(--secondary)' }}>+1.4m Variance</span>
              </div>
            </motion.div>
            <div style={{ position: 'absolute', bottom: 'var(--space-3)', right: 'var(--space-3)', display: 'flex', gap: 'var(--space-2)' }}>
              <PulseButton variant="ghost" size="sm" onClick={() => setSaturation(Math.min(99, saturation + 3))}>+</PulseButton>
              <PulseButton variant="ghost" size="sm" onClick={() => setSaturation(Math.max(10, saturation - 3))}>−</PulseButton>
            </div>
          </div>
        </GlassCard>

        {}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          {}
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
              <h4 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700 }}>NODE CAPACITY MATRIX</h4>
              <StatusChip variant="alpha" pulse>LIVE</StatusChip>
            </div>
            <div className="grid-2" style={{ gap: 'var(--space-3)' }}>
              {hospitals.map((h, i) => (
                <motion.div
                  key={h.name}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', padding: 'var(--space-3)' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', fontWeight: 700, marginBottom: 'var(--space-1)' }}>
                    <span>{h.name}</span>
                    <span style={{ color: h.color, fontSize: '0.6rem' }}>{h.status}</span>
                  </div>
                  <AnimatedCounter value={h.capacity} suffix="%" className="" decimals={0} />
                  <div className="progress-bar" style={{ marginTop: 'var(--space-1)' }}>
                    <div className="progress-bar-fill" style={{ width: `${h.capacity}%`, background: h.color }} />
                  </div>
                </motion.div>
              ))}
            </div>
          </GlassCard>

          {}
          <GlassCard>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
              <TrendingUp size={18} style={{ color: 'var(--primary)' }} />
              <h4 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700 }}>NEURAL OPTIMIZER</h4>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
              {signals.map((s, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.15 }}
                  whileHover={{ scale: 1.01, background: 'var(--bg-surface-high)' }}
                  style={{
                    background: i === 0 ? 'rgba(255, 84, 76, 0.06)' : 'var(--bg-surface)',
                    borderRadius: 'var(--radius-md)',
                    padding: 'var(--space-3)',
                    border: i === 0 ? '1px solid rgba(255, 84, 76, 0.15)' : '1px solid transparent',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.7rem', color: i === 0 ? 'var(--tertiary)' : 'var(--on-surface)', marginBottom: 'var(--space-1)', letterSpacing: '0.04em' }}>
                    {s.title}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', lineHeight: 1.4 }}>{s.text}</div>
                  {s.status && (
                    <div style={{ marginTop: 'var(--space-1)', fontSize: '0.6rem', color: 'var(--on-surface-dim)' }}>• {s.status} {s.time}</div>
                  )}
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </div>
      </div>

      {}
      <GlassCard>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <RotateCw size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0 }}>ACTIVE REROUTE MATRIX</h3>
          </div>
          <StatusChip variant="gamma">TIME SAVED: +14H 22M AGGREGATE</StatusChip>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>UNIT ID</th><th>ORIGINAL ROUTE</th><th>REROUTED TO</th><th>TIME SAVED</th><th>STATUS</th><th>ACTION</th>
            </tr>
          </thead>
          <tbody>
            {[
              { unit: 'ALS-07', from: 'H03 (Diverted)', to: 'H05 Metro Gen', saved: '8m 30s', status: 'Active' },
              { unit: 'MICU-02', from: 'H01 (Traffic)', to: 'H07 North Reach', saved: '12m 15s', status: 'En-Route' },
              { unit: 'BLS-04', from: 'H06 (Full)', to: 'H02 City Trauma', saved: '5m 45s', status: 'Completed' },
            ].map((r, i) => (
              <motion.tr key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.1 }}>
                <td><span className="text-mono" style={{ color: 'var(--primary)', fontSize: '0.8rem' }}>{r.unit}</span></td>
                <td style={{ color: 'var(--tertiary)', fontSize: '0.8rem' }}>{r.from}</td>
                <td><ArrowRight size={12} style={{ marginRight: 4, color: 'var(--secondary)' }} />{r.to}</td>
                <td style={{ color: 'var(--secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>{r.saved}</td>
                <td><StatusChip variant={r.status === 'Active' ? 'gamma' : r.status === 'En-Route' ? 'delta' : 'beta'}>{r.status}</StatusChip></td>
                <td>
                  <PulseButton variant="ghost" size="sm" onClick={() => sim.stepAction({ action_type: 'reroute', unit_id: r.unit, new_hospital_id: 'H05' })}>
                    Modify
                  </PulseButton>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </GlassCard>
    </div>
  );
}
