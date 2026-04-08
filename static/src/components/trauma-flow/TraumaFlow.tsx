import { useEffect, useRef, useState } from 'react';
import { HeartPulse, Activity, AlertCircle, Building, Crosshair, Thermometer, Box } from 'lucide-react';
import { motion } from 'framer-motion';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';
import GlassCard from '../shared/GlassCard';
import StatusChip from '../shared/StatusChip';
import PulseButton from '../shared/PulseButton';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

interface Props { sim: any; }

const hospitals = [
  { id: 'H01', name: 'CITY GENERAL', type: 'LEVEL 1 TRAUMA', status: 'ACTIVE', beds: 4, diversions: 0, util: 85 },
  { id: 'H02', name: 'MEMORIAL WEST', type: 'LEVEL 2 TRAUMA', status: 'DIVERT', beds: 0, diversions: 12, util: 100 },
  { id: 'H03', name: 'ST. JUDE NEURO', type: 'SPECIALTY (NEURO)', status: 'ACTIVE', beds: 2, diversions: 1, util: 60 },
  { id: 'H04', name: 'UNIVERSITY MED', type: 'LEVEL 1 TRAUMA', status: 'ACTIVE', beds: 12, diversions: 0, util: 45 },
];

export default function TraumaFlow({ sim }: Props) {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: { grid: { color: 'rgba(120,130,160,0.1)', drawBorder: false }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } },
      y: { grid: { color: 'rgba(120,130,160,0.1)', drawBorder: false }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } }
    },
  };

  const bedData = {
    labels: ['L1 Trauma (Adult)', 'L1 Trauma (Ped)', 'L2 Trauma', 'Neuro Surge', 'Burn Unit', 'Cardiac CATH'],
    datasets: [{
      label: 'Regional Bed Availability (%)',
      data: [15, 80, 45, 10, 5, 60],
      backgroundColor: (ctx: any) => { const v = ctx.raw; return v < 20 ? '#FF544C' : v < 50 ? '#FFB74D' : '#7DDC7A'; },
      borderRadius: 4,
    }]
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', paddingBottom: 'var(--space-6)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
        <HeartPulse size={24} style={{ color: 'var(--primary)' }} />
        <h2 style={{ margin: 0 }}>TRAUMA FLOW & SURVIVAL DYNAMICS</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.3fr) minmax(0, 1fr)', gap: 'var(--space-4)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <Building size={18} style={{ color: 'var(--primary)' }} /> HOSPITAL DIVERSION MATRIX
              </h3>
              <StatusChip variant="gamma">SYSTEM NOMINAL</StatusChip>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              {hospitals.map((h, i) => (
                <motion.div key={h.id} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                  style={{ background: 'var(--bg-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderLeft: `3px solid ${h.status === 'DIVERT' ? 'var(--tertiary)' : 'var(--secondary)'}` }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-1)' }}>
                      <span style={{ fontWeight: 800, fontFamily: 'var(--font-mono)' }}>{h.id} / {h.name}</span>
                      <StatusChip variant={h.status === 'DIVERT' ? 'alpha' : 'gamma'}>{h.status}</StatusChip>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>{h.type}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'center' }}>
                    <div style={{ textAlign: 'right' }}>
                      <div className="text-label" style={{ fontSize: '0.65rem' }}>AVAILABLE BEDS / UTIL</div>
                      <div style={{ fontSize: '1rem', fontWeight: 800, color: h.beds === 0 ? 'var(--tertiary)' : 'var(--on-surface)' }}>{h.beds} <span style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)', fontWeight: 400 }}>({h.util}%)</span></div>
                    </div>
                    <div style={{ textAlign: 'right', width: 70 }}>
                      <div className="text-label" style={{ fontSize: '0.65rem' }}>DIVERSIONS</div>
                      <div style={{ fontSize: '1rem', fontWeight: 800 }}>{h.diversions}</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </GlassCard>

          <GlassCard>
            <h3 style={{ margin: '0 0 var(--space-4) 0', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Box size={18} style={{ color: 'var(--secondary)' }} /> REGIONAL CAPACITY ARRAY
            </h3>
            <div style={{ height: 260, width: '100%' }}>
              <Bar data={bedData} options={{ ...chartOptions, indexAxis: 'x' } as any} />
            </div>
          </GlassCard>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          <GlassCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <Activity size={18} style={{ color: 'var(--priority-beta)' }} /> GOLDEN HOUR PRESERVATION
              </h3>
              <StatusChip variant="beta">+4.1% MoM</StatusChip>
            </div>
            <div className="progress-bar progress-green" style={{ height: 12, marginBottom: 8 }}>
               <div className="progress-bar-fill" style={{ width: '82.5%' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>
              <span>0%</span> <span style={{ color: 'var(--secondary)', fontWeight: 700 }}>82.5% TARGET MAINTAINED</span> <span>100%</span>
            </div>
            <p style={{ marginTop: 'var(--space-4)', fontSize: '0.8rem', color: 'var(--on-surface-variant)', lineHeight: 1.6 }}>
              The Golden Hour model measures the percentage of critical P1 traumas delivered to authoritative care within 60 minutes of incident generation. Current RL policies prioritize P1s aggressively over P2s, resulting in a 4.1% increase in preservation rates at the cost of P2 wait times.
            </p>
          </GlassCard>

          <GlassCard>
            <h3 style={{ margin: '0 0 var(--space-4) 0', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Crosshair size={18} style={{ color: 'var(--tertiary)' }} /> PROTOCOL ENFORCEMENT
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--secondary)' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: 4 }}>OXYGEN ADMIN (SpO2 &lt; 90%)</div>
                <div style={{ fontSize: '0.85rem' }}>99.2% COMPLIANCE RATE</div>
              </div>
              <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--primary)' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: 4 }}>TOURNIQUET PLACEMENT (MASSIVE HEM.)</div>
                <div style={{ fontSize: '0.85rem' }}>94.1% COMPLIANCE RATE</div>
              </div>
              <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--tertiary)' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: 4 }}>AIRWAY INTUBATION (GCS &lt; 8)</div>
                <div style={{ fontSize: '0.85rem' }}>81.4% COMPLIANCE RATE  <span style={{ color: 'var(--tertiary)', fontSize: '0.65rem' }}>▼ FLAG: SUB-OPTIMAL</span></div>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
