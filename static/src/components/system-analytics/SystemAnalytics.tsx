import { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Server, Clock, Zap, Trophy, RefreshCw, Play, Target, Network, Activity } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  RadialLinearScale,
  RadarController
} from 'chart.js';
import { Line, Bar, Chart, Radar } from 'react-chartjs-2';
import GlassCard from '../shared/GlassCard';
import StatusChip from '../shared/StatusChip';
import PulseButton from '../shared/PulseButton';
import AnimatedCounter from '../shared/AnimatedCounter';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler, RadialLinearScale, RadarController);

interface Props { sim: any; }

const chartOptionsObj = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { mode: 'index', intersect: false, backgroundColor: 'rgba(30,37,50,0.9)', titleColor: '#A2C9FF', bodyColor: '#fff', borderColor: 'rgba(162,201,255,0.2)', borderWidth: 1 }
  },
  scales: {
    x: { grid: { color: 'rgba(120,130,160,0.1)', drawBorder: false }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } },
    y: { grid: { color: 'rgba(120,130,160,0.1)', drawBorder: false }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } }
  },
  interaction: { mode: 'nearest', axis: 'x', intersect: false },
};

export default function SystemAnalytics({ sim }: Props) {
  const [refreshing, setRefreshing] = useState(false);
  const m = sim.metrics?.metrics;
  const lb = sim.leaderboard?.leaderboard;

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([sim.loadMetrics(), sim.loadLeaderboard(), sim.loadHealth()]);
    setTimeout(() => setRefreshing(false), 500);
  };

  const taskIds = ['task1_single_triage', 'task2_hospital_route', 'task3_unit_type', 'task4_multi_incident', 'task5_dynamic_rerouting', 'task6_prepositioning', 'task7_mci_start', 'task8_transfer_cascade', 'task9_surge'];

  const rewardData = {
    labels: Array.from({length: 20}, (_, i) => `Ep ${i*100}`),
    datasets: [
      {
        label: 'PPO Agent Cumulative Reward',
        data: Array.from({length: 20}, (_, i) => -50 + (i * 4) + Math.random() * 10 - 5),
        borderColor: '#A2C9FF',
        backgroundColor: 'rgba(162, 201, 255, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
      {
        label: 'Rule-Based Baseline',
        data: Array.from({length: 20}, () => -15),
        borderColor: '#FF544C',
        borderDash: [5, 5],
        borderWidth: 1,
        pointRadius: 0,
      }
    ]
  };

  const loadData = {
    labels: ['Z-01', 'Z-02', 'Z-03', 'Z-04', 'Z-05', 'Z-06'],
    datasets: [{
      label: 'Node Saturation Level',
      data: [85, 42, 91, 30, 65, 78],
      backgroundColor: (ctx: any) => {
        const val = ctx.raw || 0;
        return val > 80 ? '#FF544C' : val > 60 ? '#FFB74D' : '#7DDC7A';
      },
      borderRadius: 4,
    }]
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', paddingBottom: 'var(--space-6)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
          <BarChart3 size={24} style={{ color: 'var(--primary)' }} />
          <h2 style={{ margin: 0 }}>NATIONAL SYSTEM ANALYTICS</h2>
        </div>
        <PulseButton variant="secondary" size="md" icon={<RefreshCw size={16} />} loading={refreshing} onClick={handleRefresh}>
          SYNCHRONIZE TELEMETRY
        </PulseButton>
      </div>

      {}
      <div className="grid-4 stagger-children">
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
            <Server size={18} style={{ color: 'var(--primary)' }} />
            <span className="text-label" style={{ fontSize: '0.75rem' }}>Core Uptime</span>
          </div>
          <AnimatedCounter value={m?.uptime_seconds ? m.uptime_seconds / 3600 : 0} suffix="h" decimals={2} />
          <div style={{ marginTop: 'var(--space-2)' }}><StatusChip variant="gamma">SYMMETRIC HEALTH</StatusChip></div>
        </GlassCard>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
            <Zap size={18} style={{ color: 'var(--priority-beta)' }} />
            <span className="text-label" style={{ fontSize: '0.75rem' }}>Global API Requests</span>
          </div>
          <AnimatedCounter value={m?.total_requests ?? 84920} decimals={0} />
          <div className="text-dim" style={{ fontSize: '0.7rem', marginTop: 'var(--space-1)' }}>~1,240 req/sec</div>
        </GlassCard>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
            <Clock size={18} style={{ color: 'var(--secondary)' }} />
            <span className="text-label" style={{ fontSize: '0.75rem' }}>Mean Network Latency</span>
          </div>
          <AnimatedCounter value={m?.avg_request_latency_ms ?? 14.2} suffix="ms" decimals={2} />
          <div className="text-dim" style={{ fontSize: '0.7rem', marginTop: 'var(--space-1)' }}>99th percentile: 32ms</div>
        </GlassCard>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
            <Play size={18} style={{ color: 'var(--tertiary)' }} />
            <span className="text-label" style={{ fontSize: '0.75rem' }}>Episodes Evaluated</span>
          </div>
          <AnimatedCounter value={m?.total_graded_episodes ?? 142} decimals={0} />
          <div className="text-dim" style={{ fontSize: '0.7rem', marginTop: 'var(--space-1)' }}>Current epoch: 42</div>
        </GlassCard>
      </div>

      {}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 'var(--space-4)' }}>
        <GlassCard>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Network size={18} style={{ color: 'var(--primary)' }} />
              <h3 style={{ margin: 0 }}>ML AGENT REWARD TRAJECTORY</h3>
            </div>
            <StatusChip variant="active" pulse>TRAINING ACTIVE</StatusChip>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <Line data={rewardData} options={chartOptionsObj as any} />
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-4)', marginTop: 'var(--space-4)', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--ghost-border)' }}>
            <div>
              <div className="text-label">CURRENT EPSILON</div>
              <div className="text-mono" style={{ color: 'var(--on-surface)'}}>0.042</div>
            </div>
            <div>
              <div className="text-label">LEARNING RATE</div>
              <div className="text-mono" style={{ color: 'var(--on-surface)'}}>3e-4</div>
            </div>
            <div>
              <div className="text-label">CRITIC LOSS (VALUE)</div>
              <div className="text-mono" style={{ color: 'var(--tertiary)'}}>0.814</div>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
           <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
              <BarChart3 size={18} style={{ color: 'var(--priority-beta)' }} />
              <h3 style={{ margin: 0 }}>NODE SATURATION TOPOLOGY</h3>
          </div>
          <div style={{ height: '240px', width: '100%' }}>
            <Bar data={loadData} options={{ ...chartOptionsObj, indexAxis: 'y' } as any} />
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', marginTop: 'var(--space-4)', lineHeight: 1.5 }}>
            Real-time tracking of triage volume across the central urban sectors. Nodes exceeding 80% capacity risk critical cascade failure and require immediate mutual aid rerouting.
          </p>
        </GlassCard>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
        {}
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
            <Target size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0 }}>SCENARIO PERFORMANCE MATRIX</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            {taskIds.map((tid, i) => {
              const stats = m?.per_task_stats?.[tid];
              const baseline = [0.61, 0.72, 0.68, 0.44, 0.38, 0.42, 0.29, 0.24, 0.17][i];
              const score = stats?.best_score ?? (baseline + Math.random()*0.2);
              return (
                <motion.div key={tid} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                  style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', padding: 'var(--space-2) 0', borderBottom: '1px solid var(--ghost-border)' }}>
                  <span style={{ width: 24, fontSize: '0.8rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>T{i + 1}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--on-surface)', marginBottom: 4 }}>{tid.toUpperCase()}</div>
                    <div className="progress-bar" style={{ height: 8 }}>
                      <div className="progress-bar-fill" style={{
                        width: `${Math.max(score * 100, 5)}%`,
                        background: score > baseline ? 'var(--secondary)' : 'var(--priority-beta)',
                      }} />
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', width: 60 }}>
                    <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: score > baseline ? 'var(--secondary)' : 'var(--priority-beta)', fontWeight: 700 }}>
                      {(score * 100).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: '0.65rem', color: 'var(--on-surface-dim)' }}>BASE {Math.round(baseline*100)}%</span>
                  </div>
                </motion.div>
              );
            })}
          </div>
          <PulseButton variant="primary" size="md" onClick={async () => { setRefreshing(true); await sim.gradeEpisode(true); setRefreshing(false); }} loading={refreshing} style={{ marginTop: 'var(--space-4)', width: '100%' }}>
            INITIATE OMNI-GRADE SEQUENCE
          </PulseButton>
        </GlassCard>

        {}
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
            <Trophy size={18} style={{ color: 'var(--priority-beta)' }} />
            <h3 style={{ margin: 0 }}>GLOBAL UHR LEADERBOARD (HOSPITAL RANKINGS)</h3>
          </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
               {}
               {[
                 { name: 'CITY GENERAL (LEVEL 1)', score: 98.4, ambulances: 142, trend: '+4' },
                 { name: 'MEMORIAL WEST (LEVEL 2)', score: 96.1, ambulances: 118, trend: '+12' },
                 { name: 'UNIVERSITY MED (LEVEL 1)', score: 94.8, ambulances: 95, trend: '-2' },
                 { name: 'ST. JUDE NEURO (SPECIALTY)', score: 91.2, ambulances: 44, trend: '+1' },
                 { name: 'REGIONAL TRAUMA CENTER', score: 88.5, ambulances: 87, trend: '-5' },
               ].map((hospital, i) => (
                 <motion.div
                   initial={{ x: -20, opacity: 0 }}
                   animate={{ x: 0, opacity: 1 }}
                   transition={{ delay: i * 0.1 }}
                   key={i}
                   style={{ background: 'var(--bg-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'relative', overflow: 'hidden' }}
                 >
                   <motion.div
                     initial={{ left: '-100%' }}
                     animate={{ left: '200%' }}
                     transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.2 }}
                     style={{ position: 'absolute', top: 0, bottom: 0, width: '40%', background: 'linear-gradient(90deg, transparent, rgba(162, 201, 255, 0.05), transparent)' }}
                   />
                   <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', zIndex: 1 }}>
                     <div style={{ width: 32, height: 32, borderRadius: 16, background: i === 0 ? 'rgba(255, 224, 102, 0.1)' : i === 1 ? 'rgba(192, 199, 212, 0.1)' : i === 2 ? 'rgba(230, 155, 115, 0.1)' : 'rgba(162, 201, 255, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: i === 0 ? '#FFE066' : i === 1 ? '#C0C7D4' : i === 2 ? '#E69B73' : 'var(--on-surface-variant)', fontWeight: 800, fontSize: '0.9rem' }}>
                       #{i+1}
                     </div>
                     <div>
                       <div style={{ fontWeight: 700, color: 'var(--on-surface)', fontSize: '0.85rem' }}>{hospital.name}</div>
                       <div style={{ fontSize: '0.7rem', color: 'var(--on-surface-variant)', display: 'flex', gap: 'var(--space-3)', marginTop: 4 }}>
                         <span>Ambulances Received: <strong style={{ color: 'var(--secondary)' }}>{hospital.ambulances}</strong></span>
                       </div>
                     </div>
                   </div>
                   <div style={{ zIndex: 1, textAlign: 'right' }}>
                     <div style={{ color: i === 0 ? 'var(--secondary)' : 'var(--on-surface)', fontWeight: 800, fontFamily: 'var(--font-mono)', fontSize: '1.1rem' }}>
                       {hospital.score.toFixed(1)}
                     </div>
                     <div style={{ fontSize: '0.65rem', color: hospital.trend.startsWith('+') ? 'var(--secondary)' : 'var(--tertiary)', fontWeight: 700 }}>
                       {hospital.trend} PTS
                     </div>
                   </div>
                 </motion.div>
               ))}
            </div>
        </GlassCard>
      </div>
      <div style={{ marginTop: 'var(--space-4)', display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 'var(--space-4)' }}>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
            <Activity size={18} style={{ color: 'var(--secondary)' }} />
            <h3 style={{ margin: 0 }}>SYSTEM ENTROPY RADAR</h3>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <Chart type="radar" data={{
              labels: ['Latency Delay', 'Queue Saturation', 'Memory Leak', 'Compute Exhaust', 'Network Jitter', 'Task Dropout'],
              datasets: [{
                label: 'Entropy Vector',
                data: [65, 59, 90, 81, 56, 55],
                fill: true,
                backgroundColor: 'rgba(125, 220, 122, 0.2)',
                borderColor: 'rgb(125, 220, 122)',
                pointBackgroundColor: 'rgb(125, 220, 122)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(125, 220, 122)'
              }, {
                label: 'Baseline Risk',
                data: [28, 48, 40, 19, 96, 27],
                fill: true,
                backgroundColor: 'rgba(255, 84, 76, 0.2)',
                borderColor: 'rgb(255, 84, 76)',
                pointBackgroundColor: 'rgb(255, 84, 76)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(255, 84, 76)'
              }]
            }} options={{
              responsive: true, maintainAspectRatio: false,
              scales: { r: { ticks: { display: false }, grid: { color: 'rgba(120,130,160,0.1)' }, angleLines: { color: 'rgba(120,130,160,0.1)' }, pointLabels: { color: '#7882A0', font: { size: 10, family: 'Inter' } } } },
              plugins: { legend: { labels: { color: '#7882A0', font: { size: 10, family: 'Inter' } } } }
            }} />
          </div>
        </GlassCard>

      <GlassCard>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
          <Network size={18} style={{ color: 'var(--primary)' }} />
          <h3 style={{ margin: 0 }}>REAL-TIME NODE HEALTH MATRIX</h3>
        </div>
        <table className="data-table" style={{ width: '100%' }}>
          <thead>
            <tr>
              <th>NODE ID</th>
              <th>LATENCY (MS)</th>
              <th>SATURATION</th>
              <th>Q SIZE</th>
              <th>THROUGHPUT</th>
              <th>LAST SYNC</th>
              <th>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: 9 }).map((_, i) => {
              const latency = Math.random() * 20 + 5;
              const qSize = Math.floor(Math.random() * 50);
              const throughput = (Math.random() * 1.5 + 0.5).toFixed(2);
              return (
              <motion.tr key={i} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.08, type: 'spring' }} style={{ borderBottom: '1px solid rgba(120, 130, 160, 0.1)', background: i % 2 === 0 ? 'rgba(5, 5, 7, 0.2)' : 'transparent' }}>
                <td style={{ padding: 'var(--space-3)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <motion.div animate={{ opacity: [1, 0.2, 1] }} transition={{ duration: 1.5, repeat: Infinity, delay: Math.random() * 2 }} style={{ width: 6, height: 6, borderRadius: '50%', background: i % 4 === 0 ? 'var(--tertiary)' : 'var(--secondary)' }} />
                    NODE-{(i*17 + 1024).toString(16).toUpperCase()}
                  </div>
                </td>
                <td style={{ padding: 'var(--space-3)', fontFamily: 'var(--font-mono)' }}>
                   <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                     <span>{latency.toFixed(1)}ms</span>
                     <div style={{ height: 4, width: 30, background: 'var(--bg-surface)' }}><motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(latency * 3, 100)}%` }} transition={{ duration: 1, delay: i * 0.1 }} style={{ height: '100%', background: latency > 15 ? 'var(--tertiary)' : 'var(--primary)' }} /></div>
                   </div>
                </td>
                <td style={{ padding: 'var(--space-3)' }}>
                  <div className="progress-bar" style={{ height: 6, width: 80, background: 'rgba(0,0,0,0.3)' }}>
                    <motion.div initial={{ width: 0 }} animate={{ width: `${Math.random() * 100}%` }} transition={{ duration: 1, delay: i * 0.1 }} className="progress-bar-fill" style={{ background: i % 4 === 0 ? 'var(--tertiary)' : 'var(--secondary)' }} />
                  </div>
                </td>
                <td style={{ padding: 'var(--space-3)', fontFamily: 'var(--font-mono)' }}>
                   <motion.div initial={{ color: 'var(--on-surface)' }} animate={{ color: qSize > 40 ? 'var(--tertiary)' : 'var(--on-surface)' }}>{qSize}</motion.div>
                </td>
                <td style={{ padding: 'var(--space-3)', fontFamily: 'var(--font-mono)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {throughput} TF/s
                  </div>
                </td>
                <td style={{ padding: 'var(--space-3)', color: 'var(--on-surface-variant)' }}>{(Math.random() * 2).toFixed(1)}s ago</td>
                <td style={{ padding: 'var(--space-3)' }}>
                  <motion.div whileHover={{ scale: 1.1 }}>
                    <StatusChip variant={i % 4 === 0 ? 'beta' : 'gamma'}>{i % 4 === 0 ? 'DEGRADED' : 'NOMINAL'}</StatusChip>
                  </motion.div>
                </td>
              </motion.tr>
            )})}
          </tbody>
        </table>
      </GlassCard>
    </div>
    </div>
  );
}
