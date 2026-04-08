import { BookOpen, Target, CheckCircle, ShieldAlert, Cpu, Network, Layers, GitBranch, Zap } from 'lucide-react';
import GlassCard from '../shared/GlassCard';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler);

export default function Theory() {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 200, damping: 20 } }
  };

  const chartData = {
    labels: Array.from({length: 20}).map((_, i) => `Epoch ${i * 50}`),
    datasets: [{
      label: 'PPO Reward Optimization',
      data: Array.from({length: 20}).map((_, i) => -500 + 600 * (1 - Math.exp(-i/4)) + Math.random() * 20),
      borderColor: '#A2C9FF',
      backgroundColor: 'rgba(162, 201, 255, 0.1)',
      fill: true,
      tension: 0.4,
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: 'rgba(120,130,160,0.1)' }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } },
      y: { grid: { color: 'rgba(120,130,160,0.1)' }, ticks: { color: '#7882A0', font: { family: 'Inter', size: 10 } } }
    },
    plugins: { legend: { display: false } },
    elements: { point: { radius: 0 } }
  };

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="show" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', paddingBottom: 'var(--space-8)' }}>
      <motion.div variants={itemVariants} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
        <BookOpen size={28} style={{ color: 'var(--primary)' }} />
        <h2 style={{ margin: 0, fontSize: '1.5rem', letterSpacing: 1 }}>NATIONAL ARCHITECTURE: EMERGI-ENV V2.0</h2>
      </motion.div>

      <motion.p variants={itemVariants} style={{ color: 'var(--on-surface-variant)', fontSize: '0.95rem', lineHeight: 1.8, maxWidth: '900px' }}>
        The EMERGI-ENV V2.0 System is a specialized, production-grade Multi-Agent Reinforcement Learning (MARL) simulation designed for national-level emergency response optimization. This document serves as the canonical reference for the mathematical formulation, agent interaction protocols, and neural triage heuristics that govern the Command Nucleus.
      </motion.p>

      <div className="grid-2" style={{ gap: 'var(--space-4)', marginTop: 'var(--space-4)' }}>
        <motion.div variants={itemVariants}>
        <div style={{ height: '100%' }}>
          <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
            <Target style={{ color: 'var(--secondary)' }} size={20} />
            <h3 style={{ margin: 0 }}>Core Objective (Golden Hour Preservation)</h3>
          </div>
          <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem', lineHeight: 1.7 }}>
            The primary aim is the maximization of the <strong>Golden Hour Survival Rate</strong> across a dynamically evolving, stochastic incident landscape. Traditional fixed-rule dispatch (Closest-Available) fails under Mass Casualty Incident (MCI) conditions due to premature fleet exhaustion and sub-optimal hospital routing. EMERGI-ENV employs Proximal Policy Optimization (PPO) to learn spatiotemporal preemptive staging, allowing units to organically reposition based on predicted topological risk.
          </p>
        </GlassCard>
        </div>
        </motion.div>

        <motion.div variants={itemVariants}>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
            <Network style={{ color: 'var(--primary)' }} size={20} />
            <h3 style={{ margin: 0 }}>Markov Decision Process (MDP)</h3>
          </div>
          <ul style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem', lineHeight: 1.7, paddingLeft: 'var(--space-4)', margin: 0 }}>
            <li style={{ marginBottom: 'var(--space-2)' }}><strong>State Space (S):</strong> A 2048-dimensional tensor encoding active incidents (severity, location, age), fleet status (position, type, fatigue), hospital capacities, and graph connectivity.</li>
            <li style={{ marginBottom: 'var(--space-2)' }}><strong>Action Space (A):</strong> Discrete multidimensional actions: Dispatch (Unit, Incident, Hospital), Preempt (Unit, Node), Crew Crossover.</li>
            <li><strong>Reward (R):</strong> Non-linear function prioritizing P1 survival (+100) while heavily penalizing protocol breaches (-50) and critical degradation (-200).</li>
          </ul>
        </GlassCard>
        </motion.div>
      </div>

      <motion.div variants={itemVariants}>
        <GlassCard>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
            <Cpu style={{ color: 'var(--primary)' }} size={20} />
            <h3 style={{ margin: 0 }}>PPO Reward Optimization Trajectory</h3>
          </div>
          <div style={{ height: 200, width: '100%' }}>
            <Line data={chartData} options={chartOptions as any} />
          </div>
        </GlassCard>
      </motion.div>

      <motion.div variants={itemVariants}>
      <GlassCard>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
          <Layers style={{ color: 'var(--tertiary)' }} size={20} />
          <h3 style={{ margin: 0 }}>System Modules & Control Surfaces</h3>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-4)', marginTop: 'var(--space-3)' }}>
          <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', borderTop: '3px solid var(--primary)' }}>
            <h4 style={{ color: 'var(--on-surface)', marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 8 }}><Cpu size={14}/> Neural Triage</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)', lineHeight: 1.6 }}>Automated severity classification mapping high-dimensional vitals (SpO2, EtCO2, GCS) into discrete priority classes. Overrides human dispatcher bias during MCIs.</p>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', borderTop: '3px solid var(--secondary)' }}>
            <h4 style={{ color: 'var(--on-surface)', marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 8 }}><GitBranch size={14}/> Dynamic Rerouting</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)', lineHeight: 1.6 }}>Real-time recalculation of optimal paths accounting for active Level 1 trauma diversions and stochastic traffic choke-points.</p>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', borderTop: '3px solid var(--tertiary)' }}>
            <h4 style={{ color: 'var(--on-surface)', marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 8 }}><ShieldAlert size={14}/> Mutual Aid Cascades</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)', lineHeight: 1.6 }}>Automated inter-regional aid requests that trigger when local saturation exceeds 85%. Plunges adjacent districts into 'Standby' status.</p>
          </div>
        </div>
      </GlassCard>
      </motion.div>

      <motion.div variants={itemVariants}>
      <GlassCard>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
          <CheckCircle style={{ color: 'var(--secondary)' }} size={20} />
          <h3 style={{ margin: 0 }}>API & Engine Validation</h3>
        </div>
        <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem', lineHeight: 1.7 }}>
          The backend engine resolves logical step-functions at 1000Hz via FastAPI WebSockets. Agents query <code>/state</code> telemetry, submit high-frequency discrete choices via <code>/step</code>, and are routinely evaluated automatically through the <code>/grade</code> self-test module. The grading module utilizes a deterministic baseline heuristic (Closest-Available with Strict P1 Priority) to benchmark any deep RL policies developed.
        </p>
        <div style={{ background: 'rgba(5, 5, 7, 0.4)', padding: 'var(--space-4)', borderRadius: 'var(--radius-lg)', marginTop: 'var(--space-4)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--on-surface-variant)', border: '1px solid var(--ghost-border)' }}>

          $ uvicorn server.main:app --host 0.0.0.0 --port 7860<br/>
          <span style={{ color: 'var(--primary)' }}>[INFO]</span> Loaded 9 Grading Scenarios<br/>
          <span style={{ color: 'var(--primary)' }}>[INFO]</span> Initializing Network Topology (Nodes=50, Edges=120)<br/>
          <span style={{ color: 'var(--secondary)' }}>[SUCCESS]</span> FastAPI Engine Active on ws:
        </div>
      </GlassCard>
      </motion.div>

      <motion.div variants={itemVariants}>
      <GlassCard>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
          <Zap style={{ color: 'var(--priority-beta)' }} size={20} />
          <h3 style={{ margin: 0 }}>Modification Guidelines</h3>
        </div>
        <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem', lineHeight: 1.7 }}>
          Researchers and engineers modifying the simulation should adhere to the following strict parameters to ensure benchmark consistency:
        </p>
        <ul style={{ color: 'var(--on-surface-variant)', fontSize: '0.85rem', lineHeight: 1.7, paddingLeft: 'var(--space-4)', margin: 0 }}>
          <li><strong>Fleet Modifications:</strong> Total active units must not exceed 50 per region. ALS/BLS ratio is configurable.</li>
          <li><strong>MCI Scenarios:</strong> The <code>incident_generator</code> must maintain a Poisson distribution lambda of 0.2 for standard epochs, jumping to 5.0 during MCI events.</li>
          <li><strong>Hospital Saturation:</strong> Diversions must organically clear at a randomized rate inversely proportional to the current regional trauma influx. Hardcoded clearing is prohibited.</li>
        </ul>
      </GlassCard>
      </motion.div>
    </motion.div>
  );
}
