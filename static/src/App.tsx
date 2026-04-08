import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import TopNav, { TabId } from './components/layout/TopNav';
import CommandCenter from './components/command-center/CommandCenter';
import GridRouting from './components/grid-routing/GridRouting';
import RapidDispatch from './components/rapid-dispatch/RapidDispatch';
import TraumaFlow from './components/trauma-flow/TraumaFlow';
import BioArchives from './components/bio-archives/BioArchives';
import SystemAnalytics from './components/system-analytics/SystemAnalytics';
import Theory from './components/theory/Theory';
import EmergencyModal from './components/emergency/EmergencyModal';
import { useSimulation } from './hooks/useSimulation';
import { Toaster } from 'react-hot-toast';
import { AdminProvider } from './context/AdminContext';

const pageVariants = {
  initial: { opacity: 0, y: 12, scale: 0.99 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: -12, scale: 0.99 },
};

function AppInner() {
  const [activeTab, setActiveTab] = useState<TabId>('command');
  const [emergencyOpen, setEmergencyOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const sim = useSimulation('default');

  const fleetStatus = sim.state.done ? 'IDLE' : 'STABLE';
  const nodesCount = '12/12';

  const renderTab = () => {
    switch (activeTab) {
      case 'command':   return <CommandCenter sim={sim} />;
      case 'grid':      return <GridRouting sim={sim} />;
      case 'dispatch':  return <RapidDispatch sim={sim} />;
      case 'trauma':    return <TraumaFlow sim={sim} />;
      case 'bio':       return <BioArchives sim={sim} />;
      case 'analytics': return <SystemAnalytics sim={sim} />;
      case 'theory':    return <Theory />;
    }
  };

  return (
    <div className="dashboard-layout">
      <Toaster position="bottom-left" toastOptions={{
        style: {
          background: 'var(--bg-surface-high)',
          color: 'var(--on-surface)',
          border: '1px solid var(--ghost-border)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          backdropFilter: 'blur(10px)',
        }
      }} />
      <TopNav
        activeTab={activeTab}
        onTabChange={setActiveTab}
        simActive={!!sim.state.taskId && !sim.state.done}
        fleetStatus={fleetStatus}
        nodesCount={nodesCount}
        onEmergency={() => setEmergencyOpen(true)}
        onSettingsClick={() => setSettingsOpen(!settingsOpen)}
      />

      {}
      {sim.state.taskId && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          style={{
            background: 'var(--bg-surface)',
            borderBottom: '1px solid var(--ghost-border)',
            padding: 'var(--space-2) var(--space-6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <div style={{ display: 'flex', gap: 'var(--space-4)' }}>
            <span style={{ color: 'var(--on-surface-variant)' }}>TASK: <span style={{ color: 'var(--primary)' }}>{sim.state.taskId}</span></span>
            <span style={{ color: 'var(--on-surface-variant)' }}>EPISODE: <span style={{ color: 'var(--on-surface)' }}>{sim.state.episodeId?.slice(0, 8)}</span></span>
            <span style={{ color: 'var(--on-surface-variant)' }}>STEP: <span style={{ color: 'var(--secondary)' }}>{sim.state.step}/{sim.state.maxSteps}</span></span>
            <span style={{ color: 'var(--on-surface-variant)' }}>REWARD: <span style={{ color: sim.state.reward >= 0 ? 'var(--secondary)' : 'var(--tertiary)' }}>{sim.state.reward.toFixed(4)}</span></span>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
            <span style={{ color: 'var(--on-surface-variant)' }}>QUEUE: {sim.state.queueLength}</span>
            <span style={{ color: 'var(--on-surface-variant)' }}>ACTIVE: {sim.state.activePatients}</span>
            <span style={{ color: 'var(--on-surface-variant)' }}>RESOLVED: {sim.state.resolvedPatients}</span>
            {sim.state.done && <span style={{ color: 'var(--tertiary)', fontWeight: 700 }}>⬤ EPISODE COMPLETE</span>}
          </div>
        </motion.div>
      )}

      {}
      {sim.gradeResult && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          style={{
            background: sim.gradeResult.beats_baseline ? 'rgba(125, 220, 122, 0.08)' : 'rgba(255, 180, 77, 0.08)',
            borderBottom: `1px solid ${sim.gradeResult.beats_baseline ? 'rgba(125, 220, 122, 0.2)' : 'rgba(255, 180, 77, 0.2)'}`,
            padding: 'var(--space-2) var(--space-6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.75rem',
          }}
        >
          <span>
            <strong style={{ color: sim.gradeResult.beats_baseline ? 'var(--secondary)' : 'var(--priority-beta)' }}>
              {sim.gradeResult.beats_baseline ? '✓ BEATS BASELINE' : '✗ BELOW BASELINE'}
            </strong>
            {' — '}Score: {((sim.gradeResult.final_score ?? sim.gradeResult.score ?? 0) * 100).toFixed(1)}%
          </span>
        </motion.div>
      )}

      {}
      <main className="dashboard-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            {renderTab()}
          </motion.div>
        </AnimatePresence>
      </main>

      {}
      <EmergencyModal
        isOpen={emergencyOpen}
        onClose={() => setEmergencyOpen(false)}
        sim={sim}
      />

      {}
      <AnimatePresence>
        {sim.state.loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              bottom: 'var(--space-6)',
              right: 'var(--space-6)',
              background: 'var(--bg-surface-high)',
              borderRadius: 'var(--radius-lg)',
              padding: 'var(--space-3) var(--space-5)',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-2)',
              fontSize: '0.8rem',
              boxShadow: 'var(--glow-primary)',
              zIndex: 1000,
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(162, 201, 255, 0.1)',
            }}
          >
            <span style={{
              width: 14, height: 14,
              border: '2px solid var(--primary)',
              borderTopColor: 'transparent',
              borderRadius: '50%',
              animation: 'spin 0.7s linear infinite',
              display: 'inline-block',
            }} />
            Processing...
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function App() {
  return (
    <AdminProvider>
      <AppInner />
    </AdminProvider>
  );
}
