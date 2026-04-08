import { useState } from 'react';
import { Zap, AlertTriangle } from 'lucide-react';
import Modal from '../shared/Modal';
import PulseButton from '../shared/PulseButton';
import StatusChip from '../shared/StatusChip';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  sim: any;
}

const taskOptions = [
  { id: 'task1_single_triage', name: 'Single Triage', diff: 'EASY', color: 'var(--secondary)' },
  { id: 'task2_hospital_route', name: 'Hospital Route', diff: 'EASY', color: 'var(--secondary)' },
  { id: 'task3_unit_type', name: 'Unit Type', diff: 'EASY', color: 'var(--secondary)' },
  { id: 'task4_multi_incident', name: 'Multi Incident', diff: 'MEDIUM', color: 'var(--priority-beta)' },
  { id: 'task5_dynamic_rerouting', name: 'Dynamic Rerouting', diff: 'MEDIUM', color: 'var(--priority-beta)' },
  { id: 'task6_prepositioning', name: 'Pre-positioning', diff: 'MEDIUM', color: 'var(--priority-beta)' },
  { id: 'task7_mci_start', name: 'MCI START Triage', diff: 'HARD', color: 'var(--tertiary)' },
  { id: 'task8_transfer_cascade', name: 'Transfer Cascade', diff: 'HARD', color: 'var(--tertiary)' },
  { id: 'task9_surge', name: 'City-Wide Surge', diff: 'HARD', color: 'var(--tertiary)' },
];

export default function EmergencyModal({ isOpen, onClose, sim }: Props) {
  const [selected, setSelected] = useState('task1_single_triage');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleLaunch = async () => {
    setLoading(true);
    setResult(null);
    const res = await sim.resetEpisode(selected);
    setLoading(false);
    if (res) {
      setResult(res);
      setTimeout(onClose, 1500);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="⚡ NEW EMERGENCY SIMULATION" width="640px">
      <div style={{ marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
          <AlertTriangle size={16} style={{ color: 'var(--tertiary)' }} />
          <span style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)' }}>Select a task scenario to initialize a new simulation episode</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          {taskOptions.map((t) => (
            <div
              key={t.id}
              onClick={() => setSelected(t.id)}
              style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: selected === t.id ? 'var(--bg-surface-high)' : 'var(--bg-surface)',
                borderRadius: 'var(--radius-md)', padding: 'var(--space-3)',
                cursor: 'pointer', transition: 'all 0.2s',
                border: selected === t.id ? '1px solid var(--primary)' : '1px solid transparent',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: selected === t.id ? 'var(--primary)' : 'var(--outline-variant)' }} />
                <span style={{ fontWeight: selected === t.id ? 600 : 400 }}>{t.name}</span>
              </div>
              <StatusChip variant={t.diff === 'EASY' ? 'gamma' : t.diff === 'MEDIUM' ? 'beta' : 'alpha'}>{t.diff}</StatusChip>
            </div>
          ))}
        </div>
      </div>

      {result && (
        <div style={{ background: 'rgba(125, 220, 122, 0.08)', borderRadius: 'var(--radius-md)', padding: 'var(--space-3)', marginBottom: 'var(--space-3)', border: '1px solid rgba(125, 220, 122, 0.2)' }}>
          <span style={{ color: 'var(--secondary)', fontWeight: 600, fontSize: '0.85rem' }}>✓ Episode Initialized!</span>
          <div style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', marginTop: 'var(--space-1)', fontFamily: 'var(--font-mono)' }}>
            ID: {result.episode_id} | Max Steps: {result.max_steps}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 'var(--space-3)', justifyContent: 'flex-end' }}>
        <PulseButton variant="ghost" onClick={onClose}>Cancel</PulseButton>
        <PulseButton variant="danger" icon={<Zap size={16} />} loading={loading} onClick={handleLaunch}>
          LAUNCH SIMULATION
        </PulseButton>
      </div>
    </Modal>
  );
}
