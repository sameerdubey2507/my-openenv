import { ReactNode } from 'react';
import AnimatedCounter from './AnimatedCounter';

interface Props {
  icon: ReactNode;
  label: string;
  value: number;
  suffix?: string;
  subtext?: string;
  subtextColor?: string;
  progressValue?: number;
  progressColor?: string;
  className?: string;
}

export default function MetricCard({ icon, label, value, suffix = '', subtext, subtextColor, progressValue, progressColor = 'var(--secondary)', className = '' }: Props) {
  return (
    <div className={`card ${className}`} style={{ padding: 'var(--space-4) var(--space-5)', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className="text-label">{label}</span>
        <span style={{ color: 'var(--primary)', opacity: 0.6 }}>{icon}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-2)' }}>
        <AnimatedCounter value={value} suffix={suffix} decimals={suffix === '%' ? 1 : 0} />
        {subtext && <span style={{ fontSize: '0.75rem', color: subtextColor || 'var(--on-surface-variant)' }}>{subtext}</span>}
      </div>
      {progressValue !== undefined && (
        <div className="progress-bar" style={{ marginTop: 'var(--space-1)' }}>
          <div className="progress-bar-fill" style={{ width: `${Math.min(progressValue, 100)}%`, background: progressColor }} />
        </div>
      )}
    </div>
  );
}
