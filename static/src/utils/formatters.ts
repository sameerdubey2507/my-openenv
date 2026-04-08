export function formatNumber(n: number, decimals = 1): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(decimals) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(decimals) + 'K';
  return n.toFixed(decimals);
}

export function formatTime(minutes: number): string {
  const m = Math.floor(minutes);
  const s = Math.round((minutes - m) * 60);
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

export function formatClock(): string {
  const now = new Date();
  return now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function getPriorityColor(priority: string): string {
  switch (priority?.toUpperCase()) {
    case 'P1': case 'ALPHA': case 'IMMEDIATE': return 'var(--priority-alpha)';
    case 'P2': case 'BETA': case 'DELAYED': return 'var(--priority-beta)';
    case 'P3': case 'GAMMA': case 'MINIMAL': return 'var(--priority-gamma)';
    default: return 'var(--priority-delta)';
  }
}

export function getPriorityChipClass(priority: string): string {
  switch (priority?.toUpperCase()) {
    case 'P1': case 'ALPHA': case 'IMMEDIATE': return 'chip-alpha';
    case 'P2': case 'BETA': case 'DELAYED': return 'chip-beta';
    case 'P3': case 'GAMMA': case 'MINIMAL': return 'chip-gamma';
    default: return 'chip-delta';
  }
}
