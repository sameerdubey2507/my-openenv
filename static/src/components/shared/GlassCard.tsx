import { ReactNode, useRef } from 'react';
import { motion } from 'framer-motion';

interface Props {
  children: ReactNode;
  className?: string;
  hover3d?: boolean;
  glow?: 'primary' | 'secondary' | 'error';
  onClick?: () => void;
}

export default function GlassCard({ children, className = '', hover3d = false, glow, onClick }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!hover3d || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    ref.current.style.transform = `perspective(1000px) rotateY(${x * 6}deg) rotateX(${-y * 6}deg) scale(1.01)`;
  };

  const handleMouseLeave = () => {
    if (!hover3d || !ref.current) return;
    ref.current.style.transform = 'perspective(1000px) rotateY(0) rotateX(0) scale(1)';
  };

  const glowStyle = glow ? {
    boxShadow: glow === 'primary' ? 'var(--glow-primary)' :
               glow === 'secondary' ? 'var(--glow-secondary)' : 'var(--glow-error)'
  } : {};

  return (
    <motion.div
      ref={ref}
      className={`card ${className}`}
      style={{ ...glowStyle, transition: 'transform 0.15s ease-out, box-shadow 0.25s ease' }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {children}
    </motion.div>
  );
}
