import { motion } from 'framer-motion';

interface Props {
  children: React.ReactNode;
  variant?: 'alpha' | 'beta' | 'gamma' | 'delta' | 'active';
  pulse?: boolean;
  className?: string;
}

export default function StatusChip({ children, variant = 'delta', pulse = false, className = '' }: Props) {
  return (
    <motion.span
      className={`chip chip-${variant} ${className}`}
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.2 }}
    >
      {pulse && <span className={variant === 'alpha' ? 'live-dot-red' : 'live-dot'} />}
      {children}
    </motion.span>
  );
}
