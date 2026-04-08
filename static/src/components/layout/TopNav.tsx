import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Radio, Activity, Zap, GitBranch, HeartPulse, Database, BarChart3, Settings, Bell, Moon, BookOpen, User, LogOut, Eye, EyeOff, Shield, Check, X, Volume2, Monitor, Globe, Clock } from 'lucide-react';
import StatusChip from '../shared/StatusChip';

export type TabId = 'command' | 'grid' | 'dispatch' | 'trauma' | 'bio' | 'analytics' | 'theory';

const DOCS_URL = `${window.location.protocol}//${window.location.hostname}:7860/docs`;

const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'command', label: 'Command Center', icon: <Radio size={16} /> },
  { id: 'grid', label: 'Grid Routing', icon: <GitBranch size={16} /> },
  { id: 'dispatch', label: 'Rapid Dispatch', icon: <Zap size={16} /> },
  { id: 'trauma', label: 'Trauma Flow', icon: <HeartPulse size={16} /> },
  { id: 'bio', label: 'Bio-Archives', icon: <Database size={16} /> },
  { id: 'analytics', label: 'System Analytics', icon: <BarChart3 size={16} /> },
  { id: 'theory', label: 'Project Theory', icon: <BookOpen size={16} /> },
];

// Notification data
const INITIAL_NOTIFICATIONS = [
  { id: 1, type: 'alert', title: 'P1 Incident — Zone 3', body: 'Cardiac arrest reported at Kurla West. ALS-07 dispatched.', time: '2m ago', read: false },
  { id: 2, type: 'system', title: 'Fleet Rebalancing Complete', body: 'Optimal redistribution achieved across 12 zones.', time: '8m ago', read: false },
  { id: 3, type: 'alert', title: 'Hospital H-NOBLE at 92% Capacity', body: 'Diversion protocol activated. Redirecting to H-KEM.', time: '15m ago', read: false },
  { id: 4, type: 'info', title: 'RL Agent Checkpoint Saved', body: 'Model v3.7.2 — reward +8,452.92 — accuracy 94.1%.', time: '22m ago', read: true },
  { id: 5, type: 'system', title: 'WebSocket Reconnected', body: 'Connection to ws://0.0.0.0:7860/ws restored after 1.2s.', time: '31m ago', read: true },
  { id: 6, type: 'alert', title: 'Ambulance AMB-12 Fatigue Warning', body: 'Driver R. Sharma exceeding 6hr continuous shift.', time: '45m ago', read: true },
];

interface Props {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  simActive: boolean;
  fleetStatus: string;
  nodesCount: string;
  onEmergency: () => void;
  onSettingsClick: () => void;
}

// Dropdown panel styles
const dropdownStyle: React.CSSProperties = {
  position: 'absolute', top: '100%', right: 0, marginTop: 8,
  width: 380, maxHeight: 480,
  background: 'var(--bg-surface-high, #0d1117)', backdropFilter: 'blur(20px)',
  border: '1px solid var(--ghost-border, rgba(120,130,160,0.15))',
  borderRadius: 'var(--radius-lg, 12px)',
  boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
  zIndex: 9999, overflow: 'hidden',
};

export default function TopNav({ activeTab, onTabChange, simActive, fleetStatus, nodesCount, onEmergency, onSettingsClick }: Props) {
  const [notifOpen, setNotifOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [notifications, setNotifications] = useState(INITIAL_NOTIFICATIONS);

  // Profile / Registration state
  const [isRegistered, setIsRegistered] = useState(() => !!localStorage.getItem('emergi_user'));
  const [profileData, setProfileData] = useState(() => {
    const saved = localStorage.getItem('emergi_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [regUsername, setRegUsername] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regName, setRegName] = useState('');
  const [regRole, setRegRole] = useState('Operator');
  const [showRegPass, setShowRegPass] = useState(false);
  const [regError, setRegError] = useState('');
  const [regSuccess, setRegSuccess] = useState(false);

  // Settings state
  const [darkMode, setDarkMode] = useState(true);
  const [soundAlerts, setSoundAlerts] = useState(true);
  const [autoDispatch, setAutoDispatch] = useState(false);
  const [liveTracking, setLiveTracking] = useState(true);
  const [highContrast, setHighContrast] = useState(false);
  const [language, setLanguage] = useState('English');

  const notifRef = useRef<HTMLDivElement>(null);
  const settRef = useRef<HTMLDivElement>(null);
  const profRef = useRef<HTMLDivElement>(null);

  const unreadCount = notifications.filter(n => !n.read).length;

  // Close all when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) setNotifOpen(false);
      if (settRef.current && !settRef.current.contains(e.target as Node)) setSettingsOpen(false);
      if (profRef.current && !profRef.current.contains(e.target as Node)) setProfileOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const togglePanel = (panel: 'notif' | 'settings' | 'profile') => {
    setNotifOpen(panel === 'notif' ? !notifOpen : false);
    setSettingsOpen(panel === 'settings' ? !settingsOpen : false);
    setProfileOpen(panel === 'profile' ? !profileOpen : false);
  };

  const markAllRead = () => setNotifications(ns => ns.map(n => ({ ...n, read: true })));
  const dismissNotif = (id: number) => setNotifications(ns => ns.filter(n => n.id !== id));

  const handleRegister = () => {
    if (!regUsername.trim() || !regPassword.trim()) { setRegError('Username and password are required.'); return; }
    if (regPassword.length < 4) { setRegError('Password must be at least 4 characters.'); return; }
    const userData = {
      username: regUsername, name: regName || regUsername,
      role: regRole, registeredAt: new Date().toISOString(),
      clearance: regRole === 'Admin' ? 'LEVEL-5' : regRole === 'Supervisor' ? 'LEVEL-3' : 'LEVEL-1',
    };
    localStorage.setItem('emergi_user', JSON.stringify(userData));
    setProfileData(userData); setIsRegistered(true); setRegSuccess(true); setRegError('');
    setTimeout(() => setRegSuccess(false), 2000);
  };

  const handleLogout = () => {
    localStorage.removeItem('emergi_user');
    setProfileData(null); setIsRegistered(false);
    setRegUsername(''); setRegPassword(''); setRegName(''); setRegRole('Operator');
    setProfileOpen(false);
  };

  return (
    <nav style={{
      background: 'var(--bg-surface-low)',
      borderBottom: '1px solid var(--ghost-border)',
      position: 'sticky', top: 0, zIndex: 'var(--z-nav)' as any,
      backdropFilter: 'blur(20px)',
    }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: 'var(--space-2) var(--space-6)',
        borderBottom: '1px solid rgba(64,71,82,0.08)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <Activity size={20} style={{ color: 'var(--primary)' }} />
            <span style={{ fontWeight: 800, fontSize: '1rem', letterSpacing: '-0.02em' }}>EMERGI-ENV</span>
            <span style={{ fontSize: '0.6rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)' }}>COMMAND NUCLEUS</span>
          </div>
          <div className="hide-mobile" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
            <StatusChip variant={simActive ? 'active' : 'delta'} pulse={simActive}>
              SIMULATION: {simActive ? 'ACTIVE' : 'IDLE'}
            </StatusChip>
            <span className="text-label" style={{ fontSize: '0.7rem' }}>FLEET: {fleetStatus}</span>
            <span className="text-label" style={{ fontSize: '0.7rem' }}>NODES: {nodesCount}</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <motion.button
            className="btn btn-danger btn-sm"
            onClick={onEmergency}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{ fontWeight: 700 }}
          >
          <Zap size={14} /> NEW EMERGENCY
          </motion.button>
          <motion.button
            className="btn btn-sm"
            onClick={() => window.location.href = '/login'}
            whileHover={{ scale: 1.06, boxShadow: '0 0 28px rgba(124,58,237,0.8)' }}
            whileTap={{ scale: 0.95 }}
            animate={{ boxShadow: ['0 0 10px rgba(124,58,237,0.4)', '0 0 22px rgba(56,189,248,0.6)', '0 0 10px rgba(124,58,237,0.4)'] }}
            transition={{ boxShadow: { repeat: Infinity, duration: 2.2, ease: 'easeInOut' } }}
            style={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #7c3aed, #0ea5e9, #7c3aed)',
              backgroundSize: '200% 200%',
              color: '#fff', border: '1px solid rgba(124,58,237,0.5)',
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '6px 14px', borderRadius: '8px', cursor: 'pointer',
              fontSize: '0.8rem', letterSpacing: '0.04em',
              position: 'relative', overflow: 'hidden',
            }}
          >
            <style>{`
              @keyframes login-btn-shimmer { 0%{left:-80%} 100%{left:180%} }
              @keyframes login-btn-grad { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
              .login-btn-inner::before { content:''; position:absolute; top:0; left:-80%; width:50%; height:100%;
                background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25),transparent);
                animation:login-btn-shimmer 2.6s ease-in-out infinite; pointer-events:none; }
            `}
            </style>
            🔐 LOGIN
          </motion.button>
          <motion.button
            className="btn btn-sm"
            onClick={() => window.location.href = DOCS_URL}
            whileHover={{ scale: 1.07, boxShadow: '0 0 24px rgba(139,92,246,0.55)' }}
            whileTap={{ scale: 0.95 }}
            animate={{ boxShadow: ['0 0 8px rgba(139,92,246,0.3)', '0 0 20px rgba(59,130,246,0.5)', '0 0 8px rgba(139,92,246,0.3)'] }}
            transition={{ boxShadow: { repeat: Infinity, duration: 2, ease: 'easeInOut' } }}
            style={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4)',
              backgroundSize: '200% 200%',
              color: '#fff', border: 'none',
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '6px 14px', borderRadius: '8px', cursor: 'pointer',
              fontSize: '0.8rem', letterSpacing: '0.02em',
            }}
          >
            <BookOpen size={14} /> Functions
          </motion.button>

          {/* ═══ NOTIFICATION BELL ═══ */}
          <div ref={notifRef} style={{ position: 'relative' }}>
            <button className="btn btn-ghost btn-sm" onClick={() => togglePanel('notif')}
              style={{ padding: 'var(--space-2)', position: 'relative' }}>
              <Bell size={16} />
              {unreadCount > 0 && (
                <span style={{
                  position: 'absolute', top: 2, right: 2, width: 16, height: 16, borderRadius: '50%',
                  background: '#FF544C', fontSize: '0.55rem', fontWeight: 800, color: '#fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  boxShadow: '0 0 8px #FF544C', animation: 'pulse-glow 2s infinite',
                }}>{unreadCount}</span>
              )}
            </button>
            <AnimatePresence>
              {notifOpen && (
                <motion.div initial={{ opacity: 0, y: -10, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.2 }} style={dropdownStyle}>
                  <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(120,130,160,0.12)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Notifications</span>
                    <button onClick={markAllRead} style={{ background: 'none', border: 'none', color: 'var(--primary, #A2C9FF)', fontSize: '0.7rem', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>Mark all read</button>
                  </div>
                  <div style={{ maxHeight: 380, overflowY: 'auto' }}>
                    {notifications.map(n => (
                      <motion.div key={n.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, height: 0 }}
                        style={{ padding: '12px 20px', borderBottom: '1px solid rgba(120,130,160,0.06)', display: 'flex', gap: 12, alignItems: 'flex-start',
                          background: n.read ? 'transparent' : 'rgba(162,201,255,0.04)', cursor: 'pointer', transition: 'background 0.2s' }}
                        onClick={() => setNotifications(ns => ns.map(x => x.id === n.id ? { ...x, read: true } : x))}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(162,201,255,0.06)')}
                        onMouseLeave={e => (e.currentTarget.style.background = n.read ? 'transparent' : 'rgba(162,201,255,0.04)')}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', marginTop: 6, flexShrink: 0,
                          background: n.type === 'alert' ? '#FF544C' : n.type === 'system' ? '#FFB74D' : '#7DDC7A',
                          boxShadow: n.read ? 'none' : `0 0 8px ${n.type === 'alert' ? '#FF544C' : n.type === 'system' ? '#FFB74D' : '#7DDC7A'}` }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontWeight: 600, fontSize: '0.8rem', marginBottom: 2 }}>{n.title}</div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--on-surface-variant, #7882A0)', lineHeight: 1.4 }}>{n.body}</div>
                          <div style={{ fontSize: '0.6rem', color: 'var(--on-surface-dim, #555)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>{n.time}</div>
                        </div>
                        <button onClick={e => { e.stopPropagation(); dismissNotif(n.id); }}
                          style={{ background: 'none', border: 'none', color: 'var(--on-surface-dim, #555)', cursor: 'pointer', padding: 2, flexShrink: 0 }}>
                          <X size={12} />
                        </button>
                      </motion.div>
                    ))}
                    {notifications.length === 0 && (
                      <div style={{ padding: 40, textAlign: 'center', color: 'var(--on-surface-variant)', fontSize: '0.8rem' }}>No notifications</div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ═══ SETTINGS ═══ */}
          <div ref={settRef} style={{ position: 'relative' }}>
            <button className="btn btn-ghost btn-sm" onClick={() => togglePanel('settings')} style={{ padding: 'var(--space-2)' }}>
              <Settings size={16} />
            </button>
            <AnimatePresence>
              {settingsOpen && (
                <motion.div initial={{ opacity: 0, y: -10, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.2 }} style={dropdownStyle}>
                  <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(120,130,160,0.12)' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>System Settings</span>
                  </div>
                  <div style={{ padding: '8px 20px' }}>
                    {[
                      { icon: <Moon size={15} />, label: 'Dark Mode', desc: 'Visual theme for low-light ops', val: darkMode, set: setDarkMode },
                      { icon: <Volume2 size={15} />, label: 'Sound Alerts', desc: 'Audible P1 incident sirens', val: soundAlerts, set: setSoundAlerts },
                      { icon: <Zap size={15} />, label: 'Auto-Dispatch', desc: 'Let RL agent auto-assign units', val: autoDispatch, set: setAutoDispatch },
                      { icon: <Monitor size={15} />, label: 'Live GPS Tracking', desc: 'Real-time fleet position updates', val: liveTracking, set: setLiveTracking },
                      { icon: <Eye size={15} />, label: 'High Contrast', desc: 'Enhanced visibility for accessibility', val: highContrast, set: setHighContrast },
                    ].map(s => (
                      <div key={s.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid rgba(120,130,160,0.06)' }}>
                        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                          <span style={{ color: 'var(--primary, #A2C9FF)' }}>{s.icon}</span>
                          <div>
                            <div style={{ fontSize: '0.82rem', fontWeight: 600 }}>{s.label}</div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant, #7882A0)' }}>{s.desc}</div>
                          </div>
                        </div>
                        <div onClick={() => s.set(!s.val)} style={{
                          width: 38, height: 20, borderRadius: 10, cursor: 'pointer', transition: '0.3s',
                          background: s.val ? 'rgba(125,220,122,0.3)' : 'rgba(120,130,160,0.2)',
                          border: `1px solid ${s.val ? '#7DDC7A' : 'rgba(120,130,160,0.3)'}`,
                          position: 'relative',
                        }}>
                          <div style={{
                            width: 14, height: 14, borderRadius: '50%', position: 'absolute', top: 2,
                            left: s.val ? 20 : 2, transition: '0.3s',
                            background: s.val ? '#7DDC7A' : '#7882A0',
                            boxShadow: s.val ? '0 0 8px #7DDC7A' : 'none',
                          }} />
                        </div>
                      </div>
                    ))}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0' }}>
                      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                        <Globe size={15} style={{ color: 'var(--primary, #A2C9FF)' }} />
                        <div>
                          <div style={{ fontSize: '0.82rem', fontWeight: 600 }}>Language</div>
                          <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant, #7882A0)' }}>Interface language</div>
                        </div>
                      </div>
                      <select value={language} onChange={e => setLanguage(e.target.value)} style={{
                        background: 'var(--bg-surface, #0d1117)', border: '1px solid rgba(120,130,160,0.2)',
                        borderRadius: 6, padding: '4px 8px', color: 'var(--on-surface, #e0e0ff)',
                        fontSize: '0.72rem', fontFamily: 'var(--font-mono)', cursor: 'pointer',
                      }}>
                        {['English', 'Hindi', 'Marathi'].map(l => <option key={l} value={l}>{l}</option>)}
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ═══ PROFILE AVATAR ═══ */}
          <div ref={profRef} style={{ position: 'relative' }}>
            <div onClick={() => togglePanel('profile')} style={{
              width: 32, height: 32, borderRadius: '50%',
              background: isRegistered ? 'linear-gradient(135deg, #7DDC7A, #06b6d4)' : 'var(--primary-gradient, linear-gradient(135deg, #A2C9FF, #7882A0))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.75rem', fontWeight: 700, color: '#0d1117',
              marginLeft: 'var(--space-2)', cursor: 'pointer', transition: '0.2s',
              boxShadow: isRegistered ? '0 0 12px rgba(125,220,122,0.3)' : 'none',
            }}>
              {isRegistered ? profileData?.name?.charAt(0)?.toUpperCase() || 'U' : 'CD'}
            </div>
            <AnimatePresence>
              {profileOpen && (
                <motion.div initial={{ opacity: 0, y: -10, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.2 }} style={{ ...dropdownStyle, width: 360 }}>
                  {isRegistered && profileData ? (
                    /* ─── PROFILE VIEW ─── */
                    <div>
                      <div style={{ padding: '24px 20px', textAlign: 'center', borderBottom: '1px solid rgba(120,130,160,0.12)', background: 'rgba(125,220,122,0.04)' }}>
                        <div style={{
                          width: 56, height: 56, borderRadius: '50%', margin: '0 auto 12px',
                          background: 'linear-gradient(135deg, #7DDC7A, #06b6d4)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: '1.4rem', fontWeight: 800, color: '#0d1117',
                          boxShadow: '0 0 20px rgba(125,220,122,0.3)',
                        }}>{profileData.name?.charAt(0)?.toUpperCase()}</div>
                        <div style={{ fontWeight: 700, fontSize: '1rem' }}>{profileData.name}</div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)' }}>@{profileData.username}</div>
                      </div>
                      <div style={{ padding: '16px 20px' }}>
                        {[
                          { label: 'Role', value: profileData.role },
                          { label: 'Clearance', value: profileData.clearance },
                          { label: 'Registered', value: new Date(profileData.registeredAt).toLocaleDateString() },
                          { label: 'Status', value: 'ACTIVE', color: '#7DDC7A' },
                          { label: 'Session', value: `${Math.floor(Math.random() * 3) + 1}h ${Math.floor(Math.random() * 59)}m`, color: 'var(--primary)' },
                        ].map(r => (
                          <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(120,130,160,0.05)', fontSize: '0.8rem' }}>
                            <span style={{ color: 'var(--on-surface-variant, #7882A0)' }}>{r.label}</span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: r.color || 'var(--on-surface)' }}>{r.value}</span>
                          </div>
                        ))}
                      </div>
                      <div style={{ padding: '12px 20px', borderTop: '1px solid rgba(120,130,160,0.12)' }}>
                        <button onClick={handleLogout} style={{
                          width: '100%', padding: '10px', background: 'rgba(255,84,76,0.1)', border: '1px solid rgba(255,84,76,0.3)',
                          borderRadius: 8, color: '#FF544C', fontWeight: 700, fontSize: '0.8rem', cursor: 'pointer',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: '0.2s',
                        }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,84,76,0.2)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'rgba(255,84,76,0.1)')}>
                          <LogOut size={14} /> Sign Out
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* ─── REGISTRATION VIEW ─── */
                    <div>
                      <div style={{ padding: '20px 20px 16px', textAlign: 'center', borderBottom: '1px solid rgba(120,130,160,0.12)' }}>
                        <Shield size={28} style={{ color: 'var(--primary, #A2C9FF)', marginBottom: 8 }} />
                        <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>Operator Registration</div>
                        <div style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>Create your profile to access all features</div>
                      </div>
                      <div style={{ padding: '16px 20px' }}>
                        {regSuccess && (
                          <motion.div initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }}
                            style={{ background: 'rgba(125,220,122,0.15)', color: '#7DDC7A', padding: 10, borderRadius: 8, fontSize: '0.75rem', textAlign: 'center', marginBottom: 12, fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                            <Check size={14} /> REGISTRATION SUCCESSFUL
                          </motion.div>
                        )}
                        {regError && (
                          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                            style={{ background: 'rgba(255,84,76,0.12)', color: '#FF544C', padding: 8, borderRadius: 8, fontSize: '0.7rem', textAlign: 'center', marginBottom: 12, fontFamily: 'var(--font-mono)' }}>
                            {regError}
                          </motion.div>
                        )}
                        <div style={{ marginBottom: 10 }}>
                          <label style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4, display: 'block' }}>Display Name</label>
                          <input value={regName} onChange={e => setRegName(e.target.value)} placeholder="e.g. Dr. Sharma"
                            style={{ width: '100%', padding: '10px 12px', background: 'var(--bg-surface, rgba(0,0,0,0.4))', border: '1px solid rgba(120,130,160,0.2)', borderRadius: 8, color: 'var(--on-surface, #e0e0ff)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', outline: 'none', boxSizing: 'border-box' }}
                            onFocus={e => e.target.style.borderColor = '#A2C9FF'} onBlur={e => e.target.style.borderColor = 'rgba(120,130,160,0.2)'} />
                        </div>
                        <div style={{ marginBottom: 10 }}>
                          <label style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4, display: 'block' }}>Username</label>
                          <input value={regUsername} onChange={e => setRegUsername(e.target.value)} placeholder="e.g. op_alpha"
                            style={{ width: '100%', padding: '10px 12px', background: 'var(--bg-surface, rgba(0,0,0,0.4))', border: '1px solid rgba(120,130,160,0.2)', borderRadius: 8, color: 'var(--on-surface, #e0e0ff)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', outline: 'none', boxSizing: 'border-box' }}
                            onFocus={e => e.target.style.borderColor = '#A2C9FF'} onBlur={e => e.target.style.borderColor = 'rgba(120,130,160,0.2)'} />
                        </div>
                        <div style={{ marginBottom: 10, position: 'relative' }}>
                          <label style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4, display: 'block' }}>Password</label>
                          <input value={regPassword} onChange={e => setRegPassword(e.target.value)} type={showRegPass ? 'text' : 'password'} placeholder="••••••"
                            style={{ width: '100%', padding: '10px 12px', background: 'var(--bg-surface, rgba(0,0,0,0.4))', border: '1px solid rgba(120,130,160,0.2)', borderRadius: 8, color: 'var(--on-surface, #e0e0ff)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', outline: 'none', boxSizing: 'border-box' }}
                            onFocus={e => e.target.style.borderColor = '#A2C9FF'} onBlur={e => e.target.style.borderColor = 'rgba(120,130,160,0.2)'}
                            onKeyDown={e => e.key === 'Enter' && handleRegister()} />
                          <button onClick={() => setShowRegPass(!showRegPass)} style={{ position: 'absolute', right: 10, bottom: 10, background: 'none', border: 'none', color: 'var(--on-surface-variant)', cursor: 'pointer' }}>
                            {showRegPass ? <EyeOff size={14} /> : <Eye size={14} />}
                          </button>
                        </div>
                        <div style={{ marginBottom: 14 }}>
                          <label style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4, display: 'block' }}>Role</label>
                          <select value={regRole} onChange={e => setRegRole(e.target.value)} style={{
                            width: '100%', padding: '10px 12px', background: 'var(--bg-surface, rgba(0,0,0,0.4))',
                            border: '1px solid rgba(120,130,160,0.2)', borderRadius: 8,
                            color: 'var(--on-surface, #e0e0ff)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', cursor: 'pointer',
                          }}>
                            {['Operator', 'Supervisor', 'Admin', 'Paramedic', 'Analyst'].map(r => <option key={r} value={r}>{r}</option>)}
                          </select>
                        </div>
                        <button onClick={handleRegister} style={{
                          width: '100%', padding: '12px', borderRadius: 8, border: 'none', cursor: 'pointer',
                          background: 'linear-gradient(135deg, #A2C9FF, #06b6d4)',
                          color: '#0d1117', fontWeight: 700, fontSize: '0.85rem',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                          transition: 'transform 0.2s, box-shadow 0.2s',
                        }}
                        onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 6px 20px rgba(162,201,255,0.3)'; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none'; }}>
                          <User size={14} /> Register & Activate Profile
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Tab bar */}
      <div style={{
        display: 'flex', gap: 'var(--space-1)',
        padding: '0 var(--space-6)',
        overflowX: 'auto', scrollbarWidth: 'none',
      }}>
        {tabs.map(tab => (
          <motion.button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 'var(--space-2)',
              padding: 'var(--space-3) var(--space-4)',
              background: 'transparent', border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid var(--primary)' : '2px solid transparent',
              color: activeTab === tab.id ? 'var(--primary)' : 'var(--on-surface-variant)',
              fontFamily: 'var(--font-primary)', fontSize: '0.8125rem',
              fontWeight: activeTab === tab.id ? 600 : 400,
              cursor: 'pointer', whiteSpace: 'nowrap', transition: 'all 0.2s ease',
            }}
            whileHover={{ color: 'var(--primary)', backgroundColor: 'rgba(162,201,255,0.05)' }}
            whileTap={{ scale: 0.97 }}
          >
            {tab.icon}
            {tab.label}
          </motion.button>
        ))}
      </div>
    </nav>
  );
}
