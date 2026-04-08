import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../api/client';
import { WSManager } from '../api/websocket';
import { toast } from 'react-hot-toast';

export interface SimState {
  sessionId: string;
  taskId: string;
  episodeId: string;
  step: number;
  maxSteps: number;
  done: boolean;
  reward: number;
  queueLength: number;
  activePatients: number;
  resolvedPatients: number;
  simClockMin: number;
  observation: any;
  loading: boolean;
  connected: boolean;
  lastEvent: any;
}

const defaultState: SimState = {
  sessionId: 'default',
  taskId: '',
  episodeId: '',
  step: 0,
  maxSteps: 50,
  done: false,
  reward: 0,
  queueLength: 0,
  activePatients: 0,
  resolvedPatients: 0,
  simClockMin: 0,
  observation: null,
  loading: false,
  connected: false,
  lastEvent: null,
};

export function useSimulation(sessionId = 'default') {
  const [state, setState] = useState<SimState>({ ...defaultState, sessionId });
  const [tasks, setTasks] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [leaderboard, setLeaderboard] = useState<any>(null);
  const [scenarios, setScenarios] = useState<any[]>([]);
  const [gradeResult, setGradeResult] = useState<any>(null);
  const [protocolRules, setProtocolRules] = useState<any[]>([]);
  const [schema, setSchema] = useState<any>(null);
  const [ledger, setLedger] = useState<any>(null);
  const [actionLog, setActionLog] = useState<any[]>([]);
  const wsRef = useRef<WSManager | null>(null);

  useEffect(() => {
    const ws = new WSManager(sessionId);
    wsRef.current = ws;
    ws.connect();
    const unsub = ws.subscribe((data) => {
      setState(prev => ({ ...prev, lastEvent: data, connected: true }));
      if (data.event === 'step') {
        setState(prev => ({
          ...prev,
          step: data.step,
          done: data.done,
          reward: prev.reward + (data.reward || 0),
          queueLength: data.queue_len ?? prev.queueLength,
          activePatients: data.active ?? prev.activePatients,
        }));
      }
      if (data.event === 'graded') {
        setGradeResult(data);
      }
    });
    return () => { unsub(); ws.disconnect(); };
  }, [sessionId]);

  useEffect(() => {
    loadHealth();
    loadTasks();
    loadMetrics();
    loadLeaderboard();
    loadScenarios();
    loadProtocolRules();
    loadSchema();
    loadState();
  }, []);

  const loadHealth = useCallback(async () => {
    const res = await api.health();
    if (res.ok) setHealth(res.data);
  }, []);

  const loadTasks = useCallback(async () => {
    const res = await api.listTasks();
    if (res.ok) setTasks(res.data.tasks || []);
  }, []);

  const loadMetrics = useCallback(async () => {
    const res = await api.metrics();
    if (res.ok) setMetrics(res.data);
  }, []);

  const loadLeaderboard = useCallback(async () => {
    const res = await api.leaderboard();
    if (res.ok) setLeaderboard(res.data);
  }, []);

  const loadScenarios = useCallback(async () => {
    const res = await api.scenarios();
    if (res.ok) setScenarios(res.data.scenarios || []);
  }, []);

  const loadProtocolRules = useCallback(async () => {
    const res = await api.protocolRules();
    if (res.ok) setProtocolRules(res.data.protocol_rules || []);
  }, []);

  const loadSchema = useCallback(async () => {
    const res = await api.schema();
    if (res.ok) setSchema(res.data);
  }, []);

  const loadState = useCallback(async () => {
    const res = await api.getState(sessionId);
    if (res.ok && res.data?.state) {
      const s = res.data.state;
      setState(prev => ({
        ...prev,
        episodeId: res.data.episode_id || prev.episodeId,
        observation: s,
        queueLength: s.queue_length ?? s.incident_queue?.length ?? prev.queueLength,
        activePatients: s.active_patients ?? prev.activePatients,
        resolvedPatients: s.resolved_patients ?? prev.resolvedPatients,
        simClockMin: s.sim_clock_min ?? prev.simClockMin,
        step: s.step ?? prev.step,
      }));
    }
  }, [sessionId]);

  const loadLedger = useCallback(async () => {
    const res = await api.ledger(sessionId);
    if (res.ok) setLedger(res.data.ledger);
  }, [sessionId]);

  const resetEpisode = useCallback(async (taskId: string, seed?: number) => {
    setState(prev => ({ ...prev, loading: true }));
    const res = await api.reset(taskId, sessionId, seed);
    if (res.ok) {
      toast.success(`Simulation Reset: ${taskId}`);
      setState(prev => ({
        ...prev,
        loading: false,
        taskId: res.data.task_id,
        episodeId: res.data.episode_id,
        step: 0,
        maxSteps: res.data.max_steps,
        done: false,
        reward: 0,
        observation: res.data.observation,
        queueLength: res.data.observation?.queue_length ?? 0,
        activePatients: res.data.observation?.active_patients ?? 0,
        resolvedPatients: 0,
        simClockMin: 0,
      }));
      setGradeResult(null);
      setActionLog([]);
      return res.data;
    }
    setState(prev => ({ ...prev, loading: false }));
    return null;
  }, [sessionId]);

  const stepAction = useCallback(async (action: Record<string, unknown>) => {
    setState(prev => ({ ...prev, loading: true }));
    const res = await api.step(action, sessionId);
    if (res.ok) {
      if (action.action_type !== 'noop') {
        toast('Action executed: ' + action.action_type, { icon: '⚡' });
      }
      setActionLog(prev => [...prev, { action, result: res.data, time: new Date().toISOString() }]);
      setState(prev => ({
        ...prev,
        loading: false,
        step: res.data.step,
        done: res.data.done,
        reward: prev.reward + (res.data.reward || 0),
        observation: res.data.observation,
        simClockMin: res.data.sim_clock_min ?? prev.simClockMin,
        queueLength: res.data.observation?.queue_length ?? prev.queueLength,
        activePatients: res.data.observation?.active_patients ?? prev.activePatients,
        resolvedPatients: res.data.observation?.resolved_patients ?? prev.resolvedPatients,
      }));
      return res.data;
    }
    setState(prev => ({ ...prev, loading: false }));
    return null;
  }, [sessionId]);

  const gradeEpisode = useCallback(async (forceComplete = false) => {
    setState(prev => ({ ...prev, loading: true }));
    const res = await api.grade(sessionId, forceComplete);
    setState(prev => ({ ...prev, loading: false }));
    if (res.ok) {
      toast.success('Episode Graded! Final Score: ' + ((res.data.final_score ?? res.data.score ?? 0) * 100).toFixed(1) + '%');
      setGradeResult(res.data);
      return res.data;
    }
    toast.error('Failed to grade episode');
    return null;
  }, [sessionId]);

  return {
    state,
    tasks,
    health,
    metrics,
    leaderboard,
    scenarios,
    gradeResult,
    protocolRules,
    schema,
    ledger,
    actionLog,
    resetEpisode,
    stepAction,
    gradeEpisode,
    loadState,
    loadMetrics,
    loadLeaderboard,
    loadLedger,
    loadHealth,
    loadTasks,
  };
}
