const BASE = '';
export interface ApiResponse<T = unknown> {
  data: T;
  ok: boolean;
  error?: string;
}
async function request<T>(url: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  try {
    const res = await fetch(`${BASE}${url}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers as Record<string, string> },
      ...options,
    });
    const data = await res.json();
    if (!res.ok) return { data, ok: false, error: data?.error || res.statusText };
    return { data, ok: true };
  } catch (e) {
    return { data: null as T, ok: false, error: String(e) };
  }
}
export const api = {
  health: () => request<any>('/health'),
  listTasks: () => request<any>('/tasks'),
  getTask: (id: string) => request<any>(`/tasks/${id}`),
  listSessions: () => request<any>('/sessions'),
  deleteSession: (id: string) => request<any>(`/sessions/${id}`, { method: 'DELETE' }),
  reset: (taskId: string, sessionId = 'default', seed?: number) =>
    request<any>('/reset', {
      method: 'POST',
      body: JSON.stringify({ task_id: taskId, session_id: sessionId, seed }),
    }),
  step: (action: Record<string, unknown>, sessionId = 'default') =>
    request<any>('/step', {
      method: 'POST',
      body: JSON.stringify({ action, session_id: sessionId }),
    }),
  getState: (sessionId = 'default') =>
    request<any>(`/state?session_id=${sessionId}`),
  grade: (sessionId = 'default', forceComplete = false) =>
    request<any>('/grade', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, force_complete: forceComplete }),
    }),
  gradeBatch: (taskIds?: string[], maxSteps?: number) =>
    request<any>('/grade/batch', {
      method: 'POST',
      body: JSON.stringify({ task_ids: taskIds, max_steps_override: maxSteps }),
    }),
  metrics: () => request<any>('/metrics'),
  leaderboard: (taskId?: string) =>
    request<any>(`/leaderboard${taskId ? `?task_id=${taskId}` : ''}`),
  replay: (sessionId = 'default') =>
    request<any>(`/replay?session_id=${sessionId}&format=json`),
  ledger: (sessionId = 'default') =>
    request<any>(`/episode/ledger?session_id=${sessionId}`),
  scenarios: () => request<any>('/scenarios'),
  schema: () => request<any>('/docs/schema'),
  protocolRules: () => request<any>('/protocol/rules'),
  validate: () => request<any>('/validate', { method: 'POST' }),
};
