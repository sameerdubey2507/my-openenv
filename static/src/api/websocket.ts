type WSHandler = (data: any) => void;

export class WSManager {
  private ws: WebSocket | null = null;
  private handlers: Set<WSHandler> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private sessionId: string;
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  constructor(sessionId = 'default') {
    this.sessionId = sessionId;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/${this.sessionId}`;
    try {
      this.ws = new WebSocket(url);
      this.ws.onopen = () => {
        console.log('[WS] Connected:', this.sessionId);
        this.startPing();
      };
      this.ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          this.handlers.forEach((h) => h(data));
        } catch { }
      };
      this.ws.onclose = () => {
        console.log('[WS] Disconnected, reconnecting...');
        this.stopPing();
        this.scheduleReconnect();
      };
      this.ws.onerror = () => {
        this.ws?.close();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  private startPing() {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 25000);
  }

  private stopPing() {
    if (this.pingInterval) clearInterval(this.pingInterval);
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 3000);
  }

  subscribe(handler: WSHandler) {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  disconnect() {
    this.stopPing();
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
