import { useState, useEffect, useRef, useCallback } from 'react';
import type { Alert } from '../types';

// Derive WebSocket URL dynamically from window.location so it works on any host.
// Falls back to VITE_ALERT_WS_URL env var if set (e.g. for local dev).
function getWsUrl(): string {
  if (import.meta.env.VITE_ALERT_WS_URL) {
    return import.meta.env.VITE_ALERT_WS_URL as string;
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/alerts/ws`;
}

const RECONNECT_DELAY_MS = 3000;

function mapEnvelopeToAlert(data: Record<string, unknown>): Alert | null {
  try {
    const meta = (data.metadata ?? {}) as Record<string, unknown>;
    return {
      event_type: 'poi_match_alert',
      timestamp: (data.timestamp as string) ?? new Date().toISOString(),
      alert_id: (meta.alert_id as string) ?? `alert-${Date.now()}`,
      poi_id: (meta.poi_id as string) ?? '',
      severity: ((meta.severity as string) ?? 'low') as Alert['severity'],
      match: {
        camera_id: (meta.camera_id as string) ?? '',
        confidence: (meta.confidence as number) ?? 0,
        similarity_score: (meta.similarity_score as number) ?? 0,
        bbox: (meta.bbox as [number, number, number, number]) ?? [0, 0, 0, 0],
        frame_number: (meta.frame_number as number) ?? 0,
        thumbnail_path: (meta.thumbnail_path as string) ?? '',
      },
      poi_metadata: {
        name: (meta.name as string) ?? '',
        notes: (meta.notes as string) ?? '',
        enrollment_date: (meta.enrollment_date as string) ?? '',
        total_previous_matches: (meta.total_previous_matches as number) ?? 0,
      },
      status: 'New',
    };
  } catch {
    return null;
  }
}

export function useAlertWebSocket() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmounted = useRef(false);

  const connect = useCallback(() => {
    if (unmounted.current) return;

    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      if (!unmounted.current) setConnected(true);
    };

    ws.onmessage = (event) => {
      if (unmounted.current) return;
      try {
        const data = JSON.parse(event.data as string) as Record<string, unknown>;
        if (data.alert_type === 'POI_MATCH') {
          const alert = mapEnvelopeToAlert(data);
          if (alert) {
            setAlerts((prev) => {
              // Deduplicate by alert_id in case history already has it
              if (prev.some((a) => a.alert_id === alert.alert_id)) return prev;
              return [alert, ...prev].slice(0, 200);
            });
          }
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (!unmounted.current) {
        setConnected(false);
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    unmounted.current = false;
    connect();
    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { alerts, connected, setAlerts };
}
