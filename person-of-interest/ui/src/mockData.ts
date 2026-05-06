import type { POI, Alert, TrackingUpdate, HistoryResult } from './types';

/* ── POI Enrollments (multi-image) ─────────────────────── */

export const mockPOIs: POI[] = [
  {
    event_type: 'poi_enrollment',
    timestamp: '2026-02-18T09:15:00Z',
    poi_id: 'poi-001',
    enrolled_by: 'operator-badge-1234',
    severity: 'high',
    notes: 'Repeat offender, last seen 2026-02-10',
    reference_images: [
      { source: 'uploaded_image', embedding_id: 'emb-poi001-ref-01', vector_dim: 256, image_path: '/images/poi1.png' },
      { source: 'uploaded_image', embedding_id: 'emb-poi001-ref-02', vector_dim: 256, image_path: '/images/poi2.png' },
    ],
    status: 'active',
  },
  {
    event_type: 'poi_enrollment',
    timestamp: '2026-02-17T14:30:00Z',
    poi_id: 'poi-002',
    enrolled_by: 'operator-badge-5678',
    severity: 'medium',
    notes: 'Suspected of concealment in electronics aisle',
    reference_images: [
      { source: 'uploaded_image', embedding_id: 'emb-poi002-ref-01', vector_dim: 256, image_path: '/images/poi3.png' },
    ],
    status: 'active',
  },
  {
    event_type: 'poi_enrollment',
    timestamp: '2026-02-16T08:00:00Z',
    poi_id: 'poi-003',
    enrolled_by: 'operator-badge-1234',
    severity: 'high',
    notes: 'Known associate of poi-001',
    reference_images: [
      { source: 'uploaded_image', embedding_id: 'emb-poi003-ref-01', vector_dim: 256, image_path: '/images/poi4.png' },
      { source: 'uploaded_image', embedding_id: 'emb-poi003-ref-02', vector_dim: 256, image_path: '/images/poi1.png' },
      { source: 'uploaded_image', embedding_id: 'emb-poi003-ref-03', vector_dim: 256, image_path: '/images/poi3.png' },
    ],
    status: 'active',
  },
  {
    event_type: 'poi_enrollment',
    timestamp: '2026-02-15T10:00:00Z',
    poi_id: 'poi-004',
    enrolled_by: 'operator-badge-9012',
    severity: 'low',
    notes: 'One-time incident, monitoring only',
    reference_images: [
      { source: 'uploaded_image', embedding_id: 'emb-poi004-ref-01', vector_dim: 256, image_path: '/images/poi2.png' },
    ],
    status: 'inactive',
  },
];

/* ── POI Match Alerts ──────────────────────────────────── */

export const mockAlerts: Alert[] = [
  {
    event_type: 'poi_match_alert',
    timestamp: '2026-02-18T14:32:01.456Z',
    alert_id: 'alert-20260218-143201-poi001',
    poi_id: 'poi-001',
    severity: 'high',
    match: {
      camera_id: 'cam-01-entrance',
      confidence: 0.89,
      similarity_score: 0.91,
      bbox: [150, 60, 310, 480],
      frame_number: 12456,
      thumbnail_path: '/images/poi1.png',
    },
    poi_metadata: {
      name: "",
      notes: 'Repeat offender, last seen 2026-02-10',
      enrollment_date: '2026-02-15T10:00:00Z',
      total_previous_matches: 3,
    },
    status: 'New',
  },
  {
    event_type: 'poi_match_alert',
    timestamp: '2026-02-18T13:10:45.123Z',
    alert_id: 'alert-20260218-131045-poi002',
    poi_id: 'poi-002',
    severity: 'medium',
    match: {
      camera_id: 'cam-07-electronics',
      confidence: 0.82,
      similarity_score: 0.85,
      bbox: [200, 80, 340, 500],
      frame_number: 9832,
      thumbnail_path: '/images/poi3.png',
    },
    poi_metadata: {
      name: "",
      notes: 'Suspected of concealment in electronics aisle',
      enrollment_date: '2026-02-17T14:30:00Z',
      total_previous_matches: 1,
    },
    status: 'New',
  },
  {
    event_type: 'poi_match_alert',
    timestamp: '2026-02-18T11:05:22.789Z',
    alert_id: 'alert-20260218-110522-poi001',
    poi_id: 'poi-001',
    severity: 'high',
    match: {
      camera_id: 'cam-03-front-aisle',
      confidence: 0.91,
      similarity_score: 0.93,
      bbox: [120, 50, 290, 470],
      frame_number: 7654,
      thumbnail_path: '/images/poi2.png',
    },
    poi_metadata: {
      name: "",
      notes: 'Repeat offender, last seen 2026-02-10',
      enrollment_date: '2026-02-15T10:00:00Z',
      total_previous_matches: 2,
    },
    status: 'Acknowledged',
  },
  {
    event_type: 'poi_match_alert',
    timestamp: '2026-02-17T18:50:00.000Z',
    alert_id: 'alert-20260217-185000-poi003',
    poi_id: 'poi-003',
    severity: 'high',
    match: {
      camera_id: 'cam-12-checkout',
      confidence: 0.87,
      similarity_score: 0.88,
      bbox: [180, 70, 330, 490],
      frame_number: 15210,
      thumbnail_path: '/images/poi4.png',
    },
    poi_metadata: {
      name: "",
      notes: 'Known associate of poi-001',
      enrollment_date: '2026-02-16T08:00:00Z',
      total_previous_matches: 0,
    },
    status: 'Acknowledged',
  },
];

/* ── Tracking Updates (keyed by alert_id) ──────────────── */

export const mockTrackingUpdates: Record<string, TrackingUpdate> = {
  'alert-20260218-143201-poi001': {
    event_type: 'poi_tracking_update',
    timestamp: '2026-02-18T14:35:22.789Z',
    poi_id: 'poi-001',
    alert_id: 'alert-20260218-143201-poi001',
    current_location: {
      camera_id: 'cam-05-produce',
      bbox: [200, 90, 350, 460],
      confidence: 0.85,
    },
    trajectory: [
      { camera_id: 'cam-01-entrance', first_seen: '2026-02-18T14:32:01Z', last_seen: '2026-02-18T14:32:18Z' },
      { camera_id: 'cam-03-front-aisle', first_seen: '2026-02-18T14:33:05Z', last_seen: '2026-02-18T14:34:12Z' },
      { camera_id: 'cam-05-produce', first_seen: '2026-02-18T14:35:20Z', last_seen: null },
    ],
    time_in_store_sec: 201,
  },
  'alert-20260218-131045-poi002': {
    event_type: 'poi_tracking_update',
    timestamp: '2026-02-18T13:15:10.000Z',
    poi_id: 'poi-002',
    alert_id: 'alert-20260218-131045-poi002',
    current_location: {
      camera_id: 'cam-09-appliances',
      bbox: [210, 95, 360, 470],
      confidence: 0.80,
    },
    trajectory: [
      { camera_id: 'cam-07-electronics', first_seen: '2026-02-18T13:10:45Z', last_seen: '2026-02-18T13:12:30Z' },
      { camera_id: 'cam-09-appliances', first_seen: '2026-02-18T13:14:00Z', last_seen: null },
    ],
    time_in_store_sec: 265,
  },
};

/* ── Historical Query Result ───────────────────────────── */

export const mockHistoryResult: HistoryResult = {
  event_type: 'poi_history_result',
  query_timestamp: '2026-02-18T16:00:00Z',
  poi_id: 'poi-001',
  query_range: {
    start: '2026-02-11T00:00:00Z',
    end: '2026-02-18T16:00:00Z',
  },
  visits: [
    {
      date: '2026-02-12',
      entry_time: '2026-02-12T11:23:00Z',
      exit_time: '2026-02-12T11:45:00Z',
      cameras_visited: ['cam-01-entrance', 'cam-07-aisle-3', 'cam-01-entrance'],
      duration_sec: 1320,
      alert_id: 'alert-20260212-112300-poi001',
    },
    {
      date: '2026-02-15',
      entry_time: '2026-02-15T09:10:00Z',
      exit_time: '2026-02-15T09:32:00Z',
      cameras_visited: ['cam-01-entrance', 'cam-05-produce', 'cam-12-checkout', 'cam-01-entrance'],
      duration_sec: 1320,
      alert_id: 'alert-20260215-091000-poi001',
    },
    {
      date: '2026-02-18',
      entry_time: '2026-02-18T14:32:01Z',
      exit_time: null,
      cameras_visited: ['cam-01-entrance', 'cam-03-front-aisle', 'cam-05-produce'],
      duration_sec: null,
      alert_id: 'alert-20260218-143201-poi001',
    },
  ],
  total_visits: 3,
  search_stats: {
    vectors_searched: 892000,
    query_latency_ms: 245,
  },
};
