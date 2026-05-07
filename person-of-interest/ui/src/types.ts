/* ── POI Enrollment ─────────────────────────────────────── */

export interface ReferenceImage {
  source: string;
  embedding_id: string;
  vector_dim: number;
  image_path: string;
}

export interface POI {
  event_type: 'poi_enrollment';
  timestamp: string;
  poi_id: string;
  enrolled_by: string;
  severity: 'low' | 'medium' | 'high';
  notes: string;
  reference_images: ReferenceImage[];
  status: 'active' | 'inactive';
}

/* ── POI Match Alert ───────────────────────────────────── */

export interface AlertMatch {
  camera_id: string;
  confidence: number;
  similarity_score: number;
  bbox: [number, number, number, number];
  frame_number: number;
  thumbnail_path: string;
}

export interface AlertPOIMetadata {
  name: string;
  notes: string;
  enrollment_date: string;
  total_previous_matches: number;
}

export interface Alert {
  event_type: 'poi_match_alert';
  timestamp: string;
  alert_id: string;
  poi_id: string;
  severity: 'low' | 'medium' | 'high';
  match: AlertMatch;
  poi_metadata: AlertPOIMetadata;
  status: 'New' | 'Acknowledged'; // UI-level state
}

/* ── POI Tracking Update ───────────────────────────────── */

export interface TrajectoryStep {
  camera_id: string;
  first_seen: string;
  last_seen: string | null;
}

export interface TrackingUpdate {
  event_type: 'poi_tracking_update';
  timestamp: string;
  poi_id: string;
  alert_id: string;
  current_location: {
    camera_id: string;
    bbox: [number, number, number, number];
    confidence: number;
  };
  trajectory: TrajectoryStep[];
  time_in_store_sec: number;
}

/* ── Historical POI Query ──────────────────────────────── */

export interface Visit {
  date: string;
  entry_time: string;
  exit_time: string | null;
  cameras_visited: string[];
  duration_sec: number | null;
  alert_id: string;
}

export interface HistoryResult {
  event_type: 'poi_history_result';
  query_timestamp: string;
  poi_id: string;
  query_range: { start: string; end: string };
  visits: Visit[];
  total_visits: number;
  search_stats: {
    vectors_searched: number;
    query_latency_ms: number;
  };
}

/* ── Offline Search (detection index) ─────────────────── */

export interface ZoneAppearance {
  zone: string;
  scene_id: string;
  entry_time: string;
  exit_time: string;
  dwell_seconds: number | null;
  entry_frame_url?: string;
  exit_frame_url?: string;
}

export interface Appearance {
  faiss_id: number | null;
  track_id: string;
  camera_id: string;
  similarity: number;
  entry_similarity: number;
  exit_similarity: number | null;
  entry_timestamp: string;
  exit_timestamp: string | null;
  entry_frame_url: string | null;
  exit_frame_url: string | null;
  bbox: { x: number; y: number; width: number; height: number } | null;
  zone_appearances: ZoneAppearance[];
}

export interface SearchResult {
  event_type: 'offline_search_result';
  query_timestamp: string;
  query_range: { start: string; end: string };
  matched_poi_id?: string;
  total_appearances: number;
  appearances: Appearance[];
  rejection_reason?: string;
  search_stats: {
    search_stage?: string;
    vectors_searched?: number;
    raw_hits?: number;
    unique_tracks?: number;
    unique_tracks_above_threshold?: number;
    hits_above_threshold?: number;
    best_similarity?: number;
    poi_similarity?: number;
    detection_hits?: number;
    margin?: number;
    query_latency_ms: number;
  };
}
