import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchCameras, type CameraInfo } from '../../api/poiApi';

/** Poll interval for refreshing camera list from the backend. */
const POLL_INTERVAL_MS = 30_000;

/** Default MediaMTX WebRTC port — overridden by API response. */
const DEFAULT_WEBRTC_PORT = 8889;

const CameraStreams = () => {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [webrtcPort, setWebrtcPort] = useState(DEFAULT_WEBRTC_PORT);
  const [collapsed, setCollapsed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadCameras = useCallback(async () => {
    try {
      const data = await fetchCameras();
      setCameras(data.cameras);
      setWebrtcPort(data.mediamtx_webrtc_port);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cameras');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCameras();
    timerRef.current = setInterval(loadCameras, POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [loadCameras]);

  const playerUrl = (streamPath: string): string => {
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    return `${protocol}//${host}:${webrtcPort}/${streamPath}/`;
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm">
      {/* Header */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-gray-50 transition-colors rounded-t-xl"
      >
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${cameras.length > 0 ? 'bg-red-500 animate-pulse' : 'bg-gray-400'}`} />
          <h3 className="text-sm font-semibold text-intel-dark">
            Live Camera Feeds
            {cameras.length > 0 && (
              <span className="ml-1 text-xs font-normal text-gray-500">({cameras.length})</span>
            )}
          </h3>
        </div>
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform ${collapsed ? '' : 'rotate-180'}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Camera panels */}
      {!collapsed && (
        <div className="p-4 pt-0">
          {loading && (
            <p className="text-sm text-gray-500 py-4 text-center">Loading cameras…</p>
          )}
          {!loading && error && (
            <div className="text-sm text-red-600 py-4 text-center">
              <p>{error}</p>
              <button onClick={loadCameras} className="mt-2 text-xs text-blue-600 underline">
                Retry
              </button>
            </div>
          )}
          {!loading && !error && cameras.length === 0 && (
            <p className="text-sm text-gray-500 py-4 text-center">No cameras configured</p>
          )}
          {!loading && !error && cameras.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {cameras.map((cam) => (
                <div key={cam.camera_id} className="relative rounded-lg overflow-hidden bg-gray-900">
                  <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/60 text-white text-xs font-medium">
                    {cam.name}
                  </div>
                  <iframe
                    src={playerUrl(cam.stream_path)}
                    title={cam.name}
                    className="w-full aspect-video border-0"
                    allow="autoplay"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CameraStreams;
