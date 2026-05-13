import { useState } from 'react';

interface CameraConfig {
  name: string;
  streamPath: string;
}

const DEFAULT_CAMERAS: CameraConfig[] = [
  { name: 'Camera 01', streamPath: 'Camera_01' },
  { name: 'Camera 02', streamPath: 'Camera_02' },
];

/** Port where MediaMTX serves its built-in WebRTC player page. */
const MEDIAMTX_WEBRTC_PORT = 8889;

/**
 * Builds the MediaMTX WebRTC player URL for a given camera stream.
 * MediaMTX serves an embeddable player at  http://<host>:8889/<path>/
 */
const playerUrl = (streamPath: string): string => {
  const host = window.location.hostname;
  return `http://${host}:${MEDIAMTX_WEBRTC_PORT}/${streamPath}/`;
};

const CameraStreams = () => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm">
      {/* Header */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-gray-50 transition-colors rounded-t-xl"
      >
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <h3 className="text-sm font-semibold text-intel-dark">Live Camera Feeds</h3>
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-4 pt-0">
          {DEFAULT_CAMERAS.map((cam) => (
            <div key={cam.streamPath} className="relative rounded-lg overflow-hidden bg-gray-900">
              <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/60 text-white text-xs font-medium">
                {cam.name}
              </div>
              <iframe
                src={playerUrl(cam.streamPath)}
                title={cam.name}
                className="w-full aspect-video border-0"
                allow="autoplay"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CameraStreams;
