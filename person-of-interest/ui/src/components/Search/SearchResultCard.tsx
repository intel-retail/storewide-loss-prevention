import { useState, useEffect } from 'react';
import type { Appearance } from '../../types';

interface Props {
  appearance: Appearance;
  index: number;
}

const API_BASE = import.meta.env.VITE_API_URL ?? '';

const fmtTime = (ts: string) => {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleString('en-US', {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch {
    return ts;
  }
};

const similarityColor = (s: number) => {
  if (s >= 0.85) return 'bg-green-100 text-green-700';
  if (s >= 0.70) return 'bg-yellow-100 text-yellow-700';
  return 'bg-red-100 text-red-700';
};

// ── Lightbox ──────────────────────────────────────────────────────────────────
const Lightbox = ({ src, label, onClose }: { src: string; label: string; onClose: () => void }) => {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);
  return (
  <div
    className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
    onClick={onClose}
  >
    <div className="relative max-w-4xl max-h-[90vh] flex flex-col items-center gap-2" onClick={e => e.stopPropagation()}>
      <img
        src={src}
        alt={label}
        className="max-h-[80vh] max-w-full rounded-xl shadow-2xl object-contain"
      />
      <span className="text-white/70 text-sm">{label}</span>
      <button
        className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-white/20 hover:bg-white/40 text-white flex items-center justify-center text-lg font-bold transition-colors"
        onClick={onClose}
        aria-label="Close"
      >
        ×
      </button>
    </div>
  </div>
  );
};

// ── FrameImage ────────────────────────────────────────────────────────────────
const FrameImage = ({ url, label }: { url: string | null; label: string }) => {
  const [lightbox, setLightbox] = useState(false);
  if (!url) return null;
  const fullSrc = `${API_BASE}${url}`;
  return (
    <>
      <div className="flex flex-col items-center gap-1">
        <img
          src={fullSrc}
          alt={label}
          className="h-40 w-28 object-cover rounded-lg border border-gray-200 bg-gray-100 cursor-zoom-in hover:opacity-90 transition-opacity"
          onClick={() => setLightbox(true)}
          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
        />
        <span className="text-[10px] text-intel-gray">{label}</span>
      </div>
      {lightbox && <Lightbox src={fullSrc} label={label} onClose={() => setLightbox(false)} />}
    </>
  );
};

const AppearanceCard = ({ appearance, index }: Props) => {
  const {
    faiss_id, track_id, camera_id, similarity,
    entry_similarity, exit_similarity,
    entry_timestamp, exit_timestamp,
    entry_frame_url, exit_frame_url,
    zone_appearances,
  } = appearance;

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="p-4 space-y-4">

        {/* Header row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-intel-blue text-white text-xs flex items-center justify-center font-semibold">
              {index + 1}
            </span>
            <div>
              <p className="text-sm font-medium text-intel-dark">{camera_id}</p>
              <p className="text-[10px] text-intel-gray font-mono">
                {track_id}{faiss_id != null ? ` · id:${faiss_id}` : ''}
              </p>
            </div>
          </div>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${similarityColor(similarity)}`}>
            {(similarity * 100).toFixed(1)}% match
          </span>
        </div>

        {/* Entry + Exit frames side by side */}
        {(entry_frame_url || exit_frame_url) && (
          <div className="flex gap-4">
            <div className="flex flex-col gap-1 items-center">
              <FrameImage url={entry_frame_url} label="Entry" />
              {entry_frame_url && (
                <span className="text-[10px] text-intel-gray">
                  {(entry_similarity * 100).toFixed(1)}% · {fmtTime(entry_timestamp)}
                </span>
              )}
            </div>
            <div className="flex flex-col gap-1 items-center">
              <FrameImage url={exit_frame_url} label="Exit" />
              {exit_frame_url && exit_similarity != null && (
                <span className="text-[10px] text-intel-gray">
                  {(exit_similarity * 100).toFixed(1)}% · {fmtTime(exit_timestamp ?? '')}
                </span>
              )}
              {exit_frame_url == null && (
                <span className="text-[10px] text-intel-gray/50 italic">No exit frame</span>
              )}
            </div>
          </div>
        )}

        {/* Zone appearances */}
        {zone_appearances.length > 0 && (
          <div>
            <p className="text-xs font-medium text-intel-gray mb-2">Zone History</p>
            <div className="space-y-2">
              {zone_appearances.map((z, zi) => (
                <div key={zi} className="bg-gray-50 rounded-lg p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-intel-dark">{z.zone || z.scene_id}</span>
                    {z.dwell_seconds != null && (
                      <span className="text-[11px] text-intel-gray">
                        {z.dwell_seconds >= 60
                          ? `${Math.floor(z.dwell_seconds / 60)}m ${z.dwell_seconds % 60}s`
                          : `${z.dwell_seconds}s`}
                      </span>
                    )}
                  </div>
                  <div className="flex gap-3 text-[11px] text-intel-gray">
                    <span>In: <span className="text-intel-dark">{fmtTime(z.entry_time)}</span></span>
                    <span>Out: <span className="text-intel-dark">{fmtTime(z.exit_time)}</span></span>
                  </div>
                  {(z.entry_frame_url || z.exit_frame_url) && (
                    <div className="flex gap-3 mt-1">
                      <FrameImage url={z.entry_frame_url ?? null} label="Zone Entry" />
                      <FrameImage url={z.exit_frame_url ?? null} label="Zone Exit" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default AppearanceCard;

