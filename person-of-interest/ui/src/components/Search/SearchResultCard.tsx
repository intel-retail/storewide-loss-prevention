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

const FrameImage = ({ url, label }: { url: string | null; label: string }) => {
  if (!url) return null;
  return (
    <div className="flex flex-col items-center gap-1">
      <img
        src={`${API_BASE}${url}`}
        alt={label}
        className="h-40 w-28 object-cover rounded-lg border border-gray-200 bg-gray-100"
        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
      />
      <span className="text-[10px] text-intel-gray">{label}</span>
    </div>
  );
};

const AppearanceCard = ({ appearance, index }: Props) => {
  const { track_id, camera_id, similarity, best_match_time, entry_frame_url, last_seen_frame_url, zone_appearances } = appearance;
  const hasFrames = entry_frame_url || last_seen_frame_url;

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
              <p className="text-[10px] text-intel-gray font-mono">{track_id}</p>
            </div>
          </div>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${similarityColor(similarity)}`}>
            {(similarity * 100).toFixed(1)}% match
          </span>
        </div>

        {/* Best match time */}
        <p className="text-xs text-intel-gray">
          Best match: <span className="font-medium text-intel-dark">{fmtTime(best_match_time)}</span>
        </p>

        {/* Entry / Last-seen frames */}
        {hasFrames && (
          <div className="flex gap-4">
            <FrameImage url={entry_frame_url} label="Entry" />
            <FrameImage url={last_seen_frame_url} label="Last Seen" />
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

