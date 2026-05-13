import { useAlertWebSocket } from '../../hooks/useAlertWebSocket';
import AlertCard from './AlertCard';
import CameraStreams from './CameraStreams';
import ImagePreviewModal from '../common/ImagePreviewModal';
import { useState } from 'react';
import { clearAlerts } from '../../api/poiApi';

const ALERTS_PER_PAGE = 10;

const LiveAlerts = () => {
  const { alerts, connected, setAlerts } = useAlertWebSocket();
  const [filterPoi, setFilterPoi] = useState('');
  const [filterCamera, setFilterCamera] = useState('');
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const handleClearAlerts = async () => {
    if (!window.confirm('Clear all alerts? This cannot be undone.')) return;
    setClearing(true);
    try {
      await clearAlerts();
      setAlerts([]);
      setFilterPoi('');
      setFilterCamera('');
      setCurrentPage(1);
    } catch (err) {
      console.error('Failed to clear alerts:', err);
    } finally {
      setClearing(false);
    }
  };

  const getPoiPrimaryImage = (_poiId: string) => undefined;

  const uniquePois = [...new Set(alerts.map((a) => a.poi_id))];
  const uniqueCameras = [...new Set(alerts.map((a) => a.match.camera_id))];

  const filtered = alerts.filter((a) => {
    if (filterPoi && a.poi_id !== filterPoi) return false;
    if (filterCamera && a.match.camera_id !== filterCamera) return false;
    return true;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / ALERTS_PER_PAGE));
  const safePage = Math.min(currentPage, totalPages);
  const paginatedAlerts = filtered.slice((safePage - 1) * ALERTS_PER_PAGE, safePage * ALERTS_PER_PAGE);

  const newCount = filtered.filter((a) => a.status === 'New').length;

  return (
    <div className="flex h-full">
      {/* Sidebar filters */}
      <aside className="w-56 flex-shrink-0 bg-white border-r border-gray-200 p-4 space-y-5">
        <h3 className="text-sm font-semibold text-intel-dark">Filters</h3>

        <label className="block">
          <span className="text-xs font-medium text-intel-gray">POI</span>
          <select
            value={filterPoi}
            onChange={(e) => { setFilterPoi(e.target.value); setCurrentPage(1); }}
            className="mt-1 block w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-xs focus:border-intel-blue focus:ring-1 focus:ring-intel-blue outline-none"
          >
            <option value="">All POIs</option>
            {uniquePois.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>

        <label className="block">
          <span className="text-xs font-medium text-intel-gray">Camera</span>
          <select
            value={filterCamera}
            onChange={(e) => { setFilterCamera(e.target.value); setCurrentPage(1); }}
            className="mt-1 block w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-xs focus:border-intel-blue focus:ring-1 focus:ring-intel-blue outline-none"
          >
            <option value="">All Cameras</option>
            {uniqueCameras.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <button
          onClick={() => { setFilterPoi(''); setFilterCamera(''); setCurrentPage(1); }}
          className="w-full px-3 py-1.5 text-xs font-medium text-intel-gray bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
        >
          Reset Filters
        </button>
      </aside>

      {/* Main content */}
      <div className="flex-1 p-6 space-y-4 overflow-y-auto">
        <CameraStreams />

        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-display font-medium text-intel-dark">Live Alerts</h2>
            <p className="text-sm text-intel-gray mt-1">
              {filtered.length} alert{filtered.length !== 1 ? 's' : ''}
              {newCount > 0 && <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-700">{newCount} new</span>}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleClearAlerts}
              disabled={clearing || alerts.length === 0}
              className="px-3 py-1.5 text-xs font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {clearing ? 'Clearing…' : 'Clear All Alerts'}
            </button>
            <span className={`flex items-center gap-1.5 text-xs font-medium ${connected ? 'text-green-600' : 'text-gray-400'}`}>
              <span className={`inline-block w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-gray-400'}`} />
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        <div className="space-y-3">
          {paginatedAlerts.map((alert) => (
            <AlertCard
              key={alert.alert_id}
              alert={alert}
              onImageClick={setPreviewImage}
              poiPrimaryImage={getPoiPrimaryImage(alert.poi_id)}
            />
          ))}
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 pt-4">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={safePage <= 1}
              className="px-3 py-1.5 text-xs font-medium text-intel-blue bg-blue-50 rounded-lg hover:bg-blue-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <span className="text-xs text-intel-gray">
              Page {safePage} of {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={safePage >= totalPages}
              className="px-3 py-1.5 text-xs font-medium text-intel-blue bg-blue-50 rounded-lg hover:bg-blue-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        )}

        {filtered.length === 0 && (
          <div className="text-center py-20 text-intel-gray">
            <p className="text-lg">No alerts found</p>
            <p className="text-sm mt-1">Adjust filters or wait for new detections</p>
          </div>
        )}
      </div>

      {previewImage && <ImagePreviewModal imageUrl={previewImage} onClose={() => setPreviewImage(null)} />}
    </div>
  );
};

export default LiveAlerts;
