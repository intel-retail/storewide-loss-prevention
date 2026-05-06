import { useState, useEffect, useCallback } from 'react';
import type { POI } from '../../types';
import { listPOIs, createPOI, deletePOI } from '../../api/poiApi';
import POICard from './POICard';
import UploadModal from './UploadModal';
import POIDetailModal from './POIDetailModal';
import ImagePreviewModal from '../common/ImagePreviewModal';

const POIManagement = () => {
  const [pois, setPois] = useState<POI[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [detailPoi, setDetailPoi] = useState<POI | null>(null);

  const fetchPOIs = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await listPOIs();
      setPois(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load POIs');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPOIs();
  }, [fetchPOIs]);

  const handleDelete = async (id: string) => {
    try {
      await deletePOI(id);
      setPois((prev) => prev.filter((p) => p.poi_id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete POI');
    }
  };

  const handleSave = async ({ name, severity, files }: { name: string; severity: 'low' | 'medium' | 'high'; files: File[] }) => {
    if (files.length === 0) return;
    try {
      await createPOI({ files, severity, description: name });
      setShowUpload(false);
      await fetchPOIs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create POI');
    }
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-display font-medium text-intel-dark">Person of Interest Registry</h2>
          <p className="text-sm text-intel-gray mt-1">{pois.length} enrolled POI{pois.length !== 1 ? 's' : ''}</p>
        </div>
        <button
          onClick={() => setShowUpload(true)}
          className="px-5 py-2.5 text-sm font-medium text-white bg-intel-blue rounded-lg hover:bg-intel-blue-dark transition-colors shadow-sm"
        >
          + Enroll POI
        </button>
      </div>

      {error && (
        <div className="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-center py-20 text-intel-gray">
          <p className="text-lg">Loading POIs...</p>
        </div>
      ) : (
        <>
          {/* Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-5">
            {pois.map((poi) => (
              <POICard key={poi.poi_id} poi={poi} onDelete={handleDelete} onViewDetails={setDetailPoi} onImageClick={setPreviewImage} />
            ))}
          </div>

          {pois.length === 0 && (
            <div className="text-center py-20 text-intel-gray">
              <p className="text-lg">No POIs enrolled yet</p>
              <p className="text-sm mt-1">Click "Enroll POI" to get started</p>
            </div>
          )}
        </>
      )}

      {/* Modals */}
      {showUpload && <UploadModal onClose={() => setShowUpload(false)} onSave={handleSave} />}
      {detailPoi && <POIDetailModal poi={detailPoi} onClose={() => setDetailPoi(null)} onImageClick={setPreviewImage} />}
      {previewImage && <ImagePreviewModal imageUrl={previewImage} onClose={() => setPreviewImage(null)} />}
    </div>
  );
};

export default POIManagement;
