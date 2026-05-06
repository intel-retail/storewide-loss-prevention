import { useState } from 'react';
import type { SearchResult } from '../../types';
import { searchHistory } from '../../api/poiApi';
import AppearanceCard from './SearchResultCard';

const SearchPanel = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [result, setResult] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => setImagePreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleSearch = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await searchHistory({
        image: imageFile,
        topK: 20,
        startTime: startTime ? new Date(startTime).toISOString() : '',
        endTime: endTime ? new Date(endTime).toISOString() : '',
      });
      setResult(data);
      setSearched(true);
    } catch (err: any) {
      setError(err.message ?? 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setStartTime('');
    setEndTime('');
    setResult(null);
    setError(null);
    setSearched(false);
  };

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-white border-r border-gray-200 p-4 space-y-5 overflow-y-auto">
        <h3 className="text-sm font-semibold text-intel-dark">Search Parameters</h3>

        {/* Query image */}
        <div className="space-y-2">
          <span className="text-xs font-medium text-intel-gray">Query Image</span>
          {imagePreview ? (
            <div className="relative group">
              <img src={imagePreview} alt="Query" className="w-full aspect-square object-cover rounded-lg border border-gray-200" />
              <button
                onClick={() => { setImageFile(null); setImagePreview(null); }}
                className="absolute top-1 right-1 w-6 h-6 rounded-full bg-black/60 text-white text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                ×
              </button>
            </div>
          ) : (
            <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-intel-blue transition-colors">
              <span className="text-2xl text-intel-gray">📷</span>
              <span className="text-xs text-intel-gray mt-1">Upload Image</span>
              <input type="file" accept="image/*" onChange={handleUpload} className="hidden" />
            </label>
          )}
          <p className="text-[10px] text-intel-gray leading-tight">
            Use a surveillance screenshot or photo that includes the face (not a pre-cropped face-only image)
          </p>
        </div>

        {/* Time range */}
        <label className="block">
          <span className="text-xs font-medium text-intel-gray">Start Time</span>
          <input
            type="datetime-local"
            value={startTime}
            onChange={(e) => setStartTime(e.target.value)}
            className="mt-1 block w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-xs focus:border-intel-blue focus:ring-1 focus:ring-intel-blue outline-none"
          />
        </label>

        <label className="block">
          <span className="text-xs font-medium text-intel-gray">End Time</span>
          <input
            type="datetime-local"
            value={endTime}
            onChange={(e) => setEndTime(e.target.value)}
            className="mt-1 block w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-xs focus:border-intel-blue focus:ring-1 focus:ring-intel-blue outline-none"
          />
        </label>

        <div className="space-y-2 pt-2">
          <button
            onClick={handleSearch}
            disabled={!imageFile || loading}
            className="w-full px-4 py-2 text-sm font-medium text-white bg-intel-blue rounded-lg hover:bg-intel-blue-dark transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading && <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
            {loading ? 'Searching...' : 'Search'}
          </button>
          <button
            onClick={handleReset}
            className="w-full px-4 py-2 text-sm font-medium text-intel-gray bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
          >
            Reset
          </button>
        </div>
      </aside>

      {/* Results */}
      <div className="flex-1 p-6 space-y-4 overflow-y-auto">
        <div>
          <h2 className="text-xl font-display font-medium text-intel-dark">Historical Search</h2>
          <p className="text-sm text-intel-gray mt-1">
            {searched && result
              ? `${result.total_appearances} appearance${result.total_appearances !== 1 ? 's' : ''} found${result.matched_poi_id ? ` for POI ${result.matched_poi_id}` : ''}`
              : 'Upload a reference image and search for historical appearances'}
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Search stats */}
        {result && (
          <div className="flex flex-wrap gap-3">
            <div className="bg-blue-50 rounded-lg px-4 py-2 text-center">
              <p className="text-lg font-semibold text-intel-blue">{result.total_appearances}</p>
              <p className="text-[11px] text-intel-gray">Appearances</p>
            </div>
            {result.search_stats.poi_similarity != null && (
              <div className="bg-blue-50 rounded-lg px-4 py-2 text-center">
                <p className="text-lg font-semibold text-intel-blue">{(result.search_stats.poi_similarity * 100).toFixed(1)}%</p>
                <p className="text-[11px] text-intel-gray">POI Similarity</p>
              </div>
            )}
            {result.search_stats.best_similarity != null && (
              <div className="bg-blue-50 rounded-lg px-4 py-2 text-center">
                <p className="text-lg font-semibold text-intel-blue">{(result.search_stats.best_similarity * 100).toFixed(1)}%</p>
                <p className="text-[11px] text-intel-gray">Best Similarity</p>
              </div>
            )}
            <div className="bg-blue-50 rounded-lg px-4 py-2 text-center">
              <p className="text-lg font-semibold text-intel-blue">{result.search_stats.query_latency_ms} ms</p>
              <p className="text-[11px] text-intel-gray">Query Latency</p>
            </div>
            {result.search_stats.search_stage && (
              <div className="bg-blue-50 rounded-lg px-4 py-2 text-center">
                <p className="text-lg font-semibold text-intel-blue capitalize">{result.search_stats.search_stage.replace('_', ' ')}</p>
                <p className="text-[11px] text-intel-gray">Search Stage</p>
              </div>
            )}
          </div>
        )}

        {/* Appearances list */}
        {result && result.appearances.length > 0 ? (
          <div className="space-y-3">
            {result.appearances.map((appearance, i) => (
              <AppearanceCard key={appearance.track_id} appearance={appearance} index={i} />
            ))}
          </div>
        ) : searched && !loading ? (
          <div className="text-center py-20 text-intel-gray">
            <p className="text-lg">No appearances found</p>
            <p className="text-sm mt-1">Try a different image or expand the time range</p>
          </div>
        ) : !searched ? (
          <div className="text-center py-20 text-intel-gray">
            <p className="text-lg">Upload a query image to begin</p>
            <p className="text-sm mt-1">Historical appearances will appear here</p>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default SearchPanel;

