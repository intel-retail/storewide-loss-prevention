// API base URL — uses proxy in dev (via vite.config.ts)
const API_BASE = import.meta.env.VITE_API_URL ?? '';

export interface CreatePOIParams {
  files: File[];
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export async function listPOIs() {
  const res = await fetch(`${API_BASE}/api/v1/poi`);
  if (!res.ok) throw new Error(`Failed to list POIs: ${res.statusText}`);
  return res.json();
}

export async function createPOI({ files, severity, description }: CreatePOIParams) {
  const form = new FormData();
  files.forEach((f) => form.append('images', f));
  form.append('severity', severity);
  form.append('description', description);
  const res = await fetch(`${API_BASE}/api/v1/poi`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.error || 'Failed to create POI');
  }
  return res.json();
}

export async function deletePOI(poiId: string) {
  const res = await fetch(`${API_BASE}/api/v1/poi/${encodeURIComponent(poiId)}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Failed to delete POI: ${res.statusText}`);
  return res.json();
}

export async function clearAlerts() {
  const res = await fetch(`${API_BASE}/api/v1/alerts`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Failed to clear alerts: ${res.statusText}`);
  return res.json() as Promise<{ status: string; deleted: number }>;
}

export interface SearchParams {
  image: File;
  topK?: number;
  startTime?: string;
  endTime?: string;
}

export async function searchHistory({ image, topK = 20, startTime = '', endTime = '' }: SearchParams) {
  const form = new FormData();
  form.append('image', image);
  form.append('top_k', String(topK));
  form.append('start_time', startTime);
  form.append('end_time', endTime);
  const res = await fetch(`${API_BASE}/api/v1/search`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Search failed');
  }
  return res.json();
}
