import type { Alert } from '../../types';

interface AlertCardProps {
  alert: Alert;
  onImageClick: (url: string) => void;
  poiPrimaryImage?: string;
}

const sevBadge = (s: Alert['severity']) => {
  const c = s === 'high' ? 'bg-red-100 text-red-700' : s === 'medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-600';
  return <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold capitalize ${c}`}>{s}</span>;
};

const AlertCard = ({ alert, onImageClick, poiPrimaryImage }: AlertCardProps) => {
  const score = alert.match.similarity_score;
  const scoreColor = score >= 0.9 ? 'text-red-600' : score >= 0.8 ? 'text-yellow-600' : 'text-green-600';

  return (
    <div className="bg-white rounded-xl shadow-sm border overflow-hidden transition-all border-gray-100">
      <div className="flex gap-3 p-4">
        {/* POI reference image */}
        {poiPrimaryImage && (
          <div className="flex-shrink-0 cursor-pointer" onClick={() => onImageClick(poiPrimaryImage)}>
            <img src={poiPrimaryImage} alt="POI" className="w-20 h-24 rounded-lg object-cover bg-gray-100" />
            <p className="text-[10px] text-center text-intel-gray mt-1">POI Ref</p>
          </div>
        )}

        {/* Match thumbnail */}
        <div className="flex-shrink-0 cursor-pointer" onClick={() => onImageClick(alert.match.thumbnail_path)}>
          <img src={alert.match.thumbnail_path} alt="Detected" className="w-20 h-24 rounded-lg object-cover bg-gray-100" />
          <p className="text-[10px] text-center text-intel-gray mt-1">Detected</p>
        </div>

        {/* Details */}
        <div className="flex-1 min-w-0 space-y-1.5">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-medium text-sm text-intel-dark">{alert.poi_metadata.name || alert.poi_id}</h3>
            {sevBadge(alert.severity)}
          </div>

          <p className="text-xs text-intel-gray">
            Camera: <span className="font-medium text-intel-dark">{alert.match.camera_id}</span>
          </p>
          <p className="text-xs text-intel-gray">
            {new Date(alert.timestamp).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </p>
          <div className="flex items-center gap-3 text-xs">
            <span>Similarity: <span className={`font-semibold ${scoreColor}`}>{(score * 100).toFixed(1)}%</span></span>
            <span className="text-intel-gray">Confidence: <span className="font-medium text-intel-dark">{(alert.match.confidence * 100).toFixed(1)}%</span></span>
          </div>
          {alert.poi_metadata.total_previous_matches > 0 && (
            <p className="text-[11px] text-intel-gray">
              {alert.poi_metadata.total_previous_matches} previous match{alert.poi_metadata.total_previous_matches !== 1 ? 'es' : ''}
            </p>
          )}
          {alert.poi_metadata.notes && (
            <p className="text-[11px] text-intel-gray italic truncate" title={alert.poi_metadata.notes}>{alert.poi_metadata.notes}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AlertCard;
