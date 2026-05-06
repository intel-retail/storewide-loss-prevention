import { useState, useRef } from 'react';

interface UploadModalProps {
  onClose: () => void;
  onSave: (data: { name: string; severity: 'low' | 'medium' | 'high'; files: File[] }) => void;
}

const UploadModal = ({ onClose, onSave }: UploadModalProps) => {
  const [name, setName] = useState('');
  const [severity, setSeverity] = useState<'low' | 'medium' | 'high'>('medium');
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addFiles = (newFiles: FileList | null) => {
    if (!newFiles) return;
    const added = Array.from(newFiles);
    setFiles((prev) => [...prev, ...added]);
    added.forEach((f) => {
      const reader = new FileReader();
      reader.onloadend = () => setPreviews((prev) => [...prev, reader.result as string]);
      reader.readAsDataURL(f);
    });
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
    setPreviews((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg mx-4 p-6 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-display font-medium text-intel-dark mb-5">Enroll Person of Interest</h2>

        {/* File upload area */}
        <div
          className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center cursor-pointer hover:border-intel-blue hover:bg-blue-50/30 transition-colors mb-4"
          onClick={() => fileInputRef.current?.click()}
        >
          {previews.length === 0 ? (
            <div className="space-y-2">
              <svg className="mx-auto h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 16v-8m0 0l-3 3m3-3l3 3M4 20h16a2 2 0 002-2V6a2 2 0 00-2-2H4a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-sm text-intel-gray">Click to upload reference images</p>
              <p className="text-xs text-gray-400">PNG, JPG — up to 5 images per POI</p>
            </div>
          ) : (
            <p className="text-xs text-intel-blue font-medium">+ Add more images</p>
          )}
          <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />
        </div>

        {/* Image previews */}
        {previews.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {previews.map((p, i) => (
              <div key={i} className="relative group">
                <img src={p} alt={`ref-${i}`} className="w-20 h-20 rounded-lg object-cover border border-gray-200" />
                <button
                  onClick={() => removeFile(i)}
                  className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-red-500 text-white text-[10px] flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Severity */}
        <label className="block mb-3">
          <span className="text-sm font-medium text-intel-dark">Severity</span>
          <select
            value={severity}
            onChange={(e) => setSeverity(e.target.value as 'low' | 'medium' | 'high')}
            className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-intel-blue focus:ring-1 focus:ring-intel-blue outline-none"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </label>

        {/* POI Name */}
        <label className="block mb-4">
          <span className="text-sm font-medium text-intel-dark">POI Name</span>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. John Doe"
            className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-intel-blue focus:ring-1 focus:ring-intel-blue outline-none"
          />
        </label>

        {/* Actions */}
        <div className="flex gap-3 justify-end">
          <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-intel-gray bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors">
            Cancel
          </button>
          <button
            onClick={() => onSave({ name, severity, files })}
            disabled={files.length === 0}
            className="px-4 py-2 text-sm font-medium text-white bg-intel-blue rounded-lg hover:bg-intel-blue-dark transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Enroll POI ({files.length} image{files.length !== 1 ? 's' : ''})
          </button>
        </div>
      </div>
    </div>
  );
};

export default UploadModal;
