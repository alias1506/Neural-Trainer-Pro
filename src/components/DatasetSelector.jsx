import React, { useRef, useState, useEffect } from 'react';
import Swal from 'sweetalert2';
import Papa from 'papaparse';
import FolderTree from './FolderTree.jsx';

const CircularProgress = ({ percentage, onCancel }) => {
  const [isHovered, setIsHovered] = useState(false);
  const radius = 10;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div
      className="relative w-6 h-6 flex items-center justify-center cursor-pointer"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onCancel}
      title="Cancel Upload"
    >
      {isHovered ? (
        <svg className="w-6 h-6 text-gray-500 hover:text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      ) : (
        <svg className="transform -rotate-90 w-6 h-6">
          <circle className="text-gray-200" strokeWidth="2" stroke="currentColor" fill="transparent" r={radius} cx="12" cy="12" />
          <circle
            className="text-green-500 transition-all duration-300 ease-in-out"
            strokeWidth="2"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            stroke="currentColor"
            fill="transparent"
            r={radius}
            cx="12"
            cy="12"
          />
        </svg>
      )}
    </div>
  );
};

const SuccessCheck = () => (
  <div className="w-6 h-6 flex items-center justify-center">
    <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
    </svg>
  </div>
);

export default function DatasetSelector({ onFilesSelected, datasetInfo, onFormatChange, uploadProgress, onCancelUpload }) {
  const [dragOver, setDragOver] = useState(false);
  const folderInputRef = useRef();
  const csvInputRef = useRef();
  const [selectedFormat, setSelectedFormat] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);
  const prevUploadRef = useRef(null);

  // CSV lazy loading state
  const [csvData, setCsvData] = useState(null); // { columns: [], allRows: [] }
  const [csvPage, setCsvPage] = useState(1);
  const csvRowsPerPage = 50; // Show 50 rows per page
  const csvFileRef = useRef(null);
  const [csvFileName, setCsvFileName] = useState('');

  // Success animation after upload finishes
  useEffect(() => {
    if (prevUploadRef.current?.uploading && !uploadProgress && datasetInfo) {
      setShowSuccess(true);
      const timer = setTimeout(() => setShowSuccess(false), 2000);
      return () => clearTimeout(timer);
    }
    prevUploadRef.current = uploadProgress;
  }, [uploadProgress, datasetInfo]);

  // Parse CSV using PapaParse when a CSV file is selected (lazy loading)
  useEffect(() => {
    if (datasetInfo?.structure?.type === 'csv') {
      if (csvFileRef.current) {
        Papa.parse(csvFileRef.current, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            if (results.data && results.data.length > 0) {
              setCsvData({
                columns: results.meta.fields,
                allRows: results.data
              });
              setCsvPage(1);
            }
          },
          error: (err) => {
            console.error('CSV parse error:', err);
            setCsvData(null);
          },
        });
      } else {
        // Try to fetch from server if file ref is lost
        const findCsv = (nodes) => {
          if (!nodes) return null;
          for (const node of nodes) {
            if (!node.isFolder && node.name.endsWith('.csv')) {
              return node.name;
            }
            if (node.isFolder) {
              const found = findCsv(node.children);
              if (found) return found;
            }
          }
          return null;
        };

        const csvName = findCsv(datasetInfo.structure.tree?.children);

        if (csvName) {
          setCsvFileName(csvName);
          fetch(`http://localhost:3001/uploads/dataset/${csvName}`)
            .then(response => response.text())
            .then(csvText => {
              Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                  if (results.data && results.data.length > 0) {
                    setCsvData({
                      columns: results.meta.fields,
                      allRows: results.data
                    });
                    setCsvPage(1);
                  }
                },
                error: (err) => {
                  console.error('CSV parse error:', err);
                  setCsvData(null);
                }
              });
            })
            .catch(err => console.error('Failed to fetch CSV:', err));
        }
      }
    } else {
      setCsvData(null);
      setCsvPage(1);
    }
  }, [datasetInfo]);

  const datasetFormats = [
    { value: 'cifar-binary', label: 'CIFAR-10/100 Binary' },
    { value: 'train-test-split', label: 'Train/Test Split' },
    { value: 'class-folders', label: 'Class Folders' },
    { value: 'yolo', label: 'YOLO Format' },
    { value: 'flat-images', label: 'Flat Images' },
    { value: 'csv', label: 'CSV Dataset' },
    { value: 'custom', label: 'Custom Format' },
  ];

  // Keep selected format in sync with backend info
  useEffect(() => {
    if (datasetInfo?.structure?.type) {
      setSelectedFormat(datasetInfo.structure.type);
    }
  }, [datasetInfo]);

  const handleFormatChange = (newFormat) => {
    setSelectedFormat(newFormat);
    if (onFormatChange) onFormatChange(newFormat);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      if (files[0].name.endsWith('.csv')) {
        csvFileRef.current = files[0];
        setCsvFileName(files[0].name);
      }
      await onFilesSelected(files);
    }
  };

  const onInput = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      if (files[0].name.endsWith('.csv')) {
        csvFileRef.current = files[0];
        setCsvFileName(files[0].name);
      }
      await onFilesSelected(files);
    }
  };

  const handleRemoveDataset = async () => {
    const result = await Swal.fire({
      title: 'Remove Dataset?',
      text: 'This will clear the uploaded dataset from memory. You can upload a new one.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Yes, Remove',
      cancelButtonText: 'Cancel',
      confirmButtonColor: '#ef4444',
    });
    if (result.isConfirmed) {
      // First clear local state to update UI immediately
      setCsvData(null);
      setCsvFileName('');
      csvFileRef.current = null;
      setCsvPage(1);

      // Then notify parent component
      await onFilesSelected(null, true);

      Swal.fire({ icon: 'success', title: 'Dataset Removed', timer: 1500, showConfirmButton: false });

      // Finally cleanup server files
      try {
        await fetch('http://localhost:3001/api/cleanup-uploads', { method: 'POST' });
      } catch (e) {
        console.error('Cleanup error:', e);
      }
    }
  };

  // Calculate pagination for CSV
  const totalPages = csvData ? Math.ceil(csvData.allRows.length / csvRowsPerPage) : 0;
  const startIdx = (csvPage - 1) * csvRowsPerPage;
  const endIdx = startIdx + csvRowsPerPage;
  const currentRows = csvData ? csvData.allRows.slice(startIdx, endIdx) : [];

  return (
    <div className="flex flex-col h-full animate-fade-in gap-3">
      {/* Page Header */}
      <div className="flex-shrink-0 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Dataset Selection</h2>
          <p className="text-xs text-muted">Upload and configure your training dataset</p>
        </div>
        {datasetInfo && (
          <div className="px-2 py-1 rounded bg-green-50 border border-green-200 text-green-600 text-xs flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Dataset Ready
          </div>
        )}
      </div>

      {/* Top bar */}
      <div className="flex-shrink-0 grid grid-cols-4 gap-3">
        <div className="card card-compact flex items-center justify-between">
          <div>
            <div className="text-xs font-medium text-muted">Status</div>
            <div className="text-sm font-bold">
              {uploadProgress ? 'Uploading...' : datasetInfo ? 'Ready' : 'Waiting'}
            </div>
          </div>
          {uploadProgress ? (
            <CircularProgress percentage={uploadProgress.percentage} onCancel={onCancelUpload} />
          ) : datasetInfo ? (
            <SuccessCheck />
          ) : (
            <div className="w-2 h-2 rounded-full bg-gray-300" />
          )}
        </div>
        {/* Files count */}
        <div className="card card-compact">
          <div className="text-xs font-medium text-muted">Files</div>
          <div className="text-sm font-bold">{datasetInfo?.fileCount || 0}</div>
        </div>
        <div className="card card-compact">
          <div className="text-xs font-medium text-muted">Classes</div>
          <div className="text-sm font-bold">
            {uploadProgress ? <span className="animate-pulse">...</span> : datasetInfo?.structure?.classes?.length || 0}
          </div>
        </div>
        <div className="card card-compact">
          <div className="text-xs font-medium text-muted">Format</div>
          <div className="text-sm font-bold truncate">{datasetInfo?.structure?.type || 'N/A'}</div>
        </div>
      </div>

      {/* Main area */}
      <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
        {/* Left pane */}
        <div className="col-span-12 md:col-span-7 flex flex-col gap-3 overflow-y-auto custom-scrollbar pr-1">
          {/* Upload area */}
          <div className="card flex-shrink-0">
            <h3 className="text-sm font-semibold mb-2">Upload Dataset</h3>
            <div
              className={`border-2 border-dashed rounded-lg p-4 text-center ${dragOver ? 'border-[#6B728E] bg-gray-50' : 'border-gray-200'}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <svg className="w-7 h-7 mx-auto mb-2 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-xs text-muted mb-2">Drag & drop folder or CSV file here</p>
              <div className="flex gap-2">
                <button className="btn btn-primary flex-1 justify-center text-xs" onClick={() => folderInputRef.current.click()}>Select Folder</button>
                <button className="btn btn-secondary flex-1 justify-center text-xs" onClick={() => csvInputRef.current.click()}>Select CSV</button>
              </div>
              <input ref={folderInputRef} type="file" webkitdirectory="" directory="" onChange={onInput} className="hidden" />
              <input ref={csvInputRef} type="file" accept=".csv" onChange={onInput} className="hidden" />
            </div>
          </div>

          {/* Configuration */}
          <div className="card flex-shrink-0">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold">Configuration</h3>
              {datasetInfo && (
                <button onClick={handleRemoveDataset} className="text-xs text-red-500 hover:text-red-600 font-medium">Remove</button>
              )}
            </div>
            <div className="space-y-2">
              <div>
                <label className="label text-xs">Dataset Format</label>
                <select
                  value={selectedFormat}
                  onChange={(e) => handleFormatChange(e.target.value)}
                  className="select w-full text-xs py-1.5"
                  disabled={!datasetInfo}
                >
                  {!datasetInfo ? (
                    <option value="">Upload a dataset to select</option>
                  ) : (
                    <>
                      <option value="">Select format...</option>
                      {datasetFormats.map((f) => (
                        <option key={f.value} value={f.value}>{f.label}</option>
                      ))}
                    </>
                  )}
                </select>
              </div>
              <div>
                <label className="label text-xs">
                  Classes {datasetInfo?.structure?.classes?.length ? `(${datasetInfo.structure.classes.length})` : ''}
                </label>
                <div className="flex flex-wrap gap-1 min-h-[2rem] max-h-20 overflow-y-auto custom-scrollbar p-2 border rounded bg-gray-50">
                  {datasetInfo?.structure?.classes?.length > 0 ? (
                    datasetInfo.structure.classes.map((c, i) => (
                      <span key={i} className="px-2 py-0.5 bg-white border rounded text-[10px] text-gray-600">{c}</span>
                    ))
                  ) : (
                    <span className="text-xs text-gray-400">No classes available</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Dataset Information */}
          <div className="card flex-shrink-0">
            <h3 className="text-sm font-semibold mb-2">Dataset Information</h3>
            {datasetInfo ? (
              <div className="space-y-1">
                <div className="flex items-center justify-between py-1 border-b border-gray-100">
                  <span className="text-xs text-muted">Total Files</span>
                  <span className="text-xs font-semibold">{datasetInfo.fileCount?.toLocaleString() || 0}</span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-gray-100">
                  <span className="text-xs text-muted">Total Classes</span>
                  <span className="text-xs font-semibold">{datasetInfo.structure?.classes?.length || 0}</span>
                </div>
                <div className="flex items-center justify-between py-1 border-b border-gray-100">
                  <span className="text-xs text-muted">Dataset Type</span>
                  <span className="text-xs font-semibold capitalize">{datasetInfo.structure?.type?.replace(/-/g, ' ') || 'Unknown'}</span>
                </div>
                {datasetInfo.structure?.split && (
                  <>
                    <div className="flex items-center justify-between py-1 border-b border-gray-100">
                      <span className="text-xs text-muted">Training Samples</span>
                      <span className="text-xs font-semibold">{datasetInfo.structure.split.train || 0}</span>
                    </div>
                    <div className="flex items-center justify-between py-1 border-b border-gray-100">
                      <span className="text-xs text-muted">Test Samples</span>
                      <span className="text-xs font-semibold">{datasetInfo.structure.split.test || 0}</span>
                    </div>
                  </>
                )}
                <div className="mt-2 p-2 bg-blue-50 border border-blue-100 rounded">
                  <div className="flex items-start gap-2">
                    <svg className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-[10px] text-blue-700 leading-relaxed">
                      Your dataset is ready for training. Make sure to select the correct format above if auto-detection didn't work properly.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-5">
                <svg className="w-10 h-10 mx-auto mb-2 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-xs text-gray-400">No dataset uploaded yet</p>
                <p className="text-[10px] text-gray-300 mt-1">Upload a folder or CSV file to see details</p>
              </div>
            )}
          </div>
        </div>

        {/* Right pane – file tree or CSV preview */}
        <div className="col-span-12 md:col-span-5 flex flex-col min-h-0">
          <div className="card flex-1 flex flex-col min-h-0 p-0 overflow-hidden" style={{ maxHeight: 'calc(100vh - 180px)' }}>
            <div className="p-3 border-b bg-gray-50 flex items-center justify-between flex-shrink-0">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <svg className="w-4 h-4 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                File Explorer
              </h3>
              <span className="text-xs text-muted">{datasetInfo ? 'Structure Preview' : 'No Data'}</span>
            </div>
            <div className="flex-1 overflow-auto custom-scrollbar">
              {datasetInfo && datasetInfo.structure ? (
                <>
                  {datasetInfo.structure.type === 'csv' ? (
                    <div className="p-3 flex flex-col h-full">
                      <p className="text-xs font-medium mb-2 flex-shrink-0">{csvFileName ? `${csvFileName}` : 'CSV file'}</p>
                      {csvData && (
                        <div className="flex flex-col flex-1 min-h-0">
                          <div className="flex items-center justify-between mb-2 flex-shrink-0">
                            <div className="text-xs font-semibold text-gray-700">
                              CSV Preview ({csvData.allRows.length} total rows)
                            </div>
                            <div className="text-xs text-gray-500">
                              Page {csvPage} of {totalPages}
                            </div>
                          </div>

                          {/* CSV Table with constrained height */}
                          <div className="border rounded overflow-auto flex-1" style={{ maxHeight: '450px' }}>
                            <table className="w-full text-[10px] border-collapse">
                              <thead className="sticky top-0 bg-gray-100 z-10">
                                <tr>
                                  {csvData.columns?.map((col, i) => (
                                    <th key={i} className="border border-gray-200 px-2 py-1 text-left font-semibold text-gray-700 whitespace-nowrap">{col}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="bg-white">
                                {currentRows.map((row, ri) => (
                                  <tr key={ri} className="hover:bg-gray-50">
                                    {csvData.columns?.map((col, ci) => (
                                      <td key={ci} className="border border-gray-200 px-2 py-1 text-gray-600 whitespace-nowrap">{row[col] ?? '—'}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>

                          {/* Pagination Controls */}
                          {totalPages > 1 && (
                            <div className="flex items-center justify-between mt-2 gap-2 flex-shrink-0">
                              <button
                                onClick={() => setCsvPage(Math.max(1, csvPage - 1))}
                                disabled={csvPage === 1}
                                className="btn btn-secondary text-xs py-1 px-3 disabled:opacity-50"
                              >
                                Previous
                              </button>
                              <span className="text-xs text-gray-600">
                                Showing {startIdx + 1}-{Math.min(endIdx, csvData.allRows.length)} of {csvData.allRows.length}
                              </span>
                              <button
                                onClick={() => setCsvPage(Math.min(totalPages, csvPage + 1))}
                                disabled={csvPage === totalPages}
                                className="btn btn-secondary text-xs py-1 px-3 disabled:opacity-50"
                              >
                                Next
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="p-2">
                      <FolderTree structure={datasetInfo.structure} totalFiles={datasetInfo.fileCount} />
                    </div>
                  )}
                </>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-muted p-4">
                  <svg className="w-12 h-12 mb-2 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" /></svg>
                  <p className="text-xs">Upload a dataset to view structure</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}