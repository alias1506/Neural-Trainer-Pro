import React, { useRef, useState } from 'react'
import Swal from 'sweetalert2'
import FolderTree from './FolderTree.jsx'

export default function DatasetSelector({ onFilesSelected, datasetInfo, onFormatChange, uploadProgress }) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()
  const [selectedFormat, setSelectedFormat] = useState('')
  const [loadingClasses, setLoadingClasses] = useState(false)
  
  // Available dataset formats
  const datasetFormats = [
    { value: 'cifar-binary', label: 'CIFAR-10/100 Binary', description: 'Binary format with data_batch files' },
    { value: 'train-test-split', label: 'Train/Test Split', description: 'Folders: train/ and test/ with class subfolders' },
    { value: 'class-folders', label: 'Class Folders', description: 'Root folders as classes (auto-split 80/20)' },
    { value: 'yolo', label: 'YOLO Format', description: 'Images with .txt label files' },
    { value: 'flat-images', label: 'Flat Images', description: 'Single directory of images' },
    { value: 'csv', label: 'CSV Dataset', description: 'Tabular data in CSV format' },
    { value: 'custom', label: 'Custom Format', description: 'Auto-detect structure' }
  ]
  
  // Update selected format when datasetInfo changes
  React.useEffect(() => {
    if (datasetInfo?.structure?.type) {
      setSelectedFormat(datasetInfo.structure.type)
    }
  }, [datasetInfo])
  
  // Handle format change
  const handleFormatChange = (newFormat) => {
    setSelectedFormat(newFormat)
    if (onFormatChange) {
      onFormatChange(newFormat)
    }
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    setDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      await onFilesSelected(files)
    }
  }

  const pickFolder = async () => {
    if ('showDirectoryPicker' in window) {
      try {
        const dir = await window.showDirectoryPicker()
        const files = []
        const rootName = dir.name
        
        Swal.fire({
          icon: 'info',
          title: 'Collecting Files',
          text: `Starting to collect files from: ${rootName}`,
          timer: 2000,
          showConfirmButton: false
        })
        
        async function collect(handle, currentPath = '') {
          if (handle.kind === 'file') {
            const file = await handle.getFile()
            // Add webkitRelativePath property to preserve folder structure
            const relativePath = currentPath ? currentPath : file.name
            const fullPath = `${rootName}/${relativePath}`
            
            // Set webkitRelativePath property on the original file
            Object.defineProperty(file, 'webkitRelativePath', {
              value: fullPath,
              writable: false,
              enumerable: true,
              configurable: true
            })
            
            // Also store the full path for later use
            file._fullPath = fullPath
            
            files.push(file)
          } else if (handle.kind === 'directory') {
            // Recursively collect files from subdirectories
            for await (const [name, subHandle] of handle.entries()) {
              const newPath = currentPath ? `${currentPath}/${name}` : name
              await collect(subHandle, newPath)
            }
          }
        }
        
        // Start collecting from root
        for await (const [name, handle] of dir.entries()) {
          await collect(handle, name)
        }
        
        await onFilesSelected(files)
      } catch (error) {
        if (error.name !== 'AbortError') {
          await Swal.fire({
            icon: 'error',
            title: 'Error',
            text: `Failed to load folder: ${error.message}`,
            confirmButtonColor: '#ef4444'
          })
        }
      }
    } else {
      inputRef.current.click()
    }
  }

  const onInput = async (e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      await onFilesSelected(files)
    }
  }

  const handleRemoveDataset = async () => {
    const result = await Swal.fire({
      title: 'Remove Dataset?',
      text: 'This will clear the uploaded dataset from memory. You can upload a new one.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Yes, Remove',
      cancelButtonText: 'Cancel',
      confirmButtonColor: '#ef4444',
      cancelButtonColor: '#6b7280'
    })

    if (result.isConfirmed) {
      // Clear dataset from parent component
      await onFilesSelected(null, true) // Pass true flag to indicate removal
      await Swal.fire({
        icon: 'success',
        title: 'Dataset Removed',
        text: 'You can now upload a new dataset.',
        timer: 2000,
        showConfirmButton: false
      })
    }
  }

  return (
    <div className="bg-gray-100">
      {/* Stats Row - AdminLTE Style */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
        <div className="bg-gradient-to-br from-teal-500 to-cyan-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Upload Status</div>
              <div className="text-xl font-bold mt-1">{datasetInfo ? 'Ready' : 'Waiting'}</div>
            </div>
            <div className="text-3xl opacity-30">📁</div>
          </div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Total Files</div>
              <div className="text-xl font-bold mt-1">{datasetInfo?.fileCount || 0}</div>
            </div>
            <div className="text-3xl opacity-30">📄</div>
          </div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Classes</div>
              <div className="text-xl font-bold mt-1">
                {uploadProgress ? (
                  <span className="text-sm animate-pulse">Loading...</span>
                ) : (
                  datasetInfo?.structure?.classes?.length || 0
                )}
              </div>
            </div>
            <div className="text-3xl opacity-30">🏷️</div>
          </div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-red-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Format</div>
              <div className="text-lg font-bold mt-1">{datasetInfo?.structure?.type || 'N/A'}</div>
            </div>
            <div className="text-3xl opacity-30">🔍</div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Left Column - Upload Area */}
            <div className="lg:col-span-2 space-y-3">
              {/* Upload Card */}
              <div className="bg-white rounded-lg shadow-md overflow-hidden">
                <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-3 py-2">
                  <h3 className="font-bold flex items-center gap-2">
                    <span>📤</span>
                    Upload Dataset
                  </h3>
                </div>
                <div className={`p-4 border-4 border-dashed m-3 rounded-lg transition-all ${dragOver?'border-blue-500 bg-blue-50':'border-gray-300 bg-gray-50'}`}
                  onDragOver={(e)=>{e.preventDefault(); setDragOver(true)}}
                  onDragLeave={()=>setDragOver(false)}
                  onDrop={handleDrop}>
                  <div className="text-center">
                    <svg className="w-12 h-12 mx-auto mb-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <h3 className="text-lg font-bold text-gray-800 mb-1">Drag & Drop Your Dataset</h3>
                    <p className="text-gray-600 mb-3 text-xs">or click the button below</p>
                    <button className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 focus-ring font-semibold shadow-lg transform hover:scale-105 transition-all" onClick={pickFolder}>
                      <span className="flex items-center gap-2">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                        Select Folder
                      </span>
                    </button>
                    <input ref={inputRef} type="file" multiple webkitdirectory="true" directory="true" onChange={onInput} className="hidden" />
                  </div>
                </div>
              </div>

              {/* Dataset Format Selector - Only show after upload */}
              {datasetInfo && (
                <div className="bg-white rounded-lg shadow-md overflow-hidden">
                  <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-3 py-2">
                    <h3 className="font-bold flex items-center gap-2">
                      <span>📋</span>
                      Dataset Format
                    </h3>
                  </div>
                  <div className="p-4">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Select Format {datasetInfo.structure?.type && (
                        <span className="text-xs font-normal text-gray-500 ml-2">
                          (Auto-detected: {datasetFormats.find(f => f.value === datasetInfo.structure.type)?.label})
                        </span>
                      )}
                    </label>
                    <select
                      value={selectedFormat}
                      onChange={(e) => handleFormatChange(e.target.value)}
                      className="w-full px-4 py-2 bg-white text-gray-800 border-2 border-indigo-200 rounded-lg focus:outline-none focus:border-indigo-500 transition-colors text-sm font-medium"
                    >
                      <option value="" className="text-gray-800">Select format...</option>
                      {datasetFormats.map(format => (
                        <option key={format.value} value={format.value} className="text-gray-800">
                          {format.label} - {format.description}
                        </option>
                      ))}
                    </select>
                    {selectedFormat && (
                      <div className="mt-2 text-xs text-gray-600 bg-indigo-50 rounded px-3 py-2">
                        💡 <strong>Selected:</strong> {datasetFormats.find(f => f.value === selectedFormat)?.description}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Supported Formats Card */}
              <div className="bg-white rounded-lg shadow-md">
                <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-3 py-2">
                  <h3 className="font-bold flex items-center gap-2">
                    <span>✅</span>
                    Supported Formats
                  </h3>
                </div>
                <div className="p-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-2 border-l-4 border-blue-500">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xl">📸</span>
                        <h4 className="font-bold text-blue-600 text-xs">Computer Vision</h4>
                      </div>
                      <ul className="space-y-1 text-xs text-gray-700">
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>YOLO (Object Detection)</span>
                        </li>
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>Image Classification</span>
                        </li>
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>CIFAR-10/100</span>
                        </li>
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>Train/Test Splits</span>
                        </li>
                      </ul>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-2 border-l-4 border-green-500">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xl">📊</span>
                        <h4 className="font-bold text-green-600 text-xs">Data Science</h4>
                      </div>
                      <ul className="space-y-1 text-xs text-gray-700">
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>CSV Files</span>
                        </li>
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>Custom Formats</span>
                        </li>
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>Mixed Data</span>
                        </li>
                        <li className="flex items-center gap-1">
                          <span className="text-green-500 text-xs">✓</span>
                          <span>Any Structure</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Dataset Info - After Upload */}
              {datasetInfo && (
                <>
                  <div className="bg-white rounded-lg shadow-md">
                    <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-3 flex items-center justify-between">
                      <h3 className="font-bold text-lg flex items-center gap-2">
                        <span>✅</span>
                        Dataset Loaded Successfully
                      </h3>
                      <button 
                        onClick={handleRemoveDataset}
                        className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded-lg font-semibold text-sm transition-all shadow-md hover:shadow-lg flex items-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                        Remove Dataset
                      </button>
                    </div>
                    <div className="p-6">
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border-l-4 border-blue-500">
                          <div className="text-xs text-gray-600 uppercase font-semibold mb-1">Files</div>
                          <div className="text-2xl font-bold text-gray-800">{datasetInfo.fileCount}</div>
                        </div>
                        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4 border-l-4 border-purple-500">
                          <div className="text-xs text-gray-600 uppercase font-semibold mb-1">Classes</div>
                          <div className="text-2xl font-bold text-gray-800">
                            {uploadProgress ? (
                              <span className="text-lg text-purple-600 animate-pulse">Loading classes...</span>
                            ) : (
                              datasetInfo.structure?.classes?.length || 0
                            )}
                          </div>
                        </div>
                      </div>
                      {!uploadProgress && datasetInfo.structure?.classes && datasetInfo.structure.classes.length > 0 && (
                        <div className="mb-4">
                          <h4 className="font-semibold text-gray-700 mb-2 text-sm">Class Labels:</h4>
                          <div className="flex flex-wrap gap-2">
                            {datasetInfo.structure.classes.map((cls, idx) => (
                              <span key={idx} className="px-3 py-1 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded text-xs font-medium">
                                {cls}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {uploadProgress && (
                        <div className="mb-4 text-center">
                          <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-50 rounded-lg">
                            <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                            <span className="text-sm text-purple-700 font-medium">Detecting classes...</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Right Column - Folder Structure */}
            <div className="space-y-3">
              {/* Folder Structure Tree */}
              {datasetInfo && datasetInfo.structure && (
                <div className="bg-white rounded-lg shadow-md">
                  <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-3 py-2">
                    <h3 className="font-bold flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                      </svg>
                      Dataset Structure
                    </h3>
                  </div>
                  <div className="p-3">
                    <FolderTree structure={datasetInfo.structure} />
                  </div>
                </div>
              )}

              {/* Show upload progress when uploading */}
              {uploadProgress && uploadProgress.uploading && (
                <div className="bg-white rounded-lg shadow-md">
                  <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-3 py-2">
                    <h3 className="font-bold flex items-center gap-2">
                      <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Uploading Dataset...
                    </h3>
                  </div>
                  <div className="p-4">
                    <div className="mb-3">
                      <div className="flex justify-between text-sm text-gray-700 mb-2">
                        <span>Progress</span>
                        <span className="font-bold">{uploadProgress.percentage}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-indigo-600 h-3 rounded-full transition-all duration-300 ease-out"
                          style={{ width: `${uploadProgress.percentage}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div className="flex items-center gap-2">
                        <span>📦</span>
                        <span>{uploadProgress.filesUploaded} / {uploadProgress.totalFiles} files</span>
                      </div>
                      {uploadProgress.timeRemaining && (
                        <div className="flex items-center gap-2">
                          <span>⏱️</span>
                          <span>Est. time remaining: {uploadProgress.timeRemaining}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Show placeholder when no dataset */}
              {!datasetInfo && !uploadProgress?.uploading && (
                <div className="bg-white rounded-lg shadow-md">
                  <div className="bg-gradient-to-r from-gray-400 to-gray-600 text-white px-3 py-2">
                    <h3 className="font-bold flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                      </svg>
                      Dataset Structure
                    </h3>
                  </div>
                  <div className="p-4 text-center text-gray-500">
                    <svg className="w-16 h-16 mx-auto mb-2 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                    </svg>
                    <p className="text-sm">Upload a dataset to see folder structure</p>
                  </div>
                </div>
              )}
            </div>

          </div>
    </div>
  )
}