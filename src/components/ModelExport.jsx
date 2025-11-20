import React, { useState, useEffect } from 'react'
import Swal from 'sweetalert2'
import axios from 'axios'

export default function ModelExport({ progress, datasetInfo, onModelExported, onAllModelsCleared }) {
  const [selectedFormat, setSelectedFormat] = useState('pytorch')
  const [exporting, setExporting] = useState(false)
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [loading, setLoading] = useState(true)
  const [modelMetadata, setModelMetadata] = useState({})

  // Fetch available models on component mount (silently)
  useEffect(() => {
    fetchAvailableModels()
  }, [])

  // Refresh model list when training completes
  useEffect(() => {
    if (progress?.status === 'done') {
      // Wait a bit for file system to sync, then refresh
      const timer = setTimeout(() => {
        fetchAvailableModels()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [progress?.status])

  // Auto-select the most recent model when available
  useEffect(() => {
    if (availableModels.length > 0 && !selectedModel) {
      setSelectedModel(availableModels[0])
    }
  }, [availableModels])

  // Load model-specific metadata from training history
  useEffect(() => {
    const storedHistory = sessionStorage.getItem('trainingHistory')
    if (storedHistory) {
      try {
        const history = JSON.parse(storedHistory)
        const metadata = {}
        history.forEach(h => {
          if (h.modelPath && h.status === 'done') {
            // Extract filename from modelPath for matching
            const modelFilename = h.modelPath.split('/').pop().split('\\').pop()
            // Store metadata indexed by filename for flexible matching
            // Get final values from history arrays (last epoch)
            const finalTrainLoss = h.lossHistory && h.lossHistory.length > 0 
              ? h.lossHistory[h.lossHistory.length - 1] 
              : (h.metrics?.loss || h.metrics?.trainLoss || 0)
            const finalTrainAcc = h.accHistory && h.accHistory.length > 0
              ? h.accHistory[h.accHistory.length - 1]
              : (h.metrics?.trainAcc || h.metrics?.acc || 0)
            const finalValAcc = h.valAccHistory && h.valAccHistory.length > 0
              ? h.valAccHistory[h.valAccHistory.length - 1]
              : (h.metrics?.valAcc || 0)
            
            metadata[modelFilename] = {
              totalEpochs: h.currentEpoch || h.config?.epochs || 0,
              trainLoss: finalTrainLoss,
              trainAcc: finalTrainAcc,
              valAcc: finalValAcc,
              config: h.config
            }
          }
        })
        setModelMetadata(metadata)
      } catch (e) {
        // Silently handle training history errors
      }
    }
  }, [availableModels])

  const fetchAvailableModels = async () => {
    try {
      setLoading(true)
      const response = await axios.get('http://localhost:3001/api/list-models')
      const models = response.data.models || []
      // Force update by creating new array reference
      setAvailableModels([...models])
    } catch (error) {
      // Silently handle 404 or server errors - just show empty list
      setAvailableModels([])
    } finally {
      setLoading(false)
    }
  }

  const formatModelName = (fileName) => {
    // Extract date from filename if present (e.g., trained_model_20231120_143022.pth)
    const match = fileName.match(/(\d{8}_\d{6})/)
    if (match) {
      const dateStr = match[1]
      const year = dateStr.substring(0, 4)
      const month = dateStr.substring(4, 6)
      const day = dateStr.substring(6, 8)
      const hour = dateStr.substring(9, 11)
      const minute = dateStr.substring(11, 13)
      const second = dateStr.substring(13, 15)
      
      const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`)
      const formattedDate = date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
      
      return `TrainedModel (${formattedDate}).pth`
    }
    
    return fileName
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  const exportFormats = [
    { id: 'pytorch', name: 'PyTorch (.pth)', description: 'Native PyTorch format - Best for Python inference' },
    { id: 'onnx', name: 'ONNX (.onnx)', description: 'Open Neural Network Exchange - Cross-platform deployment' },
    { id: 'torchscript', name: 'TorchScript (.pt)', description: 'Production-optimized PyTorch - C++ deployment ready' },
    { id: 'coreml', name: 'CoreML (.mlmodel)', description: 'Apple CoreML - iOS and macOS native' },
    { id: 'tflite', name: 'TensorFlow Lite (.tflite)', description: 'Mobile-optimized - Android and embedded devices' }
  ]

  const handleDownload = async () => {
    if (!selectedModel) {
      Swal.fire({ 
        icon: 'warning', 
        title: 'No Model Selected', 
        text: 'Please select a model to export.', 
        confirmButtonColor: '#3b82f6' 
      })
      return
    }

    const format = exportFormats.find(f => f.id === selectedFormat)
    
    setExporting(true)

    try {
      // Show loading message with conversion info
      Swal.fire({
        title: selectedFormat === 'pytorch' ? 'Preparing Download...' : `Converting to ${format.name}...`,
        html: selectedFormat === 'pytorch' 
          ? 'Please wait while we prepare your model file.'
          : `Converting your PyTorch model to ${format.name} format. This may take a moment...`,
        allowOutsideClick: false,
        didOpen: () => {
          Swal.showLoading()
        }
      })

      // Get number of classes from dataset info
      const numClasses = datasetInfo?.numClasses || datasetInfo?.structure?.classes?.length || 10

      // Create download URL with conversion parameters (use relative path)
      const downloadUrl = `http://localhost:3001/api/download-model?modelPath=${encodeURIComponent(selectedModel.path)}&format=${selectedFormat}&numClasses=${numClasses}`
      
      // Create a temporary anchor element to trigger download
      const link = document.createElement('a')
      link.href = downloadUrl
      
      // Set appropriate filename based on format
      const baseFilename = selectedModel.name.replace('.pth', '')
      const formatExtensions = {
        'pytorch': '.pth',
        'onnx': '.onnx',
        'torchscript': '.pt',
        'coreml': '.mlmodel',
        'tflite': '.tflite'
      }
      link.download = `${baseFilename}${formatExtensions[selectedFormat]}`
      
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      // Close loading and show success immediately
      Swal.fire({
        icon: 'success',
        title: 'Download Started!',
        html: `
          <div class="text-left">
            <p class="mb-3 text-gray-700">Your model in ${format.name} format is being downloaded.</p>
            <p class="text-sm text-gray-600">Check your browser's download folder.</p>
            <div class="mt-4 p-3 bg-green-50 rounded">
              <p class="text-sm font-semibold text-green-800 mb-2">Format:</p>
              <p class="text-xs text-green-700">${format.description}</p>
            </div>
          </div>
        `,
        confirmButtonColor: '#10b981',
        timer: 3000
      })

      // Refresh model list in background after a short delay
      setTimeout(async () => {
        // Notify parent component about the export (for history update)
        if (onModelExported) {
          onModelExported(selectedModel.path, selectedFormat)
        }
        
        // Refresh model list after export (model will be deleted from server)
        const response = await axios.get('http://localhost:3001/api/list-models')
        const updatedModels = response.data.models || []
        setAvailableModels(updatedModels)
        
        // Clear selected model
        setSelectedModel(null)
        
        // If no models left, notify parent to reset state
        if (updatedModels.length === 0 && onAllModelsCleared) {
          onAllModelsCleared()
        }
      }, 500)
    } catch (error) {
      // Try to get more details from the error response
      let errorDetails = error.message || 'Failed to download model'
      let installHint = ''
      
      if (selectedFormat === 'onnx') {
        installHint = '<br><br><b>Install required package:</b><br><code>pip install onnx</code>'
      } else if (selectedFormat === 'torchscript') {
        installHint = '<br><br><b>TorchScript uses built-in PyTorch - no extra packages needed</b>'
      } else if (selectedFormat === 'coreml') {
        installHint = '<br><br><b>Install required package:</b><br><code>pip install coremltools</code>'
      } else if (selectedFormat === 'tflite') {
        installHint = '<br><br><b>Install required packages:</b><br><code>pip install onnx onnx-tf tensorflow</code>'
      }
      
      Swal.fire({
        icon: 'error',
        title: 'Export Failed',
        html: `
          <div class="text-left">
            <p class="mb-3 text-gray-700">${errorDetails}</p>
            ${installHint}
          </div>
        `,
        confirmButtonColor: '#ef4444',
        width: 600
      })
    } finally {
      setExporting(false)
    }
  }

  const getFormatIcon = (formatId) => {
    const icons = {
      pytorch: '🔥',
      onnx: '🌐',
      torchscript: '⚡',
      coreml: '🍎',
      tflite: '📱'
    }
    return icons[formatId] || '📦'
  }

  const getFormatColor = (formatId) => {
    const colors = {
      pytorch: 'from-orange-500 to-red-600',
      onnx: 'from-blue-500 to-cyan-600',
      torchscript: 'from-yellow-500 to-orange-600',
      coreml: 'from-gray-700 to-gray-900',
      tflite: 'from-green-500 to-emerald-600'
    }
    return colors[formatId] || 'from-gray-500 to-gray-700'
  }

  // Get current model's metadata by matching filename
  const currentMetadata = selectedModel 
    ? (modelMetadata[selectedModel.name] || modelMetadata[selectedModel.path] || {}) 
    : {}

  return (
    <div className="bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen -m-6 p-6">
      {/* Hero Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 rounded-2xl shadow-2xl p-8 mb-6 text-white overflow-hidden relative">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="relative z-10">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-extrabold mb-2 flex items-center gap-3">
                <span className="text-5xl">🚀</span>
                Model Export Center
              </h1>
              <p className="text-indigo-100 text-lg">
                Convert and download your trained models in multiple formats
              </p>
            </div>
            <div className="hidden md:block text-8xl opacity-20">📦</div>
          </div>
          
          {/* Quick Stats Bar */}
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
              <div className="text-white/80 text-xs font-medium mb-1">Available Models</div>
              <div className="text-3xl font-bold">{availableModels.length}</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
              <div className="text-white/80 text-xs font-medium mb-1">Selected Format</div>
              <div className="text-2xl font-bold flex items-center gap-2">
                <span>{getFormatIcon(selectedFormat)}</span>
                {selectedFormat.toUpperCase()}
              </div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
              <div className="text-white/80 text-xs font-medium mb-1">Model Classes</div>
              <div className="text-3xl font-bold">{datasetInfo?.numClasses || '-'}</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
              <div className="text-white/80 text-xs font-medium mb-1">Status</div>
              <div className="text-2xl font-bold">{selectedModel ? '✅ Ready' : '⏳ Select'}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Section - Model Selection */}
        <div className="xl:col-span-2 space-y-6">
          {/* Available Models */}
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-200">
            <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-3xl">📦</span>
                <div>
                  <h2 className="text-xl font-bold text-white">Your Trained Models</h2>
                  <p className="text-blue-100 text-sm">Select a model to export</p>
                </div>
              </div>
              <button
                onClick={fetchAvailableModels}
                className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-all text-white font-medium flex items-center gap-2 border border-white/30"
                title="Refresh model list"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </button>
            </div>
            
            <div className="p-6">
              {loading ? (
                <div className="text-center py-16">
                  <div className="inline-block animate-spin text-6xl mb-4">⚙️</div>
                  <p className="text-gray-600 font-medium text-lg">Loading your models...</p>
                </div>
              ) : availableModels.length === 0 ? (
                <div className="text-center py-16 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl border-2 border-dashed border-gray-300">
                  <div className="text-8xl mb-4">🎯</div>
                  <p className="text-gray-700 font-bold text-xl mb-2">No Models Available</p>
                  <p className="text-gray-500">Train a model first to see it here</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-2 max-h-[400px] overflow-y-auto overflow-x-hidden">
                  {availableModels.map((model) => (
                    <div
                      key={model.path}
                      onClick={() => setSelectedModel(model)}
                      className={`group relative p-3 rounded-lg cursor-pointer transition-all duration-200 border ${
                        selectedModel?.path === model.path
                          ? 'border-blue-500 bg-blue-50 shadow-md'
                          : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-sm'
                      }`}
                    >
                      <div className="flex items-center gap-3 overflow-hidden">
                        <div className="flex-1 min-w-0">
                          <p className="font-semibold text-sm text-gray-800 truncate">
                            {formatModelName(model.name)}
                          </p>
                          <div className="flex items-center gap-3 text-xs text-gray-500 mt-1 flex-wrap">
                            <span className="flex items-center gap-1 whitespace-nowrap">
                              <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                              </svg>
                              {formatFileSize(model.size)}
                            </span>
                            <span className="flex items-center gap-1 whitespace-nowrap">
                              <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              {new Date(model.createdAt).toLocaleString()}
                            </span>
                          </div>
                        </div>
                        {selectedModel?.path === model.path && (
                          <div className="flex-shrink-0">
                            <span className="bg-blue-500 text-white text-xs font-bold px-2 py-1 rounded whitespace-nowrap">
                              SELECTED
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Export Action Card */}
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-200">
            <div className="bg-gradient-to-r from-green-600 to-emerald-700 p-5">
              <div className="flex items-center gap-3">
                <span className="text-3xl">📥</span>
                <div>
                  <h2 className="text-xl font-bold text-white">Export & Download</h2>
                  <p className="text-green-100 text-sm">Choose format and download your model</p>
                </div>
              </div>
            </div>
            
            <div className="p-6">
              <div className="space-y-4">
                {/* Format Selection Dropdown */}
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">
                    Export Format
                  </label>
                  <div className="relative">
                    <select
                      value={selectedFormat}
                      onChange={(e) => setSelectedFormat(e.target.value)}
                      disabled={!selectedModel}
                      className="w-full px-4 py-3 pr-10 border-2 border-gray-300 rounded-xl focus:border-green-500 focus:ring-4 focus:ring-green-200 outline-none transition-all bg-white text-gray-800 font-medium appearance-none disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      {exportFormats.map((fmt) => (
                        <option key={fmt.id} value={fmt.id}>
                          {getFormatIcon(fmt.id)} {fmt.name}
                        </option>
                      ))}
                    </select>
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                      <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Format Description */}
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-lg">
                  <p className="text-sm text-gray-700 flex items-start gap-2">
                    <span className="text-xl">{getFormatIcon(selectedFormat)}</span>
                    <span>{exportFormats.find(f => f.id === selectedFormat)?.description}</span>
                  </p>
                </div>

                {/* Download Button */}
                <button
                  onClick={handleDownload}
                  disabled={!selectedModel || exporting}
                  className={`w-full py-4 px-6 rounded-xl font-bold text-lg transition-all duration-300 flex items-center justify-center gap-3 ${
                    !selectedModel || exporting
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700 shadow-lg hover:shadow-xl transform hover:scale-[1.02]'
                  }`}
                >
                  {exporting ? (
                    <>
                      <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Preparing Download...
                    </>
                  ) : (
                    <>
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download Model
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Section - Summary */}
        <div className="space-y-6">
          {/* Training Metrics Summary */}
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-200">
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-4">
              <div className="flex items-center gap-2">
                <span className="text-2xl">📊</span>
                <h3 className="text-lg font-bold text-white">Model Summary</h3>
              </div>
            </div>
            {selectedModel ? (
              <div className="p-4 space-y-3">
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-700">Total Epochs</span>
                    <span className="text-2xl font-bold text-indigo-600">{currentMetadata.totalEpochs || 0}</span>
                  </div>
                </div>
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-xl border border-green-200">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-700">Train Accuracy</span>
                    <span className="text-xl font-bold text-green-600">{currentMetadata.trainAcc ? `${(currentMetadata.trainAcc * 100).toFixed(2)}%` : 'N/A'}</span>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-xs text-gray-500 mb-1">Model File</p>
                  <p className="text-sm font-semibold text-gray-700 break-all">{formatModelName(selectedModel.name)}</p>
                  <p className="text-xs text-gray-500 mt-2">Size: {formatFileSize(selectedModel.size)}</p>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center">
                <div className="text-6xl mb-3">📋</div>
                <p className="text-gray-500 font-medium">Select a model to view summary</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}