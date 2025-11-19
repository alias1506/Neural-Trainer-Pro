import React, { useState } from 'react'
import Swal from 'sweetalert2'

export default function ModelExport({ progress, datasetInfo }) {
  const [selectedFormat, setSelectedFormat] = useState('pytorch')
  const [exporting, setExporting] = useState(false)
  // Only show models that are fully trained (status === 'done')
  const hasModel = progress.status === 'done' && progress.modelPath

  const exportFormats = [
    { id: 'pytorch', name: 'PyTorch (.pth)', description: 'Native PyTorch format - Best for Python inference' },
    { id: 'onnx', name: 'ONNX (.onnx)', description: 'Open Neural Network Exchange - Cross-platform deployment' },
    { id: 'torchscript', name: 'TorchScript (.pt)', description: 'Production-optimized PyTorch - C++ deployment ready' },
    { id: 'coreml', name: 'CoreML (.mlmodel)', description: 'Apple CoreML - iOS and macOS native' },
    { id: 'tflite', name: 'TensorFlow Lite (.tflite)', description: 'Mobile-optimized - Android and embedded devices' }
  ]

  const handleDownload = async () => {
    if (!hasModel) {
      Swal.fire({ 
        icon: 'warning', 
        title: 'No Model Available', 
        text: 'Please train a model first before exporting.', 
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
      const numClasses = datasetInfo?.structure?.classes?.length || datasetInfo?.numClasses || 10

      // Create download URL with conversion parameters
      const downloadUrl = `http://localhost:3001/api/download-model?modelPath=${encodeURIComponent(progress.modelPath)}&format=${selectedFormat}&numClasses=${numClasses}`
      
      // Create a temporary anchor element to trigger download
      const link = document.createElement('a')
      link.href = downloadUrl
      
      // Set appropriate filename based on format
      const baseFilename = progress.modelPath.split(/[\\/]/).pop().replace('.pth', '')
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

      // Close loading and show success
      Swal.fire({
        icon: 'success',
        title: 'Download Started!',
        html: `
          <div class="text-left">
            <p class="mb-3 text-gray-700">Your model in ${format.name} format is being downloaded.</p>
            <p class="text-sm text-gray-600">Check your browser's download folder.</p>
            ${selectedFormat !== 'pytorch' ? '<p class="text-sm text-orange-600 mt-2">⚠️ Original .pth file will be deleted after conversion.</p>' : ''}
            <div class="mt-4 p-3 bg-green-50 rounded">
              <p class="text-sm font-semibold text-green-800 mb-2">Format:</p>
              <p class="text-xs text-green-700">${format.description}</p>
            </div>
          </div>
        `,
        confirmButtonColor: '#10b981',
        timer: 5000
      })
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

  return (
    <div className="bg-gray-100">
      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Model Status</p><p className="text-2xl font-bold mt-1">{hasModel ? '✅ Ready' : '⏳ Pending'}</p></div><div className="text-3xl opacity-80">🎯</div></div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Epochs</p><p className="text-2xl font-bold mt-1">{progress.epoch || 0}</p></div><div className="text-3xl opacity-80">📊</div></div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Accuracy</p><p className="text-2xl font-bold mt-1">{progress.valAcc ? `${(progress.valAcc * 100).toFixed(2)}%` : 'N/A'}</p></div><div className="text-3xl opacity-80">🎯</div></div>
        </div>
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Time</p><p className="text-2xl font-bold mt-1">{progress.timeMs ? `${Math.floor(progress.timeMs / 1000)}s` : 'N/A'}</p></div><div className="text-3xl opacity-80">⏱️</div></div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          {/* Model Info */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4">
              <h3 className="text-lg font-bold">📦 Model Information</h3>
            </div>
            <div className="p-6">
              {hasModel ? (
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Model Path</p>
                    <p className="text-sm font-mono bg-gray-100 p-3 rounded break-all">{progress.modelPath}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Dataset Format</p>
                      <p className="font-semibold text-gray-800">{datasetInfo?.structure?.type || 'Unknown'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Classes</p>
                      <p className="font-semibold text-gray-800">{datasetInfo?.structure?.classes?.length || datasetInfo?.numClasses || 'N/A'}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-6xl mb-4">🤖</div>
                  <p className="text-gray-600 font-medium mb-2">No Model Available</p>
                  <p className="text-sm text-gray-500">Train a model first to export it</p>
                </div>
              )}
            </div>
          </div>

          {/* Export Download Section */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white p-4">
              <h3 className="text-lg font-bold">📥 Download Model</h3>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {/* Format Selection */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Select Export Format
                  </label>
                  <select 
                    value={selectedFormat}
                    onChange={(e) => setSelectedFormat(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all bg-white text-gray-800 font-medium"
                    disabled={!hasModel}
                  >
                    {exportFormats.map((fmt) => (
                      <option key={fmt.id} value={fmt.id}>
                        {fmt.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Format Description */}
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                  <p className="text-sm text-gray-700">
                    {exportFormats.find(f => f.id === selectedFormat)?.description}
                  </p>
                </div>

                {/* Download Button */}
                <button
                  onClick={handleDownload}
                  disabled={!hasModel || exporting}
                  className={`w-full py-4 rounded-lg font-bold text-lg transition-all shadow-lg ${
                    !hasModel || exporting
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl transform hover:scale-[1.02]'
                  }`}
                >
                  {exporting ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Preparing Download...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download Model
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          {/* Training Summary */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white p-4"><h3 className="text-lg font-bold">📈 Summary</h3></div>
            <div className="p-4 space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded"><span className="text-sm text-gray-600">Total Epochs</span><span className="font-bold text-gray-800">{progress.totalEpochs || 0}</span></div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded"><span className="text-sm text-gray-600">Final Loss</span><span className="font-bold text-red-600">{progress.trainLoss ? progress.trainLoss.toFixed(4) : 'N/A'}</span></div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded"><span className="text-sm text-gray-600">Train Acc</span><span className="font-bold text-green-600">{progress.trainAcc ? `${(progress.trainAcc * 100).toFixed(2)}%` : 'N/A'}</span></div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded"><span className="text-sm text-gray-600">Val Acc</span><span className="font-bold text-blue-600">{progress.valAcc ? `${(progress.valAcc * 100).toFixed(2)}%` : 'N/A'}</span></div>
            </div>
          </div>

          {/* Status Card */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-gray-600 to-gray-800 text-white p-4"><h3 className="text-lg font-bold">🎯 Status</h3></div>
            <div className="p-4">
              <div className={`text-center py-4 rounded-lg ${hasModel ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>
                <div className="text-4xl mb-2">{hasModel ? '✅' : '⏳'}</div>
                <p className="font-bold text-lg">{hasModel ? 'Model Ready' : 'Train First'}</p>
                <p className="text-sm mt-1 opacity-80">{hasModel ? 'Select format above' : 'Go to Training Config'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}