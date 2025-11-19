import React, { useState, useEffect, useRef } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

export default function TrainingProgress({ progress, config, onCancel }) {
  const [elapsedTime, setElapsedTime] = useState(0)
  const startTimeRef = useRef(Date.now())
  
  // Update elapsed time every second during training
  useEffect(() => {
    if (progress.status === 'training') {
      const interval = setInterval(() => {
        setElapsedTime(Date.now() - startTimeRef.current)
      }, 1000)
      return () => clearInterval(interval)
    }
  }, [progress.status])
  
  // Prepare chart data for Chart.js
  const epochs = progress.lossHistory && progress.lossHistory.length > 0
    ? progress.lossHistory.map((_, idx) => idx + 1)
    : [0]
  
  const lossData = progress.lossHistory && progress.lossHistory.length > 0
    ? progress.lossHistory.map(loss => parseFloat(loss.toFixed(4)))
    : [0]
  
  const accuracyData = progress.accHistory && progress.accHistory.length > 0
    ? progress.accHistory.map(acc => parseFloat((acc * 100).toFixed(2)))
    : [0]
  const getStatusIcon = () => {
    if (progress.status === 'preparing') return '📦'
    if (progress.status === 'training') return '🔥'
    if (progress.status === 'done') return '✅'
    if (progress.status === 'error') return '❌'
    if (progress.status === 'cancelled') return '⏸️'
    return '⏳'
  }

  const getStatusColor = () => {
    if (progress.status === 'training') return 'text-green-600'
    if (progress.status === 'done') return 'text-blue-600'
    if (progress.status === 'error') return 'text-red-600'
    if (progress.status === 'cancelled') return 'text-yellow-600'
    return 'text-gray-600'
  }

  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    if (hours > 0) return `${hours}h ${minutes % 60}m`
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`
    return `${seconds}s`
  }

  const progressPercent = progress.totalEpochs > 0 
    ? Math.round((progress.epoch / progress.totalEpochs) * 100) 
    : 0

  return (
    <div className="bg-gray-100">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        {/* Current Epoch */}
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm font-medium">Current Epoch</p>
              <p className="text-3xl font-bold mt-1">{progress.epoch || 0}/{progress.totalEpochs || 0}</p>
            </div>
            <div className="bg-white/20 rounded-full p-3">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm1 11H9V7h2v6zm0-8H9V3h2v2z"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Train Loss */}
        <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-lg shadow-lg p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-100 text-sm font-medium">Train Loss</p>
              <p className="text-3xl font-bold mt-1">{progress.trainLoss ? progress.trainLoss.toFixed(4) : 'N/A'}</p>
            </div>
            <div className="bg-white/20 rounded-full p-3">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Train Accuracy */}
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow-lg p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm font-medium">Train Accuracy</p>
              <p className="text-3xl font-bold mt-1">{progress.trainAcc ? (progress.trainAcc * 100).toFixed(2) : 0}%</p>
            </div>
            <div className="bg-white/20 rounded-full p-3">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Time Elapsed */}
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow-lg p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100 text-sm font-medium">Time Elapsed</p>
              <p className="text-3xl font-bold mt-1">{formatTime(progress.timeMs || 0)}</p>
            </div>
            <div className="bg-white/20 rounded-full p-3">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd"/>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column - Main Progress Cards */}
        <div className="lg:col-span-2 space-y-4">
          {/* Progress Bar Card */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4">
              <h3 className="text-lg font-bold">Training Progress</h3>
            </div>
            <div className="p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  {getStatusIcon()} {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}
                </span>
                <span className="text-sm font-bold text-blue-600">{progressPercent}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-indigo-600 h-4 rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                  style={{ width: `${progressPercent}%` }}
                >
                  {progressPercent > 10 && (
                    <span className="text-xs font-bold text-white">{progressPercent}%</span>
                  )}
                </div>
              </div>
              {progress.message && (
                <p className="mt-3 text-sm text-gray-600">{progress.message}</p>
              )}
            </div>
          </div>

          {/* Validation Metrics Card */}
          {progress.status !== 'idle' && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-4">
                <h3 className="text-lg font-bold">Validation Metrics</h3>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Validation Loss</p>
                    <p className="text-2xl font-bold text-red-600">
                      {progress.valLoss ? progress.valLoss.toFixed(4) : 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Validation Accuracy</p>
                    <p className="text-2xl font-bold text-green-600">
                      {progress.valAcc ? (progress.valAcc * 100).toFixed(2) + '%' : 'N/A'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Training Chart - Always visible */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-indigo-500 to-blue-600 text-white p-4">
              <h3 className="text-lg font-bold flex items-center justify-between">
                <span>📈 Training Metrics Over Time</span>
                <div className="flex items-center gap-4">
                  {progress.status === 'training' && (
                    <span className="text-sm font-normal opacity-90 flex items-center gap-2">
                      ⏱️ {formatTime(elapsedTime)}
                    </span>
                  )}
                  {progress.lossHistory && progress.lossHistory.length > 0 && (
                    <span className="text-sm font-normal opacity-90">
                      {progress.lossHistory.length} epoch{progress.lossHistory.length !== 1 ? 's' : ''} recorded
                    </span>
                  )}
                </div>
              </h3>
            </div>
            <div className="p-6 bg-gradient-to-br from-gray-50 to-gray-100 relative rounded-b-lg" style={{ height: '500px' }}>
              <Line
                data={{
                  labels: epochs,
                  datasets: [
                    {
                      label: 'Training Loss',
                      data: lossData,
                      borderColor: '#ef4444',
                      backgroundColor: 'rgba(239, 68, 68, 0.1)',
                      borderWidth: 3,
                      pointRadius: 5,
                      pointHoverRadius: 7,
                      pointBackgroundColor: '#ef4444',
                      pointBorderColor: '#fff',
                      pointBorderWidth: 2,
                      tension: 0.4,
                      yAxisID: 'y',
                    },
                    {
                      label: 'Training Accuracy (%)',
                      data: accuracyData,
                      borderColor: '#10b981',
                      backgroundColor: 'rgba(16, 185, 129, 0.1)',
                      borderWidth: 3,
                      pointRadius: 5,
                      pointHoverRadius: 7,
                      pointBackgroundColor: '#10b981',
                      pointBorderColor: '#fff',
                      pointBorderWidth: 2,
                      tension: 0.4,
                      yAxisID: 'y1',
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: {
                    mode: 'index',
                    intersect: false,
                  },
                  plugins: {
                    legend: {
                      position: 'top',
                      labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                          size: 14,
                          weight: 'bold',
                        },
                      },
                    },
                    tooltip: {
                      backgroundColor: 'rgba(0, 0, 0, 0.8)',
                      padding: 12,
                      titleFont: {
                        size: 14,
                        weight: 'bold',
                      },
                      bodyFont: {
                        size: 13,
                      },
                      callbacks: {
                        label: function(context) {
                          let label = context.dataset.label || '';
                          if (label) {
                            label += ': ';
                          }
                          if (context.parsed.y !== null) {
                            label += context.parsed.y.toFixed(4);
                          }
                          return label;
                        }
                      }
                    },
                  },
                  scales: {
                    x: {
                      display: true,
                      title: {
                        display: true,
                        text: 'Epoch',
                        font: {
                          size: 14,
                          weight: 'bold',
                        },
                        color: '#1e293b',
                      },
                      grid: {
                        color: '#e2e8f0',
                        drawOnChartArea: true,
                      },
                      ticks: {
                        color: '#475569',
                        font: {
                          size: 12,
                          weight: '500',
                        },
                      },
                    },
                    y: {
                      type: 'linear',
                      display: true,
                      position: 'left',
                      title: {
                        display: true,
                        text: 'Loss',
                        font: {
                          size: 14,
                          weight: 'bold',
                        },
                        color: '#ef4444',
                      },
                      grid: {
                        color: '#e2e8f0',
                      },
                      ticks: {
                        color: '#ef4444',
                        font: {
                          size: 12,
                          weight: '500',
                        },
                      },
                      min: 0,
                      max: progress.lossHistory && progress.lossHistory.length > 0 ? Math.max(...lossData) * 1.2 : 10,
                    },
                    y1: {
                      type: 'linear',
                      display: true,
                      position: 'right',
                      title: {
                        display: true,
                        text: 'Accuracy (%)',
                        font: {
                          size: 14,
                          weight: 'bold',
                        },
                        color: '#10b981',
                      },
                      grid: {
                        drawOnChartArea: false,
                      },
                      ticks: {
                        color: '#10b981',
                        font: {
                          size: 12,
                          weight: '500',
                        },
                      },
                      min: 0,
                      max: 100,
                    },
                  },
                }}
              />
              {(!progress.lossHistory || progress.lossHistory.length === 0) && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="text-center bg-white/90 backdrop-blur-sm px-6 py-4 rounded-lg shadow-lg">
                    <p className="text-gray-600 font-medium">Waiting for training data...</p>
                    <p className="text-sm text-gray-500 mt-1">Chart will update in real-time</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Training History */}
          {progress.lossHistory && progress.lossHistory.length > 0 && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-purple-500 to-pink-600 text-white p-4">
                <h3 className="text-lg font-bold">Epoch History</h3>
              </div>
              <div className="p-6">
                <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
                  {progress.lossHistory.map((loss, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                      <span className="text-sm font-medium text-gray-700">Epoch {idx + 1}</span>
                      <div className="flex gap-4">
                        <span className="text-sm">
                          <span className="text-gray-500">Loss:</span>{' '}
                          <span className="font-bold text-red-600">{loss.toFixed(4)}</span>
                        </span>
                        <span className="text-sm">
                          <span className="text-gray-500">Acc:</span>{' '}
                          <span className="font-bold text-green-600">
                            {progress.accHistory[idx] ? (progress.accHistory[idx] * 100).toFixed(2) + '%' : 'N/A'}
                          </span>
                        </span>
                      </div>
                    </div>
                  )).reverse()}
                </div>
              </div>
            </div>
          )}

          {/* Control Card */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-gray-500 to-gray-600 text-white p-4">
              <h3 className="text-lg font-bold">Training Control</h3>
            </div>
            <div className="p-6">
              <button 
                className="w-full px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed font-semibold shadow-lg transition-all duration-200"
                disabled={progress.status !== 'training'}
                onClick={onCancel}
              >
                {progress.status === 'training' ? '⏸️ Cancel Training' : '⏸️ Training Not Active'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column - Info Panels */}
        <div className="space-y-4">
          {/* Configuration Info */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white p-4">
              <h3 className="text-sm font-bold">Configuration</h3>
            </div>
            <div className="p-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Epochs:</span>
                <span className="font-semibold text-gray-900">{config.epochs}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Batch Size:</span>
                <span className="font-semibold text-gray-900">{config.batchSize}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Learning Rate:</span>
                <span className="font-semibold text-gray-900">{config.learningRate}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Optimizer:</span>
                <span className="font-semibold text-gray-900">{config.optimizer}</span>
              </div>
            </div>
          </div>

          {/* Status Info */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-pink-400 to-rose-500 text-white p-4">
              <h3 className="text-sm font-bold">Status</h3>
            </div>
            <div className="p-4">
              <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${
                progress.status === 'training' ? 'bg-green-100 text-green-800' :
                progress.status === 'done' ? 'bg-blue-100 text-blue-800' :
                progress.status === 'error' ? 'bg-red-100 text-red-800' :
                progress.status === 'cancelled' ? 'bg-yellow-100 text-yellow-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {getStatusIcon()} {progress.status.toUpperCase()}
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-cyan-400 to-blue-500 text-white p-4">
              <h3 className="text-sm font-bold">Training Tips</h3>
            </div>
            <div className="p-4 space-y-2 text-xs text-gray-600">
              <p>✓ Monitor loss - should decrease</p>
              <p>✓ Check accuracy - should increase</p>
              <p>✓ Watch for overfitting (val loss increases)</p>
              <p>✓ Be patient - training takes time</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}