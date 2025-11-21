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
  // X-axis: Always show all epochs from 1 to totalEpochs
  const totalEpochs = config.epochs || progress.totalEpochs || 0
  const epochs = totalEpochs > 0 
    ? Array.from({ length: totalEpochs }, (_, idx) => idx + 1)
    : [0]
  
  // Y-axis: Fill data for completed epochs, null for remaining epochs
  const completedEpochs = progress.lossHistory ? progress.lossHistory.length : 0
  
  const trainLossData = totalEpochs > 0
    ? Array.from({ length: totalEpochs }, (_, idx) => 
        idx < completedEpochs && progress.lossHistory 
          ? parseFloat(progress.lossHistory[idx].toFixed(4))
          : null
      )
    : [0]
  
  const trainAccuracyData = totalEpochs > 0
    ? Array.from({ length: totalEpochs }, (_, idx) => 
        idx < completedEpochs && progress.accHistory
          ? parseFloat((progress.accHistory[idx] * 100).toFixed(2))
          : null
      )
    : [0]

  const valLossData = totalEpochs > 0
    ? Array.from({ length: totalEpochs }, (_, idx) => 
        idx < completedEpochs && progress.valLossHistory
          ? parseFloat(progress.valLossHistory[idx].toFixed(4))
          : null
      )
    : [0]
  
  const valAccuracyData = totalEpochs > 0
    ? Array.from({ length: totalEpochs }, (_, idx) => 
        idx < completedEpochs && progress.valAccHistory
          ? parseFloat((progress.valAccHistory[idx] * 100).toFixed(2))
          : null
      )
    : [0]

  const getStatusIcon = () => {
    if (progress.status === 'preparing') return '📦'
    if (progress.status === 'training') return '🔥'
    if (progress.status === 'done') return '✅'
    if (progress.status === 'error') return '❌'
    if (progress.status === 'cancelled') return '⏸️'
    return '⏳'
  }

  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`
    return `${seconds}s`
  }

  const progressPercent = progress.totalEpochs > 0 
    ? progress.status === 'done' || (progress.epoch >= progress.totalEpochs)
      ? 100
      : Math.round((progress.epoch / progress.totalEpochs) * 100) 
    : 0

  return (
    <div className="bg-gray-100">
      {/* Top Stats Row - Realtime Updates */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
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

        {/* Val Loss */}
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg shadow-lg p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-100 text-sm font-medium">Val Loss</p>
              <p className="text-3xl font-bold mt-1">{progress.valLoss ? progress.valLoss.toFixed(4) : '0.0000'}</p>
            </div>
            <div className="bg-white/20 rounded-full p-3">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293z"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Val Accuracy */}
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm font-medium">Val Accuracy</p>
              <p className="text-3xl font-bold mt-1">{progress.valAcc ? (progress.valAcc * 100).toFixed(2) : 0}%</p>
            </div>
            <div className="bg-white/20 rounded-full p-3">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column - Progress and Config */}
        <div className="lg:col-span-2 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Progress Bar Card */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-purple-500 to-indigo-600 text-white p-4">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <span>{getStatusIcon()}</span>
                  Training Progress
                </h3>
              </div>
              <div className="p-6">
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-semibold text-gray-700">Epoch {progress.epoch} / {progress.totalEpochs}</span>
                    <span className="font-bold text-purple-600">{progressPercent}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-6 overflow-hidden shadow-inner">
                    <div 
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                      style={{ width: `${progressPercent}%` }}
                    >
                      {progressPercent > 10 && <span className="text-xs font-bold text-white">{progressPercent}%</span>}
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-600">Status</span>
                    <span className="font-semibold text-gray-800 capitalize">{progress.status}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-600">Time Elapsed</span>
                    <span className="font-semibold text-gray-800">
                      {progress.status === 'idle' 
                        ? '00:00'
                        : progress.status === 'preparing' || (progress.totalEpochs > 0 && progress.epoch === 0)
                        ? 'Calculating...'
                        : formatTime(elapsedTime)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-600">Est. Time Remaining</span>
                    <span className="font-semibold text-gray-800">
                      {progress.status === 'idle' 
                        ? '00:00'
                        : progress.status === 'preparing' || (progress.totalEpochs > 0 && progress.epoch === 0)
                        ? 'Calculating...'
                        : progress.totalEpochs > 0 && progress.epoch > 0
                        ? formatTime((elapsedTime / progress.epoch) * (progress.totalEpochs - progress.epoch))
                        : '00:00'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Configuration Card */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-blue-500 to-cyan-600 text-white p-4">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <span>⚙️</span>
                  Configuration
                </h3>
              </div>
              <div className="p-6">
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded">
                    <span className="text-sm text-gray-600 font-medium">Epochs</span>
                    <span className="font-bold text-blue-700">{config.epochs}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-green-50 rounded">
                    <span className="text-sm text-gray-600 font-medium">Batch Size</span>
                    <span className="font-bold text-green-700">{config.batchSize}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-purple-50 rounded">
                    <span className="text-sm text-gray-600 font-medium">Learning Rate</span>
                    <span className="font-bold text-purple-700">{config.learningRate}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-orange-50 rounded">
                    <span className="text-sm text-gray-600 font-medium">Optimizer</span>
                    <span className="font-bold text-orange-700 capitalize">{config.optimizer}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Combined Chart - Single Graph with All Metrics */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <span>📈</span>
                Training Metrics - Real-time
              </h3>
            </div>
            <div className="p-6">
              {(progress.status === 'preparing' || progress.status === 'training' || progress.status === 'done') ? (
                <div className="relative">
                  <Line
                    data={{
                      labels: epochs.length > 0 ? epochs : [0],
                      datasets: [
                        {
                          label: 'Train Loss',
                          data: trainLossData,
                          borderColor: 'rgb(239, 68, 68)',
                          backgroundColor: 'rgba(239, 68, 68, 0.1)',
                          fill: false,
                          tension: 0.4,
                          pointRadius: 0,
                          pointHoverRadius: 6,
                          yAxisID: 'y-loss',
                          spanGaps: false
                        },
                        {
                          label: 'Val Loss',
                          data: valLossData,
                          borderColor: 'rgb(251, 146, 60)',
                          backgroundColor: 'rgba(251, 146, 60, 0.1)',
                          fill: false,
                          tension: 0.4,
                          pointRadius: 0,
                          pointHoverRadius: 6,
                          yAxisID: 'y-loss',
                          spanGaps: false
                        },
                        {
                          label: 'Train Accuracy',
                          data: trainAccuracyData,
                          borderColor: 'rgb(34, 197, 94)',
                          backgroundColor: 'rgba(34, 197, 94, 0.1)',
                          fill: false,
                          tension: 0.4,
                          pointRadius: 0,
                          pointHoverRadius: 6,
                          yAxisID: 'y-accuracy',
                          spanGaps: false
                        },
                        {
                          label: 'Val Accuracy',
                          data: valAccuracyData,
                          borderColor: 'rgb(59, 130, 246)',
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          fill: false,
                          tension: 0.4,
                          pointRadius: 0,
                          pointHoverRadius: 6,
                          yAxisID: 'y-accuracy',
                          spanGaps: false
                        }
                      ]
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: true,
                      animation: {
                        duration: 0,
                        x: {
                          duration: 0
                        },
                        y: {
                          duration: 0
                        }
                      },
                      interaction: {
                        mode: 'index',
                        intersect: false
                      },
                      plugins: {
                        legend: {
                          display: true,
                          position: 'top',
                          labels: {
                            usePointStyle: true,
                            padding: 15
                          }
                        },
                        tooltip: {
                          mode: 'index',
                          intersect: false,
                          callbacks: {
                            label: function(context) {
                              const label = context.dataset.label || ''
                              const value = context.parsed.y
                              if (label.includes('Accuracy')) {
                                return label + ': ' + value.toFixed(2) + '%'
                              }
                              return label + ': ' + value.toFixed(4)
                            }
                          }
                        }
                      },
                      scales: {
                        x: {
                          title: {
                            display: true,
                            text: 'Epoch'
                          }
                        },
                        'y-loss': {
                          type: 'linear',
                          position: 'left',
                          beginAtZero: true,
                          title: {
                            display: true,
                            text: 'Loss',
                            color: 'rgb(239, 68, 68)'
                          },
                          ticks: {
                            color: 'rgb(239, 68, 68)'
                          }
                        },
                        'y-accuracy': {
                          type: 'linear',
                          position: 'right',
                          beginAtZero: true,
                          max: 100,
                          title: {
                            display: true,
                            text: 'Accuracy (%)',
                            color: 'rgb(34, 197, 94)'
                          },
                          ticks: {
                            color: 'rgb(34, 197, 94)'
                          },
                          grid: {
                            drawOnChartArea: false
                          }
                        }
                      }
                    }}
                    height={200}
                  />
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <div className="text-5xl mb-2">📊</div>
                  <p className="text-sm">Training data will appear here...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column - Control Panel */}
        <div className="space-y-4">
          {/* Control Card */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-red-500 to-pink-600 text-white p-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <span>🎮</span>
                Controls
              </h3>
            </div>
            <div className="p-6">
              <button
                onClick={onCancel}
                disabled={progress.status !== 'training'}
                className={`w-full py-3 rounded-lg font-bold transition-all shadow-lg ${
                  progress.status === 'training'
                    ? 'bg-gradient-to-r from-red-600 to-red-700 text-white hover:from-red-700 hover:to-red-800 hover:shadow-xl transform hover:scale-105'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {progress.status === 'training' ? '⏸️ Stop Training' : '⏹️ Not Running'}
              </button>
              
              {progress.status === 'training' && (
                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-xs text-yellow-800">
                    <strong>Note:</strong> Stopping will save current progress
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Metrics Summary Card */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-gray-600 to-gray-800 text-white p-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <span>📊</span>
                Current Metrics
              </h3>
            </div>
            <div className="p-4 space-y-3">
              <div className="bg-red-50 p-3 rounded-lg border-l-4 border-red-500">
                <p className="text-xs text-red-600 font-semibold mb-1">TRAIN LOSS</p>
                <p className="text-2xl font-bold text-red-700">{progress.trainLoss ? progress.trainLoss.toFixed(4) : 'N/A'}</p>
              </div>
              <div className="bg-green-50 p-3 rounded-lg border-l-4 border-green-500">
                <p className="text-xs text-green-600 font-semibold mb-1">TRAIN ACCURACY</p>
                <p className="text-2xl font-bold text-green-700">{progress.trainAcc ? `${(progress.trainAcc * 100).toFixed(2)}%` : 'N/A'}</p>
              </div>
              <div className="bg-orange-50 p-3 rounded-lg border-l-4 border-orange-500">
                <p className="text-xs text-orange-600 font-semibold mb-1">VAL LOSS</p>
                <p className="text-2xl font-bold text-orange-700">{progress.valLoss ? progress.valLoss.toFixed(4) : 'N/A'}</p>
              </div>
              <div className="bg-blue-50 p-3 rounded-lg border-l-4 border-blue-500">
                <p className="text-xs text-blue-600 font-semibold mb-1">VAL ACCURACY</p>
                <p className="text-2xl font-bold text-blue-700">{progress.valAcc ? `${(progress.valAcc * 100).toFixed(2)}%` : 'N/A'}</p>
              </div>
            </div>
          </div>

          {/* Epoch History Table (only if has data) */}
          {progress.lossHistory && progress.lossHistory.length > 0 && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-indigo-500 to-blue-600 text-white p-4">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <span>📜</span>
                  Epoch History
                </h3>
              </div>
              <div className="overflow-auto max-h-64">
                <table className="w-full text-sm">
                  <thead className="bg-gray-100 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left font-semibold text-gray-700">Epoch</th>
                      <th className="px-3 py-2 text-right font-semibold text-gray-700">Train Loss</th>
                      <th className="px-3 py-2 text-right font-semibold text-gray-700">Train Acc</th>
                      <th className="px-3 py-2 text-right font-semibold text-gray-700">Val Loss</th>
                      <th className="px-3 py-2 text-right font-semibold text-gray-700">Val Acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {progress.lossHistory.map((loss, idx) => (
                      <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="px-3 py-2 font-semibold text-gray-700">{idx + 1}</td>
                        <td className="px-3 py-2 text-right">
                          <span className="font-bold text-red-600">{loss.toFixed(4)}</span>
                        </td>
                        <td className="px-3 py-2 text-right">
                          <span className="font-bold text-green-600">
                            {progress.accHistory && progress.accHistory[idx] !== undefined
                              ? `${(progress.accHistory[idx] * 100).toFixed(2)}%`
                              : 'N/A'}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right">
                          {progress.valLossHistory && progress.valLossHistory[idx] !== undefined && (
                            <span className="font-bold text-orange-600">{progress.valLossHistory[idx].toFixed(4)}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {progress.valAccHistory && progress.valAccHistory[idx] !== undefined && (
                            <span className="font-bold text-blue-600">
                              {(progress.valAccHistory[idx] * 100).toFixed(2)}%
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}