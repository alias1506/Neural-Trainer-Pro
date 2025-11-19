import React, { useState } from 'react'
import Swal from 'sweetalert2'

export default function TrainingHistory({ history, onLoad, onDelete }) {
  const [filterStatus, setFilterStatus] = useState('all')
  
  // Filter history based on status
  const filteredHistory = filterStatus === 'all' 
    ? history 
    : history.filter(h => h.status === filterStatus)
  
  // Get status badge
  const getStatusBadge = (status) => {
    const badges = {
      running: { color: 'bg-blue-100 text-blue-700 border-blue-300', icon: '🔄', text: 'Running' },
      done: { color: 'bg-green-100 text-green-700 border-green-300', icon: '✅', text: 'Completed' },
      cancelled: { color: 'bg-orange-100 text-orange-700 border-orange-300', icon: '🛑', text: 'Cancelled' },
      failed: { color: 'bg-red-100 text-red-700 border-red-300', icon: '❌', text: 'Failed' }
    }
    const badge = badges[status] || badges.done
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${badge.color}`}>
        {badge.icon} {badge.text}
      </span>
    )
  }
  
  const handleDelete = async (h) => {
    const result = await Swal.fire({
      title: 'Delete Training Session?',
      text: 'This action cannot be undone.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#ef4444',
      cancelButtonColor: '#6b7280',
      confirmButtonText: 'Yes, delete it',
      cancelButtonText: 'Cancel'
    })
    
    if (result.isConfirmed) {
      onDelete(h.id)
      await Swal.fire({
        icon: 'success',
        title: 'Deleted!',
        text: 'Training session has been deleted.',
        timer: 2000,
        showConfirmButton: false
      })
    }
  }

  return (
    <div className="bg-gray-100">
      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Total Sessions</p><p className="text-2xl font-bold mt-1">{history.length}</p></div><div className="text-3xl opacity-80">📚</div></div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Completed</p><p className="text-2xl font-bold mt-1">{history.filter(h => h.status === 'done').length}</p></div><div className="text-3xl opacity-80">✅</div></div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-red-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Running</p><p className="text-2xl font-bold mt-1">{history.filter(h => h.status === 'running').length}</p></div><div className="text-3xl opacity-80">�</div></div>
        </div>
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between"><div><p className="text-sm opacity-90">Best Accuracy</p><p className="text-2xl font-bold mt-1">{history.filter(h => h.status === 'done').length > 0 ? `${(Math.max(...history.filter(h => h.status === 'done').map(h => h.metrics.acc)) * 100).toFixed(2)}%` : 'N/A'}</p></div><div className="text-3xl opacity-80">🏆</div></div>
        </div>
      </div>

      {/* Training History List */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold">📜 Training History</h2>
            <div className="flex items-center gap-2">
              <span className="text-sm opacity-90">Filter:</span>
              <select 
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-1 rounded-lg bg-white/20 backdrop-blur-sm border border-white/30 text-white text-sm font-medium focus:outline-none focus:ring-2 focus:ring-white/50"
              >
                <option value="all" className="text-gray-900">All</option>
                <option value="running" className="text-gray-900">🔄 Running</option>
                <option value="done" className="text-gray-900">✅ Completed</option>
                <option value="cancelled" className="text-gray-900">🛑 Cancelled</option>
                <option value="failed" className="text-gray-900">❌ Failed</option>
              </select>
            </div>
          </div>
        </div>
        <div className="p-6">
          {filteredHistory.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📚</div>
              <p className="text-gray-600 font-medium text-lg mb-2">
                {filterStatus === 'all' ? 'No training history yet' : `No ${filterStatus} sessions`}
              </p>
              <p className="text-sm text-gray-500">
                {filterStatus === 'all' ? 'Complete your first training session to see it here' : 'Try selecting a different filter'}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredHistory.map(h => (
                <div key={h.id} className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors border border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-semibold text-gray-900 text-lg">
                          {new Date(h.date).toLocaleString()}
                        </div>
                        {getStatusBadge(h.status || 'done')}
                      </div>
                      {h.status === 'running' && h.currentEpoch > 0 && (
                        <div className="mb-2">
                          <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                            <span>Progress: Epoch {h.currentEpoch}/{h.config.epochs}</span>
                            <span>{Math.round((h.currentEpoch / h.config.epochs) * 100)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all animate-pulse"
                              style={{ width: `${(h.currentEpoch / h.config.epochs) * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                        <div className="flex items-center gap-2">
                          <span className="text-blue-600">📊</span>
                          <span className="text-sm text-gray-600">Epochs: <strong className="text-gray-800">{h.config.epochs}</strong></span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-purple-600">📦</span>
                          <span className="text-sm text-gray-600">Batch: <strong className="text-gray-800">{h.config.batchSize}</strong></span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-orange-600">⚡</span>
                          <span className="text-sm text-gray-600">LR: <strong className="text-gray-800">{h.config.learningRate}</strong></span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-green-600">🔧</span>
                          <span className="text-sm text-gray-600"><strong className="text-gray-800">{h.config.optimizer}</strong></span>
                        </div>
                      </div>
                      {h.status === 'failed' && h.error && (
                        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                          <p className="text-xs text-red-600 font-medium mb-1">Error:</p>
                          <p className="text-sm text-red-700">{h.error}</p>
                        </div>
                      )}
                      {(h.status === 'done' || h.status === 'running') && h.metrics && (
                        <div className="flex gap-6">
                          <div className="px-4 py-2 bg-green-50 rounded-lg">
                            <span className="text-xs text-green-600 font-medium">Accuracy</span>
                            <div className="text-lg font-bold text-green-700">{(h.metrics.acc*100).toFixed(2)}%</div>
                          </div>
                          <div className="px-4 py-2 bg-red-50 rounded-lg">
                            <span className="text-xs text-red-600 font-medium">Loss</span>
                            <div className="text-lg font-bold text-red-700">{h.metrics.loss.toFixed(4)}</div>
                          </div>
                        </div>
                      )}
                      {h.status === 'cancelled' && (
                        <div className="mt-2 p-3 bg-orange-50 border border-orange-200 rounded-lg">
                          <p className="text-sm text-orange-700">
                            ⚠️ Training was cancelled by user
                          </p>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center ml-4">
                      <button 
                        className="px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 transition-all shadow-md hover:shadow-lg font-medium flex items-center gap-2" 
                        onClick={() => handleDelete(h)}
                      >
                        🗑️ Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}