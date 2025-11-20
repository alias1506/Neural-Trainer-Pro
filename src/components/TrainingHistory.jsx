import React, { useState } from 'react'
import Swal from 'sweetalert2'

export default function TrainingHistory({ history, onLoad, onDelete }) {
  const [filterStatus, setFilterStatus] = useState('all')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 10
  
  const handleClearAll = async () => {
    const result = await Swal.fire({
      title: 'Clear All History?',
      text: 'This will delete all training sessions permanently!',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#ef4444',
      cancelButtonColor: '#6b7280',
      confirmButtonText: 'Yes, clear all!',
      cancelButtonText: 'Cancel'
    })
    
    if (result.isConfirmed) {
      history.forEach(h => onDelete(h.id))
      await Swal.fire({
        icon: 'success',
        title: 'Cleared!',
        text: 'All training history has been deleted.',
        timer: 2000,
        showConfirmButton: false
      })
      setCurrentPage(1)
    }
  }
  
  // Filter history based on status
  const filteredHistory = filterStatus === 'all' 
    ? history 
    : history.filter(h => h.status === filterStatus)
  
  // Pagination
  const totalPages = Math.ceil(filteredHistory.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedHistory = filteredHistory.slice(startIndex, startIndex + itemsPerPage)
  
  // Reset to page 1 when filter changes
  React.useEffect(() => {
    setCurrentPage(1)
  }, [filterStatus])
  
  // Get status badge
  const getStatusBadge = (status, exportedFormat = null) => {
    const badges = {
      running: { color: 'bg-blue-100 text-blue-700 border-blue-300', icon: '🔄', text: 'Running' },
      done: { color: 'bg-green-100 text-green-700 border-green-300', icon: '✅', text: exportedFormat ? `Exported (${exportedFormat.toUpperCase()})` : 'Trained' },
      cancelled: { color: 'bg-orange-100 text-orange-700 border-orange-300', icon: '🛑', text: 'Cancelled' },
      failed: { color: 'bg-red-100 text-red-700 border-red-300', icon: '❌', text: 'Failed' }
    }
    const badge = badges[status] || badges.done
    return (
      <span className={`px-3 py-2 rounded-full text-xs font-bold border-2 ${badge.color} shadow-sm inline-flex items-center gap-1 whitespace-nowrap`}>
        <span>{badge.icon}</span>
        <span>{badge.text}</span>
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
    <div className="bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen p-4">
      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl shadow-lg p-5 border-l-4 border-blue-500 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-medium">Total Sessions</p>
              <p className="text-3xl font-bold text-gray-800 mt-1">{history.length}</p>
            </div>
            <div className="text-4xl">📚</div>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-5 border-l-4 border-green-500 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-medium">Completed</p>
              <p className="text-3xl font-bold text-green-600 mt-1">{history.filter(h => h.status === 'done').length}</p>
            </div>
            <div className="text-4xl">✅</div>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-5 border-l-4 border-orange-500 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-medium">Running</p>
              <p className="text-3xl font-bold text-orange-600 mt-1">{history.filter(h => h.status === 'running').length}</p>
            </div>
            <div className="text-4xl">🔥</div>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-5 border-l-4 border-purple-500 hover:shadow-xl transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-medium">Best Accuracy</p>
              <p className="text-3xl font-bold text-purple-600 mt-1">{history.filter(h => h.status === 'done').length > 0 ? `${(Math.max(...history.filter(h => h.status === 'done').map(h => h.metrics.acc)) * 100).toFixed(2)}%` : 'N/A'}</p>
            </div>
            <div className="text-4xl">🏆</div>
          </div>
        </div>
      </div>

      {/* Training History List */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-6">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl font-bold">📜 Training History</h2>
              <p className="text-blue-100 text-sm mt-1">View and manage your training sessions</p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-sm font-medium">Filter:</label>
              <select 
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-4 py-2 rounded-lg bg-white/20 backdrop-blur-sm border-2 border-white/30 text-white font-semibold focus:outline-none focus:ring-2 focus:ring-white/50 cursor-pointer hover:bg-white/30 transition-all"
              >
                <option value="all" className="text-gray-900 font-semibold">All Sessions</option>
                <option value="running" className="text-gray-900 font-semibold">🔄 Running</option>
                <option value="done" className="text-gray-900 font-semibold">✅ Completed</option>
                <option value="cancelled" className="text-gray-900 font-semibold">🛑 Cancelled</option>
                <option value="failed" className="text-gray-900 font-semibold">❌ Failed</option>
              </select>
              <button
                onClick={handleClearAll}
                disabled={history.length === 0}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed rounded-lg text-white font-bold flex items-center gap-2 transition-all"
              >
                🗑️ Clear All
              </button>
            </div>
          </div>
        </div>
        <div className="p-6">
          {filteredHistory.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-8xl mb-6">📚</div>
              <p className="text-gray-700 font-bold text-2xl mb-3">
                {filterStatus === 'all' ? 'No training history yet' : `No ${filterStatus} sessions`}
              </p>
              <p className="text-gray-500 text-lg">
                {filterStatus === 'all' ? 'Complete your first training session to see it here' : 'Try selecting a different filter'}
              </p>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="overflow-x-visible">
                <table className="w-full">
                  <thead className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-bold">ID</th>
                      <th className="px-4 py-3 text-left text-sm font-bold">Date</th>
                      <th className="px-4 py-3 text-left text-sm font-bold">Status</th>
                      <th className="px-4 py-3 text-right text-sm font-bold">Epochs</th>
                      <th className="px-4 py-3 text-right text-sm font-bold">Batch</th>
                      <th className="px-4 py-3 text-right text-sm font-bold">LR</th>
                      <th className="px-4 py-3 text-left text-sm font-bold">Optimizer</th>
                      <th className="px-4 py-3 text-right text-sm font-bold">Accuracy</th>
                      <th className="px-4 py-3 text-right text-sm font-bold">Loss</th>
                      <th className="px-4 py-3 text-center text-sm font-bold">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {paginatedHistory.map(h => (
                      <tr key={h.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3">
                          <span className="inline-block bg-blue-500 text-white rounded px-2 py-1 text-xs font-bold">#{h.id}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-gray-800 font-medium">
                            {new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                            <span className="text-gray-500 ml-1">
                              {new Date(h.date).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          {h.status === 'running' && h.currentEpoch > 0 ? (
                            <span className="inline-flex items-center gap-1 text-sm font-medium text-blue-700">
                              🔄 {Math.round((h.currentEpoch / h.config.epochs) * 100)}%
                            </span>
                          ) : (
                            getStatusBadge(h.status || 'done', h.exportedFormat)
                          )}
                        </td>
                        <td className="px-4 py-3 text-right text-sm font-medium text-gray-800">{h.config.epochs}</td>
                        <td className="px-4 py-3 text-right text-sm font-medium text-gray-800">{h.config.batchSize}</td>
                        <td className="px-4 py-3 text-right text-sm font-medium text-gray-800">{h.config.learningRate}</td>
                        <td className="px-4 py-3 text-sm font-medium text-gray-800 capitalize">{h.config.optimizer}</td>
                        <td className="px-4 py-3 text-right">
                          {(h.status === 'done' || (h.status === 'running' && h.metrics)) && h.metrics ? (
                            <span className="text-sm font-bold text-green-600">{(h.metrics.acc*100).toFixed(2)}%</span>
                          ) : (
                            <span className="text-sm text-gray-400">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          {(h.status === 'done' || (h.status === 'running' && h.metrics)) && h.metrics ? (
                            <span className="text-sm font-bold text-red-600">{h.metrics.loss.toFixed(4)}</span>
                          ) : h.status === 'failed' && h.error ? (
                            <span className="text-sm text-red-600" title={h.error}>❌ Error</span>
                          ) : h.status === 'cancelled' ? (
                            <span className="text-sm text-orange-600">⚠️ Cancelled</span>
                          ) : (
                            <span className="text-sm text-gray-400">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <button 
                            className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 transition-colors text-sm font-bold" 
                            onClick={() => handleDelete(h)}
                            title="Delete"
                          >
                            🗑️
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-6 flex items-center justify-center gap-2">
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors font-medium"
                  >
                    ← Previous
                  </button>
                  <span className="px-4 py-2 text-gray-700 font-medium">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors font-medium"
                  >
                    Next →
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
