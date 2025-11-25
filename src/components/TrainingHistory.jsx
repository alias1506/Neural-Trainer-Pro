import React, { useState, useEffect } from 'react'
import Swal from 'sweetalert2'
import Pagination from './Pagination'

export default function TrainingHistory({ history, onLoad, onDelete }) {
    const [filter, setFilter] = useState('all')
    const [searchTerm, setSearchTerm] = useState('')
    const [sortBy, setSortBy] = useState('date') // date, accuracy, duration
    const [sortOrder, setSortOrder] = useState('desc') // asc, desc
    const [viewMode, setViewMode] = useState('table') // table, grid
    const [selectedHistory, setSelectedHistory] = useState(null) // For details modal
    const [currentPage, setCurrentPage] = useState(1)
    const [itemsPerPage] = useState(10) // Items per page

    // Filter and sort history
    const filteredHistory = history
        .filter(h => {
            if (filter !== 'all' && h.status !== filter) return false
            if (searchTerm && !h.id.toLowerCase().includes(searchTerm.toLowerCase())) return false
            return true
        })
        .sort((a, b) => {
            let comparison = 0
            switch (sortBy) {
                case 'accuracy':
                    comparison = (a.metrics?.accuracy || 0) - (b.metrics?.accuracy || 0)
                    break
                case 'duration':
                    comparison = (a.durationMs || 0) - (b.durationMs || 0)
                    break
                case 'date':
                default:
                    comparison = new Date(a.date) - new Date(b.date)
            }
            return sortOrder === 'asc' ? comparison : -comparison
        })

    // Pagination calculations
    const totalPages = Math.ceil(filteredHistory.length / itemsPerPage)
    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = startIndex + itemsPerPage
    const paginatedHistory = filteredHistory.slice(startIndex, endIndex)

    // Reset to first page when filters change
    useEffect(() => {
        setCurrentPage(1)
    }, [searchTerm, filter, sortBy, sortOrder])

    const handlePageChange = (page) => {
        if (page >= 1 && page <= totalPages) {
            setCurrentPage(page)
        }
    }

    const handleViewDetails = (item) => {
        setSelectedHistory(item)
    }

    const handleDeleteHistory = async (item) => {
        const result = await Swal.fire({
            title: 'Delete Training Record?',
            html: `
                <div style="text-align: left; padding: 1rem;">
                    <p style="color: #666; margin-bottom: 0.5rem;">Are you sure you want to delete this training record?</p>
                    <p style="font-weight: 600; color: #000;">Date: ${new Date(item.date).toLocaleString()}</p>
                    <p style="color: #ef4444; font-size: 0.875rem; margin-top: 1rem;">⚠️ This action cannot be undone</p>
                </div>
            `,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Delete',
            cancelButtonText: 'Cancel',
            confirmButtonColor: '#ef4444',
            cancelButtonColor: '#6B728E'
        })

        if (result.isConfirmed) {
            onDelete(item.id)
            Swal.fire({
                icon: 'success',
                title: 'Deleted',
                text: 'Training record deleted successfully',
                timer: 2000,
                showConfirmButton: false
            })
        }
    }

    const formatDuration = (ms) => {
        if (!ms) return 'N/A'
        const seconds = Math.floor((ms / 1000) % 60)
        const minutes = Math.floor((ms / 1000 / 60) % 60)
        const hours = Math.floor(ms / 1000 / 3600)
        if (hours > 0) return `${hours}h ${minutes}m`
        if (minutes > 0) return `${minutes}m ${seconds}s`
        return `${seconds}s`
    }

    const getStatusColor = (status) => {
        switch (status) {
            case 'done':
            case 'completed':
                return 'bg-green-50 text-green-700 border-green-200'
            case 'failed':
                return 'bg-red-50 text-red-700 border-red-200'
            case 'cancelled':
                return 'bg-orange-50 text-orange-700 border-orange-200'
            default:
                return 'bg-gray-50 text-gray-700 border-gray-200'
        }
    }

    return (
        <div className="flex flex-col h-full animate-fade-in gap-4">
            {/* Header */}
            <div className="flex-shrink-0 flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-semibold">Training History</h2>
                    <p className="text-sm text-muted mt-1">View and manage past training sessions</p>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="select text-sm py-2 w-36"
                    >
                        <option value="all">All Status</option>
                        <option value="done">Completed</option>
                        <option value="failed">Failed</option>
                        <option value="cancelled">Cancelled</option>
                    </select>
                </div>
            </div>

            {/* Toolbar */}
            <div className="flex-shrink-0 card p-4">
                <div className="flex items-center justify-between gap-4">
                    {/* Left Section - Search and Sort */}
                    <div className="flex items-center gap-4 flex-1 min-w-0">
                        {/* Search */}
                        <div className="relative flex-1 max-w-md">
                            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                            <input
                                type="text"
                                placeholder="Search by ID..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="input text-sm w-full py-2"
                                style={{ paddingLeft: '2.5rem' }}
                            />
                        </div>

                        {/* Sort */}
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value)}
                            className="select text-sm py-2 w-36"
                        >
                            <option value="date">Date</option>
                            <option value="accuracy">Accuracy</option>
                            <option value="duration">Duration</option>
                        </select>

                        <button
                            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                            className="btn-icon flex-shrink-0 p-2"
                            title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                {sortOrder === 'asc' ? (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
                                ) : (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4" />
                                )}
                            </svg>
                        </button>
                    </div>

                    {/* Right Section - View Mode and Stats */}
                    <div className="flex items-center gap-4 flex-shrink-0">
                        {/* View Mode Toggle */}
                        <div className="flex items-center gap-1 bg-gray-100 rounded-md p-1">
                            <button
                                onClick={() => setViewMode('table')}
                                className={`p-2 rounded ${viewMode === 'table' ? 'bg-white shadow-sm' : 'text-muted'}`}
                                title="Table view"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                            </button>
                            <button
                                onClick={() => setViewMode('grid')}
                                className={`p-2 rounded ${viewMode === 'grid' ? 'bg-white shadow-sm' : 'text-muted'}`}
                                title="Grid view"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                                </svg>
                            </button>
                        </div>

                        {/* Stats */}
                        <div className="text-sm text-muted px-4 border-l">
                            {filteredHistory.length} of {history.length} records
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 card min-h-0 flex flex-col p-0 overflow-hidden">
                <div className="p-4 border-b bg-gray-50 flex justify-between items-center">
                    <h3 className="text-base font-semibold">Training Records</h3>
                    {history.length > 0 && (
                        <span className="text-sm text-muted">
                            Total: {history.length} session{history.length !== 1 ? 's' : ''}
                        </span>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {viewMode === 'table' ? (
                        <table className="w-full text-left text-sm table-fixed">
                            <thead className="bg-white sticky top-0 z-10 border-b shadow-sm">
                                <tr>
                                    <th className="p-4 font-medium text-muted w-[15%]">Date & Time</th>
                                    <th className="p-4 font-medium text-muted w-[25%]">Model Name</th>
                                    <th className="p-4 font-medium text-muted w-[10%]">Status</th>
                                    <th className="p-4 font-medium text-muted w-[20%]">Configuration</th>
                                    <th className="p-4 font-medium text-muted w-[15%]">Metrics</th>
                                    <th className="p-4 font-medium text-muted w-[10%]">Duration</th>
                                    <th className="p-4 font-medium text-muted text-right w-[5%]">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100">
                                {paginatedHistory.length > 0 ? paginatedHistory.map(h => (
                                    <tr key={h.id} className="hover:bg-gray-50 transition-colors animate-fade-in">
                                        <td className="p-4 truncate">
                                            <div className="flex flex-col">
                                                <span className="font-medium">{new Date(h.date).toLocaleDateString()}</span>
                                                <span className="text-xs text-muted">{new Date(h.date).toLocaleTimeString()}</span>
                                            </div>
                                        </td>
                                        <td className="p-4 truncate">
                                            <div className="flex items-center gap-2 truncate">
                                                <svg className="w-4 h-4 text-muted flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                </svg>
                                                <span className="font-medium text-sm truncate" title={h.modelPath}>{h.modelPath || 'N/A'}</span>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <span className={`px-3 py-1 rounded-full border text-xs font-medium capitalize ${getStatusColor(h.status)}`}>
                                                {h.status}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <div className="flex flex-col gap-1 text-xs text-muted">
                                                <div><span className="font-medium text-gray-700">Epochs:</span> {h.config?.epochs}</div>
                                                <div><span className="font-medium text-gray-700">Batch:</span> {h.config?.batchSize}</div>
                                                <div><span className="font-medium text-gray-700">LR:</span> {h.config?.learningRate}</div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <div className="flex flex-col gap-1 text-xs">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-green-600 font-semibold">
                                                        {h.metrics?.accuracy !== undefined ? `${(h.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                                                    </span>
                                                    <span className="text-muted">Val Acc</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <span className="text-red-600 font-semibold">
                                                        {h.metrics?.loss !== undefined ? h.metrics.loss.toFixed(4) : 'N/A'}
                                                    </span>
                                                    <span className="text-muted">Train Loss</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <span className="text-sm font-medium text-gray-600">
                                                {formatDuration(h.durationMs || h.duration)}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <div className="flex items-center justify-end gap-2">
                                                <button
                                                    onClick={() => handleViewDetails(h)}
                                                    className="p-2 text-blue-500 hover:bg-blue-50 rounded transition-colors"
                                                    title="View details"
                                                >
                                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                                    </svg>
                                                </button>
                                                <button
                                                    onClick={() => onLoad(h)}
                                                    className="p-2 text-purple-500 hover:bg-purple-50 rounded transition-colors"
                                                    title="Load configuration"
                                                >
                                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                                    </svg>
                                                </button>
                                                <button
                                                    onClick={() => handleDeleteHistory(h)}
                                                    className="p-2 text-red-500 hover:bg-red-50 rounded transition-colors"
                                                    title="Delete record"
                                                >
                                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                    </svg>
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan="7" className="p-12 text-center text-muted">
                                            <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                            </svg>
                                            <p className="font-medium">No training history found</p>
                                            <p className="text-xs mt-1">
                                                {searchTerm ? 'Try a different search term' : 'Complete a training session to see it here'}
                                            </p>
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    ) : (
                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {paginatedHistory.length > 0 ? paginatedHistory.map(h => (
                                <div key={h.id} className="card p-4 hover:shadow-md transition-all animate-fade-in cursor-pointer" onClick={() => handleViewDetails(h)}>
                                    <div className="flex items-start justify-between mb-3">
                                        <span className={`px-3 py-1 rounded-full border text-xs font-medium capitalize ${getStatusColor(h.status)}`}>
                                            {h.status}
                                        </span>
                                        <div className="flex gap-1">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    onLoad(h)
                                                }}
                                                className="p-1.5 text-purple-500 hover:bg-purple-50 rounded transition-colors"
                                                title="Load config"
                                            >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                                </svg>
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    handleDeleteHistory(h)
                                                }}
                                                className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                                                title="Delete"
                                            >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                </svg>
                                            </button>
                                        </div>
                                    </div>

                                    <div className="mb-3">
                                        <div className="text-sm font-medium">{new Date(h.date).toLocaleDateString()}</div>
                                        <div className="text-xs text-muted">{new Date(h.date).toLocaleTimeString()}</div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2 text-xs mb-3 pb-3 border-b">
                                        <div>
                                            <span className="text-muted">Accuracy</span>
                                            <div className="font-semibold text-green-600">
                                                {h.metrics?.accuracy ? `${(h.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                                            </div>
                                        </div>
                                        <div>
                                            <span className="text-muted">Loss</span>
                                            <div className="font-semibold text-red-600">
                                                {h.metrics?.loss ? h.metrics.loss.toFixed(4) : 'N/A'}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between text-xs text-muted">
                                        <span>Epochs: {h.config?.epochs || 'N/A'}</span>
                                        <span>{formatDuration(h.durationMs || h.duration)}</span>
                                    </div>
                                </div>
                            )) : (
                                <div className="col-span-full p-12 text-center text-muted">
                                    <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <p className="font-medium">No training history found</p>
                                    <p className="text-xs mt-1">
                                        {searchTerm ? 'Try a different search term' : 'Complete a training session to see it here'}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Pagination */}
                <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    totalItems={filteredHistory.length}
                    itemsPerPage={itemsPerPage}
                    onPageChange={handlePageChange}
                />
            </div>

            {/* Details Modal */}
            {selectedHistory && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 animate-fade-in" onClick={() => setSelectedHistory(null)}>
                    <div className="bg-white rounded-lg shadow-lg max-w-4xl w-full mx-4 overflow-hidden" onClick={(e) => e.stopPropagation()}>
                        <div className="p-4 border-b flex items-center justify-between bg-gray-50">
                            <h3 className="font-semibold text-lg">Training Session Details</h3>
                            <button onClick={() => setSelectedHistory(null)} className="btn-icon">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                        <div className="p-6">
                            <div className="grid grid-cols-4 gap-4 mb-6">
                                <div className="col-span-4 md:col-span-1">
                                    <label className="text-xs text-muted font-medium">Session ID</label>
                                    <p className="text-sm font-medium mt-1 font-mono truncate" title={selectedHistory.id}>{selectedHistory.id}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Status</label>
                                    <p className="text-sm font-medium mt-1">
                                        <span className={`px-3 py-1 rounded-full border text-xs capitalize ${getStatusColor(selectedHistory.status)}`}>
                                            {selectedHistory.status}
                                        </span>
                                    </p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Date & Time</label>
                                    <p className="text-sm font-medium mt-1">{new Date(selectedHistory.date).toLocaleString()}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Duration</label>
                                    <p className="text-sm font-medium mt-1">{formatDuration(selectedHistory.durationMs || selectedHistory.duration)}</p>
                                </div>
                                <div className="col-span-4">
                                    <label className="text-xs text-muted font-medium">Model Name</label>
                                    <p className="text-sm font-medium mt-1 font-mono bg-gray-50 p-2 rounded border border-gray-100">{selectedHistory.modelPath || 'N/A'}</p>
                                </div>
                            </div>

                            <div className="mb-6 pt-6 border-t">
                                <h4 className="font-semibold text-sm mb-3">Configuration</h4>
                                <div className="grid grid-cols-4 gap-3">
                                    <div className="p-3 bg-gray-50 rounded">
                                        <label className="text-xs text-muted">Epochs</label>
                                        <p className="text-lg font-semibold">{selectedHistory.config?.epochs || 'N/A'}</p>
                                    </div>
                                    <div className="p-3 bg-gray-50 rounded">
                                        <label className="text-xs text-muted">Batch Size</label>
                                        <p className="text-lg font-semibold">{selectedHistory.config?.batchSize || 'N/A'}</p>
                                    </div>
                                    <div className="p-3 bg-gray-50 rounded">
                                        <label className="text-xs text-muted">Learning Rate</label>
                                        <p className="text-lg font-semibold">{selectedHistory.config?.learningRate || 'N/A'}</p>
                                    </div>
                                    <div className="p-3 bg-gray-50 rounded">
                                        <label className="text-xs text-muted">Optimizer</label>
                                        <p className="text-lg font-semibold capitalize">{selectedHistory.config?.optimizer || 'N/A'}</p>
                                    </div>
                                    <div className="p-3 bg-gray-50 rounded">
                                        <label className="text-xs text-muted">Validation Split</label>
                                        <p className="text-lg font-semibold">
                                            {selectedHistory.config?.validationSplit === 0 ? '0 (Auto)' : (selectedHistory.config?.validationSplit || '0 (Auto)')}
                                        </p>
                                    </div>
                                    <div className="p-3 bg-gray-50 rounded">
                                        <label className="text-xs text-muted">Patience</label>
                                        <p className="text-lg font-semibold">
                                            {selectedHistory.config?.patience === 0 ? '0 (Auto)' : (selectedHistory.config?.patience || '0 (Auto)')}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {selectedHistory.metrics && (
                                <div className="pt-6 border-t">
                                    <h4 className="font-semibold text-sm mb-3">Final Metrics</h4>
                                    <div className="grid grid-cols-4 gap-3">
                                        <div className="p-3 bg-gray-50 rounded">
                                            <label className="text-xs text-muted">Accuracy</label>
                                            <p className="text-lg font-semibold text-green-600">
                                                {selectedHistory.metrics.accuracy ? `${(selectedHistory.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                                            </p>
                                        </div>
                                        <div className="p-3 bg-gray-50 rounded">
                                            <label className="text-xs text-muted">Loss</label>
                                            <p className="text-lg font-semibold text-red-600">
                                                {selectedHistory.metrics.loss ? selectedHistory.metrics.loss.toFixed(4) : 'N/A'}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="p-4 border-t bg-gray-50 flex justify-end gap-2">
                            <button onClick={() => setSelectedHistory(null)} className="btn btn-secondary py-2 text-sm">
                                Close
                            </button>
                            <button
                                onClick={() => {
                                    onLoad(selectedHistory)
                                    setSelectedHistory(null)
                                }}
                                className="btn btn-primary py-2 text-sm"
                            >
                                Load Configuration
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
