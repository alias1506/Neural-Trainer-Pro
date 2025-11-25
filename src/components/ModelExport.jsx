import React, { useState, useEffect } from 'react'
import Swal from 'sweetalert2'
import axios from 'axios'
import Pagination from './Pagination'

const API_URL = 'http://localhost:3001/api'

export default function ModelExport({ progress, datasetInfo, trainingHistory = [], onModelExported, onAllModelsCleared }) {
    const [models, setModels] = useState([])
    const [selectedModels, setSelectedModels] = useState([])
    const [exportFormat, setExportFormat] = useState('pytorch')
    const [searchQuery, setSearchQuery] = useState('')
    const [sortBy, setSortBy] = useState('date') // date, accuracy, name, size
    const [sortOrder, setSortOrder] = useState('desc') // asc, desc
    const [viewMode, setViewMode] = useState('table') // table, grid
    const [selectedModel, setSelectedModel] = useState(null) // For details modal
    const [exportHistory, setExportHistory] = useState([])
    const [currentPage, setCurrentPage] = useState(1)
    const [itemsPerPage] = useState(10) // Models per page

    // Load models from server
    useEffect(() => {
        loadModels()
    }, [])

    const loadModels = async () => {
        try {
            const response = await axios.get(`${API_URL}/list-models`)
            if (response.data.models) {
                setModels(response.data.models)
            }
        } catch (error) {
            console.error('Failed to load models:', error)
        }
    }

    // Enrich models with data from trainingHistory if server metadata is missing
    const enrichedModels = models.map(model => {
        // Normalize fields from potential legacy server response
        let normalizedModel = {
            ...model,
            id: model.id || model.name.replace('.pth', ''),
            sizeBytes: model.sizeBytes || model.size || 0,
            date: model.date || model.createdAt || model.modifiedAt || new Date().toISOString()
        }

        // Try to parse date from filename if current date is invalid or fallback
        const dateMatch = normalizedModel.name.match(/TrainedModel-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})/)
        if (dateMatch) {
            const [y, m, d, h, min, s] = dateMatch[1].split('-')
            const parsedDate = new Date(y, m - 1, d, h, min, s)
            if (!isNaN(parsedDate.getTime())) {
                normalizedModel.date = parsedDate.toISOString()
            }
        }

        // If model has accuracy, use it
        if (normalizedModel.accuracy != null) return normalizedModel

        // Try to find matching history entry
        // Match by filename (check if history path includes the model filename)
        const historyItem = trainingHistory.find(h =>
            h.modelPath && h.modelPath.includes(normalizedModel.name)
        )

        let enriched = { ...normalizedModel }

        if (historyItem) {
            enriched.sessionId = historyItem.id

            if (historyItem.metrics) {
                enriched.accuracy = historyItem.metrics.valAcc || historyItem.metrics.accuracy
                enriched.metrics = {
                    ...normalizedModel.metrics,
                    trainAcc: historyItem.metrics.trainAcc,
                    valAcc: historyItem.metrics.valAcc || historyItem.metrics.accuracy,
                    trainLoss: historyItem.metrics.trainLoss,
                    valLoss: historyItem.metrics.valLoss
                }
            }
        }

        return enriched
    })

    // Filter and sort models
    const filteredModels = enrichedModels
        .filter(model =>
            model.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
        .sort((a, b) => {
            let comparison = 0
            switch (sortBy) {
                case 'name':
                    comparison = a.name.localeCompare(b.name)
                    break
                case 'accuracy':
                    comparison = (a.accuracy || 0) - (b.accuracy || 0)
                    break
                case 'size':
                    comparison = (a.sizeBytes || 0) - (b.sizeBytes || 0)
                    break
                case 'date':
                default:
                    comparison = new Date(a.date) - new Date(b.date)
            }
            return sortOrder === 'asc' ? comparison : -comparison
        })

    // Pagination calculations
    const totalPages = Math.ceil(filteredModels.length / itemsPerPage)
    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = startIndex + itemsPerPage
    const paginatedModels = filteredModels.slice(startIndex, endIndex)

    // Check if all visible models are selected
    const isAllSelected = paginatedModels.length > 0 && paginatedModels.every(m => selectedModels.includes(m.id))

    // Reset to first page when filters change
    useEffect(() => {
        setCurrentPage(1)
    }, [searchQuery, sortBy, sortOrder])

    const handlePageChange = (page) => {
        if (page >= 1 && page <= totalPages) {
            setCurrentPage(page)
        }
    }

    const handleToggleSelect = (id) => {
        if (selectedModels.includes(id)) {
            setSelectedModels(selectedModels.filter(m => m !== id))
        } else {
            setSelectedModels([...selectedModels, id])
        }
    }

    const handleSelectAll = () => {
        if (isAllSelected) {
            // Deselect all on current page
            const pageIds = paginatedModels.map(m => m.id)
            setSelectedModels(selectedModels.filter(id => !pageIds.includes(id)))
        } else {
            // Select all on current page
            const pageIds = paginatedModels.map(m => m.id)
            // Add IDs that aren't already selected
            const newIds = pageIds.filter(id => !selectedModels.includes(id))
            setSelectedModels([...selectedModels, ...newIds])
        }
    }

    const handleExport = async () => {
        if (selectedModels.length === 0) return

        if (exportFormat === 'tflite' || exportFormat === 'coreml') {
            await Swal.fire({
                icon: 'info',
                title: 'Work in Progress',
                text: `${exportFormat === 'tflite' ? 'TFLite' : 'CoreML'} export is currently being optimized and will be available soon.`,
                confirmButtonText: 'Got it',
                customClass: {
                    popup: 'rounded-2xl shadow-xl border border-gray-100',
                    confirmButton: 'px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-medium transition-all shadow-sm'
                },
                buttonsStyling: false
            })
            return
        }

        const selectedModelData = models.filter(m => selectedModels.includes(m.id))

        const result = await Swal.fire({
            title: 'Export Models',
            html: `
                <div style="text-align: left; padding: 0.5rem;">
                    <p style="margin-bottom: 0.75rem; color: #666; font-size: 0.95rem;">Export ${selectedModels.length} model(s) as <strong>${exportFormat.toUpperCase()}</strong>?</p>
                    <div style="background: #f9fafb; padding: 0.5rem; border-radius: 8px; font-size: 0.85rem; max-height: 120px; overflow-y: auto;">
                        ${selectedModelData.map(m => `<div style="padding: 0.2rem 0; color: #555;">📦 ${m.name}</div>`).join('')}
                    </div>
                </div>
            `,
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: 'Export Now',
            cancelButtonText: 'Cancel',
            customClass: {
                popup: 'rounded-2xl shadow-2xl border border-gray-100',
                title: 'text-lg font-bold text-gray-900',
                htmlContainer: 'text-sm',
                actions: 'gap-3 mt-4',
                confirmButton: 'px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-medium transition-all shadow-sm transform active:scale-95',
                cancelButton: 'px-6 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-all shadow-sm transform active:scale-95'
            },
            buttonsStyling: false
        })

        if (result.isConfirmed) {
            try {
                Swal.fire({
                    title: 'Exporting...',
                    html: 'Please wait while we prepare your models',
                    allowOutsideClick: false,
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100'
                    },
                    didOpen: () => {
                        Swal.showLoading()
                    }
                })

                // Call the download API for each selected model
                for (const model of selectedModelData) {
                    try {
                        const downloadUrl = `${API_URL}/download-model?modelPath=${encodeURIComponent(model.path)}&format=${exportFormat}`

                        // Fetch the file as array buffer
                        const response = await fetch(downloadUrl)
                        if (!response.ok) {
                            throw new Error(`Download failed: ${response.statusText}`)
                        }

                        // Get the array buffer
                        const arrayBuffer = await response.arrayBuffer()

                        // Determine file extension and MIME type
                        let fileExt, mimeType = 'application/octet-stream';
                        switch (exportFormat) {
                            case 'pytorch':
                                fileExt = 'pth';
                                break;
                            case 'onnx':
                                fileExt = 'onnx';
                                break;
                            case 'torchscript':
                                fileExt = 'pt';
                                break;
                            case 'tflite':
                                fileExt = 'tflite';
                                break;
                            case 'coreml':
                                fileExt = 'mlmodel';
                                break;
                            default:
                                fileExt = exportFormat;
                        }

                        // Create blob with proper type
                        const blob = new Blob([arrayBuffer], { type: mimeType })

                        // Create download link
                        const url = window.URL.createObjectURL(blob)
                        const link = document.createElement('a')
                        link.href = url
                        link.download = model.name.replace('.pth', `.${fileExt}`)
                        link.style.display = 'none'
                        document.body.appendChild(link)
                        link.click()

                        // Cleanup
                        setTimeout(() => {
                            document.body.removeChild(link)
                            window.URL.revokeObjectURL(url)
                        }, 100)

                        // Small delay between downloads
                        await new Promise(resolve => setTimeout(resolve, 500))
                    } catch (downloadError) {
                        console.error('Download error:', downloadError)
                        throw downloadError
                    }
                }

                const exportRecord = {
                    id: Date.now(),
                    date: new Date().toISOString(),
                    models: selectedModelData.map(m => m.name),
                    format: exportFormat,
                    count: selectedModels.length
                }
                setExportHistory(prev => [exportRecord, ...prev])

                Swal.fire({
                    icon: 'success',
                    title: 'Export Complete!',
                    html: `
                        <div style="text-align: center; padding: 0.5rem;">
                            <p style="color: #10b981; font-size: 1rem; margin: 0.5rem 0; font-weight: 600;">✓ ${selectedModels.length} model(s) downloaded successfully</p>
                            <p style="color: #666; font-size: 0.85rem;">Format: ${exportFormat.toUpperCase()}</p>
                        </div>
                    `,
                    timer: 2500,
                    showConfirmButton: false,
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100'
                    }
                }).then(() => {
                    // Reload page after modal closes
                    window.location.reload()
                })

                setSelectedModels([])

                if (onModelExported) {
                    selectedModelData.forEach(m => onModelExported(m.path, exportFormat))
                }
            } catch (error) {
                Swal.fire({
                    icon: 'error',
                    title: 'Export Failed',
                    text: error.message || 'An error occurred during export',
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100',
                        confirmButton: 'px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-xl font-medium transition-all shadow-sm'
                    },
                    buttonsStyling: false
                })
            }
        }
    }


    const handleDeleteModel = async (model) => {
        const result = await Swal.fire({
            title: 'Delete Model?',
            text: `Are you sure you want to delete "${model.name}"?`,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Delete',
            cancelButtonText: 'Cancel',
            customClass: {
                popup: 'rounded-2xl shadow-2xl border border-gray-100',
                title: 'text-xl font-bold text-gray-900',
                htmlContainer: 'text-gray-600 mt-2',
                actions: 'gap-3 mt-6',
                confirmButton: 'px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-xl font-medium transition-all shadow-sm transform active:scale-95',
                cancelButton: 'px-6 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-all shadow-sm transform active:scale-95'
            },
            buttonsStyling: false
        })

        if (result.isConfirmed) {
            try {
                await axios.delete(`${API_URL}/delete-model/${model.id}`)

                setModels(prev => prev.filter(m => m.id !== model.id))
                setSelectedModels(prev => prev.filter(id => id !== model.id))

                Swal.fire({
                    icon: 'success',
                    title: 'Deleted',
                    text: 'Model deleted successfully',
                    timer: 2000,
                    showConfirmButton: false,
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100'
                    }
                })

                if (models.length === 1 && onAllModelsCleared) {
                    onAllModelsCleared()
                }
            } catch (error) {
                // If model not found (404), assume it's already deleted and remove from UI
                if (error.response && error.response.status === 404) {
                    setModels(prev => prev.filter(m => m.id !== model.id))
                    setSelectedModels(prev => prev.filter(id => id !== model.id))

                    Swal.fire({
                        icon: 'warning',
                        title: 'Model Not Found',
                        text: 'The model file was not found on the server but has been removed from your list.',
                        timer: 3000,
                        showConfirmButton: false,
                        customClass: {
                            popup: 'rounded-2xl shadow-xl border border-gray-100'
                        }
                    })
                    return
                }

                Swal.fire({
                    icon: 'error',
                    title: 'Delete Failed',
                    text: error.message || 'Failed to delete model',
                    confirmButtonColor: '#ef4444',
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100'
                    }
                })
            }
        }
    }

    const handleClearAll = async () => {
        const result = await Swal.fire({
            title: 'Clear All Models?',
            text: `This will permanently delete all ${models.length} trained models.`,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Clear All',
            cancelButtonText: 'Cancel',
            customClass: {
                popup: 'rounded-2xl shadow-2xl border border-gray-100',
                title: 'text-xl font-bold text-gray-900',
                htmlContainer: 'text-gray-600 mt-2',
                actions: 'gap-3 mt-6',
                confirmButton: 'px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-xl font-medium transition-all shadow-sm transform active:scale-95',
                cancelButton: 'px-6 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-all shadow-sm transform active:scale-95'
            },
            buttonsStyling: false
        })

        if (result.isConfirmed) {
            try {
                await axios.post(`${API_URL}/clear-all-models`)
                setModels([])
                setSelectedModels([])

                Swal.fire({
                    icon: 'success',
                    title: 'All Models Cleared',
                    timer: 2000,
                    showConfirmButton: false,
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100'
                    }
                })

                if (onAllModelsCleared) {
                    onAllModelsCleared()
                }
            } catch (error) {
                Swal.fire({
                    icon: 'error',
                    title: 'Clear Failed',
                    text: error.message || 'Failed to clear models',
                    confirmButtonColor: '#ef4444',
                    customClass: {
                        popup: 'rounded-2xl shadow-xl border border-gray-100'
                    }
                })
            }
        }
    }

    const handleViewDetails = (model) => {
        setSelectedModel(model)
    }

    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes'
        if (!bytes) return 'N/A'
        const sizes = ['Bytes', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(1024))
        return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
    }

    return (
        <div className="flex flex-col h-full animate-fade-in gap-4">
            {/* Header */}
            <div className="flex-shrink-0 flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-semibold">Model Export</h2>
                    <p className="text-sm text-muted mt-1">Manage and export trained models</p>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={exportFormat}
                        onChange={(e) => setExportFormat(e.target.value)}
                        className="select text-sm py-2 w-44"
                    >
                        <option value="pytorch">PyTorch (.pth)</option>
                        <option value="onnx">ONNX (.onnx)</option>
                        <option value="torchscript">TorchScript (.pt)</option>
                        <option value="tflite">TFLite (.tflite)</option>
                        <option value="coreml">CoreML (.mlmodel)</option>
                    </select>
                    <button
                        onClick={handleExport}
                        disabled={selectedModels.length === 0}
                        className="btn btn-primary py-2 px-4 text-sm"
                    >
                        Export ({selectedModels.length})
                    </button>
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
                                placeholder="Search models..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
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
                            <option value="name">Name</option>
                            <option value="accuracy">Accuracy</option>
                            <option value="size">Size</option>
                        </select>

                        <button
                            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                            className="btn-icon flex-shrink-0 p-2"
                            title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                            {filteredModels.length} of {models.length} models
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 card min-h-0 flex flex-col p-0 overflow-hidden">
                <div className="p-4 border-b bg-gray-50 flex justify-between items-center">
                    <h3 className="text-base font-semibold">Available Models</h3>
                    {models.length > 0 && (
                        <button
                            onClick={handleClearAll}
                            className="text-sm text-red-500 hover:text-red-600 font-medium transition-colors"
                        >
                            Clear All
                        </button>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {viewMode === 'table' ? (
                        <table className="w-full text-left text-sm">
                            <thead className="bg-white sticky top-0 z-10 border-b shadow-sm">
                                <tr>
                                    <th className="p-4 w-12">
                                        <input
                                            type="checkbox"
                                            className="rounded border-gray-300 cursor-pointer"
                                            checked={isAllSelected}
                                            onChange={handleSelectAll}
                                            title="Select all"
                                        />
                                    </th>
                                    <th className="p-4 font-medium text-muted">Model Name</th>
                                    <th className="p-4 font-medium text-muted">Accuracy (Train / Val)</th>
                                    <th className="p-4 font-medium text-muted">Size</th>
                                    <th className="p-4 font-medium text-muted">Date</th>
                                    <th className="p-4 font-medium text-muted text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100">
                                {paginatedModels.length > 0 ? paginatedModels.map(model => (
                                    <tr key={model.id} className="hover:bg-gray-50 transition-colors animate-fade-in">
                                        <td className="p-4">
                                            <input
                                                type="checkbox"
                                                checked={selectedModels.includes(model.id)}
                                                onChange={() => handleToggleSelect(model.id)}
                                                className="rounded border-gray-300 cursor-pointer"
                                            />
                                        </td>
                                        <td className="p-3">
                                            <div className="flex items-center gap-2">
                                                <svg className="w-4 h-4 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                </svg>
                                                <span className="font-medium">{model.name}</span>
                                            </div>
                                        </td>
                                        <td className="p-3">
                                            {model.metrics ? (
                                                <div className="flex items-center gap-3 text-sm">
                                                    <span className="text-blue-600 font-medium" title="Training Accuracy">
                                                        {(model.metrics.trainAcc * 100).toFixed(1)}%
                                                    </span>
                                                    <span className="text-gray-300">|</span>
                                                    <span className="text-green-600 font-medium" title="Validation Accuracy">
                                                        {(model.metrics.valAcc * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                            ) : model.accuracy ? (
                                                <span className="text-green-600 font-medium text-sm">
                                                    {(model.accuracy * 100).toFixed(1)}%
                                                </span>
                                            ) : (
                                                <span className="text-muted text-sm">N/A</span>
                                            )}
                                        </td>
                                        <td className="p-4 text-muted text-sm">{formatBytes(model.sizeBytes)}</td>
                                        <td className="p-4 text-muted text-sm">{new Date(model.date).toLocaleDateString()}</td>
                                        <td className="p-4">
                                            <div className="flex items-center justify-end gap-2">
                                                <button
                                                    onClick={() => handleViewDetails(model)}
                                                    className="p-2 text-blue-500 hover:bg-blue-50 rounded transition-colors"
                                                    title="View details"
                                                >
                                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                                    </svg>
                                                </button>
                                                <button
                                                    onClick={() => handleDeleteModel(model)}
                                                    className="p-2 text-red-500 hover:bg-red-50 rounded transition-colors"
                                                    title="Delete model"
                                                >
                                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                    </svg>
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan="6" className="p-12 text-center text-muted">
                                            <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                                            </svg>
                                            <p className="font-medium">No models available</p>
                                            <p className="text-[10px] mt-1">
                                                {searchQuery ? 'Try a different search term' : 'Train a model to see it here'}
                                            </p>
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    ) : (
                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {paginatedModels.length > 0 ? paginatedModels.map(model => (
                                <div key={model.id} className="card p-4 hover:shadow-md transition-all animate-fade-in cursor-pointer" onClick={() => handleViewDetails(model)}>
                                    <div className="flex items-start justify-between mb-2">
                                        <input
                                            type="checkbox"
                                            checked={selectedModels.includes(model.id)}
                                            onChange={(e) => {
                                                e.stopPropagation()
                                                handleToggleSelect(model.id)
                                            }}
                                            className="rounded border-gray-300 cursor-pointer"
                                            onClick={(e) => e.stopPropagation()}
                                        />
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                handleDeleteModel(model)
                                            }}
                                            className="p-1.5 text-red-500 hover:bg-red-50 rounded transition-colors"
                                        >
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                        </button>
                                    </div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <svg className="w-5 h-5 text-muted flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <span className="font-medium text-sm truncate">{model.name}</span>
                                    </div>
                                    <div className="flex items-center justify-between text-sm text-muted mt-3 pt-3 border-t">
                                        <div>
                                            {model.accuracy ? (
                                                <span className="px-2 py-0.5 rounded-full bg-green-50 text-green-700 border border-green-200 text-[10px] font-medium">
                                                    {(model.accuracy * 100).toFixed(1)}%
                                                </span>
                                            ) : (
                                                <span>N/A</span>
                                            )}
                                        </div>
                                        <div>{formatBytes(model.sizeBytes)}</div>
                                    </div>
                                    <div className="text-xs text-muted mt-1">
                                        {new Date(model.date).toLocaleDateString()}
                                    </div>
                                </div>
                            )) : (
                                <div className="col-span-full p-12 text-center text-muted">
                                    <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                                    </svg>
                                    <p className="font-medium">No models available</p>
                                    <p className="text-[10px] mt-1">
                                        {searchQuery ? 'Try a different search term' : 'Train a model to see it here'}
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
                    totalItems={filteredModels.length}
                    itemsPerPage={itemsPerPage}
                    onPageChange={handlePageChange}
                />
            </div>

            {/* Model Details Modal */}
            {selectedModel && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 animate-fade-in" onClick={() => setSelectedModel(null)}>
                    <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
                        <div className="p-4 border-b flex items-center justify-between bg-gray-50">
                            <h3 className="font-semibold text-lg">Model Details</h3>
                            <button onClick={() => setSelectedModel(null)} className="btn-icon">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                        <div className="p-6 overflow-y-auto custom-scrollbar" style={{ maxHeight: 'calc(80vh - 140px)' }}>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="text-xs text-muted font-medium">Model Name</label>
                                    <p className="text-sm font-medium mt-1">{selectedModel.name}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Accuracy</label>
                                    <p className="text-sm font-medium mt-1">
                                        {selectedModel.accuracy ? `${(selectedModel.accuracy * 100).toFixed(2)}%` : 'N/A'}
                                    </p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">File Size</label>
                                    <p className="text-sm font-medium mt-1">{formatBytes(selectedModel.sizeBytes)}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Created Date</label>
                                    <p className="text-sm font-medium mt-1">{new Date(selectedModel.date).toLocaleString()}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Model ID</label>
                                    <p className="text-xs text-muted mt-1 break-all font-mono">{selectedModel.sessionId || selectedModel.id}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-muted font-medium">Format</label>
                                    <p className="text-sm font-medium mt-1">PyTorch (.pth)</p>
                                </div>
                            </div>

                            {selectedModel.metrics && (
                                <div className="mt-6 pt-6 border-t">
                                    <h4 className="font-semibold text-sm mb-3">Training Metrics</h4>
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="p-3 bg-gray-50 rounded">
                                            <label className="text-xs text-muted">Train Accuracy</label>
                                            <p className="text-lg font-semibold text-blue-600">{(selectedModel.metrics.trainAcc * 100).toFixed(2)}%</p>
                                        </div>
                                        <div className="p-3 bg-gray-50 rounded">
                                            <label className="text-xs text-muted">Val Accuracy</label>
                                            <p className="text-lg font-semibold text-green-600">{(selectedModel.metrics.valAcc * 100).toFixed(2)}%</p>
                                        </div>
                                        <div className="p-3 bg-gray-50 rounded">
                                            <label className="text-xs text-muted">Train Loss</label>
                                            <p className="text-lg font-semibold text-orange-600">{selectedModel.metrics.trainLoss.toFixed(4)}</p>
                                        </div>
                                        <div className="p-3 bg-gray-50 rounded">
                                            <label className="text-xs text-muted">Val Loss</label>
                                            <p className="text-lg font-semibold text-red-600">{selectedModel.metrics.valLoss.toFixed(4)}</p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="p-4 border-t bg-gray-50 flex justify-end gap-2">
                            <button onClick={() => setSelectedModel(null)} className="btn btn-secondary py-2 text-sm">
                                Close
                            </button>
                            <button
                                onClick={() => {
                                    handleToggleSelect(selectedModel.id)
                                    setSelectedModel(null)
                                }}
                                className="btn btn-primary py-2 text-sm"
                            >
                                {selectedModels.includes(selectedModel.id) ? 'Deselect' : 'Select for Export'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
