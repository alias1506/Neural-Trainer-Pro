import React from 'react'
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

// Register ChartJS components
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
    const chartColor = '#6B728E'
    const chartFillColor = 'rgba(107, 114, 142, 0.1)'
    const { status, epoch, totalEpochs, trainLoss, trainAcc, valLoss, valAcc, timeMs, lossHistory = [], accHistory = [], valLossHistory = [], valAccHistory = [], batch, totalBatches } = progress

    // Calculate smooth progress including within-epoch progress
    let progressPercentage = 0;
    if (status === 'training' || status === 'preparing') {
        if (totalEpochs > 0) {
            const completedEpochs = Math.max(0, epoch - 1);
            const currentEpochProgress = (batch && totalBatches) ? (batch / totalBatches) : 0;
            progressPercentage = ((completedEpochs + currentEpochProgress) / totalEpochs) * 100;
            if (epoch === totalEpochs && batch === totalBatches && totalEpochs > 0 && totalBatches > 0) {
                progressPercentage = 100;
            }
            if (progressPercentage < 0) progressPercentage = 0;
            if (progressPercentage > 100) progressPercentage = 100;
        }
    } else if (status === 'completed' || status === 'done') {
        progressPercentage = 100;
    } else {
        progressPercentage = 0;
    }

    const formatTime = (ms) => {
        if (!ms) return '00:00'
        const seconds = Math.floor((ms / 1000) % 60)
        const minutes = Math.floor((ms / 1000 / 60) % 60)
        const hours = Math.floor(ms / 1000 / 3600)
        return `${hours > 0 ? hours + ':' : ''}${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
    }

    // Use only completed epoch data from history arrays (no live batch updates)
    // Add initial 0 point so graphs are visible from start
    const trainLossDataPoints = [0, ...(lossHistory || [])];
    const trainAccDataPoints = [0, ...(accHistory || [])];
    const valLossDataPoints = [0, ...(valLossHistory || [])];
    const valAccDataPoints = [0, ...(valAccHistory || [])];

    // X-axis shows total configured epochs (e.g., if user set 10 epochs, show 1-10)
    // Start from 0 to show initial resting point
    // Parse as integer to prevent string concatenation issues
    const totalConfiguredEpochs = parseInt(totalEpochs || config?.epochs || 10, 10);
    const epochLabels = [0, ...Array.from({ length: totalConfiguredEpochs }, (_, i) => i + 1)];

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Disable all animations to prevent lines from bottom
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 8,
                titleFont: { size: 11 },
                bodyFont: { size: 10 }
            }
        },
        scales: {
            x: {
                display: true,
                grid: {
                    display: true,
                    color: 'rgba(0, 0, 0, 0.05)',
                    drawBorder: true
                },
                ticks: {
                    display: true,
                    font: { size: 10 },
                    maxTicksLimit: 8,
                    color: '#666666'
                },
                border: {
                    display: true,
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            },
            y: {
                display: true,
                beginAtZero: true, // Start y-axis from 0
                grid: {
                    display: true,
                    color: 'rgba(0, 0, 0, 0.05)',
                    drawBorder: true
                },
                ticks: {
                    display: true,
                    font: { size: 10 },
                    color: '#666666'
                },
                border: {
                    display: true,
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        },
        elements: {
            point: {
                radius: 0, // Hide dots
                hitRadius: 10,
                hoverRadius: 4,
                backgroundColor: chartColor
            },
            line: {
                tension: 0.3
            }
        }
    }

    const trainLossData = {
        labels: epochLabels,
        datasets: [{
            label: 'Training Loss',
            data: trainLossDataPoints,
            borderColor: chartColor,
            backgroundColor: chartFillColor,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            spanGaps: true // Smooth transitions between points
        }]
    }

    const valLossData = {
        labels: epochLabels,
        datasets: [{
            label: 'Validation Loss',
            data: valLossDataPoints,
            borderColor: chartColor,
            backgroundColor: chartFillColor,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            spanGaps: true
        }]
    }

    const trainAccData = {
        labels: epochLabels,
        datasets: [{
            label: 'Training Accuracy',
            data: trainAccDataPoints,
            borderColor: chartColor,
            backgroundColor: chartFillColor,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            spanGaps: true
        }]
    }

    const valAccData = {
        labels: epochLabels,
        datasets: [{
            label: 'Validation Accuracy',
            data: valAccDataPoints,
            borderColor: chartColor,
            backgroundColor: chartFillColor,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            spanGaps: true
        }]
    }

    let statusText = status
    let dotClass = 'bg-gray-400'
    if (status === 'training') {
        statusText = 'Training'
        dotClass = 'bg-green-500 animate-pulse'
    } else if (status === 'upload-failed') {
        statusText = 'Upload Failed'
        dotClass = 'bg-gray-400'
    } else if (status === 'preparing') {
        statusText = 'Uploading...'
        dotClass = 'bg-blue-400 animate-pulse'
    } else if (status === 'network-error') {
        statusText = 'Network Error'
        dotClass = 'bg-gray-400'
    }

    return (
        <div className="flex flex-col gap-3 animate-fade-in" style={{ height: 'calc(100vh - 140px)' }}>
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-lg font-semibold">Training Progress</h2>
                    <div className="flex items-center gap-2 mt-0.5">
                        <span className={`w-2 h-2 rounded-full ${dotClass}`}></span>
                        <span className="text-xs text-muted capitalize">{statusText}</span>
                        <span className="text-xs text-gray-300">|</span>
                        <span className="text-xs text-muted">Time: {formatTime(timeMs)}</span>
                    </div>
                </div>
                {status === 'training' && (
                    <button onClick={onCancel} className="btn btn-secondary text-xs px-3 py-1.5">
                        Cancel
                    </button>
                )}
            </div>

            {/* Main Grid Layout */}
            <div className="flex-1 grid grid-cols-2 gap-3 min-h-0">
                {/* Left Column */}
                <div className="flex flex-col gap-3 overflow-y-auto custom-scrollbar pr-1">
                    {/* Epoch Progress */}
                    <div className="card p-3">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-muted">
                                Epoch {epoch}/{totalEpochs}
                                {batch && totalBatches && (
                                    <span className="ml-2 text-gray-400">
                                        (Batch {batch}/{totalBatches})
                                    </span>
                                )}
                            </span>
                            <span className="text-xs font-semibold" style={{ color: '#404258' }}>
                                {Math.round(progressPercentage)}%
                            </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                            <div
                                className="h-full rounded-full"
                                style={{
                                    width: `${progressPercentage}%`,
                                    backgroundColor: '#404258',
                                    transition: 'width 0.5s ease-out'
                                }}
                            ></div>
                        </div>
                    </div>

                    {/* Metrics Grid - 2x2 */}
                    <div className="grid grid-cols-2 gap-2">
                        {/* Train Loss */}
                        <div className="card p-3">
                            <div className="text-xs text-muted mb-1">Train Loss</div>
                            <div className="text-lg font-semibold text-red-600">
                                {trainLoss ? trainLoss.toFixed(4) : '—'}
                            </div>
                        </div>

                        {/* Train Acc */}
                        <div className="card p-3">
                            <div className="text-xs text-muted mb-1">Train Acc</div>
                            <div className="text-lg font-semibold text-green-600">
                                {trainAcc ? (trainAcc * 100).toFixed(2) + '%' : '—'}
                            </div>
                        </div>

                        {/* Val Loss */}
                        <div className="card p-3">
                            <div className="text-xs text-muted mb-1">Val Loss</div>
                            <div className="text-lg font-semibold text-orange-600">
                                {valLoss ? valLoss.toFixed(4) : '—'}
                            </div>
                        </div>

                        {/* Val Acc */}
                        <div className="card p-3">
                            <div className="text-xs text-muted mb-1">Val Acc</div>
                            <div className="text-lg font-semibold text-blue-600">
                                {valAcc ? (valAcc * 100).toFixed(2) + '%' : '—'}
                            </div>
                        </div>
                    </div>

                    {/* Configuration */}
                    <div className="card p-3">
                        <h3 className="text-sm font-semibold mb-2">Configuration</h3>
                        <div className="space-y-1.5">
                            <div className="flex justify-between text-xs">
                                <span className="text-muted">Learning Rate</span>
                                <span className="font-semibold">{config?.learningRate || '—'}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span className="text-muted">Batch Size</span>
                                <span className="font-semibold">{config?.batchSize || '—'}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span className="text-muted">Optimizer</span>
                                <span className="font-semibold capitalize">{config?.optimizer || '—'}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                                <span className="text-muted">Model</span>
                                <span className="font-semibold capitalize">{config?.datasetType || '—'}</span>
                            </div>
                        </div>
                    </div>

                    {/* Training Logs */}
                    <div className="card flex-1 min-h-0 flex flex-col p-0 overflow-hidden">
                        <div className="p-3 border-b bg-gray-50 flex-shrink-0">
                            <h3 className="text-sm font-semibold">Training Logs</h3>
                        </div>
                        <div className="flex-1 overflow-y-auto custom-scrollbar">
                            <table className="w-full text-xs text-left">
                                <thead className="text-muted border-b sticky top-0 bg-white z-10">
                                    <tr>
                                        <th className="pl-4 py-2 font-medium">Epoch</th>
                                        <th className="py-2 font-medium">Train Loss</th>
                                        <th className="py-2 font-medium">Train Acc</th>
                                        <th className="py-2 font-medium">Val Loss</th>
                                        <th className="pr-4 py-2 font-medium">Val Acc</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-100">
                                    {Array.isArray(lossHistory) && lossHistory.map((loss, i) => {
                                        // Only show row if ALL data is available for this epoch
                                        const hasAllData =
                                            loss !== undefined &&
                                            accHistory[i] !== undefined &&
                                            valLossHistory[i] !== undefined &&
                                            valAccHistory[i] !== undefined;

                                        // Also exclude rows where all values are zero (initial state)
                                        const hasValidData =
                                            loss > 0 ||
                                            accHistory[i] > 0 ||
                                            valLossHistory[i] > 0 ||
                                            valAccHistory[i] > 0;

                                        if (!hasAllData || !hasValidData) return null;

                                        return (
                                            <tr key={i} className="hover:bg-gray-50">
                                                <td className="pl-4 py-2">{i + 1}</td>
                                                <td className="py-2 text-red-600 font-medium">{loss.toFixed(4)}</td>
                                                <td className="py-2 text-green-600 font-medium">{(accHistory[i] * 100).toFixed(2)}%</td>
                                                <td className="py-2 text-orange-600 font-medium">{valLossHistory[i].toFixed(4)}</td>
                                                <td className="pr-4 py-2 text-blue-600 font-medium">{(valAccHistory[i] * 100).toFixed(2)}%</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                            {(!lossHistory || lossHistory.length === 0 ||
                                !lossHistory.some((loss, i) =>
                                    loss !== undefined &&
                                    accHistory[i] !== undefined &&
                                    valLossHistory[i] !== undefined &&
                                    valAccHistory[i] !== undefined
                                )) && (
                                    <div className="text-center py-8 text-muted text-xs">
                                        Waiting for first epoch...
                                    </div>
                                )}
                        </div>
                    </div>
                </div>

                {/* Right Column - Charts Grid 2x2 */}
                <div className="grid grid-rows-2 gap-3 min-h-0">
                    {/* Top Row */}
                    <div className="grid grid-cols-2 gap-3">
                        {/* Training Loss Chart */}
                        <div className="card p-0 overflow-hidden flex flex-col">
                            <div className="p-2 border-b bg-gray-50 flex-shrink-0">
                                <h4 className="text-xs font-semibold text-red-600">Training Loss</h4>
                            </div>
                            <div className="flex-1 p-2 flex items-center justify-center">
                                {trainLossDataPoints.length > 0 ? (
                                    <Line data={trainLossData} options={chartOptions} />
                                ) : (
                                    <div className="text-xs text-muted">No data yet</div>
                                )}
                            </div>
                        </div>

                        {/* Validation Loss Chart */}
                        <div className="card p-0 overflow-hidden flex flex-col">
                            <div className="p-2 border-b bg-gray-50 flex-shrink-0">
                                <h4 className="text-xs font-semibold text-orange-600">Validation Loss</h4>
                            </div>
                            <div className="flex-1 p-2 flex items-center justify-center">
                                {valLossDataPoints.length > 0 ? (
                                    <Line data={valLossData} options={chartOptions} />
                                ) : (
                                    <div className="text-xs text-muted">No data yet</div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Bottom Row */}
                    <div className="grid grid-cols-2 gap-3">
                        {/* Training Accuracy Chart */}
                        <div className="card p-0 overflow-hidden flex flex-col">
                            <div className="p-2 border-b bg-gray-50 flex-shrink-0">
                                <h4 className="text-xs font-semibold text-green-600">Training Accuracy</h4>
                            </div>
                            <div className="flex-1 p-2 flex items-center justify-center">
                                {trainAccDataPoints.length > 0 ? (
                                    <Line data={trainAccData} options={chartOptions} />
                                ) : (
                                    <div className="text-xs text-muted">No data yet</div>
                                )}
                            </div>
                        </div>

                        {/* Validation Accuracy Chart */}
                        <div className="card p-0 overflow-hidden flex flex-col">
                            <div className="p-2 border-b bg-gray-50 flex-shrink-0">
                                <h4 className="text-xs font-semibold text-blue-600">Validation Accuracy</h4>
                            </div>
                            <div className="flex-1 p-2 flex items-center justify-center">
                                {valAccDataPoints.length > 0 ? (
                                    <Line data={valAccData} options={chartOptions} />
                                ) : (
                                    <div className="text-xs text-muted">No data yet</div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
