import React from 'react'

const InfoIcon = ({ text }) => (
    <div className="group relative inline-flex items-center ml-1.5">
        <svg className="w-3.5 h-3.5 text-gray-400 hover:text-blue-500 cursor-help transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 p-2 bg-gray-900 text-white text-[10px] leading-tight rounded-md shadow-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 text-center border border-gray-700">
            {text}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-gray-900"></div>
        </div>
    </div>
)

export default function TrainingConfig({ config = {}, onChange, onStart, datasetReady, datasetInfo, isTraining }) {
    const handleChange = (field, value) => {
        onChange({ ...config, [field]: value })
    }

    const getTips = () => {
        const tips = [
            "Ensure your dataset is balanced for best results.",
            "Monitor validation loss to detect overfitting early.",
            "Start with a simple model and increase complexity as needed."
        ];

        if (config.batchSize && config.batchSize > 64) {
            tips.push("Large batch sizes may require a higher learning rate.");
        } else if (config.batchSize && config.batchSize < 16) {
            tips.push("Small batch sizes provide noisy gradients.");
        }

        if (config.epochs && config.epochs < 10) {
            tips.push("Low epoch count might lead to underfitting.");
        } else if (config.epochs && config.epochs > 100) {
            tips.push("High epoch count increases overfitting risk.");
        }

        return tips;
    }

    const getEstimatedTime = () => {
        if (!datasetInfo || !datasetInfo.fileCount) return 'N/A';

        const samples = datasetInfo.fileCount || 0;
        const epochs = config.epochs || 10;
        const batchSize = config.batchSize || 32;

        const batchesPerEpoch = Math.ceil(samples / batchSize);
        const secondsPerEpoch = batchesPerEpoch * 0.5;
        const totalSeconds = secondsPerEpoch * epochs;

        if (totalSeconds < 60) {
            return '~' + Math.round(totalSeconds) + 's';
        } else if (totalSeconds < 3600) {
            return '~' + Math.round(totalSeconds / 60) + 'm';
        } else {
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.round((totalSeconds % 3600) / 60);
            return '~' + hours + 'h ' + minutes + 'm';
        }
    }

    // Validation: epochs must be a positive integer
    const epochsValid = Number.isInteger(config.epochs) && config.epochs >= 1

    return (
        <div className="flex flex-col h-full animate-fade-in gap-4">
            <div className="flex-shrink-0 flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">Training Configuration</h2>
                    <p className="text-xs text-muted">Configure hyperparameters for your model</p>
                </div>
                {!datasetReady && (
                    <div className="px-2 py-1 rounded bg-red-50 border border-red-200 text-red-600 text-xs flex items-center gap-1">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        No dataset
                    </div>
                )}
            </div>

            <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
                <div className="col-span-12 md:col-span-8 flex flex-col gap-4 overflow-y-auto custom-scrollbar pr-1">
                    <div className="card">
                        <h3 className="text-sm font-semibold mb-3">Core Parameters</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="label text-xs flex items-center">
                                    Epochs
                                    <InfoIcon text="The number of times the entire dataset is passed forward and backward through the neural network." />
                                </label>
                                <input
                                    type="number"
                                    className={`input py-1.5 text-sm ${config.epochs === '' || !epochsValid ? 'border-red-300 focus:ring-red-200' : ''}`}
                                    value={config.epochs ?? ''}
                                    onChange={(e) => {
                                        const v = e.target.value
                                        if (v === '') {
                                            handleChange('epochs', '')
                                        } else {
                                            const n = parseInt(v, 10)
                                            handleChange('epochs', Number.isNaN(n) ? '' : Math.max(1, n))
                                        }
                                    }}
                                    min="1"
                                    max="1000"
                                    required
                                />
                                {config.epochs === '' || !epochsValid ? (
                                    <div className="text-[10px] text-red-600 mt-1">Epochs is required and must be at least 1.</div>
                                ) : null}
                            </div>
                            <div>
                                <label className="label text-xs flex items-center">
                                    Batch Size
                                    <InfoIcon text="The number of training examples utilized in one iteration." />
                                </label>
                                <input
                                    type="number"
                                    className="input py-1.5 text-sm"
                                    value={config.batchSize}
                                    onChange={(e) => handleChange('batchSize', parseInt(e.target.value) || 1)}
                                    min="1"
                                    max="512"
                                />
                            </div>
                            <div>
                                <label className="label text-xs flex items-center">
                                    Learning Rate
                                    <InfoIcon text="Controls how much to change the model in response to the estimated error." />
                                </label>
                                <input
                                    type="number"
                                    className="input py-1.5 text-sm"
                                    value={config.learningRate}
                                    onChange={(e) => handleChange('learningRate', parseFloat(e.target.value) || 0.001)}
                                    step="0.0001"
                                />
                            </div>
                            <div>
                                <label className="label text-xs flex items-center">
                                    Optimizer
                                    <InfoIcon text="Algorithm used to update model weights." />
                                </label>
                                <select
                                    className="select py-1.5 text-sm w-full"
                                    value={config.optimizer}
                                    onChange={(e) => handleChange('optimizer', e.target.value)}
                                >
                                    <option value="adam">Adam</option>
                                    <option value="sgd">SGD</option>
                                    <option value="rmsprop">RMSprop</option>
                                    <option value="adamw">AdamW</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <div className="card">
                        <h3 className="text-sm font-semibold mb-3">Advanced Settings</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="label text-xs flex items-center">
                                    Validation Split
                                    <InfoIcon text="Fraction of training data used for validation. Set to 0 for automatic 20% split (recommended to prevent overfitting)." />
                                </label>
                                <input
                                    type="number"
                                    className="input py-1.5 text-sm"
                                    value={config.validationSplit ?? 0}
                                    onChange={(e) => handleChange('validationSplit', parseFloat(e.target.value) || 0)}
                                    step="0.05"
                                    min="0"
                                    max="0.5"
                                />
                            </div>
                            <div>
                                <label className="label text-xs flex items-center">
                                    Early Stopping (Patience)
                                    <InfoIcon text="Epochs with no improvement before stopping. Set to 0 for automatic patience of 20 epochs (recommended to prevent overfitting)." />
                                </label>
                                <input
                                    type="number"
                                    className="input py-1.5 text-sm"
                                    value={config.patience ?? 0}
                                    onChange={(e) => handleChange('patience', parseInt(e.target.value) || 0)}
                                    min="0"
                                    max="50"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="card flex-1">
                        <h3 className="text-sm font-semibold mb-3">Dataset Context</h3>
                        {datasetInfo ? (
                            <div className="space-y-2">
                                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span className="text-xs text-muted">Total Samples</span>
                                    <span className="text-sm font-semibold text-gray-700">{datasetInfo.fileCount?.toLocaleString() || 0}</span>
                                </div>
                                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span className="text-xs text-muted">Classes</span>
                                    <span className="text-sm font-semibold text-gray-700">{datasetInfo.structure?.classes?.length || 0}</span>
                                </div>
                                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span className="text-xs text-muted">Format</span>
                                    <span className="text-sm font-semibold text-gray-700 capitalize">{datasetInfo.structure?.type || 'N/A'}</span>
                                </div>
                                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                                    <span className="text-xs text-muted">Dataset Type</span>
                                    <span className="text-sm font-semibold text-gray-700">
                                        {datasetInfo.structure?.type === 'csv' ? 'Tabular Data' :
                                            datasetInfo.structure?.type === 'yolo' ? 'Object Detection' :
                                                datasetInfo.structure?.type === 'cifar' ? 'Image Classification' :
                                                    'Image Classification'}
                                    </span>
                                </div>

                                <div className="flex justify-between items-center py-2">
                                    <span className="text-xs text-muted">Status</span>
                                    <span className="text-xs font-medium text-green-600 flex items-center gap-1">
                                        <span className="w-2 h-2 rounded-full bg-green-500"></span>
                                        Ready for Training
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <div className="text-xs text-muted italic p-2 text-center bg-gray-50 rounded">
                                No dataset selected. Please upload a dataset first.
                            </div>
                        )}
                    </div>
                </div>

                <div className="col-span-12 md:col-span-4 flex flex-col gap-4">
                    <div className="card">
                        <h3 className="text-sm font-semibold mb-3">Configuration Summary</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between items-center text-xs border-b border-gray-100 pb-2">
                                <span className="text-muted">Epochs</span>
                                <span className="font-medium">{epochsValid ? config.epochs : '-'}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs border-b border-gray-100 pb-2">
                                <span className="text-muted">Batch Size</span>
                                <span className="font-medium">{config.batchSize}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs border-b border-gray-100 pb-2">
                                <span className="text-muted">Learning Rate</span>
                                <span className="font-medium">{config.learningRate}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs border-b border-gray-100 pb-2">
                                <span className="text-muted">Optimizer</span>
                                <span className="font-medium uppercase">{config.optimizer}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs border-b border-gray-100 pb-2">
                                <span className="text-muted">Validation Split</span>
                                <span className="font-medium">{config.validationSplit ?? 0}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs border-b border-gray-100 pb-2">
                                <span className="text-muted">Patience</span>
                                <span className="font-medium">{config.patience ?? 0}</span>
                            </div>
                            <div className="flex justify-between items-center text-xs pt-1">
                                <span className="text-muted">Est. Training Time</span>
                                <span className="font-semibold text-blue-600">{getEstimatedTime()}</span>
                            </div>
                        </div>
                    </div>

                    <div className="card flex-1 bg-blue-50 border-blue-100 flex flex-col relative overflow-hidden">
                        <div className="absolute -bottom-4 -right-4 text-blue-100 opacity-50 pointer-events-none">
                            <svg className="w-32 h-32" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                        </div>

                        <h3 className="text-sm font-semibold mb-3 text-blue-800 flex items-center gap-2 flex-shrink-0 relative z-10">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Smart Tips
                        </h3>

                        <div className="flex-1 overflow-y-auto custom-scrollbar relative z-10 flex flex-col">
                            <ul className="space-y-3 mb-4">
                                {getTips().map((tip, index) => (
                                    <li key={index} className="text-xs text-blue-700 flex items-start gap-2">
                                        <span className="mt-1.5 w-1 h-1 rounded-full bg-blue-500 flex-shrink-0" />
                                        <span className="leading-relaxed">{tip}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    <div className="card bg-gray-50 border-gray-200">
                        <div className="text-xs text-muted mb-3 text-center">
                            Ready to start training?
                        </div>
                        <button
                            onClick={onStart}
                            disabled={!datasetReady || isTraining || !epochsValid}
                            className={`btn btn-primary w-full justify-center py-2.5 ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {isTraining ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Training in Progress...
                                </>
                            ) : (
                                <>
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    Start Training
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
