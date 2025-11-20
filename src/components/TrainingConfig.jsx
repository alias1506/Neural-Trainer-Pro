import React, { useState } from 'react'

const InfoButton = ({ info }) => {
  const [show, setShow] = useState(false)
  return (
    <div className="relative inline-block ml-2">
      <button
        type="button"
        className="inline-flex items-center justify-center w-5 h-5 text-xs font-semibold text-blue-600 bg-blue-100 rounded-full hover:bg-blue-200 focus-ring"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        aria-label="More information"
      >
        i
      </button>
      {show && (
        <div className="absolute z-10 left-0 top-6 w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-gray-700">
          {info}
        </div>
      )}
    </div>
  )
}

export default function TrainingConfig({ config, onChange, onStart, datasetReady }) {
  const set = (k, v) => onChange({ ...config, [k]: v })
  
  // Ensure config has valid defaults to prevent NaN
  const epochs = config.epochs || 10
  const batchSize = config.batchSize || 32
  const learningRate = config.learningRate || 0.001
  const optimizer = config.optimizer || 'adam'
  
  // Calculate stats
  const estimatedTime = Math.ceil((epochs * batchSize * 0.5) / 60)
  const iterations = Math.ceil(1000 / batchSize) * epochs // Assuming 1000 samples
  const memoryUsage = batchSize * 0.5 // Rough MB estimate
  
  return (
    <div className="bg-gray-100">
      {/* Stats Row - AdminLTE Style Info Boxes */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
        {/* Epochs Box */}
        <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Total Epochs</div>
              <div className="text-2xl font-bold mt-1">{epochs}</div>
            </div>
            <div className="text-3xl opacity-30">🔄</div>
          </div>
        </div>

        {/* Batch Size Box */}
        <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Batch Size</div>
              <div className="text-2xl font-bold mt-1">{batchSize}</div>
            </div>
            <div className="text-3xl opacity-30">📦</div>
          </div>
        </div>

        {/* Learning Rate Box */}
        <div className="bg-gradient-to-br from-orange-500 to-red-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Learning Rate</div>
              <div className="text-2xl font-bold mt-1">{learningRate}</div>
            </div>
            <div className="text-3xl opacity-30">⚡</div>
          </div>
        </div>

        {/* Est. Time Box */}
        <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg shadow-lg p-3 text-white">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase font-semibold opacity-80">Est. Time</div>
              <div className="text-2xl font-bold mt-1">~{estimatedTime}m</div>
            </div>
            <div className="text-3xl opacity-30">⏱️</div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Left Column - Configuration Form */}
            <div className="lg:col-span-2 space-y-4">
              {/* Hyperparameters Card */}
              <div className="bg-white rounded-lg shadow-md">
                <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-4 py-3 rounded-t-lg">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <span>⚙️</span>
                    Hyperparameters Configuration
                  </h3>
                </div>
                <div className="p-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <label className="block">
                      <div className="flex items-center mb-2">
                        <span className="font-semibold text-gray-700">Epochs</span>
                        <InfoButton info="Number of complete passes through the entire training dataset. More epochs can improve accuracy but may cause overfitting." />
                      </div>
                      <input type="number" min="1" defaultValue={epochs} onChange={(e)=>set('epochs', parseInt(e.target.value) || 10)} className="w-full border-2 border-gray-300 rounded-lg p-3 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 bg-white text-gray-800 font-medium transition-all" />
                    </label>
                    <label className="block">
                      <div className="flex items-center mb-2">
                        <span className="font-semibold text-gray-700">Batch Size</span>
                        <InfoButton info="Number of training examples used in one iteration. Larger batches train faster but use more memory. Common values: 16, 32, 64, 128." />
                      </div>
                      <input type="number" min="1" defaultValue={batchSize} onChange={(e)=>set('batchSize', parseInt(e.target.value) || 32)} className="w-full border-2 border-gray-300 rounded-lg p-3 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 bg-white text-gray-800 font-medium transition-all" />
                    </label>
                    <label className="block">
                      <div className="flex items-center mb-2">
                        <span className="font-semibold text-gray-700">Learning Rate</span>
                        <InfoButton info="Controls how much to adjust weights during training. Smaller values (0.001-0.0001) are safer but slower. Larger values train faster but may miss optimal solutions." />
                      </div>
                      <input type="number" step="0.0001" defaultValue={learningRate} onChange={(e)=>set('learningRate', parseFloat(e.target.value) || 0.001)} className="w-full border-2 border-gray-300 rounded-lg p-3 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 bg-white text-gray-800 font-medium transition-all" />
                    </label>
                    <label className="block">
                      <div className="flex items-center mb-2">
                        <span className="font-semibold text-gray-700">Optimizer</span>
                        <InfoButton info="Algorithm used to update model weights. Adam: adaptive, works well generally. SGD: simple, requires tuning. RMSprop: good for recurrent networks." />
                      </div>
                      <select defaultValue={optimizer} onChange={(e)=>set('optimizer', e.target.value)} className="w-full border-2 border-gray-300 rounded-lg p-3 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 bg-white text-gray-800 font-medium transition-all">
                        <option>adam</option>
                        <option>sgd</option>
                        <option>rmsprop</option>
                      </select>
                    </label>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <button 
                  className="w-full px-6 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed focus-ring font-bold text-lg shadow-lg transform hover:scale-105 transition-all flex items-center justify-center gap-3" 
                  disabled={!datasetReady} 
                  onClick={onStart}>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {datasetReady ? 'Start Training Now!' : 'Upload Dataset First'}
                </button>
              </div>
            </div>

            {/* Right Column - Info Panels */}
            <div className="space-y-4">
              {/* Quick Guide */}
              <div className="bg-white rounded-lg shadow-md">
                <div className="bg-gradient-to-r from-yellow-500 to-orange-600 text-white px-4 py-3 rounded-t-lg">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <span>💡</span>
                    Quick Guide
                  </h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3 text-sm">
                    <div className="flex items-start gap-2">
                      <span className="text-green-500 font-bold">✓</span>
                      <div>
                        <div className="font-semibold text-gray-800">Start Small</div>
                        <div className="text-gray-600 text-xs">Begin with 5-10 epochs for testing</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-green-500 font-bold">✓</span>
                      <div>
                        <div className="font-semibold text-gray-800">Batch Size</div>
                        <div className="text-gray-600 text-xs">32 is recommended for most cases</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-green-500 font-bold">✓</span>
                      <div>
                        <div className="font-semibold text-gray-800">Learning Rate</div>
                        <div className="text-gray-600 text-xs">0.001 is a safe default value</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-green-500 font-bold">✓</span>
                      <div>
                        <div className="font-semibold text-gray-800">Optimizer</div>
                        <div className="text-gray-600 text-xs">Adam works well for most tasks</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Training Tips */}
              <div className="bg-white rounded-lg shadow-md">
                <div className="bg-gradient-to-r from-pink-500 to-rose-600 text-white px-4 py-3 rounded-t-lg">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <span>🎯</span>
                    Pro Tips
                  </h3>
                </div>
                <div className="p-4">
                  <div className="space-y-3 text-xs text-gray-700">
                    <div className="bg-gray-50 p-3 rounded-lg border-l-4 border-pink-500">
                      <div className="font-semibold mb-1">Monitor Overfitting</div>
                      <div>If validation loss increases while training loss decreases, reduce epochs</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg border-l-4 border-pink-500">
                      <div className="font-semibold mb-1">GPU Memory</div>
                      <div>Reduce batch size if you encounter out-of-memory errors</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg border-l-4 border-pink-500">
                      <div className="font-semibold mb-1">Learning Rate</div>
                      <div>Too high causes unstable training, too low makes it very slow</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
      </div>
    </div>
  )
}