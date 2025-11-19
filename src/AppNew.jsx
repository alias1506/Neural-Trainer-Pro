import React, { useState, useEffect } from 'react'
import Swal from 'sweetalert2'
import DatasetSelector from './components/DatasetSelector.jsx'
import TrainingConfig from './components/TrainingConfig.jsx'
import TrainingProgress from './components/TrainingProgress.jsx'
import axios from 'axios'

const API_URL = 'http://localhost:3001/api'
const WS_URL = 'ws://localhost:3002'

export default function App() {
  const [section, setSection] = useState('dataset')
  const [datasetInfo, setDatasetInfo] = useState(null)
  const [config, setConfig] = useState({ epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' })
  const [progress, setProgress] = useState({ status: 'idle', epoch: 0, trainLoss: 0, trainAcc: 0, valLoss: 0, valAcc: 0, timeMs: 0, lossHistory: [], accHistory: [] })

  // Setup WebSocket
  useEffect(() => {
    const websocket = new WebSocket(WS_URL)
    
    websocket.onopen = () => {
      console.log('✓ WebSocket connected')
    }
    
    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleTrainingUpdate(data)
      } catch (e) {
        console.error('WebSocket parse error:', e)
      }
    }
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected')
    }
    
    setWs(websocket)
    
    return () => {
      websocket.close()
    }
  }, [])

  const handleTrainingUpdate = (data) => {
    console.log('Training update:', data)
    
    if (data.type === 'status') {
      setProgress(prev => ({ ...prev, status: 'preparing', message: data.message }))
    } else if (data.type === 'info') {
      setDatasetInfo(prev => ({ ...prev, ...data }))
    } else if (data.type === 'device') {
      Swal.fire({
        icon: 'info',
        title: 'Training Device',
        text: `Using ${data.device} for training`,
        timer: 2000,
        showConfirmButton: false
      })
    } else if (data.type === 'epoch') {
      setProgress(prev => ({
        ...prev,
        status: 'training',
        epoch: data.epoch,
        totalEpochs: data.totalEpochs,
        trainLoss: data.trainLoss,
        trainAcc: data.trainAcc,
        valLoss: data.valLoss,
        valAcc: data.valAcc,
        timeMs: data.elapsed * 1000,
        lossHistory: [...prev.lossHistory, data.trainLoss],
        accHistory: [...prev.accHistory, data.trainAcc]
      }))
    } else if (data.type === 'complete') {
      setProgress(prev => ({ ...prev, status: 'done' }))
      
      Swal.fire({
        icon: 'success',
        title: 'Training Complete!',
        html: `
          <div class="text-left">
            <p class="mb-2"><strong>Validation Accuracy:</strong> ${(data.finalValAcc * 100).toFixed(2)}%</p>
            <p class="mb-2"><strong>Model saved at:</strong></p>
            <p class="text-sm text-gray-600 bg-gray-100 p-2 rounded">${data.modelPath}</p>
          </div>
        `,
        confirmButtonColor: '#3b82f6'
      }).then(() => {
        setSection('dataset')
      })
    } else if (data.type === 'error') {
      setProgress(prev => ({ ...prev, status: 'error' }))
      Swal.fire({
        icon: 'error',
        title: 'Training Error',
        text: data.message,
        confirmButtonColor: '#ef4444'
      })
    }
  }

  const handleDatasetUpload = async (files) => {
    try {
      const formData = new FormData()
      
      // Add all files
      for (const file of files) {
        formData.append('files', file)
      }
      
      const response = await axios.post(`${API_URL}/upload-dataset`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      if (response.data.success) {
        const info = response.data
        setDatasetInfo(info)
        
        // Show structure detection
        await Swal.fire({
          icon: 'success',
          title: 'Dataset Uploaded',
          html: `
            <div class="text-left">
              <p class="mb-2"><strong>Files:</strong> ${info.fileCount}</p>
              <p class="mb-2"><strong>Structure:</strong> ${info.structure.structure}</p>
              <p class="text-sm text-gray-600 mt-3">Dataset is ready for training!</p>
            </div>
          `,
          confirmButtonColor: '#3b82f6'
        })
        
        setSection('config')
      }
    } catch (error) {
      console.error('Upload error:', error)
      Swal.fire({
        icon: 'error',
        title: 'Upload Failed',
        text: error.response?.data?.error || error.message,
        confirmButtonColor: '#ef4444'
      })
    }
  }

  const startTraining = async () => {
    try {
      setSection('progress')
      setProgress({ 
        status: 'preparing', 
        epoch: 0, 
        trainLoss: 0, 
        trainAcc: 0, 
        valLoss: 0, 
        valAcc: 0, 
        timeMs: 0,
        lossHistory: [],
        accHistory: [],
        message: 'Starting training...'
      })
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const response = await axios.post(`${API_URL}/train`, {
        datasetPath: datasetInfo.datasetPath,
        config
      })
      
      if (response.data.success) {
        console.log('Training started successfully')
      }
    } catch (error) {
      console.error('Training error:', error)
      setProgress(prev => ({ ...prev, status: 'error' }))
      Swal.fire({
        icon: 'error',
        title: 'Training Failed',
        text: error.response?.data?.error || error.message,
        confirmButtonColor: '#ef4444'
      })
      setSection('config')
    }
  }

  const cancelTraining = async () => {
    const result = await Swal.fire({
      title: 'Cancel Training?',
      text: 'Training will be stopped. Progress will be lost.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Yes, Cancel',
      cancelButtonText: 'Continue Training',
      confirmButtonColor: '#dc2626',
      cancelButtonColor: '#2563eb'
    })

    if (result.isConfirmed) {
      // TODO: Implement cancel endpoint
      setProgress(prev => ({ ...prev, status: 'cancelled' }))
      setSection('config')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 shadow-lg">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold">Neural Trainer Pro</h1>
          <p className="text-blue-100 mt-2">Train any image classification model with Python + PyTorch</p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-8">
        {section === 'dataset' && (
          <DatasetSelector 
            onFilesSelected={handleDatasetUpload}
            datasetInfo={datasetInfo}
          />
        )}
        
        {section === 'config' && (
          <TrainingConfig 
            config={config} 
            onChange={setConfig} 
            onStart={startTraining}
            datasetReady={!!datasetInfo}
          />
        )}
        
        {section === 'progress' && (
          <TrainingProgress 
            progress={progress} 
            config={config}
            onCancel={cancelTraining}
          />
        )}
      </main>

      <footer className="border-t border-gray-200 p-4 text-center text-sm text-gray-600 mt-12">
        <p>© 2025 Neural Trainer Pro | Python + PyTorch Backend</p>
      </footer>
    </div>
  )
}
