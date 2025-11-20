import React, { useState, useEffect } from 'react'
import Swal from 'sweetalert2'
import Sidebar from './components/Sidebar.jsx'
import DatasetSelector from './components/DatasetSelector.jsx'
import TrainingConfig from './components/TrainingConfig.jsx'
import TrainingProgress from './components/TrainingProgress.jsx'
import ModelExport from './components/ModelExport.jsx'
import TrainingHistory from './components/TrainingHistory.jsx'
import axios from 'axios'

const API_URL = 'http://localhost:3001/api'
const WS_URL = 'ws://localhost:3002'

export default function App() {
  const [collapsed, setCollapsed] = useState(false)
  const [section, setSection] = useState('dataset')
  const [uploadProgress, setUploadProgress] = useState(null)
  const [datasetInfo, setDatasetInfo] = useState(() => {
    // Load from localStorage on mount
    const saved = localStorage.getItem('datasetInfo')
    return saved ? JSON.parse(saved) : null
  })
  const [config, setConfig] = useState({ epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' })
  const [progress, setProgress] = useState({ status: 'idle', epoch: 0, trainLoss: 0, trainAcc: 0, valLoss: 0, valAcc: 0, timeMs: 0, lossHistory: [], accHistory: [], modelPath: '' })
  const [trainingHistory, setTrainingHistory] = useState(() => {
    const saved = localStorage.getItem('trainingHistory')
    return saved ? JSON.parse(saved) : []
  })

  // Save training history to localStorage
  useEffect(() => {
    localStorage.setItem('trainingHistory', JSON.stringify(trainingHistory))
  }, [trainingHistory])

  // Save datasetInfo to localStorage whenever it changes (only 1 dataset at a time)
  useEffect(() => {
    if (datasetInfo) {
      // Always overwrite with current dataset (no multiple datasets stored)
      localStorage.setItem('datasetInfo', JSON.stringify(datasetInfo))
    } else {
      // Clear localStorage when no dataset
      localStorage.removeItem('datasetInfo')
    }
  }, [datasetInfo])

  // Setup WebSocket
  useEffect(() => {
    const websocket = new WebSocket(WS_URL)
    
    websocket.onopen = () => {
      // WebSocket connected
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
      // WebSocket disconnected
    }
    
    return () => {
      websocket.close()
    }
  }, [])

  const handleTrainingUpdate = async (data) => {
    if (data.type === 'status') {
      console.clear()
      console.log('🚀 STATUS:', data.message)
      setProgress(prev => ({ ...prev, status: 'preparing', message: data.message }))
    } else if (data.type === 'info') {
      console.log('ℹ️ TRAINING INFO:')
      console.log(`  📊 Total Images: ${data.numImages}`)
      console.log(`  🏷️  Classes: ${data.numClasses}`)
      setDatasetInfo(prev => ({ ...prev, ...data }))
    } else if (data.type === 'device') {
      console.log('💻 DEVICE:', data.device)
      Swal.fire({
        icon: 'info',
        title: 'Training Device',
        text: `Using ${data.device} for training`,
        timer: 2000,
        showConfirmButton: false
      })
    } else if (data.type === 'epoch') {
      console.clear()
      console.log('═'.repeat(60))
      console.log(`📊 EPOCH ${data.epoch}/${data.totalEpochs}`)
      console.log('═'.repeat(60))
      console.log(`🔴 Train Loss:      ${data.trainLoss.toFixed(4)}`)
      console.log(`🟢 Train Accuracy:  ${(data.trainAcc * 100).toFixed(2)}%`)
      console.log(`🔵 Val Loss:        ${data.valLoss.toFixed(4)}`)
      console.log(`🟣 Val Accuracy:    ${(data.valAcc * 100).toFixed(2)}%`)
      console.log(`⏱️  Time Elapsed:    ${data.elapsed.toFixed(2)}s`)
      console.log('═'.repeat(60))
      
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
      
      // Update running history entry
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          updated[0] = {
            ...updated[0],
            currentEpoch: data.epoch,
            metrics: { acc: data.valAcc, loss: data.trainLoss }
          }
          return updated
        }
        return prev
      })
    } else if (data.type === 'complete') {
      console.clear()
      console.log('═'.repeat(60))
      console.log('✅ TRAINING COMPLETE!')
      console.log('═'.repeat(60))
      console.log(`🎯 Final Validation Accuracy: ${(data.finalValAcc * 100).toFixed(2)}%`)
      console.log(`📁 Model saved at: ${data.modelPath}`)
      console.log('═'.repeat(60))
      
      setProgress(prev => ({ ...prev, status: 'done', modelPath: data.modelPath }))
      
      // Clean up dataset after successful training
      try {
        await axios.post(`${API_URL}/clean-dataset`)
      } catch (cleanError) {
        console.warn('Dataset cleanup warning:', cleanError)
      }
      
      // Clear dataset from program state and localStorage
      setDatasetInfo(null)
      setSelectedFiles([])
      localStorage.removeItem('datasetInfo')
      
      // Update history entry to done
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          updated[0] = {
            ...updated[0],
            status: 'done',
            metrics: { acc: data.finalValAcc, loss: progress.trainLoss },
            modelPath: data.modelPath,
            endTime: new Date().toISOString()
          }
          return updated
        }
        return prev
      })
      
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
    } else if (data.type === 'cancelled') {
      console.clear()
      console.log('🛑 TRAINING CANCELLED')
      
      setProgress(prev => ({ ...prev, status: 'cancelled' }))
      
      // Update history entry to cancelled
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          updated[0] = {
            ...updated[0],
            status: 'cancelled',
            endTime: new Date().toISOString()
          }
          return updated
        }
        return prev
      })
    } else if (data.type === 'error') {
      console.clear()
      console.log('═'.repeat(60))
      console.error('❌ TRAINING ERROR')
      console.log('═'.repeat(60))
      console.error(data.message)
      console.log('═'.repeat(60))
      
      setProgress(prev => ({ ...prev, status: 'error' }))
      
      // Update history entry to failed
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          updated[0] = {
            ...updated[0],
            status: 'failed',
            error: data.message,
            endTime: new Date().toISOString()
          }
          return updated
        }
        return prev
      })
      
      Swal.fire({
        icon: 'error',
        title: 'Training Error',
        text: data.message,
        confirmButtonColor: '#ef4444'
      })
    }
  }

  const handleFormatChange = (newFormat) => {
    if (datasetInfo) {
      const updatedInfo = {
        ...datasetInfo,
        structure: {
          ...datasetInfo.structure,
          type: newFormat
        }
      }
      setDatasetInfo(updatedInfo)
      localStorage.setItem('datasetInfo', JSON.stringify(updatedInfo))
      
      Swal.fire({
        icon: 'success',
        title: 'Format Updated',
        text: `Dataset format changed to: ${newFormat}`,
        timer: 2000,
        showConfirmButton: false
      })
    }
  }

  const handleDatasetUpload = async (files, isRemove = false) => {
    // Handle dataset removal
    if (isRemove) {
      try {
        // Clean up server uploads folder
        await axios.post(`${API_URL}/clean-uploads`)
        
        setDatasetInfo(null)
        localStorage.removeItem('datasetInfo')
        
        Swal.fire({
          icon: 'success',
          title: 'Cleaned Up',
          text: 'Old datasets and models have been removed.',
          timer: 2000,
          showConfirmButton: false
        })
      } catch (error) {
        console.error('Cleanup error:', error)
      }
      // Refresh the page to reset everything
      window.location.reload()
      return
    }

    try {
      // Clear any existing dataset data before uploading new one
      setDatasetInfo(null)
      localStorage.removeItem('datasetInfo')
      
      // Clean up old datasets on server before uploading new one
      try {
        await axios.post(`${API_URL}/clean-uploads`)
      } catch (cleanError) {
        console.warn('Server cleanup warning:', cleanError)
        // Continue with upload even if cleanup fails
      }
      
      const totalFiles = files.length
      const startTime = Date.now()
      
      // Initialize upload progress
      setUploadProgress({
        uploading: true,
        percentage: 0,
        filesUploaded: 0,
        totalFiles,
        timeRemaining: null
      })
      
      const formData = new FormData()
      
      // Build a path mapping: originalFilename -> fullPath
      const pathMapping = {}
      
      // Add files to FormData and build path mapping
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const fullPath = (file.webkitRelativePath || file.name).replace(/\\/g, '/')
        
        // Create unique key for this file (using index to handle duplicate filenames)
        const fileKey = `file_${i}_${file.name}`
        pathMapping[fileKey] = fullPath
        
        // Append file with the unique key as filename
        formData.append('files', file, fileKey)
      }
      
      // Create a JSON blob and add it as a file (NOT in header due to size limits)
      const pathMappingBlob = new Blob([JSON.stringify(pathMapping)], { type: 'application/json' })
      formData.append('pathMapping', pathMappingBlob, 'pathMapping.json')
      
      const response = await axios.post(`${API_URL}/upload-dataset`, formData, {
        headers: { 
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          const elapsedTime = Date.now() - startTime
          const estimatedTotal = (elapsedTime / progressEvent.loaded) * progressEvent.total
          const timeRemaining = Math.round((estimatedTotal - elapsedTime) / 1000)
          
          setUploadProgress({
            uploading: true,
            percentage,
            filesUploaded: Math.round((percentage / 100) * totalFiles),
            totalFiles,
            timeRemaining: timeRemaining > 0 ? `${timeRemaining}s` : 'Almost done...'
          })
        }
      })
      
      // Clear upload progress
      setUploadProgress(null)
      
      if (response.data.success) {
        const info = response.data
        setDatasetInfo(info)
        // Dataset uploaded successfully - info is shown in the UI, no popup needed
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
      
      // Create a new history entry with 'running' status
      const newHistoryEntry = {
        id: Date.now(),
        date: new Date().toISOString(),
        startTime: new Date().toISOString(),
        status: 'running',
        config: config,
        currentEpoch: 0,
        metrics: { acc: 0, loss: 0 },
        modelPath: null
      }
      setTrainingHistory(prev => [newHistoryEntry, ...prev])
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const response = await axios.post(`${API_URL}/train`, {
        datasetPath: datasetInfo.datasetPath,
        config: {
          ...config,
          datasetType: datasetInfo.structure.type  // Pass the selected dataset type
        }
      })
      
      if (response.data.success) {
        // Training started successfully
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
      text: 'Training will be stopped immediately. Progress will not be saved.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Yes, Stop Now',
      cancelButtonText: 'Continue Training',
      confirmButtonColor: '#dc2626',
      cancelButtonColor: '#2563eb'
    })

    if (result.isConfirmed) {
      try {
        // Call cancel endpoint
        const response = await axios.post(`${API_URL}/cancel-training`)
        
        if (response.data.success) {
          // Update history entry to cancelled
          setTrainingHistory(prev => {
            if (prev.length > 0 && prev[0].status === 'running') {
              const updated = [...prev]
              updated[0] = {
                ...updated[0],
                status: 'cancelled',
                endTime: new Date().toISOString()
              }
              return updated
            }
            return prev
          })
          
          setProgress(prev => ({ ...prev, status: 'cancelled' }))
          
          Swal.fire({
            icon: 'info',
            title: 'Training Cancelled',
            text: 'Training has been stopped successfully.',
            confirmButtonColor: '#3b82f6',
            timer: 2000
          })
        }
      } catch (error) {
        console.error('Cancel error:', error)
        Swal.fire({
          icon: 'error',
          title: 'Cancel Failed',
          text: 'Failed to cancel training. Please try again.',
          confirmButtonColor: '#ef4444'
        })
      }
      setSection('config')
    }
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      <Sidebar 
        collapsed={collapsed} 
        onToggle={() => setCollapsed(!collapsed)}
        section={section}
        onNavigate={setSection}
      />
      
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <header className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 shadow-lg flex-shrink-0">
          <div className="max-w-7xl mx-auto">
            <h1 className="text-3xl font-bold">Neural Trainer Pro</h1>
            <p className="text-blue-100 mt-2">Train ANY dataset - YOLO, CIFAR, CSV, Image Classification & More!</p>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto bg-gray-100 custom-scrollbar">
          <div className="p-4">
            <div className="max-w-full mx-auto">
              {section === 'dataset' && (
                <DatasetSelector 
                  onFilesSelected={handleDatasetUpload}
                  datasetInfo={datasetInfo}
                  onFormatChange={handleFormatChange}
                  uploadProgress={uploadProgress}
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
              
              {section === 'export' && (
                <ModelExport 
                  progress={progress}
                  datasetInfo={datasetInfo}
                />
              )}
              
              {section === 'history' && (
                <TrainingHistory 
                  history={trainingHistory}
                  onLoad={(h) => {
                    setConfig(h.config)
                    setSection('config')
                    Swal.fire({
                      icon: 'success',
                      title: 'Config Loaded!',
                      text: 'Training configuration has been restored.',
                      timer: 2000,
                      showConfirmButton: false
                    })
                  }}
                  onDelete={(id) => {
                    setTrainingHistory(prev => prev.filter(h => h.id !== id))
                  }}
                />
              )}
            </div>
          </div>
        </main>

        <footer className="bg-gradient-to-r from-gray-700 to-gray-900 text-white px-6 py-4 shadow-lg flex-shrink-0">
          <div className="max-w-7xl mx-auto">
            <div className="flex flex-col md:flex-row items-center justify-between gap-2 text-sm">
              <p className="font-medium">© 2025 Neural Trainer Pro</p>
              <p className="text-gray-300">Python + PyTorch Backend | CPU Training</p>
              <p className="text-gray-300">Supports: YOLO • CIFAR • CSV • Custom Formats</p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
