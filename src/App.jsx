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
  const [section, setSection] = useState(() => {
    return sessionStorage.getItem('currentSection') || 'dataset'
  })
  const [uploadProgress, setUploadProgress] = useState(null)
  const [datasetInfo, setDatasetInfo] = useState(() => {
    // Load from sessionStorage on mount
    const saved = sessionStorage.getItem('datasetInfo')
    return saved ? JSON.parse(saved) : null
  })
  const [config, setConfig] = useState({ epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' })
  const [progress, setProgress] = useState({ status: 'idle', epoch: 0, trainLoss: 0, trainAcc: 0, valLoss: 0, valAcc: 0, timeMs: 0, lossHistory: [], accHistory: [], valLossHistory: [], valAccHistory: [], modelPath: '' })
  const [trainingHistory, setTrainingHistory] = useState(() => {
    const saved = sessionStorage.getItem('trainingHistory')
    if (saved) {
      const history = JSON.parse(saved)
      // Clean up any stale "running" entries on load (from previous session)
      return history.map(h => {
        if (h.status === 'running') {
          return { ...h, status: 'cancelled', endTime: h.endTime || new Date().toISOString() }
        }
        return h
      })
    }
    return []
  })

  // Save section state to sessionStorage
  useEffect(() => {
    sessionStorage.setItem('currentSection', section)
    
    // When navigating to export section, verify models exist
    // If no models, clear any orphaned dataset info
    if (section === 'export') {
      axios.get(`${API_URL}/list-models`)
        .then(response => {
          if (!response.data.models || response.data.models.length === 0) {
            // No models exist, clear dataset info
            setDatasetInfo(null)
            sessionStorage.removeItem('datasetInfo')
          }
        })
        .catch(() => {
          // Silently handle error
        })
    }
  }, [section])

  // Save training history to sessionStorage
  useEffect(() => {
    sessionStorage.setItem('trainingHistory', JSON.stringify(trainingHistory))
  }, [trainingHistory])

  // Save datasetInfo to sessionStorage whenever it changes (only 1 dataset at a time)
  useEffect(() => {
    if (datasetInfo) {
      // Always overwrite with current dataset (no multiple datasets stored)
      sessionStorage.setItem('datasetInfo', JSON.stringify(datasetInfo))
    } else {
      // Clear sessionStorage when no dataset
      sessionStorage.removeItem('datasetInfo')
    }
  }, [datasetInfo])

  // Clean up uploads folder on mount if sessionStorage is empty (browser was closed)
  useEffect(() => {
    const hasSession = sessionStorage.getItem('datasetInfo')
    if (!hasSession) {
      // Browser was closed, clean up server uploads
      axios.post(`${API_URL}/cleanup-uploads`).catch(err => {
        // Silently ignore cleanup errors
      })
    }
  }, [])

  // Setup WebSocket with retry logic
  useEffect(() => {
    let websocket = null
    let reconnectTimeout = null
    let isComponentMounted = true

    const connectWebSocket = () => {
      try {
        websocket = new WebSocket(WS_URL)
        
        // Set connection timeout
        const connectionTimeout = setTimeout(() => {
          if (websocket && websocket.readyState === WebSocket.CONNECTING) {
            websocket.close()
          }
        }, 5000)
        
        websocket.onopen = () => {
          clearTimeout(connectionTimeout)
        }
        
        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            handleTrainingUpdate(data)
          } catch (e) {
            // Ignore parse errors
          }
        }
        
        websocket.onerror = (error) => {
          clearTimeout(connectionTimeout)
        }
        
        websocket.onclose = () => {
          clearTimeout(connectionTimeout)
          // Only retry if component is still mounted and not manually closed
          if (isComponentMounted) {
            reconnectTimeout = setTimeout(connectWebSocket, 3000)
          }
        }
      } catch (error) {
        // Retry connection
        if (isComponentMounted) {
          reconnectTimeout = setTimeout(connectWebSocket, 3000)
        }
      }
    }

    connectWebSocket()
    
    return () => {
      isComponentMounted = false
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      if (websocket) websocket.close()
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
    } else if (data.type === 'batch') {
      // Real-time batch-level updates (every 10 batches)
      setProgress(prev => {
        return {
          ...prev,
          status: 'training',
          epoch: data.epoch,
          totalEpochs: data.totalEpochs,
          trainLoss: data.trainLoss,
          trainAcc: data.trainAcc,
          timeMs: data.elapsed * 1000
          // Don't update history arrays during batch updates - wait for epoch completion
        }
      })
    } else if (data.type === 'epoch') {
      // Only show console output after first epoch completes
      if (data.epoch >= 1) {
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
      }
      
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
        lossHistory: [...(prev.lossHistory || []), data.trainLoss],
        accHistory: [...(prev.accHistory || []), data.trainAcc],
        valLossHistory: [...(prev.valLossHistory || []), data.valLoss],
        valAccHistory: [...(prev.valAccHistory || []), data.valAcc]
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
      
      // Store relative path for download (remove absolute base path)
      const relativePath = data.modelPath.replace(/^.*[\\\/]uploads[\\\/]/, 'uploads/')
      setProgress(prev => ({ ...prev, status: 'done', modelPath: relativePath }))
      
      // Store numClasses and classes in datasetInfo for export
      if (data.numClasses || data.classes) {
        setDatasetInfo(prev => ({
          ...prev,
          numClasses: data.numClasses,
          classes: data.classes
        }))
      }
      
      // Clean up dataset after successful training
      try {
        await axios.post(`${API_URL}/clean-dataset`)
      } catch (cleanError) {
        // Silently handle cleanup errors
      }
      
      // Clear dataset info from state and session storage after training
      setDatasetInfo(null)
      sessionStorage.removeItem('datasetInfo')
      
      // Update history entry to done
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          // Store relative path for matching with export
          const relativePath = data.modelPath.replace(/^.*[\\\\/]/, '')
          // Get final loss from the last epoch's data
          const finalLoss = progress.lossHistory && progress.lossHistory.length > 0 
            ? progress.lossHistory[progress.lossHistory.length - 1]
            : progress.trainLoss || 0
          updated[0] = {
            ...updated[0],
            status: 'done',
            metrics: { 
              acc: data.finalValAcc, 
              loss: finalLoss, 
              trainAcc: data.finalTrainAcc,
              valAcc: data.finalValAcc,
              trainLoss: finalLoss
            },
            // Store all history arrays for export metadata
            lossHistory: progress.lossHistory || [],
            accHistory: progress.accHistory || [],
            valLossHistory: progress.valLossHistory || [],
            valAccHistory: progress.valAccHistory || [],
            modelPath: relativePath,
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
          <div class="text-center">
            <p class="mb-2 text-lg"><strong>Validation Accuracy:</strong> ${(data.finalValAcc * 100).toFixed(2)}%</p>
            <p class="text-green-600 font-semibold mt-3">✅ Model Trained Successfully!</p>
            <p class="text-sm text-gray-600 mt-2">Redirecting to export page...</p>
          </div>
        `,
        confirmButtonColor: '#10b981',
        timer: 2000,
        showConfirmButton: false
      }).then(() => {
        // Redirect to export page after notification
        setSection('export')
      })
    } else if (data.type === 'cancelled') {
      console.clear()
      console.log('🛑 TRAINING CANCELLED')
      
      // Reset progress to default state
      setProgress({ 
        status: 'idle', 
        epoch: 0, 
        totalEpochs: 0,
        trainLoss: 0, 
        trainAcc: 0, 
        valLoss: 0, 
        valAcc: 0, 
        timeMs: 0, 
        lossHistory: [], 
        accHistory: [], 
        valLossHistory: [], 
        valAccHistory: [], 
        modelPath: '' 
      })
      
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
      
      Swal.fire({
        icon: 'info',
        title: 'Training Cancelled',
        text: 'Training has been stopped.',
        confirmButtonColor: '#3b82f6',
        timer: 2000,
        showConfirmButton: false
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
        // Silently handle cleanup errors
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
        totalEpochs: config.epochs || 10,
        trainLoss: 0, 
        trainAcc: 0, 
        valLoss: 0, 
        valAcc: 0, 
        timeMs: 0,
        lossHistory: [],
        accHistory: [],
        valLossHistory: [],
        valAccHistory: [],
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
          
          // Reset progress to default state
          setProgress({ 
            status: 'idle', 
            epoch: 0, 
            totalEpochs: 0,
            trainLoss: 0, 
            trainAcc: 0, 
            valLoss: 0, 
            valAcc: 0, 
            timeMs: 0, 
            lossHistory: [], 
            accHistory: [], 
            valLossHistory: [], 
            valAccHistory: [], 
            modelPath: '' 
          })
          
          Swal.fire({
            icon: 'info',
            title: 'Training Cancelled',
            text: 'Training has been stopped successfully.',
            confirmButtonColor: '#3b82f6',
            timer: 2000
          })
        }
      } catch (error) {
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
                  onModelExported={(modelPath, format) => {
                    // Update history to mark as exported
                    setTrainingHistory(prev => prev.map(h => {
                      if (h.modelPath === modelPath) {
                        return { ...h, exportedFormat: format }
                      }
                      return h
                    }))
                    
                    // Reset progress to initial state after export
                    setProgress({ 
                      status: 'idle', 
                      epoch: 0, 
                      totalEpochs: 0,
                      trainLoss: 0, 
                      trainAcc: 0, 
                      valLoss: 0, 
                      valAcc: 0, 
                      timeMs: 0, 
                      lossHistory: [], 
                      accHistory: [], 
                      valLossHistory: [], 
                      valAccHistory: [], 
                      modelPath: '' 
                    })
                  }}
                  onAllModelsCleared={() => {
                    // Reset all state to default when no models exist
                    setDatasetInfo(null)
                    sessionStorage.removeItem('datasetInfo')
                    
                    // Reset progress to initial state
                    setProgress({ 
                      status: 'idle', 
                      epoch: 0, 
                      totalEpochs: 0,
                      trainLoss: 0, 
                      trainAcc: 0, 
                      valLoss: 0, 
                      valAcc: 0, 
                      timeMs: 0, 
                      lossHistory: [], 
                      accHistory: [], 
                      valLossHistory: [], 
                      valAccHistory: [], 
                      modelPath: '' 
                    })
                    
                    Swal.fire({
                      icon: 'info',
                      title: 'All Models Cleared',
                      text: 'All training data has been reset to default.',
                      timer: 2000,
                      showConfirmButton: false
                    })
                  }}
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
