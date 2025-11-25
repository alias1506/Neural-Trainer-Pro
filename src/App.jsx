import React, { useState, useEffect, useRef } from 'react'
import Swal from 'sweetalert2'
import Sidebar from './components/Sidebar.jsx'
import DatasetSelector from './components/DatasetSelector.jsx'
import TrainingConfig from './components/TrainingConfig.jsx'
import TrainingProgress from './components/TrainingProgress.jsx'
import ModelExport from './components/ModelExport.jsx'
import TrainingHistory from './components/TrainingHistory.jsx'
import Header from './components/Header.jsx'
import Footer from './components/Footer.jsx'
import axios from 'axios'

// Always use localhost for API and WebSocket endpoints in local development
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
  const [config, setConfig] = useState(() => {
    // Load from sessionStorage on mount to preserve user's settings
    const saved = sessionStorage.getItem('trainingConfig')
    return saved ? JSON.parse(saved) : { epochs: 30, batchSize: 32, learningRate: 0.001, optimizer: 'adam' }
  })
  const [progress, setProgress] = useState(() => {
    const saved = sessionStorage.getItem('trainingProgress')
    return saved ? JSON.parse(saved) : { status: 'idle', epoch: 0, batch: 0, totalBatches: 0, totalEpochs: 0, trainLoss: 0, trainAcc: 0, valLoss: 0, valAcc: 0, timeMs: 0, lossHistory: [], accHistory: [], valLossHistory: [], valAccHistory: [], modelPath: '' }
  })
  // Persist training progress to sessionStorage
  useEffect(() => {
    sessionStorage.setItem('trainingProgress', JSON.stringify(progress))
  }, [progress])

  // Use ref to access latest progress in WebSocket callback (avoid stale closure)
  const progressRef = useRef(progress)
  useEffect(() => {
    progressRef.current = progress
  }, [progress])

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

  // Sync current progress to history if history is missing data (Backfill fix)
  useEffect(() => {
    if (progress.status === 'done' && trainingHistory.length > 0) {
      const latestHistory = trainingHistory[0]
      // Check if latest history is the one corresponding to current progress (by checking if it's done and missing data)
      if (latestHistory.status === 'done' && (!latestHistory.durationMs || latestHistory.metrics?.loss === undefined)) {

        // Calculate duration if missing
        let durationMs = latestHistory.durationMs
        if (!durationMs && progress.timeMs) {
          durationMs = progress.timeMs
        } else if (!durationMs && latestHistory.startTime && latestHistory.endTime) {
          durationMs = new Date(latestHistory.endTime) - new Date(latestHistory.startTime)
        }

        // Get metrics from progress
        const finalLoss = progress.lossHistory && progress.lossHistory.length > 0
          ? progress.lossHistory[progress.lossHistory.length - 1]
          : progress.trainLoss

        const finalAcc = progress.accHistory && progress.accHistory.length > 0
          ? progress.accHistory[progress.accHistory.length - 1]
          : progress.trainAcc

        // Only update if we actually have new data to add
        if (durationMs || finalLoss !== undefined) {
          setTrainingHistory(prev => {
            const updated = [...prev]
            updated[0] = {
              ...updated[0],
              durationMs: durationMs || updated[0].durationMs,
              metrics: {
                ...updated[0].metrics,
                loss: finalLoss !== undefined ? finalLoss : updated[0].metrics?.loss,
                trainLoss: finalLoss !== undefined ? finalLoss : updated[0].metrics?.trainLoss,
                trainAcc: finalAcc !== undefined ? finalAcc : updated[0].metrics?.trainAcc,
                // Ensure validation metrics are preserved or fallback to progress
                accuracy: updated[0].metrics?.accuracy || progress.valAcc,
                valAcc: updated[0].metrics?.valAcc || progress.valAcc,
                valLoss: updated[0].metrics?.valLoss || progress.valLoss
              }
            }
            return updated
          })
        }
      }
    }
  }, [progress, trainingHistory])

  // Save config to sessionStorage whenever it changes
  useEffect(() => {
    sessionStorage.setItem('trainingConfig', JSON.stringify(config))
  }, [config])

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
          try { console.log('[WS] connected to', WS_URL) } catch (_) { }
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
          try { console.warn('[WS] error', error) } catch (_) { }
        }

        websocket.onclose = () => {
          clearTimeout(connectionTimeout)
          try { console.warn('[WS] disconnected, retrying...') } catch (_) { }
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
    } else if (data.type === 'batch') {
      // Update current batch metrics only (do NOT push to history arrays)
      setProgress(prev => ({
        ...prev,
        status: 'training',
        epoch: data.epoch,
        totalEpochs: data.totalEpochs,
        batch: data.batch,
        totalBatches: data.totalBatches,
        trainLoss: data.trainLoss,
        trainAcc: data.trainAcc,
        timeMs: data.elapsed * 1000
      }))
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

      // Do NOT clean up or remove dataset after training. Keep datasetInfo and files.
      // (Ensure datasetInfo is not cleared anywhere after training)

      // Update history entry to done
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          // Store relative path for matching with export
          const relativePath = data.modelPath.replace(/^.*[\\\/]/, '')

          // Get latest progress data from ref
          const currentProgress = progressRef.current

          // Get final loss from the last epoch's data or current state
          const finalTrainLoss = currentProgress.lossHistory && currentProgress.lossHistory.length > 0
            ? currentProgress.lossHistory[currentProgress.lossHistory.length - 1]
            : currentProgress.trainLoss || 0

          const finalTrainAcc = currentProgress.accHistory && currentProgress.accHistory.length > 0
            ? currentProgress.accHistory[currentProgress.accHistory.length - 1]
            : currentProgress.trainAcc || 0

          // Use duration from progress (timeMs) which comes from backend
          const durationMs = currentProgress.timeMs || (new Date() - new Date(updated[0].startTime))

          updated[0] = {
            ...updated[0],
            status: 'done',
            metrics: {
              accuracy: data.finalValAcc, // Display Validation Accuracy as main Acc
              loss: finalTrainLoss,       // Display Train Loss as main Loss
              trainAcc: data.finalTrainAcc || finalTrainAcc,
              valAcc: data.finalValAcc,
              trainLoss: finalTrainLoss,
              valLoss: currentProgress.valLoss
            },
            // Store all history arrays for export metadata
            lossHistory: currentProgress.lossHistory || [],
            accHistory: currentProgress.accHistory || [],
            valLossHistory: currentProgress.valLossHistory || [],
            valAccHistory: currentProgress.valAccHistory || [],
            modelPath: relativePath,
            endTime: new Date().toISOString(),
            durationMs: durationMs
          }
          return updated
        }
        return prev
      })

      Swal.fire({
        title: 'Model Trained Successfully',
        html: `
          <div style="display: flex; flex-direction: column; align-items: center;">
            <svg width="80" height="80" fill="none" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#10b981" opacity="0.15"/><path d="M7 13l3 3 7-7" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <div style="font-size: 1.2rem; color: #10b981; font-weight: 600; margin-top: 1rem;">Training Complete!</div>
            <div style="color: #666666; font-size: 0.95rem; margin-top: 0.5rem;">You can now export or test your model.</div>
          </div>
        `,
        background: '#FFFFFF',
        confirmButtonColor: '#10b981',
        timer: 3000,
        showConfirmButton: false,
        iconHtml: '',
        customClass: {
          popup: 'no-icon-popup'
        }
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
        batch: 0,
        totalBatches: 0,
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
      })      // Update history entry to cancelled
      setTrainingHistory(prev => {
        if (prev.length > 0 && prev[0].status === 'running') {
          const updated = [...prev]
          const currentProgress = progressRef.current
          const durationMs = currentProgress.timeMs || (new Date() - new Date(updated[0].startTime))

          updated[0] = {
            ...updated[0],
            status: 'cancelled',
            endTime: new Date().toISOString(),
            durationMs: durationMs
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
          const currentProgress = progressRef.current
          const durationMs = currentProgress.timeMs || (new Date() - new Date(updated[0].startTime))

          updated[0] = {
            ...updated[0],
            status: 'failed',
            error: data.message,
            endTime: new Date().toISOString(),
            durationMs: durationMs
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

      Swal.fire({
        icon: 'success',
        title: 'Format Updated',
        text: `Dataset format changed to: ${newFormat}`,
        timer: 2000,
        showConfirmButton: false
      })
    }
  }

  const uploadAbortController = useRef(null)

  const handleCancelUpload = () => {
    if (uploadAbortController.current) {
      uploadAbortController.current.abort()
      uploadAbortController.current = null
    }
    setUploadProgress(null)
    Swal.fire({
      icon: 'info',
      title: 'Upload Cancelled',
      text: 'The upload has been cancelled.',
      confirmButtonText: 'OK',
      confirmButtonColor: '#3b82f6'
    }).then(() => {
      window.location.reload()
    })
  }

  const handleDatasetUpload = async (files, isRemove = false) => {
    // Handle dataset removal
    if (isRemove) {
      try {
        // Clean up server uploads folder
        await axios.post(`${API_URL}/clean-uploads`)

        setDatasetInfo(null)

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

      // Create abort controller for this upload
      if (uploadAbortController.current) {
        uploadAbortController.current.abort()
      }
      uploadAbortController.current = new AbortController()

      const response = await axios.post(`${API_URL}/upload-dataset`, formData, {
        signal: uploadAbortController.current.signal,
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
      const msg = error?.message === 'canceled' || error?.code === 'ERR_CANCELED' ? 'Upload cancelled.' : (error.response?.data?.error || error.message)
      setProgress(prev => ({ ...prev, status: 'upload-failed' }))
      Swal.fire({
        icon: 'error',
        title: 'Upload Failed',
        text: msg,
        confirmButtonText: 'OK',
        confirmButtonColor: '#ef4444'
      })
    }
  }

  const startTraining = async () => {
    try {
      // Update config with datasetType before starting training
      const updatedConfig = {
        ...config,
        datasetType: datasetInfo?.structure?.type || datasetInfo?.type || 'auto'
      }
      setConfig(updatedConfig)

      setSection('progress')
      setProgress({
        status: 'preparing',
        epoch: 0,
        batch: 0,
        totalBatches: 0,
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
        config: updatedConfig,
        currentEpoch: 0,
        metrics: { acc: 0, loss: 0 },
        modelPath: null
      }
      setTrainingHistory(prev => [newHistoryEntry, ...prev])

      await new Promise(resolve => setTimeout(resolve, 100))

      const response = await axios.post(`${API_URL}/train`, {
        datasetPath: datasetInfo.datasetPath,
        config: updatedConfig
      })

      if (response.data.success) {
        // Training started successfully
      }
    } catch (error) {
      let msg = error?.message || 'Unknown error';
      if (msg.toLowerCase().includes('network')) {
        msg = 'Network Error: Unable to reach backend server. Please check that the server is running and accessible.';
        setProgress(prev => ({ ...prev, status: 'network-error' }));
      } else {
        setProgress(prev => ({ ...prev, status: 'error' }));
      }
      Swal.fire({
        icon: 'error',
        title: 'Training Failed',
        text: msg,
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
              const currentProgress = progressRef.current
              const durationMs = currentProgress.timeMs || (new Date() - new Date(updated[0].startTime))

              updated[0] = {
                ...updated[0],
                status: 'cancelled',
                endTime: new Date().toISOString(),
                durationMs: durationMs
              }
              return updated
            }
            return prev
          })

          // Reset progress to default state
          // Update progress status but keep data so user can see where it stopped
          setProgress(prev => ({
            ...prev,
            status: 'cancelled',
            message: 'Training cancelled by user'
          }))

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
      // setSection('config') // Don't redirect, stay on progress page to see charts
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
        <Header />

        <main className="flex-1 overflow-y-auto bg-gray-100 custom-scrollbar">
          <div className="p-4">
            <div className="max-w-full mx-auto">
              {section === 'dataset' && (
                <DatasetSelector
                  onFilesSelected={handleDatasetUpload}
                  datasetInfo={datasetInfo}
                  onFormatChange={handleFormatChange}
                  uploadProgress={uploadProgress}
                  onCancelUpload={handleCancelUpload}
                />
              )}

              {section === 'config' && (
                <TrainingConfig
                  config={config}
                  onChange={setConfig}
                  onStart={startTraining}
                  datasetReady={!!datasetInfo}
                  datasetInfo={datasetInfo}
                  isTraining={progress.status === 'training' || progress.status === 'preparing'}
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
                  trainingHistory={trainingHistory}
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
                    setTrainingHistory([])
                    Swal.fire({
                      icon: 'success',
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

        <Footer />
      </div>
    </div>
  )
}

